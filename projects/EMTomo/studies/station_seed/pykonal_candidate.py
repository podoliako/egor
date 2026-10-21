"""Study-only cell-centred PyKonal forward and independent forward screens.

Run from the repository root (no inverse, production imports or monkeypatches)::

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        venv/bin/python projects/EMTomo/studies/station_seed/pykonal_candidate.py

The default output contract matches EMTomo: the original shape, float32 times,
cell centres at (i + 1/2) h, an exact off-grid source, and a homogeneous sphere
of radius 3h with the containing station cell's velocity. Request dtype=float64
for continuity diagnostics. No interpolation, padding, source snapping, velocity
rounding, initial-model prior, ray changes, or inverse stopping logic is used.

Initialization follows the INSTALLED pykonal 0.4.1 solver.pyx, not the separate
forward_modeling source-box approximation. solve() skips known neighbours
(lines 275-278), but unknown=False alone does NOT freeze a trial node. Every
sphere node is therefore known=True, unknown=False, T=distance/v_station. Only
sphere nodes with an in-domain exterior axial neighbour are put on the trial
heap, to trigger exterior updates. These launch nodes intentionally remain
known, too: solve() pops them without skipping known active nodes (248-253).
This overlap is supported by the inspected implementation, but must be checked
again on a PyKonal upgrade. No post-solve analytic overwrite hides seed changes.

This is a discrete interior Dirichlet sphere, not an exact subcell spherical
interface solver. In particular the first exterior stencil samples the exterior
nodal speed; interface errors and fixed-sphere suppression of possible fast
heterogeneous re-entry paths remain limitations of this parameterization.

SCREEN WARNING: single-ULP controls alone are insufficient. This candidate has
millisecond jumps under tiny multi-ULP / finite velocity perturbations; the JSON
contains an explicit continuity verdict. It is not a production recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pykonal
import skfmm

SOURCE_RADIUS_CELLS = 3.0


def _geometry(velocity, cell_size, source_m, radius_cells):
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 3 or any(n == 0 for n in velocity.shape):
        raise ValueError("velocity must be a nonempty 3D array")
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("velocity must be finite and strictly positive")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be finite and strictly positive")
    if not np.isfinite(radius_cells) or radius_cells <= 0:
        raise ValueError("radius_cells must be finite and strictly positive")
    source = np.asarray(source_m, dtype=np.float64)
    extent = np.asarray(velocity.shape) * cell_size
    if (source.shape != (3,) or not np.isfinite(source).all()
            or np.any(source < 0) or np.any(source > extent)):
        raise ValueError(f"source {source_m} is outside the model domain")
    axes = [(np.arange(n) + 0.5) * cell_size - s for n, s in zip(velocity.shape, source)]
    distance = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2
                       + axes[2][None, None, :] ** 2)
    source_cell = tuple(np.minimum(source // cell_size, np.asarray(velocity.shape) - 1).astype(int))
    source_velocity = float(velocity[source_cell])
    inside = distance <= radius_cells * cell_size
    if not inside.any():
        raise ValueError("sphere contains no cell centre; increase radius_cells (source is not snapped)")
    return velocity, distance, inside, source_velocity


def _frontier(inside):
    """Inner axial shell; the physical domain boundary is not an exterior node."""
    boundary = np.zeros_like(inside)
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        boundary[lo] |= inside[lo] & ~inside[hi]
        boundary[hi] |= inside[hi] & ~inside[lo]
    return boundary


def _initialise_solver(velocity, h, distance, inside, source_velocity):
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = (0.5 * h,) * 3
    solver.velocity.node_intervals = (h,) * 3
    solver.velocity.npts = velocity.shape
    # ScalarField3D.values can alias its argument. Always give it a private copy.
    medium = np.array(velocity, dtype=np.float64, order="C", copy=True)
    medium[inside] = source_velocity
    solver.velocity.values = medium
    solver.traveltime.values[inside] = distance[inside] / source_velocity
    solver.known[inside] = True
    solver.unknown[inside] = False
    boundary = _frontier(inside)
    for index in np.argwhere(boundary):
        solver.trial.push(*(int(i) for i in index))
    return solver, boundary


def _march(solver, distance, inside, source_velocity):
    if not solver.solve():
        raise RuntimeError("PyKonal solve() returned False")
    times = solver.traveltime.values
    if not np.isfinite(times).all() or not solver.known.all():
        raise RuntimeError("PyKonal did not reach every cell centre")
    if not np.array_equal(times[inside], distance[inside] / source_velocity):
        raise RuntimeError("PyKonal modified the fixed analytic Dirichlet sphere")
    return times


def travel_time_field(velocity, cell_size, source_m, *, radius_cells=SOURCE_RADIUS_CELLS,
                      dtype=np.float32):
    """Independent study candidate with the production field's shape/coordinates.

    The optional radius and dtype are experiments confined to this study. A
    smaller radius that contains no cell centre raises instead of snapping.
    Sources on any physical model face/edge/corner, including the upper extent,
    are accepted even though these positions are outside the nodal-centre box.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    velocity, distance, inside, source_velocity = _geometry(
        velocity, cell_size, source_m, radius_cells)
    if inside.all():
        return (distance / source_velocity).astype(dtype)
    solver, _ = _initialise_solver(velocity, cell_size, distance, inside, source_velocity)
    return _march(solver, distance, inside, source_velocity).astype(dtype, copy=True)


def corrected_skfmm_field(velocity, cell_size, source_m, *, radius_cells=SOURCE_RADIUS_CELLS,
                          dtype=np.float32):
    """Frozen corrected EMTomo order-2 baseline; no production module imported."""
    velocity, distance, inside, source_velocity = _geometry(
        velocity, cell_size, source_m, radius_cells)
    if inside.all():
        return (distance / source_velocity).astype(dtype)
    medium = np.array(velocity, copy=True)
    medium[inside] = source_velocity
    times = np.asarray(skfmm.travel_time(
        distance - radius_cells * cell_size, medium, dx=cell_size, order=2), dtype=np.float64)
    times += radius_cells * cell_size / source_velocity
    times[inside] = distance[inside] / source_velocity
    return times.astype(dtype)


SOLVERS = {"pykonal_dirichlet": travel_time_field, "corrected_skfmm_order2": corrected_skfmm_field}


def _error(difference):
    a = np.asarray(difference, dtype=np.float64)
    if not a.size:
        return None
    return {"rms_s": float(np.sqrt(np.mean(a ** 2))), "max_abs_s": float(np.max(np.abs(a))),
            "mean_s": float(np.mean(a))}


def _homogeneous_screen():
    cases = [
        ("ulp_geometry_surface", (48, 32, 32), (1000., 875., 0.)),
        ("offgrid_surface", (32, 24, 24), (1900., 1450., 0.)),
        ("offgrid_interior", (32, 24, 24), (1900., 1450., 1300.)),
        ("lower_corner", (32, 24, 24), (0., 0., 0.)),
        ("upper_corner", (32, 24, 24), (4000., 3000., 3000.)),
        ("on_node_interior", (32, 24, 24), (1937.5, 1437.5, 1312.5)),
        ("fractional_interior", (17, 13, 11), (531.25, 827.125, 121.5)),
        ("upper_x_face", (17, 13, 11), (2125., 387.25, 600.)),
        ("whole_domain_in_sphere", (2, 2, 2), (0., 0., 0.)),
        ("single_axis_line", (1, 1, 12), (62.5, 62.5, 0.)),
    ]
    results = []
    for name, shape, source in cases:
        h = 125.
        velocity = np.full(shape, 4000.)
        _, distance, inside, v = _geometry(velocity, h, source, SOURCE_RADIUS_CELLS)
        exact = distance / v
        entry = {"case": name, "shape": list(shape), "cell_size_m": h,
                 "source_m": list(source), "inside_nodes": int(inside.sum()), "solvers": {}}
        for solver_name, solve in SOLVERS.items():
            start = perf_counter()
            actual = solve(velocity, h, source, dtype=np.float64)
            elapsed = perf_counter() - start
            assert actual.shape == shape and actual.dtype == np.float64 and np.isfinite(actual).all()
            assert np.array_equal(actual[inside], exact[inside])
            entry["solvers"][solver_name] = {
                "elapsed_s_single": elapsed, "all": _error(actual - exact),
                "outside_sphere": _error((actual - exact)[~inside]),
                "near_3h_to_6h": _error((actual - exact)[(distance > 3*h) & (distance <= 6*h)]),
                "far_gt_6h": _error((actual - exact)[distance > 6*h]),
                "inside_exact_float64": True,
            }
        results.append(entry)
    return results


def _continuity_screen():
    shape, h, source = (48, 32, 32), 125., (1000., 875., 0.)
    velocity = np.full(shape, 4000.)
    _, distance, inside, _ = _geometry(velocity, h, source, SOURCE_RADIUS_CELLS)
    block = np.zeros(shape, dtype=bool)
    block[4:8, 4:8, :4] = True
    block &= ~inside
    rng = np.random.default_rng(20261017)
    random_sign = rng.choice([-1., 1.], size=shape)
    cases = []
    for direction in (np.inf, -np.inf):
        medium = velocity.copy()
        medium[block] = np.nextafter(4000., direction)
        cases.append(("plus_1ulp_block" if direction > 0 else "minus_1ulp_block", medium))
    random_medium = np.nextafter(velocity, np.where(random_sign > 0, np.inf, -np.inf))
    random_medium[inside] = 4000.
    cases.append(("random_signed_1ulp_outside", random_medium))
    ulp = np.spacing(4000.)
    for count in (2, 4, 8, 16, 32, 64):
        medium = velocity.copy()
        medium[block] += count * ulp
        cases.append((f"plus_{count}ulp_block", medium))
    for delta in (1e-11, 1e-10, 1e-9, 1e-6, 1e-3, 1.):
        medium = velocity.copy()
        medium[block] += delta
        cases.append((f"plus_{delta:g}_m_s_block", medium))
    originals = {name: solve(velocity, h, source, dtype=np.float64) for name, solve in SOLVERS.items()}
    results = []
    repeated_jump = None
    for case, medium in cases:
        delta = float(np.max(np.abs(medium - velocity)))
        # A comparison bound for continuous arrivals in the perturbed homogeneous
        # medium, not a claim that either second-order discretization obeys it.
        physical_scale = float(distance.max()) * delta / (4000. * (4000. - delta))
        entry = {"case": case, "changed_nodes": int(np.count_nonzero(medium != velocity)),
                 "max_velocity_change_m_s": delta, "continuous_medium_comparison_bound_s": physical_scale,
                 "solvers": {}}
        for name, solve in SOLVERS.items():
            actual = solve(medium, h, source, dtype=np.float64)
            difference = actual - originals[name]
            entry["solvers"][name] = {
                "float64": _error(difference),
                "float32_storage": _error(actual.astype(np.float32).astype(float)
                                           - originals[name].astype(np.float32).astype(float)),
                "max_abs_index": [int(i) for i in np.unravel_index(np.argmax(np.abs(difference)), shape)],
            }
            if (name == "pykonal_dirichlet" and repeated_jump is None and delta < 1e-6
                    and entry["solvers"][name]["float64"]["max_abs_s"] > 1e-9):
                repeated = solve(medium, h, source, dtype=np.float64)
                assert np.array_equal(actual, repeated)
                repeated_jump = {"case": case, "bitwise_reproducible_float64": True}
        results.append(entry)
    ulp_pass = all(row["solvers"]["pykonal_dirichlet"]["float64"]["max_abs_s"] < 1e-9
                   for row in results[:3])
    small_rows = [row for row in results if row["max_velocity_change_m_s"] <= 1.1e-6]
    small_pass = all(row["solvers"]["pykonal_dirichlet"]["float64"]["max_abs_s"] < 1e-9
                     for row in small_rows)
    worst = max(small_rows, key=lambda row: row["solvers"]["pykonal_dirichlet"]["float64"]["max_abs_s"])
    return {"shape": list(shape), "cell_size_m": h, "source_m": list(source),
            "radius_m": 3*h, "block": "[4:8,4:8,:4] excluding sphere", "random_seed": 20261017,
            "diagnostic_max_abs_threshold_s": 1e-9,
            "candidate_passes_three_single_ulp_cases": ulp_pass,
            "candidate_passes_small_change_controls": small_pass,
            "worst_small_change_case": worst["case"], "repeated_jump": repeated_jump,
            "cases": results}


def _invariant_screen():
    h, shape = 125., (32, 24, 24)
    sources = [(1900., 1450., 0.), (1900., 1450., 1300.), (0., 0., 0.), (4000., 3000., 3000.)]
    no_leak = []
    for source in sources:
        base = np.full(shape, 4000.)
        _, distance, inside, _ = _geometry(base, h, source, SOURCE_RADIUS_CELLS)
        reference = travel_time_field(base, h, source, dtype=np.float64)
        for shell_speed in (2000., 6000.):
            changed = base.copy()
            changed[(distance > 2*h) & inside] = shell_speed
            before = changed.copy()
            changed.flags.writeable = False
            actual = travel_time_field(changed, h, source, dtype=np.float64)
            assert np.array_equal(changed, before)
            assert np.array_equal(actual, reference)
            no_leak.append({"source_m": list(source), "shell_velocity_m_s": shell_speed,
                            "bitwise_equal_float64": True, "max_abs_difference_s": 0.})
    source = sources[0]
    base = np.full(shape, 4000.)
    _, distance, inside, _ = _geometry(base, h, source, SOURCE_RADIUS_CELLS)
    base[(distance > 2*h) & inside] = 2000.
    base[(distance > 3*h) & (distance < 6*h)] = 4500.
    # Noncontiguous AND read-only input tests the ScalarField3D alias boundary.
    storage = np.repeat(base, 2, axis=0)
    velocity = storage[::2]
    before = velocity.copy()
    velocity.flags.writeable = False
    v, distance, inside, source_velocity = _geometry(velocity, h, source, SOURCE_RADIUS_CELLS)
    solver, boundary = _initialise_solver(v, h, distance, inside, source_velocity)
    assert np.array_equal(solver.velocity.values[~inside], before[~inside])
    assert np.all(solver.velocity.values[inside] == source_velocity)
    assert not np.shares_memory(solver.velocity.values, velocity)
    assert np.array_equal(solver.velocity.min_coords, np.full(3, h/2))
    assert np.array_equal(solver.velocity.npts, shape)
    assert solver.known[inside].all() and not solver.known[~inside].any()
    assert not solver.unknown[inside].any() and solver.unknown[~inside].all()
    assert set(solver.trial.keys) == set(map(tuple, np.argwhere(boundary)))
    actual = _march(solver, distance, inside, source_velocity).copy()
    assert np.array_equal(solver.velocity.values[~inside], before[~inside])
    assert np.array_equal(velocity, before) and np.array_equal(storage[::2], before)
    explicit = before.copy()
    explicit[inside] = source_velocity
    assert np.array_equal(actual, travel_time_field(explicit, h, source, dtype=np.float64))
    homogeneous = travel_time_field(np.full(shape, 4000.), h, source, dtype=np.float64)
    outside_effect = _error((actual - homogeneous)[distance > 6*h])
    assert outside_effect["max_abs_s"] > 1e-3
    default = travel_time_field(velocity, h, source)
    assert default.shape == shape and default.dtype == np.float32
    assert np.array_equal(default, actual.astype(np.float32))
    station_changed = np.full(shape, 4000.)
    station_index = tuple(np.minimum(np.asarray(source) // h, np.asarray(shape) - 1).astype(int))
    station_changed[station_index] = 4100.
    station_effect = travel_time_field(station_changed, h, source, dtype=np.float64) - homogeneous
    assert np.max(np.abs(station_effect[~inside])) > 1e-4
    # A tiny heterogeneous domain must also ignore nonstation interior speeds.
    tiny = np.full((2, 2, 2), 6000.)
    tiny[0, 0, 0] = 4000.
    _, tiny_distance, _, _ = _geometry(tiny, h, (0., 0., 0.), SOURCE_RADIUS_CELLS)
    assert np.array_equal(travel_time_field(tiny, h, (0., 0., 0.), dtype=np.float64), tiny_distance/4000.)
    invalid = [(-1., 0., 0.), (4000.01, 0., 0.), (np.nan, 0., 0.), (0., 0.)]
    for bad_source in invalid:
        try:
            travel_time_field(base, h, bad_source)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid source accepted: {bad_source}")
    return {"no_inner_leak": no_leak,
            "readonly_noncontiguous_input_unchanged": True,
            "private_speed_copy": True, "exterior_speeds_exact_before_and_after_solve": True,
            "fixed_seed_unchanged_before_any_overwrite": True,
            "known_unknown_heap_initialization_checked": True,
            "sphere_nodes": int(inside.sum()), "launch_nodes": int(boundary.sum()),
            "outside_heterogeneity_far_effect": outside_effect,
            "station_cell_velocity_far_effect": _error(station_effect[distance > 6*h]),
            "default_float32_shape_and_cast_exact": True,
            "whole_tiny_domain_analytic_with_heterogeneous_input": True,
            "invalid_sources_rejected": len(invalid)}


def _vertical_layer_screen():
    # Vertical rays in horizontal layers have the exact integral dz/v(z).
    # Interfaces lie on voxel faces, NOT at the velocity samples (cell centres).
    interfaces, speeds = [0., 1000., 2500., 4000.], [3000., 4500., 6000.]
    results = []
    for h in (125., 62.5):
        shape = tuple(int(extent/h) for extent in (3000., 2500., 4000.))
        z = (np.arange(shape[2]) + 0.5) * h
        layer = np.searchsorted(interfaces[1:-1], z, side="right")
        velocity = np.broadcast_to(np.asarray(speeds)[layer], shape).copy()
        ix, iy = shape[0]//2, shape[1]//2
        source = ((ix + 0.5)*h, (iy + 0.5)*h, 0.)
        exact = sum(np.clip(z - lo, 0., hi - lo) / speed
                    for lo, hi, speed in zip(interfaces[:-1], interfaces[1:], speeds))
        entry = {"shape": list(shape), "cell_size_m": h, "source_m": list(source), "solvers": {}}
        for name, solve in SOLVERS.items():
            actual = solve(velocity, h, source, dtype=np.float64)[ix, iy]
            entry["solvers"][name] = {
                "vertical_column": _error(actual - exact),
                "below_first_interface": _error((actual - exact)[z > 1000.]),
                "deepest_time_s": float(actual[-1]), "deepest_exact_s": float(exact[-1]),
                "interface_neighbour_errors_s": [float((actual-exact)[k]) for k in
                    sorted({int(boundary/h) + offset for boundary in interfaces[1:-1]
                            for offset in (-1, 0, 1)})],
            }
        results.append(entry)
    return {"interfaces_m": interfaces, "velocities_m_s": speeds,
            "analytic": "sum of vertical segment length / layer velocity; sphere wholly in first layer",
            "caveat": "FMM uses nodal speeds at cell centres, not an exact voxel-face interface integrator",
            "resolutions": results}


def _runtime_screen(repeats):
    results = []
    for shape, heterogeneous in [((48, 32, 32), False), ((160, 56, 56), False), ((160, 56, 56), True)]:
        h, source = 125., (1000., 875., 0.)
        velocity = np.full(shape, 4000.)
        if heterogeneous:
            x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
            velocity = np.broadcast_to(3500. + 14*z + 180*np.sin(x/9)*np.cos(y/7), shape).copy()
        entry = {"shape": list(shape), "nodes": int(np.prod(shape)), "heterogeneous": heterogeneous,
                 "source_m": list(source), "cell_size_m": h, "solvers": {}}
        # Warm up both separately, then alternate measurement order to reduce bias.
        for solve in SOLVERS.values():
            solve(velocity, h, source)
        samples = {name: [] for name in SOLVERS}
        for repeat in range(repeats):
            items = list(SOLVERS.items())
            if repeat % 2:
                items.reverse()
            for name, solve in items:
                start = perf_counter()
                solve(velocity, h, source)
                samples[name].append(perf_counter() - start)
        for name, values in samples.items():
            entry["solvers"][name] = {"seconds": values, "median_s": float(np.median(values)),
                                      "min_s": float(min(values)), "max_s": float(max(values))}
        entry["pykonal_over_skfmm_median_ratio"] = (
            entry["solvers"]["pykonal_dirichlet"]["median_s"]
            / entry["solvers"]["corrected_skfmm_order2"]["median_s"])
        results.append(entry)
    return {"repeats": repeats, "warmups_per_solver_per_case": 1,
            "scope": "serial complete field call, float32 output; includes validation, geometry, copies and seeding; excludes imports",
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="runtime samples per solver and geometry")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if version("pykonal") != "0.4.1":
        raise RuntimeError("Reinspect known/trial initialization before screening another PyKonal version")
    start = perf_counter()
    installed_source = Path(pykonal.__file__).with_name("solver.pyx")
    result = {
        "scope": "forward-only; production, G, inverse, priors, stopping rules, dependencies and venv unchanged",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pykonal": version("pykonal"), "skfmm": skfmm.__version__,
                        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}},
        "provenance": {"candidate_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "installed_solver_pyx_sha256": hashlib.sha256(installed_source.read_bytes()).hexdigest(),
                       "baseline": "frozen corrected skfmm order=2, phi=d-R; speed[d<=R]=v_station; +R/v_station; inner analytic overwrite"},
        "candidate": {"radius_cells": SOURCE_RADIUS_CELLS, "default_dtype": "float32", "screen_errors_dtype": "float64",
                      "grid": "unchanged input shape; min_coords=(h/2,h/2,h/2), node_intervals=(h,h,h)",
                      "dirichlet": "all sphere nodes known=True, unknown=False; inner axial shell also queued as protected launch nodes",
                      "offgrid_source": "exact Euclidean sphere times; containing voxel selects station velocity; no source snapping"},
    }
    print("Checking input/medium/Dirichlet invariants", flush=True)
    result["invariants"] = _invariant_screen()
    print("Screening homogeneous analytic fields", flush=True)
    result["homogeneous"] = _homogeneous_screen()
    print("Screening velocity continuity (ULPs and finite perturbations)", flush=True)
    result["continuity"] = _continuity_screen()
    print("Screening exact vertical arrivals through heterogeneous layers", flush=True)
    result["vertical_layers"] = _vertical_layer_screen()
    print("Timing warmed serial field calls", flush=True)
    result["runtime"] = _runtime_screen(args.repeats)
    result["verdict"] = {
        "continuity_status": ("passes_tested_controls" if result["continuity"]["candidate_passes_small_change_controls"]
                              else "reject_as_continuity_fix"),
        "ready_for_inverse": False,
        "reason": "Passing single-ULP cases does not offset millisecond jumps at tiny larger perturbations; no inverse validated.",
    }
    result["limitations"] = [
        "Forward screen only; no inverse, gradient/ray Jacobian consistency, event interpolation or flat4 validation.",
        "Fixed nodal Dirichlet sphere approximates a subcell spherical boundary; exterior stencil sampling differs from skfmm phi initialization.",
        "All interior times remain fixed even if a heterogeneous fast exterior could yield faster re-entry arrivals; retains EMTomo sphere convention.",
        "Sharp voxel-face layers are sampled as nodal speeds; vertical analytic errors include interface discretization bias.",
        "PyKonal has second-order <= time comparisons (solver.pyx 311-315); its exact jump mechanism is not established by this screen.",
        "The candidate fails tiny multi-ULP/finite perturbation continuity; zero single-ULP differences must not be interpreted as a fix.",
        "Known launch nodes on the trial heap rely on the inspected 0.4.1 implementation, not a documented general Dirichlet API.",
        "Runtime is serial on this host under its current load, not multiprocessing throughput or a memory benchmark.",
    ]
    result["screen_elapsed_s"] = perf_counter() - start
    path = Path(__file__).with_name("pykonal_screen.json")
    # Compact summaries only: no fields, models, or pointwise error arrays.
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "screen_elapsed_s": result["screen_elapsed_s"],
                      "verdict": result["verdict"],
                      "candidate_passes_three_single_ulp_cases": result["continuity"]["candidate_passes_three_single_ulp_cases"],
                      "worst_small_change_case": result["continuity"]["worst_small_change_case"],
                      "runtime": result["runtime"]["cases"]}, indent=2))


if __name__ == "__main__":
    main()
