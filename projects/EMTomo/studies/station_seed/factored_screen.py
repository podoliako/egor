"""Independent forward-only physical/continuity screen; writes ONLY this JSON sibling.

Run from the repository venv, in bounded batches if desired::

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python projects/EMTomo/studies/station_seed/factored_screen.py \
        --reset --sections homogeneous invariants symmetry ulp
    ... factored_screen.py --sections layered
    ... factored_screen.py --sections boundary

No production/inverse imports, model files, dependencies, solver edits, priors,
or stopping changes. Candidate source is compiled unchanged EXCEPT Numba's disk
cache flag is disabled in memory, preventing writes outside the two-file scope.
The exact frozen baseline functions are AST-extracted from flat4_controls.py to
avoid importing its unrelated inverse machinery. Their float32 outputs are
checked against a separately evaluated native-float64 version of the formula.
For fixed-physical-radius controls only, the extracted baseline's RADIUS_CELLS
constant is parameterized; those cases are explicitly labelled.

Analytic references describe the continuum medium, not either discrete operator.
Layer interfaces are voxel faces. Sphere voxelization and nodal slowness averaging
are intentional sources of error, not reasons to adjust the solver in this screen.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import types

sys.dont_write_bytecode = True

import numpy as np
import numba
import skfmm

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "factored_screen.json"
SECTIONS = ("homogeneous", "invariants", "symmetry", "ulp", "layered", "boundary")
MAX_NODES = 1_999_999
TOLERANCE = 1e-12
MAX_CYCLES = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate():
    path = HERE / "factored_candidate.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    disabled = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "njit":
            for keyword in node.keywords:
                if keyword.arg == "cache" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    keyword.value.value = False
                    disabled += 1
    module = types.ModuleType("_factored_screen_candidate")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module, disabled


CANDIDATE, DISABLED_CACHES = load_candidate()
BASELINE_PATH = HERE / "flat4_controls.py"
BASELINE_TREE = ast.parse(BASELINE_PATH.read_text(), filename=str(BASELINE_PATH))
BASELINE_NAMESPACE = {"np": np, "skfmm": skfmm, "RADIUS_CELLS": 3.0}
exec(compile(ast.Module(body=[node for node in BASELINE_TREE.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name in ("_frozen_field", "homogenized_field")],
                        type_ignores=[]), str(BASELINE_PATH), "exec"), BASELINE_NAMESPACE)


def geometry(velocity, h, source, radius_cells):
    axes = [(np.arange(n) + 0.5) * h - s for n, s in zip(velocity.shape, source)]
    distance = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2
                       + axes[2][None, None, :] ** 2)
    cell = tuple(np.minimum(np.asarray(source) // h, np.asarray(velocity.shape) - 1).astype(int))
    return distance, distance <= radius_cells * h, float(velocity[cell]), cell


def error_stats(delta, mask=None):
    delta = np.asarray(delta, dtype=np.float64)
    selected = np.ones(delta.shape, dtype=bool) if mask is None else np.broadcast_to(mask, delta.shape)
    values = delta[selected]
    if not values.size:
        return {"count": 0}
    worst = np.unravel_index(np.argmax(np.where(selected, np.abs(delta), -1.0)), delta.shape)
    return {"count": int(values.size), "mean_s": float(values.mean()),
            "rms_s": float(np.sqrt(np.mean(values * values))),
            "max_abs_s": float(np.max(np.abs(values))), "min_signed_s": float(values.min()),
            "max_signed_s": float(values.max()), "p95_abs_s": float(np.quantile(np.abs(values), 0.95)),
            "worst_index": [int(i) for i in worst], "worst_signed_s": float(delta[worst])}


def baseline64(velocity, h, source, radius_cells):
    velocity = np.asarray(velocity, dtype=np.float64)
    distance, inside, station_velocity, _ = geometry(velocity, h, source, radius_cells)
    if inside.all():
        return distance / station_velocity
    # Match frozen .copy() exactly: skfmm is sensitive to non-C-order speeds.
    medium = velocity.copy()
    medium[inside] = station_velocity
    times = np.asarray(skfmm.travel_time(distance - radius_cells * h, medium, dx=h, order=2), dtype=np.float64)
    times += radius_cells * h / station_velocity
    times[inside] = distance[inside] / station_velocity
    return times


def solve_one(name, velocity, h, source, radius_cells):
    started = perf_counter()
    try:
        nodes = int(np.prod(np.asarray(velocity.shape) + 1))
        if nodes > MAX_NODES:
            raise ValueError(f"screen node budget exceeded: {nodes}")
        if name == "factored":
            field, info = CANDIDATE.travel_time_field(
                velocity, h, source, radius_cells=radius_cells, tolerance=TOLERANCE,
                max_cycles=MAX_CYCLES, diagnostics=True, dtype=np.float64)
        else:
            field = baseline64(velocity, h, source, radius_cells)
            native_seconds = perf_counter() - started
            BASELINE_NAMESPACE["RADIUS_CELLS"] = radius_cells
            frozen = BASELINE_NAMESPACE["homogenized_field"](velocity, h, source)
            equal = bool(np.array_equal(field.astype(np.float32), frozen))
            if not equal:
                raise AssertionError("native baseline float32 cast differs from extracted frozen baseline")
            info = {"native_float64_seconds": native_seconds,
                    "frozen_float32_seconds": perf_counter() - started - native_seconds,
                    "frozen_float32_bitwise_verified": equal,
                    "baseline_radius": "original_3h" if radius_cells == 3.0 else "parameterized_frozen_radius",
                    "convergence_info": "skfmm exposes no sweep count/residual"}
        if field.shape != velocity.shape or field.dtype != np.float64 or not np.isfinite(field).all():
            raise AssertionError("invalid field shape, dtype, or finite values")
        info["wall_seconds"] = perf_counter() - started
        info["finite_nonnegative"] = bool(np.all(field >= 0))
        return field, info
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                      "wall_seconds": perf_counter() - started}


def pair(name, velocity, h, source, radius_cells=3.0, reference=None, masks=None):
    distance, inside, station_velocity, _ = geometry(velocity, h, source, radius_cells)
    entry = {"case": name, "shape_cells": list(velocity.shape),
             "nodes": int(np.prod(np.asarray(velocity.shape) + 1)), "h_m": h,
             "source_m": list(source), "radius_m": radius_cells * h,
             "radius_cells": radius_cells, "seed_cell_count": int(inside.sum()), "solvers": {}}
    fields = {}
    subsets = {"all": None, "outside_seed": ~inside, "far_gt_2R": distance > 2 * radius_cells * h}
    subsets.update(masks or {})
    for solver in ("factored", "corrected_skfmm_order2"):
        field, info = solve_one(solver, velocity, h, source, radius_cells)
        entry["solvers"][solver] = info
        if field is None:
            continue
        fields[solver] = field
        info["inside_analytic"] = error_stats(field - distance / station_velocity, inside)
        if reference is not None:
            info["errors"] = {key: error_stats(field - reference, mask) for key, mask in subsets.items()}
            info["float32_storage_errors"] = error_stats(field.astype(np.float32).astype(float) - reference)
    if len(fields) == 2:
        entry["factored_minus_baseline"] = error_stats(fields["factored"] - fields["corrected_skfmm_order2"])
    brief = []
    for solver, info in entry["solvers"].items():
        if "error" in info:
            brief.append(f"{solver}: ERROR {info['error']}")
        else:
            suffix = "" if reference is None else f", rms={info['errors']['all']['rms_s']:.6g}s max={info['errors']['all']['max_abs_s']:.6g}s"
            brief.append(f"{solver}: {info['wall_seconds']:.3f}s{suffix}")
    print(name + " | " + " | ".join(brief), flush=True)
    return entry, fields


def heterogeneous(shape, h):
    x, y, z = np.meshgrid(*[(np.arange(n) + 0.5) * h for n in shape], indexing="ij", sparse=True)
    return np.asarray(3100.0 + 1500.0 * (z >= 1000.0)
                      + 230.0 * np.sin(x / 480.0) * np.cos(y / 370.0)
                      + 170.0 * np.sin(z / 310.0) + 40.0 * np.cos((x + y + z) / 230.0))


def homogeneous_screen():
    h, shape = 125.0, (24, 18, 14)
    extent = np.asarray(shape) * h
    cases = [("surface_top_center", shape, tuple(extent * (0.5, 0.5, 0))),
             ("first_cell_center", shape, (1562.5, 1187.5, 62.5)),
             ("fractional_interior", shape, (1234.5, 1045.25, 384.75)),
             ("upper_x_face", shape, (extent[0], 833.125, 514.75)),
             ("source_at_output_center", shape, (1437.5, 937.5, 812.5)),
             ("whole_domain_seed", (2, 2, 2), (0.0, 0.0, 0.0)),
             ("thin_domain", (1, 1, 14), (62.5, 62.5, 0.0))]
    for corner in np.ndindex(2, 2, 2):
        cases.append(("corner_" + "".join(map(str, corner)), shape, tuple(extent * corner)))
    rows = []
    for name, dimensions, source in cases:
        velocity = np.full(dimensions, 4000.0)
        distance, _, _, _ = geometry(velocity, h, source, 3.0)
        row, _ = pair(name, velocity, h, source, reference=distance / 4000.0)
        rows.append(row)
    return {"reference": "exact Euclidean source distance / 4000 m/s", "cases": rows}


def invariant_screen():
    h, shape, source = 125.0, (20, 14, 16), (912.25, 831.75, 0.0)
    original = heterogeneous(shape, h)
    distance, inside, station_velocity, station_cell = geometry(original, h, source, 3.0)
    storage = np.repeat(original, 2, axis=0)
    readonly = storage[::2]
    readonly.flags.writeable = False
    entry, reference = pair("readonly_noncontiguous", readonly, h, source)
    checks = {"readonly_noncontiguous_input_unchanged": bool(np.array_equal(storage[::2], original)),
              "field_does_not_alias_input": all(not np.shares_memory(field, readonly) for field in reference.values())}
    rows = [entry]
    for speed in (1000.0, 7000.0):
        changed = original.copy()
        changed[inside] = speed
        changed[station_cell] = station_velocity
        row, fields = pair(f"nonstation_inside_changed_to_{speed:g}", changed, h, source)
        row["outside_input_bitwise_unchanged"] = bool(np.array_equal(changed[~inside], original[~inside]))
        row["against_original"] = {solver: {"bitwise_equal": bool(np.array_equal(field, reference[solver])),
                                                  "difference": error_stats(field - reference[solver])}
                                   for solver, field in fields.items() if solver in reference}
        rows.append(row)
    explicit = original.copy()
    explicit[inside] = station_velocity
    row, explicit_fields = pair("explicitly_homogenized_equivalence", explicit, h, source)
    row["against_original"] = {solver: {"bitwise_equal": bool(np.array_equal(field, reference[solver])),
                                              "difference": error_stats(field - reference[solver])}
                               for solver, field in explicit_fields.items() if solver in reference}
    rows.append(row)
    outside = original.copy()
    outside[(distance > 750.0) & (distance < 1250.0)] *= 1.15
    row, outside_fields = pair("outside_heterogeneity_must_matter", outside, h, source)
    row["against_original_far"] = {solver: error_stats(field - reference[solver], distance > 1000.0)
                                    for solver, field in outside_fields.items() if solver in reference}
    rows.append(row)

    # Audit actual internal nodal seed before the public cell-centre overwrite.
    slow = 1.0 / original
    slow[inside] = 1.0 / station_velocity
    nodes = CANDIDATE._node_slowness(slow)
    independent_nodes = np.empty(nodes.shape)
    for index in np.ndindex(nodes.shape):
        neighbors = tuple(slice(max(0, i - 1), min(n, i + 1)) for i, n in zip(index, shape))
        independent_nodes[index] = np.mean(slow[neighbors])
    offsets = [np.arange(n + 1) * h - s for n, s in zip(shape, source)]
    radius = np.sqrt(offsets[0][:, None, None] ** 2 + offsets[1][None, :, None] ** 2
                     + offsets[2][None, None, :] ** 2)
    known = radius <= 3.0 * h
    factor = np.full(nodes.shape, max(float(nodes.max()), 1.0 / station_velocity))
    factor[known] = 1.0 / station_velocity
    frozen = factor[known].copy()
    cycles, change = CANDIDATE._solve(factor, nodes, radius, known, *offsets, h, TOLERANCE, MAX_CYCLES)
    centre_factor = sum(factor[a:a + shape[0], b:b + shape[1], c:c + shape[2]] / 8.0
                        for a in (0, 1) for b in (0, 1) for c in (0, 1))
    reconstructed = distance * centre_factor
    reconstructed[inside] = distance[inside] * (1.0 / station_velocity)
    checks.update({"homogenized_fine_slowness_outside_exact": bool(np.array_equal(slow[~inside], (1.0 / original)[~inside])),
                   "homogenized_fine_slowness_inside_exact": bool(np.all(slow[inside] == 1.0 / station_velocity)),
                   "node_average_max_abs_difference_s_per_m": float(np.max(np.abs(nodes - independent_nodes))),
                   "internal_nodal_seed_bitwise_unchanged": bool(np.array_equal(factor[known], frozen)),
                   "reconstructed_pipeline_equals_public": bool("factored" in reference and np.array_equal(reconstructed, reference["factored"])),
                   "internal_factor_min": float(factor.min()), "internal_factor_max": float(factor.max()),
                   "effective_slowness_min": float(min(nodes.min(), 1.0 / station_velocity)),
                   "effective_slowness_max": float(max(nodes.max(), 1.0 / station_velocity)),
                   "audit_cycles": int(cycles), "audit_change": float(change)})
    rejected = []
    for invalid in ((-1.0, 0.0, 0.0), (shape[0] * h + 1.0, 0.0, 0.0), (np.nan, 0.0, 0.0)):
        try:
            CANDIDATE.travel_time_field(original, h, invalid)
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    checks["outside_or_nonfinite_sources_rejected"] = rejected
    default = CANDIDATE.travel_time_field(original, h, source)
    checks["default_float32_contract"] = bool(default.dtype == np.float32 and default.shape == shape
                                                and "factored" in reference
                                                and np.array_equal(default, reference["factored"].astype(np.float32)))
    checks["internal_factor_within_effective_slowness_bounds"] = bool(
        factor.min() >= checks["effective_slowness_min"] * (1 - 1e-12)
        and factor.max() <= checks["effective_slowness_max"] * (1 + 1e-12))
    checks["node_average_matches_independent_valid_neighbors"] = checks["node_average_max_abs_difference_s_per_m"] <= 1e-18
    print("invariants:", json.dumps(checks), flush=True)
    return {"checks": checks, "cases": rows}


def symmetry_screen():
    h, shape, source = 125.0, (24, 16, 20), np.array((912.25, 831.75, 0.0))
    velocity = heterogeneous(shape, h)
    base, fields = pair("symmetry_reference", velocity, h, source)
    rows = []
    for axis in range(3):
        reflected_source = source.copy()
        reflected_source[axis] = shape[axis] * h - source[axis]
        row, changed = pair(f"reflect_axis_{axis}", np.flip(velocity, axis), h, reflected_source)
        row["untransformed_difference"] = {solver: error_stats(np.flip(field, axis) - fields[solver])
                                            for solver, field in changed.items() if solver in fields}
        rows.append(row)
    for permutation in ((1, 0, 2), (2, 1, 0), (1, 2, 0)):
        row, changed = pair("permute_" + "".join(map(str, permutation)), np.transpose(velocity, permutation),
                            h, source[list(permutation)])
        row["untransformed_difference"] = {solver: error_stats(np.transpose(field, np.argsort(permutation)) - fields[solver])
                                            for solver, field in changed.items() if solver in fields}
        rows.append(row)
    return {"reference": base, "cases": rows,
            "scope": "exact grid reflections/permutations, not arbitrary rotations; source avoids ambiguous interior cell faces"}


def step_ulps(values, count, direction):
    result = values.copy()
    target = np.inf if direction > 0 else -np.inf
    for _ in range(count):
        result = np.nextafter(result, target)
    return result


def ulp_screen():
    h, shape, source = 125.0, (40, 20, 24), (2412.25, 1231.75, 0.0)
    velocity = heterogeneous(shape, h)
    distance, inside, _, station_cell = geometry(velocity, h, source, 3.0)
    block = np.zeros(shape, dtype=bool)
    block[10:25, 5:15, 2:13] = True
    block &= ~inside
    base, fields = pair("heterogeneous_ulp_reference", velocity, h, source)
    perturbations = []
    for count in (1, 8, 64):
        for direction in (-1, 1):
            changed = velocity.copy()
            changed[block] = step_ulps(velocity[block], count, direction)
            perturbations.append((f"block_{direction:+d}x{count}_ulp", changed))
    rng = np.random.default_rng(20261017)
    signs = rng.choice((-1, 1), size=shape)
    for count in (1, 64):
        changed = velocity.copy()
        for direction in (-1, 1):
            mask = (~inside) & (signs == direction)
            changed[mask] = step_ulps(velocity[mask], count, direction)
        perturbations.append((f"random_signed_{count}_ulp_outside", changed))
    for direction in (-1, 1):
        changed = velocity.copy()
        changed[station_cell] = step_ulps(np.asarray(velocity[station_cell]), 64, direction)
        perturbations.append((f"station_{direction:+d}x64_ulp", changed))
    for delta in (1e-6, 1e-3):
        changed = velocity.copy()
        changed[block] += delta
        perturbations.append((f"block_plus_{delta:g}_m_s", changed))
    rows = []
    for name, changed in perturbations:
        row, perturbed = pair(name, changed, h, source)
        relative = float(np.max(np.abs(changed - velocity) / velocity))
        row["max_velocity_change_m_s"] = float(np.max(np.abs(changed - velocity)))
        row["changed_cells"] = int(np.count_nonzero(changed != velocity))
        row["comparison_relative_input_bound"] = relative
        row["changes"] = {}
        for solver, field in perturbed.items():
            if solver not in fields:
                continue
            delta = field - fields[solver]
            bound = float(fields[solver].max()) * relative / (1.0 - relative)
            stats = error_stats(delta)
            row["changes"][solver] = {"float64": stats,
                "float32": error_stats(field.astype(np.float32).astype(float) - fields[solver].astype(np.float32).astype(float)),
                "max_output_ulps": float(np.max(np.abs(delta) / np.spacing(fields[solver]))),
                "relative_comparison_bound_s": bound,
                "exceeds_comparison_bound_plus_1e-12_s": stats["max_abs_s"] > bound + 1e-12}
        rows.append(row)
    repeated, repeat_info = solve_one("factored", velocity, h, source, 3.0)
    return {"reference": base, "cases": rows, "random_seed": 20261017,
            "candidate_repeat_bitwise_equal": bool(repeated is not None and "factored" in fields and np.array_equal(repeated, fields["factored"])),
            "repeat_diagnostics": repeat_info,
            "comparison_note": "Fixed geometry/mask: positive homogeneous monotone discrete comparison predicts relative slowness stability; 1e-12 s is an explicit rounding allowance, not a physical accuracy tolerance."}


def two_layer_reference(shape, h, source, depth=750.0, upper=2500.0, lower=5500.0):
    """Exact first arrivals for source in upper layer, including critical head waves."""
    x, y, z = np.meshgrid(*[(np.arange(n) + 0.5) * h for n in shape], indexing="ij", sparse=True)
    rho = np.broadcast_to(np.sqrt((x - source[0]) ** 2 + (y - source[1]) ** 2), shape)
    z = np.broadcast_to(z, shape)
    exact = np.sqrt(rho * rho + (z - source[2]) ** 2) / upper
    branch = np.zeros(shape, dtype=np.int8)
    above = z < depth
    critical_tangent = upper / np.sqrt(lower * lower - upper * upper)
    vertical_sum = 2 * depth - source[2] - z
    head = rho / lower + vertical_sum * np.sqrt(1 / upper ** 2 - 1 / lower ** 2)
    valid_head = above & (rho >= vertical_sum * critical_tangent) & (head < exact)
    exact[valid_head] = head[valid_head]
    branch[valid_head] = 2
    below = ~above
    horizontal = rho[below]
    d1, d2 = depth - source[2], z[below] - depth
    lo = np.zeros_like(horizontal)
    hi = np.full_like(horizontal, np.nextafter(1.0 / lower, 0.0))
    for _ in range(64):
        p = (lo + hi) / 2
        lateral = d1 * p * upper / np.sqrt(1 - (p * upper) ** 2) + d2 * p * lower / np.sqrt(1 - (p * lower) ** 2)
        less = lateral < horizontal
        lo = np.where(less, p, lo)
        hi = np.where(less, hi, p)
    p = (lo + hi) / 2
    c1, c2 = np.sqrt(1 - (p * upper) ** 2), np.sqrt(1 - (p * lower) ** 2)
    exact[below] = d1 / (upper * c1) + d2 / (lower * c2)
    branch[below] = 1
    mismatch = d1 * p * upper / c1 + d2 * p * lower / c2 - horizontal
    return exact, branch, {"max_snell_horizontal_closure_m": float(np.max(np.abs(mismatch))),
                           "critical_angle_deg": float(np.degrees(np.arcsin(upper / lower))),
                           "branches_counts": {"direct": int(np.sum(branch == 0)), "transmitted": int(np.sum(branch == 1)),
                                               "head_wave": int(np.sum(branch == 2))}}


def boundary_cells(shape):
    mask = np.zeros(shape, dtype=bool)
    for axis in range(3):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = 0, -1
        mask[tuple(low)] = True
        mask[tuple(high)] = True
    return mask


def refinement_rates(rows, key="all"):
    result = {}
    for radius_policy in sorted({row["radius_policy"] for row in rows}):
        selected = sorted((row for row in rows if row["radius_policy"] == radius_policy), key=lambda row: -row["h_m"])
        result[radius_policy] = {}
        for solver in ("factored", "corrected_skfmm_order2"):
            data = []
            for row in selected:
                info = row["solvers"][solver]
                if "errors" not in info or not info["errors"][key].get("count"):
                    continue
                stats = info["errors"][key]
                item = {"h_m": row["h_m"], "rms_s": stats["rms_s"], "max_abs_s": stats["max_abs_s"]}
                if data and item["rms_s"] > 0:
                    item["rms_order_from_previous"] = float(np.log(data[-1]["rms_s"] / item["rms_s"]) / np.log(data[-1]["h_m"] / item["h_m"]))
                data.append(item)
            result[radius_policy][solver] = data
    return result


def layered_screen():
    extent, source = (6000.0, 1500.0, 2000.0), (3000.0, 750.0, 0.0)
    rows = []
    for h in (125.0, 62.5, 31.25):
        shape = tuple(int(e / h) for e in extent)
        z = (np.arange(shape[2]) + 0.5) * h
        velocity = np.broadcast_to(np.where(z < 750.0, 2500.0, 5500.0), shape).copy()
        exact, branch, analytic = two_layer_reference(shape, h, source)
        rho = np.sqrt(((np.arange(shape[0]) + 0.5) * h - source[0])[:, None, None] ** 2
                      + ((np.arange(shape[1]) + 0.5) * h - source[1])[None, :, None] ** 2)
        masks = {"upper_direct": branch == 0, "transmitted": branch == 1, "head_wave": branch == 2,
                 "oblique_transmitted": (branch == 1) & (rho > 1000.0),
                 "near_interface_2h": np.broadcast_to(np.abs(z - 750.0) <= 2 * h, shape),
                 "boundary_cells": boundary_cells(shape)}
        policies = (("fixed_R375", 375.0 / h), ("shrinking_R3h", 3.0))
        for policy, radius_cells in policies:
            if h == 125.0 and policy == "shrinking_R3h":
                duplicate = json.loads(json.dumps(rows[-1]))
                duplicate["case"] += "_shared_with_R3h"
                duplicate["radius_policy"] = policy
                duplicate["shared_identical_solve_not_retimed"] = True
                rows.append(duplicate)
                continue
            row, fields = pair(f"layers_h{h:g}_{policy}", velocity, h, source, radius_cells, exact, masks)
            row.update({"radius_policy": policy, "analytic_checks": analytic})
            if policy == "fixed_R375":
                fixed_fields = fields
            else:
                row["same_grid_shrinking_minus_fixed_radius"] = {
                    solver: error_stats(field - fixed_fields[solver])
                    for solver, field in fields.items() if solver in fixed_fields}
            rows.append(row)
    h, shape, source = 125.0, (48, 12, 16), (3062.5, 812.5, 62.5)
    z = (np.arange(shape[2]) + 0.5) * h
    velocity = np.broadcast_to(np.where(z < 750.0, 2500.0, 5500.0), shape).copy()
    exact, branch, analytic = two_layer_reference(shape, h, source)
    first_cell, _ = pair("layers_source_first_cell_center", velocity, h, source, reference=exact,
                         masks={"head_wave": branch == 2, "transmitted": branch == 1})
    first_cell["analytic_checks"] = analytic
    return {"reference": "Two planar layers z<750:2500 m/s, z>=750:5500 m/s; direct, Snell transmitted, critical head-wave minimum. Interfaces at voxel faces; source sphere entirely upper medium.",
            "fixed_physical_domain_m": list(extent), "fixed_source_m": [3000.0, 750.0, 0.0],
            "cases": rows, "refinement": refinement_rates(rows), "first_cell_source": first_cell,
            "caveat": "Different grids sample different receiver centres; full-volume norms approximate continuum norms, not pointwise nested-grid errors. Nodal slowness averaging smears the interface; seed masks also change."}


def boundary_screen():
    extent, source, radius = (3000.0, 2000.0, 2000.0), (1500.0, 1000.0, 0.0), 375.0
    rows = []
    for h in (125.0, 62.5, 31.25):
        shape = tuple(int(e / h) for e in extent)
        velocity = np.full(shape, 6000.0)
        distance, inside, _, _ = geometry(velocity, h, source, radius / h)
        velocity[inside] = 1000.0
        exact = np.minimum(distance, radius) / 1000.0 + np.maximum(distance - radius, 0.0) / 6000.0
        top = np.zeros(shape, dtype=bool)
        top[:, :, 0] = True
        row, fields = pair(f"boundary_slow_sphere_h{h:g}", velocity, h, source, radius / h, exact,
                           {"top_row_outside": top & ~inside, "outside_gt_Rplus2h": distance > radius + 2 * h,
                            "far_gt_Rplus375m": distance > radius + 375.0, "boundary_cells": boundary_cells(shape)})
        row["radius_policy"] = "fixed_R375"
        scale = h * (1 / 1000.0 - 1 / 6000.0)
        row["one_h_interface_time_scale_s"] = scale
        row["gross_early_screen_threshold_s"] = 2 * scale
        row["gross_early_threshold_note"] = "Diagnostic 2h slowness-contrast allowance; NOT a rigorous discretization error bound. Also test the tighter analytic voxel-medium envelope below."
        delta_radius = np.sqrt(3.0) * h / 2
        lower_radius, upper_radius = max(0.0, radius - delta_radius), radius + delta_radius
        lower_time = np.minimum(distance, lower_radius) / 1000.0 + np.maximum(distance - lower_radius, 0.0) / 6000.0
        upper_time = np.minimum(distance, upper_radius) / 1000.0 + np.maximum(distance - upper_radius, 0.0) / 6000.0
        row["voxelization_envelope"] = {
            "half_cell_diagonal_m": float(delta_radius), "radial_time_allowance_s": float(delta_radius * (1 / 1000.0 - 1 / 6000.0)),
            "explanation": "Centre-labelled voxel medium contains the slow ball R-halfdiag and lies within slow ball R+halfdiag. Their exact radial times bound continuum travel in that voxel medium. This is not a theorem for the nodal averaged finite-difference operator."}
        for solver, field in fields.items():
            info = row["solvers"][solver]
            info["worst_early_error_in_h_contrast_units"] = max(0.0, -info["errors"]["outside_seed"]["min_signed_s"]) / scale
            info["gross_early_beyond_2h_contrast_scale"] = info["errors"]["outside_seed"]["min_signed_s"] < -2 * scale
            info["worst_violation_mandatory_seed_exit_s"] = max(0.0, float(np.max(radius / 1000.0 - field[~inside])))
            info["early_beyond_voxel_continuum_envelope_s"] = max(0.0, float(np.max(lower_time - field)))
            info["late_beyond_voxel_continuum_envelope_s"] = max(0.0, float(np.max(field - upper_time)))
        rows.append(row)
    return {"reference": "Exact radial continuum T=min(r,R)/1000 + max(r-R,0)/6000; source on physical top boundary, slow hemisphere R=375 m.",
            "fixed_physical_domain_m": list(extent), "source_m": list(source), "cases": rows,
            "refinement": refinement_rates(rows, "outside_gt_Rplus2h"),
            "caveat": "The cell-centre sphere is voxelized; adjacent-cell nodal slowness averages smear its boundary, while the analytic nodal seed uses an exact sphere. Errors combine all three effects; threshold is a gross-leakage screen only."}


def summarize(results):
    findings = []
    for section in SECTIONS:
        content = results.get(section)
        if not content:
            continue
        if "section_error" in content:
            findings.append({"severity": "failure", "section": section, "message": content["section_error"]})
        for row in content.get("cases", []):
            for solver, info in row.get("solvers", {}).items():
                if "error" in info:
                    findings.append({"severity": "failure", "section": section, "case": row["case"], "solver": solver, "message": info["error"]})
    for row in results.get("layered", {}).get("cases", []):
        a, b = (row["solvers"][name] for name in ("factored", "corrected_skfmm_order2"))
        if "errors" in a and "errors" in b:
            ratio = a["errors"]["all"]["rms_s"] / b["errors"]["all"]["rms_s"]
            findings.append({"severity": "accuracy_comparison", "section": "layered", "case": row["case"],
                             "candidate_over_baseline_rms": ratio, "candidate_worse": ratio > 1})
    for row in results.get("boundary", {}).get("cases", []):
        info = row["solvers"]["factored"]
        if "errors" in info:
            baseline = row["solvers"]["corrected_skfmm_order2"]
            ratio = (info["errors"]["all"]["rms_s"] / baseline["errors"]["all"]["rms_s"]
                     if "errors" in baseline else None)
            findings.append({"severity": "boundary_diagnostic", "case": row["case"],
                             "gross_early_flag": info["gross_early_beyond_2h_contrast_scale"],
                             "early_beyond_voxel_continuum_envelope_s": info["early_beyond_voxel_continuum_envelope_s"],
                             "candidate_over_baseline_rms": ratio,
                             "min_error_s": info["errors"]["outside_seed"]["min_signed_s"],
                             "max_error_s": info["errors"]["outside_seed"]["max_signed_s"]})
    for row in results.get("ulp", {}).get("cases", []):
        for solver, info in row.get("changes", {}).items():
            if info["exceeds_comparison_bound_plus_1e-12_s"]:
                findings.append({"severity": "continuity_warning", "case": row["case"], "solver": solver,
                                 "max_abs_change_s": info["float64"]["max_abs_s"]})
    for row in results.get("symmetry", {}).get("cases", []):
        for solver, stats in row.get("untransformed_difference", {}).items():
            if stats["max_abs_s"] > 1e-9:
                findings.append({"severity": "symmetry_warning", "case": row["case"], "solver": solver,
                                 "max_abs_change_s": stats["max_abs_s"]})
    return findings


def quantitative_summary(results):
    rows = []
    for section in SECTIONS:
        content = results.get(section, {})
        rows.extend(row for row in content.get("cases", []) if not row.get("shared_identical_solve_not_retimed"))
        for key in ("reference", "first_cell_source"):
            if isinstance(content.get(key), dict) and "solvers" in content[key]:
                rows.append(content[key])
    candidate_infos = [row["solvers"]["factored"] for row in rows if "error" not in row["solvers"]["factored"]]
    baseline_infos = [row["solvers"]["corrected_skfmm_order2"] for row in rows
                      if "error" not in row["solvers"]["corrected_skfmm_order2"]]
    summary = {"paired_cases": len(rows), "successful_candidate_fields": len(candidate_infos),
               "successful_baseline_fields": len(baseline_infos),
               "largest_grid_nodes": max((row["nodes"] for row in rows), default=0),
               "all_successful_baselines_frozen_float32_verified": all(info["frozen_float32_bitwise_verified"] for info in baseline_infos),
               "candidate_cycles_min": min((info["cycles_of_8_sweeps"] for info in candidate_infos), default=None),
               "candidate_cycles_max": max((info["cycles_of_8_sweeps"] for info in candidate_infos), default=None),
               "candidate_max_abs_relative_hamiltonian_residual": max(
                   (max(abs(info["relative_hamiltonian_residual"]["min"]), abs(info["relative_hamiltonian_residual"]["max"]))
                    for info in candidate_infos), default=None),
               "timings_note": "Candidate wall_seconds excludes JIT warmup. Baseline wall_seconds includes a second frozen float32 verification solve; use native_float64_seconds for single-solve comparisons."}
    homogeneous = results.get("homogeneous", {}).get("cases", [])
    summary["homogeneous_worst_candidate_abs_error_s"] = max(
        (row["solvers"]["factored"]["errors"]["all"]["max_abs_s"] for row in homogeneous
         if "errors" in row["solvers"]["factored"]), default=None)
    summary["symmetry_worst_abs_s"] = {
        solver: max((row["untransformed_difference"][solver]["max_abs_s"]
                     for row in results.get("symmetry", {}).get("cases", [])
                     if solver in row.get("untransformed_difference", {})), default=None)
        for solver in ("factored", "corrected_skfmm_order2")}
    summary["heterogeneous_ulp_worst_abs_change_s"] = {
        solver: max((row["changes"][solver]["float64"]["max_abs_s"]
                     for row in results.get("ulp", {}).get("cases", [])
                     if "ulp" in row["case"] and solver in row.get("changes", {})), default=None)
        for solver in ("factored", "corrected_skfmm_order2")}
    checks = results.get("invariants", {}).get("checks", {})
    flags = [value for value in checks.values() if isinstance(value, bool)]
    flags.extend(checks.get("outside_or_nonfinite_sources_rejected", []))
    flags.extend(comparison["bitwise_equal"] for row in results.get("invariants", {}).get("cases", [])
                 for comparison in row.get("against_original", {}).values())
    summary["all_recorded_invariant_flags_true"] = bool(flags) and all(flags)
    summary["execution_failures"] = [finding for finding in results["findings"] if finding["severity"] == "failure"]
    summary["interpretation"] = "Continuity, homogeneous exactness and iterative residual do NOT establish heterogeneous physical accuracy. Layered subregions and radial stress must be judged separately; no solver adjustment was made."
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    protected = [HERE / name for name in ("factored_candidate.py", "flat4_controls.py")]
    protected += [HERE.parents[1] / name for name in ("eikonal.py", "tomography/tomography_em.py")]
    before = {str(path.relative_to(HERE.parents[1])): digest(path) for path in protected}
    results = {} if args.reset or not OUTPUT.exists() else json.loads(OUTPUT.read_text())
    if results and (results["provenance"]["protected_sha256_before"] != before
                    or results["provenance"]["screen_sha256"] != digest(Path(__file__))):
        raise RuntimeError("screen/protected source hashes changed since earlier batch; rerun with --reset rather than mix versions")
    results.setdefault("scope", "Independent forward-only screen. Only factored_screen.py and factored_screen.json written; no production/solver/test edits, inverse, prior, dependencies, model files, or stopping changes.")
    results.setdefault("provenance", {"created_utc": datetime.now(timezone.utc).isoformat(),
                       "protected_sha256_before": before, "screen_sha256": digest(Path(__file__)),
                       "baseline": "AST-extracted unmodified flat4_controls.homogenized_field/_frozen_field; native float64 formula cast checked bitwise against frozen float32 on EVERY solve",
                       "candidate_loading": f"Only {DISABLED_CACHES} njit cache=True flags disabled in memory; arithmetic, iteration and tolerances unchanged",
                       "python": sys.version, "platform": platform.platform(),
                       "versions": {"numpy": np.__version__, "numba": numba.__version__, "skfmm": skfmm.__version__},
                       "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                       "candidate_tolerance": TOLERANCE, "candidate_max_cycles": MAX_CYCLES,
                       "node_budget": MAX_NODES, "error_sign": "numerical minus analytic reference; no demeaning or fitted offset"})
    warm_started = perf_counter()
    CANDIDATE.travel_time_field(np.full((4, 4, 4), 4000.0), 125.0, (0.0, 0.0, 0.0), dtype=np.float64)
    results.setdefault("batches", []).append({"started_utc": datetime.now(timezone.utc).isoformat(),
                                              "sections": args.sections, "jit_warmup_seconds": perf_counter() - warm_started})
    runners = {"homogeneous": homogeneous_screen, "invariants": invariant_screen, "symmetry": symmetry_screen,
               "ulp": ulp_screen, "layered": layered_screen, "boundary": boundary_screen}
    for section in args.sections:
        started = perf_counter()
        try:
            results[section] = runners[section]()
        except Exception as exc:
            results[section] = {"section_error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            print(section + ": ERROR " + results[section]["section_error"], flush=True)
        results[section]["section_wall_seconds"] = perf_counter() - started
        results["provenance"]["protected_sha256_after"] = {str(path.relative_to(HERE.parents[1])): digest(path) for path in protected}
        results["provenance"]["protected_files_unchanged"] = results["provenance"]["protected_sha256_after"] == before
        results["findings"] = summarize(results)
        results["summary"] = quantitative_summary(results)
        results["completed_sections"] = [name for name in SECTIONS if name in results]
        OUTPUT.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(f"Saved {section}: {results[section]['section_wall_seconds']:.3f}s -> {OUTPUT.name}", flush=True)
    print("Summary:", json.dumps(results["summary"], indent=2), flush=True)
    print("Findings:", json.dumps(results["findings"], indent=2), flush=True)


if __name__ == "__main__":
    main()
