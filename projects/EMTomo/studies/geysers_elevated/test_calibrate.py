"""Small production-wiring tests: no real fields, forward modeling, or updates."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projects.EMTomo.studies.geysers_elevated import calibrate as calibration


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "projects" / "EMTomo"))
    import eikonal
    import experiment_runner
    from instruments import likelihood
    from tomography import tomography_em, tomography_events

    def forbidden(*args, **kwargs):
        pytest.fail("Calibration must not run real forward, EM, rays, or updates")

    for module, names in (
        (eikonal, ("travel_time_field",)),
        (tomography_em, ("make_tomography_step", "run_em", "run_events", "solve_slowness_update", "warm_up_jit")),
        (tomography_events, ("process_event", "compute_G_all_stations", "compute_G_all_stations_serial",
                             "accumulate_normal_equations")),
        (experiment_runner, ("prepare_inversion",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    # In particular, calibration must use eikonal's API, not em's imported alias.
    monkeypatch.setattr(tomography_em, "station_travel_time_fields", forbidden)
    calls = SimpleNamespace(refined=[], fields=[], noise=[], nll=[], normalization=[], refine=[], select=[])
    fine = SimpleNamespace(cell_size=100.0)
    velocity = np.arange(16.0).reshape(16, 1, 1) + 4000.0

    class Initial:
        def __init__(self):
            self.velocity = velocity

        def refined(self, subdivision, interpolation):
            calls.refined.append((subdivision, interpolation))
            return fine

    class Prepared:
        @property
        def reference_model(self):
            raise AssertionError("reference model is oracle information")

        @property
        def reference_events(self):
            raise AssertionError("reference events are oracle information")

        @property
        def event_ids(self):
            raise AssertionError("sampling must use observation rows")

    prepared = Prepared()
    prepared.initial_model = Initial()
    prepared.station_locs = np.array([[0., 0., 0.], [100., 0., 0.]])
    prepared.arrivals_table = np.arange(8., dtype=float).reshape(4, 2)
    prepared.noise_sigmas = (0.1, 0.01)
    config = SimpleNamespace(subdivision=2, slowness_interpolation="trilinear", n_workers=3,
                             weights_min_distance=1, weight_noise_relative_sigma=None,
                             weight_noise_absolute_sigma_s=0.025, weight_model_sigma_s=0.2,
                             n_candidates=5, weights_top_n=1, temperature=0.5, candidate_mode="hard")
    times = np.zeros((2, 16, 1, 1), dtype=np.float32)
    grid = np.arange(16., dtype=float).reshape(16, 1, 1)
    # Refinement reverses grid ranking; smaller K must not truncate sorted K15.
    refined_costs = np.array([20., 18., 18., 16., 14., 12., 10., 8., 6., 4., 3., 2., 1., 0.5, 0., -1.])
    marker = object()

    def fields(model, stations, *, n_workers):
        assert model is fine and stations is prepared.station_locs
        calls.fields.append(n_workers)
        return times

    def resolve(cfg, sigmas):
        assert cfg is config and sigmas is prepared.noise_sigmas
        noise = experiment_runner.resolve_pick_noise(cfg, sigmas)
        calls.noise.append(noise)
        return noise

    def normalization(actual, noise):
        assert actual is times and noise is calls.noise[-1]
        calls.normalization.append(True)
        return marker

    def nll(actual, observed, noise, norm):
        assert actual is times and noise is calls.noise[-1] and norm is marker
        calls.nll.append(observed.copy())
        return grid

    select_native = tomography_events.select_candidate_cells

    def select(cost, k, distance):
        assert cost is grid and distance == config.weights_min_distance
        calls.select.append(k)
        return select_native(cost, k, distance)

    def refine(actual, observed, cell, noise):
        assert actual is times and noise is calls.noise[-1]
        calls.refine.append((observed.copy(), cell.copy()))
        return cell.astype(float) + 0.1, float(refined_costs[cell[0]])

    monkeypatch.setattr(eikonal, "station_travel_time_fields", fields)
    monkeypatch.setattr(likelihood, "likelihood_normalization", normalization)
    monkeypatch.setattr(likelihood, "negative_log_likelihood", nll)
    monkeypatch.setattr(tomography_events, "select_candidate_cells", select)
    monkeypatch.setattr(tomography_events, "refine_hypocentre_in_cell", refine)
    runtime = SimpleNamespace(np=np, em=tomography_em, resolve_pick_noise=resolve,
                              prepare_inversion=forbidden)
    return SimpleNamespace(prepared=prepared, config=config, runtime=runtime, calls=calls, fine=fine,
                           costs=refined_costs, grid=grid, times=times, events=tomography_events,
                           likelihood=likelihood, select_native=select_native)


def test_production_wiring_native_k_costs_ledger_and_no_mutation(wired, tmp_path):
    w = wired
    before = (w.prepared.initial_model.velocity.tobytes(), w.prepared.arrivals_table.tobytes(),
              w.prepared.station_locs.tobytes(), copy.deepcopy(vars(w.config)))
    result = calibration.calibrate(w.prepared, w.config, w.runtime, tmp_path)
    assert result == json.loads((tmp_path / "calibration.json").read_text())
    json.dumps(result, allow_nan=False)
    assert list(tmp_path.iterdir()) == [tmp_path / "calibration.json"]
    assert w.calls.refined == [(2, "trilinear")]
    assert w.calls.fields == [3] and len(w.calls.normalization) == 1
    assert len(w.calls.noise) == 1
    assert w.calls.noise[0].relative_sigma == 0.1
    assert w.calls.noise[0].absolute_sigma_s == 0.025
    assert len(w.calls.nll) == 4 and len(w.calls.refine) == 4 * 15
    assert w.calls.select == [15, 1, 5, 10] * 4
    assert result["event_indices"] == [0, 1, 2, 3]
    assert result["shortlist_prefix_verified"] is True
    for event in result["events"]:
        for k in (1, 5, 10, 15):
            detail = event["per_k"][str(k)]
            native = w.select_native(w.grid, k, 1)
            native_costs = [w.costs[cell[0]] for cell in native]
            order = np.argsort(native_costs, kind="stable")
            assert detail["costs"] == np.asarray(native_costs)[order].tolist()
            assert detail["ranked_cells"] == native[order].tolist()
            assert detail["best_cost"] == min(native_costs)
            assert detail["retained_count"] == k
        assert event["per_k"]["5"]["best_cost"] == 14.0
        assert event["per_k"]["15"]["best_cost"] == 0.0
    assert result["cases"] == [
        {"candidate_mode": mode, "n_candidates": k, "weights_top_n": 1 if mode == "hard" else k,
         "temperature": 1.0 if mode == "hard" else result["temperature"]}
        for mode, ks in (("hard", (1, 5, 10, 15)), ("soft", (5, 10, 15))) for k in ks]
    assert before == (w.prepared.initial_model.velocity.tobytes(), w.prepared.arrivals_table.tobytes(),
                      w.prepared.station_locs.tobytes(), vars(w.config))


def test_evenly_spaced_observation_indices_and_determinism(wired, tmp_path):
    w = wired
    w.prepared.arrivals_table = np.arange(3776 * 2., dtype=float).reshape(3776, 2)
    first = calibration.calibrate(w.prepared, w.config, w.runtime, tmp_path / "first")
    second = calibration.calibrate(w.prepared, w.config, w.runtime, tmp_path / "second")
    assert first == second
    assert first["sample_count"] == 128
    indices = first["event_indices"]
    assert indices == np.linspace(0, 3775, 128, dtype=int).tolist()
    assert len(set(indices)) == 128 and indices[0] == 0 and indices[-1] == 3775
    assert max(np.diff(indices)) - min(np.diff(indices)) <= 1
    assert [observed[0] for observed in w.calls.nll[:128]] == [2 * i for i in indices]


def test_nonprefix_native_selection_refines_extra_cell_not_wrong_prefix(wired, monkeypatch, tmp_path):
    w = wired

    def nonprefix(cost, k, distance):
        cells = w.select_native(cost, k, distance)
        if k == 5:
            cells[-1] = [15, 0, 0]
        return cells

    monkeypatch.setattr(w.events, "select_candidate_cells", nonprefix)
    result = calibration.calibrate(w.prepared, w.config, w.runtime, tmp_path)
    assert not result["shortlist_prefix_verified"]
    assert result["prefix_failures"] == [{"event_index": i, "k": 5} for i in range(4)]
    assert len(w.calls.refine) == 4 * 16
    for event in result["events"]:
        assert event["per_k"]["5"]["best_cost"] == -1.0
        assert event["per_k"]["15"]["best_cost"] == 0.0
    assert any("not all K15 prefixes" in warning for warning in result["warnings"])


@pytest.mark.parametrize("spacing,expected_reason", [
    (0.0, "proper_likelihood_already_sufficient"),
    (0.01, "proper_likelihood_already_sufficient"),
    (3.0, "target_reached"),
    (1e4, "upper_bound_saturated"),
])
def test_temperature_monotonic_bounded_stable_and_never_sharpens(wired, spacing, expected_reason):
    weights = wired.likelihood.candidate_weights
    costs = np.arange(15., dtype=float) * spacing
    original = costs.copy()
    ess = []
    for t in np.linspace(1, 16, 61):
        p = weights(costs, t)
        assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0)
        ess.append(1 / float(p @ p))
    assert np.all(np.diff(ess) >= -1e-12)
    t, diagnostics = calibration._choose_temperature([costs, costs + 1e6], np, weights)
    shifted_t, _ = calibration._choose_temperature([costs + 1e9] * 2, np, weights)
    assert t == pytest.approx(shifted_t, abs=1e-6)
    assert 1 <= t <= 16 and diagnostics["reason"] == expected_reason
    if expected_reason == "target_reached":
        assert diagnostics["median_ess_at_selected_temperature"] == pytest.approx(3., abs=1e-9)
    elif expected_reason == "upper_bound_saturated":
        assert t == 16 and diagnostics["saturated"]
    else:
        assert t == 1 and not diagnostics["saturated"]
    np.testing.assert_array_equal(costs, original)


def test_native_weights_extreme_costs_no_nan(wired):
    costs = np.array([-1e200, 0., 1e200])
    for t in (1., 16.):
        p = wired.likelihood.candidate_weights(costs, t)
        np.testing.assert_array_equal(p, [1., 0., 0.])
    t, diagnostics = calibration._choose_temperature([costs], np, wired.likelihood.candidate_weights)
    assert t == 16 and diagnostics["saturated"]


def test_duplicate_locations_do_not_claim_spatial_diversity(wired, monkeypatch, tmp_path):
    def refine(*args):
        return np.array([1.0, 1.0, 1.0]), 0.0

    monkeypatch.setattr(wired.events, "refine_hypocentre_in_cell", refine)
    result = calibration.calibrate(wired.prepared, wired.config, wired.runtime, tmp_path)
    assert result["temperature"] == 1.
    metrics = result["per_k"]["15"]["at_temperature"]
    assert metrics["ess"]["median"] == pytest.approx(15.)
    assert metrics["unique_hypotheses"]["median"] == 1
    assert metrics["unique_location_ess"]["median"] == pytest.approx(1.)
    assert any("not spatial diversity" in warning for warning in result["warnings"])
    # Strict threshold, in physical metres, including transitive near-duplicates.
    positions = np.array([[0., 0., 0.], [0.006, 0., 0.], [0.012, 0., 0.], [0.023, 0., 0.]])
    labels = calibration._location_groups(positions, 100., np)
    assert labels[0] == labels[1] == labels[2] and labels[3] != labels[2]
    assert calibration._location_groups(np.array([[0., 0., 0.], [1., 0., 0.]]), 1., np) == [0, 1]


def test_shortlist_shortfall_reports_saturation(wired, monkeypatch, tmp_path):
    def select(cost, k, distance):
        return wired.select_native(cost, min(k, 2), distance)

    monkeypatch.setattr(wired.events, "select_candidate_cells", select)
    result = calibration.calibrate(wired.prepared, wired.config, wired.runtime, tmp_path)
    assert result["temperature"] == 16 and result["selection"]["saturated"]
    assert result["per_k"]["15"]["shortlist_shortfall_events"] == 4
    assert all(event["per_k"]["15"]["retained_count"] == 2 for event in result["events"])


@pytest.mark.parametrize("observations", [np.empty((0, 2)), np.array([[np.nan, 1.]]), np.ones((2, 1))])
def test_invalid_observations_fail_before_fields_or_writes(wired, tmp_path, observations):
    wired.prepared.arrivals_table = observations
    with pytest.raises(ValueError, match="observations"):
        calibration.calibrate(wired.prepared, wired.config, wired.runtime, tmp_path)
    assert not wired.calls.fields and not list(tmp_path.iterdir())


def test_import_is_snapshot_safe_without_numpy_or_production():
    script = (
        "import importlib.util, sys; "
        f"spec=importlib.util.spec_from_file_location('isolated_calibration', {str(Path(calibration.__file__))!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert not any(name in sys.modules for name in "
        "('numpy', 'eikonal', 'tomography', 'instruments', 'experiment_runner'))"
    )
    subprocess.run([sys.executable, "-S", "-c", script], check=True, timeout=10,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
