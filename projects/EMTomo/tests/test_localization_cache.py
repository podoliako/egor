"""Control-to-E-step reuse must preserve the complete numerical calculation."""
from dataclasses import replace
import json
import weakref

import numpy as np
import pytest

from config import InversionConfig
from eikonal import station_travel_time_fields
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise
from tomography import tomography_em as em
from tomography import tomography_events as events
from tomography import tomography_validation as validation
from velocity_model import VelocityModel


@pytest.fixture
def problem():
    model = VelocityModel(np.linspace(2900., 3300., 12).reshape(3, 2, 2), 200.)
    stations = np.array([[x, y, z] for x in (0., 600.)
                         for y in (0., 400.) for z in (0., 400.)])
    config = InversionConfig(cell_size=200., subdivision=2, n_candidates=4,
                             weights_top_n=3, weights_min_distance=1, n_workers=1,
                             lambda_reg=1., coverage_damping_power=0., log_g_per_weight=True)
    times = station_travel_time_fields(model.refined(2, "nearest"), stations, 1)
    positions = [[1.3, 1.2, 1.7], [3.1, 1.4, 0.9], [2.3, 2.2, 1.6], [2.8, .8, 1.1]]
    observed = np.array([sample_cell_centered_trilinear_batch(times, p) + origin
                         for p, origin in zip(positions, [1., -2., 10., 0.])])
    observed += np.random.default_rng(13).normal(0., .002, observed.shape)
    observed[0, 0] = np.nan
    mask = np.zeros(observed.shape, dtype=bool)
    mask[0, [1, 6]] = True
    mask[1, 3] = True
    mask[2, 2] = True
    training = observed.copy()
    training[mask] = np.nan
    return model, training, observed, mask, stations, PickNoise(.03, .01, .02), config


def _fields_settings(model, times, stations, noise, config, misfit=False):
    fields = events.IterationFields.from_times(
        times, metric_to_cell_coord(stations, model.cell_size / config.subdivision),
        model.cell_size / config.subdivision, config.subdivision, noise,
        config.slowness_interpolation,
    )
    settings = events.EventSettings(
        subdivision=config.subdivision, slowness_interpolation=config.slowness_interpolation,
        n_candidates=config.n_candidates, weights_top_n=config.weights_top_n,
        weights_min_distance=config.weights_min_distance, candidate_mode=config.candidate_mode,
        temperature=config.temperature, noise=noise, fast_refinement=config.fast_refinement,
        log_g_per_weight=True, log_misfit=misfit,
    )
    return fields, settings


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("mode,misfit", [("soft", False), ("hard", True)])
def test_cached_candidates_weights_normal_equations_and_diagnostics_are_exact(
    problem, workers, fast, mode, misfit, monkeypatch,
):
    model, training, observed, mask, stations, noise, config = problem
    config = replace(config, n_workers=workers, fast_refinement=fast)
    reference_score = validation.evaluate_control_rmse(
        model, training, observed, mask, stations, noise, config,
    )
    score, cache = validation.evaluate_control_rmse(
        model, training, observed, mask, stations, noise, config, return_cache=True,
    )
    assert score == reference_score
    assert set(cache.candidates) == {0, 1, 2}  # Event 3 must still be localized normally.
    assert all(len(value.costs) == 4 for value in cache.candidates.values())
    assert not cache.times.flags.writeable
    # Reweight and truncate from the entire cached shortlist, not the previous MAP.
    config = replace(config, candidate_mode=mode, temperature=2.3, weights_top_n=2)
    cache.validate(model, training, stations, noise, config)
    fields, settings = _fields_settings(model, cache.times, stations, noise, config, misfit)
    reference = events.run_events(training, fields, settings, workers)
    original_refine = events.refine_hypocentre_in_cell
    refinement_calls = []

    def checked_refine(times, picks, *args, **kwargs):
        # Only the fully observed, unscored event may be localized on a cache hit.
        np.testing.assert_array_equal(picks, training[3])
        refinement_calls.append(1)
        return original_refine(times, picks, *args, **kwargs)

    monkeypatch.setattr(events, "refine_hypocentre_in_cell", checked_refine)
    diagnostics = {}
    actual = events.run_events(training, fields, settings, workers,
                               candidates=cache.candidates, diagnostics=diagnostics)
    if workers == 1:
        assert len(refinement_calls) == 4
    assert diagnostics["reused_localizations"] == 3
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_array_equal(a, b)
    for (ai, a), (bi, b) in zip(actual[2], reference[2]):
        assert ai == bi
        _assert_same(vars(a), vars(b))


@pytest.mark.parametrize("solver", ["dense", "matrix_free"])
def test_step_reuses_fields_and_has_identical_velocity_update(problem, solver, monkeypatch):
    model, training, observed, mask, stations, noise, config = problem
    config = replace(config, linear_solver=solver)
    reference = em.make_tomography_step(model, training, stations, noise, config)
    _, cache = validation.evaluate_control_rmse(
        model, training, observed, mask, stations, noise, config, return_cache=True,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Cached step must not rebuild FMM fields or refine the model")

    monkeypatch.setattr(em, "station_travel_time_fields", forbidden)
    monkeypatch.setattr(VelocityModel, "refined", forbidden)
    diagnostics = {}
    actual = em.make_tomography_step(model, training, stations, noise, config,
                                     localization_cache=cache, diagnostics=diagnostics)
    for a, b in zip(actual, reference):
        np.testing.assert_array_equal(a, b)
    assert diagnostics["wall_s"]["station_fields_s"] == 0.
    assert diagnostics["events"]["reused_localizations"] == 3


def test_held_out_values_cannot_change_cached_candidates(problem):
    model, training, observed, mask, stations, noise, config = problem
    score, cache = validation.evaluate_control_rmse(*problem, return_cache=True)
    changed = observed.copy()
    changed[mask] += 100.
    changed[~mask] = -1e9
    other_score, other = validation.evaluate_control_rmse(
        model, training, changed, mask, stations, noise, config, return_cache=True,
    )
    assert other_score != score
    assert other.fingerprint == cache.fingerprint
    np.testing.assert_array_equal(other.times, cache.times)
    for event in cache.candidates:
        _assert_same(vars(other.candidates[event]), vars(cache.candidates[event]))


@pytest.mark.parametrize("change", ["model", "cell_size", "training", "mask", "stations", "noise",
                                    "subdivision", "slowness_interpolation", "n_candidates",
                                    "weights_min_distance", "fast_refinement"])
def test_stale_cache_is_rejected(problem, change):
    model, training, observed, mask, stations, noise, config = problem
    _, cache = validation.evaluate_control_rmse(*problem, return_cache=True)
    if change == "model":
        model = VelocityModel(model.velocity + 1., model.cell_size)
    elif change == "cell_size":
        model = VelocityModel(model.velocity, model.cell_size * 2)
    elif change == "training":
        training = training + .001
    elif change == "mask":
        training = training.copy()
        training[0, 2] = np.nan
    elif change == "stations":
        stations = stations + 1.
    elif change == "noise":
        noise = PickNoise(.02, .01, .02)
    else:
        config = replace(config, **{change: {
            "subdivision": 1, "slowness_interpolation": "trilinear", "n_candidates": 5,
            "weights_min_distance": 2, "fast_refinement": False,
        }[change]})
    with pytest.raises(ValueError, match="Localization cache does not match"):
        em.make_tomography_step(model, training, stations, noise, config, localization_cache=cache)


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_complete_run_same_stopping_models_scores_and_saved_weights(problem, mode, tmp_path, monkeypatch):
    model, _, observed, _, stations, noise, config = problem
    config = replace(config, n_cycles=4, candidate_mode=mode, n_workers=2,
                     runs_dir=str(tmp_path), control_min_training_picks=3,
                     control_fraction=.2, temperature_schedule=(2., 1.5, 1., .8))
    # Preserve the actual per-update field objects only as weak references. The
    # parent must not pin the old control fields during the following solver.
    original_fields = validation.station_travel_time_fields
    original_solve = em.solve_slowness_update
    previous_fields = []

    def fields(*args, **kwargs):
        value = original_fields(*args, **kwargs)
        previous_fields.append(weakref.ref(value))
        return value

    def solve(*args, **kwargs):
        assert all(reference() is None for reference in previous_fields)
        return original_solve(*args, **kwargs)

    reference = em.run_em(config, model, observed, stations, noise, reuse_control_localization=False)
    monkeypatch.setattr(validation, "station_travel_time_fields", fields)
    monkeypatch.setattr(em, "solve_slowness_update", solve)
    actual = em.run_em(config, model, observed, stations, noise)
    assert json.loads((actual.run_dir / "stopping.json").read_text()) == json.loads(
        (reference.run_dir / "stopping.json").read_text())
    np.testing.assert_array_equal(np.load(actual.run_dir / "final_model.npy"),
                                  np.load(reference.run_dir / "final_model.npy"))
    actual_iters = sorted(actual.run_dir.glob("iter_*"))
    assert len(actual_iters) == len(list(reference.run_dir.glob("iter_*")))
    for iteration in actual_iters:
        expected = reference.run_dir / iteration.name
        for filename in ("model.npy", "delta_s.npy", "sensitivity_diagonal.npy", "coverage_confidence.npy"):
            np.testing.assert_array_equal(np.load(iteration / filename), np.load(expected / filename))
        ac = json.loads((iteration / "control.json").read_text())
        ec = json.loads((expected / "control.json").read_text())
        ac.pop("wall_s"); ec.pop("wall_s")
        assert ac == ec
        assert (iteration / "complete.json").exists()
        for event in iteration.glob("event_*"):
            for filename in ("weights.npz", "rays.npz", "ray_status.npz"):
                with np.load(event / filename) as a, np.load(expected / event.name / filename) as b:
                    _assert_same(dict(a), dict(b))
        performance = json.loads((iteration / "performance.json").read_text())
        if iteration.name != "iter_0":
            assert performance["events"]["reused_localizations"] > 0
