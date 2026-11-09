"""Held-out RMSE controls stopping; truth and output settings do not."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import InversionConfig
from instruments.likelihood import PickNoise
from main import CONFIG, _parser
from tomography import tomography_em as em
from tomography.tomography_logging import TomographyLogger
from velocity_model import VelocityModel


def _run(monkeypatch, tmp_path, scores, *, save_runs=True, with_reference=False,
         n_stations=10, logger=None, **overrides):
    initial = VelocityModel(np.full((1, 1, 1), 5000.), 1000.)
    arrivals = np.arange(2 * n_stations, dtype=float).reshape(2, n_stations)
    before = arrivals.copy()
    stations = np.zeros((n_stations, 3))
    updates, evaluations = [], []
    training_tables = []

    def step(model, training, current_stations, noise, config, *, iteration, logger):
        assert iteration == len(updates)
        training_tables.append(training.copy())
        # Deliberately worsen velocity RMSE against truth at EVERY update.
        target = np.full(model.shape, 5010. + 10. * iteration)
        updates.append(target)
        return 1. / target - 1. / model.velocity, np.ones(model.shape), np.ones(model.shape)

    def evaluate(model, training, observed, mask, current_stations, noise, config):
        index = len(evaluations)
        assert len(updates) == index + 1  # Always score after the full velocity update.
        np.testing.assert_allclose(model.velocity, updates[-1])
        np.testing.assert_array_equal(observed, before)
        np.testing.assert_array_equal(training, training_tables[-1])
        assert np.isnan(training[mask]).all()
        np.testing.assert_array_equal(training[~mask], observed[~mask])
        evaluations.append(mask.copy())
        return scores[index]

    monkeypatch.setattr(em, "make_tomography_step", step)
    monkeypatch.setattr(em, "evaluate_control_rmse", evaluate)
    config = InversionConfig(**{
        "n_cycles": len(scores), "save_runs": save_runs, "runs_dir": str(tmp_path),
        "max_velocity_step_fraction": None,
        "temperature_schedule": tuple(4. if i < 3 else 1. for i in range(len(scores))),
        **overrides,
    })
    result = em.run_em(config, initial, arrivals, stations, PickNoise(0., .1, 0.),
                       reference_model=initial if with_reference else None, logger=logger)
    np.testing.assert_array_equal(arrivals, before)
    np.testing.assert_array_equal(initial.velocity, 5000.)
    return SimpleNamespace(logger=result, updates=updates, evaluations=evaluations,
                           training=training_tables, arrivals=arrivals, config=config)


@pytest.mark.parametrize("save_runs", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("scores, expected_updates, stopped", [
    ([100., 101., 102., 90.], 3, True),
    ([100., 99., 98.01, 90.], 3, True),  # Exactly 1% is insufficient.
    ([100., 99.2, 98.6, 90.], 3, True),
    ([100., 99.8, 98., 97.5, 97.4, 80.], 5, True),  # Improvement resets patience.
    ([100., 98., 96., 94., 92.], 5, False),
    ([20., 20., 20., 20.], 3, True),
    ([0., 0., 0., 0.], 3, True),
    ([0., 1., 2., 0.], 3, True),
    ([10.], 1, False),
    ([10., 20.], 2, False),
])
def test_control_stopping_after_complete_updates(
    monkeypatch, tmp_path, capsys, scores, expected_updates, stopped, save_runs, with_reference,
):
    result = _run(monkeypatch, tmp_path, scores, save_runs=save_runs, with_reference=with_reference)
    assert len(result.updates) == len(result.evaluations) == expected_updates
    mask = result.evaluations[0]
    assert mask.sum() == 2  # Default: 10% of 20 finite picks.
    for current_mask in result.evaluations:
        np.testing.assert_array_equal(current_mask, mask)
    assert ("[EM] Stopping" in capsys.readouterr().out) == stopped

    if not save_runs:
        assert result.logger is None
        assert list(tmp_path.iterdir()) == []
        return

    run_dir = result.logger.run_dir
    np.testing.assert_allclose(np.load(run_dir / "final_model.npy"), result.updates[-1])
    np.testing.assert_array_equal(np.load(run_dir / "control_mask.npy"), mask)
    split = json.loads((run_dir / "control_split.json").read_text())
    assert split["seed"] == 42
    assert split["n_control"] == 2
    assert split["actual_fraction"] == .1
    assert (run_dir / "timing_summary.json").exists()
    assert not (run_dir / f"iter_{expected_updates}").exists()
    for iteration in range(expected_updates):
        directory = run_dir / f"iter_{iteration}"
        for filename in ("complete.json", "delta_s.npy", "sensitivity_diagonal.npy",
                         "coverage_confidence.npy", "temperature.json", "control.json"):
            assert (directory / filename).exists()
        control = json.loads((directory / "control.json").read_text())
        assert control["rmse_s"] == scores[iteration]
        assert control["stop"] == (stopped and iteration == expected_updates - 1)
        if iteration == 0 or scores[iteration - 1] == 0:
            assert control["relative_improvement"] is None
        else:
            assert control["relative_improvement"] == pytest.approx(
                (scores[iteration - 1] - scores[iteration]) / scores[iteration - 1])
    quality_path = run_dir / "quality.jsonl"
    assert quality_path.exists() == with_reference
    if with_reference:
        quality = [json.loads(line) for line in quality_path.read_text().splitlines()]
        np.testing.assert_allclose([row["rms_m_s"] for row in quality],
                                   10. * np.arange(1, expected_updates + 1))
    ending = json.loads((run_dir / "stopping.json").read_text())
    assert ending["completed_updates"] == expected_updates
    assert ending["reason"] == ("control_rmse_plateau" if stopped else "max_cycles")
    assert ending["control_rmse_s"] == scores[expected_updates - 1]
    params = json.loads((run_dir / "meta.json").read_text())["run_params"]
    assert params["early_stopping"] == em.METHOD_DESCRIPTION["early_stopping"]
    assert params["control_n_picks"] == 2


@pytest.mark.parametrize("options, n_stations", [({"control_fraction": 0.}, 10), ({}, 5), ({}, 2)])
def test_no_control_uses_all_picks_and_never_stops_on_truth(monkeypatch, tmp_path, options, n_stations):
    result = _run(monkeypatch, tmp_path, [100.] * 5, with_reference=True,
                  n_stations=n_stations, **options)
    assert len(result.updates) == 5
    assert result.evaluations == []
    for training in result.training:
        np.testing.assert_array_equal(training, result.arrivals)
    assert not np.load(result.logger.run_dir / "control_mask.npy").any()
    assert not (result.logger.run_dir / "iter_0/control.json").exists()
    ending = json.loads((result.logger.run_dir / "stopping.json").read_text())
    assert ending["reason"] == "max_cycles"
    assert ending["control_rmse_s"] is None


@pytest.mark.parametrize("patience, threshold, scores, expected", [
    (1, .01, [10.] * 5, 3),  # Even configurable patience cannot bypass three updates.
    (3, .01, [10.] * 5, 4),
    (2, .05, [100., 98., 96., 94.], 3),
    (2, 0., [100., 99.9, 99.8, 99.7], 4),
])
def test_configurable_threshold_and_patience(monkeypatch, tmp_path, patience, threshold, scores, expected):
    result = _run(monkeypatch, tmp_path, scores, early_stopping_patience=patience,
                  early_stopping_min_relative_improvement=threshold)
    assert len(result.updates) == expected


def test_control_is_saved_before_iteration_publication(monkeypatch, tmp_path):
    save_json = TomographyLogger._save_json
    published = []

    def checked_save(path, values):
        if path.name == "complete.json":
            assert path.with_name("control.json").exists()
            published.append(values["iter"])
        save_json(path, values)

    monkeypatch.setattr(TomographyLogger, "_save_json", staticmethod(checked_save))
    _run(monkeypatch, tmp_path, [10.] * 4)
    assert published == [0, 1, 2]


def test_failed_control_save_does_not_publish_iteration(monkeypatch, tmp_path):
    logger = TomographyLogger(base_dir=tmp_path)
    save_json = logger._save_json

    def fail_save(path, values):
        if path.name == "control.json":
            raise OSError("control write failed")
        save_json(path, values)

    monkeypatch.setattr(logger, "_save_json", fail_save)
    with pytest.raises(OSError, match="control write failed"):
        _run(monkeypatch, tmp_path, [10.] * 4, logger=logger)
    assert not (logger.run_dir / "iter_0/complete.json").exists()
    assert not (logger.run_dir / "final_model.npy").exists()


@pytest.mark.parametrize("name", ["control_fraction", "early_stopping_min_relative_improvement"])
@pytest.mark.parametrize("value", [-.1, 1., np.nan, np.inf, True, "0.1", None])
def test_invalid_fraction_controls(name, value):
    with pytest.raises(ValueError, match=name):
        InversionConfig(**{name: value})


@pytest.mark.parametrize("name, value", [
    ("control_seed", -1), ("control_seed", 1.5), ("control_seed", True),
    ("control_min_training_picks", 1), ("control_min_training_picks", 5.5),
    ("control_min_training_picks", np.bool_(True)),
    ("early_stopping_patience", 0), ("early_stopping_patience", 1.1),
    ("early_stopping_patience", True),
])
def test_invalid_integer_controls(name, value):
    with pytest.raises(ValueError, match=name):
        InversionConfig(**{name: value})


def test_control_cli_roundtrip():
    args = vars(_parser().parse_args([
        "example", "--control-fraction", ".2", "--control-seed", "7",
        "--control-min-training-picks", "6", "--early-stopping-patience", "3",
        "--early-stopping-min-relative-improvement", ".005",
    ]))
    for name in ("experiment_id", "experiments_root", "validate_only"):
        args.pop(name)
    config = replace(CONFIG, **args)
    assert config.control_fraction == .2
    assert config.control_seed == 7
    assert config.control_min_training_picks == 6
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_relative_improvement == .005


def test_real_em_run_scores_final_model_and_excludes_control_from_event_logs(tmp_path):
    from instruments.instruments_coords import sample_cell_centered_trilinear_batch

    truth = VelocityModel(np.full((4, 4, 4), 2000.), 100.)
    initial = VelocityModel(np.full(truth.shape, 2100.), truth.cell_size)
    stations = np.array([[0., 0., 0.], [400., 0., 0.], [0., 400., 0.],
                         [0., 0., 400.], [400., 400., 400.], [400., 400., 0.]])
    fields = em.station_travel_time_fields(truth.refined(2), stations, n_workers=1)
    arrivals = np.stack([
        sample_cell_centered_trilinear_batch(fields, [3., 4., 3.]) + 2.5,
        sample_cell_centered_trilinear_batch(fields, [4., 3., 4.]) + 1.5,
    ]).astype(float)
    original = arrivals.copy()
    noise = PickNoise(0., .01, 0.)
    config = InversionConfig(n_cycles=3, n_workers=1, subdivision=2, n_candidates=2,
                             runs_dir=str(tmp_path))
    logger = em.run_em(config, initial, arrivals, stations, noise)
    run_dir = logger.run_dir
    mask = np.load(run_dir / "control_mask.npy")
    assert mask.sum() == 1
    training = arrivals.copy()
    training[mask] = np.nan
    for iteration in range(3):
        directory = run_dir / f"iter_{iteration}"
        assert (directory / "complete.json").exists()
        for event in np.flatnonzero(mask.any(axis=1)):
            with np.load(directory / f"event_{event}/weights.npz") as weights:
                np.testing.assert_array_equal(weights["observed_mask"], ~mask[event])
            with np.load(directory / f"event_{event}/ray_status.npz") as rays:
                np.testing.assert_array_equal(rays["observed_mask"], ~mask[event])
    final_model = VelocityModel(np.load(run_dir / "final_model.npy"), initial.cell_size)
    score = json.loads((run_dir / "iter_2/control.json").read_text())
    expected_rmse = em.evaluate_control_rmse(
        final_model, training, arrivals, mask, stations, noise, config,
    )
    assert score["rmse_s"] == pytest.approx(expected_rmse)
    assert np.isfinite(expected_rmse)
    assert not (run_dir / "quality.jsonl").exists()
    np.testing.assert_array_equal(arrivals, original)
