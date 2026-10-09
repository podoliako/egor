"""Small saved-experiment workflows, without using reference events as priors."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from main import CONFIG, main
import experiment_runner
from experiment_data import load_tomography_experiment
from instruments.likelihood import PickNoise
from projects.forward_modeling import (
    ForwardConfig, PointSet, VelocityGrid, load_arrivals, run_experiment, save_inputs,
)


@pytest.fixture
def small_experiment(tmp_path):
    # Deliberately not aligned with 1-km inversion cells or each other.
    values = np.full((2, 2, 2), 4750.)
    values[1, 0, 1] = 5250.
    stations = PointSet(("S-C", "S-A", "S-B"),
                        [[150., 180., 0.], [1770., 1770., 0.], [1120., 250., 0.]])
    events = PointSet(("E-B", "E-A", "E-C"),
                      [[333., 425., 620.], [1610., 1460., 1430.], [1200., 890., 820.]])
    save_inputs(tmp_path, "small", VelocityGrid(values, 1000.), stations, events)
    run_experiment(tmp_path, "small", config=ForwardConfig(refinement=2))
    return tmp_path, values, stations, events


def test_prepared_inversion_uses_saved_observations_and_reference_only_for_metrics(small_experiment):
    root, truth, stations, events = small_experiment
    config = replace(CONFIG, cell_size=500., initial_velocity_m_s=5000.)
    prepared = experiment_runner.prepare_inversion("small", config, root)
    assert prepared.initial_model.velocity.shape == (4, 4, 4)
    np.testing.assert_array_equal(prepared.initial_model.velocity, 5000.)
    np.testing.assert_array_equal(prepared.reference_model.velocity, truth.repeat(2, 0).repeat(2, 1).repeat(2, 2))
    assert prepared.station_ids == stations.ids
    assert prepared.event_ids == events.ids
    np.testing.assert_array_equal(prepared.station_locs, stations.coordinates_m)
    observed = {(a.event_id, a.station_id): a.arrival_time_s for a in load_arrivals(root, "small")}
    np.testing.assert_array_equal(prepared.arrivals_table, [
        [observed[(event, station)] for station in stations.ids] for event in events.ids
    ])
    assert not hasattr(prepared, "reference_event_coordinates_m")


def test_approximate_initial_gradient_is_independent_of_true_velocity(small_experiment):
    root = small_experiment[0]
    config = replace(CONFIG, cell_size=500., initial_gradient_m_s=(4700., 5500.))
    prepared = experiment_runner.prepare_inversion("small", config, root)
    expected = 4700. + 800. * (np.arange(4) + 0.5) / 4
    np.testing.assert_array_equal(prepared.initial_model.velocity, np.broadcast_to(expected, (4, 4, 4)))
    assert not np.array_equal(prepared.initial_model.velocity, prepared.reference_model.velocity)


def test_approximate_horizontal_layers_do_not_use_true_velocities(small_experiment):
    root = small_experiment[0]
    config = replace(CONFIG, cell_size=500., initial_layer_boundaries_km=(0.5, 1., 1.5),
                     initial_layer_velocities_m_s=(4700., 4900., 5100., 5300.))
    prepared = experiment_runner.prepare_inversion("small", config, root)
    np.testing.assert_array_equal(prepared.initial_model.velocity[0, 0], [4700., 4900., 5100., 5300.])
    np.testing.assert_array_equal(prepared.initial_model.velocity, np.broadcast_to(
        [4700., 4900., 5100., 5300.], (4, 4, 4),
    ))
    assert not np.array_equal(prepared.initial_model.velocity, prepared.reference_model.velocity)


@pytest.mark.parametrize("boundaries, velocities, gradient, message", [
    (None, (4700., 4900.), None, "require both"),
    ((1.,), None, None, "require both"),
    ((), (5000.,), None, "initial_layer_boundaries_km"),
    ((1., 1.), (4700., 4900., 5100.), None, "initial_layer_boundaries_km"),
    ((1.5, 1.), (4700., 4900., 5100.), None, "initial_layer_boundaries_km"),
    ((0.,), (4700., 4900.), None, "initial_layer_boundaries_km"),
    ((2.,), (4700., 4900.), None, "initial_layer_boundaries_km"),
    ((float("nan"),), (4700., 4900.), None, "initial_layer_boundaries_km"),
    ((1.,), (4700.,), None, "initial_layer_velocities_m_s"),
    ((1.,), (4700., 0.), None, "initial_layer_velocities_m_s"),
    ((1.,), (4700., float("inf")), None, "initial_layer_velocities_m_s"),
    ((1.,), (4700., 4900.), (4700., 5500.), "mutually exclusive"),
])
def test_invalid_horizontal_layers_rejected(small_experiment, boundaries, velocities, gradient, message):
    with pytest.raises(ValueError, match=message):
        experiment_runner.prepare_inversion("small", replace(
            CONFIG, cell_size=1000., initial_layer_boundaries_km=boundaries,
            initial_layer_velocities_m_s=velocities, initial_gradient_m_s=gradient,
        ), small_experiment[0])


@pytest.mark.parametrize("endpoints", [(0., 5500.), (4700., -1.), (float("nan"), 5500.), (4700.,)])
def test_invalid_initial_gradient_rejected(small_experiment, endpoints):
    with pytest.raises(ValueError, match="initial_gradient_m_s"):
        experiment_runner.prepare_inversion(
            "small", replace(CONFIG, cell_size=1000., initial_gradient_m_s=endpoints), small_experiment[0],
        )


def test_saved_run_passes_exact_stations_and_no_event_truth(small_experiment, monkeypatch):
    root, truth, stations, events = small_experiment
    calls = []
    monkeypatch.setattr(experiment_runner, "warm_up_jit", lambda: calls.append("warmup"))
    monkeypatch.setattr(experiment_runner, "run_em", lambda *args, **kwargs: calls.append((args, kwargs)))
    config = replace(CONFIG, cell_size=1000., n_cycles=1, n_workers=1,
                     weights_top_n=2, weights_min_distance=3,
                     candidate_mode="hard", temperature=0.5, save_runs=False)
    assert main(config, experiment_id="small", experiments_root=root) is None
    assert calls[0] == "warmup"
    (passed_config, initial, arrivals, station_locs, noise), kwargs = calls[1]
    assert passed_config is config
    np.testing.assert_array_equal(station_locs, stations.coordinates_m)
    assert arrivals.shape == (len(events.ids), len(stations.ids))
    assert initial.shape == (2, 2, 2) and np.all(initial.velocity == 5000.)
    np.testing.assert_array_equal(kwargs["reference_model"].velocity, truth)
    assert kwargs["run_name"].endswith("_small")
    assert kwargs["source_experiment"]["id"] == "small"
    assert set(kwargs) == {"reference_model", "source_experiment", "run_name"}


@pytest.fixture
def metadata_experiment(tmp_path):
    """Complete saved input/output without running a forward solver."""
    stations = PointSet(("S1", "S2"), [[100., 100., 0.], [900., 900., 0.]])
    events = PointSet(("E1",), [[500., 500., 500.]])
    input_dir = save_inputs(tmp_path, "metadata", VelocityGrid(np.full((2, 2, 2), 5000.), 1000.), stations, events)
    output_dir = tmp_path / "output" / "metadata"
    output_dir.mkdir(parents=True)
    (output_dir / "arrivals.csv").write_text(
        "station_id,event_id,arrival_time_s\nS1,E1,0\nS2,E1,0.1\n", encoding="utf-8",
    )
    metadata = {
        "time_reference": "earliest_station_arrival_per_event",
        "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
                  "arrival_time": "s", "elapsed_time": "s"},
        "input_sha256": {name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
                         for name in ("model.npz", "stations.csv", "events.csv")},
        "noise": {"enabled": True, "model": "independent_gaussian_relative_plus_absolute",
                  "config": {"relative_sigma": 0.03, "absolute_sigma_s": 0.07, "seed": 42}},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path, metadata_path


def test_saved_weight_sigmas_use_metadata_and_overrides(metadata_experiment, monkeypatch):
    root, _ = metadata_experiment
    passed = []
    monkeypatch.setattr(experiment_runner, "warm_up_jit", lambda: None)
    monkeypatch.setattr(experiment_runner, "run_em", lambda *args, **kwargs: passed.append(args[4]))
    config = replace(CONFIG, cell_size=1000., save_runs=False)
    prepared = experiment_runner.prepare_inversion("metadata", config, root)
    assert prepared.noise_sigmas == (0.03, 0.07)
    experiment_runner.run_saved_experiment("metadata", config, root)
    assert passed[-1] == PickNoise(0.03, 0.07, 0.2)
    experiment_runner.run_saved_experiment("metadata", replace(
        config, weight_noise_relative_sigma=0., weight_noise_absolute_sigma_s=0.4,
        weight_model_sigma_s=0.6,
    ), root)
    assert passed[-1] == PickNoise(0., 0.4, 0.6)


@pytest.mark.parametrize("noise", [None, {"enabled": False, "config": None}])
def test_absent_or_disabled_noise_has_zero_sigmas(metadata_experiment, noise):
    root, path = metadata_experiment
    metadata = json.loads(path.read_text())
    if noise is None:
        del metadata["noise"]
    else:
        metadata["noise"] = noise
    path.write_text(json.dumps(metadata))
    assert load_tomography_experiment("metadata", root).noise_sigmas == (0., 0.)


@pytest.mark.parametrize("noise, message", [
    ({"enabled": True, "model": "other", "config": {"relative_sigma": 0., "absolute_sigma_s": 0.}}, "noise.model"),
    ({"enabled": True, "model": "independent_gaussian_relative_plus_absolute", "config": {"relative_sigma": -1., "absolute_sigma_s": 0.}}, "relative_sigma"),
    ({"enabled": True, "model": "independent_gaussian_relative_plus_absolute", "config": {"relative_sigma": float("nan"), "absolute_sigma_s": 0.}}, "relative_sigma"),
    ({"enabled": True, "model": "independent_gaussian_relative_plus_absolute", "config": {"relative_sigma": 0., "absolute_sigma_s": float("inf")}}, "absolute_sigma_s"),
    ({"enabled": True, "model": "independent_gaussian_relative_plus_absolute", "config": {"relative_sigma": 0.}}, "absolute_sigma_s"),
    ({"enabled": True, "model": "independent_gaussian_relative_plus_absolute", "config": {"relative_sigma": True, "absolute_sigma_s": 0.}}, "relative_sigma"),
    ({"enabled": "true"}, "noise.enabled"),
])
def test_invalid_metadata_noise_rejected(metadata_experiment, noise, message):
    root, path = metadata_experiment
    metadata = json.loads(path.read_text())
    metadata["noise"] = noise
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_tomography_experiment("metadata", root)


@pytest.mark.parametrize("overrides, message", [
    ({"weight_noise_relative_sigma": -1.}, "weight_noise_relative_sigma"),
    ({"weight_noise_absolute_sigma_s": float("nan")}, "weight_noise_absolute_sigma_s"),
    ({"weight_model_sigma_s": float("inf")}, "weight_model_sigma_s"),
    ({"weight_noise_relative_sigma": 0., "weight_noise_absolute_sigma_s": 0.,
      "weight_model_sigma_s": 0.}, "must be positive"),
])
def test_invalid_weight_config_rejected_before_em(metadata_experiment, monkeypatch, overrides, message):
    root, _ = metadata_experiment
    monkeypatch.setattr(experiment_runner, "warm_up_jit", lambda: pytest.fail("unexpected JIT"))
    with pytest.raises(ValueError, match=message):
        config = replace(CONFIG, cell_size=1000., **overrides)
        experiment_runner.run_saved_experiment("metadata", config, root, validate_only=True)


def test_validate_only_skips_jit_and_inversion(small_experiment, monkeypatch):
    root = small_experiment[0]
    def forbidden():
        raise AssertionError("must not run the inversion")
    monkeypatch.setattr(experiment_runner, "warm_up_jit", forbidden)
    config = replace(CONFIG, cell_size=1000.)
    prepared = main(config, experiment_id="small", experiments_root=root, validate_only=True)
    assert prepared.arrivals_table.shape == (3, 3)


@pytest.mark.parametrize("initial", [
    [], ["--initial-gradient-m-s", "4700", "5500"],
    ["--initial-layer-boundaries-km", "1", "--initial-layer-velocities-m-s", "4700", "4900"],
    ["--weights-top-n", "2", "--weights-min-distance", "3", "--temperature", "0.5"],
])
def test_cli_validate_only_on_small_experiment(small_experiment, initial):
    root = small_experiment[0]
    completed = subprocess.run(
        [sys.executable, "main.py", "small", "--experiments-root", str(root),
         "--cell-size-m", "1000", *initial, "--validate-only"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "3 events, 3 stations, inversion grid (2, 2, 2)" in completed.stdout


@pytest.mark.parametrize("options", [
    ["--weights-top-n", "0"], ["--weights-min-distance", "0"],
    ["--temperature", "0"], ["--temperature", "nan"],
])
def test_invalid_weight_cli_options_rejected(small_experiment, options):
    result = subprocess.run(
        [sys.executable, "main.py", "small", "--experiments-root", str(small_experiment[0]),
         *options, "--validate-only"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr


@pytest.mark.parametrize("options", [
    ["--weight-noise-relative-sigma", "-0.1"],
    ["--weight-noise-relative-sigma", "nan"],
    ["--weight-noise-absolute-sigma-s", "inf"],
    ["--weight-noise-absolute-sigma-s", "-1"],
    ["--weight-model-sigma-s", "-0.1"],
    ["--weight-model-sigma-s", "nan"],
    ["--weight-noise-relative-sigma", "0", "--weight-noise-absolute-sigma-s", "0",
     "--weight-model-sigma-s", "0"],
])
def test_invalid_weight_sigma_cli_rejected(metadata_experiment, options):
    root, _ = metadata_experiment
    result = subprocess.run(
        [sys.executable, "main.py", "metadata", "--experiments-root", str(root),
         "--cell-size-m", "1000", *options, "--validate-only"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr


def test_weight_sigma_cli_accepts_valid_overrides(metadata_experiment):
    root, _ = metadata_experiment
    result = subprocess.run(
        [sys.executable, "main.py", "metadata", "--experiments-root", str(root),
         "--cell-size-m", "1000", "--weight-noise-relative-sigma", "0",
         "--weight-noise-absolute-sigma-s", "0.5", "--weight-model-sigma-s", "0",
         "--validate-only"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cell_size", [750., 3000., 0., -10.])
def test_mismatched_grid_is_rejected_before_em(small_experiment, cell_size):
    root = small_experiment[0]
    with pytest.raises(ValueError, match="cell_size"):
        experiment_runner.prepare_inversion("small", replace(CONFIG, cell_size=cell_size), root)


def test_real_one_iteration_on_saved_small_experiment(small_experiment, tmp_path):
    root = small_experiment[0]
    config = replace(CONFIG, cell_size=1000., n_cycles=1, subdivision=1,
                     n_workers=1, weights_top_n=1, save_runs=False,
                     runs_dir=str(tmp_path / "not-created"))
    assert main(config, experiment_id="small", experiments_root=root) is None
    assert not (tmp_path / "not-created").exists()


def test_layered_run_records_initial_profile_and_completes(small_experiment, tmp_path):
    config = replace(CONFIG, cell_size=1000., n_cycles=1, subdivision=1,
                     n_workers=1, weights_top_n=1,
                     initial_layer_boundaries_km=(1.,),
                     initial_layer_velocities_m_s=(4700., 5300.),
                     runs_dir=str(tmp_path / "layered-runs"))
    logger = main(config, experiment_id="small", experiments_root=small_experiment[0])
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["run_params"]["initial_layer_boundaries_km"] == [1.]
    assert meta["run_params"]["initial_layer_velocities_m_s"] == [4700., 5300.]
    np.testing.assert_array_equal(np.load(logger.run_dir / "initial_model.npy")[0, 0], [4700., 5300.])
    assert (logger.run_dir / "iter_0" / "delta_s.npy").is_file()


def test_gradient_run_records_parameter_and_completes(small_experiment, tmp_path):
    root = small_experiment[0]
    config = replace(CONFIG, cell_size=1000., n_cycles=1, subdivision=1,
                     n_workers=1, weights_top_n=1,
                     initial_gradient_m_s=(4700., 5500.), runs_dir=str(tmp_path / "runs"))
    logger = main(config, experiment_id="small", experiments_root=root)
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert "smoothness_reg" not in meta["run_params"]
    assert meta["run_params"]["initial_gradient_m_s"] == [4700., 5500.]
    np.testing.assert_array_equal(np.load(logger.run_dir / "initial_model.npy")[:, :, 0], 4900.)
    np.testing.assert_array_equal(np.load(logger.run_dir / "initial_model.npy")[:, :, 1], 5300.)
    assert meta["source_experiment"]["id"] == "small"
    assert (logger.run_dir / "iter_0" / "delta_s.npy").is_file()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("candidate_mode", ["soft", "hard"])
def test_saved_multicandidate_weights_use_likelihood_in_serial_and_parallel(small_experiment, tmp_path, workers, candidate_mode):
    config = replace(CONFIG, cell_size=1000., n_cycles=1, subdivision=1,
                     n_workers=workers, weights_top_n=2, weight_model_sigma_s=2.,
                     candidate_mode=candidate_mode,
                     runs_dir=str(tmp_path / f"runs-{workers}-{candidate_mode}"))
    logger = main(config, experiment_id="small", experiments_root=small_experiment[0])
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["run_params"]["weight_likelihood"].startswith("gaussian_independent")
    assert meta["run_params"]["normal_equations"] == "station_precision_weighted_profiled_origin"
    assert meta["run_params"]["candidate_mode"] == candidate_mode
    assert meta["run_params"]["weight_noise_relative_sigma"] == 0.
    assert meta["run_params"]["weight_noise_absolute_sigma_s"] == 0.
    assert meta["run_params"]["weight_model_sigma_s"] == 2.
    weights = []
    for event in range(3):
        with np.load(logger.run_dir / "iter_0" / f"event_{event}" / "weights.npz") as data:
            assert len(data["weight_values"]) == 2
            weights.append(data["weight_values"].copy())
    if candidate_mode == "hard":
        assert np.all(np.isin(weights, [0., 1.]))
    else:
        assert not np.allclose(weights[0], weights[1])
        assert np.all(np.asarray(weights) > 0)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.)


def test_fine_grid_ray_logging_for_viewer(small_experiment, tmp_path):
    config = replace(CONFIG, cell_size=1000., n_cycles=1, subdivision=2, n_workers=1,
                     weights_top_n=1, log_g_per_weight=True, runs_dir=str(tmp_path / "runs"))
    logger = main(config, experiment_id="small", experiments_root=small_experiment[0])
    with np.load(logger.run_dir / "iter_0" / "event_0" / "weight_0" / "G_stations_sparse.npz") as G:
        np.testing.assert_array_equal(G["shape"], [4, 4, 4])
        assert len(G["offsets"]) == 4 and G["values"].size > 0
    assert (logger.run_dir / "final_model.npy").is_file()
    quality = json.loads((logger.run_dir / "quality.jsonl").read_text().splitlines()[0])
    assert set(quality) == {"iter", "avg_abs_pct_dev", "rms_m_s"}


@pytest.mark.parametrize("top_n, n_candidates", [(6, 5), (3, 2)])
def test_top_n_cannot_exceed_candidates(top_n, n_candidates):
    with pytest.raises(ValueError, match="n_candidates"):
        replace(CONFIG, weights_top_n=top_n, n_candidates=n_candidates)
