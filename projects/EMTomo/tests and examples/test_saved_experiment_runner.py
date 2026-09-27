"""Small saved-experiment workflows, without using reference events as priors."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from main import CONFIG, main
import experiment_runner
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
    config = replace(CONFIG, cell_size=500., grid_shape=(99, 99, 99), background_vp=5000.)
    prepared = experiment_runner.prepare_inversion("small", config, root)
    assert prepared.initial_model.grid.vp.shape == (4, 4, 4)
    np.testing.assert_array_equal(prepared.initial_model.grid.vp, 5000.)
    np.testing.assert_array_equal(prepared.reference_model.grid.vp, truth.repeat(2, 0).repeat(2, 1).repeat(2, 2))
    assert prepared.station_ids == stations.ids
    assert prepared.event_ids == events.ids
    np.testing.assert_array_equal(prepared.station_locs, stations.coordinates_m)
    observed = {(a.event_id, a.station_id): a.arrival_time_s for a in load_arrivals(root, "small")}
    np.testing.assert_array_equal(prepared.arrivals_table, [
        [observed[(event, station)] for station in stations.ids] for event in events.ids
    ])
    assert not hasattr(prepared, "reference_event_coordinates_m")


def test_saved_run_passes_exact_stations_and_no_event_truth(small_experiment, monkeypatch):
    root, _, stations, events = small_experiment
    calls = []
    monkeypatch.setattr(experiment_runner, "warm_up_jit", lambda: calls.append("warmup"))
    monkeypatch.setattr(experiment_runner, "run_em", lambda **kwargs: calls.append(kwargs) or None)
    config = replace(CONFIG, cell_size=1000., n_cycles=1, n_workers=1, save_runs=False)
    assert main(config, experiment_id="small", experiments_root=root) is None
    assert calls[0] == "warmup"
    passed = calls[1]
    assert passed["event_locs"] is None
    assert passed["station_locs"] == [tuple(c) for c in stations.coordinates_m]
    assert passed["arrivals_table"].shape == (len(events.ids), len(stations.ids))
    assert passed["initial_model"].grid.vp.shape == (2, 2, 2)
    assert np.all(passed["initial_model"].grid.vp == 5000.)
    assert passed["run_name"].endswith("_small")
    assert passed["true_model"] is not passed["initial_model"]
    assert "true_model_fine" not in passed


def test_validate_only_skips_jit_and_inversion(small_experiment, monkeypatch):
    root = small_experiment[0]
    def forbidden():
        raise AssertionError("must not run the inversion")
    monkeypatch.setattr(experiment_runner, "warm_up_jit", forbidden)
    config = replace(CONFIG, cell_size=1000.)
    prepared = main(config, experiment_id="small", experiments_root=root, validate_only=True)
    assert prepared.arrivals_table.shape == (3, 3)


def test_cli_validate_only_on_small_experiment(small_experiment):
    root = small_experiment[0]
    completed = subprocess.run(
        [sys.executable, "main.py", "small", "--experiments-root", str(root),
         "--cell-size-m", "1000", "--validate-only"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "3 events, 3 stations, inversion grid (2, 2, 2)" in completed.stdout


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
