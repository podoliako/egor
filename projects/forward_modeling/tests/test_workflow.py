"""End-to-end checks with the actual FMM, not an I/O solver mock."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from projects.forward_modeling import (
    ForwardConfig, PointSet, VelocityGrid, load_arrivals, run_experiment, save_inputs,
)


def _save_case(root):
    model = VelocityGrid(np.full((6, 6, 6), 2000.), 100.)
    stations = PointSet(["001", "edge"], [[599.1, 400.4, 570.2], [0., 0., 0.]])
    events = PointSet(["event"], [[110.2, 150.5, 190.1]])
    save_inputs(root, "case", model, stations, events)


def test_real_cli_roundtrip_with_accuracy_check(tmp_path):
    _save_case(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "projects.forward_modeling", "case", "--root", str(tmp_path),
         "--refinement", "1", "--max-difference-s", "0.1"],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    arrivals = load_arrivals(tmp_path, "case")
    assert {a.station_id for a in arrivals} == {"001", "edge"}
    assert min(a.arrival_time_s for a in arrivals) == 0
    metadata = json.loads((tmp_path / "output" / "case" / "metadata.json").read_text())
    assert metadata["convergence"]["max_absolute_time_difference_s"] < 0.1
    assert metadata["convergence"]["fine_refinement"] == 2
    assert metadata["max_difference_s"] == 0.1
    assert metadata["solver_name"] == "pykonal.EikonalSolver"


def test_real_solver_rejects_unconverged_result(tmp_path):
    _save_case(tmp_path)
    with pytest.raises(RuntimeError, match="(?i)convergence"):
        run_experiment(tmp_path, "case", ForwardConfig(refinement=1), max_difference_s=1e-12)
    assert not (tmp_path / "output" / "case").exists()
    assert list((tmp_path / "output").iterdir()) == []
    assert (tmp_path / "input" / "case" / "model.npz").is_file()
