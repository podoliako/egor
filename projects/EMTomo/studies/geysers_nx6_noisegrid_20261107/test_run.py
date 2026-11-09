"""Solver-free checks that actual noise cannot leak into the assumed likelihood."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import asdict

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("nx6_noisegrid_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from config import InversionConfig
from experiment_runner import resolve_pick_noise


def reference():
    return asdict(InversionConfig(n_cycles=12, n_candidates=50, weights_top_n=50,
                                  weights_min_distance=7, cell_size=1000, subdivision=8,
                                  temperature=1.0, n_workers=4, weight_model_sigma_s=0.02,
                                  weight_noise_relative_sigma=None,
                                  weight_noise_absolute_sigma_s=None))


@pytest.mark.parametrize("factor", [0.5, 0.75, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_assumed_noise_is_fixed_despite_actual_noise(factor, mode):
    original = reference()
    config = InversionConfig(**runner.case_config(original, str(factor), mode))
    noise = resolve_pick_noise(config, (0.01 * factor, 0.05 * factor))
    assert asdict(noise) == runner.ASSUMED
    assert config.n_cycles == 12
    assert (config.n_candidates, config.weights_top_n, config.weights_min_distance) == (50, 50, 7)
    assert config.temperature == 1
    assert config.candidate_mode == mode
    assert original["weight_noise_relative_sigma"] is None
    assert original["weight_noise_absolute_sigma_s"] is None


def test_soft_hard_only_differ_by_mode_and_name():
    soft = runner.case_config(reference(), "x2", "soft")
    hard = runner.case_config(reference(), "x2", "hard")
    assert {k for k in soft if soft[k] != hard[k]} == {"candidate_mode", "run_name"}
