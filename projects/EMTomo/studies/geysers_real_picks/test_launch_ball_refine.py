"""Solver-free checks of ball-holdout continuation configuration."""
from dataclasses import asdict
import numpy as np
import pytest
from config import InversionConfig
from studies.geysers_real_picks.launch_ball_refine import continuation_config, refine_cells


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_continuation_retains_split_stopping_likelihood_and_parent_field(mode):
    original = asdict(InversionConfig(candidate_mode=mode, cell_size=1000., subdivision=8,
                                      n_cycles=15, control_fraction=0., n_workers=12,
                                      n_candidates=50, weights_top_n=50, weights_min_distance=7,
                                      initial_gradient_m_s=[3500., 6000.], coverage_damping_power=0.,
                                      lambda_reg=1., linear_solver='auto'))
    config = continuation_config(original, mode)
    allowed = {'cell_size', 'subdivision', 'initial_gradient_m_s', 'runs_dir', 'run_name'}
    assert {k for k in config if config[k] != original[k]} <= allowed
    assert config['cell_size'] == 500 and config['subdivision'] == 4
    assert config['n_cycles'] == 15 and config['control_fraction'] == 0
    parent = np.random.default_rng(42).uniform(3000, 6500, (3, 2, 4))
    refined = refine_cells(parent)
    np.testing.assert_array_equal(refined.repeat(4, 0).repeat(4, 1).repeat(4, 2),
                                  parent.repeat(8, 0).repeat(8, 1).repeat(8, 2))
    from tomography.tomography_em import _normal_solver
    assert _normal_solver(InversionConfig(**config), 46 * 22 * 14) == 'matrix_free'


def test_continuation_rejects_pick_control():
    original = asdict(InversionConfig(cell_size=1000., subdivision=8, n_cycles=15,
                                      candidate_mode='soft', control_fraction=.1))
    with pytest.raises(ValueError, match='ball-holdout parent'):
        continuation_config(original, 'soft')
