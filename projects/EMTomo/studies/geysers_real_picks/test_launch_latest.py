"""Solver-free regressions for the new real-pick queue."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from config import InversionConfig
from experiment_runner import initial_velocity
from studies.geysers_real_picks.launch_latest import (
    build_config, control_descriptor, check_finished, refine_cells, build_refined_config,
)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_refinement_preserves_calculation_grid_and_parent_parameters(mode, tmp_path):
    source = asdict(InversionConfig(
        candidate_mode=mode, cell_size=1000., subdivision=8,
        initial_gradient_m_s=[3500., 6000.], n_candidates=50, weights_top_n=50,
        weights_min_distance=7, n_cycles=7, n_workers=12, linear_solver='auto'))
    refined = build_refined_config(source, mode, tmp_path)
    allowed = {'cell_size', 'subdivision', 'initial_gradient_m_s', 'runs_dir', 'run_name'}
    assert all(value == source[key] for key, value in refined.items() if key not in allowed)
    assert refined['cell_size'] == 500 and refined['subdivision'] == 4
    assert refined['cell_size'] / refined['subdivision'] == source['cell_size'] / source['subdivision'] == 125
    assert refined['initial_gradient_m_s'] is None
    from tomography.tomography_em import _normal_solver
    assert _normal_solver(InversionConfig(**refined), 46 * 22 * 14) == 'matrix_free'
    with pytest.raises(ValueError, match='mode'):
        build_refined_config(source, 'hard' if mode == 'soft' else 'soft', tmp_path)


def test_refine_cells_is_exact_at_unchanged_fmm_grid():
    parent = np.random.default_rng(42).uniform(3500, 6000, (3, 2, 4))
    initial = refine_cells(parent)
    assert initial.shape == (6, 4, 8)
    np.testing.assert_array_equal(initial.repeat(4, 0).repeat(4, 1).repeat(4, 2),
                                  parent.repeat(8, 0).repeat(8, 1).repeat(8, 2))
    np.testing.assert_array_equal(parent, initial[::2, ::2, ::2])
    assert not np.shares_memory(parent, initial)


@pytest.mark.parametrize('bad', [np.ones((2, 2)), np.full((2, 2, 2), np.nan), np.zeros((2, 2, 2))])
def test_refine_cells_rejects_invalid_parent(bad):
    with pytest.raises(ValueError, match='parent velocity'):
        refine_cells(bad)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_inherit_parameters_except_explicit_geometry_start_and_output(mode, tmp_path):
    source = asdict(InversionConfig(
        cell_size=800., n_candidates=50, weights_top_n=50, weights_min_distance=7,
        n_workers=12, n_cycles=7, weight_noise_relative_sigma=.01,
        weight_noise_absolute_sigma_s=.05, weight_model_sigma_s=.02,
        lambda_reg=1., coverage_damping_power=0., max_velocity_step_fraction=.02,
        initial_layer_boundaries_km=[.8], initial_layer_velocities_m_s=[3500., 6000.]))
    source['n_events'] = 3465
    info = dict(cell_size_m=1000., vp_at_top_boundary_m_s=3500., vp_at_bottom_boundary_m_s=6000.)
    new, changes = build_config(source, mode, info, tmp_path)
    assert 'n_events' not in new
    for key, value in new.items():
        if key not in changes:
            assert value == source[key]
    config = InversionConfig(**new)
    assert config.candidate_mode == mode
    assert config.initial_layer_boundaries_km is None
    assert config.n_workers == 12 and config.control_fraction == .1
    model = initial_velocity(config, (23, 11, 7))
    np.testing.assert_allclose(model[0, 0], 3500 + 2500 * (np.arange(7) + .5) / 7)


def test_nested_control_is_common_and_preserves_original_missing_picks():
    arrivals = np.arange(36, dtype=float).reshape(3, 12)
    arrivals[0, 6:] = np.nan
    original = arrivals.copy()
    config = InversionConfig(control_fraction=.1, control_min_training_picks=5)
    mask, descriptor = control_descriptor(arrivals, config)
    np.testing.assert_array_equal(arrivals, original)
    assert not np.any(mask & ~np.isfinite(original))
    assert descriptor['n_control'] == 3 and descriptor['n_training'] == 27
    other_mask, other = control_descriptor(arrivals, config)
    np.testing.assert_array_equal(mask, other_mask)
    assert descriptor == other


def test_completion_accepts_early_stop_not_seven_required_updates(tmp_path):
    config = InversionConfig(n_cycles=7)
    initial = np.full((2, 2, 2), 5000.)
    mask = np.zeros((1, 6), dtype=bool)
    mask[0, 0] = True
    np.save(tmp_path / 'initial_model.npy', initial)
    np.save(tmp_path / 'final_model.npy', initial)
    np.save(tmp_path / 'control_mask.npy', mask)
    for i in range(3):
        directory = tmp_path / f'iter_{i}'
        directory.mkdir()
        (directory / 'complete.json').write_text(json.dumps({'iter': i}))
        (directory / 'control.json').write_text(json.dumps(dict(
            rmse_s=.1, n_control=1, insufficient_steps=i, stop=i == 2)))
    stopping = dict(completed_updates=3, reason='control_rmse_plateau', control_rmse_s=.1)
    (tmp_path / 'stopping.json').write_text(json.dumps(stopping))
    assert check_finished(tmp_path, config, initial, mask, {'n_control': 1}) == stopping
    (tmp_path / 'iter_1/complete.json').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        check_finished(tmp_path, config, initial, mask, {'n_control': 1})
