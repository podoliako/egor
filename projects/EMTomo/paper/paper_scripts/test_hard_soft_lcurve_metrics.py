"""Small artifact-only regressions; no tomography or forward solver imports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'lcurve_metrics', Path(__file__).with_name('make_hard_soft_lcurve_metrics.py'))
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def test_volume_uses_original_discontinuities_not_slowness_average():
    truth = np.array([4000., 6000.]).reshape(2, 1, 1)
    truth = np.repeat(np.repeat(truth, 2, axis=1), 2, axis=2)
    model = np.array([[[4800.]]])
    assert metrics.volume_reference(model, 2., truth, 1.) == pytest.approx(
        np.sqrt((800.**2 + 1200.**2) / 2))
    # Diagnostic reference would be 4800 and have zero error, a different metric.
    assert metrics.rmse(model, np.array([[[4800.]]])) == 0


def test_aligned_checkerboard_volume():
    truth = np.array([[[4750.]], [[5250.]]])
    model = np.full((4, 2, 2), 5000.)
    assert metrics.volume_reference(model, 1., truth, 2.) == 250.


def test_soft_map_is_largest_weight_not_mean_or_nearest_truth():
    positions = np.array([[0, 0, 0], [2, 0, 0]])
    truth = np.array([.5, .5, .5])
    assert metrics.map_error(positions, [.1, .9], truth, 1., [3, 1, 1]) == 2.
    assert metrics.map_error(positions, [.5, .5], truth, 1., [3, 1, 1]) == 0.
    with pytest.raises(ValueError, match='Invalid saved hypotheses'):
        metrics.map_error(positions, [.1, .1], truth, 1., [3, 1, 1])


def test_exact_legacy_mappings_and_exclusions():
    selections = metrics.rows(metrics.SELECTIONS)
    legacy = {name: metrics.read(metrics.ROOT / f'paper/figures/{name}_comparison_metrics.json')
              for name in ('onesided', 'two_clusters', 'real_geometry', 'fault')}
    tables = metrics.table_pairs(selections, legacy)
    assert [len(t) for t in tables.values()] == [4, 4, 5, 5, 5, 4]
    assert tables['real_geometry_noise'][2] is tables['real_geometry_candidates'][1]
    assert all(r['pair_id'] != 'section4_pair06' for t in tables.values() for r in t)
    assert len({r['pair_id'] for t in tables.values() for r in t}) == 26
    assert len({r['hard_run_id'] for r in tables['fault_retention']}) == 1
    assert [r['soft_weights_top_n'] for r in tables['fault_retention']] == ['40', '50', '75', '100', '128']


def test_report_uses_iter_k_for_models_and_weights_and_k_minus_one_for_quality():
    report = metrics.read(metrics.OUTPUT)
    assert report['sections'] == ['4.2', '4.3', '4.4', '4.5']
    assert len(report['records']) == 44
    for record in report['records']:
        k = record['selected_update_k']
        assert record['model_path'].endswith(f'/iter_{k}/model.npy')
        assert record['weights_directory'].endswith(f'/iter_{k}')
        assert record['weights_iteration_index'] == k
        assert record['quality_iteration_index'] == k - 1
        assert record['complete_iteration_indices'] == list(range(12))
        assert record['block_slowness_reference_rmse_m_s'] == pytest.approx(record['quality_rms_m_s'])
    separation = report['tables']['fault_separation']
    assert [(r['hard']['selected_update_k'], r['soft']['selected_update_k'])
            for r in separation] == [(5, 5), (5, 5), (4, 5), (4, 5)]
