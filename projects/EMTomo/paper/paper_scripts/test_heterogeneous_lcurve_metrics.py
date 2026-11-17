"""Focused tests: saved metrics need no inversion or forward solver imports."""
import json
from pathlib import Path

import numpy as np
import pytest

from make_heterogeneous_lcurve_metrics import (
    AUDIT, OUTPUT, PRIMARY, arrival_matrix, map_error, pure_functions,
    verify_stopping, volume_reference,
)


def test_arrivals_follow_catalogue_order_not_arrival_or_sorted_order():
    arrivals = [dict(event_id=e, station_id=s, arrival_time_s=v)
                for e, s, v in [('a', 'x', 1), ('z', 'x', 2), ('a', 'y', 3), ('z', 'y', 4)]]
    np.testing.assert_array_equal(arrival_matrix(arrivals, ['z', 'a'], ['y', 'x']), [[4, 2], [3, 1]])
    with pytest.raises(ValueError, match='Duplicate arrival'):
        arrival_matrix(arrivals + arrivals[:1], ['z', 'a'], ['y', 'x'])
    with pytest.raises(ValueError, match='Missing synthetic'):
        arrival_matrix(arrivals[:-1], ['z', 'a'], ['y', 'x'])


def test_volume_rmse_preserves_subcell_truth_variance():
    truth = np.full((2, 2, 2), 3000.)
    truth[1] = 5000.
    model = np.full((1, 1, 1), 4000.)
    assert volume_reference(model, 2., truth, 1.) == 1000.


def test_map_uses_largest_weight_not_mean_and_first_tie():
    positions = [[0., 0., 0.], [1., 0., 0.]]
    assert map_error(positions, [.5, .5], [50., 50., 50.], 100., [2, 2, 2]) == 0.
    assert map_error(positions, [.2, .8], [50., 50., 50.], 100., [2, 2, 2]) == 100.


def test_ast_loader_never_executes_module_imports(tmp_path):
    source = tmp_path / 'helper.py'
    source.write_text('import nonexistent_solver\nraise RuntimeError("must not run")\n'
                      'def pure(x):\n    return np.asarray(x) + 1\n')
    namespace = pure_functions(source, ('pure',))
    np.testing.assert_array_equal(namespace['pure']([1, 2]), [2, 3])


def test_control_plateau_is_distinct_from_selected_update():
    params = dict(early_stopping_min_relative_improvement=.01, early_stopping_patience=2, n_cycles=20)
    controls = [dict(iter=i, rmse_s=value, insufficient_steps=i, stop=i == 2)
                for i, value in enumerate([1., .995, .994])]
    stopping = dict(completed_updates=3, reason='control_rmse_plateau', control_rmse_s=.994, insufficient_steps=2)
    verify_stopping(controls, stopping, params)
    with pytest.raises(ValueError, match='Invalid stopping summary'):
        verify_stopping(controls, dict(stopping, reason='max_cycles'), params)


def test_report_primary_audit_indices_truth_and_population():
    report = json.loads(OUTPUT.read_text())
    assert report['primary_pairs'] == list(PRIMARY)
    assert report['audit_pairs'] == list(AUDIT)
    assert report['experiment']['truth_cell_size_m'] == 100
    assert report['experiment']['extent_m'] == [20000, 8000, 8000]
    assert len(set(report['experiment']['event_ids'])) == 3465
    assert len(report['records']) == 8
    for record in report['records']:
        k = record['selected_update_k']
        assert record['weights_iteration_index'] == k
        assert record['quality_iteration_index'] == k - 1
        assert k < record['actual_updates'] <= record['requested_updates']
        assert record['n_events'] == 3465
        assert record['control_event_subset_count'] == 3244
        assert record['control_split']['n_control'] == 9009
        assert record['control_split']['n_training'] == 81081
        assert record['velocity_rmse_m_s'] != record['block_slowness_diagnostic_rmse_m_s']
        for key in ('model', 'initial', 'control_mask', 'meta', 'manifest'):
            assert report['input_maps'][record['run_id']][key] in report['source_files_sha256']
        assert len(record['row_id_weights_sha256']) == 64
    primary = [r for r in report['records'] if r['role'] == 'primary']
    assert {r['start'] for r in primary} == {'supplied_1d', 'homogeneous_5000_m_s'}
    assert [(r['selected_update_k'], r['actual_updates']) for r in primary] == [(2, 9), (2, 9), (5, 20), (5, 20)]
    assert all(r['frozen_csv_row']['included_in_plot'] == 'False'
               for r in primary if r['start'] == 'homogeneous_5000_m_s')
