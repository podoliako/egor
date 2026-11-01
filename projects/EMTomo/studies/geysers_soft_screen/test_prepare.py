"""Plan/subset/noise tests with tiny local arrays; never calculate forward times."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('soft_screen_prepare', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_all_24_proposals_and_13_controls():
    specs = prepare.soft_specs()
    assert len(specs) == 24
    keys = set()
    for s in specs:
        cycles = s.get('n_cycles', 12)
        if 'schedule' in s:
            assert len(s['schedule']) == cycles
            assert s['schedule'][-1] == 1
        keys.add((s['k'], s['n'], s['d'], s.get('lambda_reg', 1),
                  s.get('coverage_damping_power', 0), cycles,
                  s.get('stations', 26), s.get('noise', 0)))
    assert len(keys) == 13
    assert [s['temperature'] for s in specs[:6]] == [2, 4, 2, 4, 2, 4]
    assert specs[7]['schedule'] == [4]*4 + [2]*4 + [1]*4
    assert specs[15]['schedule'] == [4]*4 + [2]*4 + [1]*16
    assert [s.get('stations', 26) for s in specs[16:]] == [12, 12, 8, 8, 26, 26, 12, 8]


def test_nested_station_selection_uses_xy_only():
    xyz = np.random.default_rng(3).normal(size=(26, 3))
    order = prepare.station_order(xyz)
    assert sorted(order) == list(range(26))
    assert set(order[:8]) < set(order[:12])
    xyz[:, 2] += np.arange(26)*100000
    assert order == prepare.station_order(xyz)
    assert order == prepare.station_order(xyz.copy())


@pytest.mark.parametrize('sigma', [0., .02])
def test_derivation_noise_then_normalization_no_forward(tmp_path, monkeypatch, sigma):
    from projects.forward_modeling.experiments import save_inputs
    from projects.forward_modeling.model import VelocityGrid, PointSet
    from projects.EMTomo.experiment_data import load_tomography_experiment
    from projects.forward_modeling.noise import NoiseConfig, add_arrival_noise
    from projects.forward_modeling import solver
    monkeypatch.setattr(prepare, 'EXPERIMENTS', tmp_path)
    monkeypatch.setattr(prepare, 'SOURCE_ID', 'toy')
    def no_forward(*args, **kwargs):
        pytest.fail('A derived dataset must not run the forward solver')
    monkeypatch.setattr(solver, 'compute_travel_times', no_forward)
    model = VelocityGrid(np.full((2, 2, 2), 4000.), 1000.)
    stations = PointSet(('A', 'B', 'C', 'D'), np.array([[0, 0, 0], [2000, 0, 0],
                                                                  [0, 2000, 0], [2000, 2000, 0]]))
    events = PointSet(('e1', 'e2'), np.array([[500, 500, 500], [1500, 1500, 1500]]))
    save_inputs(tmp_path, 'toy', model, stations, events, {'toy': True})
    output = tmp_path / 'output/toy'
    output.mkdir(parents=True)
    (output / 'clean_times.npz').write_bytes(b'source-cache-provenance')
    metadata = dict(noise={'enabled': False, 'model': 'independent_gaussian_relative_plus_absolute'},
                    units=dict(coordinates='m', cell_size='m', velocity='m/s',
                   arrival_time='s', elapsed_time='s'), time_reference='earliest_station_arrival_per_event')
    clean = np.array([[.3, 2., 1., 3.], [1., .3, 2., 1.5]])
    indices = [0, 2]
    identifier = prepare.prepare_dataset(2, sigma, model, stations, events, clean, indices, metadata, {})
    data = load_tomography_experiment(identifier, tmp_path)
    selected = clean[:, indices]
    if sigma:
        selected = add_arrival_noise(selected, events.ids, ('A', 'C'), NoiseConfig(sigma, sigma, 42))
    np.testing.assert_array_equal(data.arrival_times_s, selected-selected.min(axis=1, keepdims=True))
    assert data.station_ids == ('A', 'C') and data.noise_sigmas == (sigma, sigma)
    assert json.loads((tmp_path / 'output' / identifier / 'metadata.json').read_text())['no_new_forward_solve']
    assert prepare.prepare_dataset(2, sigma, model, stations, events, clean, indices, metadata, {}) == identifier
