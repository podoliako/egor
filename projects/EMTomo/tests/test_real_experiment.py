"""Real training input cannot supply event priors, truth, or held-out observations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import InversionConfig
from experiment_data import load_real_pick_experiment
import experiment_runner as runner
from velocity_model import VelocityModel


@pytest.fixture
def archive(tmp_path):
    arrays = dict(station_ids=np.array(['BG.A', 'BG.B', 'BG.C']), event_ids=np.array(['1', '2']),
                  station_coordinates_m=np.array([[0, 0, 0], [500, 500, 0], [1000, 500, 0]]),
                  arrival_times_s=np.array([[0., np.nan, .5], [np.nan, .3, 0.]]))
    metadata = dict(schema='emtomo.real_picks.v1', domain_extent_m=[2000, 2000, 2000],
                    time_reference='earliest_training_pick_per_event', minimum_training_picks=2,
                    units={'coordinates': 'm', 'arrival_time': 's'})
    def write():
        np.savez(tmp_path / 'arrivals.npz', **arrays)
        metadata['arrays_sha256'] = hashlib.sha256((tmp_path / 'arrivals.npz').read_bytes()).hexdigest()
        (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    write()
    return tmp_path, arrays, metadata, write


def test_sparse_training_only(archive):
    path, arrays, _, _ = archive
    data = load_real_pick_experiment(path)
    np.testing.assert_array_equal(data.arrival_times_s, arrays['arrival_times_s'])
    assert not data.arrival_times_s.flags.writeable
    assert not hasattr(data, 'reference_event_coordinates_m')
    assert not hasattr(data, 'source_model')
    assert not hasattr(data, 'origin_time')


@pytest.mark.parametrize('change', ['inf', 'negative', 'few', 'nonzero', 'shape', 'ids', 'bounds', 'truth'])
def test_invalid_archive(archive, change):
    path, arrays, _, write = archive
    if change == 'inf': arrays['arrival_times_s'][0, 1] = np.inf
    if change == 'negative': arrays['arrival_times_s'][0, 0] = -.1
    if change == 'few': arrays['arrival_times_s'][0, 2] = np.nan
    if change == 'nonzero': arrays['arrival_times_s'][0, 0] = .1
    if change == 'shape': arrays['arrival_times_s'] = arrays['arrival_times_s'][:, :2]
    if change == 'ids': arrays['event_ids'][1] = '1'
    if change == 'bounds': arrays['station_coordinates_m'][0, 0] = -1
    if change == 'truth': arrays['event_coordinates_m'] = np.zeros((2, 3))
    write()
    with pytest.raises(ValueError): load_real_pick_experiment(path)


@pytest.mark.parametrize('field,value', [('schema', 'other'), ('time_reference', 'all_picks'),
                                        ('domain_extent_m', [0, 1, 2]), ('minimum_training_picks', True),
                                        ('units', {'coordinates': 'km'})])
def test_invalid_metadata(archive, field, value):
    path, _, meta, write = archive
    meta[field] = value
    write()
    with pytest.raises(ValueError): load_real_pick_experiment(path)


def test_hash(archive):
    path = archive[0]
    with (path / 'arrivals.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'): load_real_pick_experiment(path)


def test_real_runner_no_catalog_no_truth(archive, monkeypatch):
    path = archive[0]
    config = InversionConfig(cell_size=1000, weight_noise_relative_sigma=0,
                             weight_noise_absolute_sigma_s=.1)
    start = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    prepared = runner.run_real_experiment(path, config, initial_model=start, validate_only=True)
    assert prepared.reference_model is None and prepared.noise_sigmas is None
    assert not hasattr(prepared, 'event_locs')
    monkeypatch.setattr(runner, 'warm_up_jit', lambda: None)
    calls = []
    monkeypatch.setattr(runner, 'run_em', lambda *a, **kw: calls.append((a, kw)))
    runner.run_real_experiment(path, config, initial_model=start)
    args, kw = calls[0]
    assert kw['reference_model'] is None
    assert kw['source_experiment']['event_ids'] == ['1', '2']
    np.testing.assert_array_equal(args[2], prepared.arrivals_table)
    with pytest.raises(ValueError, match='explicit'):
        runner.run_real_experiment(path, replace(config, weight_noise_absolute_sigma_s=None), initial_model=start)
    with pytest.raises(ValueError, match='tile'):
        runner.prepare_real_inversion(path, config, initial_model=VelocityModel(np.ones((1, 1, 1)), 1000.))


def test_real_artifact():
    path = Path(__file__).resolve().parents[1] / 'data/geysers_2011/real_experiment_v1/emtomo'
    if not path.exists(): pytest.skip('Prepared real data not installed')
    data = load_real_pick_experiment(path)
    assert data.arrival_times_s.shape == (3471, 31)
    assert np.isfinite(data.arrival_times_s).sum() == 42973
    assert np.min(np.isfinite(data.arrival_times_s).sum(axis=1)) == 6
    assert data.domain_extent_m == (23000, 11000, 7000)
