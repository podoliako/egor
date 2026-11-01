"""Regional profile, train-only native exports, and non-executing launch audit."""
import csv
from datetime import datetime
import json

import numpy as np
import pytest

from studies.geysers_real_picks import finalize as f
from studies.geysers_real_picks.tomodd_inputs import Domain, GradientModel, _model_files
from studies.geysers_real_picks.run_real import main


def test_gradient_nodes_and_continuation():
    model = GradientModel((1.166, 4.166, 9.166), (4.1, 5.47, 5.75))
    domain = Domain(38.80275, -122.77070, 135, (23, 11, 7), 1166, (13, 3.5))
    files, (nx, ny, nz) = _model_files(domain, model)
    lines = files['MOD'].splitlines()
    depths = np.fromstring(lines[3], sep=' ')
    vp = np.array([np.fromstring(s, sep=' ') for s in lines[4:4 + ny * nz]]).reshape(nz, ny, nx)
    expected = np.interp(depths + 1.166, model.depths_km, model.vp_km_s)
    np.testing.assert_allclose(vp, np.broadcast_to(expected[:, None, None], vp.shape), atol=5.1e-7)
    np.testing.assert_allclose(depths[1:-1], np.arange(7) + .5 - 1.166)
    assert vp[0, 0, 0] == 4.1
    assert vp[-1, 0, 0] == 5.75


@pytest.mark.parametrize('depth,speed', [((0,), (4,)), ((0, 0), (4, 5)), ((0, 1), (4, -1)),
                                        ((0, float('nan')), (4, 5)), ((0, 1), (4,))])
def test_bad_gradient(depth, speed):
    with pytest.raises(ValueError): GradientModel(depth, speed)


def test_official_profile():
    profile = f.regional_profile()
    assert profile['source_depth_km'] == [0, 3, 8, 20, 22]
    assert profile['vp_km_s'] == [4.1, 5.47, 5.75, 6.02, 7.9]
    assert profile['local_depth_km'] == [1.166, 4.166, 9.166, 21.166, 23.166]


def test_native_actual_split_and_initialization():
    directory = f.DATA / 'methods'
    if not directory.exists(): pytest.skip('Prepared real data not installed')
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        assert f.sha(directory / name) == digest
    assert manifest['inversions_launched'] is False
    native = json.loads((directory / 'tomodd/manifest.json').read_text())
    assert native['noise']['sigma_km'] == 0
    assert native['missing_support_files'] == []
    assert native['absolute_observations'] == 42973
    assert native['domain']['horizontal_offset_km'] == [13, 3.5]
    for event in native['events']:
        assert event['catalog_xyz_km'] == event['initial_xyz_km']
    ids = {int(e['event_id']) for e in native['events']}
    aliases = {s['alias']: ('BG', s['station']) for s in native['stations']}
    train = {(int(r['event_id']), r['network'], r['station']): r for r in f.rows(f.DATA / 'train.csv')}
    held = {(int(r['event_id']), r['network'], r['station']) for r in f.rows(f.DATA / 'holdout.csv')}
    assert {k[0] for k in train} == ids
    origin = {e['event_id']: datetime.fromisoformat(e['export_origin_time_utc'].replace('Z', '+00:00'))
              for e in native['events']}
    absolute = {}
    for line in (directory / 'tomodd/absolute.dat').read_text().splitlines():
        words = line.split()
        if words[0] == '#':
            event = int(words[1])
        else:
            key = (event, *aliases[words[0]])
            assert key not in absolute and key not in held and key in train
            picked = datetime.fromisoformat(train[key]['pick_time_utc'].replace('Z', '+00:00'))
            assert float(words[1]) == (picked - origin[event]).total_seconds()
            absolute[key] = float(words[1])
    assert absolute.keys() == train.keys()
    count = 0
    for line in (directory / 'tomodd/dt.ct').read_text().splitlines():
        words = line.split()
        if words[0] == '#':
            a, b = map(int, words[1:])
        else:
            station = aliases[words[0]]
            assert (a, *station) in train and (b, *station) in train
            assert (a, *station) not in held and (b, *station) not in held
            assert float(words[1]) == absolute[a, *station]
            assert float(words[2]) == absolute[b, *station]
            count += 1
    assert count == native['differential_observations'] == 54317
    start = np.load(directory / 'initial_velocity_m_s.npy', allow_pickle=False)
    assert start.shape == (23, 11, 7)
    lines = (directory / 'tomodd/MOD').read_text().splitlines()
    _, nx, ny, nz = lines[0].split()
    nx, ny, nz = map(int, (nx, ny, nz))
    vp = np.array([np.fromstring(s, sep=' ') for s in lines[4:4 + ny * nz]]).reshape(nz, ny, nx)
    np.testing.assert_allclose(start[0, 0], vp[1:-1, 1, 1] * 1000, atol=.00051)
    configs = json.loads((directory / 'emtomo_configs.json').read_text())
    for mode, config in configs.items():
        assert config['candidate_mode'] == mode
        assert config['weight_noise_absolute_sigma_s'] is None
        assert config['n_candidates'] == 128 and config['n_cycles'] == 24
        assert config['initial_layer_boundaries_km'] is None
    assert configs['soft']['temperature_schedule'] == [4.] * 4 + [2.] * 4 + [1.] * 16
    assert configs['hard']['temperature_schedule'] is None


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_validate_only_default(mode, monkeypatch):
    import experiment_runner
    monkeypatch.setattr(experiment_runner, 'run_em', lambda *a, **kw: pytest.fail('Solver must not execute'))
    result = main([mode])
    assert result.reference_model is None
    with pytest.raises(SystemExit): main([mode, '--execute'])
