"""Prepare or run the clean Geysers faulted-layer forward experiment only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parents[1]))
from projects.forward_modeling.experiments import load_inputs, run_experiment, save_inputs
from projects.forward_modeling.model import ForwardConfig, VelocityGrid
from projects.EMTomo.experiment_data import load_tomography_experiment

EXPERIMENTS = ROOT.parent / 'forward_modeling/experiments'
SOURCE_ID = 'geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001'
EXPERIMENT_ID = 'geysers11_20110401-20110420_fault3_x6p5_elev1166_26st_3776ev_clean'
STATE = ROOT / 'runs/geysers_faulted_layers'
PARAMETERS = {
    'extent_km': [20., 7., 7.], 'fault_x_km': 6.5,
    'left': {'boundaries_km': [.7, 1.9], 'vp_km_s': [3.6, 4.4, 5.]},
    'right': {'boundaries_km': [1.1, 2.9], 'vp_km_s': [4., 4.8, 5.5]},
    'initial': {'boundaries_km': [.85, 2.3], 'vp_km_s': [3.85, 4.55, 5.3]},
}
CONFIG = ForwardConfig(refinement=2, source_radius_cells=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_model(cell_size_m=125.):
    shape = tuple(int(round(km * 1000 / cell_size_m)) for km in PARAMETERS['extent_km'])
    if not np.allclose(np.asarray(shape) * cell_size_m, np.asarray(PARAMETERS['extent_km']) * 1000):
        raise ValueError('Cell size must divide all domain lengths')
    x = (np.arange(shape[0]) + .5) * cell_size_m / 1000
    z = (np.arange(shape[2]) + .5) * cell_size_m / 1000
    profiles = {}
    for side in ('left', 'right'):
        block = PARAMETERS[side]
        profiles[side] = np.asarray(block['vp_km_s'])[np.searchsorted(block['boundaries_km'], z, side='right')] * 1000
    field = np.where(x[:, None] < PARAMETERS['fault_x_km'], profiles['left'], profiles['right'])
    return VelocityGrid(np.broadcast_to(field[:, None, :], shape), cell_size_m)


def prepare(experiments_root=EXPERIMENTS):
    source_dir = experiments_root / 'input' / SOURCE_ID
    source_model, stations, events = load_inputs(experiments_root, SOURCE_ID)
    if (len(stations.ids), len(events.ids)) != (26, 3776):
        raise ValueError('Expected the unchanged 26-station / 3776-event elevated geometry')
    # Pin the geometry used in the completed noise series, not a new catalogue selection.
    expected = {'stations.csv': '59895d722cff63e2f087438704631ae6740f2f76dd7065ce64bad8e6b69be4e3',
                'events.csv': 'd7f6af385a08edf1b4f622e35c08d6f40bbc06d2d60e3ab9e42a730e3dedd786'}
    if any(sha(source_dir / name) != digest for name, digest in expected.items()):
        raise ValueError('Source station/event geometry differs from the approved Geysers series')
    model = build_model()
    if source_model.velocity.shape != model.velocity.shape or source_model.cell_size_m != model.cell_size_m:
        raise ValueError('Unexpected source grid')
    source_generation = json.loads((source_dir / 'generation.json').read_text())
    effective = {}
    for side in ('left', 'right'):
        ix = 0 if side == 'left' else -1
        profile = model.velocity[ix, 0]
        effective[side] = ((np.flatnonzero(np.diff(profile)) + 1) * model.cell_size_m / 1000).tolist()
    generation = {
        'generator': 'geysers_faulted_layers/experiment.py', 'experiment_id': EXPERIMENT_ID,
        'source_experiment_id': SOURCE_ID,
        'source_input_sha256': {name: sha(source_dir / name) for name in
                                ('model.npz', 'stations.csv', 'events.csv', 'generation.json')},
        'station_count': len(stations.ids), 'event_count': len(events.ids),
        'pair_count': len(stations.ids) * len(events.ids),
        'selection': 'All stations and events from the elevated source, unchanged IDs/order/XYZ.',
        'parameters': PARAMETERS, 'effective_boundaries_km': effective,
        'cell_size_m': model.cell_size_m, 'array_axes': ['x', 'y', 'z'],
        'model_policy': 'Piecewise constant voxels sampled at cell centres; independent of y; no spheres.',
        'noise': {'enabled': False}, 'forward_config': asdict(CONFIG),
        'initial_model_usage': 'Approximate 1-D parameters for future inversion ONLY; not used by forward solver.',
        'top_elevation_m': source_generation['top_elevation_m'],
        'vertical_coordinates': source_generation['vertical_coordinates'],
        'shared_datum_assumption': source_generation['shared_datum_assumption'],
    }
    destination = experiments_root / 'input' / EXPERIMENT_ID
    if destination.exists():
        saved_model, saved_stations, saved_events = load_inputs(experiments_root, EXPERIMENT_ID)
        matches = (np.array_equal(saved_model.velocity, model.velocity)
                   and saved_model.cell_size_m == model.cell_size_m
                   and saved_model.origin_m == model.origin_m
                   and json.loads((destination / 'generation.json').read_text()) == generation)
        for actual, expected_points in ((saved_stations, stations), (saved_events, events)):
            matches = matches and actual.ids == expected_points.ids and np.array_equal(actual.coordinates_m, expected_points.coordinates_m)
        if not matches:
            raise ValueError('Existing experiment differs; refusing to overwrite')
    else:
        save_inputs(experiments_root, EXPERIMENT_ID, model, stations, events, generation)
    return generation


def run(workers):
    STATE.mkdir(parents=True, exist_ok=True)
    status = {'experiment_id': EXPERIMENT_ID, 'status': 'running', 'workers': workers,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(STATE / 'status.json', status)
    try:
        prepare()
        print(f'Forward only: {EXPERIMENT_ID}; workers={workers}; noise=off', flush=True)
        output = run_experiment(EXPERIMENTS, EXPERIMENT_ID, CONFIG, workers=workers,
                                noise=None, save_absolute_times=True)
        loaded = load_tomography_experiment(EXPERIMENT_ID, EXPERIMENTS)
        metadata = json.loads((output / 'metadata.json').read_text())
        with np.load(output / 'clean_times.npz', allow_pickle=False) as cache:
            clean = cache['absolute_times_s']
            assert clean.shape == (3776, 26) and np.isfinite(clean).all() and (clean >= 0).all()
            assert tuple(cache['station_ids']) == loaded.station_ids
            assert tuple(cache['event_ids']) == loaded.event_ids
            np.testing.assert_array_equal(cache['station_coordinates_m'], loaded.station_coordinates_m)
            np.testing.assert_array_equal(cache['event_coordinates_m'], loaded.reference_event_coordinates_m)
            np.testing.assert_allclose(clean - clean.min(axis=1, keepdims=True), loaded.arrival_times_s, rtol=0, atol=1e-14)
        assert loaded.noise_sigmas == (0., 0.) and metadata['noise']['enabled'] is False
        assert sha(output / 'clean_times.npz') == metadata['absolute_times_cache']['sha256']
        status.update(status='completed', output_dir=str(output), pair_count=3776 * 26,
                      elapsed_seconds=metadata['elapsed_seconds'])
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(STATE / 'status.json', status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    else:
        run(args.workers)
