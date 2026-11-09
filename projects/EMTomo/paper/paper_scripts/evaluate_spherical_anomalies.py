"""Read saved artifacts for exactly spherical_anomalies_comparison.json.

python -B paper/paper_scripts/evaluate_spherical_anomalies.py
python -B paper/paper_scripts/evaluate_spherical_anomalies.py --check

Only the new metrics JSON can be written (exclusive creation). No solver,
plotting module, live input, or inverse/forward computation is imported/run.
"""
from __future__ import annotations

import argparse

import csv
import hashlib
import io
import itertools
import json
import re
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

from evaluate_metrics import native_helpers, positions_to_metres, require, sha256

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
STUDY = ROOT / 'runs/matched_geometry_20261021'
DD = ROOT / 'runs/tomodd_guard_fix_20261027'
SNAPSHOT = STUDY / 'snapshot'
FIGURE = PAPER / 'figures/spherical_anomalies_comparison.json'
OUTPUT = PAPER / 'figures/spherical_anomalies_metrics.json'
SHAPE = (160, 56, 56)
CENTERS = tuple((np.arange(n) + .5) * .125 for n in SHAPE)
FILES = {}


def record(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    require(expected is None or digest == expected, f'Hash mismatch: {path}')
    key = str(path.relative_to(ROOT))
    require(key not in FILES or FILES[key] == digest, f'Input changed: {path}')
    FILES[key] = digest
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def points(path, id_key):
    with record(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = tuple(row[id_key] for row in rows)
    xyz = np.array([[float(row[f'{a}_m']) for a in 'xyz'] for row in rows])
    require(len(ids) == len(set(ids)) and np.isfinite(xyz).all(), 'Invalid catalog')
    return ids, xyz


def numeric_ids(ids):
    require(all(re.fullmatch(r'(?:EVT_)?[0-9]+', value) for value in ids), 'Unsupported ID')
    result = np.array([int(value.removeprefix('EVT_')) for value in ids])
    require(len(set(result)) == len(ids) and (result > 0).all(), 'Nonunique numeric IDs')
    return result


def align(rows, ids):
    require(rows.shape == (len(ids), 4) and np.isfinite(rows).all(), 'Invalid DD catalog')
    require(len(set(rows[:, 0])) == len(ids) and set(rows[:, 0]) == set(ids), 'DD ID mismatch')
    by_id = {value: i for i, value in enumerate(rows[:, 0])}
    return rows[[by_id[value] for value in ids]]


def geographic_xyz(rows, mapping):
    lat, lon = mapping['center_lat'], mapping['center_lon']
    return np.column_stack(((rows[:, 2] - lon) * 111 * np.cos(np.deg2rad(lat)) + 10,
                            (rows[:, 1] - lat) * 111 + 3.5, rows[:, 3])) * 1000


def p90(xyz, truth):
    errors = np.linalg.norm(xyz - truth, axis=1)
    require(np.isfinite(errors).all() and len(errors) == len(truth), 'Invalid errors')
    return float(np.quantile(errors, .9, method='linear'))


def velocity(field, truth, masks):
    require(field.shape == truth.shape and np.isfinite(field).all() and (field > 0).all(),
            'Invalid native velocity')
    return {name + '_rmse_m_s': float(np.sqrt(np.mean((field[mask] - truth[mask])**2)))
            for name, mask in masks.items()}


def manual_trilinear(nodes, axes, query):
    """Independent eight-corner evaluator; includes fixed exterior intervals."""
    lower = [np.searchsorted(a, query[:, j], side='right') - 1 for j, a in enumerate(axes)]
    require(all(((i >= 0) & (i < len(a) - 1)).all() for i, a in zip(lower, axes)),
            'Interpolation would extrapolate')
    fractions = [(query[:, j] - a[i]) / (a[i + 1] - a[i])
                 for j, (a, i) in enumerate(zip(axes, lower))]
    value = np.zeros(len(query))
    for bits in itertools.product((0, 1), repeat=3):
        w = np.prod([f if b else 1 - f for f, b in zip(fractions, bits)], axis=0)
        value += w * nodes[tuple(i + b for i, b in zip(lower, bits))]
    return value


def em_locations(run, meta, ids, truth):
    cycle = run / 'iter_11'
    require({p.name for p in cycle.glob('event_*') if p.is_dir()} ==
            {f'event_{j}' for j in range(len(ids))}, 'Missing/extra EM event rows')
    digest, maps, means, sums, counts = hashlib.sha256(), [], [], [], []
    for j, event_id in enumerate(ids):
        path = cycle / f'event_{j}/weights.npz'
        payload = path.read_bytes()
        digest.update(f'{j}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['weight_shape'], SHAPE)
            xyz = positions_to_metres(saved['positions'], meta['grid_info']['fine_cell_size'])
            w = np.asarray(saved['weight_values'], dtype=float)
        require(w.shape == (len(xyz),) and 1 <= len(w) <= meta['run_params']['weights_top_n']
                and np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, 'Bad E-step mass')
        require(((xyz >= 0) & (xyz <= [20000, 7000, 7000])).all(), 'E-step outside volume')
        sums.append(float(w.sum()))
        counts.append(len(w))
        w = w / w.sum()
        maps.append(xyz[np.argmax(w)])
        means.append(w @ xyz)
    return {'events': len(ids), 'p90_3d_error_m': p90(np.array(maps), truth),
            'weighted_mean_p90_3d_error_m': p90(np.array(means), truth),
            'point_estimator': 'maximum saved posterior mass; first saved atom on exact ties',
            'e_step_cycle': 11, 'pre_last_M_step': True,
            'row_index_source_event_id_weights_sha256': digest.hexdigest(),
            'saved_mass_range': [min(sums), max(sums)], 'saved_atom_count_range': [min(counts), max(counts)]}


def observation_checks(work, source, ids, station_ids, stations, initial, config):
    event_index = {value: i for i, value in enumerate(ids)}
    station_index = {value: j for j, value in enumerate(station_ids)}
    arrivals = np.full((len(ids), len(station_ids)), np.nan)
    path = source.parents[1] / 'output' / source.name / 'arrivals.csv'
    with record(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            i, j = event_index[row['event_id']], station_index[row['station_id']]
            require(np.isnan(arrivals[i, j]), 'Duplicate saved pick')
            arrivals[i, j] = float(row['arrival_time_s'])
    require(np.isfinite(arrivals).all(), 'Missing saved picks')
    gauge = np.linalg.norm(initial[:, None, :] - stations[None, :, :] / 1000, axis=2).min(axis=1) / 5
    expected = arrivals + gauge[:, None]
    numeric = numeric_ids(ids)
    by_id = {value: i for i, value in enumerate(numeric)}
    maximum, absolute_count, differential_count, pairs = 0., 0, 0, []
    for name in ('absolute.dat', 'dt.ct'):
        current, seen_stations = None, set()
        with record(work / name).open() as stream:
            for line in stream:
                tokens = line.split()
                if tokens[0] == '#':
                    if current is not None:
                        require(seen_stations == set(station_ids), 'Incomplete observation block')
                    current = tuple(by_id[int(value)] for value in tokens[1:])
                    require(len(current) == (1 if name == 'absolute.dat' else 2), 'Invalid header')
                    seen_stations = set()
                    if name == 'dt.ct':
                        pairs.append(current)
                else:
                    station = tokens[0]
                    require(station not in seen_stations and tokens[-2:] == ['1.00', 'P'], 'Bad pick')
                    seen_stations.add(station)
                    actual = np.array([float(value) for value in tokens[1:-2]])
                    target = expected[list(current), station_index[station]]
                    maximum = max(maximum, float(np.max(np.abs(actual - target))))
                    if name == 'absolute.dat':
                        absolute_count += 1
                    else:
                        differential_count += 1
            require(seen_stations == set(station_ids), 'Incomplete last observation block')
    require(maximum <= 5.01e-9, 'DD inputs are not the saved noisy picks plus gauge')
    require(absolute_count == config['observations']['absolute'] == arrivals.size and
            differential_count == config['observations']['differential'] == len(pairs) * len(station_ids)
            and len(pairs) == len(set(pairs)) == config['observations']['event_pairs'], 'Wrong counts')
    degree = np.bincount(np.array(pairs).ravel(), minlength=len(ids))
    distances = np.array([np.linalg.norm(initial[a] - initial[b]) for a, b in pairs])
    require(degree.min() >= 1 and degree.max() <= 4 and distances.max() <= 3, 'Wrong pair graph')
    return {'all_saved_picks_matched': True, 'max_serialization_error_s': maximum,
            'absolute_count': absolute_count, 'differential_count': differential_count,
            'event_pairs': len(pairs), 'degree_range': [int(degree.min()), int(degree.max())],
            'pair_distance_range_km': [float(distances.min()), float(distances.max())]}


def evaluate():
    figure = read_json(FIGURE)
    for path, digest in figure['sha256'].items():
        record(Path(path), digest)
    manifest = read_json(STUDY / 'manifest.json')
    require(read_json(STUDY / 'status.json')['status'] == 'completed', 'EM study incomplete')
    require(read_json(DD / 'status.json')['state'] == 'completed', 'DD study incomplete')
    build = read_json(DD / 'build/manifest.json')
    require(build['source_change_proof']['changed_lines_1based'] == [840, 841], 'Wrong guard fix')
    helper_path = SNAPSHOT / 'benchmark/tomodd_benchmark/compare_emtomo_flat4.py'
    record(helper_path, manifest['snapshot_sha256'][str(helper_path.relative_to(SNAPSHOT))])
    native = native_helpers(helper_path)
    report = {'schema_version': 1, 'selection': str(FIGURE.relative_to(ROOT)),
              'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
              'definitions': {
                  'velocity': 'Equal-volume RMSE of Vp in m/s on all 501760 native 125-m truth centers; no smoothing',
                  'sphere': 'Each declared geometric sphere intersected with abs(truth-flat4 background)>1e-7 m/s, matching frozen region_masks',
                  'EM': 'Native 1-km constant Vp cells (nearest slowness), not trilinear cells',
                  'DD': 'Native padded trilinear Vp including fixed exterior nodes, NOT interpolated slowness',
                  'location': 'P90 Euclidean 3D error in metres, full catalog, equal event weights; NumPy linear quantile',
                  'EM_checkpoint': 'iter_11 saved E-step on pre-update velocity; final_model after 12 M-steps; no final relocation',
                  'soft_location': 'Primary table uses MAP; weighted-mean P90 supplied separately, never posterior expected distance',
                  'identity': 'EM event_j = source CSV row j; DD EVT_ prefix removed or original integer catalog ID, aligned explicitly',
                  'oracles': 'Velocity-only whole-volume L2 references, not inversions or per-sphere lower bounds'},
              'geometries': {}, 'rows': [], 'oracles': {}}
    common_truth = common_oracle = None
    for geometry, n_events, n_stations in (('synthetic', 1920, 52), ('geysers', 3776, 26)):
        selected = [row for row in figure['rows'] if row['geometry'] == geometry]
        require(len(selected) == 3 and all(row['em_runs'] == selected[0]['em_runs'] and
                row['dd_run'] == selected[0]['dd_run'] for row in selected), 'Inconsistent figure selection')
        dd = Path(selected[0]['dd_run'])
        require(dd == DD / 'outputs' / f'{geometry}-dd1-d40-s5', 'Wrong DD run')
        config, info = read_json(dd / 'config.json'), read_json(dd / 'worker_info.json')
        require(info['state'] == 'completed' and info['dd1_input_reuse']['all_inputs_byte_identical'], 'Wrong DD status/inputs')
        record(Path(config['solver']['binary']), build['artifact_sha256']['tomoDD-SE'])
        require(info['solver_sha256'] == config['solver']['binary_sha256'] == build['artifact_sha256']['tomoDD-SE'], 'Wrong active binary')
        require(config['solver']['iteration_sets'] == [[3, 1., .08, -9., 1., 40., 1, 0.]], 'Wrong DD schedule')
        source = SNAPSHOT / 'projects/forward_modeling/experiments/input' / config['source_experiment']
        for name, digest in config['source_forward_metadata']['input_sha256'].items():
            record(source / name, digest)
        output = source.parents[1] / 'output' / source.name
        for name in ('arrivals.csv', 'metadata.json'):
            path = output / name
            record(path, manifest['snapshot_sha256'][str(path.relative_to(SNAPSHOT))])
        require(read_json(output / 'metadata.json') == config['source_forward_metadata'], 'DD forward metadata differs')
        ids, truth_xyz = points(source / 'events.csv', 'event_id')
        station_ids, stations = points(source / 'stations.csv', 'station_id')
        require(len(ids) == n_events and len(station_ids) == n_stations, 'Wrong geometry counts')
        generation = read_json(source / 'generation.json')
        with np.load(record(source / 'model.npz'), allow_pickle=False) as saved:
            truth = saved['velocity'].astype(float).transpose(2, 1, 0)
            require(saved['velocity'].shape == SHAPE and float(saved['cell_size_m']) == 125, 'Wrong truth grid')
            np.testing.assert_array_equal(saved['origin_m'], [0, 0, 0])
        if common_truth is None:
            common_truth = truth.copy()
            parameters = native.generation_parameters(generation)
        else:
            np.testing.assert_array_equal(truth, common_truth)
        regions = native.region_masks(truth, CENTERS, parameters)
        masks = {key: value for key, value in regions.items() if key in ('volume', 'spheres') or key.startswith('sphere_')}
        require(len(masks) == 8, 'Expected six individual spheres')
        report['truth'] = {'shape_xyz': list(SHAPE), 'cell_size_m': 125, 'extent_km': [20, 7, 7],
                           'parameters': parameters, 'region_cells': {key: int(value.sum()) for key, value in masks.items()}}
        for mode, top_n, run_path in zip(('hard', 'soft'), (1, 10), selected[0]['em_runs']):
            case_id = f'{geometry}-em-{mode}-k15-top{top_n}-dmin1'
            result = read_json(STUDY / 'cases' / case_id / 'result.json')
            require(result['status'] == 'completed' and result['run_dir'] == run_path, 'Wrong EM run')
            run = Path(run_path)
            meta = read_json(run / 'meta.json')
            params, grid = meta['run_params'], meta['grid_info']
            case = next(c for c in manifest['cases'] if c['id'] == case_id)
            for key, value in case['config'].items():
                if key in params:
                    require(params[key] == value, f'EM config differs: {key}')
            require(params['candidate_mode'] == mode and params['weights_top_n'] == top_n and
                    params['n_candidates'] == 15 and params['weights_min_distance'] == 1 and params['n_cycles'] == 12,
                    'Wrong EM candidate mode/checkpoint')
            require(meta['source_experiment'] == {'id': source.name, 'model_sha256': sha256(source / 'model.npz')}, 'Wrong EM source')
            for path in list(source.iterdir()) + [output / name for name in ('arrivals.csv', 'metadata.json')]:
                if path.name in ('events.csv', 'stations.csv', 'generation.json', 'model.npz', 'arrivals.csv', 'metadata.json'):
                    record(path, params['provenance']['snapshot_sha256'][str(path.relative_to(SNAPSHOT))])
            require(params['n_events'] == n_events and meta['event_locs'] == [], 'EM must have no catalog initialization')
            np.testing.assert_allclose(meta['station_locs'], stations, rtol=0, atol=1e-7)
            require(grid['fine_shape'] == list(SHAPE) and grid['coarse_shape'] == [20, 7, 7] and
                    grid['fine_cell_size'] == 125 and grid['coarse_cell_size'] == 1000 and
                    params['subdivision'] == 8 and params['slowness_interpolation'] == 'nearest', 'Wrong EM grid')
            require(sorted(int(p.parent.name.split('_')[1]) for p in run.glob('iter_*/complete.json')) == list(range(12)), 'Incomplete EM cycles')
            for step in range(12):
                require(read_json(run / f'iter_{step}/complete.json')['iter'] == step, 'Wrong cycle')
            cells = np.load(record(run / 'final_model.npy', result['final_model_sha256']), allow_pickle=False)
            initial = np.load(record(run / 'initial_model.npy'), allow_pickle=False)
            background = np.array([3800., 4600., 5200., 5700.])[np.searchsorted([.7, 1.9, 3.4], np.arange(7) + .5)]
            np.testing.assert_array_equal(initial, np.broadcast_to(background, (20, 7, 7)))
            field = native.tile_cells(cells)
            indices = np.indices(SHAPE)
            np.testing.assert_array_equal(field.transpose(2, 1, 0), cells[tuple(indices // 8)])
            report['rows'].append({'geometry': geometry, 'method': 'EMTomo ' + mode,
                                   'run': str(run.relative_to(ROOT)), 'velocity': velocity(field, truth, masks),
                                   'location': em_locations(run, meta, ids, truth_xyz),
                                   'parameters': {k: v for k, v in params.items() if k != 'provenance'},
                                   'checks': {'initial_layers_verified': True, 'native_cell_tiling_verified': True,
                                              'source_inputs_and_arrivals_hash_matched': True}})
        work = dd / 'work/inversion'
        for name, digest in info['generated_input_sha256'].items():
            record(work / name, digest)
        record(work / 'tomoDD-SE.inp', info['generated_control_sha256'])
        with np.load(record(dd / 'state.npz'), allow_pickle=False) as saved:
            state = {key: saved[key].copy() for key in saved.files}
        np.testing.assert_array_equal(state['physical_velocity_km_s'].transpose(2, 1, 0), truth / 1000)
        np.testing.assert_allclose(state['true_event_xyz_km'] * 1000, truth_xyz, rtol=0, atol=1e-9)
        numeric = numeric_ids(ids)
        np.testing.assert_array_equal(state['true_events'][:, 0], numeric)
        np.testing.assert_allclose(geographic_xyz(state['true_events'], config['coordinate_mapping']), truth_xyz, rtol=0, atol=1e-6)
        final_rows = np.loadtxt(record(work / 'tomoDD.reloc'), usecols=(0, 1, 2, 3), ndmin=2)
        np.testing.assert_array_equal(align(final_rows, numeric), align(state['final_events'], numeric))
        xyz = geographic_xyz(align(final_rows, numeric), config['coordinate_mapping'])
        history = native.read_velocity_history(record(work / 'tomoDD.vel'), (9, 9, 22), 3)
        np.testing.assert_allclose(history, state['final_vp'] * 1000, rtol=0, atol=1e-9)
        log = record(work / 'tomoDD-SE.log').read_text()
        for step in range(1, 4):
            require(re.search(rf'\bIteration\s+{step}\s+finished\b', log), 'DD iteration incomplete')
            align(np.loadtxt(record(work / f'tomoDD.reloc.001.{step:03d}'), usecols=(0, 1, 2, 3), ndmin=2), numeric)
        grid = config['tomodd_grid']
        axes = tuple(np.array(grid[key]) for key in ('depths_km', 'y_km', 'x_km'))
        mapping = config['coordinate_mapping']
        np.testing.assert_allclose(state['lon_nodes'], mapping['center_lon'] + axes[2] / (111 * np.cos(np.deg2rad(mapping['center_lat']))), rtol=0, atol=1e-12)
        np.testing.assert_allclose(state['lat_nodes'], mapping['center_lat'] + axes[1] / 111, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(state['depths'], axes[0])
        np.testing.assert_array_equal(state['initial_vp'][1:-1, 1:-1, 1:-1] * 1000,
                                      np.broadcast_to(background[:, None, None], (7, 7, 20)))
        exterior = np.ones((9, 9, 22), dtype=bool)
        exterior[1:-1, 1:-1, 1:-1] = False
        np.testing.assert_array_equal(state['final_vp'][exterior], state['initial_vp'][exterior])
        field = native.evaluate_native(history, axes, CENTERS)
        z, y, x = np.meshgrid(CENTERS[2], CENTERS[1] - 3.5, CENTERS[0] - 10, indexing='ij')
        query = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
        manual = manual_trilinear(history, axes, query).reshape(truth.shape)
        interpolation_error = float(np.max(np.abs(field - manual)))
        np.testing.assert_allclose(field, manual, rtol=0, atol=1e-9)
        checks = observation_checks(work, source, ids, station_ids, stations, state['initial_event_xyz_km'], config)
        checks.update(native_trilinear_max_difference_m_s=interpolation_error,
                      final_vp_matches_iteration_3=True, exterior_nodes_unchanged=True,
                      all_events_retained_and_aligned=True, shared_truth_verified=True)
        report['rows'].append({'geometry': geometry, 'method': 'tomoDD', 'run': str(dd.relative_to(ROOT)),
                               'velocity': velocity(field, truth, masks),
                               'location': {'events': n_events, 'p90_3d_error_m': p90(xyz, truth_xyz)},
                               'checks': checks})
        report['geometries'][geometry] = {
            'events': n_events, 'stations': n_stations,
            'event_xyz_range_m': [truth_xyz.min(axis=0).tolist(), truth_xyz.max(axis=0).tolist()],
            'station_xyz_range_m': [stations.min(axis=0).tolist(), stations.max(axis=0).tolist()],
            'source': str(source.relative_to(ROOT)), 'noise': config['source_forward_metadata']['noise'],
            'forward': {key: config['source_forward_metadata'][key] for key in
                        ('solver_name', 'pykonal_version', 'config', 'receiver_interpolation', 'time_reference')},
            'dd_grid': grid, 'dd_solver': config['solver'], 'dd_observations': config['observations'],
            'dd_initial_hypocenters': config['initial_hypocenters'],
            'dd_absolute_DD_amplitude_weights': info['absolute_dd_weight'],
            'coordinate_mapping': mapping,
            'geometry_generation': {k: generation[k] for k in
                                    ('generator', 'selection', 'top_elevation_m', 'vertical_coordinates', 'shared_datum_assumption') if k in generation}}
        em_oracle = native.tile_cells(native.constant_oracle(truth))
        dd_oracle = native.evaluate_native(state['oracle_vp'] * 1000, axes, CENTERS)
        if common_oracle is not None:
            np.testing.assert_allclose(dd_oracle, common_oracle, rtol=0, atol=1e-6)
        common_oracle = dd_oracle
        report['oracles'][geometry] = {'EMTomo': velocity(em_oracle, truth, masks), 'tomoDD': velocity(dd_oracle, truth, masks)}
        for row in report['rows']:
            if row['geometry'] == geometry:
                oracle = em_oracle if row['method'].startswith('EMTomo') else dd_oracle
                require(row['velocity']['volume_rmse_m_s'] >= velocity(oracle, truth, masks)['volume_rmse_m_s'], 'Below representation oracle')
    for path in (Path(__file__), Path(__file__).with_name('evaluate_metrics.py'), Path(__file__).with_name('make_checkerboard_location_ecdf.py')):
        record(path)
    report['sha256'] = dict(sorted(FILES.items()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and compare, without writing')
    args = parser.parse_args()
    if not args.check:
        require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Report already exists; use --check')
    report = evaluate()
    if args.check:
        require(json.loads(OUTPUT.read_text()) == report, 'Recomputed report differs')
        print('PASS: full recomputation equals saved JSON')
    else:
        with OUTPUT.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in report['rows']:
        print(row['geometry'], row['method'], json.dumps(row['velocity']),
              'P90_3D_m=', row['location']['p90_3d_error_m'])


if __name__ == '__main__':
    main()
