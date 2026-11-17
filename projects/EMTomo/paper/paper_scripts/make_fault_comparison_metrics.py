"""Extract section 4.5 metrics from saved artifacts only (NumPy; no solver imports).

From projects/EMTomo:
    python3 -B paper/paper_scripts/make_fault_comparison_metrics.py
    python3 -B paper/paper_scripts/make_fault_comparison_metrics.py --check

Only writes paper/figures/fault_comparison_metrics.json; refuses to overwrite.
--check recomputes everything and compares without writing. A POSIX alarm bounds
extraction to 150 seconds by default (--timeout-seconds to change it).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import signal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'paper/figures/fault_comparison_metrics.json'
EXPERIMENT = 'gfsoftscreen28_26st_noise02_3776ev'
STUDIES = ('topn_sweep_20261111', 'fault_dmin_sweep_20261112',
           'fault_dmin_top128_20261113')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def xyz(records):
    return np.array([[float(row[a + '_m']) for a in 'xyz'] for row in records])


def rmse(model, reference):
    require(model.shape == reference.shape and np.isfinite(model).all()
            and (model > 0).all(), 'Invalid velocity model')
    return float(np.sqrt(np.mean((model - reference) ** 2)))


def expand(model):
    for axis in range(3):
        model = np.repeat(model, 8, axis=axis)
    return model


def map_error(positions, weights, truth, side_m=125.):
    positions, weights = np.asarray(positions), np.asarray(weights)
    require(positions.ndim == 2 and positions.shape[1] == 3
            and len(positions) > 0 and weights.shape == (len(positions),)
            and np.isfinite(positions).all() and np.isfinite(weights).all()
            and (weights >= 0).all() and np.isclose(weights.sum(), 1., atol=1e-6),
            'Invalid saved hypotheses')
    estimate = (positions[np.argmax(weights)] + .5) * side_m
    require(np.all(estimate >= 0) and np.all(estimate <= [20000, 7000, 7000]),
            'MAP estimate outside domain')
    return float(np.linalg.norm(estimate - truth))


def extract():
    hashes = {}

    def track(path):
        path = Path(path)
        # Paths relative to EMTomo, including ../forward_modeling, are portable.
        import os
        hashes[os.path.relpath(path, ROOT)] = sha256(path)
        return path

    manifests = {name: read(track(ROOT / 'studies' / name / 'manifest.json'))
                 for name in STUDIES}
    experiments_root = Path(manifests[STUDIES[0]]['experiments_root'])
    source = experiments_root / 'input' / EXPERIMENT
    generation = read(track(source / 'generation.json'))
    metadata = read(track(experiments_root / 'output' / EXPERIMENT / 'metadata.json'))
    original = experiments_root / 'input' / generation['source_experiment_id']
    original_generation = read(track(original / 'generation.json'))
    parameters = original_generation['parameters']
    for name, expected in metadata['input_sha256'].items():
        require(sha256(track(source / name)) == expected, f'Stale input: {name}')
    for name, expected in generation['source_input_sha256'].items():
        require(sha256(track(original / name)) == expected, f'Stale original: {name}')
    for name in ('events.csv', 'stations.csv', 'model.npz'):
        require(sha256(source / name) == sha256(original / name), f'Changed {name}')
    events, stations = rows(source / 'events.csv'), rows(source / 'stations.csv')
    ids = [r['event_id'] for r in events]
    station_ids = [r['station_id'] for r in stations]
    truth_xyz = xyz(events)
    require(len(ids) == len(set(ids)) == 3776 and len(set(station_ids)) == 26,
            'Unexpected catalogue or station IDs')
    require(np.isfinite(truth_xyz).all(), 'Nonfinite event truth')
    # Match the forward loader's preserved CSV row order, not numeric ID sorting.
    arrivals_path = track(experiments_root / 'output' / EXPERIMENT / 'arrivals.csv')
    arrivals = rows(arrivals_path)
    pairs = [(r['event_id'], r['station_id']) for r in arrivals]
    require(len(pairs) == len(set(pairs)) == 98176
            and set(pairs) == {(e, s) for e in ids for s in station_ids},
            'Incomplete or misaligned event/station arrival IDs')
    with np.load(source / 'model.npz', allow_pickle=False) as saved:
        truth = saved['velocity'].astype(float)
        require(truth.shape == (160, 56, 56) and float(saved['cell_size_m']) == 125
                and np.array_equal(saved['origin_m'], [0, 0, 0]), 'Unexpected truth grid')
    x = (np.arange(160) + .5) * .125
    z = (np.arange(56) + .5) * .125
    profiles = {block: np.asarray(parameters[block]['vp_km_s'])[
        np.searchsorted(parameters[block]['boundaries_km'], z, side='right')] * 1000
        for block in ('left', 'right')}
    analytic = np.where(x[:, None, None] < parameters['fault_x_km'],
                        profiles['left'][None, None, :], profiles['right'][None, None, :])
    require(np.array_equal(truth, np.broadcast_to(analytic, truth.shape)),
            'Saved truth disagrees with original fault specification')
    # Execute only this pure NumPy helper, never import inversion/FMM modules.
    helper_path = track(ROOT / 'velocity_model.py')
    tree = ast.parse(helper_path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'block_average_slowness')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(helper_path), 'exec'), namespace)
    reference = namespace['block_average_slowness'](truth, 125., (20, 7, 7), 1000.)
    track(ROOT / 'experiment_runner.py')
    track(ROOT / 'paper/figures/real_geometry_comparison_metrics.json')
    pair_csv = track(ROOT / 'paper/exp_data/hard_soft_last_velocity_rmse_topn_sweep.csv')
    baseline = next(r for r in rows(pair_csv) if r['pair_id'] == 'section5_pair22')
    topn = manifests[STUDIES[0]]
    require({p['hard_run_id'] for p in topn['pairs'] if p.get('source_pair_id') == 'section5_pair22'}
            == {baseline['hard_run_id']}, 'Baseline hard control disagreement')
    require({v['source_run_id'] for k, v in topn['cases'].items() if k.startswith('s5-')}
            == {baseline['soft_run_id']}, 'Baseline soft control disagreement')
    selected = [(f'dmin2-{mode}-top40', baseline[f'{mode}_run_id'],
                 {'pair_id': 'section5_pair22', 'path': str(pair_csv.relative_to(ROOT))}, None)
                for mode in ('hard', 'soft')]
    for study, manifest in manifests.items():
        for case_id, case in manifest['cases'].items():
            if case['experiment_id'] != EXPERIMENT:
                continue
            result_path = track(ROOT / 'studies' / study / 'cases' / case_id / 'result.json')
            result = read(result_path)
            require(result['status'] == 'completed' and result['completed_updates'] == 12,
                    f'Incomplete case {case_id}')
            selected.append((case_id, result['run_id'],
                             {'manifest': f'studies/{study}/manifest.json', 'case_id': case_id,
                              'result': str(result_path.relative_to(ROOT)),
                              'source_run_id': case['source_run_id']}, case['config']))
    require(len(selected) == 14 and len({r[1] for r in selected}) == 14, 'Expected 14 distinct runs')
    records = []
    shared = None
    initial_baseline = None
    for case_id, run_id, provenance, config in selected:
        run = ROOT / 'runs' / run_id
        meta = read(track(run / 'meta.json'))
        params = meta['run_params']
        require(meta['run_id'] == run_id and meta['source_experiment']['id'] == EXPERIMENT
                and meta['source_experiment']['model_sha256'] == sha256(source / 'model.npz'),
                f'Wrong run/source: {run_id}')
        require(params['n_events'] == 3776 and params['n_cycles'] == 12
                and params['n_candidates'] == 128 and params['cell_size'] == 1000
                and params['subdivision'] == 8 and params['slowness_interpolation'] == 'nearest',
                'Unexpected inversion protocol')
        require(meta['grid_info']['fine_shape'] == [160, 56, 56]
                and meta['grid_info']['fine_cell_size'] == 125
                and np.array_equal(meta['station_locs'], xyz(stations)), 'Geometry mismatch')
        common_keys = ('cell_size', 'subdivision', 'slowness_interpolation', 'n_cycles',
                       'n_candidates', 'initial_layer_boundaries_km', 'initial_layer_velocities_m_s',
                       'temperature', 'weight_noise_relative_sigma', 'weight_noise_absolute_sigma_s',
                       'weight_model_sigma_s', 'lambda_reg', 'coverage_damping_power',
                       'coverage_floor', 'coverage_reference_percentile', 'max_velocity_step_fraction',
                       'linear_solver', 'solver_rtol', 'solver_maxiter')
        common = {k: params[k] for k in common_keys}
        if shared is None:
            shared = common
        require(common == shared, 'Non-shared numerical settings')
        if config is not None:
            for key, value in config.items():
                if key in params and key not in ('run_name', 'runs_dir', 'save_runs'):
                    require(params[key] == value, f'Manifest/config disagreement: {case_id}/{key}')
        for i in range(12):
            require(read(track(run / f'iter_{i}/complete.json'))['iter'] == i,
                    f'Missing completion marker {case_id}/{i}')
        quality = [json.loads(line) for line in track(run / 'quality.jsonl').read_text().splitlines()]
        require([q['iter'] for q in quality] == list(range(12)), 'Unexpected quality history')
        initial = np.load(track(run / 'initial_model.npy'), allow_pickle=False)
        final = np.load(track(run / 'final_model.npy'), allow_pickle=False)
        expected_initial = np.broadcast_to(np.asarray(params['initial_layer_velocities_m_s'])[
            np.searchsorted(params['initial_layer_boundaries_km'], np.arange(7) + .5,
                            side='right')][None, None, :], (20, 7, 7))
        require(np.array_equal(initial, expected_initial), 'Initial profile mismatch')
        if initial_baseline is None:
            initial_baseline = initial
        require(np.array_equal(initial, initial_baseline), 'Initial models differ')
        score = rmse(final, reference)
        require(np.isclose(score, quality[-1]['rms_m_s'], rtol=1e-10, atol=1e-8),
                f'Final model/quality RMS disagreement: {case_id}')
        cycle = run / 'iter_11'
        require({p.name for p in cycle.glob('event_*') if p.is_dir()}
                == {f'event_{i}' for i in range(3776)}, 'Missing/extra events')
        errors, digest, counts = [], hashlib.sha256(), []
        for i, identifier in enumerate(ids):
            payload = (cycle / f'event_{i}/weights.npz').read_bytes()
            digest.update(f'{i}\t{identifier}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                require(np.array_equal(saved['weight_shape'], [160, 56, 56]), 'Weight grid mismatch')
                count = len(saved['positions'])
                require(1 <= count <= params['weights_top_n'], 'Candidate count mismatch')
                if params['candidate_mode'] == 'hard':
                    require(np.count_nonzero(saved['weight_values']) == 1,
                                                'Hard run must have exactly one nonzero hypothesis weight')
                errors.append(map_error(saved['positions'], saved['weight_values'], truth_xyz[i]))
                counts.append(count)
        records.append(dict(case_id=case_id, run_id=run_id, run_dir=f'runs/{run_id}',
                            provenance=provenance, run_params=params, candidate_mode=params['candidate_mode'],
                            tested=128, used=1 if params['candidate_mode'] == 'hard' else params['weights_top_n'],
                            dmin_cells=params['weights_min_distance'],
                            dmin_m=params['weights_min_distance'] * 125.,
                            velocity_rmse_m_s=score, quality_rms_m_s=quality[-1]['rms_m_s'],
                            original_voxel_volume_rmse_m_s=rmse(expand(final), truth),
                            p90_location_error_m=float(np.percentile(errors, 90, method='linear')),
                            n_events=len(errors), saved_candidate_count_range=[min(counts), max(counts)],
                            row_id_weights_sha256=digest.hexdigest()))
        print(f'{case_id}: RMS={score:.6f} m/s; P90={records[-1]["p90_location_error_m"]:.6f} m', flush=True)
    track(Path(__file__))
    return dict(experiment_id=EXPERIMENT, original_truth_experiment_id=generation['source_experiment_id'],
                geometry=dict(parameters=parameters,
                              effective_boundaries_km=original_generation['effective_boundaries_km'],
                              n_stations=26, n_events=3776, n_observed_arrivals=len(pairs),
                              station_xyz_min_m=xyz(stations).min(axis=0).tolist(),
                              station_xyz_max_m=xyz(stations).max(axis=0).tolist(),
                              event_xyz_min_m=truth_xyz.min(axis=0).tolist(),
                              event_xyz_max_m=truth_xyz.max(axis=0).tolist(),
                              top_elevation_m=original_generation['top_elevation_m'],
                              vertical_coordinates=original_generation['vertical_coordinates'],
                              shared_datum_assumption=original_generation['shared_datum_assumption']),
                grids=dict(truth_voxel_size_m=125, truth_shape=[160, 56, 56],
                           generation_solver='PyKonal; saved clean absolute times, no new solve',
                           generation_config=metadata['config'], generation_spacing_m=62.5,
                           generation_velocity_representation=metadata['velocity_representation'],
                           inversion_cell_size_m=1000, inversion_shape=[20, 7, 7],
                           inversion_subdivision=8, inversion_computational_spacing_m=125),
                noise=metadata['noise'], shared_run_params=shared,
                velocity_metric=dict(name='equal-volume cell RMSE against block-averaged-slowness reference',
                                     units='m/s', checkpoint='final_model.npy after 12 updates',
                                     reference='1 / volume_mean(1 / original truth velocity) per 1-km cell',
                                     initial_rmse_m_s=rmse(initial_baseline, reference),
                                     verification='All 14 final_model.npy RMSEs match quality.jsonl iter 11',
                                     caveat='Unlike aligned checkerboards, NOT whole-volume RMSE against original truth; boundaries cross cells.',
                                     supplemental='original_voxel_volume_rmse_m_s repeats constant inversion cells onto original 125-m truth voxels',
                                     initial_original_voxel_volume_rmse_m_s=rmse(expand(initial_baseline), truth)),
                hypocenter_metric=dict(name='90th percentile of full Euclidean 3D error', units='m',
                                       checkpoint='iter_11 E-step preceding twelfth velocity update',
                                       estimator='largest saved weight (first argmax on ties), not posterior mean',
                                       position_conversion='(positions + 0.5) * 125 m',
                                       event_alignment='event_i to events.csv row i, original IDs/order/XYZ hash-identical; complete arrival ID cross-product checked',
                                       percentile='numpy.percentile(errors, 90, method=linear); equal event weights',
                                       weights_digest='SHA256(row_index TAB event_id TAB weights_file_SHA256 LF)'),
                controls='dmin2 hard top40 is reused for all dmin2 soft comparisons; dmin4/7 hard top50 and top128 are distinct controls, not independent replicates.',
                uncertainty='One synthetic realization, seed 42; no replicate uncertainty estimates.',
                adjacent_section_provenance='real_geometry_comparison_metrics.json hashes original inputs, manifests, run metadata/final models and row-ID-weight digests; same final12/preceding iter11 MAP protocol, but its aligned checkerboard reference equals voxel truth.',
                records=records, source_files_sha256=dict(sorted(hashes.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=150)
    args = parser.parse_args()
    require(args.timeout_seconds > 0, 'Timeout must be positive')

    def timeout(signum, frame):
        raise TimeoutError(f'Extraction exceeded {args.timeout_seconds} seconds')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        report = extract()
        if args.check:
            require(read(OUTPUT) == report, 'Metrics report differs from recomputation')
            print('Verified: report exactly matches recomputation; no files written.')
        else:
            with OUTPUT.open('x') as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(OUTPUT)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
