"""Section 4.6 metrics from saved artifacts only; never import inversion/FMM code.

Run with the repository venv Python and -B. Default creates the report exclusively;
--check verifies exact recomputation without writing, --refresh replaces this report.
All extraction is bounded to 300 seconds. Cap20 pairs are primary; cap7 is audit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from make_hard_soft_lcurve_metrics import map_error, read, require, rmse, rows, volume_reference, xyz

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'paper/figures/heterogeneous_lcurve_comparison_metrics.json'
SELECTIONS = ROOT / 'paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv'
PRIMARY = ('section6_pair01', 'section6_pair05')
AUDIT = ('section6_pair02', 'section6_pair06')


def pure_functions(path, names, constants=None):
    """Load only named NumPy functions, excluding module imports and side effects."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Missing pure helper')
    namespace = {'np': np, **(constants or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def arrival_matrix(arrivals, event_ids, station_ids):
    """Align by IDs in catalogue row order, rejecting duplicates or missing pairs."""
    ei, si = {v: i for i, v in enumerate(event_ids)}, {v: i for i, v in enumerate(station_ids)}
    require(len(ei) == len(event_ids) and len(si) == len(station_ids), 'Duplicate catalogue IDs')
    values = np.full((len(ei), len(si)), np.nan)
    for row in arrivals:
        require(row['event_id'] in ei and row['station_id'] in si, 'Unknown arrival ID')
        index = ei[row['event_id']], si[row['station_id']]
        require(np.isnan(values[index]), 'Duplicate arrival pair')
        value = float(row['arrival_time_s'])
        require(np.isfinite(value), 'Nonfinite arrival')
        values[index] = value
    require(np.isfinite(values).all(), 'Missing synthetic arrival pair')
    return values


def verify_stopping(controls, stopping, params):
    """Validate control stopping independently of the frozen L-curve selection."""
    require(len(controls) == stopping['completed_updates'], 'Wrong control history length')
    insufficient, previous = 0, None
    for i, score in enumerate(controls):
        value = score['rmse_s']
        require(score['iter'] == i and np.isfinite(value) and value >= 0, 'Invalid control score')
        if previous is not None:
            insufficient = (0 if value < previous * (1 - params['early_stopping_min_relative_improvement'])
                            else insufficient + 1)
        stop = i + 1 >= 3 and insufficient >= params['early_stopping_patience']
        require(score['insufficient_steps'] == insufficient and score['stop'] == stop
                and (not stop or i == len(controls) - 1), 'Control stopping mismatch')
        previous = value
    require(stopping['reason'] == ('control_rmse_plateau' if stop else 'max_cycles')
            and stopping['control_rmse_s'] == previous
            and stopping['insufficient_steps'] == insufficient
            and (stop or len(controls) == params['n_cycles']), 'Invalid stopping summary')


def extract():
    hashes = {}

    def track(path, expected=None):
        path = Path(path)
        key = os.path.relpath(path, ROOT)
        if key not in hashes:
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(expected is None or hashes[key] == expected, f'Stale input: {key}')
        return path

    def local(path):
        # Manifest paths are absolute on the original machine; preserve project-relative identity.
        suffix = str(path).split('/projects/', 1)
        require(len(suffix) == 2, f'Unrecognized source path: {path}')
        return ROOT.parent / suffix[1]

    selections = {s['pair_id']: s for s in rows(track(SELECTIONS)) if s['pair_id'] in PRIMARY + AUDIT}
    require(set(selections) == set(PRIMARY + AUDIT), 'Missing section 6 selections')
    identifiers = {s['experiment_id'] for s in selections.values()}
    require(len(identifiers) == 1, 'Audit is not the same synthetic realization')
    identifier = identifiers.pop()
    experiments = ROOT.parent / 'forward_modeling/experiments'
    source, forward = experiments / 'input' / identifier, experiments / 'output' / identifier
    generation = read(track(source / 'generation.json'))
    metadata = read(track(forward / 'metadata.json'))
    for name, digest in metadata['input_sha256'].items():
        track(source / name, digest)
    require(metadata['time_reference'] == 'earliest_station_arrival_per_event', 'Arrival convention changed')
    require(metadata['config']['refinement'] == 1 and generation['forward_plan']['numerical_cell_size_m'] == 100,
            'Unexpected forward grid')
    for key in ('model_source', 'model_description_source'):
        track(local(generation[key]['path']), generation[key]['sha256'])
    with np.load(track(source / 'model.npz'), allow_pickle=False) as saved:
        truth, truth_side = saved['velocity'].astype(float), float(saved['cell_size_m'])
        require(truth.shape == (200, 80, 80) and truth_side == 100
                and np.array_equal(saved['origin_m'], [0, 0, 0]), 'Wrong original truth grid')
    with np.load(local(generation['model_source']['path']), allow_pickle=False) as saved:
        require(np.array_equal(truth, saved['velocity']) and truth_side == float(saved['cell_size_m'])
                and np.array_equal(saved['origin_m'], [0, 0, 0]), 'Forward truth differs from model_previous')
    script_dir = Path(__file__).parent
    analytic = pure_functions(track(script_dir / 'make_realistic_anomalies.py'),
        ('scaled_coordinates', 'components', 'velocity'), dict(SIZE_SCALE=2., AMPLITUDE_SCALE=1.5,
        FAULT_THICKNESS_SCALE=1., SURFACE_LAYER_THICKNESS_SCALE=1., SEED=42,
        INTRUSION_CENTER_KM=(15., 3.8, 4.8)))
    x, y, z = [(np.arange(n) + .5) * .1 for n in truth.shape]
    rebuilt = analytic['velocity'](x[:, None, None], y[None, :, None], z[None, None, :]) * 1000
    require(np.array_equal(truth, rebuilt.astype(np.float32)), 'Analytic previous-version truth mismatch')
    for name in ('make_realistic_anomalies_initial.py', 'make_realistic_anomalies_800m.py'):
        track(script_dir / name)
    geometry = generation['geometry_source']
    for name, digest in geometry['input_sha256'].items():
        track(experiments / 'input' / geometry['experiment_id'] / name, digest)
        if name in ('events.csv', 'stations.csv'):
            require((source / name).read_bytes() == (experiments / 'input' / geometry['experiment_id'] / name).read_bytes(),
                    'Geometry changed from saved catalogue')
    events, stations = rows(source / 'events.csv'), rows(source / 'stations.csv')
    ids, station_ids = [r['event_id'] for r in events], [r['station_id'] for r in stations]
    locations, station_xyz = xyz(events), xyz(stations)
    arrivals = arrival_matrix(rows(track(forward / 'arrivals.csv')), ids, station_ids)
    require(arrivals.shape == (3465, 26), 'Unexpected observation geometry')
    require(np.isfinite(locations).all() and (locations >= 0).all()
            and (locations <= np.array(truth.shape) * truth_side).all(), 'Invalid truth locations')
    records, studies, input_maps = [], {}, {}
    for pair_id in PRIMARY + AUDIT:
        selection = selections[pair_id]
        manifest_path = local(selection['study_manifest'])
        manifest = read(track(manifest_path))
        study = manifest_path.parent
        studies[os.path.relpath(manifest_path, ROOT)] = dict(control_split=manifest['control_split'],
            auto_stopping=manifest['auto_stopping'], assumed_noise=manifest['assumed_noise'])
        for category in ('input', 'output'):
            for name, digest in manifest[f'{category}_sha256'].items():
                track(experiments / category / identifier / name, digest)
        snapshot = local(manifest['snapshot'])
        helper_paths = {}
        for name in ('velocity_model.py', 'experiment_runner.py', 'experiment_data.py',
                     'tomography/tomography_em.py', 'tomography/tomography_events.py',
                     'tomography/tomography_validation.py', 'tomography/tomography_logging.py'):
            relative = 'projects/EMTomo/' + name
            helper_paths[name] = track(snapshot / relative, manifest['runtime_sha256'][relative])
        split = pure_functions(helper_paths['tomography/tomography_validation.py'], ('split_control_arrivals',))['split_control_arrivals']
        block = pure_functions(helper_paths['velocity_model.py'], ('block_average_slowness',))['block_average_slowness']
        for mode in ('hard', 'soft'):
            run_id = selection[f'{mode}_run_id']
            run = ROOT / 'runs' / run_id
            meta = read(track(run / 'meta.json'))
            params, grid = meta['run_params'], meta['grid_info']
            case_id = meta['source_experiment']['case_id']
            case = manifest['cases'][case_id]
            require(meta['run_id'] == run_id and params['candidate_mode'] == mode
                    and meta['source_experiment']['id'] == identifier
                    and meta['source_experiment']['model_sha256'] == metadata['input_sha256']['model.npz'], 'Run identity mismatch')
            require(params['n_cycles'] == int(selection['requested_updates'])
                    and params['slowness_interpolation'] == 'nearest'
                    and params['n_events'] == len(ids) and np.array_equal(meta['station_locs'], station_xyz),
                    'Run geometry/protocol mismatch')
            for key, value in case['config'].items():
                if key in params:
                    require(params[key] == value, f'Manifest/run parameter mismatch: {key}')
            for key in ('n_candidates', 'weights_top_n'):
                require(params[key] == int(selection[f'{mode}_{key}']), f'CSV mismatch: {key}')
            require(params['weights_min_distance'] == int(selection['weights_min_distance'])
                    and params['coverage_damping_power'] == float(selection['coverage_damping_power']), 'CSV protocol mismatch')
            training, expected_mask = split(arrivals, fraction=params['control_fraction'],
                seed=params['control_seed'], min_training_picks=params['control_min_training_picks'])
            mask = np.load(track(run / 'control_mask.npy'), allow_pickle=False)
            require(mask.dtype == bool and np.array_equal(mask, expected_mask), 'Saved control mask mismatch')
            split_info = read(track(run / 'control_split.json'))
            actual_split = dict(requested_fraction=params['control_fraction'], seed=params['control_seed'],
                n_control=int(mask.sum()), n_training=int(np.isfinite(training).sum()),
                scored_events=int(mask.any(axis=1).sum()), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
            require(actual_split == manifest['control_split'], 'Manifest control split mismatch')
            require(all(split_info[key] == actual_split[key] for key in ('requested_fraction', 'seed', 'n_control', 'n_training'))
                    and np.isnan(training[mask]).all() and np.array_equal(training[~mask], arrivals[~mask]), 'Pick leakage')
            require(np.isfinite(training).any(axis=1).all(), 'Some events are not training events')
            stopping = read(track(run / 'stopping.json'))
            completed = stopping['completed_updates']
            require({p.name for p in run.glob('iter_*') if p.is_dir()} == {f'iter_{i}' for i in range(completed)}, 'Incomplete run')
            controls = []
            for i in range(completed):
                require(read(track(run / f'iter_{i}/complete.json'))['iter'] == i, 'Incomplete update')
                controls.append(read(track(run / f'iter_{i}/control.json')))
                require(controls[-1]['n_control'] == int(mask.sum()), 'Wrong control count')
            verify_stopping(controls, stopping, params)
            k = int(selection[f'{mode}_selected_update_number'])
            require(0 < k < completed and int(selection[f'{mode}_quality_iteration_index']) == k - 1
                    and int(selection[f'{mode}_residual_iteration_index']) == k
                    and int(selection[f'{mode}_n_lcurve_checkpoints']) == completed - 1
                    and selection['selection_uses_truth'] == 'False', 'Frozen selection indexing mismatch')
            quality = [json.loads(line) for line in track(run / 'quality.jsonl').read_text().splitlines()]
            require([q['iter'] for q in quality] == list(range(completed)), 'Invalid quality history')
            model_path, initial_path = track(run / f'iter_{k}/model.npy'), track(run / 'initial_model.npy')
            model, initial = np.load(model_path, allow_pickle=False), np.load(initial_path, allow_pickle=False)
            require(list(model.shape) == grid['coarse_shape'] == [25, 10, 10]
                    and params['cell_size'] == 800 and grid['fine_cell_size'] == 100
                    and grid['fine_shape'] == [200, 80, 80], 'Unexpected inversion grid')
            initial_source = case['initial_model_source']
            if case_id.startswith('1d-'):
                for key in ('path', 'snapshot_file'):
                    track(local(initial_source[key]), initial_source['sha256'])
                with np.load(local(initial_source['snapshot_file']), allow_pickle=False) as saved:
                    require(np.array_equal(initial, saved['velocity']) and float(saved['cell_size_m']) == 800
                            and np.array_equal(saved['origin_m'], [0, 0, 0]), 'Supplied 1D start mismatch')
                require(np.array_equal(initial, np.broadcast_to(initial[0, 0, :], initial.shape)), 'Not a 1D start')
            else:
                require(initial_source == dict(kind='homogeneous', velocity_m_s=5000.)
                        and initial.shape == model.shape and np.all(initial == 5000), 'Homogeneous start mismatch')
            diagnostic = rmse(model, block(truth, truth_side, model.shape, params['cell_size']))
            require(np.isclose(diagnostic, quality[k-1]['rms_m_s'], rtol=1e-10, atol=1e-8)
                    and np.isclose(quality[k-1]['avg_abs_pct_dev'], float(selection[f'{mode}_avg_abs_pct_dev_at_lcurve']), rtol=1e-10),
                    'V_k does not match quality[k-1] or CSV')
            cycle = run / f'iter_{k}'
            require({p.name for p in cycle.glob('event_*') if p.is_dir()} == {f'event_{i}' for i in range(len(ids))}, 'Saved event set mismatch')
            errors, digest, counts = [], hashlib.sha256(), []
            for i, event_id in enumerate(ids):
                payload = (cycle / f'event_{i}/weights.npz').read_bytes()
                digest.update(f'{i}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
                with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                    positions, weights = saved['positions'], saved['weight_values']
                    require(np.array_equal(saved['weight_shape'], grid['fine_shape'])
                            and 1 <= len(positions) <= params['weights_top_n'], 'Invalid weights grid/count')
                    require(mode != 'hard' or np.count_nonzero(weights) == 1, 'Non-hard weights')
                    errors.append(map_error(positions, weights, locations[i], grid['fine_cell_size'], grid['fine_shape']))
                    counts.append(len(positions))
            errors = np.asarray(errors)
            record = dict(pair_id=pair_id, role='primary' if pair_id in PRIMARY else 'audit_only',
                start='supplied_1d' if case_id.startswith('1d-') else 'homogeneous_5000_m_s',
                run_id=run_id, mode=mode, selected_update_k=k, quality_iteration_index=k-1,
                weights_iteration_index=k, requested_updates=params['n_cycles'], actual_updates=completed,
                velocity_rmse_m_s=volume_reference(model, params['cell_size'], truth, truth_side),
                initial_velocity_rmse_m_s=volume_reference(initial, params['cell_size'], truth, truth_side),
                block_slowness_diagnostic_rmse_m_s=diagnostic, hypocenter_p90_m=float(np.percentile(errors, 90, method='linear')),
                n_events=len(errors), control_event_subset_p90_m=float(np.percentile(errors[mask.any(axis=1)], 90, method='linear')),
                control_event_subset_count=int(mask.any(axis=1).sum()), saved_candidate_count_range=[min(counts), max(counts)],
                row_id_weights_sha256=digest.hexdigest(), run_params=params, grid_info=grid,
                initial_model_source=initial_source, control_split=actual_split, stopping=stopping,
                control_history=controls, frozen_csv_row=selection)
            records.append(record)
            input_maps[run_id] = dict(model=os.path.relpath(model_path, ROOT), initial=os.path.relpath(initial_path, ROOT),
                weights_directory=os.path.relpath(cycle, ROOT), control_mask=os.path.relpath(run / 'control_mask.npy', ROOT),
                meta=os.path.relpath(run / 'meta.json', ROOT), manifest=os.path.relpath(manifest_path, ROOT))
            print(f'{pair_id} {mode}: k={k}, updates={completed}, RMSE={record["velocity_rmse_m_s"]:.9f}, '
                  f'P90={record["hypocenter_p90_m"]:.9f}, initial={record["initial_velocity_rmse_m_s"]:.9f}', flush=True)
    track(Path(__file__))
    track(script_dir / 'make_hard_soft_lcurve_metrics.py')
    return dict(schema_version=1, section='4.6', primary_pairs=list(PRIMARY), audit_pairs=list(AUDIT),
        selection='Frozen CSV k, never chosen using truth. Plot exclusion does not exclude homogeneous starts from the article.',
        velocity_metric='Equal-volume RMSE against original 100-m piecewise-constant truth throughout 20x8x8 km; repeat 800-m inversion cells, not coarse block-slowness truth.',
        hypocenter_metric='Linear P90 of 3D Euclidean MAP errors over all 3465 events, using iter_k training-only E-step weights in V_k; first argmax on ties; coordinates=(positions+0.5)*100 m.',
        timing='iter_k/model.npy is V_k after k updates, verified against quality[k-1]. Control stopping is separate from the L-curve selection; stopping update retained.',
        event_alignment='event_i follows source events.csv row i, not sorted ID. Runtime loader row-order protocol and input hashes preserved; legacy meta does not independently store event IDs.',
        weights_hash_protocol='SHA256 of concatenated UTF-8 lines: row_index TAB event_id TAB SHA256(weights.npz) NEWLINE, in source catalogue order.',
        caveats=['Synthetic geology with real catalogue/station geometry, not observed velocities or independently verified event truth.',
                 'No real missing-pick mask: complete 3465x26 synthetic picks before seeded 10% pick holdout. All events retain training picks; P90 is not a held-out-event score.',
                 '3244 events have at least one held-out pick. Their subset P90 is audit only, not the article all-event P90.',
                 'Supplied 1D start is truth-derived: arithmetic horizontal mean Vp, Gaussian depth smoothing sigma=0.4 km (reflect), then 800-m depth-bin averaging.',
                 'Cap7 pairs reuse the same arrivals/noise realization and split; they are budget/runtime audits, not independent noise replicates.',
                 'Cap20 1D stops at 9 updates by control plateau; homogeneous reaches 20-update cap. No rollback to a truth-selected model.',
                 'Station elevation/catalogue depth datum compatibility is an unverified source assumption.',
                 'No inversions, forward modelling or FMM executed; extraction capped at 300 seconds.'],
        experiment=dict(id=identifier, generation=generation, forward_metadata=metadata,
            truth_shape=list(truth.shape), truth_cell_size_m=truth_side, extent_m=(np.array(truth.shape)*truth_side).tolist(),
            velocity_range_m_s=[float(truth.min()), float(truth.max())], analytic_previous_truth_verified=True,
            event_ids=ids, station_ids=station_ids, n_events=len(ids), n_stations=len(station_ids), n_arrivals=int(arrivals.size),
            event_xyz_min_m=locations.min(axis=0).tolist(), event_xyz_max_m=locations.max(axis=0).tolist(),
            station_xyz_min_m=station_xyz.min(axis=0).tolist(), station_xyz_max_m=station_xyz.max(axis=0).tolist()),
        studies=studies, records=records, input_maps=input_maps, source_files_sha256=dict(sorted(hashes.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true')
    group.add_argument('--refresh', action='store_true')
    args = parser.parse_args()

    def timeout(signum, frame):
        raise TimeoutError('Extraction exceeded 300 seconds')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    started = time.monotonic()
    try:
        report = extract()
        if args.check:
            require(read(OUTPUT) == report, 'Report differs from exact recomputation')
            print('Verified exact recomputation; no files written.')
        elif args.refresh:
            temporary = OUTPUT.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            temporary.replace(OUTPUT)
        else:
            with OUTPUT.open('x') as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'Elapsed extraction seconds: {time.monotonic() - started:.3f}')
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
