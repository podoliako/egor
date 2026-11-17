"""Sections 4.2--4.5 at the frozen CSV L-curve selections; no solver imports.

Run from projects/EMTomo:
    python -B paper/paper_scripts/make_hard_soft_lcurve_metrics.py
    python -B paper/paper_scripts/make_hard_soft_lcurve_metrics.py --check

Creates hard_soft_lcurve_comparison_metrics.json (exclusive unless --refresh).
--check recomputes without writing. Extraction is bounded to 300 seconds.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'paper/figures/hard_soft_lcurve_comparison_metrics.json'
SELECTIONS = ROOT / 'paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv'


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def volume_reference(model, model_side, truth, truth_side):
    """Repeat the coarser field onto the nested common volume grid, not average truth."""
    ratio = max(model_side, truth_side) / min(model_side, truth_side)
    require(np.isclose(ratio, round(ratio)), 'Non-nested evaluation grids')
    if model_side >= truth_side:
        for axis in range(3):
            model = np.repeat(model, round(ratio), axis=axis)
    else:
        for axis in range(3):
            truth = np.repeat(truth, round(ratio), axis=axis)
    return rmse(model, truth)


def map_error(positions, weights, truth, side, shape):
    positions, weights = np.asarray(positions), np.asarray(weights)
    require(positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 0
            and weights.shape == (len(positions),) and np.isfinite(positions).all()
            and np.isfinite(weights).all() and (weights >= 0).all()
            and np.isclose(weights.sum(), 1., atol=1e-6)
            and (positions >= 0).all() and (positions < np.asarray(shape)).all(),
            'Invalid saved hypotheses')
    return float(np.linalg.norm((positions[np.argmax(weights)] + .5) * side - truth))


def table_pairs(selections, legacy):
    """Use legacy JSON row identities, never its final-checkpoint metric values."""
    result = {}
    for table, name in [('onesided', 'onesided'), ('two_clusters', 'two_clusters'),
                        ('real_geometry', 'real_geometry')]:
        old = legacy[name]
        by_id = {r['id']: Path(r['run_dir']).name for r in old['rows']}
        result[table] = []
        for pair in old['comparisons']:
            matches = [r for r in selections if r['hard_run_id'] == by_id[pair['hard']]
                       and r['soft_run_id'] == by_id[pair['soft']]]
            require(len(matches) == 1, f'Ambiguous CSV mapping: {table}/{pair}')
            result[table].append(matches[0])
    real = result.pop('real_geometry')
    result['real_geometry_noise'] = real[:5]
    result['real_geometry_candidates'] = [real[5], real[2], *real[6:]]
    result['fault_retention'] = [next(r for r in selections if r['pair_id'] == name)
        for name in ['section5_pair22', *[f'section5_topnsweep_k128_top{n}'
                                        for n in (50, 75, 100, 128)]]]
    result['fault_separation'] = [next(r for r in selections if r['pair_id'] ==
        f'section5_dminsweep_k128_top{n}_dmin{d}') for d in (4, 7) for n in (50, 128)]
    fault_runs = {r['run_id'] for r in legacy['fault']['records']}
    require({r[f'{mode}_run_id'] for t in ('fault_retention', 'fault_separation')
             for r in result[t] for mode in ('hard', 'soft')} == fault_runs,
            'Fault legacy/CSV run mapping mismatch')
    require([len(v) for v in result.values()] == [4, 4, 5, 5, 5, 4], 'Wrong table sizes')
    return result


def extract():
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[os.path.relpath(path, ROOT)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    selections = rows(track(SELECTIONS))
    require(len(selections) == 56 and len({r['pair_id'] for r in selections}) == 56,
            'Expected 56 unique frozen selection rows')
    legacy = {name: read(track(ROOT / f'paper/figures/{name}_comparison_metrics.json'))
              for name in ('onesided', 'two_clusters', 'real_geometry', 'fault')}

    manifest = read(track(ROOT / 'studies/topn_sweep_20261111/manifest.json'))
    # Resolve relative to this project rather than the manifest's machine-specific absolute path.
    experiments_root = ROOT.parent / 'forward_modeling/experiments'
    require(experiments_root.is_dir(), 'Missing saved forward experiments')
    require('experiments_root' in manifest, 'Malformed study manifest')
    tables = table_pairs(selections, legacy)
    # Load just the pure NumPy diagnostic reference helper; no FMM/inversion imports.
    helper_path = track(ROOT / 'velocity_model.py')
    tree = ast.parse(helper_path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'block_average_slowness')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(helper_path), 'exec'), namespace)
    block_average = namespace['block_average_slowness']
    experiments, source_cache, records = {}, {}, {}

    def source_data(identifier):
        if identifier in source_cache:
            return source_cache[identifier]
        source = experiments_root / 'input' / identifier
        metadata = read(track(experiments_root / 'output' / identifier / 'metadata.json'))
        generation = read(track(source / 'generation.json'))
        for name, expected in metadata['input_sha256'].items():
            track(source / name)
            require(hashes[os.path.relpath(source / name, ROOT)] == expected,
                    f'Stale input {identifier}/{name}')
        require(metadata['time_reference'] == 'earliest_station_arrival_per_event',
                'Wrong arrival convention')
        events, stations = rows(source / 'events.csv'), rows(source / 'stations.csv')
        ids, station_ids = [r['event_id'] for r in events], [r['station_id'] for r in stations]
        require(len(set(ids)) == len(ids) and len(set(station_ids)) == len(station_ids),
                'Duplicate source IDs')
        arrivals = rows(track(experiments_root / 'output' / identifier / 'arrivals.csv'))
        pairs = {(r['event_id'], r['station_id']) for r in arrivals}
        require(len(arrivals) == len(pairs) == len(ids) * len(station_ids)
                and pairs == {(e, s) for e in ids for s in station_ids}, 'Arrival ID mismatch')
        parent_id = generation.get('source_experiment_id', identifier)
        parent = experiments_root / 'input' / parent_id
        # Cropped checkerboards legitimately differ from their parent experiment.
        # Only the fault's noise-only derivative must preserve the complete truth/catalogue.
        original_id = parent_id if identifier == 'gfsoftscreen28_26st_noise02_3776ev' else identifier
        original = experiments_root / 'input' / original_id
        if parent_id != identifier:
            for name, expected in generation.get('source_input_sha256', {}).items():
                track(parent / name)
                require(hashes[os.path.relpath(parent / name, ROOT)] == expected,
                        f'Stale parent {parent_id}/{name}')
        if original_id != identifier:
            for name, expected in generation.get('source_input_sha256', {}).items():
                track(original / name)
                require(hashes[os.path.relpath(original / name, ROOT)] == expected,
                        f'Stale original {original_id}/{name}')
            for name in ('model.npz', 'events.csv', 'stations.csv'):
                require((source / name).read_bytes() == track(original / name).read_bytes(),
                        f'Changed original {name}')
        with np.load(track(original / 'model.npz'), allow_pickle=False) as saved:
            truth, side = saved['velocity'].astype(float), float(saved['cell_size_m'])
            require(np.array_equal(saved['origin_m'], [0, 0, 0]), 'Nonzero truth origin')
        truth_xyz = xyz(events)
        require(np.isfinite(truth_xyz).all() and np.isfinite(truth).all()
                and (truth > 0).all(), 'Invalid original truth')
        experiments[identifier] = dict(original_truth_experiment_id=original_id,
            source_input_directory=os.path.relpath(source, ROOT), n_events=len(ids),
            n_stations=len(station_ids), n_observed_arrivals=len(arrivals),
            truth_shape=list(truth.shape), truth_cell_size_m=side,
            generation=generation, forward_metadata=metadata,
            original_generation=read(track(original / 'generation.json')),
            event_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            event_xyz_min_m=truth_xyz.min(axis=0).tolist(),
            event_xyz_max_m=truth_xyz.max(axis=0).tolist())
        source_cache[identifier] = (truth, side, ids, truth_xyz, xyz(stations))
        return source_cache[identifier]

    for selection in {r['pair_id']: r for group in tables.values() for r in group}.values():
        for mode in ('hard', 'soft'):
            run_id = selection[f'{mode}_run_id']
            k = int(selection[f'{mode}_selected_update_number'])
            require(int(selection[f'{mode}_quality_iteration_index']) == k - 1
                    and int(selection[f'{mode}_residual_iteration_index']) == k,
                    'CSV checkpoint index mismatch')
            if run_id in records:
                require(records[run_id]['selected_update_k'] == k, 'Inconsistent reused selection')
                continue
            run = ROOT / 'runs' / run_id
            meta = read(track(run / 'meta.json'))
            params, grid = meta['run_params'], meta['grid_info']
            identifier = meta['source_experiment']['id']
            truth, truth_side, ids, locations, stations = source_data(identifier)
            require(identifier == selection['experiment_id'] and meta['run_id'] == run_id
                    and params['candidate_mode'] == mode, 'Run identity mismatch')
            require(meta['source_experiment']['model_sha256'] ==
                    experiments[identifier]['forward_metadata']['input_sha256']['model.npz'],
                    'Run/source model mismatch')
            require(params['n_events'] == len(ids) == int(selection['n_events'])
                    and np.array_equal(meta['station_locs'], stations), 'Run geometry mismatch')
            if 'event_ids' in meta['source_experiment']:
                require(meta['source_experiment']['event_ids'] == ids, 'Run event order mismatch')
            require(params['slowness_interpolation'] == 'nearest' and params['n_cycles'] == 12
                    and 0 < k < params['n_cycles'], 'Unavailable selected E-step or wrong protocol')
            for key in ('n_candidates', 'weights_top_n'):
                require(params[key] == int(selection[f'{mode}_{key}']), f'CSV/{key} mismatch')
            require(params['weights_min_distance'] == int(selection['weights_min_distance']),
                    'CSV/dmin mismatch')
            if selection['section'] == '4':
                require(params['coverage_damping_power'] == 2, 'Exclude cov0 three-update runs')
            for i in range(params['n_cycles']):
                require(read(track(run / f'iter_{i}/complete.json'))['iter'] == i,
                        f'Incomplete iteration {run_id}/{i}')
            quality = [json.loads(line) for line in track(run / 'quality.jsonl').read_text().splitlines()]
            require([q['iter'] for q in quality] == list(range(12)), 'Invalid quality history')
            model_path = track(run / f'iter_{k}/model.npy')
            model = np.load(model_path, allow_pickle=False)
            require(list(model.shape) == grid['coarse_shape'], 'Model grid mismatch')
            reference = block_average(truth, truth_side, model.shape, params['cell_size'])
            diagnostic = rmse(model, reference)
            require(np.isclose(diagnostic, quality[k-1]['rms_m_s'], rtol=1e-10, atol=1e-8),
                    f'iter_k/quality[k-1] mismatch: {run_id}')
            require(np.isclose(quality[k-1]['avg_abs_pct_dev'],
                    float(selection[f'{mode}_avg_abs_pct_dev_at_lcurve']), rtol=1e-10),
                    'Frozen selection diagnostic mismatch')
            cycle = run / f'iter_{k}'
            require({p.name for p in cycle.glob('event_*') if p.is_dir()} ==
                    {f'event_{i}' for i in range(len(ids))}, 'Missing/extra saved events')
            errors, digest, counts = [], hashlib.sha256(), []
            for i, event_id in enumerate(ids):
                payload = (cycle / f'event_{i}/weights.npz').read_bytes()
                digest.update(f'{i}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
                with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                    positions, weights = saved['positions'], saved['weight_values']
                    require(np.array_equal(saved['weight_shape'], grid['fine_shape'])
                            and 1 <= len(positions) <= params['weights_top_n'], 'Weight grid/count mismatch')
                    if mode == 'hard':
                        require(np.count_nonzero(weights) == 1, 'Non-hard saved weights')
                    errors.append(map_error(positions, weights, locations[i],
                                            grid['fine_cell_size'], grid['fine_shape']))
                    counts.append(len(positions))
            initial = np.load(track(run / 'initial_model.npy'), allow_pickle=False)
            records[run_id] = dict(run_id=run_id, experiment_id=identifier, mode=mode,
                selected_update_k=k, quality_iteration_index=k-1, weights_iteration_index=k,
                model_path=os.path.relpath(model_path, ROOT),
                weights_directory=os.path.relpath(cycle, ROOT), run_params=params, grid_info=grid,
                started_at=meta['started_at'], complete_iteration_indices=list(range(12)),
                velocity_rmse_m_s=volume_reference(model, params['cell_size'], truth, truth_side),
                block_slowness_reference_rmse_m_s=diagnostic, quality_rms_m_s=quality[k-1]['rms_m_s'],
                initial_velocity_rmse_m_s=volume_reference(initial, params['cell_size'], truth, truth_side),
                hypocenter_p90_m=float(np.percentile(errors, 90, method='linear')),
                n_events=len(errors), saved_candidate_count_range=[min(counts), max(counts)],
                row_id_weights_sha256=digest.hexdigest())
            print(f'{selection["pair_id"]} {mode} k={k}: '
                  f'RMSE={records[run_id]["velocity_rmse_m_s"]:.6f}; '
                  f'P90={records[run_id]["hypocenter_p90_m"]:.6f}', flush=True)
    output_tables = {}
    for name, group in tables.items():
        output_tables[name] = []
        for selection in group:
            item = dict(pair_id=selection['pair_id'], frozen_csv_row=selection)
            for mode in ('hard', 'soft'):
                record = records[selection[f'{mode}_run_id']]
                item[mode] = {key: record[key] for key in ('run_id', 'selected_update_k',
                    'velocity_rmse_m_s', 'hypocenter_p90_m')}
                item[mode].update(tested=record['run_params']['n_candidates'],
                    used=1 if mode == 'hard' else record['run_params']['weights_top_n'],
                    dmin_m=record['run_params']['weights_min_distance'] * record['grid_info']['fine_cell_size'])
            output_tables[name].append(item)
    track(Path(__file__))
    return dict(schema_version=1, sections=['4.2', '4.3', '4.4', '4.5'],
        selection='Frozen CSV selections copied verbatim; no L-curve recalculation or truth-based selection.',
        timing=dict(velocity='iter_k/model.npy is V_k after k velocity updates; checked against quality[k-1].',
                    hypocenters='iter_k weights are the E-step using V_k, before update k+1; NOT iter_(k-1).',
                    extraction='Saved artifacts only; no inversions or FMM; 300-second maximum runtime.'),
        velocity_metric='Equal-volume RMSE against original piecewise-constant truth; repeat the coarser grid onto the nested common grid. Fault uses original 125-m voxels, not block-slowness truth.',
        hypocenter_metric='P90 of Euclidean 3D errors, equal event weights, numpy linear percentile. Largest saved weight in both modes; first argmax on ties, not posterior mean. (positions+0.5)*fine_cell_size.',
        event_alignment='event_i follows source events.csv row i, not sorted event ID; forward metadata input hashes, unique IDs and full arrival ID cross-product verified. Legacy runs do not independently save event IDs; alignment relies on the preserved loader row-order protocol.',
        caveats=[
            'Section 4.1 and results.tex are not modified. Legacy metric JSONs supply row mappings/context only, not selected metrics.',
            'Each mode has its own frozen selected update k; equal selected k is not imposed.',
            'Hard-50 controls are reused for three comparisons in 4.2/4.3; nominal-noise candidate-50 pair is shared between both 4.4 tables.',
            'Fault dmin2 hard top40 is reused for all five retention pairs; dmin4/7 top50/128 have separate matched hard runs, not independent replicates.',
            'Real-geometry noise uses cov2 twelve-update runs only, not the cov0 three-update experiment. Likelihood noise stays nominal as actual noise is scaled.',
            'Single synthetic realization; no replicate uncertainty. Realistic catalogue coordinates define synthetic truth, not independently verified earthquake locations. Fault vertical-datum assumption is unverified; see original_generation.',
            'Fault block-slowness diagnostic RMSE is only a cross-check and generally differs from original-truth whole-volume velocity RMSE.',
            'Worker counts and explicit held-out-pick settings differ between historical and sweep runs; full per-run numerical settings and forward metadata are retained.'],
        tables=output_tables, records=list(records.values()), experiments=experiments,
        source_files_sha256=dict(sorted(hashes.items())))


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
            require(read(OUTPUT) == report, 'Report differs from recomputation')
            print('Verified exact recomputation; no files written.')
        else:
            if args.refresh:
                temporary = OUTPUT.with_suffix('.json.tmp')
                with temporary.open('w') as stream:
                    stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
                temporary.replace(OUTPUT)
            else:
                with OUTPUT.open('x') as stream:
                    stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(OUTPUT)
        print(f'Elapsed extraction seconds: {time.monotonic() - started:.3f}')
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
