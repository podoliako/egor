"""Read-only saved-weight diagnostic. Only writes new files beside this script."""
import csv
import hashlib
import io
import json
from pathlib import Path
import signal
import time

import numpy as np

ROOT = Path('/mnt/disk01/egor')
BASE = ROOT / 'projects/EMTomo'
OUT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def estimates(positions, weights, side, shape):
    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    require(positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 0,
            'Invalid positions shape')
    require(weights.shape == (len(positions),) and np.isfinite(weights).all()
            and (weights >= 0).all(), 'Invalid weights')
    total = weights.sum()
    require(np.isclose(total, 1., atol=1e-6, rtol=0), 'Unnormalized weights')
    require(np.isfinite(positions).all() and (positions >= 0).all()
            and (positions < np.asarray(shape)).all(), 'Invalid positions')
    coordinates = (positions + .5) * side
    w = weights / total
    mode = coordinates[np.argmax(weights)]
    mean = np.sum(weights[:, None] * coordinates, axis=0) / total
    spread = np.sqrt(np.sum(w * np.sum((coordinates - mean) ** 2, axis=1)))
    return mode, mean, float(spread), float(1 / np.sum(w ** 2)), float(abs(total - 1))


def metrics(errors):
    return dict(p90_m=float(np.percentile(errors, 90, method='linear')),
                mean3d_m=float(np.mean(errors)), median3d_m=float(np.median(errors)),
                spatial_rmse_m=float(np.sqrt(np.mean(errors ** 2))),
                p95_m=float(np.percentile(errors, 95, method='linear')),
                max_m=float(np.max(errors)))


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Diagnostic exceeded 300 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    report_paths = [BASE / 'paper/figures' / name for name in
                    ('hard_soft_lcurve_comparison_metrics.json',
                     'heterogeneous_lcurve_comparison_metrics.json')]
    a, b = map(read, report_paths)
    pairs = {}
    for series, rows in a['tables'].items():
        for row in rows:
            pid = row['pair_id']
            if pid not in pairs:
                pairs[pid] = dict(pair_id=pid, series=[], hard=row['hard'], soft=row['soft'])
            pairs[pid]['series'].append(series)
    for pid in b['primary_pairs']:
        rr = [r for r in b['records'] if r['pair_id'] == pid]
        pairs[pid] = dict(pair_id=pid, series=['heterogeneous'], start=rr[0]['start'],
                         **{r['mode']:r for r in rr})
    require(len(pairs) == 28, 'Expected 28 distinct article pairs')
    records = {r['run_id']:r for r in a['records']}
    for r in b['records']:
        if r['pair_id'] in b['primary_pairs']:
            records[r['run_id']] = dict(r, experiment_id=b['experiment']['id'],
                weights_directory=b['input_maps'][r['run_id']]['weights_directory'])
    selected = {p[m]['run_id'] for p in pairs.values() for m in ('hard', 'soft')}
    sources, audits, runs = {}, {}, {}
    event_path = OUT / 'per_event.csv'
    with event_path.open('x', newline='') as stream:
        columns = ['run_id','mode','selected_update_k','row_index','event_id']
        columns += [f'{kind}_{axis}_m' for kind in ('truth','map','mean') for axis in 'xyz']
        columns += ['map_error_m','mean_error_m','mean_minus_map_error_m',
                    'map_to_mean_shift_m','posterior_rms_spread_m','ess']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rid in sorted(selected):
            r = records[rid]
            eid = r['experiment_id']
            if eid not in sources:
                exp = a['experiments'].get(eid, b['experiment'])
                source = BASE / exp.get('source_input_directory',
                    '../forward_modeling/experiments/input/' + eid)
                expected = exp['forward_metadata']['input_sha256']
                checked = {}
                for name, digest in expected.items():
                    actual = sha((source / name).read_bytes())
                    require(actual == digest, f'Input hash mismatch: {eid}/{name}')
                    checked[name] = actual
                with (source / 'events.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                ids = [row['event_id'] for row in rows]
                require(len(set(ids)) == len(ids), 'Duplicate event IDs')
                truth = np.array([[float(row[c + '_m']) for c in 'xyz'] for row in rows])
                order_digest = sha('\n'.join(ids).encode())
                if 'event_order_sha256' in exp:
                    require(order_digest == exp['event_order_sha256'], 'Event order mismatch')
                else:
                    require(ids == exp['event_ids'], 'Heterogeneous event order mismatch')
                audits[eid] = dict(source_directory=str(source.resolve()), input_sha256=checked,
                                   event_order_sha256=order_digest, n_events=len(ids))
                sources[eid] = ids, truth
            ids, truth = sources[eid]
            require(len(ids) == r['n_events'], 'Wrong event count')
            k = r['selected_update_k']
            require(k == r['weights_iteration_index'], 'Wrong E-step iteration')
            cycle = BASE / r['weights_directory']
            require(cycle.name == f'iter_{k}', 'Wrong weights directory')
            require({p.name for p in cycle.glob('event_*')} ==
                    {f'event_{i}' for i in range(len(ids))}, 'Missing/extra events')
            grid = r['grid_info']
            digest = hashlib.sha256()
            errors, shifts, spreads, esses, normalization = [], [], [], [], []
            for i, event_id in enumerate(ids):
                payload = (cycle / f'event_{i}/weights.npz').read_bytes()
                digest.update(f'{i}\t{event_id}\t{sha(payload)}\n'.encode())
                with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                    positions, weights = saved['positions'], saved['weight_values']
                    require(np.array_equal(saved['weight_shape'], grid['fine_shape']), 'Wrong grid')
                    require(1 <= len(positions) <= r['run_params']['weights_top_n'], 'Wrong count')
                    mode, mean, spread, ess, deviation = estimates(
                        positions, weights, grid['fine_cell_size'], grid['fine_shape'])
                    if r['mode'] == 'hard':
                        require(np.count_nonzero(weights) == 1 and np.array_equal(mode, mean),
                                'Hard mean not exactly MAP')
                emap, emean = np.linalg.norm(mode-truth[i]), np.linalg.norm(mean-truth[i])
                shift = np.linalg.norm(mean-mode)
                errors.append((emap, emean))
                shifts.append(shift)
                spreads.append(spread)
                esses.append(ess)
                normalization.append(deviation)
                event = dict(run_id=rid, mode=r['mode'], selected_update_k=k,
                             row_index=i, event_id=event_id)
                for kind, xyz in [('truth',truth[i]), ('map',mode), ('mean',mean)]:
                    event.update({f'{kind}_{axis}_m':float(v) for axis,v in zip('xyz',xyz)})
                event.update(map_error_m=emap, mean_error_m=emean,
                    mean_minus_map_error_m=emean-emap, map_to_mean_shift_m=shift,
                    posterior_rms_spread_m=spread, ess=ess)
                writer.writerow(event)
            errors = np.array(errors)
            mm, mn = metrics(errors[:,0]), metrics(errors[:,1])
            delta = abs(mm['p90_m'] - r['hypocenter_p90_m'])
            require(delta <= 1e-8, f'MAP P90 mismatch: {rid}, {delta}')
            require(digest.hexdigest() == r['row_id_weights_sha256'], f'Weight digest mismatch {rid}')
            runs[rid] = dict(run_id=rid, experiment_id=eid, mode=r['mode'], selected_update_k=k,
                n_events=len(ids), map=mm, mean=mn,
                fraction_mean_better=float(np.mean(errors[:,1] < errors[:,0])),
                fraction_mean_worse=float(np.mean(errors[:,1] > errors[:,0])),
                fraction_equal=float(np.mean(errors[:,1] == errors[:,0])),
                average_map_to_mean_shift_m=float(np.mean(shifts)),
                p95_map_to_mean_shift_m=float(np.percentile(shifts,95)),
                average_posterior_rms_spread_m=float(np.mean(spreads)), average_ess=float(np.mean(esses)),
                max_weight_sum_deviation=float(max(normalization)),
                map_p90_crosscheck_absolute_difference_m=delta,
                row_id_weights_sha256=digest.hexdigest(), digest_matches_published=True)
            print(f'{len(runs)}/{len(selected)} {r["mode"]}: P90 {mm["p90_m"]:.3f} -> {mn["p90_m"]:.3f}', flush=True)
    pair_rows = []
    for pid,p in pairs.items():
        h,s = runs[p['hard']['run_id']], runs[p['soft']['run_id']]
        for mode in ('hard','soft'):
            require(abs(runs[p[mode]['run_id']]['map']['p90_m'] - p[mode]['hypocenter_p90_m']) <= 1e-8,
                    'Table MAP mismatch')
            require(runs[p[mode]['run_id']]['selected_update_k'] == p[mode]['selected_update_k'],
                    'Table k mismatch')
        row = dict(pair_id=pid, series=';'.join(p['series']), start=p.get('start',''),
                   hard_run_id=h['run_id'], soft_run_id=s['run_id'],
                   hard_k=h['selected_update_k'], soft_k=s['selected_update_k'], n_events=s['n_events'])
        params = records[s['run_id']]['run_params']
        row.update(tested=params['n_candidates'], top_n=params['weights_top_n'],
                   dmin_cells=params['weights_min_distance'], temperature=params['temperature'])
        for key in h['map']:
            row.update({f'hard_map_{key}':h['map'][key], f'soft_map_{key}':s['map'][key],
                        f'soft_mean_{key}':s['mean'][key]})
        for key in ('p90_m','spatial_rmse_m'):
            for estimator in ('map','mean'):
                row[f'soft_{estimator}_gain_{key}_percent'] = 100*(1-s[estimator][key]/h['map'][key])
        for key in ('fraction_mean_better','fraction_mean_worse','fraction_equal',
                    'average_map_to_mean_shift_m','average_posterior_rms_spread_m','average_ess'):
            row[key] = s[key]
        pair_rows.append(row)
    series_rows = []
    for series in [*a['tables'], 'heterogeneous', 'ALL_DEDUPLICATED']:
        rr = [r for r in pair_rows if series == 'ALL_DEDUPLICATED' or series in r['series'].split(';')]
        row = dict(series=series, pairs=len(rr))
        for key in ('p90_m','spatial_rmse_m'):
            for estimator in ('map','mean'):
                gains = [r[f'soft_{estimator}_gain_{key}_percent'] for r in rr]
                row[f'{estimator}_{key}_soft_better_hard_count'] = sum(g > 0 for g in gains)
                row[f'{estimator}_{key}_average_gain_percent'] = float(np.mean(gains))
            change = [r[f'soft_mean_{key}']-r[f'soft_map_{key}'] for r in rr]
            row[f'mean_better_map_{key}_count'] = sum(v < 0 for v in change)
            row[f'mean_minus_map_{key}_average_m'] = float(np.mean(change))
            row[f'mean_minus_map_{key}_range_m'] = f'{min(change):.6f} to {max(change):.6f}'
        series_rows.append(row)
    elapsed = time.monotonic()-started
    result = dict(schema_version=1, elapsed_seconds=elapsed, distinct_pairs=len(pairs), unique_runs=len(runs),
        protocol=dict(selection='Published fixed L-curve k for each mode; iter_k weights in V_k',
            coordinate_convention='(positions + 0.5) * fine_cell_size, metres',
            weighted_mean='sum(saved_weights * coordinates) / sum(saved_weights)',
            weights='Actual saved weighting, no temperature tuning; finite/nonnegative, sum within 1e-6 absolute',
            p90='numpy linear percentile, all events equally weighted',
            spatial_rmse='sqrt(mean(||estimate - truth||^2))',
            comparison='hard MAP baseline for both soft estimators; strict event better/worse comparisons',
            series_aggregation='Unweighted average over selected pairs, not pooled events or independent replicates'),
        caveats=['Postprocessing only; no changes to velocities, article, existing outputs, or inversion.',
            'No FMM, forward models or new solves. No truth-based selection or temperature tuning.',
            'The mean minimizes posterior expected squared coordinate loss for the saved discrete distribution, not necessarily realized truth error, P90, or mean distance.',
            'Multimodal weighted means can lie between modes in low-probability/physically implausible locations; no new solve evaluates them.',
            'Saved weights are a retained-candidate approximation, not an independently calibrated full posterior.',
            'Single synthetic realizations; reused hard controls and shared section4_pair07 are not independent replicates.',
            'Legacy event_i alignment relies on loader row-order protocol; catalogue hashes/order and row-ID-weight digests verified.',
            'Heterogeneous section includes PRIMARY pair01 and pair05 only; audit pairs excluded.'],
        source_reports_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in report_paths},
        input_audits=audits, runs=list(runs.values()), pairs=pair_rows, series=series_rows)
    with (OUT/'diagnostic.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    write_csv(OUT/'pairs.csv', pair_rows)
    write_csv(OUT/'series.csv', series_rows)
    signal.alarm(0)
    print(f'Complete: {len(pairs)} distinct pairs, {len(runs)} unique runs, {elapsed:.2f}s')


if __name__ == '__main__':
    main()
