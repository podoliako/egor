"""Plot full-catalogue MAP hypocentre-error P90 from saved last E-step weights.

python -B paper/paper_scripts/make_geysers_noise_hypocentre_p90.py
Reads all 18 runs, writes PNG/PDF, summary/event CSVs and JSON provenance. No solves.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_metrics import positions_to_metres, sha256
from make_geysers_noise_quality import OUTPUT, ROOT, STEM, STYLES, STUDY, common

NAME = 'geysers_emtomo_pick_noise_hypocentre_p90'


def map_error(positions, weights, truth_m, fine_side_m):
    xyz = positions_to_metres(positions, fine_side_m)
    values = np.asarray(weights, dtype=float)
    if (values.shape != (len(xyz),) or not np.isfinite(values).all()
            or (values < 0).any() or values.sum() <= 0):
        raise ValueError('Invalid saved hypothesis weights')
    estimate = xyz[np.argmax(values)]
    return float(np.linalg.norm(estimate - truth_m)), estimate


def main():
    manifest = common.read(STUDY / 'manifest.json')
    common.verify_manifest(manifest)
    assert common.read(STUDY / 'status.json')['status'] == 'completed'
    source = Path(manifest['source_experiments_root']) / 'input' / manifest['source_experiment_id']
    events_path = source / 'events.csv'
    with events_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['event_id'] for row in rows]
    truth = np.array([[float(row[axis + '_m']) for axis in 'xyz'] for row in rows])
    assert len(ids) == len(set(ids)) == 3776 and np.isfinite(truth).all()
    records_path = OUTPUT / f'{STEM}.json'
    runs = common.read(records_path)['records']
    assert len(runs) == 18
    hashes = {str(path): sha256(path) for path in (STUDY / 'manifest.json', events_path, records_path)}
    summaries = []
    with (OUTPUT / f'{NAME}_events.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'event_id', 'error_3d_m', 'map_x_m', 'map_y_m', 'map_z_m'])
        for row in runs:
            run = Path(row['run_dir'])
            variant = row['variant']
            if row['baseline']:
                reference = manifest['references'][variant]
                assert row['case_id'] == reference['case_id'] and str(run) == reference['run_dir']
                experiment = manifest['source_experiment_id']
            else:
                case = next(case for case in manifest['cases'] if case['id'] == row['case_id'])
                result = common.read(STUDY / 'cases' / case['id'] / 'result.json')
                assert result['status'] == 'completed' and result['run_dir'] == str(run)
                assert case['variant'] == variant
                experiment = case['experiment_id']
                derived = Path(manifest['experiments_root']) / 'input' / experiment / 'events.csv'
                assert sha256(derived) == hashes[str(events_path)]
            meta_path = run / 'meta.json'
            meta = common.read(meta_path)
            assert meta['source_experiment']['id'] == experiment
            assert meta['run_params']['n_events'] == 3776 and meta['run_params']['n_cycles'] == 12
            assert meta['grid_info']['fine_shape'] == [160, 56, 56]
            assert meta['grid_info']['fine_cell_size'] == 125
            expected = manifest['references'][variant]['config']
            for key in ('candidate_mode', 'n_candidates', 'weights_top_n', 'weights_min_distance'):
                assert meta['run_params'][key] == expected[key]
            cycle = run / 'iter_11'
            assert common.read(cycle / 'complete.json')['iter'] == 11
            assert {p.name for p in cycle.glob('event_*') if p.is_dir()} == {f'event_{i}' for i in range(3776)}
            errors, digest = [], hashlib.sha256()
            for i, identifier in enumerate(ids):
                payload = (cycle / f'event_{i}/weights.npz').read_bytes()
                digest.update(f'{i}\t{identifier}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
                with np.load(io.BytesIO(payload), allow_pickle=False) as weights:
                    assert np.array_equal(weights['weight_shape'], [160, 56, 56])
                    assert 1 <= len(weights['positions']) <= expected['weights_top_n']
                    error, estimate = map_error(weights['positions'], weights['weight_values'], truth[i], 125.)
                assert np.all(estimate >= 0) and np.all(estimate <= [20000, 7000, 7000])
                errors.append(error)
                writer.writerow([row['case_id'], identifier, error, *estimate])
            errors = np.asarray(errors)
            assert errors.shape == (3776,) and np.isfinite(errors).all()
            score = dict(case_id=row['case_id'], variant=variant, run_dir=str(run),
                         relative_sigma=row['relative_sigma'], absolute_sigma_s=row['absolute_sigma_s'],
                         multiplier=row['multiplier'], events=len(errors),
                         p90_error_m=float(np.percentile(errors, 90, method='linear')),
                         median_error_m=float(np.median(errors)), rms_error_m=float(np.sqrt(np.mean(errors**2))))
            summaries.append(score)
            hashes[str(meta_path)] = sha256(meta_path)
            hashes[str(cycle / 'complete.json')] = sha256(cycle / 'complete.json')
            score['row_id_weights_sha256'] = digest.hexdigest()
            print(f'{variant} noise x{row["multiplier"]:g}: P90={score["p90_error_m"]:.3f} m', flush=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.6, 5.7), layout='constrained')
    for variant, (label, color, marker, style) in STYLES.items():
        rows = sorted((r for r in summaries if r['variant'] == variant), key=lambda r: r['multiplier'])
        assert [r['multiplier'] for r in rows] == [.5, 1, 2, 3, 5, 10]
        ax.plot([r['relative_sigma'] * 100 for r in rows], [r['p90_error_m'] for r in rows],
                label=label, color=color, marker=marker, linestyle=style, linewidth=1.8,
                markersize=6, markerfacecolor='white', markeredgewidth=1.4)
    ax.set(title='90th-percentile hypocentre location error\nGeysers geometry · last saved E-step',
           xlabel='Relative pick-noise σ (%)', ylabel='90th percentile of 3D location error (m)',
           xticks=[.5, 1, 2, 3, 5, 10])
    upper = ax.secondary_xaxis('top', functions=(lambda x: x * 10, lambda x: x / 10))
    upper.set(xlabel='Absolute pick-noise σ (ms)', xticks=[5, 10, 20, 30, 50, 100])
    ax.grid(alpha=.22)
    ax.legend(frameon=False, loc='best')
    fig.supxlabel('MAP locations · 3776 events · before the final velocity update', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{NAME}.{extension}', dpi=200)
    plt.close(fig)
    with (OUTPUT / f'{NAME}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = dict(records=summaries, metric='np.percentile(Euclidean 3D errors in metres, 90, method=linear)',
                  point_estimate='Saved hypothesis of maximum weight; first argmax for exact ties; no posterior mean',
                  coordinate_conversion='(saved positions + 0.5) * fine_cell_size_m (125 m)',
                  checkpoint='iter_11: final saved E-step before the 12th velocity update, not post-final relocation',
                  mapping='event_i matched to original events.csv row i; identical source event IDs/order at all noise levels',
                  weights_digest='SHA256(row_index TAB event_id TAB weights_file_sha256 LF)',
                  noise_policy='Paired seed 42, both actual noise sigmas scale together, fixed inversion settings',
                  uncertainty='One realization per level; no replicate-based confidence intervals', source_sha256=hashes,
                  event_errors_sha256=sha256(OUTPUT / f'{NAME}_events.csv'), script_sha256=sha256(Path(__file__)))
    (OUTPUT / f'{NAME}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (OUTPUT / f'{NAME}.md').write_text(
        '# Hypocentre-error 90th percentile versus pick noise\n\n'
        'Full catalogue (3776 events), Euclidean 3D error in metres, NumPy linear percentile. '
        'For hard and soft, use the maximum-weight saved hypothesis (MAP); no averaging of soft locations. '
        'The saved coordinates are continuous fine-grid indices: (positions+0.5)*125 m. '
        'Last saved E-step, iter_11, BEFORE the final velocity update; no new relocation or solve. '
        'Original 1%+10 ms baseline and five additional noise levels; all three configurations fixed. '
        'One paired seed, no uncertainty intervals. Raw per-event errors and MAP coordinates are in the event CSV.\n\n'
        'Rebuild: `python -B paper/paper_scripts/make_geysers_noise_hypocentre_p90.py`.\n')


if __name__ == '__main__':
    main()
