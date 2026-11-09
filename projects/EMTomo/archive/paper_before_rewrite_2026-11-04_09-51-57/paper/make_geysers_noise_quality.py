"""Two three-line noise-sensitivity plots from saved final EMTomo models only.

python -B paper/make_geysers_noise_quality.py
No forward or inverse solve. Outputs PNG/PDF plus shared CSV/JSON/Markdown.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_metrics import BOUNDARIES, SPEEDS, native_helpers, sha256, template_amplitude

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies/geysers_pick_noise_20261027'))
import common

STUDY = ROOT / 'runs/geysers_pick_noise_20261027'
OUTPUT = ROOT / 'paper/figures'
STEM = 'geysers_emtomo_pick_noise_quality'
STYLES = {
    'hard-k15-top1-dmin1': ('Hard · top 1 · dmin 1', '#0072B2', 'o', '-'),
    'soft-k15-top10-dmin1': ('Soft · top 10 · dmin 1', '#D55E00', 's', '--'),
    'soft-k15-top10-dmin2': ('Soft · top 10 · dmin 2', '#009E73', '^', '-.'),
}


def quality(field, truth, masks):
    if (field.shape != truth.shape or not np.isfinite(field).all() or (field <= 0).any()
            or len(masks) != 6 or any(mask.shape != truth.shape or not mask.any() for mask in masks)):
        raise ValueError('Invalid field or six-sphere masks')
    squared = (field - truth) ** 2
    individual = [float(np.sqrt(squared[mask].mean())) for mask in masks]
    return dict(volume_rmse_m_s=float(np.sqrt(squared.mean())),
                mean_anomaly_rmse_m_s=float(np.mean(individual)), sphere_rmse_m_s=individual,
                pooled_anomaly_rmse_m_s=float(np.sqrt(squared[np.logical_or.reduce(masks)].mean())))


def amplitude_quality(field, truth, background, masks):
    amplitudes = [template_amplitude(field, truth, background, mask) for mask in masks]
    return dict(mean_anomaly_amplitude=float(np.mean(amplitudes)), sphere_amplitudes=amplitudes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amplitude', action='store_true',
                        help='Plot mean signed template amplitude instead; preserve previous RMSE figures')
    args = parser.parse_args()
    stem = STEM + ('_amplitude' if args.amplitude else '')
    manifest = common.read(STUDY / 'manifest.json')
    common.verify_manifest(manifest)
    status = common.read(STUDY / 'status.json')
    assert status['status'] == 'completed' and all(c['status'] == 'completed' for c in status['cases'].values())
    paths = [STUDY / name for name in ('manifest.json', 'status.json', 'summary.json', 'forward_complete.json')]
    hashes = {str(path): sha256(path) for path in paths}
    summary = common.read(STUDY / 'summary.json')
    old_scores = {row['case_id']: row['volume_rmse_m_s'] for row in summary['records']}
    source = Path(manifest['source_experiments_root']) / 'input' / manifest['source_experiment_id']
    canonical = source.parent / 'flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001'
    helper = Path(manifest['runtime_root']) / 'benchmark/tomodd_benchmark/compare_emtomo_flat4.py'
    native = native_helpers(helper)
    with np.load(source / 'model.npz', allow_pickle=False) as saved:
        truth = saved['velocity'].transpose(2, 1, 0).copy()
        assert float(saved['cell_size_m']) == 125 and np.array_equal(saved['origin_m'], [0, 0, 0])
    assert sha256(source / 'model.npz') == sha256(canonical / 'model.npz')
    parameters = native.generation_parameters(common.read(canonical / 'generation.json'))
    centers = tuple((np.arange(n) + .5) * .125 for n in (160, 56, 56))
    regions = native.region_masks(truth, centers, parameters)
    background = np.asarray(SPEEDS)[np.searchsorted(BOUNDARIES, centers[2], side='right')][:, None, None]
    names = [name for name in regions if name.startswith('sphere_')]
    masks = [regions[name] for name in names]
    assert len(names) == 6 and np.array_equal(np.logical_or.reduce(masks), regions['spheres'])
    for path in (source / 'model.npz', canonical / 'model.npz', canonical / 'generation.json', helper):
        hashes[str(path)] = sha256(path)
    records = []
    for variant, reference in manifest['references'].items():
        runs = [dict(case_id=reference['case_id'], multiplier=1., relative_sigma=.01,
                     absolute_sigma_s=.01, run_dir=reference['run_dir'], baseline=True)]
        for case in manifest['cases']:
            if case['variant'] != variant:
                continue
            result_path = STUDY / 'cases' / case['id'] / 'result.json'
            result = common.read(result_path)
            assert result['status'] == 'completed' and result['completed_cycles'] == 12
            assert sha256(Path(result['run_dir']) / 'final_model.npy') == result['final_model_sha256']
            hashes[str(result_path)] = sha256(result_path)
            level = next(x for x in manifest['levels'] if x['id'] == case['level_id'])
            runs.append(dict(case_id=case['id'], multiplier=level['multiplier'],
                             relative_sigma=level['relative_sigma'], absolute_sigma_s=level['absolute_sigma_s'],
                             run_dir=result['run_dir'], baseline=False))
        assert sorted(row['multiplier'] for row in runs) == [.5, 1, 2, 3, 5, 10]
        for row in sorted(runs, key=lambda row: row['multiplier']):
            run = Path(row['run_dir'])
            history = [json.loads(line) for line in (run / 'quality.jsonl').read_text().splitlines()]
            assert [entry['iter'] for entry in history] == list(range(12))
            assert all(common.read(run / f'iter_{i}/complete.json')['iter'] == i for i in range(12))
            coarse = np.load(run / 'final_model.npy', allow_pickle=False)
            assert coarse.shape == (20, 7, 7)
            field = coarse.repeat(8, 0).repeat(8, 1).repeat(8, 2).transpose(2, 1, 0)
            scores = quality(field, truth, masks)
            if args.amplitude:
                scores.update(amplitude_quality(field, truth, background, masks))
            np.testing.assert_allclose(scores['volume_rmse_m_s'], old_scores[row['case_id']], rtol=0, atol=1e-9)
            records.append(dict(row, variant=variant, **scores))
            for name in ('final_model.npy', 'meta.json', 'quality.jsonl'):
                hashes[str(run / name)] = sha256(run / name)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    plots = [('mean_anomaly_amplitude', 'anomalies', 'Mean recovered amplitude of six spherical anomalies',
              'Mean signed template amplitude (%)')] if args.amplitude else [
        ('volume_rmse_m_s', 'volume', 'Whole-volume velocity recovery', 'Volume RMSE (m/s)'),
        ('mean_anomaly_rmse_m_s', 'anomalies', 'Mean recovery error of six spherical anomalies',
         'Mean per-anomaly RMSE (m/s)'),
    ]
    for metric, suffix, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(8.6, 5.5), layout='constrained')
        for variant, (label, color, marker, style) in STYLES.items():
            rows = [row for row in records if row['variant'] == variant]
            ax.plot([row['relative_sigma'] * 100 for row in rows],
                    [row[metric] * (100 if args.amplitude else 1) for row in rows],
                    label=label, color=color, marker=marker, linestyle=style, linewidth=1.8,
                    markersize=6, markerfacecolor='white', markeredgewidth=1.4)
        ax.set(title=title + '\nGeysers geometry · final update 12', xlabel='Relative pick-noise σ (%)',
               ylabel=ylabel, xticks=[.5, 1, 2, 3, 5, 10])
        upper = ax.secondary_xaxis('top', functions=(lambda x: x * 10, lambda x: x / 10))
        upper.set(xlabel='Absolute pick-noise σ (ms)', xticks=[5, 10, 20, 30, 50, 100])
        if args.amplitude:
            ax.axhline(100, color='.45', linewidth=1, linestyle=':', zorder=0)
            ax.axhline(0, color='.65', linewidth=.8, zorder=0)
            ax.set_ylim(min(-5, ax.get_ylim()[0]), max(105, ax.get_ylim()[1]))
        ax.grid(alpha=.22)
        ax.legend(frameon=False, loc='best')
        for extension in ('png', 'pdf'):
            fig.savefig(OUTPUT / f'{stem}_{suffix}.{extension}', dpi=200)
        plt.close(fig)
    report = dict(records=records, sphere_order=names, sphere_cells=[int(mask.sum()) for mask in masks],
                  sphere_parameters=parameters['anomalies'], checkpoint='final_model.npy after update 12',
                  volume_metric='sqrt(mean over all physical 125-m cells of squared Vp error)',
                  anomaly_metric='arithmetic mean of six within-sphere RMSEs; equal weight per anomaly',
                  mask='Native region_masks: distance <= radius AND nonzero truth-background contrast',
                  uncertainty='One paired seed (42); no independent replicates or confidence intervals',
                  noise_policy='Both actual sigmas scale together; inversion likelihood/configs remain fixed',
                  source_sha256=hashes, script_sha256=sha256(Path(__file__)))
    if args.amplitude:
        report['amplitude_metric'] = ('For each sphere: sum[(Vp_est-background)*(Vp_true-background)] / '
                                      'sum[(Vp_true-background)^2]; arithmetic mean of six signed coefficients. '
                                      '1=correct amplitude, 0=no template projection, negative=opposite contrast; '
                                      'not clipped and not a complete reconstruction-error metric.')
    (OUTPUT / f'{stem}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    columns = ['variant', 'multiplier', 'relative_sigma', 'absolute_sigma_s', 'volume_rmse_m_s',
               'mean_anomaly_rmse_m_s', 'pooled_anomaly_rmse_m_s', *names, 'case_id', 'baseline', 'run_dir']
    if args.amplitude:
        columns += ['mean_anomaly_amplitude', *(name + '_amplitude' for name in names)]
    with (OUTPUT / f'{stem}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            output = {k: v for k, v in row.items() if k not in ('sphere_rmse_m_s', 'sphere_amplitudes')}
            output.update(zip(names, row['sphere_rmse_m_s']))
            if args.amplitude:
                output.update(zip((name + '_amplitude' for name in names), row['sphere_amplitudes']))
            writer.writerow(output)
    caption = ('# Geysers EMTomo pick-noise sensitivity\n\n'
               'Two plots with three lines each, all six noise levels including the original baseline.\n'
               'Final update 12 only; native piecewise-constant reconstruction on 125-m truth cells.\n'
               'Volume: RMSE over the entire physical domain. Anomalies: compute RMSE separately inside\n'
               'each of the six sphere supports, then take their arithmetic mean (not volume-weighted).\n'
               'Lower is better for both plots. Relative and absolute noise increase together; the\n'
               'top and bottom axes describe the same noise levels, not independent experiments.\n'
               'All inversion settings, including assumed noise, remain fixed. One paired seed,\n'
               'no replicate-based uncertainty estimates. See CSV/JSON for all scores and provenance.\n\n'
               'Rebuild: `python -B paper/make_geysers_noise_quality.py`. No solves are launched.\n')
    if args.amplitude:
        caption = ('# Mean signed anomaly amplitude versus pick noise\n\n'
                   + report['amplitude_metric'] + '\n\n'
                   'The figure shows percentages; 100% is the target, not a guaranteed optimum for other errors. '
                   'Known flat4 background is subtracted, not fitted. Fast/slow signs are normalized by '
                   'each true template. Equal weight per sphere; no absolute values or clipping. '
                   'Noise artifacts, background bias, shape and location errors can affect the projection, '
                   'and averaging can hide differences between spheres. A larger value is not necessarily '
                   'a better reconstruction; use alongside RMSE.\n\n'
                   'Same 18 saved final models, six paired noise levels, fixed inversion settings. '
                   'No solves. Rebuild: `python -B paper/make_geysers_noise_quality.py --amplitude`.\n')
    (OUTPUT / f'{stem}.md').write_text(caption)
    metric = 'mean_anomaly_amplitude' if args.amplitude else 'mean_anomaly_rmse_m_s'
    print(json.dumps([{k: row[k] for k in ('variant', 'multiplier', 'volume_rmse_m_s', metric)}
                     for row in records], indent=2))


if __name__ == '__main__':
    main()
