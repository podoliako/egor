"""Bounded read-only evaluation of retained final models; no numerical solves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import common


def report(manifest):
    common.verify_manifest(manifest)
    import numpy as np
    state = Path(manifest['state_dir'])
    source = Path(manifest['source_experiments_root']) / 'input' / manifest['source_experiment_id']
    with np.load(source / 'model.npz', allow_pickle=False) as saved:
        truth = saved['velocity'].copy()
    if truth.shape != (160, 56, 56):
        raise ValueError('Unexpected truth shape')
    records = []

    def metric(run):
        coarse = np.load(Path(run) / 'final_model.npy', allow_pickle=False)
        if coarse.shape != (20, 7, 7) or not np.isfinite(coarse).all():
            raise ValueError('Invalid final model')
        field = coarse.repeat(8, axis=0).repeat(8, axis=1).repeat(8, axis=2)
        return float(np.sqrt(np.mean((field - truth) ** 2)))

    for variant in common.VARIANTS:
        ref = manifest['references'][variant]
        records.append(dict(case_id=ref['case_id'], variant=variant, multiplier=1.,
                            relative_sigma=.01, absolute_sigma_s=.01, status='baseline_reused',
                            run_dir=ref['run_dir'], volume_rmse_m_s=metric(ref['run_dir'])))
    for case in manifest['cases']:
        path = state / 'cases' / case['id'] / 'result.json'
        result = common.read(path) if path.exists() else {}
        level = next(x for x in manifest['levels'] if x['id'] == case['level_id'])
        row = dict(case_id=case['id'], variant=case['variant'], multiplier=level['multiplier'],
                   relative_sigma=level['relative_sigma'], absolute_sigma_s=level['absolute_sigma_s'],
                   status=result.get('status', 'not_started'), run_dir=result.get('run_dir'),
                   volume_rmse_m_s=None)
        if result.get('status') == 'completed':
            if common.sha(Path(result['run_dir']) / 'final_model.npy') != result['final_model_sha256']:
                raise ValueError('Completed final model hash changed')
            row['volume_rmse_m_s'] = metric(result['run_dir'])
        if result.get('error'):
            row['error'] = result['error']
        records.append(row)
    summary = dict(study_id=manifest['study_id'], evaluated_utc=common.now(),
                   manifest_sha256=common.sha(state / 'manifest.json'),
                   completed_new_inversions=sum(r['status'] == 'completed' for r in records),
                   expected_new_inversions=15, records=records, final_checkpoint=12,
                   assumed_noise_fixed=manifest['policy']['assumed_noise_fixed'],
                   note='Fixed-configuration robustness, not matched-noise likelihood tuning. Locations are pre-final-M-step.')
    common.atomic(state / 'summary.json', summary)
    with (state / 'summary.csv').open('w', newline='') as stream:
        columns = ['case_id', 'variant', 'multiplier', 'relative_sigma', 'absolute_sigma_s',
                   'status', 'volume_rmse_m_s', 'run_dir', 'error']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Geysers spheres: fixed-configuration pick-noise sensitivity', '',
             f'New inversions completed: {summary["completed_new_inversions"]}/15.', '',
             'One noiseless forward solve; paired seed 42. Original 1% +10 ms baseline reused.',
             'All three inversion configurations, including assumed likelihood noise, stay fixed.', '',
             '| Noise multiplier | Configuration | Status | Volume RMSE (m/s) |',
             '|---:|---|---|---:|']
    for row in sorted(records, key=lambda r: (r['multiplier'], r['variant'])):
        value = '—' if row['volume_rmse_m_s'] is None else f'{row["volume_rmse_m_s"]:.3f}'
        lines.append(f'| {row["multiplier"]:g} | {row["variant"]} | {row["status"]} | {value} |')
    (state / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    report(common.read(args.manifest))
