"""Refresh the top-N sweep and both distance sweeps without duplicate pairs."""
import csv
import importlib.util
from pathlib import Path

STATE = Path(__file__).resolve().parent
BASE = STATE.parents[1]
TOPN = STATE.parent / 'topn_sweep_20261111'
DATA = BASE / 'paper/exp_data'


def main():
    spec = importlib.util.spec_from_file_location('sweep_export', TOPN / 'export.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = DATA / 'hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv'
    rows = module.export_study(study=TOPN, project=BASE, output=target)
    combined = list(rows)
    for study, filename in (
        (STATE, 'hard_soft_lcurve_avg_abs_pct_dev_fault_dmin.csv'),
        (STATE.parent / 'fault_dmin_top128_20261113',
         'hard_soft_lcurve_avg_abs_pct_dev_fault_dmin_top128.csv'),
    ):
        distance = module.export_study(study=study, project=BASE, output=DATA / filename)
        manifest = module.read(study / 'manifest.json')
        ids = {p['pair_id'] for p in manifest['pairs']}
        additions = [r for r in distance if r['pair_id'] in ids]
        if len(additions) != len(ids) or ids.intersection(r['pair_id'] for r in combined):
            raise ValueError('Missing or duplicate distance comparisons')
        combined.extend(additions)
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(target)
    print(f'Combined CSV: {len(combined)} pairs, {len(combined) - len(rows)} added distance comparisons')


if __name__ == '__main__':
    main()
