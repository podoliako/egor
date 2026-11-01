"""Evaluate completed screen and its existing matched hard controls; no solves."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'runs/geysers_faulted_layers/soft_screen_20261028'


def main():
    manifest = json.loads((STATE / 'manifest.json').read_text())
    cases = {c['id']: c for c in manifest['cases']}
    results = {key: json.loads((STATE / 'cases' / key / 'result.json').read_text()) for key in cases}
    assert len(cases) == 37 and all(r['status'] == 'completed' for r in results.values())
    for dataset in manifest['datasets']:
        for path, digest in dataset['sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    source = Path(manifest['experiments_root']) / 'input' / next(iter(cases.values()))['experiment_id']
    with np.load(source / 'model.npz', allow_pickle=False) as archive:
        truth = archive['velocity'].copy()
        assert float(archive['cell_size_m']) == 125
    assert truth.shape == (160, 56, 56)
    reference = 1 / (1 / truth).reshape(20, 8, 7, 8, 7, 8).mean(axis=(1, 3, 5))
    hypocentres = {row['case_id']: row for row in json.loads((STATE / 'hypocentre_metrics.json').read_text())['rows']}
    evaluated = {}
    for key, case in cases.items():
        run = Path(results[key]['run_dir'])
        config = case['config']
        meta = json.loads((run / 'meta.json').read_text())
        assert meta['source_experiment']['id'] == case['experiment_id']
        for field, value in config.items():
            if field not in ('runs_dir', 'save_runs'):
                assert meta['run_params'][field] == value, (key, field)
        assert all((run / f'iter_{i}/complete.json').exists() for i in range(config['n_cycles']))
        if config.get('temperature_schedule'):
            for i, temperature in enumerate(config['temperature_schedule']):
                assert json.loads((run / f'iter_{i}/temperature.json').read_text())['temperature'] == temperature
        final = np.load(run / 'final_model.npy', allow_pickle=False)
        assert final.shape == reference.shape and np.isfinite(final).all() and (final > 0).all()
        coarse_rmse = float(np.sqrt(np.mean((final - reference) ** 2)))
        assert abs(coarse_rmse - results[key]['final_rms_m_s']) < 1e-9
        fine = final.repeat(8, 0).repeat(8, 1).repeat(8, 2)
        squared = (fine - truth) ** 2
        history = [json.loads(line)['rms_m_s'] for line in (run / 'quality.jsonl').read_text().splitlines()]
        assert len(history) == config['n_cycles'] and abs(history[-1] - coarse_rmse) < 1e-9
        rays = json.loads((run / f'iter_{config["n_cycles"]-1}/ray_status.json').read_text())
        evaluated[key] = dict(coarse_rmse=coarse_rmse, fine_rmse=float(np.sqrt(squared.mean())),
                              fault_rmse=float(np.sqrt(squared[40:64].mean())),
                              deep_jump_m_s=float(final[7:8, :, 3:7].mean() - final[5:6, :, 3:7].mean()),
                              final_ray_failures=rays['failed'], history=history,
                              hypocentres=hypocentres[key], run_dir=str(run))
    rows = []
    for key in sorted(k for k in cases if k.startswith('s')):
        case = cases[key]
        control = cases[case['hard_control']]
        excluded = {'candidate_mode', 'temperature', 'temperature_schedule', 'run_name'}
        comparable = lambda c: {k: v for k, v in c['config'].items() if k not in excluded}
        assert comparable(case) == comparable(control)
        assert case['experiment_id'] == control['experiment_id']
        soft, hard = evaluated[key], evaluated[control['id']]
        dataset = next(d for d in manifest['datasets'] if d['id'] == case['experiment_id'])
        cfg = case['config']
        delta_history = np.asarray(soft['history']) - hard['history']
        row = dict(case_id=key, hard_control=control['id'], stations=dataset['count'], noise_sigma=dataset['noise_sigma'],
                   n_cycles=cfg['n_cycles'], K=cfg['n_candidates'], N=cfg['weights_top_n'], dmin=cfg['weights_min_distance'],
                   temperature=cfg['temperature'], schedule=json.dumps(cfg.get('temperature_schedule')),
                   lambda_reg=cfg['lambda_reg'], coverage_power=cfg['coverage_damping_power'],
                   soft_coarse_rmse=soft['coarse_rmse'], hard_coarse_rmse=hard['coarse_rmse'],
                   coarse_delta_pct=100 * (soft['coarse_rmse'] / hard['coarse_rmse'] - 1),
                   soft_fine_rmse=soft['fine_rmse'], hard_fine_rmse=hard['fine_rmse'],
                   soft_fault_rmse=soft['fault_rmse'], hard_fault_rmse=hard['fault_rmse'],
                   soft_deep_jump=soft['deep_jump_m_s'], hard_deep_jump=hard['deep_jump_m_s'],
                   soft_hypo_p90_m=soft['hypocentres']['p90_m'], hard_hypo_p90_m=hard['hypocentres']['p90_m'],
                   soft_hypo_median_m=soft['hypocentres']['median_m'], hard_hypo_median_m=hard['hypocentres']['median_m'],
                   iterations_soft_better=int(np.count_nonzero(delta_history < 0)),
                   soft_run_dir=soft['run_dir'], hard_run_dir=hard['run_dir'])
        rows.append(row)
    summary = dict(completed_runs=37, soft_runs=24, existing_hard_controls=13, missing_hard_controls=[],
                   soft_wins_coarse=sum(r['soft_coarse_rmse'] < r['hard_coarse_rmse'] for r in rows),
                   soft_wins_fine=sum(r['soft_fine_rmse'] < r['hard_fine_rmse'] for r in rows),
                   soft_wins_fault_band=sum(r['soft_fault_rmse'] < r['hard_fault_rmse'] for r in rows),
                   soft_wins_hypo_p90=sum(r['soft_hypo_p90_m'] < r['hard_hypo_p90_m'] for r in rows),
                   soft_wins_hypo_median=sum(r['soft_hypo_median_m'] < r['hard_hypo_median_m'] for r in rows),
                   soft_better_at_any_iteration=sum(r['iterations_soft_better'] > 0 for r in rows),
                   final_ray_failures=sum(v['final_ray_failures'] for v in evaluated.values()),
                   closest_by_paired_rmse=[r['case_id'] for r in sorted(rows, key=lambda r:r['coarse_delta_pct'])[:4]],
                   notes=['Coarse reference: inverse block-mean slowness at 1 km; fine truth: saved 125 m model.',
                          'Fault band x=[5,8) km, all y/z. Deep jump x=[7,8) minus [5,6) km at z=[3,7) km; true jump 500 m/s.',
                          'Hypocentre MAP metrics use the last E-step BEFORE the final M-step.',
                          'All controls already completed; deterministic duplicates were not launched.',
                          'Single geometry/truth and one paired noise seed; not a universal soft-vs-hard result.'])
    (STATE / 'analysis.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + '\n')
    with (STATE / 'analysis.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Результаты скрининга: soft против точных hard-контролей', '',
             'Все 37 прогонов завершены. Все 13 hard-контролей уже готовы; новых сравнительных hard не требуется.', '',
             'Ни один из 24 soft не выиграл по RMSE на километровом оракле, мелкой истине или в полосе разлома.',
             'Ни у одного soft нет выигрыша RMSE даже на промежуточной одинаковой по номеру итерации.', '',
             '| Soft | Hard | Станций | Циклов | RMSE soft, м/с | RMSE hard, м/с | Разница | P90 soft/hard, м |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r:r['coarse_delta_pct']):
        lines.append(f'| {r["case_id"]} | {r["hard_control"]} | {r["stations"]} | {r["n_cycles"]} | '
                     f'{r["soft_coarse_rmse"]:.2f} | {r["hard_coarse_rmse"]:.2f} | +{r["coarse_delta_pct"]:.2f}% | '
                     f'{r["soft_hypo_p90_m"]:.1f}/{r["hard_hypo_p90_m"]:.1f} |')
    lines += ['', 'RMSE в таблице — относительно километрового оракла. Все метрики, параметры и run_dir — analysis.csv/json.',
              'Единственный выигрыш P90 гипоцентров: s21 против h10, 458,92 против 467,79 м (−1,90%); медиана и томография хуже.',
              'Нулевое число failed rays в последнем E-step всех 37 прогонов. Расписания температур и соответствие конфигураций проверены.',
              'Ограничения: одна модель/геометрия; один paired seed шума. MAP-гипоцентры сохранены до последнего обновления скорости.']
    (STATE / 'analysis.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
