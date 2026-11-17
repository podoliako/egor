"""Frozen-runtime top-N sweep. Explicit prepare / validate / launch commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

STATE = Path(__file__).resolve().parent
BASE = STATE.parents[1]
REPO = BASE.parents[1]
PYTHON = str(REPO / 'venv/bin/python')
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    with (STATE / '.prepare.lock').open('x') as stream:
        stream.write(now())
    sys.path.insert(0, str(BASE))
    from config import InversionConfig
    from experiment_data import DEFAULT_EXPERIMENTS_ROOT
    rows = {r['pair_id']: r for r in csv.DictReader((BASE / 'paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev.csv').open())}
    manifest = dict(created_at=now(), cases={}, pairs=[], experiments_root=str(DEFAULT_EXPERIMENTS_ROOT), concurrent_runs=4, workers_per_run=12, runtime_sha256={})
    def case(source, mode, k, top):
        case_id = f"s{source['section']}-{mode}-k{k}-top{top}"
        meta = read(BASE / 'runs' / source[f'{mode}_run_id'] / 'meta.json')
        params = meta['run_params']
        config = {f.name: params[f.name] for f in fields(InversionConfig) if f.name in params}
        config.update(n_candidates=k, weights_top_n=top, candidate_mode=mode, n_workers=12,
                      control_fraction=0.0, n_cycles=12, runs_dir=str(BASE / 'runs'), save_runs=True,
                      run_name='topnsweep11-' + case_id)
        config = asdict(InversionConfig(**config))
        manifest['cases'][case_id] = dict(config=config, experiment_id=source['experiment_id'], source_run_id=source[f'{mode}_run_id'])
        (STATE / 'cases' / case_id).mkdir(parents=True)
        return case_id
    def pair(source, k, top, new_hard):
        soft = case(source, 'soft', k, top)
        hard = case(source, 'hard', k, top) if new_hard else None
        manifest['pairs'].append(dict(pair_id=f"section{source['section']}_topnsweep_k{k}_top{top}", source_pair_id=source['pair_id'],
                                      soft_case_id=soft, hard_case_id=hard, hard_run_id=None if hard else source['hard_run_id']))
    for source_id in ('section2_pair01', 'section3_pair01'):
        for k, top in ((50, 10), (50, 25), (100, 100)):
            pair(rows[source_id], k, top, k == 100)
    for k in (30, 100, 150, 200):
        pair(rows['section4_pair07'], k, k, True)
    for top in (50, 75, 100, 128):
        pair(rows['section5_pair22'], 128, top, False)
    assert len(manifest['cases']) == 20 and len(manifest['pairs']) == 14
    previous = read(BASE / 'studies/geysers_realistic_previous_cov0_auto_opt20/manifest.json')
    runtime = set(previous['runtime_sha256'])
    runtime.update(str(p.relative_to(REPO)) for p in BASE.glob('*.py') if p.name not in ('server.py', 'main.py'))
    for package in (BASE / 'tomography', BASE / 'instruments', REPO / 'projects/forward_modeling'):
        runtime.update(str(p.relative_to(REPO)) for p in package.rglob('*.py') if '__pycache__' not in p.parts)
    for name in sorted(runtime):
        target = STATE / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
        manifest['runtime_sha256'][name] = sha(target)
    write(STATE / 'manifest.json', manifest)
    print('Prepared 20 runs / 14 new comparisons', flush=True)


def setup():
    manifest = read(STATE / 'manifest.json')
    snapshot = STATE / 'snapshot'
    for name, checksum in manifest['runtime_sha256'].items():
        if sha(snapshot / name) != checksum:
            raise ValueError(f'Snapshot changed: {name}')
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / 'projects/EMTomo'))
    return manifest


def validate_case(manifest, case_id):
    import numpy as np
    from config import InversionConfig
    from experiment_runner import prepare_inversion
    item = manifest['cases'][case_id]
    config = InversionConfig(**item['config'])
    prepared = prepare_inversion(item['experiment_id'], config, manifest['experiments_root'])
    old = BASE / 'runs' / item['source_run_id']
    if not np.array_equal(prepared.initial_model.velocity, np.load(old / 'initial_model.npy')):
        raise ValueError(f'Initial model mismatch: {case_id}')
    meta = read(old / 'meta.json')
    if prepared.arrivals_table.shape[0] != meta['run_params']['n_events']:
        raise ValueError('Event count mismatch')
    from experiment_data import _sha256
    model_hash = _sha256(Path(manifest['experiments_root']) / 'input' / item['experiment_id'] / 'model.npz')
    if meta['source_experiment']['model_sha256'] != model_hash:
        raise ValueError('Truth model hash mismatch')
    return config, prepared, model_hash


def worker(case_id):
    manifest = setup()
    path = STATE / 'cases' / case_id / 'result.json'
    with (path.parent / '.reserved').open('x') as stream:
        stream.write(now())
    status = dict(status='running', pid=os.getpid(), started_at=now())
    write(path, status)
    try:
        config, prepared, model_hash = validate_case(manifest, case_id)
        from experiment_runner import resolve_pick_noise
        from tomography.tomography_em import run_em, warm_up_jit
        from tomography.tomography_logging import TomographyLogger
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name, run_version=config.run_version)
        status.update(run_id=logger.run_id, run_dir=str(logger.run_dir))
        write(path, status)
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
               resolve_pick_noise(config, prepared.noise_sigmas), reference_model=prepared.reference_model,
               logger=logger, source_experiment=dict(id=manifest['cases'][case_id]['experiment_id'],
               model_sha256=model_hash, study_manifest=str(STATE / 'manifest.json'), case_id=case_id))
        complete = list(logger.run_dir.glob('iter_*/complete.json'))
        if len(complete) != 12 or not (logger.run_dir / 'final_model.npy').exists():
            raise ValueError(f'Expected 12 complete iterations, got {len(complete)}')
        status.update(status='completed', completed_updates=len(complete))
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        status['finished_at'] = now()
        write(path, status)


def env():
    result = dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED='1')
    result.pop('PYTHONPATH', None)
    result.pop('PYTHONHOME', None)
    return result


def queue():
    manifest = read(STATE / 'manifest.json')
    status = dict(status='running', pid=os.getpid(), started_at=now(), cases={})
    write(STATE / 'queue_status.json', status)
    def execute(case_id):
        with (STATE / 'cases' / case_id / 'worker.log').open('xb') as log:
            process = subprocess.Popen([PYTHON, str(STATE / 'run.py'), 'worker', '--case-id', case_id],
                                       cwd=BASE, env=env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            return process.wait()
    with ThreadPoolExecutor(max_workers=manifest['concurrent_runs']) as pool:
        futures = {pool.submit(execute, c): c for c in manifest['cases']}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                code = future.result()
                status['cases'][case_id] = dict(returncode=code)
            except Exception:
                status['cases'][case_id] = dict(error=traceback.format_exc())
            write(STATE / 'queue_status.json', status)
    with (STATE / 'export.log').open('ab') as log:
        result = subprocess.run([PYTHON, str(STATE / 'export.py')], cwd=BASE, env=env(), stdout=log, stderr=subprocess.STDOUT)
    status.update(status='completed' if result.returncode == 0 and all(v.get('returncode') == 0 for v in status['cases'].values()) else 'failed',
                  export_returncode=result.returncode, finished_at=now())
    write(STATE / 'queue_status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'launch', 'queue', 'worker'))
    parser.add_argument('--case-id')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'validate':
        manifest = setup()
        for case_id in manifest['cases']:
            config, prepared, _ = validate_case(manifest, case_id)
            print(case_id, prepared.arrivals_table.shape, 'validated', flush=True)
        write(STATE / 'validation.json', dict(validated_at=now(), manifest_sha256=sha(STATE / 'manifest.json'), cases=list(manifest['cases'])))
    elif args.action == 'launch':
        validation = read(STATE / 'validation.json')
        assert validation['manifest_sha256'] == sha(STATE / 'manifest.json')
        with (STATE / '.launch.lock').open('x') as reservation:
            reservation.write(now())
        with (STATE / 'queue.log').open('xb') as log:
            process = subprocess.Popen([PYTHON, str(STATE / 'run.py'), 'queue'], cwd=BASE, env=env(),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        write(STATE / 'launch.json', dict(pid=process.pid, launched_at=now()))
        print('Detached queue PID:', process.pid)
    elif args.action == 'queue':
        queue()
    else:
        worker(args.case_id)


if __name__ == '__main__':
    main()
