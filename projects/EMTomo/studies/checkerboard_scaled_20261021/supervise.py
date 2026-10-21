"""Dependency-aware scaled-checkerboard queue; status.json is updated atomically.

Refuses any existing status.json (no implicit resume or duplicate runs).
SIGTERM/SIGINT terminate this supervisor's child process groups only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_status(path, status):
    status['updated_utc'] = now()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(path)


def verify_snapshot(manifest):
    root = Path(manifest['snapshot_root'])
    for name, expected in manifest['snapshot_sha256'].items():
        with (root / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Snapshot changed: {name}')


def dependency_state(case, rows):
    dependencies = [rows[key]['status'] for key in case.get('depends_on', [])]
    if any(value in ('failed', 'blocked', 'interrupted', 'not_started') for value in dependencies):
        return 'blocked'
    return 'ready' if all(value == 'completed' for value in dependencies) else 'waiting'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_snapshot(manifest)
    state = Path(manifest['state_dir'])
    path = state / 'status.json'
    cases = manifest['cases']
    status = {'study_id': manifest['study_id'], 'status': 'running', 'pid': os.getpid(),
              'started_utc': now(), 'cases': {c['id']: {'status': 'queued', 'method': c['method'],
                        'log': str(state / 'logs' / (c['id'] + '.log'))} for c in cases}}
    with path.open('x') as stream:
        json.dump(status, stream, indent=2)
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(f'Stopping after signal {signum}', flush=True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            for case_id, (process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    status['cases'][case_id].update(status='completed' if code == 0 else 'failed',
                                                   returncode=code, finished_utc=now())
                    del active[case_id]
                    print(f'{case_id}: exited {code}', flush=True)
            if stopping:
                raise InterruptedError('Queue stopped by signal')
            for case in cases:
                row = status['cases'][case['id']]
                if row['status'] != 'queued':
                    continue
                dependency = dependency_state(case, status['cases'])
                if dependency == 'blocked':
                    row.update(status='blocked', reason='An upstream forward calculation failed')
                    continue
                if dependency == 'waiting':
                    continue
                count = sum(status['cases'][key]['method'] == case['method'] for key in active)
                if count >= manifest['parallel_limits'][case['method']]:
                    continue
                env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='')
                for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
                    env[key] = '1'
                env['NUMBA_CACHE_DIR'] = str(state / 'cache' / case['id'] / 'numba')
                env['MPLCONFIGDIR'] = str(state / 'cache' / case['id'] / 'matplotlib')
                env['MPLBACKEND'] = 'Agg'
                log = Path(row['log']).open('x')
                try:
                    process = subprocess.Popen(case['command'], cwd=manifest['snapshot_root'], env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, close_fds=True)
                except BaseException:
                    log.close()
                    raise
                active[case['id']] = process, log
                row.update(status='running', pid=process.pid, started_utc=now())
                print(f"Started {case['id']}: pid={process.pid}", flush=True)
                write_status(path, status)
            write_status(path, status)
            if not active and all(row['status'] != 'queued' for row in status['cases'].values()):
                break
            time.sleep(5)
        status['status'] = 'completed' if all(row['status'] == 'completed' for row in status['cases'].values()) else 'failed'
    except BaseException as error:
        status.update(status='interrupted' if stopping else 'failed', error=str(error))
        for process, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for case_id, (process, log) in active.items():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            log.close()
            status['cases'][case_id].update(status='interrupted', returncode=process.returncode, finished_utc=now())
        for row in status['cases'].values():
            if row['status'] == 'queued':
                row['status'] = 'not_started'
        raise
    finally:
        status['finished_utc'] = now()
        write_status(path, status)
    return 0 if status['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
