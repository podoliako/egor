"""SSH-safe, initial-only queue: one forward stage, then fifteen fresh EM workers.

Usage: pipeline.py launch|status|stop [--state PATH]. Finished, failed and stopped
states are never implicitly resumed: use a NEW state directory for another study.
The internal supervise command requires the launcher's inherited OS flock.
"""
from __future__ import annotations

import argparse

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

DEFAULT_STATE = Path('/mnt/disk01/egor/projects/EMTomo/runs/geysers_pick_noise_20261027')
PYTHON = '/mnt/disk01/egor/venv/bin/python3'
HEARTBEAT_S = 2.0
POLL_S = .1
HANDSHAKE_S = 30.0
CLEANUP_GRACE_S = 2.0
REPORT_TIMEOUT_S = 120.0
TIMEOUTS = dict(forward_timeout_s=10800, case_timeout_s=14400, queue_timeout_s=86400)
LEGACY_TIMEOUTS = dict(forward_timeout_s='forward_s', case_timeout_s='case_s', queue_timeout_s='overall_s')
LEVELS = ('x0p5', 'x2', 'x3', 'x5', 'x10')
VARIANTS = ('hard-k15-top1-dmin1', 'soft-k15-top10-dmin1', 'soft-k15-top10-dmin2')
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def identity(pid):
    """Use the /proc birth token, never a bare, recyclable PID."""
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else 'proc:' + fields[19]
    except (OSError, IndexError):
        if Path('/proc').exists():
            return None
    # Useful for solve-free tests on macOS; the production host uses /proc.
    try:
        output = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'stat='],
                                         text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        birth, state = output.rsplit(None, 1)
        return None if state.startswith('Z') else 'ps:' + birth
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def alive(record):
    return bool(record and record.get('birth') and identity(record.get('pid')) == record['birth'])


def acquire(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / 'queue.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise RuntimeError('Queue exclusive lock is held; refusing duplicate launch') from None
    return fd


def initial_only(state):
    for name in ('status.json', 'forward_complete.json', 'queue.log', 'forward.log',
                 'summary.json', 'report.md', 'report.log'):
        if os.path.lexists(state / name):
            raise RuntimeError(f'Existing {name}; initial-only queue refuses resume or overwrite')
    if any((state / 'cases').glob('*/result.json')) or any((state / 'cases').glob('*/run.log')):
        raise RuntimeError('Existing case outputs; refusing resume or overwrite')
    if (state / 'forward' / 'solve_started.json').exists():
        raise RuntimeError('Forward already attempted; refusing another queue')


def environment(manifest):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(manifest['runtime_root']))
    env.update({name: '1' for name in THREAD_VARIABLES})
    return env


def timeout(manifest, name):
    """Accept the shared manifest's named limits, including the original nesting."""
    limits = manifest.get('timeouts', {})
    value = manifest.get(name, limits.get(name, limits.get(LEGACY_TIMEOUTS[name])))
    ceiling = TIMEOUTS[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= ceiling:
        raise ValueError(f'Invalid finite timeout {name}: {value!r}')
    return value


def validate_queue(manifest, state):
    if manifest.get('schema_version', 1) != 1 or Path(manifest['state_dir']).resolve() != state:
        raise ValueError('Manifest schema/state mismatch')
    if Path(manifest['runner_root']).resolve() != state / 'runner':
        raise ValueError('Runner must be the frozen state/runner directory')
    for key, value in dict(forward_workers=32, inversion_workers_per_run=12,
                           max_parallel_runs=3, blas_threads=1).items():
        if manifest['resources'][key] != value:
            raise ValueError(f'Unexpected resources.{key}')
    expected = {(level, variant) for level in LEVELS for variant in VARIANTS}
    cases = manifest['cases']
    if len(cases) != 15 or {(c['level_id'], c['variant']) for c in cases} != expected:
        raise ValueError('Expected exactly fifteen cases: five noise levels, three variants')
    if any(c['id'] != f"{c['level_id']}-{c['variant']}" for c in cases):
        raise ValueError('Unexpected case ID')
    for name in TIMEOUTS:
        timeout(manifest, name)


def terminate_all(items):
    """Kill owned session groups, even after a leader exits leaving pool workers."""
    for item in items:
        try:
            os.killpg(item['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + CLEANUP_GRACE_S
    while items and time.monotonic() < deadline:
        exists = False
        for item in items:
            item['process'].poll()
            try:
                os.killpg(item['process'].pid, 0)
                exists = True
            except ProcessLookupError:
                pass
        if not exists:
            break
        time.sleep(.05)
    for item in items:
        try:
            os.killpg(item['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for item in items:
        try:
            item['process'].wait(timeout=2)
        except subprocess.TimeoutExpired:
            print(f"Unreaped child PID {item['process'].pid}", flush=True)
        finally:
            item['handle'].close()


def runner_script(manifest, name):
    path = Path(manifest['runner_root']) / name
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != manifest['runner_sha256'].get(name):
        raise ValueError(f'Frozen runner modified or unregistered: {name}')
    return path


def forward_ready(manifest):
    return importlib.import_module('em_worker').verify_forward(manifest)


def case_completed(state, case_id):
    result = read(state / 'cases' / case_id / 'result.json')
    if result.get('case_id') != case_id or result.get('status') != 'completed' or not result.get('final_model_sha256'):
        raise ValueError(f'Case did not publish a validated completion: {case_id}')
    return result


class Supervisor:
    def __init__(self, state):
        self.state = state
        self.manifest: dict[str, Any] | None = None
        self.active = []
        self.started = time.monotonic()
        self.mutex = threading.RLock()
        self.done = threading.Event()
        self.heartbeat_error = None
        self.stop_requested = None
        self.status: dict[str, Any] = dict(status='running', stage='preflight', phase='preflight', pid=os.getpid(),
                           birth=identity(os.getpid()), started_utc=now(), active=[], finished_cases=[],
                           cases={}, manifest=str(state / 'manifest.json'))
        self.status['supervisor'] = dict(pid=self.status['pid'], birth=self.status['birth'])

    def transition(self, **fields):
        with self.mutex:
            if 'stage' in fields:
                fields['phase'] = fields['stage']
            self.status.update(fields)
            print(json.dumps(dict(utc=now(), transition=fields)), flush=True)
            self.publish()

    def publish(self):
        with self.mutex:
            self.status.update(updated_utc=now(), heartbeat_unix_s=time.time())
            self.status['active'] = [dict(id=i['id'], pid=i['process'].pid, birth=i['birth'],
                                          pgid=i['process'].pid, started_utc=i['started_utc'],
                                          log=str(i['log']), state='running') for i in self.active]
            # Preserve viewer run directories as soon as workers publish them.
            for identifier, record in self.status['cases'].items():
                path = self.state / 'cases' / identifier / 'result.json'
                if path.is_file():
                    result = read(path)
                    record['run_dir'] = result.get('run_dir')
            atomic(self.state / 'status.json', self.status)

    def heartbeat(self):
        while not self.done.wait(HEARTBEAT_S):
            try:
                self.publish()
            except BaseException as error:
                self.heartbeat_error = error
                return

    def check_deadline(self):
        if self.stop_requested is not None:
            raise KeyboardInterrupt(f'received signal {self.stop_requested}')
        if self.heartbeat_error:
            raise RuntimeError(f'Heartbeat persistence failed: {self.heartbeat_error}')
        assert self.manifest is not None
        if time.monotonic() - self.started >= timeout(self.manifest, 'queue_timeout_s'):
            raise TimeoutError('Overall queue deadline exceeded')

    def spawn(self, identifier, script, extra, log, limit):
        assert self.manifest is not None
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open('x')
        try:
            command = [PYTHON, '-B', str(runner_script(self.manifest, script)),
                       '--manifest', str(self.state / 'manifest.json'), *extra]
            process = subprocess.Popen(command, cwd=self.manifest['runtime_root'], env=environment(self.manifest),
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        except BaseException:
            handle.close()
            raise
        with self.mutex:
            self.active.append(dict(id=identifier, process=process, birth=identity(process.pid),
                                    started=time.monotonic(), started_utc=now(), timeout=limit, log=log, handle=handle))
            if identifier in self.status['cases']:
                self.status['cases'][identifier].update(status='running', pid=process.pid,
                                                       birth=self.active[-1]['birth'], started_utc=now())
            self.transition(child_started=identifier)

    def record(self, outcome):
        with self.mutex:
            self.status['finished_cases'].append(outcome)
            identifier = outcome['id']
            if identifier in self.status['cases']:
                record = self.status['cases'][identifier]
                record.update(outcome)
                directory = self.state / 'cases' / identifier
                directory.mkdir(parents=True, exist_ok=True)
                atomic(directory / 'supervisor.json', outcome)
                path = directory / 'result.json'
                result = read(path) if path.exists() else dict(case_id=identifier, run_dir=None)
                if outcome['status'] != 'completed':
                    # Keep partial worker provenance, traceback, metrics and run_dir.
                    result.update(status=outcome['status'], supervisor_outcome=outcome,
                                  finished_utc=outcome['finished_utc'])
                    atomic(path, result)
                record['run_dir'] = result.get('run_dir')
            print(json.dumps(dict(utc=now(), child_finished=outcome)), flush=True)
            self.publish()

    def wait_one(self, *, reporting=False):
        while True:
            if not reporting:
                self.check_deadline()
            current = time.monotonic()
            for item in list(self.active):
                process = item['process']
                code = process.poll()
                timed_out = code is None and current - item['started'] >= item['timeout']
                if code is None and not timed_out:
                    continue
                terminate_all([item])
                with self.mutex:
                    self.active.remove(item)
                outcome = dict(id=item['id'], exit_code=code if code is not None else process.returncode,
                               timed_out=timed_out, status='timed_out' if timed_out else 'failed', finished_utc=now())
                if not timed_out and code == 0:
                    try:
                        if item['id'] == 'forward':
                            forward_ready(self.manifest)
                        elif item['id'] != 'report':
                            case_completed(self.state, item['id'])
                        outcome['status'] = 'completed'
                    except Exception as error:
                        outcome['error'] = str(error)
                self.record(outcome)
                return outcome
            time.sleep(POLL_S)

    def execute(self):
        self.manifest = read(self.state / 'manifest.json')
        validate_queue(self.manifest, self.state)
        with self.mutex:
            self.status['cases'] = {c['id']: dict(id=c['id'], level_id=c['level_id'], variant=c['variant'],
                                                  status='pending', run_dir=None,
                                                  log=str(self.state / 'cases' / c['id'] / 'run.log'))
                                    for c in self.manifest['cases']}
        importlib.import_module('common').verify_manifest(self.manifest)
        self.check_deadline()
        self.transition(stage='forward')
        self.spawn('forward', 'forward.py', [], self.state / 'forward.log', timeout(self.manifest, 'forward_timeout_s'))
        if self.wait_one()['status'] != 'completed':
            raise RuntimeError('Forward execution/validation failed; ALL inversions blocked; see forward.log')
        self.transition(stage='inversions')
        pending = list(self.manifest['cases'])
        while pending or self.active:
            self.check_deadline()
            while pending and len(self.active) < self.manifest['resources']['max_parallel_runs']:
                case = pending.pop(0)
                try:
                    forward_ready(self.manifest)
                    self.check_deadline()
                    self.spawn(case['id'], 'em_worker.py', ['--case-id', case['id']],
                               self.state / 'cases' / case['id'] / 'run.log', timeout(self.manifest, 'case_timeout_s'))
                except TimeoutError:
                    raise
                except Exception as error:
                    self.record(dict(id=case['id'], status='failed', error=str(error),
                                     exit_code=None, timed_out=False, finished_utc=now()))
            if self.active:
                self.wait_one()
        complete = all(c['status'] == 'completed' for c in self.status['cases'].values())
        self.transition(status='completed' if complete else 'completed_with_failures', stage='finished')

    def finish(self):
        # Once cleanup starts, repeated stop signals cannot interrupt group reaping.
        terminate_all(self.active)
        for item in list(self.active):
            with self.mutex:
                self.active.remove(item)
            self.record(dict(id=item['id'], status='cancelled', exit_code=item['process'].returncode,
                             timed_out=False, finished_utc=now()))
        with self.mutex:
            for record in self.status['cases'].values():
                if record['status'] == 'pending':
                    record['status'] = 'blocked' if self.status.get('stage') in ('preflight', 'forward') else 'cancelled'
            self.publish()
        # Reporting gets a separate bounded 120 s cleanup allowance after a queue
        # deadline/stop, so partial scientific results can still be summarized.
        if self.manifest is not None and (Path(self.manifest['runner_root']) / 'report.py').is_file():
            previous_stage = self.status['stage']
            try:
                self.transition(stage='report')
                self.spawn('report', 'report.py', [], self.state / 'report.log', REPORT_TIMEOUT_S)
                outcome = self.wait_one(reporting=True)
                if outcome['status'] != 'completed':
                    self.transition(report_error=outcome)
            except Exception as error:
                self.transition(report_error=str(error))
            finally:
                terminate_all(self.active)
                with self.mutex:
                    self.active.clear()
                self.transition(stage=previous_stage)
        self.transition(finished_utc=now(), elapsed_s=time.monotonic() - self.started)


def supervise(state, lock_fd):
    state = Path(state).resolve()
    info, expected = os.fstat(lock_fd), (state / 'queue.lock').stat()
    if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Inherited lock descriptor does not match state/queue.lock')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock_fd, False)
    if (state / 'status.json').exists():
        os.close(lock_fd)
        raise FileExistsError('Existing status.json; refusing resume or overwrite')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    queue = Supervisor(state)

    def interrupted(signum, frame):
        # Do not raise asynchronously between Popen and recording child ownership.
        queue.stop_requested = signum

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    ticker = threading.Thread(target=queue.heartbeat, daemon=True, name='queue-heartbeat')
    try:
        queue.publish()
        ticker.start()
        try:
            queue.execute()
        except BaseException as error:
            queue.transition(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            traceback.print_exc()
        finally:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            queue.finish()
    finally:
        queue.done.set()
        if ticker.is_alive():
            ticker.join(timeout=5)
        os.close(lock_fd)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if queue.status['status'] == 'completed' else 1


def launch(state):
    state = Path(state).resolve()
    fd = acquire(state)
    try:
        initial_only(state)
        manifest = importlib.import_module('prepare').initialize(state)
        validate_queue(manifest, state)
        importlib.import_module('common').verify_manifest(manifest)
        # SIG_IGN survives exec, protecting even the interval before supervise's
        # Python signal handler is installed. No unsafe preexec_fn is needed.
        old_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            with (state / 'queue.log').open('x') as stream:
                process = subprocess.Popen([PYTHON, '-B', str(runner_script(manifest, 'pipeline.py')),
                                            'supervise', '--state', str(state), '--lock-fd', str(fd)],
                                           cwd=manifest['runtime_root'], env=environment(manifest),
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True, pass_fds=(fd,))
        finally:
            signal.signal(signal.SIGHUP, old_hup)
        deadline = time.monotonic() + HANDSHAKE_S
        while time.monotonic() < deadline:
            if (state / 'status.json').is_file():
                status = read(state / 'status.json')
                if (status.get('pid') == process.pid and alive(status) and status.get('status') == 'running'
                        and status.get('stage') == 'forward'
                        and any(i['id'] == 'forward' and alive(i) for i in status.get('active', []))
                        and time.time() - status.get('heartbeat_unix_s', 0) <= 5
                        and process.poll() is None):
                    return status
            if process.poll() is not None:
                raise RuntimeError(f'Queue exited at startup ({process.returncode}); inspect {state / "queue.log"}')
            time.sleep(.05)
        raise TimeoutError('No ready forward heartbeat within 30s; inspect queue.log/status.json; do not relaunch')
    finally:
        # Never unlock explicitly: the detached child owns this open-file-description lock.
        os.close(fd)


def get_status(state):
    status = read(Path(state) / 'status.json')
    status['scheduler_alive'] = alive(status)
    status['heartbeat_age_s'] = max(0, time.time() - status.get('heartbeat_unix_s', 0))
    for item in status.get('active', []):
        item['alive'] = alive(item)
    return status


def stop(state):
    """Birth-checked, scoped stop; reporting may take up to its 120 s allowance."""
    state = Path(state)
    status = read(state / 'status.json')
    if alive(status):
        try:
            os.kill(status['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + REPORT_TIMEOUT_S + 20
        while alive(status) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(status):
            raise TimeoutError('Supervisor has not stopped; inspect scoped status/child records')
    # Recover live, identity-matched orphan leaders after an external supervisor kill.
    # Hold the same lock before repairing a dead supervisor's persistent history.
    latest = read(state / 'status.json')
    if latest.get('status') != 'running' and not latest.get('active'):
        return get_status(state)
    fd = acquire(state)
    try:
        latest = read(state / 'status.json')
        if alive(latest):
            raise RuntimeError('Supervisor identity changed during stop; refusing orphan cleanup')
        groups = []
        for item in latest.get('active', []):
            try:
                if alive(item) and os.getpgid(item['pid']) == item['pid'] and os.getsid(item['pid']) == item['pid']:
                    groups.append(item)
                    os.killpg(item['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
        if groups:
            time.sleep(CLEANUP_GRACE_S)
            for item in groups:
                try:
                    # The group remains ours if its leader died; kill descendants too.
                    if identity(item['pid']) in (None, item['birth']):
                        os.killpg(item['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for item in latest.get('active', []):
            outcome = dict(id=item['id'], status='cancelled', finished_utc=now(),
                           error='Supervisor died; scoped stop recovery requested', exit_code=None, timed_out=False)
            latest.setdefault('finished_cases', []).append(outcome)
            record = latest.get('cases', {}).get(item['id'])
            if record is not None:
                record.update(outcome)
                directory = state / 'cases' / item['id']
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / 'result.json'
                result = read(path) if path.exists() else dict(case_id=item['id'], run_dir=None)
                result.update(status='cancelled', supervisor_outcome=outcome, finished_utc=outcome['finished_utc'])
                atomic(path, result)
                atomic(directory / 'supervisor.json', outcome)
        for record in latest.get('cases', {}).values():
            if record['status'] == 'pending':
                record['status'] = 'cancelled'
        latest.update(status='stopped', active=[], stage='finished', phase='finished', finished_utc=now(),
                      error='Supervisor was not alive; stop performed scoped orphan recovery', updated_utc=now())
        atomic(state / 'status.json', latest)
        with (state / 'queue.log').open('a') as stream:
            stream.write(json.dumps(dict(utc=now(), transition='orphan_stop_recovery')) + '\n')
    finally:
        os.close(fd)
    return get_status(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'stop', 'supervise'))
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args(argv)
    if args.action == 'supervise':
        if args.lock_fd is None:
            parser.error('supervise requires inherited --lock-fd; use launch')
        return supervise(args.state, args.lock_fd)
    if args.lock_fd is not None:
        parser.error('--lock-fd is only for supervise')
    result = {'launch': launch, 'status': get_status, 'stop': stop}[args.action](args.state)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
