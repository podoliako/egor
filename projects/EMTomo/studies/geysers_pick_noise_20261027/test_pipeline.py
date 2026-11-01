"""Solve-free queue/worker tests. No test launches the real study queue."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


pipeline = module('pick_noise_test_pipeline', HERE / 'pipeline.py')
worker = module('pick_noise_test_worker', HERE / 'em_worker.py')
config_module = module('pick_noise_test_config', HERE.parents[1] / 'config.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def plan(tmp_path):
    state = tmp_path / 'state'
    runner = state / 'runner'
    runner.mkdir(parents=True)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    manifest = dict(schema_version=1, study_id='geysers_pick_noise_20261027', state_dir=str(state),
                    runner_root=str(runner), runner_sha256={'pipeline.py': 'a' * 64},
                    runtime_root=str(runtime), runtime_sha256={'numerical.py': 'b' * 64},
                    experiments_root=str(state / 'experiments'), source_input_sha256={},
                    source_output_sha256={}, levels=[], cases=[], references={},
                    resources=dict(forward_workers=32, max_parallel_runs=3,
                                   inversion_workers_per_run=12, blas_threads=1), **pipeline.TIMEOUTS)
    for level_id, multiplier in zip(pipeline.LEVELS, (.5, 2, 3, 5, 10)):
        experiment_id = 'test_' + level_id
        manifest['levels'].append(dict(id=level_id, multiplier=multiplier, relative_sigma=.01 * multiplier,
                                       absolute_sigma_s=.01 * multiplier, seed=42, experiment_id=experiment_id))
        for variant in pipeline.VARIANTS:
            manifest['cases'].append(dict(id=f'{level_id}-{variant}', level_id=level_id,
                                           variant=variant, experiment_id=experiment_id))
    manifest['runner_sha256'] = {}
    for name in ('pipeline.py', 'forward.py', 'em_worker.py'):
        path = runner / name
        path.write_text('# fake runner, NEVER executed by these tests\n')
        manifest['runner_sha256'][name] = sha(path)
    pipeline.atomic(state / 'manifest.json', manifest)
    return manifest


def add_report(plan):
    path = Path(plan['runner_root']) / 'report.py'
    path.write_text('# fake report, NEVER executed by these tests\n')
    plan['runner_sha256']['report.py'] = sha(path)
    save_plan(plan)


def save_plan(plan):
    pipeline.atomic(Path(plan['state_dir']) / 'manifest.json', plan)


@pytest.fixture
def fake_queue(plan, monkeypatch):
    """Only mocked child processes; even 'forward.py' is never executed."""
    state = Path(plan['state_dir'])
    calls = []
    cleanup = []
    failures = {}
    never_finish = set()
    common = SimpleNamespace(verify_manifest=Mock())
    monkeypatch.setitem(sys.modules, 'common', common)
    gate = Mock(return_value={})
    monkeypatch.setattr(pipeline, 'forward_ready', gate)
    monkeypatch.setattr(pipeline, 'POLL_S', .001)
    monkeypatch.setattr(pipeline, 'identity', lambda pid: f'birth:{pid}' if pid else None)

    class Process:
        def __init__(self, command, **kwargs):
            assert command[0] == '/mnt/disk01/egor/venv/bin/python3'
            assert Path(command[2]).parent == state / 'runner'
            assert kwargs['start_new_session'] is True
            assert kwargs['stdin'] == subprocess.DEVNULL
            assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
            assert 'pass_fds' not in kwargs  # Workers never retain the queue lock.
            self.identifier = command[-1] if '--case-id' in command else Path(command[2]).stem
            self.command, self.kwargs = command, kwargs
            self.pid = 900000 + len(calls)
            self.returncode = None
            self.polls = 0
            calls.append(self)
            if '--case-id' in command:
                path = state / 'cases' / self.identifier / 'result.json'
                pipeline.atomic(path, dict(case_id=self.identifier, status='completed',
                                           run_dir=str(state / 'fake-runs' / self.identifier),
                                           final_model_sha256='c' * 64, provenance={'preserve': True}))

        def poll(self):
            self.polls += 1
            if self.identifier not in never_finish and self.polls >= 2 and self.returncode is None:
                self.returncode = failures.get(self.identifier, 0)
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    def terminate(items):
        for item in items:
            cleanup.append(item['id'])
            if item['process'].returncode is None:
                item['process'].returncode = -signal.SIGTERM
            item['handle'].close()

    monkeypatch.setattr(pipeline.subprocess, 'Popen', Process)
    monkeypatch.setattr(pipeline, 'terminate_all', terminate)
    return SimpleNamespace(state=state, calls=calls, cleanup=cleanup, failures=failures,
                           never_finish=never_finish, gate=gate, common=common)


def test_manifest_queue_contract(plan):
    pipeline.validate_queue(plan, Path(plan['state_dir']))
    assert pipeline.HANDSHAKE_S <= 30
    assert pipeline.HEARTBEAT_S <= 5
    assert pipeline.REPORT_TIMEOUT_S <= 120
    for name, value in pipeline.TIMEOUTS.items():
        assert pipeline.timeout(plan, name) == value
    nested = deepcopy(plan)
    nested['timeouts'] = {pipeline.LEGACY_TIMEOUTS[k]: nested.pop(k) for k in pipeline.TIMEOUTS}
    pipeline.validate_queue(nested, Path(nested['state_dir']))
    plan['cases'].pop()
    with pytest.raises(ValueError, match='fifteen'):
        pipeline.validate_queue(plan, Path(plan['state_dir']))


@pytest.mark.parametrize('value', [None, 0, -1, float('inf'), float('nan'), True, 86401])
def test_finite_timeout_rejects_invalid(plan, value):
    plan['queue_timeout_s'] = value
    with pytest.raises(ValueError, match='finite timeout'):
        pipeline.timeout(plan, 'queue_timeout_s')


def test_exclusive_lock_and_initial_only(tmp_path):
    fd = pipeline.acquire(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='duplicate'):
            pipeline.acquire(tmp_path)
    finally:
        os.close(fd)
    fd = pipeline.acquire(tmp_path)
    os.close(fd)
    path = tmp_path / 'status.json'
    path.write_text('{"status":"failed"}')
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match='refuses resume'):
        pipeline.launch(tmp_path)
    assert path.read_bytes() == original


def test_inherited_lock_survives_launcher_fd_close(tmp_path):
    """A tiny lock-holder, not the queue supervisor or a numerical worker."""
    fd = pipeline.acquire(tmp_path)
    script = 'import time; print("ready",flush=True); time.sleep(.2)'
    process = subprocess.Popen([sys.executable, '-B', '-c', script], pass_fds=(fd,),
                               start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        os.close(fd)
        fd = None
        with pytest.raises(RuntimeError, match='duplicate'):
            pipeline.acquire(tmp_path)
        process.wait(timeout=5)
        released = pipeline.acquire(tmp_path)
        os.close(released)
    finally:
        if fd is not None:
            os.close(fd)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def test_birth_token_prevents_recycled_pid(monkeypatch):
    monkeypatch.setattr(pipeline, 'identity', lambda pid: 'new-birth')
    assert not pipeline.alive(dict(pid=123, birth='old-birth'))
    assert not pipeline.alive(dict(pid=123))
    assert pipeline.alive(dict(pid=123, birth='new-birth'))


def test_real_process_birth_token():
    token = pipeline.identity(os.getpid())
    assert token and pipeline.alive(dict(pid=os.getpid(), birth=token))


def test_queue_all_fifteen_success_and_report(plan, fake_queue, monkeypatch):
    add_report(plan)
    queue = pipeline.Supervisor(fake_queue.state)
    maximum = []
    original = queue.spawn

    def spawn(*args, **kwargs):
        original(*args, **kwargs)
        maximum.append(len(queue.active))

    monkeypatch.setattr(queue, 'spawn', spawn)
    queue.execute()
    queue.finish()
    assert queue.status['status'] == 'completed'
    assert len(queue.status['cases']) == 15
    assert all(c['status'] == 'completed' and c['run_dir'] for c in queue.status['cases'].values())
    assert max(maximum) == 3
    assert [p.identifier for p in fake_queue.calls] == ['forward', *[c['id'] for c in plan['cases']], 'report']
    assert fake_queue.gate.call_count == 16  # after forward and before EACH case
    status = pipeline.read(fake_queue.state / 'status.json')
    assert status['phase'] == 'finished' and not status['active']
    assert status['supervisor'] == {'pid': os.getpid(), 'birth': f'birth:{os.getpid()}'}


def test_case_failure_does_not_block_others(plan, fake_queue):
    failed = plan['cases'][1]['id']
    fake_queue.failures[failed] = 7
    queue = pipeline.Supervisor(fake_queue.state)
    queue.execute()
    queue.finish()
    assert queue.status['status'] == 'completed_with_failures'
    assert len(fake_queue.calls) == 16
    assert sum(c['status'] == 'completed' for c in queue.status['cases'].values()) == 14
    result = pipeline.read(fake_queue.state / 'cases' / failed / 'result.json')
    assert result['status'] == 'failed' and result['provenance']['preserve']
    assert result['supervisor_outcome']['exit_code'] == 7


@pytest.mark.parametrize('invalid_ledger', [False, True])
def test_forward_failure_blocks_every_inversion(plan, fake_queue, invalid_ledger):
    if invalid_ledger:
        fake_queue.gate.side_effect = ValueError('changed cache')
    else:
        fake_queue.failures['forward'] = 9
    queue = pipeline.Supervisor(fake_queue.state)
    with pytest.raises(RuntimeError, match='ALL inversions blocked'):
        queue.execute()
    queue.transition(status='failed')
    queue.finish()
    assert [p.identifier for p in fake_queue.calls] == ['forward']
    assert all(c['status'] == 'blocked' for c in queue.status['cases'].values())


def test_pre_case_hash_gate_blocks_changed_data(plan, fake_queue):
    fake_queue.gate.side_effect = [None] + [ValueError('changed derived arrivals')] * 15
    queue = pipeline.Supervisor(fake_queue.state)
    queue.execute()
    queue.finish()
    assert [p.identifier for p in fake_queue.calls] == ['forward']
    assert queue.status['status'] == 'completed_with_failures'
    assert all(c['status'] == 'failed' for c in queue.status['cases'].values())


def test_case_timeout_cleans_group_and_continues(plan, fake_queue):
    plan['case_timeout_s'] = .1
    save_plan(plan)
    slow = plan['cases'][0]['id']
    fake_queue.never_finish.add(slow)
    queue = pipeline.Supervisor(fake_queue.state)
    queue.execute()
    queue.finish()
    assert queue.status['cases'][slow]['status'] == 'timed_out'
    assert slow in fake_queue.cleanup
    assert sum(c['status'] == 'completed' for c in queue.status['cases'].values()) == 14


def test_forward_timeout_blocks_all_inversions(plan, fake_queue):
    plan['forward_timeout_s'] = .01
    save_plan(plan)
    fake_queue.never_finish.add('forward')
    queue = pipeline.Supervisor(fake_queue.state)
    with pytest.raises(RuntimeError, match='ALL inversions blocked'):
        queue.execute()
    queue.transition(status='failed')
    queue.finish()
    assert [p.identifier for p in fake_queue.calls] == ['forward']
    assert queue.status['finished_cases'][0]['status'] == 'timed_out'
    assert all(c['status'] == 'blocked' for c in queue.status['cases'].values())


def test_overall_timeout_supervisor_cleans_and_reports(plan, fake_queue):
    plan['queue_timeout_s'] = .01
    save_plan(plan)
    fake_queue.never_finish.add('forward')
    add_report(plan)
    fd = pipeline.acquire(fake_queue.state)
    assert pipeline.supervise(fake_queue.state, fd) == 1
    status = pipeline.read(fake_queue.state / 'status.json')
    assert status['status'] == 'failed'
    assert 'deadline' in status['error']
    assert not status['active']
    assert fake_queue.cleanup == ['forward', 'report']
    assert all(c['status'] == 'blocked' for c in status['cases'].values())
    fd = pipeline.acquire(fake_queue.state)  # all inherited lock ownership released
    os.close(fd)


def test_parent_stop_cleans_remaining_children_and_preserves_results(plan, fake_queue, monkeypatch):
    original = pipeline.Supervisor.wait_one

    def interrupted(self, **kwargs):
        if self.status['stage'] == 'inversions':
            raise KeyboardInterrupt('test stop')
        return original(self, **kwargs)

    monkeypatch.setattr(pipeline.Supervisor, 'wait_one', interrupted)
    fd = pipeline.acquire(fake_queue.state)
    assert pipeline.supervise(fake_queue.state, fd) == 1
    status = pipeline.read(fake_queue.state / 'status.json')
    assert status['status'] == 'stopped' and not status['active']
    assert len(fake_queue.calls) == 4
    for case in plan['cases'][:3]:
        result = pipeline.read(fake_queue.state / 'cases' / case['id'] / 'result.json')
        assert result['status'] == 'cancelled' and result['provenance']['preserve']
        assert case['id'] in fake_queue.cleanup
    assert all(c['status'] == 'cancelled' for c in status['cases'].values())


def test_stop_signal_during_popen_cannot_leave_untracked_child(plan, fake_queue, monkeypatch):
    original = pipeline.subprocess.Popen

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    fd = pipeline.acquire(fake_queue.state)
    assert pipeline.supervise(fake_queue.state, fd) == 1
    status = pipeline.read(fake_queue.state / 'status.json')
    assert status['status'] == 'stopped' and not status['active']
    assert fake_queue.cleanup == ['forward']


def test_modified_frozen_worker_is_not_executed(plan, fake_queue):
    (fake_queue.state / 'runner' / 'em_worker.py').write_text('tampered runner')
    queue = pipeline.Supervisor(fake_queue.state)
    queue.execute()
    queue.finish()
    assert queue.status['status'] == 'completed_with_failures'
    assert [p.identifier for p in fake_queue.calls] == ['forward']
    assert all('Frozen runner modified' in c['error'] for c in queue.status['cases'].values())


def test_report_timeout_is_bounded_and_does_not_erase_scientific_success(plan, fake_queue, monkeypatch):
    add_report(plan)
    monkeypatch.setattr(pipeline, 'REPORT_TIMEOUT_S', .005)
    fake_queue.never_finish.add('report')
    queue = pipeline.Supervisor(fake_queue.state)
    queue.execute()
    queue.finish()
    assert queue.status['status'] == 'completed'
    assert queue.status['report_error']['timed_out']
    assert not queue.active


def test_heartbeat_runs_during_slow_preflight(plan, fake_queue, monkeypatch):
    monkeypatch.setattr(pipeline, 'HEARTBEAT_S', .005)
    ticks = []

    def slow_verify(manifest):
        time.sleep(.03)
        ticks.append(pipeline.read(fake_queue.state / 'status.json')['heartbeat_unix_s'])
        time.sleep(.03)
        ticks.append(pipeline.read(fake_queue.state / 'status.json')['heartbeat_unix_s'])

    fake_queue.common.verify_manifest.side_effect = slow_verify
    fd = pipeline.acquire(fake_queue.state)
    assert pipeline.supervise(fake_queue.state, fd) == 0
    assert ticks[1] > ticks[0]


def test_rejected_manual_supervisor_never_overwrites_history(plan):
    state = Path(plan['state_dir'])
    path = state / 'status.json'
    pipeline.atomic(path, {'status': 'completed', 'pid': 99, 'birth': 'old'})
    before = path.read_bytes()
    fd = pipeline.acquire(state)
    with pytest.raises(FileExistsError, match='refusing resume'):
        pipeline.supervise(state, fd)
    assert path.read_bytes() == before


def test_launch_requires_running_forward_heartbeat_and_inherits_lock(plan, monkeypatch):
    state = Path(plan['state_dir'])
    initialize = Mock(return_value=plan)
    monkeypatch.setitem(sys.modules, 'prepare', SimpleNamespace(initialize=initialize))
    monkeypatch.setitem(sys.modules, 'common', SimpleNamespace(verify_manifest=Mock()))
    monkeypatch.setattr(pipeline, 'identity', lambda pid: f'birth:{pid}')
    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        with pytest.raises(RuntimeError, match='duplicate'):
            pipeline.acquire(state)
        assert os.fstat(kwargs['pass_fds'][0])
        pipeline.atomic(state / 'status.json', dict(pid=1234, birth='birth:1234', status='running', stage='forward',
                        heartbeat_unix_s=time.time(), active=[dict(id='forward', pid=1235, birth='birth:1235')]))
        return SimpleNamespace(pid=1234, poll=lambda: None)

    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    result = pipeline.launch(state)
    assert result['status'] == 'running'
    initialize.assert_called_once_with(state)
    assert captured['command'][:3] == [pipeline.PYTHON, '-B', str(state / 'runner' / 'pipeline.py')]
    assert captured['stdin'] == subprocess.DEVNULL and captured['start_new_session']
    assert captured['stderr'] == subprocess.STDOUT
    assert captured['stdout'].name == str(state / 'queue.log')


@pytest.mark.parametrize('state_status', ['completed', 'failed', 'running'])
def test_launch_does_not_accept_merely_created_process(plan, monkeypatch, state_status):
    state = Path(plan['state_dir'])
    monkeypatch.setitem(sys.modules, 'prepare', SimpleNamespace(initialize=lambda path: plan))
    monkeypatch.setitem(sys.modules, 'common', SimpleNamespace(verify_manifest=Mock()))
    monkeypatch.setattr(pipeline, 'HANDSHAKE_S', .01)
    monkeypatch.setattr(pipeline, 'identity', lambda pid: 'birth')

    def popen(*args, **kwargs):
        pipeline.atomic(state / 'status.json', dict(pid=555, birth='birth', status=state_status,
                        stage='forward', heartbeat_unix_s=time.time(), active=[]))
        return SimpleNamespace(pid=555, poll=lambda: None)

    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    with pytest.raises(TimeoutError, match='No ready forward heartbeat'):
        pipeline.launch(state)


def test_stop_never_signals_recycled_supervisor_pid(tmp_path, monkeypatch):
    pipeline.atomic(tmp_path / 'status.json', dict(status='running', pid=777, birth='old', active=[]))
    monkeypatch.setattr(pipeline, 'identity', lambda pid: 'recycled')
    kill = Mock()
    monkeypatch.setattr(pipeline.os, 'kill', kill)
    pipeline.stop(tmp_path)
    kill.assert_not_called()


def test_orphan_stop_preserves_partial_history_and_kills_scoped_group(tmp_path, monkeypatch):
    directory = tmp_path / 'cases' / 'case-a'
    directory.mkdir(parents=True)
    pipeline.atomic(directory / 'result.json', dict(case_id='case-a', status='running', run_dir='/partial/run',
                                                   provenance={'keep': True}))
    pipeline.atomic(tmp_path / 'status.json', dict(status='running', pid=777, birth='dead',
                    active=[dict(id='case-a', pid=888, birth='child')],
                    cases={'case-a': dict(status='running'), 'case-b': dict(status='pending')}, finished_cases=[]))
    identities = {888: 'child'}
    monkeypatch.setattr(pipeline, 'identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(pipeline.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(pipeline.os, 'getsid', lambda pid: pid)
    monkeypatch.setattr(pipeline, 'CLEANUP_GRACE_S', 0)
    signals = []

    def killpg(pid, signum):
        signals.append((pid, signum))
        identities.pop(pid, None)  # leader exits on TERM, its descendants still need KILL

    monkeypatch.setattr(pipeline.os, 'killpg', killpg)
    status = pipeline.stop(tmp_path)
    assert signals == [(888, signal.SIGTERM), (888, signal.SIGKILL)]
    assert status['status'] == 'stopped' and not status['active']
    assert all(c['status'] == 'cancelled' for c in status['cases'].values())
    result = pipeline.read(directory / 'result.json')
    assert result['provenance'] == {'keep': True} and result['run_dir'] == '/partial/run'
    assert result['status'] == 'cancelled'


def test_status_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / 'status.json'
    pipeline.atomic(path, dict(status='running', pid=777, birth='birth', heartbeat_unix_s=time.time(),
                              active=[dict(id='forward', pid=888, birth='child')]))
    original = path.read_bytes()
    monkeypatch.setattr(pipeline, 'identity', lambda pid: {777: 'birth', 888: 'child'}.get(pid))
    status = pipeline.get_status(tmp_path)
    assert status['scheduler_alive'] and status['active'][0]['alive']
    assert path.read_bytes() == original


def test_cleanup_kills_descendant_after_leader_has_exited(tmp_path, monkeypatch):
    """A tiny sleeper process tree, never a forward or inversion process."""
    monkeypatch.setattr(pipeline, 'CLEANUP_GRACE_S', .05)
    script = ("import subprocess,sys; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
              "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
              "print(p.pid,flush=True)")
    process = subprocess.Popen([sys.executable, '-B', '-c', script], start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    child_pid = None
    handle = (tmp_path / 'tiny-child.log').open('w')
    try:
        child_pid = int(process.stdout.readline())
        process.wait(timeout=5)
        assert pipeline.identity(child_pid)
        pipeline.terminate_all([dict(process=process, handle=handle)])
        deadline = time.monotonic() + 3
        while pipeline.identity(child_pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert pipeline.identity(child_pid) is None
        assert handle.closed
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()
        handle.close()


@pytest.fixture
def worker_case(plan, tmp_path, monkeypatch):
    project = tmp_path / 'project'
    monkeypatch.setattr(worker, 'PROJECT', project)
    base = config_module.InversionConfig(
        cell_size=1000., subdivision=8, n_cycles=12, n_candidates=15, n_workers=12,
        candidate_mode='hard', weights_top_n=1, weights_min_distance=1,
        weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.01, weight_model_sigma_s=.02,
        run_name='original', runs_dir='original/runs', initial_layer_boundaries_km=(.7, 1.9, 3.4),
        initial_layer_velocities_m_s=(3800., 4600., 5200., 5700.))
    initial = np.full(worker.GRID, 4500., dtype=np.float64)
    for variant, (mode, top, distance) in worker.MODES.items():
        reference = tmp_path / variant
        reference.mkdir()
        np.save(reference / 'initial_model.npy', initial)
        original = replace(base, candidate_mode=mode, weights_top_n=top, weights_min_distance=distance)
        plan['references'][variant] = dict(config=worker.canonical(asdict(original)), run_dir=str(reference), case_id='old-' + variant)
    for case in plan['cases']:
        ref = plan['references'][case['variant']]
        case['config'] = dict(ref['config'], run_name='gpnoise27-' + case['id'], runs_dir=str(project / 'runs'))
        case['reference_run'] = ref['run_dir']
    plan['source_input_sha256'] = {'model.npz': 'd' * 64}
    save_plan(plan)
    case = plan['cases'][-1]  # largest actual noise, fixed likelihood still .01/.01/.02
    level = plan['levels'][-1]
    data = SimpleNamespace(event_ids=tuple(range(3776)), station_ids=tuple(range(26)),
                           arrivals_table=np.zeros(worker.SHAPE), station_locs=np.zeros((26, 3)),
                           noise_sigmas=(level['relative_sigma'], level['absolute_sigma_s']),
                           initial_model=SimpleNamespace(velocity=initial, shape=worker.GRID), reference_model=object())
    ledger = dict(clean_sha256='e' * 64, levels={level['id']: {'output_sha256': {'arrivals.csv': 'f' * 64}}})
    monkeypatch.setattr(worker, 'verify_forward', Mock(return_value=ledger))
    prepare = Mock(return_value=data)

    def resolve(config, sigmas):
        return SimpleNamespace(relative_sigma=config.weight_noise_relative_sigma,
                               absolute_sigma_s=config.weight_noise_absolute_sigma_s, model_sigma_s=config.weight_model_sigma_s)

    class Logger:
        def __init__(self, *, base_dir, run_name, **kwargs):
            self.run_dir = Path(base_dir) / ('run_' + run_name + '-mock')
            self.run_dir.mkdir(parents=True)
            self.meta = None

        def save_meta(self, **kwargs):
            self.meta = kwargs

    runtime = SimpleNamespace(config_module=config_module,
                              runner=SimpleNamespace(prepare_inversion=prepare, resolve_pick_noise=resolve),
                              em=SimpleNamespace(TomographyLogger=Logger, warm_up_jit=Mock(), run_em=Mock()))
    common = SimpleNamespace(read=pipeline.read, atomic=pipeline.atomic, now=pipeline.now, sha=sha,
                             verify_manifest=Mock(), load_runtime=Mock(return_value=runtime))
    return SimpleNamespace(manifest=plan, case=case, data=data, runtime=runtime, common=common, level=level)


def test_worker_all_fifteen_configs_preserve_every_scientific_field(worker_case):
    m = worker_case.manifest
    for case in m['cases']:
        config = worker.build_config(m, case, config_module.InversionConfig)
        original = m['references'][case['variant']]['config']
        for key, value in worker.canonical(asdict(config)).items():
            if key not in worker.LOGGING_FIELDS:
                assert value == original[key]


@pytest.mark.parametrize('field,value', [('lambda_reg', 9.), ('weight_noise_relative_sigma', .1),
                                        ('weight_noise_absolute_sigma_s', .1), ('weight_model_sigma_s', .2),
                                        ('fast_refinement', False), ('n_workers', 1), ('n_cycles', 1)])
def test_worker_rejects_any_scientific_config_change(worker_case, field, value):
    worker_case.case['config'][field] = value
    with pytest.raises(ValueError, match='Scientific configuration changed'):
        worker.build_config(worker_case.manifest, worker_case.case, config_module.InversionConfig)


def test_worker_fixed_noise_and_result_written_before_iteration_one(worker_case, monkeypatch):
    w = worker_case
    path = Path(w.manifest['state_dir']) / 'cases' / w.case['id'] / 'result.json'
    seen = []

    def solve(config, initial, arrivals, stations, noise, **kwargs):
        saved = pipeline.read(path)
        assert saved['status'] == 'running' and saved['run_dir'] == str(kwargs['logger'].run_dir)
        assert w.runtime.em.warm_up_jit.call_count == 1
        assert (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) == (.01, .01, .02)
        assert w.data.noise_sigmas == (.1, .1)
        kwargs['logger'].save_meta(run_params={'test': True})
        seen.append(kwargs['logger'].meta['run_params']['provenance'])

    w.runtime.em.run_em.side_effect = solve
    monkeypatch.setattr(worker, 'validate_outputs', Mock(return_value=dict(final_model_sha256='a' * 64,
                                                                         final_quality={'iter': 11})))
    result = worker.run(w.manifest, w.case['id'], w.common, manifest_path=Path(w.manifest['state_dir']) / 'manifest.json')
    assert result['status'] == 'completed'
    assert seen[0]['actual_noise']['relative_sigma'] == .1
    assert seen[0]['fixed_assumed_noise'] == dict(relative_sigma=.01, absolute_sigma_s=.01, model_sigma_s=.02)
    assert seen[0]['runtime_sha256'] == w.manifest['runtime_sha256']
    assert worker.verify_forward.call_count == 2
    assert w.common.load_runtime.call_count == 1


@pytest.mark.parametrize('problem', ['geometry', 'noise', 'initial', 'resolver'])
def test_worker_fails_closed_before_solve(worker_case, problem):
    w = worker_case
    if problem == 'geometry':
        w.data.event_ids = (0,)
    elif problem == 'noise':
        w.data.noise_sigmas = (.01, .01)
    elif problem == 'initial':
        w.data.initial_model.velocity[0, 0, 0] += 1
    else:
        w.runtime.runner.resolve_pick_noise = lambda *args: SimpleNamespace(relative_sigma=.1, absolute_sigma_s=.1, model_sigma_s=.02)
    with pytest.raises(ValueError):
        worker.run(w.manifest, w.case['id'], w.common, manifest_path=Path(w.manifest['state_dir']) / 'manifest.json')
    w.runtime.em.run_em.assert_not_called()
    result = pipeline.read(Path(w.manifest['state_dir']) / 'cases' / w.case['id'] / 'result.json')
    assert result['status'] == 'failed' and result['traceback']


def test_worker_failure_history_and_duplicate_attempt_are_preserved(worker_case):
    w = worker_case
    w.runtime.em.run_em.side_effect = RuntimeError('mock solve failed')
    manifest_path = Path(w.manifest['state_dir']) / 'manifest.json'
    with pytest.raises(RuntimeError, match='mock solve failed'):
        worker.run(w.manifest, w.case['id'], w.common, manifest_path=manifest_path)
    path = manifest_path.parent / 'cases' / w.case['id'] / 'result.json'
    result = pipeline.read(path)
    assert result['status'] == 'failed' and result['run_dir']
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match='refusing resume'):
        worker.run(w.manifest, w.case['id'], w.common, manifest_path=manifest_path)
    assert path.read_bytes() == before
    assert w.runtime.em.run_em.call_count == 1


def test_worker_output_validation_checks_every_cycle_and_event(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'SHAPE', (2, 26))  # Tiny on disk, same exhaustive loop.
    np.save(tmp_path / 'final_model.npy', np.full(worker.GRID, 4000.))
    (tmp_path / 'quality.jsonl').write_text(''.join(json.dumps(dict(iter=i, rms=.1)) + '\n' for i in range(12)))
    for i in range(12):
        cycle = tmp_path / f'iter_{i}'
        cycle.mkdir()
        pipeline.atomic(cycle / 'complete.json', dict(iter=i))
        for event in range(2):
            directory = cycle / f'event_{event}'
            directory.mkdir()
            (directory / 'weights.npz').write_bytes(b'mock weights')
    common = SimpleNamespace(read=pipeline.read, sha=sha)
    result = worker.validate_outputs(tmp_path, np, common)
    assert result['completed_cycles'] == 12 and result['final_quality']['iter'] == 11
    (tmp_path / 'iter_11' / 'event_1' / 'weights.npz').unlink()
    with pytest.raises(ValueError, match='Missing event weights: 11/1'):
        worker.validate_outputs(tmp_path, np, common)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_worker_rejects_invalid_final_model(tmp_path, value):
    final = np.full(worker.GRID, 4000.)
    final[0, 0, 0] = value
    np.save(tmp_path / 'final_model.npy', final)
    with pytest.raises(ValueError, match='Invalid final'):
        worker.validate_outputs(tmp_path, np, SimpleNamespace())


@pytest.fixture
def forward_ledger(plan):
    state = Path(plan['state_dir'])
    (state / 'forward').mkdir()
    cache = state / 'forward' / 'clean_times.npz'
    cache.write_bytes(b'mock immutable clean cache')
    plan['source_input_sha256'] = {name: hashlib.sha256(name.encode()).hexdigest() for name in worker.INPUTS}
    binding = worker.manifest_digest(plan)
    cache_hash = sha(cache)
    pipeline.atomic(state / 'forward' / 'clean_times.json', dict(status='completed', manifest_digest=binding,
                    shape=list(worker.SHAPE), dtype='float64', time_reference='absolute_propagation_time_s',
                    source_input_sha256=plan['source_input_sha256'], runtime_sha256=plan['runtime_sha256'], clean_sha256=cache_hash))
    ledger = dict(status='completed', manifest_digest=binding, clean_sha256=cache_hash,
                  baseline_reproduction=dict(status='completed', allclose=True, pairs=3776 * 26, max_abs_difference_s=0.), levels={})
    root = Path(plan['experiments_root'])
    for level in plan['levels']:
        record = dict(status='completed', manifest_digest=binding, experiment_id=level['experiment_id'],
                      level_id=level['id'], pairs=3776 * 26)
        for category, names in (('input', worker.INPUTS), ('output', worker.OUTPUTS)):
            directory = root / category / level['experiment_id']
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_text(name)
            record[category + '_sha256'] = {name: sha(directory / name) for name in names}
        ledger['levels'][level['id']] = record
    pipeline.atomic(state / 'forward_complete.json', ledger)
    common = SimpleNamespace(read=pipeline.read, sha=sha, verify_manifest=Mock())
    return SimpleNamespace(common=common, ledger=ledger, state=state)


@pytest.mark.parametrize('tamper', [None, 'cache', 'arrivals', 'inputs', 'ledger', 'baseline', 'runtime'])
def test_forward_hash_gate_checks_cache_and_all_derived_data(plan, forward_ledger, tamper):
    f = forward_ledger
    if tamper == 'cache':
        (f.state / 'forward' / 'clean_times.npz').write_bytes(b'changed')
    elif tamper in ('arrivals', 'inputs'):
        category, filename = ('output', 'arrivals.csv') if tamper == 'arrivals' else ('input', 'events.csv')
        path = Path(plan['experiments_root']) / category / plan['levels'][-1]['experiment_id'] / filename
        path.write_text('changed')
    elif tamper == 'ledger':
        f.ledger['levels'].pop('x10')
        pipeline.atomic(f.state / 'forward_complete.json', f.ledger)
    elif tamper == 'baseline':
        f.ledger['baseline_reproduction']['allclose'] = False
        pipeline.atomic(f.state / 'forward_complete.json', f.ledger)
    elif tamper == 'runtime':
        f.common.verify_manifest.side_effect = ValueError('runtime changed')
    if tamper is None:
        assert worker.verify_forward(plan, f.common)['status'] == 'completed'
    else:
        with pytest.raises(ValueError):
            worker.verify_forward(plan, f.common)
