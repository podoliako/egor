"""Four frozen-input tomoDD sensitivity cases. No solver runs with --validate-only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'runs/matched_geometry_20261021'
SNAPSHOT = OLD / 'snapshot'
BASELINE = OLD / 'outputs/synthetic-tomodd-sigma1'
DEFAULT_ROOT = PROJECT / 'runs/tomodd_sensitivity_20261026'
WORKER_PATH = HERE.parent / 'matched_geometry_20261021/tomodd_worker.py'
SOLVER_TIMEOUT_S = 1200
WORKER_TIMEOUT_S = 1500
VALIDATION_TIMEOUT_S = 240
SCRIPT_VERSION = 'retry-compact80-v2'
REUSABLE_CASE = 'dd1-d40-s5'
PROVENANCE_KEYS = ('input_sha256', 'support_sha256', 'snapshot_source_sha256',
                   'solver_sha256', 'solver_manifest_sha256', 'worker_sha256')
CASES = (
    {'id': 'dd03-d40-s5', 'dd_weight': 0.3, 'damping': 40.0, 'smoothing': [5.0] * 3},
    {'id': 'dd1-d40-s5', 'dd_weight': 1.0, 'damping': 40.0, 'smoothing': [5.0] * 3},
    {'id': 'dd03-d80-s10', 'dd_weight': 0.3, 'damping': 80.0, 'smoothing': [10.0] * 3},
    {'id': 'dd03-d20-s2', 'dd_weight': 0.3, 'damping': 20.0, 'smoothing': [2.0] * 3},
)
CONSTANT_FILES = ('MOD', 'grid.dat', 'event.dat', 'station.dat', 'dt.ct', 'absolute.dat',
                  'dt.cc', 'ak135.15.SKS', 'layer-16.dat')
SCHEDULE_MARKER = '* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTDD DAMP JOINT THRE'
SMOOTH_MARKER = '* weight1 weight2 weight3 CC_format'

spec = importlib.util.spec_from_file_location('matched_tomodd_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
require = worker.require


def solver_schedule(case):
    q = case['dd_weight']
    return ((3, q, 0.08, -9.0, 1.0 / q, case['damping'], 1, 0.0),)


def case_config(case, root):
    return dict(case, output=str(root / 'outputs' / case['id']), absolute_weight=1.0,
                solver_schedule=solver_schedule(case), timeout_s=SOLVER_TIMEOUT_S)


def replace_data_line(text, marker, transform):
    lines = text.splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if line.strip() == marker]
    require(len(indices) == 1, f'Unexpected control layout: {marker}')
    index = indices[0] + 1
    lines[index] = transform(lines[index])
    return ''.join(lines)


def data_tokens(text, marker):
    result = []
    replace_data_line(text, marker, lambda line: result.extend(line.split()) or line)
    return result


def check_data_line_lengths(text):
    lengths = [len(line) for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith('*')]
    require(all(length <= 80 for length in lengths), 'Solver control data line exceeds 80 columns')
    return max(lengths, default=0)


def render_control(original, cfg, case, **kwargs):
    """Keep precision without overflowing the frozen solver's character line*80."""
    text = original(cfg, **kwargs)

    def precise_wtdd(line):
        tokens = line.split()
        require(len(tokens) == 13, 'Unexpected schedule field count')
        tokens[9] = format(1.0 / case['dd_weight'], '.17g')
        return ' '.join(tokens) + '\n'

    text = replace_data_line(text, SCHEDULE_MARKER, precise_wtdd)
    check_control(text, case)
    return text


def check_control(text, case):
    check_data_line_lengths(text)
    values = list(map(float, data_tokens(text, SCHEDULE_MARKER)))
    expected = [3, -9, -9, -9, -9, case['dd_weight'], .08, -9, -9,
                1 / case['dd_weight'], case['damping'], 1, 0]
    require(values == expected, 'Serialized schedule differs from declared case')
    require(list(map(float, data_tokens(text, SMOOTH_MARKER))) == case['smoothing'] + [1],
            'Serialized smoothing differs from declared case')
    require(math.isclose(values[5] * values[9], 1.0, rel_tol=0, abs_tol=2e-16),
            'Serialized effective absolute amplitude is not unity')
    # Every other control, including P-only, ray step and joint flags, is frozen.
    baseline = (BASELINE / 'work/inversion/tomoDD-SE.inp').read_text()
    for marker in (SCHEDULE_MARKER, SMOOTH_MARKER):
        text_line = []
        replace_data_line(text, marker, lambda line: text_line.append(line) or line)
        baseline = replace_data_line(baseline, marker, lambda line: text_line[0])
    require(text == baseline, 'Other generated solver controls differ from baseline')


def check_generated(inversion, baseline=None):
    baseline = BASELINE if baseline is None else baseline
    hashes = {}
    for name in CONSTANT_FILES:
        actual = worker.digest(inversion / name)
        require(actual == worker.digest(baseline / 'work/inversion' / name),
                f'Generated input differs from baseline: {name}')
        hashes[name] = actual
    return hashes


class InputsValidated(Exception):
    """Stop adapter.run at the control hook, before it can invoke the binary."""


def configure_adapter(adapter, case, output, info, validate_only):
    # Called only AFTER the original worker's default-protocol preflight.
    def solver_config(iterations, node_spacing_km):
        require((iterations, node_spacing_km) == (3, 1.0), 'Unexpected solver geometry/iterations')
        return adapter.SolverConfig(
            profile=f'tomodd-sensitivity-20261026-{case["id"]}',
            smoothing_lon_lat_depth=tuple(case['smoothing']),
            iteration_sets=solver_schedule(case))

    original = adapter._input_text

    def input_text(cfg, **kwargs):
        text = render_control(original, cfg, case, **kwargs)
        info['generated_input_sha256'] = check_generated(output / 'work/inversion')
        tokens = data_tokens(text, SCHEDULE_MARKER)
        import numpy as np
        q, wtdd = float(tokens[5]), float(tokens[9])
        info['serialized_weights'] = {
            'WTCTP': tokens[5], 'WTDD': tokens[9], 'absolute_amplitude': q * wtdd,
            'absolute_amplitude_error': q * wtdd - 1.0,
            'float32_product': float(np.float32(q) * np.float32(wtdd)),
            'unmodified_renderer_absolute_amplitude': q * float(format(wtdd, 'g')),
            'note': 'WTDD rendered with .17g; schedule whitespace compacted for line*80; REAL still rounds.',
        }
        info['generated_control_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        info['max_control_data_line_length'] = check_data_line_lengths(text)
        if validate_only:
            raise InputsValidated()
        return text

    adapter._solver_config = solver_config
    adapter._input_text = input_text


def prepare_case(case, root, output):
    args = argparse.Namespace(geometry='synthetic', output=output,
                              benchmark_root=SNAPSHOT / 'benchmark',
                              forward_root=SNAPSHOT / 'projects/forward_modeling',
                              solver_build=SNAPSHOT / 'solver', support_root=SNAPSHOT / 'support')
    adapter = worker.load_adapter(args.benchmark_root)
    arguments, info = worker.preflight(args, adapter)
    baseline_info = json.loads((BASELINE / 'worker_info.json').read_text())
    require(baseline_info['state'] == 'completed', 'Baseline is not completed')
    for key in ('input_sha256', 'support_sha256', 'snapshot_source_sha256', 'solver_sha256'):
        require(info[key] == baseline_info[key], f'Frozen provenance differs from baseline: {key}')
    arguments['timeout_s'] = SOLVER_TIMEOUT_S
    info.update(case_config(case, root), baseline=str(BASELINE),
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()},
                smoothing_lon_lat_depth=case['smoothing'], absolute_dd_weight=[1., case['dd_weight']],
                script_version=SCRIPT_VERSION, script_sha256=worker.digest(Path(__file__)))
    # Record the eventual retained location, not validate-only's temporary directory.
    info['arguments']['run_dir'] = str(root / 'outputs' / case['id'])
    info['timeout_s'] = SOLVER_TIMEOUT_S
    return adapter, arguments, info


def validate_reuse(case, source, adapter, fresh):
    """Read-only import of the one approved completed attempt-one case."""
    require(case['id'] == REUSABLE_CASE, 'Only dd1-d40-s5 may be reused')
    source = source.resolve(strict=True)
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    matches = [c for c in manifest['cases'] if c['id'] == case['id']]
    require(len(matches) == 1, 'Original manifest must contain exactly one completed-case declaration')
    declared = matches[0]
    expected = json.loads(json.dumps(case_config(case, source)))
    require(all(declared[k] == v for k, v in expected.items()), 'Original case differs from expected config')
    require(manifest['baseline'] == str(BASELINE) and manifest['snapshot'] == str(SNAPSHOT),
            'Original manifest uses different frozen roots')
    saved_script = source / 'run_study_attempt1.py'
    require(worker.digest(saved_script) == manifest['script_sha256'], 'Original saved script hash mismatch')
    require(manifest['worker_sha256'] == fresh['worker_sha256'], 'Original helper hash mismatch')
    output = Path(expected['output'])
    require(output.resolve(strict=True) == output, 'Original case output may not redirect via symlinks')
    status = json.loads((source / 'status.json').read_text())['cases'][case['id']]
    old = json.loads((output / 'worker_info.json').read_text())
    require(status['state'] == old['state'] == 'completed' and status['exit_code'] == 0,
            'Original case did not complete successfully')
    require(status['output'] == old['output'] == str(output) and old['id'] == case['id'],
            'Original output identity mismatch')
    require(old['script_sha256'] == declared['script_sha256'] == manifest['script_sha256'],
            'Original worker script hash mismatch')
    for key in PROVENANCE_KEYS:
        require(old[key] == declared[key] == fresh[key], f'Reused frozen provenance differs: {key}')
    for key, value in expected.items():
        require(old[key] == value, f'Reused worker config differs: {key}')
    arguments = dict(json.loads(json.dumps(fresh['arguments'])), run_dir=str(output))
    require(old['arguments'] == declared['arguments'] == arguments, 'Reused run arguments differ')
    config = json.loads((output / 'config.json').read_text())
    solver = config['solver']
    require(solver['iteration_sets'] == expected['solver_schedule']
            and solver['smoothing_lon_lat_depth'] == case['smoothing'], 'Reused solver schedule/smoothing differs')
    require(solver['binary_sha256'] == fresh['solver_sha256']
            and solver['binary'] == arguments['binary'] and solver['timeout_s'] == SOLVER_TIMEOUT_S
            and solver['ray_derivative_step_km'] == arguments['ray_step_km'], 'Reused binary/ray/timeout differs')
    require(config['initial_hypocenters']['sigma_km'] == arguments['initial_location_sigma_km']
            and config['initial_hypocenters']['seed'] == arguments['initial_location_seed'], 'Reused sigma/seed differs')
    require(config['source_experiment'] == arguments['experiment_id']
            and config['source_root'] == arguments['forward_root']
            and config['tomodd_grid']['main_node_spacing_km'] == arguments['node_spacing_km']
            and config['tomodd_grid']['free_buffer_layers'] == arguments['free_buffer_layers']
            and config['observations']['max_pair_degree'] == arguments['max_pair_degree']
            and config['observations']['max_pair_distance_km'] == arguments['max_pair_distance_km'],
            'Reused geometry/picks/pair settings differ')
    generated = check_generated(output / 'work/inversion')
    require(generated == old['generated_input_sha256'] == declared['generated_input_sha256']
            == fresh['generated_input_sha256'], 'Reused generated-input manifest hashes differ')
    control_path = output / 'work/inversion/tomoDD-SE.inp'
    control = control_path.read_text()
    check_control(control, case)
    control_hash = worker.digest(control_path)
    require(control_hash == old['generated_control_sha256'] == declared['generated_control_sha256'],
            'Reused control manifest hash mismatch')
    results = json.loads((output / 'results.json').read_text())
    require(results['event_count'] == fresh['events'], 'Reused event count differs')
    artifacts = worker.check_completed(output, adapter, results)
    retained = ('config.json', 'worker_info.json', 'results.json', 'state.npz', 'truth.npz',
                'work/inversion/tomoDD-SE.inp', *artifacts)
    hashes = {}
    for name in retained:
        path = output / name
        require(path.is_file() and path.stat().st_size > 0, f'Missing reused artifact: {name}')
        hashes[name] = worker.digest(path)
    return {'output': str(output), 'arguments': arguments, 'reused_from': str(output),
            'reuse_validated': True, 'result_script_sha256': old['script_sha256'],
            'preflight_control_sha256': fresh['generated_control_sha256'],
            'generated_control_sha256': control_hash,
            'max_control_data_line_length': check_data_line_lengths(control),
            'reuse_provenance': {'study_root': str(source), 'manifest': str(manifest_path),
                                'manifest_sha256': worker.digest(manifest_path),
                                'script': str(saved_script), 'script_sha256': manifest['script_sha256'],
                                'artifact_sha256': hashes}}


def write_json(path, value, exclusive=False):
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    else:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)


def run_case(case, root, validate_only, reuse_completed_from=None):
    output = root / 'outputs' / case['id']
    started = time.monotonic()
    info = {'id': case['id'], 'output': str(output), 'state': 'failed',
            'started_utc': worker.now(), 'worker_pid': os.getpid()}
    if not validate_only:
        worker.new_output(output)
        output.mkdir(exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='tomodd-sensitivity-') as temporary:
            os.environ['MPLCONFIGDIR'] = str(Path(temporary) / 'mpl')
            os.environ['MPLBACKEND'] = 'Agg'
            os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'y'
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ[name] = '1'
            generated = Path(temporary) / 'generated' if validate_only else output
            adapter, arguments, prepared = prepare_case(case, root, generated)
            info.update(prepared)
            configure_adapter(adapter, case, generated, info, validate_only)
            if validate_only:
                try:
                    adapter.run(**arguments)
                except InputsValidated:
                    pass
                else:
                    raise RuntimeError('Validation hook was not reached')
                if reuse_completed_from is not None and case['id'] == REUSABLE_CASE:
                    info.update(validate_reuse(case, reuse_completed_from, adapter, info))
                info.update(state='validated', solver_launched=False)
            else:
                # The adapter requires the exclusively claimed directory to stay empty.
                info['state'] = 'running'
                metrics = adapter.run(**arguments)
                require('generated_input_sha256' in info, 'Generated-input guard was not reached')
                info['iteration_artifacts'] = worker.check_completed(output, adapter, metrics)
                for name in ('config.json', 'results.json', 'state.npz', 'truth.npz'):
                    require((output / name).is_file(), f'Missing retained artifact: {name}')
                info.update(state='completed')
    except BaseException as error:
        info.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        info.update(elapsed_s=time.monotonic() - started, finished_utc=worker.now())
        if not validate_only:
            write_json(output / 'worker_info.json', info)
    print(json.dumps(info), flush=True)
    return 0


def check_root(root):
    require(not root.is_symlink(), f'Study root may not be a symlink: {root}')
    root = root.resolve()
    for protected in (OLD.resolve(), HERE.resolve()):
        require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                f'Study root overlaps protected tree: {protected}')
    require(not root.exists() or (root.is_dir() and not any(root.iterdir())),
            f'Refusing existing/nonempty study results: {root}')
    return root


def stop_child(process):
    # The solver inherits the worker's new session/process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_children(root, jobs, validate_only, log_dir, status=None, reuse_completed_from=None):
    """Finite foreground queue; skip only the explicitly verified completed dd1."""
    results = {}
    if not validate_only and status is not None:
        for case in CASES:
            record = status['cases'][case['id']]
            if 'reused_from' in record:
                require(case['id'] == REUSABLE_CASE and record['state'] == 'completed'
                        and record.get('reuse_validated') is True, 'Unvalidated case reuse in queue')
                results[case['id']] = dict(record)
    pending = [case for case in CASES if case['id'] not in results]
    active = {}
    limit = VALIDATION_TIMEOUT_S if validate_only else WORKER_TIMEOUT_S
    deadline = time.monotonic() + math.ceil(len(CASES) / jobs) * (limit + 15) + 30

    def publish():
        if status is not None:
            write_json(root / 'status.json', status)

    try:
        while pending or active:
            require(time.monotonic() < deadline, 'Overall child queue deadline exceeded')
            while pending and len(active) < jobs:
                case = pending.pop(0)
                path = log_dir / f'{case["id"]}.log'
                command = [sys.executable, '-B', str(Path(__file__).resolve()),
                           '--study-root', str(root), '--case', case['id']]
                if validate_only:
                    command.append('--validate-only')
                    if reuse_completed_from is not None:
                        command.extend(['--reuse-completed-from', str(reuse_completed_from)])
                with path.open('x') as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True,
                                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                active[case['id']] = (process, time.monotonic(), path)
                if status is not None:
                    status['cases'][case['id']].update(state='running', pid=process.pid,
                                                       started_utc=worker.now())
                    publish()
            for ident, (process, started, path) in list(active.items()):
                timed_out = time.monotonic() - started > limit
                if timed_out:
                    stop_child(process)
                code = process.poll()
                if code is None:
                    continue
                # Reap explicitly, even when poll has already obtained the exit code.
                process.wait(timeout=10)
                record = {'state': 'failed', 'exit_code': code, 'elapsed_s': time.monotonic() - started,
                          'finished_utc': worker.now(), 'timed_out': timed_out}
                try:
                    require(code == 0 and not timed_out, f'Worker failed; inspect {path}')
                    info = (json.loads(path.read_text()) if validate_only else
                            json.loads((root / 'outputs' / ident / 'worker_info.json').read_text()))
                    require(info['state'] == ('validated' if validate_only else 'completed'),
                            f'Worker did not pass: {ident}')
                    record.update(state=info['state'], info=info)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    record['error'] = str(error)
                if record['state'] == 'failed' and not timed_out:
                    # A crashed/killed worker may have left its solver alive.
                    stop_child(process)
                results[ident] = record
                del active[ident]
                if status is not None:
                    status['cases'][ident].update({k: v for k, v in record.items() if k != 'info'})
                    publish()
            if active:
                time.sleep(.1)
    finally:
        for ident, (process, started, path) in active.items():
            stop_child(process)
            if status is not None:
                status['cases'][ident].update(state='failed', error='Parent interrupted; child group killed',
                                               exit_code=process.returncode, finished_utc=worker.now())
        if status is not None:
            for case in pending:
                status['cases'][case['id']].update(state='cancelled')
            publish()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--jobs', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--reuse-completed-from', type=Path,
                        help='Read-only attempt-one study; reuse only verified completed dd1-d40-s5')
    parser.add_argument('--case', choices=[c['id'] for c in CASES], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case:
        root = args.study_root.resolve()
        case = next(c for c in CASES if c['id'] == args.case)
        if not args.validate_only:
            manifest = json.loads((root / 'manifest.json').read_text())
            require(manifest['script_sha256'] == worker.digest(Path(__file__)), 'Study script changed')
            declared = next(c for c in manifest['cases'] if c['id'] == args.case)
            require('reused_from' not in declared, 'Refusing to rerun a reused completed case')
            require(all(declared[k] == json.loads(json.dumps(v))
                        for k, v in case_config(case, root).items()), 'Case differs from manifest')
        return run_case(case, root, args.validate_only, args.reuse_completed_from)
    root = check_root(args.study_root)
    reuse_source = args.reuse_completed_from
    if reuse_source is not None:
        reuse_source = reuse_source.resolve(strict=True)
        require(not root.is_relative_to(reuse_source) and not reuse_source.is_relative_to(root),
                'Retry root must not overlap original study')
    # Validation exercises adapter.run's actual writers, then stops before the solver.
    with tempfile.TemporaryDirectory(prefix='tomodd-preflight-') as temporary:
        validated = run_children(root, args.jobs, True, Path(temporary),
                                 reuse_completed_from=reuse_source)
        failures = {k: v for k, v in validated.items() if v['state'] != 'validated'}
        if failures:
            for ident in failures:
                print(f'{ident}: validation failed\n{(Path(temporary) / (ident + ".log")).read_text()}',
                      file=sys.stderr)
            return 1
    cases = [validated[c['id']]['info'] for c in CASES]
    reuse_metadata = {}
    if reuse_source is not None:
        imported = next(c for c in cases if c['id'] == REUSABLE_CASE)
        require(imported.get('reuse_validated') is True, 'Completed dd1 reuse was not validated')
        imported['state'] = 'completed'
        reuse_metadata = {'reuse_completed_from': str(reuse_source),
                         'original_manifest_sha256': imported['reuse_provenance']['manifest_sha256'],
                         'original_script_sha256': imported['result_script_sha256']}
    if args.validate_only:
        print(json.dumps({'state': 'validated', 'solver_launched': False, 'baseline': str(BASELINE),
                          'script_version': SCRIPT_VERSION, 'script_sha256': worker.digest(Path(__file__)),
                          **reuse_metadata, 'cases': cases}, indent=2))
        return 0
    root = check_root(args.study_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'created_utc': worker.now(), 'baseline': str(BASELINE), 'snapshot': str(SNAPSHOT),
                'script_version': SCRIPT_VERSION, 'script_sha256': worker.digest(Path(__file__)),
                'worker_sha256': worker.digest(WORKER_PATH), **reuse_metadata,
                'baseline_sha256': {name: worker.digest(BASELINE / name)
                                    for name in ('config.json', 'worker_info.json', 'results.json')},
                'cases': cases,
                'jobs': args.jobs, 'solver_timeout_s': SOLVER_TIMEOUT_S,
                'worker_timeout_s': WORKER_TIMEOUT_S, 'comparison_depth_km': 1.5}
    write_json(root / 'manifest.json', manifest, exclusive=True)
    status = {'state': 'running', 'started_utc': worker.now(), 'parent_pid': os.getpid(),
              'cases': {c['id']: {'state': 'pending', 'output': c['output']} for c in cases}}
    for case in cases:
        if case.get('reuse_validated'):
            status['cases'][case['id']].update(state='completed', reused_from=case['reused_from'],
                                               reuse_validated=True, exit_code=0)
    write_json(root / 'status.json', status, exclusive=True)
    try:
        (root / 'outputs').mkdir()
        (root / 'logs').mkdir()
        results = run_children(root, args.jobs, False, root / 'logs', status)
        passed = len(results) == len(CASES) and all(r['state'] == 'completed' for r in results.values())
        status['state'] = 'completed' if passed else 'failed'
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['finished_utc'] = worker.now()
        write_json(root / 'status.json', status)
    print(json.dumps({'state': status['state'], 'root': str(root)}))
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
