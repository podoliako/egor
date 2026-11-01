"""Reproducible horizontal-guard-only production build and two bounded inversions.

Run --build-only, tests, --validate-only, then --run. Every phase is exclusive;
failed history is retained and existing outputs are never reused/overwritten.
All writes, including caches and temporary files, stay in this study/run scope.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
for thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_variable] = '1'
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT / 'runs/tomodd_guard_fix_20261027'
SNAPSHOT = PROJECT / 'runs/matched_geometry_20261021/snapshot'
FROZEN = SNAPSHOT / 'solver'
BUILD = ROOT / 'build'
PYTHON = Path('/mnt/disk01/egor/venv/bin/python3')
SOLVER_TIMEOUT = 1200
WORKER_TIMEOUT = 1500
VALIDATION_TIMEOUT = 240
GEOMETRIES = ('synthetic', 'geysers')
REFERENCES = {
    'synthetic': PROJECT / 'runs/tomodd_sensitivity_20261026/outputs/dd1-d40-s5',
    'geysers': PROJECT / 'runs/tomodd_dd1_geometry_20261026/outputs/geysers-dd1-d40-s5',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load('guard_fix_shared', HERE.parent / 'tomodd_sensitivity_20261026/run_study.py')
guard = load('guard_fix_source_contract', HERE.parent / 'tomodd_homogeneous_20261026/check_regional_guard.py')
worker = shared.worker
require = worker.require
digest = worker.digest
CASE = next(c for c in shared.CASES if c['id'] == 'dd1-d40-s5')


def read(path):
    return json.loads(path.read_text())


def save(path, value, exclusive=False):
    shared.write_json(path, value, exclusive=exclusive)


def verify_hashes(hashes):
    for name, expected in hashes.items():
        require(digest(Path(name)) == expected, f'Protected/provenance file changed: {name}')


def study_hashes():
    return {str(p): digest(p) for p in sorted(HERE.iterdir()) if p.is_file()}


def status(value):
    save(ROOT / 'status.json', value)
    with (ROOT / 'status_history.jsonl').open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def frozen_contract():
    worker.verify_build(FROZEN)
    manifest = read(FROZEN / 'manifest.json')
    for name, checksum in manifest['artifact_sha256'].items():
        require(digest(FROZEN / name) == checksum, f'Frozen artifact mismatch: {name}')
    compiler = manifest['recipe']['tools']['fc']
    require(digest(Path(compiler['path'])) == compiler['sha256'], 'Frozen compiler changed')
    require(digest(FROZEN / 'RaySPDR2.f') == manifest['recipe']['patched_ray_sha256'],
            'Frozen source changed')
    return manifest


def patch_proof(original, patched):
    proof = guard.validate_patch(original, patched)
    expected = ''.join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), patched.decode().splitlines(keepends=True),
        fromfile='a/RaySPDR2.f', tofile='b/RaySPDR2.f', n=0))
    require(expected == (HERE / 'horizontal_guard.patch').read_text(), 'Tracked patch differs')
    return proof


def link_plan(manifest):
    """Ask the frozen Makefile for its link recipe without rebuilding/writing it."""
    objects = [name for name in manifest['artifact_sha256']
               if name.endswith('.o') and name != 'forward_probe.o']
    recipe = manifest['recipe']
    command = [recipe['tools']['make']['path'], '--no-print-directory', '--dry-run',
               '--always-make', *[f'--old-file={name}' for name in objects], 'tomoDD-SE',
               f'FC={recipe["tools"]["fc"]["path"]}',
               'LDFLAGS=' + ' '.join(recipe['ldflags']), 'LIBS=']
    result = subprocess.run(command, cwd=FROZEN, capture_output=True, text=True,
                            check=True, timeout=30)
    lines = result.stdout.strip().splitlines()
    require(len(lines) == 1, f'Make must emit exactly one link command: {result.stdout}')
    tokens = shlex.split(lines[0])
    ordered = [token for token in tokens if token.endswith('.o')]
    require(set(ordered) == set(objects) and len(ordered) == len(objects), 'Link object set differs')
    require(tokens == [recipe['tools']['fc']['path'], *recipe['ldflags'], *ordered,
                       '-o', 'tomoDD-SE'], 'Unexpected Makefile link recipe')
    return command, result.stdout, ordered


def protection_ledger():
    trees = [FROZEN, SNAPSHOT / 'benchmark', SNAPSHOT / 'support', *REFERENCES.values(),
             PROJECT / 'runs/tomodd_homogeneous_20261026/regional_guard_control']
    trees += [PROJECT / f'runs/matched_geometry_20261021/outputs/{g}-tomodd-sigma1'
              for g in GEOMETRIES]
    files = {p for tree in trees for p in tree.rglob('*') if p.is_file()}
    files.update((Path(shared.__file__), Path(worker.__file__), Path(guard.__file__)))
    for geometry in GEOMETRIES:
        old = read(REFERENCES[geometry] / 'worker_info.json')
        files.update(Path(p) for p in old['input_sha256'])
    return {str(p): digest(p) for p in sorted(files)}


def build():
    require(ROOT.is_dir() and not any(ROOT.iterdir()), 'Build requires an empty new run root')
    started = time.monotonic()
    status({'state': 'building', 'started_utc': worker.now()})
    try:
        frozen = frozen_contract()
        ledger = protection_ledger()
        save(ROOT / 'protected_sha256.json', ledger, exclusive=True)
        BUILD.mkdir(exist_ok=False)
        original = (FROZEN / 'RaySPDR2.f').read_bytes()
        patched = guard.patch_source(original)
        proof = patch_proof(original, patched)
        (BUILD / 'RaySPDR2.f').write_bytes(patched)
        (BUILD / 'horizontal_guard.patch').write_bytes((HERE / 'horizontal_guard.patch').read_bytes())
        make_command, make_stdout, objects = link_plan(frozen)
        (BUILD / 'make_dry_run.log').write_text(make_stdout)
        recipe = frozen['recipe']
        fc = recipe['tools']['fc']['path']
        flags = recipe['fflags'].copy()
        require(flags[-1] == '-I./include', 'Unexpected frozen include flag')
        flags[-1] = f'-I{FROZEN / "include"}'
        compile_command = [fc, *flags, *recipe['ray_type_flags'], '-c',
                           str(BUILD / 'RaySPDR2.f'), '-o', str(BUILD / 'RaySPDR2.o')]

        def relink(corrected):
            paths = [(BUILD if corrected and name == 'RaySPDR2.o' else FROZEN) / name
                     for name in objects]
            return [fc, *recipe['ldflags'], *map(str, paths), '-o',
                    str(BUILD / ('tomoDD-SE' if corrected else 'original-tomoDD-SE'))]

        commands = [compile_command, relink(False), relink(True)]
        save(BUILD / 'commands.json', {'make_dry_run': make_command, 'cwd_make': str(FROZEN),
             'commands': commands, 'cwd_build': str(BUILD), 'timeout_s_each': 120}, exclusive=True)
        timings = []
        for index, command in enumerate(commands, 1):
            tick = time.monotonic()
            with (BUILD / f'command_{index}.log').open('x') as log:
                subprocess.run(command, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=120)
            timings.append(time.monotonic() - tick)
        require(digest(BUILD / 'original-tomoDD-SE') == worker.BINARY_SHA256,
                'Unmodified frozen-object relink did not reproduce original inversion')
        require(digest(BUILD / 'tomoDD-SE') != worker.BINARY_SHA256, 'Corrected binary is unchanged')
        manifest = {
            'status': 'complete', 'production_inversion': True,
            'source_change_proof': proof, 'frozen_recipe_signature': frozen['signature'],
            'original_manifest': str(FROZEN / 'manifest.json'),
            'original_manifest_sha256': digest(FROZEN / 'manifest.json'),
            'original_binary_sha256': worker.BINARY_SHA256,
            'original_relink_byte_identical': True, 'recipe': recipe,
            'compiler': recipe['tools']['fc'], 'makefile_sha256': digest(FROZEN / 'Makefile'),
            'link_object_order': objects,
            'unchanged_link_object_sha256': {name: digest(FROZEN / name) for name in objects
                                              if name != 'RaySPDR2.o'},
            'original_ray_object_sha256': digest(FROZEN / 'RaySPDR2.o'),
            'original_source_sha256': digest(FROZEN / 'RaySPDR2.f'),
            'artifact_sha256': {name: digest(BUILD / name) for name in
                ('RaySPDR2.f', 'horizontal_guard.patch', 'RaySPDR2.o', 'tomoDD-SE',
                 'original-tomoDD-SE', 'commands.json', 'make_dry_run.log')},
            'commands': commands, 'make_dry_run_command': make_command,
            'command_elapsed_s': timings, 'elapsed_s': time.monotonic() - started,
            'study_source_sha256': study_hashes(),
        }
        save(BUILD / 'manifest.json', manifest, exclusive=True)
        verify_hashes(ledger)
        save(ROOT / 'manifest.json', {
            'schema': 1, 'created_utc': worker.now(), 'study': str(HERE), 'root': str(ROOT),
            'build_manifest': str(BUILD / 'manifest.json'),
            'build_manifest_sha256': digest(BUILD / 'manifest.json'),
            'protected_sha256_file': str(ROOT / 'protected_sha256.json'),
            'protected_sha256_file_sha256': digest(ROOT / 'protected_sha256.json'),
            'study_source_sha256': study_hashes(),
            'case': CASE, 'solver_timeout_s': SOLVER_TIMEOUT, 'worker_timeout_s': WORKER_TIMEOUT,
            'snapshot': str(SNAPSHOT), 'references': {g: str(p) for g, p in REFERENCES.items()},
            'supervision': 'two foreground workers, separate process groups; finite deadlines',
        }, exclusive=True)
        status({'state': 'built', 'elapsed_s': time.monotonic() - started})
    except BaseException as error:
        status({'state': 'failed', 'phase': 'build', 'error': f'{type(error).__name__}: {error}',
                'elapsed_s': time.monotonic() - started})
        raise


def verify_corrected():
    frozen = frozen_contract()
    root_manifest = read(ROOT / 'manifest.json')
    require(digest(BUILD / 'manifest.json') == root_manifest['build_manifest_sha256'],
            'Corrected build manifest changed')
    manifest = read(BUILD / 'manifest.json')
    require(manifest['status'] == 'complete' and manifest['production_inversion'] is True,
            'Not a completed production inversion build')
    verify_hashes(root_manifest['study_source_sha256'])
    # A driver-only revision may follow a solver-free validation failure. The
    # original build source must remain available and match its sealed ledger.
    archive = root_manifest.get('build_source_archive', {})
    require(set(archive).issubset(manifest['study_source_sha256']), 'Unknown archived build source')
    for name, checksum in manifest['study_source_sha256'].items():
        path = Path(archive.get(name, name))
        if name in archive:
            require(path.resolve().is_relative_to(ROOT / 'history'), 'Build source archive escapes history')
        require(digest(path) == checksum, f'Original build source changed: {name}')
    require(manifest['original_manifest_sha256'] == digest(FROZEN / 'manifest.json'),
            'Original build provenance changed')
    require(manifest['frozen_recipe_signature'] == frozen['signature'], 'Frozen recipe differs')
    require(manifest['compiler'] == frozen['recipe']['tools']['fc'], 'Compiler provenance differs')
    require(manifest['recipe'] == frozen['recipe'], 'Frozen compilation recipe differs')
    require(manifest['makefile_sha256'] == digest(FROZEN / 'Makefile'), 'Makefile differs')
    patch_proof((FROZEN / 'RaySPDR2.f').read_bytes(), (BUILD / 'RaySPDR2.f').read_bytes())
    for name, checksum in manifest['artifact_sha256'].items():
        require(digest(BUILD / name) == checksum, f'Corrected build artifact changed: {name}')
    for name, checksum in manifest['unchanged_link_object_sha256'].items():
        require(digest(FROZEN / name) == checksum == frozen['artifact_sha256'][name],
                f'Unchanged linked object differs: {name}')
    require(digest(BUILD / 'original-tomoDD-SE') == worker.BINARY_SHA256,
            'Original relink proof differs')
    require(digest(BUILD / 'tomoDD-SE') != worker.BINARY_SHA256, 'Corrected binary equals old binary')
    require(digest(ROOT / 'protected_sha256.json') == root_manifest['protected_sha256_file_sha256'],
            'Protection ledger changed')
    return manifest


def check_reference(info, geometry):
    baseline = read(shared.BASELINE / 'worker_info.json')
    reference = read(REFERENCES[geometry] / 'worker_info.json')
    for label, old in (('baseline', baseline), ('dd1 reference', reference)):
        require(old['state'] == 'completed', f'{label} is incomplete')
        for key in shared.PROVENANCE_KEYS:
            require(info[key] == old[key], f'{label} frozen provenance differs: {key}')
    expected = json.loads(json.dumps(dict(info['arguments'],
        run_dir=str(REFERENCES[geometry]), timeout_s=SOLVER_TIMEOUT)))
    require(expected == reference['arguments'], 'Old dd1 input/run parameters differ')
    config = read(REFERENCES[geometry] / 'config.json')
    require(config['solver']['iteration_sets'] == json.loads(json.dumps(shared.solver_schedule(CASE)))
            and config['solver']['smoothing_lon_lat_depth'] == CASE['smoothing'],
            'Old dd1 schedule/smoothing differs')
    return reference


def reference_control_layout(text, geometry):
    """Restore only reference schedule whitespace, after the shared renderer guards."""
    reference = (REFERENCES[geometry] / 'work/inversion/tomoDD-SE.inp').read_bytes()
    old_line = []
    shared.replace_data_line(reference.decode(), shared.SCHEDULE_MARKER,
                             lambda line: old_line.append(line) or line)

    def same_tokens(line):
        require(line.split() == old_line[0].split(), 'Reference schedule tokens differ')
        return old_line[0]

    result = shared.replace_data_line(text, shared.SCHEDULE_MARKER, same_tokens)
    shared.check_control(result, CASE)
    require(result.encode() == reference, 'Non-schedule control bytes differ from old dd1')
    return result


def check_dd1_inputs(folder, text, geometry, reference):
    hashes = {}
    for name in shared.CONSTANT_FILES:
        path, old = folder / name, REFERENCES[geometry] / 'work/inversion' / name
        require(path.read_bytes() == old.read_bytes(), f'Old dd1 input bytes differ: {name}')
        hashes[name] = digest(path)
    require(hashes == reference['generated_input_sha256'], 'Old dd1 input ledger differs')
    old_control = REFERENCES[geometry] / 'work/inversion/tomoDD-SE.inp'
    require(text.encode() == old_control.read_bytes(), 'Old dd1 control bytes differ')
    require(hashlib.sha256(text.encode()).hexdigest() == reference['generated_control_sha256'],
            'Old dd1 control ledger differs')
    shared.check_control(text, CASE)
    return {'reference': str(REFERENCES[geometry]), 'all_inputs_byte_identical': True,
            'control_byte_identical': True, 'input_sha256': hashes,
            'control_sha256': digest(old_control)}


def run_worker(geometry, validate_only):
    started = time.monotonic()
    phase = 'validation' if validate_only else 'outputs'
    output = ROOT / phase / f'{geometry}-dd1-d40-s5'
    output.mkdir(exist_ok=False)
    info = {'state': 'failed', 'geometry': geometry, 'output': str(output),
            'started_utc': worker.now(), 'worker_pid': os.getpid()}
    try:
        shared.BASELINE = PROJECT / f'runs/matched_geometry_20261021/outputs/{geometry}-tomodd-sigma1'
        args = argparse.Namespace(geometry=geometry, output=output, benchmark_root=SNAPSHOT / 'benchmark',
             forward_root=SNAPSHOT / 'projects/forward_modeling', solver_build=FROZEN,
             support_root=SNAPSHOT / 'support')
        adapter = worker.load_adapter(args.benchmark_root)
        arguments, prepared = worker.preflight(args, adapter)
        info.update(prepared)
        reference = check_reference(info, geometry)
        corrected = verify_corrected()
        # Do not weaken/replace verify_build: original preflight has already succeeded.
        info['original_solver_provenance'] = {key: info[key] for key in
            ('solver_sha256', 'solver_manifest_sha256', 'solver_build_manifest')}
        info['original_solver_provenance']['binary'] = str(arguments['binary'])
        arguments.update(binary=BUILD / 'tomoDD-SE', timeout_s=SOLVER_TIMEOUT)
        info.update(id=f'{geometry}-dd1-d40-s5', parameter_set=CASE['id'], baseline=str(shared.BASELINE),
            solver_sha256=corrected['artifact_sha256']['tomoDD-SE'],
            solver_manifest_sha256=digest(BUILD / 'manifest.json'),
            solver_build_manifest=str(BUILD / 'manifest.json'),
            corrected_solver_provenance={'binary': str(arguments['binary']),
                'binary_sha256': corrected['artifact_sha256']['tomoDD-SE'],
                'build_manifest': str(BUILD / 'manifest.json'),
                'build_manifest_sha256': digest(BUILD / 'manifest.json')},
            solver_schedule=shared.solver_schedule(CASE), smoothing_lon_lat_depth=CASE['smoothing'],
            absolute_dd_weight=[1., 1.], timeout_s=SOLVER_TIMEOUT,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()},
            script_sha256=digest(Path(__file__)), shared_script_sha256=digest(Path(shared.__file__)))
        shared.configure_adapter(adapter, CASE, output, info, False)
        original_input_text = adapter._input_text

        def checked_input_text(cfg, **kwargs):
            text = original_input_text(cfg, **kwargs)
            info['shared_rendered_control_sha256'] = info['generated_control_sha256']
            text = reference_control_layout(text, geometry)
            info['generated_control_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            info['max_control_data_line_length'] = shared.check_data_line_lengths(text)
            info['dd1_input_reuse'] = check_dd1_inputs(output / 'work/inversion', text, geometry, reference)
            if validate_only:
                (output / 'work/inversion/tomoDD-SE.inp').write_text(text)
                raise shared.InputsValidated()
            return text

        adapter._input_text = checked_input_text
        actual_run = subprocess.run

        def timed_solver(command, **kwargs):
            require(command == [str(BUILD / 'tomoDD-SE'), 'tomoDD-SE.inp'], 'Unexpected solver command')
            require(kwargs['timeout'] == SOLVER_TIMEOUT, 'Solver is not bounded')
            require(digest(BUILD / 'tomoDD-SE') == info['solver_sha256'], 'Solver changed before launch')
            work = output / 'work/inversion'
            check_dd1_inputs(work, (work / 'tomoDD-SE.inp').read_text(), geometry, reference)
            info['solver_launched'] = True
            tick = time.monotonic()
            try:
                result = actual_run(command, **kwargs)
                info['solver_exit_code'] = result.returncode
                return result
            finally:
                info['solver_elapsed_s'] = time.monotonic() - tick

        adapter.subprocess = SimpleNamespace(run=timed_solver, STDOUT=subprocess.STDOUT)
        if validate_only:
            try:
                adapter.run(**arguments)
            except shared.InputsValidated:
                pass
            else:
                raise RuntimeError('Validation failed to stop before the solver')
            info.update(state='validated', solver_launched=False)
        else:
            metrics = adapter.run(**arguments)
            require(metrics['event_count'] == info['events'], 'Event count changed')
            info['iteration_artifacts'] = worker.check_completed(output, adapter, metrics)
            for name in ('config.json', 'results.json', 'state.npz', 'truth.npz'):
                require((output / name).is_file(), f'Missing retained artifact: {name}')
            config = read(output / 'config.json')
            require(config['solver']['binary_sha256'] == info['solver_sha256']
                    and config['solver']['build_manifest'] == info['solver_build_manifest'],
                    'Adapter did not record corrected solver provenance')
            check_dd1_inputs(output / 'work/inversion',
                             (output / 'work/inversion/tomoDD-SE.inp').read_text(), geometry, reference)
            info.update(state='completed', results=metrics)
    except BaseException as error:
        info.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        info.update(elapsed_s=time.monotonic() - started, finished_utc=worker.now())
        save(output / 'worker_info.json', info, exclusive=True)
    print(json.dumps({k: info[k] for k in ('geometry', 'state', 'elapsed_s')}), flush=True)


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Kill surviving solver descendants even if the worker has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def supervise(validate_only):
    verify_corrected()
    phase = 'validation' if validate_only else 'outputs'
    if not validate_only:
        validation = read(ROOT / 'validation/summary.json')
        require(validation['state'] == 'validated', 'Both validations must pass before inversion')
        verify_hashes(validation['artifact_sha256'])
        verify_hashes(validation['study_source_sha256'])
    (ROOT / phase).mkdir(exist_ok=False)
    started = time.monotonic()
    children, cases = {}, {}
    status({'state': 'validating' if validate_only else 'running', 'started_utc': worker.now()})
    try:
        for geometry in GEOMETRIES:
            command = [str(PYTHON), '-B', str(Path(__file__)), '--worker', geometry]
            if validate_only:
                command.append('--validate-only')
            log = (ROOT / phase / f'{geometry}.log').open('x')
            try:
                process = subprocess.Popen(command, cwd=PROJECT, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            children[geometry] = (process, time.monotonic())
            cases[geometry] = {'state': 'running', 'pid': process.pid, 'command': command,
                               'output': str(ROOT / phase / f'{geometry}-dd1-d40-s5')}
        bound = VALIDATION_TIMEOUT if validate_only else WORKER_TIMEOUT
        while any(c['state'] == 'running' for c in cases.values()):
            require(time.monotonic() - started < bound + 20, 'Finite supervisor deadline exceeded')
            for geometry, (process, tick) in children.items():
                case = cases[geometry]
                if case['state'] != 'running':
                    continue
                if process.poll() is None and time.monotonic() - tick > bound:
                    stop_group(process)
                    case['error'] = f'Worker exceeded {bound}s deadline'
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    stop_group(process)
                case.update(exit_code=code, supervisor_elapsed_s=time.monotonic() - tick)
                path = Path(case['output']) / 'worker_info.json'
                info = read(path) if path.is_file() else {}
                expected = 'validated' if validate_only else 'completed'
                case['state'] = expected if code == 0 and info.get('state') == expected else 'failed'
                if info.get('error'):
                    case['error'] = info['error']
                case['worker_info'] = info
                status({'state': 'validating' if validate_only else 'running', 'cases': cases,
                        'elapsed_s': time.monotonic() - started})
            time.sleep(.2)
        expected = 'validated' if validate_only else 'completed'
        require(all(c['state'] == expected for c in cases.values()), 'One or more workers failed')
        verify_corrected()
        verify_hashes(read(ROOT / 'protected_sha256.json'))
        artifacts = {}
        for case in cases.values():
            output = Path(case['output'])
            # Retain an integrity ledger for every generated artifact, including iteration snapshots.
            artifacts.update({str(p): digest(p) for p in output.rglob('*') if p.is_file()})
        summary = {'state': expected, 'cases': cases, 'elapsed_s': time.monotonic() - started,
                   'finished_utc': worker.now(), 'protected_files_unchanged': True,
                   'artifact_sha256': artifacts, 'study_source_sha256': study_hashes()}
        save(ROOT / phase / 'summary.json', summary, exclusive=True)
        manifest = read(ROOT / 'manifest.json')
        manifest[phase] = {'summary': str(ROOT / phase / 'summary.json'),
                           'summary_sha256': digest(ROOT / phase / 'summary.json'),
                           'elapsed_s': summary['elapsed_s'], 'state': expected}
        save(ROOT / 'manifest.json', manifest)
        status({'state': expected, 'cases': cases, 'elapsed_s': summary['elapsed_s'],
                'finished_utc': worker.now(), 'protected_files_unchanged': True})
        print(json.dumps({'state': expected, 'elapsed_s': summary['elapsed_s'],
                          'cases': {g: c['state'] for g, c in cases.items()}}), flush=True)
    except BaseException as error:
        for process, _ in children.values():
            stop_group(process)
        status({'state': 'failed', 'phase': phase, 'cases': cases,
                'error': f'{type(error).__name__}: {error}', 'elapsed_s': time.monotonic() - started})
        raise


def interrupted(signum, frame):
    raise TimeoutError(f'Received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', choices=GEOMETRIES)
    args = parser.parse_args()
    require(sum((args.build_only, args.run, bool(args.worker), args.validate_only and not args.worker)) == 1,
            'Select exactly one phase')
    require(not ROOT.is_symlink() and not HERE.is_symlink(), 'Scope must not redirect through symlinks')
    cache = ROOT / 'cache'
    if not args.build_only:
        cache.mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
        MPLCONFIGDIR=str(cache / (args.worker or 'supervisor')), TMPDIR=str(cache),
        GFORTRAN_UNBUFFERED_ALL='y', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.build_only:
        # GCC needs temporary storage, confined to the build directory created by build().
        os.environ['TMPDIR'] = str(BUILD)
        build()
    elif args.worker:
        run_worker(args.worker, args.validate_only)
    else:
        supervise(args.validate_only)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
