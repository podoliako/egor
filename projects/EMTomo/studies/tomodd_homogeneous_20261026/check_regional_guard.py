"""Private four-offset horizontal-guard causal control; full original-input pairs.

Only regional_guard_control/ is writable. No production binary, inversion,
frozen object/source, previous sealed output, or article is modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'runs/tomodd_homogeneous_20261026'
GLOBAL = OLD / 'global_homogeneous'
OUTPUT = OLD / 'regional_guard_control'
FROZEN = PROJECT / 'runs/matched_geometry_20261021/snapshot/solver'
PRIVATE = OUTPUT / 'build'
CASES = {'artificial': (1920, 52, 600), 'geysers': (3776, 26, 1200)}
LABELS = ('geysers_outlier', 'geysers_maxpositive', 'artificial_maxpositive')
CRITERIA = {'ray_only': {'rms_ms': 1., 'max_abs_ms': 5.},
            'total': {'rms_ms': 5., 'max_abs_ms': 10.}}
OLD_GUARD = (b'      if(lo.GT.xn(1)+0.1.AND.lo.LT.xn(nx)-0.1.AND.la.GT.yn(1)+0.1\n'
             b'     &   .AND.la.LT.yn(ny)-0.1)then')
NEW_GUARD = (b'      if(lo.GT.xn(1).AND.lo.LT.xn(nx).AND.la.GT.yn(1)\n'
             b'     &   .AND.la.LT.yn(ny))then')
LIMITATION = ('This isolates the horizontal exclusion guard in the fixed-TRUE-source regional-homogeneous '
              'forward control with original AK135 support. It does not establish inversion degradation '
              'or justify a production fix. No production-time/Jacobian integration investigation is repeated.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(expected):
    for name, checksum in expected.items():
        require(digest(name) == checksum, f'Input/sealed file changed: {name}')


def patch_source(original):
    require(original.count(OLD_GUARD) == 1, 'Expected unique original four-offset guard')
    return original.replace(OLD_GUARD, NEW_GUARD, 1)


def validate_patch(original, patched):
    require(patched == patch_source(original), 'Private source differs beyond the four guard offsets')
    require(patched.replace(NEW_GUARD, OLD_GUARD, 1) == original, 'Patch does not restore original bytes')
    before_lines, after_lines = original.splitlines(keepends=True), patched.splitlines(keepends=True)
    require(len(before_lines) == len(after_lines), 'Source line count changed')
    changed = [i + 1 for i, (a, b) in enumerate(zip(before_lines, after_lines)) if a != b]
    require(changed == [840, 841], f'Unexpected changed lines: {changed}')
    require(OLD_GUARD.count(b'+0.1') + OLD_GUARD.count(b'-0.1') == 4, 'Not exactly four offsets')
    require(b'      if(dp.GE.zn(nz)-1)then' in patched, 'Bottom guard changed')
    return {'changed_lines_1based': changed, 'removed_horizontal_offsets': 4,
            'entire_remaining_source_byte_identical': True, 'bottom_guard_unchanged': True}


def self_tests(original):
    patched = patch_source(original)
    details = validate_patch(original, patched)
    for bad in (patched + b'\n', patched.replace(b'if(dp.GE.zn(nz)-1)', b'if(dp.GE.zn(nz)-2)', 1),
                patched.replace(b'      nloop  = 200', b'      nloop  = 201', 1)):
        require(bad != patched, 'Source guard test fixture did not mutate source')
        try:
            validate_patch(original, bad)
        except ValueError:
            continue
        raise AssertionError('Source-only change guard accepted unrelated mutation')
    return {'status': 'passed', 'checks': ['exact_four_offset_patch', 'byte_exact_reverse_patch',
            'only_lines_840_841_changed', 'bottom_guard_preserved', 'reject_added_newline',
            'reject_bottom_change', 'reject_other_numerical_change'], **details}


def input_hashes():
    expected = json.loads((GLOBAL / 'input_sha256.json').read_text())
    for name, checksum in json.loads((GLOBAL / 'output_sha256.json').read_text()).items():
        expected[str(GLOBAL / name)] = checksum
    expected[str(GLOBAL / 'output_sha256.json')] = digest(GLOBAL / 'output_sha256.json')
    expected[str(Path(__file__).resolve())] = digest(__file__)
    build = json.loads((FROZEN / 'manifest.json').read_text())
    compiler = build['recipe']['tools']['fc']
    expected[compiler['path']] = compiler['sha256']
    for name in ('forward_probe', 'forward_probe.o', 'RaySPDR2.o', 'transform_r_gfortran.o'):
        require(expected[str(FROZEN / name)] == build['artifact_sha256'][name], 'Frozen object ledger mismatch')
    require(expected[str(FROZEN / 'RaySPDR2.f')] == build['recipe']['patched_ray_sha256'],
            'Frozen source provenance differs')
    verify(expected)
    return expected, build


def compile_private(build):
    PRIVATE.mkdir(exist_ok=False)
    source = PRIVATE / 'RaySPDR2.f'
    shutil.copyfile(FROZEN / 'RaySPDR2.f', source)
    original = source.read_bytes()
    source.write_bytes(patch_source(original))
    proof = validate_patch(original, source.read_bytes())
    difference = ''.join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                         source.read_text().splitlines(keepends=True),
                         fromfile=str(FROZEN / 'RaySPDR2.f'), tofile=str(source)))
    (PRIVATE / 'horizontal_guard_only.diff').write_text(difference)
    compiler = build['recipe']['tools']['fc']['path']
    flags = list(build['recipe']['fflags'])
    require(flags[-1] == '-I./include', 'Unexpected frozen include option')
    flags[-1] = f'-I{FROZEN / "include"}'
    ray_flags = build['recipe']['ray_type_flags']
    require(ray_flags == ['-cpp', '-Dmins=ray_min_segment_km'], 'Missing frozen REAL*8 mins flags')
    obj = PRIVATE / 'RaySPDR2.o'
    binary = PRIVATE / 'forward_probe'
    commands = [[compiler, *flags, *ray_flags, '-c', str(source), '-o', str(obj)],
                [compiler, '-O3', '-m64', '-mcmodel=medium', str(FROZEN / 'forward_probe.o'),
                 str(obj), str(FROZEN / 'transform_r_gfortran.o'), '-lm', '-o', str(binary)]]
    save(PRIVATE / 'commands.json', {'commands': commands, 'timeout_s_each': 120}, exclusive=True)
    started = time.monotonic()
    for i, command in enumerate(commands):
        with (PRIVATE / f'command_{i + 1}.log').open('x') as log:
            subprocess.run(command, cwd=PRIVATE, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    manifest = {'status': 'complete', 'diagnostic_only_not_production': True, 'commands': commands,
                'compile_link_elapsed_s': time.monotonic() - started,
                'source_change_proof': proof, 'source_sha256': digest(source),
                'original_source_sha256': digest(FROZEN / 'RaySPDR2.f'),
                'diff_sha256': digest(PRIVATE / 'horizontal_guard_only.diff'),
                'private_ray_object_sha256': digest(obj), 'private_probe_sha256': digest(binary),
                'frozen_linked_objects_sha256': {name: digest(FROZEN / name)
                                                for name in ('forward_probe.o', 'transform_r_gfortran.o')},
                'frozen_recipe_signature': build['signature']}
    save(PRIVATE / 'manifest.json', manifest, exclusive=True)
    return manifest


def load_runtime():
    path = HERE / 'run_check.py'
    spec = importlib.util.spec_from_file_location('regional_guard_original_check', path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    require(check.CRITERIA == CRITERIA, 'Predeclared criteria differ')
    check.source_contract()
    _, _, solver = check.load_runtime()
    return check, solver


def copy_inputs(folder, geometry):
    folder.mkdir(exist_ok=False)
    for name in ('MOD', 'ak135.15.SKS', 'layer-16.dat'):
        shutil.copyfile(OLD / geometry / name, folder / name)
        require(digest(folder / name) == digest(OLD / geometry / name), 'Original input copy mismatch')
    return {name: digest(folder / name) for name in ('MOD', 'ak135.15.SKS', 'layer-16.dat')}


def run_probe(folder, geometry, sources, receivers, solver, timeout):
    inputs = copy_inputs(folder, geometry)
    started = time.monotonic()
    times, nseg = solver.forward_times(PRIVATE, folder, sources, receivers, timeout=timeout)
    wall = time.monotonic() - started
    child, = folder.glob('small_domain_solver_probe_*')
    manifest = json.loads((child / 'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['binary_sha256'] == digest(PRIVATE / 'forward_probe'),
            'Private probe provenance mismatch')
    require(manifest['input_sha256'] == inputs, 'Probe received changed input files')
    return times, nseg, wall, child, inputs


def target_checks(check, solver):
    (OUTPUT / 'target_checks').mkdir(exist_ok=False)
    targets = json.loads((OLD / 'pair_reproduction/targets.json').read_text())
    controls = json.loads((OLD / 'path_inspection/results.json').read_text())['global_p5_controls']
    results = {}
    for label in LABELS:
        target = targets[label]
        times, nseg, wall, _, _ = run_probe(OUTPUT / 'target_checks' / label, target['geometry'],
                                          [target['source_lat_lon_depth_km']],
                                          [target['receiver_lat_lon_depth_km']], solver, 120)
        measured, segments = float(times[0, 0]), int(nseg[0, 0])
        errors = {'ray_only': np.array([[measured - target['t_chord_s']]]),
                  'total': np.array([[measured - target['t_cart_s']]])}
        results[label] = {'event_id': target['event_id'], 'station_id': target['station_id'],
                          'source_xyz_km': target['source_xyz_km'], 'receiver_xyz_km': target['receiver_xyz_km'],
                          'private_time_s': measured, 'nseg': segments,
                          'ray_error_ms': (measured - target['t_chord_s']) * 1000,
                          'stock_ray_error_ms': target['original_ray_error_ms'],
                          'delta_vs_stock_ms': (measured - target['original_t_probe_s']) * 1000,
                          'delta_vs_global_p5_ms': (measured - controls[label]['time_s']) * 1000,
                          'criteria_pass': {key: check.passes(check.metrics(value), key) for key, value in errors.items()},
                          'probe_wall_s': wall}
        print(f'target {label}: ray error {results[label]["ray_error_ms"]:+.9f}ms nseg={segments}', flush=True)
    save(OUTPUT / 'target_checks/results.json', results, exclusive=True)
    return results


def comparison(times, nseg, baseline, check):
    delta = times - baseline['t_probe_s']
    return {'time_changed_pairs_exact': int(np.count_nonzero(delta)),
            'time_changed_over_1ms_pairs': int(np.count_nonzero(np.abs(delta) > .001)),
            'nseg_changed_pairs': int(np.count_nonzero(nseg != baseline['nseg'])),
            'delta_private_minus_baseline': check.metrics(delta)}


def histogram(values):
    return {str(int(k)): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def full_geometry(name, check, solver):
    ne, ns, timeout = CASES[name]
    with np.load(OLD / name / 'arrays.npz', allow_pickle=False) as archive:
        stock = {key: archive[key] for key in archive.files}
    with np.load(GLOBAL / name / 'arrays.npz', allow_pickle=False) as archive:
        global5 = {key: archive[key] for key in archive.files}
    for key in ('sources_lat_lon_depth_km', 'receivers_lat_lon_depth_km', 't_cart_s', 't_chord_s'):
        np.testing.assert_array_equal(stock[key], global5[key])
    sources, receivers = stock['sources_lat_lon_depth_km'], stock['receivers_lat_lon_depth_km']
    require(sources.shape == (ne, 3) and receivers.shape == (ns, 3), 'Incorrect original geometry counts')
    print(f'{name}: private guard full {ne * ns} pairs; original support; timeout {timeout}s', flush=True)
    times, nseg, wall, child, inputs = run_probe(OUTPUT / name, name, sources, receivers, solver, timeout)
    require(times.shape == nseg.shape == (ne, ns) and times.size == ne * ns, 'Incomplete full-pair output')
    original_child, = (OLD / name).glob('small_domain_solver_probe_*')
    require((child / 'coordinates.txt').read_bytes() == (original_child / 'coordinates.txt').read_bytes(),
            'Serialized full-pair request differs from stock')
    ray, embedding, total, residual = check.decomposition(times, stock['t_chord_s'], stock['t_cart_s'])
    metrics = {key: check.metrics(value) for key, value in
               (('ray_only', ray), ('embedding_serialization', embedding), ('total', total))}
    results = {'geometry': name, 'events': ne, 'stations': ns, 'complete_pairs': ne * ns,
               'request_byte_identical_to_stock': True, 'input_sha256_identical_to_stock': inputs,
               'metrics': metrics, 'criteria_pass': {key: check.passes(metrics[key], key) for key in CRITERIA},
               'nseg_histogram': histogram(nseg), 'stock_nseg_histogram': histogram(stock['nseg']),
               'global_p5_nseg_histogram': histogram(global5['nseg']),
               'vs_stock': comparison(times, nseg, stock, check),
               'vs_global_p5': comparison(times, nseg, global5, check),
               'decomposition_exact': residual == 0., 'decomposition_max_abs_residual_s': residual,
               'probe_wall_s': wall}
    keep = ('event_ids', 'station_ids', 'true_event_xyz_km', 'station_xyz_km',
            'sources_lat_lon_depth_km', 'receivers_lat_lon_depth_km', 't_cart_s', 't_chord_s')
    np.savez_compressed(OUTPUT / name / 'arrays.npz', **{key: stock[key] for key in keep},
                        t_probe_s=times, nseg=nseg, ray_error_s=ray, embedding_serialization_error_s=embedding,
                        total_error_s=total, stock_t_probe_s=stock['t_probe_s'], stock_nseg=stock['nseg'],
                        global_p5_t_probe_s=global5['t_probe_s'], global_p5_nseg=global5['nseg'],
                        delta_vs_stock_s=times-stock['t_probe_s'], delta_vs_global_p5_s=times-global5['t_probe_s'])
    save(OUTPUT / name / 'metrics.json', results, exclusive=True)
    print(f'{name}: ray RMS/max {metrics["ray_only"]["rms_ms"]:.9f}/{metrics["ray_only"]["max_abs_ms"]:.9f}ms; '
          f'criteria {results["criteria_pass"]}', flush=True)
    return results


def render(targets, results, build, failures):
    lines = ['# Private regional horizontal-guard causal control', '',
             'Only four horizontal ±0.1° offsets were removed in the private RaySPDR2.f '
             '(lines 840–841). Strict full node bounds replace the inner guard. Bottom guard and '
             'all other source bytes are unchanged. The frozen forward_probe.o and transform object '
             'are linked with the private ray object; no production tomoDD binary was built.', '',
             '**Predeclared, unchanged criteria:** ray-only RMS ≤1 ms / max ≤5 ms; '
             'total RMS ≤5 ms / max ≤10 ms. Full original serialized TRUE geometry, regional MOD=5 '
             'and **unmodified original AK135 support** are used. No sampling or threshold tuning.', '',
             '## Three target checks before full runs', '',
             '| Target | Private time (s) | Ray error (ms) | nseg | Δ stock (ms) | Δ global-P5 (ms) |',
             '|---|---:|---:|---:|---:|---:|']
    for label, r in targets.items():
        lines.append(f'| {label} | {r["private_time_s"]:.12f} | {r["ray_error_ms"]:+.9f} | '
                     f'{r["nseg"]} | {r["delta_vs_stock_ms"]:+.9f} | {r["delta_vs_global_p5_ms"]:+.9f} |')
    lines += ['', '## Complete full-pair results', '',
              '| Geometry | Pairs | Ray RMS (ms) | Ray max abs (ms) | Total RMS (ms) | Total max abs (ms) | Criteria | nseg histogram | Probe wall (s) |',
              '|---|---:|---:|---:|---:|---:|---|---|---:|']
    for name, r in results.items():
        ray, total = r['metrics']['ray_only'], r['metrics']['total']
        lines.append(f'| {name} | {r["complete_pairs"]} | {ray["rms_ms"]:.9f} | {ray["max_abs_ms"]:.9f} | '
                     f'{total["rms_ms"]:.9f} | {total["max_abs_ms"]:.9f} | {r["criteria_pass"]} | '
                     f'{r["nseg_histogram"]} | {r["probe_wall_s"]:.6f} |')
    lines += ['', '## Differences from stock and global-P5', '',
              'Delta = private-guard time minus the named baseline. Global-P5 uses the stock guard/core '
              'and P=5 support; stock uses the original guard/core and original support.', '',
              '| Geometry | Baseline | Changed times | Δ RMS (ms) | Δ max abs (ms) | Changed nseg |',
              '|---|---|---:|---:|---:|---:|']
    for name, r in results.items():
        for baseline in ('stock', 'global_p5'):
            c = r[f'vs_{baseline}']
            m = c['delta_private_minus_baseline']
            lines.append(f'| {name} | {baseline} | {c["time_changed_pairs_exact"]} | {m["rms_ms"]:.9f} | '
                         f'{m["max_abs_ms"]:.9f} | {c["nseg_changed_pairs"]} |')
    passed = len(results) == 2 and not failures and all(all(r['criteria_pass'].values()) for r in results.values())
    lines += ['', '## Conclusion and limits', '',
              (('The private guard-only control passes both unchanged criteria for both geometries. '
                'With unchanged original support and every other core/source/input component fixed, '
                'this isolates the artificial 0.1° horizontal exclusion as causal for the observed '
                'stock regional-homogeneous forward-control failures.') if passed else
               'The predeclared full-control success condition was not met; retained metrics/failures '
               'must be interpreted without tuning the guard, thresholds or inputs.'), '', LIMITATION, '',
              'The three targets were executed first with the actual private probe. All full-pair '
              'requests are byte-identical to stock. Original MOD/support input hashes, source/object '
              'hashes and all old sealed outputs were verified before and after. '
              'Full bias/RMS/P95/max/event-demeaned metrics and arrays are retained in JSON/NPZ; '
              'source diff, source-test results, compiler/link flags and logs, object/binary hashes '
              'and probe raw requests/results/manifests are retained alongside this compact report.', '',
              f'Private compile/link wall: {build["compile_link_elapsed_s"]:.6f}s; compiler/link commands '
              'bounded at 120s; target probes at 120s; full probes at 600s/1200s.']
    if failures:
        lines += ['', '## Retained failures', '', '```json', json.dumps(failures, indent=2), '```']
    return '\n'.join(lines) + '\n'


def main():
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and not any(OUTPUT.iterdir()),
            'regional_guard_control must be precreated, empty and not a symlink')
    ledger = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, '-B', *sys.argv], 'criteria': CRITERIA,
              'private_diagnostic_core_build_only': True, 'production_modified': False,
              'full_probe_timeouts_s': {name: x[2] for name, x in CASES.items()},
              'limitations': LIMITATION}
    save(OUTPUT / 'manifest.json', ledger, exclusive=True)
    started = time.monotonic()
    results, failures = {}, {}
    try:
        cache = OUTPUT / 'cache'
        cache.mkdir(exist_ok=False)
        os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg', XDG_CACHE_HOME=str(cache),
                          TMPDIR=str(cache), NUMBA_CACHE_DIR=str(cache), GFORTRAN_UNBUFFERED_ALL='y',
                          OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        tests = self_tests((FROZEN / 'RaySPDR2.f').read_bytes())
        save(OUTPUT / 'source_tests.json', tests, exclusive=True)
        expected, frozen_build = input_hashes()
        save(OUTPUT / 'input_sha256.json', expected, exclusive=True)
        build = compile_private(frozen_build)
        # After compilation, disable optional fc-list executable discovery during adapter import.
        # Actual probe invocations use absolute paths. No old audit guard is installed.
        os.environ['PATH'] = str(cache)
        check, solver = load_runtime()
        targets = target_checks(check, solver)
        for name in CASES:
            try:
                results[name] = full_geometry(name, check, solver)
            except Exception as error:
                failures[name] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
                save(OUTPUT / f'{name}_failure.json', failures[name], exclusive=True)
        verify(expected)
        validate_patch((FROZEN / 'RaySPDR2.f').read_bytes(), (PRIVATE / 'RaySPDR2.f').read_bytes())
        save(OUTPUT / 'metrics.json', {'targets': targets, 'geometries': results, 'failures': failures,
                                     'criteria': CRITERIA}, exclusive=True)
        (OUTPUT / 'report.md').write_text(render(targets, results, build, failures))
        ledger.update(status='failed' if failures else 'complete', all_old_hashes_unchanged=True,
                      verified_input_files=len(expected), complete_pairs=sum(r['complete_pairs'] for r in results.values()),
                      all_criteria_pass=len(results) == 2 and not failures and all(
                          all(r['criteria_pass'].values()) for r in results.values()))
    except BaseException as error:
        ledger.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        ledger.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save(OUTPUT / 'manifest.json', ledger)
        hashes = {str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
        save(OUTPUT / 'output_sha256.json', hashes, exclusive=True)
    print(json.dumps({key: ledger[key] for key in ('status', 'complete_pairs', 'all_criteria_pass', 'elapsed_s')}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
