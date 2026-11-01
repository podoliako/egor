"""Narrow, forward-only fresh-process/order reproduction of sealed diagnostic pairs.

Run from EMTomo with:
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_homogeneous_20261026/reproduce_pairs.py
Only pair_reproduction/ is writable. Each unchanged frozen probe has a 120 s timeout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'runs/tomodd_homogeneous_20261026'
OUTPUT = OLD / 'pair_reproduction'
SNAPSHOT = PROJECT / 'runs/matched_geometry_20261021/snapshot'
BINARY = SNAPSHOT / 'solver/forward_probe'
TIMEOUT_S = 120


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


def write_guard(path):
    if not isinstance(path, int):
        require(Path(os.fsdecode(path)).resolve().is_relative_to(OUTPUT.resolve()),
                f'Write outside pair_reproduction: {path}')


def install_guard():
    def audit(event, args):
        if event == 'open':
            if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                write_guard(args[0])
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
            write_guard(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            write_guard(args[0])
            write_guard(args[1])
        elif event == 'subprocess.Popen':
            executable, argv, cwd, _ = args
            if Path(executable).resolve() != BINARY.resolve() or len(argv) != 1:
                # Optional fontconfig discovery handles OSError without launching anything.
                raise PermissionError('Only the existing frozen forward_probe may execute')
            write_guard(cwd)
        elif event in ('os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
            raise PermissionError(f'Unapproved process launch: {event}')
    sys.addaudithook(audit)


def input_hashes():
    expected = json.loads((OLD / 'input_sha256.json').read_text())
    for ledger in ('inspection_sha256.json', 'run_sha256.json'):
        for name, checksum in json.loads((OLD / ledger).read_text()).items():
            path = str(OLD / name)
            require(path not in expected or expected[path] == checksum, 'Conflicting old hash ledgers')
            expected[path] = checksum
        expected[str(OLD / ledger)] = digest(OLD / ledger)
    expected[str(Path(__file__).resolve())] = digest(__file__)
    verify(expected)
    return expected


def load_check():
    # Import utilities only: do not call the old main or install its wider write guard.
    path = HERE / 'run_check.py'
    spec = importlib.util.spec_from_file_location('sealed_homogeneous_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_chord(source, receiver):
    lat1, lon1 = np.deg2rad(source[:2])
    lat2, lon2 = np.deg2rad(receiver[:2])
    h = np.sin((lat1 - lat2) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
    r1, r2 = 6371. - source[2], 6371. - receiver[2]
    return float(np.sqrt((r1 - r2) ** 2 + 4 * r1 * r2 * h) / 5.)


def select_targets(check):
    targets = {}
    for geometry in ('geysers', 'artificial'):
        with np.load(OLD / geometry / 'arrays.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        shape = arrays['t_probe_s'].shape
        selections = [(f'{geometry}_maxpositive', np.unravel_index(np.argmax(arrays['ray_error_s']), shape))]
        if geometry == 'geysers':
            i, = np.flatnonzero(arrays['event_ids'] == '3157429')
            j, = np.flatnonzero(arrays['station_ids'] == 'RGP')
            selections += [('geysers_outlier', (i, j)),
                           ('geysers_preceding', np.unravel_index(int(i) * shape[1] + int(j) - 1, shape))]
        probe, = (OLD / geometry).glob('small_domain_solver_probe_*')
        requests = (probe / 'coordinates.txt').read_text().splitlines()
        raw_results = (probe / 'forward.tsv').read_text().splitlines()
        require(int(requests[0]) == arrays['t_probe_s'].size, 'Original request count changed')
        for label, (i, j) in selections:
            i, j = int(i), int(j)
            index = i * shape[1] + j
            source = arrays['sources_lat_lon_depth_km'][i]
            receiver = arrays['receivers_lat_lon_depth_km'][j]
            raw = np.asarray([float(x) for x in requests[index + 1].split()])
            np.testing.assert_array_equal(raw, np.concatenate((receiver, source)))
            row = raw_results[index + 1].split()
            require(int(row[0]) == index + 1 and float(row[1]) == arrays['t_probe_s'][i, j]
                    and int(row[2]) == arrays['nseg'][i, j], 'Original raw result/array mismatch')
            analytic = float(check.chord_times(source[None, :], receiver[None, :])[0, 0])
            require(analytic == arrays['t_chord_s'][i, j], 'Saved chord does not match identical endpoints')
            require(abs(analytic - independent_chord(source, receiver)) < 5e-12,
                    'Independent chord self-check failed')
            target = {'label': label, 'geometry': geometry, 'event_id': str(arrays['event_ids'][i]),
                      'station_id': str(arrays['station_ids'][j]), 'event_index': i, 'station_index': j,
                      'original_pair_number': index + 1,
                      'source_xyz_km': arrays['true_event_xyz_km'][i].tolist(),
                      'receiver_xyz_km': arrays['station_xyz_km'][j].tolist(),
                      'source_lat_lon_depth_km': source.tolist(), 'receiver_lat_lon_depth_km': receiver.tolist(),
                      't_cart_s': float(arrays['t_cart_s'][i, j]), 't_chord_s': analytic,
                      'original_t_probe_s': float(arrays['t_probe_s'][i, j]),
                      'original_nseg': int(arrays['nseg'][i, j]),
                      'original_ray_error_ms': float(arrays['ray_error_s'][i, j] * 1000)}
            targets[label] = target
    outlier = targets['geysers_outlier']
    require(outlier['original_ray_error_ms'] < -300, 'Selected event is not the original outlier')
    require(targets['geysers_preceding']['original_pair_number'] + 1 == outlier['original_pair_number'],
            'Preceding pair is not adjacent in the original request')
    return targets


def plan():
    jobs = []
    for label in ('geysers_outlier', 'geysers_maxpositive', 'artificial_maxpositive'):
        for repeat in (1, 2):
            jobs.append((f'{label}_alone_{repeat}', [(label, False)]))
        for repeat in (1, 2):
            jobs.append((f'{label}_reversed_{repeat}', [(label, True)]))
        jobs.append((f'{label}_twice', [(label, False), (label, False)]))
    jobs += [('preceding_alone', [('geysers_preceding', False)]),
             ('preceding_then_outlier', [('geysers_preceding', False), ('geysers_outlier', False)]),
             ('outlier_then_preceding', [('geysers_outlier', False), ('geysers_preceding', False)])]
    return jobs


def run_job(name, sequence, targets, solver):
    folder = OUTPUT / name
    folder.mkdir(exist_ok=False)
    geometry, = {targets[label]['geometry'] for label, _ in sequence}
    for file in ('MOD', 'ak135.15.SKS', 'layer-16.dat'):
        shutil.copyfile(OLD / geometry / file, folder / file)
    sources, receivers = [], []
    for label, reversed_endpoints in sequence:
        target = targets[label]
        source, receiver = target['source_lat_lon_depth_km'], target['receiver_lat_lon_depth_km']
        sources.append(receiver if reversed_endpoints else source)
        receivers.append(source if reversed_endpoints else receiver)
    pairs = np.column_stack((np.arange(len(sequence)), np.arange(len(sequence))))
    started = time.monotonic()
    times, segments = solver.forward_times(BINARY.parent, folder, sources, receivers, pairs=pairs, timeout=TIMEOUT_S)
    elapsed = time.monotonic() - started
    probe, = folder.glob('small_domain_solver_probe_*')
    manifest = json.loads((probe / 'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['binary_sha256'] == digest(BINARY),
            'Probe provenance mismatch')
    rows = []
    for index, ((label, reversed_endpoints), measured, nseg) in enumerate(zip(sequence, times, segments)):
        target = targets[label]
        analytic = independent_chord(np.asarray(sources[index]), np.asarray(receivers[index]))
        require(abs(analytic - target['t_chord_s']) < 5e-12, 'Reversed endpoint analytic symmetry failed')
        # Preserve the original analytic convention for exact baseline comparisons.
        chord = target['t_chord_s']
        rows.append({'job': name, 'row': index + 1, 'target': label, 'reversed_endpoints': reversed_endpoints,
                     'measured_s': float(measured), 'analytic_chord_s': chord,
                     'independent_chord_s': analytic, 'cartesian_s': target['t_cart_s'], 'nseg': int(nseg),
                     'ray_error_ms': float((measured - chord) * 1000),
                     'delta_from_original_ms': float((measured - target['original_t_probe_s']) * 1000),
                     'source_xyz_km': target['receiver_xyz_km'] if reversed_endpoints else target['source_xyz_km'],
                     'receiver_xyz_km': target['source_xyz_km'] if reversed_endpoints else target['receiver_xyz_km']})
    result = {'job': name, 'probe_wall_s': elapsed, 'timeout_s': TIMEOUT_S,
              'probe_directory': str(probe.relative_to(OUTPUT)), 'rows': rows}
    save(folder / 'results.json', result, exclusive=True)
    return result


def render(targets, results, failures):
    lines = ['# Narrow pair reproduction', '',
             'Existing unchanged forward_probe; identical serialized endpoints; completed homogeneous MOD; '
             'fresh process per job; 120 s per probe. No build, patch, inversion or sealed-file writes.', '',
             '| Target | Event / station | Source XYZ (km) | Receiver XYZ (km) | Original probe (s) | Original nseg |',
             '|---|---|---|---|---:|---:|']
    for label, target in targets.items():
        lines.append(f'| {label} | {target["event_id"]} / {target["station_id"]} | '
                     f'{target["source_xyz_km"]} | {target["receiver_xyz_km"]} | '
                     f'{target["original_t_probe_s"]:.12f} | {target["original_nseg"]} |')
    lines += ['', '| Job / row | Target | Reversed | Measured (s) | Chord (s) | Probe − chord (ms) | Δ original (ms) | nseg |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        for row in result['rows']:
            lines.append(f'| {row["job"]} / {row["row"]} | {row["target"]} | {row["reversed_endpoints"]} | '
                         f'{row["measured_s"]:.12f} | {row["analytic_chord_s"]:.12f} | '
                         f'{row["ray_error_ms"]:+.6f} | {row["delta_from_original_ms"]:+.6f} | {row["nseg"]} |')
    outlier_alone = [r for job in results for r in job['rows']
                     if r['job'].startswith('geysers_outlier_alone_')]
    if len(outlier_alone) == 2:
        reproduced = all(r['measured_s'] == targets['geysers_outlier']['original_t_probe_s']
                         and r['nseg'] == targets['geysers_outlier']['original_nseg'] for r in outlier_alone)
        lines += ['', f'Outlier exactly reproduces original time and segment count in both fresh-alone processes: **{reproduced}**.',
                  'A fresh-alone disappearance would be evidence of a state/order dependency; persistence '
                  'does not exclude other state effects. Endpoint reversal and sequence rows are reported '
                  'without changing the model or thresholds. No internal mechanism is inferred here.', '',
                  'The unchanged cellvel regional guard is 0.1 degree and finite-difference gradient '
                  'evaluations can encounter fallback; mechanism analysis is left to the parent task.']
    if failures:
        lines += ['', '## Retained failures', '', '```json', json.dumps(failures, indent=2), '```']
    return '\n'.join(lines) + '\n'


def main():
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and not any(OUTPUT.iterdir()),
            'pair_reproduction must be a precreated empty nonsymlink directory')
    ledger = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, '-B', *sys.argv], 'python': sys.version, 'numpy': np.__version__,
              'timeout_s_per_probe': TIMEOUT_S, 'build_or_patch': False}
    save(OUTPUT / 'manifest.json', ledger, exclusive=True)
    install_guard()
    started = time.monotonic()
    results, failures = [], {}
    expected = None
    try:
        cache = OUTPUT / 'cache'
        cache.mkdir(exist_ok=False)
        os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg', XDG_CACHE_HOME=str(cache),
                          TMPDIR=str(cache), NUMBA_CACHE_DIR=str(cache), GFORTRAN_UNBUFFERED_ALL='y',
                          OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        expected = input_hashes()
        save(OUTPUT / 'input_sha256.json', expected, exclusive=True)
        check = load_check()
        check.source_contract()
        targets = select_targets(check)
        save(OUTPUT / 'targets.json', targets, exclusive=True)
        # The existing matched worker load_adapter guard binds the frozen package exactly once.
        _, _, solver = check.load_runtime()
        for name, sequence in plan():
            try:
                result = run_job(name, sequence, targets, solver)
                results.append(result)
                print(name + ': ' + '; '.join(f'{r["measured_s"]:.12f}s '
                      f'{r["ray_error_ms"]:+.6f}ms nseg={r["nseg"]}' for r in result['rows']), flush=True)
            except Exception as error:
                failures[name] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
                save(OUTPUT / name / 'failure.json', failures[name], exclusive=True)
                print(f'{name}: FAILED: {error}', flush=True)
        verify(expected)
        save(OUTPUT / 'results.json', {'targets': targets, 'jobs': results, 'failures': failures}, exclusive=True)
        (OUTPUT / 'report.md').write_text(render(targets, results, failures))
        ledger.update(status='failed' if failures else 'complete', sealed_and_input_hashes_unchanged=True,
                      verified_input_file_count=len(expected), jobs_complete=len(results))
    except BaseException as error:
        ledger.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        ledger.update(elapsed_s=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat(), failures=failures)
        save(OUTPUT / 'manifest.json', ledger)
        hashes = {str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
        save(OUTPUT / 'output_sha256.json', hashes, exclusive=True)
    print(json.dumps({'status': ledger['status'], 'jobs_complete': len(results),
                      'elapsed_s': ledger['elapsed_s'], 'old_artifacts_unchanged': True}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
