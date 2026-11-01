"""Superseding full-pair true-global-P-homogeneity test; no compile or inversion.

Only global_homogeneous/ is writable. Existing serialized TRUE endpoints, MOD,
layer support and frozen forward_probe are reused. Only copied AK135 P speeds
change to 5 km/s. No old audit guard is installed; all caches/temp files are
explicitly confined to the new directory and bytecode writes are disabled.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'runs/tomodd_homogeneous_20261026'
OUTPUT = OLD / 'global_homogeneous'
SNAPSHOT = PROJECT / 'runs/matched_geometry_20261021/snapshot'
BUILD = SNAPSHOT / 'solver'
CASES = {'artificial': (1920, 52, 600), 'geysers': (3776, 26, 1200)}
CRITERIA = {'ray_only': {'rms_ms': 1., 'max_abs_ms': 5.},
            'total': {'rms_ms': 5., 'max_abs_ms': 10.}}
SUPPORT_PATTERN = re.compile(r'^([ \t]*\S+[ \t]+\S+[ \t]+)(\S+)(.*)$')
LIMITATION = ('Forward-only fixed-TRUE-source evidence. The probe bypasses the main input reader; '
              'the exact previously serialized endpoints are reused. No conclusion about inversion '
              'impact or production PBR versus ttmder path integration follows from this test.')


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
    for path, checksum in expected.items():
        require(digest(path) == checksum, f'Input/sealed artifact changed: {path}')


def support_p5(text):
    """Replace only the third token; preserve all other tokens and whitespace."""
    lines = []
    for line in text.splitlines(keepends=True):
        require(len(line.split()) == 5, 'AK135 support must contain five fields per row')
        newline = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        body = line[:-len(newline)] if newline else line
        match = SUPPORT_PATTERN.fullmatch(body)
        require(match is not None, 'Unsupported AK135 row')
        lines.append(match[1] + '5.0000' + match[3] + newline)
    result = ''.join(lines)
    validate_support(text, result)
    return result


def validate_support(original, altered):
    before_lines, after_lines = original.splitlines(keepends=True), altered.splitlines(keepends=True)
    require(len(before_lines) == len(after_lines) > 0, 'Support row count changed')
    for before, after in zip(before_lines, after_lines):
        require(len(before.split()) == len(after.split()) == 5, 'Support column count changed')
        b, a = before.split(), after.split()
        require(b[:2] + b[3:] == a[:2] + a[3:] and float(a[2]) == 5., 'Non-P column changed')
        before_mask = SUPPORT_PATTERN.sub(lambda m: m[1] + '<P>' + m[3], before.rstrip('\r\n'))
        after_mask = SUPPORT_PATTERN.sub(lambda m: m[1] + '<P>' + m[3], after.rstrip('\r\n'))
        require(before_mask == after_mask and before.endswith('\n') == after.endswith('\n'),
                'Whitespace or non-P bytes changed')
    before, after = np.loadtxt(original.splitlines()), np.loadtxt(altered.splitlines())
    np.testing.assert_array_equal(before[:, [0, 1, 3, 4]], after[:, [0, 1, 3, 4]])
    require(np.all(after[:, 2] == 5.), 'Not all support P velocities equal 5')


def self_tests():
    original = ' 6371.000  0.000   5.8000  3.4600  2.7200\n 6356 15 6.5 3.8 2.9\n'
    altered = support_p5(original)
    require(altered == ' 6371.000  0.000   5.0000  3.4600  2.7200\n 6356 15 5.0000 3.8 2.9\n',
            'Support token/whitespace preservation test failed')
    validate_support(original, altered)
    require(support_p5(altered) == altered, 'Support transformation is not idempotent')
    for bad in (altered.replace('3.4600', '3.0000'), altered.replace('6371.000', '6370.000'),
                altered.replace('5.0000', '5.1000'), altered + altered.splitlines()[0] + '\n'):
        try:
            validate_support(original, bad)
        except (ValueError, AssertionError):
            continue
        raise AssertionError('Support mutation guard failed')
    return {'status': 'passed', 'tests': ['only_P_token_changes_with_whitespace_preserved',
            'other_columns_exactly_equal', 'transformation_idempotent', 'reject_Vs_change',
            'reject_radius_change', 'reject_non5_P', 'reject_row_count_change']}


def input_hashes():
    inspection = OLD / 'path_inspection'
    expected = json.loads((inspection / 'input_sha256.json').read_text())
    for name, checksum in json.loads((inspection / 'inspection_sha256.json').read_text()).items():
        expected[str(inspection / name)] = checksum
    expected[str(inspection / 'inspection_sha256.json')] = digest(inspection / 'inspection_sha256.json')
    expected[str(Path(__file__).resolve())] = digest(__file__)
    build = json.loads((BUILD / 'manifest.json').read_text())
    for name in ('forward_probe', 'RaySPDR2.o', 'transform_r_gfortran.o'):
        require(expected[str(BUILD / name)] == build['artifact_sha256'][name], 'Frozen artifact ledger mismatch')
    require(build['artifact_sha256']['forward_probe'] == '75f13999e572e01831e1214258b2fe384d75cfe0dec5f543b4b4b37845bd696f',
            'Unexpected frozen probe')
    require(build['artifact_sha256']['RaySPDR2.o'] == '3e18dffd53b6da144c7d5ea4f74f8929d542f045e8c15a0f9d8287182e73722a',
            'Unexpected frozen ray object')
    verify(expected)
    return expected


def load_runtime():
    path = HERE / 'run_check.py'
    spec = importlib.util.spec_from_file_location('global_homogeneous_sealed_check', path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    require(check.CRITERIA == CRITERIA, 'Criteria differ from predeclared original')
    check.source_contract()
    _, _, solver = check.load_runtime()
    return check, solver


def histogram(values):
    return {str(int(key)): int(count) for key, count in zip(*np.unique(values, return_counts=True))}


def run_geometry(name, check, solver):
    ne, ns, timeout = CASES[name]
    folder = OUTPUT / name
    folder.mkdir(exist_ok=False)
    status = {'status': 'preparing', 'pairs_expected': ne * ns, 'timeout_s': timeout}
    save(folder / 'status.json', status, exclusive=True)
    started = time.monotonic()
    try:
        with np.load(OLD / name / 'arrays.npz', allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        sources, receivers = old['sources_lat_lon_depth_km'], old['receivers_lat_lon_depth_km']
        require(sources.shape == (ne, 3) and receivers.shape == (ns, 3)
                and old['t_probe_s'].shape == (ne, ns), 'Wrong original full-pair counts')
        np.testing.assert_array_equal(check.chord_times(sources, receivers), old['t_chord_s'])
        np.testing.assert_array_equal(check.distances(old['true_event_xyz_km'], old['station_xyz_km']) / 5., old['t_cart_s'])
        for file in ('MOD', 'ak135.15.SKS', 'layer-16.dat'):
            shutil.copyfile(OLD / name / file, folder / file)
        support = folder / 'ak135.15.SKS'
        before = support.read_text()
        support.write_text(support_p5(before))
        validate_support(before, support.read_text())
        require(digest(support) != digest(OLD / name / 'ak135.15.SKS'), 'Support intervention did not change bytes')
        mod_lines = (folder / 'MOD').read_text().splitlines()
        require(tuple(map(int, mod_lines[0].split()[1:])) == (22, 9, 9), 'Unexpected MOD dimensions')
        require(np.all(np.array([list(map(float, line.split())) for line in mod_lines[4:85]]) == 5.),
                'Regional MOD is not 5 km/s at all nodes including padding')
        intervention = {'diagnostic_input_only_not_production': True,
                        'only_AK135_P_column_changed': True, 'non_P_columns_and_whitespace_preserved': True,
                        'original_support_sha256': digest(OLD / name / 'ak135.15.SKS'),
                        'altered_support_sha256': digest(support),
                        'unchanged_files_sha256': {file: digest(folder / file) for file in ('MOD', 'layer-16.dat')}}
        for file, checksum in intervention['unchanged_files_sha256'].items():
            require(checksum == digest(OLD / name / file), 'Unexpected non-P input change')
        save(folder / 'intervention.json', intervention, exclusive=True)
        status['status'] = 'probing'
        save(folder / 'status.json', status)
        print(f'{name}: global P=5; all {ne * ns} pairs; timeout {timeout}s', flush=True)
        probe_started = time.monotonic()
        probe, nseg = solver.forward_times(BUILD, folder, sources, receivers, timeout=timeout)
        probe_wall = time.monotonic() - probe_started
        require(probe.shape == nseg.shape == (ne, ns) and probe.size == ne * ns, 'Incomplete full-pair result')
        child, = folder.glob('small_domain_solver_probe_*')
        original_child, = (OLD / name).glob('small_domain_solver_probe_*')
        require((child / 'coordinates.txt').read_bytes() == (original_child / 'coordinates.txt').read_bytes(),
                'Full serialized endpoint request differs from original')
        child_manifest = json.loads((child / 'manifest.json').read_text())
        require(child_manifest['status'] == 'complete'
                and child_manifest['binary_sha256'] == digest(BUILD / 'forward_probe'), 'Probe provenance mismatch')
        for file, checksum in child_manifest['input_sha256'].items():
            require(checksum == digest(folder / file) == digest(child / file), 'Probe input copy mismatch')
        ray, embedding, total, residual = check.decomposition(probe, old['t_chord_s'], old['t_cart_s'])
        errors = {'ray_only': ray, 'embedding_serialization': embedding, 'total': total}
        delta = probe - old['t_probe_s']
        transitions, counts = np.unique(np.column_stack((old['nseg'].ravel(), nseg.ravel())), axis=0, return_counts=True)
        changes = {'time_changed_pairs_exact': int(np.count_nonzero(probe != old['t_probe_s'])),
                   'time_unchanged_pairs_exact': int(np.count_nonzero(probe == old['t_probe_s'])),
                   'time_changed_over_1us_pairs': int(np.count_nonzero(np.abs(delta) > 1e-6)),
                   'time_changed_over_1ms_pairs': int(np.count_nonzero(np.abs(delta) > .001)),
                   'nseg_changed_pairs': int(np.count_nonzero(nseg != old['nseg'])),
                   'either_time_or_nseg_changed_pairs': int(np.count_nonzero((probe != old['t_probe_s']) | (nseg != old['nseg']))),
                   'delta_global_minus_original_support': check.metrics(delta),
                   'original_nseg_histogram': histogram(old['nseg']), 'global_nseg_histogram': histogram(nseg),
                   'nseg_transitions': {f'{int(pair[0])}->{int(pair[1])}': int(count)
                                        for pair, count in zip(transitions, counts)}}
        data = SimpleNamespace(event_ids=old['event_ids'].tolist(), station_ids=old['station_ids'].tolist(),
                               event_xyz_km=old['true_event_xyz_km'], station_xyz_km=old['station_xyz_km'])
        result = {'geometry': name, 'events': ne, 'stations': ns, 'pairs': ne * ns,
                  'all_pairs_complete': True, 'serialized_request_byte_identical': True,
                  'original_request_sha256': digest(original_child / 'coordinates.txt'),
                  'global_request_sha256': digest(child / 'coordinates.txt'),
                  'metrics': {key: check.metrics(values) for key, values in errors.items()},
                  'criteria_pass': {key: check.passes(check.metrics(errors[key]), key) for key in CRITERIA},
                  'decomposition_exact_float64': residual == 0., 'decomposition_max_abs_residual_s': residual,
                  'changes_from_regional5_original_support': changes,
                  'worst_pairs': {key: check.worst(values, data, sources, receivers, old['t_cart_s'], old['t_chord_s'], probe)
                                  for key, values in errors.items()},
                  'original_actual_station_sha256': digest(OLD / name / 'station.dat'),
                  'intervention': intervention, 'probe_wall_s': probe_wall, 'elapsed_s': time.monotonic() - started,
                  'limitations': LIMITATION}
        keep = ('event_ids', 'station_ids', 'true_event_xyz_km', 'station_xyz_km',
                'sources_lat_lon_depth_km', 'receivers_lat_lon_depth_km', 't_cart_s', 't_chord_s')
        np.savez_compressed(folder / 'arrays.npz', **{key: old[key] for key in keep},
                            t_probe_s=probe, nseg=nseg, ray_error_s=ray, embedding_serialization_error_s=embedding,
                            total_error_s=total, original_support_t_probe_s=old['t_probe_s'],
                            original_support_nseg=old['nseg'], delta_time_s=delta, delta_nseg=nseg - old['nseg'])
        save(folder / 'metrics.json', result, exclusive=True)
        status.update(status='complete', criteria_pass=result['criteria_pass'])
        print(f'{name}: complete; ray RMS/max {result["metrics"]["ray_only"]["rms_ms"]:.9f}/'
              f'{result["metrics"]["ray_only"]["max_abs_ms"]:.9f}ms; criteria {result["criteria_pass"]}', flush=True)
        return result
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        status['elapsed_s'] = time.monotonic() - started
        save(folder / 'status.json', status)


def render(results, failures):
    lines = ['# Superseding true-global-homogeneity test — full pairs', '',
             '**This is the superseding test of true global P=5 km/s homogeneity.** '
             'The original sealed report remains unchanged and describes regional P=5 with nonhomogeneous AK135 support. '
             'This additive test changes only the copied AK135 P column to 5 km/s, with unchanged original frozen probe, '
             'serialized TRUE endpoints, MOD and layer support. No compile, inversion, sampling or article edits.', '',
             '## Unchanged predeclared criteria', '',
             'Ray-only RMS ≤1 ms and maximum absolute error ≤5 ms; total RMS ≤5 ms and maximum absolute error ≤10 ms '
             '(10 ms absolute pick-noise reference). No thresholds were tuned.', '',
             '| Geometry | Error (ms) | Signed bias | RMS | P95 abs | Max abs | Event-demeaned RMS |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        for key, m in result['metrics'].items():
            lines.append(f'| {name} | {key} | {m["signed_bias_ms"]:.9f} | {m["rms_ms"]:.9f} | '
                         f'{m["p95_abs_ms"]:.9f} | {m["max_abs_ms"]:.9f} | {m["per_event_demeaned_rms_ms"]:.9f} |')
    for name, result in results.items():
        changes = result['changes_from_regional5_original_support']
        lines += ['', f'## {name}', '', f'{result["events"]} × {result["stations"]} = {result["pairs"]} complete pairs. '
                  f'Probe wall {result["probe_wall_s"]:.6f} s. Criteria: `{result["criteria_pass"]}`.',
                  f'Full coordinate request byte-identical to original: **True**. Exact decomposition residual: '
                  f'{result["decomposition_max_abs_residual_s"]} s.', '',
                  'Changes are measured against regional MOD=5 with original nonhomogeneous AK135 support. '
                  'Exact time-change counts include float32 output changes; 1 μs/1 ms counts separate material changes.',
                  '```json', json.dumps(changes, indent=2), '```', '',
                  'Input-only intervention and hashes:', '```json', json.dumps(result['intervention'], indent=2), '```', '',
                  'Worst pair IDs and original XYZ, serialized endpoints, and all three times:',
                  '```json', json.dumps(result['worst_pairs'], indent=2), '```']
    lines += ['', '## Definitions, provenance and scope', '',
              '`t_cart` is original supplied Cartesian distance / 5; `t_chord` is the saved radius-6371-km spherical chord '
              'on serialized TRUE endpoints; `t_probe` is the unchanged frozen probe with globally P=5 support. '
              'Ray-only = probe − chord; embedding/serialization = chord − Cartesian; total = probe − Cartesian. '
              'The exact same source/station arrays and source-major requests are reused and verified.', '',
              'Original actual station.dat hashes, binary/object hashes, all previous input/sealed-output ledgers, '
              'and this script hash are verified before and after. `input_sha256.json` records those hashes; '
              '`output_sha256.json` seals every new artifact except itself. Raw probe requests/results/manifests '
              'and source-major arrays, baseline times/segments and deltas are retained.', '',
              LIMITATION]
    if failures:
        lines += ['', '## Retained failures', '', '```json', json.dumps(failures, indent=2), '```']
    return '\n'.join(lines) + '\n'


def main():
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and not any(OUTPUT.iterdir()),
            'global_homogeneous must be precreated, empty and not a symlink')
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, '-B', *sys.argv], 'python': sys.version, 'numpy': np.__version__,
                'title': 'Superseding true-global-homogeneity test', 'criteria': CRITERIA,
                'timeouts_s': {name: values[2] for name, values in CASES.items()},
                'compile_or_inversion': False, 'limitations': LIMITATION}
    save(OUTPUT / 'manifest.json', manifest, exclusive=True)
    started = time.monotonic()
    results, failures = {}, {}
    try:
        cache = OUTPUT / 'cache'
        cache.mkdir(exist_ok=False)
        # Empty executable-search directory makes optional fontconfig discovery fall back;
        # the probe is invoked by absolute path. No old or new audit guard is installed.
        os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg', XDG_CACHE_HOME=str(cache),
                          TMPDIR=str(cache), NUMBA_CACHE_DIR=str(cache), PATH=str(cache),
                          GFORTRAN_UNBUFFERED_ALL='y', OPENBLAS_NUM_THREADS='1',
                          OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        tests = self_tests()
        save(OUTPUT / 'tests.json', tests, exclusive=True)
        expected = input_hashes()
        save(OUTPUT / 'input_sha256.json', expected, exclusive=True)
        check, solver = load_runtime()
        for name in CASES:
            try:
                results[name] = run_geometry(name, check, solver)
            except Exception as error:
                failures[name] = f'{type(error).__name__}: {error}'
                print(f'{name}: FAILED: {error}', flush=True)
        verify(expected)
        save(OUTPUT / 'metrics.json', {'criteria': CRITERIA, 'geometries': results, 'failures': failures}, exclusive=True)
        (OUTPUT / 'report.md').write_text(render(results, failures))
        manifest.update(status='failed' if failures else 'complete', all_old_hashes_unchanged=True,
                        verified_input_files=len(expected), complete_pairs=sum(r['pairs'] for r in results.values()),
                        all_criteria_pass=not failures and len(results) == 2 and all(
                            all(r['criteria_pass'].values()) for r in results.values()))
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat(),
                        failures=failures)
        save(OUTPUT / 'manifest.json', manifest)
        hashes = {str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
        save(OUTPUT / 'output_sha256.json', hashes, exclusive=True)
    print(json.dumps({key: manifest[key] for key in ('status', 'complete_pairs', 'all_criteria_pass', 'elapsed_s')}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
