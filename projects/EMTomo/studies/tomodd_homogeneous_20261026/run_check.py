"""Forward-only, full-pair homogeneous endpoint diagnostic; never builds/inverts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUTPUT = PROJECT / 'runs/tomodd_homogeneous_20261026'
OLD = PROJECT / 'runs/matched_geometry_20261021'
SNAPSHOT = OLD / 'snapshot'
FORWARD = SNAPSHOT / 'projects/forward_modeling'
WORKER = HERE.parent / 'matched_geometry_20261021/tomodd_worker.py'
CASES = {
    'artificial': (PROJECT / 'runs/tomodd_sensitivity_20261026/outputs/dd1-d40-s5', 1920, 52, 600),
    'geysers': (PROJECT / 'runs/tomodd_dd1_geometry_20261026/outputs/geysers-dd1-d40-s5', 3776, 26, 1200),
}
CRITERIA = {'ray_only': {'rms_ms': 1.0, 'max_abs_ms': 5.0},
            'total': {'rms_ms': 5.0, 'max_abs_ms': 10.0}}
LIMITATION = ('The probe bypasses the main input reader; serialized endpoints and scalar types '
              'are reproduced from actual files and verified frozen source. This homogeneous, '
              'fixed-TRUE-source harness does not establish the cause of inversion degradation '
              'and cannot definitively exclude bugs outside its exercised path.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_root(path, expected=OUTPUT):
    path = Path(path)
    require(not path.is_symlink() and path.resolve() == expected.absolute(), 'Unapproved output root')
    require(path.is_dir(), 'Output root must be precreated')
    for child in path.iterdir():
        require(child.name in ('validation', 'failed_preprobe_001')
                and child.is_dir() and not child.is_symlink(),
                'Output root already claimed or contains unexpected files')
        if child.name == 'failed_preprobe_001':
            failed = json.loads((child / 'manifest.json').read_text())
            require(failed['status'] == 'failed' and not any((child / name).exists() for name in CASES),
                    'Only an archived pre-probe startup failure is permitted')


def check_write(path, root=OUTPUT):
    if isinstance(path, int):
        return
    require(Path(os.fsdecode(path)).resolve().is_relative_to(root.resolve()),
            f'Write outside approved run root: {path}')


def check_process(executable, argv, cwd):
    if (Path(executable).resolve() != (SNAPSHOT / 'solver/forward_probe').resolve()
            or len(argv) != 1):
        # OSError lets optional Matplotlib font discovery fall back without launching fc-list.
        raise PermissionError('Only the frozen forward_probe may execute')
    check_write(cwd)


def install_audit_guard():
    """Confine library/cache writes and allow only the existing probe subprocess."""
    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check_write(path)
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
            check_write(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check_write(args[0])
            check_write(args[1])
        elif event == 'subprocess.Popen':
            executable, argv, cwd, _ = args
            check_process(executable, argv, cwd)
        elif event in ('os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
            raise ValueError(f'Unapproved process launch: {event}')
    sys.addaudithook(audit)


def verify_hashes(expected):
    for path, checksum in expected.items():
        require(digest(path) == checksum, f'Provenance hash mismatch: {path}')


def source_contract():
    """Fail closed if inspected reader/promotion/call conventions change."""
    files = {name: (SNAPSHOT / 'solver' / name).read_text().lower()
             for name in ('getdata_SPDR.f', 'trialsrc_FDD_shot.f', 'tomoDD-SE.f', 'forward_probe.f')}
    compact = {name: ''.join(text.split()) for name, text in files.items()}
    reader = compact['getdata_SPDR.f']
    for token in ('realev_lat(maxeve)', 'realev_lon(maxeve)', 'realev_dep(maxeve)',
                  'realsta_lat(maxsta)', 'realsta_lon(maxsta)', 'realsta_eve(maxsta)',
                  'sta_eve(i)=-sta_eve(i)/1000.0'):
        require(token in reader, f'Reader type/sign contract changed: {token}')
    trial = compact['trialsrc_FDD_shot.f']
    for token in ('src_lon(i)=ev_lon(i)', 'src_lat(i)=ev_lat(i)', 'src_dep(i)=ev_dep(i)'):
        require(token in trial, f'Source promotion changed: {token}')
    main = compact['tomoDD-SE.f']
    for token in ('realsta_elv(maxsta)', 'realtlat', 'realtlon', 'realtdep,dep0,dep1',
                      'tlat=sta_lat(i)', 'tlon=sta_lon(i)', 'tdep=sta_elv(i)',
                  'tlat=src_lat(j)', 'tlon=src_lon(j)', 'tdep=src_dep(j)',
                  'callpbr(isp,aar,bbr,hr,aas,bbs,hs,rp_geo,nseg2,tmp_ttp(k,j))'):
        require(token in main, f'Inversion endpoint contract changed: {token}')
    probe = compact['forward_probe.f']
    require('read(*,*,iostat=ios)rlat,rlon,rdep,slat,slon,sdep' in probe
            and 'callpbr(0,rlat,rlon,rdep,slat,slon,sdep,w,ni,tt)' in probe,
            'Probe receiver-first/fresh-scalar contract changed')


def provenance():
    expected = {}
    parent = json.loads((OLD / 'manifest.json').read_text())
    for name, checksum in parent['snapshot_sha256'].items():
        expected[str(SNAPSHOT / name)] = checksum
    expected[str(OLD / 'manifest.json')] = digest(OLD / 'manifest.json')
    build = SNAPSHOT / 'solver'
    manifest = json.loads((build / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Incomplete build')
    recipe = manifest['recipe']
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    require(signature == manifest['signature'] == '2d6bca5eb5b1891a004fb0a3ec80951e767660bdcaf6c63554344a1ff6b200dd',
            'Build recipe/signature mismatch')
    require(recipe['min_segment_km'] == .05 and recipe['coordinate_roundoff_fix'] is True,
            'Unexpected ray build parameters')
    for name, checksum in manifest['artifact_sha256'].items():
        expected[str(build / name)] = checksum
    for name, key in (('RaySPDR2.f', 'patched_ray_sha256'),
                      ('transform_r_gfortran.c', 'patched_transform_sha256'),
                      ('include/tomoFDD.inc', 'patched_include_sha256'),
                      ('forward_probe.f', 'probe_source_sha256')):
        expected[str(build / name)] = recipe[key]
    wrapper = SNAPSHOT / 'benchmark/tomodd_benchmark/small_domain_solver.py'
    require(digest(wrapper) == recipe['wrapper_sha256'], 'Forward wrapper differs from build recipe')
    link = manifest['commands'][-1]
    require(all(name in link for name in ('forward_probe.o', 'RaySPDR2.o', 'transform_r_gfortran.o', 'forward_probe')),
            'Unexpected recorded probe linkage')
    require('RaySPDR2.f' in (build / 'Makefile').read_text(), 'Production Makefile uses another ray source')
    for case, (prior, ne, ns, _) in CASES.items():
        info = json.loads((prior / 'worker_info.json').read_text())
        cfg = json.loads((prior / 'config.json').read_text())
        require(info['state'] == 'completed' and (info['events'], info['stations']) == (ne, ns),
                f'Incomplete/incorrect prior run: {case}')
        require(Path(cfg['source_root']).resolve() == FORWARD.resolve(), 'Non-frozen geometry source')
        require(cfg['coordinate_mapping']['center_lat'] == 30.3
                and cfg['coordinate_mapping']['center_lon'] == 103.3, 'Unexpected mapping')
        require(info['solver_manifest_sha256'] == digest(build / 'manifest.json'), 'Prior solver manifest changed')
        require(info['solver_sha256'] == manifest['artifact_sha256']['tomoDD-SE'], 'Prior production binary differs')
        expected.update(info['input_sha256'])
        for name, checksum in info['snapshot_source_sha256'].items():
            expected[str(SNAPSHOT / 'benchmark' / name)] = checksum
        for name, checksum in info['generated_input_sha256'].items():
            expected[str(prior / 'work/inversion' / name)] = checksum
        for name, checksum in info['support_sha256'].items():
            expected[str(SNAPSHOT / 'support' / name)] = checksum
            require(digest(prior / 'work/inversion' / name) == checksum, 'Prior support differs')
        for name in ('config.json', 'worker_info.json', 'truth.npz', 'state.npz', 'results.json'):
            expected[str(prior / name)] = digest(prior / name)
    expected[str(WORKER)] = digest(WORKER)
    for name in ('run_check.py', 'test_run_check.py', 'README.md'):
        expected[str(HERE / name)] = digest(HERE / name)
    verify_hashes(expected)
    source_contract()
    return expected


def load_runtime():
    spec = importlib.util.spec_from_file_location('homogeneous_matched_worker', WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    adapter = worker.load_adapter(SNAPSHOT / 'benchmark')
    from tomodd_benchmark import synthetic, small_domain_solver
    return adapter, synthetic, small_domain_solver


def load_geometry(case, adapter):
    prior, ne, ns, _ = CASES[case]
    cfg = json.loads((prior / 'config.json').read_text())
    data = adapter.load_external_data(FORWARD, cfg['source_experiment'])
    require((len(data.event_ids), len(data.station_ids)) == (ne, ns), 'Incorrect full-pair dimensions')
    np.testing.assert_array_equal(data.lengths_km, [20., 7., 7.])
    np.testing.assert_array_equal(data.origin_km, [0., 0., 0.])
    grid, exp = adapter.build_tomodd_geometry(data, node_spacing_km=1., free_buffer_layers=0)
    require(grid.initial_vp.shape == (9, 9, 22) and np.all(grid.initial_vp == 5.),
            'Model must be 5 km/s at ALL nodes including padding')
    for name in ('truth.npz', 'state.npz'):
        with np.load(prior / name, allow_pickle=False) as archive:
            for key, values in (('true_event_xyz_km', data.event_xyz_km), ('true_events', exp.true_events),
                                ('stations', exp.stations), ('lon_nodes', exp.lon_nodes),
                                ('lat_nodes', exp.lat_nodes), ('depths', exp.depths)):
                np.testing.assert_array_equal(archive[key], values, err_msg=f'{name}: {key}')
    np.testing.assert_array_equal(exp.true_events[:, 0], adapter._numeric_event_ids(data.event_ids))
    return data, grid, exp


def parse_events(path, expected_ids):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    require([int(row[9]) for row in rows] == list(expected_ids), 'TRUE event file ID/order mismatch')
    return np.asarray([[row[2], row[3], row[4]] for row in rows], dtype=np.float32).astype(np.float64)


def parse_stations(path, expected_ids):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    require([row[0] for row in rows] == list(expected_ids), 'Station file ID/order mismatch')
    values = np.asarray([row[1:4] for row in rows], dtype=np.float32)
    values[:, 2] = -values[:, 2] / np.float32(1000.)
    return values.astype(np.float64)


def spherical_xyz(coords):
    coords = np.asarray(coords, dtype=np.float64)
    lat, lon = np.deg2rad(coords[:, :2]).T
    radius = 6371. - coords[:, 2]
    return radius[:, None] * np.column_stack((np.cos(lat) * np.cos(lon),
                                             np.cos(lat) * np.sin(lon), np.sin(lat)))


def distances(sources, receivers):
    return np.linalg.norm(np.asarray(sources)[:, None, :] - np.asarray(receivers)[None, :, :], axis=2)


def chord_times(sources, receivers):
    return distances(spherical_xyz(sources), spherical_xyz(receivers)) / 5.


def metrics(error_s):
    error = np.asarray(error_s, dtype=np.float64) * 1000.
    require(error.ndim == 2 and error.size > 0 and np.isfinite(error).all(), 'Invalid error matrix')
    demeaned = error - error.mean(axis=1, keepdims=True)
    return {'signed_bias_ms': float(error.mean()), 'rms_ms': float(np.sqrt(np.mean(error ** 2))),
            'p95_abs_ms': float(np.percentile(np.abs(error), 95)),
            'max_abs_ms': float(np.max(np.abs(error))),
            'per_event_demeaned_rms_ms': float(np.sqrt(np.mean(demeaned ** 2)))}


def decomposition(t_probe, t_chord, t_cart):
    ray = t_probe - t_chord
    embedding = t_chord - t_cart
    total = t_probe - t_cart
    residual = total - (ray + embedding)
    require(np.array_equal(total, ray + embedding), 'Error decomposition is not exactly equal in float64')
    return ray, embedding, total, float(np.max(np.abs(residual)))


def stats(values):
    values = np.asarray(values)
    return {'min': float(values.min()), 'mean': float(values.mean()),
            'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max())}


def passes(result, kind):
    return all(result[key] <= limit for key, limit in CRITERIA[kind].items())


def worst(error, data, sources, receivers, t_cart, t_chord, t_probe):
    i, j = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    return {'event_index': int(i), 'station_index': int(j), 'event_id': data.event_ids[i],
            'station_id': data.station_ids[j], 'event_xyz_km': data.event_xyz_km[i].tolist(),
            'station_xyz_km': data.station_xyz_km[j].tolist(),
            'source_serialized_lat_lon_depth': sources[i].tolist(),
            'receiver_serialized_lat_lon_depth': receivers[j].tolist(),
            'signed_error_ms': float(error[i, j] * 1000.), 't_cart_s': float(t_cart[i, j]),
            't_chord_s': float(t_chord[i, j]), 't_probe_s': float(t_probe[i, j])}


def run_geometry(case, adapter, synthetic, solver):
    started = time.monotonic()
    prior, _, _, timeout = CASES[case]
    folder = OUTPUT / case
    folder.mkdir(exist_ok=False)
    status = {'status': 'preparing', 'started_utc': now(), 'timeout_s': timeout}
    save(folder / 'status.json', status, exclusive=True)
    try:
        data, grid, exp = load_geometry(case, adapter)
        synthetic.write_events(folder / 'true_event.dat', exp.true_events)
        sources = parse_events(folder / 'true_event.dat', exp.true_events[:, 0].astype(int))
        # Byte comparison tests the transfer; PBR receives the actual old file, not the regenerated one.
        synthetic.write_stations(folder / 'regenerated_station.dat', exp)
        actual_station = prior / 'work/inversion/station.dat'
        require((folder / 'regenerated_station.dat').read_bytes() == actual_station.read_bytes(),
                'Actual station.dat differs from frozen adapter transfer')
        shutil.copyfile(actual_station, folder / 'station.dat')
        receivers = parse_stations(folder / 'station.dat', data.station_ids)
        adapter._write_external_mod(folder / 'MOD', exp, np.full_like(grid.initial_vp, 5.))
        lines = (folder / 'MOD').read_text().splitlines()
        require(tuple(map(int, lines[0].split()[1:])) == (22, 9, 9), 'Incorrect MOD dimensions')
        vp = np.array([list(map(float, row.split())) for row in lines[4:4 + 9 * 9]])
        require(vp.shape == (81, 22) and np.all(vp == 5.), 'Nonhomogeneous serialized MOD')
        for name in ('ak135.15.SKS', 'layer-16.dat'):
            shutil.copyfile(SNAPSHOT / 'support' / name, folder / name)
        t_cart = distances(data.event_xyz_km, data.station_xyz_km) / 5.
        t_chord = chord_times(sources, receivers)
        wrong_height = receivers.copy()
        wrong_height[:, 2] *= -1
        t_wrong_height = chord_times(sources, wrong_height)
        t_reversed = chord_times(sources, receivers[::-1])
        arrays = {'event_ids': np.asarray(data.event_ids), 'station_ids': np.asarray(data.station_ids),
                  'true_event_xyz_km': data.event_xyz_km, 'station_xyz_km': data.station_xyz_km,
                  'sources_lat_lon_depth_km': sources, 'receivers_lat_lon_depth_km': receivers,
                  't_cart_s': t_cart, 't_chord_s': t_chord,
                  't_wrong_height_s': t_wrong_height, 't_reversed_station_s': t_reversed}
        np.savez_compressed(folder / 'endpoints_analytic.npz', **arrays)
        status.update(status='probing', pair_count=int(t_cart.size), prepared_elapsed_s=time.monotonic() - started)
        save(folder / 'status.json', status)
        print(f'{case}: probing all {t_cart.size} pairs; timeout {timeout}s', flush=True)
        probe_started = time.monotonic()
        t_probe, nseg = solver.forward_times(SNAPSHOT / 'solver', folder, sources, receivers, timeout=timeout)
        probe_elapsed = time.monotonic() - probe_started
        require(t_probe.shape == t_cart.shape and nseg.shape == t_cart.shape, 'Incomplete full-pair output')
        ray, embedding, total, residual = decomposition(t_probe, t_chord, t_cart)
        errors = {'ray_only': ray, 'embedding_serialization': embedding, 'total': total}
        np.savez_compressed(folder / 'arrays.npz', **arrays, t_probe_s=t_probe, nseg=nseg,
                            ray_error_s=ray, embedding_serialization_error_s=embedding, total_error_s=total)
        probe_dir, = folder.glob('small_domain_solver_probe_*')
        probe_manifest = json.loads((probe_dir / 'manifest.json').read_text())
        require(probe_manifest['status'] == 'complete'
                and probe_manifest['binary_sha256'] == digest(SNAPSHOT / 'solver/forward_probe'),
                'Probe provenance incomplete')
        for name, checksum in probe_manifest['input_sha256'].items():
            require(digest(folder / name) == digest(probe_dir / name) == checksum, 'Probe input copy changed')
        result = {'geometry': case, 'events': len(sources), 'stations': len(receivers),
                  'pairs': int(t_cart.size), 'prior_run': str(prior),
                  'actual_station_sha256': digest(actual_station),
                  'regenerated_station_sha256': digest(folder / 'regenerated_station.dat'),
                  'station_bytes_identical': True, 'true_event_id_xyz_alignment': True,
                  'serialized_endpoint_type': 'read REAL*4 then promoted REAL*8; station -height/1000 in REAL*4',
                  'metrics': {key: metrics(value) for key, value in errors.items()},
                  'decomposition_max_abs_residual_s': residual, 'decomposition_exact_float64': True,
                  'worst_pairs': {key: worst(value, data, sources, receivers, t_cart, t_chord, t_probe)
                                  for key, value in errors.items()},
                  'lengths_km': {'cartesian': stats(5 * t_cart), 'chord': stats(5 * t_chord),
                                 'probe_equivalent': stats(5 * t_probe)},
                  'segments': stats(nseg),
                  'segment_count_histogram': {str(int(k)): int(v) for k, v in zip(*np.unique(nseg, return_counts=True))},
                  'chord_length_per_segment_km': stats(5 * t_chord / nseg),
                  'detectability_analytic_only': {
                      'wrong_height_minus_correct': metrics(t_wrong_height - t_chord),
                      'reversed_station_association_minus_correct': metrics(t_reversed - t_chord),
                      'height_flip_is_null': bool(np.array_equal(t_wrong_height, t_chord))},
                  'probe_wall_s': probe_elapsed, 'elapsed_s': time.monotonic() - started,
                  'probe_directory': str(probe_dir.relative_to(OUTPUT)), 'limitations': LIMITATION}
        result['criteria_pass'] = {key: passes(result['metrics'][key], key) for key in CRITERIA}
        save(folder / 'metrics.json', result, exclusive=True)
        status.update(status='complete', probe_wall_s=probe_elapsed, criteria_pass=result['criteria_pass'])
        print(f'{case}: complete in {probe_elapsed:.3f}s probe wall time; criteria {result["criteria_pass"]}', flush=True)
        return result
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=now(), elapsed_s=time.monotonic() - started)
        save(folder / 'status.json', status)


def report(results, failures):
    lines = ['# Homogeneous forward-only diagnostic', '',
             'No picks/noise, relocation, inverse solve, build, or original-solver writes. '
             'All TRUE-source × station pairs; homogeneous 5 km/s at all 22×9×9 nodes including padding.', '',
             '## Predeclared criteria (unchanged)', '',
             'Ray-only RMS ≤ 1 ms and max ≤ 5 ms (prior published ray-numeric criteria). '
             'Total RMS ≤ 5 ms and max ≤ 10 ms, relative to 10 ms absolute pick noise.', '',
             '`t_cart`: original supplied Cartesian XYZ Euclidean distance / 5. '
             '`t_chord`: standard-latitude spherical Earth chord / 5, radius 6371 km, '
             'using serialized endpoints consumed by PBR. `t_probe`: frozen production-ray probe.', '',
             '| Geometry | Error (ms) | Bias | RMS | P95 abs | Max abs | Event-demeaned RMS |',
             '|---|---|---:|---:|---:|---:|---:|']
    for case, result in results.items():
        for key, m in result['metrics'].items():
            lines.append(f'| {case} | {key} | {m["signed_bias_ms"]:.6f} | {m["rms_ms"]:.6f} | '
                         f'{m["p95_abs_ms"]:.6f} | {m["max_abs_ms"]:.6f} | {m["per_event_demeaned_rms_ms"]:.6f} |')
    for case, result in results.items():
        lines += ['', f'## {case}', '',
                  f'{result["events"]} × {result["stations"]} = {result["pairs"]} pairs; '
                  f'probe wall {result["probe_wall_s"]:.3f}s; geometry elapsed {result["elapsed_s"]:.3f}s.',
                  f'Criteria: {json.dumps(result["criteria_pass"])}. Exact float64 decomposition residual: '
                  f'{result["decomposition_max_abs_residual_s"]} s.',
                  f'Actual station.dat SHA256: `{result["actual_station_sha256"]}`; '
                  'regenerated bytes identical; station IDs/order and TRUE event ID/XYZ alignment verified.', '',
                  'Lengths (km), segment counts, and chord length / segment (not actual per-segment samples):',
                  '```json', json.dumps({key: result[key] for key in
                                        ('lengths_km', 'segments', 'segment_count_histogram',
                                         'chord_length_per_segment_km')}, indent=2), '```', '',
                  'Analytic-only detectability checks; wrong settings were never passed to the probe:',
                  '```json', json.dumps(result['detectability_analytic_only'], indent=2), '```', '',
                  'Worst pairs (original XYZ, IDs, serialized endpoints and all three times):',
                  '```json', json.dumps(result['worst_pairs'], indent=2), '```']
    lines += ['', '## Scope and provenance', '', LIMITATION, '',
              'Both geometries use the frozen model-centered mapping (x−10 km, y−3.5 km), '
              'synthetic center 30.3°N/103.3°E, 111 km/degree latitude and cosine-scaled longitude. '
              'No original physical geographic coordinates are sent to PBR. '
              'Event latitude/longitude are written to 5 decimals, depth to .3 km, with frozen '
              '`synthetic.write_events`; station endpoints come from actual inversion `station.dat`. '
              'REAL*4 input/sign conversion followed by REAL*8 promotion matches inspected '
              '`getdata_SPDR.f`, `trialsrc_FDD_shot.f`, and `tomoDD-SE.f`. '
              'Probe reads receiver first and calls `pbr(0,receiver,source,...)` with fresh scalars.', '',
              'The snapshot ledger, complete build artifact hashes, patched source hashes, recipe signature, '
              'recorded link command, production Makefile, support files and prior input ledgers '
              'were checked before execution and rechecked afterward. The probe and inversion share '
              'the recorded `RaySPDR2.o`; no fresh relink was performed. '
              '`input_sha256.json` records all read input/source/binary hashes. '
              '`run_sha256.json` covers all final run files except itself and mutable root status. '
              'Probe raw requests/results/manifests/logs and arrays are retained.', '',
              'The homogeneous checks test coordinate transfer/serialization and production ray numerics '
              'only. Passing criteria is not proof of a general absence of bugs or an explanation '
              'of heterogeneous inversion behavior. No production code changes are made.']
    if failures:
        lines += ['', '## Failures (no sampling or threshold changes)', '', '```json', json.dumps(failures, indent=2), '```']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Verify frozen hashes/geometry; no probe')
    args = parser.parse_args(argv)
    check_root(OUTPUT)
    ledger = {'status': 'running', 'started_utc': now(), 'pid': os.getpid(), 'command': sys.argv,
              'python': sys.executable, 'numpy': np.__version__, 'criteria': CRITERIA,
              'scope': 'forward-only; full pairs; fixed TRUE sources; no picks/noise/inversion/build',
              'validate_only': args.validate_only, 'limitations': LIMITATION}
    save(OUTPUT / 'manifest.json', ledger, exclusive=True)
    install_audit_guard()
    started = time.monotonic()
    results, failures = {}, {}
    expected = None
    try:
        cache = OUTPUT / 'cache'
        cache.mkdir(exist_ok=False)
        os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg', XDG_CACHE_HOME=str(cache),
                          NUMBA_CACHE_DIR=str(cache), TMPDIR=str(cache), GFORTRAN_UNBUFFERED_ALL='y',
                          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        expected = provenance()
        save(OUTPUT / 'input_sha256.json', expected, exclusive=True)
        adapter, synthetic, solver = load_runtime()
        if args.validate_only:
            for case in CASES:
                load_geometry(case, adapter)
            ledger['status'] = 'validated'
        else:
            for case in CASES:
                try:
                    results[case] = run_geometry(case, adapter, synthetic, solver)
                except Exception as error:
                    failures[case] = f'{type(error).__name__}: {error}'
                    print(f'{case}: FAILED: {error}', flush=True)
            save(OUTPUT / 'metrics.json', {'criteria': CRITERIA, 'geometries': results, 'failures': failures}, exclusive=True)
            (OUTPUT / 'report.md').write_text(report(results, failures))
            ledger['status'] = 'failed' if failures else 'complete'
            ledger['all_criteria_pass'] = bool(results) and not failures and all(
                all(r['criteria_pass'].values()) for r in results.values())
        verify_hashes(expected)
        ledger['inputs_unchanged_after_run'] = True
    except BaseException as error:
        ledger.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        ledger.update(finished_utc=now(), elapsed_s=time.monotonic() - started, failures=failures)
        save(OUTPUT / 'manifest.json', ledger)
        run_hashes = {str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob('*'))
                      if p.is_file() and p.name not in ('run_sha256.json', 'manifest.json')}
        # Include immutable child probe manifests, excluding only the mutable root ledger.
        for p in OUTPUT.rglob('manifest.json'):
            if p != OUTPUT / 'manifest.json':
                run_hashes[str(p.relative_to(OUTPUT))] = digest(p)
        save(OUTPUT / 'run_sha256.json', run_hashes, exclusive=True)
    print(json.dumps({'status': ledger['status'], 'elapsed_s': ledger['elapsed_s'],
                      'all_criteria_pass': ledger.get('all_criteria_pass')}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
