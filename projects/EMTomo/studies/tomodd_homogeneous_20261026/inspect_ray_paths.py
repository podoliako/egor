"""Private path-dump driver linked to unchanged frozen ray objects; input-only P=5 control.

Run with the existing venv Python and -B. All generated artifacts stay in
runs/tomodd_homogeneous_20261026/path_inspection/. Commands/probes time out at 120 s.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
REPRO = OLD / 'pair_reproduction'
OUTPUT = OLD / 'path_inspection'
SNAPSHOT = PROJECT / 'runs/matched_geometry_20261021/snapshot'
BUILD = SNAPSHOT / 'solver'
PRIVATE = OUTPUT / 'build/forward_probe_paths'
FC = '/usr/bin/x86_64-linux-gnu-gfortran-13'
TIMEOUT = 120
LABELS = ('geysers_outlier', 'geysers_maxpositive', 'artificial_maxpositive')


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
        require(digest(name) == checksum, f'Input/sealed artifact changed: {name}')


def check_write(path):
    if not isinstance(path, int):
        require(Path(os.fsdecode(path)).resolve().is_relative_to(OUTPUT.resolve()),
                f'Write outside path_inspection: {path}')


def install_guard():
    def audit(event, args):
        if event == 'open':
            if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check_write(args[0])
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
            check_write(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check_write(args[0])
            check_write(args[1])
        elif event == 'subprocess.Popen':
            executable, argv, cwd, _ = args
            resolved = Path(executable).resolve()
            allowed = {Path(FC).resolve(), PRIVATE.resolve(), (BUILD / 'forward_probe').resolve()}
            if resolved not in allowed:
                raise PermissionError('Only compiler, private driver and frozen forward probe may execute')
            check_write(cwd)
            if resolved == Path(FC).resolve():
                require('-c' not in argv or str(OUTPUT / 'build/forward_probe_paths.f') in argv,
                        'Only the private driver source may be compiled')
                require(not any(str(arg).endswith('RaySPDR2.f') for arg in argv), 'Ray core rebuild forbidden')
        elif event in ('os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
            raise PermissionError(f'Unapproved process launch: {event}')
    sys.addaudithook(audit)


def input_hashes():
    expected = json.loads((REPRO / 'input_sha256.json').read_text())
    for name, checksum in json.loads((REPRO / 'output_sha256.json').read_text()).items():
        expected[str(REPRO / name)] = checksum
    expected[str(REPRO / 'output_sha256.json')] = digest(REPRO / 'output_sha256.json')
    expected[str(Path(__file__).resolve())] = digest(__file__)
    build = json.loads((BUILD / 'manifest.json').read_text())
    require(build['recipe']['tools']['fc']['path'] == FC, 'Compiler path differs from frozen build')
    expected[FC] = build['recipe']['tools']['fc']['sha256']
    for name in ('RaySPDR2.o', 'transform_r_gfortran.o', 'forward_probe', 'forward_probe.o'):
        require(expected[str(BUILD / name)] == build['artifact_sha256'][name], 'Conflicting object provenance')
    verify(expected)
    return expected


def load_runtime():
    path = HERE / 'run_check.py'
    spec = importlib.util.spec_from_file_location('path_inspection_sealed_check', path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    check.source_contract()
    _, _, solver = check.load_runtime()
    return check, solver


def extend_source(original):
    declaration = '      integer i,j,n,ni,ios\n'
    extra = '''      integer dumpk,dump_nbl,dump_reg
      real*8 dump_la,dump_lo,dump_r,dump_v,dump_depth
      real*8 dump_dpi,dump_r2d
'''
    require(original.count(declaration) == 1 and original.count('         flush(20)\n') == 1,
            'Unexpected frozen driver layout')
    source = original.replace(declaration, declaration + extra)
    source = source.replace("      open(20,file='forward.tsv',status='replace')\n", """      open(21,file='path_nodes.tsv',status='replace')
      write(21,'(a)') '# pair node radius_km colat_deg lon_deg Vp_km_s cell_lat_deg cell_lon_deg clamped_depth_km nbl regional'
      open(22,file='guard.tsv',status='replace')
      write(22,'(a)') '# lon_lower lon_upper lat_lower lat_upper depth_upper dpi r2d'
      dump_dpi=asin(1.)/90.
      dump_r2d=90./asin(1.)
      write(22,'(7(es25.17e3,1x))')dble(xn(1)+0.1),dble(xn(nx)-0.1),dble(yn(1)+0.1),dble(yn(ny)-0.1),dble(zn(nz)-1),dump_dpi,dump_r2d
      close(22)
      open(20,file='forward.tsv',status='replace')
""")
    source = source.replace('         flush(20)\n', '''         flush(20)
C        Observe only after pbr; cellvel mutates latitude/longitude copies.
         do dumpk=1,ni+1
            dump_la=w(2,dumpk)
            dump_lo=w(3,dumpk)
            dump_r=w(1,dumpk)
            call cellvel(0,dump_la,dump_lo,dump_r,dump_v,dump_nbl)
            dump_depth=max(0.0d0,6371.0d0-dump_r)
            dump_reg=0
            if(dump_lo.gt.xn(1)+0.1.and.dump_lo.lt.xn(nx)-0.1.and.dump_la.gt.yn(1)+0.1.and.dump_la.lt.yn(ny)-0.1.and.dump_depth.lt.zn(nz)-1)dump_reg=1
            write(21,'(2(i10,1x),7(es25.17e3,1x),2(i12,1x))')j,dumpk,w(1,dumpk),w(2,dumpk),w(3,dumpk),dump_v,dump_la,dump_lo,dump_depth,dump_nbl,dump_reg
         enddo
         flush(21)
''')
    return source.replace('      close(20)\n', '      close(20)\n      close(21)\n')


def compile_driver(solver):
    folder = OUTPUT / 'build'
    folder.mkdir(exist_ok=False)
    original = (BUILD / 'forward_probe.f').read_text()
    require(original == solver.PROBE_SOURCE, 'Frozen driver differs from frozen PROBE_SOURCE')
    shutil.copyfile(BUILD / 'forward_probe.f', folder / 'original_forward_probe.f')
    source = folder / 'forward_probe_paths.f'
    source.write_text(extend_source(original))
    obj = folder / 'forward_probe_paths.o'
    flags = ['-O3', '-std=legacy', '-fallow-argument-mismatch', '-fno-automatic',
             '-ffixed-line-length-none', '-m64', '-mcmodel=medium', f'-I{BUILD / "include"}']
    frozen = json.loads((BUILD / 'manifest.json').read_text())
    require(flags[:-1] == frozen['recipe']['fflags'][:-1], 'Compile flags differ from frozen driver')
    commands = [[FC, *flags, '-c', str(source), '-o', str(obj)],
                [FC, '-O3', '-m64', '-mcmodel=medium', str(obj), str(BUILD / 'RaySPDR2.o'),
                 str(BUILD / 'transform_r_gfortran.o'), '-lm', '-o', str(PRIVATE)]]
    save(folder / 'commands.json', {'commands': commands, 'timeout_s_each': TIMEOUT}, exclusive=True)
    started = time.monotonic()
    for index, command in enumerate(commands):
        with (folder / f'command_{index + 1}.log').open('x') as log:
            subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=TIMEOUT)
    manifest = {'status': 'complete', 'driver_only_compiled': True, 'commands': commands,
                'elapsed_s': time.monotonic() - started,
                'original_source_sha256': digest(BUILD / 'forward_probe.f'),
                'extended_source_sha256': digest(source), 'private_object_sha256': digest(obj),
                'private_binary_sha256': digest(PRIVATE),
                'frozen_linked_objects_sha256': {name: digest(BUILD / name)
                                                for name in ('RaySPDR2.o', 'transform_r_gfortran.o')}}
    save(folder / 'manifest.json', manifest, exclusive=True)
    return manifest


def copy_inputs(folder, geometry):
    folder.mkdir(exist_ok=False)
    for name in ('MOD', 'ak135.15.SKS', 'layer-16.dat'):
        shutil.copyfile(OLD / geometry / name, folder / name)
    lines = (folder / 'MOD').read_text().splitlines()
    require(tuple(map(int, lines[0].split()[1:])) == (22, 9, 9), 'Wrong regional MOD dimensions')
    require(np.all(np.array([list(map(float, row.split())) for row in lines[4:85]]) == 5.),
            'Regional MOD is not homogeneous including padding')


def run_driver(label, target):
    folder = OUTPUT / 'original_paths' / label
    copy_inputs(folder, target['geometry'])
    request = '1\n' + ' '.join(format(float(x), '.17g') for x in
                               (*target['receiver_lat_lon_depth_km'], *target['source_lat_lon_depth_km'])) + '\n'
    (folder / 'coordinates.txt').write_text(request)
    manifest = {'status': 'running', 'binary': str(PRIVATE), 'binary_sha256': digest(PRIVATE),
                'timeout_s': TIMEOUT, 'target': label,
                'input_sha256': {name: digest(folder / name) for name in
                                  ('MOD', 'ak135.15.SKS', 'layer-16.dat', 'coordinates.txt')}}
    save(folder / 'manifest.json', manifest, exclusive=True)
    started = time.monotonic()
    try:
        with (folder / 'probe.log').open('x') as log:
            subprocess.run([str(PRIVATE)], cwd=folder, input=request, text=True,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=TIMEOUT)
        rows = np.loadtxt(folder / 'forward.tsv', ndmin=2)
        require(rows.shape == (1, 3) and rows[0, 0] == 1, 'Malformed/incomplete private driver result')
        measured, nseg = float(rows[0, 1]), int(rows[0, 2])
        require(measured == target['original_t_probe_s'] and nseg == target['original_nseg'],
                'Private driver does not exactly reproduce original frozen time/nseg; interpretation forbidden')
        nodes = np.loadtxt(folder / 'path_nodes.tsv', ndmin=2)
        require(nodes.shape == (nseg + 1, 11) and np.isfinite(nodes).all(), 'Invalid path node dump')
        np.testing.assert_array_equal(nodes[:, 1], np.arange(1, nseg + 2))
        np.testing.assert_array_equal(nodes[:, 0], np.ones(nseg + 1))
        require(np.all(nodes[:, 5] > 0), 'Nonpositive dumped node velocity')
        manifest.update(status='complete', exact_original_time_and_nseg=True, time_s=measured, nseg=nseg)
        return folder, nodes, np.loadtxt(folder / 'guard.tsv'), measured, nseg
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - started
        save(folder / 'manifest.json', manifest)


def xyz_from_nodes(nodes, degrees_per_radian):
    radius = nodes[:, 2]
    colat = nodes[:, 3] / degrees_per_radian
    lon = nodes[:, 4] / degrees_per_radian
    return radius[:, None] * np.column_stack((np.sin(colat) * np.cos(lon),
                                             np.sin(colat) * np.sin(lon), np.cos(colat)))


def departures(xyz):
    start, stop = xyz[0], xyz[-1]
    direction = stop - start
    fractions = np.clip((xyz - start) @ direction / (direction @ direction), 0., 1.)
    nearest = start + fractions[:, None] * direction
    return np.linalg.norm(xyz - nearest, axis=1)


def analyze_path(target, path):
    folder, nodes, guard, measured, nseg = path
    lon_min, lon_max, lat_min, lat_max, depth_max, dpi, r2d = guard
    regional = ((nodes[:, 7] > lon_min) & (nodes[:, 7] < lon_max)
                & (nodes[:, 6] > lat_min) & (nodes[:, 6] < lat_max) & (nodes[:, 8] < depth_max))
    np.testing.assert_array_equal(regional.astype(int), nodes[:, 10].astype(int))
    # Invert the actual Fortran output scaling to recover internal angular distances.
    # Absolute longitude rotation is immaterial to polyline lengths/departures.
    xyz = xyz_from_nodes(nodes, r2d)
    xyz_standard = xyz_from_nodes(nodes, 180. / np.pi)
    lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    standard_lengths = np.linalg.norm(np.diff(xyz_standard, axis=0), axis=1)
    speeds = nodes[:, 5]
    segment_times = lengths * .5 * (1. / speeds[:-1] + 1. / speeds[1:])
    t_poly = float(segment_times.sum())
    outside = ~regional
    outside_segments = outside[:-1] | outside[1:]
    excess_over_v5 = float(np.sum(lengths / 5. - segment_times))
    deviation = departures(xyz)
    np.savez_compressed(folder / 'path_arrays.npz', raw_nodes=nodes, guard=guard, xyz_km=xyz,
                        xyz_standard_degrees_km=xyz_standard, segment_lengths_km=lengths,
                        segment_times_s=segment_times, departure_from_returned_endpoint_chord_km=deviation,
                        node_regional=regional)
    original_support = np.loadtxt(folder / 'ak135.15.SKS')
    global_speeds = np.unique(original_support[:, 2])
    outside_speeds = np.unique(speeds[outside])
    require(all(np.any(np.isclose(v, global_speeds, rtol=0, atol=1e-12)) for v in outside_speeds),
            'Outside-node speed does not match AK135 support')
    return {'target': target['label'], 'event_id': target['event_id'], 'station_id': target['station_id'],
            'source_xyz_km': target['source_xyz_km'], 'receiver_xyz_km': target['receiver_xyz_km'],
            'frozen_time_exactly_reproduced': True, 'frozen_tt_s': measured, 'nseg': nseg,
            't_chord_s': target['t_chord_s'], 'ray_error_ms': (measured - target['t_chord_s']) * 1000,
            'path_polyline_length_km': float(lengths.sum()),
            'returned_endpoint_chord_km': float(np.linalg.norm(xyz[-1] - xyz[0])),
            'path_excess_length_km': float(lengths.sum() - np.linalg.norm(xyz[-1] - xyz[0])),
            'max_departure_from_returned_endpoint_chord_km': float(deviation.max()),
            'max_departure_node': int(np.argmax(deviation)) + 1,
            'standard_degree_polyline_length_km': float(standard_lengths.sum()),
            'fortran_dpi': float(dpi), 'fortran_r2d': float(r2d),
            'polyline_reconstruction': 'Invert dumped Fortran r2d; straight chord uses returned endpoints',
            't_poly_trapezoid_s': t_poly, 't_poly_minus_frozen_tt_ms': (t_poly - measured) * 1000,
            't_poly_rounded_float32_matches_frozen_tt': bool(float(np.float32(t_poly)) == measured),
            't_poly_v5_s': float(lengths.sum() / 5.),
            'node_slowness_saving_vs_v5_ms': excess_over_v5 * 1000,
            'node_count': nseg + 1, 'regional_nodes': int(regional.sum()), 'fallback_nodes': int(outside.sum()),
            'fallback_node_indices_1based': (np.flatnonzero(outside) + 1).tolist(),
            'fallback_touching_segment_count': int(outside_segments.sum()),
            'fallback_touching_segment_length_km': float(lengths[outside_segments].sum()),
            'node_speed_min_km_s': float(speeds.min()), 'node_speed_max_km_s': float(speeds.max()),
            'node_speeds_unique_km_s': np.unique(speeds).tolist(),
            'outside_node_speeds_unique_km_s': outside_speeds.tolist(),
            'region_guard': dict(zip(('lon_lower', 'lon_upper', 'lat_lower', 'lat_upper', 'depth_upper'),
                                    map(float, guard[:5]))),
            'final_node_region_does_not_trace_gradient_stencil': True}


def p5_support(text):
    lines = []
    for line in text.splitlines():
        parts = line.split()
        require(len(parts) == 5, 'Unexpected AK135 table format')
        parts[2] = '5.0000'
        lines.append(' '.join(parts))
    result = '\n'.join(lines) + '\n'
    before = np.loadtxt(text.splitlines())
    after = np.loadtxt(result.splitlines())
    np.testing.assert_array_equal(before[:, [0, 1, 3, 4]], after[:, [0, 1, 3, 4]])
    require(np.all(after[:, 2] == 5.), 'P=5 table construction failed')
    return result


def run_control(label, target, solver):
    folder = OUTPUT / 'global_p5_control' / label
    copy_inputs(folder, target['geometry'])
    path = folder / 'ak135.15.SKS'
    original_hash = digest(path)
    path.write_text(p5_support(path.read_text()))
    changed_hash = digest(path)
    require(changed_hash != original_hash, 'Support intervention did not alter the table')
    save(folder / 'intervention.json', {'diagnostic_input_only_not_production': True,
                                      'modified': 'All AK135 P velocities set to 5 km/s; all other columns unchanged',
                                      'original_support_sha256': original_hash,
                                      'altered_support_sha256': changed_hash,
                                      'regional_MOD_sha256': digest(folder / 'MOD')}, exclusive=True)
    started = time.monotonic()
    times, segments = solver.forward_times(BUILD, folder, [target['source_lat_lon_depth_km']],
                                          [target['receiver_lat_lon_depth_km']], timeout=TIMEOUT)
    measured, nseg = float(times[0, 0]), int(segments[0, 0])
    result = {'target': label, 'original_unchanged_frozen_probe': True, 'time_s': measured, 'nseg': nseg,
              't_chord_s': target['t_chord_s'], 'ray_error_ms': (measured - target['t_chord_s']) * 1000,
              'change_from_original_ms': (measured - target['original_t_probe_s']) * 1000,
              'elapsed_s': time.monotonic() - started, 'altered_support_sha256': changed_hash,
              'only_support_P_column_changed': True}
    save(folder / 'results.json', result, exclusive=True)
    return result


def self_checks():
    xyz = np.array([[0., 0., 0.], [1., 2., 0.], [2., 0., 0.]])
    np.testing.assert_array_equal(departures(xyz), [0., 2., 0.])
    trial = p5_support('6371 0 5.8 3.46 2.72\n6356 15 5.8 3.46 2.72\n')
    require(np.all(np.loadtxt(trial.splitlines())[:, 2] == 5.), 'Support self-test failed')


def render(paths, controls, build_manifest):
    lines = ['# Frozen-core private path inspection', '',
             'All three private-driver times and segment counts exactly match the sealed frozen-probe '
             'targets before path interpretation. Only the driver was compiled; the exact frozen '
             '`RaySPDR2.o` and `transform_r_gfortran.o` were linked unchanged.', '',
             '| Target | Frozen tt (s) | nseg | Polyline (km) | Max chord departure (km) | Fallback nodes | Speed min/max (km/s) | Poly trapezoid − tt (ms) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        p = paths[label]
        lines.append(f'| {label} | {p["frozen_tt_s"]:.12f} | {p["nseg"]} | {p["path_polyline_length_km"]:.9f} | '
                     f'{p["max_departure_from_returned_endpoint_chord_km"]:.9f} | '
                     f'{p["fallback_nodes"]}/{p["node_count"]} | {p["node_speed_min_km_s"]:.9f}/{p["node_speed_max_km_s"]:.9f} | '
                     f'{p["t_poly_minus_frozen_tt_ms"]:+.9f} |')
    lines += ['', '## Input-only globally P=5 control', '',
              'Only the AK135 support P-speed column was changed, in new diagnostic directories. '
              'Regional MOD remains exactly the completed diagnostic MOD; Vs, radius, depth and density '
              'columns are unchanged. Controls execute the original unchanged frozen forward_probe.', '',
              '| Target | Original ray error (ms) | Control time (s) | Control ray error (ms) | Control nseg | Δ original time (ms) |',
              '|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        p, c = paths[label], controls[label]
        lines.append(f'| {label} | {p["ray_error_ms"]:+.6f} | {c["time_s"]:.12f} | '
                     f'{c["ray_error_ms"]:+.6f} | {c["nseg"]} | {c["change_from_original_ms"]:+.6f} |')
    lines += ['', '## Quantitative path evidence', '']
    for label in LABELS:
        p = paths[label]
        lines += [f'### {label}', '', f'Source XYZ km: `{p["source_xyz_km"]}`; receiver XYZ km: `{p["receiver_xyz_km"]}`.',
                  f'Fallback node indices (1-based): `{p["fallback_node_indices_1based"]}`; '
                  f'fallback speeds km/s: `{p["outside_node_speeds_unique_km_s"]}`.',
                  f'Segments touching fallback nodes: {p["fallback_touching_segment_count"]}, '
                  f'combined length {p["fallback_touching_segment_length_km"]:.9f} km. '
                  f'Trapezoidal slowness saving versus 5 km/s on the same polyline: '
                  f'{p["node_slowness_saving_vs_v5_ms"]:.6f} ms.',
                  f'Polyline at constant 5 km/s: {p["t_poly_v5_s"]:.12f} s; '
                  f'node-speed trapezoid: {p["t_poly_trapezoid_s"]:.12f} s; '
                  f'float32 rounding matches frozen tt: {p["t_poly_rounded_float32_matches_frozen_tt"]}.', '']
    lines += ['## Interpretation limits and artifacts', '',
              'Node regional flags use the post-cellvel normalized longitude, corrected latitude and '
              'clamped depth, with the exact REAL*4 grid-guard expressions dumped by the driver '
              '(strict 0.1-degree horizontal guard and bottom zn(nz)-1). Cellvel receives separate '
              'scalar copies, so its coordinate mutation does not change returned w nodes. '
              'Node dumps are post-pbr observations, not a trace of all gradient-stencil evaluations.', '',
              'Polyline calculations invert the actual Fortran r2d value dumped by the driver; the '
              'alternative standard-degree reconstruction is also retained. Chord departure is the '
              'maximum node distance to the straight segment connecting returned path endpoints. '
              'Reported trapezoidal times use the dumped velocities, not a new ray solve.', '',
              'A non-5 final-node speed with a false regional flag demonstrates support-model fallback '
              'on the returned ray. The P=5 intervention tests dependence on support P velocities '
              'without changing the numerical core. If positive errors persist under the globally P=5 '
              'control, this test does not localize their remaining cause. In particular, it does not '
              'prove any specific precision/gradient instability or identify a production fix.', '',
              'All raw requests/results, float64 node/velocity/nbl dumps, NPZ path arrays, commands, '
              'compile/link logs, private source/object/binary, copied inputs, altered-support hashes '
              'and provenance are retained. No frozen/production source, old sealed output, article '
              'or inversion was changed.', '',
              f'Private compile/link elapsed: {build_manifest["elapsed_s"]:.6f} s. Each command/probe timeout: {TIMEOUT} s.']
    return '\n'.join(lines) + '\n'


def main():
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and not any(OUTPUT.iterdir()),
            'path_inspection must be precreated, empty and not a symlink')
    ledger = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, '-B', *sys.argv], 'timeout_s_each_command': TIMEOUT,
              'numerical_core_rebuilt_or_modified': False, 'python': sys.version, 'numpy': np.__version__}
    save(OUTPUT / 'manifest.json', ledger, exclusive=True)
    install_guard()
    started = time.monotonic()
    try:
        cache = OUTPUT / 'cache'
        cache.mkdir(exist_ok=False)
        os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg', XDG_CACHE_HOME=str(cache),
                          TMPDIR=str(cache), NUMBA_CACHE_DIR=str(cache), GFORTRAN_UNBUFFERED_ALL='y',
                          OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        self_checks()
        expected = input_hashes()
        save(OUTPUT / 'input_sha256.json', expected, exclusive=True)
        check, solver = load_runtime()
        targets_all = json.loads((REPRO / 'targets.json').read_text())
        targets = {label: targets_all[label] for label in LABELS}
        save(OUTPUT / 'targets.json', targets, exclusive=True)
        build_manifest = compile_driver(solver)
        (OUTPUT / 'original_paths').mkdir(exist_ok=False)
        raw_paths = {label: run_driver(label, targets[label]) for label in LABELS}
        # Gate ALL path interpretation on exact reproduction of ALL three targets.
        ledger['all_three_private_times_and_nseg_exact'] = True
        save(OUTPUT / 'manifest.json', ledger)
        paths = {label: analyze_path(targets[label], raw_paths[label]) for label in LABELS}
        (OUTPUT / 'global_p5_control').mkdir(exist_ok=False)
        controls = {label: run_control(label, targets[label], solver) for label in LABELS}
        save(OUTPUT / 'results.json', {'paths': paths, 'global_p5_controls': controls,
                                     'all_three_private_times_and_nseg_exact': True}, exclusive=True)
        (OUTPUT / 'report.md').write_text(render(paths, controls, build_manifest))
        verify(expected)
        ledger.update(status='complete', verified_input_and_old_files=len(expected), all_old_hashes_unchanged=True)
        for label in LABELS:
            p, c = paths[label], controls[label]
            print(f'{label}: exact original tt/nseg; poly={p["path_polyline_length_km"]:.9f}km '
                  f'departure={p["max_departure_from_returned_endpoint_chord_km"]:.9f}km '
                  f'fallback={p["fallback_nodes"]}/{p["node_count"]} '
                  f'V={p["node_speed_min_km_s"]:.9f}..{p["node_speed_max_km_s"]:.9f} '
                  f'global-P5 error={c["ray_error_ms"]:+.6f}ms nseg={c["nseg"]}', flush=True)
    except BaseException as error:
        ledger.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        ledger.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save(OUTPUT / 'manifest.json', ledger)
        hashes = {str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
        save(OUTPUT / 'output_sha256.json', hashes, exclusive=True)
    print(json.dumps({'status': ledger['status'], 'elapsed_s': ledger['elapsed_s'],
                      'all_old_hashes_unchanged': ledger['all_old_hashes_unchanged']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
