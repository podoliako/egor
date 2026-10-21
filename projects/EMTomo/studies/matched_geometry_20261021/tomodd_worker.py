
"""Foreground worker for the two matched-geometry tomoDD runs; never builds/picks.

--benchmark-root is the SNAPSHOT directory containing tomodd_benchmark/, not
that package directory itself. Python >=3.11, numpy, scipy and matplotlib are
required. --validate-only reads inputs and constructs the initial graph but
never creates --output or executes a solver. Matplotlib's temporary cache is
confined to this study and removed on exit; Python bytecode writes are disabled.
Normal execution exclusively claims a nonexistent output, retains the adapter's
config/results/truth/state and work/inversion history, and adds worker_info.json.
No timeout is imposed (the old 'nolimit' protocol); supervise the process group
externally if a wall-time limit is needed. No background processes are launched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

LIVE_ROOT = Path('/home/egor/tomodd_benchmark')
DEFAULT_BUILD = LIVE_ROOT / 'data/small_domain_solver/mins_0p05_2d6bca5eb5b1'
DEFAULT_SUPPORT = Path('/home/egor/tomodd_pipeline/USTC-tomoDD-SE-training2020/Example/RealData-Inversion')
BINARY_SHA256 = '33f41e0eddb2d6f82b860331f2f1803cb4b240f4671826610ceb948fef58c3f4'
BUILD_SIGNATURE = '2d6bca5eb5b1891a004fb0a3ec80951e767660bdcaf6c63554344a1ff6b200dd'
EXPERIMENTS = {
    'synthetic': ('flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001', 1920, 52),
    'geysers': ('geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001', 3776, 26),
}
BOUNDARIES = (0.7, 1.9, 3.4)
VELOCITIES = (3800.0, 4600.0, 5200.0, 5700.0)
SCHEDULE = ((3, 0.1, 0.08, -9.0, 10.0, 40.0, 1, 0.0),)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(UTC).isoformat(timespec='seconds')


def new_output(path):
    # Check before resolving, so even a dangling final symlink is rejected.
    require(not path.exists() and not path.is_symlink(), f'Output already exists: {path}')
    path = path.resolve()
    require(path.parent.is_dir(), f'Output parent must already exist: {path.parent}')
    require(os.access(path.parent, os.W_OK | os.X_OK), f'Output parent is not writable: {path.parent}')
    return path


def load_adapter(root):
    """Bind the package explicitly; never fall through to PYTHONPATH/live imports."""
    root = root.resolve(strict=True)
    require(root != LIVE_ROOT.resolve(), '--benchmark-root must be a snapshot, not the live tree')
    package = root / 'tomodd_benchmark'
    init = package / '__init__.py'
    require(init.is_file(), f'Expected snapshot package: {init}')
    for path in package.rglob('*.py'):
        require(path.resolve().is_relative_to(root), f'Snapshot source escapes its root: {path}')
    require(not any(name == 'tomodd_benchmark' or name.startswith('tomodd_benchmark.')
                    for name in sys.modules), 'tomoDD package already imported; use a fresh worker process')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('tomodd_benchmark', init,
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from tomodd_benchmark import external_checkerboard as adapter
    for name, loaded in tuple(sys.modules.items()):
        if name == 'tomodd_benchmark' or name.startswith('tomodd_benchmark.'):
            require(Path(loaded.__file__).resolve().is_relative_to(root),
                    f'Imported non-snapshot module: {name}: {loaded.__file__}')
    return adapter


def verify_build(build):
    manifest_path = build / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    recipe = manifest['recipe']
    require(manifest['status'] == 'complete' and manifest['signature'] == BUILD_SIGNATURE,
            'Expected completed mins_0p05_2d6bca5eb5b1 build')
    # Match build_solver's signature algorithm without calling its build API.
    require(hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest() == BUILD_SIGNATURE,
            'Build recipe signature mismatch')
    require(recipe['min_segment_km'] == 0.05 and recipe['coordinate_roundoff_fix'] is True
            and recipe['max_events'] == 4000
            and recipe['ray_type_flags'] == ['-cpp', '-Dmins=ray_min_segment_km'],
            'Unexpected solver recipe')
    require(platform.system() == 'Linux' and platform.machine() == recipe['machine'],
            'Existing solver requires Linux x86_64; it cannot run on macOS/ARM')
    expected = {
        'tomoDD-SE': BINARY_SHA256,
        'RaySPDR2.f': recipe['patched_ray_sha256'],
        'transform_r_gfortran.c': recipe['patched_transform_sha256'],
        'include/tomoFDD.inc': recipe['patched_include_sha256'],
    }
    require(manifest['artifact_sha256']['tomoDD-SE'] == BINARY_SHA256,
            'Unexpected binary hash in build manifest')
    for name, checksum in expected.items():
        require(digest(build / name) == checksum, f'Modified solver build file: {build / name}')
    binary = build / 'tomoDD-SE'
    require(os.access(binary, os.X_OK), f'Solver is not executable: {binary}')
    return binary, manifest_path


def run_arguments(args, binary):
    return {
        'forward_root': args.forward_root, 'experiment_id': EXPERIMENTS[args.geometry][0],
        'run_dir': args.output, 'iterations': 3, 'timeout_s': 0,
        'max_pair_distance_km': 3.0, 'max_pair_degree': 4, 'node_spacing_km': 1.0,
        'free_buffer_layers': 0, 'initial_location_sigma_km': 1.0,
        'initial_location_seed': 42, 'initial_gradient_vp_m_s': None,
        'initial_layer_boundaries_km': BOUNDARIES,
        'initial_layer_velocities_m_s': VELOCITIES, 'binary': binary, 'ray_step_km': 0.05,
    }


def preflight(args, adapter):
    import numpy as np

    binary, manifest_path = verify_build(args.solver_build)
    # config.SUPPORT_DIR is derived from the package's sibling pipeline, which
    # does not exist in a code-only snapshot. Override only this process's adapter.
    adapter.SUPPORT_DIR = args.support_root
    support = {name: digest(args.support_root / name) for name in adapter.SUPPORT_FILES}
    solver = adapter._solver_config(3, 1.0)
    require(solver.iteration_sets == SCHEDULE and solver.smoothing_lon_lat_depth == (5.0, 5.0, 5.0),
            'Snapshot solver defaults differ from the verified old protocol')
    experiment_id, event_count, station_count = EXPERIMENTS[args.geometry]
    input_dir = args.forward_root / 'experiments/input' / experiment_id
    picks_dir = args.forward_root / 'experiments/output' / experiment_id
    files = [input_dir / name for name in ('model.npz', 'stations.csv', 'events.csv', 'generation.json')]
    files += [picks_dir / name for name in ('arrivals.csv', 'metadata.json')]
    hashes = {str(path): digest(path) for path in files}
    data = adapter.load_external_data(args.forward_root, experiment_id)
    require((len(data.event_ids), len(data.station_ids)) == (event_count, station_count),
            'Unexpected event/station counts')
    require(data.velocity_km_s.shape == (160, 56, 56) and data.cell_size_km == 0.125
            and np.array_equal(data.origin_km, [0, 0, 0]), 'Unexpected truth grid/datum')
    for values in (data.velocity_km_s, data.event_xyz_km, data.station_xyz_km, data.arrivals_s):
        require(np.isfinite(values).all(), 'Nonfinite input values')
    require((data.velocity_km_s > 0).all(), 'Nonpositive truth velocities')
    for name, checksum in data.metadata['input_sha256'].items():
        require(hashes.get(str(input_dir / name)) == checksum, f'Forward input hash mismatch: {name}')
    adapter._numeric_event_ids(data.event_ids)
    grid, _ = adapter.build_tomodd_geometry(
        data, 1.0, 0, None, BOUNDARIES, tuple(v / 1000 for v in VELOCITIES))
    require(grid.initial_vp.shape == (9, 9, 22), 'Unexpected padded solver grid')
    initial, offsets = adapter.perturb_initial_event_xyz(data.event_xyz_km, data.lengths_km, 1.0, 42)
    pairs = adapter.select_event_pairs(initial, 3.0, 4)
    absolute = data.arrivals_s.size
    differential = len(pairs) * station_count
    require(event_count <= 4000 and station_count <= 60
            and absolute + differential <= min(adapter.MAX_DATA, 1100000), 'Solver capacity exceeded')
    arguments = run_arguments(args, binary)
    return arguments, {
        'geometry': args.geometry, 'experiment_id': experiment_id,
        'benchmark_root': str(args.benchmark_root), 'output': str(args.output),
        'adapter': str(Path(adapter.__file__).resolve()),
        'snapshot_source_sha256': {str(p.relative_to(args.benchmark_root)): digest(p)
                                   for p in sorted((args.benchmark_root / 'tomodd_benchmark').rglob('*.py'))},
        'worker_sha256': digest(Path(__file__)),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()},
        'solver_schedule': SCHEDULE, 'smoothing_lon_lat_depth': (5.0, 5.0, 5.0),
        'absolute_dd_weight': (1.0, 0.1), 'timeout_s': None,
        'solver_build_manifest': str(manifest_path), 'solver_manifest_sha256': digest(manifest_path),
        'solver_sha256': BINARY_SHA256, 'support_root': str(args.support_root),
        'support_sha256': support, 'input_sha256': hashes,
        'events': event_count, 'stations': station_count, 'event_pairs': len(pairs),
        'absolute_observations': int(absolute), 'differential_observations': differential,
        'initial_coordinate_rms_km': np.sqrt(np.mean(offsets ** 2, axis=0)).tolist(),
        'boundary_handling': 'coordinate-wise rejection resampling inside the physical volume',
        'python': sys.version, 'numpy': np.__version__,
    }


def check_completed(output, adapter, metrics):
    work = output / 'work/inversion'
    log = (work / 'tomoDD-SE.log').read_text(errors='replace')
    history = (work / 'tomoDD.vel').read_text(errors='replace')
    ids = adapter.parse_relocated_events(work / 'tomoDD.reloc')[:, 0]
    require(metrics['final_retained_event_count'] == metrics['event_count']
            and not metrics['unrelocated_event_ids'], 'Incomplete final event catalog')
    artifacts = []
    for step in range(1, 4):
        require(re.search(rf'\bIteration\s+{step}\s+finished\b', log), f'Iteration {step} not completed')
        require(len(re.findall(rf'P-wave velocity at iteration:\s*{step}\s*$', history, re.MULTILINE)) == 1,
                f'Missing/duplicate velocity snapshot {step}')
        path = work / f'tomoDD.reloc.001.{step:03d}'
        rows = adapter.parse_relocated_events(path)
        require(len(rows) == len(ids) and set(rows[:, 0]) == set(ids),
                f'Incomplete relocation snapshot: {path}')
        artifacts.append(str(path.relative_to(output)))
    return artifacts + ['work/inversion/tomoDD.vel', 'work/inversion/Vp_model.dat',
                            'work/inversion/tomoDD-SE.log']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=EXPERIMENTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--forward-root', type=Path, required=True)
    parser.add_argument('--solver-build', type=Path, default=DEFAULT_BUILD,
                        help='Existing verified build, or an unchanged copy; never rebuilt')
    parser.add_argument('--support-root', type=Path, default=DEFAULT_SUPPORT,
                        help='Read-only directory with ak135.15.SKS and layer-16.dat')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    args.output = new_output(args.output)
    for name in ('benchmark_root', 'forward_root', 'solver_build', 'support_root'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    for root in (args.benchmark_root, args.forward_root, args.solver_build, args.support_root):
        require(not args.output.is_relative_to(root), f'Output must be outside input trees: {root}')
    # pyplot/font discovery otherwise writes to ~/.cache or /tmp. This is the
    # only transient write during validate-only; the requested output stays absent.
    with tempfile.TemporaryDirectory(prefix='.tomodd-cache-', dir=Path(__file__).resolve().parent) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'y'
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = '1'
        adapter = load_adapter(args.benchmark_root)
        arguments, info = preflight(args, adapter)
        if args.validate_only:
            print(json.dumps(dict(info, state='validated', solver_launched=False), indent=2))
            return 0
        # Atomic exclusive claim also rejects an empty directory created during
        # preflight. The adapter accepts this empty directory and fills it itself.
        args.output.mkdir(exist_ok=False)
        started = time.monotonic()
        info.update(state='running', started_utc=now(), worker_pid=os.getpid())
        print(json.dumps(info), flush=True)
        try:
            metrics = adapter.run(**arguments)
            info['iteration_artifacts'] = check_completed(args.output, adapter, metrics)
            info.update(state='completed', results=metrics)
        except BaseException as error:
            info.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            info.update(finished_utc=now(), elapsed_s=time.monotonic() - started)
            (args.output / 'worker_info.json').write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps({'state': info['state'], 'output': str(args.output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
