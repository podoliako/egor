"""Fixed-length soft -> hard real-pick runs with a spatial (ball) event holdout.

python -B studies/geysers_real_picks/launch_ball.py --state runs/NEW_DIR [--start]

Same configuration, start model, training archive and frozen numerical Python as the
1 km linear 3.5-6.0 km/s queue, except:
  * all picks of every event whose catalog hypocentre lies inside a ball are removed
    from the inversion (the catalog is used only to define the ball, never passed on);
  * n_cycles=15 and control_fraction=0: no random pick control, no automatic stopping;
  * new run names.
Held-out events and their training-archive picks are saved under heldout/ for
post-hoc evaluation. The 4775 external holdout picks remain unopened, as before.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch_latest import THREAD_ENV, read, sha, write  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'runs/geysers_real_linear35_60_latest_20261110_063859'
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
N_CYCLES = 15
ARTIFACTS = ['initial_model.npy', 'final_model.npy', 'stopping.json']


def catalog_xyz(event_ids):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'paper/paper_scripts'))
    from make_geysers_real_location_shifts import geographic_to_local, validate_geometry
    geometry = read(DATA / 'geometry.json')
    validate_geometry(geometry)
    with (DATA / 'catalog.csv').open(newline='') as stream:
        catalog = {row['event_id']: row for row in csv.DictReader(stream)}
    rows = [catalog[i] for i in event_ids]
    return np.asarray(geographic_to_local(np.array([float(r['latitude']) for r in rows]),
                                          np.array([float(r['longitude']) for r in rows]),
                                          np.array([float(r['depth_km']) for r in rows]), geometry))


def prepare(state, centre, radius):
    import numpy as np
    parent = read(PARENT / 'manifest.json')
    if read(PARENT / 'status.json')['status'] != 'completed':
        raise ValueError('Parent queue must be completed')
    for name, digest in parent['frozen_sha256'].items():
        if sha(PARENT / name) != digest:
            raise ValueError(f'Parent frozen artifact changed: {name}')
    with np.load(PARENT / 'inputs/emtomo/arrivals.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if arrays['arrival_times_s'].shape != (3471, 31) or np.isfinite(arrays['arrival_times_s']).sum() != 42973:
        raise ValueError('Unexpected parent training archive')
    ids = arrays['event_ids'].astype(str)
    xyz = catalog_xyz(ids)
    inside = np.linalg.norm(xyz - np.asarray(centre, float), axis=1) <= radius
    if not 0 < inside.sum() < len(ids):
        raise ValueError('Ball must select some but not all events')

    state.mkdir(parents=True, exist_ok=False)
    for sub in ('runner', 'inputs/emtomo', 'heldout'):
        (state / sub).mkdir(parents=True)
    shutil.copy2(__file__, state / 'runner/worker.py')
    shutil.copy2(HERE / 'launch_latest.py', state / 'runner/launch_latest.py')
    shutil.copy2(PARENT / 'runner/pipeline.py', state / 'runner/pipeline.py')
    for name in parent['frozen_sha256']:
        if name.startswith('snapshot/') and name.endswith('.py'):
            (state / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PARENT / name, state / name)
    shutil.copy2(PARENT / 'inputs/initial_velocity_m_s.npy', state / 'inputs/initial_velocity_m_s.npy')
    shutil.copy2(PARENT / 'inputs/initial_model.json', state / 'inputs/initial_model.json')

    train = dict(arrays, event_ids=arrays['event_ids'][~inside],
                 arrival_times_s=arrays['arrival_times_s'][~inside])
    with (state / 'inputs/emtomo/arrivals.npz').open('xb') as stream:
        np.savez(stream, **train)
    metadata = read(PARENT / 'inputs/emtomo/metadata.json')
    metadata['arrays_sha256'] = sha(state / 'inputs/emtomo/arrivals.npz')
    metadata['spatial_holdout'] = dict(centre_m=list(centre), radius_m=radius,
                                       removed_events=int(inside.sum()))
    write(state / 'inputs/emtomo/metadata.json', metadata)
    with (state / 'heldout/arrivals.npz').open('xb') as stream:
        np.savez(stream, event_ids=arrays['event_ids'][inside], station_ids=arrays['station_ids'],
                 station_coordinates_m=arrays['station_coordinates_m'],
                 arrival_times_s=arrays['arrival_times_s'][inside], catalog_xyz_m=xyz[inside])
    z = xyz[inside, 2]
    ball = dict(centre_m=list(centre), radius_m=radius, frame='local XYZ m, z positive down from +1166 m',
                selection='catalog hypocentre within radius of centre (catalog used only for this split)',
                n_events_total=len(ids), n_heldout_events=int(inside.sum()),
                heldout_fraction=float(inside.mean()),
                n_heldout_picks=int(np.isfinite(arrays['arrival_times_s'][inside]).sum()),
                n_training_events=int((~inside).sum()),
                n_training_picks=int(np.isfinite(train['arrival_times_s']).sum()),
                heldout_depth_below_sea_level_km=dict(
                    min=float((z.min() - 1166) / 1000), median=float((np.median(z) - 1166) / 1000),
                    max=float((z.max() - 1166) / 1000)),
                all_events_median_depth_below_sea_level_km=float((np.median(xyz[:, 2]) - 1166) / 1000),
                heldout_event_ids=ids[inside].tolist())
    write(state / 'heldout/ball.json', ball)

    cases = []
    for parent_case in parent['cases']:
        config = dict(parent_case['config'])
        config.update(n_cycles=N_CYCLES, control_fraction=0.0,
                      run_name=f"geysers-real-linear35-60-ball10-it15-{config['candidate_mode']}")
        cases.append(dict(id=parent_case['id'], config=config, completion_artifacts=ARTIFACTS,
                          parent_config_changes={k: {'parent': parent_case['config'].get(k), 'new': v}
                                                 for k, v in config.items()
                                                 if parent_case['config'].get(k) != v}))
    if [c['id'] for c in cases] != ['soft', 'hard']:
        raise ValueError('Expected soft then hard parent cases')
    for case in cases:
        (state / 'cases' / case['id']).mkdir(parents=True)
    manifest = dict(
        project_root=str(ROOT), created_utc=datetime.now(timezone.utc).isoformat(), cases=cases,
        case_timeout_s=parent['case_timeout_s'], parent_queue=str(PARENT),
        parent_manifest_sha256=sha(PARENT / 'manifest.json'),
        source_run=parent['source_run'], initial_model=parent['initial_model'],
        spatial_holdout={k: v for k, v in ball.items() if k != 'heldout_event_ids'},
        external_holdout_policy=parent['external_holdout_policy'],
        catalog_policy=parent['catalog_policy'] + '; catalog used only to define the held-out ball',
        likelihood_policy=parent['likelihood_policy'],
        stopping_policy=f'Fixed {N_CYCLES} updates; control_fraction=0, no automatic stopping',
        runtime_policy='Exact verified numerical Python snapshot inherited from parent queue',
        resources=parent['resources'],
        frozen_sha256={str(p.relative_to(state)): sha(p) for p in sorted(state.rglob('*')) if p.is_file()})
    write(state / 'manifest.json', manifest)
    for case in cases:
        subprocess.run([sys.executable, '-B', str(state / 'runner/worker.py'), '--state', str(state),
                        '--case', case['id'], '--validate-only'], check=True,
                       env=dict(os.environ, **THREAD_ENV), timeout=90)
    return manifest


def check_finished(directory, n_cycles, initial):
    import numpy as np
    stopping = read(directory / 'stopping.json')
    if stopping['reason'] != 'max_cycles' or stopping['completed_updates'] != n_cycles:
        raise ValueError(f'Unexpected stopping: {stopping}')
    done = {int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json')}
    if done != set(range(n_cycles)):
        raise ValueError('Incomplete iteration checkpoints')
    final = np.load(directory / 'final_model.npy', allow_pickle=False)
    if final.shape != initial.shape or not np.isfinite(final).all() or np.any(final <= 0):
        raise ValueError('Invalid final velocity')
    np.testing.assert_array_equal(np.load(directory / 'initial_model.npy', allow_pickle=False), initial)
    return stopping


def worker(state, mode, validate_only):
    manifest = read(state / 'manifest.json')
    for name, digest in manifest['frozen_sha256'].items():
        if sha(state / name) != digest:
            raise ValueError(f'Frozen input/runtime changed: {name}')
    runtime = state / 'snapshot'
    os.environ.update(THREAD_ENV)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    import numpy as np
    from config import InversionConfig
    from experiment_runner import prepare_real_inversion, resolve_pick_noise, initial_velocity
    from velocity_model import VelocityModel
    case = next(c for c in manifest['cases'] if c['id'] == mode)
    config = InversionConfig(**case['config'])
    continuation = 'parent_model_file' in case
    expected_grid = (500, 4) if continuation else (1000, 8)
    if (config.n_cycles != N_CYCLES or config.control_fraction != 0
            or (config.cell_size, config.subdivision) != expected_grid):
        raise ValueError('Unexpected stopping/grid configuration')
    initial_path = state / case.get('initial_model_file', 'inputs/initial_velocity_m_s.npy')
    initial = VelocityModel(np.load(initial_path, allow_pickle=False), config.cell_size)
    if continuation:
        from launch_latest import refine_cells
        parent = np.load(state / case['parent_model_file'], allow_pickle=False)
        np.testing.assert_array_equal(initial.velocity, refine_cells(parent))
        np.testing.assert_array_equal(initial.velocity.repeat(4, 0).repeat(4, 1).repeat(4, 2),
                                      parent.repeat(8, 0).repeat(8, 1).repeat(8, 2))
    else:
        np.testing.assert_allclose(initial.velocity, initial_velocity(config, initial.shape), rtol=0, atol=1e-10)
    prepared = prepare_real_inversion(state / 'inputs/emtomo', config, initial_model=initial)
    held = read(state / 'heldout/ball.json')
    if (prepared.reference_model is not None
            or prepared.arrivals_table.shape != (held['n_training_events'], 31)
            or int(np.isfinite(prepared.arrivals_table).sum()) != held['n_training_picks']
            or set(prepared.event_ids) & set(held['heldout_event_ids'])):
        raise ValueError('Unexpected real-data preparation or held-out leakage')
    noise = resolve_pick_noise(config, prepared.noise_sigmas)
    if (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) != (.01, .05, .02):
        raise ValueError('Likelihood settings differ from the parent queue')
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and (name in ('config', 'experiment_runner', 'experiment_data', 'velocity_model',
                                 'eikonal', 'interpolation', 'raytracing')
                         or name.split('.')[0] in ('tomography', 'instruments', 'projects')):
            if not Path(filename).resolve().is_relative_to(runtime):
                raise ValueError(f'Non-frozen import: {name}')
    if validate_only:
        print(json.dumps(dict(case=mode, validated=True, solver_calls=0,
                              training_events=held['n_training_events'],
                              heldout_events=held['n_heldout_events']), indent=2), flush=True)
        return
    path = state / 'cases' / mode / 'result.json'
    if path.exists() and read(path).get('run_dir'):
        raise FileExistsError('Refusing to rerun an attempted inversion')
    result = dict(status='running', case_id=mode, started_at=time.time(), config=case['config'])
    write(path, result)
    try:
        from tomography.tomography_em import run_em, warm_up_jit
        from tomography.tomography_logging import TomographyLogger
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                  run_version=config.run_version)
        result['run_dir'] = str(logger.run_dir)
        write(path, result)
        run_em(config, initial, prepared.arrivals_table, prepared.station_locs, noise,
               reference_model=None, logger=logger,
               source_experiment=dict(kind='real_picks',
                                                     id=('geysers_real_v1_ball10_refine500_it15' if continuation
                                                         else 'geysers_real_v1_linear35_60_ball10_it15'),
                                      queue_manifest=str(state / 'manifest.json'), case_id=mode,
                                      arrays_sha256=sha(state / 'inputs/emtomo/arrivals.npz'),
                                      initial_model_sha256=sha(initial_path),
                                      parent_run=case.get('parent_run'),
                                      parameters_source_run=case.get('parent_run', manifest['source_run']),
                                      spatial_holdout=manifest['spatial_holdout'],
                                      event_ids=list(prepared.event_ids), station_ids=list(prepared.station_ids)))
        stopping = check_finished(logger.run_dir, config.n_cycles, initial.velocity)
        result.update(status='completed', stopping=stopping,
                      artifact_sha256={name: sha(logger.run_dir / name) for name in ARTIFACTS})
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_at'] = time.time()
        write(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--centre-m', type=float, nargs=3, default=(9500., 3000., 2500.))
    parser.add_argument('--radius-m', type=float, default=1400.)
    parser.add_argument('--case', choices=('soft', 'hard'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    state = args.state.resolve()
    if args.case:
        worker(state, args.case, args.validate_only)
        return
    sys.path.insert(0, str(ROOT))
    prepare(state, tuple(args.centre_m), args.radius_m)
    print(json.dumps({k: v for k, v in read(state / 'heldout/ball.json').items()
                      if k != 'heldout_event_ids'}, indent=2), flush=True)
    if args.start:
        with (state / 'queue.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(state / 'runner/pipeline.py'),
                                        '--state', str(state)], cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (state / 'supervisor.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(state=str(state), supervisor_pid=process.pid), indent=2))


if __name__ == '__main__':
    main()
