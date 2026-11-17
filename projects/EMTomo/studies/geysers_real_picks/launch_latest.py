"""Freeze and launch soft -> hard real-pick runs using an existing run's settings.

Only the starting model, grid spacing, candidate mode and output names change.
The original external holdout remains excluded; automatic stopping uses a new
control split of the prepared training archive. Numerical imports in workers
are activated only after verifying the frozen snapshot.
"""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENV = dict.fromkeys(('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'), '1')
ARTIFACTS = ['initial_model.npy', 'final_model.npy', 'stopping.json',
             'control_mask.npy', 'control_split.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_config(source_params, mode, model_info, runs_dir):
    from config import InversionConfig
    if mode not in ('soft', 'hard'):
        raise ValueError('Expected soft or hard')
    parameters = {f.name: source_params[f.name] for f in fields(InversionConfig)
                  if f.name in source_params}
    overrides = dict(cell_size=model_info['cell_size_m'],
                     initial_velocity_m_s=model_info['vp_at_top_boundary_m_s'],
                     initial_gradient_m_s=[model_info['vp_at_top_boundary_m_s'],
                                           model_info['vp_at_bottom_boundary_m_s']],
                     initial_layer_boundaries_km=None, initial_layer_velocities_m_s=None,
                     candidate_mode=mode, save_runs=True, runs_dir=str(runs_dir),
                     run_name=f'geysers-real-linear35-60-cov0-auto-p12-{mode}')
    parameters.update(overrides)
    config = asdict(InversionConfig(**parameters))
    changes = {key: {'source': source_params.get(key), 'new': value}
               for key, value in overrides.items() if source_params.get(key) != value}
    return config, changes


def control_descriptor(arrivals, config):
    import numpy as np
    from tomography.tomography_validation import split_control_arrivals
    training, mask = split_control_arrivals(
        arrivals, fraction=config.control_fraction, seed=config.control_seed,
        min_training_picks=config.control_min_training_picks)
    if not np.isnan(training[mask]).all():
        raise ValueError('Control observations leaked into training')
    np.testing.assert_array_equal(training[~mask], arrivals[~mask])
    if np.min(np.isfinite(training).sum(axis=1)) < config.control_min_training_picks:
        raise ValueError('Too few training picks')
    return mask, dict(n_input=int(np.isfinite(arrivals).sum()), n_control=int(mask.sum()),
                      n_training=int(np.isfinite(training).sum()),
                      scored_events=int(mask.any(axis=1).sum()),
                      mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def prepare(state, source_run):
    import numpy as np
    from config import InversionConfig
    from experiment_runner import initial_velocity
    data = ROOT / 'data/geysers_2011/real_experiment_v1'
    model_dir = ROOT / 'data/geysers_2011/initial_models/linear_vp_3p5_6p0'
    split = read(data / 'manifest.json')
    for name in ('emtomo/metadata.json', 'emtomo/arrivals.npz', 'geometry.json'):
        if sha(data / name) != split['output_sha256'][name]:
            raise ValueError(f'Training input changed: {name}')
    model_info = read(model_dir / 'model.json')
    model_path = model_dir / model_info['array_file']
    if sha(model_path) != model_info['array_sha256']:
        raise ValueError('Initial model changed')
    source_meta = source_run / 'meta.json'
    params = read(source_meta)['run_params']
    cases = []
    for mode in ('soft', 'hard'):
        config, changes = build_config(params, mode, model_info, ROOT / 'runs')
        np.testing.assert_allclose(initial_velocity(InversionConfig(**config), tuple(model_info['shape'])),
                                   np.load(model_path, allow_pickle=False), rtol=0, atol=1e-10)
        cases.append(dict(id=mode, config=config, source_parameter_changes=changes,
                          completion_artifacts=ARTIFACTS))
    with np.load(data / 'emtomo/arrivals.npz', allow_pickle=False) as archive:
        arrivals = archive['arrival_times_s']
        if arrivals.shape != (3471, 31) or np.isfinite(arrivals).sum() != 42973:
            raise ValueError('Unexpected real-pick selection')
        _, control = control_descriptor(arrivals, InversionConfig(**cases[0]['config']))
    state.mkdir(parents=True, exist_ok=False)
    (state / 'runner').mkdir()
    shutil.copy2(__file__, state / 'runner/worker.py')
    shutil.copy2(ROOT / 'studies/geysers_soft_screen/pipeline.py', state / 'runner/pipeline.py')
    for source in (ROOT, ROOT / 'tomography', ROOT / 'instruments', ROOT.parent / 'forward_modeling'):
        target = state / 'snapshot' / source.relative_to(ROOT.parents[1])
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob('*.py'):
            shutil.copy2(path, target / path.name)
    (state / 'inputs').mkdir()
    shutil.copytree(data / 'emtomo', state / 'inputs/emtomo')
    shutil.copy2(model_path, state / 'inputs/initial_velocity_m_s.npy')
    shutil.copy2(model_dir / 'model.json', state / 'inputs/initial_model.json')
    for case in cases:
        (state / 'cases' / case['id']).mkdir(parents=True)
    manifest = dict(
        project_root=str(ROOT), created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases, case_timeout_s=21600, source_run=str(source_run),
        source_run_meta_sha256=sha(source_meta), source_run_params=params,
        source_split_manifest_sha256=sha(data / 'manifest.json'),
        initial_model=model_info, control_split=control,
        external_holdout_policy='4775 external holdout picks are neither opened nor copied',
        catalog_policy='No catalog locations/origins, published Vp or true model passed to inversion',
        likelihood_policy='Inherited 1% relative + 0.05 s absolute + 0.02 s model sigma; assumptions, not calibrated pick accuracy',
        runtime_policy='Current numerical Python source frozen without modifying user work',
        resources=dict(workers_per_run=12, simultaneous_runs=1, order=['soft', 'hard'], thread_environment=THREAD_ENV),
        frozen_sha256={str(p.relative_to(state)): sha(p) for p in sorted(state.rglob('*')) if p.is_file()})
    write(state / 'manifest.json', manifest)
    env = dict(os.environ, **THREAD_ENV)
    for case in cases:
        subprocess.run([sys.executable, '-B', str(state / 'runner/worker.py'), '--state', str(state),
                        '--case', case['id'], '--validate-only'], check=True, env=env, timeout=90)
    return manifest


def refine_cells(velocity):
    """Split each velocity cell into eight identical children, without smoothing."""
    import numpy as np
    velocity = np.asarray(velocity)
    if velocity.ndim != 3 or not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError('Expected finite positive 3-D parent velocity')
    refined = velocity.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                np.testing.assert_array_equal(refined[x::2, y::2, z::2], velocity)
    return refined


def build_refined_config(source_config, mode, runs_dir):
    from config import InversionConfig
    if source_config['candidate_mode'] != mode:
        raise ValueError('Parent mode must match continuation mode')
    if source_config['cell_size'] != 1000 or source_config['subdivision'] != 8:
        raise ValueError('Expected 1 km parent cells and 125 m calculation grid')
    parameters = dict(source_config)
    parameters.update(cell_size=500., subdivision=4, initial_gradient_m_s=None,
                      initial_layer_boundaries_km=None, initial_layer_velocities_m_s=None,
                      runs_dir=str(runs_dir), save_runs=True,
                      run_name=f'geysers-real-refine500-from-linear35-60-cov0-auto-p12-{mode}')
    return asdict(InversionConfig(**parameters))


def prepare_refinement(state, parent_queue):
    import numpy as np
    from config import InversionConfig
    parent_manifest = read(parent_queue / 'manifest.json')
    if read(parent_queue / 'status.json')['status'] != 'completed':
        raise ValueError('Parent queue must be completed successfully')
    for name, digest in parent_manifest['frozen_sha256'].items():
        if sha(parent_queue / name) != digest:
            raise ValueError(f'Parent frozen artifact changed: {name}')
    parent_models, cases, parent_hashes = {}, [], {}
    for mode in ('soft', 'hard'):
        record = parent_queue / 'cases' / mode / 'result.json'
        result = read(record)
        if result['status'] != 'completed':
            raise ValueError(f'Parent {mode} is not completed')
        directory = Path(result['run_dir'])
        for name, digest in result['artifact_sha256'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Changed parent result: {directory / name}')
            parent_hashes[str(directory / name)] = digest
        parent_hashes[str(record)] = sha(record)
        original = next(c['config'] for c in parent_manifest['cases'] if c['id'] == mode)
        config = build_refined_config(original, mode, ROOT / 'runs')
        velocity = np.load(directory / 'final_model.npy', allow_pickle=False)
        if velocity.shape != (23, 11, 7):
            raise ValueError('Unexpected parent velocity grid')
        mask = np.load(directory / 'control_mask.npy', allow_pickle=False)
        check_finished(directory, InversionConfig(**original),
                       np.load(directory / 'initial_model.npy', allow_pickle=False),
                       mask, parent_manifest['control_split'])
        refined = refine_cells(velocity)
        # The same continuous piecewise-constant start is seen by the 125 m FMM grid.
        old_fine = velocity.repeat(8, 0).repeat(8, 1).repeat(8, 2)
        new_fine = refined.repeat(4, 0).repeat(4, 1).repeat(4, 2)
        np.testing.assert_array_equal(new_fine, old_fine)
        parent_models[mode] = (directory, refined)
        changes = {key: {'source': original.get(key), 'new': value}
                   for key, value in config.items() if value != original.get(key)}
        cases.append(dict(id=mode, config=config, source_parameter_changes=changes,
                          completion_artifacts=ARTIFACTS,
                          initial_model_file=f'inputs/{mode}/initial_velocity_m_s.npy',
                          parent_model_file=f'inputs/{mode}/parent_final_model.npy',
                          parent_control_mask_file=f'inputs/{mode}/parent_control_mask.npy',
                          parent_run=str(directory), parent_stopping=result['stopping']))
    inputs = parent_queue / 'inputs/emtomo'
    with np.load(inputs / 'arrivals.npz', allow_pickle=False) as archive:
        mask, control = control_descriptor(archive['arrival_times_s'], InversionConfig(**cases[0]['config']))
    if control != parent_manifest['control_split']:
        raise ValueError('Continuation must use identical training/control split')
    for mode, (directory, _) in parent_models.items():
        np.testing.assert_array_equal(mask, np.load(directory / 'control_mask.npy', allow_pickle=False))
    state.mkdir(parents=True, exist_ok=False)
    (state / 'runner').mkdir()
    shutil.copy2(__file__, state / 'runner/worker.py')
    shutil.copy2(parent_queue / 'runner/pipeline.py', state / 'runner/pipeline.py')
    # Reuse exactly the parent's verified numerical Python, not mutable live code or caches.
    for name in parent_manifest['frozen_sha256']:
        if name.startswith('snapshot/') and name.endswith('.py'):
            target = state / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_queue / name, target)
    shutil.copytree(inputs, state / 'inputs/emtomo')
    for mode, (directory, refined) in parent_models.items():
        target = state / 'inputs' / mode
        target.mkdir()
        with (target / 'initial_velocity_m_s.npy').open('xb') as stream:
            np.save(stream, refined, allow_pickle=False)
        shutil.copy2(directory / 'final_model.npy', target / 'parent_final_model.npy')
        shutil.copy2(directory / 'control_mask.npy', target / 'parent_control_mask.npy')
        (state / 'cases' / mode).mkdir(parents=True)
    manifest = dict(
        project_root=str(ROOT), created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases, case_timeout_s=parent_manifest['case_timeout_s'], source_run=str(parent_queue),
        parent_queue=str(parent_queue), parent_manifest_sha256=sha(parent_queue / 'manifest.json'),
        parent_result_sha256=parent_hashes, control_split=control,
        external_holdout_policy=parent_manifest['external_holdout_policy'],
        catalog_policy=parent_manifest['catalog_policy'],
        likelihood_policy=parent_manifest['likelihood_policy'],
        initial_policy='Each mode starts from its own final parent Vp; 2x2x2 identical children, no smoothing',
        calculation_grid_policy='500 m velocity cells, subdivision=4, unchanged 125 m calculation grid',
        runtime_policy='Exact verified numerical Python snapshot inherited from parent queue',
        auto_stopping_policy='Fresh stopping counter, up to 7 additional updates; same control mask and thresholds',
        resources=parent_manifest['resources'],
        frozen_sha256={str(p.relative_to(state)): sha(p) for p in sorted(state.rglob('*')) if p.is_file()})
    write(state / 'manifest.json', manifest)
    for case in cases:
        subprocess.run([sys.executable, '-B', str(state / 'runner/worker.py'), '--state', str(state),
                        '--case', case['id'], '--validate-only'], check=True,
                       env=dict(os.environ, **THREAD_ENV), timeout=90)
    return manifest


def check_finished(directory, config, initial, mask, control):
    import numpy as np
    stopping = read(directory / 'stopping.json')
    n = stopping['completed_updates']
    if not isinstance(n, int) or not 3 <= n <= config.n_cycles:
        raise ValueError('Invalid stopping update count')
    expected = set(range(n))
    if {int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json')} != expected:
        raise ValueError('Incomplete iteration checkpoints')
    final = np.load(directory / 'final_model.npy', allow_pickle=False)
    if final.shape != initial.shape or not np.isfinite(final).all() or np.any(final <= 0):
        raise ValueError('Invalid final velocity')
    np.testing.assert_array_equal(np.load(directory / 'initial_model.npy', allow_pickle=False), initial)
    np.testing.assert_array_equal(np.load(directory / 'control_mask.npy', allow_pickle=False), mask)
    previous, insufficient, stop = None, 0, False
    for i in range(n):
        score = read(directory / f'iter_{i}/control.json')
        rmse = score['rmse_s']
        if not np.isfinite(rmse) or rmse < 0 or score['n_control'] != control['n_control']:
            raise ValueError('Invalid control score')
        if previous is not None:
            insufficient = 0 if rmse < previous * (1 - config.early_stopping_min_relative_improvement) else insufficient + 1
        stop = i + 1 >= 3 and insufficient >= config.early_stopping_patience
        if score['insufficient_steps'] != insufficient or score['stop'] != stop or (stop and i != n - 1):
            raise ValueError('Inconsistent automatic stopping')
        previous = rmse
    if (stopping['reason'] != ('control_rmse_plateau' if stop else 'max_cycles')
            or stopping['control_rmse_s'] != previous or (not stop and n != config.n_cycles)):
        raise ValueError('Inconsistent stopping summary')
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
    initial_path = state / case.get('initial_model_file', 'inputs/initial_velocity_m_s.npy')
    initial = VelocityModel(np.load(initial_path, allow_pickle=False), config.cell_size)
    if 'parent_model_file' in case:
        parent = np.load(state / case['parent_model_file'], allow_pickle=False)
        np.testing.assert_array_equal(initial.velocity, refine_cells(parent))
        if config.cell_size != 500 or config.subdivision != 4:
            raise ValueError('Continuation must retain 125 m calculation cells')
    else:
        np.testing.assert_allclose(initial.velocity, initial_velocity(config, initial.shape), rtol=0, atol=1e-10)
    prepared = prepare_real_inversion(state / 'inputs/emtomo', config, initial_model=initial)
    if prepared.reference_model is not None or prepared.arrivals_table.shape != (3471, 31):
        raise ValueError('Unexpected real-data preparation')
    noise = resolve_pick_noise(config, prepared.noise_sigmas)
    mask, control = control_descriptor(prepared.arrivals_table, config)
    if control != manifest['control_split']:
        raise ValueError('Control split changed')
    if 'parent_control_mask_file' in case:
        np.testing.assert_array_equal(mask, np.load(state / case['parent_control_mask_file'], allow_pickle=False))
    if (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) != (.01, .05, .02):
        raise ValueError('Likelihood settings differ from requested source run')
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and (name in ('config', 'experiment_runner', 'experiment_data', 'velocity_model',
                                 'eikonal', 'interpolation', 'raytracing')
                         or name.split('.')[0] in ('tomography', 'instruments', 'projects')):
            if not Path(filename).resolve().is_relative_to(runtime):
                raise ValueError(f'Non-frozen import: {name}')
    if validate_only:
        print(json.dumps(dict(case=mode, validated=True, solver_calls=0, control=control), indent=2), flush=True)
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
                                      id=('geysers_real_v1_refine500_from_linear35_60_train90'
                                          if 'parent_run' in case else 'geysers_real_v1_linear35_60_train90'),
                                      queue_manifest=str(state / 'manifest.json'), case_id=mode,
                                      arrays_sha256=sha(state / 'inputs/emtomo/arrivals.npz'),
                                      initial_model_sha256=sha(initial_path),
                                      parameters_source_run=case.get('parent_run', manifest['source_run']),
                                      parent_run=case.get('parent_run'),
                                      event_ids=list(prepared.event_ids), station_ids=list(prepared.station_ids)))
        stopping = check_finished(logger.run_dir, config, initial.velocity, mask, control)
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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--source-run', type=Path)
    sources.add_argument('--parent-queue', type=Path, help='Refine completed 1 km runs to 500 m, retaining 125 m FMM cells')
    parser.add_argument('--case', choices=('soft', 'hard'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    state = args.state.resolve()
    if args.case:
        worker(state, args.case, args.validate_only)
        return
    if args.source_run is None and args.parent_queue is None:
        parser.error('--source-run or --parent-queue is required when preparing a queue')
    sys.path.insert(0, str(ROOT))
    manifest = (prepare_refinement(state, args.parent_queue.resolve()) if args.parent_queue
                else prepare(state, args.source_run.resolve()))
    if args.start:
        with (state / 'queue.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(state / 'runner/pipeline.py'),
                                        '--state', str(state)], cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (state / 'supervisor.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(state=str(state), supervisor_pid=process.pid,
                              order=manifest['resources']['order']), indent=2))


if __name__ == '__main__':
    main()
