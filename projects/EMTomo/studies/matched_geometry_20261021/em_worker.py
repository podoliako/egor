"""Run one matched EMTomo case from a frozen runtime and frozen observations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(manifest):
    root = Path(manifest['snapshot_root'])
    for name, expected in manifest['snapshot_sha256'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError(f'Snapshot modified: {path}')


def prepare(manifest, case):
    import importlib
    root = Path(manifest['snapshot_root'])
    runtime = root / 'projects/EMTomo'
    if any(name in sys.modules for name in ('config', 'experiment_runner', 'projects')):
        raise ValueError('Use a fresh worker process, without preloaded runtime modules')
    sys.path[:0] = [str(runtime), str(root)]
    config_module = importlib.import_module('config')
    runner = importlib.import_module('experiment_runner')
    em = importlib.import_module('tomography.tomography_em')
    for name, module in tuple(sys.modules.items()):
        if name in ('config', 'eikonal', 'experiment_runner', 'experiment_data', 'raytracing',
                    'interpolation', 'velocity_model') or name.startswith(('projects.', 'instruments.', 'tomography.')):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise ValueError(f'Non-snapshot import: {name}: {location}')
    config = config_module.InversionConfig(**case['config'])
    expected = dict(cell_size=1000., subdivision=8, n_cycles=12, n_candidates=15,
                    slowness_interpolation='nearest', lambda_reg=1., coverage_damping_power=2.,
                    coverage_floor=.05, coverage_reference_percentile=75., temperature=1.,
                    weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.01,
                    weight_model_sigma_s=.02, max_velocity_step_fraction=.02, n_workers=12,
                    initial_layer_boundaries_km=[.7, 1.9, 3.4],
                    initial_layer_velocities_m_s=[3800., 4600., 5200., 5700.], save_runs=True)
    actual = json.loads(json.dumps(asdict(config)))
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f'Unapproved configuration: {key}: {actual[key]}')
    mode = (config.candidate_mode, config.weights_top_n, config.weights_min_distance)
    if mode not in (('hard', 1, 1), ('soft', 10, 1), ('soft', 10, 2)):
        raise ValueError(f'Unexpected candidate configuration: {mode}')
    prepared = runner.prepare_inversion(case['experiment_id'], config,
                                       experiments_root=Path(manifest['forward_root']) / 'experiments')
    count = (1920, 52) if case['geometry'] == 'synthetic' else (3776, 26)
    if (len(prepared.event_ids), len(prepared.station_ids)) != count or tuple(prepared.noise_sigmas) != (.01, .01):
        raise ValueError('Unexpected geometry or pick noise')
    if prepared.initial_model.shape != (20, 7, 7):
        raise ValueError('Unexpected inversion grid')
    return config, runner, em, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    manifest = json.loads(args.manifest.read_text())
    case = next(c for c in manifest['cases'] if c['id'] == args.case_id)
    if case['method'] != 'emtomo':
        raise ValueError('Expected EMTomo case')
    verify(manifest)
    config, runner, em, prepared = prepare(manifest, case)
    if args.validate_only:
        print(json.dumps({'case': case['id'], 'status': 'validated', 'config': asdict(config),
                          'events': len(prepared.event_ids), 'stations': len(prepared.station_ids),
                          'method': em.METHOD_DESCRIPTION}), flush=True)
        return
    state = Path(manifest['state_dir']) / 'cases' / case['id']
    state.mkdir(exist_ok=False)
    result_path = state / 'result.json'
    provenance = dict(study=manifest['study_id'], case_id=case['id'],
                      manifest_path=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
                      snapshot_root=manifest['snapshot_root'], snapshot_sha256=manifest['snapshot_sha256'],
                      worker_sha256=digest(__file__), effective_config=asdict(config),
                      final_locations_are_pre_last_velocity_update=True)

    class StudyLogger(em.TomographyLogger):
        def save_meta(self, **kwargs):
            kwargs['run_params']['provenance'] = provenance
            super().save_meta(**kwargs)

    result = dict(case_id=case['id'], status='running', started_utc=now(), worker_pid=os.getpid(),
                  config=asdict(config), run_dir=None, provenance=provenance)
    atomic_json(result_path, result)
    started = time.monotonic()
    try:
        logger = StudyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                             run_version=config.run_version,
                             run_tags={'ncand': 15, 'topn': config.weights_top_n,
                                       'mode': config.candidate_mode, 'dmin': config.weights_min_distance})
        result['run_dir'] = str(logger.run_dir.resolve())
        atomic_json(result_path, result)
        print(f"Case {case['id']}; run directory: {logger.run_dir}", flush=True)
        em.warm_up_jit()
        em.run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
                  runner.resolve_pick_noise(config, prepared.noise_sigmas),
                  reference_model=prepared.reference_model,
                  source_experiment={'id': case['experiment_id'], 'model_sha256': digest(
                      Path(manifest['forward_root']) / 'experiments/input' / case['experiment_id'] / 'model.npz')},
                  run_name=config.run_name, logger=logger)
        import numpy as np
        final = np.load(logger.run_dir / 'final_model.npy', allow_pickle=False)
        if final.shape != (20, 7, 7) or not np.isfinite(final).all() or not (final > 0).all():
            raise ValueError('Invalid final velocity model')
        quality = [json.loads(line) for line in (logger.run_dir / 'quality.jsonl').read_text().splitlines()]
        if [row['iter'] for row in quality] != list(range(12)):
            raise ValueError('Incomplete quality history')
        for step in range(12):
            cycle = logger.run_dir / f'iter_{step}'
            if json.loads((cycle / 'complete.json').read_text())['iter'] != step:
                raise ValueError(f'Incomplete iteration {step}')
            for event in range(len(prepared.event_ids)):
                if not (cycle / f'event_{event}/weights.npz').is_file():
                    raise ValueError(f'Missing event weights: {step}/{event}')
        verify(manifest)
        result.update(status='completed', final_model_sha256=digest(logger.run_dir / 'final_model.npy'))
    except BaseException as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=now(), elapsed_s=time.monotonic() - started)
        atomic_json(result_path, result)


if __name__ == '__main__':
    main()
