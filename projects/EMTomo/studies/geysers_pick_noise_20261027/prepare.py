"""Freeze the approved 5x3 matrix, retaining the original numerical runtime."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import common

HERE = Path(__file__).resolve().parent
DEFAULT_STATE = common.PROJECT / 'runs' / common.STUDY_ID
RUNNERS = ('common.py', 'prepare.py', 'forward.py', 'em_worker.py', 'pipeline.py', 'report.py')


def initialize(state):
    state = Path(state).resolve()
    if not state.is_relative_to(common.PROJECT / 'runs') or state == common.PROJECT / 'runs':
        raise ValueError('New state must be inside project/runs')
    state.mkdir(parents=True, exist_ok=True)
    path = state / 'manifest.json'
    if path.exists():
        manifest = common.read(path)
        common.verify_manifest(manifest)
        return manifest
    if any(p.name != 'queue.lock' for p in state.iterdir()):
        raise FileExistsError('Unpublished preparation exists; inspect instead of overwriting')
    original = common.read(common.BASE / 'manifest.json')
    if common.read(common.BASE / 'status.json')['status'] != 'completed':
        raise ValueError('Original matched study is incomplete')
    snapshot = common.BASE / 'snapshot'
    sources = {name: checksum for name, checksum in original['snapshot_sha256'].items()
               if name.endswith('.py') and name.startswith(('projects/', 'instruments/'))}
    common.check_hashes(snapshot, sources)
    current_versions = {name: version(name) for name in original['versions']}
    if current_versions != original['versions']:
        raise ValueError(f'Numerical environment differs: {current_versions}')
    source = snapshot / 'projects/forward_modeling/experiments'
    input_dir = source / 'input' / common.EXPERIMENT
    output_dir = source / 'output' / common.EXPERIMENT
    metadata = common.read(output_dir / 'metadata.json')
    if (metadata['config'] != dict(refinement=2, source_radius_cells=2)
            or metadata['noise']['config'] != dict(relative_sigma=.01, absolute_sigma_s=.01, seed=42)):
        raise ValueError('Unexpected original forward settings')
    input_hashes = {name: common.sha(input_dir / name)
                    for name in ('model.npz', 'events.csv', 'stations.csv', 'generation.json')}
    if metadata['input_sha256'] != input_hashes:
        raise ValueError('Original forward inputs changed')
    references = {}
    for variant in common.VARIANTS:
        case_id = f'geysers-em-{variant}'
        old_case = next(c for c in original['cases'] if c['id'] == case_id)
        result_path = common.BASE / 'cases' / case_id / 'result.json'
        result = common.read(result_path)
        run = Path(result['run_dir'])
        if result['status'] != 'completed' or common.sha(run / 'final_model.npy') != result['final_model_sha256']:
            raise ValueError(f'Incomplete original inversion: {variant}')
        references[variant] = dict(case_id=case_id, config=old_case['config'], run_dir=str(run),
                                   result_sha256=common.sha(result_path),
                                   run_sha256={name: common.sha(run / name)
                                               for name in ('meta.json', 'initial_model.npy', 'final_model.npy')})
    levels = [dict(id=key, multiplier=factor, relative_sigma=.01 * factor, absolute_sigma_s=.01 * factor,
                   seed=42, experiment_id=f'geysers_spheres_picknoise_20261027_{key}')
              for key, factor in common.MULTIPLIERS]
    cases = []
    for level in levels:
        for variant in common.VARIANTS:
            identifier = f'{level["id"]}-{variant}'
            config = dict(references[variant]['config'], run_name=f'gpnoise27-{identifier}',
                          runs_dir=str(common.PROJECT / 'runs'))
            cases.append(dict(id=identifier, variant=variant, config=config, level_id=level['id'],
                              experiment_id=level['experiment_id'], reference_run=references[variant]['run_dir']))
    runner = state / 'runner'
    runner.mkdir()
    for name in RUNNERS:
        shutil.copyfile(HERE / name, runner / name)
    (state / 'experiments').mkdir()
    (state / 'cases').mkdir()
    manifest = dict(
        schema_version=1, study_id=common.STUDY_ID, created_utc=common.now(), state_dir=str(state),
        source_manifest=str(common.BASE / 'manifest.json'),
        source_manifest_sha256=common.sha(common.BASE / 'manifest.json'),
        runtime_root=str(snapshot), runtime_sha256=sources,
        runner_root=str(runner), runner_sha256={name: common.sha(runner / name) for name in RUNNERS},
        source_experiments_root=str(source), source_experiment_id=common.EXPERIMENT,
        source_input_sha256=input_hashes,
        source_output_sha256={name: common.sha(output_dir / name) for name in ('arrivals.csv', 'metadata.json')},
        experiments_root=str(state / 'experiments'), levels=levels, references=references, cases=cases,
        resources=dict(forward_workers=32, inversion_workers_per_run=12, max_parallel_runs=3, blas_threads=1),
        forward_timeout_s=10800, case_timeout_s=14400, queue_timeout_s=86400,
        python='/mnt/disk01/egor/venv/bin/python3', versions=current_versions,
        forward_spec=dict(refinement=2, source_radius_cells=2, noiseless_absolute_times_saved=True,
                          source_order='events', baseline_reproduction_atol_s=1e-8),
        policy=dict(scientific_configs_unchanged=True, allowed_config_differences=['run_name', 'runs_dir'],
                    paired_noise_seed=42, assumed_noise_fixed=dict(relative_sigma=.01, absolute_sigma_s=.01,
                                                                model_sigma_s=.02),
                    temperature_recalibrated=False, final_checkpoint=12,
                    event_locations='last E-step before final M-step',
                    old_baseline_reused=True, new_inversions=15, forward_solves=1))
    common.verify_manifest(manifest)
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return manifest


def check_baselines(manifest):
    from dataclasses import asdict
    runtime = common.load_runtime(manifest)
    import numpy as np
    for variant in common.VARIANTS:
        ref = manifest['references'][variant]
        config = runtime.config_module.InversionConfig(**ref['config'])
        if json.loads(json.dumps(asdict(config))) != ref['config']:
            raise ValueError('Effective config differs from saved original config')
        data = runtime.runner.prepare_inversion(common.EXPERIMENT, config,
                                               experiments_root=Path(manifest['source_experiments_root']))
        expected = np.load(Path(ref['run_dir']) / 'initial_model.npy', allow_pickle=False)
        if not np.array_equal(data.initial_model.velocity, expected):
            raise ValueError(f'Initial model changed: {variant}')
        if (len(data.event_ids), len(data.station_ids)) != (3776, 26):
            raise ValueError('Wrong catalogue shape')
        noise = runtime.runner.resolve_pick_noise(config, (.1, .1))
        if (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) != (.01, .01, .02):
            raise ValueError('Assumed noise not fixed')
        print(f'Validated baseline {variant}: exact config and initial model; noise resolver fixed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--check-baselines', action='store_true')
    args = parser.parse_args()
    result = initialize(args.state)
    if args.check_baselines:
        check_baselines(result)
    print(json.dumps(dict(state_dir=result['state_dir'], cases=len(result['cases']),
                          status='prepared'), indent=2))
