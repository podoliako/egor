"""Continue a completed ball-holdout queue with 500 m velocity cells.

Keep the exact training archive and ball event IDs, parent's numerical snapshot,
125 m calculation grid, and 15 fixed additional updates. Soft then hard.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from launch_latest import THREAD_ENV, build_refined_config, read, refine_cells, sha, write
from launch_ball import ARTIFACTS, N_CYCLES, check_finished


def continuation_config(original, mode):
    config = build_refined_config(original, mode, ROOT / 'runs')
    if original['n_cycles'] != N_CYCLES or original['control_fraction'] != 0:
        raise ValueError('Expected fixed 15-update ball-holdout parent')
    config['run_name'] = f'geysers-real-ball10-refine500-it15-{mode}'
    return config


def prepare(state, parent):
    import numpy as np
    manifest = read(parent / 'manifest.json')
    if read(parent / 'status.json')['status'] != 'completed':
        raise ValueError('Parent queue must be completed')
    for name, digest in manifest['frozen_sha256'].items():
        if sha(parent / name) != digest:
            raise ValueError(f'Changed parent frozen file: {name}')
    ball = read(parent / 'heldout/ball.json')
    with np.load(parent / 'inputs/emtomo/arrivals.npz', allow_pickle=False) as arrays:
        if (arrays['arrival_times_s'].shape != (ball['n_training_events'], 31)
                or np.isfinite(arrays['arrival_times_s']).sum() != ball['n_training_picks']
                or set(arrays['event_ids'].astype(str)) & set(ball['heldout_event_ids'])):
            raise ValueError('Parent training archive leaks held-out events or has wrong counts')
    models, cases, parent_hashes = {}, [], {}
    for mode in ('soft', 'hard'):
        record = parent / f'cases/{mode}/result.json'
        result = read(record)
        if result['status'] != 'completed':
            raise ValueError(f'Parent {mode} incomplete')
        directory = Path(result['run_dir'])
        for name, digest in result['artifact_sha256'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Changed parent artifact: {directory / name}')
            parent_hashes[str(directory / name)] = digest
        parent_hashes[str(record)] = sha(record)
        original = next(c['config'] for c in manifest['cases'] if c['id'] == mode)
        config = continuation_config(original, mode)
        velocity = np.load(directory / 'final_model.npy', allow_pickle=False)
        if velocity.shape != (23, 11, 7):
            raise ValueError('Expected 1 km parent model')
        check_finished(directory, N_CYCLES, np.load(directory / 'initial_model.npy', allow_pickle=False))
        if np.load(directory / 'control_mask.npy', allow_pickle=False).any():
            raise ValueError('Unexpected internal control in ball parent')
        refined = refine_cells(velocity)
        np.testing.assert_array_equal(refined.repeat(4, 0).repeat(4, 1).repeat(4, 2),
                                      velocity.repeat(8, 0).repeat(8, 1).repeat(8, 2))
        models[mode] = (directory, refined)
        cases.append(dict(id=mode, config=config, completion_artifacts=ARTIFACTS,
                          initial_model_file=f'inputs/{mode}/initial_velocity_m_s.npy',
                          parent_model_file=f'inputs/{mode}/parent_final_model.npy',
                          parent_run=str(directory), parent_stopping=result['stopping'],
                          parent_config_changes={k: {'parent': original.get(k), 'new': v}
                                                 for k, v in config.items() if original.get(k) != v}))
    state.mkdir(parents=True, exist_ok=False)
    (state / 'runner').mkdir()
    for source, target in ((HERE / 'launch_ball.py', 'worker.py'),
                           (HERE / 'launch_latest.py', 'launch_latest.py'),
                           (Path(__file__), 'prepare_refinement.py'),
                           (parent / 'runner/pipeline.py', 'pipeline.py')):
        shutil.copy2(source, state / 'runner' / target)
    for name in manifest['frozen_sha256']:
        if name.startswith('snapshot/') and name.endswith('.py'):
            target = state / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent / name, target)
    shutil.copytree(parent / 'inputs/emtomo', state / 'inputs/emtomo')
    (state / 'heldout').mkdir()
    # Keep the exact ball definition/IDs; neither copy nor open held-out arrivals.
    shutil.copy2(parent / 'heldout/ball.json', state / 'heldout/ball.json')
    for name in ('arrivals.npz', 'metadata.json'):
        if sha(state / 'inputs/emtomo' / name) != sha(parent / 'inputs/emtomo' / name):
            raise ValueError('Training archive changed during copy')
    for mode, (directory, refined) in models.items():
        target = state / 'inputs' / mode
        target.mkdir()
        np.save(target / 'initial_velocity_m_s.npy', refined, allow_pickle=False)
        shutil.copy2(directory / 'final_model.npy', target / 'parent_final_model.npy')
        (state / 'cases' / mode).mkdir(parents=True)
    prepared = dict(
        project_root=str(ROOT), created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases, case_timeout_s=manifest['case_timeout_s'], source_run=str(parent),
        parent_queue=str(parent), parent_manifest_sha256=sha(parent / 'manifest.json'),
        parent_result_sha256=parent_hashes, spatial_holdout=manifest['spatial_holdout'],
        heldout_ball_sha256=sha(parent / 'heldout/ball.json'),
        heldout_arrivals_policy='Not opened or copied; unchanged ball archive remains in parent queue',
        external_holdout_policy=manifest['external_holdout_policy'],
        catalog_policy=manifest['catalog_policy'], likelihood_policy=manifest['likelihood_policy'],
        initial_policy='Own mode final Vp after 15 updates, split 2x2x2 with identical children; no smoothing',
        calculation_grid_policy='500 m velocity cells, subdivision4, unchanged 125 m calculation grid',
        stopping_policy=f'Fixed {N_CYCLES} additional updates; control_fraction=0, no automatic stopping',
        runtime_policy='Exact verified numerical Python snapshot inherited from parent ball queue',
        resources=manifest['resources'],
        frozen_sha256={str(p.relative_to(state)): sha(p) for p in sorted(state.rglob('*')) if p.is_file()})
    write(state / 'manifest.json', prepared)
    for case in cases:
        subprocess.run([sys.executable, '-B', str(state / 'runner/worker.py'), '--state', str(state),
                        '--case', case['id'], '--validate-only'], check=True,
                       env=dict(os.environ, **THREAD_ENV), timeout=90)
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-queue', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    state = args.state.resolve()
    manifest = prepare(state, args.parent_queue.resolve())
    if args.start:
        with (state / 'queue.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(state / 'runner/pipeline.py'),
                                        '--state', str(state)], cwd=ROOT,
                                       env=dict(os.environ, **THREAD_ENV), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (state / 'supervisor.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(state=str(state), supervisor_pid=process.pid,
                              order=manifest['resources']['order']), indent=2), flush=True)
