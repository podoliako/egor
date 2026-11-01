"""Freeze and optionally start tomoDD -> hard -> soft with 1% pick uncertainty.

Reuse the tested soft-screen supervisor: lock, heartbeat, bounded workers,
continue-on-failure, no accidental reexecution, and SIGHUP protection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(state):
    methods = DATA / 'methods'
    provenance = json.loads((methods / 'manifest.json').read_text())
    if sha(DATA / 'manifest.json') != provenance['split_manifest_sha256']:
        raise ValueError('Split manifest changed')
    split = json.loads((DATA / 'manifest.json').read_text())
    for root, hashes in ((DATA, split['output_sha256']), (methods, provenance['output_sha256'])):
        for name, digest in hashes.items():
            if sha(root / name) != digest:
                raise ValueError(f'Prepared artifact changed: {root / name}')
    binary = Path(provenance['native_solver']['path'])
    if sha(binary) != provenance['native_solver']['sha256']:
        raise ValueError('Corrected tomoDD binary changed')
    configs = json.loads((methods / 'emtomo_configs.json').read_text())
    state.mkdir(parents=True, exist_ok=False)
    (state / 'runner').mkdir()
    shutil.copy2(ROOT / 'studies/geysers_soft_screen/pipeline.py', state / 'runner/pipeline.py')
    shutil.copy2(Path(__file__).with_name('worker.py'), state / 'runner/worker.py')
    # Copy only numerical Python source, not old runs, caches, or forward datasets.
    for source in (ROOT, ROOT / 'tomography', ROOT / 'instruments', ROOT.parent / 'forward_modeling'):
        target = state / 'snapshot' / source.relative_to(ROOT.parents[1])
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob('*.py'):
            shutil.copy2(path, target / path.name)
    (state / 'inputs').mkdir()
    shutil.copytree(DATA / 'emtomo', state / 'inputs/emtomo')
    shutil.copy2(methods / 'initial_velocity_m_s.npy', state / 'inputs/initial_velocity_m_s.npy')
    shutil.copy2(methods / 'regional_model.json', state / 'inputs/regional_model.json')
    (state / 'solver').mkdir()
    shutil.copy2(binary, state / 'solver/tomoDD-SE')
    native_work = state / 'cases/tomodd/work'
    shutil.copytree(methods / 'tomodd', native_work)
    cases = [dict(id='tomodd', config=dict(n_cycles=3, candidate_mode='tomodd'),
                  completion_artifacts=['Vp_model.dat', 'tomoDD.reloc', 'tomoDD.reloc.001.001',
                                        'tomoDD.reloc.001.002', 'tomoDD.reloc.001.003', 'tomoDD.vel'])]
    for mode in ('hard', 'soft'):
        config = configs[mode]
        config.update(weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=0.,
                      run_name=f'geysers-real-{mode}-pick01', runs_dir=str(ROOT / 'runs'))
        (state / 'cases' / mode).mkdir()
        cases.append(dict(id=mode, config=config))
    import numpy as np
    with np.load(state / 'inputs/emtomo/arrivals.npz', allow_pickle=False) as inputs:
        event_ids = inputs['event_ids'].tolist()
    manifest = dict(
        project_root=str(ROOT), runtime_root=str(state / 'snapshot'),
        created_utc=datetime.now(timezone.utc).isoformat(), case_timeout_s=21600,
        cases=cases, event_ids=event_ids, source_methods_manifest_sha256=sha(methods / 'manifest.json'),
        source_split_manifest_sha256=sha(DATA / 'manifest.json'),
        pick_noise=dict(relative_sigma=.01, absolute_sigma_s=0., model_sigma_s=.02,
                        time='predicted station travel time, not first-pick relative time'),
        tomodd_policy='unchanged dd1-d40-s5 equal pick weights; catalog unperturbed; not Gaussian 1% reweighting',
        heldout_policy='4775 held-out picks never copied into numerical inputs',
        resources=dict(workers_per_emtomo=24, simultaneous_cases=1, blas_threads=1),
        frozen_sha256={str(p.relative_to(state)): sha(p) for p in sorted(state.rglob('*')) if p.is_file()},
    )
    (state / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, **{name: '1' for name in
                             ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')})
    for case in cases:
        subprocess.run([sys.executable, '-B', str(state / 'runner/worker.py'), '--state', str(state),
                        '--case', case['id'], '--validate-only'], check=True, env=env, timeout=60)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True, help='New state directory')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    state = args.state.resolve()
    prepare(state)
    if args.start:
        with (state / 'queue.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(state / 'runner/pipeline.py'),
                                      '--state', str(state)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (state / 'supervisor.pid').write_text(str(child.pid) + '\n')
        print(json.dumps(dict(state=str(state), supervisor_pid=child.pid, order=['tomodd', 'hard', 'soft'])))


if __name__ == '__main__':
    main()
