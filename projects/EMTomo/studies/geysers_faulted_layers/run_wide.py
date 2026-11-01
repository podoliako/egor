"""Matched wide-search hard/soft pair on the clean faulted Geysers experiment.

python -B studies/geysers_faulted_layers/run_wide.py hard|soft [--temperature 8] [--validate-only]
Both modes use the same saved accelerated runtime, candidates and regularization.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'runs/geysers_faulted_layers/wide_cov0'


def main(mode, validate_only=False, temperature=1.):
    manifest = json.loads((STATE / 'manifest.json').read_text())
    runtime = Path(manifest['runtime_root'])
    for name, digest in manifest['runtime_sha256'].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for name, digest in manifest['data_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Experiment data changed: {name}')
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    from config import InversionConfig
    from main import main as run_saved
    import numpy as np

    old = json.loads((ROOT / 'runs/geysers_pick_noise_20261027/manifest.json').read_text())
    parameters = dict(old['references']['hard-k15-top1-dmin1']['config'])
    parameters.update(candidate_mode=mode, temperature=temperature, n_candidates=64, weights_top_n=20,
                      weights_min_distance=4, coverage_damping_power=0.,
                      initial_layer_boundaries_km=(.85, 2.3),
                      initial_layer_velocities_m_s=(3850., 4550., 5300.),
                      weight_noise_relative_sigma=0., weight_noise_absolute_sigma_s=0.,
                      linear_solver='dense', compact_rays=True, fast_refinement=True,
                      run_name=f'gfault28-{mode}-k64-top20-dmin4-cov0', runs_dir=str(ROOT / 'runs'))
    label = mode if temperature == 1 else f'{mode}-t{temperature:g}'
    if temperature != 1:
        parameters['run_name'] += f'-t{temperature:g}'
    config = InversionConfig(**parameters)
    prepared = run_saved(config, experiment_id=manifest['experiment_id'],
                         experiments_root=manifest['experiments_root'], validate_only=True)
    assert prepared.initial_model.shape == (20, 7, 7)
    assert prepared.arrivals_table.shape == (3776, 26) and prepared.noise_sigmas == (0., 0.)
    np.testing.assert_array_equal(prepared.initial_model.velocity[0, 0],
                                  [3850., 4550., 5300., 5300., 5300., 5300., 5300.])
    print(json.dumps(asdict(config), indent=2), flush=True)
    if validate_only:
        print('Validated: no EM, no JIT warmup, no run output.', flush=True)
        return
    status = dict(status='running', mode=mode, config=asdict(config),
                  experiment_id=manifest['experiment_id'], runtime_root=str(runtime),
                  manifest_sha256=hashlib.sha256((STATE / 'manifest.json').read_bytes()).hexdigest(),
                  started_utc=datetime.now(timezone.utc).isoformat())

    def save_status():
        temporary = STATE / f'{label}_status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(STATE / f'{label}_status.json')

    save_status()
    try:
        logger = run_saved(config, experiment_id=manifest['experiment_id'],
                           experiments_root=manifest['experiments_root'])
        assert (logger.run_dir / 'final_model.npy').is_file()
        status.update(status='completed', run_dir=str(logger.run_dir))
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('hard', 'soft'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.)
    args = parser.parse_args()
    main(args.mode, args.validate_only, args.temperature)
