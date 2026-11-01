"""Validate prepared hard/soft input; --execute is required to start an inversion.

Real pick uncertainty is not a synthetic noise injection. Supply a likelihood
sigma explicitly at launch, chosen from training data only, never from holdout.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import InversionConfig
from experiment_runner import run_real_experiment
from velocity_model import VelocityModel
from studies.geysers_real_picks.finalize import DATA, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('hard', 'soft'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--absolute-pick-sigma-s', type=float)
    parser.add_argument('--relative-pick-sigma', type=float, default=0.)
    args = parser.parse_args(argv)
    if args.execute and args.absolute_pick_sigma_s is None:
        parser.error('--execute requires an explicit --absolute-pick-sigma-s')
    methods = args.data / 'methods'
    manifest = json.loads((methods / 'manifest.json').read_text())
    if sha(args.data / 'manifest.json') != manifest['split_manifest_sha256']:
        raise ValueError('Split manifest changed')
    # The EMTomo launcher does not open the catalog, holdout, or native DD files.
    for name in ('emtomo_configs.json', 'initial_velocity_m_s.npy', 'regional_model.json'):
        if sha(methods / name) != manifest['output_sha256'][name]:
            raise ValueError(f'Method artifact changed: {name}')
    split = json.loads((args.data / 'manifest.json').read_text())
    for name in ('emtomo/arrivals.npz', 'emtomo/metadata.json'):
        if sha(args.data / name) != split['output_sha256'][name]:
            raise ValueError(f'Training artifact changed: {name}')
    config = InversionConfig(**json.loads((methods / 'emtomo_configs.json').read_text())[args.mode])
    if args.absolute_pick_sigma_s is not None:
        config = replace(config, weight_noise_absolute_sigma_s=args.absolute_pick_sigma_s,
                         weight_noise_relative_sigma=args.relative_pick_sigma)
    start = VelocityModel(np.load(methods / 'initial_velocity_m_s.npy', allow_pickle=False), config.cell_size)
    result = run_real_experiment(args.data / 'emtomo', config, initial_model=start, validate_only=not args.execute)
    if not args.execute:
        print(json.dumps({'mode': args.mode, 'validation': 'passed', 'solver_launched': False,
                          'events': len(result.event_ids), 'stations': len(result.station_ids),
                          'observed_picks': int(np.isfinite(result.arrivals_table).sum()),
                          'initial_shape': list(result.initial_model.shape),
                          'catalog_input': False, 'truth_metrics': False,
                          'pick_sigma_status': 'explicit' if args.absolute_pick_sigma_s is not None else 'pending'}, indent=2))
    return result


if __name__ == '__main__':
    main()
