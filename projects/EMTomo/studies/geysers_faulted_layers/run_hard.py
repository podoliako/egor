"""Run the last Geysers hard configuration on the clean faulted-layer data.

Uses the frozen numerical runtime of the completed pick-noise series, not
potentially changed live code. --validate-only does not run EM or write runs.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'runs/geysers_pick_noise_20261027/manifest.json'
EXPERIMENT_ID = 'geysers11_20110401-20110420_fault3_x6p5_elev1166_26st_3776ev_clean'
EXPERIMENTS = ROOT.parent / 'forward_modeling/experiments'
STATE = ROOT / 'runs/geysers_faulted_layers'


def main(validate_only=False):
    manifest = json.loads(MANIFEST.read_text())
    runtime = Path(manifest['runtime_root'])
    for name, expected in manifest['runtime_sha256'].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen runtime changed: {name}')
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'projects/EMTomo'))
    from config import InversionConfig
    from main import main as run_saved
    import numpy as np

    generation = json.loads((EXPERIMENTS / 'input' / EXPERIMENT_ID / 'generation.json').read_text())
    parameters = dict(manifest['references']['hard-k15-top1-dmin1']['config'])
    initial = generation['parameters']['initial']
    parameters.update(initial_layer_boundaries_km=tuple(initial['boundaries_km']),
                      initial_layer_velocities_m_s=tuple(v * 1000 for v in initial['vp_km_s']),
                      weight_noise_relative_sigma=0., weight_noise_absolute_sigma_s=0.,
                      run_name='gfault28-hard-k15-top1-dmin1-clean', runs_dir=str(ROOT / 'runs'))
    config = InversionConfig(**parameters)
    prepared = run_saved(config, experiment_id=EXPERIMENT_ID,
                         experiments_root=EXPERIMENTS, validate_only=True)
    assert prepared.initial_model.shape == (20, 7, 7)
    assert prepared.arrivals_table.shape == (3776, 26)
    assert prepared.noise_sigmas == (0., 0.)
    np.testing.assert_array_equal(prepared.initial_model.velocity[0, 0],
                                  [3850., 4550., 5300., 5300., 5300., 5300., 5300.])
    print(json.dumps(asdict(config), indent=2), flush=True)
    if validate_only:
        print('Validation passed; no EM, no JIT warmup, no run output.', flush=True)
        return
    input_dir = EXPERIMENTS / 'input' / EXPERIMENT_ID
    output_dir = EXPERIMENTS / 'output' / EXPERIMENT_ID
    sources = [input_dir / name for name in ('model.npz', 'stations.csv', 'events.csv', 'generation.json')]
    sources += [output_dir / name for name in ('arrivals.csv', 'metadata.json')]
    status = dict(status='running', experiment_id=EXPERIMENT_ID, config=asdict(config),
                  runtime_root=str(runtime), started_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def save_status():
        temporary = STATE / 'hard_status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(STATE / 'hard_status.json')

    save_status()
    try:
        logger = run_saved(config, experiment_id=EXPERIMENT_ID, experiments_root=EXPERIMENTS)
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
    parser.add_argument('--validate-only', action='store_true')
    main(parser.parse_args().validate_only)
