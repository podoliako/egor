"""One frozen-runtime inversion from the prepared screen manifest."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def main(state, identifier, validate_only=False):
    manifest = json.loads((state / 'manifest.json').read_text())
    case = next(c for c in manifest['cases'] if c['id'] == identifier)
    runtime = Path(manifest['runtime_root'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == case['experiment_id'])
    for name, digest in manifest['runtime_sha256'].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen runtime changed: {name}')
    for name, digest in dataset['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Experiment changed: {name}')
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    from config import InversionConfig
    from experiment_runner import prepare_inversion, resolve_pick_noise
    from tomography.tomography_em import run_em, warm_up_jit
    from tomography.tomography_logging import TomographyLogger
    import numpy as np

    config = InversionConfig(**case['config'])
    prepared = prepare_inversion(case['experiment_id'], config, manifest['experiments_root'])
    assert prepared.arrivals_table.shape == (3776, dataset['count'])
    assert prepared.noise_sigmas == (dataset['noise_sigma'], dataset['noise_sigma'])
    noise = resolve_pick_noise(config, prepared.noise_sigmas)
    assert noise.relative_sigma == noise.absolute_sigma_s == dataset['noise_sigma']
    np.testing.assert_array_equal(prepared.initial_model.velocity[0, 0],
                                  [3850., 4550., 5300., 5300., 5300., 5300., 5300.])
    print(f'Validated {identifier}: {case["experiment_id"]}', flush=True)
    if validate_only:
        print(json.dumps(asdict(config), indent=2), flush=True)
        return
    directory = state / 'cases' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('run_dir') or previous.get('status') != 'running':
            raise FileExistsError(f'Refusing to overwrite attempted case {identifier}')
    result = dict(status='running', case_id=identifier, config=asdict(config),
                  hard_control=case.get('hard_control'), experiment_id=case['experiment_id'],
                  started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)

    save()
    try:
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                  run_version=config.run_version,
                                  run_tags=dict(mode=config.candidate_mode, ncand=config.n_candidates,
                                                topn=config.weights_top_n, dmin=config.weights_min_distance,
                                                cov=config.coverage_damping_power))
        result['run_dir'] = str(logger.run_dir)
        save()
        source_model = Path(manifest['experiments_root']) / 'input' / case['experiment_id'] / 'model.npz'
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs, noise,
               reference_model=prepared.reference_model,
               source_experiment=dict(id=case['experiment_id'], model_sha256=dataset['sha256'][str(source_model)]),
               run_name=config.run_name, logger=logger)
        final = np.load(logger.run_dir / 'final_model.npy', allow_pickle=False)
        assert final.shape == prepared.initial_model.shape and np.isfinite(final).all() and (final > 0).all()
        assert all((logger.run_dir / f'iter_{i}/complete.json').exists() for i in range(config.n_cycles))
        if config.temperature_schedule is not None:
            for i, temperature in enumerate(config.temperature_schedule):
                assert json.loads((logger.run_dir / f'iter_{i}/temperature.json').read_text())['temperature'] == temperature
        result.update(status='completed', final_rms_m_s=float(np.sqrt(np.mean(
                      (final - prepared.reference_model.velocity) ** 2))))
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    main(args.state.resolve(), args.case, args.validate_only)
