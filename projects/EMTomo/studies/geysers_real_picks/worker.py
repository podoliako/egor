"""One hashed real-data inversion; lifecycle is managed by the existing pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_native(work, expected_ids):
    import numpy as np
    log = (work / 'stdout.log').read_text(errors='replace')
    if 'writing out results' not in log or log.count('updating the velocities') != 3:
        raise ValueError('Native solver did not complete three velocity updates')
    shape = tuple(map(int, (work / 'MOD').read_text().splitlines()[0].split()[1:]))
    velocity = np.fromstring((work / 'Vp_model.dat').read_text(), sep=' ')
    if velocity.size != int(np.prod(shape)) or not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError('Invalid final native Vp model')
    ids = set()
    for line in (work / 'tomoDD.reloc').read_text().splitlines():
        fields = line.split()
        event = int(fields[0])
        if event in ids or not np.isfinite([float(x) for x in fields[1:10]]).all():
            raise ValueError('Duplicate or nonfinite native relocation')
        ids.add(event)
    if not ids or not ids <= expected_ids:
        raise ValueError('Unexpected native event IDs')
    return dict(retained_events=len(ids), missing_event_ids=sorted(expected_ids - ids),
                vp_min_km_s=float(velocity.min()), vp_max_km_s=float(velocity.max()))


def main(state, identifier, validate_only=False):
    manifest = json.loads((state / 'manifest.json').read_text())
    case = next(c for c in manifest['cases'] if c['id'] == identifier)
    for name, digest in manifest['frozen_sha256'].items():
        if sha(state / name) != digest:
            raise ValueError(f'Frozen input/runtime changed: {name}')
    runtime = state / 'snapshot'
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    import numpy as np
    directory = state / 'cases' / identifier
    if identifier != 'tomodd':
        from config import InversionConfig
        from experiment_runner import prepare_real_inversion, resolve_pick_noise
        from velocity_model import VelocityModel
        config = InversionConfig(**case['config'])
        initial = VelocityModel(np.load(state / 'inputs/initial_velocity_m_s.npy'), config.cell_size)
        prepared = prepare_real_inversion(state / 'inputs/emtomo', config, initial_model=initial)
        noise = resolve_pick_noise(config, prepared.noise_sigmas)
        if (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) != (.01, 0., .02):
            raise ValueError('Unexpected real-pick likelihood noise')
        if prepared.arrivals_table.shape != (3471, 31) or np.isfinite(prepared.arrivals_table).sum() != 42973:
            raise ValueError('Unexpected training selection')
    if validate_only:
        print(f'Validated {identifier}; solver not launched', flush=True)
        return
    path = directory / 'result.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('run_dir') or previous.get('status') != 'running':
            raise FileExistsError('Refusing to overwrite an attempted inversion')
    result = dict(status='running', case_id=identifier, config=case['config'], started_at=time.time())
    write(path, result)
    try:
        if identifier == 'tomodd':
            work = directory / 'work'
            if (work / 'stdout.log').exists():
                raise FileExistsError('Native work directory was already executed')
            result['run_dir'] = str(work)
            write(path, result)
            with (work / 'stdout.log').open('x') as log:
                completed = subprocess.run([str(state / 'solver/tomoDD-SE'), 'tomoDD-SE.inp'],
                                           cwd=work, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=manifest['case_timeout_s'] - 60)
            if completed.returncode:
                raise RuntimeError(f'Native solver exit code {completed.returncode}')
            expected = {int(e) for e in manifest['event_ids']}
            result.update(check_native(work, expected))
            for name in case['completion_artifacts']:
                if not (work / name).is_file() or not (work / name).stat().st_size:
                    raise ValueError(f'Missing native completion artifact: {name}')
            result['artifact_sha256'] = {n: sha(work / n) for n in case['completion_artifacts']}
        else:
            from tomography.tomography_em import run_em, warm_up_jit
            from tomography.tomography_logging import TomographyLogger
            warm_up_jit()
            logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                      run_version=config.run_version,
                                      run_tags=dict(mode=config.candidate_mode, ncand=config.n_candidates,
                                                    topn=config.weights_top_n, dmin=config.weights_min_distance,
                                                    cov=config.coverage_damping_power))
            result['run_dir'] = str(logger.run_dir)
            write(path, result)
            run_em(config, initial, prepared.arrivals_table, prepared.station_locs, noise,
                   reference_model=None, logger=logger,
                   source_experiment=dict(kind='real_picks', id='geysers_real_v1_train90',
                                          queue_manifest=str(state / 'manifest.json'),
                                          arrays_sha256=sha(state / 'inputs/emtomo/arrivals.npz'),
                                          event_ids=list(prepared.event_ids), station_ids=list(prepared.station_ids)))
            final = np.load(logger.run_dir / 'final_model.npy', allow_pickle=False)
            if final.shape != initial.shape or not np.isfinite(final).all() or np.any(final <= 0):
                raise ValueError('Invalid final EMTomo velocity')
            if not all((logger.run_dir / f'iter_{i}/complete.json').is_file() for i in range(config.n_cycles)):
                raise ValueError('Incomplete EMTomo iteration history')
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_at'] = time.time()
        write(path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    main(args.state.resolve(), args.case, args.validate_only)
