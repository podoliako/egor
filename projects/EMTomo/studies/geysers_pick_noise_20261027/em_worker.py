"""One fresh-process EM inversion; imports numerical code only after activation.

There is deliberately no resume: a case result is a durable attempt record, even
when validation, a solve, or an external stop fails partway through that attempt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path('/mnt/disk01/egor/projects/EMTomo')
SHAPE = (3776, 26)
GRID = (20, 7, 7)
INPUTS = ('model.npz', 'stations.csv', 'events.csv', 'generation.json')
OUTPUTS = ('arrivals.csv', 'metadata.json')
LOGGING_FIELDS = {'run_name', 'runs_dir'}
MODES = {'hard-k15-top1-dmin1': ('hard', 1, 1),
         'soft-k15-top10-dmin1': ('soft', 10, 1),
         'soft-k15-top10-dmin2': ('soft', 10, 2)}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def verify_forward(manifest, common=None):
    """Hash-only gate shared with the supervisor (no numerical/runtime imports)."""
    common = common or importlib.import_module('common')
    common.verify_manifest(manifest)
    state = Path(manifest['state_dir'])
    binding = manifest_digest(manifest)
    cache = common.read(state / 'forward' / 'clean_times.json')
    ledger = common.read(state / 'forward_complete.json')
    clean_hash = common.sha(state / 'forward' / 'clean_times.npz')
    if (cache.get('status') != 'completed' or cache.get('manifest_digest') != binding
            or cache.get('shape') != list(SHAPE) or cache.get('dtype') != 'float64'
            or cache.get('time_reference') != 'absolute_propagation_time_s'
            or cache.get('source_input_sha256') != manifest['source_input_sha256']
            or cache.get('runtime_sha256') != manifest['runtime_sha256']
            or cache.get('clean_sha256') != clean_hash
            or ledger.get('status') != 'completed' or ledger.get('manifest_digest') != binding
            or ledger.get('clean_sha256') != clean_hash):
        raise ValueError('Invalid or modified clean forward cache/completion ledger')
    baseline = ledger.get('baseline_reproduction', {})
    if (baseline.get('status') != 'completed' or baseline.get('allclose') is not True
            or baseline.get('pairs') != SHAPE[0] * SHAPE[1]
            or not 0 <= baseline.get('max_abs_difference_s', float('inf')) <= 1e-8):
        raise ValueError('Forward baseline reproduction gate was not passed')
    if set(ledger.get('levels', {})) != {level['id'] for level in manifest['levels']}:
        raise ValueError('Incomplete forward levels')
    root = Path(manifest['experiments_root'])
    for level in manifest['levels']:
        record = ledger['levels'][level['id']]
        if (record.get('status') != 'completed' or record.get('manifest_digest') != binding
                or record.get('experiment_id') != level['experiment_id']
                or record.get('level_id') != level['id']
                or record.get('pairs') != SHAPE[0] * SHAPE[1]
                or record.get('input_sha256') != manifest['source_input_sha256']):
            raise ValueError(f"Invalid forward level ledger: {level['id']}")
        for category, names in (('input', INPUTS), ('output', OUTPUTS)):
            actual = {name: common.sha(root / category / level['experiment_id'] / name) for name in names}
            if actual != record.get(category + '_sha256'):
                raise ValueError(f"Changed forward {category}: {level['id']}")
    return ledger


def build_config(manifest, case, config_type):
    """Equality is over every config field, not merely a shortlist of settings."""
    reference = manifest['references'][case['variant']]
    original = canonical(reference['config'])
    specified = canonical(case['config'])
    config = config_type(**specified)
    effective = canonical(asdict(config))
    for key in set(original) | set(effective) | set(specified):
        if key not in LOGGING_FIELDS and (key not in original or key not in effective
                                         or original[key] != effective[key]):
            raise ValueError(f'Scientific configuration changed: {key}')
    expected = dict(n_cycles=12, n_candidates=15, n_workers=12, save_runs=True,
                    weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.01,
                    weight_model_sigma_s=.02)
    for key, value in expected.items():
        if effective.get(key) != value:
            raise ValueError(f'Unapproved configuration: {key}')
    if (config.candidate_mode, config.weights_top_n, config.weights_min_distance) != MODES[case['variant']]:
        raise ValueError('Candidate variant mismatch')
    if Path(config.runs_dir).resolve() != PROJECT / 'runs' or not config.run_name.startswith('gpnoise27-'):
        raise ValueError('Run output must be viewer-discoverable under project/runs/run_gpnoise27-*')
    if Path(case['reference_run']).resolve() != Path(reference['run_dir']).resolve():
        raise ValueError('Case reference run mismatch')
    return config


def prepare_case(manifest, case, common):
    # load_runtime must establish frozen paths before importing config, EM or numpy.
    runtime = common.load_runtime(manifest)
    np = importlib.import_module('numpy')
    config = build_config(manifest, case, runtime.config_module.InversionConfig)
    level = next(level for level in manifest['levels'] if level['id'] == case['level_id'])
    if case['experiment_id'] != level['experiment_id']:
        raise ValueError('Case experiment does not match its noise level')
    data = runtime.runner.prepare_inversion(case['experiment_id'], config,
                                           experiments_root=Path(manifest['experiments_root']))
    if (len(data.event_ids), len(data.station_ids)) != SHAPE or np.shape(data.arrivals_table) != SHAPE:
        raise ValueError('Expected exactly 3776 events x 26 stations')
    actual_noise = (level['relative_sigma'], level['absolute_sigma_s'])
    if tuple(data.noise_sigmas) != actual_noise:
        raise ValueError('Prepared noise sigmas differ from the actual noise level')
    initial = np.asarray(data.initial_model.velocity)
    baseline = np.load(Path(case['reference_run']) / 'initial_model.npy', allow_pickle=False)
    if (initial.shape != GRID or initial.dtype != baseline.dtype
            or not np.array_equal(initial, baseline) or not np.isfinite(initial).all() or not (initial > 0).all()):
        raise ValueError('Initial model differs from the baseline initial_model.npy')
    noise = runtime.runner.resolve_pick_noise(config, data.noise_sigmas)
    if (noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s) != (.01, .01, .02):
        raise ValueError('Fixed likelihood noise resolver changed')
    return runtime, np, config, data, noise, level


def validate_outputs(run_dir, np, common):
    final_path = run_dir / 'final_model.npy'
    final = np.load(final_path, allow_pickle=False)
    if final.shape != GRID or not np.isfinite(final).all() or not (final > 0).all():
        raise ValueError('Invalid final velocity model')
    quality_path = run_dir / 'quality.jsonl'
    quality = [json.loads(line) for line in quality_path.read_text().splitlines() if line.strip()]
    if [row['iter'] for row in quality] != list(range(12)):
        raise ValueError('Incomplete quality history: expected iterations 0..11')
    # Reject non-finite metrics before publishing a successful JSON result.
    canonical(quality)
    for step in range(12):
        cycle = run_dir / f'iter_{step}'
        if common.read(cycle / 'complete.json').get('iter') != step:
            raise ValueError(f'Incomplete iteration {step}')
        for event in range(SHAPE[0]):
            if not (cycle / f'event_{event}' / 'weights.npz').is_file():
                raise ValueError(f'Missing event weights: {step}/{event}')
    return dict(final_model_sha256=common.sha(final_path), quality_sha256=common.sha(quality_path),
                final_quality=quality[-1], quality=quality, completed_cycles=12,
                event_weights_per_cycle=SHAPE[0])


def run(manifest, case_id, common, *, manifest_path):
    matches = [case for case in manifest['cases'] if case['id'] == case_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown or duplicate case ID: {case_id}')
    case = matches[0]
    directory = Path(manifest['state_dir']) / 'cases' / case_id
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / 'worker.lock', os.O_CREAT | os.O_RDWR, 0o600)
    owned = False
    result_path = directory / 'result.json'
    started = time.monotonic()
    result = dict(case_id=case_id, level_id=case['level_id'], variant=case['variant'], status='running',
                  started_utc=common.now(), worker_pid=os.getpid(), run_dir=None)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(result_path):
            raise FileExistsError('Existing case result; refusing resume or overwrite')
        owned = True
        common.atomic(result_path, result)
        ledger = verify_forward(manifest, common)
        runtime, np, config, data, noise, level = prepare_case(manifest, case, common)
        provenance = dict(
            study_id=manifest['study_id'], case_id=case_id,
            manifest_path=str(manifest_path), manifest_sha256=common.sha(manifest_path),
            manifest_digest=manifest_digest(manifest), runtime_root=manifest['runtime_root'],
            runtime_sha256=manifest['runtime_sha256'], runner_sha256=manifest['runner_sha256'],
            source_input_sha256=manifest['source_input_sha256'],
            source_output_sha256=manifest['source_output_sha256'],
            reference_run=case['reference_run'], reference_case_id=manifest['references'][case['variant']]['case_id'],
            initial_model_sha256=common.sha(Path(case['reference_run']) / 'initial_model.npy'),
            actual_noise=dict(relative_sigma=level['relative_sigma'], absolute_sigma_s=level['absolute_sigma_s'],
                              seed=level['seed'], multiplier=level['multiplier']),
            fixed_assumed_noise=dict(relative_sigma=.01, absolute_sigma_s=.01, model_sigma_s=.02),
            forward_output_sha256=ledger['levels'][level['id']]['output_sha256'],
            clean_sha256=ledger['clean_sha256'], effective_config=canonical(asdict(config)),
            final_locations_are_pre_last_velocity_update=True)

        class StudyLogger(runtime.em.TomographyLogger):
            def save_meta(self, **kwargs):
                kwargs['run_params']['provenance'] = provenance
                super().save_meta(**kwargs)

        logger = StudyLogger(base_dir=config.runs_dir, run_name=config.run_name, run_version=config.run_version,
                             run_tags={'ncand': config.n_candidates, 'topn': config.weights_top_n,
                                       'mode': config.candidate_mode, 'dmin': config.weights_min_distance})
        result.update(run_dir=str(logger.run_dir.resolve()), config=canonical(asdict(config)), provenance=provenance)
        # A live viewer and the supervisor can find this run before the first EM cycle.
        common.atomic(result_path, result)
        print(f"Case {case_id}; run directory: {logger.run_dir}", flush=True)
        runtime.em.warm_up_jit()
        runtime.em.run_em(config, data.initial_model, data.arrivals_table, data.station_locs, noise,
                          reference_model=data.reference_model,
                          source_experiment={'id': case['experiment_id'],
                                             'model_sha256': manifest['source_input_sha256']['model.npz']},
                          run_name=config.run_name, logger=logger)
        metrics = validate_outputs(logger.run_dir, np, common)
        verify_forward(manifest, common)
        result.update(status='completed', **metrics)
    except BaseException as error:
        if owned:
            result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if owned:
                result.update(finished_utc=common.now(), elapsed_s=time.monotonic() - started)
                common.atomic(result_path, result)
        finally:
            os.close(fd)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--case-id', required=True)
    args = parser.parse_args(argv)
    if not args.manifest.is_absolute():
        parser.error('--manifest must be absolute')
    sys.dont_write_bytecode = True
    common = importlib.import_module('common')
    run(common.read(args.manifest), args.case_id, common, manifest_path=args.manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
