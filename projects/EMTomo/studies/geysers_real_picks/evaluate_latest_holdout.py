"""Evaluate fixed latest real-pick models; never update velocities or fit to holdout.

Run --fit initial/hard/soft in separate processes, then --score-only. The original
38676-pick inversion training mask is retained, excluding internal control too.
All models use the same 50-candidate MAP location protocol and likelihood noise.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
QUEUE = ROOT / 'runs/geysers_real_refine500_fmm125_20261110_110138'
INITIAL = ROOT / 'data/geysers_2011/initial_models/linear_vp_3p5_6p0/model.json'
OUT = ROOT / 'paper/exp_data/geysers_real_holdout_latest'
MODES = ('initial', 'hard', 'soft')
H = 125.
NOISE = (.01, .05, .02)
SHORTLIST, DMIN = 50, 7
_STATE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Changed input: {path}')


def mask_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def training():
    split, manifest = read(DATA / 'manifest.json'), read(QUEUE / 'manifest.json')
    for name in ('emtomo/arrivals.npz', 'emtomo/metadata.json', 'geometry.json'):
        verified(DATA / name, split['output_sha256'][name])
    source = QUEUE / 'inputs/emtomo/arrivals.npz'
    verified(source, split['output_sha256']['emtomo/arrivals.npz'])
    with np.load(source, allow_pickle=False) as arrays:
        event_ids, station_ids = arrays['event_ids'].copy(), arrays['station_ids'].copy()
        stations, observed = arrays['station_coordinates_m'].copy(), arrays['arrival_times_s'].copy()
    masks = []
    for mode in ('hard', 'soft'):
        result = read(QUEUE / f'cases/{mode}/result.json')
        path = Path(result['run_dir']) / 'control_mask.npy'
        verified(path, result['artifact_sha256']['control_mask.npy'])
        masks.append(np.load(path, allow_pickle=False))
    np.testing.assert_array_equal(*masks)
    control = masks[0]
    if control.dtype != np.bool_ or control.shape != observed.shape:
        raise ValueError('Invalid control mask')
    if mask_sha(control) != manifest['control_split']['mask_sha256']:
        raise ValueError('Control mask differs from frozen split')
    if not np.isfinite(observed[control]).all():
        raise ValueError('Control mask selects missing observations')
    train = observed.copy()
    train[control] = np.nan
    if (train.shape != (3471, 31) or np.isfinite(train).sum() != 38676
            or control.sum() != 4297 or np.isfinite(observed).sum() != 42973
            or np.isfinite(train).sum(axis=1).min() < 5):
        raise ValueError('Unexpected training/control selection')
    return event_ids, station_ids, stations, train, observed, control


def activate_runtime():
    manifest = read(QUEUE / 'manifest.json')
    for name, digest in manifest['frozen_sha256'].items():
        verified(QUEUE / name, digest)
    hashes = {name: digest for name, digest in manifest['frozen_sha256'].items()
              if name.startswith('snapshot/') and name.endswith('.py')}
    if not hashes:
        raise ValueError('No frozen numerical code')
    runtime = QUEUE / 'snapshot'
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    return hashes


def model(name):
    manifest = read(QUEUE / 'manifest.json')
    paths = [QUEUE / 'manifest.json']
    if name == 'initial':
        info = read(INITIAL)
        path = INITIAL.parent / info['array_file']
        verified(path, info['array_sha256'])
        coarse, cell = np.load(path, allow_pickle=False), info['cell_size_m']
        paths += [INITIAL, path]
        note = 'Original linear 3.5--6.0 km/s baseline, sampled on 1-km cells'
    else:
        record = QUEUE / f'cases/{name}/result.json'
        result = read(record)
        if result['status'] != 'completed':
            raise ValueError(f'{name}: incomplete run')
        directory = Path(result['run_dir'])
        for key, digest in result['artifact_sha256'].items():
            verified(directory / key, digest)
            paths.append(directory / key)
        config = read(directory / 'meta.json')['run_params']
        case = next(c for c in manifest['cases'] if c['id'] == name)
        # The logger omits output controls and adds descriptive provenance fields.
        differences = {key: (config.get(key), value) for key, value in case['config'].items()
                       if key not in ('save_runs', 'runs_dir') and config.get(key) != value}
        if differences:
            raise ValueError(f'Run/queue numerical config mismatch: {differences}')
        if (config['n_candidates'] != SHORTLIST or config['weights_min_distance'] != DMIN
                or tuple(config[k] for k in ('weight_noise_relative_sigma',
                    'weight_noise_absolute_sigma_s', 'weight_model_sigma_s')) != NOISE):
            raise ValueError('Protocol differs from inversion')
        stopping = read(directory / 'stopping.json')
        markers = sorted(int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json'))
        if markers != list(range(stopping['completed_updates'])):
            raise ValueError('Incomplete iterations')
        coarse = np.load(directory / 'final_model.npy', allow_pickle=False)
        cell = config['cell_size']
        paths += [record, directory / 'meta.json']
        note = f'{name}: final 500-m model after {stopping["completed_updates"]} continuation updates'
    if not np.isfinite(coarse).all() or np.any(coarse <= 0):
        raise ValueError('Invalid velocities')
    np.testing.assert_allclose(np.asarray(coarse.shape) * cell, [23000, 11000, 7000])
    factor = round(cell / H)
    if factor * H != cell:
        raise ValueError('Nonintegral FMM refinement')
    fine = coarse.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)
    return fine, dict(note=note, source_sha256={str(p): sha(p) for p in paths})


def fitted_origin(predicted, observed, noise):
    """Precision-weighted origin using only finite training observations."""
    selected = np.isfinite(observed)
    if selected.sum() < 2:
        raise ValueError('Insufficient training picks')
    sigma = noise.sigmas(predicted[selected])
    precision = 1 / sigma**2
    if not np.isfinite(precision).all() or np.any(precision <= 0):
        raise ValueError('Invalid noise precision')
    return float(np.average(observed[selected] - predicted[selected], weights=precision))


def fit_event(index):
    from instruments.likelihood import negative_log_likelihood
    from instruments.instruments_coords import sample_cell_centered_trilinear_batch
    from tomography.tomography_math import select_candidate_cells
    from evaluate_holdout import safe_refine
    fields, observed, noise = _STATE['fields'], _STATE['train'][index], _STATE['noise']
    selected = np.flatnonzero(np.isfinite(observed))
    cost = negative_log_likelihood(fields, observed[selected], noise, station_indices=selected)
    if not np.isfinite(cost).all():
        raise ValueError(f'Nonfinite likelihood for event {index}')
    candidates = select_candidate_cells(cost, SHORTLIST, DMIN)
    del cost
    refined = [safe_refine(fields, observed[selected], cell, noise, selected) for cell in candidates]
    position, nll, _ = min(refined, key=lambda item: item[1])
    times = sample_cell_centered_trilinear_batch(fields, position)
    origin = fitted_origin(times, observed, noise)
    return index, position, times + origin, origin, float(nll), sum(item[2] for item in refined)


def fit(name, workers):
    frozen = activate_runtime()
    from eikonal import station_travel_time_fields
    from velocity_model import VelocityModel
    from instruments.likelihood import PickNoise, negative_log_likelihood
    from tomography.tomography_math import refine_hypocentre_in_cell
    event_ids, station_ids, stations, train, _, _ = training()
    fine, provenance = model(name)
    binding = dict(model=provenance, runtime_sha256=frozen,
        training_sha256=sha(DATA / 'emtomo/arrivals.npz'), train_mask_sha256=mask_sha(np.isfinite(train)),
        evaluator_sha256=sha(Path(__file__)), fallback_sha256=sha(Path(__file__).with_name('evaluate_holdout.py')),
        grid_m=H, shortlist=SHORTLIST, dmin=DMIN, noise=list(NOISE),
        origin='TRAIN-only precision-weighted offset with all three likelihood noise components',
        prediction='MAP location, identical protocol for initial/hard/soft; no velocity updates',
        training_picks=int(np.isfinite(train).sum()), internal_control_used_for_location=False,
        fallback='Known Powell zero-direction error only: bounded coordinate descent')
    metadata, arrays_path = OUT / f'{name}_fit.json', OUT / f'{name}_predictions.npz'
    if metadata.exists():
        old = read(metadata)
        if old['binding'] != binding:
            raise ValueError('Existing fit has different provenance; choose a new output')
        verified(arrays_path, old['predictions_sha256'])
        print(f'{name}: verified existing predictions', flush=True)
        return
    started = time.monotonic()
    print(f'{name}: computing 31 station fields, {workers} workers', flush=True)
    fields = station_travel_time_fields(VelocityModel(fine, H), stations, workers)
    del fine
    if not np.isfinite(fields).all():
        raise ValueError('Nonfinite FMM fields')
    noise = PickNoise(*NOISE)
    # JIT once before fork; each worker is single-threaded.
    toy = np.zeros((3, 2, 2, 2), dtype=np.float32)
    negative_log_likelihood(toy, np.zeros(3), noise, station_indices=np.arange(3))
    refine_hypocentre_in_cell(toy, np.zeros(3), (0, 0, 0), noise, station_indices=np.arange(3))
    _STATE.update(fields=fields, train=train, noise=noise)
    predictions = np.empty_like(train)
    positions = np.empty((len(event_ids), 3))
    origins, costs = np.empty(len(event_ids)), np.empty(len(event_ids))
    fallbacks = np.zeros(len(event_ids), dtype=int)
    print(f'{name}: training-only MAP fits for {len(event_ids)} events', flush=True)
    with mp.get_context('fork').Pool(workers) as pool:
        for done, (i, position, predicted, origin, cost, fallback) in enumerate(
                pool.imap_unordered(fit_event, range(len(event_ids))), 1):
            predictions[i], positions[i], origins[i], costs[i], fallbacks[i] = predicted, position, origin, cost, fallback
            if done % 250 == 0 or done == len(event_ids):
                elapsed = time.monotonic() - started
                print(f'{name}: {done}/{len(event_ids)}, {elapsed:.1f}s', flush=True)
                write_json(OUT / f'{name}_progress.json', dict(completed=done, total=len(event_ids), elapsed_s=elapsed))
    _STATE.clear()
    for values in (predictions, positions, origins, costs):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite fitted values')
    temporary = arrays_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, predicted_relative_s=predictions, position_m=(positions + .5) * H,
            origin_relative_s=origins, train_nll=costs, candidate_fallback_counts=fallbacks,
            event_ids=event_ids, station_ids=station_ids)
    temporary.replace(arrays_path)
    for path, digest in provenance['source_sha256'].items():
        verified(path, digest)
    write_json(metadata, dict(binding=binding, predictions_sha256=sha(arrays_path),
        elapsed_s=time.monotonic() - started, holdout_opened=False,
        candidate_fallbacks=int(fallbacks.sum()), events_with_fallback=int(np.count_nonzero(fallbacks))))
    print(f'{name}: predictions saved, holdout not opened', flush=True)


def heldout_indices(held, event_ids, station_ids, original):
    events = {str(value): i for i, value in enumerate(event_ids)}
    stations = {str(value): i for i, value in enumerate(station_ids)}
    keys = [(row['event_id'], row['network'] + '.' + row['station']) for row in held]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate held-out event/station pair')
    ei = np.array([events[e] for e, _ in keys])
    si = np.array([stations[s] for _, s in keys])
    if not np.isnan(original[ei, si]).all():
        raise ValueError('External holdout overlaps training or internal control')
    observed = np.array([float(row['seconds_from_train_reference']) for row in held])
    if not np.isfinite(observed).all():
        raise ValueError('Nonfinite held-out observation')
    return ei, si, observed


def paired_bootstrap(event_indices, hard, soft, samples=2000, seed=42):
    """Resample scored events, retaining all their picks and model pairing."""
    _, inverse = np.unique(event_indices, return_inverse=True)
    n = inverse.max() + 1
    count = np.bincount(inverse)
    hard_ss = np.bincount(inverse, weights=np.square(hard))
    soft_ss = np.bincount(inverse, weights=np.square(soft))
    rng = np.random.default_rng(seed)
    differences, gains = [], []
    for _ in range(samples):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)
        denominator = np.dot(weights, count)
        h = np.sqrt(np.dot(weights, hard_ss) / denominator)
        s = np.sqrt(np.dot(weights, soft_ss) / denominator)
        differences.append(s - h)
        gains.append(100 * (1 - s / h))
    return dict(samples=samples, seed=seed, resampling_unit='event with all its held-out picks',
        soft_minus_hard_rmse_s_ci95=np.percentile(differences, [2.5, 97.5]).tolist(),
        soft_rmse_gain_pct_ci95=np.percentile(gains, [2.5, 97.5]).tolist(),
        caveat='Conditional on this fixed split and fitted models; no independent-event or tuning-adjusted inference')


def score():
    from cv_metrics import summarize
    event_ids, station_ids, _, train, original, control = training()
    ledgers, predictions = {}, {}
    # Verify all TRAIN-only fits before first opening the external holdout.
    for name in MODES:
        ledger = read(OUT / f'{name}_fit.json')
        if ledger['holdout_opened'] is not False:
            raise ValueError('Fit opened holdout')
        b = ledger['binding']
        if (b['train_mask_sha256'] != mask_sha(np.isfinite(train))
                or b['evaluator_sha256'] != sha(Path(__file__))
                or b['noise'] != list(NOISE) or b['shortlist'] != SHORTLIST or b['dmin'] != DMIN):
            raise ValueError('Stale or incompatible predictions')
        for path, digest in b['model']['source_sha256'].items():
            verified(path, digest)
        path = OUT / f'{name}_predictions.npz'
        verified(path, ledger['predictions_sha256'])
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['event_ids'], event_ids)
            np.testing.assert_array_equal(a['station_ids'], station_ids)
            predictions[name] = a['predicted_relative_s'].copy()
        ledgers[name] = ledger
    split = read(DATA / 'manifest.json')
    verified(DATA / 'holdout.csv', split['output_sha256']['holdout.csv'])
    with (DATA / 'holdout.csv').open() as stream:
        held = list(csv.DictReader(stream))
    ei, si, observed = heldout_indices(held, event_ids, station_ids, original)
    if len(held) != 4775 or len(np.unique(ei)) != 3186:
        raise ValueError('Unexpected external holdout')
    report = dict(protocol='Fixed velocity models; identical 50-candidate/dmin7 MAP relocation and weighted origin from actual38676 TRAIN picks only; 125-m FMM.',
        scope='Velocity-model prediction test, not a soft posterior-predictive score; no inversion rerun.',
        training_picks=38676, internal_control_picks=4297, external_holdout_picks=len(held),
        external_holdout_events=len(np.unique(ei)), likelihood_noise=list(NOISE), models={},
        caveats=['The external split was inspected for earlier models and is not a pristine tuning-independent test.',
                 'These are held-out picks of training events, not independent held-out events.'],
        source_sha256={str(DATA / 'holdout.csv'): sha(DATA / 'holdout.csv'),
                       str(DATA / 'manifest.json'): sha(DATA / 'manifest.json'),
                       str(Path(__file__)): sha(Path(__file__))})
    errors, table, pick_rows = {}, [], []
    for name in MODES:
        predicted = predictions[name]
        errors[name] = predicted[ei, si] - observed
        h = summarize(errors[name])
        c = summarize((predicted - original)[control])
        t = summarize((predicted - train)[np.isfinite(train)])
        report['models'][name] = dict(holdout=h, control=c, train=t,
            binding=ledgers[name]['binding'], candidate_fallbacks=ledgers[name]['candidate_fallbacks'])
        if name != 'initial':
            result = read(QUEUE / f'cases/{name}/result.json')
            stopping = read(Path(result['run_dir']) / 'stopping.json')
            np.testing.assert_allclose(c['rmse_s'], stopping['control_rmse_s'], rtol=0, atol=1e-9)
        table.append(dict(model=name, training_picks=38676, holdout_picks=len(held),
            train_rmse_s=t['rmse_s'], control_rmse_s=c['rmse_s'], **{'holdout_' + k: v for k, v in h.items()}))
        for j, row in enumerate(held):
            pick_rows.append(dict(model=name, event_id=row['event_id'], station_id=str(station_ids[si[j]]),
                observed_relative_s=observed[j], predicted_relative_s=predicted[ei[j], si[j]], residual_s=errors[name][j]))
    initial = report['models']['initial']['holdout']['rmse_s']
    hard, soft = (report['models'][name]['holdout']['rmse_s'] for name in ('hard', 'soft'))
    report['soft_gain_vs_hard_pct'] = 100 * (1 - soft / hard)
    for row in table:
        row['holdout_rmse_gain_vs_initial_pct'] = 100 * (1 - row['holdout_rmse_s'] / initial)
        report['models'][row['model']]['holdout_rmse_gain_vs_initial_pct'] = row['holdout_rmse_gain_vs_initial_pct']
    report['paired_event_bootstrap'] = paired_bootstrap(ei, errors['hard'], errors['soft'])
    for filename, records in [('summary.csv', table), ('holdout_predictions.csv', pick_rows)]:
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    write_json(OUT / 'summary.json', report)
    print(json.dumps(dict(metrics=table, gain_pct=report['soft_gain_vs_hard_pct'],
                          bootstrap=report['paired_event_bootstrap']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fit', choices=MODES)
    group.add_argument('--score-only', action='store_true')
    parser.add_argument('--workers', type=int, default=40)
    args = parser.parse_args()
    if not OUT.is_dir():
        raise FileNotFoundError(OUT)
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        raise ValueError('Invalid worker count for available CPUs')
    if args.fit:
        fit(args.fit, args.workers)
    else:
        score()


if __name__ == '__main__':
    main()
