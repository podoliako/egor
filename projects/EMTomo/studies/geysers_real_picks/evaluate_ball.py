"""Evaluate fixed ball-holdout models with a common within-event pick split.

All ball events were excluded from velocity inversion. Locations and origin times
are fitted on 80% of their finite picks; remaining picks assess prediction.
Catalog coordinates are used only for comparison, never for localization.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                      'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
import numpy as np
import evaluate_latest_holdout as locator

ROOT = Path(__file__).resolve().parents[2]
MODES = ('initial', 'soft', 'hard')


def split_picks(observed, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.zeros(observed.shape, dtype=bool)
    for row, values in enumerate(observed):
        finite = np.flatnonzero(np.isfinite(values))
        count = min(max(1, round(len(finite) * .2)), len(finite) - 5)
        if count < 1:
            raise ValueError('Every held-out event needs at least six picks')
        mask[row, rng.choice(finite, count, replace=False)] = True
    return mask


def paired_interval(initial, candidate, mask):
    a = np.where(mask, initial**2, 0).sum(axis=1)
    b = np.where(mask, candidate**2, 0).sum(axis=1)
    counts = mask.sum(axis=1)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(a), size=(5000, len(a)))
    delta = np.sqrt(b[indices].sum(axis=1) / counts[indices].sum(axis=1)) - np.sqrt(
        a[indices].sum(axis=1) / counts[indices].sum(axis=1))
    return np.quantile(delta, [.025, .975]).tolist()


def evaluate(queue, out, workers):
    out.mkdir(parents=True, exist_ok=False)
    locator.QUEUE = queue
    frozen = locator.activate_runtime()
    from eikonal import station_travel_time_fields
    from velocity_model import VelocityModel
    from instruments.likelihood import PickNoise, negative_log_likelihood
    from tomography.tomography_math import refine_hypocentre_in_cell

    manifest = locator.read(queue / 'manifest.json')
    if locator.read(queue / 'status.json')['status'] != 'completed':
        raise ValueError('Queue is not completed')
    source = queue / 'heldout/arrivals.npz'
    with np.load(source, allow_pickle=False) as arrays:
        observed = arrays['arrival_times_s'].copy()
        event_ids, station_ids = arrays['event_ids'].copy(), arrays['station_ids'].copy()
        stations, catalog = arrays['station_coordinates_m'].copy(), arrays['catalog_xyz_m'].copy()
    ball = locator.read(queue / 'heldout/ball.json')
    with np.load(queue / 'inputs/emtomo/arrivals.npz', allow_pickle=False) as arrays:
        if set(event_ids.astype(str)) & set(arrays['event_ids'].astype(str)):
            raise ValueError('Held-out events overlap inversion training')
    validation = split_picks(observed)
    train = observed.copy()
    train[validation] = np.nan
    np.savez_compressed(out / 'observations.npz', event_ids=event_ids, station_ids=station_ids,
                        observed_s=observed, validation_mask=validation, catalog_xyz_m=catalog,
                        centre_m=ball['centre_m'], radius_m=ball['radius_m'])
    protocol = dict(velocity='Fixed saved initial/final models; no inversion updates',
                    location='MAP K50/dmin7; same protocol for all models, no catalog start or ball constraint',
                    pick_split='Per event, random 20% rounded, at least 1 validation and 5 fitting picks; seed42',
                    origin='Precision-weighted on fitting picks only', grid_m=125, noise=list(locator.NOISE),
                    catalog='Reference only, not ground truth; not used for location fitting',
                    external_holdout='4775 external holdout picks not opened',
                    n_events=len(event_ids), n_fit_picks=int(np.isfinite(train).sum()),
                    n_validation_picks=int(validation.sum()),
                    caveat='Single spatial block and one pick split; bootstrap intervals are conditional on this block')
    binding = dict(queue=str(queue), evaluator_sha256=locator.sha(Path(__file__)),
                   locator_sha256=locator.sha(Path(locator.__file__)),
                   refinement_sha256=locator.sha(Path(__file__).with_name('evaluate_holdout.py')),
                   observations_sha256=locator.sha(source), runtime_sha256=frozen, protocol=protocol)
    locator.write_json(out / 'protocol.json', binding)
    summary = dict(queue=str(queue), protocol=protocol, models={})
    residuals = {}
    noise = PickNoise(*locator.NOISE)
    toy = np.zeros((3, 2, 2, 2), dtype=np.float32)
    negative_log_likelihood(toy, np.zeros(3), noise, station_indices=np.arange(3))
    refine_hypocentre_in_cell(toy, np.zeros(3), (0, 0, 0), noise, station_indices=np.arange(3))
    initial = np.load(queue / 'inputs/initial_velocity_m_s.npy', allow_pickle=False)
    for mode in MODES:
        started = time.monotonic()
        if mode == 'initial':
            velocity, model_path = initial, queue / 'inputs/initial_velocity_m_s.npy'
        else:
            result = locator.read(queue / f'cases/{mode}/result.json')
            if result['status'] != 'completed':
                raise ValueError(f'Incomplete {mode}')
            directory = Path(result['run_dir'])
            for name, digest in result['artifact_sha256'].items():
                locator.verified(directory / name, digest)
            np.testing.assert_array_equal(initial, np.load(directory / 'initial_model.npy', allow_pickle=False))
            model_path = directory / 'final_model.npy'
            velocity = np.load(model_path, allow_pickle=False)
        if velocity.shape != (23, 11, 7) or not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError('Invalid velocity model')
        digest = locator.sha(model_path)
        print(f'{mode}: station fields, {workers} workers', flush=True)
        fine = velocity.repeat(8, 0).repeat(8, 1).repeat(8, 2)
        fields = station_travel_time_fields(VelocityModel(fine, locator.H), stations, workers)
        del fine
        if not np.isfinite(fields).all():
            raise ValueError('Nonfinite station fields')
        locator._STATE.update(fields=fields, train=train, noise=noise)
        predictions, positions = np.empty_like(train), np.empty((len(event_ids), 3))
        origins, costs, fallbacks = np.empty(len(event_ids)), np.empty(len(event_ids)), np.zeros(len(event_ids), int)
        with mp.get_context('fork').Pool(workers) as pool:
            for done, (i, pos, pred, origin, cost, fallback) in enumerate(
                    pool.imap_unordered(locator.fit_event, range(len(event_ids))), 1):
                positions[i], predictions[i], origins[i], costs[i], fallbacks[i] = pos, pred, origin, cost, fallback
                if done % 50 == 0 or done == len(event_ids):
                    progress = dict(mode=mode, completed=done, total=len(event_ids), elapsed_s=time.monotonic()-started)
                    locator.write_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        locator._STATE.clear()
        del fields
        positions = (positions + .5) * locator.H
        if not all(np.isfinite(x).all() for x in (positions, predictions, origins, costs)):
            raise ValueError('Nonfinite fit')
        np.savez_compressed(out / f'{mode}_predictions.npz', position_m=positions,
                            predicted_relative_s=predictions, origin_relative_s=origins,
                            fit_nll=costs, candidate_fallback_counts=fallbacks,
                            event_ids=event_ids, station_ids=station_ids)
        residual = predictions - observed
        residuals[mode] = residual
        distance = np.linalg.norm(positions - catalog, axis=1)
        metrics = dict(validation_rmse_s=float(np.sqrt(np.mean(residual[validation]**2))),
                       validation_mae_s=float(np.mean(np.abs(residual[validation]))),
                       fit_rmse_s=float(np.sqrt(np.mean(residual[np.isfinite(train)]**2))),
                       catalog_distance_median_m=float(np.median(distance)),
                       catalog_distance_rmse_m=float(np.sqrt(np.mean(distance**2))),
                       catalog_distance_p90_m=float(np.quantile(distance, .9)),
                       elapsed_s=time.monotonic()-started, candidate_fallbacks=int(fallbacks.sum()),
                       model_sha256=digest, predictions_sha256=locator.sha(out / f'{mode}_predictions.npz'))
        if mode != 'initial':
            metrics['validation_rmse_delta_vs_initial_ci95_s'] = paired_interval(residuals['initial'], residual, validation)
        locator.verified(model_path, digest)
        summary['models'][mode] = metrics
        locator.write_json(out / 'summary.json', summary)
        print(mode, json.dumps(metrics), flush=True)
    for name, digest in manifest['frozen_sha256'].items():
        locator.verified(queue / name, digest)
    locator.write_json(out / 'complete.json', dict(status='completed', summary_sha256=locator.sha(out / 'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    evaluate(args.queue.resolve(), args.output.resolve(), args.workers)
