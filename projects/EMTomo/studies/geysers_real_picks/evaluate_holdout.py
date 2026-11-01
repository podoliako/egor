"""Fixed-model CV: fit locations/origins on TRAIN only, then open HOLDOUT.

python -B studies/geysers_real_picks/evaluate_holdout.py
Uses a common 125-m FMM/location protocol for initial, hard, soft and tomoDD
velocity models. No velocity inversion, catalog-location prior or holdout refit.
"""
from collections import defaultdict
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/geysers_real_20261031_pick01'
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
OUT = SOURCE / 'holdout_evaluation_v2'
MODELS = ('initial', 'hard', 'soft', 'tomodd')
WORKERS = 24
H = 125.
NOISE_VALUES = (.01, 0., .02)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def training():
    split = json.loads((DATA / 'manifest.json').read_text())
    for name in ('emtomo/arrivals.npz', 'emtomo/metadata.json', 'geometry.json'):
        assert sha(DATA / name) == split['output_sha256'][name]
    path = SOURCE / 'inputs/emtomo/arrivals.npz'
    assert sha(path) == split['output_sha256']['emtomo/arrivals.npz']
    with np.load(path, allow_pickle=False) as a:
        event_ids, station_ids = tuple(a['event_ids']), tuple(a['station_ids'])
        stations, train = a['station_coordinates_m'].copy(), a['arrival_times_s'].copy()
    assert train.shape == (3471, 31) and np.isfinite(train).sum() == 42973
    return event_ids, station_ids, stations, train


def activate_runtime():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    hashes = {name: value for name, value in manifest['frozen_sha256'].items()
              if name.startswith('snapshot/') and name.endswith('.py')}
    for name, digest in hashes.items():
        assert sha(SOURCE / name) == digest, name
    runtime = SOURCE / 'snapshot'
    sys.path[:0] = [str(runtime / 'projects/EMTomo'), str(runtime)]
    return hashes


def velocity_model(name):
    """Return XYZ fine-grid m/s and exact checkpoint provenance."""
    if name == 'initial':
        path = SOURCE / 'inputs/initial_velocity_m_s.npy'
        coarse = np.load(path, allow_pickle=False)
        paths, note = [path], 'Initial regional 1-D baseline'
    elif name in ('hard', 'soft'):
        record = SOURCE / f'cases/{name}/result.json'
        result = json.loads(record.read_text())
        directory = Path(result['run_dir'])
        meta_path = directory / 'meta.json'
        meta = json.loads(meta_path.read_text())
        complete = sorted(int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json'))
        assert complete == list(range(len(complete))) and complete
        paths = [record, meta_path]
        final = directory / 'final_model.npy'
        if final.exists():
            assert len(complete) == meta['run_params']['n_cycles']
            coarse = np.load(final, allow_pickle=False)
            paths.append(final)
            note = f'{name}: completed {len(complete)} velocity updates'
        else:
            last = directory / f'iter_{complete[-1]}'
            v = np.load(last / 'model.npy', allow_pickle=False)
            ds = np.load(last / 'delta_s.npy', allow_pickle=False).reshape(v.shape)
            coarse = 1 / np.maximum(1/v + ds, 1e-12)
            step = meta['run_params']['max_velocity_step_fraction']
            if step is not None:
                coarse = np.clip(coarse, v*(1-step), v*(1+step))
            paths += [last / 'model.npy', last / 'delta_s.npy']
            next_model = directory / f'iter_{complete[-1]+1}/model.npy'
            if next_model.exists():
                np.testing.assert_array_equal(coarse, np.load(next_model, allow_pickle=False))
                paths.append(next_model)
            note = f'{name}: last valid model after {len(complete)} updates; run failed, not final'
    else:
        from scipy.interpolate import RegularGridInterpolator
        work = SOURCE / 'cases/tomodd/work'
        record = SOURCE / 'cases/tomodd/result.json'
        result = json.loads(record.read_text())
        assert result['status'] == 'completed'
        assert sha(work / 'Vp_model.dat') == result['artifact_sha256']['Vp_model.dat']
        lines = (work / 'MOD').read_text().splitlines()
        nx, ny, nz = map(int, lines[0].split()[1:])
        lon, lat, depth = (np.fromstring(line, sep=' ') for line in lines[1:4])
        native = np.fromstring((work / 'Vp_model.dat').read_text(), sep=' ').reshape(nz, ny, nx)
        gpath = DATA / 'geometry.json'
        g = json.loads(gpath.read_text())
        axes = [(np.arange(n)+.5)*H/1000 for n in (184, 88, 56)]
        x, y, z = np.meshgrid(*axes, indexing='ij')
        ox, oy = np.asarray(g['geographic_reference_in_new_xy_m'])/1000
        angle = np.deg2rad(g['x_azimuth_degrees'])
        north = (x-ox)*np.cos(angle)-(y-oy)*np.sin(angle)
        east = (x-ox)*np.sin(angle)+(y-oy)*np.cos(angle)
        latitude = g['reference_latitude']+north/110.574
        longitude = g['reference_longitude']+east/(111.320*np.cos(np.deg2rad(g['reference_latitude'])))
        points = np.stack((z-g['top_elevation_m']/1000, latitude, longitude), axis=-1)
        fine = RegularGridInterpolator((depth, lat, lon), native, bounds_error=True)(points)*1000
        provenance = dict(note='tomoDD: final 3 velocity updates; native model retained 3467/3471 events',
                          source_sha256={str(p): sha(p) for p in (record, work/'MOD', work/'Vp_model.dat', gpath)})
        return fine, provenance
    assert coarse.shape == (23, 11, 7) and np.isfinite(coarse).all() and (coarse > 0).all()
    fine = coarse.repeat(8, axis=0).repeat(8, axis=1).repeat(8, axis=2)
    return fine, dict(note=note, source_sha256={str(p): sha(p) for p in paths})


_FIT = {}


def safe_refine(fields, observed, cell, noise, selected):
    """Retain a candidate if Powell hits its known zero-direction reduction bug."""
    from tomography.tomography_math import refine_hypocentre_in_cell
    try:
        position, value = refine_hypocentre_in_cell(fields, observed, cell, noise, station_indices=selected)
        return position, value, False
    except ValueError as error:
        if str(error) != 'zero-size array to reduction operation maximum which has no identity':
            raise
    from scipy.optimize import minimize_scalar
    from instruments.hypocentre_objective import make_refinement_objective
    from instruments.instruments_coords import cell_coord_bounds
    objective = make_refinement_objective(fields, observed, noise, station_indices=selected)
    bounds = cell_coord_bounds(tuple(int(v) for v in cell), fields.shape[1:])
    position = np.asarray(cell, dtype=float)
    value = float(objective(position))
    for sweep in range(4):
        previous = value
        for axis, (lo, hi) in enumerate(bounds):
            if hi <= lo:
                continue
            def scalar(x):
                point = position.copy()
                point[axis] = x
                return float(objective(point))
            result = minimize_scalar(scalar, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
            for x in (lo, hi, result.x):
                candidate = scalar(x)
                if np.isfinite(candidate) and candidate < value:
                    position[axis], value = x, candidate
        if previous-value <= 1e-10*max(1, abs(previous)):
            break
    if not np.isfinite(value):
        raise ValueError('Nonfinite fallback likelihood')
    return position, value, True


def fit_event(index):
    from instruments.likelihood import negative_log_likelihood
    from instruments.instruments_coords import sample_cell_centered_trilinear_batch
    from tomography.tomography_math import select_candidate_cells, refine_hypocentre_in_cell
    from cv_metrics import profile_origin
    fields, observations, noise = _FIT['fields'], _FIT['train'][index], _FIT['noise']
    selected = np.flatnonzero(np.isfinite(observations))
    observed = observations[selected]
    cost = negative_log_likelihood(fields, observed, noise, station_indices=selected)
    candidates = select_candidate_cells(cost, 128, 2)
    del cost
    refined = [safe_refine(fields, observed, cell, noise, selected) for cell in candidates]
    best = min(range(len(refined)), key=lambda i: refined[i][1])
    position, nll, _ = refined[best]
    fallbacks = sum(value[2] for value in refined)
    absolute = sample_cell_centered_trilinear_batch(fields, position)
    origin = profile_origin(absolute, observations, relative_sigma=.01, model_sigma=.02)
    return index, position, absolute+origin, origin, float(nll), fallbacks


def fit(name):
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = activate_runtime()
    from eikonal import station_travel_time_fields
    from velocity_model import VelocityModel
    from instruments.likelihood import PickNoise, negative_log_likelihood
    from tomography.tomography_math import refine_hypocentre_in_cell
    event_ids, station_ids, stations, train = training()
    fine, provenance = velocity_model(name)
    binding = dict(model=provenance, runtime_sha256=frozen, training_sha256=sha(SOURCE/'inputs/emtomo/arrivals.npz'),
                   evaluator_sha256=sha(Path(__file__)), metrics_sha256=sha(Path(__file__).with_name('cv_metrics.py')),
                   grid_m=H, shortlist=128, dmin=2, origin='TRAIN-only precision weighted offset',
                   noise=list(NOISE_VALUES), prediction='MAP location, identical protocol for all models',
                   fallback='Only for known Powell zero-direction error: bounded coordinate descent, all models')
    metadata_path, arrays_path = OUT/f'{name}_fit.json', OUT/f'{name}_predictions.npz'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        assert old['binding'] == binding and sha(arrays_path) == old['predictions_sha256']
        print(f'{name}: verified existing training-only predictions', flush=True)
        return
    start = time.monotonic()
    write_json(OUT/f'{name}_progress.json', dict(stage='station_fields', completed_events=0))
    print(f'{name}: station fields for fixed velocity model', flush=True)
    fields = station_travel_time_fields(VelocityModel(fine, H), stations, WORKERS)
    del fine
    noise = PickNoise(*NOISE_VALUES)
    # Compile before fork; no ray tracing, normal equations or velocity updates.
    toy = np.zeros((3, 2, 2, 2), dtype=np.float32)
    negative_log_likelihood(toy, np.zeros(3), noise, station_indices=np.arange(3))
    refine_hypocentre_in_cell(toy, np.zeros(3), (0, 0, 0), noise, station_indices=np.arange(3))
    _FIT.update(fields=fields, train=train, noise=noise)
    predictions = np.empty_like(train)
    positions = np.empty((len(event_ids), 3))
    origins, costs = np.empty(len(event_ids)), np.empty(len(event_ids))
    fallback_counts = np.zeros(len(event_ids), dtype=int)
    print(f'{name}: fit all {len(event_ids)} locations/origins using TRAIN only', flush=True)
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for done, (i, position, predicted, origin, cost, fallbacks) in enumerate(pool.imap_unordered(fit_event, range(len(event_ids))), 1):
            predictions[i], positions[i], origins[i], costs[i] = predicted, position, origin, cost
            fallback_counts[i] = fallbacks
            if done % 250 == 0 or done == len(event_ids):
                write_json(OUT/f'{name}_progress.json', dict(stage='train_only_location', completed_events=done,
                                                           total_events=len(event_ids), elapsed_s=time.monotonic()-start))
                print(f'{name}: {done}/{len(event_ids)}', flush=True)
    _FIT.clear()
    assert np.isfinite(predictions).all() and np.isfinite(origins).all()
    temporary = arrays_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, predicted_relative_s=predictions, origin_relative_s=origins,
                            position_m=(positions+.5)*H, train_nll=costs, candidate_fallback_counts=fallback_counts,
                            event_ids=np.asarray(event_ids), station_ids=np.asarray(station_ids))
    temporary.replace(arrays_path)
    for path, digest in provenance['source_sha256'].items():
        assert sha(path) == digest, 'Model changed during evaluation'
    write_json(metadata_path, dict(binding=binding, predictions_sha256=sha(arrays_path),
                                    elapsed_s=time.monotonic()-start, holdout_opened=False,
                                    candidate_fallbacks=int(fallback_counts.sum()),
                                    events_with_fallback=int(np.count_nonzero(fallback_counts))))
    print(f'{name}: training-only prediction artifact saved in {time.monotonic()-start:.1f}s', flush=True)


def score():
    from cv_metrics import summarize
    event_ids, station_ids, stations, train = training()
    event_index, station_index = ({value:i for i,value in enumerate(ids)} for ids in (event_ids, station_ids))
    split = json.loads((DATA/'manifest.json').read_text())
    for name in ('train.csv', 'holdout.csv'):
        assert sha(DATA/name) == split['output_sha256'][name]
    train_rows, held = rows(DATA/'train.csv'), rows(DATA/'holdout.csv')
    key = lambda r: (r['event_id'], r['network']+'.'+r['station'])
    assert len(held) == 4775 and len(train_rows) == 42973
    assert not {key(r) for r in train_rows} & {key(r) for r in held}
    for r in train_rows:
        e, s = key(r)
        assert train[event_index[e], station_index[s]] == float(r['seconds_from_train_reference'])
    report = dict(protocol='Common 125-m skfmm FMM, 128-candidate MAP relocalization and origin fit on TRAIN only.',
                  uncertainty='Relative sigma=.01, model sigma=.02s; no holdout-based tuning.',
                  scope='Velocity-model prediction test; NOT native end-to-end relocation scoring or a rerun of velocity inversion.',
                  holdout_picks=len(held), holdout_events=len({r['event_id'] for r in held}),
                  train_picks=len(train_rows), source_split_sha256=sha(DATA/'manifest.json'), models={})
    prediction_rows, error_vectors = [], {}
    for name in MODELS:
        ledger = json.loads((OUT/f'{name}_fit.json').read_text())
        assert ledger['holdout_opened'] is False
        assert sha(OUT/f'{name}_predictions.npz') == ledger['predictions_sha256']
        with np.load(OUT/f'{name}_predictions.npz', allow_pickle=False) as a:
            assert tuple(a['event_ids']) == event_ids and tuple(a['station_ids']) == station_ids
            predicted = a['predicted_relative_s']
        errors = []
        per_event = defaultdict(list)
        for r in held:
            event, station = key(r)
            e, s = event_index[event], station_index[station]
            assert np.isnan(train[e, s]), 'Held-out observation leaked into training'
            observed = float(r['seconds_from_train_reference'])
            estimate = float(predicted[e, s])
            error = estimate-observed
            errors.append(error)
            per_event[event].append(error)
            prediction_rows.append(dict(model=name, event_id=event, station_id=station,
                                        observed_relative_s=observed, predicted_relative_s=estimate,
                                        residual_pred_minus_observed_s=error))
        error_vectors[name] = np.asarray(errors)
        report['models'][name] = dict(note=ledger['binding']['model']['note'],
                                     candidate_fallbacks=ledger['candidate_fallbacks'],
                                     events_with_fallback=ledger['events_with_fallback'],
                                     train=summarize((predicted-train)[np.isfinite(train)]),
                                     holdout=summarize(errors),
                                     mean_event_mae_s=float(np.mean([np.mean(np.abs(v)) for v in per_event.values()])),
                                     mean_event_mse_s=float(np.mean([np.mean(np.square(v)) for v in per_event.values()])))
    for values in report['models'].values():
        values['holdout_rmse_improvement_vs_initial_pct'] = 100*(1-values['holdout']['rmse_s']/report['models']['initial']['holdout']['rmse_s'])
    path = OUT/'holdout_predictions.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
        writer.writeheader(); writer.writerows(prediction_rows)
    write_json(OUT/'summary.json', report)
    text = ['# Real-pick holdout prediction', '', report['protocol'], '',
            '| Model | Train RMSE, ms | Holdout RMSE, ms | MAE, ms | Median abs, ms | P90 abs, ms | Bias, ms |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in report['models'].items():
        h = values['holdout']
        numbers = [values['train']['rmse_s'], *[h[k] for k in ('rmse_s', 'mae_s', 'median_abs_s', 'p90_abs_s', 'bias_s')]]
        text.append('| '+name+' | '+' | '.join(f'{v*1000:.2f}' for v in numbers)+' |')
    text += ['', 'Hard is the last valid 21-update model; soft completed 24; tomoDD completed 3 updates.',
             'All 4775 held-out picks scored. Location and origin were re-estimated from training only for every model.',
             'Common forward operator samples native tomoDD nodal Vp onto 125-m voxels in the shared domain;',
             'this deliberately does not reproduce the native tomoDD ray solver or use its relocated origins.',
             'This is one fixed pick split; it does not establish geological truth or independent-event generalization.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for name, errors in error_vectors.items():
        absolute = np.sort(np.abs(errors))
        ax.plot(absolute, np.arange(1, len(absolute)+1)/len(absolute), label=name)
    ax.set(xlabel='Absolute held-out pick error (s)', ylabel='Fraction of held-out picks',
           xlim=(0, max(float(np.percentile(np.abs(v), 99)) for v in error_vectors.values())), ylim=(0, 1),
           title='Geysers: 4775 hidden P picks | train-only location and origin')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT/'holdout_absolute_error_ecdf.png', dpi=180)
    plt.close(fig)
    print((OUT/'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', choices=MODELS)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if args.fit:
        fit(args.fit)
    else:
        if not args.score_only:
            for name in MODELS:
                subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--fit', name],
                               check=True, cwd=ROOT, timeout=1800)
        score()
