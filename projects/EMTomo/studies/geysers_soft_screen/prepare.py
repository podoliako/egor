"""Prepare 24 soft configurations and 13 shared hard controls; no forward solves."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
STATE = PROJECT / 'runs/geysers_faulted_layers/soft_screen_20261028'
EXPERIMENTS = PROJECT.parent / 'forward_modeling/experiments'
SOURCE_ID = 'geysers11_20110401-20110420_fault3_x6p5_elev1166_26st_3776ev_clean'
sys.path.insert(0, str(PROJECT.parents[1]))
from projects.forward_modeling.experiments import _staged_directory, load_inputs, save_inputs
from projects.forward_modeling.model import PointSet
from projects.forward_modeling.noise import NoiseConfig, add_arrival_noise


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def station_order(xyz):
    """Greedy farthest-point sampling in physical XY, without looking at truth/data."""
    xy = np.asarray(xyz)[:, :2]
    chosen = [int(np.argmax(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))]
    while len(chosen) < len(xy):
        distances = np.min(np.sum((xy[:, None] - xy[chosen][None]) ** 2, axis=2), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return chosen


def soft_specs():
    specs = []
    supports = [(64, 20, 2), (128, 40, 2), (128, 40, 3)]
    schedule_a = [4.] * 4 + [2.] * 4 + [1.] * 4
    schedule_b = [8.] * 3 + [4.] * 3 + [2.] * 3 + [1.] * 3
    for k, n, d in supports:
        for temperature in (2., 4.):
            specs.append(dict(k=k, n=n, d=d, temperature=temperature))
    for schedule in (schedule_a, schedule_b):
        for k, n, d in supports:
            specs.append(dict(k=k, n=n, d=d, schedule=schedule))
    for changes in ({'lambda_reg': .3}, {'lambda_reg': 3.}, {'coverage_damping_power': .5},
                    {'n_cycles': 24, 'schedule': [4.] * 4 + [2.] * 4 + [1.] * 16}):
        specs.append(dict(k=128, n=40, d=2, schedule=schedule_a, **{
            key: value for key, value in changes.items() if key != 'schedule'}))
        if 'schedule' in changes:
            specs[-1]['schedule'] = changes['schedule']
    for count in (12, 8):
        specs.append(dict(k=128, n=40, d=2, stations=count))
        specs.append(dict(k=128, n=40, d=2, stations=count, schedule=schedule_a))
    for count, noise in ((26, .02), (26, .03), (12, .02), (8, .02)):
        specs.append(dict(k=128, n=40, d=2, stations=count, noise=noise))
    assert len(specs) == 24
    return specs


def prepare_dataset(count, sigma, model, stations, events, clean, indices, source_metadata, source_hashes):
    if count == 26 and sigma == 0:
        return SOURCE_ID
    noise_label = 'clean' if sigma == 0 else f'noise{int(round(sigma * 100)):02d}'
    identifier = f'gfsoftscreen28_{count}st_{noise_label}_3776ev'
    selected = PointSet(tuple(stations.ids[i] for i in indices), stations.coordinates_m[indices])
    generation = dict(generator='geysers_soft_screen.prepare', source_experiment_id=SOURCE_ID,
                      source_input_sha256=source_hashes, station_ids=list(selected.ids),
                      selection='Nested deterministic farthest-point XY selection; no truth or arrivals used.',
                      noise_sigmas=[sigma, sigma], noise_seed=42,
                      source_absolute_times_sha256=sha(EXPERIMENTS / 'output' / SOURCE_ID / 'clean_times.npz'))
    destination = EXPERIMENTS / 'input' / identifier
    if destination.exists():
        actual, old_stations, old_events = load_inputs(EXPERIMENTS, identifier)
        assert np.array_equal(actual.velocity, model.velocity)
        assert old_stations.ids == selected.ids and np.array_equal(old_stations.coordinates_m, selected.coordinates_m)
        assert old_events.ids == events.ids and np.array_equal(old_events.coordinates_m, events.coordinates_m)
        assert json.loads((destination / 'generation.json').read_text()) == generation
    else:
        save_inputs(EXPERIMENTS, identifier, model, selected, events, generation)
    times = np.array(clean[:, indices], copy=True)
    noise = NoiseConfig(sigma, sigma, 42)
    if sigma:
        times = add_arrival_noise(times, events.ids, selected.ids, noise)
    times -= times.min(axis=1, keepdims=True)
    metadata = dict(source_metadata)
    metadata.pop('absolute_times_cache', None)
    metadata.update(solver_name='derived_from_saved_pykonal_absolute_times', elapsed_seconds=0.,
                    input_sha256={name: sha(destination / name) for name in
                                  ('model.npz', 'stations.csv', 'events.csv', 'generation.json')},
                    no_new_forward_solve=True, derived_from=generation)
    metadata['noise'] = dict(source_metadata['noise'], enabled=bool(sigma),
                             config=asdict(noise) if sigma else None)
    output = EXPERIMENTS / 'output' / identifier
    if output.exists():
        assert json.loads((output / 'metadata.json').read_text()) == metadata
    else:
        with _staged_directory(output) as temporary:
            with (temporary / 'arrivals.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('station_id', 'event_id', 'arrival_time_s'))
                writer.writerows((sta, event, float(times[e, s])) for e, event in enumerate(events.ids)
                                 for s, sta in enumerate(selected.ids))
            write_json(temporary / 'metadata.json', metadata)
    return identifier


def prepare():
    if (STATE / 'manifest.json').exists():
        raise FileExistsError('Plan already exists; use its frozen runner, do not silently re-prepare')
    runtime = STATE / 'snapshot'
    assert (runtime / 'projects/EMTomo/config.py').exists(), 'Prepare the numerical snapshot first'
    sys.path.insert(0, str(runtime / 'projects/EMTomo'))
    from config import InversionConfig
    from experiment_data import load_tomography_experiment

    model, stations, events = load_inputs(EXPERIMENTS, SOURCE_ID)
    assert (len(events.ids), len(stations.ids)) == (3776, 26)
    source_input = EXPERIMENTS / 'input' / SOURCE_ID
    source_output = EXPERIMENTS / 'output' / SOURCE_ID
    source_metadata = json.loads((source_output / 'metadata.json').read_text())
    assert sha(source_output / 'clean_times.npz') == source_metadata['absolute_times_cache']['sha256']
    with np.load(source_output / 'clean_times.npz', allow_pickle=False) as cache:
        clean = cache['absolute_times_s'].copy()
        assert tuple(cache['event_ids']) == events.ids and tuple(cache['station_ids']) == stations.ids
        np.testing.assert_array_equal(cache['station_coordinates_m'], stations.coordinates_m)
        np.testing.assert_array_equal(cache['event_coordinates_m'], events.coordinates_m)
    source_hashes = {name: sha(source_input / name) for name in
                     ('model.npz', 'stations.csv', 'events.csv', 'generation.json')}
    order = station_order(stations.coordinates_m)
    subsets = {count: sorted(order[:count]) for count in (8, 12, 26)}
    datasets = {}
    for count, sigma in ((26, 0.), (12, 0.), (8, 0.), (26, .02), (26, .03), (12, .02), (8, .02)):
        identifier = prepare_dataset(count, sigma, model, stations, events, clean, subsets[count],
                                     source_metadata, source_hashes)
        loaded = load_tomography_experiment(identifier, EXPERIMENTS)
        assert loaded.arrival_times_s.shape == (3776, count) and loaded.noise_sigmas == (sigma, sigma)
        expected = clean[:, subsets[count]]
        if sigma:
            expected = add_arrival_noise(expected, events.ids, loaded.station_ids, NoiseConfig(sigma, sigma, 42))
        np.testing.assert_array_equal(loaded.arrival_times_s, expected - expected.min(axis=1, keepdims=True))
        directory = EXPERIMENTS / 'input' / identifier
        output = EXPERIMENTS / 'output' / identifier
        paths = [directory / name for name in ('model.npz', 'stations.csv', 'events.csv', 'generation.json')]
        paths += [output / name for name in ('arrivals.csv', 'metadata.json')]
        datasets[(count, sigma)] = dict(id=identifier, count=count, noise_sigma=sigma,
                                        sha256={str(p): sha(p) for p in paths})
    base = asdict(InversionConfig(cell_size=1000., subdivision=8, n_cycles=12, n_workers=24,
                  initial_layer_boundaries_km=(.85, 2.3), initial_layer_velocities_m_s=(3850., 4550., 5300.),
                  n_candidates=128, weights_top_n=40, weights_min_distance=2,
                  weight_noise_relative_sigma=0., weight_noise_absolute_sigma_s=0., weight_model_sigma_s=.02,
                  lambda_reg=1., coverage_damping_power=0., max_velocity_step_fraction=.02,
                  linear_solver='dense', runs_dir=str(PROJECT / 'runs')))
    soft = []
    for number, spec in enumerate(soft_specs(), 1):
        dataset = datasets[(spec.get('stations', 26), spec.get('noise', 0.))]
        config = dict(base, candidate_mode='soft', n_candidates=spec['k'], weights_top_n=spec['n'],
                      weights_min_distance=spec['d'], temperature=spec.get('temperature', 1.),
                      temperature_schedule=spec.get('schedule'),
                      weight_noise_relative_sigma=dataset['noise_sigma'],
                      weight_noise_absolute_sigma_s=dataset['noise_sigma'])
        for key in ('lambda_reg', 'coverage_damping_power', 'n_cycles'):
            if key in spec:
                config[key] = spec[key]
        config['run_name'] = f'gfss28-s{number:02d}'
        InversionConfig(**config)
        soft.append(dict(id=f's{number:02d}', experiment_id=dataset['id'], config=config))
    controls, seen = [], {}
    for case in soft:
        config = dict(case['config'], candidate_mode='hard', temperature=1., temperature_schedule=None)
        config.pop('run_name')
        key = json.dumps([case['experiment_id'], config], sort_keys=True)
        if key not in seen:
            identifier = f'h{len(controls)+1:02d}'
            seen[key] = identifier
            config['run_name'] = f'gfss28-{identifier}'
            controls.append(dict(id=identifier, experiment_id=case['experiment_id'], config=config))
        case['hard_control'] = seen[key]
    assert len(controls) == 13
    ordered, used = [], set()
    # Prioritize the five most informative proposals, then finish the full screen.
    for number in [3, 8, 19, 20, 23] + [n for n in range(1, 25) if n not in (3, 8, 19, 20, 23)]:
        case = soft[number-1]
        hard = next(c for c in controls if c['id'] == case['hard_control'])
        for item in (hard, case):
            if item['id'] not in used:
                ordered.append(item)
                used.add(item['id'])
    manifest = dict(project_root=str(PROJECT), experiments_root=str(EXPERIMENTS), runtime_root=str(runtime),
                    runtime_sha256={str(p.relative_to(runtime)): sha(p) for p in sorted(runtime.rglob('*.py'))},
                    source_absolute_times_sha256=sha(source_output / 'clean_times.npz'),
                    datasets=list(datasets.values()), station_selection={str(n): [stations.ids[i] for i in idx]
                                                                       for n, idx in subsets.items()},
                    cases=ordered, case_timeout_s=14400, forward_solves=0, soft_runs=24, hard_controls=13)
    write_json(STATE / 'manifest.json', manifest)
    print(json.dumps(dict(stations=manifest['station_selection'], soft=24, hard=13,
                          order=[c['id'] for c in ordered]), indent=2))


if __name__ == '__main__':
    prepare()
