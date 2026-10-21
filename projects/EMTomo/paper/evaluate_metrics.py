"""Read saved artifacts only; never import or execute a forward/inverse solver.

python -B evaluate_metrics.py             # create supplemental_metrics.json exclusively
python -B evaluate_metrics.py --check     # recompute and compare without writing
python -B -m unittest -v test_evaluate_metrics.py

Requires NumPy/SciPy and the original benchmark artifacts. Evaluation is bounded
by 120 seconds (SIGALRM on POSIX). The only possible write is the new report next
to this script; an existing report, including a symlink, is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

BENCHMARK = Path('/home/egor/tomodd_benchmark')
PAPER = Path(__file__).resolve().parent
COMPARISONS = {
    'synthetic': 'comparison_flat4_synthetic_em8_tomodd3_zero_noise',
    'geysers': 'comparison_flat4_geysers_em12_tomodd3_zero_noise',
}
SOFT_RUNS = {
    5: 'run_gyelev-soft-c1000-k5-geysers11-20110401-20110420-flat4-elev1166-26st-3776ev-noise01-pick001_v1.4_lam-1.0_cov-2_attempt-6efe8b00_20261017_215525',
    10: 'run_gyelev-soft-c1000-k10-geysers11-20110401-20110420-flat4-elev1166-26st-3776ev-noise01-pick001_v1.4_lam-1.0_cov-2_attempt-806f38cf_20261017_220852',
    15: 'run_gyelev-soft-c1000-k15-geysers11-20110401-20110420-flat4-elev1166-26st-3776ev-noise01-pick001_v1.4_lam-1.0_cov-2_attempt-c644cd69_20261017_222715',
}
SHAPE = (160, 56, 56)
COARSE = (20, 7, 7)
BOUNDARIES = (0.7, 1.9, 3.4)
SPEEDS = (3800., 4600., 5200., 5700.)
HELPERS = ('require', 'positive_model', 'read_velocity_history', 'evaluate_native',
           'tile_cells', 'constant_oracle', 'generation_parameters', 'region_masks')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def native_helpers(path):
    """Compile only audited pure helper definitions, NOT the module's imports.

    compare_emtomo_flat4 imports external_checkerboard, which imports plotting,
    solver runners and sparse fitting machinery. AST extraction keeps the exact
    native history/interpolation/mask implementation without those side effects.
    The original source hash and helper names are recorded in the report.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(set(HELPERS) <= definitions.keys(), 'Native evaluation helpers changed')
    selected = ast.Module(body=[definitions[name] for name in HELPERS], type_ignores=[])
    namespace = dict(np=np, re=re, Path=Path, RegularGridInterpolator=RegularGridInterpolator,
                     BOUNDARIES=BOUNDARIES, SPEEDS=SPEEDS)
    exec(compile(selected, str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in HELPERS})


def positions_to_metres(positions, fine_side_m):
    positions = np.asarray(positions, dtype=float)
    require(positions.ndim == 2 and positions.shape[1] == 3
            and np.isfinite(positions).all(), 'Expected finite (atoms, 3) positions')
    require(np.isfinite(fine_side_m) and fine_side_m > 0, 'Invalid fine cell size')
    return (positions + 0.5) * fine_side_m


def distribution_statistics(locations_m, weights, truth_m):
    """ES, MAP error, mean error, and mean-centred squared spread for one event."""
    x, w, truth = (np.asarray(a, dtype=float) for a in (locations_m, weights, truth_m))
    require(x.ndim == 2 and x.shape[1] == 3 and len(x) > 0
            and w.shape == (len(x),) and truth.shape == (3,), 'Invalid distribution shape')
    require(np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(truth).all()
            and (w >= 0).all() and np.isfinite(w.sum()) and w.sum() > 0,
            'Distribution needs finite nonnegative weights with positive mass')
    w = w / w.sum()
    distances = np.linalg.norm(x - truth, axis=1)
    pairwise = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    mean = w @ x
    return {
        'energy_score_m': float(w @ distances - 0.5 * (w @ pairwise @ w)),
        'map_error_m': float(distances[np.argmax(w)]),
        'weighted_mean_error_m': float(np.linalg.norm(mean - truth)),
        'spread_squared_m2': float(w @ np.sum((x - mean)**2, axis=1)),
    }


def summarize_locations(statistics):
    require(len(statistics) > 0, 'Empty catalog')
    columns = {key: np.array([row[key] for row in statistics]) for key in statistics[0]}
    errors = columns['map_error_m']
    return {
        'events': len(statistics),
        'energy_score_mean_m': float(columns['energy_score_m'].mean()),
        'map_rms_m': float(np.sqrt(np.mean(errors**2))),
        'map_median_m': float(np.median(errors)),
        'map_p95_m': float(np.percentile(errors, 95)),
        'weighted_mean_rms_m': float(np.sqrt(np.mean(columns['weighted_mean_error_m']**2))),
        'weighted_rms_spread_m': float(np.sqrt(columns['spread_squared_m2'].mean())),
    }


def template_amplitude(field, truth, background, mask):
    field, truth = np.asarray(field, dtype=float), np.asarray(truth, dtype=float)
    background = np.broadcast_to(np.asarray(background, dtype=float), truth.shape)
    mask = np.asarray(mask, dtype=bool)
    require(field.shape == truth.shape == mask.shape and mask.any(), 'Invalid template shapes/support')
    template = (truth - background)[mask]
    estimate = (field - background)[mask]
    denominator = np.sum(template**2)
    require(np.isfinite(template).all() and np.isfinite(estimate).all() and denominator > 0,
            'Template requires finite values and nonzero contrast')
    return float(np.sum(estimate * template) / denominator)


def config_differences(hard, soft, prefix=''):
    result = {}
    for key in sorted(hard.keys() | soft.keys()):
        a, b = hard.get(key), soft.get(key)
        name = prefix + key
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(config_differences(a, b, name + '.'))
        elif a != b:
            result[name] = {'hard': a, 'soft': b}
    return result


class Evaluation:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.files = {}
        self.native_path = benchmark / 'tomodd_benchmark/compare_emtomo_flat4.py'
        self.record(self.native_path)
        self.native = native_helpers(self.native_path)

    def record(self, path, expected=None):
        path = Path(path)
        value = sha256(path)
        require(expected is None or value == expected, f'Hash mismatch: {path}')
        require(str(path) not in self.files or self.files[str(path)] == value,
                f'Input changed during evaluation: {path}')
        self.files[str(path)] = value
        return value

    def points(self, path, id_field):
        self.record(path)
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(set(reader.fieldnames or []) == {id_field, 'x_m', 'y_m', 'z_m'},
                    f'Unexpected point columns: {path}')
            rows = list(reader)
        ids = tuple(row[id_field] for row in rows)
        xyz = np.array([[float(row[f'{axis}_m']) for axis in 'xyz'] for row in rows])
        require(len(ids) > 0 and len(set(ids)) == len(ids) and np.isfinite(xyz).all(),
                f'Invalid full source catalog: {path}')
        return ids, xyz

    def source(self, provenance):
        source = Path(provenance['source_directory'])
        for name, value in provenance['input_sha256'].items():
            self.record(source / name, value)
        output = source.parent.parent / 'output' / source.name
        output_hashes = {name: self.record(output / name) for name in ('arrivals.csv', 'metadata.json')}
        require(output_hashes['metadata.json'] == provenance['forward_metadata_sha256'],
                'Forward metadata changed')
        parameters = self.native.generation_parameters(read_json(source / 'generation.json'))
        if parameters is None:
            canonical = self.benchmark / 'runs' / COMPARISONS['synthetic'] / 'provenance.json'
            self.record(canonical)
            canonical_source = Path(read_json(canonical)['source_directory'])
            self.record(canonical_source / 'model.npz', provenance['canonical_truth_sha256'])
            self.record(canonical_source / 'generation.json', read_json(canonical)['input_sha256']['generation.json'])
            parameters = self.native.generation_parameters(read_json(canonical_source / 'generation.json'))
        require(parameters is not None and parameters['horizontal_boundaries_km'] == list(BOUNDARIES[:2])
                and parameters['dipping_boundary_km'] == [3.4, 3.4]
                and parameters['layer_velocities_m_s'] == list(SPEEDS), 'Not the known flat4 background')
        with np.load(source / 'model.npz', allow_pickle=False) as data:
            truth_xyz = self.native.positive_model(data['velocity'], SHAPE, 'truth')
            require(float(data['cell_size_m']) == 125 and np.array_equal(data['origin_m'], [0, 0, 0]),
                    'Unexpected truth cell size/origin')
        ids, true_m = self.points(source / 'events.csv', 'event_id')
        station_ids, stations_m = self.points(source / 'stations.csv', 'station_id')
        truth = truth_xyz.transpose(2, 1, 0)
        centers = tuple((np.arange(n) + .5) * .125 for n in SHAPE)
        background = np.asarray(SPEEDS)[np.searchsorted(BOUNDARIES, centers[2], side='right')][:, None, None]
        regions = self.native.region_masks(truth, centers, parameters)
        counts = {name: int(mask.sum()) for name, mask in regions.items()}
        require(counts == provenance['evaluation']['region_cells'], 'Native region masks changed')
        sphere_names = [name for name in regions if name.startswith('sphere_')]
        require(len(sphere_names) == 6, 'Expected six spheres')
        return dict(source=source, ids=ids, true_m=true_m, stations_m=stations_m,
                    station_ids=station_ids, truth=truth, centers=centers, background=background,
                    regions=regions, sphere_names=sphere_names, parameters=parameters,
                    input_hashes=provenance['input_sha256'], output_hashes=output_hashes)

    def velocity(self, field, context):
        truth, regions = context['truth'], context['regions']
        require(field.shape == truth.shape and np.isfinite(field).all() and (field > 0).all(),
                'Invalid evaluated velocity field')
        return {
            'volume_rmse_m_s': float(np.sqrt(np.mean((field - truth)**2))),
            'spheres_rmse_m_s': float(np.sqrt(np.mean((field[regions['spheres']] - truth[regions['spheres']])**2))),
            'sphere_template_amplitudes': {
                name: template_amplitude(field, truth, context['background'], regions[name])
                for name in context['sphere_names']},
        }

    def weights(self, cycle, meta, context):
        ids, truth_m = context['ids'], context['true_m']
        actual = {p.name for p in cycle.iterdir() if p.is_dir() and p.name.startswith('event_')}
        require(actual == {f'event_{j}' for j in range(len(ids))}, f'Incomplete/extra event rows: {cycle}')
        checksum, statistics, mass_sums, counts = hashlib.sha256(), [], [], []
        for j, event_id in enumerate(ids):
            path = cycle / f'event_{j}/weights.npz'
            payload = path.read_bytes()
            checksum.update(f'{j}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                require(tuple(data['weight_shape']) == SHAPE, f'Wrong weight geometry: {path}')
                positions, values = data['positions'], data['weight_values']
                xyz = positions_to_metres(positions, meta['grid_info']['fine_cell_size'])
            require(1 <= len(xyz) <= meta['run_params']['weights_top_n'], f'Invalid atom count: {path}')
            require(((xyz >= 0) & (xyz <= np.array(SHAPE) * 125)).all(), f'Atom outside volume: {path}')
            statistics.append(distribution_statistics(xyz, values, truth_m[j]))
            mass_sums.append(float(values.sum()))
            counts.append(len(values))
        return summarize_locations(statistics), {
            'weights_directory': str(cycle), 'row_id_weights_sha256': checksum.hexdigest(),
            'weights_mapping': 'SHA256(row_index TAB source_event_id TAB weights_file_sha256 LF), CSV row order',
            'saved_weight_sum_range': [min(mass_sums), max(mass_sums)],
            'saved_atoms_range': [min(counts), max(counts)],
            'weights_normalized_per_event_for_scoring': True,
        }

    def em(self, run, updates, context, expected=None):
        self.record(run / 'meta.json', expected['meta_sha256'] if expected else None)
        meta = read_json(run / 'meta.json')
        params, grid = meta['run_params'], meta['grid_info']
        require(meta['source_experiment'] == {'id': context['source'].name,
                'model_sha256': context['input_hashes']['model.npz']}, 'EM source mismatch')
        require(params['n_events'] == len(context['ids']) and meta['event_locs'] == [],
                'Unexpected EM event count/initialization')
        require(np.allclose(meta['station_locs'], context['stations_m'], rtol=0, atol=1e-7),
                'EM station geometry mismatch')
        require(grid['fine_shape'] == list(SHAPE) and grid['coarse_shape'] == list(COARSE)
                and grid['fine_cell_size'] == 125 and grid['coarse_cell_size'] == 1000
                and params['subdivision'] == 8 and params['cell_size'] == 1000
                and params['fine_side_m'] == 125 and params['slowness_interpolation'] == 'nearest',
                'EM geometry/interpolation mismatch')
        recorded = params.get('provenance', {})
        for key, actual in (('input_sha256', context['input_hashes']), ('output_sha256', context['output_hashes'])):
            require(key not in recorded or recorded[key] == actual, f'EM {key} mismatch')
        require(0 < updates <= params['n_cycles'], 'Unavailable EM updates')
        quality_path = run / 'quality.jsonl'
        self.record(quality_path, expected['quality_log_sha256'] if expected else None)
        quality = [json.loads(line) for line in quality_path.read_text().splitlines() if line.strip()]
        require([row['iter'] for row in quality[:updates]] == list(range(updates)), 'Incomplete EM history')
        for step in range(updates):
            complete = run / f'iter_{step}/complete.json'
            self.record(complete)
            require(read_json(complete)['iter'] == step, f'Incomplete checkpoint: {complete}')
        model_path = run / ('final_model.npy' if updates == params['n_cycles'] else f'iter_{updates}/model.npy')
        self.record(model_path, expected['model_sha256'] if expected else None)
        initial_hash = self.record(run / 'initial_model.npy', expected['initial_model_sha256'] if expected else None)
        model = self.native.positive_model(np.load(model_path, allow_pickle=False), COARSE, 'EM model')
        scores, info = self.weights(run / f'iter_{updates - 1}', meta, context)
        if expected:
            require(info['row_id_weights_sha256'] == expected['row_id_weights_sha256'], 'Hard weights changed')
            require(updates - 1 == expected['e_step_cycle'] and str(model_path) == expected['model_path'],
                    'Comparison checkpoint selection changed')
        info.update(run=str(run), updates=updates, model_path=str(model_path), e_step_cycle=updates - 1,
                    e_step_is_pre_update_not_post_M_relocation=True, initial_model_sha256=initial_hash,
                    source_input_hashes_recorded=recorded.get('input_sha256'),
                    source_output_hashes_recorded=recorded.get('output_sha256'), run_params=params)
        return dict(location=scores, velocity=self.velocity(self.native.tile_cells(model), context), provenance=info), meta

    def comparison(self, label):
        directory = self.benchmark / 'runs' / COMPARISONS[label]
        for filename in ('provenance.json', 'metrics.json'):
            self.record(directory / filename)
        provenance, metrics = (read_json(directory / name) for name in ('provenance.json', 'metrics.json'))
        context = self.source(provenance)
        require(len(context['ids']) == (1920 if label == 'synthetic' else 3776)
                and len(context['station_ids']) == (52 if label == 'synthetic' else 26), 'Wrong full catalog')
        updates = 8 if label == 'synthetic' else 12
        hard, hard_meta = self.em(Path(provenance['emtomo']['run']), updates, context, provenance['emtomo'])
        require(hard_meta['run_params']['candidate_mode'] == 'hard'
                and hard_meta['run_params']['weights_top_n'] == 1, 'Expected hard point mass')
        fields = {'emtomo': hard['velocity']}
        native = self.native
        oracle_nodes = None
        for name, info in provenance['tomodd'].items():
            for path, value in info['files_sha256'].items():
                self.record(Path(path), value)
            run = Path(info['run'])
            config = read_json(run / 'config.json')
            require(config['source_forward_metadata']['input_sha256'] == context['input_hashes'], 'TomoDD source mismatch')
            axes = tuple(np.array(config['tomodd_grid'][key]) for key in ('depths_km', 'y_km', 'x_km'))
            require(tuple(map(len, axes)) == (9, 9, 22), 'Unexpected native node grid')
            with np.load(run / 'state.npz', allow_pickle=False) as state:
                require(np.allclose(state['physical_velocity_km_s'] * 1000, context['truth'].transpose(2, 1, 0),
                                    rtol=0, atol=1e-9), 'Saved state truth mismatch')
                require(np.allclose(state['true_event_xyz_km'] * 1000, context['true_m'], rtol=0, atol=1e-7),
                        'Saved state event geometry mismatch')
                require(float(state['cell_size_km']) == .125, 'Saved state spacing mismatch')
                oracle = native.positive_model(state['oracle_vp'] * 1000, (9, 9, 22), 'saved oracle')
                if oracle_nodes is not None:
                    require(np.allclose(oracle, oracle_nodes, rtol=0, atol=1e-6), 'Oracle differs between starts')
                oracle_nodes = oracle
                fields['trilinear_oracle'] = self.velocity(native.evaluate_native(oracle, axes, context['centers']), context)
                fields['tomo_initial'] = self.velocity(native.evaluate_native(state['initial_vp'] * 1000, axes, context['centers']), context)
            require(info['iteration'] == 3, 'Expected TomoDD iteration 3')
            nodes = native.read_velocity_history(run / 'work/inversion/tomoDD.vel', (9, 9, 22), 3)
            fields[name] = self.velocity(native.evaluate_native(nodes, axes, context['centers']), context)
        fields['constant_oracle'] = self.velocity(native.tile_cells(native.constant_oracle(context['truth'])), context)
        initial = np.load(Path(provenance['emtomo']['run']) / 'initial_model.npy', allow_pickle=False)
        fields['em_initial'] = self.velocity(native.tile_cells(initial), context)
        for name, scores in fields.items():
            for region, key in (('volume', 'volume_rmse_m_s'), ('spheres', 'spheres_rmse_m_s')):
                require(np.isclose(scores[key], metrics['velocity'][name][region]['rmse_m_s'], rtol=0, atol=1e-8),
                        f'Existing common-grid metric mismatch: {label}/{name}/{region}')
        old_location = metrics['location']['emtomo_last_pre_update_E_step']
        for key, old in (('map_rms_m', 'rms_km'), ('map_median_m', 'median_km'), ('map_p95_m', 'p95_km')):
            require(np.isclose(hard['location'][key], 1000 * old_location[old], rtol=0, atol=1e-7),
                    'Existing hard location metric mismatch')
        result = dict(source_experiment=provenance['source_experiment'], source_directory=str(context['source']),
                      input_sha256=context['input_hashes'], output_sha256=context['output_hashes'],
                      events=len(context['ids']), stations=len(context['station_ids']),
                      region_cells={key: int(mask.sum()) for key, mask in context['regions'].items()},
                      hard=hard, velocity=fields, existing_comparison_metrics_verified=True)
        return result, context, hard_meta

    def run(self):
        # Evidence for the saved continuous-index and fine-centre layer conventions.
        for path in (PAPER.parent / 'instruments/instruments_coords.py',
                     PAPER.parent / 'tomography/tomography_logging.py',
                     PAPER.parent.parent / 'forward_modeling/dipping_layers_generator.py'):
            self.record(path)
        result = {}
        for label in COMPARISONS:
            result[label], context, hard_meta = self.comparison(label)
            if label != 'geysers':
                continue
            scan = {}
            for k, name in SOFT_RUNS.items():
                soft, meta = self.em(PAPER.parent / 'runs' / name, 12, context)
                params = meta['run_params']
                require(params['candidate_mode'] == 'soft' and params['weights_top_n'] == k
                        and params['n_candidates'] == k and params['n_cycles'] == 12, 'Wrong soft configuration')
                differences = config_differences(hard_meta['run_params'], params)
                allowed = {'candidate_mode', 'n_candidates', 'weights_top_n', 'run_name', 'provenance.case_id',
                           'effective_config.candidate_mode', 'effective_config.n_candidates',
                           'effective_config.weights_top_n', 'effective_config.run_name'}
                require(set(differences) <= allowed, f'Unexpected soft/hard config differences: {differences}')
                require(soft['provenance']['source_input_hashes_recorded'] == context['input_hashes']
                        and soft['provenance']['source_output_hashes_recorded'] == context['output_hashes'],
                        'Soft scan requires full recorded input/output hashes')
                require(soft['provenance']['initial_model_sha256'] == result[label]['hard']['provenance']['initial_model_sha256'],
                        'Soft initialization differs')
                soft['config_differences_from_hard'] = differences
                scan[f'K{k}'] = soft
            result[label]['supplementary_sensitivity_scan'] = dict(
                matched_soft_weight_effect=False, best_K_selected=False, runs=scan,
                interpretation='Shortlist n_candidates and retained weights_top_n both change from hard K=1; not an isolated soft-weight effect.')
        return {
            'schema_version': 1,
            'method': {
                'grid': dict(shape_xyz=list(SHAPE), cell_size_m=125, samples=501760, origin_m=[0, 0, 0]),
                'sphere_order': list(context['sphere_names']),
                'sphere_parameters': context['parameters']['anomalies'],
                'background': dict(boundaries_km=list(BOUNDARIES), speeds_m_s=list(SPEEDS),
                                   convention='Native fine-cell depth centres; searchsorted(side=right), depth >= 3.4 km in layer 4; no fitted/demeaned background.'),
                'sphere_mask': 'Native center-distance <= radius AND abs(truth - known background) > 1e-7 m/s; identical to source comparisons.',
                'template_amplitude': 'sum_sphere((vhat-background)*(vtrue-background)) / sum_sphere((vtrue-background)^2); dimensionless, signed, not clipped',
                'energy_score': 'mean_events[sum_j w_j ||h_j-h_true|| - 0.5 sum_jl w_j w_l ||h_j-h_l||], metres',
                'hard_energy_score': 'Mean Euclidean error, NOT location RMS.',
                'weighted_mean_rms': 'sqrt(mean_events(||sum_j w_j h_j - h_true||^2)), metres',
                'weighted_rms_spread': 'sqrt(mean_events(sum_j w_j ||h_j - sum_l w_l h_l||^2)), metres',
                'map': 'Largest normalized saved weight; first saved atom breaks ties; RMS/median/P95 over full catalog.',
                'coordinates': '(saved continuous positions + 0.5) * fine_cell_size_m; weight_indices are NOT locations',
                'oracle': 'Constant-cell arithmetic Vp mean computed analytically; native trilinear uses saved oracle_vp only. No oracle/inverse/forward fitting executed.',
                'native_helpers': list(HELPERS), 'native_source': str(self.native_path),
                'deadline_seconds': 120,
            },
            'caveats': [
                'EM locations are previous E-step: synthetic after 8 uses iter_7, Geysers after 12 uses iter_11. They are NOT post-M-step relocations.',
                'All source rows scored with equal event weight; no event subset or best-K selection.',
                'Geysers uses synthetic arrivals with catalog geometry, not observed picks; top datum already elevation 1166 m, no further correction.',
                'Soft Geysers runs are outside the primary comparisons: supplementary sensitivity scan, not a matched hard/soft ablation.',
                'No matched synthetic soft run was supplied/found; synthetic soft metrics unavailable.',
                'Legacy synthetic hard run lacks recorded source input/output hashes: verified comparison hashes, source ID/model hash, catalog count and station coordinates, but original arrival ingestion cannot be retrospectively hash-proven.',
                'Template amplitudes are descriptive projections against the known background, not regional RMSE lower bounds; layer/background reconstruction errors can contribute.',
            ],
            'datasets': result,
            'files_sha256': dict(sorted(self.files.items())),
            'evaluator_sha256': sha256(Path(__file__)),
            'software': {'python': sys.version.split()[0], 'numpy': np.__version__,
                         'scipy': __import__('scipy').__version__},
        }


@contextmanager
def deadline(seconds=120):
    require(0 < seconds <= 120, 'Evaluation timeout must be in (0, 120] seconds')
    def expired(signum, frame):
        raise TimeoutError(f'Read-only evaluation exceeded {seconds:g} seconds')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, default=BENCHMARK)
    parser.add_argument('--check', action='store_true', help='Recompute report and compare; no writes')
    parser.add_argument('--timeout-seconds', type=float, default=120, help='Positive, at most 120')
    args = parser.parse_args(argv)
    report_path = PAPER / 'supplemental_metrics.json'
    require(not report_path.is_symlink(), 'Refusing symlink report')
    require(args.check or not report_path.exists(), 'Report exists; use --check (never overwritten)')
    with deadline(args.timeout_seconds):
        report = Evaluation(args.benchmark_root.resolve()).run()
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.check:
            require(report_path.read_text() == text, 'Recomputed report differs')
            print(f'Verified reproducible report: {report_path}')
        else:
            with report_path.open('x') as stream:
                stream.write(text)
            print(f'Created {report_path}')


if __name__ == '__main__':
    main()
