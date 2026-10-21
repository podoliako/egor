"""One forward/EMTomo/tomoDD case for the scaled checkerboard study."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(manifest):
    root = Path(manifest['snapshot_root'])
    for name, expected in manifest['snapshot_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Snapshot changed: {name}')
    for name, expected in manifest['build_sha256'].items():
        if digest(Path(manifest['state_dir']) / name) != expected:
            raise ValueError(f'Build changed: {name}')


def greedy_pairs(event_xyz_km, max_distance_km, max_degree):
    """Same sorted greedy edges as the adapter, without rejecting absolute-only events."""
    import numpy as np
    candidates = []
    for first in range(len(event_xyz_km)):
        distances = np.linalg.norm(event_xyz_km[first + 1:] - event_xyz_km[first], axis=1)
        for offset in np.flatnonzero(distances <= max_distance_km):
            second = first + 1 + int(offset)
            candidates.append((float(distances[offset]), first, second))
    candidates.sort()
    degree = np.zeros(len(event_xyz_km), dtype=int)
    selected = []
    for _, first, second in candidates:
        if degree[first] >= max_degree or degree[second] >= max_degree:
            continue
        selected.append((first, second))
        degree[first] += 1
        degree[second] += 1
    return selected


def runtime(manifest):
    import importlib
    root = Path(manifest['snapshot_root'])
    sys.path[:0] = [str(root / 'projects/EMTomo'), str(root / 'benchmark'), str(root)]
    from projects.forward_modeling.experiments import load_inputs, run_experiment
    from projects.forward_modeling.model import ForwardConfig
    from projects.forward_modeling.noise import NoiseConfig
    config = importlib.import_module('config')
    runner = importlib.import_module('experiment_runner')
    em = importlib.import_module('tomography.tomography_em')
    from tomodd_benchmark import external_checkerboard as adapter
    for name, loaded in tuple(sys.modules.items()):
        if name in ('config', 'experiment_data', 'experiment_runner', 'eikonal', 'interpolation',
                    'velocity_model', 'raytracing') or name.startswith(('projects.', 'tomography.', 'instruments.', 'tomodd_benchmark.')):
            path = getattr(loaded, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError(f'Live module imported: {name}: {path}')
    adapter.SUPPORT_DIR = root / 'support'
    adapter.select_event_pairs = greedy_pairs
    return config.InversionConfig, runner, em, adapter, load_inputs, run_experiment, ForwardConfig, NoiseConfig


def validate_inputs(manifest, case, api):
    import numpy as np
    InversionConfig, runner, em, adapter, load_inputs, _, _, _ = api
    forward_root = Path(manifest['forward_root'])
    model, stations, events = load_inputs(forward_root / 'experiments', case['experiment_id'])
    if len(stations.ids) != 98 or len(events.ids) != 1200:
        raise ValueError('Incorrect event/station counts')
    np.testing.assert_allclose(np.array(model.velocity.shape) * model.cell_size_m, [24000, 12000, 12000])
    np.testing.assert_array_equal(np.unique(model.velocity), [4750., 5250.])
    if case['method'] != 'forward':
        forward = Path(manifest['state_dir']) / 'cases' / f"{case['geometry']}-forward/result.json"
        ledger = json.loads(forward.read_text())
        if ledger['status'] != 'completed':
            raise ValueError('Forward computation not completed')
        for path, expected in ledger['output_sha256'].items():
            if digest(path) != expected:
                raise ValueError(f'Forward output changed: {path}')
        data = runner.prepare_inversion(case['experiment_id'], InversionConfig(**manifest['em_config']),
                                        experiments_root=forward_root / 'experiments')
        if tuple(data.noise_sigmas) != (.01, .01) or data.initial_model.shape != (24, 12, 12):
            raise ValueError('Unexpected noise/grid')
        if not np.all(data.initial_model.velocity == 5000.):
            raise ValueError('Initial model is not homogeneous 5000 m/s')
        if case['method'] == 'tomodd':
            external = adapter.load_external_data(forward_root, case['experiment_id'])
            initial, _ = adapter.perturb_initial_event_xyz(external.event_xyz_km, external.lengths_km, 1., 42)
            pairs = adapter.select_event_pairs(initial, 3., 4)
            if 117600 + len(pairs) * 98 > adapter.MAX_DATA:
                raise ValueError('MAXDATA capacity exceeded')
    return model


def run_forward(manifest, case, api):
    _, runner, _, _, _, run_experiment, ForwardConfig, NoiseConfig = api
    output = run_experiment(root=Path(manifest['forward_root']) / 'experiments',
                            experiment_id=case['experiment_id'],
                            config=ForwardConfig(refinement=case['refinement'], source_radius_cells=2),
                            workers=12, noise=NoiseConfig(relative_sigma=.01, absolute_sigma_s=.01, seed=42))
    # Ingestion also verifies hashes, units, IDs, all 117600 picks and event time references.
    from experiment_data import load_tomography_experiment
    data = load_tomography_experiment(case['experiment_id'], Path(manifest['forward_root']) / 'experiments')
    assert data.arrival_times_s.shape == (1200, 98) and data.noise_sigmas == (.01, .01)
    return dict(output=str(output), output_sha256={str(output / name): digest(output / name)
                                                for name in ('arrivals.csv', 'metadata.json')})


def run_em(manifest, case, api, result, result_path):
    import numpy as np
    InversionConfig, runner, em, *_ = api
    config = InversionConfig(**{**manifest['em_config'], 'run_name': case['run_name']})
    prepared = runner.prepare_inversion(case['experiment_id'], config,
                                       experiments_root=Path(manifest['forward_root']) / 'experiments')
    forward = json.loads((Path(manifest['state_dir']) / 'cases' / f"{case['geometry']}-forward/result.json").read_text())
    provenance = dict(study_id=manifest['study_id'], case_id=case['id'],
                      manifest_sha256=manifest['_manifest_sha256'], snapshot_root=manifest['snapshot_root'],
                      code_and_input_sha256=manifest['snapshot_sha256'], output_sha256=forward['output_sha256'],
                      effective_config=asdict(config), final_event_locations_precede_last_velocity_update=True)

    class Logger(em.TomographyLogger):
        def save_meta(self, **kwargs):
            kwargs['run_params']['provenance'] = provenance
            super().save_meta(**kwargs)

    logger = Logger(base_dir=config.runs_dir, run_name=config.run_name, run_version=config.run_version,
                    run_tags={'ncand': 15, 'topn': 1, 'mode': 'hard', 'dmin': 1})
    result.update(run_dir=str(logger.run_dir.resolve()), config=asdict(config), provenance=provenance)
    save(result_path, result)
    print(f"Run directory: {logger.run_dir}", flush=True)
    em.warm_up_jit()
    em.run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
              runner.resolve_pick_noise(config, prepared.noise_sigmas), reference_model=prepared.reference_model,
              source_experiment={'id': case['experiment_id'], 'model_sha256': digest(
                  Path(manifest['forward_root']) / 'experiments/input' / case['experiment_id'] / 'model.npz')},
              run_name=config.run_name, logger=logger)
    final = np.load(logger.run_dir / 'final_model.npy', allow_pickle=False)
    assert final.shape == (24, 12, 12) and np.isfinite(final).all() and (final > 0).all()
    quality = [json.loads(row) for row in (logger.run_dir / 'quality.jsonl').read_text().splitlines()]
    assert [row['iter'] for row in quality] == list(range(12))
    for step in range(12):
        cycle = logger.run_dir / f'iter_{step}'
        assert json.loads((cycle / 'complete.json').read_text())['iter'] == step
        assert len(list(cycle.glob('event_*/weights.npz'))) == 1200
    return dict(final_model_sha256=digest(logger.run_dir / 'final_model.npy'))


def run_dd(manifest, case, api):
    _, _, _, adapter, *_ = api
    output = Path(manifest['state_dir']) / 'outputs' / case['id']
    if output.exists():
        raise FileExistsError(output)
    import matplotlib
    matplotlib.use('Agg')
    metrics = adapter.run(forward_root=Path(manifest['forward_root']), experiment_id=case['experiment_id'],
                          run_dir=output, iterations=3, timeout_s=0, max_pair_distance_km=3., max_pair_degree=4,
                          node_spacing_km=1., free_buffer_layers=0, initial_location_sigma_km=1.,
                          initial_location_seed=42, binary=Path(manifest['binary']), ray_step_km=.05)
    assert metrics['final_retained_event_count'] == 1200 and not metrics['unrelocated_event_ids']
    log = (output / 'work/inversion/tomoDD-SE.log').read_text(errors='replace')
    import re
    for step in (1, 2, 3):
        assert re.search(rf'Iteration\s+{step}\s+finished', log)
        assert len(adapter.parse_relocated_events(output / f'work/inversion/tomoDD.reloc.001.{step:03d}')) == 1200
    import numpy as np
    with np.load(output / 'state.npz', allow_pickle=False) as saved:
        pairs = greedy_pairs(saved['initial_event_xyz_km'], 3., 4)
    degree = np.zeros(1200, dtype=int)
    for first, second in pairs:
        degree[first] += 1
        degree[second] += 1
    data = adapter.load_external_data(Path(manifest['forward_root']), case['experiment_id'])
    isolated = [data.event_ids[i] for i in np.flatnonzero(degree == 0)]
    return dict(run_dir=str(output), metrics=metrics,
                graph_policy='Unmodified greedy edges; isolated events retain all absolute picks',
                absolute_only_event_ids=isolated, pair_count=len(pairs),
                results_sha256=digest(output / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    manifest = json.loads(args.manifest.read_text())
    manifest['_manifest_sha256'] = digest(args.manifest)
    case = next(case for case in manifest['cases'] if case['id'] == args.case_id)
    verify(manifest)
    api = runtime(manifest)
    validate_inputs(manifest, case, api)
    if args.validate_only:
        print(json.dumps({'case': case['id'], 'status': 'validated', 'solver_launched': False}), flush=True)
        return
    folder = Path(manifest['state_dir']) / 'cases' / case['id']
    folder.mkdir(exist_ok=False)
    path = folder / 'result.json'
    result = dict(case_id=case['id'], method=case['method'], status='running', started_utc=now(),
                  worker_pid=os.getpid(), experiment_id=case['experiment_id'])
    save(path, result)
    print(f"Starting {case['id']}: {case['method']}, experiment {case['experiment_id']}", flush=True)
    started = time.monotonic()
    try:
        if case['method'] == 'forward':
            extra = run_forward(manifest, case, api)
        elif case['method'] == 'emtomo':
            extra = run_em(manifest, case, api, result, path)
        else:
            extra = run_dd(manifest, case, api)
        verify(manifest)
        result.update(extra, status='completed')
    except BaseException as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=now(), elapsed_s=time.monotonic() - started)
        save(path, result)


if __name__ == '__main__':
    main()
