"""Create new scaled inputs and freeze the approved nine-job plan. No solves here."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    base = Path(__file__).resolve().parents[2]
    state = base / 'runs/checkerboard_scaled_20261021'
    snapshot = state / 'snapshot'
    manifest_path = state / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    sys.path[:0] = [str(snapshot / 'projects/EMTomo'), str(snapshot)]
    from config import InversionConfig
    from projects.forward_modeling.experiments import load_inputs, save_inputs
    from projects.forward_modeling.model import VelocityGrid, PointSet
    old_comparison = Path('/home/egor/tomodd_benchmark/runs/comparison_nx4_nx6_nx8_locnoise2km_seed42/metrics.json')
    old_cases = json.loads(old_comparison.read_text())['cases']
    source_root = base.parent / 'forward_modeling/experiments'
    forward_root = snapshot / 'projects/forward_modeling'
    definitions = []
    source_hashes = {str(old_comparison): sha(old_comparison)}
    for key, old_case in old_cases.items():
        if key not in ('nx4', 'nx6', 'nx8'):
            raise ValueError(key)
        metadata = old_case['metadata']
        old_id = metadata['source_experiment']
        source = source_root / 'input' / old_id
        for name in ('model.npz', 'events.csv', 'stations.csv', 'generation.json'):
            source_hashes[str(source / name)] = sha(source / name)
        assert sha(source / 'model.npz') == metadata['forward_model_sha256']
        assert sha(source / 'events.csv') == metadata['events_sha256']
        assert sha(source / 'stations.csv') == metadata['stations_sha256']
        model, stations, events = load_inputs(source_root, old_id)
        scaled_model = VelocityGrid(model.velocity, model.cell_size_m / 10,
                                    tuple(np.asarray(model.origin_m) / 10))
        scaled_stations = PointSet(stations.ids, stations.coordinates_m / 10)
        scaled_events = PointSet(events.ids, events.coordinates_m / 10)
        generation = json.loads((source / 'generation.json').read_text())
        parameters = generation['parameters']
        parameters['lengths_km'] = [value / 10 for value in parameters['lengths_km']]
        parameters['min_depth_km'] /= 10
        generation['derived_from'] = dict(experiment_id=old_id, linear_scale=.1,
            source_input_sha256={name: source_hashes[str(source / name)] for name in
                                 ('model.npz', 'events.csv', 'stations.csv', 'generation.json')},
            coordinates='Original event/station identities and coordinates divided by ten; not regenerated')
        new_id = f'checkerboard_24x12x12_{key}_98st_1200ev_grid20x10x6_minz3_noise01_pick001_seed42'
        destination = save_inputs(forward_root / 'experiments', new_id, scaled_model,
                                  scaled_stations, scaled_events, generation=generation)
        reloaded, new_stations, new_events = load_inputs(forward_root / 'experiments', new_id)
        np.testing.assert_array_equal(reloaded.velocity, model.velocity)
        np.testing.assert_array_equal(new_stations.coordinates_m, stations.coordinates_m / 10)
        np.testing.assert_array_equal(new_events.coordinates_m, events.coordinates_m / 10)
        assert np.array_equal(np.array(reloaded.velocity.shape) * reloaded.cell_size_m, [24000, 12000, 12000])
        refinement = int(round(reloaded.cell_size_m / 125))
        assert reloaded.cell_size_m / refinement == 125
        definitions.append(dict(geometry=key, experiment_id=new_id, source_experiment=old_id,
                                refinement=refinement, input_dir=str(destination)))
    definitions.sort(key=lambda row: row['geometry'])
    for name in ('stations.csv', 'events.csv'):
        assert len({sha(Path(d['input_dir']) / name) for d in definitions}) == 1
    build = json.loads((state / 'build_reference.json').read_text())
    binary = Path(build['binary'])
    solver_manifest = json.loads((binary.parent / 'manifest.json').read_text())
    assert solver_manifest['status'] == 'complete'
    assert solver_manifest['recipe']['min_segment_km'] == .05
    assert solver_manifest['recipe']['coordinate_roundoff_fix']
    previous_build = base / 'runs/matched_geometry_20261021/snapshot/solver'
    differences = []
    for file in previous_build.rglob('*'):
        if file.is_file() and file.suffix in ('.f', '.c', '.h', '.inc'):
            relative = file.relative_to(previous_build)
            other = binary.parent / relative
            if file.read_bytes() != other.read_bytes():
                differences.append(str(relative))
    assert differences == ['include/tomoFDD.inc'], differences
    previous_include = (previous_build / 'include/tomoFDD.inc').read_text()
    new_include = (binary.parent / 'include/tomoFDD.inc').read_text()
    import re
    assert re.sub(r'(MAXOBS\s*=\s*)120', r'\g<1>60', new_include) == previous_include
    common = asdict(InversionConfig(cell_size=1000., subdivision=8, initial_velocity_m_s=5000.,
          n_cycles=12, n_candidates=15, weights_top_n=1, weights_min_distance=1, candidate_mode='hard',
          temperature=1., weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.01,
          weight_model_sigma_s=.02, lambda_reg=1., coverage_damping_power=2.,
          max_velocity_step_fraction=.02, n_workers=12, runs_dir=str(base / 'runs')))
    cases = []
    forward_ids = [f"{d['geometry']}-forward" for d in definitions]
    for method in ('forward', 'tomodd', 'emtomo'):
        for definition in definitions:
            case = {**definition, 'id': f"{definition['geometry']}-{method}", 'method': method,
                    'depends_on': [] if method == 'forward' else forward_ids,
                    'run_name': f"cbscale21-{definition['geometry']}-hard-k15"}
            case['command'] = [sys.executable, '-B', str(snapshot / 'runner/worker.py'),
                               '--manifest', str(manifest_path), '--case-id', case['id']]
            cases.append(case)
    snapshot_hashes = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts
                      and not p.name.endswith(('.pyc', '.nbc', '.nbi'))}
    # The forward outputs do not exist yet; workers hash and publish them on completion.
    build_hashes = {str(p.relative_to(state)): sha(p) for p in sorted(binary.parent.rglob('*')) if p.is_file()}
    manifest = dict(study_id='checkerboard_scaled_20261021', schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(), state_dir=str(state), snapshot_root=str(snapshot),
        forward_root=str(forward_root), snapshot_sha256=snapshot_hashes, build_sha256=build_hashes,
        binary=str(binary), source_sha256=source_hashes, em_config=common, cases=cases,
        parallel_limits={'forward': 3, 'emtomo': 3, 'tomodd': 2},
        python=sys.executable, versions={name: importlib.metadata.version(name) for name in
           ('numpy', 'scipy', 'numba', 'scikit-fmm', 'pykonal', 'matplotlib')},
        protocol=dict(linear_scale=.1, lengths_km=[24,12,12], inversion_cell_km=1,
            checker_sizes_km=[6,4,3], velocity_m_s=[4750,5250], initial_velocity_m_s=5000,
            forward_grid_m=125, forward_workers=12, source_radius_cells=2,
            pick_noise_relative_sigma=.01, pick_noise_absolute_sigma_s=.01, seed=42,
            tomodd_initial_coordinate_sigma_km=1, tomodd_iterations=3,
            tomodd_smoothing=[5,5,5], tomodd_damping=40, tomodd_pair_degree=4,
            tomodd_pair_max_distance_km=3, tomodd_ray_step_km=.05,
            tomodd_build_change='MAXOBS 60 -> 120 only; same solver sources/patches as sphere series',
            not_scale_equivalent_to_old_experiment=True,
            note='Current matched sphere numerical runtime, homogeneous checkerboard initialization; no truth-based checkpoint choice'))
    with manifest_path.open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    (state / 'source_git_status.txt').write_text(subprocess.check_output(
        ['git','--no-pager','--no-optional-locks','status','--short'], cwd=base, text=True))
    print(f'Manifest: {manifest_path}')
    print(f'Jobs: {len(cases)}; copied model sizes/refinements: {[(d["geometry"],d["refinement"]) for d in definitions]}')
    print('Verified: three byte-identical scaled catalogs; solver source differs only in MAXOBS capacity.')


if __name__ == '__main__':
    main()
