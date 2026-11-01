"""Frozen-runtime and provenance helpers; no numerical imports until activation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path('/mnt/disk01/egor/projects/EMTomo')
STUDY_ID = 'geysers_pick_noise_20261027'
BASE = PROJECT / 'runs/matched_geometry_20261021'
EXPERIMENT = 'geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001'
VARIANTS = ('hard-k15-top1-dmin1', 'soft-k15-top10-dmin1', 'soft-k15-top10-dmin2')
MULTIPLIERS = (('x0p5', .5), ('x2', 2.), ('x3', 3.), ('x5', 5.), ('x10', 10.))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_hashes(root, expected):
    root = Path(root).resolve()
    for name, checksum in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or sha(path) != checksum:
            raise ValueError(f'Frozen/provenance file changed: {path}')


def scientific(config):
    return {k: v for k, v in config.items() if k not in ('run_name', 'runs_dir')}


def verify_manifest(manifest):
    if manifest['study_id'] != STUDY_ID or manifest['schema_version'] != 1:
        raise ValueError('Unexpected study/schema')
    state = Path(manifest['state_dir']).resolve()
    if not state.is_relative_to(PROJECT / 'runs') or state == PROJECT / 'runs':
        raise ValueError('State must be a dedicated project/runs directory')
    if Path(manifest['runtime_root']) != BASE / 'snapshot':
        raise ValueError('Expected the original matched-series frozen runtime')
    if sha(BASE / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Original matched-series manifest changed')
    original = read(BASE / 'manifest.json')
    expected_runtime = {name: digest for name, digest in original['snapshot_sha256'].items()
                        if name.endswith('.py') and name.startswith(('projects/', 'instruments/'))}
    if manifest['runtime_sha256'] != expected_runtime:
        raise ValueError('Numerical source ledger differs from original matched runtime')
    check_hashes(manifest['runtime_root'], expected_runtime)
    if Path(manifest['runner_root']) != state / 'runner':
        raise ValueError('Unexpected frozen runner directory')
    check_hashes(manifest['runner_root'], manifest['runner_sha256'])
    if Path(manifest['experiments_root']) != state / 'experiments':
        raise ValueError('Derived experiments must stay in the new study')
    source = BASE / 'snapshot/projects/forward_modeling/experiments'
    if Path(manifest['source_experiments_root']) != source or manifest['source_experiment_id'] != EXPERIMENT:
        raise ValueError('Unexpected original source experiment')
    check_hashes(source / 'input' / EXPERIMENT, manifest['source_input_sha256'])
    check_hashes(source / 'output' / EXPERIMENT, manifest['source_output_sha256'])
    expected_levels = [dict(id=key, multiplier=factor, relative_sigma=.01 * factor,
                            absolute_sigma_s=.01 * factor, seed=42,
                            experiment_id=f'geysers_spheres_picknoise_20261027_{key}')
                       for key, factor in MULTIPLIERS]
    if manifest['levels'] != expected_levels:
        raise ValueError('Noise grid changed')
    if set(manifest['references']) != set(VARIANTS):
        raise ValueError('Expected three matched configurations')
    for variant in VARIANTS:
        ref = manifest['references'][variant]
        case_id = f'geysers-em-{variant}'
        old_case = next(c for c in original['cases'] if c['id'] == case_id)
        result_file = BASE / 'cases' / case_id / 'result.json'
        result = read(result_file)
        if (ref['case_id'] != case_id or ref['config'] != old_case['config']
                or ref['run_dir'] != result['run_dir'] or result['status'] != 'completed'
                or sha(result_file) != ref['result_sha256']):
            raise ValueError(f'Original case provenance changed: {variant}')
        check_hashes(ref['run_dir'], ref['run_sha256'])
    if len(manifest['cases']) != 15 or len({c['id'] for c in manifest['cases']}) != 15:
        raise ValueError('Expected fifteen distinct inversions')
    expected_cases = {(level['id'], variant) for level in expected_levels for variant in VARIANTS}
    if {(c['level_id'], c['variant']) for c in manifest['cases']} != expected_cases:
        raise ValueError('Incomplete experiment matrix')
    for case in manifest['cases']:
        level = next(x for x in expected_levels if x['id'] == case['level_id'])
        ref = manifest['references'][case['variant']]
        if (case['id'] != f"{case['level_id']}-{case['variant']}"
                or case['experiment_id'] != level['experiment_id']
                or case['reference_run'] != ref['run_dir']
                or scientific(case['config']) != scientific(ref['config'])
                or case['config']['runs_dir'] != str(PROJECT / 'runs')
                or case['config']['run_name'] != f"gpnoise27-{case['id']}"):
            raise ValueError(f'Case configuration changed: {case["id"]}')


def load_runtime(manifest):
    verify_manifest(manifest)
    if any(name in sys.modules for name in ('config', 'experiment_runner', 'projects', 'tomography')):
        raise ValueError('Use a fresh process for frozen-runtime activation')
    root = Path(manifest['runtime_root'])
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / 'projects/EMTomo'), str(root)]
    config_module = importlib.import_module('config')
    runner = importlib.import_module('experiment_runner')
    em = importlib.import_module('tomography.tomography_em')
    for name, module in tuple(sys.modules.items()):
        if (name in ('config', 'experiment_runner', 'experiment_data', 'eikonal', 'raytracing',
                     'interpolation', 'velocity_model') or
                name.startswith(('projects.', 'instruments.', 'tomography.'))):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise ValueError(f'Non-frozen import: {name}: {location}')
    return SimpleNamespace(config_module=config_module, runner=runner, em=em)
