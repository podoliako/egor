"""Solver-free completion checks for the real-data queue."""
import hashlib
import json

import numpy as np
import pytest

from studies.geysers_real_picks.worker import check_native
from studies.geysers_soft_screen.pipeline import result_for


def test_generic_completion_artifacts(tmp_path):
    artifact = tmp_path / 'Vp_model.dat'
    artifact.write_text('4.1 4.2\n')
    result = dict(status='completed', run_dir=str(tmp_path),
                  artifact_sha256={artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()})
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    case = dict(completion_artifacts=['Vp_model.dat'])
    assert result_for(case, path, tmp_path) == result
    artifact.write_text('changed')
    with pytest.raises(ValueError, match='changed'): result_for(case, path, tmp_path)
    for names in ([], ['../outside'], ['/absolute']):
        with pytest.raises(ValueError): result_for(dict(completion_artifacts=names), path, tmp_path)


def test_native_completion_requires_iterations_and_finite_outputs(tmp_path):
    (tmp_path / 'stdout.log').write_text('updating the velocities\n' * 3 + 'writing out results ...')
    (tmp_path / 'MOD').write_text('.001 2 2 2\n')
    (tmp_path / 'Vp_model.dat').write_text('4.1 ' * 8)
    row = '1 38.8 -122.7 2 0 0 0 1 1 1\n'
    (tmp_path / 'tomoDD.reloc').write_text(row)
    result = check_native(tmp_path, {1, 2})
    assert result['retained_events'] == 1 and result['missing_event_ids'] == [2]
    (tmp_path / 'Vp_model.dat').write_text('nan ' * 8)
    with pytest.raises(ValueError, match='Vp'): check_native(tmp_path, {1, 2})
    (tmp_path / 'Vp_model.dat').write_text('4.1 ' * 8)
    (tmp_path / 'tomoDD.reloc').write_text(row * 2)
    with pytest.raises(ValueError, match='Duplicate'): check_native(tmp_path, {1, 2})
    (tmp_path / 'tomoDD.reloc').write_text(row)
    (tmp_path / 'stdout.log').write_text('normal exit but no inversion')
    with pytest.raises(ValueError, match='three'): check_native(tmp_path, {1, 2})
