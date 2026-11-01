"""Local optimizer fallback tests; no observed-data or forward-solver access."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('holdout_eval', Path(__file__).with_name('evaluate_holdout.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_normal_refinement_unchanged(monkeypatch):
    import tomography.tomography_math as math
    point = np.array([.1, .2, .3])
    monkeypatch.setattr(math, 'refine_hypocentre_in_cell', lambda *a, **k: (point, -5.))
    got, cost, fallback = evaluation.safe_refine(None, None, None, None, None)
    np.testing.assert_array_equal(got, point)
    assert cost == -5 and fallback is False


def test_known_powell_error_uses_bounded_training_objective(monkeypatch):
    import tomography.tomography_math as math
    import instruments.hypocentre_objective as objectives
    def broken(*args, **kwargs):
        raise ValueError('zero-size array to reduction operation maximum which has no identity')
    monkeypatch.setattr(math, 'refine_hypocentre_in_cell', broken)
    target = np.array([.2, .3, .4])
    calls = []
    def objective(point):
        calls.append(np.array(point))
        return float(np.sum((point-target)**2))
    monkeypatch.setattr(objectives, 'make_refinement_objective', lambda *a, **k: objective)
    fields = np.zeros((3, 2, 2, 2), dtype=np.float32)
    point, cost, fallback = evaluation.safe_refine(fields, np.zeros(3), [0, 0, 0], None, np.arange(3))
    assert fallback is True and cost < 1e-12
    np.testing.assert_allclose(point, target, atol=1e-6)
    assert all(np.all((p >= 0) & (p <= .5)) for p in calls)


@pytest.mark.parametrize('error', [ValueError('other error'), RuntimeError('unrelated failure')])
def test_other_failures_not_hidden(monkeypatch, error):
    import tomography.tomography_math as math
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(math, 'refine_hypocentre_in_cell', broken)
    with pytest.raises(type(error), match=str(error)):
        evaluation.safe_refine(None, None, None, None, None)
