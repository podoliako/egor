"""Synthetic-only tests: no model, real picks, solver, or filesystem reads."""
import numpy as np
import pytest

from studies.geysers_real_picks.cv_metrics import profile_origin, summarize


def test_constant_offset_exact():
    times = np.array([3., 7., 11.])
    assert profile_origin(times, times - 3.) == -3.
    assert profile_origin([0., 2.], [0., 2.]) == 0.


def test_heteroscedastic_weighted():
    # Variances are 2 and 5, so normalized precisions are 5/7 and 2/7.
    assert profile_origin([1., 2.], [0., 3.], 1., 1.) == pytest.approx(-3 / 7)
    assert profile_origin([1., 2.], [0., 3.], 0., 1.) == 0.


def test_absent_prediction_cannot_change_origin():
    observed = np.array([0., np.nan, 2., np.nan])
    origin = profile_origin([3., 0., 5., 1.], observed)
    assert origin == profile_origin([3., 1e100, 5., 0.], observed)
    assert origin == pytest.approx(-3.)


def test_origin_fitting_ignores_holdout():
    times = np.array([3., 1., 7., 2.])
    train = np.array([True, False, True, False])
    for held_values in ([100., -100.], [-1e6, 1e6]):
        picks = times - 3.
        picks[~train] = held_values
        masked = np.where(train, picks, np.nan)
        assert profile_origin(times, masked) == -3.


@pytest.mark.parametrize('predicted,observed', [
    ([[1., 2.]], [[0., 1.]]), ([1., 2.], [[0., 1.]]), ([1.], [0., 1.]),
    ([1., np.nan], [0., 1.]), ([1., np.inf], [0., 1.]), ([-1., 2.], [0., 1.]),
    ([], []), ([1., 2.], [np.nan, np.nan]), ([1., 2.], [0., np.nan]),
    ([1., 2.], [0., np.inf]), ([1., 2.], [0., -np.inf]),
])
def test_bad_inputs(predicted, observed):
    with pytest.raises(ValueError):
        profile_origin(predicted, observed)


@pytest.mark.parametrize('relative,model', [
    (-.01, .02), (np.nan, .02), (np.inf, .02), (.01, -.02),
    (.01, np.nan), (.01, np.inf), (.01, 0.), (0., 0.),
    (True, .02), (.01, False), ([.01], .02), (.01, [.02]),
    (1e308, .02), (0., 1e-300),
])
def test_bad_noise(relative, model):
    with pytest.raises(ValueError):
        profile_origin([1., 2.], [0., 1.], relative, model)


def test_metric_numerics():
    result = summarize([-.2, -.1, 0., .1, .3])
    assert result == pytest.approx({
        'count': 5, 'rmse_s': np.sqrt(.03), 'mae_s': .14,
        'median_abs_s': .1, 'p90_abs_s': .26, 'bias_s': .02,
        'fraction_abs_gt_0p1_s': .4, 'fraction_abs_gt_0p2_s': .2,
    })
    assert isinstance(result['count'], int)
    assert summarize([0.]) == dict.fromkeys(result, 0) | {'count': 1}
    assert summarize([-.2])['fraction_abs_gt_0p2_s'] == 0.


@pytest.mark.parametrize('values', [[], [np.nan], [0., np.inf], [0., -np.inf], [[1.]], 1.])
def test_bad_residuals(values):
    with pytest.raises(ValueError):
        summarize(values)
