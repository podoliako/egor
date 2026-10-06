"""Likelihood weights must depend on absolute noise, not the candidate gap."""

import numpy as np
import pytest

from instruments.instruments_weights import (
    candidate_posterior_weights, compute_epicenter_weight_matrix,
    compute_weights_from_misfit,
)


def test_two_candidate_probability_tracks_gap_and_noise():
    observed = np.array([0., 1.])
    closest = np.array([1., 2.])
    weak = candidate_posterior_weights(
        np.stack([closest, [1., 2.2]]), observed,
        relative_sigma=0., absolute_sigma_s=0.1, model_sigma_s=0.,
    )
    strong = candidate_posterior_weights(
        np.stack([closest, [1., 2.4]]), observed,
        relative_sigma=0., absolute_sigma_s=0.1, model_sigma_s=0.,
    )
    # With two stations, a 0.2-s difference has chi² = 0.2² / (2 * 0.1²).
    assert weak[0] == pytest.approx(1. / (1. + np.exp(-1.)))
    assert strong[0] == pytest.approx(1. / (1. + np.exp(-4.)))
    assert strong[0] > weak[0]
    diffuse = candidate_posterior_weights(
        np.stack([closest, [1., 2.4]]), observed,
        relative_sigma=0., absolute_sigma_s=0.4, model_sigma_s=0.,
    )
    assert .5 < diffuse[0] < weak[0]


def test_unknown_event_origin_and_pick_reference_cancel():
    predicted = np.array([[1., 2., 3.], [1.1, 2., 3.]])
    observed = np.array([0., 1., 2.])
    kw = dict(relative_sigma=0., absolute_sigma_s=0.1, model_sigma_s=0.2)
    expected = candidate_posterior_weights(predicted, observed, **kw)
    np.testing.assert_allclose(candidate_posterior_weights(predicted, observed + 30., **kw), expected)
    np.testing.assert_allclose(candidate_posterior_weights(predicted, observed - 1., **kw), expected)


def test_heteroscedastic_marginal_likelihood_includes_normalization():
    predictions = np.array([[1., 2.], [2., 3.]])
    observed = np.array([0., 1.])  # Both candidates fit exactly up to origin shift.
    weights = candidate_posterior_weights(
        predictions, observed, relative_sigma=.1,
        absolute_sigma_s=.05, model_sigma_s=.2,
    )
    def norm(row):
        sigmas = np.hypot(np.hypot(.1 * row, .05), .2)
        return np.prod(1 / sigmas) / np.sqrt(np.sum(1 / sigmas**2))
    expected = np.array([norm(row) for row in predictions])
    np.testing.assert_allclose(weights, expected / expected.sum())
    assert weights[0] > weights[1]


def test_extreme_misfit_is_stable_and_temperature_tempers_likelihood():
    predicted = np.array([[1., 2.], [1., 100.]])
    observed = np.array([0., 1.])
    kw = dict(relative_sigma=0., absolute_sigma_s=.05, model_sigma_s=.1)
    np.testing.assert_array_equal(candidate_posterior_weights(predicted, observed, **kw), [1., 0.])
    first = candidate_posterior_weights(predicted, observed, temperature=1., **kw)
    warmer = candidate_posterior_weights(predicted, observed, temperature=10000., **kw)
    assert warmer[0] < first[0]
    assert np.isclose(warmer.sum(), 1.)


def test_generic_misfit_requires_absolute_scale():
    small = compute_weights_from_misfit([0., 1.], misfit_scale=1.)
    large = compute_weights_from_misfit([0., 10.], misfit_scale=1.)
    assert large[0] > small[0]
    with pytest.raises(TypeError, match="misfit_scale"):
        compute_weights_from_misfit([0., 1.])
    fields = np.zeros((2, 2, 1, 1))
    fields[1, :, 0, 0] = [1., 1.2]
    grid = compute_epicenter_weight_matrix(fields, np.array([0., 1.]),
                                            misfit_scale=.01)
    np.testing.assert_allclose(np.sum(grid), 1.)


@pytest.mark.parametrize("kwargs", [
    dict(relative_sigma=0., absolute_sigma_s=0., model_sigma_s=0.),
    dict(relative_sigma=-1., absolute_sigma_s=.1, model_sigma_s=.1),
    dict(relative_sigma=0., absolute_sigma_s=.1, model_sigma_s=float("nan")),
    dict(relative_sigma=0., absolute_sigma_s=.1, model_sigma_s=.1, temperature=0.),
])
def test_bad_likelihood_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        candidate_posterior_weights(np.array([[1., 2.], [1., 2.2]]),
                                    np.array([0., 1.]), **kwargs)
