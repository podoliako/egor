"""Hypothesis likelihood: depends on absolute noise and marginalizes the origin time."""

import numpy as np
import pytest

from instruments.likelihood import (
    PickNoise, candidate_weights, likelihood_normalization, negative_log_likelihood,
)


def _weights(predicted, observed, relative=0., absolute=0.1, model=0., temperature=1.):
    """Weights of candidates given as rows ``(n_candidates, n_stations)``."""
    nll = negative_log_likelihood(np.asarray(predicted).T, observed, PickNoise(relative, absolute, model))
    return candidate_weights(nll, temperature)


def test_two_candidate_probability_tracks_gap_and_noise():
    observed = np.array([0., 1.])
    closest = [1., 2.]
    weak = _weights([closest, [1., 2.2]], observed)
    strong = _weights([closest, [1., 2.4]], observed)
    # With two stations, a 0.2-s difference has chi² = 0.2² / (2 * 0.1²).
    assert weak[0] == pytest.approx(1. / (1. + np.exp(-1.)))
    assert strong[0] == pytest.approx(1. / (1. + np.exp(-4.)))
    diffuse = _weights([closest, [1., 2.4]], observed, absolute=0.4)
    assert .5 < diffuse[0] < weak[0]


def test_unknown_origin_time_cancels():
    predicted = [[1., 2., 3.], [1.1, 2., 3.]]
    observed = np.array([0., 1., 2.])
    expected = _weights(predicted, observed, model=0.2)
    np.testing.assert_allclose(_weights(predicted, observed + 30., model=0.2), expected)
    np.testing.assert_allclose(_weights(predicted, observed - 1., model=0.2), expected)


def test_heteroscedastic_likelihood_matches_explicit_formula():
    rng = np.random.default_rng(3)
    predicted = rng.uniform(1., 30., size=4) + rng.normal(0., 0.2, size=(6, 4))
    observed = predicted[0] - 3. + rng.normal(0., 0.1, size=4)
    noise = PickNoise(0.03, 0.05, 0.2)
    sigmas = noise.sigmas(predicted)
    precision = 1 / sigmas ** 2
    residuals = observed - predicted
    means = np.sum(precision * residuals, axis=1) / precision.sum(axis=1)
    chi2 = np.sum(precision * (residuals - means[:, None]) ** 2, axis=1)
    likelihood = np.exp(-chi2 / 2) / (np.prod(sigmas, axis=1) * np.sqrt(precision.sum(axis=1)))
    nll = negative_log_likelihood(predicted.T, observed, noise)
    np.testing.assert_allclose(candidate_weights(nll), likelihood / likelihood.sum())
    np.testing.assert_allclose(nll - nll.min(), -np.log(likelihood / likelihood.max()), atol=1e-9)


def test_homoscedastic_likelihood_ranks_like_pairwise_misfit():
    rng = np.random.default_rng(5)
    fields = rng.uniform(0., 20., size=(5, 3, 4, 2)).astype(np.float32)
    observed = rng.uniform(0., 5., size=5)
    residuals = observed[:, None, None, None] - fields
    pairwise = 5 * np.sum(residuals ** 2, axis=0) - np.sum(residuals, axis=0) ** 2
    nll = negative_log_likelihood(fields, observed, PickNoise(0., 0.1, 0.2))
    np.testing.assert_allclose(nll, pairwise / (2 * 5 * 0.05), rtol=1e-6)


def test_extreme_misfit_is_stable_and_temperature_tempers_likelihood():
    predicted = [[1., 2.], [1., 100.]]
    observed = np.array([0., 1.])
    np.testing.assert_array_equal(_weights(predicted, observed, 0., .05, .1), [1., 0.])
    warmer = _weights(predicted, observed, 0., .05, .1, temperature=10000.)
    assert 0 < warmer[1] < warmer[0]
    assert np.isclose(warmer.sum(), 1.)


@pytest.mark.parametrize("kwargs", [
    dict(relative_sigma=0., absolute_sigma_s=0., model_sigma_s=0.),
    dict(relative_sigma=-1., absolute_sigma_s=.1, model_sigma_s=.1),
    dict(relative_sigma=0., absolute_sigma_s=.1, model_sigma_s=float("nan")),
])
def test_bad_noise_rejected(kwargs):
    with pytest.raises(ValueError):
        PickNoise(**kwargs)


def test_bad_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        candidate_weights([0., 1.], temperature=0.)


def test_precomputed_normalization_is_reused_exactly():
    rng = np.random.default_rng(8)
    fields = rng.uniform(1., 20., size=(4, 3, 2, 5)).astype(np.float32)
    observed = rng.uniform(0., 3., size=4)
    noise = PickNoise(0.02, 0.05, 0.1)
    normalization = likelihood_normalization(fields, noise)
    np.testing.assert_allclose(negative_log_likelihood(fields, observed, noise, normalization),
                               negative_log_likelihood(fields, observed, noise))
    assert likelihood_normalization(fields, PickNoise(0., 0.05, 0.1)) == 0.0
