"""Synthetic checks for latest fixed-model holdout evaluation; no real data reads."""
import numpy as np
import pytest

from studies.geysers_real_picks.evaluate_latest_holdout import (
    fitted_origin, heldout_indices, mask_sha, paired_bootstrap,
)


class Noise:
    def sigmas(self, times):
        return np.sqrt((.01 * times)**2 + .05**2 + .02**2)


def test_origin_matches_all_three_noise_components():
    times = np.array([1., 2., 50.])
    observed = np.array([0., 2., np.nan])
    weights = 1 / Noise().sigmas(times[:2])**2
    expected = np.average([-1., 0.], weights=weights)
    assert fitted_origin(times, observed, Noise()) == pytest.approx(expected)
    times[2] = 10000
    assert fitted_origin(times, observed, Noise()) == pytest.approx(expected)


def test_origin_needs_training():
    with pytest.raises(ValueError, match='Insufficient'):
        fitted_origin(np.ones(2), np.array([0., np.nan]), Noise())


def test_holdout_maps_ids_not_row_order():
    rows = [dict(event_id='e1', network='BG', station='A', seconds_from_train_reference='2.5')]
    ei, si, observed = heldout_indices(rows, ['e2', 'e1'], ['BG.B', 'BG.A'], np.full((2, 2), np.nan))
    np.testing.assert_array_equal(ei, [1])
    np.testing.assert_array_equal(si, [1])
    np.testing.assert_array_equal(observed, [2.5])


def test_holdout_overlap_and_duplicates_rejected():
    row = dict(event_id='e1', network='BG', station='A', seconds_from_train_reference='2.5')
    with pytest.raises(ValueError, match='overlaps'):
        heldout_indices([row], ['e1'], ['BG.A'], np.zeros((1, 1)))
    with pytest.raises(ValueError, match='Duplicate'):
        heldout_indices([row, row], ['e1'], ['BG.A'], np.full((1, 1), np.nan))


def test_mask_hash_uses_c_order_uint8():
    mask = np.array([[True, False], [True, True]])
    assert mask_sha(mask) == mask_sha(np.asfortranarray(mask))
    assert mask_sha(mask) != mask_sha(~mask)


def test_bootstrap_preserves_pairs_and_event_groups():
    events = np.array([0, 0, 1, 2, 2])
    hard = np.array([1., -2., 3., -4., 5.])
    same = paired_bootstrap(events, hard, hard, samples=40)
    assert same['soft_minus_hard_rmse_s_ci95'] == [0., 0.]
    half = paired_bootstrap(events, hard, hard / 2, samples=40)
    np.testing.assert_allclose(half['soft_rmse_gain_pct_ci95'], [50., 50.])
    assert half == paired_bootstrap(events, hard, hard / 2, samples=40)
