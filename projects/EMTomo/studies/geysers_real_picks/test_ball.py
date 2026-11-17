"""Small, solver-free checks for the spatial-holdout evaluation protocol."""
import numpy as np
import pytest

from evaluate_ball import paired_interval, split_picks


def test_split_is_reproducible_and_excludes_missing_picks():
    observed = np.ones((4, 31))
    observed[0, 6:] = np.nan
    observed[1, 13:] = np.nan
    mask = split_picks(observed)
    np.testing.assert_array_equal(mask, split_picks(observed))
    assert np.isfinite(observed[mask]).all()
    assert np.all(mask.sum(axis=1) >= 1)
    assert np.all((np.isfinite(observed) & ~mask).sum(axis=1) >= 5)
    assert mask[0].sum() == 1
    assert mask[1].sum() == 3
    assert mask[2].sum() == 6


def test_split_rejects_insufficient_picks():
    with pytest.raises(ValueError, match='at least six'):
        split_picks(np.ones((1, 5)))


def test_paired_interval_direction_and_missing_values():
    baseline = np.array([[2., np.nan], [2., np.nan]])
    better = baseline / 2
    mask = np.array([[True, False], [True, False]])
    np.testing.assert_allclose(paired_interval(baseline, better, mask), [-1, -1])
    np.testing.assert_allclose(paired_interval(baseline, baseline, mask), [0, 0])
