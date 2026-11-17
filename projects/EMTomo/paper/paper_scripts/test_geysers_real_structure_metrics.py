"""Shared-mask structure metrics tests; synthetic inputs, no seismic solvers."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_geysers_real_structure_metrics import (
    compare_grid, convergence, descriptive_metrics, midpoint_grid,
    published_mask, sha, verify_hashes,
)


def test_mask_is_strict_and_requires_both_finite_fields():
    values = np.array([[5., .4], [5., .400001], [np.nan, 1.],
                       [5., np.nan], [np.inf, 1.], [5., np.inf], [5., .1]])
    np.testing.assert_array_equal(published_mask(values), [False, True, False, False, False, False, False])


def test_difference_sign_units_rms_mae_and_pearson():
    result = descriptive_metrics([1000., 4000., 4000.], [2000., 3000., 5000.])
    assert result['rms_difference_m_s'] == 1000.
    assert result['mae_m_s'] == 1000.
    assert result['mean_signed_difference_m_s'] == pytest.approx(-1000 / 3)
    assert result['pearson_vp'] == pytest.approx(np.corrcoef([1., 4., 4.], [2., 3., 5.])[0, 1])
    assert descriptive_metrics([5000., 5000.], [4000., 6000.])['pearson_vp'] is None


@pytest.mark.parametrize('a,b', [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])])
def test_invalid_metric_vectors_rejected(a, b):
    with pytest.raises(ValueError, match='shared mask'):
        descriptive_metrics(a, b)


def test_midpoints_are_uniform_in_km_and_tile_full_domain():
    points, shape = midpoint_grid([1000, 500, 250], 125)
    np.testing.assert_array_equal(shape, [8, 4, 2])
    np.testing.assert_allclose(points[0], [.0625, .0625, .0625])
    np.testing.assert_allclose(points[-1], [.9375, .4375, .1875])
    np.testing.assert_allclose(points[1] - points[0], [0, 0, .125])
    assert len(points) * 125 ** 3 == 1000 * 500 * 250
    with pytest.raises(ValueError, match='exactly tile'):
        midpoint_grid([1000, 500, 250], 200)


class MaskedField:
    def sample(self, points):
        # Exclude x >= 0.75 by support and 0.5 <= x < 0.75 at the strict threshold.
        x = points[..., 0]
        vp = np.where(x < .75, 4. + x, np.nan)
        resolution = np.where(x < .5, .8, .4)
        return np.stack((vp, resolution), axis=-1)


def test_shared_mask_volume_containing_cells_and_batch_invariance():
    models = {
        'initial': (np.full((1, 1, 1), 5000.), 1.),
        'hard': (np.full((2, 2, 2), 4500.), .5),
        'soft': (np.full((2, 2, 2), 4000.), .5),
    }
    fine = compare_grid(models, MaskedField(), [1000] * 3, 125, batch_size=13)
    assert fine == compare_grid(models, MaskedField(), [1000] * 3, 125, batch_size=512)
    assert fine['total_samples'] == 512
    assert fine['finite_published_vp_samples'] == 384
    assert fine['finite_published_resolvability_le_threshold_samples'] == 128
    assert fine['shared_mask_samples'] == 256
    assert fine['shared_mask_volume_km3'] == .5
    assert fine['shared_mask_volume_m3'] == 5e8
    assert fine['metrics']['initial']['mean_signed_difference_m_s'] == pytest.approx(750.)
    assert fine['metrics']['hard']['mean_signed_difference_m_s'] == pytest.approx(250.)
    assert fine['metrics']['soft']['mean_signed_difference_m_s'] == pytest.approx(-250.)
    for name in models:
        assert fine['metrics'][name]['pearson_vp'] is None
    coarse = compare_grid(models, MaskedField(), [1000] * 3, 250)
    change = convergence(fine, coarse)
    assert change['shared_mask_volume_change_km3'] == 0
    assert change['metric_changes']['initial']['mean_signed_difference_m_s'] == pytest.approx(0)
    assert change['metric_changes']['initial']['pearson_vp'] is None


def test_nonfinite_model_is_not_silently_masked():
    with pytest.raises(ValueError, match='refusing to change the shared mask'):
        compare_grid({'initial': (np.full((1, 1, 1), np.nan), 1.)}, MaskedField(), [1000] * 3, 125)


def test_hash_verification_fails_closed_without_writing_files():
    path = Path(__file__).resolve()
    assert list(verify_hashes({str(path): sha(path)}).values()) == [sha(path)]
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify_hashes({str(path): '0' * 64})
