"""Small estimator tests; full artifact/provenance checks run in the plot generator."""
import unittest
import numpy as np

from make_soft_velocity_gain_vs_weighted_mean_hypocenter_gain import point_locations


class WeightedMeanTests(unittest.TestCase):
    def test_full_normalized_mean_and_cell_centres(self):
        positions = np.array([[0., 0., 0.], [2., 4., 6.]])
        weights = np.array([0.25, 0.75]) * (1 + 1e-8)
        mean, maximum, total = point_locations(positions, weights, 125., [10]*3)
        np.testing.assert_allclose(mean, [250., 437.5, 625.])
        np.testing.assert_array_equal(maximum, [312.5, 562.5, 812.5])
        self.assertAlmostEqual(total, 1 + 1e-8)

    def test_hard_ignores_zero_weight_search_candidates(self):
        positions = np.array([[0., 0., 0.], [8., 8., 8.], [4., 4., 4.]])
        mean, maximum, _ = point_locations(positions, np.array([0., 1., 0.]), 125., [10]*3)
        np.testing.assert_array_equal(mean, maximum)
        np.testing.assert_array_equal(mean, [1062.5]*3)

    def test_tiny_positive_weight_not_thresholded(self):
        mean, _, _ = point_locations(np.array([[0., 0., 0.], [8., 8., 8.]]),
                                     np.array([1-1e-8, 1e-8]), 125., [10]*3)
        np.testing.assert_allclose(mean, [62.50001]*3, rtol=0, atol=1e-12)

    def test_invalid_weights_rejected(self):
        positions = np.zeros((2, 3))
        for weights in ([0., 0.], [-1., 2.], [np.nan, 1.], [0.2, 0.2]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                point_locations(positions, np.array(weights), 125., [10]*3)


if __name__ == "__main__":
    unittest.main()
