import unittest

import numpy as np

from diagnose import estimates, metrics


class DiagnosticTests(unittest.TestCase):
    def test_cell_centres_and_mean(self):
        mode, mean, spread, ess, _ = estimates([[0, 0, 0], [2, 0, 0]], [.75, .25], 100, [3, 1, 1])
        np.testing.assert_array_equal(mode, [50, 50, 50])
        np.testing.assert_array_equal(mean, [100, 50, 50])
        self.assertAlmostEqual(spread ** 2, 7500)
        self.assertAlmostEqual(ess, 1.6)

    def test_exact_hard_and_first_tie(self):
        mode, mean, _, ess, _ = estimates([[0, 0, 0], [2, 0, 0]], [0, 1], 125, [3, 1, 1])
        np.testing.assert_array_equal(mode, mean)
        self.assertEqual(ess, 1)
        mode, *_ = estimates([[0, 0, 0], [2, 0, 0]], [.5, .5], 125, [3, 1, 1])
        np.testing.assert_array_equal(mode, [62.5, 62.5, 62.5])

    def test_invalid_weights(self):
        for weights in ([.5, -.1], [float('nan'), 1], [float('inf'), 0], [0, 0], [.4, .4]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                estimates([[0, 0, 0], [2, 0, 0]], weights, 100, [3, 1, 1])

    def test_normalizes_saved_roundoff(self):
        mode, mean, *_ = estimates([[0, 0, 0], [2, 0, 0]], [.7500003, .2500001], 100, [3, 1, 1])
        np.testing.assert_allclose(mean, [100, 50, 50], atol=1e-12)

    def test_spatial_rmse_and_linear_percentile(self):
        result = metrics(np.array([0., 5.]))
        self.assertEqual(result['p90_m'], 4.5)
        self.assertAlmostEqual(result['spatial_rmse_m'], np.sqrt(12.5))


if __name__ == '__main__':
    unittest.main()
