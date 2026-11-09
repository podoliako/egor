"""Targeted analytic tests; no saved runs, solvers, or filesystem writes required."""
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_metrics as metrics


class EnergyScoreTests(unittest.TestCase):
    def test_point_mass_is_euclidean_error(self):
        score = metrics.distribution_statistics([[3, 4, 0]], [7], [0, 0, 0])
        self.assertEqual(score['energy_score_m'], 5)
        self.assertEqual(score['map_error_m'], 5)
        self.assertEqual(score['weighted_mean_error_m'], 5)
        self.assertEqual(score['spread_squared_m2'], 0)

    def test_hard_catalog_energy_is_mean_not_rms(self):
        scores = [metrics.distribution_statistics([[d, 0, 0]], [1], [0, 0, 0]) for d in (0, 10)]
        summary = metrics.summarize_locations(scores)
        self.assertEqual(summary['energy_score_mean_m'], 5)
        self.assertAlmostEqual(summary['map_rms_m'], np.sqrt(50))
        self.assertEqual(summary['map_median_m'], 5)
        self.assertEqual(summary['map_p95_m'], 9.5)
        self.assertEqual(summary['weighted_rms_spread_m'], 0)

    def test_manual_two_atom_distribution(self):
        # Distances to truth 1 and 3, pair distance 4. ES = 2.5 - .75 = 1.75.
        score = metrics.distribution_statistics([[0, 0, 0], [4, 0, 0]], [.25, .75], [1, 0, 0])
        self.assertEqual(score['energy_score_m'], 1.75)
        self.assertEqual(score['map_error_m'], 3)
        self.assertEqual(score['weighted_mean_error_m'], 2)
        self.assertEqual(score['spread_squared_m2'], 3)
        summary = metrics.summarize_locations([score])
        self.assertAlmostEqual(summary['weighted_rms_spread_m'], np.sqrt(3))

    def test_translation_invariance(self):
        x = np.array([[1., 2., 3.], [-4., 5., 1.], [7., -8., 9.]])
        truth = np.array([3., -2., 1.])
        shift = np.array([10000., -500., 89.])
        a = metrics.distribution_statistics(x, [.2, .3, .5], truth)
        b = metrics.distribution_statistics(x + shift, [.2, .3, .5], truth + shift)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=9)

    def test_weight_normalization_and_zero_mass_atom(self):
        a = metrics.distribution_statistics([[0, 0, 0], [4, 0, 0]], [1, 3], [1, 0, 0])
        b = metrics.distribution_statistics([[0, 0, 0], [4, 0, 0], [999, 999, 999]],
                                            [.25, .75, 0], [1, 0, 0])
        self.assertEqual(a, b)

    def test_invalid_weights_rejected(self):
        for weights in ([0, 0], [-1, 2], [np.nan, 1], [np.inf, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                metrics.distribution_statistics([[0, 0, 0], [1, 0, 0]], weights, [0, 0, 0])

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            metrics.distribution_statistics([[0, 0, 0]], [1, 2], [0, 0, 0])

    def test_map_tie_uses_first_atom(self):
        score = metrics.distribution_statistics([[0, 0, 0], [4, 0, 0]], [.5, .5], [1, 0, 0])
        self.assertEqual(score['map_error_m'], 1)


class TemplateAmplitudeTests(unittest.TestCase):
    def setUp(self):
        # Nonconstant background and both fast/slow contrasts; masked exterior
        # has unrelated errors so the test detects accidental full-volume fits.
        self.background = np.array([3800., 4600., 5200., 5700.])
        self.truth = self.background + [100., -200., 50., 0.]
        self.mask = np.array([True, True, True, False])

    def test_background_zero_truth_one_and_scaled(self):
        for scale in (0., 1., .3, 2., -1.):
            field = self.background + scale * (self.truth - self.background)
            field[-1] = 100000.
            with self.subTest(scale=scale):
                self.assertAlmostEqual(metrics.template_amplitude(
                    field, self.truth, self.background, self.mask), scale)

    def test_broadcast_depth_background(self):
        background = np.array([3800., 4600.])[:, None, None]
        truth = np.broadcast_to(background, (2, 3, 4)) + 10
        self.assertEqual(metrics.template_amplitude(truth, truth, background,
                                                   np.ones(truth.shape, dtype=bool)), 1)

    def test_zero_contrast_rejected(self):
        with self.assertRaises(ValueError):
            metrics.template_amplitude(self.background, self.background, self.background, self.mask)


class CoordinatesAndSafetyTests(unittest.TestCase):
    def test_continuous_indices_to_metres(self):
        positions = np.array([[0., 1., 2.], [1.25, 2.5, 3.75]])
        expected = [[62.5, 187.5, 312.5], [218.75, 375., 531.25]]
        np.testing.assert_array_equal(metrics.positions_to_metres(positions, 125), expected)
        # Independent inverse of the native metric_to_cell_coord convention.
        np.testing.assert_allclose(np.array(expected) / 125 - .5, positions)

    def test_invalid_cell_size_rejected(self):
        for cell_size in (0, -1, np.nan):
            with self.subTest(cell_size=cell_size), self.assertRaises(ValueError):
                metrics.positions_to_metres([[0, 0, 0]], cell_size)

    def test_existing_report_refused_before_reading_runs(self):
        with patch.object(metrics.Path, 'is_symlink', return_value=False), \
             patch.object(metrics.Path, 'exists', return_value=True), \
             patch.object(metrics, 'Evaluation') as evaluation:
            with self.assertRaisesRegex(ValueError, 'Report exists'):
                metrics.main([])
            evaluation.assert_not_called()

    def test_symlink_report_refused_even_in_check_mode(self):
        with patch.object(metrics.Path, 'is_symlink', return_value=True), \
             patch.object(metrics, 'Evaluation') as evaluation:
            with self.assertRaisesRegex(ValueError, 'symlink'):
                metrics.main(['--check'])
            evaluation.assert_not_called()

    def test_timeout_cannot_exceed_120_seconds(self):
        with self.assertRaises(ValueError):
            with metrics.deadline(121):
                self.fail('Invalid deadline accepted')

    def test_recursive_config_difference(self):
        self.assertEqual(metrics.config_differences({'n_candidates': 1, 'p': {'mode': 'hard'}},
                                                    {'n_candidates': 5, 'p': {'mode': 'soft'}}),
                         {'n_candidates': {'hard': 1, 'soft': 5},
                          'p.mode': {'hard': 'hard', 'soft': 'soft'}})


if __name__ == '__main__':
    unittest.main()
