"""Focused solver-free tests; temporary files stay in the approved run tree."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('homogeneous_check', Path(__file__).with_name('run_check.py'))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=check.OUTPUT / 'validation')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cartesian_distances_original_xyz(self):
        np.testing.assert_array_equal(check.distances([[10, 3.5, 5]], [[13, 7.5, 5], [10, 3.5, 0]]), [[5, 5]])

    def test_spherical_chord_standard_latitude_not_surface_arc(self):
        actual = check.chord_times([[0, 0, 0]], [[0, 90, 0], [90, 0, 0]])
        np.testing.assert_allclose(actual, np.full((1, 2), np.sqrt(2) * 6371 / 5), rtol=1e-14)
        self.assertLess(actual[0, 0], np.pi * 6371 / 10)

    def test_spherical_radius_depth_and_symmetry(self):
        source = [[30.3, 103.3, 5]]
        receiver = [[30.3, 103.3, 0], [30.3, 103.3, -1]]
        np.testing.assert_allclose(check.chord_times(source, receiver), [[1., 1.2]], atol=2e-13)
        np.testing.assert_allclose(check.chord_times(receiver, source).T, check.chord_times(source, receiver))

    def test_event_serialized_float32_then_float64_and_ids(self):
        path = self.root / 'true_event.dat'
        path.write_text('20260101 00000000 30.31235 103.31235 1.235 2 0 0 0 7 0\n')
        values = check.parse_events(path, [7])
        self.assertEqual(values.dtype, np.float64)
        np.testing.assert_array_equal(values, np.array([[30.31235, 103.31235, 1.235]], dtype=np.float32).astype(float))
        self.assertNotEqual(values[0, 1], 103.31235)
        with self.assertRaisesRegex(ValueError, 'ID/order'):
            check.parse_events(path, [8])

    def test_station_height_sign_and_float32_division(self):
        path = self.root / 'station.dat'
        path.write_text('A 30.31235 103.31235 -1166\nB 30.30000 103.30000 125\n')
        values = check.parse_stations(path, ['A', 'B'])
        np.testing.assert_array_equal(values[:, 2], np.array([np.float32(1166) / np.float32(1000),
                                                            -np.float32(125) / np.float32(1000)], dtype=float))
        self.assertGreater(values[0, 2], 0)
        self.assertLess(values[1, 2], 0)
        with self.assertRaisesRegex(ValueError, 'ID/order'):
            check.parse_stations(path, ['B', 'A'])

    def test_metrics_signed_bias_quantile_and_event_gauge(self):
        values = np.array([[.001, .003], [-.001, .001]])
        result = check.metrics(values)
        self.assertAlmostEqual(result['signed_bias_ms'], 1)
        self.assertAlmostEqual(result['rms_ms'], np.sqrt(3))
        self.assertAlmostEqual(result['p95_abs_ms'], 2.7)
        self.assertEqual(result['max_abs_ms'], 3)
        self.assertEqual(result['per_event_demeaned_rms_ms'], 1)
        shifted = check.metrics(values + np.array([[1.], [-3.]]))
        self.assertAlmostEqual(shifted['per_event_demeaned_rms_ms'], 1)
        with self.assertRaises(ValueError):
            check.metrics([[np.nan]])

    def test_three_time_decomposition_exact(self):
        cart = np.array([[1., 2.], [3., 4.]])
        chord = cart + np.array([[.002, -.003], [.004, -.005]])
        probe = chord + .0001
        ray, embedding, total, residual = check.decomposition(probe, chord, cart)
        np.testing.assert_array_equal(ray + embedding, total)
        np.testing.assert_array_equal(total, probe - cart)
        self.assertEqual(residual, 0)

    def test_fixed_criteria_are_inclusive_and_independent(self):
        self.assertEqual(check.CRITERIA, {'ray_only': {'rms_ms': 1., 'max_abs_ms': 5.},
                                         'total': {'rms_ms': 5., 'max_abs_ms': 10.}})
        self.assertTrue(check.passes({'rms_ms': 1., 'max_abs_ms': 5.}, 'ray_only'))
        self.assertFalse(check.passes({'rms_ms': 1.000001, 'max_abs_ms': 4.}, 'ray_only'))
        self.assertFalse(check.passes({'rms_ms': 4., 'max_abs_ms': 10.000001}, 'total'))

    def test_wrong_height_and_association_are_analytically_detectable(self):
        sources = np.array([[30.3, 103.3, 4.]])
        receivers = np.array([[30.3, 103.3, 1.166], [30.3, 103.32, .2]])
        correct = check.chord_times(sources, receivers)
        wrong = receivers.copy()
        wrong[:, 2] *= -1
        self.assertGreater(check.metrics(check.chord_times(sources, wrong) - correct)['rms_ms'], 100)
        self.assertGreater(check.metrics(check.chord_times(sources, receivers[::-1]) - correct)['rms_ms'], 100)
        receivers[:, 2] = 0
        wrong = receivers.copy()
        wrong[:, 2] *= -1
        np.testing.assert_array_equal(check.chord_times(sources, wrong), check.chord_times(sources, receivers))

    def test_exclusive_claim_and_output_root_guard(self):
        check.check_root(self.root, expected=self.root)
        claim = self.root / 'manifest.json'
        check.save(claim, {'status': 'claimed'}, exclusive=True)
        with self.assertRaises(FileExistsError):
            check.save(claim, {'status': 'overwritten'}, exclusive=True)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            check.check_root(self.root, expected=self.root)
        with self.assertRaisesRegex(ValueError, 'Unapproved'):
            check.check_root(self.root)
        link = self.root / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            check.check_root(link, expected=self.root)

    def test_write_scope_resolves_symlinks_and_parent_traversal(self):
        check.check_write(self.root / 'new', root=self.root)
        with self.assertRaisesRegex(ValueError, 'outside approved'):
            check.check_write(self.root / '../escape', root=self.root)
        link = self.root / 'escape'
        link.symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            check.check_write(link / 'file', root=self.root)

    def test_hash_guard(self):
        path = self.root / 'input'
        path.write_bytes(b'original')
        hashes = {str(path): check.digest(path)}
        check.verify_hashes(hashes)
        path.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            check.verify_hashes(hashes)

    def test_frozen_reader_source_contract_without_solver(self):
        check.source_contract()

    def test_process_guard_blocks_font_discovery_and_inversion_with_oserror(self):
        with self.assertRaises(PermissionError):
            check.check_process('fc-list', ['fc-list', '--help'], None)
        with self.assertRaises(PermissionError):
            check.check_process(check.SNAPSHOT / 'solver/tomoDD-SE', ['tomoDD-SE'], self.root)
        binary = check.SNAPSHOT / 'solver/forward_probe'
        check.check_process(binary, [str(binary)], self.root)
        with self.assertRaises(PermissionError):
            check.check_process(binary, [str(binary), 'extra'], self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
