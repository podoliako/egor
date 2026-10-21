import unittest

import numpy as np

from compare_slices import choose_planes, plane_grid, sample_blocks, sample_nodes


class SliceTests(unittest.TestCase):
    def test_horizontal_matches_exact_fine_layer(self):
        xyz, extent, valid = plane_grid({'name': 'horizontal'})
        values = np.arange(160 * 56 * 56).reshape(160, 56, 56)
        np.testing.assert_array_equal(sample_blocks(values, .125, xyz, valid), values[:, :, 11].T)
        self.assertEqual(extent, (0., 20., 0., 7.))

    def test_blocks_are_not_smoothed_and_mask_is_preserved(self):
        values = np.array([[[3.]], [[7.]]])
        xyz = np.array([[[.99, .5, .5], [1.01, .5, .5], [2.1, .5, .5]]])
        result = sample_blocks(values, 1., xyz, np.array([[True, True, False]]))
        np.testing.assert_array_equal(result[0, :2], [3., 7.])
        self.assertTrue(np.isnan(result[0, 2]))

    def test_native_nodes_reproduce_linear_field(self):
        axes = (np.array([0., 7.]), np.array([-3.5, 3.5]), np.array([-10., 10.]))
        z, y, x = np.meshgrid(*axes, indexing='ij')
        nodes = 4000 + 2 * (x + 10) + 3 * (y + 3.5) + 4 * z
        xyz = np.array([[[1., 2., 3.], [15., 4., 2.]]])
        expected = 4000 + 2 * xyz[..., 0] + 3 * xyz[..., 1] + 4 * xyz[..., 2]
        np.testing.assert_allclose(sample_nodes(nodes, axes, xyz, np.ones((1, 2), bool)), expected)

    def test_truth_only_planes_cover_all_centers_with_small_offsets(self):
        centers = np.array([[3.5, 2.5, 1.2], [8, 4.8, 2], [12.5, 2.4, 2.6],
                            [16.5, 4.5, 1.6], [6, 2.2, 3.6], [11, 5, 4]])
        radii = np.array([.9, 1.1, 1.2, 1., 1.1, 1.])
        planes = choose_planes(centers, radii)
        self.assertEqual(planes[0]['sphere_ids'], [1, 2, 5, 6])
        self.assertEqual(planes[1]['sphere_ids'], [2, 3, 4, 5])
        for plane in planes:
            basis = np.array([plane['u'], plane['v'], plane['normal']])
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            self.assertLess(plane['max_selected_distance_over_radius'], .15)
            xyz, _, valid = plane_grid(plane)
            np.testing.assert_allclose((xyz - plane['origin']) @ plane['normal'], 0, atol=1e-12)
            self.assertTrue(valid.any())


if __name__ == '__main__':
    unittest.main()
