import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes, label

from sensitivity_envelope import expanded_envelope, cell_boundary_segments


def test_convex_envelope_fills_holes_and_includes_raw():
    raw = np.zeros((40, 50), dtype=bool)
    raw[10:30, 10:40] = True
    raw[15:25, 20:30] = False
    display = expanded_envelope(raw, 100, padding_m=0)
    assert display[raw].all()
    assert display[15:25, 20:30].all()
    np.testing.assert_array_equal(display, binary_fill_holes(display))


def test_metric_expansion_and_single_point():
    raw = np.zeros((30, 30), dtype=bool)
    raw[15, 15] = True
    display = expanded_envelope(raw, 100, padding_m=500)
    assert display[15, 20]
    assert not display[15, 21]
    assert not display[20, 20]  # Euclidean, not square dilation.
    assert label(display)[1] == 1


def test_hull_connects_disconnected_components():
    raw = np.zeros((30, 30), dtype=bool)
    raw[5:8, 5:8] = True
    raw[20:23, 20:23] = True
    display = expanded_envelope(raw, 100, padding_m=0)
    assert display[raw].all()
    assert display[15, 15]
    assert label(display, structure=np.ones((3, 3)))[1] == 1


def test_full_domain_remains_full():
    raw = np.ones((10, 12), dtype=bool)
    np.testing.assert_array_equal(expanded_envelope(raw, 100), raw)


def test_cell_edges_are_axis_aligned_and_on_native_grid():
    mask = np.array([[True, True], [True, False]])
    segments = cell_boundary_segments(mask, .8, origin=(1., 0.))
    assert len(segments) == 8
    delta = segments[:, 1] - segments[:, 0]
    assert np.all(np.isclose(delta[:, 0], 0) | np.isclose(delta[:, 1], 0))
    np.testing.assert_allclose(np.abs(delta).sum(axis=1), .8)
    grid = (segments - [1., 0.]) / .8
    np.testing.assert_allclose(grid, np.rint(grid), atol=1e-12)


def test_full_mask_has_only_outer_edges():
    segments = cell_boundary_segments(np.ones((2, 3), dtype=bool), 1.)
    assert len(segments) == 10


@pytest.mark.parametrize("spacing,padding", [(0, 500), (np.nan, 500), (100, -1)])
def test_invalid_parameters(spacing, padding):
    with pytest.raises(ValueError):
        expanded_envelope(np.ones((3, 3), dtype=bool), spacing, padding)
