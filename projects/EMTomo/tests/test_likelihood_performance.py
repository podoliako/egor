"""Reference equivalence for the tiled and fused likelihood paths (no large runs)."""
import numpy as np
import pytest
from scipy.optimize import minimize

from instruments.hypocentre_objective import (
    _refinement_objective_kernel,
    make_refinement_objective,
)
from instruments.instruments_coords import cell_coord_bounds, sample_cell_centered_trilinear_batch
from instruments.likelihood import (
    PickNoise,
    likelihood_normalization,
    negative_log_likelihood,
    negative_log_likelihood_batch,
)


def _reference_objective(fields, observed, noise):
    def objective(coord):
        return float(negative_log_likelihood(
            sample_cell_centered_trilinear_batch(fields, coord), observed, noise,
        ))
    return objective


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("relative", [0.0, 0.03])
@pytest.mark.parametrize("tile_size", [1, 7, 1024, 2048])
@pytest.mark.parametrize("normalization_kind", ["auto", "precomputed", "scalar", "broadcast"])
def test_batch_matches_single_events(dtype, relative, tile_size, normalization_kind):
    rng = np.random.default_rng(401)
    fields = rng.uniform(0.0, 30.0, (5, 3, 347)).astype(dtype)
    observations = rng.uniform(-5.0, 20.0, (9, 5))
    noise = PickNoise(relative, 0.05, 0.2)
    normalization = {
        "auto": None,
        "precomputed": likelihood_normalization(fields, noise),
        "scalar": -2.5,
        "broadcast": rng.normal(size=(1, 347)),
    }[normalization_kind]
    expected = np.stack([
        negative_log_likelihood(fields, observed, noise, normalization)
        for observed in observations
    ])
    actual = negative_log_likelihood_batch(fields, observations, noise, normalization, tile_size)
    assert actual.shape == (9, 3, 347)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("relative", [0.0, 0.02])
@pytest.mark.parametrize("batch_size", [0, 1, 3, 8, 9])
def test_batch_noncontiguous_and_caller_sized_output(dtype, relative, batch_size):
    rng = np.random.default_rng(11)
    fields = rng.uniform(1.0, 15.0, (7, 3, 4, 10)).astype(dtype)[:, ::-1, :, ::2]
    observations = rng.normal(size=(batch_size * 2, 14))[::2, ::2]
    noise = PickNoise(relative, 0.1, 0.2)
    normalization = likelihood_normalization(fields, noise)
    if relative:
        normalization = np.asfortranarray(normalization)
    actual = negative_log_likelihood_batch(fields, observations, noise, normalization, tile_size=17)
    assert actual.shape == (batch_size, 3, 4, 5)
    for index, observed in enumerate(observations):
        np.testing.assert_array_equal(actual[index], negative_log_likelihood(fields, observed, noise))


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (2, 3, 4)])
def test_batch_grid_shapes(shape):
    fields = np.full((2, *shape), 3.0)
    observations = np.array([[1.0, 2.0], [3.0, 4.0]])
    noise = PickNoise(0.02, 0.1, 0.2)
    actual = negative_log_likelihood_batch(fields, observations, noise)
    expected = np.stack([negative_log_likelihood(fields, obs, noise) for obs in observations])
    assert actual.shape == (2, *shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("relative", [0.0, 0.07])
def test_batch_matches_centered_residual_formula(relative):
    rng = np.random.default_rng(17)
    fields = rng.uniform(0.0, 40.0, (11, 19))
    observed = rng.uniform(-4.0, 3.0, (3, 11))
    noise = PickNoise(relative, 0.1, 0.2)
    variance = (relative * fields) ** 2 + noise.constant_variance
    precision = 1.0 / variance
    residual = observed[:, :, None] - fields
    mean = (precision * residual).sum(axis=1) / precision.sum(axis=0)
    expected = 0.5 * (precision * (residual - mean[:, None]) ** 2).sum(axis=1)
    if relative:
        expected += 0.5 * np.log(variance).sum(axis=0) + 0.5 * np.log(precision.sum(axis=0))
    np.testing.assert_allclose(negative_log_likelihood_batch(fields, observed, noise), expected,
                               rtol=2e-14, atol=2e-10)


@pytest.mark.parametrize("observed", [np.zeros(3), np.zeros((2, 2)), np.zeros((1, 3, 1)),
                                     [[0.0, np.nan, 1.0]], [[0.0, 1.0, np.inf]]])
def test_batch_invalid_observations(observed):
    with pytest.raises(ValueError):
        negative_log_likelihood_batch(np.ones((3, 4)), observed, PickNoise())


@pytest.mark.parametrize("fields", [np.ones(()), np.ones((1, 4)), np.ones((0, 4))])
def test_batch_invalid_fields(fields):
    with pytest.raises(ValueError):
        negative_log_likelihood_batch(fields, np.ones((2, 3)), PickNoise())


@pytest.mark.parametrize("tile_size", [0, -1, 1.5, True, np.bool_(False), "8", None])
def test_batch_invalid_tile_size(tile_size):
    with pytest.raises(ValueError, match="tile_size"):
        negative_log_likelihood_batch(np.ones((3, 4)), np.ones((2, 3)), PickNoise(), tile_size=tile_size)


def test_batch_numpy_integer_tile_size():
    actual = negative_log_likelihood_batch(np.ones((3, 4)), np.ones((2, 3)), PickNoise(),
                                           tile_size=np.int64(2))
    np.testing.assert_array_equal(actual, np.zeros((2, 4)))


@pytest.mark.parametrize("normalization", [np.ones((2, 4)), np.ones(3)])
def test_batch_rejects_normalization_that_adds_event_or_grid_axes(normalization):
    with pytest.raises(ValueError, match="normalization"):
        negative_log_likelihood_batch(np.ones((3, 4)), np.ones((2, 3)), PickNoise(), normalization)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("relative", [0.0, 0.025])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("shape", [(4, 5, 6), (1, 4, 1)])
def test_objective_matches_staged_sampler_at_edges_and_interiors(dtype, relative, strided, shape):
    rng = np.random.default_rng(912)
    fields = rng.uniform(0.0, 30.0, (9, *shape)).astype(dtype)
    observed = rng.uniform(-5.0, 5.0, 18)[::2]
    if strided:
        fields = fields[:, ::-1, ::-1, ::-1]
    noise = PickNoise(relative, 0.08, 0.1)
    objective = make_refinement_objective(fields, observed, noise)
    reference = _reference_objective(fields, observed, noise)
    coordinates = [np.zeros(3), np.array(shape) - 1.0, [-10.0, 2.3, 80.0], [1.0, 2.0, 3.0],
                   [1e-12, 1.0 - 1e-12, 1.0 + 1e-12]]
    coordinates.extend(rng.uniform(-1.0, np.array(shape), size=(40, 3)))
    for coord in coordinates:
        assert objective(coord) == reference(coord)


def test_legacy_float32_intermediates_round_each_numpy_operation():
    """Exercise NumPy 1.x semantics even when the host uses NumPy 2.x."""
    rng = np.random.default_rng(99)
    fields = rng.uniform(0.0, 40.0, (8, 2, 2, 2)).astype(np.float32)
    observed = rng.normal(size=8)
    noise = PickNoise(0.03, 0.1, 0.2)

    def lerp(a, b, weight):
        return a * np.float32(1.0 - weight) + b * np.float32(weight)

    for coord in rng.uniform(size=(40, 3)):
        di, dj, dk = coord
        c00 = lerp(fields[:, 0, 0, 0], fields[:, 1, 0, 0], di)
        c10 = lerp(fields[:, 0, 1, 0], fields[:, 1, 1, 0], di)
        c01 = lerp(fields[:, 0, 0, 1], fields[:, 1, 0, 1], di)
        c11 = lerp(fields[:, 0, 1, 1], fields[:, 1, 1, 1], di)
        predicted = lerp(lerp(c00, c10, dj), lerp(c01, c11, dj), dk).astype(np.float64)
        expected = float(negative_log_likelihood(predicted, observed, noise))
        actual = _refinement_objective_kernel(fields, observed, coord, noise.relative_sigma,
                                              noise.constant_variance, True)
        assert actual == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("relative", [0.0, 0.03])
@pytest.mark.parametrize("cell", [(1, 2, 1), (0, 0, 0), (3, 3, 3)])
def test_full_powell_evaluation_and_iteration_trajectories(dtype, relative, cell):
    rng = np.random.default_rng(29)
    stations = rng.uniform(-2.0, 6.0, (8, 3))
    grid = np.moveaxis(np.indices((4, 4, 4), dtype=np.float64), 0, -1)
    fields = np.linalg.norm(grid[None] - stations[:, None, None, None], axis=-1).astype(dtype)
    observed = sample_cell_centered_trilinear_batch(fields, [1.23, 1.78, 1.11]) + 0.7
    noise = PickNoise(relative, 0.05, 0.2)
    reference = _reference_objective(fields, observed, noise)
    compiled = make_refinement_objective(fields, observed, noise)

    def run(objective):
        evaluations = []
        iterations = []

        def traced(coord):
            value = objective(coord)
            evaluations.append(np.r_[coord.copy(), value])
            return value

        result = minimize(traced, np.asarray(cell, dtype=np.float64), method="Powell",
                          bounds=cell_coord_bounds(cell, fields.shape[1:]),
                          options={"xtol": 1e-3, "ftol": 1e-8, "maxiter": 50},
                          callback=lambda coord: iterations.append(coord.copy()))
        return result, np.array(evaluations), np.array(iterations)

    expected, expected_evaluations, expected_iterations = run(reference)
    actual, evaluations, iterations = run(compiled)
    assert actual.success == expected.success
    assert actual.nit == expected.nit
    assert actual.nfev == expected.nfev
    np.testing.assert_array_equal(evaluations, expected_evaluations)
    np.testing.assert_array_equal(iterations, expected_iterations)
    np.testing.assert_array_equal(actual.x, expected.x)
    assert actual.fun == expected.fun


@pytest.mark.parametrize("fields", [np.ones((1, 2, 2, 2)), np.ones((3, 2, 2)),
                                   np.ones((3, 2, 0, 2)), np.ones((3, 2, 2, 2), dtype=np.int32)])
def test_objective_invalid_fields(fields):
    with pytest.raises(ValueError, match="station_fields"):
        make_refinement_objective(fields, np.ones(3), PickNoise())


@pytest.mark.parametrize("observed", [np.ones(2), np.ones((3, 1)), [0.0, np.nan, 1.0],
                                     [0.0, 1.0, np.inf]])
def test_objective_invalid_observations(observed):
    with pytest.raises(ValueError, match="observation"):
        make_refinement_objective(np.ones((3, 2, 2, 2)), observed, PickNoise())


@pytest.mark.parametrize("coord", [0.0, [0.0, 1.0], np.zeros((3, 1)),
                                  [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
def test_objective_invalid_coordinates(coord):
    objective = make_refinement_objective(np.ones((3, 2, 2, 2)), np.ones(3), PickNoise())
    with pytest.raises(ValueError, match="cell_coord"):
        objective(coord)
