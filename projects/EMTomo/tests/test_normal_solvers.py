"""Dense, sparse-input and matrix-free normal-equation regression tests."""

import pickle

import numpy as np
import pytest
from scipy import sparse

import tomography.tomography_math as math
from tomography.tomography_math import (
    MatrixFreeNormalEquations,
    accumulate_normal_equations,
    solve_slowness_update,
)


def _add(hessian, rhs, rows, residual, valid=None, sigmas=None, weight=1.0):
    n_stations = rows.shape[0]
    accumulate_normal_equations(
        hessian, rhs, rows, residual,
        np.ones(n_stations, dtype=bool) if valid is None else valid,
        np.ones(n_stations) if sigmas is None else sigmas, weight,
    )


def _fixture(seed=13, n_vox=24):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(9, n_vox))
    rows[rng.random(rows.shape) < 0.75] = 0.0
    rows[:, -2:] = 0.0
    valid = np.array([True, True, False, True, True, True, False, True, True])
    residual = rng.normal(size=9)
    sigmas = rng.uniform(0.2, 2.0, size=9)
    # Invalid/unreached stations are ignored even if their data are unusable.
    rows[~valid] = np.nan
    residual[~valid] = np.nan
    sigmas[~valid] = np.nan
    return rows, residual, valid, sigmas


def _pairs(rows, residual, valid, sigmas, weight):
    rows, residual, precision = rows[valid], residual[valid], 1 / sigmas[valid] ** 2
    hessian = np.zeros((rows.shape[1], rows.shape[1]))
    rhs = np.zeros(rows.shape[1])
    for i in range(len(rows)):
        for j in range(i):
            g = rows[i] - rows[j]
            w = weight * precision[i] * precision[j] / precision.sum()
            hessian += w * np.outer(g, g)
            rhs += w * g * (residual[i] - residual[j])
    return hessian, rhs


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("matrix_free", [False, True])
def test_assembly_matches_explicit_precision_weighted_pairs(sparse_input, matrix_free):
    rows, residual, valid, sigmas = _fixture()
    hessian = MatrixFreeNormalEquations(24) if matrix_free else np.zeros((24, 24))
    rhs = np.zeros(24)
    data = sparse.csr_matrix(rows) if sparse_input else rows.reshape(9, 2, 3, 4)
    _add(hessian, rhs, data, residual, valid, sigmas, 0.37)
    expected_h, expected_b = _pairs(rows, residual, valid, sigmas, 0.37)
    np.testing.assert_allclose(rhs, expected_b, atol=2e-15)
    if matrix_free:
        np.testing.assert_allclose(hessian.diagonal(), expected_h.diagonal(), atol=2e-15)
        vector = np.random.default_rng(4).normal(size=24)
        np.testing.assert_allclose(hessian.matvec(vector), expected_h @ vector, atol=2e-14)
        np.testing.assert_allclose(hessian.aslinearoperator() @ vector, expected_h @ vector, atol=2e-14)
        assert hessian._blocks[0].rows.shape[1] <= 22
        assert sparse.issparse(hessian._blocks[0].rows) == sparse_input
    else:
        np.testing.assert_allclose(hessian, expected_h, atol=2e-15)


@pytest.mark.parametrize("coverage_power", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_multiple_hypotheses_solvers_and_coverage_are_equivalent(coverage_power, sparse_input):
    n_vox = 24
    hessian, rhs = np.zeros((n_vox, n_vox)), np.zeros(n_vox)
    operator, operator_rhs = MatrixFreeNormalEquations(n_vox), np.zeros(n_vox)
    for seed, weight in [(7, 0.2), (9, 0.7), (42, 0.1)]:
        rows, residual, valid, sigmas = _fixture(seed)
        data = sparse.csr_matrix(rows) if sparse_input else rows
        _add(hessian, rhs, data, residual, valid, sigmas, weight)
        _add(operator, operator_rhs, data, residual, valid, sigmas, weight)
    np.testing.assert_allclose(operator_rhs, rhs, atol=1e-14)
    options = dict(coverage_damping_power=coverage_power, coverage_floor=0.1,
                   coverage_reference_percentile=65.0)
    diagnostics = {}
    direct = solve_slowness_update(hessian, rhs, (2, 3, 4), 0.05, diagnostics=diagnostics, **options)
    assert diagnostics["solver"] == "cholesky"
    assert diagnostics["iterations"] == 0
    legacy_diagnostics = {}
    legacy = solve_slowness_update(hessian, rhs, (2, 3, 4), 0.05, solver="legacy",
                                   diagnostics=legacy_diagnostics, **options)
    assert legacy_diagnostics["solver"] == "solve"
    cg_diagnostics = {}
    iterative = solve_slowness_update(operator, operator_rhs, (2, 3, 4), 0.05,
                                      diagnostics=cg_diagnostics, **options)
    assert cg_diagnostics["solver"] == "cg"
    assert cg_diagnostics["iterations"] > 0
    assert cg_diagnostics["relative_residual"] <= 1e-10
    for expected, reference, actual in zip(direct, legacy, iterative):
        np.testing.assert_allclose(reference, expected, atol=1e-12)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        assert np.all(np.isfinite(actual))


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("valid,weight", [([True, False], 1), ([False, False], 1),
                                         ([True, True], 0), ([True, True], -1)])
def test_skipped_hypotheses_do_not_inspect_invalid_inputs(matrix_free, sparse_input, valid, weight):
    hessian = MatrixFreeNormalEquations(3) if matrix_free else np.zeros((3, 3))
    rhs = np.zeros(3)
    rows = np.full((2, 3), np.nan)
    if sparse_input:
        rows = sparse.csr_matrix(rows)
    _add(hessian, rhs, rows, [np.nan, np.nan], valid, [np.nan, np.nan], weight)
    np.testing.assert_array_equal(rhs, 0)
    np.testing.assert_array_equal(hessian.diagonal(), 0)
    if matrix_free:
        assert not hessian._blocks


@pytest.mark.parametrize("sparse_input", [False, True])
def test_merge_is_in_place_rhs_external_and_blocks_own_their_data(sparse_input):
    chunks, rhs_chunks = [], []
    reference, reference_rhs = MatrixFreeNormalEquations(24), np.zeros(24)
    for seed in (1, 2):
        rows, residual, valid, sigmas = _fixture(seed)
        if sparse_input:
            rows = sparse.csr_matrix(rows)
        chunk, rhs = MatrixFreeNormalEquations(24), np.zeros(24)
        _add(chunk, rhs, rows, residual, valid, sigmas)
        _add(reference, reference_rhs, rows, residual, valid, sigmas)
        if sparse_input:
            rows.data[:] = 100
        else:
            rows[:] = 100
        sigmas[:] = 100
        chunks.append(chunk)
        rhs_chunks.append(rhs)
    second_diag = chunks[1].diagonal()
    # Event workers can serialize compact chunks without materializing H.
    second = pickle.loads(pickle.dumps(chunks[1]))
    assert chunks[0].merge(second) is chunks[0]
    chunks[0].merge(MatrixFreeNormalEquations(24))
    np.testing.assert_allclose(sum(rhs_chunks), reference_rhs)
    np.testing.assert_allclose(chunks[0].diagonal(), reference.diagonal())
    np.testing.assert_array_equal(chunks[1].diagonal(), second_diag)
    vector = np.arange(24, dtype=float)
    np.testing.assert_allclose(chunks[0].matvec(vector), reference.matvec(vector))
    before = chunks[0].diagonal()
    with pytest.raises(ValueError, match="incompatible"):
        chunks[0].merge(MatrixFreeNormalEquations(23))
    np.testing.assert_array_equal(chunks[0].diagonal(), before)


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("n_vox", [0, 3])
def test_empty_and_zero_systems(matrix_free, n_vox):
    hessian = MatrixFreeNormalEquations(n_vox) if matrix_free else np.zeros((n_vox, n_vox))
    diagnostics = {}
    result = solve_slowness_update(hessian, np.zeros(n_vox), (n_vox, 1, 1), 0.1, diagnostics=diagnostics)
    for array in result:
        assert array.shape == (n_vox, 1, 1)
        np.testing.assert_array_equal(array, 0)
    assert diagnostics == dict(solver="zero", iterations=0, relative_residual=0.0)
    if n_vox:
        if matrix_free:
            with pytest.raises(RuntimeError, match="zero system|residual"):
                solve_slowness_update(hessian, np.ones(n_vox), (n_vox,), 0.1)
        else:
            result = solve_slowness_update(hessian, np.ones(n_vox), (n_vox,), 0.1,
                                            diagnostics=diagnostics)
            np.testing.assert_array_equal(result[0], 0)
            assert diagnostics["solver"] == "lstsq"
            assert diagnostics["relative_residual"] == pytest.approx(1.0)


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("identical", [False, True])
def test_zero_or_identical_station_rows(sparse_input, identical):
    rows = np.tile([0.0, 2.0, 3.0] if identical else [0.0, 0.0, 0.0], (4, 1))
    operator, rhs = MatrixFreeNormalEquations(3), np.zeros(3)
    _add(operator, rhs, sparse.csr_matrix(rows) if sparse_input else rows,
         [1, 2, 3, 4], sigmas=[1, 1, 1, 1])
    np.testing.assert_array_equal(operator.diagonal(), 0)
    np.testing.assert_array_equal(operator.matvec(np.array([1.0, 2.0, 3.0])), 0)
    np.testing.assert_array_equal(rhs, 0)
    delta, _, _ = solve_slowness_update(operator, rhs, (3,), 0.1)
    np.testing.assert_array_equal(delta, 0)


@pytest.mark.parametrize("seed", [0, 1, 6, 13])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_identical_rows_with_unequal_precisions_have_no_spurious_curvature(seed, sparse_input):
    rows = np.tile([2., 3.], (17, 1))
    sigmas = np.random.default_rng(seed).uniform(0.2, 2.0, 17)
    operator, rhs = MatrixFreeNormalEquations(2), np.zeros(2)
    _add(operator, rhs, sparse.csr_matrix(rows) if sparse_input else rows,
         np.arange(17.), sigmas=sigmas)
    # Squared rounding error in the mean is acceptable; a first-order error
    # from subtracting nearly equal precision sums is not.
    np.testing.assert_allclose(operator.diagonal(), 0, atol=1e-26)
    np.testing.assert_array_equal(operator.matvec(np.ones(2)), 0)


@pytest.mark.parametrize("matrix_free", [False, True])
def test_nonzero_system_zero_rhs(matrix_free):
    hessian = MatrixFreeNormalEquations(3) if matrix_free else np.zeros((3, 3))
    rhs = np.zeros(3)
    _add(hessian, rhs, np.eye(3), [1, 1, 1])
    result = solve_slowness_update(hessian, rhs, (3,), 0.1)
    np.testing.assert_array_equal(result[0], 0)
    assert np.all(result[1] > 0)
    assert np.all(np.isfinite(result[2]))


@pytest.mark.parametrize("matrix_free", [False, True])
def test_consistent_singular_system_and_uncovered_voxels(matrix_free):
    hessian = MatrixFreeNormalEquations(3) if matrix_free else np.zeros((3, 3))
    rhs = np.zeros(3)
    _add(hessian, rhs, np.array([[1., 1., 0.], [0., 0., 0.]]), [2., 0.])
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (3,), 0.0, diagnostics=diagnostics)
    np.testing.assert_allclose(result[0], [1, 1, 0], atol=1e-14)
    assert diagnostics["solver"] == ("cg" if matrix_free else "lstsq")
    assert diagnostics["relative_residual"] <= 1e-10


@pytest.mark.parametrize("hessian", [np.array([[2., 1.], [0., 2.]]),
                                     np.array([[1., 2.], [2., 1.]])])
def test_dense_asymmetric_and_indefinite_use_general_solve(hessian):
    rhs = np.array([1., 3.])
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (2,), 0.0, diagnostics=diagnostics)
    np.testing.assert_allclose(result[0], np.linalg.solve(hessian, rhs))
    assert diagnostics["solver"] == "solve"


@pytest.mark.parametrize("solver", ["auto", "legacy"])
def test_iterative_tolerance_does_not_reject_a_finite_direct_solution(solver):
    rng = np.random.default_rng(11)
    rows = rng.normal(size=(12, 12))
    hessian = rows.T @ rows
    rhs = rng.normal(size=12)
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (12,), .1, solver=solver,
                                   rtol=1e-30, diagnostics=diagnostics)
    assert np.all(np.isfinite(result[0]))
    assert diagnostics["relative_residual"] < 1e-12


@pytest.mark.parametrize("size", [1, 7, 25])
def test_symmetry_check_matches_roundoff_policy(size):
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(size, size))
    matrix += matrix.T
    for perturbation in (0.0, 1e-15, 1e-8):
        changed = matrix.copy()
        changed[0, -1] += perturbation
        tolerance = 10 * np.finfo(float).eps * np.max(np.abs(changed))
        assert math._symmetric_to_roundoff(changed) == np.allclose(changed, changed.T, rtol=0., atol=tolerance)


def test_cholesky_failure_falls_back_without_mutating_inputs(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("forced Cholesky failure")

    monkeypatch.setattr(math.linalg, "cho_factor", fail)
    hessian, rhs = np.diag([1., 2.]), np.array([2., 3.])
    original_h, original_rhs = hessian.copy(), rhs.copy()
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (2,), 0.2, diagnostics=diagnostics)
    np.testing.assert_allclose(result[0], rhs / [1.3, 2.3])
    assert diagnostics["solver"] == "solve"
    np.testing.assert_array_equal(hessian, original_h)
    np.testing.assert_array_equal(rhs, original_rhs)


@pytest.mark.parametrize("solver", ["auto", "legacy"])
def test_inconsistent_singular_dense_system_preserves_lstsq_semantics(solver):
    hessian, rhs = np.diag([1., 0.]), np.array([1., 1.])
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (2,), 0.0, solver=solver, diagnostics=diagnostics)
    expected = np.linalg.lstsq(hessian, rhs, rcond=None)[0]
    np.testing.assert_array_equal(result[0], expected)
    assert diagnostics["solver"] == "lstsq"
    assert diagnostics["relative_residual"] == pytest.approx(1 / np.sqrt(2))


@pytest.mark.parametrize("bad_result", [np.full(2, np.nan), np.full(2, np.inf)])
def test_dense_lstsq_still_rejects_nonfinite_solutions(monkeypatch, bad_result):
    monkeypatch.setattr(np.linalg, "lstsq", lambda *args, **kwargs: (bad_result,))
    with np.errstate(invalid="ignore"):
        with pytest.raises(RuntimeError, match="lstsq.*actual relative residual"):
            solve_slowness_update(np.zeros((2, 2)), np.ones(2), (2,), 0.0)


def test_cg_iteration_limit_reports_failure_and_diagnostics():
    rows, residual, valid, sigmas = _fixture()
    hessian, rhs = MatrixFreeNormalEquations(24), np.zeros(24)
    _add(hessian, rhs, rows, residual, valid, sigmas)
    diagnostics = {}
    with pytest.raises(RuntimeError, match="cg.*iterations=1.*actual relative residual"):
        solve_slowness_update(hessian, rhs, (24,), 0.1, maxiter=1, diagnostics=diagnostics)
    assert diagnostics["solver"] == "cg"
    assert diagnostics["iterations"] == 1
    assert diagnostics["relative_residual"] > 1e-10


@pytest.mark.parametrize("bad_result,info", [(np.zeros(3), 0), (np.full(3, np.nan), 0),
                                           (np.zeros(3), -1)])
def test_cg_status_is_not_a_substitute_for_actual_residual(monkeypatch, bad_result, info):
    hessian, rhs = MatrixFreeNormalEquations(3), np.zeros(3)
    _add(hessian, rhs, np.eye(3), [1., 2., 4.])
    monkeypatch.setattr(math, "cg", lambda *args, **kwargs: (bad_result, info))
    with pytest.raises(RuntimeError, match="cg.*actual relative residual"):
        solve_slowness_update(hessian, rhs, (3,), 0.1)


def test_matrix_free_rejects_negative_damping_and_inconsistent_zero_row():
    hessian, rhs = MatrixFreeNormalEquations(2), np.zeros(2)
    _add(hessian, rhs, np.array([[1., 0.], [0., 0.]]), [1., 0.])
    with pytest.raises(RuntimeError, match="positive semidefinite"):
        solve_slowness_update(hessian, rhs, (2,), -0.1)
    with pytest.raises(RuntimeError, match="zero row"):
        solve_slowness_update(hessian, [1., 1.], (2,), 0.0)


def test_matrix_free_large_sparse_assembly_and_solve_never_densify(monkeypatch):
    n_vox = 100_000
    rows = sparse.csr_matrix(([1., 2., 3., 4.], ([0, 1, 2, 3], [0, 2, 4, n_vox - 1])),
                             shape=(4, n_vox))

    def forbidden(*args, **kwargs):
        raise AssertionError("Matrix-free assembly/solve attempted dense conversion or direct solve")

    for name in ("toarray", "todense"):
        monkeypatch.setattr(sparse.csr_matrix, name, forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(math.linalg, "cho_factor", forbidden)
    monkeypatch.setattr(np, "diag", forbidden)
    for name in ("zeros", "empty", "ones", "full"):
        original = getattr(np, name)

        def guard(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and min(shape) > 4:
                raise AssertionError(f"Unexpected dense allocation {shape}")
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, name, guard)
    hessian, rhs = MatrixFreeNormalEquations(n_vox), np.zeros(n_vox)
    _add(hessian, rhs, rows, [1., -1., 2., 3.])
    assert hessian._blocks[0].rows.shape == (4, 4)
    assert hessian._blocks[0].rows.nnz == 4
    diagnostics = {}
    result = solve_slowness_update(hessian, rhs, (n_vox,), 0.1, diagnostics=diagnostics, maxiter=30)
    assert all(np.all(np.isfinite(array)) for array in result)
    assert diagnostics["relative_residual"] <= 1e-10


def test_csr_duplicate_entries_and_explicit_zeros():
    # CSR can contain unsummed duplicates and explicitly stored zeros.
    rows = sparse.csr_matrix((np.array([1., 2., 0., 4.]), np.array([0, 0, 1, 2]),
                              np.array([0, 3, 4])), shape=(2, 4))
    expected, expected_rhs = np.zeros((4, 4)), np.zeros(4)
    _add(expected, expected_rhs, rows.toarray(), [1., 3.], sigmas=[1., 2.])
    hessian, rhs = MatrixFreeNormalEquations(4), np.zeros(4)
    _add(hessian, rhs, rows, [1., 3.], sigmas=[1., 2.])
    np.testing.assert_allclose(rhs, expected_rhs)
    np.testing.assert_allclose(hessian.diagonal(), expected.diagonal())
    np.testing.assert_allclose(hessian.matvec(np.arange(4.)), expected @ np.arange(4.))
    np.testing.assert_array_equal(hessian._blocks[0].active, [0, 2])
    assert rows.nnz == 4  # Caller storage is unchanged.


@pytest.mark.parametrize("sigmas", [[0., 1.], [-1., 1.], [np.inf, 1.], [np.nan, 1.], [1.]])
def test_bad_station_sigmas_rejected(sigmas):
    with pytest.raises(ValueError, match="sigma"):
        _add(MatrixFreeNormalEquations(2), np.zeros(2), np.eye(2), [1., 2.], sigmas=sigmas)


@pytest.mark.parametrize("options", [dict(solver="unknown"), dict(rtol=0), dict(rtol=np.nan),
                                     dict(maxiter=0), dict(maxiter=1.5), dict(coverage_floor=0),
                                     dict(coverage_damping_power=-1),
                                     dict(coverage_reference_percentile=101)])
def test_invalid_solver_options(options):
    with pytest.raises(ValueError):
        solve_slowness_update(np.eye(2), [1., 2.], (2,), 0.1, **options)


def test_solver_storage_mismatch_and_voxel_mismatch():
    with pytest.raises(ValueError, match="requires"):
        solve_slowness_update(np.eye(2), [1., 2.], (2,), 0.1, solver="cg")
    with pytest.raises(ValueError, match="requires"):
        solve_slowness_update(MatrixFreeNormalEquations(2), [1., 2.], (2,), 0.1, solver="legacy")
    with pytest.raises(ValueError, match="voxel"):
        solve_slowness_update(MatrixFreeNormalEquations(3), [1., 2.], (2,), 0.1)
    with pytest.raises(ValueError, match="voxel"):
        _add(MatrixFreeNormalEquations(3), np.zeros(2), np.eye(2), [1., 2.])


@pytest.mark.parametrize("bad_rhs", [[np.nan, 0.], [np.inf, 0.]])
def test_nonfinite_rhs_never_returns_a_result(bad_rhs):
    with pytest.raises(ValueError, match="finite"):
        solve_slowness_update(np.eye(2), bad_rhs, (2,), 0.1)


def test_scatter_compiled_and_uncompiled_agree():
    scatter = math._scatter_add_dense
    if not hasattr(scatter, "py_func"):
        pytest.skip("Optional compiled scatter unavailable")
    active = np.array([0, 2, 4])
    gram = np.arange(9., dtype=float).reshape(3, 3)
    compiled, python = np.zeros((5, 5)), np.zeros((5, 5))
    scatter(compiled, active, gram)
    scatter.py_func(python, active, gram)
    np.testing.assert_array_equal(compiled, python)
    np.testing.assert_array_equal(compiled[np.ix_(active, active)], gram)


def _block_reference_matvec(operator, vector):
    """The original per-hypothesis implementation, independent of packing."""
    result = np.zeros(operator.n_vox)
    for block in operator._blocks:
        projected = block.rows @ vector[block.active]
        projected -= projected[0]
        centered = projected - np.dot(block.precision, projected) / block.precision.sum()
        result[block.active] += block.weight * np.asarray(
            block.rows.T @ (block.precision * centered)
        ).ravel()
    return result


def _mixed_operator():
    operator, rhs = MatrixFreeNormalEquations(101), np.zeros(101)
    rng = np.random.default_rng(32)
    for index, n_stations in enumerate([2, 5, 9, 17]):
        rows = np.zeros((n_stations, 101))
        active = rng.choice(101, size=12, replace=False)
        rows[:, active] = rng.normal(size=(n_stations, active.size))
        rows[rng.random(rows.shape) < 0.65] = 0
        rows[0] = 0  # Empty CSR station rows must retain their centering weight.
        _add(operator, rhs, sparse.csr_matrix(rows) if index % 2 else rows,
             rng.normal(size=n_stations), sigmas=rng.uniform(0.2, 3.0, size=n_stations),
             weight=0.1 + index)
    return operator


@pytest.mark.parametrize("backend", ["compiled", "scipy"])
def test_packed_matvec_matches_block_reference_for_thousands_of_mixed_blocks(monkeypatch, backend):
    if backend == "scipy":
        monkeypatch.setattr(math, "_packed_matvec", math._packed_matvec_scipy)
    elif not hasattr(math._packed_matvec, "py_func"):
        pytest.skip("Optional compiled matvec unavailable")
    operator = _mixed_operator()
    for _ in range(10):
        operator.merge(operator)
    assert len(operator._blocks) == 4096
    assert operator._packed is None
    vector = np.random.default_rng(5).normal(size=operator.n_vox)
    expected = _block_reference_matvec(operator, vector)
    np.testing.assert_allclose(operator.matvec(vector), expected, rtol=2e-12, atol=2e-10)
    packed = operator._packed
    assert packed.block_ptr.size == 4097
    assert packed.indptr.size == packed.precision.size + 1
    assert packed.indptr[-1] == packed.data.size == packed.indices.size
    assert packed.indices.min() >= 0 and packed.indices.max() < operator.n_vox
    # Packing maps each block's local CSR columns back to global voxel indices.
    global_rows = sparse.csr_matrix((packed.data, packed.indices, packed.indptr),
                                    shape=(packed.precision.size, operator.n_vox))
    start, stop = packed.block_ptr[:2]
    block = operator._blocks[0]
    np.testing.assert_allclose((global_rows @ vector)[start:stop], block.rows @ vector[block.active])

    class NoBlockTraversal(list):
        def __iter__(self):
            raise AssertionError("A cached matvec traversed Python hypothesis blocks")

    operator._blocks = NoBlockTraversal(operator._blocks)
    np.testing.assert_allclose(operator.matvec(vector), expected, rtol=2e-12, atol=2e-10)
    assert operator._packed is packed
    if backend == "compiled":
        assert math._packed_matvec.targetoptions["fastmath"] is False
        assert not math._packed_matvec.targetoptions.get("parallel", False)


@pytest.mark.parametrize("backend", ["compiled", "scipy"])
def test_packed_cache_invalidates_after_accumulation_and_merge(monkeypatch, backend):
    if backend == "scipy":
        monkeypatch.setattr(math, "_packed_matvec", math._packed_matvec_scipy)
    operator = _mixed_operator()
    vector = np.random.default_rng(13).normal(size=operator.n_vox)
    linear_operator = operator.aslinearoperator()
    linear_operator @ vector
    original = operator._packed
    _add(operator, np.zeros(101), sparse.eye(3, 101, format="csr"), [1., -2., 3.],
         sigmas=[1., 2., 0.5])
    assert operator._packed is None
    np.testing.assert_allclose(linear_operator @ vector, _block_reference_matvec(operator, vector),
                               rtol=1e-13, atol=1e-13)
    assert operator._packed is not original
    before_merge = operator._packed
    other = _mixed_operator()
    other_value = other.matvec(vector)
    other_packed = other._packed
    operator.merge(other)
    assert operator._packed is None
    np.testing.assert_allclose(linear_operator @ vector, _block_reference_matvec(operator, vector),
                               rtol=1e-13, atol=1e-13)
    assert operator._packed is not before_merge
    assert other._packed is other_packed
    np.testing.assert_array_equal(other.matvec(vector), other_value)
    # Packed and unpacked chunks remain serializable for event workers.
    restored = pickle.loads(pickle.dumps(operator))
    np.testing.assert_array_equal(restored.matvec(vector), operator.matvec(vector))
    cached = operator._packed
    with pytest.raises(ValueError, match="incompatible"):
        operator.merge(MatrixFreeNormalEquations(102))
    assert operator._packed is cached


@pytest.mark.parametrize("backend", ["compiled", "scipy"])
def test_packed_empty_and_identical_station_predictions(monkeypatch, backend):
    if backend == "scipy":
        monkeypatch.setattr(math, "_packed_matvec", math._packed_matvec_scipy)
    operator = MatrixFreeNormalEquations(4)
    vector = np.array([1., -2., 3., -4.])
    np.testing.assert_array_equal(operator.matvec(vector), 0)
    empty_packed = operator._packed
    rng = np.random.default_rng(11)
    rows = np.tile([1e8, 0., 3e8, 0.], (17, 1))
    _add(operator, np.zeros(4), sparse.csr_matrix(rows), rng.normal(size=17),
         sigmas=rng.uniform(0.2, 2.0, size=17))
    assert operator._packed is None
    np.testing.assert_array_equal(operator.matvec(vector), 0)
    assert operator._packed is not empty_packed
    np.testing.assert_array_equal(operator.matvec(vector), _block_reference_matvec(operator, vector))
