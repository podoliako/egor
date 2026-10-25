from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg

try:
    from numba import njit
except ImportError:  # Compiled kernels are optional; SciPy fallbacks remain available.
    njit = None

from instruments.hypocentre_objective import make_refinement_objective
from instruments.instruments_coords import cell_coord_bounds, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise, negative_log_likelihood


def select_candidate_cells(cost: np.ndarray, n: int, min_distance: int = 1) -> np.ndarray:
    """Lowest-cost cells, pairwise separated by at least ``min_distance`` (Chebyshev, in cells)."""
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("cost must be a 3-D array")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be an integer >= 1")
    if not isinstance(min_distance, (int, np.integer)) or min_distance < 1:
        raise ValueError("min_distance must be an integer >= 1")

    flat = values.ravel()
    # Each selected cell can exclude at most (2d - 1)^3 better-ranked cells.
    n_considered = min(flat.size, n * (2 * min_distance - 1) ** 3 + n)
    while True:
        if n_considered < flat.size:
            order = np.argpartition(flat, n_considered - 1)[:n_considered]
            order = order[np.argsort(flat[order], kind="stable")]
        else:
            order = np.argsort(flat, kind="stable")
        selected = []
        for flat_index in order:
            if not np.isfinite(flat[flat_index]):
                continue
            index = np.asarray(np.unravel_index(int(flat_index), values.shape), dtype=np.int64)
            if any(np.max(np.abs(index - previous)) < min_distance for previous in selected):
                continue
            selected.append(index)
            if len(selected) == n:
                return np.asarray(selected, dtype=np.int64)
        if n_considered >= flat.size:
            break
        n_considered = flat.size
    if not selected:
        raise ValueError("No finite hypocentre candidates available")
    return np.asarray(selected, dtype=np.int64)


def refine_hypocentre_in_cell(
    station_fields: np.ndarray,
    observed: np.ndarray,
    cell_index,
    noise: PickNoise,
    *,
    accelerated: bool = True,
) -> tuple[np.ndarray, float]:
    """Continuous maximum-likelihood position of one hypothesis within its cell."""
    start = np.asarray(cell_index, dtype=np.float64)
    bounds = cell_coord_bounds(tuple(int(v) for v in cell_index), station_fields.shape[1:])

    if accelerated:
        objective = make_refinement_objective(station_fields, observed, noise)
    else:
        def objective(coord):
            predicted = sample_cell_centered_trilinear_batch(station_fields, coord)
            return float(negative_log_likelihood(predicted, observed, noise))

    start_value = objective(start)
    result = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-3, "ftol": 1e-8, "maxiter": 50},
    )
    refined = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(refined)):
        return start, start_value
    refined_value = objective(refined)
    if not np.isfinite(refined_value) or refined_value > start_value:
        return start, start_value
    return refined, refined_value


@dataclass(frozen=True)
class _NormalBlock:
    active: np.ndarray
    rows: np.ndarray | sparse.csr_matrix
    precision: np.ndarray
    weight: float


@dataclass(frozen=True)
class _PackedNormalEquations:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    block_ptr: np.ndarray
    precision: np.ndarray
    precision_sums: np.ndarray
    weights: np.ndarray
    max_stations: int


def _packed_matvec(vector, indptr, indices, data, block_ptr, precision,
                   precision_sums, weights, max_stations):
    result = np.zeros(vector.size, dtype=np.float64)
    projected = np.empty(max_stations, dtype=np.float64)
    # Blocks are serial: their output voxels overlap. Do not enable fastmath;
    # subtracting a common station value before centring preserves null modes.
    for block in range(weights.size):
        start, stop = block_ptr[block], block_ptr[block + 1]
        for row in range(start, stop):
            value = 0.0
            for entry in range(indptr[row], indptr[row + 1]):
                value += data[entry] * vector[indices[entry]]
            projected[row - start] = value
        baseline = projected[0]
        weighted_sum = 0.0
        for row in range(start, stop):
            projected[row - start] -= baseline
            weighted_sum += precision[row] * projected[row - start]
        mean = weighted_sum / precision_sums[block]
        for row in range(start, stop):
            value = weights[block] * precision[row] * (projected[row - start] - mean)
            for entry in range(indptr[row], indptr[row + 1]):
                result[indices[entry]] += data[entry] * value
    return result


def _packed_matvec_scipy(vector, indptr, indices, data, block_ptr, precision,
                         precision_sums, weights, max_stations):
    """Batched sparse fallback without a Python loop over hypotheses."""
    if weights.size == 0:
        return np.zeros(vector.size, dtype=np.float64)
    rows = sparse.csr_matrix((data, indices, indptr), shape=(precision.size, vector.size), copy=False)
    counts = np.diff(block_ptr)
    projected = rows @ vector
    projected -= np.repeat(projected[block_ptr[:-1]], counts)
    means = np.add.reduceat(precision * projected, block_ptr[:-1]) / precision_sums
    centered = projected - np.repeat(means, counts)
    return np.asarray(rows.T @ (np.repeat(weights, counts) * precision * centered)).ravel()


if njit is not None:
    _packed_matvec = njit(cache=True, fastmath=False)(_packed_matvec)
else:
    _packed_matvec = _packed_matvec_scipy


class MatrixFreeNormalEquations:
    """Compact normal-equation blocks; the RHS is accumulated separately.

    Each block owns its reached-station rows on active voxels only. CSR input
    remains sparse, including during matrix-vector products. No voxel Gram
    matrix is stored. Blocks are not modified after assembly, so merged chunks
    can safely share them. The first matvec lazily packs one global-index CSR
    payload for all blocks; subsequent products avoid per-hypothesis Python or
    SciPy calls. Accumulation and merging invalidate this derived cache.
    """

    def __init__(self, n_vox: int):
        if not isinstance(n_vox, (int, np.integer)) or n_vox < 0:
            raise ValueError("n_vox must be a non-negative integer")
        self.n_vox = int(n_vox)
        self.diag = np.zeros(self.n_vox, dtype=np.float64)
        self._blocks: list[_NormalBlock] = []
        self._packed: _PackedNormalEquations | None = None

    def diagonal(self) -> np.ndarray:
        return self.diag.copy()

    def merge(self, other: MatrixFreeNormalEquations) -> MatrixFreeNormalEquations:
        """Merge compatible chunks in place; callers must sum their RHS arrays."""
        if not isinstance(other, MatrixFreeNormalEquations) or other.n_vox != self.n_vox:
            raise ValueError("Cannot merge normal equations with incompatible voxel counts/types")
        self.diag += other.diag
        self._blocks.extend(other._blocks)
        self._packed = None
        return self

    def _pack(self) -> _PackedNormalEquations:
        if self._packed is not None:
            return self._packed
        n_rows = sum(block.precision.size for block in self._blocks)
        nnz = sum(block.rows.nnz if sparse.issparse(block.rows) else np.count_nonzero(block.rows)
                  for block in self._blocks)
        indptr = np.empty(n_rows + 1, dtype=np.int64)
        indices = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float64)
        block_ptr = np.empty(len(self._blocks) + 1, dtype=np.int64)
        precision = np.empty(n_rows, dtype=np.float64)
        precision_sums = np.empty(len(self._blocks), dtype=np.float64)
        weights = np.empty(len(self._blocks), dtype=np.float64)
        row_offset = entry_offset = max_stations = 0
        indptr[0] = block_ptr[0] = 0
        for i, block in enumerate(self._blocks):
            rows = sparse.csr_matrix(block.rows, copy=False)
            row_stop = row_offset + block.precision.size
            entry_stop = entry_offset + rows.nnz
            indptr[row_offset + 1:row_stop + 1] = rows.indptr[1:] + entry_offset
            indices[entry_offset:entry_stop] = block.active[rows.indices]
            data[entry_offset:entry_stop] = rows.data
            precision[row_offset:row_stop] = block.precision
            precision_sums[i] = block.precision.sum()
            weights[i] = block.weight
            block_ptr[i + 1] = row_stop
            max_stations = max(max_stations, block.precision.size)
            row_offset, entry_offset = row_stop, entry_stop
        self._packed = _PackedNormalEquations(
            indptr, indices, data, block_ptr, precision, precision_sums, weights, max_stations,
        )
        return self._packed

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(self.n_vox)
        packed = self._pack()
        return _packed_matvec(
            vector, packed.indptr, packed.indices, packed.data, packed.block_ptr,
            packed.precision, packed.precision_sums, packed.weights, packed.max_stations,
        )

    def aslinearoperator(self) -> LinearOperator:
        return LinearOperator(
            (self.n_vox, self.n_vox), matvec=self.matvec, rmatvec=self.matvec, dtype=np.float64,
        )


def _symmetric_to_roundoff(matrix):
    largest = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            largest = max(largest, abs(matrix[i, j]))
    tolerance = 10 * np.finfo(np.float64).eps * largest
    for i in range(matrix.shape[0]):
        for j in range(i):
            if abs(matrix[i, j] - matrix[j, i]) > tolerance:
                return False
    return True


def _scatter_add_dense(hessian, active, gram):
    for i in range(active.size):
        for j in range(active.size):
            hessian[active[i], active[j]] += gram[i, j]


if njit is not None:
    _scatter_add_dense = njit(cache=True)(_scatter_add_dense)
    _symmetric_to_roundoff = njit(cache=True)(_symmetric_to_roundoff)
else:
    def _symmetric_to_roundoff(matrix):
        tolerance = 10 * np.finfo(float).eps * np.max(np.abs(matrix))
        return np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance)

    def _scatter_add_dense(hessian, active, gram):
        hessian[np.ix_(active, active)] += gram


def _compact_station_rows(station_sensitivities, valid, n_vox):
    """Filter reached stations, then gather active columns just once."""
    if sparse.issparse(station_sensitivities):
        if station_sensitivities.shape != (valid.size, n_vox):
            raise ValueError("Sparse station sensitivities must have shape (stations, n_vox)")
        rows = sparse.csr_matrix(station_sensitivities, dtype=np.float64)[valid].copy()
        rows.sum_duplicates()
        rows.eliminate_zeros()
        active = np.unique(rows.indices)
        rows = rows[:, active]
        values = rows.data
    else:
        rows = np.asarray(station_sensitivities).reshape(valid.size, n_vox)[valid]
        active = np.flatnonzero(np.any(rows != 0.0, axis=0))
        rows = np.asarray(rows[:, active], dtype=np.float64)
        values = rows
    if not np.all(np.isfinite(values)):
        raise ValueError("Valid station sensitivities must be finite")
    return active, rows


def _sparse_centered_diagonal(rows, precision, mean):
    # Sum squared deviations rather than subtracting two nearly equal moments.
    # Implicit zeros contribute mean**2 times their total station precision.
    row_precision = np.repeat(precision, np.diff(rows.indptr))
    present_precision = np.bincount(rows.indices, weights=row_precision, minlength=rows.shape[1])
    diagonal = np.bincount(
        rows.indices, weights=row_precision * np.square(rows.data - mean[rows.indices]),
        minlength=rows.shape[1],
    )
    missing_precision = np.maximum(precision.sum() - present_precision, 0.0)
    # Different reduction orders can leave an O(eps) difference even when no
    # entries are missing; that error would swamp a near-zero variance.
    counts = np.bincount(rows.indices, minlength=rows.shape[1])
    missing_precision[counts == rows.shape[0]] = 0.0
    return diagonal + missing_precision * np.square(mean)


def accumulate_normal_equations(
    hessian: np.ndarray | MatrixFreeNormalEquations,
    rhs: np.ndarray,
    station_sensitivities: np.ndarray | sparse.csr_matrix,
    station_residuals: np.ndarray,
    valid_stations: np.ndarray,
    station_sigmas: np.ndarray,
    weight: float,
) -> None:
    """Add one weighted hypothesis to dense or matrix-free equations in place.

    Dense sensitivities may have shape ``[S, *model_shape]`` or ``[S, M]``;
    sparse sensitivities have shape ``[S, M]``. Only reached/valid stations are
    used. The unknown origin time is profiled out by precision-weighted
    centring, equivalent to pair weights ``p_i * p_j / sum(p)``.
    """
    valid = np.asarray(valid_stations, dtype=bool)
    if np.count_nonzero(valid) < 2 or weight <= 0.0:
        return
    if not np.isfinite(weight):
        raise ValueError("Hypothesis weight must be finite")
    n_vox = rhs.size
    if isinstance(hessian, MatrixFreeNormalEquations) and hessian.n_vox != n_vox:
        raise ValueError("Normal equations and RHS must have matching voxel counts")
    residual = np.asarray(station_residuals, dtype=np.float64)[valid]
    sigmas = np.asarray(station_sigmas, dtype=np.float64)
    if sigmas.shape != valid.shape:
        raise ValueError("station_sigmas must match valid_stations shape")
    sigmas = sigmas[valid]
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Valid station sigmas must be finite and positive")
    if not np.all(np.isfinite(residual)):
        raise ValueError("Valid station residuals must be finite")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        precision = 1.0 / np.square(sigmas)
    if not np.all(np.isfinite(precision)) or not np.isfinite(precision.sum()) or precision.sum() <= 0:
        raise ValueError("Station precisions must have a finite, positive sum")

    active, rows = _compact_station_rows(station_sensitivities, valid, n_vox)
    if active.size == 0:
        return
    residual_centered = residual - np.average(residual, weights=precision)
    if isinstance(hessian, MatrixFreeNormalEquations):
        mean = np.asarray(rows.T @ precision).ravel() / precision.sum()
        if sparse.issparse(rows):
            diagonal = _sparse_centered_diagonal(rows, precision, mean)
            contribution = np.asarray(rows.T @ (precision * residual_centered)).ravel()
            contribution -= mean * np.dot(precision, residual_centered)
        else:
            centered = rows - mean
            diagonal = np.sum(precision[:, None] * np.square(centered), axis=0)
            contribution = centered.T @ (precision * residual_centered)
        hessian.diag[active] += weight * diagonal
        hessian._blocks.append(_NormalBlock(active, rows, precision, float(weight)))
        hessian._packed = None
    else:
        if sparse.issparse(rows):
            rows = rows.toarray()
        rows_centered = rows - np.average(rows, axis=0, weights=precision)
        gram = rows_centered.T @ (precision[:, None] * rows_centered)
        gram *= weight
        _scatter_add_dense(hessian, active, gram)
        contribution = rows_centered.T @ (precision * residual_centered)
    rhs[active] += weight * contribution


def solve_slowness_update(
    hessian,
    rhs,
    model_shape,
    lambda_reg: float,
    coverage_damping_power: float = 0.0,
    coverage_floor: float = 0.05,
    coverage_reference_percentile: float = 75.0,
    solver: str = "auto",
    rtol: float = 1e-10,
    maxiter: int | None = None,
    diagnostics: dict | None = None,
):
    """Solve the damped normal system, returning the original three arrays.

    Damping acts on the increment, relative to the mean of ``diag(H)``. The
    coverage/confidence formulas are identical for dense and matrix-free data.
    ``auto`` uses checked Cholesky (then LU/lstsq) for dense inputs and Jacobi
    CG for matrix-free inputs. ``legacy`` forces dense LU/lstsq; ``cg`` requires
    matrix-free input. CG supports nonnegative damping and consistent singular
    PSD systems, but never falls back to dense storage. Nonconvergence or an
    unacceptable actual residual raises RuntimeError for CG/direct solves.
    Dense lstsq retains its original semantics: finite least-squares solutions
    are returned even for inconsistent systems, with their residual reported.

    ``diagnostics``, if supplied, receives ``solver``, ``iterations`` (CG steps;
    zero for direct solves), and ``relative_residual``. Returns
    ``(delta_s, sensitivity_diagonal, coverage_confidence)`` as before.
    """
    n_vox = int(np.prod(model_shape))
    matrix_free = isinstance(hessian, MatrixFreeNormalEquations)
    if matrix_free:
        if hessian.n_vox != n_vox:
            raise ValueError("Normal equations and model_shape must have matching voxel counts")
        diagonal = hessian.diagonal()
    else:
        hessian = np.asarray(hessian, dtype=np.float64).reshape(n_vox, n_vox)
        diagonal = np.diag(hessian)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n_vox)
    if solver not in ("auto", "legacy", "cg"):
        raise ValueError("solver must be 'auto', 'legacy', or 'cg'")
    if (matrix_free and solver == "legacy") or (not matrix_free and solver == "cg"):
        raise ValueError("'legacy' requires dense equations; 'cg' requires matrix-free equations")
    if not np.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError("rtol must be finite and in (0, 1)")
    if maxiter is not None and (not isinstance(maxiter, (int, np.integer)) or maxiter < 1):
        raise ValueError("maxiter must be a positive integer or None")
    if not np.isfinite(coverage_damping_power) or coverage_damping_power < 0.0:
        raise ValueError("coverage_damping_power must be finite and >= 0")
    if not 0.0 < coverage_floor <= 1.0:
        raise ValueError("coverage_floor must be in (0, 1]")
    if not 0.0 <= coverage_reference_percentile <= 100.0:
        raise ValueError("coverage_reference_percentile must be in [0, 100]")
    if (not np.isfinite(lambda_reg) or not np.all(np.isfinite(rhs))
            or not np.all(np.isfinite(diagonal))
            or (not matrix_free and not np.all(np.isfinite(hessian)))):
        raise ValueError("Normal equations, RHS, and damping must be finite")
    if matrix_free and (lambda_reg < 0.0 or np.any(diagonal < 0.0)):
        raise RuntimeError("Matrix-free CG requires a positive semidefinite system and nonnegative damping")

    sensitivity = np.maximum(diagonal, 0.0)
    positive = sensitivity[sensitivity > 0.0]
    reference = float(np.percentile(positive, coverage_reference_percentile)) if positive.size else 1.0
    confidence = np.clip(sensitivity / max(reference, np.finfo(float).tiny), 0.0, 1.0)
    scale = max(float(np.sum(diagonal)) / n_vox, 0.0) if n_vox else 0.0
    damping = lambda_reg * scale / np.power(np.maximum(confidence, coverage_floor), coverage_damping_power)
    if not np.all(np.isfinite(damping)):
        raise RuntimeError("Normal-equation damping is not finite")

    iterations = 0
    info = 0
    chosen = "zero"
    if matrix_free:
        system = LinearOperator(
            (n_vox, n_vox), matvec=lambda x: hessian.matvec(x) + damping * x, dtype=np.float64,
        )
    else:
        system = hessian.copy()
        system.flat[::n_vox + 1] += damping

    if not np.any(rhs):
        delta_s = np.zeros(n_vox, dtype=np.float64)
    elif matrix_free:
        chosen = "cg"
        system_diagonal = diagonal + damping
        if not np.any(system_diagonal) or np.any((system_diagonal == 0) & (rhs != 0)):
            raise RuntimeError("Matrix-free CG cannot solve a zero system or a nonzero RHS on a zero row")
        inverse_diagonal = np.ones(n_vox, dtype=np.float64)
        np.divide(1.0, system_diagonal, out=inverse_diagonal, where=system_diagonal > 0)
        preconditioner = LinearOperator(
            (n_vox, n_vox), matvec=lambda x: inverse_diagonal * x, dtype=np.float64,
        )

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        delta_s, info = cg(
            system, rhs, M=preconditioner, rtol=rtol, atol=0.0,
            maxiter=maxiter, callback=count_iteration,
        )
    else:
        delta_s = None
        if solver == "auto":
            # Cholesky reads only one triangle: never use it on an asymmetric
            # matrix, except for symmetry errors at floating-point roundoff.
            if _symmetric_to_roundoff(system):
                try:
                    factor = linalg.cho_factor(system, lower=True, check_finite=False)
                    delta_s = linalg.cho_solve(factor, rhs, check_finite=False)
                    chosen = "cholesky"
                except np.linalg.LinAlgError:
                    pass
        if delta_s is None:
            try:
                delta_s = np.linalg.solve(system, rhs)
                chosen = "solve"
            except np.linalg.LinAlgError:
                delta_s = np.linalg.lstsq(system, rhs, rcond=None)[0]
                chosen = "lstsq"

    # Check the original operator, not CG's recursively updated residual.
    residual = system @ delta_s - rhs
    rhs_norm = float(linalg.norm(rhs, check_finite=False))
    residual_norm = float(linalg.norm(residual, check_finite=False))
    relative_residual = residual_norm / rhs_norm if rhs_norm else residual_norm
    if diagnostics is not None:
        diagnostics.update(solver=chosen, iterations=iterations, relative_residual=relative_residual)
    if (info != 0 or not np.all(np.isfinite(delta_s)) or not np.isfinite(relative_residual)
            or (matrix_free and relative_residual > rtol)):
        raise RuntimeError(
            f"Normal-equation {chosen} solve failed: info={info}, iterations={iterations}, "
            f"actual relative residual={relative_residual:.3e}, required rtol={rtol:.3e}"
        )
    return (
        delta_s.reshape(model_shape),
        sensitivity.reshape(model_shape),
        confidence.reshape(model_shape),
    )
