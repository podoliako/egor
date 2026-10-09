"""Cell-centred velocity model on a uniform cubic grid."""
from __future__ import annotations

import numpy as np

from interpolation import prolongate_cell_centered_trilinear


class VelocityModel:
    """Velocities in m/s; index ``(i, j, k)`` is the centre of the cell
    ``[(i, j, k) * cell_size, (i + 1, j + 1, k + 1) * cell_size)``, z down."""

    def __init__(self, velocity: np.ndarray, cell_size: float):
        velocity = np.array(velocity, dtype=np.float64)
        if velocity.ndim != 3 or 0 in velocity.shape:
            raise ValueError("velocity must be a non-empty 3-D array")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError("velocity must be finite and positive")
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be finite and positive")
        velocity.flags.writeable = False
        self.velocity = velocity
        self.cell_size = float(cell_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.velocity.shape

    def refined(self, subdivision: int, interpolation: str = "nearest") -> "VelocityModel":
        """Split every cell into ``subdivision**3`` cells.

        ``nearest`` copies the parent value; ``trilinear`` interpolates slowness
        between cell centres and converts it back to velocity.
        """
        if subdivision < 1:
            raise ValueError("subdivision must be >= 1")
        if subdivision == 1:
            return self
        if interpolation == "nearest":
            fine = self.velocity
            for axis in range(3):
                fine = np.repeat(fine, subdivision, axis=axis)
        elif interpolation == "trilinear":
            fine = 1.0 / prolongate_cell_centered_trilinear(1.0 / self.velocity, subdivision)
        else:
            raise ValueError("interpolation must be 'nearest' or 'trilinear'")
        return VelocityModel(fine, self.cell_size / subdivision)


def block_average_slowness(
    velocity: np.ndarray,
    cell_size: float,
    target_shape: tuple[int, int, int],
    target_cell_size: float,
) -> np.ndarray:
    """Volume-average slowness of ``velocity`` over each target cell.

    Both grids start at the origin and must cover the same extent; cells need
    not nest. Returns velocity (inverse of the mean slowness).
    """
    slowness = 1.0 / np.asarray(velocity, dtype=np.float64)
    for axis, (n_source, n_target) in enumerate(zip(slowness.shape, target_shape)):
        if not np.isclose(n_source * cell_size, n_target * target_cell_size, rtol=1e-12, atol=0):
            raise ValueError("source and target grids must cover the same extent")
        source_edges = np.arange(n_source + 1) * cell_size
        target_edges = np.arange(n_target + 1) * target_cell_size
        overlap = np.clip(
            np.minimum(target_edges[1:, None], source_edges[None, 1:])
            - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
            0.0,
            None,
        ) / target_cell_size
        slowness = np.moveaxis(np.tensordot(overlap, slowness, axes=(1, axis)), 0, axis)
    return 1.0 / slowness
