"""Single source of EMTomo inversion settings and their validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InversionConfig:
    # Inversion grid; the cell must tile the forward experiment domain exactly.
    cell_size: float = 10_000.0
    subdivision: int = 3
    slowness_interpolation: str = "nearest"

    # Starting model: homogeneous unless a gradient or horizontal layers are set.
    initial_velocity_m_s: float = 5000.0
    initial_gradient_m_s: tuple[float, float] | None = None
    initial_layer_boundaries_km: tuple[float, ...] | None = None
    initial_layer_velocities_m_s: tuple[float, ...] | None = None

    # Hypocentre hypotheses and their likelihood.
    n_cycles: int = 7
    # Cells shortlisted by grid likelihood and refined within the cell; the best
    # weights_top_n refined positions become the event's weighted hypotheses.
    n_candidates: int = 5
    weights_top_n: int = 1
    weights_min_distance: int = 1
    candidate_mode: str = "soft"
    temperature: float = 1.0
    # None takes the pick-noise sigmas from the experiment metadata.
    weight_noise_relative_sigma: float | None = None
    weight_noise_absolute_sigma_s: float | None = None
    weight_model_sigma_s: float = 0.2

    # Linearized update.
    lambda_reg: float = 0.01
    coverage_damping_power: float = 1.0
    coverage_floor: float = 0.05
    coverage_reference_percentile: float = 75.0
    max_velocity_step_fraction: float | None = 0.03

    # Runtime and output. Change run_version for every method release.
    n_workers: int = 25
    run_name: str = "em"
    run_version: str = "1.4"
    runs_dir: str = "runs"
    save_runs: bool = True
    log_g_per_weight: bool = False

    # Execution controls; these do not change the inversion model or Powell settings.
    likelihood_batch_size: int = 8
    likelihood_batch_memory_mb: int = 128
    compact_rays: bool = True
    fast_refinement: bool = True
    linear_solver: str = "auto"
    matrix_free_min_voxels: int = 8192
    solver_rtol: float = 1e-10
    solver_maxiter: int | None = None

    def __post_init__(self):
        def positive(name):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        def nonnegative(name):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")

        for name in ("cell_size", "initial_velocity_m_s", "temperature"):
            positive(name)
        for name in ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s",
                     "weight_model_sigma_s", "lambda_reg", "coverage_damping_power"):
            nonnegative(name)
        for name in ("subdivision", "n_cycles", "n_candidates", "weights_top_n",
                     "weights_min_distance", "n_workers", "likelihood_batch_size",
                     "likelihood_batch_memory_mb", "matrix_free_min_voxels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.solver_maxiter is not None:
            if (isinstance(self.solver_maxiter, bool)
                    or not isinstance(self.solver_maxiter, (int, np.integer))
                    or self.solver_maxiter < 1):
                raise ValueError("solver_maxiter must be a positive integer or None")
        for name in ("compact_rays", "fast_refinement"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if not isinstance(self.linear_solver, str) or self.linear_solver not in (
            "auto", "dense", "matrix_free", "legacy",
        ):
            raise ValueError("linear_solver must be 'auto', 'dense', 'matrix_free', or 'legacy'")
        if self.linear_solver == "matrix_free" and (self.lambda_reg is None or self.lambda_reg <= 0):
            raise ValueError("linear_solver='matrix_free' requires lambda_reg > 0")
        if (not isinstance(self.solver_rtol, (int, float, np.integer, np.floating))
                or not np.isfinite(self.solver_rtol) or not 0.0 < self.solver_rtol < 1.0):
            raise ValueError("solver_rtol must be finite and in (0, 1)")
        if self.weights_top_n > self.n_candidates:
            raise ValueError("weights_top_n must not exceed n_candidates")
        if self.slowness_interpolation not in ("nearest", "trilinear"):
            raise ValueError("slowness_interpolation must be 'nearest' or 'trilinear'")
        if self.candidate_mode not in ("soft", "hard"):
            raise ValueError("candidate_mode must be 'soft' or 'hard'")
        if not 0.0 < self.coverage_floor <= 1.0:
            raise ValueError("coverage_floor must be in (0, 1]")
        if not 0.0 <= self.coverage_reference_percentile <= 100.0:
            raise ValueError("coverage_reference_percentile must be in [0, 100]")
        step = self.max_velocity_step_fraction
        if step is not None and not 0.0 < step < 1.0:
            raise ValueError("max_velocity_step_fraction must be in (0, 1) or None")
        has_layers = (self.initial_layer_boundaries_km is not None
                      or self.initial_layer_velocities_m_s is not None)
        if has_layers and self.initial_gradient_m_s is not None:
            raise ValueError("initial layers and initial_gradient_m_s are mutually exclusive")
