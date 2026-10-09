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
                     "weights_min_distance", "n_workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
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
