"""EMTomo inversion setup for a saved, ID-validated forward experiment.

Observed arrivals and station positions come from the experiment. Source
velocities are used only for optional quality metrics, never to initialize the
inversion. True event positions are not passed to the inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from experiment_data import DEFAULT_EXPERIMENTS_ROOT, load_tomography_experiment
from tomography.tomography import run_em, warm_up_jit
from velocity_model import VelocityModel


@dataclass(frozen=True)
class PreparedInversion:
    initial_model: VelocityModel
    reference_model: VelocityModel
    station_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    station_locs: list[tuple[float, float, float]]
    arrivals_table: np.ndarray
    noise_sigmas: tuple[float, float]


def resolve_weight_sigmas(config, noise_sigmas: tuple[float, float]) -> tuple[float, float, float]:
    """Resolve explicit overrides against the observation noise, then validate."""
    values = (
        noise_sigmas[0] if config.weight_noise_relative_sigma is None else config.weight_noise_relative_sigma,
        noise_sigmas[1] if config.weight_noise_absolute_sigma_s is None else config.weight_noise_absolute_sigma_s,
        config.weight_model_sigma_s,
    )
    names = ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s", "weight_model_sigma_s")
    resolved = []
    for name, value in zip(names, values):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and nonnegative")
        try:
            sigma = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite and nonnegative") from error
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        resolved.append(sigma)
    if not any(value > 0 for value in resolved):
        raise ValueError("weight noise/model sigmas must have positive combined variance")
    return tuple(resolved)


def prepare_inversion(experiment_id: str, config, experiments_root=DEFAULT_EXPERIMENTS_ROOT) -> PreparedInversion:
    """Validate observations and match the inversion geometry to the input domain.

    The saved velocity is only sampled for reference metrics; initial velocities
    come from the explicitly configured homogeneous, gradient or horizontal layers.
    """
    data = load_tomography_experiment(experiment_id, experiments_root)
    side = float(config.cell_size)
    if not np.isfinite(side) or side <= 0:
        raise ValueError("Inversion cell_size must be finite and positive")
    extent = np.asarray(data.source_model.velocity.shape) * data.source_model.cell_size_m
    ratio = extent / side
    shape = np.rint(ratio).astype(np.intp)
    if np.any(shape < 1) or not np.allclose(shape, ratio, rtol=1e-12, atol=0):
        raise ValueError(f"Inversion cell_size={side:g} m must exactly tile experiment dimensions {tuple(extent)} m")
    if not np.isfinite(config.background_vp) or config.background_vp <= 0:
        raise ValueError("background_vp must be finite and positive")
    model_config = {
        "lon": config.lon, "lat": config.lat, "height": config.height,
        "azimuth": config.azimuth, "side_size": side,
        "n_x": int(shape[0]), "n_y": int(shape[1]), "n_z": int(shape[2]),
    }
    initial = VelocityModel.from_config(model_config)
    boundaries = config.initial_layer_boundaries_km
    velocities = config.initial_layer_velocities_m_s
    if boundaries is not None or velocities is not None:
        if config.initial_gradient_m_s is not None:
            raise ValueError("initial layers and initial_gradient_m_s are mutually exclusive")
        if boundaries is None or velocities is None:
            raise ValueError("initial layers require both boundaries and velocities")
        boundaries = np.asarray(boundaries, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if (boundaries.ndim != 1 or len(boundaries) < 1 or not np.all(np.isfinite(boundaries))
                or boundaries[0] <= 0 or boundaries[-1] >= extent[2] / 1000
                or np.any(np.diff(boundaries) <= 0)):
            raise ValueError("initial_layer_boundaries_km must increase strictly inside the depth range")
        if (velocities.ndim != 1 or len(velocities) != len(boundaries) + 1
                or not np.all(np.isfinite(velocities)) or np.any(velocities <= 0)):
            raise ValueError("initial_layer_velocities_m_s must contain one finite positive speed per layer")
        depths_km = (np.arange(shape[2], dtype=np.float64) + 0.5) * side / 1000
        profile = velocities[np.searchsorted(boundaries, depths_km, side="right")]
        initial.set_vp_array(np.broadcast_to(profile, tuple(shape)))
    elif config.initial_gradient_m_s is None:
        initial.fill_linear_gradient("vp", config.background_vp, config.background_vp)
    else:
        endpoints = np.asarray(config.initial_gradient_m_s, dtype=np.float64)
        if endpoints.shape != (2,) or not np.all(np.isfinite(endpoints)) or np.any(endpoints <= 0):
            raise ValueError("initial_gradient_m_s must contain two finite positive velocities")
        depths = (np.arange(shape[2], dtype=np.float64) + 0.5) / shape[2]
        profile = endpoints[0] + (endpoints[1] - endpoints[0]) * depths
        initial.set_vp_array(np.broadcast_to(profile, tuple(shape)))

    reference = VelocityModel.from_config(model_config)
    # Sample at the centres of inversion cells. The reference truth is only
    # supplied to quality metrics / logger, never to the solver's initial model.
    source_shape = np.asarray(data.source_model.velocity.shape)
    cell_indices = [np.minimum(((np.arange(n) + 0.5) * side / data.source_model.cell_size_m).astype(np.intp), source_shape[axis] - 1)
                    for axis, n in enumerate(shape)]
    reference.set_vp_array(data.source_model.velocity[np.ix_(*cell_indices)])
    stations = [tuple(float(value) for value in row) for row in data.station_coordinates_m]
    return PreparedInversion(initial, reference, data.station_ids, data.event_ids,
                             stations, data.arrival_times_s, data.noise_sigmas)


def run_saved_experiment(experiment_id: str, config, experiments_root=DEFAULT_EXPERIMENTS_ROOT,
                         *, validate_only: bool = False):
    """Run EM on saved observations or validate and report the prepared inputs."""
    prepared = prepare_inversion(experiment_id, config, experiments_root)
    relative_sigma, absolute_sigma_s, model_sigma_s = resolve_weight_sigmas(config, prepared.noise_sigmas)
    print(
        f"Experiment {experiment_id}: {len(prepared.event_ids)} events, "
        f"{len(prepared.station_ids)} stations, inversion grid "
        f"{prepared.initial_model.grid.vp.shape}, cell {config.cell_size:g} m",
        flush=True,
    )
    if validate_only:
        return prepared

    model_path = Path(experiments_root) / "input" / experiment_id / "model.npz"
    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    warm_up_jit()
    logger = run_em(
        n_cycles=config.n_cycles,
        initial_model=prepared.initial_model,
        arrivals_table=prepared.arrivals_table,
        station_locs=prepared.station_locs,
        weights_top_n=config.weights_top_n,
        weights_min_distance=config.weights_min_distance,
        candidate_mode=config.candidate_mode,
        temperature=config.temperature,
        weight_noise_relative_sigma=relative_sigma,
        weight_noise_absolute_sigma_s=absolute_sigma_s,
        weight_model_sigma_s=model_sigma_s,
        lambda_reg=config.lambda_reg,
        subdivision=config.subdivision,
        smoothness_reg=config.smoothness_reg,
        initial_gradient_m_s=config.initial_gradient_m_s,
        initial_layer_boundaries_km=config.initial_layer_boundaries_km,
        initial_layer_velocities_m_s=config.initial_layer_velocities_m_s,
        coverage_damping_power=config.coverage_damping_power,
        coverage_floor=config.coverage_floor,
        coverage_reference_percentile=config.coverage_reference_percentile,
        max_velocity_step_fraction=config.max_velocity_step_fraction,
        run_name=f"{config.run_name}_{experiment_id}",
        run_version=config.run_version,
        slowness_interpolation=config.slowness_interpolation,
        v_bounds=config.v_bounds,
        v_reg_strength=config.v_reg_strength,
        v_left_mode=config.v_left_mode,
        v_right_mode=config.v_right_mode,
        v_left_rate=config.v_left_rate,
        v_right_rate=config.v_right_rate,
        v_left_power=config.v_left_power,
        v_right_power=config.v_right_power,
        true_model=prepared.reference_model,
        source_experiment={"id": experiment_id, "model_sha256": digest.hexdigest()},
        # Event reference coordinates are withheld from the inversion; the
        # forward input directory remains the source of truth for future display.
        event_locs=None,
        save_runs=config.save_runs,
        runs_dir=config.runs_dir,
        n_workers=config.n_workers,
        log_G_per_weight=config.log_g_per_weight,
    )
    return logger
