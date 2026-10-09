"""EMTomo inversion setup for a saved, ID-validated forward experiment.

Observed arrivals and station positions come from the experiment. The source
velocity model is used only for quality metrics, never to initialize the
inversion. True event positions are not passed to the inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import InversionConfig
from experiment_data import DEFAULT_EXPERIMENTS_ROOT, _sha256, load_tomography_experiment
from instruments.likelihood import PickNoise
from tomography.tomography_em import run_em, warm_up_jit
from velocity_model import VelocityModel, block_average_slowness


@dataclass(frozen=True)
class PreparedInversion:
    initial_model: VelocityModel
    reference_model: VelocityModel
    station_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    station_locs: np.ndarray
    arrivals_table: np.ndarray
    noise_sigmas: tuple[float, float]


def resolve_pick_noise(config: InversionConfig, noise_sigmas: tuple[float, float]) -> PickNoise:
    """Explicit config overrides win over the experiment's observation-noise metadata."""
    relative, absolute = noise_sigmas
    if config.weight_noise_relative_sigma is not None:
        relative = config.weight_noise_relative_sigma
    if config.weight_noise_absolute_sigma_s is not None:
        absolute = config.weight_noise_absolute_sigma_s
    return PickNoise(float(relative), float(absolute), float(config.weight_model_sigma_s))


def initial_velocity(config: InversionConfig, shape: tuple[int, int, int]) -> np.ndarray:
    """Homogeneous, linear-gradient or horizontally layered starting velocities at cell centres."""
    depth_m = (np.arange(shape[2], dtype=np.float64) + 0.5) * config.cell_size
    bottom_m = shape[2] * config.cell_size
    boundaries = config.initial_layer_boundaries_km
    velocities = config.initial_layer_velocities_m_s
    if boundaries is not None or velocities is not None:
        if boundaries is None or velocities is None:
            raise ValueError("initial layers require both boundaries and velocities")
        boundaries = np.asarray(boundaries, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if (boundaries.ndim != 1 or len(boundaries) < 1 or not np.all(np.isfinite(boundaries))
                or boundaries[0] <= 0 or boundaries[-1] >= bottom_m / 1000
                or np.any(np.diff(boundaries) <= 0)):
            raise ValueError("initial_layer_boundaries_km must increase strictly inside the depth range")
        if (velocities.ndim != 1 or len(velocities) != len(boundaries) + 1
                or not np.all(np.isfinite(velocities)) or np.any(velocities <= 0)):
            raise ValueError("initial_layer_velocities_m_s must contain one finite positive speed per layer")
        profile = velocities[np.searchsorted(boundaries, depth_m / 1000, side="right")]
    elif config.initial_gradient_m_s is not None:
        endpoints = np.asarray(config.initial_gradient_m_s, dtype=np.float64)
        if endpoints.shape != (2,) or not np.all(np.isfinite(endpoints)) or np.any(endpoints <= 0):
            raise ValueError("initial_gradient_m_s must contain two finite positive velocities")
        profile = endpoints[0] + (endpoints[1] - endpoints[0]) * depth_m / bottom_m
    else:
        profile = np.full(shape[2], config.initial_velocity_m_s)
    return np.broadcast_to(profile, shape)


def prepare_inversion(experiment_id: str, config: InversionConfig,
                      experiments_root=DEFAULT_EXPERIMENTS_ROOT) -> PreparedInversion:
    """Validate observations and match the inversion grid to the experiment domain."""
    data = load_tomography_experiment(experiment_id, experiments_root)
    source = data.source_model
    extent = np.asarray(source.velocity.shape) * source.cell_size_m
    ratio = extent / config.cell_size
    shape = tuple(int(n) for n in np.rint(ratio))
    if min(shape) < 1 or not np.allclose(shape, ratio, rtol=1e-12, atol=0):
        raise ValueError(f"Inversion cell_size={config.cell_size:g} m must exactly tile "
                         f"experiment dimensions {tuple(extent)} m")
    initial = VelocityModel(initial_velocity(config, shape), config.cell_size)
    # Truth for metrics: the slowness average of the source model over each inversion cell.
    reference = VelocityModel(
        block_average_slowness(source.velocity, source.cell_size_m, shape, config.cell_size),
        config.cell_size,
    )
    return PreparedInversion(initial, reference, data.station_ids, data.event_ids,
                             np.asarray(data.station_coordinates_m, dtype=np.float64),
                             data.arrival_times_s, data.noise_sigmas)


def run_saved_experiment(experiment_id: str, config: InversionConfig,
                         experiments_root=DEFAULT_EXPERIMENTS_ROOT, *, validate_only: bool = False):
    """Run EM on saved observations, or only validate and return the prepared inputs."""
    prepared = prepare_inversion(experiment_id, config, experiments_root)
    noise = resolve_pick_noise(config, prepared.noise_sigmas)
    print(
        f"Experiment {experiment_id}: {len(prepared.event_ids)} events, "
        f"{len(prepared.station_ids)} stations, inversion grid "
        f"{prepared.initial_model.shape}, cell {config.cell_size:g} m",
        flush=True,
    )
    if validate_only:
        return prepared

    model_sha256 = _sha256(Path(experiments_root) / "input" / experiment_id / "model.npz")
    warm_up_jit()
    return run_em(
        config,
        prepared.initial_model,
        prepared.arrivals_table,
        prepared.station_locs,
        noise,
        reference_model=prepared.reference_model,
        source_experiment={"id": experiment_id, "model_sha256": model_sha256},
        run_name=f"{config.run_name}_{experiment_id}",
    )
