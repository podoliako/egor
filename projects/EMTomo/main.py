import cProfile
import csv
import pstats
from dataclasses import dataclass
from pathlib import Path
from pstats import SortKey

import numpy as np

from instruments.instruments import (
    generate_synthetic_arrivals_table,
    snap_metric_points_to_cell_centers,
)
from tomography.tomography import run_em, warm_up_jit
from velocity_model import VelocityModel


@dataclass(frozen=True)
class ExampleConfig:
    """All parameters of the synthetic tomography example."""

    # Model geometry and geographic reference: 350 x 150 x 70 km.
    cell_size: float = 10_000.0
    grid_shape: tuple[int, int, int] = (35, 15, 7)
    lon: float = 37.6173
    lat: float = 55.7558
    height: float = 50.0
    azimuth: float = 45.0

    # Initial model: homogeneous background; loading a saved model is disabled.
    # initial_model_path: str | None = "runs/run_20260903_195317/iter_17/model.npy"

    # Station layout and true velocity model. Stations form a uniform surface grid;
    # events are distributed uniformly in the model volume.
    station_grid_shape: tuple[int, int] = (7, 5)
    station_locations_csv: str | None = None
    event_locations_csv: str | None = None
    background_vp: float = 5000.0
    checkerboard_anomaly_fraction: float = 0.05
    checkerboard_cell_size: float = 20_000.0
    checkerboard_block_shape: tuple[int, int, int] | None = None
    checkerboard_rotation_degrees: float = 45.0

    # Synthetic arrival generation: events are placed on a uniform volume grid.
    subdivision: int = 3
    event_grid_shape: tuple[int, int, int] = (35, 15, 7)  # 3,675 hypocentres.
    random_seed: int = 7
    event_depth_bias: float = 0.0
    event_z_offset: float = 250.0
    slowness_interpolation: str = "nearest"
    arrival_noise_std: float = 0.01  # Gaussian pick noise: 10 ms per station.
    synthetic_arrivals_cache: str | None = None

    # EM inversion. Change the version for every method release.
    run_name: str = "em"
    run_version: str = "1.1"
    n_cycles: int = 7
    weights_top_n: int = 1
    weights_min_distance: int = 1
    temperature: float = 1
    lambda_reg: float = 0.01
    coverage_damping_power: float = 1
    coverage_floor: float = 0.05
    coverage_reference_percentile: float = 75.0
    max_velocity_step_fraction: float = 0.03
    v_bounds: tuple[float, float] = (4000.0, 6000.0)
    v_reg_strength: float = 0.0
    v_left_mode: str = "lin"
    v_right_mode: str = "lin"
    v_left_rate: float = 0.0
    v_right_rate: float = 0.0
    v_left_power: float = 0.0
    v_right_power: float = 0.0

    # Runtime and output.
    n_workers: int = 25
    save_runs: bool = True
    runs_dir: str = "runs"
    log_g_per_weight: bool = False
    profiling_stats_limit: int = 30


CONFIG = ExampleConfig()


def build_top_surface_stations(
    n_stations_x: int,
    n_stations_y: int,
    model_n_x: int,
    model_n_y: int,
    cell_size: float,
):
    """Return an evenly spaced station grid above the model's top surface."""
    model_width_x = model_n_x * cell_size
    model_width_y = model_n_y * cell_size
    return [
        (
            (i + 0.5) * model_width_x / n_stations_x,
            (j + 0.5) * model_width_y / n_stations_y,
            0.0,
        )
        for i in range(n_stations_x)
        for j in range(n_stations_y)
    ]


def build_uniform_volume_events(
    event_grid_shape: tuple[int, int, int],
    model_shape: tuple[int, int, int],
    cell_size: float,
    depth_bias: float = 0.0,
):
    """Return a regular event grid, optionally concentrated toward the bottom.

    ``depth_bias=0`` gives uniformly spaced depths. Positive values transform
    normalized depth quantiles with exponent ``1 / (1 + depth_bias)``, reducing
    vertical spacing toward the bottom while keeping every event inside the model.
    """
    if min(event_grid_shape) <= 0:
        raise ValueError("event_grid_shape values must be positive")
    if depth_bias < 0.0 or not np.isfinite(depth_bias):
        raise ValueError("depth_bias must be a finite value >= 0")

    event_n_x, event_n_y, event_n_z = event_grid_shape
    model_n_x, model_n_y, model_n_z = model_shape
    model_width_x = model_n_x * cell_size
    model_width_y = model_n_y * cell_size
    model_depth = model_n_z * cell_size
    depth_exponent = 1.0 / (1.0 + depth_bias)
    event_depths = model_depth * (
        (np.arange(event_n_z, dtype=np.float64) + 0.5) / event_n_z
    ) ** depth_exponent

    return [
        (
            (i + 0.5) * model_width_x / event_n_x,
            (j + 0.5) * model_width_y / event_n_y,
            float(event_depths[k]),
        )
        for i in range(event_n_x)
        for j in range(event_n_y)
        for k in range(event_n_z)
    ]


def load_metric_points_csv(filepath: str) -> list[tuple[float, float, float]]:
    """Load EMTomo local metric coordinates from a CSV with x_m, y_m, z_m."""
    path = Path(filepath)
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"x_m", "y_m", "z_m"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required_columns)}")
        points = [
            (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            for row in reader
        ]
    if not points:
        raise ValueError(f"No metric points found in {path}")
    return points


def load_or_generate_synthetic_arrivals(
    config: ExampleConfig,
    forward_true_model: VelocityModel,
    stations_metric: list[tuple[float, float, float]],
    events_metric: list[tuple[float, float, float]],
) -> list[list[float]]:
    """Load a geometry-validated arrival cache or generate and optionally save it."""
    cache_path = (
        Path(config.synthetic_arrivals_cache)
        if config.synthetic_arrivals_cache is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            arrivals = np.asarray(cache["arrivals"], dtype=np.float64)
            cached_stations = np.asarray(cache["stations"], dtype=np.float64)
            cached_events = np.asarray(cache["events"], dtype=np.float64)
        if not np.array_equal(cached_stations, np.asarray(stations_metric)):
            raise ValueError(f"Station geometry does not match arrival cache: {cache_path}")
        if not np.array_equal(cached_events, np.asarray(events_metric)):
            raise ValueError(f"Event geometry does not match arrival cache: {cache_path}")
        print(f"Loaded synthetic arrivals: {cache_path}", flush=True)
        return arrivals.tolist()

    arrivals_table, _ = generate_synthetic_arrivals_table(
        forward_true_model,
        station_locs=stations_metric,
        event_locs=events_metric,
        random_seed=config.random_seed,
        subdivision=1,
        slowness_interpolation=config.slowness_interpolation,
        arrival_noise_std=config.arrival_noise_std,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            arrivals=np.asarray(arrivals_table, dtype=np.float64),
            stations=np.asarray(stations_metric, dtype=np.float64),
            events=np.asarray(events_metric, dtype=np.float64),
        )
        print(f"Saved synthetic arrivals: {cache_path}", flush=True)
    return arrivals_table


def load_initial_vp(model: VelocityModel, filepath: str) -> None:
    """Load a saved coarse-grid Vp array as the inversion starting model."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Initial model file not found: {path}")

    vp = np.load(path, allow_pickle=False)
    expected_shape = model.grid.vp.shape
    if vp.shape != expected_shape:
        raise ValueError(
            f"Initial Vp shape {vp.shape} does not match the configured "
            f"coarse-grid shape {expected_shape}"
        )
    if not np.all(np.isfinite(vp)) or np.any(vp <= 0.0):
        raise ValueError("Initial Vp must contain only finite positive velocities")

    model.set_vp_array(vp)


def build_true_model(model: VelocityModel, config: ExampleConfig) -> None:
    """Create a physical 3D checkerboard rotated around the vertical axis."""
    if config.checkerboard_cell_size <= 0.0:
        raise ValueError("checkerboard_cell_size must be positive")
    if config.checkerboard_block_shape is not None and (
        len(config.checkerboard_block_shape) != 3
        or min(config.checkerboard_block_shape) <= 0
    ):
        raise ValueError(
            "checkerboard_block_shape must contain three positive values"
        )

    n_x, n_y, n_z = model.grid.vp.shape
    cell_size = model.geometry.side_size
    x = (np.arange(n_x, dtype=np.float64) + 0.5) * cell_size
    y = (np.arange(n_y, dtype=np.float64) + 0.5) * cell_size
    z = (np.arange(n_z, dtype=np.float64) + 0.5) * cell_size
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    # Rotate horizontal coordinates about the model centre. Rotation does not
    # affect the vertical checkerboard axis.
    x -= n_x * cell_size / 2.0
    y -= n_y * cell_size / 2.0
    angle = np.deg2rad(config.checkerboard_rotation_degrees)
    rotated_x = np.cos(angle) * x + np.sin(angle) * y
    rotated_y = -np.sin(angle) * x + np.cos(angle) * y
    checker_sizes = (
        tuple(size * config.cell_size for size in config.checkerboard_block_shape)
        if config.checkerboard_block_shape is not None
        else (config.checkerboard_cell_size,) * 3
    )
    checker_index = (
        np.floor(rotated_x / checker_sizes[0]).astype(np.int64)
        + np.floor(rotated_y / checker_sizes[1]).astype(np.int64)
        + np.floor(z / checker_sizes[2]).astype(np.int64)
    )
    anomaly_sign = np.where(checker_index % 2 == 0, 1.0, -1.0)
    model.set_vp_array(
        config.background_vp
        * (1.0 + anomaly_sign * config.checkerboard_anomaly_fraction)
    )


def main(config: ExampleConfig = CONFIG) -> None:
    n_x, n_y, n_z = config.grid_shape
    model_config = {
        "lon": config.lon,
        "lat": config.lat,
        "height": config.height,
        "azimuth": config.azimuth,
        "side_size": config.cell_size,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
    }

    # Snap surface stations to centres of the upper fine-grid cells so their
    # metric coordinates match the cell-centred FMM source locations exactly.
    fine_cell_size = config.cell_size / config.subdivision
    fine_shape: tuple[int, int, int] = (
        n_x * config.subdivision,
        n_y * config.subdivision,
        n_z * config.subdivision,
    )
    raw_stations_metric = (
        load_metric_points_csv(config.station_locations_csv)
        if config.station_locations_csv is not None
        else build_top_surface_stations(
            *config.station_grid_shape,
            n_x,
            n_y,
            config.cell_size,
        )
    )
    stations_metric = snap_metric_points_to_cell_centers(
        raw_stations_metric, fine_cell_size, fine_shape
    )

    initial_model = VelocityModel.from_config(model_config)
    initial_model.fill_linear_gradient(
        "vp", config.background_vp, config.background_vp
    )

    # To resume from a saved coarse-grid model instead, replace the block above with:
    # load_initial_vp(initial_model, "runs/run_20260903_195317/iter_17/model.npy")

    # This coarse reference is used for quality metrics and run visualisation.
    # It samples the same physical pattern as the detailed forward model below.
    true_model = VelocityModel.from_config(model_config)
    build_true_model(true_model, config)

    forward_model_config = {
        **model_config,
        "side_size": fine_cell_size,
        "n_x": fine_shape[0],
        "n_y": fine_shape[1],
        "n_z": fine_shape[2],
    }
    forward_true_model = VelocityModel.from_config(forward_model_config)
    build_true_model(forward_true_model, config)

    events_metric = (
        load_metric_points_csv(config.event_locations_csv)
        if config.event_locations_csv is not None
        else build_uniform_volume_events(
            config.event_grid_shape,
            config.grid_shape,
            config.cell_size,
            depth_bias=config.event_depth_bias,
        )
    )
    arrivals_table = load_or_generate_synthetic_arrivals(
        config,
        forward_true_model,
        stations_metric,
        events_metric,
    )

    warm_up_jit()

    profiler = cProfile.Profile()
    profiler.enable()
    logger = run_em(
        n_cycles=config.n_cycles,
        initial_model=initial_model,
        arrivals_table=arrivals_table,
        station_locs=stations_metric,
        weights_top_n=config.weights_top_n,
        weights_min_distance=config.weights_min_distance,
        temperature=config.temperature,
        lambda_reg=config.lambda_reg,
        subdivision=config.subdivision,
        coverage_damping_power=config.coverage_damping_power,
        coverage_floor=config.coverage_floor,
        coverage_reference_percentile=config.coverage_reference_percentile,
        max_velocity_step_fraction=config.max_velocity_step_fraction,
        run_name=config.run_name,
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
        true_model=true_model,
        true_model_fine=forward_true_model,
        event_locs=events_metric,
        save_runs=config.save_runs,
        runs_dir=config.runs_dir,
        n_workers=config.n_workers,
        log_G_per_weight=config.log_g_per_weight,
    )
    profiler.disable()

    print(f"Run saved: {logger.run_dir}")
    logger.save_profiling(profiler)
    pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(
        config.profiling_stats_limit
    )


if __name__ == "__main__":
    main()
