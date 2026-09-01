import cProfile
import math
import pstats
from dataclasses import dataclass
from pstats import SortKey

from instruments.instruments import (
    generate_synthetic_arrivals_table,
    snap_metric_points_to_cell_centers,
)
from tomography.tomography import run_em, warm_up_jit
from velocity_model import VelocityModel


@dataclass(frozen=True)
class ExampleConfig:
    """All parameters of the synthetic tomography example."""

    # Model geometry and geographic reference.
    cell_size: float = 500.0
    grid_shape: tuple[int, int, int] = (9, 9, 5)
    lon: float = 37.6173
    lat: float = 55.7558
    height: float = 50.0
    azimuth: float = 45.0

    # Station layout and true velocity model.
    station_grid_shape: tuple[int, int] = (8, 8)
    background_vp: float = 100.0
    central_anomaly_fraction: float = 0.08

    # Synthetic arrival generation.
    subdivision: int = 8
    n_events: int = 250
    random_seed: int = 7
    event_depth_bias: float = 0.0
    event_z_offset: float = 250.0
    slowness_interpolation: str = "nearest"

    # EM inversion.
    n_cycles: int = 8
    weights_top_n: int = 10
    weights_min_distance: int = 2
    temperature: float = 0.5
    lambda_reg: float = 0.03
    coverage_damping_power: float = 1.3
    coverage_floor: float = 0.05
    coverage_reference_percentile: float = 75.0
    max_velocity_step_fraction: float = 0.03
    v_bounds: tuple[float, float] = (80.0, 120.0)
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


def build_true_model(model: VelocityModel, config: ExampleConfig) -> None:
    """Create a smooth spherical central anomaly in the coarse-cell grid."""
    model.fill_linear_gradient("vp", config.background_vp, config.background_vp)

    shape = model.grid.vp.shape
    center = tuple((size - 1) / 2.0 for size in shape)
    taper_radius = min(center)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                radius = math.sqrt(
                    (i - center[0]) ** 2
                    + (j - center[1]) ** 2
                    + (k - center[2]) ** 2
                )
                if radius >= taper_radius:
                    continue

                anomaly_fraction = config.central_anomaly_fraction * 0.5 * (
                    1.0 + math.cos(math.pi * radius / taper_radius)
                )
                model.set_vp(
                    i,
                    j,
                    k,
                    config.background_vp * (1.0 + anomaly_fraction),
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
    fine_shape = tuple(size * config.subdivision for size in config.grid_shape)
    stations_metric = snap_metric_points_to_cell_centers(
        build_top_surface_stations(
            *config.station_grid_shape,
            n_x,
            n_y,
            config.cell_size,
        ),
        fine_cell_size,
        fine_shape,
    )

    initial_model = VelocityModel.from_config(model_config)
    initial_model.fill_linear_gradient(
        "vp", config.background_vp, config.background_vp
    )

    true_model = VelocityModel.from_config(model_config)
    build_true_model(true_model, config)

    arrivals_table, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        n_events=config.n_events,
        random_seed=config.random_seed,
        subdivision=config.subdivision,
        slowness_interpolation=config.slowness_interpolation,
        depth_bias=config.event_depth_bias,
        z_offset=config.event_z_offset,
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
