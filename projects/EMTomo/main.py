import cProfile
import math
import pstats
from pstats import SortKey

from instruments.instruments import (
    generate_synthetic_arrivals_table,
    snap_metric_points_to_cell_centers,
)
from tomography.tomography import run_em, warm_up_jit
from velocity_model import VelocityModel


CELL_SIZE = 500.0
GRID_SHAPE = (9, 9, 9)
STATION_GRID_SHAPE = (8, 8)
SUBDIVISION = 8
N_EVENTS = 250
N_CYCLES = 8
N_WORKERS = 25

BACKGROUND_VP = 100.0
CENTRAL_ANOMALY_FRACTION = 0.08
V_BOUNDS = (80.0, 120.0)
V_REG_STRENGTH = 1


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


def build_true_model(model: VelocityModel) -> None:
    """Create a +8% central anomaly with a smooth spherical taper to zero."""
    model.fill_linear_gradient("vp", BACKGROUND_VP, BACKGROUND_VP)

    shape = model.grid.vp.shape
    center = tuple((size - 1) / 2 for size in shape)
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

                anomaly_fraction = CENTRAL_ANOMALY_FRACTION * 0.5 * (
                    1.0 + math.cos(math.pi * radius / taper_radius)
                )
                model.set_vp(i, j, k, BACKGROUND_VP * (1.0 + anomaly_fraction))


if __name__ == "__main__":
    n_x, n_y, n_z = GRID_SHAPE
    model_config = {
        "lon": 37.6173,
        "lat": 55.7558,
        "height": 50.0,
        "azimuth": 45.0,
        "side_size": CELL_SIZE,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
    }

    # Snap surface stations to centres of the upper fine-grid cells so their
    # metric coordinates match the cell-centred FMM source locations exactly.
    fine_cell_size = CELL_SIZE / SUBDIVISION
    fine_shape = (n_x * SUBDIVISION, n_y * SUBDIVISION, n_z * SUBDIVISION)
    stations_metric = snap_metric_points_to_cell_centers(
        build_top_surface_stations(*STATION_GRID_SHAPE, n_x, n_y, CELL_SIZE),
        fine_cell_size,
        fine_shape,
    )

    initial_model = VelocityModel.from_config(model_config)
    initial_model.fill_linear_gradient("vp", BACKGROUND_VP, BACKGROUND_VP)

    true_model = VelocityModel.from_config(model_config)
    build_true_model(true_model)

    arrivals_table, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        n_events=N_EVENTS,
        random_seed=7,
        subdivision=SUBDIVISION,
        slowness_interpolation="trilinear",
        depth_bias=0.0,
        z_offset=2 * CELL_SIZE,
    )

    warm_up_jit()

    profiler = cProfile.Profile()
    profiler.enable()
    logger = run_em(
        n_cycles=N_CYCLES,
        initial_model=initial_model,
        arrivals_table=arrivals_table,
        station_locs=stations_metric,
        weights_top_n=1,
        weights_min_distance=2,
        temperature=0.02,
        lambda_reg=0.03,
        subdivision=SUBDIVISION,
        slowness_interpolation="trilinear",
        v_bounds=V_BOUNDS,
        v_reg_strength=V_REG_STRENGTH,
        v_left_mode="lin",
        v_right_mode="lin",
        v_left_rate=0.0,
        v_right_rate=0.0,
        v_left_power=0.0,
        v_right_power=0.0,
        true_model=true_model,
        event_locs=events_metric,
        save_runs=True,
        runs_dir="runs",
        n_workers=N_WORKERS,
        log_G_per_weight=False,
    )
    profiler.disable()

    print(f"Run saved: {logger.run_dir}")
    logger.save_profiling(profiler)
    pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)
