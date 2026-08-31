import cProfile
import pstats
from pstats import SortKey

from instruments.instruments import generate_synthetic_arrivals_table
from tomography.tomography import run_em, warm_up_jit
from velocity_model import VelocityModel


CELL_SIZE = 500.0
GRID_SHAPE = (8, 8, 8)
SUBDIVISION = 15
N_EVENTS = 300
N_CYCLES = 60
N_WORKERS = 20

V_BOUNDS = (50.0, 150.0)
V_REG_STRENGTH = 0.00055


def build_top_cell_center_stations(n_x: int, n_y: int, cell_size: float):
    """Return one station above the center of every cell in the top layer."""
    return [
        ((i + 0.5) * cell_size, (j + 0.5) * cell_size, 0.0)
        for i in range(n_x)
        for j in range(n_y)
    ]


def build_true_model(model: VelocityModel) -> None:
    model.fill_linear_gradient("vp", 100.0, 100.0)

    for i in (2, 3, 4):
        for j in range(model.grid.vp.shape[1]):
            for k in (2, 3):
                velocity = 102.0 if j in (0, 2) else 110.0 if i == 3 else 105.0
                model.set_vp(i, j, k, velocity)


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

    # 8 × 8 stations: one above the center of each top-layer cell.
    stations_metric = build_top_cell_center_stations(n_x, n_y, CELL_SIZE)

    initial_model = VelocityModel.from_config(model_config)
    initial_model.fill_linear_gradient("vp", 100.0, 100.0)

    true_model = VelocityModel.from_config(model_config)
    build_true_model(true_model)

    arrivals_table, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        n_events=N_EVENTS,
        random_seed=7,
        subdivision=SUBDIVISION,
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
        temperature=0.001,
        lambda_reg=0.0005,
        subdivision=SUBDIVISION,
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
