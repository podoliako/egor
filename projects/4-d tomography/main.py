from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid
from wave_propagation import WavePropagator
from instruments import compute_epicenter_weight_matrix, generate_synthetic_arrivals_table
from raytracing import trace_ray_from_timefield, rasterize_path_binary, rasterize_path_lengths
from math import *
from components.graphics import simple_scatter, simple_heatmap
from tomography import make_tomography_step, run_em

import cProfile
import pstats
from pstats import SortKey


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    CELL_SIZE = 500.0
    SUBDIVISION = 7
    model_config = {
        'lon': 37.6173,
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,
        'side_size': CELL_SIZE,
        'n_x': 10,
        'n_y': 3,
        'n_z': 10
    }

    n_stations = 15
    n_events = 17
    stations_metric = [
        (i * CELL_SIZE * model_config['n_x'] / n_stations, CELL_SIZE, 0)
        for i in range(n_stations)
    ]
    events_metric = [
        (
            (i + 1) * CELL_SIZE * model_config['n_x'] / (n_events + 1),
            CELL_SIZE,
            (j + 1) * CELL_SIZE * model_config['n_z'] / (n_events + 1),
        )
        for i in range(n_events)
        for j in range(n_events)
    ]

    initial_model = VelocityModel.from_config(model_config)
    true_model = VelocityModel.from_config(model_config)

    for i in range(true_model.grid.vp.shape[0]):
        for j in range(true_model.grid.vp.shape[1]):
            for k in range(true_model.grid.vp.shape[2]):
                if j != 1:
                    true_model.set_vp(i, j, k, 1)
                    initial_model.set_vp(i, j, k, 1)
                else:
                    true_model.set_vp(i, j, k, np.random.normal(100, 0.3))
                    initial_model.set_vp(i, j, k, 100)
                # elif (i % 2 == 0 and k % 2 == 0) or (i % 2 != 0 and k % 2 != 0):
                #     true_model.set_vp(i, j, k, 90)
                #     initial_model.set_vp(i, j, k, 9)
                # else:
                #     true_model.set_vp(i, j, k, 100.1)

    simple_heatmap(true_model.get_geo_grid().vp[:, 0, :], filename='true_model_3.png')
    simple_heatmap(initial_model.get_geo_grid().vp[:, 0, :], filename='initial_model_3.png')

    full_arr, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        event_locs=events_metric,
        random_seed=7,
        subdivision=SUBDIVISION,
    )

    # print("Events (metric):", events_metric)

    X = [e[0] for e in events_metric]
    Y = [e[2] for e in events_metric]
    simple_scatter(X, Y)

    # print(full_arr)

    logger = run_em(
        n_cycles=10,
        initial_model=initial_model,
        arrivals_table=full_arr,
        station_locs=stations_metric,
        weights_top_n=1,
        lambda_reg=0.001,
        subdivision=SUBDIVISION,
        # --- сохранение ---
        true_model=true_model,
        event_locs=events_metric,
        save_runs=True,
        runs_dir="runs",
    )

    print(f"Run saved: {logger.run_dir}")

    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)