from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
from velocity_model import VelocityModel
from instruments import generate_synthetic_arrivals_table
from math import *
from components.graphics import simple_scatter, simple_heatmap
from tomography import make_tomography_step, run_em, warm_up_jit

import cProfile
import pstats
from pstats import SortKey


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    CELL_SIZE = 500.0
    SUBDIVISION = 13
    model_config = {
        'lon': 37.6173,
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,
        'side_size': CELL_SIZE,
        'n_x': 10,
        'n_y': 3,
        'n_z': 1
    }

    n_stations = 70
    # n_events = 10
    middle_y = (model_config['n_y']) * CELL_SIZE / 2
    middle_z = (model_config['n_z']) * CELL_SIZE / 2
    n_y = model_config['n_y']
    n_x = model_config['n_x']
    # middle_y = CELL_SIZE * 1

    stations_metric = [
        (
            np.random.uniform(CELL_SIZE/2, (n_x)*CELL_SIZE - CELL_SIZE/2), 
            np.random.uniform(CELL_SIZE/2, (n_y)*CELL_SIZE - CELL_SIZE/2),
            0
         )
        for i in range(n_stations)
    ]

    # events_metric = [
    #     (
    #         (i + 1) * CELL_SIZE * model_config['n_x'] / (n_events + 1),
    #         middle_y,
    #         (j + 1) * CELL_SIZE * model_config['n_z'] / (n_events + 1),
    #     )
    #     for i in range(n_events)
    #     for j in range(n_events)
    # ]

    initial_model = VelocityModel.from_config(model_config)
    true_model = VelocityModel.from_config(model_config)

    initial_model.fill_linear_gradient('vp', 100, 100)
    true_model.fill_linear_gradient('vp', 100, 100)    

    for i in range(true_model.grid.vp.shape[0]):
        for j in range(true_model.grid.vp.shape[1]):
            for k in range(true_model.grid.vp.shape[2]):
                # if i >= 4 and i <= 5 and k >= 5 and k <= 6:
                #     true_model.set_vp(i, j, k, 101)
                
                # if j != 1:
                #     true_model.set_vp(i, j, k, 1)
                #     initial_model.set_vp(i, j, k, 1)
                # else:
                #     true_model.set_vp(i, j, k, np.random.normal(100, 0.001))
                #     initial_model.set_vp(i, j, k, 100)
                if (i % 2 == 0 and k % 2 == 0) or (i % 2 != 0 and k % 2 != 0):
                    true_model.set_vp(i, j, k, 100 + np.random.normal(0, 2))
                    initial_model.set_vp(i, j, k, 100)
                else:
                    true_model.set_vp(i, j, k, 100 + np.random.normal(0, 2))
                    initial_model.set_vp(i, j, k, 100)
                # if (i % 2 == 0 and k % 2 == 0) or (i % 2 != 0 and k % 2 != 0):
                #     true_model.set_vp(i, j, k, 98)
                #     initial_model.set_vp(i, j, k, 99)
                # else:
                #     true_model.set_vp(i, j, k, 102)
                #     initial_model.set_vp(i, j, k, 101)


    full_arr, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        n_events=350,
        random_seed=7,
        subdivision=SUBDIVISION,
        depth_bias=3,
        x_offset=CELL_SIZE*1,
        y_offset=0
    )

    warm_up_jit()

    logger = run_em(
        n_cycles=5,
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
        n_workers=2,
    )

    print(f"Run saved: {logger.run_dir}")
    
    profiler.disable()
    logger.save_profiling(profiler)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)