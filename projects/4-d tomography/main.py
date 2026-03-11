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
    model_config = {
        'lon': 37.6173,
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,
        'side_size': CELL_SIZE,  # метров на ячейку (cell_size coarse сетки)
        'n_x': 10,
        'n_y': 1,
        'n_z': 10
    }

    # --- Координаты станций в МЕТРАХ (x_m, y_m, z_m) ---
    # Было: (0,1,0), (10,1,0), ... — это были индексы coarse сетки
    # Стало: умножаем на cell_size (100 м), поверхность → z_m=0
    #
    # cell_size coarse = side_size = 100 м (зависит от вашей модели, уточните)
    # Пример: индекс i=10 → x_m = 10 * 100 = 1000 м
  # метров, cell_size coarse сетки при subdivision=1

    n_stations = 10
    n_events = 15
    stations_metric = [ (i*CELL_SIZE*model_config['n_x']/n_stations, 0, 0) for i in range(n_stations)]
    events_metric = [
        (i * CELL_SIZE * model_config['n_x'] / n_events,
        0,
        j * CELL_SIZE * model_config['n_z'] / n_events)
        for i in range(n_events)
        for j in range(n_events)
    ]
    # Конвертируем: (i, j, k) → (i*cell_size, j*cell_size, k*cell_size) метров


    # subdivision для fine сетки при вычислениях
    SUBDIVISION = 10  # можно менять: 1, 2, 3, ...

    initial_model = VelocityModel.from_config(model_config)
    initial_model.fill_linear_gradient('vp', top_value=100.0, bottom_value=100.0)

    true_model = VelocityModel.from_config(model_config)
    # true_model.fill_linear_gradient('vp', top_value=95.0, bottom_value=105.0)

    for i in range(true_model.grid.vp.shape[0]):
        for j in range(true_model.grid.vp.shape[1]):
            for k in range(true_model.grid.vp.shape[2]):
                if (i % 2 == 0 and k % 2 == 0):
                    true_model.set_vp(i, j, k, 110)
                else:
                    true_model.set_vp(i, j, k, 90)

    simple_heatmap(true_model.get_geo_grid().vp[:, 0, :], filename='true_model_3.png')
    simple_heatmap(initial_model.get_geo_grid().vp[:, 0, :], filename='initial_model_3.png')

    # Генерируем синтетику на true_model с нужным subdivision
    full_arr, events_metric = generate_synthetic_arrivals_table(
        true_model,
        station_locs=stations_metric,
        event_locs=events_metric,
        random_seed=7,
        subdivision=SUBDIVISION,
    )

    print("Events (metric):", events_metric)

    X = [e[0] for e in events_metric]
    Y = [e[2] for e in events_metric]
    simple_scatter(X, Y)

    print(full_arr)

    run_em(
        5,
        initial_model,
        full_arr,
        stations_metric,
        weights_top_n=1,
        lambda_reg=99999,
        subdivision=SUBDIVISION,
    )

    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)