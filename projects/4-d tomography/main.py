from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid
from wave_propagation import WavePropagator
from math import *
from components.graphics import simple_scatter

def quick_test(model):
    print(f"Vp at surface (0,0,0): {model.get_vp(0, 0, 0):.1f} m/s")
    print(f"Vp at depth (0,0,15): {model.get_vp(0, 0, 15):.1f} m/s")
    print(f"Vp at depth (0,0,29): {model.get_vp(0, 0, 29):.1f} m/s")
    
    print(f"Vs at surface (0,0,0): {model.get_vs(0, 0, 0):.1f} m/s")
    print(f"Vs at depth (0,0,15): {model.get_vs(0, 0, 15):.1f} m/s")
    print(f"Vs at depth (0,0,29): {model.get_vs(0, 0, 29):.1f} m/s")

def demo():
    config = {
        'lon': 37.6173,  # Москва, например
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,  # 45 градусов от севера
        'side_size': 100.0,  # 100 метров на сторону
        'n_x': 10,
        'n_y': 10,
        'n_z': 10
    }

    model = VelocityModel.from_config(config)

    # Заполняем градиентом по глубине
    model.fill_linear_gradient('vp', top_value=100.0, bottom_value=100.0)
    model.fill_linear_gradient('vs', top_value=1000.0, bottom_value=3000.0)

    n_subdivision = 2
    grid = model.get_geo_grid(n_subdivision)

    propogator = WavePropagator(solver='skfmm')

    times = propogator.compute_from_geo_grid(grid, (0, 0, 0), 'P')
    print(times)

    est_time = times[-1,-1,-1]
    theoretical_time = (3**0.5)*(10*n_subdivision-1)/n_subdivision
    print(f'Subdivision: {n_subdivision}')
    print(f"Geo grid shape: {grid.shape}")
    print(f'Estimated time: {est_time}')
    print(f'Time from corner to corner: {theoretical_time}')
    print(f'Error: {round(float(100*abs(theoretical_time-est_time)/theoretical_time), 3)}%')

def calculate_dist_in_grid(grid:GridGeometry, target_point, start_point=(0,0,0)):
    t = target_point
    s = start_point
    dist = grid.cell_size * sqrt(((t[0] - s[0])**2) + ((t[1] - s[1])**2) + ((t[2] - s[2])**2))
    dist = round(dist, 2)
    return dist

def calculate_dist_matrix(grid:GridGeometry):
    shape = grid.shape
    dists = np.empty(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dists[i][j][k] = calculate_dist_in_grid(grid, (i, j, k))

    return dists

def homogenius_media_test(grid, velocity, times_exp):
    dist_martix = calculate_dist_matrix(grid)
    times_theory = dist_martix / velocity
    err_matrix = np.round(100 * abs(times_theory - times_exp)/times_theory, 3)

    dtype = [('dist', float), ('err', float)]
    structured_arr = np.empty(dist_martix.shape, dtype=dtype)

    structured_arr['dist'] = dist_martix
    structured_arr['err'] = err_matrix

    return structured_arr.flatten().flatten()




if __name__ == '__main__':
    config = {
        'lon': 37.6173, 
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,
        'side_size': 100.0, 
        'n_x': 300,
        'n_y': 1,
        'n_z': 50
    }

    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', top_value=100.0, bottom_value=100.0)
    
    n_subdivision = 5
    grid = model.get_geo_grid(n_subdivision)
    
    solver = 'skfmm'
    propogator = WavePropagator(solver=solver)
    times_exp = propogator.compute_from_geo_grid(grid, (0, 0, 0), 'P')

    res = homogenius_media_test(grid, 100, times_exp)

    print(res)

    dist_threshold = 5000
    x = [item[0] if item[0] > dist_threshold else np.nan for item in res]
    y = [item[1] if item[0] > dist_threshold else np.nan for item in res]
    simple_scatter(x, y, s=0.1, 
                   x_label='distance from (0,0,0), m', 
                   y_label='err, %', 
                   title=f'Solver: {solver}, Grid shape: {str(grid.shape)}', 
                   dpi=700)



