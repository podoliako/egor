from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid
from wave_propagation import WavePropagator
from instruments import compute_pairwise_misfit_matrix_and_sse, compute_epicenter_weight_matrix, generate_synthetic_arrivals_table
from raytracing import trace_ray_from_timefield, rasterize_path_binary, rasterize_path_lengths
from math import *
from components.graphics import simple_scatter, simple_heatmap
from tomography import run_tomography_prototype

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

def big_homogenius_media_test(model):
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

def missfit_test(model):
    n_subdivision = 1
    grid = model.get_geo_grid(n_subdivision)

    stations = [
        {'loc':(0,0,0), 'arrival_unix':145}, 
        {'loc':(99,0,0), 'arrival_unix':50},
        {'loc':(199,0,0), 'arrival_unix':50},
        {'loc':(299,0,0), 'arrival_unix':146}]

    miss_matrix = compute_pairwise_misfit_matrix_and_sse(grid, stations, (149,0,25), solver='skfmm')
    return miss_matrix


def weights_test(model, stations):
    n_subdivision = 1
    grid = model.get_geo_grid(n_subdivision)
    
    return compute_epicenter_weight_matrix(grid, stations, solver='skfmm', abs_misfit_threshold=220)


def ray_tracing_G_test(model):
    n_subdivision = 1
    grid = model.get_geo_grid(n_subdivision)

    origin = (0, 0, 0)
    spacing = (1, 1, 1)
    station = (150, 1, 0)
    epic = (50, 1, 25)

    solver = 'skfmm'
    propogator = WavePropagator(solver=solver)
    T = propogator.compute_from_geo_grid(grid, station, 'P')

    path = trace_ray_from_timefield(T, station, epic, origin, spacing)
    print(path)
    # G3 = rasterize_path_binary(path, T.shape, origin, spacing)   # 3D 0/1 matrix
    G3 = rasterize_path_lengths(path, T.shape, voxel_size=spacing)

    return G3

def synthetic_arrivals(model):
    events = [(25,1,25), (35,1,35), (66,1,13), (160,1,45), (290,1,25)]
    stations = [(0,0,0), (50,0,0), (100,0,0), (150,0,0), (200,0,0), (250,0,0)]
    synth_arrivals = generate_synthetic_arrivals_table(model, event_locs=events, station_locs=stations)
    return synth_arrivals

def tomography(initial_model, arrivlas):
    res = run_tomography_prototype(initial_model, arrivlas)
    return res

if __name__ == '__main__':
    modle_config = {
        'lon': 37.6173, 
        'lat': 55.7558,
        'height': 50.0,
        'azimuth': 45.0,
        'side_size': 100.0, 
        'n_x': 300,
        'n_y': 3,
        'n_z': 50
    }

    stations = [
        {'loc':(0,1,0), 'arrival_unix':0}, 
        {'loc':(99,1,0), 'arrival_unix':0},
        {'loc':(199,1,0), 'arrival_unix':100},
        {'loc':(299,1,0), 'arrival_unix':195}]

    initial_model = VelocityModel.from_config(modle_config)
    initial_model.fill_linear_gradient('vp', top_value=100.0, bottom_value=100.0)

    true_model = VelocityModel.from_config(modle_config)
    true_model.fill_linear_gradient('vp', top_value=50.0, bottom_value=150.0)

    arr = synthetic_arrivals(true_model)
    print(arr)

    tm = tomography(initial_model, arr)

    print(tm)
    # res = ray_tracing_G_test(model)
    # print(res)
    # simple_heatmap(res[:,1,:])



