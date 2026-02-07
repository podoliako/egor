import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid
from wave_propagation import WavePropagator



def quick_test(model):
    print(f"Vp at surface (0,0,0): {model.get_vp(0, 0, 0):.1f} m/s")
    print(f"Vp at depth (0,0,15): {model.get_vp(0, 0, 15):.1f} m/s")
    print(f"Vp at depth (0,0,29): {model.get_vp(0, 0, 29):.1f} m/s")
    
    print(f"Vs at surface (0,0,0): {model.get_vs(0, 0, 0):.1f} m/s")
    print(f"Vs at depth (0,0,15): {model.get_vs(0, 0, 15):.1f} m/s")
    print(f"Vs at depth (0,0,29): {model.get_vs(0, 0, 29):.1f} m/s")


if __name__ == '__main__':
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


    grid = model.get_geo_grid(6)
    print(grid.cell_size)

    propogator = WavePropagator(solver='skfmm')

    times = propogator.compute_from_geo_grid(grid, (0, 0, 0), 'P')
    print(times)


