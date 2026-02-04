"""
Simple tests for velocity model.
Run with: python -m pytest test_velocity_model.py
or just: python test_velocity_model.py
"""
import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid


def test_grid_creation():
    """Test basic grid creation."""
    grid = VelocityGrid((10, 20, 30))
    assert grid.shape == (10, 20, 30)
    assert grid.vp.shape == (10, 20, 30)
    assert grid.vs.shape == (10, 20, 30)
    print("✓ Grid creation test passed")


def test_set_get_values():
    """Test setting and getting individual values."""
    grid = VelocityGrid((5, 5, 5))
    
    grid.set_vp(2, 3, 4, 5000.0)
    assert grid.get_vp(2, 3, 4) == 5000.0
    
    grid.set_vs(1, 2, 3, 3000.0)
    assert grid.get_vs(1, 2, 3) == 3000.0
    
    print("✓ Set/get values test passed")


def test_linear_gradient():
    """Test linear gradient filling."""
    grid = VelocityGrid((10, 10, 10))
    grid.fill_linear_gradient('vp', 1000.0, 5000.0)
    
    # Check top and bottom
    assert np.allclose(grid.vp[:, :, 0], 1000.0)
    assert np.allclose(grid.vp[:, :, 9], 5000.0)
    
    # Check middle is approximately average
    assert np.allclose(grid.vp[:, :, 5], 3000.0, atol=500)
    
    print("✓ Linear gradient test passed")


def test_array_operations():
    """Test bulk array operations."""
    grid = VelocityGrid((3, 3, 3))
    
    vp_values = np.ones((3, 3, 3)) * 4000.0
    grid.set_vp_array(vp_values)
    
    assert np.allclose(grid.vp, 4000.0)
    print("✓ Array operations test passed")


def test_geometry():
    """Test geometry class."""
    geom = GridGeometry(
        lon=30.0, lat=60.0, height=100.0,
        azimuth=90.0, side_size=50.0,
        n_x=10, n_y=20, n_z=30
    )
    
    config = geom.to_dict()
    assert config['lon'] == 30.0
    assert config['n_x'] == 10
    
    geom2 = GridGeometry.from_dict(config)
    assert geom2.lon == 30.0
    assert geom2.azimuth == 90.0
    
    print("✓ Geometry test passed")


def test_full_model():
    """Test complete model workflow."""
    config = {
        'lon': 0.0, 'lat': 0.0, 'height': 0.0,
        'azimuth': 0.0, 'side_size': 100.0,
        'n_x': 5, 'n_y': 5, 'n_z': 5
    }
    
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', 2000.0, 6000.0)
    
    # Test convenience methods
    model.set_vp(2, 2, 2, 5000.0)
    assert model.get_vp(2, 2, 2) == 5000.0
    
    print("✓ Full model test passed")


def test_save_load():
    """Test JSON serialization."""
    import os
    
    config = {
        'lon': 10.0, 'lat': 20.0, 'height': 0.0,
        'azimuth': 45.0, 'side_size': 100.0,
        'n_x': 3, 'n_y': 3, 'n_z': 3
    }
    
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', 1000.0, 3000.0)
    model.set_vs(1, 1, 1, 1500.0)
    
    # Save and load
    filepath = '/tmp/test_model.json'
    model.to_json(filepath, include_data=True)
    
    loaded = VelocityModel.from_json(filepath)
    
    assert loaded.geometry.lon == 10.0
    assert loaded.geometry.azimuth == 45.0
    assert loaded.get_vs(1, 1, 1) == 1500.0
    assert np.allclose(loaded.grid.vp[0, 0, 0], 1000.0)
    
    os.remove(filepath)
    print("✓ Save/load test passed")


def test_geo_grid_no_subdivision():
    """Test geo grid with subdivision=1 (1:1 mapping)."""
    config = {
        'lon': 0.0, 'lat': 0.0, 'height': 0.0,
        'azimuth': 0.0, 'side_size': 100.0,
        'n_x': 5, 'n_y': 5, 'n_z': 5
    }
    
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', 2000.0, 4000.0)
    
    geo = model.get_geo_grid(subdivision=1)
    
    assert geo.shape == (5, 5, 5)
    assert geo.cell_size == 100.0
    assert geo.subdivision == 1
    assert np.allclose(geo.vp, model.grid.vp)
    
    print("✓ Geo grid no subdivision test passed")


def test_geo_grid_subdivision():
    """Test geo grid with subdivision."""
    config = {
        'lon': 0.0, 'lat': 0.0, 'height': 0.0,
        'azimuth': 0.0, 'side_size': 100.0,
        'n_x': 2, 'n_y': 2, 'n_z': 2
    }
    
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', 1000.0, 3000.0)
    
    geo = model.get_geo_grid(subdivision=2)
    
    # Check dimensions
    assert geo.shape == (4, 4, 4)
    assert geo.cell_size == 50.0
    assert geo.subdivision == 2
    
    # Check corners match original values
    assert np.isclose(geo.vp[0, 0, 0], model.grid.vp[0, 0, 0])
    
    print("✓ Geo grid subdivision test passed")


def test_geo_grid_interpolation():
    """Test different interpolation methods."""
    config = {
        'lon': 0.0, 'lat': 0.0, 'height': 0.0,
        'azimuth': 0.0, 'side_size': 100.0,
        'n_x': 3, 'n_y': 3, 'n_z': 3
    }
    
    model = VelocityModel.from_config(config)
    model.grid.vp[:] = 1000.0
    model.grid.vp[1, 1, 1] = 5000.0  # One high value
    
    # Trilinear should smooth
    geo_tri = model.get_geo_grid(subdivision=2, interpolation='trilinear')
    
    # Nearest should be blocky
    geo_near = model.get_geo_grid(subdivision=2, interpolation='nearest')
    
    # Values should differ
    assert not np.allclose(geo_tri.vp, geo_near.vp)
    
    print("✓ Geo grid interpolation test passed")


def test_custom_interpolation():
    """Test custom interpolation function."""
    config = {
        'lon': 0.0, 'lat': 0.0, 'height': 0.0,
        'azimuth': 0.0, 'side_size': 100.0,
        'n_x': 2, 'n_y': 2, 'n_z': 2
    }
    
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient('vp', 1000.0, 2000.0)
    
    def constant_interp(values, i, j, k, di, dj, dk):
        """Always return 9999."""
        return 9999.0
    
    geo = model.get_geo_grid(subdivision=2, interpolation=constant_interp)
    
    assert np.allclose(geo.vp, 9999.0)
    
    print("✓ Custom interpolation test passed")


if __name__ == '__main__':
    print("Running velocity model tests...\n")
    test_grid_creation()
    test_set_get_values()
    test_linear_gradient()
    test_array_operations()
    test_geometry()
    test_full_model()
    test_save_load()
    test_geo_grid_no_subdivision()
    test_geo_grid_subdivision()
    test_geo_grid_interpolation()
    test_custom_interpolation()
    print("\n✅ All tests passed!")