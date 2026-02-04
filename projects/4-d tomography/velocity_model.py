"""
Seismic velocity model with geographic grid reference.
"""
import numpy as np
import json
from typing import Dict, Tuple, Optional


class GridGeometry:
    """
    Geometry and spatial reference of the velocity grid.
    
    Parameters:
    -----------
    lon : float
        Longitude of reference cube center (degrees)
    lat : float
        Latitude of reference cube center (degrees)
    height : float
        Height relative to ground level of reference cube center (meters)
    azimuth : float
        Azimuth of grid orientation (degrees, clockwise from North)
    side_size : float
        Cube side size in meters
    n_x, n_y, n_z : int
        Number of cubes along each axis (z goes downward)
    """
    
    def __init__(self, lon: float, lat: float, height: float, 
                 azimuth: float, side_size: float,
                 n_x: int, n_y: int, n_z: int):
        self.lon = lon
        self.lat = lat
        self.height = height
        self.azimuth = azimuth
        self.side_size = side_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        
    def to_dict(self) -> Dict:
        """Export geometry to dictionary."""
        return {
            'lon': self.lon,
            'lat': self.lat,
            'height': self.height,
            'azimuth': self.azimuth,
            'side_size': self.side_size,
            'n_x': self.n_x,
            'n_y': self.n_y,
            'n_z': self.n_z
        }
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'GridGeometry':
        """Create geometry from dictionary."""
        return cls(
            lon=config['lon'],
            lat=config['lat'],
            height=config['height'],
            azimuth=config['azimuth'],
            side_size=config['side_size'],
            n_x=config['n_x'],
            n_y=config['n_y'],
            n_z=config['n_z']
        )


class VelocityGrid:
    """
    Storage for velocity parameters.
    
    Parameters are stored as separate numpy arrays for efficiency.
    Indexing: grid[i, j, k] where k increases with depth.
    """
    
    def __init__(self, shape: Tuple[int, int, int]):
        """
        Initialize empty velocity grid.
        
        Parameters:
        -----------
        shape : tuple of (n_x, n_y, n_z)
        """
        self.shape = shape
        self.vp = np.zeros(shape, dtype=np.float32)
        self.vs = np.zeros(shape, dtype=np.float32)
    
    def set_vp(self, i: int, j: int, k: int, value: float):
        """Set P-wave velocity at grid point (i, j, k)."""
        self.vp[i, j, k] = value
    
    def get_vp(self, i: int, j: int, k: int) -> float:
        """Get P-wave velocity at grid point (i, j, k)."""
        return self.vp[i, j, k]
    
    def set_vs(self, i: int, j: int, k: int, value: float):
        """Set S-wave velocity at grid point (i, j, k)."""
        self.vs[i, j, k] = value
    
    def get_vs(self, i: int, j: int, k: int) -> float:
        """Get S-wave velocity at grid point (i, j, k)."""
        return self.vs[i, j, k]
    
    def set_vp_array(self, values: np.ndarray):
        """Set all Vp values at once. Array must match grid shape."""
        if values.shape != self.shape:
            raise ValueError(f"Array shape {values.shape} doesn't match grid shape {self.shape}")
        self.vp = values.astype(np.float32)
    
    def set_vs_array(self, values: np.ndarray):
        """Set all Vs values at once. Array must match grid shape."""
        if values.shape != self.shape:
            raise ValueError(f"Array shape {values.shape} doesn't match grid shape {self.shape}")
        self.vs = values.astype(np.float32)
    
    def fill_linear_gradient(self, param: str, top_value: float, bottom_value: float):
        """
        Fill parameter with linear gradient from top (k=0) to bottom (k=n_z-1).
        
        Parameters:
        -----------
        param : str
            'vp' or 'vs'
        top_value : float
            Value at surface (k=0)
        bottom_value : float
            Value at bottom (k=n_z-1)
        """
        if param not in ['vp', 'vs']:
            raise ValueError("param must be 'vp' or 'vs'")
        
        n_z = self.shape[2]
        gradient = np.linspace(top_value, bottom_value, n_z)
        
        # Broadcast gradient along k axis
        values = np.zeros(self.shape, dtype=np.float32)
        for k in range(n_z):
            values[:, :, k] = gradient[k]
        
        if param == 'vp':
            self.vp = values
        else:
            self.vs = values
    
    def to_dict(self) -> Dict:
        """Export velocity data to dictionary (as lists for JSON serialization)."""
        return {
            'vp': self.vp.tolist(),
            'vs': self.vs.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VelocityGrid':
        """Create velocity grid from dictionary."""
        vp_array = np.array(data['vp'], dtype=np.float32)
        shape = vp_array.shape
        grid = cls(shape)
        grid.vp = vp_array
        grid.vs = np.array(data['vs'], dtype=np.float32)
        return grid


class VelocityModel:
    """
    Complete velocity model combining geometry and velocity data.
    """
    
    def __init__(self, geometry: GridGeometry, grid: Optional[VelocityGrid] = None):
        """
        Initialize velocity model.
        
        Parameters:
        -----------
        geometry : GridGeometry
            Spatial reference and grid dimensions
        grid : VelocityGrid, optional
            Velocity data. If None, creates empty grid.
        """
        self.geometry = geometry
        if grid is None:
            shape = (geometry.n_x, geometry.n_y, geometry.n_z)
            self.grid = VelocityGrid(shape)
        else:
            self.grid = grid
    
    @classmethod
    def from_config(cls, config: Dict) -> 'VelocityModel':
        """
        Create model from configuration dictionary.
        
        Parameters:
        -----------
        config : dict
            Must contain geometry parameters and optionally 'data' with vp/vs arrays
        """
        geometry = GridGeometry.from_dict(config)
        
        if 'data' in config:
            grid = VelocityGrid.from_dict(config['data'])
        else:
            grid = None
        
        return cls(geometry, grid)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'VelocityModel':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_config(config)
    
    def to_json(self, filepath: str, include_data: bool = True):
        """
        Save model to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        include_data : bool
            If True, includes velocity arrays. If False, only saves geometry.
        """
        config = self.geometry.to_dict()
        
        if include_data:
            config['data'] = self.grid.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Convenience methods that delegate to grid
    def set_vp(self, i: int, j: int, k: int, value: float):
        """Set P-wave velocity at grid point (i, j, k)."""
        self.grid.set_vp(i, j, k, value)
    
    def get_vp(self, i: int, j: int, k: int) -> float:
        """Get P-wave velocity at grid point (i, j, k)."""
        return self.grid.get_vp(i, j, k)
    
    def set_vs(self, i: int, j: int, k: int, value: float):
        """Set S-wave velocity at grid point (i, j, k)."""
        self.grid.set_vs(i, j, k, value)
    
    def get_vs(self, i: int, j: int, k: int) -> float:
        """Get S-wave velocity at grid point (i, j, k)."""
        return self.grid.get_vs(i, j, k)
    
    def set_vp_array(self, values: np.ndarray):
        """Set all Vp values at once."""
        self.grid.set_vp_array(values)
    
    def set_vs_array(self, values: np.ndarray):
        """Set all Vs values at once."""
        self.grid.set_vs_array(values)
    
    def fill_linear_gradient(self, param: str, top_value: float, bottom_value: float):
        """Fill parameter with linear gradient in depth."""
        self.grid.fill_linear_gradient(param, top_value, bottom_value)
    
    def __repr__(self) -> str:
        return (f"VelocityModel(grid_size=({self.geometry.n_x}, {self.geometry.n_y}, "
                f"{self.geometry.n_z}), cell_size={self.geometry.side_size}m, "
                f"center=({self.geometry.lon:.4f}, {self.geometry.lat:.4f}))")
