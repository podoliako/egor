"""
Seismic velocity model with geographic grid reference.
"""
import numpy as np
import json
from typing import Dict, Tuple, Optional, Callable, Union


class GeoGrid:
    """
    Refined geometric grid for raytracing.
    
    This is generated from VelocityModel by subdividing cells and interpolating.
    Each cell in the velocity model is split into subdivision^3 geo cells.
    
    Attributes:
    -----------
    shape : tuple
        Shape of the geo grid (n_x * subdivision, n_y * subdivision, n_z * subdivision)
    cell_size : float
        Size of each geo cell in meters
    vp : np.ndarray
        P-wave velocities at geo grid resolution
    vs : np.ndarray
        S-wave velocities at geo grid resolution
    subdivision : int
        Subdivision factor used to generate this grid
    """
    
    def __init__(self, shape: Tuple[int, int, int], cell_size: float, subdivision: int):
        self.shape = shape
        self.cell_size = cell_size
        self.subdivision = subdivision
        self.vp = np.zeros(shape, dtype=np.float32)
        self.vs = np.zeros(shape, dtype=np.float32)
    
    def __repr__(self) -> str:
        return (f"GeoGrid(shape={self.shape}, cell_size={self.cell_size:.2f}m, "
                f"subdivision={self.subdivision})")


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


# Interpolation strategies
def trilinear_interpolation(values: np.ndarray, i: int, j: int, k: int, 
                            di: float, dj: float, dk: float) -> float:
    """
    Trilinear interpolation within a cell.
    
    Parameters:
    -----------
    values : np.ndarray
        3D array of values (e.g., vp or vs)
    i, j, k : int
        Base cell indices
    di, dj, dk : float
        Fractional position within cell [0..1]
    
    Returns:
    --------
    Interpolated value
    """
    n_x, n_y, n_z = values.shape
    
    # Get the 8 corner values
    # Handle boundary conditions
    i1 = min(i + 1, n_x - 1)
    j1 = min(j + 1, n_y - 1)
    k1 = min(k + 1, n_z - 1)
    
    # 8 corners of the cube
    c000 = values[i, j, k]
    c100 = values[i1, j, k]
    c010 = values[i, j1, k]
    c110 = values[i1, j1, k]
    c001 = values[i, j, k1]
    c101 = values[i1, j, k1]
    c011 = values[i, j1, k1]
    c111 = values[i1, j1, k1]
    
    # Interpolate along x
    c00 = c000 * (1 - di) + c100 * di
    c01 = c001 * (1 - di) + c101 * di
    c10 = c010 * (1 - di) + c110 * di
    c11 = c011 * (1 - di) + c111 * di
    
    # Interpolate along y
    c0 = c00 * (1 - dj) + c10 * dj
    c1 = c01 * (1 - dj) + c11 * dj
    
    # Interpolate along z
    return c0 * (1 - dk) + c1 * dk


def nearest_neighbor_interpolation(values: np.ndarray, i: int, j: int, k: int,
                                   di: float, dj: float, dk: float) -> float:
    """
    Nearest neighbor interpolation (no interpolation, just repeat values).
    
    Parameters: same as trilinear_interpolation
    """
    return values[i, j, k]


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
    
    def get_geo_grid(self, subdivision: int = 1, 
                     interpolation: Union[str, Callable] = 'trilinear') -> GeoGrid:
        """
        Generate refined geometric grid for raytracing.
        
        Each velocity model cell is subdivided into subdivision^3 geo cells.
        Values are interpolated based on the chosen interpolation strategy.
        
        Parameters:
        -----------
        subdivision : int
            Subdivision factor (1 = no subdivision, 2 = 8x cells, 3 = 27x cells)
        interpolation : str or callable
            Interpolation method:
            - 'trilinear': Trilinear interpolation (default, smooth)
            - 'nearest': Nearest neighbor (fast, blocky)
            - callable: Custom interpolation function with signature
                       func(values, i, j, k, di, dj, dk) -> float
        
        Returns:
        --------
        GeoGrid with refined resolution
        
        Examples:
        ---------
        >>> # 1:1 mapping
        >>> geo = model.get_geo_grid(subdivision=1)
        >>> 
        >>> # 27x refinement with smooth interpolation
        >>> geo = model.get_geo_grid(subdivision=3)
        >>> 
        >>> # Custom interpolation
        >>> def my_interp(values, i, j, k, di, dj, dk):
        >>>     return values[i, j, k]  # custom logic
        >>> geo = model.get_geo_grid(subdivision=2, interpolation=my_interp)
        """
        if subdivision < 1:
            raise ValueError("subdivision must be >= 1")
        
        # Select interpolation function
        if isinstance(interpolation, str):
            if interpolation == 'trilinear':
                interp_func = trilinear_interpolation
            elif interpolation == 'nearest':
                interp_func = nearest_neighbor_interpolation
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation}")
        elif callable(interpolation):
            interp_func = interpolation
        else:
            raise TypeError("interpolation must be str or callable")
        
        # Create geo grid
        geo_shape = (
            self.geometry.n_x * subdivision,
            self.geometry.n_y * subdivision,
            self.geometry.n_z * subdivision
        )
        geo_cell_size = self.geometry.side_size / subdivision
        geo_grid = GeoGrid(geo_shape, geo_cell_size, subdivision)
        
        # Fill geo grid with interpolated values
        for gi in range(geo_shape[0]):
            for gj in range(geo_shape[1]):
                for gk in range(geo_shape[2]):
                    # Map geo index to velocity model index
                    i = gi // subdivision
                    j = gj // subdivision
                    k = gk // subdivision
                    
                    # Fractional position within velocity cell [0..1]
                    di = (gi % subdivision) / subdivision
                    dj = (gj % subdivision) / subdivision
                    dk = (gk % subdivision) / subdivision
                    
                    # Interpolate
                    geo_grid.vp[gi, gj, gk] = interp_func(
                        self.grid.vp, i, j, k, di, dj, dk
                    )
                    geo_grid.vs[gi, gj, gk] = interp_func(
                        self.grid.vs, i, j, k, di, dj, dk
                    )
        
        return geo_grid
    
    def __repr__(self) -> str:
        return (f"VelocityModel(grid_size=({self.geometry.n_x}, {self.geometry.n_y}, "
                f"{self.geometry.n_z}), cell_size={self.geometry.side_size}m, "
                f"center=({self.geometry.lon:.4f}, {self.geometry.lat:.4f}))")