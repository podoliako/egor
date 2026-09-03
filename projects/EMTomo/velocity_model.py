"""
Seismic velocity model with geographic grid reference.
"""
import numpy as np
import json
from typing import Callable, Dict, Optional, Tuple, Union

from interpolation import (
    nearest_neighbor_interpolation,
    prolongate_cell_centered_trilinear,
)


def _prolongate_slowness_as_velocity(values: np.ndarray, subdivision: int) -> np.ndarray:
    """Interpolate positive cell-centred velocities through slowness."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(values <= 0.0):
        # An uninitialized component (commonly Vs in a P-only model) is not
        # physically usable for FMM; preserve its values without dividing by 0.
        return prolongate_cell_centered_trilinear(values, subdivision)
    fine_slowness = prolongate_cell_centered_trilinear(1.0 / values, subdivision)
    return 1.0 / fine_slowness


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
    
    Coordinate convention:
    ----------------------
    ``(lon, lat, height)`` identifies the geographic position of the centre of
    the model's *top face*, i.e. local metric point
    ``(n_x * side_size / 2, n_y * side_size / 2, 0)``.  This is a geometric
    point, not the centre of a particular cell, so it remains unambiguous for
    both odd and even grid dimensions.

    EMTomo calculations use local metric coordinates.  The local axes are:

    - ``x``: direction ``azimuth`` clockwise from geographic North;
    - ``y``: direction ``azimuth + 90°`` (to the right of ``x``);
    - ``z``: positive downward from the top face.

    A value at array index ``(i, j, k)`` represents the centre of the cell
    ``[(i, j, k) * side_size, (i + 1, j + 1, k + 1) * side_size)``.
    Geographic conversion is intentionally left to an external projection
    layer (for example, a UCVM adapter).

    Parameters:
    -----------
    lon, lat : float
        Longitude and latitude of the model top-face centre (degrees).
    height : float
        Height of the model top-face centre relative to ground level (meters).
    azimuth : float
        Orientation of local x (degrees, clockwise from North).
    side_size : float
        Uniform cell side length in meters.
    n_x, n_y, n_z : int
        Number of cells along each local axis; z increases downward.
    """

    def __init__(self, lon: float, lat: float, height: float,
                 azimuth: float, side_size: float,
                 n_x: int, n_y: int, n_z: int):
        if side_size <= 0:
            raise ValueError("side_size must be positive")
        if min(n_x, n_y, n_z) <= 0:
            raise ValueError("n_x, n_y, and n_z must be positive")
        self.lon = lon
        self.lat = lat
        self.height = height
        self.azimuth = azimuth
        self.side_size = side_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Number of cells as ``(n_x, n_y, n_z)``."""
        return (self.n_x, self.n_y, self.n_z)

    @property
    def top_face_center_local(self) -> Tuple[float, float, float]:
        """Local metric coordinates of the geographic reference point."""
        return (
            self.n_x * self.side_size / 2.0,
            self.n_y * self.side_size / 2.0,
            0.0,
        )

    def cell_center_local(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Return the local metric centre of cell ``(i, j, k)`` in meters."""
        if not (0 <= i < self.n_x and 0 <= j < self.n_y and 0 <= k < self.n_z):
            raise IndexError(f"cell index {(i, j, k)} out of bounds for shape {self.shape}")
        return (
            (i + 0.5) * self.side_size,
            (j + 0.5) * self.side_size,
            (k + 0.5) * self.side_size,
        )
        
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
    
    def get_geo_grid(
        self,
        subdivision: int = 1,
        interpolation: Union[str, Callable] = 'nearest',
        slowness_interpolation: str = 'nearest',
    ) -> GeoGrid:
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
            - 'trilinear': Cell-centred trilinear interpolation of velocity
            - 'nearest': Nearest neighbor (fast, blocky)
            - callable: Custom interpolation function with signature
                       func(values, i, j, k, di, dj, dk) -> float
            slowness_interpolation : {'nearest', 'trilinear'}
                Tomography parameterization for positive velocity components.
                'trilinear' prolongs slowness and converts it back to velocity.
                It takes precedence over ``interpolation``.
        
        Returns:
        --------
        GeoGrid with refined resolution
        
        Examples:
        ---------
        >>> # 1:1 mapping
        >>> geo = model.get_geo_grid(subdivision=1)
        >>> 
        >>> # 27x refinement with smooth slowness interpolation
        >>> geo = model.get_geo_grid(
        ...     subdivision=3, slowness_interpolation='trilinear'
        ... )
        >>> 
        >>> # Custom interpolation
        >>> def my_interp(values, i, j, k, di, dj, dk):
        >>>     return values[i, j, k]  # custom logic
        >>> geo = model.get_geo_grid(subdivision=2, interpolation=my_interp)
        """
        if subdivision < 1:
            raise ValueError("subdivision must be >= 1")
        
        if slowness_interpolation not in {'nearest', 'trilinear'}:
            raise ValueError("slowness_interpolation must be 'nearest' or 'trilinear'")

        # Select interpolation function for the legacy velocity/custom modes.
        if isinstance(interpolation, str):
            if interpolation == 'trilinear':
                interp_func = None
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
        
        if slowness_interpolation == 'trilinear':
            geo_grid.vp[:] = _prolongate_slowness_as_velocity(self.grid.vp, subdivision)
            geo_grid.vs[:] = _prolongate_slowness_as_velocity(self.grid.vs, subdivision)
            return geo_grid

        if interpolation == 'trilinear':
            geo_grid.vp[:] = prolongate_cell_centered_trilinear(self.grid.vp, subdivision)
            geo_grid.vs[:] = prolongate_cell_centered_trilinear(self.grid.vs, subdivision)
            return geo_grid

        # Fill geo grid with nearest-neighbour or custom interpolation.
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
