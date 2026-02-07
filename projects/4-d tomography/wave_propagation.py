"""
Wave propagation calculations for seismic tomography.

Computes travel times from a source point to all cells in the velocity grid
using eikonal equation solvers.
"""
import numpy as np
from typing import Tuple, Optional, Callable, Union, Protocol
from abc import ABC, abstractmethod


class EikonalSolver(Protocol):
    """
    Protocol for eikonal equation solvers.
    
    Any solver must implement the solve() method with this signature.
    """
    
    def solve(self, velocity: np.ndarray, source_idx: Tuple[int, int, int],
              cell_size: float) -> np.ndarray:
        """
        Solve eikonal equation to compute travel times.
        
        Parameters:
        -----------
        velocity : np.ndarray
            3D array of velocities (m/s), shape (n_x, n_y, n_z)
        source_idx : tuple of (i, j, k)
            Source cell indices
        cell_size : float
            Cell size in meters
        
        Returns:
        --------
        np.ndarray
            Travel times to each cell (seconds), same shape as velocity
        """
        ...


class SKFMMSolver:
    """
    Fast Marching Method solver using scikit-fmm.
    
    This is the default solver - efficient and accurate for smooth velocity fields.
    Requires: pip install scikit-fmm
    """
    
    def __init__(self, order: int = 2):
        """
        Initialize SKFMM solver.
        
        Parameters:
        -----------
        order : int
            Approximation order (1 or 2). Order 2 is more accurate but slower.
        """
        try:
            import skfmm
            self.skfmm = skfmm
        except ImportError:
            raise ImportError(
                "scikit-fmm is required for SKFMMSolver. "
                "Install with: pip install scikit-fmm"
            )
        self.order = order
    
    def solve(self, velocity: np.ndarray, source_idx: Tuple[int, int, int],
              cell_size: float) -> np.ndarray:
        """
        Solve using Fast Marching Method.
        
        The eikonal equation: |∇T| = 1/v(x)
        where T is travel time and v is velocity.
        """
        # Create phi array (distance field)
        # Negative inside source, positive outside
        phi = np.ones_like(velocity, dtype=np.float64)
        phi[source_idx] = -1
        
        # Solve eikonal equation
        # dx parameter accounts for cell size
        travel_time = self.skfmm.travel_time(
            phi=phi,
            speed=velocity.astype(np.float64),
            dx=cell_size,
            order=self.order
        )
        
        return travel_time.astype(np.float32)


class SimpleFDSolver:
    """
    Simple Finite Difference solver (fallback, slower but no dependencies).
    
    Uses basic Dijkstra-like algorithm for travel time computation.
    Good for testing but not recommended for production.
    """
    
    def solve(self, velocity: np.ndarray, source_idx: Tuple[int, int, int],
              cell_size: float) -> np.ndarray:
        """
        Solve using simple finite difference sweeping.
        """
        from scipy.ndimage import distance_transform_edt
        
        n_x, n_y, n_z = velocity.shape
        
        # Initialize travel times to infinity
        travel_time = np.full(velocity.shape, np.inf, dtype=np.float32)
        travel_time[source_idx] = 0.0
        
        # Simple sweeping - not optimal but works
        # In reality, should use proper Dijkstra or Fast Sweeping
        changed = True
        max_iter = 100
        iteration = 0
        
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            
            # Sweep in all directions
            for i in range(n_x):
                for j in range(n_y):
                    for k in range(n_z):
                        if (i, j, k) == source_idx:
                            continue
                        
                        # Check all 6 neighbors
                        min_neighbor_time = np.inf
                        
                        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), 
                                          (0,1,0), (0,0,-1), (0,0,1)]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            
                            if (0 <= ni < n_x and 0 <= nj < n_y and 
                                0 <= nk < n_z):
                                min_neighbor_time = min(
                                    min_neighbor_time,
                                    travel_time[ni, nj, nk]
                                )
                        
                        # Update travel time
                        if min_neighbor_time < np.inf:
                            dt = cell_size / velocity[i, j, k]
                            new_time = min_neighbor_time + dt
                            
                            if new_time < travel_time[i, j, k]:
                                travel_time[i, j, k] = new_time
                                changed = True
        
        return travel_time


class WavePropagator:
    """
    Wave propagation calculator with pluggable solvers.
    
    Computes travel times from a source to all cells in a velocity grid.
    """
    
    def __init__(self, solver: Optional[Union[str, EikonalSolver]] = 'skfmm'):
        """
        Initialize wave propagator.
        
        Parameters:
        -----------
        solver : str or EikonalSolver
            Solver to use:
            - 'skfmm': Fast Marching Method (default, requires scikit-fmm)
            - 'simple': Simple finite difference (slow, no dependencies)
            - Custom solver object implementing EikonalSolver protocol
        """
        if isinstance(solver, str):
            if solver == 'skfmm':
                self.solver = SKFMMSolver()
            elif solver == 'simple':
                self.solver = SimpleFDSolver()
            else:
                raise ValueError(f"Unknown solver: {solver}")
        else:
            # Custom solver
            self.solver = solver
    
    def compute_travel_times(
        self,
        velocity: np.ndarray,
        source_idx: Tuple[int, int, int],
        cell_size: float,
        wave_type: str = 'P'
    ) -> np.ndarray:
        """
        Compute travel times from source to all cells.
        
        Parameters:
        -----------
        velocity : np.ndarray
            3D velocity array (m/s), shape (n_x, n_y, n_z)
            For wave_type='P' use Vp, for 'S' use Vs
        source_idx : tuple of (i, j, k)
            Source position in grid indices
        cell_size : float
            Size of each cell in meters
        wave_type : str
            'P' or 'S' (for documentation, doesn't affect calculation)
        
        Returns:
        --------
        np.ndarray
            Travel times in seconds, same shape as velocity
        
        Examples:
        ---------
        >>> propagator = WavePropagator(solver='skfmm')
        >>> travel_times = propagator.compute_travel_times(
        ...     velocity=vp_array,
        ...     source_idx=(10, 20, 0),
        ...     cell_size=100.0
        ... )
        """
        # Validate inputs
        if not isinstance(velocity, np.ndarray) or velocity.ndim != 3:
            raise ValueError("velocity must be a 3D numpy array")
        
        i, j, k = source_idx
        n_x, n_y, n_z = velocity.shape
        
        if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
            raise ValueError(
                f"source_idx {source_idx} out of bounds for grid shape {velocity.shape}"
            )
        
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        
        if wave_type not in ['P', 'S']:
            raise ValueError("wave_type must be 'P' or 'S'")
        
        # Solve eikonal equation
        travel_times = self.solver.solve(velocity, source_idx, cell_size)
        
        return travel_times
    
    def compute_from_geo_grid(
        self,
        geo_grid,  # GeoGrid object
        source_idx: Tuple[int, int, int],
        wave_type: str = 'P'
    ) -> np.ndarray:
        """
        Convenience method to compute travel times from a GeoGrid object.
        
        Parameters:
        -----------
        geo_grid : GeoGrid
            Geographic grid with velocity data
        source_idx : tuple
            Source position in geo grid indices
        wave_type : str
            'P' or 'S' wave
        
        Returns:
        --------
        np.ndarray
            Travel times in seconds
        """
        velocity = geo_grid.vp if wave_type == 'P' else geo_grid.vs
        
        return self.compute_travel_times(
            velocity=velocity,
            source_idx=source_idx,
            cell_size=geo_grid.cell_size,
            wave_type=wave_type
        )


# Convenience function
def compute_travel_times(
    velocity: np.ndarray,
    source_idx: Tuple[int, int, int],
    cell_size: float,
    solver: str = 'skfmm',
    wave_type: str = 'P'
) -> np.ndarray:
    """
    Convenience function to compute travel times.
    
    Parameters:
    -----------
    velocity : np.ndarray
        3D velocity array (m/s)
    source_idx : tuple
        Source position (i, j, k)
    cell_size : float
        Cell size in meters
    solver : str
        'skfmm' or 'simple'
    wave_type : str
        'P' or 'S'
    
    Returns:
    --------
    np.ndarray
        Travel times in seconds
    
    Examples:
    ---------
    >>> times = compute_travel_times(vp_array, (10, 20, 0), 100.0)
    """
    propagator = WavePropagator(solver=solver)
    return propagator.compute_travel_times(velocity, source_idx, cell_size, wave_type)