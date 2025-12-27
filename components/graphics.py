"""Graphics and visualization utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import inspect

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.stats import ttest_ind

from .utilities import generate_name


# Configuration
DEFAULT_DPI = 300


def _get_pictures_dir():
    """Get pictures directory relative to the calling script."""
    # Get the file that called the graphics function
    frames = inspect.stack()
    frame = frames[4]  # Go up the call stack
    caller_file = frame.filename
    caller_dir = Path(caller_file).parent
    return caller_dir / 'pictures'


def _ensure_pictures_dir():
    """Ensure pictures directory exists."""
    pics_dir = _get_pictures_dir()
    pics_dir.mkdir(parents=True, exist_ok=True)
    return pics_dir


def save_figure(fig, filename, dpi=DEFAULT_DPI, **kwargs):
    """
    Save matplotlib figure to pictures directory next to calling script.
    
    Args:
        fig: matplotlib figure object
        filename: output filename (can be just name or full path)
        dpi: dots per inch for output
        **kwargs: additional arguments for plt.savefig
    """
    pics_dir = _ensure_pictures_dir()
    
    # If filename is just a name, put it in pictures dir
    filepath = Path(filename)
    if not filepath.is_absolute() and filepath.parent == Path('.'):
        filepath = pics_dir / filename
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Default kwargs
    save_kwargs = {'dpi': dpi, 'bbox_inches': 'tight'}
    save_kwargs.update(kwargs)
    
    plt.savefig(filepath, **save_kwargs)
    print(f"Saved: {filepath}")


def plot_vels_2d(grid, v_data, filename='2d_vels.png'):
    """Plot 2D velocity distribution on a map."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    contour = ax.tricontourf(
        list(grid[0]), list(grid[1]), list(v_data),
        levels=20, cmap='viridis', alpha=0.8
    )
    
    cbar = plt.colorbar(contour, shrink=0.7, label='Скорость Vs (м/с)')
    cbar.ax.tick_params(labelsize=10)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    plt.tight_layout()
    save_figure(fig, filename)


def plot_hist(values, trimmed=False, quantile=0.95, filename=None):
    """Plot histogram of values."""
    if filename is None:
        filename = f"{generate_name(['hist'])}.png"
    
    if trimmed:
        values = sorted(values)
        lower = round((1 - quantile) * len(values))
        upper = round(quantile * len(values))
        values = values[lower:upper]
    
    fig = plt.figure()
    plt.hist(values, bins=140)
    save_figure(fig, filename)


def plot_ray_path(path, start_point, end_point, elev=30, azim=-70, multi_view=True, filename=None):
    """Plot 3D ray path compared to straight line."""
    path = np.array(path)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', label='Recovered Path')
    ax.plot(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        [start_point[2], end_point[2]],
        color='red', linestyle='--', label='Straight Line'
    )
    ax.scatter(*start_point, color='green', s=50, label='Start Point')
    ax.scatter(*end_point, color='black', s=50, label='End Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray Path vs Straight Line')
    ax.legend()
    ax.grid(True)
    
    timestr = time.strftime("%Y%m%d%H%M%S")
    
    if multi_view:
        for i in range(-1, 2):
            current_az = i * 70
            ax.view_init(elev=elev, azim=current_az)
            fname = filename or f'ray_path_{timestr}_{elev}_{current_az}.png'
            save_figure(fig, fname)
    else:
        ax.view_init(elev=elev, azim=azim)
        fname = filename or f'ray_path_{timestr}_{elev}_{azim}.png'
        save_figure(fig, fname)


def plot_wave_front_points(time_matrix, time_from_origin, dt, filename=None):
    """Plot wave front as scattered points in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    mask = (time_matrix >= time_from_origin - dt) & (time_matrix <= time_from_origin + dt)
    z_idx, y_idx, x_idx = np.where(mask)
    
    times = time_matrix[mask]
    norm_times = (times - (time_from_origin - dt)) / (2 * dt)
    
    sc = ax.scatter(
        x_idx, y_idx, z_idx, c=norm_times, cmap='viridis',
        marker='o', s=1, alpha=0.6
    )
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Normalized Time')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Wave Front at t = {time_from_origin} ± {dt}')
    
    fname = filename or f'wavefront_{generate_name([time_from_origin, dt])}.png'
    save_figure(fig, fname)


def _generate_cube(x, y, z, size):
    """Generate vertices for a cube centered at (x, y, z)."""
    h = size / 2
    vertices = [
        [x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h],
        [x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    return faces


def plot_wave_front(time_matrix, time_from_origin, dt, max_points=50_000, filename=None):
    """Plot wave front as voxels in 3D."""
    fig = plt.figure(figsize=(10, 8), dpi=500)
    ax = fig.add_subplot(111, projection='3d')

    mask = (time_matrix >= time_from_origin - dt) & (time_matrix <= time_from_origin + dt)
    z, y, x = np.where(mask)

    if len(x) > max_points:
        step = len(x) // max_points
        x, y, z = x[::step], y[::step], z[::step]
        print(f"Downsampled from {len(mask.nonzero()[0])} to {len(x)} points")

    cube_size = 1.0
    for xi, yi, zi in zip(x, y, z):
        cube = art3d.Poly3DCollection(
            _generate_cube(xi, yi, zi, cube_size),
            alpha=0.5, linewidth=0.05, edgecolor='k', facecolor='blue'
        )
        ax.add_collection3d(cube)

    ax.set_xlim([0, time_matrix.shape[2]])
    ax.set_ylim([0, time_matrix.shape[1]])
    ax.set_zlim([0, time_matrix.shape[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Wavefront at t = {time_from_origin} ± {dt} | {len(x)} cubes')
    
    plt.tight_layout()
    fname = filename or f'wavefront_{generate_name([time_from_origin, dt])}.png'
    save_figure(fig, fname, dpi=500)


def plot_fat_ray(time_matrix_sx, time_matrix_rx, source_coords, receiver_coords, T, 
                 max_points=50_000, filename=None):
    """Plot fat ray (Fresnel zone) between source and receiver."""
    fig = plt.figure(figsize=(10, 8), dpi=500)
    ax = fig.add_subplot(111, projection='3d')

    t_sr = time_matrix_sx[receiver_coords]
    t_rs = time_matrix_rx[source_coords]
    t_travel = 0.5 * (t_sr + t_rs)

    condition = time_matrix_sx + time_matrix_rx - t_travel <= T
    z, y, x = np.where(condition)

    if len(x) > max_points:
        step = len(x) // max_points
        x, y, z = x[::step], y[::step], z[::step]
        print(f"Downsampled from {len(x) * step} to {len(x)} points")

    cube_size = 1.0
    for xi, yi, zi in zip(x, y, z):
        cube = art3d.Poly3DCollection(
            _generate_cube(xi, yi, zi, cube_size),
            alpha=0.5, linewidth=0.05, edgecolor='k', facecolor='blue'
        )
        ax.add_collection3d(cube)

    ax.scatter(*source_coords[::-1], color='red', s=100, label='Source')
    ax.scatter(*receiver_coords[::-1], color='green', s=100, label='Receiver')

    ax.set_xlim([0, time_matrix_sx.shape[2]])
    ax.set_ylim([0, time_matrix_sx.shape[1]])
    ax.set_zlim([0, time_matrix_sx.shape[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Cells where t_sx + t_rx - t_travel ≤ {T} | {len(x)} cubes')
    ax.legend()
    
    for i in range(-1, 2):
        current_az = i * 70
        ax.view_init(elev=30, azim=current_az)
        plt.tight_layout()
        fname = filename or f'fat_ray_{generate_name([T, current_az])}.png'
        save_figure(fig, fname, dpi=500)


def simple_plot(x, y, filename=None):
    """Simple line plot."""
    fig = plt.figure()
    plt.plot(x, y)
    fname = filename or f"{generate_name(['plot'])}.png"
    save_figure(fig, fname)


def simple_scatter(x, y, filename=None):
    """Simple scatter plot."""
    fig = plt.figure(dpi=300)
    plt.scatter(x, y, s=2)
    plt.grid()
    fname = filename or f"{generate_name(['scatter'])}.png"
    save_figure(fig, fname)


def plot_and_compare_stddev_by_date(x, y, cutoff_str='2015-03-01',
                                   group_names=('До 2015', 'После 2015'),
                                   xlabel='Дата землетрясения',
                                   ylabel='std, с.',
                                   title='По приходам S волн',
                                   filename='stddev_compare_S.png',
                                   dpi=500):
    """
    Plot and statistically compare two groups split by date.
    
    Returns:
        dict: Statistics including means, t-test results, and raw data
    """
    # Prepare data
    dates = pd.to_datetime(x, errors='coerce')
    y = np.array(y, dtype=float)
    cutoff = pd.Timestamp(cutoff_str)
    df = pd.DataFrame({'date': dates, 'y': y})
    df = df.dropna().sort_values('date')

    groupA = df[df['date'] < cutoff]
    groupB = df[df['date'] >= cutoff]
    
    # Plot
    plt.rcParams.update({'axes.labelsize': 12, 'font.size': 11})

    colA = "#5A56C5"
    colB = "#C8626BED"

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=dpi)
    
    ax.scatter(
        groupA['date'], groupA['y'], s=8, color=colA, alpha=0.85,
        label=f"{group_names[0]} (n={len(groupA)})", 
        edgecolor='blue', linewidth=0.3
    )
    ax.scatter(
        groupB['date'], groupB['y'], s=8, color=colB, alpha=0.85,
        label=f"{group_names[1]} (n={len(groupB)})", 
        edgecolor='red', linewidth=0.3
    )

    meanA, meanB = groupA['y'].mean(), groupB['y'].mean()
    
    if not groupA.empty:
        ax.hlines(
            meanA, groupA['date'].min(), groupA['date'].max(),
            colors=colA, lw=2, linestyles='--', label='Среднее до 2015'
        )
    if not groupB.empty:
        ax.hlines(
            meanB, groupB['date'].min(), groupB['date'].max(),
            colors=colB, lw=2, linestyles='--', label='Среднее после 2015'
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.23)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    ax.legend(
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        frameon=False, borderaxespad=0.3, fontsize=11
    )
    
    save_figure(fig, filename, dpi=dpi, transparent=False)
    
    plt.close(fig)

    # Statistics
    if len(groupA) > 1 and len(groupB) > 1:
        t_stat, p_val = ttest_ind(
            groupA['y'], groupB['y'], 
            equal_var=False, nan_policy='omit'
        )
    else:
        t_stat, p_val = np.nan, np.nan

    print('--- Welch t-test for independent samples ---')
    print(f'Before {cutoff_str}:  n={len(groupA)}, mean={meanA:.3f}')
    print(f'After {cutoff_str}:   n={len(groupB)}, mean={meanB:.3f}')
    if np.isfinite(t_stat):
        print(f'Welch t-test: t = {t_stat:.3f}, p = {p_val:.4g}')
    else:
        print('Insufficient data for t-test')
    
    return {
        'groupA_mean': meanA, 'groupB_mean': meanB,
        't_stat': t_stat, 'p_val': p_val,
        'nA': len(groupA), 'nB': len(groupB),
        'groupA_y': groupA['y'].values,
        'groupB_y': groupB['y'].values
    }


def plot_spatial_distribution(df, param_nm, filename,
                              vmin=None, vmax=None,
                              lon_bounds=None, lat_bounds=None,
                              interpolation_method='cubic',
                              figsize=(14, 10), cmap='viridis', dpi=500):
    """Plot spatial distribution of parameter on a map with interpolation."""
    df_clean = df[['lon', 'lat', param_nm]].dropna()
    if df_clean.empty:
        print("No data to plot")
        return

    lons = df_clean['lon'].values
    lats = df_clean['lat'].values
    values = df_clean[param_nm].values

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    if lon_bounds is None:
        lon_margin = (lons.max() - lons.min()) * 0.1
        lon_bounds = (lons.min() - lon_margin, lons.max() + lon_margin)

    if lat_bounds is None:
        lat_margin = (lats.max() - lats.min()) * 0.1
        lat_bounds = (lats.min() - lat_margin, lats.max() + lat_margin)

    ax.set_extent(
        [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')

    grid_res = 200
    grid_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_res)
    grid_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_res)
    X, Y = np.meshgrid(grid_lon, grid_lat)

    try:
        Z = griddata((lons, lats), values, (X, Y), method=interpolation_method)
    except Exception:
        Z = griddata((lons, lats), values, (X, Y), method='linear')

    cf = ax.contourf(
        X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.7, transform=ccrs.PlateCarree()
    )

    ax.scatter(
        lons, lats, c=values, s=30, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='black', linewidth=0.5, alpha=0.9, zorder=5,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(param_nm, fontsize=12)

    plt.tight_layout()
    save_figure(fig, filename, dpi=dpi)
    plt.close()


def plot_spatial_distribution_series(df, param_nm, experiment_name,
                                     time_from_col='t_from', time_to_col='t_to',
                                     interpolation_method='cubic',
                                     figsize=(14, 10), cmap='viridis', dpi=500):
    """
    Plot time series of spatial distributions.
    
    Returns:
        list: Paths to saved files
    """
    pics_dir = _ensure_pictures_dir()
    out_dir = pics_dir / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = df[[time_from_col, time_to_col]].drop_duplicates()
    periods = periods.sort_values(by=time_from_col)

    # Unified color scale
    vmin = df[param_nm].min()
    vmax = df[param_nm].max()

    # Unified map bounds
    lon_min, lon_max = df['node_lon'].min(), df['node_lon'].max()
    lat_min, lat_max = df['node_lat'].min(), df['node_lat'].max()

    lon_margin = (lon_max - lon_min) * 0.1
    lat_margin = (lat_max - lat_min) * 0.1

    lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)
    lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)

    saved_files = []

    for i, row in periods.iterrows():
        t_from = row[time_from_col]
        t_to = row[time_to_col]

        df_p = df[(df[time_from_col] == t_from) & (df[time_to_col] == t_to)]

        fname = out_dir / f"{i:03d}_{param_nm}_{t_from}_to_{t_to}.png"
        fname = Path(str(fname).replace(":", "-"))

        plot_spatial_distribution(
            df_p, param_nm, filename=fname,
            vmin=vmin, vmax=vmax,
            lon_bounds=lon_bounds, lat_bounds=lat_bounds,
            interpolation_method=interpolation_method,
            figsize=figsize, cmap=cmap, dpi=dpi
        )

        saved_files.append(fname)

    return saved_files


def plot_delta_t(df, filename=None):
    """Plot relationship between P-wave and S-wave arrival times."""
    # Convert timedelta to seconds if needed
    for col in ['delta_t_p', 'delta_t_s']:
        if df[col].dtype == 'timedelta64[ns]':
            df[col] = df[col].dt.total_seconds()

    df = df.dropna(subset=['delta_t_p', 'delta_t_s'])

    fig = plt.figure(figsize=(8, 6), dpi=500)
    
    color_map = {
        'g': 'tab:blue',
        'n': 'tab:orange',
        'b': 'tab:green',
        None: 'tab:gray'
    }
    
    plt.scatter(
        df['delta_t_p'], df['delta_t_s'],
        c=df['wave_subtype'].map(color_map),
        alpha=0.7, s=0.3
    )

    plt.xlabel("Δt (P-wave), sec")
    plt.ylabel("Δt (S-wave), sec")
    plt.title("S-wave vs P-wave arrival time relationship")

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=color, markersize=8)
        for label, color in [('g', 'tab:blue'), ('n', 'tab:orange'), 
                             ('b', 'tab:green'), ('no subtype', 'tab:gray')]
    ]
    plt.legend(handles=handles, title="Wave subtype")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    fname = filename or f"{generate_name(['delta_t'])}.png"
    save_figure(fig, fname, dpi=500)


def plot_events_stations(events_df, stations_df, edges_df, filename='events-stations.png'):
    """Plot events, stations and their connections on a map."""
    fig = plt.figure(figsize=(10, 8), dpi=500)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Plot events
    ax.scatter(
        events_df['lon'], events_df['lat'],
        color='red', s=30, marker='*', label='Events'
    )

    # Plot stations
    ax.scatter(
        stations_df['lon'], stations_df['lat'],
        color='green', s=30, marker='^', label='Stations'
    )

    # Plot connections
    for _, row in edges_df.iterrows():
        ev = events_df.loc[events_df['event_id'] == row['event_id']].iloc[0]
        st = stations_df.loc[stations_df['station_id'] == row['station_id']].iloc[0]

        ax.plot(
            [ev['lon'], st['lon']],
            [ev['lat'], st['lat']],
            color='gray', linewidth=0.5, alpha=0.1
        )

    ax.legend()
    plt.title("Events, Stations and Connections")
    save_figure(fig, filename, dpi=500)