"""Graphics and visualization utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import inspect

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, art3d
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.stats import ttest_ind
from PIL import Image

from .utilities import generate_name, datetime


# Configuration
DEFAULT_DPI = 700


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


def save_figure(fig, filename, dpi=DEFAULT_DPI, subdir=None, **kwargs):
    """
    Save matplotlib figure to pictures directory next to calling script.
    
    Args:
        fig: matplotlib figure object
        filename: output filename (can be just name or full path)
        dpi: dots per inch for output
        subdir: optional subdirectory within pictures dir
        **kwargs: additional arguments for plt.savefig
    """
    pics_dir = _ensure_pictures_dir()
    
    # If subdir specified, create it
    if subdir:
        pics_dir = pics_dir / subdir
        pics_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    return filepath


def plot_vels_2d(grid, v_data, filename=None):
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

    fname = filename or generate_name(['vels_2d']) + '.png'
    save_figure(fig, fname)


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

    fname = filename or generate_name(['hist']) + '.png'
    save_figure(fig, fname)


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
            fname = filename or generate_name(['ray_path', {timestr}, {elev}, {current_az}]) + '.png'
            save_figure(fig, fname)
    else:
        ax.view_init(elev=elev, azim=azim)
        fname = filename or generate_name(['ray_path', {timestr}, {elev}, {azim}]) + '.png'
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
    
    fname = filename or generate_name(['wavefront', time_from_origin, dt]) + '.png'
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
    fname = filename or generate_name(['wavefront_cubic', time_from_origin, dt]) + '.png'
    save_figure(fig, fname)


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
        fname = filename or generate_name(['fat_ray', T, current_az]) + '.png'
        save_figure(fig, fname, dpi=500)


def simple_plot(x, y, filename=None):
    """Simple line plot."""
    fig = plt.figure()
    plt.plot(x, y)
    fname = filename or generate_name(['plot']) + '.png'
    save_figure(fig, fname)


def simple_scatter(x, y, filename=None, s=2):
    """Simple scatter plot."""
    fig = plt.figure(dpi=300)
    plt.scatter(x, y, s=s)
    plt.grid()
    fname = filename or generate_name(['scatter']) + '.png'
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


def plot_spatial_distribution(df, param_nm, vmin=None, vmax=None, 
                              lon_bounds=None, lat_bounds=None, 
                              interpolation_method='linear', figsize=(14, 10), 
                              cmap='viridis', filename=None, title=None,
                              subdir=None, ax=None, color_bar=True):
    """
    Plot spatial distribution of parameter on a map with interpolation.
    
    Args:
        df: DataFrame with 'lon', 'lat', and param_nm columns
        param_nm: parameter name to plot
        vmin, vmax: color scale limits
        lon_bounds, lat_bounds: map extent (auto-calculated if None)
        interpolation_method: 'cubic', 'linear', or 'nearest'
        figsize: figure size (ignored if ax is provided)
        cmap: colormap
        filename: output filename (if None, auto-generated)
        title: plot title (if None, uses param_nm)
        subdir: subdirectory for saving
        ax: existing axes to plot on (if None, creates new figure)
        
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    df_clean = df[['lon', 'lat', param_nm]].dropna()
    
    if df_clean.empty:
        print("No data to plot")
        return None, None
    
    lons = df_clean['lon'].values
    lats = df_clean['lat'].values
    values = df_clean[param_nm].values
    
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        own_fig = True
    else:
        fig = ax.get_figure()
        own_fig = False
    
    # Calculate bounds if not provided
    if lon_bounds is None:
        lon_margin = (lons.max() - lons.min()) * 0.1
        lon_bounds = (lons.min() - lon_margin, lons.max() + lon_margin)
    
    if lat_bounds is None:
        lat_margin = (lats.max() - lats.min()) * 0.1
        lat_bounds = (lats.min() - lat_margin, lats.max() + lat_margin)
    
    # Set map extent and features
    ax.set_extent(
        [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], 
        crs=ccrs.PlateCarree()
    )
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
    
    # Interpolation grid
    grid_res = 200
    grid_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_res)
    grid_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_res)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate
    try:
        Z = griddata((lons, lats), values, (X, Y), method=interpolation_method)
    except Exception:
        Z = griddata((lons, lats), values, (X, Y), method='linear')
    
    # Plot contours and points
    cf = ax.contourf(
        X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, 
        alpha=0.7, transform=ccrs.PlateCarree()
    )
    
    ax.scatter(
        lons, lats, c=values, s=1, cmap=cmap, vmin=vmin, vmax=vmax, 
        alpha=0.9, zorder=5, transform=ccrs.PlateCarree()
    )
    
    # Colorbar
    if color_bar is True:
        cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label(param_nm, fontsize=12)
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Save only if we created our own figure
    if own_fig:
        plt.tight_layout()
        fname = filename or generate_name(['spatial_distribution', param_nm]) + '.png'
        save_figure(fig, fname, subdir=subdir)
        plt.close()
    
    return fig, ax


def plot_spatial_distribution_timeseries(df, param_nm, vmin=None, vmax=None, 
                                         lon_bounds=None, lat_bounds=None, 
                                         interpolation_method='cubic', 
                                         figsize=(14, 10), cmap='viridis',
                                         output_dir='spatial_timeseries'):
    """
    Plot spatial distribution for each date and save frames for GIF creation.
    
    Args:
        df: DataFrame with columns 'lon', 'lat', 'dt', and param_nm
        param_nm: parameter name to plot
        vmin, vmax: color scale limits (consistent across all frames)
        lon_bounds, lat_bounds: map extent
        interpolation_method: 'cubic', 'linear', or 'nearest'
        figsize: figure size
        cmap: colormap
        output_dir: directory name for saving frames
    
    Returns:
        Path to output directory with frames
    """
    
    # Clean data and get unique dates
    df_clean = df[['lon', 'lat', 'dt', param_nm]].dropna()
    if df_clean.empty:
        print("No data to plot")
        return None
    
    dates = sorted(df_clean['dt'].unique())
    print(f"Creating {len(dates)} frames...")
    
    # Calculate global bounds and value ranges if not provided
    if lon_bounds is None or lat_bounds is None or vmin is None or vmax is None:
        lons = df_clean['lon'].values
        lats = df_clean['lat'].values
        values = df_clean[param_nm].values
        
        if lon_bounds is None:
            lon_margin = (lons.max() - lons.min()) * 0.1
            lon_bounds = (lons.min() - lon_margin, lons.max() + lon_margin)
        
        if lat_bounds is None:
            lat_margin = (lats.max() - lats.min()) * 0.1
            lat_bounds = (lats.min() - lat_margin, lats.max() + lat_margin)
        
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
    
    # Plot each date
    for i, date in enumerate(dates):
        df_date = df_clean[df_clean['dt'] == date]
        
        if df_date.empty:
            continue
        
        # Use the main plot function
        fig, ax = plot_spatial_distribution(
            df_date, 
            param_nm,
            vmin=vmin,
            vmax=vmax,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            interpolation_method=interpolation_method,
            figsize=figsize,
            cmap=cmap,
            filename=None,  # Don't save yet
            title=f'{param_nm} - {date}'
        )
        
        if fig is None:
            continue
        
        # Save with custom filename in subdirectory
        filename = f'frame_{i:04d}_{date}.png'
        save_figure(fig, filename, subdir=output_dir)
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dates)} frames")
    
    frames_dir = _ensure_pictures_dir() / output_dir
    print(f"\nAll frames saved to: {frames_dir}")
    print(f"\nTo create GIF, run:")
    print(f"  from PIL import Image")
    print(f"  import glob")
    print(f"  frames = [Image.open(f) for f in sorted(glob.glob('{frames_dir}/frame_*.png'))]")
    print(f"  frames[0].save('{frames_dir.parent}/animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)")
    
    return frames_dir

def create_gif_from_pngs(input_folder, output_path='animation.gif', duration=100, loop=0):
    """
    Создает GIF из PNG изображений в папке.
    
    Параметры:
    ----------
    input_folder : str
        Путь к папке с PNG изображениями
    output_path : str, optional
        Путь для сохранения GIF (по умолчанию 'animation.gif')
    duration : int, optional
        Длительность каждого кадра в миллисекундах (по умолчанию 100)
    loop : int, optional
        Количество повторений (0 = бесконечно, по умолчанию 0)
    
    Возвращает:
    -----------
    str : Путь к созданному GIF файлу
    """
    
    # Получаем список PNG файлов
    folder_path = Path(input_folder)
    png_files = sorted(folder_path.glob('*.png'))
    
    if not png_files:
        raise ValueError(f"В папке {input_folder} не найдено PNG файлов")
    
    # Загружаем изображения
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        # Конвертируем в RGB, если нужно (GIF не поддерживает RGBA напрямую)
        if img.mode == 'RGBA':
            # Создаем белый фон
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Используем альфа-канал как маску
            images.append(rgb_img)
        else:
            images.append(img.convert('RGB'))
    
    # Сохраняем как GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    
    print(f"GIF создан: {output_path}")
    print(f"Количество кадров: {len(images)}")
    
    return output_path


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
    
    fname = filename or generate_name(['delta_t']) + '.png'
    save_figure(fig, fname, dpi=500)


def plot_events_stations(events_df, stations_df, edges_df, filename=None):
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
    fname = filename or generate_name(['events_stations']) + '.png'
    save_figure(fig, fname, dpi=500)