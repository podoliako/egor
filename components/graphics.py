from .utilities import generate_name

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
from pathlib import Path


def plot_vels_2d(grid, v_data, file_name='2d_vels.png'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    contour = ax.tricontourf(list(grid[0]), list(grid[1]), list(v_data), 
                        levels=20, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(contour, shrink=0.7, label='Скорость Vs (м/с)')
    cbar.ax.tick_params(labelsize=10)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hist(values, trimmed=False):
    file_name = f'pictures/{generate_name(['hist'])}.png'
    if trimmed is True:
        quantile = 0.95
        values = list(values)
        values.sort()
        values = values[round((1-quantile)*len(values)):round(quantile*len(values))]
    plt.hist(values, bins=140)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ray_path(path, start_point, end_point, elev=30, azim=-70, multy=True):
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
    
    if multy is True:
        for i in range(-1,2,1):
            current_az = i*70
            ax.view_init(elev=elev, azim=current_az)
            timestr = time.strftime("%Y%m%d%H%M%S")
            file_name = f'ray_path_{timestr}_{elev}_{current_az}.png'
            plt.savefig(f'pictures/{file_name}')
   
    else:
        ax.view_init(elev=elev, azim=azim)
        timestr = time.strftime("%Y%m%d%H%M%S")
        file_name = f'ray_path_{timestr}_{elev}_{azim}.png'
        plt.savefig(f'pictures/{file_name}')


def plot_wave_front_points(time_matrix, time_from_origin, dt):
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find indices where time is within the specified range
    mask = (time_matrix >= time_from_origin - dt) & (time_matrix <= time_from_origin + dt)
    z_idx, y_idx, x_idx = np.where(mask)
    
    # Get the corresponding time values for coloring
    times = time_matrix[mask]
    
    # Normalize times for colormap (optional)
    norm_times = (times - (time_from_origin - dt)) / (2 * dt)
    
    # Plot each point (could be slow for large datasets)
    sc = ax.scatter(x_idx, y_idx, z_idx, c=norm_times, cmap='viridis', 
                    marker='o', s=1, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Normalized Time')
    
    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set title
    ax.set_title(f'Wave Front at t = {time_from_origin} ± {dt}')
    
    plt.savefig(f'pictures/wavefront_{generate_name([time_from_origin, dt])}.png')


def plot_wave_front_slow(time_matrix, time_from_origin, dt):
    """
    Рисует кубики (voxels) в 3D, где время в ячейках лежит в диапазоне time_from_origin ± dt.

    Параметры:
    - time_matrix: 3D массив (z, y, x) с временами достижения волной
    - time_from_origin: центральное время
    - dt: допустимое отклонение (±dt)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Находим индексы ячеек, попадающих в диапазон
    mask = (time_matrix >= time_from_origin - dt) & (time_matrix <= time_from_origin + dt)
    
    # Получаем координаты этих ячеек
    z, y, x = np.where(mask)

    # Если точек слишком много, можно сделать downsampling (например, брать каждую 5-ю)
    if len(x) > 100_000:
        step = max(1, len(x) // 50_000)  # Ограничиваем ~50k кубиков для скорости
        x, y, z = x[::step], y[::step], z[::step]

    # Создаем 3D-массив для voxels (False = пусто, True = кубик)
    voxels = np.zeros(time_matrix.shape, dtype=bool)
    voxels[z, y, x] = True  # Заполняем только нужные кубики

    # Рисуем voxels (однотонным цветом)
    ax.voxels(voxels, facecolors='blue', edgecolor='k', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Wavefront at t = {time_from_origin} ± {dt}')
    plt.tight_layout()
    plt.savefig(f'pictures/wavefront_{generate_name([time_from_origin, dt])}.png')
    
def plot_wave_front(time_matrix, time_from_origin, dt, max_points=50_000):
    fig = plt.figure(figsize=(10, 8), dpi=500)

    ax = fig.add_subplot(111, projection='3d')

    # Находим индексы ячеек в диапазоне
    mask = (time_matrix >= time_from_origin - dt) & (time_matrix <= time_from_origin + dt)
    z, y, x = np.where(mask)

    # Downsampling, если точек слишком много
    if len(x) > max_points:
        step = len(x) // max_points
        x, y, z = x[::step], y[::step], z[::step]
        print(f"Уменьшено число точек с {len(mask.nonzero()[0])} до {len(x)} для скорости")

    # Размер кубика (можно настроить)
    cube_size = 1.0

    # Рисуем каждый кубик
    for xi, yi, zi in zip(x, y, z):
        # Создаем куб в координатах (xi, yi, zi)
        cube = art3d.Poly3DCollection(_generate_cube(xi, yi, zi, cube_size), 
                                     alpha=0.5, linewidth=0.05, edgecolor='k', facecolor='blue')
        ax.add_collection3d(cube)

    ax.set_xlim([0, time_matrix.shape[2]])
    ax.set_ylim([0, time_matrix.shape[1]])
    ax.set_zlim([0, time_matrix.shape[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Wavefront at t = {time_from_origin} ± {dt} | {len(x)} cubes')
    plt.tight_layout()
    plt.savefig(f'pictures/wavefront_{generate_name([time_from_origin, dt])}.png')


def _generate_cube(x, y, z, size):
    """Генерирует вершины куба с центром в (x, y, z)."""
    h = size / 2
    vertices = [
        [x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h],  # Низ
        [x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h]   # Верх
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Низ
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Верх
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Перед
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Зад
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Право
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # Лево
    ]
    return faces

def plot_fat_ray(time_matrix_sx, time_matrix_rx, source_coords, receiver_coords, T, max_points=50_000):
    fig = plt.figure(figsize=(10, 8), dpi=500)
    ax = fig.add_subplot(111, projection='3d')

    # Вычисляем t_travel = 0.5 * (t_sr + t_rs)
    t_sr = time_matrix_sx[receiver_coords]  # время от источника до ресивера
    t_rs = time_matrix_rx[source_coords]    # время от ресивера до источника
    t_travel = 0.5 * (t_sr + t_rs)

    # Вычисляем условие t_sx + t_rx - t_travel ≤ T
    condition = time_matrix_sx + time_matrix_rx - t_travel <= T
    z, y, x = np.where(condition)

    # Даунсэмплинг, если точек слишком много
    if len(x) > max_points:
        step = len(x) // max_points
        x, y, z = x[::step], y[::step], z[::step]
        print(f"Уменьшено число точек с {len(x) * step} до {len(x)} для скорости")

    # Размер кубика (можно настроить)
    cube_size = 1.0

    # Рисуем каждый кубик
    for xi, yi, zi in zip(x, y, z):
        cube = art3d.Poly3DCollection(_generate_cube(xi, yi, zi, cube_size),
                                    alpha=0.5, linewidth=0.05, edgecolor='k', facecolor='blue')
        ax.add_collection3d(cube)

    # Отмечаем источник и ресивер красным и зелёным
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
    for i in range(-1,2,1):
        current_az = i*70
        ax.view_init(elev=30, azim=current_az)
        ax.view_init(elev=30, azim=current_az)
        plt.tight_layout()
        plt.savefig(f'pictures/fat_ray_{generate_name([T, current_az])}.png')

def simple_plot(x, y):
    plt.plot(x, y)
    plt.savefig(f'pictures/{generate_name(['plot'])}.png')

def simple_scatter(x, y):
    plt.figure(dpi=300)
    plt.scatter(x, y, s=2)
    plt.grid()
    plt.savefig(f'pictures/{generate_name(['scatter'])}.png')


def plot_and_compare_stddev_by_date(
        x, y,
        cutoff_str='2015-03-01',
        group_names=('До 2015', 'После 2015'),
        xlabel='Дата землетрясения',
        ylabel='std, с.',
        title='По приходам S волн',
        outpath='stddev_compare_S.png',
        dpi=500,
        show=False
    ):

    # --- 1. Подготовка данных ---
    dates = pd.to_datetime(x, errors='coerce')
    y = np.array(y, dtype=float)
    cutoff = pd.Timestamp(cutoff_str)
    df = pd.DataFrame({'date': dates, 'y': y})
    df = df.dropna().sort_values('date')

    groupA = df[df['date'] < cutoff]
    groupB = df[df['date'] >= cutoff]
    
    # --- 2. График ---
    plt.rcParams.update({'axes.labelsize': 12, 'font.size': 11})

    # Интересные цвета: глубокий бирюзовый и «малиновый»
    colA = "#5A56C5"  # бирюзовый/аквамарин (blue-green)
    colB = "#C8626BED"  # ярко-малиновый

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=dpi)
    ax.scatter(groupA['date'], groupA['y'], s=8, color=colA, alpha=0.85,
               label=f"{group_names[0]} (n={len(groupA)})", edgecolor='blue', linewidth=0.3)
    ax.scatter(groupB['date'], groupB['y'], s=8, color=colB, alpha=0.85,
               label=f"{group_names[1]} (n={len(groupB)})", edgecolor='red', linewidth=0.3)

    meanA, meanB = groupA['y'].mean(), groupB['y'].mean()
    if not groupA.empty:
        ax.hlines(meanA, groupA['date'].min(), groupA['date'].max(),
                  colors=colA, lw=2, linestyles='--', label=f'Среднее до 2015')
    if not groupB.empty:
        ax.hlines(meanB, groupB['date'].min(), groupB['date'].max(),
                  colors=colB, lw=2, linestyles='--', label=f'Среднее после 2015')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.23)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # место справа под легенду

    # --- Легенда вне графика ---
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.3,
        fontsize=11
    )
    # --- 3. Сохраняем график ---
    if outpath.lower().endswith('.png'):
        plt.savefig(outpath, dpi=dpi, bbox_inches='tight', transparent=False)
    else:
        plt.savefig(outpath, bbox_inches='tight', transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- 4. Статистика ---
    if len(groupA) > 1 and len(groupB) > 1:
        t_stat, p_val = ttest_ind(groupA['y'], groupB['y'], equal_var=False, nan_policy='omit')
    else:
        t_stat, p_val = np.nan, np.nan

    print('--- Welch t-критерий для независимых выборок ---')
    print(f'До {cutoff_str}:   n={len(groupA)}, среднее={meanA:.3f}')
    print(f'После {cutoff_str}: n={len(groupB)}, среднее={meanB:.3f}')
    if np.isfinite(t_stat):
        print(f'Welch t-test: t = {t_stat:.3f}, p = {p_val:.4g}')
    else:
        print('Недостаточно данных для t-test')
    
    return {'groupA_mean': meanA, 'groupB_mean': meanB,
            't_stat': t_stat, 'p_val': p_val,
            'nA': len(groupA), 'nB': len(groupB),
            'groupA_y': groupA['y'].values,
            'groupB_y': groupB['y'].values}

def plot_spatial_distribution(df, param_nm, output_filename,
                               vmin=None, vmax=None,
                               lon_bounds=None, lat_bounds=None,
                               interpolation_method='cubic',
                               figsize=(14, 10), cmap='viridis', dpi=300):

    df_clean = df[['node_lon', 'node_lat', param_nm]].dropna()
    if df_clean.empty:
        return

    lons = df_clean['node_lon'].values
    lats = df_clean['node_lat'].values
    values = df_clean[param_nm].values

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    if lon_bounds is None:
        lon_margin = (lons.max() - lons.min()) * 0.1
        lon_bounds = (lons.min() - lon_margin, lons.max() + lon_margin)

    if lat_bounds is None:
        lat_margin = (lats.max() - lats.min()) * 0.1
        lat_bounds = (lats.min() - lat_margin, lats.max() + lat_margin)

    ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
                  crs=ccrs.PlateCarree())

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
    except:
        Z = griddata((lons, lats), values, (X, Y), method='linear')

    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                     alpha=0.7, transform=ccrs.PlateCarree())

    ax.scatter(lons, lats, c=values, s=30, cmap=cmap, vmin=vmin, vmax=vmax,
               edgecolors='black', linewidth=0.5, alpha=0.9, zorder=5,
               transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(param_nm, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_delta_t(df, plt_nm=generate_name(['delta_t'])):
    """
    Строит график зависимости delta_t_s от delta_t_p.
    Принимает DataFrame с колонками delta_t_p, delta_t_s, wave_subtype.
    """
    # Преобразуем timedelta → секунды, если нужно
    for col in ['delta_t_p', 'delta_t_s']:
        if df[col].dtype == 'timedelta64[ns]':
            df[col] = df[col].dt.total_seconds()

    # Удалим пустые значения
    df = df.dropna(subset=['delta_t_p', 'delta_t_s'])

    plt.figure(figsize=(8, 6), dpi=500)
    scatter = plt.scatter(
        df['delta_t_p'],
        df['delta_t_s'],
        c=df['wave_subtype'].map({'g': 'tab:blue', 'n': 'tab:orange', 'b': 'tab:green', None: 'tab:gray'}),
        alpha=0.7,
        s=0.3
    )

    plt.xlabel("Δt (P-wave), sec")
    plt.ylabel("Δt (S-wave), sec")
    plt.title("Зависимость времени прихода S-волн от P-волн")

    # Легенда
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='g', markerfacecolor='tab:blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='n', markerfacecolor='tab:orange', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='b', markerfacecolor='tab:green', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='(no subtype)', markerfacecolor='tab:gray', markersize=8)
    ]
    plt.legend(handles=handles, title="Wave subtype")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plt_nm)

def plot_spatial_distribution_series(df, param_nm, experiment_name,
                                     time_from_col='t_from', time_to_col='t_to',
                                     interpolation_method='cubic',
                                     figsize=(14, 10), cmap='viridis', dpi=300):

    out_dir = Path("pictures") / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = df[[time_from_col, time_to_col]].drop_duplicates()
    periods = periods.sort_values(by=time_from_col)

    # единый color scale для всей серии
    vmin = df[param_nm].min()
    vmax = df[param_nm].max()

    # единые границы карты
    lon_min = df['node_lon'].min()
    lon_max = df['node_lon'].max()
    lat_min = df['node_lat'].min()
    lat_max = df['node_lat'].max()

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
            df_p,
            param_nm,
            output_filename=fname,
            vmin=vmin, vmax=vmax,
            lon_bounds=lon_bounds, lat_bounds=lat_bounds,
            interpolation_method=interpolation_method,
            figsize=figsize, cmap=cmap, dpi=dpi
        )

        saved_files.append(fname)

    return saved_files

def plot_events_stations(events_df, stations_df, edges_df, plot_nm='events-stations.png'):
    """
    Рисует события, станции и линии event→station на карте.
    """
    
    fig = plt.figure(figsize=(10, 8), dpi=500)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # --- Рисуем события ---
    ax.scatter(
        events_df['lon'], events_df['lat'],
        color='red', s=30, marker='*', label='Events'
    )

    # --- Рисуем станции ---
    ax.scatter(
        stations_df['lon'], stations_df['lat'],
        color='green', s=30, marker='^', label='Stations'
    )

    # --- Рисуем линии event → station ---
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
    plt.savefig(plot_nm)
