from components.dwh import pd, load_to_postgres, load_to_postgis, get_travel_times_by_region, get_region_borders
from components.utilities import make_grid_3d_time, generate_name, Earth

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import math
import json


def meters_to_deg_lat(r_m):
    return r_m / 111_320.0

def meters_to_deg_lon(r_m, lat):
    return r_m / (111_320.0 * np.cos(np.radians(lat)))

def make_json_serializable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except:
        return None

def replace_nan(data):
    # Если словарь — обрабатываем значения
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}

    # Если список — обрабатываем элементы
    if isinstance(data, list):
        return [replace_nan(item) for item in data]

    # Если значение — float nan
    if isinstance(data, float) and math.isnan(data):
        return "Nan"

    # Если значение строка "nan" (на всякий случай)
    if isinstance(data, str) and data.lower() == "nan":
        return "Nan"

    return data

def estimate_vp_vs_ratio(df):
    df["ratio_ts_tp"] = df["delta_t_s"] / df["delta_t_p"]

    X = df[["delta_t_p"]].values
    y = df["delta_t_s"].values
    n_points = len(df)

    # free regression
    model_free = LinearRegression()
    try:
        model_free.fit(X, y)
    except Exception as e:
        return {"error": str(e)}, None

    k_free = float(model_free.coef_[0])
    b_free = float(model_free.intercept_)
    y_pred_free = model_free.predict(X)
    rmse_free = float(np.sqrt(mean_squared_error(y, y_pred_free)))
    r2_free = float(r2_score(y, y_pred_free))

    # zero-intercept regression
    model_zero = LinearRegression(fit_intercept=False)
    model_zero.fit(X, y)
    k_zero = float(model_zero.coef_[0])
    y_pred_zero = model_zero.predict(X)
    rmse_zero = float(np.sqrt(mean_squared_error(y, y_pred_zero)))
    r2_zero = float(r2_score(y, y_pred_zero))

    return {
        "n_points": n_points,
        "regression_free": {
            "k": k_free,
            "b": b_free,
            "rmse": rmse_free,
            "r2": r2_free
        },
        "regression_zero_intercept": {
            "k": k_zero,
            "rmse": rmse_zero,
            "r2": r2_zero
        }
    }

def get_travel_times_for_node(
    travel_time_df: pd.DataFrame,
    lon: float,
    lat: float,
    r_events: float,
    r_stations: float,
    dttm_from,
    dttm_to
):
    """
    r_events, r_stations — в метрах
    lon, lat — в градусах
    """

    earth = Earth()

    df = travel_time_df.copy()
    df['event_dttm'] = pd.to_datetime(df['event_dttm'])

    # ---------------------------
    # 1. ФИЛЬТР ПО ВРЕМЕНИ (сразу)
    # ---------------------------
    df = df[
        (df.event_dttm >= dttm_from) &
        (df.event_dttm <= dttm_to)
    ]
    if df.empty:
        return df

    # ---------------------------
    # 2. BBOX ДЛЯ СОБЫТИЙ
    # ---------------------------
    dlat_e = meters_to_deg_lat(r_events)
    dlon_e = meters_to_deg_lon(r_events, lat)

    df = df[
        (df.e_lat.between(lat - dlat_e, lat + dlat_e)) &
        (df.e_lon.between(lon - dlon_e, lon + dlon_e))
    ]
    if df.empty:
        return df

    # ---------------------------
    # 3. BBOX ДЛЯ СТАНЦИЙ
    # ---------------------------
    dlat_s = meters_to_deg_lat(r_stations)
    dlon_s = meters_to_deg_lon(r_stations, lat)

    df = df[
        (df.s_lat.between(lat - dlat_s, lat + dlat_s)) &
        (df.s_lon.between(lon - dlon_s, lon + dlon_s))
    ]
    if df.empty:
        return df

    # ---------------------------
    # 4. ТОЧНЫЕ ECEF РАССТОЯНИЯ
    # ---------------------------
    x0, y0, z0 = earth.geodetic_to_ecef(lat, lon, 0.0)

    xe, ye, ze = Earth.geodetic_to_ecef_vec(
        df.e_lat.values,
        df.e_lon.values,
        0.0,
        earth.a,
        earth.e_sq
    )

    xs, ys, zs = Earth.geodetic_to_ecef_vec(
        df.s_lat.values,
        df.s_lon.values,
        0.0,
        earth.a,
        earth.e_sq
    )

    dist_events = np.sqrt((xe-x0)**2 + (ye-y0)**2 + (ze-z0)**2)
    dist_stations = np.sqrt((xs-x0)**2 + (ys-y0)**2 + (zs-z0)**2)

    mask = (dist_events <= r_events) & (dist_stations <= r_stations)
    return df.loc[mask]

def run_experiment_vp_vs(
    region_nm, 
    n_steps_lat, 
    n_steps_lon,
    r_events,
    r_stations,
    dttm_from, 
    dttm_to, 
    n_steps_time
):
    experiment_nm = generate_name(['exp', dttm_from, dttm_to, region_nm])

    borders = get_region_borders(region_nm) 
    
    travel_times_df = get_travel_times_by_region(
        region_nm=region_nm,
        dttm_from=dttm_from,
        dttm_to=dttm_to)

    grid = make_grid_3d_time(
        lat_min=borders['lat_min'], lat_max=borders['lat_max'],
        lon_min=borders['lon_min'], lon_max=borders['lon_max'],
        n_steps_lat=n_steps_lat,
        n_steps_lon=n_steps_lon,
        dttm_from=dttm_from,
        dttm_to=dttm_to,
        n_steps_time=n_steps_time
    )

    # ------------ PROGRESS BAR ------------
    print(f"Running experiment '{experiment_nm}'...")

    rows = []
    node_number = 0
    for point in tqdm(grid, desc="Processing grid points", unit="pt", total=n_steps_lat*n_steps_lon*n_steps_time):

        node_df = get_travel_times_for_node(
            travel_time_df=travel_times_df,
            lat=point["lat"],
            lon=point["lon"],
            r_events=r_events,
            r_stations=r_stations,
            dttm_from=point["t_from"],
            dttm_to=point["t_to"])
        
        node_df['node_number'] = node_number
        node_number += 1

        params = estimate_vp_vs_ratio(node_df)   
        params = json.dumps(replace_nan(params))

        # if time_travel_df is None and node_df is not None and node_df.shape[0] >= 1:
        #     time_travel_df = node_df
        # elif time_travel_df is not None and node_df is not None:
        #     time_travel_df = pd.concat([time_travel_df, node_df], ignore_index=True)
        # else:
        #     pass  # no data case

        rows.append({
            "experiment_nm": experiment_nm,
            "lat": float(point["lat"]),
            "lon": float(point["lon"]),
            "r_events": float(r_events),
            "r_stations": float(r_stations),
            "t_from": str(point["t_from"]),
            "t_to": str(point["t_to"]),
            "params": json.loads(json.dumps(params, default=list))
        })
        # load_to_postgis(exp_df, 'experiments')

    exp_df = pd.DataFrame(rows)
    load_to_postgis(exp_df, 'experiments')
    # if time_travel_df is not None:
    #     time_travel_df['experiment_nm'] = experiment_nm
    #     time_travel_df = time_travel_df.drop(['event_dttm', 'ratio_ts_tp'], axis=1)
    #     load_to_postgres(time_travel_df, 'travel_times')
        
    return exp_df

def prepare_events_x_stations_data(df):
    # 1. Уникальные события
    events_df = (
        df[['event_id', 'e_lon', 'e_lat']]
        .drop_duplicates()
        .rename(columns={'e_lon': 'lon', 'e_lat': 'lat'})
    )
    events_df['type'] = 'event'
    
    # 2. Создаём уникальный station_id (так как station_nm + network_nm уникальны)
    df['station_id'] = df['station_nm'] + "_" + df['network_nm']
    
    # 3. Уникальные станции
    stations_df = (
        df[['station_id', 'station_nm', 'network_nm', 's_lon', 's_lat']]
        .drop_duplicates()
        .rename(columns={'s_lon': 'lon', 's_lat': 'lat'})
    )
    stations_df['type'] = 'station'
    
    # 4. Связи (линии)
    edges_df = df[['event_id', 'station_id']].drop_duplicates()
    
    return events_df, stations_df, edges_df

def get_unique_subset(df, uniqe_by, subset_columns=None):
    if subset_columns is None:
        subset_columns = uniqe_by
    unique_combinations = df.drop_duplicates(subset=uniqe_by)
    result = unique_combinations[subset_columns].reset_index(drop=True)
    return result

def build_grid_travel_times(grid, travel_time_df, events_df, stations_df):
    """
    Оптимизированная версия с использованием NumPy массивов.
    Возвращает разреженное представление данных.
    """
    from scipy.sparse import lil_matrix
    
    # Размерность сетки
    n_lat, n_lon, n_time = grid.shape
    
    # Словарь для хранения списков измерений
    grid_data = defaultdict(list)
    
    # Создаем индексы для быстрого поиска
    events_dict = {
        row['event_id']: (row['geo_ranges']['ranges'], row['time_range'])
        for _, row in events_df.iterrows()
    }
    
    stations_dict = {
        row['station_nm']: row['geo_ranges']['ranges']
        for _, row in stations_df.iterrows()
    }
    
    # Группируем travel_time_df по (event_id, station_nm) для эффективности
    grouped = travel_time_df.groupby(['event_id', 'station_nm'])
    
    for (event_id, station_nm), group in grouped:
        if event_id not in events_dict or station_nm not in stations_dict:
            continue
        
        event_geo_ranges, event_time_range = events_dict[event_id]
        station_geo_ranges = stations_dict[station_nm]
        
        # Конвертируем ranges в множества для быстрого пересечения
        event_nodes = _ranges_to_set(event_geo_ranges)
        station_nodes = _ranges_to_set(station_geo_ranges)
        common_nodes = event_nodes & station_nodes
        
        # Извлекаем все пары (delta_t_s, delta_t_p) для этой группы
        measurements = list(zip(group['delta_t_s'].values, group['delta_t_p'].values))
        
        # Добавляем измерения во все подходящие узлы
        it_start = event_time_range['it_start']
        it_end = event_time_range['it_end']
        
        for ix, iy in common_nodes:
            for it in range(it_start, it_end + 1):
                grid_data[(ix, iy, it)].extend(measurements)
    
    return dict(grid_data)

def build_final_experiment_df(grid, grid_travel_times):
    """
    Создает финальный датафрейм эксперимента с индексами узлов
    и всеми параметрами в JSON.
    
    Parameters:
    -----------
    grid : Grid3DTime
        Сетка
    grid_travel_times : dict
        Результат build_grid_travel_times_optimized
    
    Returns:
    --------
    pd.DataFrame с колонками:
        - ix, iy, it: индексы узла
        - params: JSON с координатами, временем и статистиками
    """
    rows = []
    
    for (ix, iy, it), measurements in grid_travel_times.items():
        # Получаем параметры узла из сетки
        node = grid[ix, iy, it]
        
        # Конвертируем измерения в numpy массив
        measurements_array = np.array(measurements)
        delta_t_s_values = measurements_array[:, 0]
        delta_t_p_values = measurements_array[:, 1]
        
        n_measurements = len(measurements)
        
        # Собираем все параметры в словарь
        params = {
            # Координаты и время узла
            'lon': float(node['lon']),
            'lat': float(node['lat']),
            'dttm': node['dttm'].isoformat(),
            
            # Количество измерений
            'n_measurements': n_measurements,
            
            # Статистики по delta_t_s
            'delta_t_s': {
                'mean': float(delta_t_s_values.mean()),
                'std': float(delta_t_s_values.std()),
                'min': float(delta_t_s_values.min()),
                'max': float(delta_t_s_values.max()),
            },
            
            # Статистики по delta_t_p
            'delta_t_p': {
                'mean': float(delta_t_p_values.mean()),
                'std': float(delta_t_p_values.std()),
                'min': float(delta_t_p_values.min()),
                'max': float(delta_t_p_values.max()),
            },
            
            # Линейная регрессия
            'linear_regression': {}
        }
        
        # Линейная регрессия: delta_t_p = slope * delta_t_s + intercept
        if n_measurements >= 2:
            try:
                lr_result = stats.linregress(delta_t_p_values, delta_t_s_values)
                params['linear_regression'] = {
                    'slope': float(lr_result.slope),
                    'intercept': float(lr_result.intercept),
                    'r_value': float(lr_result.rvalue),
                    'r_squared': float(lr_result.rvalue ** 2),
                    'p_value': float(lr_result.pvalue),
                    'stderr': float(lr_result.stderr),
                    'significant': bool(lr_result.pvalue < 0.05),
                }
            except Exception as e:
                params['linear_regression'] = {
                    'slope': None,
                    'intercept': None,
                    'r_value': None,
                    'r_squared': None,
                    'p_value': None,
                    'stderr': None,
                    'significant': False,
                    'error': str(e)
                }
        else:
            params['linear_regression'] = {
                'slope': None,
                'intercept': None,
                'r_value': None,
                'r_squared': None,
                'p_value': None,
                'stderr': None,
                'significant': False,
                'error': 'insufficient_data'
            }
        
        rows.append({
            'ix': ix,
            'iy': iy,
            'it': it,
            'params': params
        })
    
    # Создаем датафрейм
    df = pd.DataFrame(rows)
    
    # Сортируем по индексам для удобства
    df = df.sort_values(['it', 'ix', 'iy']).reset_index(drop=True)
    
    return df

def _ranges_to_set(ranges):
    """Вспомогательная функция: конвертирует ranges в множество (ix, iy)."""
    nodes = set()
    for ix, iy_start, iy_end in ranges:
        for iy in range(iy_start, iy_end + 1):
            nodes.add((ix, iy))
    return nodes

