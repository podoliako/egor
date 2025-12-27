from components.dwh import pd, load_to_postgres, load_to_postgis, get_travel_times_by_region, get_region_borders
from components.utilities import make_grid_3d_time, generate_name, Grid3DTime

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import math
import json


def _ranges_to_set(ranges):
    """Вспомогательная функция: конвертирует ranges в множество (ix, iy)."""
    nodes = set()
    for ix, iy_start, iy_end in ranges:
        for iy in range(iy_start, iy_end + 1):
            nodes.add((ix, iy))
    return nodes

def get_unique_subset(df, uniqe_by, subset_columns=None):
    if subset_columns is None:
        subset_columns = uniqe_by
    unique_combinations = df.drop_duplicates(subset=uniqe_by)
    result = unique_combinations[subset_columns].reset_index(drop=True)
    return result

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

def get_travel_times_stations_and_events(region_nm, dttm_from, dttm_to):
    tt_df = get_travel_times_by_region(region_nm, dttm_from, dttm_to)
    events_df = get_unique_subset(tt_df, ['event_id'], ['event_id', 'e_lon', 'e_lat', 'event_dttm'])
    stations_df = get_unique_subset(tt_df,['station_nm', 'network_nm'], ['station_nm', 'network_nm', 's_lon', 's_lat']) 
    return tt_df, events_df, stations_df

def create_grid(region_nm, dttm_from, dttm_to, n_steps_lat, n_steps_lon, n_steps_time):
    borders = get_region_borders(region_nm)
    grid = make_grid_3d_time(
        lat_min=borders['lat_min'], lat_max=borders['lat_max'],
        lon_min=borders['lon_min'], lon_max=borders['lon_max'],
        n_steps_lat=n_steps_lat, n_steps_lon=n_steps_lon,
        dttm_from=dttm_from,
        dttm_to=dttm_to,
        n_steps_time=n_steps_time
    )
    return grid

def add_grid_ranges(grid, events_df, stations_df, r_events_km, r_stations_km, r_time_days):
    events_df['geo_ranges'] = events_df.apply(
    lambda row: grid.find_nodes_in_radius(row['e_lon'], row['e_lat'], r_events_km),
    axis=1)

    events_df['time_range'] = events_df.apply(
    lambda row: grid.find_time_range(row['event_dttm'], r_time_days), 
    axis=1)

    stations_df['geo_ranges'] = stations_df.apply(
    lambda row: grid.find_nodes_in_radius(row['s_lon'], row['s_lat'], r_stations_km),
    axis=1)

    return events_df, stations_df

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

def run_experiment_vp_vs(exp_params):
    region_nm = exp_params['region_nm'] 
    dttm_from = exp_params['dttm_from'] 
    dttm_to = exp_params['dttm_to'] 
    n_steps_lat = exp_params['n_steps_lat'] 
    n_steps_lon = exp_params['n_steps_lon'] 
    n_steps_time = exp_params['n_steps_time']
    r_events_km = exp_params['r_events_km']
    r_stations_km = exp_params['r_stations_km']
    r_time_days = exp_params['r_time_days']
    
    experiment_nm = generate_name(['exp', region_nm, dttm_from, dttm_to])

    dwh_log = pd.DataFrame([{'experiment_nm': experiment_nm, 'initial_params': json.dumps(exp_params)}])
    load_to_postgres(dwh_log, 'experiments')

    grid = create_grid(region_nm, dttm_from, dttm_to, n_steps_lat, n_steps_lon, n_steps_time)
    tt_df, events_df, stations_df = get_travel_times_stations_and_events(region_nm, dttm_from, dttm_to)
    events_df, stations_df = add_grid_ranges(grid, events_df, stations_df, r_events_km, r_stations_km, r_time_days)
    grid_travel_times = build_grid_travel_times(grid, tt_df, events_df, stations_df)
    result_df = build_final_experiment_df(grid, grid_travel_times)

    dwh_result = result_df.copy()
    dwh_result['experiment_nm'] = experiment_nm
    dwh_result['params'] = dwh_result['params'].apply(lambda x: json.dumps(x))
    load_to_postgres(dwh_result, 'experiment_nodes')

    return result_df


