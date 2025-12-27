import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import psycopg2
import pandas as pd
# from utilities import make_grid_2d
from components.graphics import plot_spatial_distribution, plot_delta_t, plot_spatial_distribution_series, plot_events_stations
from components.dwh import get_travel_times_by_region, get_region_borders
from instruments import prepare_events_x_stations_data, get_travel_times_for_node, get_unique_subset, build_grid_travel_times, build_final_experiment_df
from components.utilities import make_grid_3d_time, extract_param
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def plot_node_example():
    tt_df = get_travel_times_by_region('Kamchatka', '2010-01-01', '2010-12-31')
    print(tt_df)

    node_df = get_travel_times_for_node(tt_df, 160, 53, 75000, 500000, '2010-01-01', '2010-12-31')
    print(node_df)

    ev_df, st_df, ar_df = prepare_events_x_stations_data(node_df)
    plot_events_stations(ev_df, st_df, ar_df)

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

def _test_grid():
    grid = create_grid()
    # Событие
    e_lon, e_lat = 37.6, 55.7  # Москва
    radius = 500  # км

    # Компактный результат
    result = grid.find_nodes_in_radius(e_lon, e_lat, radius)
    print(f"Найдено узлов: {result['count']}")
    print(f"Компактное представление: {len(result['ranges'])} диапазонов")

    print(f"Диапозоны: {result['ranges']}")

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

def run_experiment_vp_vs(
        region_nm, 
        dttm_from, 
        dttm_to, 
        n_steps_lat, 
        n_steps_lon, 
        n_steps_time,
        r_events_km,
        r_stations_km,
        r_time_days):
    
    grid = create_grid(region_nm, dttm_from, dttm_to, n_steps_lat, n_steps_lon, n_steps_time)
    tt_df, events_df, stations_df = get_travel_times_stations_and_events(region_nm, dttm_from, dttm_to)
    events_df, stations_df = add_grid_ranges(grid, events_df, stations_df, r_events_km, r_stations_km, r_time_days)
    grid_travel_times = build_grid_travel_times(grid, tt_df, events_df, stations_df)
    result_df = build_final_experiment_df(grid, grid_travel_times)
    return result_df


if __name__ == '__main__':
    exp_params = {
        'region_nm': 'Kamchatka',
        'dttm_from': '2007-01-01', 
        'dttm_to': '2008-05-01', 
        'n_steps_lat': 45, 
        'n_steps_lon': 60, 
        'n_steps_time': 2,
        'r_events_km': 400,
        'r_stations_km': 550,
        'r_time_days': 500
    }

    res = run_experiment_vp_vs(
        exp_params['region_nm'], 
        exp_params['dttm_from'], 
        exp_params['dttm_to'], 
        exp_params['n_steps_lat'], 
        exp_params['n_steps_lon'], 
        exp_params['n_steps_time'],
        exp_params['r_events_km'],
        exp_params['r_stations_km'],
        exp_params['r_time_days'])
    
    res['lon'] = extract_param(res, 'lon')
    res['lat'] = extract_param(res, 'lat')
    res['n_measurements'] = extract_param(res, 'n_measurements')
    res['lr_slope'] = extract_param(res, ['linear_regression', 'slope'])

    res = res.loc[res['n_measurements'] >= 100]

    res = res.loc[(res['lon'] >= 156) & (res['lon'] <= 167) 
            & (res['lat'] >= 48.5) & (res['lat'] <= 58)
            & (res['lat'] >= res['lon']*0.86 - 89)
            & (res['lat'] <= res['lon']*0.86 - 82)]

    res = res.loc[res['it'] == 0]
    print(res)
    plot_spatial_distribution(res, 'lr_slope', 'spatial_test.png')