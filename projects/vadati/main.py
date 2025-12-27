import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import psycopg2
import pandas as pd
# from utilities import make_grid_2d
from components.graphics import plot_spatial_distribution, plot_delta_t, plot_spatial_distribution_series, plot_events_stations
from components.dwh import get_travel_times_by_region, get_region_borders, get_experiment
from instruments import run_experiment_vp_vs
from components.utilities import make_grid_3d_time, extract_param
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    exp_params = {
        'region_nm': 'Kamchatka',
        'dttm_from': '2007-01-01', 
        'dttm_to': '2011-12-31', 
        'n_steps_lat': 135, 
        'n_steps_lon': 180, 
        'n_steps_time': 60,
        'r_events_km': 70,
        'r_stations_km': 200,
        'r_time_days': 150
    }
    
    exp_df = get_experiment('exp_Kamchatka_2007-01-01_2011-12-31_20251227180807')

    print(exp_df.info())
    # res = run_experiment_vp_vs(exp_params)
    
    # res['lon'] = extract_param(res, 'lon')
    # res['lat'] = extract_param(res, 'lat')
    # res['n_measurements'] = extract_param(res, 'n_measurements')
    # res['lr_slope'] = extract_param(res, ['linear_regression', 'slope'])

    # res = res.loc[res['n_measurements'] >= 100]

    # res = res.loc[(res['lon'] >= 156) & (res['lon'] <= 167) 
    #         & (res['lat'] >= 48.5) & (res['lat'] <= 58)
    #         & (res['lat'] >= res['lon']*0.86 - 89)
    #         & (res['lat'] <= res['lon']*0.86 - 82)]

    # res = res.loc[res['it'] == 0]
    # print(res)
    # plot_spatial_distribution(res, 'lr_slope', 'spatial_test_3.png')