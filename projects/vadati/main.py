import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import psycopg2
import pandas as pd
# from utilities import make_grid_2d
from components.graphics import plot_spatial_distribution, plot_delta_t, plot_spatial_distribution_series, plot_events_stations
from components.dwh import get_travel_times_by_region
from instruments import run_experiment_vp_vs, prepare_events_x_stations_data, get_travel_times_for_node, estimate_vp_vs_ratio
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



# df = get_kamchatka_ts_tv()

# print(df)
# 



# df = get_tp_ts(-116, 34, 1000000, 2000000, '2015-01-01', '2021-02-01')
# print(df)
# df = df.loc[(df['delta_t_s'].dt.seconds < 500) or (df['delta_t_s'].dt.seconds > 0)]
# plot_delta_t(df)
# res = estimate_ts_tp_ratio(-116, 34, 100000, 200000, '2010-01-01', '2012-02-01')
# print(res)

# exp_df = run_experiment_vp_vs('Kamchatka', 60, 45, 400000, 550000, '2007-01-01', '2022-12-31', 12)

# exp_nm = '20251214105253_exp_2007-01-01_2022-12-31_Kamchatka'
# df = get_experiment(exp_nm)
# df = df.loc[(df['node_lon'] >= 156) & (df['node_lon'] <= 167) 
#             & (df['node_lat'] >= 48.5) & (df['node_lat'] <= 58)
#             & (df['node_lat'] >= df['node_lon']*0.86 - 89)
#             & (df['node_lat'] <= df['node_lon']*0.86 - 82)]
# df = df.loc[(df['n_points'] >= 100)]
# print(df)
# plot_spatial_distribution_series(df, 'k_free', exp_nm)

# df = df.loc[(df['n_points'] >= 5000)]
# plot_spatial_distribution(df, 'k_zero_intercept', 'spatial_fixe.png')

# df = get_tp_ts(158.85, 54.047, 400000, 550000, '2007-01-01', '2008-05-01')
# print(df)

# plot_delta_t(df)

def plot_node_example():
    tt_df = get_travel_times_by_region('Kamchatka', '2010-01-01', '2010-12-31')
    print(tt_df)

    node_df = get_travel_times_for_node(tt_df, 160, 53, 75000, 500000, '2010-01-01', '2010-12-31')
    print(node_df)

    ev_df, st_df, ar_df = prepare_events_x_stations_data(node_df)
    plot_events_stations(ev_df, st_df, ar_df)

if __name__ == '__main__':
    plot_node_example()
