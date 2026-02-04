from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from components.graphics import plot_spatial_distribution, plot_delta_t, create_gif_from_pngs, plot_events_stations
from components.dwh import get_travel_times_by_region, get_region_borders, get_experiment
from components.utilities import make_grid_3d_time, extract_param

from instruments import run_experiment_vp_vs

import warnings
warnings.filterwarnings('ignore')

def make_picture():    
    exp_df = get_experiment('exp_Kamchatka_2007-01-01_2022-12-31_20251227220012')
    exp_df = exp_df.loc[exp_df['n_measurements'] >= 125]
    frame_df = exp_df.loc[exp_df['it'] == 20]
    plot_spatial_distribution(frame_df, 
                              'slope', 
                              vmin=1.5, 
                              vmax=1.9, 
                              lon_bounds=(156, 165), 
                              lat_bounds=(50,57), 
                              color_bar=False, 
                              title=f'Кадр: {20}', 
                              subdir='picture_test')

def make_pictures():    
    exp_df = get_experiment('exp_Kamchatka_2007-01-01_2022-12-31_20251227220012')
    exp_df = exp_df.loc[exp_df['n_measurements'] >= 125]

    for i in range(192):
        frame_df = exp_df.loc[exp_df['it'] == i]
        plot_spatial_distribution(frame_df, 'slope', vmin=1.5, vmax=1.9, lon_bounds=(156, 165), lat_bounds=(50,57), color_bar=False, title=f'Кадр: {i}', subdir='animation_test_3')
    
def make_gif():
    folder = '/mnt/disk01/egor/projects/vadati/pictures/animation_test_2'
    create_gif_from_pngs(folder, output_path='animation_2.gif')


exp_params = {
        'region_nm': 'Kamchatka',
        'dttm_from': '2007-01-01', 
        'dttm_to': '2022-12-31', 
        'n_steps_lat': 135, 
        'n_steps_lon': 180, 
        'n_steps_time': 192,
        'r_events_km': 70,
        'r_stations_km': 200,
        'r_time_days': 150
    }

if __name__ == '__main__':
    # make_picture()
    # res = run_experiment_vp_vs(exp_params)
    make_pictures()
    # make_gif()


    # res['lon'] = extract_param(res, 'lon')
    # res['lat'] = extract_param(res, 'lat')
    # res['n_measurements'] = extract_param(res, 'n_measurements')
    # res['lr_slope'] = extract_param(res, ['linear_regression', 'slope'])

    

    # res = res.loc[(res['lon'] >= 156) & (res['lon'] <= 167) 
    #         & (res['lat'] >= 48.5) & (res['lat'] <= 58)
    #         & (res['lat'] >= res['lon']*0.86 - 89)
    #         & (res['lat'] <= res['lon']*0.86 - 82)]

    # res = res.loc[res['it'] == 0]
    # print(res)
    # plot_spatial_distribution(res, 'lr_slope', 'spatial_test_3.png')