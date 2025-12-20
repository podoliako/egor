from utilities import Point, add_cell_coords, pd, get_tables_summary
from raytracing import \
      calculate_time_martix, load_time_matrix, get_ray_path, calculate_time_martix_event,\
      calculate_time_martix_station, load_time_matrix_coords, get_arrival_time

from dwh import get_event, get_station, get_arrivals_by_event
from graphics import plot_ray_path, plot_wave_front, plot_fat_ray, plot_hist, simple_plot, simple_scatter, plot_and_compare_stddev_by_date
from model import VelocityModel
from aggregates import add_delta_time, calculate_arrival_table, get_events_inside_model, compute_tables
import warnings
warnings.filterwarnings('ignore')


# CBX	IR	POINT (-116.6636 32.3131)
# COA	IR	POINT (-115.123 32.8635)
# COK	IR	POINT (-115.727 32.8492)
# CRR	IR	POINT (-115.96917 32.88683)
# ECBX	IR	POINT (-115.0518 31.4725)
# ERPC	IR	POINT (-115.663 32.7435) (1, -59, 270) 2012-01-10 11:55:19.040
# IKP	IR	POINT (-116.10948 32.65012)
# SPIG	IR	POINT (-115.46581 31.04581)
# WESC	IR	POINT (-115.73161 32.75903)
# YUH	IR	POINT (-115.923 32.6477)

# POINT (-115.638 32.5)	3000.0	2012-01-10 11:55:14.100 (30, -36, 0)

# -------------------------------------------------------------
# 601626669	earthquake	known	POINT (-115.5352 33.0466)	2339.2	2012-08-26 19:31:22.370 (23, 0, 0)
# COK	IR	POINT (-115.727 32.8492)	Pg	2012-08-26 19:31:28.150 (1, -180, -219)
# 601626669	ERPC	IR	POINT (-115.663 32.7435)	2012-08-26 19:31:29.430

# top_mid = Point(0, -115.5352, 33.0466)

# model = VelocityModel(top_mid_point=top_mid,
#                             cube_side=100,
#                             n_north=3001,
#                             n_east=3001,
#                             n_depth=25)

# model.load_cvh()

# vp = model.vp

# print(vp)

# model.save('big_model.npz')

# model = VelocityModel.load('big_model_21-07.npz')
# model = VelocityModel.load('cube_test.npz')


# print(model.find_cell_coords(0,  32.7435, -115.663))

# print(min(model.vp.reshape(model.n_depth*model.n_east*model.n_north)))
# cords = model.find_cell_coords(target_depth=2339.2, target_lon=-115.638, target_lat=33.0466)
# print(cords)

# cords = model.find_cell_coords(target_depth=0, target_lon=-115.727, target_lat=32.8492)
# print(cords)

# print(model.vp[10][1500-180][1500-219])
# print(model.)
# travel_time = get_travel_times(model.vp, (23, 1500, 1500), model.cube_side)

# np.savez('travel_time_big_____.npz', height=travel_time)

# M = np.load('times/travel_time_big.npz')
# MM = M['height']

# print(MM)
# print(MM[1][1500-120][1500-336])

# path = get_ray_path(MM, (23, 1500, 1500), (3, 1500-1000, 1500+1000))

# for p in path:
#     print(round(p[0], 2), round(p[1], 2), round(p[2], 2))

# print(path)

# plot_ray_path(path, (23, 1500, 1500), (3, 1500-1000, 1500+1000))

# print(path)


# model = VelocityModel.from_event(14771060, 50, 4001)




# model = VelocityModel.load('20250825203458_100_150_1851_1851')



# calculate_time_martix_station('MBIG', 'IR', model, 'P')
# calculate_time_martix(14771060, model, 'P')
# depth, lon, lat = get_event(14771060)
# source_coords = model.find_cell_coords(depth, lat, lon)

# depth, lon, lat = get_station('MBIG', 'IR')
# receiver_coords = model.find_cell_coords(depth, lat, lon)

# time_rx = load_time_matrix_coords(receiver_coords, model, 'P')
# time_sx = load_time_matrix(14771060, model, 'P')

# path = get_ray_path(time_sx, source_coords, receiver_coords)
# plot_ray_path(path, source_coords, receiver_coords)

# print(times[reciever_coords[0], reciever_coords[1], reciever_coords[2]])

# plot_fat_ray(time_sx, time_rx, source_coords, receiver_coords, 0.02)


# plot_wave_front(times, 15, 0.01)



# print(times[coords[0], coords[1], coords[2]])

# print(coords)

# path = get_ray_path(times, coords, (0,0,0))

# plot_ray_path(path=path, start_point=coords, end_point=(0,0,0))


# print(m[54])

# model.save()

# 14771060
# 9443229
# z, x, y = model.find_cell_coords(0,  -115.92, 32.647)

# print(z, x, y)




# time = get_arrival_time(14771060, model, (1, 1196, 1171), 'P')

# calculate_arrival_table(model, 13479474, 'P')
# print(df)
# plot_hist(df['deviation_fmm'])

# df = get_events_inside_model(model, 3.0, 11)
# print(df)

# cords = model.find_cell_coords(6000, 32.127, -115.49)
# print(cords)

# compute_tables(model, 3, 11, 'S')
# print(model.top_mid_point.lon, model.top_mid_point.lat)


path = 'models/20250825203458_100_150_1851_1851/fmm_times/P_wawes'
df = get_tables_summary(path)
print(df)
x = pd.to_datetime(df['event_dt']).dt.floor('D')
y = df['diviation_std']

plot_and_compare_stddev_by_date(x, y)

# print(df['deviation_fmm_x_mean'].std())



# top_point = Point(0, -115.2, 32.4)

# model = VelocityModel(top_mid_point=top_point, cube_side=100, n_depth=150, n_east=1851, n_north=1851)
# model.save()
# model.load_cvh()
# model.save()