# from dwh import get_all_events
# import numpy as np
# import pandas as pd


import numpy as np
import requests
import matplotlib.pyplot as plt

# Центральная точка
lat0 = 32.419499
lon0 = -115.305991

# Радиус и шаг сетки
radius_m = 5000       # 1 км
step_m = 50         # шаг 100 м

# Перевод метров в градусы
deg_per_m_lat = 1 / 111320
deg_per_m_lon = 1 / (111320 * np.cos(np.radians(lat0)))

# Массив смещений
offsets = np.arange(-radius_m, radius_m + step_m, step_m)
lats = lat0 + offsets * deg_per_m_lat
lons = lon0 + offsets * deg_per_m_lon
grid_lats, grid_lons = np.meshgrid(lats, lons)

def get_elevations(lat_array, lon_array):
    coords = [f"{lat},{lon}" for lat, lon in zip(lat_array.flatten(), lon_array.flatten())]
    elevations = []
    batch_size = 100
    for i in range(0, len(coords), batch_size):
        batch = "|".join(coords[i:i + batch_size])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={batch}"
        r = requests.get(url)
        data = r.json()
        elevations.extend([pt['elevation'] for pt in data['results']])
    return np.array(elevations).reshape(lat_array.shape)

Z = get_elevations(grid_lats, grid_lons)

# Рисуем тепловую карту
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
           origin='lower', cmap='terrain', aspect='auto')

plt.colorbar(label='Высота, м')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title('Рельеф вокруг точки (32.419499°N, 115.305991°W)')
plt.show()


plt.savefig('geo_5.png')





# min_lon, max_lon = -115.9, -115.3
# min_lat, max_lat = 32.50, 32.80

# df = get_all_events('2010-01-01', '2020-01-01', min_lon, min_lat, max_lon, max_lat)

# print(df)


# n_lon = 5   # число разбиений по долготе
# n_lat = 5   # число разбиений по широте


# lon_edges = np.linspace(min_lon, max_lon, n_lon+1)
# lat_edges = np.linspace(min_lat, max_lat, n_lat+1)

# sector_centers = []
# for i in range(n_lon):
#     for j in range(n_lat):
#         sector_centers.append({
#             'sector_idx': (i,j),
#             'center_lon': (lon_edges[i] + lon_edges[i+1])/2,
#             'center_lat': (lat_edges[j] + lat_edges[j+1])/2
#         })
# sector_df = pd.DataFrame(sector_centers)


# def find_sector(lon, lat):
#     lon_bin = np.digitize(lon, lon_edges) - 1
#     lat_bin = np.digitize(lat, lat_edges) - 1
#     if 0 <= lon_bin < n_lon and 0 <= lat_bin < n_lat:
#         return (lon_bin, lat_bin)
#     else:
#         return None

# df['sector_idx'] = df.apply(lambda r: find_sector(r['event_lon'], r['event_lat']), axis=1)
# df = df[df['sector_idx'].notnull()]


# df['event_dttm'] = pd.to_datetime(df['event_dttm'])
# df['year_month'] = df['event_dttm'].dt.to_period("Y")

# all_months = pd.period_range('2010-01', '2019-12', freq='Y')

# counts = (
#     df.groupby(['sector_idx','year_month'])
#       .size().reset_index(name='cnt')
# )


# sector_idx_all = sector_df['sector_idx'].unique()
# multi_index = pd.MultiIndex.from_product([sector_idx_all, all_months],
#                                          names=['sector_idx','year_month'])
# counts_all = counts.set_index(['sector_idx','year_month']).reindex(multi_index, fill_value=0).reset_index()


# minmax = counts_all.groupby('sector_idx')['cnt'].agg(min_events_month='min',
#                                                      max_events_month='max').reset_index()

# result = pd.merge(minmax, sector_df, on='sector_idx', how='left')
# result = result[['center_lon','center_lat','min_events_month','max_events_month']]

# print(result)