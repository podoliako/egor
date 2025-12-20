from math import radians, degrees, sin, cos, sqrt, atan2, floor
import numpy as np
import time
import pandas as pd
from pathlib import Path
import glob
import re
import os
import requests
import datetime

class Point:
    def __init__(self, depth, lon, lat):
        self.lon = lon
        self.lat = lat
        self.depth = depth

class Earth:
    def __init__(self):
        # WGS84 параметры
        self.a = 6378137.0                    # Большая полуось (экваториальный радиус)
        self.f = 1 / 298.257223563            # Сплюснутость
        self.b = self.a * (1 - self.f)        # Малая полуось
        self.e_sq = self.f * (2 - self.f)     # Эксцентриситет в квадрате

    def geodetic_to_ecef(self, lat, lon, h):
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        N = self.a / sqrt(1 - self.e_sq * sin(lat_rad) ** 2)

        x = (N + h) * cos(lat_rad) * cos(lon_rad)
        y = (N + h) * cos(lat_rad) * sin(lon_rad)
        z = (N * (1 - self.e_sq) + h) * sin(lat_rad)

        return x, y, z
    
    def geodetic_to_ecef_vec(lat, lon, h, a, e_sq):
        lat = np.radians(lat)
        lon = np.radians(lon)

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        N = a / np.sqrt(1 - e_sq * sin_lat**2)

        x = (N + h) * cos_lat * np.cos(lon)
        y = (N + h) * cos_lat * np.sin(lon)
        z = (N * (1 - e_sq) + h) * sin_lat
        return x, y, z
    
    def ecef_to_geodetic(self, x, y, z):
        # Метод Bowring
        eps = self.e_sq / (1.0 - self.e_sq)
        p = sqrt(x**2 + y**2)
        theta = atan2(z * self.a, p * self.b)
        lon = atan2(y, x)
        lat = atan2(z + eps * self.b * sin(theta)**3,
                        p - self.e_sq * self.a * cos(theta)**3)
        N = self.a / sqrt(1 - self.e_sq * sin(lat)**2)
        h = p / cos(lat) - N

        return degrees(lat), degrees(lon), h
    
    def enu_to_ecef(self, dx, dy, dz, ref_lat, ref_lon, ref_h):
        lat_rad = radians(ref_lat)
        lon_rad = radians(ref_lon)

        t = [
            [-sin(lon_rad), -sin(lat_rad) * cos(lon_rad), cos(lat_rad) * cos(lon_rad)],
            [cos(lon_rad), -sin(lat_rad) * sin(lon_rad), cos(lat_rad) * sin(lon_rad)],
            [0, cos(lat_rad), sin(lat_rad)]
        ]

        x_offset = t[0][0]*dx + t[0][1]*dy + t[0][2]*dz
        y_offset = t[1][0]*dx + t[1][1]*dy + t[1][2]*dz
        z_offset = t[2][0]*dx + t[2][1]*dy + t[2][2]*dz

        x0, y0, z0 = self.geodetic_to_ecef(ref_lat, ref_lon, ref_h)
        return x0 + x_offset, y0 + y_offset, z0 + z_offset
    
    def ecef_to_enu(self, x, y, z, ref_lat, ref_lon, ref_h):
        lat_rad = radians(ref_lat)
        lon_rad = radians(ref_lon)

        x0, y0, z0 = self.geodetic_to_ecef(ref_lat, ref_lon, ref_h)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        # Транспонированная матрица вращения
        t = [
            [-sin(lon_rad),                cos(lon_rad),               0],
            [-sin(lat_rad) * cos(lon_rad), -sin(lat_rad) * sin(lon_rad), cos(lat_rad)],
            [cos(lat_rad) * cos(lon_rad),  cos(lat_rad) * sin(lon_rad),  sin(lat_rad)]
        ]

        e = t[0][0]*dx + t[0][1]*dy + t[0][2]*dz
        n = t[1][0]*dx + t[1][1]*dy + t[1][2]*dz
        u = t[2][0]*dx + t[2][1]*dy + t[2][2]*dz

        return e, n, u
    
def get_elevation(points):
    req_string = ''
    for p in points:
        appendix = f'{str(p.lat)},{str(p.lon)}|'
        req_string += appendix
    req_string = req_string[:-1]

    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={req_string}')
    
    response = requests.get(query).json()
    result = [r['elevation'] for r in response['results']]
    return result

def generate_cells(top_mid_point, cube_side, n_x, n_y, n_depth):
    Earth_model = Earth()
    # Проверка на нечетность
    if n_x % 2 == 0:
        raise ValueError("n_x must be odd")
    if n_y % 2 == 0:
        raise ValueError("n_y must be odd")
    
    # Рассчитываем количество шагов в каждом направлении от центра
    half_north = (n_x - 1) // 2
    half_east = (n_y - 1) // 2
    
    depths = []
    lats = []
    lons = []
    
    # Проходим по всем слоям: от верхнего в глубину
    for dz in range(n_depth):
        # Смещение вниз (в метрах)
        down_offset = -dz * cube_side  # отрицательное, так как вниз
        
        # Проходим по восток-запад: от -half_east до +half_east
        for dx in range(-half_east, half_east + 1):

            # Проходим по север-юг: от -half_north до +half_north
            for dy in range(-half_north, half_north + 1):
            
                # Рассчитываем смещение в ENU
                local_east = dx * cube_side
                local_north = dy * cube_side
                local_up = down_offset
                
                # Преобразуем ENU в ECEF
                x, y, z = Earth_model.enu_to_ecef(
                    local_east, local_north, local_up,
                    top_mid_point.lat, top_mid_point.lon, -top_mid_point.depth
                )
                
                # Преобразуем ECEF обратно в геодезические координаты
                lat, lon, alt = Earth_model.ecef_to_geodetic(x, y, z)
                depth = -alt
                
                # Сохраняем результаты
                depths.append(round(depth, 5))
                lats.append(round(lat, 5))
                lons.append(round(lon, 5))
    
    return depths, lons, lats


def generate_name(params, with_time=True):
    name = ''
    for param in params:
        name += str(param) + '_'

    name = name[:-1]


    if with_time is True:
        timestr = time.strftime("%Y%m%d%H%M%S")
        name = timestr + '_' + name
    return name

def find_latest_npz(folder):
    files = glob.glob(f'{folder}/*.npz')
    version_files = []
    for file in files:
        # Extract version number using regular expression
        match = re.search(r'_v(\d+)\.npz$', file)
        if match:
            version = int(match.group(1))
            version_files.append((version, file))
    
    if not version_files:
        return None
    version_files.sort()
    latest_version, latest_file = version_files[-1]
    return latest_file


def add_cell_coords(model, df):
    rows = df.apply(lambda row: pd.Series(model.find_cell_coords(0, row['lat'], row['lon'])), axis=1)
    df[['z', 'x', 'y']] = rows
    df = df.dropna()
    df[['z', 'x', 'y']] = df[['z', 'x', 'y']].astype(int)
    return df

def add_cell_coords(model, df):
    rows = df.apply(lambda row: pd.Series(model.find_cell_coords(0, row['lat'], row['lon'])), axis=1)
    df[['z', 'x', 'y']] = rows
    df = df.dropna()
    df[['z', 'x', 'y']] = df[['z', 'x', 'y']].astype(int)
    return df


def get_tables_summary(folder_path):  
    summary_list = []
    csv_files = glob.glob(os.path.join(folder_path, '*table_fmm*.csv'))

    for file in csv_files:
        df = pd.read_csv(file)
        if 'arrival_dttm' in df.columns and 'deviation_fmm_x_mean' in df.columns:
            event_dt = df['arrival_dttm'].iloc[0]
            # mean = df['fmm_event_dttm'].head(10).mean()
            diviation_std = (pd.to_datetime(df['fmm_event_dttm']).astype(int) / 10**9).std()
            summary_list.append({'event_dt': event_dt, 'diviation_std': diviation_std})
        else:
            print(f"Пропущен файл {file}, нет нужных столбцов")
    
    summary_df = pd.DataFrame(summary_list)
    return summary_df


def make_grid_3d_time(lat_min, lat_max, lon_min, lon_max,
                      n_steps_lat, n_steps_lon,
                      dttm_from, dttm_to, n_steps_time):
    
    n_steps_time += 1
    dttm_from = datetime.datetime.strptime(dttm_from, "%Y-%m-%d")
    dttm_to = datetime.datetime.strptime(dttm_to, "%Y-%m-%d")

    Earth_model = Earth()

    lat0 = (lat_min + lat_max) / 2
    lon0 = (lon_min + lon_max) / 2
    h0 = 0

    corners = [(lat_min, lon_min),
               (lat_min, lon_max),
               (lat_max, lon_min),
               (lat_max, lon_max)]

    enu_points = []
    for lat, lon in corners:
        x, y, z = Earth_model.geodetic_to_ecef(lat, lon, h0)
        e, n, _ = Earth_model.ecef_to_enu(x, y, z, lat0, lon0, h0)
        enu_points.append((e, n))

    east_vals  = [p[0] for p in enu_points]
    north_vals = [p[1] for p in enu_points]

    east_min,  east_max  = min(east_vals),  max(east_vals)
    north_min, north_max = min(north_vals), max(north_vals)

    step_east  = (east_max  - east_min)  / (n_steps_lat - 1)
    step_north = (north_max - north_min) / (n_steps_lon - 1)

    # список времён
    times = [
        dttm_from + i * (dttm_to - dttm_from) / (n_steps_time - 1)
        for i in range(n_steps_time)
    ]

    def generator():
        for it in range(n_steps_time - 1):
            t_from = times[it]
            t_to   = times[it + 1]

            for ix in range(n_steps_lat):
                e = east_min + ix * step_east
                for iy in range(n_steps_lon):
                    n = north_min + iy * step_north

                    x, y, z = Earth_model.enu_to_ecef(e, n, 0, lat0, lon0, h0)
                    lat, lon, _ = Earth_model.ecef_to_geodetic(x, y, z)

                    yield {
                        "t_from": t_from,
                        "t_to": t_to,
                        "ix": ix,
                        "iy": iy,
                        "lat": lat,
                        "lon": lon
                    }

    return generator()