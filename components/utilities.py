from math import radians, degrees, sin, cos, sqrt, atan2, floor
import numpy as np
import time
import pandas as pd
from pathlib import Path
import glob
import json
import re
import os
import requests
import datetime
from haversine import haversine

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

class Grid3DTime:
    """3D пространственно-временная сетка с удобной индексацией."""
    
    def __init__(self, lat_min, lat_max, lon_min, lon_max,
                 n_steps_lat, n_steps_lon,
                 dttm_from, dttm_to, n_steps_time):
        self.n_steps_lat = n_steps_lat
        self.n_steps_lon = n_steps_lon
        self.n_steps_time = n_steps_time
        
        # Парсим даты
        if isinstance(dttm_from, str):
            dttm_from = datetime.datetime.strptime(dttm_from, "%Y-%m-%d")
        if isinstance(dttm_to, str):
            dttm_to = datetime.datetime.strptime(dttm_to, "%Y-%m-%d")
        
        self.dttm_from = dttm_from
        self.dttm_to = dttm_to
        
        # Предвычисляем сетку
        self.lat_arr = np.linspace(lat_min, lat_max, n_steps_lat)
        self.lon_arr = np.linspace(lon_min, lon_max, n_steps_lon)
        
        # Массив временных меток
        total_seconds = (self.dttm_to - self.dttm_from).total_seconds()
        self.time_arr = [
            self.dttm_from + datetime.timedelta(seconds=i * total_seconds / (self.n_steps_time - 1))
            for i in range(self.n_steps_time)
        ]
    
    def __getitem__(self, idx):
        """Доступ по индексам [x, y, t]."""
        if isinstance(idx, tuple) and len(idx) == 3:
            ix, iy, it = idx
            
            # Проверка границ
            if not (0 <= ix < self.n_steps_lat):
                raise IndexError(f"x индекс {ix} вне диапазона [0, {self.n_steps_lat})")
            if not (0 <= iy < self.n_steps_lon):
                raise IndexError(f"y индекс {iy} вне диапазона [0, {self.n_steps_lon})")
            if not (0 <= it < self.n_steps_time):
                raise IndexError(f"t индекс {it} вне диапазона [0, {self.n_steps_time})")
            
            return {
                'lat': self.lat_arr[ix],
                'lon': self.lon_arr[iy],
                'dttm': self.time_arr[it],
                'ix': ix,
                'iy': iy,
                'it': it
            }
        else:
            raise TypeError("Индекс должен быть кортежем из 3 элементов (x, y, t)")
    
    def iter_all(self):
        """Итератор по всем точкам сетки."""
        for it in range(self.n_steps_time):
            for ix in range(self.n_steps_lat):
                for iy in range(self.n_steps_lon):
                    yield self[ix, iy, it]
    
    def iter_time_slice(self, it):
        """Итератор по временному срезу."""
        for ix in range(self.n_steps_lat):
            for iy in range(self.n_steps_lon):
                yield self[ix, iy, it]
    
    def iter_spatial_point(self, ix, iy):
        """Итератор по временной эволюции одной точки."""
        for it in range(self.n_steps_time):
            yield self[ix, iy, it]

    def find_time_range(self, event_dttm, time_window_days):
        """
        Находит временной диапазон узлов сетки для события.
        
        Parameters:
        -----------
        event_dttm : datetime
            Время события
        time_window_hours : float
            Временное окно в каждую из сторон (в днях)
        
        Returns:
        --------
        dict с ключами:
            'it_start': int - начальный индекс времени
            'it_end': int - конечный индекс времени  
            'count': int - количество временных шагов
        """

        # Парсим дату если строка
        if isinstance(event_dttm, str):
            event_dttm = datetime.datetime.strptime(event_dttm, "%Y-%m-%d %H:%M:%S.%f")
        
        # Вычисляем временное окно
        delta = datetime.timedelta(days=time_window_days)
        time_min = event_dttm - delta
        time_max = event_dttm + delta
        
        # Находим индексы
        # Используем бинарный поиск по массиву time_arr
        it_start = 0
        it_end = self.n_steps_time - 1
        
        # Ищем начальный индекс
        for i, t in enumerate(self.time_arr):
            if t >= time_min:
                it_start = i
                break
        
        # Ищем конечный индекс
        for i in range(len(self.time_arr) - 1, -1, -1):
            if self.time_arr[i] <= time_max:
                it_end = i
                break
        
        # Ограничиваем границами сетки
        it_start = max(0, it_start)
        it_end = min(self.n_steps_time - 1, it_end)
        
        return {
            'it_start': it_start,
            'it_end': it_end,
            'count': it_end - it_start + 1
        }
    

    def find_nodes_in_radius(self, e_lon, e_lat, radius_km):
        """
        Находит узлы сетки внутри круга радиуса R от события.
        Использует библиотеку haversine для расчета расстояний.
        """
        # Оценка ограничивающего прямоугольника
        lat_degree_dist = 111.0
        lon_degree_dist = 111.0 * np.cos(np.radians(e_lat))
        
        delta_lat = radius_km / lat_degree_dist
        delta_lon = radius_km / lon_degree_dist if lon_degree_dist > 0 else 180
        
        ix_min = np.searchsorted(self.lat_arr, e_lat - delta_lat, side='left')
        ix_max = np.searchsorted(self.lat_arr, e_lat + delta_lat, side='right')
        iy_min = np.searchsorted(self.lon_arr, e_lon - delta_lon, side='left')
        iy_max = np.searchsorted(self.lon_arr, e_lon + delta_lon, side='right')
        
        ix_min = max(0, ix_min)
        ix_max = min(self.n_steps_lat, ix_max)
        iy_min = max(0, iy_min)
        iy_max = min(self.n_steps_lon, iy_max)
        
        ranges = []
        total_count = 0
        
        event_point = (e_lat, e_lon)  # haversine требует (lat, lon)
        
        for ix in range(ix_min, ix_max):
            lat = self.lat_arr[ix]
            iy_start = None
            
            for iy in range(iy_min, iy_max):
                lon = self.lon_arr[iy]
                grid_point = (lat, lon)
                
                dist = haversine(event_point, grid_point)
                
                if dist <= radius_km:
                    if iy_start is None:
                        iy_start = iy
                    iy_end = iy
                else:
                    if iy_start is not None:
                        ranges.append((ix, iy_start, iy_end))
                        total_count += (iy_end - iy_start + 1)
                        iy_start = None
            
            if iy_start is not None:
                ranges.append((ix, iy_start, iy_end))
                total_count += (iy_end - iy_start + 1)
        
        return {
            'ranges': ranges,
            'count': total_count
        }
    
    @property
    def shape(self):
        """Размерность сетки."""
        return (self.n_steps_lat, self.n_steps_lon, self.n_steps_time)


def make_grid_3d_time(lat_min, lat_max, lon_min, lon_max,
                      n_steps_lat, n_steps_lon,
                      dttm_from, dttm_to, n_steps_time):
    """Создает 3D пространственно-временную сетку."""
    return Grid3DTime(lat_min, lat_max, lon_min, lon_max,
                     n_steps_lat, n_steps_lon,
                     dttm_from, dttm_to, n_steps_time)
    
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
        name = name + '_' + timestr
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

def extract_param(df, param_path):
    """
    Извлекает параметр из JSON в отдельную колонку.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм с колонкой params
    param_path : str or list
        Путь к параметру, например 'lon' или ['linear_regression', 'slope']
    
    Returns:
    --------
    pd.Series с извлеченными значениями
    """
    if isinstance(param_path, str):
        return df['params'].apply(lambda x: x.get(param_path))
    else:
        # Для вложенных параметров
        def get_nested(d, path):
            for key in path:
                if d is None:
                    return None
                d = d.get(key)
            return d
        return df['params'].apply(lambda x: get_nested(x, param_path))
