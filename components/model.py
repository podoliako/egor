import numpy as np
import pandas as pd
from datetime import timedelta
from utilities import Point, Earth, Path, generate_cells, generate_name, find_latest_npz
import ucvm
from dwh import get_event
from math import radians, sin, cos


class VelocityModel:
    def __init__(self, top_mid_point, cube_side, n_north, n_east, n_depth):
        self.top_mid_point = top_mid_point
        self.cube_side = cube_side
        self.n_north = n_north
        self.n_east = n_east
        self.n_depth = n_depth

        self.model_name = generate_name([cube_side, n_depth, n_north, n_east])
        self.model_version = 0

        self.vp = None
        self.vs = None

    def _set_velocity(self, vp_flat, vs_flat):
        expected_size = self.n_depth * self.n_east * self.n_north
        assert len(vp_flat) == expected_size, f"vp size {len(vp_flat)} != expected {expected_size}"
        assert len(vs_flat) == expected_size, f"vs size {len(vs_flat)} != expected {expected_size}"

        # Порядок итерирования: глубина (depth), долгота (lon), широта (lat)
        self.vp = np.reshape(vp_flat, (self.n_depth, self.n_east, self.n_north)).astype(np.float32)
        self.vs = np.reshape(vs_flat, (self.n_depth, self.n_east, self.n_north)).astype(np.float32)

    def load_cvh(self):
        depths, lons, lats = generate_cells(
            top_mid_point=self.top_mid_point,
            cube_side=self.cube_side,
            n_north=self.n_north,
            n_east=self.n_east,
            n_depth=self.n_depth
        )

        vp, vs = ucvm.get_velocities(depths=depths, lons=lons, lats=lats, cube_side=self.cube_side)
        self._set_velocity(vp, vs)

    def save(self):
        self.model_version += 1
        folder = f'models/{self.model_name}'
        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(folder+'/time_matrices').mkdir(parents=True, exist_ok=True)
        Path(folder+'/fmm_times').mkdir(parents=True, exist_ok=True)

        file_name = f'{folder}/model_v{str(self.model_version)}.npz'

        np.savez_compressed(file_name,
            mid_depth = self.top_mid_point.depth,
            mid_lon = self.top_mid_point.lon,
            mid_lat = self.top_mid_point.lat,
            cube_side = self.cube_side,
            n_north = self.n_north,
            n_east = self.n_east,
            n_depth = self.n_depth,
            model_name = np.array(self.model_name, dtype='<U27'),
            model_version = self.model_version,
            vp = self.vp,
            vs = self.vs
        )

    def find_cell_coords(self, target_depth, target_lat, target_lon, for_np=True):
        n_north = self.n_north
        n_east = self.n_east
        n_depth = self.n_depth
        top_mid_point = self.top_mid_point
        cube_side = self.cube_side

        planet = Earth()
            
        target_x, target_y, target_z = planet.geodetic_to_ecef(target_lat, target_lon, -target_depth)
        ref_x, ref_y, ref_z = planet.geodetic_to_ecef(top_mid_point.lat, top_mid_point.lon, -top_mid_point.depth)
        
        delta_x = target_x - ref_x
        delta_y = target_y - ref_y
        delta_z = target_z - ref_z
        
        lat_rad = radians(top_mid_point.lat)
        lon_rad = radians(top_mid_point.lon)
        
        east = -sin(lon_rad)*delta_x + cos(lon_rad)*delta_y
        north = -sin(lat_rad)*cos(lon_rad)*delta_x - sin(lat_rad)*sin(lon_rad)*delta_y + cos(lat_rad)*delta_z
        up = cos(lat_rad)*cos(lon_rad)*delta_x + cos(lat_rad)*sin(lon_rad)*delta_y + sin(lat_rad)*delta_z
        
        dx = round(east / cube_side)
        dy = round(north / cube_side)
        dz = round(-up / cube_side)
    
        half_east = (n_east - 1) // 2
        half_north = (n_north - 1) // 2
        
        if (abs(dx) > half_east or 
            abs(dy) > half_north or 
            dz < 0 or dz >= n_depth):
            return (np.nan, np.nan, np.nan)
        
        if for_np is True:
            dx += half_east
            dy += half_north

        return (int(dz), int(dx), int(dy))
    
    @classmethod
    def load(cls, model_name, version='latest'):
        file_name = None
        if version == 'latest':
            file_name = find_latest_npz(f'models/{model_name}')
        else:
            file_name = f'models/{model_name}/model_v{version}.npz'

        loaded_data = np.load(file_name, allow_pickle=True)
        
        instance = cls(
            top_mid_point=Point(depth=loaded_data['mid_depth'],
                                lon=loaded_data['mid_lon'],
                                lat=loaded_data['mid_lat']),
            cube_side=loaded_data['cube_side'],
            n_north=loaded_data['n_north'],
            n_east=loaded_data['n_east'],
            n_depth=loaded_data['n_depth'],
        )
        
        instance.model_name=model_name
        instance.model_version=int(loaded_data['model_version'])


        instance.vp = loaded_data['vp']
        instance.vs = loaded_data['vs']
        
        loaded_data.close()
        return instance

    @classmethod
    def from_event(cls, event_id, cube_side, n_horizontal):
        n_east = n_north = n_horizontal
        depth, lon, lat = get_event(event_id)

        n_depth = int((1.2*depth) // cube_side)

        instance = cls(
            top_mid_point=Point(depth=0,
                                lon=lon,
                                lat=lat),
            cube_side=cube_side,
            n_north=n_north,
            n_east=n_east,
            n_depth=n_depth
        )
        
        instance.vp = None
        instance.vs = None

        instance.load_cvh()
        instance.save()
        return instance

    
    
def add_model_time_arrival(model, df):
    model_times = []

    for _, row in df.iterrows():
        # Создаем точки
        event_point = Point(row['event_lon'], row['event_lat'], row['event_depth'])
        station_point = Point(row['station_lon'], row['station_lat'], 0.0)

        # Определяем тип волны
        wave_type = 'vp' if row['arrival_type'][0].upper() == 'P' else 'vs'

        # Считаем время прибытия по модели
        try:
            arrival_time = model.calculate_time_arrival(event_point, station_point, wave_type=wave_type)
        except Exception:
            arrival_time = timedelta(seconds=0)  # или np.nan

        model_times.append(arrival_time)

    df = df.copy()
    df['model_time_arrival'] = model_times
    df['deviation'] = df['model_time_arrival'].dt.total_seconds()  - df['actual_time_diff'].dt.total_seconds() 
    return df





