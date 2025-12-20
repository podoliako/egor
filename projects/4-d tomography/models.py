import sys
sys.path.append('/mnt/disk01/egor/')
from utilities import Point, Earth, np, generate_name, cos, sin, radians, get_elevation, floor


class GeoModel:
    def __init__(self, top_mid_point:Point, azimuth:float, cube_side:float, n_x:int, n_y:int, n_z:int):
        self.top_mid_point = top_mid_point
        self.cube_side = cube_side
        self.azimuth = azimuth
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.earth = Earth()

        self.ref_h = get_elevation([self.top_mid_point])[0]

        self.model_name = generate_name(['geo', cube_side, azimuth, n_x, n_y, n_z])
        self.model_version = 0


    def get_cell_center_geodetic(self, ix: int, iy: int, iz: int) -> Point:
        """
        Преобразует целочисленные индексы ячейки (x, y, z) в ее географические координаты (центр ячейки).
        Глубина (depth) рассчитывается как расстояние от центра ячейки до поверхности земли.
        Положительная глубина означает, что центр ячейки находится ниже поверхности.
        """
        if not (0 <= ix < self.n_x and 0 <= iy < self.n_y and 0 <= iz < self.n_z):
            raise ValueError("Индексы ячейки выходят за пределы модели")

        # 1. Вычисляем смещение в локальной системе координат модели (в метрах)
        # Опорная точка модели (top_mid_point) соответствует локальным координатам ((n_x-1)/2, (n_y-1)/2, 0)
        # Центр ячейки (ix, iy, iz) имеет локальные координаты (ix + 0.5, iy + 0.5, iz + 0.5)
        # Ось Z модели направлена ВНИЗ от верхней поверхности.
        
        x_ref_local = (self.n_x - 1) / 2.0
        y_ref_local = (self.n_y - 1) / 2.0
        
        # Смещение центра ячейки относительно опорной точки в локальной системе
        dx_local = (ix + 0.5 - x_ref_local) * self.cube_side
        dy_local = (iy + 0.5 - y_ref_local) * self.cube_side
        # z=0 - верхний слой. Центр ячейки этого слоя на глубине -0.5 * cube_side
        dz_local = -(iz + 0.5) * self.cube_side - self.top_mid_point.depth

        # 2. Поворачиваем локальные смещения, чтобы согласовать их с системой ENU (East, North, Up)
        # Локальная ось X направлена по азимуту, Y - на 90 градусов против часовой стрелки.
        # Стандартная система ENU: N - север, E - восток.
        azimuth_rad = radians(self.azimuth)
        
        # Матрица поворота из локальной системы в ENU
        dE = dx_local * sin(azimuth_rad) + dy_local * cos(azimuth_rad)
        dN = dx_local * cos(azimuth_rad) - dy_local * sin(azimuth_rad)
        dU = dz_local # Ось Z/Up совпадает

        # 3. Преобразуем смещение ENU в глобальные координаты ECEF
        ref_lat = self.top_mid_point.lat
        ref_lon = self.top_mid_point.lon
        # Получаем реальную высоту поверхности в опорной точке
        ref_h = get_elevation([self.top_mid_point])[0]

        x_ecef, y_ecef, z_ecef = self.earth.enu_to_ecef(dE, dN, dU, ref_lat, ref_lon, ref_h)

        # 4. Преобразуем ECEF в геодезические координаты (lat, lon, h)
        lat, lon, h = self.earth.ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
        
        # 5. Рассчитываем глубину относительно поверхности земли
        surface_elevation = get_elevation([Point(0, lon, lat)])[0]
        depth = surface_elevation - h

        return Point(depth=depth, lon=lon, lat=lat)

    def get_cell_indices(self, lat: float, lon: float, depth: float):
        """
        Принимает географические координаты и глубину, возвращает целочисленные индексы (x, y, z) ячейки,
        в которой находится точка. Возвращает None, если точка вне модели.
        """
        # 1. Рассчитываем абсолютную высоту точки над эллипсоидом (h)
        surface_elevation = get_elevation([Point(0, lon, lat)])[0]
        h = surface_elevation - depth

        # 2. Преобразуем геодезические координаты точки в ECEF
        x_ecef, y_ecef, z_ecef = self.earth.geodetic_to_ecef(lat, lon, h)

        # 3. Преобразуем ECEF в локальное смещение ENU относительно опорной точки модели
        ref_lat = self.top_mid_point.lat
        ref_lon = self.top_mid_point.lon
        ref_h = get_elevation([self.top_mid_point])[0] - self.top_mid_point.depth
        
        dE, dN, dU = self.earth.ecef_to_enu(x_ecef, y_ecef, z_ecef, ref_lat, ref_lon, ref_h)
        
        # 4. Поворачиваем смещение из системы ENU в локальную систему модели
        azimuth_rad = radians(self.azimuth)
        
        # Обратная матрица поворота
        dx_local = dE * sin(azimuth_rad) + dN * cos(azimuth_rad)
        dy_local = -dN * sin(azimuth_rad) + dE * cos(azimuth_rad)
        dz_local = dU
        
        # 5. Преобразуем смещение в метрах в непрерывные координаты ячеек
        x_ref_local = (self.n_x - 1) / 2.0
        y_ref_local = (self.n_y - 1) / 2.0
        
        x_coord = (dx_local / self.cube_side) + x_ref_local
        y_coord = (dy_local / self.cube_side) + y_ref_local
        # z_coord = 0 соответствует верхней поверхности, ось z направлена вниз
        z_coord = -dz_local / self.cube_side
        
        # 6. Получаем целочисленные индексы и проверяем, находятся ли они в границах модели
        ix = floor(x_coord)
        iy = floor(y_coord)
        iz = floor(z_coord)

        if 0 <= ix < self.n_x and 0 <= iy < self.n_y and 0 <= iz < self.n_z:
            return ix, iy, iz
        else:
            return None
    