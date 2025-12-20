from models import GeoModel
from utilities import Point

top_mid = Point(-1000, -115.26, 32.38)

model = GeoModel(top_mid, 58, 100, 300, 150, 200)

# points = [model.get_cell_center_geodetic(151, 75, 9)]

print(model.get_cell_indices(32.417878, -115.308017, 0))


# for p in points:
    # print(p.lat, p.lon, p.de)