from utilities import generate_name, np
import skfmm
from dwh import get_event, get_station
from model import VelocityModel


def _get_arrival_matrix(velocities, start_point, cell_side):
    # Размер 3D сетки
    shape = velocities.shape
    speed = velocities
    speed[(speed==0)] = 1
    phi = np.ones(shape)
    phi[start_point[0], start_point[1], start_point[2]] = 0  # Начальная точка волны

    # Вычисляем решение уравнения Эйкена: время достижения каждой точки
    travel_time = skfmm.travel_time(phi, speed, self_test=False)

    travel_time *= cell_side
    travel_time = np.round(travel_time, 4)
    return travel_time

def calculate_time_martix_event(event_id:int, model:VelocityModel, wave_type):
    depth, lon, lat = get_event(event_id)
    event_coords = model.find_cell_coords(target_depth=depth, target_lat=lat, target_lon=lon)
    calculate_time_martix(event_coords, model, wave_type)

def calculate_time_martix_station(station_nm, network_nm, model:VelocityModel, wave_type):
    depth, lon, lat = get_station(station_nm, network_nm)
    event_coords = model.find_cell_coords(target_depth=depth, target_lat=lat, target_lon=lon)
    calculate_time_martix(event_coords, model, wave_type)

def calculate_time_martix(event_coords, model:VelocityModel, wave_type):
    vels = None
    if wave_type == 'P':
        vels = model.vp
    elif wave_type == 'S':
        vels = model.vs
    else:
        raise TypeError(f'Unknown wave type: {wave_type}')
    
    matrix = _get_arrival_matrix(velocities=vels, start_point=event_coords, cell_side=model.cube_side)

    folder = f'models/{model.model_name}/time_matrices'
    file_name = f'{generate_name(['skfmm', event_coords[0], event_coords[1], event_coords[2],
                                   wave_type], with_time=False)}.npz'

    np.savez_compressed(f'{folder}/{file_name}', matrix=matrix)

def load_time_matrix(event_id:int, model:VelocityModel, wave_type):
    depth, lon, lat = get_event(event_id)
    event_coords = model.find_cell_coords(target_depth=depth, target_lat=lat, target_lon=lon)

    matrix = load_time_matrix_coords(event_coords, model, wave_type)
    return matrix

def load_time_matrix_coords(coords, model:VelocityModel, wave_type):
    folder = f'models/{model.model_name}/time_matrices'
    file_name = f'{generate_name(['skfmm', coords[0], coords[1], coords[2],
                                   wave_type], with_time=False)}.npz'
    try:
        raw_marix = np.load(f'{folder}/{file_name}')
        matrix = raw_marix['matrix']
    except FileNotFoundError:
        print('Time matrix was not found, trying to create...')
        calculate_time_martix(coords, model, wave_type)
        raw_marix = np.load(f'{folder}/{file_name}')
        matrix = raw_marix['matrix']

    return matrix

def get_ray_path(travel_time, start_point, end_point, step_size=0.01, max_steps=1000000):
    point = np.array(end_point, dtype=float)
    path = [tuple(point)]
    
    grad = np.gradient(travel_time)
    
    for _ in range(max_steps):
        # Интерполируем градиент в текущей точке (взяли ближайший сосед)
        ix = np.clip(np.round(point).astype(int), 0, np.array(travel_time.shape) - 1)
        grad_vector = np.array([g[ix[0], ix[1], ix[2]] for g in grad])

        # Если градиент почти нулевой — выход (мы у источника)
        if np.linalg.norm(grad_vector) < 1e-6:
            break
        
        # Делаем шаг против градиента
        grad_dir = -grad_vector / np.linalg.norm(grad_vector)
        point += grad_dir * step_size
        
        path.append(tuple(point))
        
        # Проверяем, дошли ли до стартовой точки (с запасом)
        if np.linalg.norm(point - start_point) < 1.0:
            break
            
    return path

def get_arrival_from_matrix(martix, start_point, finish_point):
    z_start, x_start, y_start = start_point
    z_finish, x_finish, y_finish = finish_point
    start_time = martix[z_start, x_start, y_start]
    finish_time = martix[z_finish, x_finish, y_finish]
    time_delta = finish_time - start_time
    time_delta = round(time_delta, 2)
    return time_delta

def get_arrival_time(event_id, model:VelocityModel, finish_point, wave_type):
    depth, lon, lat = get_event(event_id)
    start_point = model.find_cell_coords(depth, lat, lon)
    matrix = load_time_matrix(event_id, model, wave_type)
    arrival_time = get_arrival_from_matrix(matrix, start_point, finish_point)
    return arrival_time