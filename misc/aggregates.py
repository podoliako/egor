from dwh import get_event, get_arrivals_by_event, get_events_in_cylinder
from componets.utilities import pd, add_cell_coords, generate_name
from raytracing import get_arrival_from_matrix, load_time_matrix
from model import VelocityModel


def add_delta_time(model, df, event_id, wave_type):
    depth, lon, lat = get_event(event_id)
    start_point = model.find_cell_coords(depth, lat, lon)
    matrix = load_time_matrix(event_id, model, wave_type)
    column = df.apply(lambda row: pd.Series(get_arrival_from_matrix(matrix, start_point, (row['z'], row['x'], row['y']))), axis=1)
    if column.empty:
        return "no stations"
    df['fmm_time_delta'] = column
    df['fmm_time_delta'] = pd.to_timedelta(df['fmm_time_delta'], unit='s')
    return df


def calculate_arrival_table(model, event_id, wave_type):
    df = get_arrivals_by_event(event_id)
    df = add_cell_coords(model, df)
    df = add_delta_time(model, df, event_id, wave_type)
    if type(df) == str and df == 'no stations':
        return
    df['fmm_event_dttm'] = df['arrival_dttm'] - df['fmm_time_delta']
    mean_dttm = df['fmm_event_dttm'].mean()
    df['deviation_fmm_x_mean'] = df.apply(lambda row: row['fmm_event_dttm'] - mean_dttm, axis=1).dt.total_seconds() 

    folder = f'models/{model.model_name}/fmm_times/'
    file_name = f'{generate_name(['table', 'fmm', event_id])}.csv'
    df.to_csv(folder + file_name)

def get_events_inside_model(model:VelocityModel, min_mag, min_stations, year):
    depth = (model.n_depth) * model.cube_side*0.9
    radius = (((model.n_north - 1) // 2) * model.cube_side)
    point = model.top_mid_point
    lon = point.lon
    lat = point.lat

    df = get_events_in_cylinder(lon, lat, depth, radius, min_mag, min_stations, year)
    return df

def compute_tables(model:VelocityModel, min_mag, min_stations, wave_type, year=1970):
    events = get_events_inside_model(model, min_mag, min_stations, year)['event_id']
    n_events = len(events)
    print(f'There are {n_events} to process')
    i = 1
    for event_id in events:
        print(f'Processing {i}/{n_events} table...')
        calculate_arrival_table(model, event_id, wave_type)
        i+=1
    