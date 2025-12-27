import psycopg2
import pandas as pd
import geopandas as gpd
import sqlalchemy
from shapely.geometry import Point

DB_PARAMS = {
    'dbname': 'gis',
    'user': 'gis',
    'password': '123456',
    'host': '10.0.62.59',
    'port': '55432'
}

DB_URL = 'postgresql://gis:123456@10.0.62.59:55432/gis'


def execute_query(query):
    """Execute SQL query and return DataFrame."""
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_to_postgres(df, df_name):
    """Load DataFrame to PostgreSQL table."""
    engine = sqlalchemy.create_engine(DB_URL)
    try:
        df.to_sql(
            df_name,
            engine,
            if_exists='append',
            index=False,
        )
    except Exception as ex:
        print(f"Operation failed: {ex}")
    finally:
        engine.dispose()


def load_to_postgis(df, df_name):
    """Load DataFrame with geometry to PostGIS table."""
    geom = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    engine = sqlalchemy.create_engine(DB_URL)
    try:
        gdf.to_postgis(
            df_name,
            engine,
            if_exists='append',
            index=False
        )
    except Exception as ex:
        print(f"Operation failed: {ex}")
    finally:
        engine.dispose()


def get_all_arrivals(from_dt, to_dt, min_lon=-120.5, min_lat=31.0, max_lon=-113.5, max_lat=36.5):
    """Get all arrivals within date and spatial bounds."""
    query = f"""
        WITH station_locs AS (
            SELECT 
                s.station_nm, 
                s.network_nm,
                s.loc
            FROM stations s
            WHERE s.loc IS NOT NULL
        )
        SELECT 
            ST_X(e.loc) as event_lon,
            ST_Y(e.loc) as event_lat,
            e.depth as event_depth,
            a.arrival_type,
            a.arrival_dttm - e.event_dttm as actual_time_diff,
            ST_X(st.loc) as station_lon,
            ST_Y(st.loc) as station_lat
        FROM events e
        JOIN arrivals a ON e.event_id = a.event_id
        JOIN station_locs st ON a.station_nm = st.station_nm AND a.network_nm = st.network_nm
        WHERE e.loc IS NOT NULL 
            AND a.arrival_dttm IS NOT NULL 
            AND e.event_dttm::date >= '{from_dt}' 
            AND e.event_dttm::date <= '{to_dt}'
            AND e.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326) 
            AND st.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326)
            AND e.depth > 0
    """
    df = execute_query(query)
    df = df.loc[df['arrival_type'].isin(['P', 'S', 'Pg', 'Sg'])]
    return df


def get_event(event_id):
    """Get event location and depth by event_id."""
    query = f"""
        SELECT ST_X(e.loc) as lon, ST_Y(e.loc) as lat, e.depth, e.event_dttm 
        FROM events e
        WHERE e.event_id = {event_id}
    """
    df = execute_query(query)
    return df['depth'][0], df['lon'][0], df['lat'][0]


def get_station(st_name, network_nm):
    """Get station location by station name and network."""
    query = f"""
        SELECT ST_X(s.loc) as lon, ST_Y(s.loc) as lat 
        FROM stations s
        WHERE s.station_nm = '{st_name}' AND s.network_nm = '{network_nm}'
    """
    df = execute_query(query)
    return 0, df['lon'][0], df['lat'][0]


def get_arrivals_by_event(event_id):
    """Get all arrivals for a specific event."""
    query = f"""
        SELECT ST_X(s.loc) as lon, ST_Y(s.loc) as lat, a.arrival_dttm 
        FROM arrivals a
        INNER JOIN stations s 
            ON a.station_nm = s.station_nm AND a.network_nm = s.network_nm 
        WHERE a.event_id = {event_id}
        ORDER BY a.arrival_dttm
    """
    return execute_query(query)


def get_events_in_cylinder(lon, lat, depth, radius, min_mag, min_stations, year):
    """Get events within cylindrical region with minimum magnitude and station count."""
    query = f"""
        SELECT DISTINCT 
            e.event_id, 
            e.event_type, 
            e.event_type_certainty, 
            ST_X(e.loc) as lon, 
            ST_Y(e.loc) as lat, 
            e.depth, 
            e.event_dttm, 
            COUNT(DISTINCT s.station_nm) 
        FROM events e
        INNER JOIN magnitudes m
            ON m.event_id = e.event_id 
            AND m.mag_value >= {min_mag} 
            AND e.depth < {depth} 
            AND ST_Distance(
                ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857), 
                ST_Transform(e.loc::geometry, 3857)
            ) < {radius} 
            AND e.event_type_certainty = 'known'
        INNER JOIN arrivals a ON a.event_id = e.event_id 
        INNER JOIN stations s 
            ON a.network_nm = s.network_nm 
            AND a.station_nm = s.station_nm 
            AND ST_Distance(
                ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857), 
                ST_Transform(s.loc::geometry, 3857)
            ) < {radius}
        WHERE date_part('year', event_dttm) >= {year}
        GROUP BY e.event_id, e.event_type, e.event_type_certainty, lon, lat, e.depth, e.event_dttm
        HAVING COUNT(DISTINCT s.station_nm) >= {min_stations}
        ORDER BY e.event_dttm
    """
    return execute_query(query)


def get_all_events(from_dt, to_dt, min_lon=-120.5, min_lat=31.0, max_lon=-113.5, max_lat=36.5):
    """Get all events within date and spatial bounds."""
    query = f"""
        SELECT 
            ST_X(e.loc) as event_lon,
            ST_Y(e.loc) as event_lat,
            e.depth as event_depth,
            e.event_dttm as event_dttm
        FROM events e
        WHERE e.event_dttm::date >= '{from_dt}' 
            AND e.event_dttm::date <= '{to_dt}'
            AND e.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326)
    """
    return execute_query(query)


def get_experiment(experiment_nm):
    """Get experiment results by experiment name."""
    query = f"""
        select 
            ix, 
            iy, 
            it,
            (params->>'lon')::float as lon,
            (params->>'lat')::float as lat,
            (params->>'dttm')::date as dt,
            (params->>'n_measurements')::int as n_measurements,
            (params->'linear_regression'->>'slope')::float as slope
        from experiment_nodes 
        where experiment_nm = '{experiment_nm}'
        order by it, ix, iy 
    """
    return execute_query(query)


def get_tp_ts_by_exp(experiment_nm):
    """Get travel times by experiment name."""
    query = f"""
        SELECT * FROM travel_times
        WHERE experiment_nm = '{experiment_nm}'
    """
    return execute_query(query)


def get_travel_times_by_region(region_nm, dttm_from, dttm_to):
    """Get P and S wave travel times for events and stations within a region."""
    query = f"""
        SELECT
            MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.pick_id END) AS p_pick_id,
            MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.pick_id END) AS s_pick_id,
            e.event_id,
            ST_X(e.loc) as e_lon,
            ST_Y(e.loc) as e_lat,
            ST_X(s.loc) as s_lon,
            ST_Y(s.loc) as s_lat,
            s.station_nm,
            s.network_nm,
            MIN(e.event_dttm) AS event_dttm,
            EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_p,
            EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_s,
            CASE
                WHEN a.arrival_type LIKE '%g' THEN 'g'
                WHEN a.arrival_type LIKE '%n' THEN 'n'
                WHEN a.arrival_type LIKE '%b' THEN 'b'
                ELSE NULL
            END AS wave_subtype
        FROM regions r
        JOIN events e 
            ON ST_Within(e.loc, r.region_geo)
        JOIN arrivals a 
            ON a.event_id = e.event_id
        JOIN stations s 
            ON s.station_nm = a.station_nm 
            AND s.network_nm = a.network_nm
            AND ST_Within(s.loc, r.region_geo)
        WHERE r.region_nm = '{region_nm}' 
            AND a.arrival_type IN ('P','S','Pg','Sg','Pn','Sn','Pb','Sb')
            AND e.event_dttm::date >= '{dttm_from}'
            AND e.event_dttm::date <= '{dttm_to}'
            AND EXTRACT(EPOCH FROM a.arrival_dttm - e.event_dttm) > 0
        GROUP BY
            e.event_id,
            s.station_nm,
            s.network_nm,
            wave_subtype
        HAVING MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.pick_id END) IS NOT NULL
            AND MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.pick_id END) IS NOT NULL
    """
    return execute_query(query)


def get_region_borders(region_nm):
    """Get min/max lon/lat borders of a region."""
    query = f"""
        SELECT 
            ST_XMin(r.region_geo) as lon_min,
            ST_XMax(r.region_geo) as lon_max,
            ST_YMin(r.region_geo) as lat_min,
            ST_YMax(r.region_geo) as lat_max
        FROM regions r
        WHERE r.region_nm = '{region_nm}'
    """
    df = execute_query(query)
    return df.iloc[0].to_dict()
