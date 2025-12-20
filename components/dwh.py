import psycopg2
import pandas as pd
import geopandas as gpd
import sqlalchemy

from shapely.geometry import Point

db_params = {
    'dbname': 'gis',
    'user': 'gis',
    'password': '123456',
    'host': '10.0.62.59',
    'port': '55432' 
}


def load_to_postgres(df, df_name):
    engine = sqlalchemy.create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')
    try:
        df.to_sql(
            df_name,
            engine,
            if_exists='append',
            index=False,          
        )
        # print(f"[{df_name}] Records created successfully")
    except Exception as ex:
        print("Operation failed: {0}".format(ex))


def load_to_postgis(df, df_name):
    geom = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    engine = sqlalchemy.create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')
    try:
        gdf.to_postgis(
            df_name,
            engine,
            if_exists='append',
            index=False
        )
        # print(f"[{df_name}] Records created successfully")
    except Exception as ex:
        print("Operation failed: {0}".format(ex))
    finally:
        engine.dispose()


def get_all_arrivals(from_dt, to_dt, min_lon=-120.5, min_lat=31.0, max_lon=-113.5, max_lat=36.5):
    conn = psycopg2.connect(**db_params)
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
        WHERE e.loc IS NOT NULL AND a.arrival_dttm IS NOT NULL AND e.event_dttm::date >= '{from_dt}' AND e.event_dttm::date <= '{to_dt}'
        and e.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326) 
        and st.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326)
        and e.depth > 0
        """
    df = pd.read_sql(query, conn)
    df = df.loc[df['arrival_type'].isin(['P', 'S', 'Pg', 'Sg'])]

    return df

def get_event(event_id):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select ST_X(e.loc) as lon, ST_Y(e.loc) as lat, e."depth", e.event_dttm from events e
        where e.event_id  = {event_id}
        """
    df = pd.read_sql(query, conn)
    return df['depth'][0], df['lon'][0], df['lat'][0]

def get_station(st_name, network_nm):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select ST_X(s.loc) as lon, ST_Y(s.loc) as lat from stations s
        where s.station_nm  = '{st_name}' and s.network_nm = '{network_nm}'
        """
    df = pd.read_sql(query, conn)
    return 0, df['lon'][0], df['lat'][0]

def get_arrivals_by_event(event_id):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select ST_X(s.loc) as lon, ST_Y(s.loc) as lat, a.arrival_dttm from arrivals a
        inner join stations s on
        a.station_nm = s.station_nm and a.network_nm = s.network_nm 
        where a.event_id = {event_id}
        order by a.arrival_dttm
        """
    df = pd.read_sql(query, conn)
    return df

def get_events_in_cylinder(lon, lat, depth, radius, min_mag, min_stations, year):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select distinct e.event_id, e.event_type, e.event_type_certainty, ST_X(e.loc) as lon, ST_Y(e.loc) as lat, e.depth, e.event_dttm, count(distinct s.station_nm) from events e
        inner join magnitudes m
        on m.event_id = e.event_id and m.mag_value >= {min_mag} and e."depth" < {depth} and 
        ST_Distance(ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326),3857), ST_Transform(e.loc::geometry, 3857)) < {radius} and e.event_type_certainty = 'known'
        inner join arrivals a on a.event_id = e.event_id 
        inner join stations s on a.network_nm = s.network_nm and a.station_nm = s.station_nm and 
        ST_Distance(ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326),3857), ST_Transform(s.loc::geometry, 3857)) < {radius}
        where date_part('year', event_dttm) >= {year}
        group by 1,2,3,4,5,6,7
        having count(distinct s.station_nm) >= {min_stations}
        order by e.event_dttm
        """
    df = pd.read_sql(query, conn)
    return df

def get_all_events(from_dt, to_dt, min_lon=-120.5, min_lat=31.0, max_lon=-113.5, max_lat=36.5):
    conn = psycopg2.connect(**db_params)
    query = f"""
        SELECT 
            ST_X(e.loc) as event_lon,
            ST_Y(e.loc) as event_lat,
            e.depth as event_depth,
            e.event_dttm as event_dttm
        FROM events e
        WHERE e.event_dttm::date >= '{from_dt}' AND e.event_dttm::date <= '{to_dt}'
        and e.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326) 
        """
    df = pd.read_sql(query, conn)
    return df

# def get_tp_ts(lon, lat, r_events, r_stations, dttm_from, dttm_to, drop_nan=True):
#     # Connect to DB
#     conn = psycopg2.connect(**db_params)

#     query = f"""
#     SELECT
#         MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.pick_id END) AS p_pick_id,
#         MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.pick_id END) AS s_pick_id,
#         e.event_id,
#         s.station_nm,
#         s.network_nm,
#         MIN(e.event_dttm) AS event_dttm,
#         -- Time deltas
#         EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_p,
#         EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_s,
#         -- Wave subtype
#         CASE
#             WHEN a.arrival_type LIKE '%g' THEN 'g'
#             WHEN a.arrival_type LIKE '%n' THEN 'n'
#             WHEN a.arrival_type LIKE '%b' THEN 'b'
#             ELSE NULL
#         END AS wave_subtype
#     FROM events e
#     JOIN arrivals a ON a.event_id = e.event_id
#     JOIN stations s ON s.station_nm = a.station_nm 
#                     AND s.network_nm = a.network_nm
#     WHERE 
#         -- Events radius filter
#         ST_DWithin(
#             e.loc,
#             ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography,
#             {r_events}
#         )
#         -- Stations radius filter
#         AND ST_DWithin(
#             s.loc,
#             ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography,
#             {r_stations}
#         )
#         AND a.arrival_type IN ('P','S','Pg','Sg','Pn','Sn','Pb','Sb')
#         AND e.event_dttm::date >= '{dttm_from}'
#         AND e.event_dttm::date <= '{dttm_to}'
#         AND EXTRACT(EPOCH FROM a.arrival_dttm - e.event_dttm) > 0
#         -- AND e.event_type_certainty = 'known'
#     GROUP BY
#         e.event_id,
#         s.station_nm,
#         s.network_nm,
#         wave_subtype
#     """

#     df = pd.read_sql(query, conn)
#     if drop_nan is True:
#         df = df.dropna(subset=["delta_t_p", "delta_t_s"])
#         df = df.reset_index(drop=True)
#     return df


def get_experiment(experiment_nm):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select
        ST_X(e.geometry) as node_lon,
        ST_Y(e.geometry) as node_lat,
        e.t_from,
        e.t_to, 
        (e.params::jsonb->>'n_points')::numeric as n_points,
        ((e.params::jsonb->>'regression_free')::jsonb->>'k')::numeric as k_free,
        ((e.params::jsonb->>'regression_free')::jsonb->>'r2')::numeric as r2_free,
        ((e.params::jsonb->>'regression_zero_intercept')::jsonb->>'k')::numeric as k_zero_intercept,
        ((e.params::jsonb->>'regression_zero_intercept')::jsonb->>'r2')::numeric as r2_k_zero_intercept
        from experiments e
        where e.experiment_nm = '{experiment_nm}';
    """
    df = pd.read_sql(query, conn)
    return df

def get_tp_ts_by_exp(experiment_nm):
    conn = psycopg2.connect(**db_params)
    query = f"""
        select * from travel_times
        where experiment_nm = '{experiment_nm}';
        """
    df = pd.read_sql(query, conn)
    return df

def get_df_by_query(query):
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(query, conn)
    return df

def get_travel_times_by_region(region_nm, dttm_from, dttm_to,):
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
        -- Time deltas
        EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_p,
        EXTRACT(EPOCH FROM MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm END)) AS delta_t_s,
        -- Wave subtype
        CASE
            WHEN a.arrival_type LIKE '%g' THEN 'g'
            WHEN a.arrival_type LIKE '%n' THEN 'n'
            WHEN a.arrival_type LIKE '%b' THEN 'b'
            ELSE NULL
        END AS wave_subtype
    FROM regions r
    JOIN events e 
        on ST_Within(e.loc, r.region_geo)
    JOIN arrivals a 
        ON a.event_id = e.event_id
    JOIN stations s 
        ON s.station_nm = a.station_nm 
        AND s.network_nm = a.network_nm
        AND ST_Within(s.loc, r.region_geo)
    where TRUE 
        AND r.region_nm = '{region_nm}' 
        AND a.arrival_type IN ('P','S','Pg','Sg','Pn','Sn','Pb','Sb')
        AND e.event_dttm::date >= '{dttm_from}'
        AND e.event_dttm::date <= '{dttm_to}'
        AND EXTRACT(EPOCH FROM a.arrival_dttm - e.event_dttm) > 0
    GROUP BY
        e.event_id,
        s.station_nm,
        s.network_nm,
        wave_subtype
    HAVING TRUE
        AND MAX(CASE WHEN a.arrival_type LIKE 'P%' THEN a.pick_id END) is not null
        AND MAX(CASE WHEN a.arrival_type LIKE 'S%' THEN a.pick_id END) is not null
    """
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(query, conn)
    return df

def get_region_borders(region_nm):
    query = f"""
    select 
        ST_XMin(r.region_geo) as lon_min,
        ST_XMax(r.region_geo) as lon_max,
        ST_YMin(r.region_geo) as lat_min,
        ST_YMax(r.region_geo) as lat_max
    from regions r
    where r.region_nm = '{region_nm}'
    """
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql(query, conn)
    res = df.iloc[0].to_dict()
    return res