import psycopg2
import pandas as pd
import matplotlib.pyplot as plt


db_params = {
    'dbname': 'gis',
    'user': 'gis',
    'password': '123456',
    'host': '10.0.62.59',
    'port': '55432' 
}


def get_kamchatka_ts_tv():
    conn = psycopg2.connect(**db_params)
    query = f"""
    SELECT
    e.event_id,
    s.station_nm,
    s.network_nm,
    MIN(e.event_dttm) AS event_dttm,
    -- Агрегируем по типам волн
    MAX(CASE 
        WHEN a.arrival_type LIKE 'P%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_p,
    MAX(CASE 
        WHEN a.arrival_type LIKE 'S%' THEN a.arrival_dttm - e.event_dttm 
    END) AS delta_t_s,
    -- Определяем wave_subtype
    CASE
        WHEN a.arrival_type LIKE '%g' THEN 'g'
        WHEN a.arrival_type LIKE '%n' THEN 'n'
        WHEN a.arrival_type LIKE '%b' THEN 'b'
        ELSE NULL
    END AS wave_subtype
    FROM events e
    JOIN arrivals a ON a.event_id = e.event_id
    JOIN stations s ON s.station_nm = a.station_nm AND s.network_nm = a.network_nm
    WHERE 
    ST_Within(e.loc, ST_MakeEnvelope(-115, 32, -120, 35, 4326))
    AND ST_Within(s.loc, ST_MakeEnvelope(-115, 35, -120, 32, 4326))
    AND a.arrival_type IN ('P', 'S', 'Pg', 'Sg', 'Pn', 'Sn', 'Pb', 'Sb')
    and e.event_dttm::date <= '2025-01-01' and e.event_dttm::date >= '2015-01-01'
    GROUP BY
    e.event_id,
    s.station_nm,
    s.network_nm,
    wave_subtype; 
    """
    df = pd.read_sql(query, conn)
    return df

df = get_kamchatka_ts_tv()
df = df.loc[(df['delta_t_p'].dt.seconds < 500) & (df['delta_t_s'].dt.seconds < 500)]
print(df)


def plot_delta_t(df):
    """
    Строит график зависимости delta_t_s от delta_t_p.
    Принимает DataFrame с колонками delta_t_p, delta_t_s, wave_subtype.
    """
    # Преобразуем timedelta → секунды, если нужно
    for col in ['delta_t_p', 'delta_t_s']:
        if df[col].dtype == 'timedelta64[ns]':
            df[col] = df[col].dt.total_seconds()

    # Удалим пустые значения
    df = df.dropna(subset=['delta_t_p', 'delta_t_s'])

    plt.figure(figsize=(8, 6), dpi=500)
    scatter = plt.scatter(
        df['delta_t_p'],
        df['delta_t_s'],
        c=df['wave_subtype'].map({'g': 'tab:blue', 'n': 'tab:orange', 'b': 'tab:green', None: 'tab:gray'}),
        alpha=0.7,
        s=0.3
    )

    # # Диагональная линия y=x
    # lims = [
    #     min(df['delta_t_p'].min(), df['delta_t_s'].min()),
    #     max(df['delta_t_p'].max(), df['delta_t_s'].max())
    # ]
    # plt.plot(lims, lims, 'k--', lw=1)

    plt.xlabel("Δt (P-wave), sec")
    plt.ylabel("Δt (S-wave), sec")
    plt.title("Зависимость времени прихода S-волн от P-волн")

    # Легенда
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='g', markerfacecolor='tab:blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='n', markerfacecolor='tab:orange', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='b', markerfacecolor='tab:green', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='(no subtype)', markerfacecolor='tab:gray', markersize=8)
    ]
    plt.legend(handles=handles, title="Wave subtype")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plot_7.png')

plot_delta_t(df)