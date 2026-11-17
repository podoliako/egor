"""Observed P arrivals versus straight-line 3-D station distance; no solves."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
EVENT = '3259971'
STEM = 'event3259971_arrival_vs_distance_catalog_soft'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    location = next(r for r in rows(HERE / 'real_picks_hard_soft_hypocentres_projection_1000m_max1.csv') if r['event_id'] == EVENT)
    stations = {r['station_id']: r for r in rows(DATA / 'stations.csv')}
    points = {mode: np.array([float(location[f'{mode}_{a}_m']) for a in 'xyz']) for mode in ('catalog', 'soft')}
    result = []
    for split, filename in [('fit', 'train.csv'), ('holdout', 'holdout.csv')]:
        for pick in rows(DATA / filename):
            if pick['event_id'] != EVENT:
                continue
            station_id = pick['network'] + '.' + pick['station']
            station = stations[station_id]
            xyz = np.array([float(station[a + '_m']) for a in 'xyz'])
            result.append(dict(event_id=EVENT, station_id=station_id, split=split,
                               phase=pick['phase'], pick_time_utc=pick['pick_time_utc'],
                               arrival_from_catalog_origin_s=float(pick['seconds_from_catalog_origin']),
                               **{mode + '_distance_km': float(np.linalg.norm(xyz - point) / 1000) for mode, point in points.items()}))
    result.sort(key=lambda r: r['arrival_from_catalog_origin_s'])
    assert len(result) == 7 and sum(r['split'] == 'fit' for r in result) == 6
    assert all(r['phase'] == 'P' for r in result)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'catalog': '#336fa5', 'soft': '#e18435'}
    fits = {}
    fit_rows = [r for r in result if r['split'] == 'fit']
    for mode in points:
        distances = np.array([r[mode + '_distance_km'] for r in fit_rows])
        times = np.array([r['arrival_from_catalog_origin_s'] for r in fit_rows])
        slope, intercept = np.polyfit(distances, times, 1)
        residuals = times - (slope * distances + intercept)
        r_squared = 1 - np.sum(residuals**2) / np.sum((times - times.mean())**2)
        fits[mode] = dict(slope_s_per_km=float(slope), intercept_s=float(intercept),
                          r_squared=float(r_squared), n_picks=len(fit_rows),
                          selection='Six fit picks only; STY holdout excluded; free intercept')
        line_x = np.linspace(0, max(distances) * 1.04, 200)
        ax.plot(line_x, slope * line_x + intercept, color=colors[mode], lw=1.5,
                ls='--', alpha=.8, zorder=2)
        for r in result:
            x, y = r[mode + '_distance_km'], r['arrival_from_catalog_origin_s']
            ax.scatter(x, y, color=colors[mode], marker='D' if r['split'] == 'holdout' else 'o',
                       s=65, edgecolors='white', linewidths=.7, zorder=3)
            ax.annotate(r['station_id'].split('.')[1], (x, y), xytext=(6, 5 if r['station_id'] != 'BG.STY' else -14),
                        textcoords='offset points', color=colors[mode], fontsize=9)
    ax.set(xlabel='Straight-line 3-D source–station distance (km)',
           ylabel='Observed P arrival − catalog origin time (s)',
           title='Event 3259971 · observed arrivals vs distance\n2011-04-03 · catalog origin 05:26:46.940 UTC')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4)
    for i, (mode, fit) in enumerate(fits.items()):
        ax.text(.03, .58 - i * .11,
                f"{mode.capitalize()}: t = {fit['slope_s_per_km']:.3f} r + {fit['intercept_s']:.3f} s"
                f"\n$R^2$ = {fit['r_squared']:.3f}",
                transform=ax.transAxes, color=colors[mode], fontsize=10)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=colors[m], label=l) for m, l in [('catalog', 'Catalog hypocentre'), ('soft', 'Final Soft MAP hypocentre')]] +
              [Line2D([], [], marker='D', ls='none', color='0.45', label='Held-out pick (STY)')], loc='upper center', frameon=False)
    fig.text(.5, .035, 'Dashed: linear least-squares fits to the six used picks, with free intercept; STY excluded.\nCommon catalog time reference; geometric 3-D distances, not ray lengths. Lines extrapolated to zero distance.',
             ha='center', fontsize=9, color='0.35')
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(HERE.parent / 'png' / (STEM + '.png'), dpi=220)
    fig.savefig(HERE / (STEM + '.pdf'))
    plt.close(fig)
    with (HERE / (STEM + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (HERE / (STEM + '.json')).write_text(json.dumps(dict(event_id=EVENT, positions_m={m: p.tolist() for m, p in points.items()},
        distance='Euclidean 3-D distance using stations.csv x_m,y_m,z_m in the same local frame',
        time='Observed pick minus catalog origin; same reference for both locations', linear_fits=fits, picks=result), indent=2) + '\n')
    print('Linear fits:', json.dumps(fits, indent=2))
    for r in result:
        print(r['station_id'], r['split'], 't=', r['arrival_from_catalog_origin_s'], 'catalog km=', r['catalog_distance_km'], 'soft km=', r['soft_distance_km'])
    print(HERE.parent / 'png' / (STEM + '.png'))


if __name__ == '__main__':
    main()
