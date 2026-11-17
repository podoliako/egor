"""Event 3157652: observed P/S arrivals versus catalog and Soft 3-D distance."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
EVENT = '3157652'
STEM = 'event3157652_arrival_vs_distance_catalog_soft_PS_linear'
sys.path.insert(0, str(ROOT / 'paper/paper_scripts'))
from make_geysers_real_location_shifts import geographic_to_local


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    cat = next(r for r in rows(DATA / 'catalog.csv') if r['event_id'] == EVENT)
    origin = datetime.fromisoformat(cat['origin_time_utc'].replace('Z', '+00:00'))
    geometry = json.loads((DATA / 'geometry.json').read_text())
    with np.load(ROOT / 'paper/exp_data/geysers_real_holdout_latest/soft_predictions.npz') as data:
        indices = np.flatnonzero(data['event_ids'].astype(str) == EVENT)
        assert len(indices) == 1
        soft = data['position_m'][indices[0]].copy()
    points = dict(catalog=geographic_to_local(float(cat['latitude']), float(cat['longitude']), float(cat['depth_km']), geometry), soft=soft)
    stations = {r['station_id']: r for r in rows(DATA / 'stations.csv')}
    result = []

    def add(station_id, phase, split, timestamp):
        station = stations[station_id]
        xyz = np.array([float(station[a + '_m']) for a in 'xyz'])
        result.append(dict(event_id=EVENT, station_id=station_id, phase=phase, split=split,
                           pick_time_utc=timestamp.isoformat(), arrival_from_catalog_origin_s=(timestamp-origin).total_seconds(),
                           **{m + '_distance_km': float(np.linalg.norm(xyz-p)/1000) for m, p in points.items()}))

    for split, filename in [('fit', 'train.csv'), ('holdout', 'holdout.csv')]:
        for pick in rows(DATA / filename):
            if pick['event_id'] == EVENT:
                add(pick['network']+'.'+pick['station'], 'P', split, datetime.fromisoformat(pick['pick_time_utc'].replace('Z', '+00:00')))
    raw_path = ROOT / 'data/geysers_2011/real_picks/raw/2011-04-10.html'
    active = False
    for line in raw_path.read_text().splitlines():
        if re.search(r'\s'+EVENT+r'L', line):
            active = True
        elif active and re.fullmatch(r'\s*'+EVENT+r'\s*', line):
            break
        elif active and ' S ' in line:
            match = re.search(r'(\d{12})\s+(\d+)\s+S\s', line)
            assert match, line
            minute = datetime.strptime(match[1], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            add(line[5:7].strip()+'.'+line[:5].strip(), 'S', 'independent S', minute+timedelta(seconds=int(match[2])/100))
    assert sum(r['phase']=='P' for r in result) == 11
    assert sum(r['phase']=='S' for r in result) == 4
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(12, 7.5))
    colors = dict(catalog='#336fa5', soft='#e18435')
    fits = {}
    xmax = np.ceil(max(r[m+'_distance_km'] for r in result for m in points)+1)
    for mode in points:
        for phase in ('P', 'S'):
            selected = [r for r in result if r['phase']==phase and r['split']!='holdout']
            x = np.array([r[mode+'_distance_km'] for r in selected])
            y = np.array([r['arrival_from_catalog_origin_s'] for r in selected])
            slope, intercept = np.polyfit(x,y,1)
            residual = y-(slope*x+intercept)
            r2 = 1-np.sum(residual**2)/np.sum((y-y.mean())**2)
            fits[mode+'_'+phase] = dict(slope_s_per_km=float(slope), intercept_s=float(intercept), r_squared=float(r2), n_picks=len(x))
            xx = np.linspace(0, min(xmax, x.max()*1.04), 200)
            ax.plot(xx, slope*xx+intercept, color=colors[mode], ls='--' if phase=='P' else ':', lw=1.8,
                    label=f'{mode.capitalize()} {phase}: t = {slope:.3f} r {intercept:+.3f}; R² = {r2:.3f}')
        for r in result:
            x,y = r[mode+'_distance_km'],r['arrival_from_catalog_origin_s']
            marker = 'D' if r['split']=='holdout' else ('^' if r['phase']=='S' else 'o')
            ax.scatter(x,y,marker=marker,s=65,color=colors[mode],edgecolors='white',linewidths=.6,zorder=3)
            offset = (5, 7 if r['phase']=='S' else -12)
            ax.annotate(r['station_id'].split('.')[1], (x,y), xytext=offset, textcoords='offset points', color=colors[mode], fontsize=8)
    ax.set(xlim=(0,xmax), ylim=(0,4.1), xlabel='Straight-line 3-D source–station distance (km)',
           ylabel='Observed arrival − catalog origin time (s)',
           title='Event 3157652 · observed P and S arrivals vs distance\n2011-04-10 · catalog origin 20:56:27.520 UTC')
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([],[],marker=m,ls='none',color='0.4',label=l) for m,l in [('o','P pick'),('^','S pick (not used in relocation)'),('D','Held-out P pick (NEG)')]]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,-.14), ncol=2, frameon=False, fontsize=9)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)
    fig.text(.5,.025,'Separate free-intercept linear fits: 10 used P picks (including LCK) and 4 S picks; held-out NEG excluded.\nCommon catalog time reference; geometric distances, not ray lengths. Soft MAP is close to the domain boundary.', ha='center',fontsize=9,color='0.35')
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.30)
    fig.savefig(HERE.parent/'png'/(STEM+'.png'),dpi=220)
    fig.savefig(HERE/(STEM+'.pdf'))
    plt.close(fig)
    with (HERE/(STEM+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    (HERE/(STEM+'.json')).write_text(json.dumps(dict(event_id=EVENT,positions_m={m:p.tolist() for m,p in points.items()},
        fits=fits,fit_policy='Separate OLS with free intercept; all non-holdout points of each phase; no outlier removal',
        raw_S_source=str(raw_path.relative_to(ROOT)),picks=result),indent=2)+'\n')
    print(json.dumps(fits,indent=2))
    print(HERE.parent/'png'/(STEM+'.png'))


if __name__ == '__main__':
    main()
