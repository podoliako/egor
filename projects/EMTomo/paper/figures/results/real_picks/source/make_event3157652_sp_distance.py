"""Illustrative constant-speed S-P distance check, not a layered-model inversion."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STEM = 'event3157652_SP_distance_catalog_soft'
VP = 5.0
RATIO = 1.73
VS = VP / RATIO


def main():
    source = HERE / 'event3157652_arrival_vs_distance_catalog_soft_PS_linear.csv'
    with source.open(newline='') as stream:
        picks = list(csv.DictReader(stream))
    pairs = {}
    for pick in picks:
        pairs.setdefault(pick['station_id'], {})[pick['phase']] = pick
    result = []
    for station, phases in pairs.items():
        if not {'P', 'S'} <= phases.keys():
            continue
        dt = float(phases['S']['arrival_from_catalog_origin_s']) - float(phases['P']['arrival_from_catalog_origin_s'])
        assert dt > 0
        result.append(dict(station_id=station, s_minus_p_s=dt, sp_distance_km=dt/(1/VS-1/VP),
                           catalog_distance_km=float(phases['P']['catalog_distance_km']),
                           soft_distance_km=float(phases['P']['soft_distance_km'])))
    assert len(result) == 4
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(8.5,7.5))
    colors = dict(catalog='#336fa5', soft='#e18435')
    upper = np.ceil(max(r[m+'_distance_km'] for r in result for m in colors)+1)
    ax.plot([0,upper],[0,upper],color='0.5',ls='--',lw=1,label='Equal distances (y = x)')
    for mode, label in [('catalog','Catalog hypocentre'),('soft','Final Soft MAP hypocentre')]:
        x = [r['sp_distance_km'] for r in result]
        y = [r[mode+'_distance_km'] for r in result]
        ax.scatter(x,y,color=colors[mode],s=65,label=label,zorder=3)
        for r in result:
            ax.annotate(r['station_id'].split('.')[1],(r['sp_distance_km'],r[mode+'_distance_km']),
                        xytext=(7,4),textcoords='offset points',color=colors[mode],fontsize=10)
    ax.set(xlim=(0,upper),ylim=(0,upper),aspect='equal',
           xlabel='Distance inferred from S − P (km)',
           ylabel='Geometric 3-D source–station distance (km)',
           title='Event 3157652 · S−P distance check\nAssumed Vp = 5.0 km/s; Vp/Vs = 1.73')
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right',frameon=False)
    fig.text(.5,.045,'r = (tS − tP) / (1/Vs − 1/Vp); origin time cancels.\nIllustrative homogeneous-speed estimate, not ray tracing in the recovered model.',ha='center',fontsize=9,color='0.35')
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(HERE.parent/'png'/(STEM+'.png'),dpi=220)
    fig.savefig(HERE/(STEM+'.pdf'))
    plt.close(fig)
    with (HERE/(STEM+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    report=dict(event_id='3157652',assumed_vp_km_s=VP,assumed_vs_km_s=VS,assumed_vp_vs=RATIO,
                assumption='Illustrative constant speeds, not estimated from these picks or extracted from the recovered model',stations=result)
    (HERE/(STEM+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
