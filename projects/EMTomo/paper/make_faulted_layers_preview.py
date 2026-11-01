"""Draw a proposed faulted truth and 1-D start; no forward/inverse calculation.

Run: python -B paper/make_faulted_layers_preview.py
All distances below are km, velocities are km/s. No experiment inputs are changed.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


OUTPUT = Path(__file__).resolve().parent / 'figures'
STEM = 'faulted_layers_x6p5_3layers_preview'
PARAMETERS = {
    'status': 'proposed_geometry_only_no_solves',
    'extent_km': [20.0, 7.0, 7.0],
    'fault_x_km': 6.5,
    'section_y_km': 3.5,
    'left': {'boundaries_km': [0.7, 1.9],
             'vp_km_s': [3.6, 4.4, 5.0]},
    'right': {'boundaries_km': [1.1, 2.9],
              'vp_km_s': [4.0, 4.8, 5.5]},
    'initial': {'boundaries_km': [0.85, 2.3],
                'vp_km_s': [3.85, 4.55, 5.3]},
    'interpretation': 'Two finite layers over a basal half-space; invariant along y; no spheres.',
}


def profile(z, block):
    return np.asarray(block['vp_km_s'])[np.searchsorted(block['boundaries_km'], z, side='right')]


def main():
    p = PARAMETERS
    length, _, depth = p['extent_km']
    fault = p['fault_x_km']
    x_edges = np.linspace(0, length, 801)
    z_edges = np.linspace(0, depth, 561)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    z = (z_edges[:-1] + z_edges[1:]) / 2
    truth = np.where(x[None, :] < fault,
                     profile(z, p['left'])[:, None], profile(z, p['right'])[:, None])
    initial = np.broadcast_to(profile(z, p['initial'])[:, None], truth.shape)
    left_h = np.diff([0, *p['left']['boundaries_km']])
    right_h = np.diff([0, *p['right']['boundaries_km']])
    assert np.all(right_h > left_h)
    for block in ('left', 'right', 'initial'):
        assert np.all(np.diff(p[block]['boundaries_km']) > 0)
        assert np.all(np.diff(p[block]['vp_km_s']) > 0)
    assert np.all(np.asarray(p['right']['vp_km_s']) > p['left']['vp_km_s'])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True, sharey=True,
                             layout='constrained')
    speeds = [v for name in ('left', 'right', 'initial') for v in p[name]['vp_km_s']]
    norm = Normalize(min(speeds), max(speeds))
    for ax, field, title in zip(axes, (truth, initial),
                               ('Proposed truth: thicker, faster layers to the right',
                                'Initial model: approximate intermediate 1-D layering')):
        image = ax.pcolormesh(x_edges, z_edges, field, cmap='viridis', norm=norm,
                              shading='flat', rasterized=True)
        ax.set(xlim=(0, length), ylim=(depth, 0), ylabel='Depth z (km)', title=title)
        ax.set_aspect('equal')
        ax.set_yticks(np.arange(0, depth + 1))
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('x (km)')

    for name, xmin, xmax in (('left', 0, fault), ('right', fault, length)):
        block = p[name]
        boundaries = block['boundaries_km']
        axes[0].hlines(boundaries, xmin, xmax, color='white', linewidth=1.1)
        edges = [0, *boundaries, depth]
        for top, bottom, speed in zip(edges[:-1], edges[1:], block['vp_km_s']):
            axes[0].text((xmin + xmax) / 2, (top + bottom) / 2, f'{speed:.2f}',
                         ha='center', va='center', color='white', weight='bold',
                         bbox=dict(facecolor='black', alpha=.28, edgecolor='none', pad=2))
    axes[0].axvline(fault, color='black', linewidth=1.6)
    axes[0].text(fault + .2, depth - .2, f'Fault at x = {fault:g} km', rotation=90,
                 ha='left', va='bottom', fontsize=9,
                 bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
    block = p['initial']
    axes[1].hlines(block['boundaries_km'], 0, length, color='white', linewidth=1.1)
    edges = [0, *block['boundaries_km'], depth]
    for top, bottom, speed in zip(edges[:-1], edges[1:], block['vp_km_s']):
        axes[1].text(length / 2, (top + bottom) / 2, f'{speed:.2f}',
                     ha='center', va='center', color='white', weight='bold',
                     bbox=dict(facecolor='black', alpha=.28, edgecolor='none', pad=2))
    fig.colorbar(image, ax=axes, label='Vp (km/s)', shrink=.9, pad=.02)
    fig.suptitle('Vertical section at y = 3.5 km | Geometry preview only', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}.{extension}', dpi=180)
    plt.close(fig)
    (OUTPUT / f'{STEM}.json').write_text(json.dumps(p, indent=2) + '\n')
    print('Preview only; no forward/inverse solver called.')
    print(OUTPUT / f'{STEM}.png')


if __name__ == '__main__':
    main()
