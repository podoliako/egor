"""Compare the saved 1-km ball-holdout runs with GJI 2021 at Y=3 km.

No solvers, no smoothing of inversion cells, no extrapolation of published Vp.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Patch
import numpy as np

from make_geysers_published_vp_sections import GEOMETRY, SOURCE, LayeredModel, validate
from make_geysers_real_published_comparison import completed_model, sha

ROOT = Path(__file__).resolve().parents[2]
STEM = 'geysers_real_ball10_it15_soft_hard_gji2021_y3km_1km'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    queue, output = args.queue.resolve(), args.output.resolve()
    targets = [output / sub / f'{STEM}_{layout}.{extension}'
               for layout in ('stacked', 'side_by_side')
               for sub, extension in (('png', 'png'), ('source', 'pdf'))]
    targets += [output / f'{STEM}.json']
    for path in targets:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    geometry = json.loads(GEOMETRY.read_text())
    np.testing.assert_array_equal(geometry['domain_extent_m'], [23000, 11000, 7000])
    initial = queue / 'inputs/initial_velocity_m_s.npy'
    sources = {str(p): sha(p) for p in (GEOMETRY, SOURCE, initial, queue / 'manifest.json',
                                        queue / 'heldout/ball.json', Path(__file__),
                                        Path(__file__).with_name('make_geysers_published_vp_sections.py'),
                                        Path(__file__).with_name('make_geysers_real_published_comparison.py'))}
    fields, labels = [], []
    for mode in ('soft', 'hard'):
        velocity, provenance, cell = completed_model(queue, mode, initial, geometry)
        if cell != 1:
            raise ValueError('This comparison requires saved 1 km velocity cells')
        fields.append(velocity[:, 3, :].T / 1000)
        labels.append(f'EMTomo {mode} | {provenance["completed_updates"]} updates | 1 km cells')
        sources.update(provenance['source_sha256'])
    table = np.genfromtxt(SOURCE, delimiter=',', names=True)
    published = LayeredModel(table, geometry)
    validate(published, table, geometry)
    xedges, zedges = np.linspace(0, 23, 231), np.linspace(0, 7, 71)
    x, z = (xedges[:-1] + xedges[1:]) / 2, (zedges[:-1] + zedges[1:]) / 2
    xx, zz = np.meshgrid(x, z)
    points = np.stack((xx, np.full_like(xx, 3.), zz), axis=-1)
    sampled = published.sample(points)
    vp, resolution = sampled[..., 0], sampled[..., 1]
    if not np.isfinite(vp).any():
        raise ValueError('Published model has no coverage on the section')
    fields.append(vp)
    labels.append('Guo & Thurber | GJI 2021 | published interpolation')
    finite = np.concatenate([field[np.isfinite(field)] for field in fields])
    norm = Normalize(np.floor(finite.min() * 5) / 5, np.ceil(finite.max() * 5) / 5)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    ball = json.loads((queue / 'heldout/ball.json').read_text())
    centre, radius = np.asarray(ball['centre_m']) / 1000, ball['radius_m'] / 1000
    intersection = radius**2 - (3. - centre[1])**2
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'hatch.linewidth': .35})
    for layout, shape, size in (('stacked', (3, 1), (13, 12)),
                                ('side_by_side', (1, 3), (21, 5))):
        fig, axes = plt.subplots(*shape, figsize=size, layout='constrained')
        axes = np.asarray(axes).reshape(-1)
        for index, (ax, field, label) in enumerate(zip(axes, fields, labels)):
            edges = (np.arange(24), np.arange(8)) if index < 2 else (xedges, zedges)
            image = ax.pcolormesh(*edges, np.ma.masked_invalid(field), cmap=cmap, norm=norm,
                                  shading='flat', rasterized=True)
            if index == 2:
                low = np.ma.masked_where(~np.isfinite(vp), (resolution <= .4).astype(float))
                if np.any(low == 1):
                    ax.contourf(x, z, low, levels=[.5, 1.5], colors='none', hatches=['////'])
            if intersection >= 0:
                ax.add_patch(Circle(centre[[0, 2]], np.sqrt(intersection), fill=False,
                                    linestyle='--', linewidth=1.4, edgecolor='black'))
            ax.set(title=label, xlabel='X (km)', ylabel='Depth below model top (km)',
                   xlim=(0, 23), ylim=(7, 0), aspect='equal')
            ax.set_xticks([0, 5, 10, 15, 20, 23])
        fig.colorbar(image, ax=list(axes), label='Vp (km/s) | shared scale', shrink=.85, pad=.02)
        fig.suptitle('Geysers | identical XZ section at Y=3 km | ball-holdout inversion\n'
                     'EMTomo: saved 1 km cells, no smoothing; Y boundary uses cell [3, 4) km', fontsize=14)
        legend = [Patch(facecolor='none', edgecolor='black', linestyle='--', label='Holdout sphere intersection'),
                  Patch(facecolor='white', edgecolor='.5', hatch='////', label='GJI 2021 resolvability ≤ 0.4'),
                  Patch(facecolor='white', edgecolor='.5', label='White: outside published coverage')]
        fig.legend(handles=legend, loc='outside lower center', ncol=3, fontsize=10)
        for sub, extension in (('png', 'png'), ('source', 'pdf')):
            path = output / sub / f'{STEM}_{layout}.{extension}'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(path)
        plt.close(fig)
    report = dict(queue=str(queue), section_y_km=3., em_cell_size_m=1000,
                  em_y_cell_km=[3, 4], em_policy='Actual saved piecewise-constant cells; no smoothing',
                  published_article_doi='10.1093/gji/ggab017',
                  published_policy='Native published field interpolated at XZ points at exact Y=3 km; not re-inverted or coarsened',
                  published_interpolation='Delaunay-linear in projected XY; linear between source depth levels',
                  published_plot_sampling_m=100, plotting_samples_are_resolution=False,
                  outside_policy='NaN/white; no extrapolation or background fill',
                  low_resolvability_hatched_threshold=.4,
                  depth_policy='Source sea-level depth plus 1.166 km; same local axes as EMTomo',
                  published_is_ground_truth=False, solver_calls=0,
                  color_limits_km_s=[float(norm.vmin), float(norm.vmax)],
                  source_sha256=sources, outputs=[str(p) for p in targets[:-1]])
    for path, digest in sources.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Input changed during plotting: {path}')
    targets[-1].write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(targets[-1])


if __name__ == '__main__':
    main()
