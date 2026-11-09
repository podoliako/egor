"""Plot published GJI 2021 Vp on the real-pick domain, without extrapolation.

Run from projects/EMTomo:
    python -B paper/paper_scripts/make_geysers_published_vp_sections.py
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/geysers_2021/Geysers_Vpmodel_Guo-Thurber2021GJI.CSV'
GEOMETRY = ROOT / 'data/geysers_2011/real_experiment_v1/geometry.json'
OUTPUT = ROOT / 'paper/figures/real_picks'
STEM = 'geysers_gji2021_vp_domain_middle_sections'
SPACING_KM = 0.1


def project_nodes(latitude, longitude, depth, geometry):
    north = (latitude - geometry['reference_latitude']) * geometry['north_km_per_latitude_degree']
    east = (longitude - geometry['reference_longitude']) * (
        111.320 * np.cos(np.deg2rad(geometry['reference_latitude'])))
    angle = np.deg2rad(geometry['x_azimuth_degrees'])
    offset = np.asarray(geometry['geographic_reference_in_new_xy_m']) / 1000
    return np.column_stack((
        north * np.cos(angle) + east * np.sin(angle) + offset[0],
        -north * np.sin(angle) + east * np.cos(angle) + offset[1],
        depth + geometry['top_elevation_m'] / 1000,
    ))


class LayeredModel:
    """Linear horizontal triangles and vertical segments; no outside fill."""

    def __init__(self, table, geometry):
        self.nodes = project_nodes(table['Latitude'], table['Longitude'], table['Depthkm'], geometry)
        if not np.isfinite(self.nodes).all():
            raise ValueError('Non-finite source coordinates')
        if not np.isfinite(table['Vpkms']).all() or np.any(table['Vpkms'] <= 0):
            raise ValueError('Invalid source velocity')
        resolution = table['Scaled_resolvability']
        if not np.isfinite(resolution).all() or np.any((resolution < 0) | (resolution > 1)):
            raise ValueError('Invalid scaled resolvability')
        self.z = np.unique(self.nodes[:, 2])
        first = self.nodes[:, 2] == self.z[0]
        self.xy = self.nodes[first, :2]
        if len(np.unique(self.xy, axis=0)) != len(self.xy) or len(self.z) < 2:
            raise ValueError('Duplicate horizontal nodes or insufficient depth levels')
        values = []
        for z in self.z:
            mask = self.nodes[:, 2] == z
            # A shared triangulation is valid only if every layer has the same nodes/order.
            np.testing.assert_array_equal(self.nodes[mask, :2], self.xy)
            values.append(np.column_stack((table['Vpkms'][mask], resolution[mask])))
        self.values = np.asarray(values)
        self.triangulation = Delaunay(self.xy)

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        result = np.full((len(points), 2), np.nan)
        simplex = self.triangulation.find_simplex(points[:, :2])
        valid = ((simplex >= 0) & np.isfinite(points).all(axis=1)
                 & (points[:, 2] >= self.z[0]) & (points[:, 2] <= self.z[-1]))
        if not valid.any():
            return result.reshape(*shape, 2)
        queries, simplex = points[valid], simplex[valid]
        transform = self.triangulation.transform[simplex]
        bary = np.einsum('nij,nj->ni', transform[:, :2], queries[:, :2] - transform[:, 2])
        bary = np.column_stack((bary, 1 - bary.sum(axis=1)))
        vertices = self.triangulation.simplices[simplex]
        lower = np.clip(np.searchsorted(self.z, queries[:, 2], side='right') - 1, 0, len(self.z) - 2)
        fraction = (queries[:, 2] - self.z[lower]) / (self.z[lower + 1] - self.z[lower])
        low = np.einsum('ni,nij->nj', bary, self.values[lower[:, None], vertices])
        high = np.einsum('ni,nij->nj', bary, self.values[(lower + 1)[:, None], vertices])
        result[valid] = low * (1 - fraction[:, None]) + high * fraction[:, None]
        return result.reshape(*shape, 2)


def section_points(extent, horizontal, vertical, fixed):
    edges = [np.linspace(0, extent[axis], round(extent[axis] / SPACING_KM) + 1)
             for axis in (horizontal, vertical)]
    centres = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    uu, vv = np.meshgrid(*centres)
    points = np.broadcast_to(extent / 2, (*uu.shape, 3)).copy()
    points[..., horizontal], points[..., vertical] = uu, vv
    return edges, centres, points


def validate(model, table, geometry):
    reproduced = model.sample(model.nodes)
    np.testing.assert_allclose(reproduced[:, 0], table['Vpkms'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(reproduced[:, 1], table['Scaled_resolvability'], rtol=0, atol=1e-10)
    centre_xy = model.xy.mean(axis=0)
    outside = np.array([
        [*centre_xy, model.z[0] - .01], [*centre_xy, model.z[-1] + .01],
        [model.xy[:, 0].min() - 1, centre_xy[1], model.z.mean()],
        [model.xy[:, 0].max() + 1, centre_xy[1], model.z.mean()],
    ])
    assert np.isnan(model.sample(outside)).all(), 'Extrapolation must remain absent'
    midpoint = model.nodes[:len(model.xy)].copy()
    midpoint[:, 2] = (model.z[0] + model.z[1]) / 2
    np.testing.assert_allclose(model.sample(midpoint), model.values[:2].mean(axis=0), rtol=0, atol=1e-10)
    reference = project_nodes(np.array([geometry['reference_latitude']]),
                              np.array([geometry['reference_longitude']]), np.array([0.]), geometry)
    np.testing.assert_allclose(reference[0], [
        * (np.asarray(geometry['geographic_reference_in_new_xy_m']) / 1000),
        geometry['top_elevation_m'] / 1000])


def main():
    geometry = json.loads(GEOMETRY.read_text())
    extent = np.asarray(geometry['domain_extent_m'], dtype=float) / 1000
    np.testing.assert_array_equal(geometry['domain_origin_m'], [0, 0, 0])
    table = np.genfromtxt(SOURCE, delimiter=',', names=True)
    model = LayeredModel(table, geometry)
    validate(model, table, geometry)
    specs = [('XZ', 0, 2, 1), ('YZ', 1, 2, 0), ('XY', 0, 1, 2)]
    sections = [(spec, *section_points(extent, *spec[1:])) for spec in specs]
    samples = [model.sample(points) for _, _, _, points in sections]
    finite = np.concatenate([values[..., 0][np.isfinite(values[..., 0])] for values in samples])
    norm = Normalize(np.floor(finite.min() * 5) / 5, np.ceil(finite.max() * 5) / 5)
    plt.rcParams.update({'font.size': 10, 'hatch.linewidth': .3})
    fig = plt.figure(figsize=(16, 10), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[extent[0], extent[1]],
                           height_ratios=[extent[2], extent[1]])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    note = fig.add_subplot(grid[1, 1])
    note.axis('off')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    metadata = []
    for ax, (spec, edges, centres, points), values in zip(axes, sections, samples):
        name, horizontal, vertical, fixed = spec
        vp, resolution = values[..., 0], values[..., 1]
        covered = np.isfinite(vp)
        image = ax.pcolormesh(*edges, np.ma.masked_invalid(vp), cmap=cmap, norm=norm,
                              shading='flat', rasterized=True)
        low = np.ma.masked_where(~covered, (resolution <= .4).astype(float))
        if np.any(low == 1):
            hatching = ax.contourf(*centres, low, levels=[.5, 1.5], colors='none', hatches=['////'])
            hatching.set_edgecolor('0.45')
        labels = ['X (km; azimuth 135°)', 'Y (km; azimuth 225°)', 'Z below model top (km)']
        fixed_name = 'XYZ'[fixed]
        ax.set(xlabel=labels[horizontal], ylabel=labels[vertical],
               title=f'{name} section | {fixed_name} = {extent[fixed] / 2:g} km',
               xlim=(0, extent[horizontal]), ylim=(0, extent[vertical]))
        if vertical == 2:
            ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        metadata.append({'plane': name, 'fixed_axis': fixed_name,
                         'fixed_coordinate_km': float(extent[fixed] / 2),
                         'sample_count': int(vp.size), 'covered_samples': int(covered.sum()),
                         'blank_samples': int((~covered).sum()),
                         'interpolated_resolvability_gt_0p4_samples': int((covered & (resolution > .4)).sum()),
                         'vp_range_km_s': [float(vp[covered].min()), float(vp[covered].max())]})
    note.legend(handles=[
        Patch(facecolor='white', edgecolor='0.45', hatch='////',
              label='Interpolated resolvability ≤ 0.4'),
        Patch(facecolor='white', edgecolor='0.6', label='White: outside source coverage'),
    ], loc='upper left', frameon=False, fontsize=10)
    note.text(.04, .68,
              'Published model: Guo & Thurber\nGJI 2021 · DOI: 10.1093/gji/ggab017\n'
              'Not the JGR 2022 temporal models.\n\n'
              'Our domain: 23 × 11 × 7 km\nModel top: +1.166 km above sea level\n'
              'Z = source sea-level depth + 1.166 km\n\n'
              'Linear interpolation within source\ntriangles and between depth levels.\n'
              'No extrapolation or background fill.\n\n'
              '100 m plotting samples are not\n100 m tomographic resolution.\n'
              'Low-resolvability values are hatched.\n\n'
              'Geographic datum is not specified\nin the source Readme; sea-level\n'
              'compatibility is assumed.',
              transform=note.transAxes, va='top', fontsize=10, linespacing=1.45)
    fig.colorbar(image, ax=axes, label='Published Vp (km/s)', shrink=.85, pad=.025)
    fig.suptitle('Published Geysers Vp in our real-pick domain | three middle orthogonal sections', fontsize=15)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}.{extension}', dpi=190)
    plt.close(fig)
    report = {
        'source_article_doi': '10.1093/gji/ggab017', 'truth_available': False,
        'domain_extent_km': extent.tolist(), 'geometry': geometry,
        'plot_sampling_km': SPACING_KM,
        'interpolation': 'Delaunay-linear in projected XY, linear between source depth levels',
        'outside_policy': 'NaN / white; no extrapolation, nearest fill or initial-model substitution',
        'resolution_policy': 'Interpolate published scaled resolvability with the same weights; hatch <= 0.4',
        'color_limits_km_s': [float(norm.vmin), float(norm.vmax)], 'sections': metadata,
        'checks_passed': ['all source Vp/resolvability nodes reproduced within 1e-10',
                          'outside horizontal/depth support returns NaN',
                          'vertical midpoint interpolation', 'geographic/sea-level reference mapping'],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SOURCE, GEOMETRY, Path(__file__))},
    }
    (OUTPUT / f'{STEM}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(OUTPUT / f'{STEM}.png')


if __name__ == '__main__':
    main()
