"""Matched final-checkpoint slices. Plane selection uses sphere truth geometry only."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'paper'))
from evaluate_metrics import native_helpers, sha256, template_amplitude

LENGTHS = np.array([20., 7., 7.])
VP_LIMITS = (2839.211944580078, 6516.418171324893)
METHODS = ('tomodd', 'hard', 'soft_d1', 'soft_d2')
LABELS = {'truth': 'Physical truth', 'initial': 'Initial: constant cells',
          'oracle_em': 'Oracle: constant cells', 'oracle_dd': 'Oracle: trilinear',
          'tomodd': 'tomoDD, iteration 3', 'hard': 'EMTomo hard, 15 → 1',
          'soft_d1': 'EMTomo soft, 15 → 10, dmin 1',
          'soft_d2': 'EMTomo soft, 15 → 10, dmin 2'}


def fit_plane(points):
    center = points.mean(axis=0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    u = np.array([1., 0., 0.]) - normal[0] * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return center, normal, u, v


def choose_planes(centers, radii):
    candidates = []
    for ids in itertools.combinations(range(len(centers)), 4):
        origin, normal, u, v = fit_plane(centers[list(ids)])
        distances = abs((centers - origin) @ normal)
        score = float(np.max(distances[list(ids)] / radii[list(ids)]))
        candidates.append((score, ids, origin, normal, u, v, distances))
    candidates.sort(key=lambda row: (row[0], row[1]))
    first = candidates[0]
    missing = set(range(len(centers))) - set(first[1])
    second = next(row for row in candidates if missing <= set(row[1]))
    return [dict(name=f'oblique_{letter}', sphere_ids=[i + 1 for i in ids],
                 origin=origin, normal=normal, u=u, v=v, distances=distances,
                 max_selected_distance_over_radius=score)
            for letter, (score, ids, origin, normal, u, v, distances) in zip(('A', 'B'), (first, second))]


def plane_grid(plane, spacing=.125):
    if plane['name'] == 'horizontal':
        u = (np.arange(160) + .5) * spacing
        v = (np.arange(56) + .5) * spacing
        uu, vv = np.meshgrid(u, v)
        xyz = np.stack((uu, vv, np.full_like(uu, 1.4375)), axis=-1)
        return xyz, (0., 20., 0., 7.), np.ones(uu.shape, dtype=bool)
    corners = np.array(list(itertools.product(*zip(np.zeros(3), LENGTHS))))
    projected = np.column_stack(((corners - plane['origin']) @ plane['u'],
                                 (corners - plane['origin']) @ plane['v']))
    low = np.floor(projected.min(0) / spacing) * spacing
    high = np.ceil(projected.max(0) / spacing) * spacing
    u = np.arange(low[0] + spacing / 2, high[0], spacing)
    v = np.arange(low[1] + spacing / 2, high[1], spacing)
    uu, vv = np.meshgrid(u, v)
    xyz = plane['origin'] + uu[..., None] * plane['u'] + vv[..., None] * plane['v']
    valid = ((xyz >= 0) & (xyz < LENGTHS)).all(axis=-1)
    return xyz, (low[0], high[0], low[1], high[1]), valid


def sample_blocks(values_xyz, spacing, xyz, valid):
    result = np.full(valid.shape, np.nan)
    indices = np.floor(xyz[valid] / spacing).astype(int)
    result[valid] = values_xyz[tuple(indices.T)]
    return result


def sample_nodes(nodes_zyx, axes_zyx, xyz, valid):
    points = xyz[valid][:, [2, 1, 0]] - np.array([0., 3.5, 10.])
    result = np.full(valid.shape, np.nan)
    result[valid] = RegularGridInterpolator(axes_zyx, nodes_zyx, bounds_error=True)(points)
    return result


def score(field, truth, mask=None):
    selected = np.isfinite(truth) if mask is None else mask
    residual = field[selected] - truth[selected]
    return dict(rmse_m_s=float(np.sqrt(np.mean(residual**2))), mae_m_s=float(np.mean(abs(residual))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    state = args.study_root
    manifest = json.loads((state / 'manifest.json').read_text())
    status = json.loads((state / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('All eight final checkpoints must be complete')
    snapshot = Path(manifest['snapshot_root'])
    helper = snapshot / 'benchmark/tomodd_benchmark/compare_emtomo_flat4.py'
    assert sha256(helper) == manifest['snapshot_sha256'][str(helper.relative_to(snapshot))]
    native = native_helpers(helper)
    hashes = {str(helper): sha256(helper), str(state / 'manifest.json'): sha256(state / 'manifest.json')}
    def record(path):
        hashes[str(path)] = sha256(path)
        return path
    synthetic_case = next(c for c in manifest['cases'] if c['geometry'] == 'synthetic')
    source = Path(manifest['forward_root']) / 'experiments/input' / synthetic_case['experiment_id']
    parameters = native.generation_parameters(json.loads(record(source / 'generation.json').read_text()))
    centers = np.array([a['center_km'] for a in parameters['anomalies']])
    radii = np.array([a['radius_km'] for a in parameters['anomalies']])
    grid_centers = tuple((np.arange(n) + .5) * .125 for n in (160, 56, 56))
    models, volume_metrics = {}, {}
    reference = None
    for geometry in ('synthetic', 'geysers'):
        run = state / 'outputs' / f'{geometry}-tomodd-sigma1'
        info = json.loads(record(run / 'worker_info.json').read_text())
        assert info['state'] == 'completed'
        config = json.loads(record(run / 'config.json').read_text())
        axes = tuple(np.array(config['tomodd_grid'][key]) for key in ('depths_km', 'y_km', 'x_km'))
        nodes = native.read_velocity_history(record(run / 'work/inversion/tomoDD.vel'), (9, 9, 22), 3)
        with np.load(record(run / 'state.npz'), allow_pickle=False) as data:
            truth_xyz = data['physical_velocity_km_s'] * 1000
            oracle_dd = data['oracle_vp'] * 1000
            np.testing.assert_allclose(nodes, data['final_vp'] * 1000, rtol=0, atol=1e-8)
        if reference is not None:
            np.testing.assert_allclose(truth_xyz, reference[0], rtol=0, atol=1e-8)
            np.testing.assert_allclose(oracle_dd, reference[1], rtol=0, atol=1e-6)
        reference = truth_xyz, oracle_dd
        truth = truth_xyz.transpose(2, 1, 0)
        regions = native.region_masks(truth, grid_centers, parameters)
        background_volume = np.array([3800., 4600., 5200., 5700.])[
            np.searchsorted([.7, 1.9, 3.4], grid_centers[2], side='right')][:, None, None]
        models[geometry] = {'tomodd': ('nodes', nodes, axes)}
        for case in manifest['cases']:
            if case['method'] != 'emtomo' or case['geometry'] != geometry:
                continue
            result = json.loads(record(state / 'cases' / case['id'] / 'result.json').read_text())
            assert result['status'] == 'completed'
            run_em = Path(result['run_dir'])
            path = record(run_em / 'final_model.npy')
            assert hashes[str(path)] == result['final_model_sha256']
            config_em = case['config']
            key = 'hard' if config_em['candidate_mode'] == 'hard' else f"soft_d{config_em['weights_min_distance']}"
            model = native.positive_model(np.load(path), (20, 7, 7), key)
            models[geometry][key] = ('blocks', model, 1.)
            for i in range(12):
                assert (run_em / f'iter_{i}/complete.json').is_file()
            initial = np.load(record(run_em / 'initial_model.npy'))
            if 'initial_em' in locals():
                np.testing.assert_array_equal(initial, initial_em)
            initial_em = initial
        volume_metrics[geometry] = {}
        for key, (kind, model, geometry_data) in models[geometry].items():
            field = native.evaluate_native(model, geometry_data, grid_centers) if kind == 'nodes' else native.tile_cells(model)
            volume_metrics[geometry][key] = dict(volume=score(field, truth), spheres=score(field, truth, regions['spheres']),
                sphere_amplitudes={name: template_amplitude(field, truth, background_volume, mask)
                                   for name, mask in regions.items() if name.startswith('sphere_')})
        saved = json.loads(record(run / 'results.json').read_text())
        np.testing.assert_allclose(volume_metrics[geometry]['tomodd']['volume']['rmse_m_s'],
                                   saved['final_field_rmse_km_s'] * 1000, rtol=0, atol=1e-6)
    oracle_em = native.constant_oracle(truth)
    planes = [dict(name='horizontal', requested_z_km=1.435, sampled_z_km=1.4375)] + choose_planes(centers, radii)
    all_slices = []
    for plane in planes:
        xyz, extent, valid = plane_grid(plane)
        fields = {'truth': sample_blocks(truth_xyz, .125, xyz, valid),
                  'initial': sample_blocks(initial_em, 1., xyz, valid),
                  'oracle_em': sample_blocks(oracle_em, 1., xyz, valid),
                  'oracle_dd': sample_nodes(oracle_dd, axes, xyz, valid)}
        for geometry in models:
            for key, (kind, model, geometry_data) in models[geometry].items():
                fields[f'{geometry}/{key}'] = (sample_nodes(model, geometry_data, xyz, valid) if kind == 'nodes'
                                               else sample_blocks(model, geometry_data, xyz, valid))
        # Use the layered background at the same truth voxel centers, not a smoothed background.
        depths = (np.floor(xyz[..., 2] / .125) + .5) * .125
        background = np.array([3800., 4600., 5200., 5700.])[np.searchsorted([.7, 1.9, 3.4], depths, side='right')]
        background = np.where(valid, background, np.nan)
        all_slices.append((plane, xyz, extent, valid, fields, background))
    contrast_limit = float(np.ceil(max(np.nanmax(abs(f - bg)) for _, _, _, _, fields, bg in all_slices
                                      for f in fields.values()) / 100) * 100)
    args.output_dir.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(state / 'cache/comparison-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    reports = []
    for plane, xyz, extent, valid, fields, background in all_slices:
        sections = {key: score(value, fields['truth']) for key, value in fields.items()}
        for contrast in (False, True):
            fig, axs = plt.subplots(3, 4, figsize=(19, 8 if plane['name'] == 'horizontal' else 9), constrained_layout=True)
            order = ['truth', 'initial', 'oracle_em', 'oracle_dd'] + [f'{g}/{key}' for g in models for key in METHODS]
            for ax, key in zip(axs.flat, order):
                value = fields[key] - background if contrast else fields[key]
                limits = (-contrast_limit, contrast_limit) if contrast else VP_LIMITS
                image = ax.imshow(value, extent=extent, origin='lower', interpolation='nearest', aspect='equal',
                                  cmap='RdBu_r' if contrast else 'Reds', vmin=limits[0], vmax=limits[1])
                title = LABELS[key.split('/')[-1]]
                if '/' in key:
                    title = ('Artificial' if key.startswith('synthetic/') else 'Geysers') + ': ' + title
                    title += f"\nSlice RMSE {sections[key]['rmse_m_s']:.1f} m/s"
                ax.set(title=title, xlabel='x (km)' if plane['name'] == 'horizontal' else 'u (km)',
                       ylabel='y (km)' if plane['name'] == 'horizontal' else 'v (km)')
                ax.tick_params(labelsize=8)
                ax.title.set_fontsize(10)
                if key == 'truth':
                    for i, (center, radius) in enumerate(zip(centers, radii)):
                        if plane['name'] == 'horizontal':
                            distance = abs(center[2] - 1.4375); location = center[:2]
                        else:
                            distance = plane['distances'][i]
                            location = [(center - plane['origin']) @ plane['u'], (center - plane['origin']) @ plane['v']]
                        if distance < radius:
                            ax.add_patch(Circle(location, np.sqrt(radius**2 - distance**2), fill=False,
                                                edgecolor='black', linewidth=.7, linestyle='--'))
                            ax.text(*location, str(i + 1), ha='center', va='center', fontsize=9)
            description = ('z = 1.4375 km (requested 1.435 km)' if plane['name'] == 'horizontal' else
                           f"{plane['name']}: near sphere centers {', '.join(map(str, plane['sphere_ids']))}; u,v are distances in plane")
            fig.suptitle(f'Matched final checkpoints: EMTomo 12 updates / tomoDD 3 iterations, initial location sigma 1 km\n'
                         + description + ('\nVelocity contrast relative to the known true layered background' if contrast else ''))
            fig.colorbar(image, ax=list(axs.flat), shrink=.75,
                         label='Vp - true layered background (m/s)' if contrast else 'Vp (m/s)', extend='both')
            stem = plane['name'] + ('_contrast' if contrast else '_vp')
            for suffix in ('png', 'pdf'):
                fig.savefig(args.output_dir / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
            plt.close(fig)
        # Archive evaluated values to make numerical inspection independent of plotting.
        np.savez_compressed(args.output_dir / f"{plane['name']}_samples.npz", xyz_km=xyz, valid=valid,
                            background_m_s=background, **{key.replace('/', '_'): value for key, value in fields.items()})
        info = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plane.items()}
        info.update(sample_spacing_km=.125, extent=list(extent), valid_samples=int(valid.sum()), metrics=sections,
                    vp_saturated_samples={key: int((((f < VP_LIMITS[0]) | (f > VP_LIMITS[1])) & valid).sum()) for key, f in fields.items()})
        reports.append(info)
    report = dict(volume_metrics=volume_metrics, planes=reports, vp_color_limits=VP_LIMITS,
                  contrast_color_limits=[-contrast_limit, contrast_limit], source_sha256=hashes,
                  script_sha256=sha256(Path(__file__)),
                  plane_selection='Enumerate 4-sphere subsets, fit PCA plane, minimize maximum selected distance/radius. '
                                  'Second plane must include centers omitted from first. Uses truth geometry only.',
                  sampling='Native nearest cells for truth/EM; native trilinear Vp for tomoDD. No extra smoothing.',
                  oracle_caveat='Oracles minimize full-volume velocity L2, not slice-specific errors.')
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(volume_metrics=volume_metrics, planes=[{k: v for k, v in p.items() if k in
          ('name', 'sphere_ids', 'origin', 'normal', 'distances', 'metrics')} for p in reports]), indent=2))


if __name__ == '__main__':
    main()
