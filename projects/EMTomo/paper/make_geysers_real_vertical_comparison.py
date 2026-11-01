"""Plot saved real-pick initial/hard/soft/tomoDD mid-Y sections; no solvers.

python -B paper/make_geysers_real_vertical_comparison.py
Native tomoDD is interpolated in geographic lon/lat/sea-level depth, not in the
rotated EMTomo Cartesian axes. An incomplete hard run is explicitly labelled.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runs/geysers_real_20261031_pick01'
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
OUTPUT = ROOT / 'paper/figures'
STEM = 'geysers_real_pick01_vertical_y5p5km'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def em_model(mode):
    result = json.loads((STATE / f'cases/{mode}/result.json').read_text())
    directory = Path(result['run_dir'])
    meta = json.loads((directory / 'meta.json').read_text())
    completed = sorted(int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json'))
    assert completed == list(range(len(completed))) and completed
    paths = [directory / 'meta.json', STATE / f'cases/{mode}/result.json']
    if (directory / 'final_model.npy').exists():
        assert len(completed) == meta['run_params']['n_cycles']
        paths.append(directory / 'final_model.npy')
        model = np.load(paths[-1], allow_pickle=False)
        label = f'EMTomo {mode} | final, {len(completed)} updates'
    else:
        last = directory / f'iter_{completed[-1]}'
        v = np.load(last / 'model.npy', allow_pickle=False)
        ds = np.load(last / 'delta_s.npy', allow_pickle=False).reshape(v.shape)
        model = 1 / np.maximum(1 / v + ds, 1e-12)
        step = meta['run_params']['max_velocity_step_fraction']
        if step is not None:
            model = np.clip(model, v * (1-step), v * (1+step))
        paths.extend([last / 'model.npy', last / 'delta_s.npy'])
        following = directory / f'iter_{completed[-1]+1}/model.npy'
        if following.exists():
            np.testing.assert_array_equal(model, np.load(following, allow_pickle=False))
            paths.append(following)
        label = f'EMTomo {mode} | {len(completed)} updates, run stopped early'
    assert model.shape == (23, 11, 7) and np.isfinite(model).all() and (model > 0).all()
    return model / 1000, label, paths, directory


def main():
    geometry = json.loads((DATA / 'geometry.json').read_text())
    extent = np.asarray(geometry['domain_extent_m']) / 1000
    np.testing.assert_array_equal(extent, [23, 11, 7])
    y = float(extent[1] / 2)
    xedges, zedges = np.arange(185) * .125, np.arange(57) * .125
    x, z = (xedges[:-1] + xedges[1:]) / 2, (zedges[:-1] + zedges[1:]) / 2
    xx, zz = np.meshgrid(x, z, indexing='ij')
    hard, hard_label, hard_paths, hard_dir = em_model('hard')
    soft, soft_label, soft_paths, soft_dir = em_model('soft')
    initial_path = STATE / 'inputs/initial_velocity_m_s.npy'
    initial = np.load(initial_path, allow_pickle=False) / 1000
    for directory in (hard_dir, soft_dir):
        np.testing.assert_array_equal(initial, np.load(directory / 'initial_model.npy') / 1000)
    ix, iz = np.floor(x).astype(int), np.floor(z).astype(int)
    fields = [a[:, int(y), :][ix[:, None], iz[None, :]] for a in (initial, hard, soft)]

    work = STATE / 'cases/tomodd/work'
    result_path = STATE / 'cases/tomodd/result.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'completed'
    assert sha(work / 'Vp_model.dat') == result['artifact_sha256']['Vp_model.dat']
    lines = (work / 'MOD').read_text().splitlines()
    nx, ny, nz = map(int, lines[0].split()[1:])
    lon, lat, depth = (np.fromstring(line, sep=' ') for line in lines[1:4])
    assert tuple(map(len, (lon, lat, depth))) == (nx, ny, nz)
    assert all(np.all(np.diff(axis) > 0) for axis in (lon, lat, depth))
    native = np.fromstring((work / 'Vp_model.dat').read_text(), sep=' ')
    assert native.size == nx*ny*nz
    native = native.reshape(nz, ny, nx)
    # Confirm MOD axis order/units using its independently specified regional start.
    mod_values = np.fromstring('\n'.join(lines[4:]), sep=' ')
    assert mod_values.size == 2*nx*ny*nz
    initial_native = mod_values[:nx*ny*nz].reshape(nz, ny, nx)
    regional_path = STATE / 'inputs/regional_model.json'
    regional = json.loads(regional_path.read_text())
    predicted = np.interp(depth + geometry['top_elevation_m']/1000,
                          regional['local_depth_km'], regional['vp_km_s'])
    np.testing.assert_allclose(initial_native, np.broadcast_to(predicted[:, None, None], initial_native.shape),
                               rtol=0, atol=1e-6)
    # Geographic reference is at (13,3.5) km, NOT at the new domain centre.
    offset_x, offset_y = np.asarray(geometry['geographic_reference_in_new_xy_m']) / 1000
    angle = np.deg2rad(geometry['x_azimuth_degrees'])
    dx, dy = xx-offset_x, y-offset_y
    north, east = dx*np.cos(angle)-dy*np.sin(angle), dx*np.sin(angle)+dy*np.cos(angle)
    latitude = geometry['reference_latitude'] + north/geometry['north_km_per_latitude_degree']
    lon_scale = 111.320*np.cos(np.deg2rad(geometry['reference_latitude']))
    longitude = geometry['reference_longitude'] + east/lon_scale
    sea_depth = zz - geometry['top_elevation_m']/1000
    points = np.stack((sea_depth, latitude, longitude), axis=-1)
    fields.append(RegularGridInterpolator((depth, lat, lon), native, bounds_error=True)(points))
    assert all(np.isfinite(a).all() and np.all((a > 1) & (a < 10)) for a in fields)
    labels = ['Initial 1-D | actual EMTomo 1 km cells', hard_label, soft_label,
              'tomoDD | final, 3 velocity updates']
    vmin = float(np.floor(min(a.min() for a in fields)*5)/5)
    vmax = float(np.ceil(max(a.max() for a in fields)*5)/5)
    norm = Normalize(vmin, vmax)

    def panel(ax, index):
        im = ax.pcolormesh(xedges, zedges, fields[index].T, cmap='viridis', norm=norm,
                           shading='flat', rasterized=True)
        ax.set(xlim=(0, 23), ylim=(7, 0), xlabel='x (km)', ylabel='Depth below model top (km)',
               title=labels[index])
        ax.set_aspect('equal')
        ax.set_xticks([0, 5, 10, 15, 20, 23])
        ax.set_yticks([0, 2, 4, 6, 7])
        return im

    # Identically sized panels: one initial section above three side-by-side runs.
    fig = plt.figure(figsize=(18, 6), layout='constrained')
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, 1])]
    axes += [fig.add_subplot(grid[1, i]) for i in range(3)]
    for i, ax in enumerate(axes):
        image = panel(ax, i)
    fig.colorbar(image, ax=axes, label='Vp (km/s)', shrink=.85, pad=.02)
    fig.suptitle(f'Geysers real P arrivals | middle XZ section at Y = {y:g} km', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}_side_by_side.{extension}', dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), layout='constrained')
    for i, ax in enumerate(axes):
        image = panel(ax, i)
    fig.colorbar(image, ax=axes, label='Vp (km/s)', shrink=.9, pad=.02)
    fig.suptitle(f'Geysers real P arrivals | middle XZ section at Y = {y:g} km', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}_stacked.{extension}', dpi=180)
    plt.close(fig)
    paths = [DATA / 'geometry.json', initial_path, regional_path, result_path, work / 'MOD', work / 'Vp_model.dat']
    paths += hard_paths + soft_paths
    metadata = dict(section_y_km=y, extent_km=extent.tolist(), color_limits_km_s=[vmin, vmax],
                    labels=labels, hard_status='failed; last fully completed 21 updates, not final 24',
                    tomoDD_retained_events=result['retained_events'],
                    native_projection='geographic lon/lat/sea-level depth interpolated at rotated local XZ coordinates',
                    em_projection='nearest 1 km cell model, no smoothing',
                    initial_policy='Actual EMTomo cell-centre regional start; tomoDD samples the same profile on its native nodes.',
                    truth_available=False, no_solver_calls=True,
                    source_sha256={str(p): sha(p) for p in paths})
    (OUTPUT / f'{STEM}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metadata.items() if k != 'source_sha256'}, indent=2))
    print(OUTPUT / f'{STEM}_side_by_side.png')


if __name__ == '__main__':
    main()
