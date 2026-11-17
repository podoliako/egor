"""Post-relocation location shifts, descriptive only; no relocation or solvers.

Run from projects/EMTomo once all six input files are ready:
    python -B paper/paper_scripts/make_geysers_real_location_shifts.py
    python -B paper/paper_scripts/make_geysers_real_location_shifts.py --input-dir PATH

Each MODE_fit.json must contain predictions_sha256, holdout_opened=false and
binding with the existing evaluator keys: model.source_sha256, training_sha256
(the original arrivals.npz file hash), grid_m=125, shortlist=50, dmin=7,
noise=[.01,.05,.02], origin containing 'TRAIN', prediction containing 'MAP'.
Additionally require binding.train_mask_sha256 to bind actual38676 explicitly:
    sha256(np.ascontiguousarray(actual_train_mask, dtype=np.uint8).tobytes()).hexdigest()
The mask is finite(original arrivals) & ~original internal control, in the
original arrivals event/station order, C order, one byte (0/1) per entry.
Unknown binding fields are preserved. Missing/incompatible provenance fails
closed; the forthcoming generator's binding schema must be checked at handoff.
Catalog is read only AFTER loading/validating all training-only fitted positions.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/geysers_2011/real_experiment_v1'
QUEUE = ROOT / 'runs/geysers_real_refine500_fmm125_20261110_110138'
PARENT = ROOT / 'runs/geysers_real_linear35_60_latest_20261110_063859'
INITIAL = ROOT / 'data/geysers_2011/initial_models/linear_vp_3p5_6p0/model.json'
DEFAULT_INPUT = ROOT / 'paper/exp_data/geysers_real_holdout_latest'
STEM = 'geysers_real_location_shifts_latest'
OUTPUT = ROOT / 'paper/exp_data' / STEM
FIGURE = ROOT / 'paper/figures/real_picks' / STEM
MODES = ('initial', 'hard', 'soft')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def verify_file(path, expected, hashes):
    path = Path(path)
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    hashes[path_label(path)] = actual


def mask_sha256(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def validate_geometry(geometry):
    if (geometry['z_direction'] != 'positive_down'
            or geometry['x_azimuth_degrees'] != 135.
            or geometry['y_azimuth_degrees'] != 225.
            or geometry['top_elevation_m'] != 1166.):
        raise ValueError('Expected the real-domain x135/y225, positive-down, +1166 m geometry')
    np.testing.assert_array_equal(geometry['domain_origin_m'], [0, 0, 0])


def geographic_to_local(latitude, longitude, depth_km, geometry):
    """Return local XYZ metres; depth is catalog depth below sea level."""
    latitude, longitude, depth = np.broadcast_arrays(latitude, longitude, depth_km)
    north = (latitude - geometry['reference_latitude']) * geometry['north_km_per_latitude_degree'] * 1000
    east = (longitude - geometry['reference_longitude']) * 111320 * np.cos(np.deg2rad(geometry['reference_latitude']))
    angle = np.deg2rad(geometry['x_azimuth_degrees'])
    offset = geometry['geographic_reference_in_new_xy_m']
    return np.stack((north * np.cos(angle) + east * np.sin(angle) + offset[0],
                     -north * np.sin(angle) + east * np.cos(angle) + offset[1],
                     geometry['top_elevation_m'] + 1000 * depth), axis=-1)


def local_to_geographic(position_m, geometry):
    """Inverse transform, returning latitude, longitude, sea-level depth km."""
    position = np.asarray(position_m, dtype=float)
    xy = position[..., :2] - geometry['geographic_reference_in_new_xy_m']
    angle = np.deg2rad(geometry['x_azimuth_degrees'])
    north = xy[..., 0] * np.cos(angle) - xy[..., 1] * np.sin(angle)
    east = xy[..., 0] * np.sin(angle) + xy[..., 1] * np.cos(angle)
    return np.stack((geometry['reference_latitude'] + north / (1000 * geometry['north_km_per_latitude_degree']),
                     geometry['reference_longitude'] + east / (111320 * np.cos(np.deg2rad(geometry['reference_latitude']))),
                     (position[..., 2] - geometry['top_elevation_m']) / 1000), axis=-1)


def unique_ids(values, label):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'US':
        raise ValueError(f'{label}: expected a one-dimensional string array')
    ids = values.astype(str).tolist()
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{label}: empty or duplicate IDs')
    return ids


def alignment_indices(actual, expected, label):
    ids = unique_ids(actual, label)
    if set(ids) != set(expected):
        raise ValueError(f'{label}: missing or extra IDs')
    index = {value: i for i, value in enumerate(ids)}
    return np.array([index[value] for value in expected], dtype=int)


def validate_predictions(arrays, event_ids, station_ids):
    """Align by IDs, not file row order; reject missing, extra or nonfinite data."""
    required = {'position_m', 'event_ids', 'station_ids', 'predicted_relative_s', 'origin_relative_s', 'train_nll'}
    if not required.issubset(arrays):
        raise ValueError(f'Missing prediction fields: {sorted(required - set(arrays))}')
    event_order = alignment_indices(arrays['event_ids'], event_ids, 'prediction events')
    alignment_indices(arrays['station_ids'], station_ids, 'prediction stations')
    n, s = len(event_ids), len(station_ids)
    for key, shape in [('position_m', (n, 3)), ('predicted_relative_s', (n, s)),
                       ('origin_relative_s', (n,)), ('train_nll', (n,))]:
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'Invalid prediction field {key}: expected finite {shape}')
    return np.asarray(arrays['position_m'], dtype=float)[event_order]


def validate_binding(ledger, training_hash, train_mask_hash, model_path, model_hash, hashes):
    if ledger.get('holdout_opened') is not False:
        raise ValueError('Fit must declare holdout_opened=false')
    binding = ledger['binding']
    for key, expected in [('shortlist', 50), ('dmin', 7), ('grid_m', 125.), ('noise', [.01, .05, .02])]:
        if binding.get(key) != expected:
            raise ValueError(f'Incompatible binding.{key}; expected {expected!r}')
    if binding.get('training_sha256') != training_hash:
        raise ValueError('binding.training_sha256 must match original manifest-verified arrivals.npz')
    if binding.get('train_mask_sha256') != train_mask_hash:
        raise ValueError('Missing/mismatched binding.train_mask_sha256: bind the actual38676 boolean mask '
                         'as SHA256 of C-order uint8 bytes in original arrivals event/station order')
    if 'train' not in binding.get('origin', '').lower() or 'map' not in binding.get('prediction', '').lower():
        raise ValueError('Binding must identify TRAIN-only origin and MAP prediction protocol')
    sources = binding['model']['source_sha256']
    normalized = {str((ROOT / name).resolve()): digest for name, digest in sources.items()}
    if normalized.get(str(model_path.resolve())) != model_hash:
        raise ValueError('Binding does not reference the expected final velocity model/hash')
    for name, digest in sources.items():
        verify_file(ROOT / name, digest, hashes)
    return binding


def shift_summary(position_m, reference_m):
    """Equal-event descriptive distances and signed shifts; never truth errors."""
    position, reference = np.asarray(position_m, dtype=float), np.asarray(reference_m, dtype=float)
    if (position.shape != reference.shape or position.ndim != 2 or position.shape[1] != 3
            or not len(position) or not np.isfinite(position).all() or not np.isfinite(reference).all()):
        raise ValueError('Summary requires matching nonempty finite N x 3 positions')
    delta = position - reference
    distance = np.linalg.norm(delta, axis=1)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    return dict(event_count=len(position), median_distance_3d_m=float(np.median(distance)),
                p90_distance_3d_m=float(np.percentile(distance, 90)),
                median_horizontal_distance_m=float(np.median(horizontal)),
                p90_horizontal_distance_m=float(np.percentile(horizontal, 90)),
                median_signed_xyz_m=np.median(delta, axis=0).tolist(),
                median_absolute_vertical_shift_m=float(np.median(np.abs(delta[:, 2]))),
                fraction_deeper=float(np.mean(delta[:, 2] > 0)),
                fraction_shallower=float(np.mean(delta[:, 2] < 0)),
                fraction_same_depth=float(np.mean(delta[:, 2] == 0)))


def summarize(positions, catalog):
    return dict(against_catalog={mode: shift_summary(positions[mode], catalog) for mode in MODES},
                against_initial_fitted={mode: shift_summary(positions[mode], positions['initial']) for mode in ('hard', 'soft')},
                hard_soft=dict(convention='soft minus hard; distance is symmetric',
                               **shift_summary(positions['soft'], positions['hard'])))


def indexed_rows(path, key):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = unique_ids(np.asarray([row[key] for row in rows]), str(path))
    return dict(zip(ids, rows))


def load_inputs(input_dir):
    required = [input_dir / f'{mode}_{suffix}' for mode in MODES for suffix in ('predictions.npz', 'fit.json')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Inputs not ready; no waiting or outputs written:\n' + '\n'.join(missing))
    hashes = {}
    manifest_path = DATA / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    hashes[path_label(manifest_path)] = sha(manifest_path)
    for name in ('geometry.json', 'emtomo/arrivals.npz'):
        verify_file(DATA / name, manifest['output_sha256'][name], hashes)
    geometry = json.loads((DATA / 'geometry.json').read_text())
    validate_geometry(geometry)
    with np.load(DATA / 'emtomo/arrivals.npz', allow_pickle=False) as arrays:
        event_ids = unique_ids(arrays['event_ids'], 'source events')
        station_ids = unique_ids(arrays['station_ids'], 'source stations')
        observed = np.isfinite(arrays['arrival_times_s'])
        stations = arrays['station_coordinates_m'].copy()
    if observed.shape != (3471, 31) or len(event_ids) != 3471 or len(station_ids) != 31 or observed.sum() != 42973:
        raise ValueError('Unexpected original real-pick training set')
    info = json.loads(INITIAL.read_text())
    hashes[path_label(INITIAL)] = sha(INITIAL)
    initial_path = INITIAL.parent / info['array_file']
    verify_file(initial_path, info['array_sha256'], hashes)
    model_sources = {'initial': (initial_path, info['array_sha256'])}
    controls = []
    for mode in ('hard', 'soft'):
        # Check original control mask as well as its unchanged continuation copy.
        for queue in (PARENT, QUEUE):
            record_path = queue / 'cases' / mode / 'result.json'
            record = json.loads(record_path.read_text())
            hashes[path_label(record_path)] = sha(record_path)
            if record['status'] != 'completed':
                raise ValueError(f'Run not complete: {record_path}')
            directory = Path(record['run_dir'])
            control_path = directory / 'control_mask.npy'
            verify_file(control_path, record['artifact_sha256']['control_mask.npy'], hashes)
            control = np.load(control_path, allow_pickle=False)
            if control.dtype != np.bool_ or control.shape != observed.shape or np.any(control & ~observed):
                raise ValueError('Invalid internal control mask')
            controls.append(control)
            if queue == QUEUE:
                model_path = directory / 'final_model.npy'
                model_hash = record['artifact_sha256']['final_model.npy']
                verify_file(model_path, model_hash, hashes)
                model_sources[mode] = (model_path, model_hash)
    if not all(np.array_equal(controls[0], control) for control in controls[1:]):
        raise ValueError('Parent/continuation hard/soft internal control masks differ')
    actual_train = observed & ~controls[0]
    if actual_train.sum() != 38676 or controls[0].sum() != 4297:
        raise ValueError('Expected actual38676 train picks excluding 4297 internal control picks')
    train_hash = mask_sha256(actual_train)
    positions, fits = {}, {}
    for mode in MODES:
        fit_path, prediction_path = input_dir / f'{mode}_fit.json', input_dir / f'{mode}_predictions.npz'
        ledger = json.loads(fit_path.read_text())
        hashes[path_label(fit_path)] = sha(fit_path)
        verify_file(prediction_path, ledger['predictions_sha256'], hashes)
        validate_binding(ledger, manifest['output_sha256']['emtomo/arrivals.npz'], train_hash,
                         *model_sources[mode], hashes)
        with np.load(prediction_path, allow_pickle=False) as arrays:
            positions[mode] = validate_predictions(arrays, event_ids, station_ids)
        fits[mode] = ledger
    # Post-relocation only: catalog coordinates never enter model fitting or selection.
    for name in ('catalog.csv', 'stations.csv'):
        verify_file(DATA / name, manifest['output_sha256'][name], hashes)
    catalog_rows = indexed_rows(DATA / 'catalog.csv', 'event_id')
    if not set(event_ids).issubset(catalog_rows):
        raise ValueError('Retained events missing from diagnostic catalog')
    selected = [catalog_rows[event_id] for event_id in event_ids]
    geographic = np.array([[float(row[k]) for k in ('latitude', 'longitude', 'depth_km')] for row in selected])
    catalog = geographic_to_local(*geographic.T, geometry)
    if not np.isfinite(catalog).all():
        raise ValueError('Nonfinite diagnostic catalog coordinates')
    station_rows = indexed_rows(DATA / 'stations.csv', 'station_id')
    if set(station_rows) != set(station_ids):
        raise ValueError('Station table differs from prediction/source station IDs')
    station_geo = np.array([[float(station_rows[s][k]) for k in ('latitude', 'longitude', 'elevation_m')] for s in station_ids])
    projected_stations = geographic_to_local(station_geo[:, 0], station_geo[:, 1], -station_geo[:, 2] / 1000, geometry)
    np.testing.assert_allclose(stations, projected_stations, rtol=0, atol=1e-6)
    hashes[path_label(Path(__file__))] = sha(Path(__file__))
    provenance = dict(source_sha256=hashes, fits=fits, training=dict(
        original_picks=42973, internal_control_picks=4297, actual_training_picks=38676,
        train_mask_sha256=train_hash, mask_encoding='C-order uint8 0/1, original arrivals event/station order'),
        catalog_total_events=len(catalog_rows), catalog_extra_events_not_used=len(catalog_rows) - len(event_ids),
        event_ids=event_ids, station_ids=station_ids)
    return positions, catalog, geographic, stations, geometry, provenance


def event_rows(event_ids, positions, catalog, geographic):
    for index, event_id in enumerate(event_ids):
        row = dict(event_id=event_id, catalog_latitude=geographic[index, 0],
                   catalog_longitude=geographic[index, 1], catalog_depth_km=geographic[index, 2])
        for axis, value in zip('xyz', catalog[index]):
            row[f'catalog_{axis}_m'] = value
        for mode in MODES:
            delta = positions[mode][index] - catalog[index]
            for axis, value, shift in zip('xyz', positions[mode][index], delta):
                row[f'{mode}_{axis}_m'] = value
                row[f'{mode}_minus_catalog_d{axis}_m'] = shift
            row[f'{mode}_catalog_distance_3d_m'] = np.linalg.norm(delta)
            row[f'{mode}_catalog_horizontal_distance_m'] = np.linalg.norm(delta[:2])
            row[f'{mode}_deeper_than_catalog'] = bool(delta[2] > 0)
        for target, reference in [('hard', 'initial'), ('soft', 'initial'), ('soft', 'hard')]:
            delta = positions[target][index] - positions[reference][index]
            for axis, value in zip('xyz', delta):
                row[f'{target}_minus_{reference}_d{axis}_m'] = value
            row[f'{target}_{reference}_distance_3d_m'] = np.linalg.norm(delta)
        yield row


def make_figure(positions, catalog, stations, geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout='constrained')
    colors = dict(initial='#4477AA', hard='#EE6677', soft='#228833')
    labels = ['X (km; azimuth 135 deg)', 'Y (km; azimuth 225 deg)', 'Z below model top (km; positive down)']
    all_points = np.concatenate([catalog, stations, *positions.values()]) / 1000
    # Include out-of-domain catalog/fitted locations rather than clipping the diagnostic.
    lower = np.minimum(all_points.min(axis=0), np.zeros(3))
    upper = np.maximum(all_points.max(axis=0), np.asarray(geometry['domain_extent_m']) / 1000)
    padding = np.maximum((upper - lower) * .03, .1)
    for row, mode in enumerate(MODES):
        for column, (name, horizontal, vertical) in enumerate([('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]):
            ax = axes[row, column]
            start, end = catalog / 1000, positions[mode] / 1000
            delta = end - start
            ax.scatter(start[:, horizontal], start[:, vertical], s=5, color='0.55', alpha=.25, rasterized=True)
            ax.quiver(start[:, horizontal], start[:, vertical], delta[:, horizontal], delta[:, vertical],
                      angles='xy', scale_units='xy', scale=1, color=colors[mode], alpha=.13,
                      width=.0012, headwidth=3, rasterized=True)
            ax.scatter(end[:, horizontal], end[:, vertical], s=6, color=colors[mode], alpha=.42, rasterized=True)
            ax.scatter(stations[:, horizontal] / 1000, stations[:, vertical] / 1000,
                       marker='^', s=36, facecolor='white', edgecolor='black', linewidth=.7, zorder=5)
            ax.set(xlabel=labels[horizontal], ylabel=labels[vertical], title=f'{mode.capitalize()} fitted MAP | {name}',
                   xlim=(lower[horizontal] - padding[horizontal], upper[horizontal] + padding[horizontal]),
                   ylim=(lower[vertical] - padding[vertical], upper[vertical] + padding[vertical]))
            if vertical == 2:
                ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.15)
    handles = [Line2D([], [], marker='o', linestyle='', color='0.55', label='Catalog (diagnostic reference)'),
               Line2D([], [], marker='^', linestyle='', markerfacecolor='white', color='black', label='Stations')]
    handles += [Line2D([], [], marker='o', linestyle='', color=colors[mode], label=f'{mode.capitalize()} fitted MAP') for mode in MODES]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=9)
    fig.suptitle('Training-only fitted locations: arrows from catalog to fitted MAP\n'
                 'Descriptive shifts, not location errors; catalog/sea-level datum compatibility is assumed', fontsize=13)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    positions, catalog, geographic, stations, geometry, provenance = load_inputs(args.input_dir.resolve())
    extent = np.asarray(geometry['domain_extent_m'])
    report = dict(
        schema_version=1, input_dir=path_label(args.input_dir), geometry=geometry, provenance=provenance,
        truth_available=False, accuracy_evaluated=False, catalog_used_for_fitting=False, no_solver_calls=True,
        summary=summarize(positions, catalog),
        summary_policy='Equal weight per event, all 3471 retained IDs, no outlier trimming; numpy linear P90',
        shift_convention='Fitted minus catalog or named reference; positive dZ means deeper; fractions use strict sign',
        out_of_domain_counts={name: int(np.any((points < 0) | (points > extent), axis=1).sum())
                              for name, points in dict(catalog=catalog, **positions).items()},
        plot_policy='All events and stations; XY/XZ/YZ, common per-axis limits including out-of-domain points; arrows unscaled',
        caveats=[
            'Catalog positions are a diagnostic reference, not truth; distances are not location errors or accuracy.',
            'Catalog was joined only after loading/validating training-only MAP predictions; no relocation is run here.',
            'Catalog sea-level depth is converted as local Z = 1166 + 1000 * depth_km; datum compatibility is assumed.',
            'MAP locations and shifts do not describe posterior uncertainty, resolution or statistical significance.',
            'All methods must use the same 50 candidates/dmin7/noise(.01,.05,.02) and original actual38676 mask.',
            'Initial means locations fitted in the original initial velocity model, not unrelocated catalog locations.',
            'Small hard-soft separation would not establish accuracy; no geological interpretation is supplied.',
            'Bindings and hashes establish artifact consistency, not an independent audit of the fitting implementation.',
        ],
    )
    rows = list(event_rows(provenance['event_ids'], positions, catalog, geographic))
    json.dumps(report, allow_nan=False)
    figure = make_figure(positions, catalog, stations, geometry)
    # Recheck every consumed file immediately before any final output is written.
    for path, digest in provenance['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Input changed during diagnostics: {path}')
    for extension in ('png', 'pdf'):
        figure.savefig(FIGURE.with_suffix('.' + extension), dpi=180)
    import matplotlib.pyplot as plt
    plt.close(figure)
    report['figure_sha256'] = {path_label(FIGURE.with_suffix('.' + ext)): sha(FIGURE.with_suffix('.' + ext)) for ext in ('png', 'pdf')}
    with OUTPUT.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report['event_csv_sha256'] = sha(OUTPUT.with_suffix('.csv'))
    OUTPUT.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))
    print(f'Outputs: {OUTPUT}.json/.csv and {FIGURE}.png/.pdf')


if __name__ == '__main__':
    main()
