"""Descriptive real-pick Vp differences from Guo–Thurber GJI 2021, not accuracy.

From projects/EMTomo (no solvers, no figure/manuscript edits):
    python -B paper/paper_scripts/make_geysers_real_structure_metrics.py

125/250 m midpoint quadrature uses one published-coverage/resolvability mask
for initial/hard/soft at each spacing. These spacings are not resolution claims.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from make_geysers_real_published_comparison import (
    ROOT, INITIAL_DIR, completed_model, sample_cells, sha,
)
from make_geysers_published_vp_sections import GEOMETRY, SOURCE, LayeredModel, validate

QUEUE = ROOT / 'runs/geysers_real_refine500_fmm125_20261110_110138'
PARENT = ROOT / 'runs/geysers_real_linear35_60_latest_20261110_063859'
FIGURE = ROOT / 'paper/figures/real_picks/geysers_real_refine500_fmm125_with_emtomo_oracle.json'
OUTPUT = ROOT / 'paper/exp_data/geysers_real_structure_comparison'
THRESHOLD = 0.4
METRICS = ('rms_difference_m_s', 'mean_signed_difference_m_s',
           'mae_m_s', 'pearson_vp')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def verify_hashes(hashes):
    verified = {}
    for name, expected in hashes.items():
        path = ROOT / name
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
        verified[relative(path)] = actual
    return verified


def midpoint_grid(extent_m, spacing_m):
    extent = np.asarray(extent_m, dtype=float)
    if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError('Expected three finite positive domain extents')
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('Spacing must be finite and positive')
    shape = np.rint(extent / spacing_m).astype(int)
    if not np.allclose(shape * spacing_m, extent, rtol=0, atol=1e-9):
        raise ValueError('Spacing must exactly tile the domain')
    axes = [(np.arange(n) + .5) * spacing_m / 1000 for n in shape]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return points, shape


def published_mask(values):
    return np.isfinite(values).all(axis=-1) & (values[..., 1] > THRESHOLD)


def descriptive_metrics(model_m_s, published_m_s):
    model = np.asarray(model_m_s, dtype=float)
    reference = np.asarray(published_m_s, dtype=float)
    if (model.shape != reference.shape or model.ndim != 1 or not model.size
            or not np.isfinite(model).all() or not np.isfinite(reference).all()):
        raise ValueError('Metrics require nonempty matching finite vectors on the shared mask')
    difference = model - reference
    centred_model, centred_reference = model - model.mean(), reference - reference.mean()
    denominator = np.linalg.norm(centred_model) * np.linalg.norm(centred_reference)
    pearson = float(np.clip(np.dot(centred_model, centred_reference) / denominator, -1, 1)) if denominator else None
    return dict(rms_difference_m_s=float(np.sqrt(np.mean(difference ** 2))),
                mean_signed_difference_m_s=float(np.mean(difference)),
                mae_m_s=float(np.mean(np.abs(difference))), pearson_vp=pearson)


def compare_grid(models, published, extent_m, spacing_m, batch_size=65536):
    """Models map names to (XYZ m/s array, cell size km); no per-model masking."""
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    for name, (array, cell_km) in models.items():
        if array.ndim != 3 or not np.isfinite(array).all() or np.any(array <= 0):
            raise ValueError(f'Invalid model: {name}; refusing to change the shared mask')
        np.testing.assert_allclose(np.asarray(array.shape) * cell_km * 1000,
                                   extent_m, rtol=0, atol=1e-9)
    points, shape = midpoint_grid(extent_m, spacing_m)
    values = np.empty((len(points), 2))
    for start in range(0, len(points), batch_size):
        values[start:start + batch_size] = published.sample(points[start:start + batch_size])
    finite_vp = np.isfinite(values[:, 0])
    finite_both = np.isfinite(values).all(axis=1)
    mask = published_mask(values)
    count = int(mask.sum())
    if not count:
        raise ValueError('Empty shared published coverage/resolvability mask')
    sample_volume = float(spacing_m ** 3)
    reference = values[mask, 0] * 1000
    metrics = {name: descriptive_metrics(sample_cells(array, points[mask], cell_km) * 1000, reference)
               for name, (array, cell_km) in models.items()}
    return dict(
        spacing_m=spacing_m, shape_xyz=shape.tolist(), total_samples=len(points),
        finite_published_vp_samples=int(finite_vp.sum()),
        outside_finite_published_vp_samples=int((~finite_vp).sum()),
        finite_vp_nonfinite_resolvability_samples=int((finite_vp & ~finite_both).sum()),
        finite_published_resolvability_le_threshold_samples=int((finite_both & ~mask).sum()),
        shared_mask_samples=count, excluded_samples=int((~mask).sum()),
        sample_volume_m3=sample_volume, domain_volume_km3=float(np.prod(extent_m) / 1e9),
        finite_published_vp_volume_km3=float(finite_vp.sum() * sample_volume / 1e9),
        shared_mask_volume_m3=count * sample_volume,
        shared_mask_volume_km3=count * sample_volume / 1e9,
        shared_mask_fraction_of_domain=count / len(points),
        shared_mask_fraction_of_finite_published_vp=count / int(finite_vp.sum()),
        mask_sha256=hashlib.sha256(np.packbits(mask, bitorder='little').tobytes()).hexdigest(),
        metrics=metrics,
    )


def load_inputs():
    figure = json.loads(FIGURE.read_text())
    # Fail closed on changed figure sources, including loaders and all final artifacts.
    hashes = verify_hashes(figure['source_sha256'])
    for mode in ('hard', 'soft'):
        hashes.update(verify_hashes(figure[mode]['source_sha256']))
    hashes[relative(FIGURE)] = sha(FIGURE)
    geometry = json.loads(GEOMETRY.read_text())
    np.testing.assert_array_equal(geometry['domain_origin_m'], [0, 0, 0])
    if geometry != figure['geometry'] or Path(figure['queue']).resolve() != QUEUE:
        raise ValueError('Figure geometry or queue mismatch')
    info_path = INITIAL_DIR / 'model.json'
    info = json.loads(info_path.read_text())
    initial_path = INITIAL_DIR / info['array_file']
    hashes.update(verify_hashes({str(initial_path): info['array_sha256']}))
    initial = np.load(initial_path, allow_pickle=False)
    np.testing.assert_array_equal(initial.shape, info['shape'])
    np.testing.assert_array_equal(info['domain_extent_m'], geometry['domain_extent_m'])
    np.testing.assert_array_equal(info['domain_origin_m'], geometry['domain_origin_m'])
    if info['array_units'] != 'm/s' or info['axis_order'] != ['x', 'y', 'z']:
        raise ValueError('Unexpected initial units or axes')
    expected = 3500 + 2500 * (np.arange(initial.shape[2]) + .5) / initial.shape[2]
    np.testing.assert_allclose(initial, np.broadcast_to(expected, initial.shape), rtol=0, atol=1e-10)
    models = {'initial': (initial, info['cell_size_m'] / 1000)}
    provenance = {'initial': dict(array=relative(initial_path), metadata=relative(info_path),
                                 sha256=sha(initial_path), cell_size_m=info['cell_size_m'],
                                 shape_xyz=list(initial.shape), role='Original baseline, not continuation start')}
    manifest_path = QUEUE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest['parent_queue']).resolve() != PARENT:
        raise ValueError('Unexpected parent queue')
    for mode in ('hard', 'soft'):
        case = next(c for c in manifest['cases'] if c['id'] == mode)
        parent, parent_provenance, parent_cell_km = completed_model(PARENT, mode, initial_path, geometry)
        parent_result = PARENT / 'cases' / mode / 'result.json'
        parent_record = json.loads(parent_result.read_text())
        if Path(parent_record['run_dir']) != Path(case['parent_run']):
            raise ValueError(f'{mode}: wrong parent run')
        if case['parent_stopping'] != parent_provenance['stopping']:
            raise ValueError(f'{mode}: wrong parent stopping metadata')
        copied_parent = QUEUE / case['parent_model_file']
        hashes.update(verify_hashes({str(copied_parent): parent_record['artifact_sha256']['final_model.npy']}))
        start_path = QUEUE / case['initial_model_file']
        start = np.load(start_path, allow_pickle=False)
        np.testing.assert_array_equal(start, parent.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        final, final_provenance, cell_km = completed_model(QUEUE, mode, start_path, geometry)
        np.testing.assert_allclose(parent_cell_km, 2 * cell_km, rtol=0, atol=1e-12)
        record_path = QUEUE / 'cases' / mode / 'result.json'
        record = json.loads(record_path.read_text())
        final_path = Path(record['run_dir']) / 'final_model.npy'
        if 'final_model.npy' not in record['artifact_sha256']:
            raise ValueError(f'{mode}: missing final-model hash')
        if final_provenance['source_sha256'] != figure[mode]['source_sha256']:
            raise ValueError(f'{mode}: artifacts differ from figure provenance')
        if final_provenance['stopping'] != figure[mode]['stopping']:
            raise ValueError(f'{mode}: stopping differs from figure provenance')
        hashes.update(verify_hashes(parent_provenance['source_sha256']))
        hashes[relative(start_path)] = sha(start_path)
        models[mode] = (final, cell_km)
        provenance[mode] = dict(
            array=relative(final_path), sha256=sha(final_path), result=relative(record_path),
            cell_size_m=cell_km * 1000, shape_xyz=list(final.shape),
            completed_updates=final_provenance['completed_updates'], stopping=final_provenance['stopping'],
            parent_result=relative(parent_result), parent_run=relative(case['parent_run']),
            parent_completed_updates=parent_provenance['completed_updates'],
            continuation_start=relative(start_path),
            continuation_policy='Own verified parent final split into 2x2x2 identical cells',
        )
    table = np.genfromtxt(SOURCE, delimiter=',', names=True)
    published = LayeredModel(table, geometry)
    validate(published, table, geometry)
    hashes[relative(Path(__file__))] = sha(Path(__file__))
    return models, published, geometry, provenance, dict(sorted(hashes.items()))


def convergence(fine, coarse):
    changes = {}
    for mode in fine['metrics']:
        changes[mode] = {}
        for key in METRICS:
            a, b = fine['metrics'][mode][key], coarse['metrics'][mode][key]
            changes[mode][key] = None if a is None or b is None else a - b
    volume_change = fine['shared_mask_volume_km3'] - coarse['shared_mask_volume_km3']
    return dict(fine_spacing_m=fine['spacing_m'], coarse_spacing_m=coarse['spacing_m'],
                difference_convention='125 m minus 250 m; independently sampled same mask rule',
                shared_mask_volume_change_km3=volume_change,
                shared_mask_volume_relative_change=volume_change / coarse['shared_mask_volume_km3'],
                metric_changes=changes,
                caveat='Two-grid sensitivity check only; no formal convergence order or uncertainty bound')


def main():
    models, published, geometry, provenance, hashes = load_inputs()
    grids = [compare_grid(models, published, geometry['domain_extent_m'], spacing) for spacing in (125., 250.)]
    report = dict(
        schema_version=1, comparison='Descriptive Vp differences from Guo–Thurber GJI 2021',
        source_article_doi='10.1093/gji/ggab017', truth_available=False, accuracy_evaluated=False,
        no_solver_calls=True, queue=relative(QUEUE), parent_queue=relative(PARENT),
        figure_provenance=relative(FIGURE), geometry=geometry, models=provenance,
        difference_convention='EMTomo model Vp minus published Vp, m/s',
        metric_definitions=dict(rms_difference_m_s='sqrt(mean((model - published)^2))',
                                mean_signed_difference_m_s='mean(model - published)',
                                mae_m_s='mean(abs(model - published)); descriptive, not truth error',
                                pearson_vp='Pearson correlation of absolute Vp; null if either vector is constant'),
        sampling=dict(primary_spacing_m=125, convergence_spacing_m=250,
                      grid='Uniform XYZ cell midpoints over the full zero-origin real domain; C-order, Z fastest',
                      weighting='Equal volume per midpoint; volume = selected count * spacing^3',
                      emtomo='Containing piecewise-constant cells via sample_cells; no smoothing',
                      published='LayeredModel: Delaunay-linear XY and linear depth, Vp and scaled resolvability',
                      outside_policy='No extrapolation, background substitution or oracle projection',
                      mask='finite(published Vp) AND finite(interpolated scaled resolvability) AND resolvability > 0.4',
                      resolvability_threshold=THRESHOLD, shared_mask_for_all_models=True,
                      nonfinite_emtomo_policy='Fail, never silently change mask per model',
                      mask_hash_encoding='SHA256 of numpy.packbits(flat mask, bitorder=little); grid shape/order above'),
        grids=grids, convergence=convergence(*grids), source_sha256=hashes,
        checks_passed=['All figure source hashes and completed final artifacts verified',
                       'Parent final hashes, original parent starts and exact 2x2x2 continuation starts verified',
                       'Completion markers, stopping metadata, units, model shapes and domain checked',
                       'Published source nodes, vertical interpolation, outside NaN and coordinate mapping validated'],
        caveats=[
            'Published Vp is an external structural reference, not truth; these are not error/accuracy metrics.',
            'Published scaled resolvability is not EMTomo resolvability; no EMTomo resolution claim is made.',
            '125/250 m are quadrature spacings, not model or tomographic resolution.',
            'Volume is a midpoint approximation to the thresholded finite support, not exact boundary integration.',
            'Pearson uses absolute Vp and can reflect the common depth trend; no anomaly-specific inference or p-value.',
            'Original initial is 1 km piecewise constant; 500 m continuations start from their own parent finals.',
            'Published geographic datum is unspecified; sea-level compatibility is assumed, not independently verified.',
            'No geological interpretation, truth validation, holdout evaluation or seismic solver is performed.',
        ],
        manuscript_todo=dict(color='red', text='TODO: Describe coincident anomalies; no interpretation supplied here.',
                             latex=r'\textcolor{red}{TODO: Describe coincident anomalies.}',
                             manuscript_modified=False),
    )
    json_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    rows = []
    for grid in grids:
        for mode, metrics in grid['metrics'].items():
            rows.append(dict(
                spacing_m=grid['spacing_m'], model=mode, total_samples=grid['total_samples'],
                finite_published_vp_samples=grid['finite_published_vp_samples'],
                shared_mask_samples=grid['shared_mask_samples'],
                shared_mask_volume_km3=grid['shared_mask_volume_km3'], mask_sha256=grid['mask_sha256'],
                difference_convention='model_minus_published', accuracy_evaluated=False,
                **metrics,
            ))
    OUTPUT.with_suffix('.json').write_text(json_text)
    with OUTPUT.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'grids': grids, 'convergence': report['convergence']}, indent=2, allow_nan=False))
    print(OUTPUT.with_suffix('.json'))
    print(OUTPUT.with_suffix('.csv'))


if __name__ == '__main__':
    main()
