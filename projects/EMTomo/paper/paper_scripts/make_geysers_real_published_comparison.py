"""Compare initial Vp, real-pick runs and GJI 2021 without running solvers.

Run with --queue PATH for completed hard/soft runs from the linear start.
Without --queue, reproduce the older comparison to the NCSN-GEY-start runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.geysers_real_picks.evaluate_holdout import velocity_model, H, SOURCE as RUN_SOURCE
from make_geysers_published_vp_sections import (
    GEOMETRY, SOURCE, OUTPUT, LayeredModel, section_points, validate,
)

INITIAL_DIR = ROOT / 'data/geysers_2011/initial_models/linear_vp_3p5_6p0'
STEM = 'geysers_real_initial_hard_soft_gji2021_middle_sections'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_cells(model, points, cell_km):
    """Sample actual piecewise-constant XYZ cells, without visual smoothing."""
    indices = np.floor(points / cell_km).astype(int)
    if np.any(indices < 0) or np.any(indices >= np.asarray(model.shape)):
        raise ValueError('Query outside cell model')
    return model[tuple(np.moveaxis(indices, -1, 0))] / 1000


def volume_slowness_oracle(model, extent, cell_km=1., samples_per_axis=16):
    """Project the published field onto constant cells, without outside fill.

    This is a representation reference, not an inversion or a Vp-L2 optimum.
    For LayeredModel, convex horizontal support and common depth bounds make
    all eight corners a sufficient full-cell coverage test.
    """
    shape = np.rint(np.asarray(extent) / cell_km).astype(int)
    np.testing.assert_allclose(shape * cell_km, extent, rtol=0, atol=1e-12)
    origins = np.moveaxis(np.indices(shape), 0, -1).reshape(-1, 3) * cell_km
    corners = np.asarray([(x, y, z) for x in (0., cell_km)
                          for y in (0., cell_km) for z in (0., cell_km)])
    covered = np.isfinite(model.sample(origins[:, None] + corners)[..., 0]).all(axis=1)
    offsets = (np.arange(samples_per_axis) + .5) * cell_km / samples_per_axis
    offsets = np.stack(np.meshgrid(offsets, offsets, offsets, indexing='ij'), axis=-1).reshape(-1, 3)
    result = np.full(len(origins), np.nan)
    indices = np.flatnonzero(covered)
    for start in range(0, len(indices), 32):
        batch = indices[start:start + 32]
        values = model.sample(origins[batch, None] + offsets)[..., 0]
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Invalid samples inside fully covered oracle cell')
        result[batch] = 1000 / np.mean(1 / values, axis=1)
    return result.reshape(tuple(shape))


def completed_model(queue, mode, initial_path, geometry):
    """Load only complete, hashed final results with the same initial model."""
    record = queue / 'cases' / mode / 'result.json'
    result = json.loads(record.read_text())
    if result['status'] != 'completed':
        raise ValueError(f'{mode} is not completed')
    directory = Path(result['run_dir'])
    paths = [record, directory / 'meta.json']
    for name, digest in result['artifact_sha256'].items():
        path = directory / name
        if sha(path) != digest:
            raise ValueError(f'Changed completed artifact: {path}')
        paths.append(path)
    if sha(directory / 'initial_model.npy') != sha(initial_path):
        raise ValueError(f'{mode} initial model differs from column 1')
    meta = json.loads((directory / 'meta.json').read_text())
    if meta['source_experiment']['kind'] != 'real_picks':
        raise ValueError('Expected a real-pick run')
    config = meta['run_params']
    stopping = json.loads((directory / 'stopping.json').read_text())
    updates = stopping['completed_updates']
    markers = sorted(int(p.parent.name.split('_')[-1]) for p in directory.glob('iter_*/complete.json'))
    if markers != list(range(updates)):
        raise ValueError(f'{mode} completion markers disagree with stopping')
    paths.extend(directory / f'iter_{i}/complete.json' for i in markers)
    model = np.load(directory / 'final_model.npy', allow_pickle=False)
    if model.ndim != 3 or not np.isfinite(model).all() or np.any(model <= 0):
        raise ValueError(f'Invalid {mode} final velocity')
    np.testing.assert_allclose(np.asarray(model.shape) * config['cell_size'], geometry['domain_extent_m'])
    return model, dict(note=f'{mode}: final after {updates} updates; {stopping["reason"]}',
                       completed_updates=updates, stopping=stopping,
                       source_sha256={str(p): sha(p) for p in paths}), config['cell_size'] / 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, help='Completed real-pick queue, including refined-grid continuations')
    parser.add_argument('--stem', help='Output basename (PNG, PDF and JSON)')
    parser.add_argument('--with-oracle', action='store_true',
                        help='Insert a volume-mean-slowness projection of GJI 2021 on the inversion cell grid' )
    args = parser.parse_args(argv)
    queue = args.queue.resolve() if args.queue else None
    manifest = json.loads((queue / 'manifest.json').read_text()) if queue else None
    refinement = bool(manifest and manifest.get('parent_queue'))
    stem = args.stem or (STEM + '_refine500_latest' if refinement else
                         STEM + '_linear35_60_latest' if queue else STEM)
    geometry = json.loads(GEOMETRY.read_text())
    extent = np.asarray(geometry['domain_extent_m'], dtype=float) / 1000
    np.testing.assert_array_equal(geometry['domain_origin_m'], [0, 0, 0])
    initial_info = json.loads((INITIAL_DIR / 'model.json').read_text())
    initial_path = INITIAL_DIR / initial_info['array_file']
    assert sha(initial_path) == initial_info['array_sha256']
    initial = np.load(initial_path, allow_pickle=False)
    np.testing.assert_array_equal(initial.shape, initial_info['shape'])
    expected = 3500 + 2500 * (np.arange(initial.shape[2]) + .5) / initial.shape[2]
    np.testing.assert_allclose(initial, np.broadcast_to(expected, initial.shape), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(initial.shape) * initial_info['cell_size_m'] / 1000, extent)
    extra_sources = []
    if queue:
        starts = {}
        for mode in ('hard', 'soft'):
            case = next(c for c in manifest['cases'] if c['id'] == mode)
            starts[mode] = queue / case.get('initial_model_file', 'inputs/initial_velocity_m_s.npy')
            if refinement:
                parent_path = queue / case['parent_model_file']
                parent = np.load(parent_path, allow_pickle=False)
                expanded = parent.repeat(2, 0).repeat(2, 1).repeat(2, 2)
                np.testing.assert_array_equal(np.load(starts[mode], allow_pickle=False), expanded)
                extra_sources += [parent_path, starts[mode]]
        hard, hard_provenance, hard_cell_km = completed_model(queue, 'hard', starts['hard'], geometry)
        soft, soft_provenance, soft_cell_km = completed_model(queue, 'soft', starts['soft'], geometry)
        actual_initial_path = starts['hard']
        if not refinement:
            assert sha(actual_initial_path) == sha(initial_path)
        else:
            for mode, provenance in [('hard', hard_provenance), ('soft', soft_provenance)]:
                case = next(c for c in manifest['cases'] if c['id'] == mode)
                provenance.update(parent_run=case['parent_run'], parent_stopping=case['parent_stopping'],
                                  initial_policy='Own parent final model split into 2x2x2 identical cells; no smoothing')
    else:
        hard, hard_provenance = velocity_model('hard')
        soft, soft_provenance = velocity_model('soft')
        hard_cell_km = soft_cell_km = H / 1000
        for array in (hard, soft):
            np.testing.assert_allclose(np.asarray(array.shape) * H / 1000, extent)
        actual_initial_path = RUN_SOURCE / 'inputs/initial_velocity_m_s.npy'
        assert sha(actual_initial_path) != sha(initial_path), 'Recheck the start mismatch note'
    table = np.genfromtxt(SOURCE, delimiter=',', names=True)
    published = LayeredModel(table, geometry)
    validate(published, table, geometry)
    oracle, oracle_info = None, None
    np.testing.assert_allclose(hard_cell_km, soft_cell_km)
    oracle_cell_km = hard_cell_km if queue else 1.
    samples_per_axis = max(2, round(oracle_cell_km / .0625))
    if args.with_oracle:
        oracle = volume_slowness_oracle(published, extent, oracle_cell_km, samples_per_axis)
        coarse_oracle = volume_slowness_oracle(published, extent, oracle_cell_km, samples_per_axis // 2)
        np.testing.assert_array_equal(np.isfinite(oracle), np.isfinite(coarse_oracle))
        finite_cells = np.isfinite(oracle)
        difference = np.abs(oracle[finite_cells] - coarse_oracle[finite_cells])
        oracle_info = dict(
            policy='Inverse volume-averaged slowness of published Vp, approximated by midpoint quadrature',
            interpretation='Representation reference relative to GJI 2021, not true Vp, not an inversion or a Vp-L2 optimum',
            cell_size_m=oracle_cell_km * 1000, shape_xyz=list(oracle.shape),
            quadrature_spacing_m=oracle_cell_km * 1000 / samples_per_axis,
            samples_per_axis=samples_per_axis, convergence_comparison_samples_per_axis=samples_per_axis // 2,
            convergence_max_change_m_s=float(difference.max()),
            convergence_rms_change_m_s=float(np.sqrt(np.mean(difference ** 2))),
            fully_covered_cells=int(finite_cells.sum()), blank_cells=int((~finite_cells).sum()),
            outside_policy='Whole cell blank if any corner lies outside published horizontal/depth support; no extrapolation',
        )
        print('Oracle:', json.dumps(oracle_info))

    specs = [('XZ', 0, 2, 1), ('YZ', 1, 2, 0), ('XY', 0, 1, 2)]
    sections = []
    for spec in specs:
        edges, centres, points = section_points(extent, *spec[1:])
        article = published.sample(points)
        arrays = [sample_cells(initial, points, initial_info['cell_size_m'] / 1000),
                  sample_cells(hard, points, hard_cell_km),
                  sample_cells(soft, points, soft_cell_km), article[..., 0]]
        if oracle is not None:
            arrays.insert(-1, sample_cells(oracle, points, oracle_cell_km))
        sections.append((spec, edges, centres, arrays, article[..., 1]))
    finite = np.concatenate([a[np.isfinite(a)] for _, _, _, arrays, _ in sections for a in arrays])
    limits = [float(np.floor(finite.min() * 5) / 5), float(np.ceil(finite.max() * 5) / 5)]
    norm = Normalize(*limits)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    plt.rcParams.update({'font.size': 10, 'hatch.linewidth': .35, 'hatch.color': '0.4'})
    columns = 5 if args.with_oracle else 4
    fig, axes = plt.subplots(3, columns, figsize=(5 * columns, 12),
                             gridspec_kw={'height_ratios': [7/23, 7/11, 11/23]})
    fig.subplots_adjust(left=.055, right=.93, bottom=.15, top=.84, hspace=.38, wspace=.28)
    if queue:
        titles = ['Initial (used by both runs)\nLinear 3.5–6.0 km/s',
                  f'EMTomo hard\nFinal; {hard_provenance["completed_updates"]} updates, auto-stop',
                  f'EMTomo soft\nFinal; {soft_provenance["completed_updates"]} updates, auto-stop',
                  'Guo & Thurber (GJI 2021)\nPublished Vp, not ground truth']
    else:
        titles = ['Initial (proposed)\nLinear 3.5–6.0 km/s',
                  'EMTomo hard (old GEY start)\n21 updates; stopped early',
                  'EMTomo soft (old GEY start)\n24 updates; final',
                  'Guo & Thurber (GJI 2021)\nPublished Vp, not ground truth']
    if refinement:
        titles[0] = 'Original initial (first stage)\nLinear 3.5–6.0 km/s'
        for col, mode, provenance in [(1, 'hard', hard_provenance), (2, 'soft', soft_provenance)]:
            before = provenance['parent_stopping']['completed_updates']
            after = provenance['completed_updates']
            titles[col] = (f'EMTomo {mode} | {hard_cell_km*1000:g} m cells\n'
                           f'Final; {after} new updates ({before} + {after} total)')
    model_names = ['original_initial' if refinement else 'initial' if queue else 'proposed_initial',
                   'hard', 'soft', 'published_gji2021']
    if args.with_oracle:
        titles.insert(-1, f'EMTomo oracle w.r.t. GJI 2021\n{oracle_cell_km*1000:g} m cells; mean slowness')
        model_names.insert(-1, 'emtomo_representation_oracle_gji2021')
    labels = ['X (km; azimuth 135°)', 'Y (km; azimuth 225°)', 'Z below box top (km)']
    records = []
    for row, (spec, edges, centres, arrays, resolution) in enumerate(sections):
        name, horizontal, vertical, fixed = spec
        records.append({'plane': name, 'fixed_axis': 'XYZ'[fixed],
                        'fixed_coordinate_km': float(extent[fixed] / 2), 'models': []})
        for col, vp in enumerate(arrays):
            ax = axes[row, col]
            covered = np.isfinite(vp)
            image = ax.pcolormesh(*edges, np.ma.masked_invalid(vp), cmap=cmap, norm=norm,
                                 shading='flat', rasterized=True)
            if col == columns - 1:
                low = np.ma.masked_where(~covered, (resolution <= .4).astype(float))
                if np.any(low == 1):
                    ax.contourf(*centres, low, levels=[.5, 1.5], colors='none', hatches=['////'])
            if vertical == 2:
                ax.axhline(geometry['top_elevation_m'] / 1000, color='0.25',
                           linestyle='--', linewidth=.65)
            ax.set(xlim=(0, extent[horizontal]), ylim=(0, extent[vertical]),
                   xlabel=labels[horizontal], ylabel=labels[vertical])
            if vertical == 2:
                ax.invert_yaxis()
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(titles[col], fontsize=12, pad=12)
            ax.text(.015, .97, f'{name} | {"XYZ"[fixed]} = {extent[fixed]/2:g} km',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox={'facecolor': 'white', 'alpha': .82, 'edgecolor': 'none', 'pad': 2})
            records[-1]['models'].append({
                'model': model_names[col],
                'covered_samples': int(covered.sum()), 'blank_samples': int((~covered).sum()),
                'vp_range_km_s': [float(vp[covered].min()), float(vp[covered].max())],
            })
    cax = fig.add_axes([.946, .235, .012, .54])
    fig.colorbar(image, cax=cax, label='Vp (km/s) — common scale')
    fig.suptitle('Geysers real-pick domain: three middle orthogonal sections', fontsize=18, y=.975)
    start_note = ('Real P picks | 500 m continuation from each method’s own previous final model; '
                  'column 1 is the original first-stage baseline.\n' if refinement else
                  'Real P picks, 1–20 April 2011 | Both inversions used the initial model in column 1.\n')
    subtitle = (
        start_note + 'Final internal-control RMSE: hard '
        f'{hard_provenance["stopping"]["control_rmse_s"]*1000:.2f} ms; soft '
        f'{soft_provenance["stopping"]["control_rmse_s"]*1000:.2f} ms. External holdout not evaluated.'
        if queue else
        'Initial is the NEW proposed model; hard/soft are EXISTING 31 Oct 2026 runs from NCSN GEY.\n'
        'These are NOT inversion results from the new 3.5–6.0 km/s start.')
    fig.text(.5, .925, subtitle, ha='center', va='top', fontsize=12, color='0.25')
    oracle_note = ('\nOracle: published model projected as inverse volume-mean slowness; '
                   'not an inversion or ground truth. Partially covered cells remain entirely blank.'
                   if args.with_oracle else '')
    cells_note = (f'Hard/soft: actual {hard_cell_km*1000:g} m cells; original initial: 1 km cells. '
                  'Calculation grid remains 125 m; no smoothing. ' if refinement else
                  'Initial/hard/soft: actual 1 km cell values, no smoothing. ')
    fig.text(.055, .082,
             'Domain: 23 × 11 × 7 km. Z = sea-level depth + 1.166 km; dashed line: sea level. '
             'Local planar geographic mapping; exact vertical-datum compatibility is unverified.\n'
             + cells_note + 'Published: linear interpolation within source coverage only; white = no coverage.\n'
             'Hatching: published scaled resolvability ≤ 0.4. '
             '100 m plotting samples do not imply 100 m resolution. Article DOI: 10.1093/gji/ggab017.' + oracle_note,
             ha='left', va='top', fontsize=10, linespacing=1.6)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{stem}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    if oracle is not None:
        oracle_path = OUTPUT / f'{stem}_emtomo_oracle.npz'
        np.savez_compressed(oracle_path, velocity_m_s=oracle, cell_size_m=oracle_cell_km*1000,
                            origin_m=np.zeros(3), fully_covered=np.isfinite(oracle))
        oracle_info['saved_array'] = str(oracle_path.relative_to(ROOT))
        oracle_info['saved_array_sha256'] = sha(oracle_path)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
                     (GEOMETRY, SOURCE, initial_path, INITIAL_DIR / 'model.json', actual_initial_path,
                      Path(__file__), Path(__file__).with_name('make_geysers_published_vp_sections.py'),
                      ROOT / 'studies/geysers_real_picks/evaluate_holdout.py')}
    for path in extra_sources:
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    if queue:
        source_hashes[str((queue / 'manifest.json').relative_to(ROOT))] = sha(queue / 'manifest.json')
    report = {
        'truth_available': False, 'no_solver_calls': True,
        'initial_policy': 'Linear 3.5–6.0 km/s across box boundaries, sampled at cell centres',
        'start_mismatch': ('Column 1 is the original baseline; each 500 m continuation starts from its own parent final model'
                           if refinement else None if queue else
                           'Existing hard/soft used NCSN GEY, not the proposed initial shown in column 1'),
        'continuation_from_per_mode_parent': refinement,
        'queue': str(queue) if queue else None,
        'external_holdout_evaluated': False,
        'hard': hard_provenance, 'soft': soft_provenance,
        'columns': model_names, 'emtomo_oracle': oracle_info,
        'source_article_doi': '10.1093/gji/ggab017',
        'domain_extent_km': extent.tolist(), 'geometry': geometry,
        'color_limits_km_s': limits, 'sections': records,
        'published_outside_policy': 'NaN / white; no extrapolation or background substitution',
        'published_resolution_policy': 'Interpolate scaled resolvability; hatch <= 0.4',
        'cell_sampling': 'Nearest containing cell; no smoothing',
        'checks_passed': ['initial hash and linear profile', 'domain extents match all arrays',
                          'final artifacts and checkpoints verified; per-mode parent starts reproduced exactly' if refinement else
                          'completed final artifacts hashed, checkpoint counts and common start verified' if queue
                          else 'hard last valid checkpoint verified by existing evaluator',
                          'all published source nodes reproduced', 'published outside support is NaN',
                          'published vertical interpolation and geographic reference mapping'],
        'source_sha256': source_hashes,
    }
    (OUTPUT / f'{stem}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'color_limits_km_s': limits, 'sections': records}, indent=2))
    print(OUTPUT / f'{stem}.png')


if __name__ == '__main__':
    main()
