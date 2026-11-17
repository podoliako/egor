"""Plot saved ball-holdout evaluation artifacts only; never run a solver.

--evaluation is the directory containing summary.json, observations.npz and the
three *_predictions.npz files. --output contains png/ and source/ (PDF).
Relative summary['queue'] paths are resolved against the evaluation directory;
relative result['run_dir'] paths are resolved against manifest['project_root'].
Existing figure/sidecar paths are never overwritten. The external pick holdout
is not accessed. Optional velocity panels need only the queue manifest, case
result JSONs and saved velocity arrays, not any queue arrival archives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = 'geysers_real_ball10_it15'
MODELS = ('initial', 'soft', 'hard')
LABELS = {'initial': 'Initial', 'soft': 'Soft', 'hard': 'Hard'}
COLORS = {'initial': '#666666', 'soft': '#0072B2', 'hard': '#D55E00'}
METRICS = ('validation_rmse_s', 'validation_mae_s', 'catalog_distance_median_m',
           'catalog_distance_rmse_m', 'fit_rmse_s')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def record(self, path):
        path = path.resolve(strict=True)
        self.hashes[str(path)] = sha256(path)
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def npz(self, path):
        with np.load(self.record(path), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def npy(self, path):
        return np.load(self.record(path), allow_pickle=False)


def load_evaluation(directory, inputs):
    summary = inputs.json(directory / 'summary.json')
    if 'protocol' not in summary:
        raise ValueError('summary.json must describe the evaluation protocol')
    for name in MODELS:
        for metric in METRICS:
            value = summary['models'][name][metric]
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f'Invalid summary metric: {name}.{metric}')
    obs = inputs.npz(directory / 'observations.npz')
    observed = np.asarray(obs['observed_s'], dtype=float)
    mask = obs['validation_mask']
    if observed.ndim != 2 or 0 in observed.shape:
        raise ValueError('observed_s must be a nonempty event-by-station matrix')
    n, s = observed.shape
    if mask.dtype != np.bool_ or mask.shape != observed.shape or not mask.any():
        raise ValueError('validation_mask must be a nonempty boolean mask matching observed_s')
    if not np.isfinite(observed[mask]).all():
        raise ValueError('Every validation pick must have a finite observed time')
    for key, shape in (('event_ids', (n,)), ('station_ids', (s,)),
                       ('catalog_xyz_m', (n, 3)), ('centre_m', (3,))):
        if obs[key].shape != shape:
            raise ValueError(f'{key}: expected shape {shape}, got {obs[key].shape}')
    for key in ('event_ids', 'station_ids'):
        if len(np.unique(obs[key])) != len(obs[key]):
            raise ValueError(f'{key} must be unique')
    radius = np.asarray(obs['radius_m'])
    if radius.ndim != 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius_m must be a finite positive scalar')
    if not all(np.isfinite(obs[key]).all() for key in ('catalog_xyz_m', 'centre_m')):
        raise ValueError('Catalog coordinates and ball centre must be finite')
    predictions, residuals, event_rmse, distances = {}, {}, {}, {}
    counts = mask.sum(axis=1)
    valid_events = counts > 0
    for name in MODELS:
        pred = inputs.npz(directory / f'{name}_predictions.npz')
        if pred['position_m'].shape != (n, 3) or not np.isfinite(pred['position_m']).all():
            raise ValueError(f'{name}: position_m must be finite with shape {(n, 3)}')
        relative = pred['predicted_relative_s']
        if relative.shape != observed.shape or not np.isfinite(relative[mask]).all():
            raise ValueError(f'{name}: invalid validation predictions')
        predictions[name] = pred
        residual = np.zeros_like(observed)
        residual[mask] = relative[mask] - observed[mask]
        residuals[name] = residual[mask]
        event_rmse[name] = np.sqrt((residual**2).sum(axis=1)[valid_events] / counts[valid_events])
        distances[name] = np.linalg.norm(pred['position_m'] - obs['catalog_xyz_m'], axis=1)
    return summary, obs, predictions, residuals, event_rmse, distances


def dashboard(summary, obs, residuals, event_rmse, distances):
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    bins = np.histogram_bin_edges(np.concatenate(list(residuals.values())), bins=45)
    for name in MODELS:
        ax.hist(residuals[name], bins=bins, density=True, histtype='step', linewidth=1.8,
                color=COLORS[name], label=LABELS[name])
    ax.axvline(0, color='black', linewidth=.7, linestyle=':')
    ax.set(title='Validation residual distributions', xlabel='Prediction − observation (s)',
           ylabel='Probability density')
    ax.legend()
    upper = max(float(v.max()) for v in event_rmse.values()) * 1.05
    upper = max(upper, .001)
    for col, name in enumerate(('soft', 'hard'), start=1):
        ax = fig.add_subplot(grid[0, col])
        ax.scatter(event_rmse['initial'], event_rmse[name], s=17, alpha=.6,
                   color=COLORS[name], edgecolors='none', rasterized=True)
        ax.plot([0, upper], [0, upper], '--', color='.4', linewidth=1)
        improved = np.mean(event_rmse[name] < event_rmse['initial'])
        ax.set(xlim=(0, upper), ylim=(0, upper), xlabel='Initial validation RMSE (s)',
               ylabel=f'{LABELS[name]} validation RMSE (s)',
               title=f'Paired events: initial vs {name}\n{improved:.1%} below identity (lower is better)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
    ax = fig.add_subplot(grid[1, 0])
    for name in MODELS:
        x = np.sort(distances[name]) / 1000
        ax.step(np.r_[0, x], np.r_[0, np.arange(1, len(x) + 1) / len(x)],
                where='post', color=COLORS[name], label=LABELS[name], linewidth=1.7)
    ax.set(title='Distance to catalog reference (not truth)', xlabel='3-D catalog distance (km)',
           ylabel='Fraction of events', ylim=(0, 1.02), xlim=(0, None))
    ax.grid(alpha=.2)
    ax.legend()
    ax = fig.add_subplot(grid[1, 1:])
    ax.axis('off')
    rows = ['Validation RMSE (s)', 'Validation MAE (s)', 'Catalog distance median (m)',
            'Catalog distance RMS (m)', 'Non-validation fit RMSE (s)']
    cells = [[f"{float(summary['models'][name][metric]):.4g}" for name in MODELS]
             for metric in METRICS]
    table = ax.table(cellText=cells, rowLabels=rows, colLabels=[LABELS[n] for n in MODELS],
                     cellLoc='center', bbox=(.39, .26, .60, .65))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.text(0, .10, 'Metrics above are supplied by summary.json.\n'
            'Catalog locations are a reference, not ground truth; distance is not location accuracy.',
            transform=ax.transAxes, fontsize=10, va='top')
    count = int((obs['validation_mask'].sum(axis=1) > 0).sum())
    fig.suptitle(f'Geysers ball holdout | saved evaluation | {len(obs["event_ids"])} events\n'
                 f'Locations fitted using non-validation picks only; '
                 f'{obs["validation_mask"].sum()} validation picks in {count} events', fontsize=15)
    return fig


def locations(obs, predictions):
    catalog = obs['catalog_xyz_m'] / 1000
    centre, radius = obs['centre_m'] / 1000, float(obs['radius_m']) / 1000
    cloud = np.concatenate([catalog] + [predictions[n]['position_m'] / 1000 for n in MODELS]
                           + [(centre - radius)[None, :], (centre + radius)[None, :]])
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    pad = np.maximum((hi - lo) * .06, .1)
    limits = list(zip(lo - pad, hi + pad))
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout='constrained')
    for row, name in enumerate(MODELS):
        position = predictions[name]['position_m'] / 1000
        for col, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row, col]
            ax.scatter(catalog[:, i], catalog[:, j], s=20, facecolors='none', edgecolors='.35',
                       linewidths=.7, label='Catalog reference (not truth)', rasterized=True)
            ax.scatter(position[:, i], position[:, j], s=12, alpha=.65, color=COLORS[name],
                       edgecolors='none', label=f'{LABELS[name]} fitted location', rasterized=True)
            ax.add_patch(Circle(centre[[i, j]], radius, fill=False, linestyle='--',
                                color='black', linewidth=1.2, label='Holdout sphere projection'))
            ax.set(xlim=limits[i], ylim=limits[j], xlabel=f'{"XYZ"[i]} (km)',
                   ylabel=f'{"XYZ"[j]} (km)', title=f'{LABELS[name]} | {"XYZ"[i]}{"XYZ"[j]}')
            if j == 2:
                ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.2)
        axes[row, 0].legend(fontsize=8, loc='best')
    fig.suptitle('Geysers ball holdout | catalog reference vs fitted locations\n'
                 'Fitted using non-validation picks only; shared coordinate limits; '
                 'Z positive down from model top', fontsize=14)
    return fig


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_velocity(summary, evaluation, inputs):
    if not summary.get('queue'):
        return None, 'summary.json has no queue path'
    queue = resolve_path(summary['queue'], evaluation)
    manifest_path = queue if queue.suffix == '.json' else queue / 'manifest.json'
    queue = manifest_path.parent
    try:
        manifest = inputs.json(manifest_path)
        cases = {case['id']: case for case in manifest['cases']}
        for name in ('soft', 'hard'):
            if float(cases[name]['config']['cell_size']) != 1000:
                raise ValueError('Velocity sections require the ball-run 1000 m grid')
        fields = {'initial': inputs.npy(queue / 'inputs/initial_velocity_m_s.npy')}
        for name in ('soft', 'hard'):
            result = inputs.json(queue / 'cases' / name / 'result.json')
            if result.get('status') != 'completed':
                return None, f'{name} queue case is not completed'
            project = resolve_path(manifest.get('project_root', str(ROOT)), queue)
            run = resolve_path(result['run_dir'], project)
            fields[name] = inputs.npy(run / 'final_model.npy')
    except FileNotFoundError as error:
        return None, f'optional queue artifact missing: {error.filename}'
    shape = fields['initial'].shape
    if len(shape) != 3 or min(shape) == 0 or shape[1] <= 3:
        raise ValueError('Velocity volume must include Y=3 km on the 1 km cell grid')
    for name, field in fields.items():
        if field.shape != shape or not np.isfinite(field).all() or np.any(field <= 0):
            raise ValueError(f'Invalid saved velocity volume: {name}')
    return fields, None


def velocity_sections(fields, obs):
    # At an internal cell boundary, use the cell on its positive-Y side: [3, 4) km.
    y = 3.0
    slices = {name: field[:, 3, :] / 1000 for name, field in fields.items()}
    vmin = min(float(field.min()) for field in slices.values())
    vmax = max(float(field.max()) for field in slices.values())
    if vmin == vmax:
        vmin, vmax = vmin - .01, vmax + .01
    norm = Normalize(vmin, vmax)
    nx, _, nz = fields['initial'].shape
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True,
                             layout='constrained')
    centre, radius = obs['centre_m'] / 1000, float(obs['radius_m']) / 1000
    catalog = obs['catalog_xyz_m'] / 1000
    intersection_sq = radius**2 - (y - centre[1])**2
    for ax, name in zip(axes, MODELS):
        image = ax.pcolormesh(np.arange(nx + 1), np.arange(nz + 1), slices[name].T,
                              shading='flat', cmap='viridis', norm=norm, rasterized=True)
        ax.scatter(catalog[:, 0], catalog[:, 2], s=16, facecolors='none', edgecolors='white',
                   linewidths=.7, label='Held-out catalog (all Y projected; not truth)',
                   rasterized=True)
        if intersection_sq >= 0:
            ax.add_patch(Circle(centre[[0, 2]], np.sqrt(intersection_sq), fill=False,
                                color='black', linestyle='--', linewidth=1.5,
                                label='Sphere intersection at Y=3 km'))
        ax.set(title=LABELS[name], xlabel='X (km)', ylabel='Z below model top (km)',
               xlim=(0, nx), ylim=(nz, 0), aspect='equal')
    axes[0].legend(loc='upper right', fontsize=9, facecolor='.3', labelcolor='white')
    fig.colorbar(image, ax=list(axes), label='Vp (km/s) — shared scale', shrink=.85)
    fig.suptitle('Geysers ball holdout | XZ velocity sections at Y=3 km\n'
                 'Saved 1 km cells, no smoothing; boundary uses Y cell [3, 4) km', fontsize=14)
    return fig, {'section_y_km': y, 'sampled_y_cell_km': [3, 4],
                 'velocity_limits_km_s': [vmin, vmax],
                 'catalog_overlay': 'All held-out catalog events projected into XZ, not a Y slab',
                 'sphere_intersects_section': bool(intersection_sq >= 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.evaluation.expanduser().resolve(strict=True)
    if not evaluation.is_dir():
        parser.error('--evaluation must be a directory')
    output = args.output.expanduser().resolve()
    for directory in (output, output / 'png', output / 'source'):
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(directory)
    inputs = Inputs()
    inputs.record(Path(__file__))
    summary, obs, predictions, residuals, event_rmse, distances = load_evaluation(evaluation, inputs)
    evaluation_hashes = dict(inputs.hashes)
    fields, skipped = load_velocity(summary, evaluation, inputs)
    if skipped:
        warnings.warn(f'Velocity sections skipped: {skipped}', stacklevel=1)
    names = ['evaluation', 'locations'] + (['velocity_sections'] if fields is not None else [])
    targets = {name: [output / 'png' / f'{STEM}_{name}.png',
                      output / 'source' / f'{STEM}_{name}.pdf',
                      output / f'{STEM}_{name}.json'] for name in names}
    for paths in targets.values():
        for path in paths:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
    for directory in (output, output / 'png', output / 'source'):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    common = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluation=str(evaluation),
                  protocol=summary['protocol'], catalog_is_ground_truth=False,
                  location_fit_policy='Non-validation picks only (supplied evaluator protocol)',
                  residual_definition='predicted_relative_s - observed_s on validation_mask; no recentering',
                  solver_calls=0, external_holdout_opened=False,
                  n_events=len(obs['event_ids']), n_validation_picks=int(obs['validation_mask'].sum()),
                  n_events_without_validation=int((obs['validation_mask'].sum(axis=1) == 0).sum()),
                  velocity_sections_skipped=skipped)
    for name in names:
        extra = {}
        if name == 'evaluation':
            fig = dashboard(summary, obs, residuals, event_rmse, distances)
        elif name == 'locations':
            fig = locations(obs, predictions)
        else:
            fig, extra = velocity_sections(fields, obs)
        png, pdf, sidecar = targets[name]
        try:
            fig.savefig(png, dpi=200, bbox_inches='tight')
            fig.savefig(pdf, bbox_inches='tight')
        finally:
            plt.close(fig)
        report = dict(common, figure=name, outputs=[str(png), str(pdf)],
                      source_sha256=inputs.hashes if name == 'velocity_sections' else evaluation_hashes,
                      **extra)
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for path in targets[name]:
            print(path)


if __name__ == '__main__':
    main()
