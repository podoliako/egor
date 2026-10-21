"""Plot completed matched tomoDD cases; read saved iteration 3, never rerun a solver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'paper'))
from evaluate_metrics import native_helpers, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--depth-km', type=float, default=1.435)
    args = parser.parse_args()
    manifest = json.loads((args.study_root / 'manifest.json').read_text())
    snapshot = Path(manifest['snapshot_root'])
    helper = snapshot / 'benchmark/tomodd_benchmark/compare_emtomo_flat4.py'
    assert sha256(helper) == manifest['snapshot_sha256'][str(helper.relative_to(snapshot))]
    native = native_helpers(helper)
    centers = tuple((np.arange(n) + .5) * .125 for n in (160, 56, 56))
    if not np.isfinite(args.depth_km) or not 0 <= args.depth_km <= 7:
        raise ValueError('Depth must be inside [0, 7] km')
    index = int(np.argmin(abs(centers[2] - args.depth_km)))
    fields, reference, metrics, hashes = {}, None, {}, {}
    for geometry in ('synthetic', 'geysers'):
        run = args.study_root / 'outputs' / f'{geometry}-tomodd-sigma1'
        info = json.loads((run / 'worker_info.json').read_text())
        assert info['state'] == 'completed'
        config = json.loads((run / 'config.json').read_text())
        axes = tuple(np.asarray(config['tomodd_grid'][key]) for key in ('depths_km', 'y_km', 'x_km'))
        history = run / 'work/inversion/tomoDD.vel'
        nodes = native.read_velocity_history(history, (9, 9, 22), 3)
        with np.load(run / 'state.npz', allow_pickle=False) as state:
            truth = state['physical_velocity_km_s'].transpose(2, 1, 0) * 1000
            np.testing.assert_allclose(nodes, state['final_vp'] * 1000, rtol=0, atol=1e-8)
            initial = native.evaluate_native(state['initial_vp'] * 1000, axes, centers)
            oracle = native.evaluate_native(state['oracle_vp'] * 1000, axes, centers)
        if reference is not None:
            for first, second in zip(reference, (truth, initial, oracle)):
                np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)
        reference = truth, initial, oracle
        field = native.evaluate_native(nodes, axes, centers)
        results = json.loads((run / 'results.json').read_text())
        volume_rmse = float(np.sqrt(np.mean((field - truth)**2)))
        np.testing.assert_allclose(volume_rmse, results['final_field_rmse_km_s'] * 1000, rtol=0, atol=1e-6)
        fields[geometry] = field[index]
        error = field[index] - truth[index]
        metrics[geometry] = dict(slice_rmse_m_s=float(np.sqrt(np.mean(error**2))),
                                 slice_mae_m_s=float(np.mean(abs(error))), volume_rmse_m_s=volume_rmse,
                                 slice_vp_range_m_s=[float(field[index].min()), float(field[index].max())])
        for name in ('state.npz', 'config.json', 'results.json', 'worker_info.json', 'work/inversion/tomoDD.vel'):
            hashes[str(run / name)] = sha256(run / name)
    fields.update(truth=truth[index], initial=initial[index], oracle=oracle[index])
    # Union of the two existing paper figures' limits, shared by every new panel.
    limits = (2839.211944580078, 6516.418171324893)
    args.output_dir.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(args.study_root / 'cache' / 'slice-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles = {'truth': 'Physical truth', 'initial': 'Initial: native trilinear',
              'oracle': 'Oracle: native trilinear',
              'synthetic': 'TomoDD: artificial geometry', 'geysers': 'TomoDD: Geysers geometry'}
    for stem, order in (
        ('tomodd_comparison', ('truth', 'oracle', 'synthetic', 'geysers')),
        ('tomodd_synthetic', ('truth', 'synthetic', 'initial', 'oracle')),
        ('tomodd_geysers', ('truth', 'geysers', 'initial', 'oracle')),
    ):
        fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)
        for ax, key in zip(axs.flat, order):
            image = ax.imshow(fields[key], extent=(0, 20, 0, 7), origin='lower',
                              interpolation='nearest', aspect='equal', cmap='Reds',
                              vmin=limits[0], vmax=limits[1])
            title = titles[key]
            if key in metrics:
                title += f"; slice RMSE {metrics[key]['slice_rmse_m_s']:.1f} m/s"
            ax.set(title=title, xlabel='x (km)', ylabel='y (km)')
        fig.colorbar(image, ax=list(axs.flat), shrink=.8, label='Vp (m/s): white slow → red fast',
                     extend='both')
        fig.suptitle(f'Matched tomoDD: iteration 3; initial location sigma = 1 km per coordinate\n'
                     f'Requested z={args.depth_km:g} km; sampled z={centers[2][index]:.4f} km')
        for suffix in ('png', 'pdf'):
            fig.savefig(args.output_dir / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    report = dict(requested_depth_km=args.depth_km, sampled_depth_km=float(centers[2][index]),
                  depth_index=index, iteration=3, initial_hypocenter_sigma_km_per_coordinate=1,
                  seed=42, color_limits_m_s=limits,
                  color_scale_note='Common union of the two original paper panels; values outside limits saturate.',
                  metrics=metrics, source_sha256=hashes,
                  helper_sha256=sha256(helper), script_sha256=sha256(Path(__file__)))
    for key in metrics:
        metrics[key]['saturated_pixels'] = int(((fields[key] < limits[0]) | (fields[key] > limits[1])).sum())
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
