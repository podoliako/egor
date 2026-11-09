"""Plot saved faulted-layer truth and its EMTomo 1-km slowness-average oracle.

python -B paper/make_faulted_layers_oracle.py
Only reads experiment inputs; never runs a forward or inverse solver.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))
from projects.forward_modeling.experiments import load_inputs
from velocity_model import block_average_slowness

EXPERIMENT_ID = 'geysers11_20110401-20110420_fault3_x6p5_elev1166_26st_3776ev_clean'
EXPERIMENTS = ROOT.parent / 'forward_modeling/experiments'
OUTPUT = ROOT / 'paper/figures'
STEM = 'faulted_layers_x6p5_3layers_oracle_1km'


def main():
    model, _, _ = load_inputs(EXPERIMENTS, EXPERIMENT_ID)
    input_dir = EXPERIMENTS / 'input' / EXPERIMENT_ID
    generation = json.loads((input_dir / 'generation.json').read_text())
    truth = model.velocity
    assert truth.shape == (160, 56, 56) and model.cell_size_m == 125
    oracle = block_average_slowness(truth, model.cell_size_m, (20, 7, 7), 1000.)
    blocks = (1 / truth).reshape(20, 8, 7, 8, 7, 8)
    np.testing.assert_allclose(1 / oracle, blocks.mean(axis=(1, 3, 5)), rtol=1e-14)
    np.testing.assert_allclose(oracle[:, 0, :], oracle[:, -1, :], rtol=1e-14)
    # The fault bisects x=[6,7] km: all deep cells there mix both basal speeds.
    np.testing.assert_allclose(oracle[6, :, 4:], 2 / (1 / 5000 + 1 / 5500), rtol=1e-14)
    expanded = oracle.repeat(8, axis=0).repeat(8, axis=1).repeat(8, axis=2)
    velocity_rmse = float(np.sqrt(np.mean((expanded - truth) ** 2)))
    section_y_km = 3.5
    fields = (truth[:, int(section_y_km * 1000 / model.cell_size_m), :],
              oracle[:, int(section_y_km), :])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True, sharey=True,
                             layout='constrained')
    norm = Normalize(float(truth.min()) / 1000, float(truth.max()) / 1000)
    for ax, field, cell_km, title in zip(
            axes, fields, (.125, 1.),
            ('True model: saved 125 m voxels',
             'EMTomo oracle: 1 km cells, volume-averaged slowness')):
        image = ax.pcolormesh(np.arange(field.shape[0] + 1) * cell_km,
                              np.arange(field.shape[1] + 1) * cell_km,
                              field.T / 1000, cmap='viridis', norm=norm,
                              shading='flat', rasterized=True)
        ax.set(xlim=(0, 20), ylim=(7, 0), xlabel='x (km)', ylabel='Depth z (km)', title=title)
        ax.set_aspect('equal')
        ax.set_yticks(np.arange(8))
        ax.tick_params(labelbottom=True)
    fault = generation['parameters']['fault_x_km']
    axes[0].axvline(fault, color='black', linewidth=1.2)
    for side, xmin, xmax in (('left', 0, fault), ('right', fault, 20)):
        axes[0].hlines(generation['effective_boundaries_km'][side], xmin, xmax,
                       color='white', linewidth=1)
    axes[1].vlines(np.arange(1, 20), 0, 7, color='white', alpha=.3, linewidth=.5)
    axes[1].hlines(np.arange(1, 7), 0, 20, color='white', alpha=.3, linewidth=.5)
    fig.colorbar(image, ax=axes, label='Vp (km/s)', shrink=.9, pad=.02)
    fig.suptitle(f'Vertical section at y = {section_y_km:g} km | No inversion', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}.{extension}', dpi=180)
    plt.close(fig)
    metadata = {
        'experiment_id': EXPERIMENT_ID, 'section_y_km': section_y_km,
        'source_model_sha256': hashlib.sha256((input_dir / 'model.npz').read_bytes()).hexdigest(),
        'oracle': 'inverse of volume-averaged slowness; same function as EMTomo reference_model',
        'oracle_shape_xyz': list(oracle.shape), 'oracle_cell_size_m': 1000.,
        'velocity_volume_rmse_m_s_against_125m_truth': velocity_rmse,
        'not_a_velocity_L2_optimum': True, 'no_forward_or_inverse_solve': True,
    }
    (OUTPUT / f'{STEM}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(OUTPUT / f'{STEM}.png')


if __name__ == '__main__':
    main()
