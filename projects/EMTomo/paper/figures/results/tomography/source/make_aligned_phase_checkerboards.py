"""Aligned cuts X=Y=4.5 km, Z=1.5 km; display-only sign flip in all 4.4 views.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_aligned_phase_checkerboards.py
Right-edge crops and original inversion checkpoints remain unchanged.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from make_matched_checkerboards import build_blocks
from make_right_cropped_checkerboards import SPECS as RIGHT_SPECS
from make_checkerboard_pairs import draw_results, draw_truth, save
from make_comparison import HERE, ROOT, PNG, sha

RESULTS = "checkerboards_aligned45_phase_flipped_results"
TRUTH = "checkerboards_aligned45_phase_flipped_true"
BASELINE_KM_S = 5.
SPECS = [(*spec[:-1], [4500., 4500., 1500.]) for spec in RIGHT_SPECS]


def prepare():
    blocks, metrics_path = build_blocks(SPECS, require_equal_volumes=False, require_equal_slices=False)
    raw = {f"s{b['section'].replace('.', '_')}_{key}": value.copy()
           for b in blocks for key, value in b["slices"].items()}
    for block in blocks:
        if block["section"] == "4.4":
            block["slices"] = {key: 2 * BASELINE_KM_S - values for key, values in block["slices"].items()}
    for block in blocks[1:]:
        for plane in ("xy", "xz", "yz"):
            np.testing.assert_array_equal(block["slices"]["true_"+plane], blocks[0]["slices"]["true_"+plane])
    return blocks, raw, metrics_path


def main():
    blocks, raw, metrics_path = prepare()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    fig = draw_results(blocks, stem=RESULTS)
    fig._suptitle.set_text("Aligned checkerboards · XY: Z = 1.5 km · XZ: Y = 4.5 km · YZ: X = 4.5 km")
    for ax in fig.axes[12:18]:
        ax.texts[0].set_text(ax.texts[0].get_text() + " · −ΔVp")
    fig.axes[-1].set_ylabel("Display Vp (km/s)")
    fig.text(.5, -.025,
             "4.4 only, ALL views and Hard / Soft / True: ΔVp → −ΔVp around 5 km/s (display Vp = 10 − raw Vp). "
             "Explicit phase convention, not original physical velocities. 4.2/4.3 signs unchanged.",
             ha="center", fontsize=9, color="0.25")
    save(fig, RESULTS)
    plt.close(fig)
    fig = draw_truth(blocks, stem=TRUTH, matching_volumes=False)
    fig.texts[1].set_text("Identical displayed True · XY: Z = 1.5 km · XZ: Y = 4.5 km · YZ: X = 4.5 km")
    fig.texts[2].set_text("Same native cuts in every series. Displayed True in 4.4 has its anomaly sign reversed about 5 km/s.\n"
                         "Right-edge crops preserve X = 0. No reflection, coordinate shift or velocity smoothing.")
    fig.axes[-1].set_ylabel("Display Vp (km/s)")
    save(fig, TRUTH)
    plt.close(fig)
    arrays = {"raw_" + key: value for key, value in raw.items()}
    arrays.update({f"display_s{b['section'].replace('.', '_')}_{key}": value
                   for b in blocks for key, value in b["slices"].items()})
    data_path = HERE / f"{RESULTS}.npz"
    np.savez_compressed(data_path, **arrays)
    report = dict(
        variant="Display-only phase reversal in 4.4, all three planes, equal native cuts across series",
        native_cuts_xyz_m=[4500., 4500., 1500.], anomaly_baseline_km_s=BASELINE_KM_S,
        transform="4.4 all XY/XZ/YZ True/Hard/Soft: display_Vp_km_s = 10 - raw_Vp_km_s",
        transform_scope="All 4.4 plotted slices; 4.2/4.3 signs unchanged",
        displayed_true_slices_equal=True, raw_true_slices_equal=False,
        source_models_modified=False, previous_figures_unchanged=True,
        right_edge_crops_unchanged=True, no_reflections=True, no_velocity_smoothing=True,
        rows=["4.2", "4.3", "4.4"],
        columns=["XY Hard", "XY Soft", "XZ Hard", "XZ Soft", "YZ Hard", "YZ Soft"],
        blocks=[dict(section=b["section"], crop=b["crop"], records=b["records"],
                     source_directory=b["source_directory"], input_sha256=b["input_sha256"]) for b in blocks],
        data_sha256=sha(data_path),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (Path(__file__).resolve(), HERE/"make_checkerboard_pairs.py",
                        HERE/"make_matched_checkerboards.py", HERE/"make_right_cropped_checkerboards.py",
                        HERE/"make_comparison.py", metrics_path)},
    )
    (HERE/f"{RESULTS}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PASS: equal X/Y=4.5 km, Z=1.5 km; only 4.4 phase reversal; exact displayed True equality")
    print(PNG/f"{RESULTS}.png")
    print(PNG/f"{TRUTH}.png")


if __name__ == "__main__":
    main()
