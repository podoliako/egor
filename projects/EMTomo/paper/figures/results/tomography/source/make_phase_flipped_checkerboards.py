"""Display-only XY sign reversal in 4.4, all horizontal cuts Z=1.5 km.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_phase_flipped_checkerboards.py
Right crops and XZ/YZ cuts remain unchanged. Never modifies inversion models.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from make_matched_checkerboards import build_blocks
from make_right_cropped_checkerboards import SPECS as RIGHT_SPECS
from make_checkerboard_pairs import draw_results, draw_truth, save
from make_comparison import HERE, ROOT, PNG, sha

RESULTS = "checkerboards_z15_phase_flipped_results"
TRUTH = "checkerboards_z15_phase_flipped_true"
BASELINE_KM_S = 5.
SPECS = [(*spec[:-1], [spec[-1][0], spec[-1][1], 1500.]) for spec in RIGHT_SPECS]


def prepare():
    blocks, metrics_path = build_blocks(SPECS, require_equal_volumes=False, require_equal_slices=False)
    raw = {f"s{b['section'].replace('.', '_')}_{key}": value.copy()
           for b in blocks for key, value in b["slices"].items()}
    for b in blocks:
        if b["section"] == "4.4":
            for mode in ("true", "hard", "soft"):
                key = mode + "_xy"
                b["slices"][key] = 2 * BASELINE_KM_S - b["slices"][key]
    for block in blocks[1:]:
        for plane in ("xy", "xz", "yz"):
            np.testing.assert_array_equal(block["slices"]["true_"+plane], blocks[0]["slices"]["true_"+plane])
    return blocks, raw, metrics_path


def main():
    blocks, raw, metrics_path = prepare()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    fig = draw_results(blocks, stem=RESULTS)
    fig._suptitle.set_text("Checkerboard display variant · XY at Z = 1.5 km · 4.4 XY anomaly sign reversed")
    for ax in fig.axes[12:14]:
        ax.texts[0].set_text("Z = 1.5 km · ΔVp sign reversed")
    fig.axes[-1].set_ylabel("Display Vp (km/s)")
    fig.text(.5, -.025,
             "4.4 XY only (Hard / Soft / True): display Vp = 10 − original Vp [km/s], i.e. ΔVp → −ΔVp around 5 km/s. "
             "Not an unchanged physical velocity section; XZ/YZ are unmodified.",
             ha="center", fontsize=9, color="0.25")
    save(fig, RESULTS)
    plt.close(fig)
    fig = draw_truth(blocks, stem=TRUTH, matching_volumes=False)
    fig.texts[1].set_text("Matched displayed True slices · All XY cuts Z = 1.5 km · 4.4 XY phase reversed")
    fig.axes[-1].set_ylabel("Display Vp (km/s)")
    fig.text(.5, -.045,
             "This shared XY pattern equals raw True in 4.2/4.3 and sign-reversed True in 4.4. "
             "XZ/YZ remain actual unchanged sections.",
             ha="center", fontsize=9, color="0.25")
    save(fig, TRUTH)
    plt.close(fig)
    arrays = {"raw_" + key: value for key, value in raw.items()}
    arrays.update({f"display_s{b['section'].replace('.', '_')}_{key}": value
                   for b in blocks for key, value in b["slices"].items()})
    data_path = HERE / f"{RESULTS}.npz"
    np.savez_compressed(data_path, **arrays)
    report = dict(
        variant="Explicit display-only phase reversal, not original velocity values in 4.4 XY",
        horizontal_z_m=1500., anomaly_baseline_km_s=BASELINE_KM_S,
        transform="4.4 XY: display_Vp_km_s = 10 - raw_Vp_km_s, for True/Hard/Soft equally",
        transform_scope="4.4 XY only; all XZ/YZ and all 4.2/4.3 values unchanged",
        displayed_true_slices_equal=True, raw_true_xy_slices_equal=False,
        source_models_modified=False, previous_figures_unchanged=True,
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
    print("PASS: XY Z=1.5 km in all series; 4.4 XY sign-flipped equally for True/Hard/Soft; exact displayed True equality")
    print(PNG/f"{RESULTS}.png")
    print(PNG/f"{TRUTH}.png")


if __name__ == "__main__":
    main()
