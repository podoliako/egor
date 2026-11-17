"""Final aligned checkerboards: no overall title or bottom explanatory notes.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_checkerboards_final.py
Retains explicit 4.4 phase-reversal labels and unchanged source data.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from make_aligned_phase_checkerboards import prepare
from make_checkerboard_pairs import draw_results, draw_truth, save
from make_comparison import HERE, ROOT, PNG, sha

RESULTS = "checkerboards_4_2-4_4_results_final"
TRUTH = "checkerboards_4_2-4_4_true_final"


def build_figures(blocks):
    results = draw_results(blocks, stem=RESULTS)
    headings = {"Horizontal XY", "Longitudinal XZ", "Transverse YZ"}
    for text in list(results.texts):
        if text.get_text() not in headings and not text.get_text().startswith("§"):
            text.remove()
    for ax in results.axes[12:18]:
        ax.texts[0].set_text(ax.texts[0].get_text() + " · −ΔVp")
    results.axes[-1].set_ylabel("Display Vp (km/s)")
    truth = draw_truth(blocks, stem=TRUTH, matching_volumes=False)
    for text in list(truth.texts):
        text.remove()
    truth.axes[-1].set_ylabel("Display Vp (km/s)")
    return results, truth


def main():
    blocks, raw, metrics_path = prepare()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    results, truth = build_figures(blocks)
    save(results, RESULTS)
    save(truth, TRUTH)
    plt.close(results)
    plt.close(truth)
    arrays = {"raw_" + key: value for key, value in raw.items()}
    arrays.update({f"display_s{b['section'].replace('.', '_')}_{key}": value
                   for b in blocks for key, value in b["slices"].items()})
    data_path = HERE / f"{RESULTS}.npz"
    np.savez_compressed(data_path, **arrays)
    report = dict(
        outputs=[RESULTS, TRUTH], overall_titles=False, bottom_notes=False,
        layout="Rows 4.2/4.3/4.4; paired Hard/Soft columns for XY/XZ/YZ; shared True separate",
        native_cuts_xyz_m=[4500., 4500., 1500.],
        transform="All 4.4 True/Hard/Soft views: display Vp = 10 - original Vp in km/s (delta sign reversal about 5 km/s)",
        transform_labels="−ΔVp retained on each 4.4 result panel; Display Vp on colorbars",
        exact_display_true_equality=True, original_models_modified=False, previous_figures_unchanged=True,
        blocks=[dict(section=b["section"], crop=b["crop"], records=b["records"],
                     source_directory=b["source_directory"], input_sha256=b["input_sha256"]) for b in blocks],
        data_sha256=sha(data_path),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (Path(__file__).resolve(), HERE/"make_aligned_phase_checkerboards.py",
                        HERE/"make_checkerboard_pairs.py", HERE/"make_matched_checkerboards.py",
                        HERE/"make_right_cropped_checkerboards.py", HERE/"make_comparison.py", metrics_path)},
    )
    (HERE/f"{RESULTS}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(PNG/f"{RESULTS}.png")
    print(PNG/f"{TRUTH}.png")


if __name__ == "__main__":
    main()
