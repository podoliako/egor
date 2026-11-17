"""Right-edge crops, preserving native X=0, with exactly matched True slices.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_right_cropped_checkerboards.py
Three-dimensional True volumes need not match; all three displayed planes must.
"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from make_matched_checkerboards import build_blocks, EXTENT_M
from make_checkerboard_pairs import draw_results, draw_truth, LIMITS
from make_comparison import HERE, ROOT, PNG, sha

RESULTS = "checkerboards_right_crop_results_pairs"
TRUTH = "checkerboards_right_crop_true_only"
SPECS = [
    ("4.2", "Checkerboard: one-sided stations", "onesided", "section2_pair01",
     [0., 0., 0.], [12000., 6000., 4500.]),
    ("4.3", "Checkerboard: two station clusters", "two_clusters", "section3_pair01",
     [0., 0., 0.], [6500., 4500., 4500.]),
    ("4.4", "Checkerboard: Geysers geometry", "real_geometry_noise", "section4_pair07",
     [0., 0., 0.], [9000., 3500., 1600.]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-42-z16", action="store_true",
                        help="Keep 4.2 Z=1.6 km; use Z=1.6 in 4.3 and Z=4.5 in 4.4")
    args = parser.parse_args()
    specs = SPECS
    results_stem, truth_stem = RESULTS, TRUTH
    if args.keep_42_z16:
        depths = {"4.2": 1600., "4.3": 1600., "4.4": 4500.}
        specs = [(*spec[:-1], [spec[-1][0], spec[-1][1], depths[spec[0]]]) for spec in SPECS]
        results_stem += "_z16_42"
        truth_stem += "_z16_42"
    blocks, metrics_path = build_blocks(specs, require_equal_volumes=False)
    for b in blocks:
        np.testing.assert_array_equal(b["crop"]["lower_native_xyz_m"], [0, 0, 0])
    # No color/phase inversion: opposite 3-D phases genuinely produce equal slices at these cuts.
    assert not np.array_equal(blocks[0]["fields"]["true"][0], blocks[2]["fields"]["true"][0])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    plt.close(draw_results(blocks, stem=results_stem))
    plt.close(draw_truth(blocks, stem=truth_stem, matching_volumes=False))
    arrays = {f"s{b['section'].replace('.', '_')}_{key}": value for b in blocks for key, value in b["slices"].items()}
    for b in blocks:
        arrays[f"s{b['section'].replace('.', '_')}_true_volume_m_s"] = b["fields"]["true"][0]
    data_path = HERE / f"{results_stem}.npz"
    np.savez_compressed(data_path, **arrays)
    report = dict(
        crop_policy="All native lower bounds remain zero; remove only X=20–24 km in 4.2/4.3 and Y,Z=8–12 km in 4.2",
        displayed_extent_m=EXTENT_M.tolist(), original_X_zero_preserved=True,
        exact_true_slice_equality=True, exact_true_volume_equality=False,
        cut_changes_from_full_domain_figure=(
                    "4.2 unchanged; 4.3: Y 3.5→4.5 km, Z 4.5→1.6 km; 4.4: Z 1.6→4.5 km"
                    if args.keep_42_z16 else
                    "4.2: Z 1.6→4.5 km; 4.3: Y 3.5→4.5 km; all other native cuts unchanged"),
        inversion_checkpoints_unchanged=True, previous_figures_unchanged=True,
        no_reflections=True, no_color_reversal=True, no_velocity_modification=True,
        rows=["4.2", "4.3", "4.4"],
        columns=["XY Hard", "XY Soft", "XZ Hard", "XZ Soft", "YZ Hard", "YZ Soft"],
        color_limits_km_s=list(LIMITS), outputs=[results_stem, truth_stem],
        caveats=["Three matching 2-D planes do not imply equal 3-D True volumes",
                 "Different section depths sample different parts of the inversion fields",
                 "Full acquisition geometry and resolution remain different",
                 "Axes are native coordinates (no +1 km fault-frame shift for 4.4)"],
        blocks=[dict(section=b["section"], crop=b["crop"], records=b["records"],
                     source_directory=b["source_directory"], input_sha256=b["input_sha256"]) for b in blocks],
        data_sha256=sha(data_path),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (Path(__file__).resolve(), HERE/"make_checkerboard_pairs.py",
                        HERE/"make_matched_checkerboards.py", HERE/"make_comparison.py", metrics_path)},
    )
    (HERE / f"{results_stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PASS: native X=0 preserved; exact XY/XZ/YZ True equality, no requirement of equal 3-D phase")
    print(PNG / f"{truth_stem}.png")
    print(PNG / f"{results_stem}.png")


if __name__ == "__main__":
    main()
