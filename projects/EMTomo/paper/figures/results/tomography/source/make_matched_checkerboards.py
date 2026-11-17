"""Separate cropped 4.2–4.4 comparison with exactly identical True volumes/views.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_matched_checkerboards.py
The original comparison and its section artifacts are never overwritten.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from make_comparison import HERE, ROOT, PNG, load_block, sample_section, draw_block, read, sha

STEM = "checkerboards_matched_true"
EXTENT_M = np.array([20000., 8000., 8000.])
SPECS = [
    ("4.2", "Checkerboard: one-sided stations", "onesided", "section2_pair01",
     [4000., 0., 0.], [12000., 3500., 1600.]),
    ("4.3", "Checkerboard: two station clusters", "two_clusters", "section3_pair01",
     [4000., 0., 0.], [6500., 3500., 3500.]),
    ("4.4", "Checkerboard: Geysers geometry", "real_geometry_noise", "section4_pair07",
     [0., 0., 0.], [9000., 3500., 1600.]),
]


def crop_field(field, cell_m, lower_m):
    begin = np.rint(lower_m / cell_m).astype(int)
    shape = np.rint(EXTENT_M / cell_m).astype(int)
    np.testing.assert_allclose(begin * cell_m, lower_m, rtol=0, atol=1e-9)
    np.testing.assert_allclose(shape * cell_m, EXTENT_M, rtol=0, atol=1e-9)
    if np.any(begin < 0) or np.any(begin + shape > np.array(field.shape)):
        raise ValueError("Crop outside native field")
    return field[tuple(slice(i, i+n) for i, n in zip(begin, shape))].copy()


def build_blocks(specs=SPECS, require_equal_volumes=True, require_equal_slices=True):
    metrics_path = ROOT / "paper/figures/article_comparison_metrics.json"
    metrics = read(metrics_path)
    blocks = []
    for section, title, table, pair, lower, cuts in specs:
        lower, cuts = np.asarray(lower), np.asarray(cuts)
        records = next(r for r in metrics["tables"][table] if r["pair_id"] == pair)
        # Validate the original complete inputs/checkpoints and their full-volume RMSE first.
        block = load_block(section, title, pair, records["hard"]["experiment_id"], records,
                           cuts[1], cuts[0], [0, 0, 0], [4.6, 5.4])
        original_extent = block["extent_m"].copy()
        block["fields"] = {mode: (crop_field(field, cell, lower), cell)
                           for mode, (field, cell) in block["fields"].items()}
        block["extent_m"] = EXTENT_M.copy()
        local_cuts = cuts - lower
        block.update(section_x_native_m=float(local_cuts[0]),
                     section_y_native_m=float(local_cuts[1]),
                     section_z_native_m=float(local_cuts[2]),
                     section_x_common_m=float(local_cuts[0]),
                     section_y_common_m=float(local_cuts[1]),
                     crop=dict(original_extent_m=original_extent.tolist(),
                               lower_native_xyz_m=lower.tolist(), upper_native_xyz_m=(lower+EXTENT_M).tolist(),
                               section_native_xyz_m=cuts.tolist(), section_crop_local_xyz_m=local_cuts.tolist(),
                               coordinate_policy="Crop-local coordinates; not the fault-test geographic frame"))
        arrays = {}
        for mode, (field, cell) in block["fields"].items():
            for plane, axis in [("xy", 2), ("yz", 0), ("xz", 1)]:
                arrays[mode+"_"+plane] = sample_section(field, cell, EXTENT_M, axis, local_cuts[axis])
        if not all(np.isfinite(a).all() and a.min() >= 4.6 and a.max() <= 5.4 for a in arrays.values()):
            raise ValueError("Invalid slice or values outside shared colour limits")
        block["slices"] = arrays
        blocks.append(block)
    for block in blocks[1:]:
        if require_equal_volumes:
            np.testing.assert_array_equal(block["fields"]["true"][0], blocks[0]["fields"]["true"][0])
        if require_equal_slices:
            for plane in ("xy", "yz", "xz"):
                np.testing.assert_array_equal(block["slices"]["true_"+plane], blocks[0]["slices"]["true_"+plane])
    return blocks, metrics_path


def main():
    blocks, metrics_path = build_blocks()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    fig = plt.figure(figsize=(17, 24))
    grid = fig.add_gridspec(3, 1, left=.075, right=.95, top=.925, bottom=.055, hspace=.18)
    fig.suptitle("Matched checkerboards · identical cropped True fields", fontsize=17, y=.99)
    fig.text(.5, .972, "Rows: True / Soft / Hard · Columns: XY / YZ / XZ · Native cell values, no smoothing",
             ha="center", fontsize=10)
    fig.text(.5, .957, "Common displayed extent: 20 × 8 × 8 km · Crop-local coordinates · Different section levels, identical True slices",
             ha="center", fontsize=10, color="0.3")
    for i, block in enumerate(blocks):
        draw_block(fig, grid[i, 0], block, chr(97+i))
    fig.text(.5, .022,
             "4.2: crop native X = 4–24, Y = 0–8, Z = 0–8 km; Y cut 6 → 3.5 km.\n"
             "4.3: crop native X = 4–24 km; Z cut 4.5 → 3.5 km. 4.4: unchanged native domain and cuts.\n"
             "Cuts otherwise unchanged; axis origins reset to crop starts. This aligns periodic truth, not acquisition geometry or resolution.",
             ha="center", fontsize=9, linespacing=1.5, color="0.3")
    fig.savefig(PNG / f"{STEM}.png", dpi=200, bbox_inches="tight", pad_inches=.12)
    fig.savefig(HERE / f"{STEM}.pdf", dpi=200, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    exports = {f"s{b['section'].replace('.', '_')}_{key}": value for b in blocks for key, value in b["slices"].items()}
    for b in blocks:
        exports[f"s{b['section'].replace('.', '_')}_true_volume_m_s"] = b["fields"]["true"][0]
    data_path = HERE / f"{STEM}.npz"
    np.savez_compressed(data_path, **exports)
    report = dict(
        purpose="Separate feasibility demonstration: identical cropped True volumes and corresponding 2-D views",
        truth_volume_shape_xyz=list(blocks[0]["fields"]["true"][0].shape),
        truth_cell_size_m=blocks[0]["fields"]["true"][1], displayed_extent_m=EXTENT_M.tolist(),
        exact_truth_volume_equality=True, exact_truth_slice_equality=True,
        main_comparison_unchanged=True, inversions_rerun=False,
        crop_selection="Whole 4 km checkerboard blocks only, matching extent and phase; not chosen by inversion error",
        section_policy="Small Y adjustment in 4.2, small Z adjustment in 4.3; all other native cuts retained",
        caveats=["Cropping only 4.2 cannot match 4.3 and 4.4: their X extents are 24 and 20 km",
                 "Axis origins are crop-local, not the previous fault-test frame",
                 "Matching True fields does not make acquisition, noise or inversion resolution identical",
                 "Selected V3 checkpoints unchanged; recorded RMSE remains original full-domain RMSE, not crop RMSE"],
        blocks=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in b.items()
                 if k not in ("fields", "slices")} for b in blocks],
        data_sha256=sha(data_path),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (Path(__file__).resolve(), HERE / "make_comparison.py", metrics_path)},
    )
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PASS: identical 5 × 2 × 2 native True volumes and all three True section arrays across 4.2–4.4")
    print(PNG / f"{STEM}.png")


if __name__ == "__main__":
    main()
