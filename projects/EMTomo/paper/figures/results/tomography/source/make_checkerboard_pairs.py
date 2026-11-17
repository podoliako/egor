"""Split matched 4.2–4.4 checkerboards into a shared True view and 3×6 results.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_checkerboard_pairs.py
Existing figures are not overwritten; matched crops/cuts are reused unchanged.
"""
from pathlib import Path
import json

from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np

from make_matched_checkerboards import build_blocks, EXTENT_M
from make_comparison import HERE, ROOT, PNG, SPACING_M, sha

RESULTS = "checkerboards_matched_results_pairs"
TRUTH = "checkerboards_matched_true_only"
PLANES = [("xy", 0, 1, 2, "Horizontal XY"),
          ("xz", 0, 2, 1, "Longitudinal XZ"),
          ("yz", 1, 2, 0, "Transverse YZ")]
LIMITS = (4.6, 5.4)


def panel(ax, block, mode, plane, h, v, fixed):
    extent = EXTENT_M / 1000
    values = block["slices"][mode + "_" + plane]
    ax.imshow(values, origin="upper", extent=(0, extent[h], extent[v], 0),
              cmap="RdBu_r", norm=Normalize(*LIMITS), interpolation="nearest",
              aspect="equal", rasterized=True)
    if mode != "true":
        truth = block["slices"]["true_" + plane]
        columns = np.flatnonzero(np.any(np.diff(truth, axis=1) != 0, axis=0)) + 1
        rows = np.flatnonzero(np.any(np.diff(truth, axis=0) != 0, axis=1)) + 1
        for i in columns:
            ax.axvline(i * SPACING_M / 1000, color="black", linewidth=.45, alpha=.8)
        for i in rows:
            ax.axhline(i * SPACING_M / 1000, color="black", linewidth=.45, alpha=.8)
    ax.set(xlim=(0, extent[h]), ylim=(0, extent[v]) if plane == "xy" else (extent[v], 0),
           xlabel="X (km)" if h == 0 else "Y (km)",
           ylabel="Y (km)" if plane == "xy" else "Depth (km)")
    ax.xaxis.set_major_locator(MultipleLocator(4 if extent[h] >= 16 else 2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(labelsize=8, length=3, pad=2)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    local = block["crop"]["section_crop_local_xyz_m"][fixed] / 1000
    ax.annotate(f'{"XYZ"[fixed]} = {local:g} km', xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                ha="left", va="bottom", annotation_clip=False)


def colorbar(fig, right, bottom_in, height_in):
    fw, fh = fig.get_size_inches()
    cax = fig.add_axes([right + .022, bottom_in / fh, .009, height_in / fh])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*LIMITS), cmap="RdBu_r"),
                 cax=cax, label="Vp (km/s)")
    cax.tick_params(labelsize=8)


def save(fig, stem):
    fig.savefig(PNG / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=.12)
    fig.savefig(HERE / f"{stem}.pdf", dpi=220, bbox_inches="tight", pad_inches=.12)


def draw_results(blocks, stem=RESULTS):
    fig = plt.figure(figsize=(21, 8.4))
    fw, fh = fig.get_size_inches()
    left = .065
    scale = .17  # Inches per km, identical across every panel.
    within_gap, group_gap = .32, .65
    row_height, row_gap = 8 * scale, .65
    top_in = 6.75
    right = left
    for row, block in enumerate(blocks):
        bottom_in = top_in - row * (row_height + row_gap) - row_height
        x_in = left * fw
        for plane, h, v, fixed, heading in PLANES:
            width = EXTENT_M[h] / 1000 * scale
            group_start = x_in
            for mode in ("hard", "soft"):
                ax = fig.add_axes([x_in / fw, bottom_in / fh, width / fw, row_height / fh])
                panel(ax, block, mode, plane, h, v, fixed)
                if row == 0:
                    ax.set_title(mode.capitalize(), fontsize=11, pad=21)
                x_in += width + within_gap
            group_end = x_in - within_gap
            if row == 0:
                fig.text((group_start + group_end) / 2 / fw, (top_in + .72) / fh,
                         heading, ha="center", fontsize=13, weight="bold")
            x_in += group_gap - within_gap
        right = (x_in - group_gap) / fw
        fig.text(left - .038, (bottom_in + row_height / 2) / fh,
                 f'§{block["section"]}', ha="center", va="center", fontsize=12, weight="bold", rotation=90)
    content_bottom = top_in - 2 * (row_height + row_gap) - row_height
    colorbar(fig, right, content_bottom + (top_in - content_bottom) * .25,
             (top_in - content_bottom) * .5)
    fig.suptitle("Checkerboard inversion results · matched cropped domains", fontsize=17, y=.985)
    fig.text(.5, .075,
             "Rows: 4.2 one-sided stations · 4.3 two station clusters · 4.4 Geysers geometry. All models: selected V₃ (3 updates).\n"
             "Crop-local coordinates; matched True patterns shown separately. Thin black grid: original checkerboard interfaces.\n"
             "Native cell values, no smoothing. Native crops and plane levels are recorded in the accompanying provenance.",
             ha="center", va="top", fontsize=9, linespacing=1.5, color="0.3")
    save(fig, stem)
    return fig


def draw_truth(blocks, stem=TRUTH, matching_volumes=True):
    fig = plt.figure(figsize=(12.8, 3.9))
    fw, fh = fig.get_size_inches()
    scale, gap = .205, .7
    x_in, bottom = .75, .85
    height = 8 * scale
    block = blocks[0]
    for plane, h, v, fixed, heading in PLANES:
        width = EXTENT_M[h] / 1000 * scale
        ax = fig.add_axes([x_in / fw, bottom / fh, width / fw, height / fh])
        panel(ax, block, "true", plane, h, v, fixed)
        # True images are equal despite distinct section levels; do not imply one shared cut.
        ax.texts[0].remove()
        ax.set_title(heading, fontsize=11, pad=9)
        x_in += width + gap
    colorbar(fig, (x_in - gap) / fw, bottom + height * .2, height * .6)
    fig.suptitle("Common True checkerboard · sections 4.2–4.4", fontsize=16, y=.97)
    subtitle = ("Exactly identical cropped True volumes and corresponding slices · 20 × 8 × 8 km"
                if matching_volumes else
                "Exactly identical True slices · Different 3-D checkerboard phases · Right-edge crops")
    fig.text(.5, .78, subtitle,
             ha="center", fontsize=10, color="0.3")
    fig.text(.5, .055,
             "XY / XZ / YZ shown once; fixed coordinates differ between series but sample identical checkerboard phases.\n"
             "Crop-local axes; actual cut coordinates are recorded in the results figure and provenance.",
             ha="center", fontsize=9, linespacing=1.5, color="0.3")
    save(fig, stem)
    return fig


def main():
    blocks, metrics_path = build_blocks()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    plt.close(draw_results(blocks))
    plt.close(draw_truth(blocks))
    data = {f"s{b['section'].replace('.', '_')}_{key}": value for b in blocks for key, value in b["slices"].items()}
    data_path = HERE / f"{RESULTS}.npz"
    np.savez_compressed(data_path, **data)
    report = dict(results_layout=dict(rows=["4.2", "4.3", "4.4"],
                                     columns=["XY Hard", "XY Soft", "XZ Hard", "XZ Soft", "YZ Hard", "YZ Soft"]),
                  true_layout="One shared row: XY / XZ / YZ; exact truth equality checked before plotting",
                  source_crop_policy="Unchanged make_matched_checkerboards.build_blocks crops and cuts",
                  original_and_matched_figures_unchanged=True,
                  outputs=[RESULTS, TRUTH], color_limits_km_s=list(LIMITS),
                  data_sha256=sha(data_path),
                  blocks=[dict(section=b["section"], crop=b["crop"], records=b["records"],
                               source_directory=b["source_directory"], input_sha256=b["input_sha256"]) for b in blocks],
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 (Path(__file__).resolve(), HERE / "make_comparison.py",
                                  HERE / "make_matched_checkerboards.py", metrics_path)})
    (HERE / f"{RESULTS}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(PNG / f"{TRUTH}.png")
    print(PNG / f"{RESULTS}.png")


if __name__ == "__main__":
    main()
