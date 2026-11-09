"""Shared six-sphere truth: surface plan view and two constant-Y sections.

Run from EMTomo: python -B paper/make_spherical_anomalies_truth.py
Reads saved truth only; no forward/inverse solve or replacement model generation.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from make_checkerboard_geometry import map_frame

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/matched_geometry_20261021/snapshot/projects/forward_modeling/experiments/input"
ARTIFICIAL = INPUT / "flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001"
GEYSERS = INPUT / "geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001"
OUTPUT = Path(__file__).resolve().parent / "figures"
STEM = "spherical_anomalies_truth"
SECTIONS = ((2.4, (1, 3, 5)), (4.8, (2, 4, 6)))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with np.load(ARTIFICIAL / "model.npz", allow_pickle=False) as data:
        velocity = data["velocity"] / 1000
        cell = float(data["cell_size_m"]) / 1000
        origin = data["origin_m"] / 1000
    with np.load(GEYSERS / "model.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["velocity"] / 1000, velocity)
        np.testing.assert_array_equal(data["origin_m"] / 1000, origin)
        assert float(data["cell_size_m"]) / 1000 == cell
    parameters = json.loads((ARTIFICIAL / "generation.json").read_text())["parameters"]
    spheres = parameters["anomalies"]
    extent = np.array(velocity.shape) * cell
    assert velocity.shape == (160, 56, 56) and cell == .125
    assert np.array_equal(origin, [0, 0, 0]) and len(spheres) == 6
    assert np.isfinite(velocity).all() and np.all(velocity[:, :, 0] == 3.8)
    assert np.array_equal(extent, [20, 7, 7])

    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 9,
        "mathtext.fontset": "stix", "axes.labelsize": 9,
        "axes.titlesize": 10, "axes.linewidth": .6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30",
        "savefig.facecolor": "white",
    })
    # Preserve the checkerboard's velocity anchors, extending its range for layers/spheres.
    anchors = ((3.5, "#454545"), (4., "#7a7a7a"), (4.75, "#ffffff"),
               (5.25, "#e32400"), (6., "#61177c"), (6.5, "#351041"))
    norm = Normalize(3.5, 6.5)
    cmap = LinearSegmentedColormap.from_list(
        STEM, [(norm(v), color) for v, color in anchors], N=2048)
    assert velocity.min() >= norm.vmin and velocity.max() <= norm.vmax
    fig = plt.figure(figsize=(16, 2.8), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, .025), wspace=.08)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    top = axes[0]
    image = top.imshow(velocity[:, :, 0].T, origin="lower", extent=(0, 20, 0, 7),
                       cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    top.set(title="(a)  Plan view · surface model", ylabel="$Y$ (km)", ylim=(0, 7))
    records = []
    for ax, (y, expected), letter, style in zip(axes[1:], SECTIONS, "bc", ("-", "--")):
        index = int(np.floor(y / cell))
        sampled_y = (index + .5) * cell
        visible = tuple(i for i, sphere in enumerate(spheres, 1)
                        if abs(sampled_y - sphere["center_km"][1]) < sphere["radius_km"])
        assert visible == tuple(sorted(expected))
        assert all(abs(sampled_y - spheres[i - 1]["center_km"][1])
                   < .5 * spheres[i - 1]["radius_km"] for i in expected)
        ax.imshow(velocity[:, index, :].T, origin="upper", extent=(0, 20, 7, 0),
                  cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.set(title=f"({letter})  Section · $Y = {y:g}$ km",
               ylabel="Depth (km)", ylim=(7, 0))
        top.axhline(y, color="#262b30", lw=.9, ls=style, zorder=3)
        for tip, tail in (((0, y), (-.65, y)), ((20, y), (20.65, y))):
            top.annotate("", xy=tip, xytext=tail,
                         arrowprops=dict(arrowstyle="->", color="#262b30", lw=.9,
                                         mutation_scale=9, shrinkA=0, shrinkB=0),
                         annotation_clip=False, zorder=7)
        top.text(19.4, y + .22, f"({letter})  $Y = {y:g}$ km",
                 ha="right", va="bottom", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5), zorder=6)
        records.append({"requested_y_km": y, "native_cell_index": index,
                        "native_cell_y_bounds_km": [index * cell, (index + 1) * cell],
                        "native_cell_center_y_km": sampled_y, "visible_spheres": list(visible)})
    for ax in axes:
        ax.set(xlabel="$X$ (km)", xlim=(0, 20), xticks=np.arange(0, 21, 4),
               yticks=(0, 2, 4, 6, 7))
        ax.tick_params(direction="out", pad=3)
        map_frame(ax, interval=2, thickness=.06)
    color_ax = fig.add_subplot(grid[0, 3])
    color_ax.set_box_aspect((7 / 20) / .025)
    bar = fig.colorbar(image, cax=color_ax, orientation="vertical", ticks=np.arange(3.5, 6.51, .5))
    color_ax.set_title("$V$ (km/s)", fontsize=8, pad=8)
    bar.outline.set_linewidth(.45)
    color_ax.tick_params(labelsize=8, width=.45, length=2)
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    report = {
        "shared_truth_verified": True, "extent_km": extent.tolist(),
        "native_cell_size_km": cell, "sections": records,
        "surface_sampling": "Native top cell, z in [0, 0.125) km; constant 3.8 km/s",
        "sampling": "Native piecewise-constant truth; positive-coordinate cell at a boundary; no smoothing",
        "sphere_parameters": spheres, "color_anchors_km_s": dict(anchors),
        "color_limits_km_s": [norm.vmin, norm.vmax], "colorbar_orientation": "vertical",
        "velocity_range_km_s": [float(velocity.min()), float(velocity.max())],
        "frame_interval_km": 2, "stations_or_events": False,
        "sha256": {str(p): sha256(p) for p in
                   (ARTIFICIAL / "model.npz", ARTIFICIAL / "generation.json",
                    GEYSERS / "model.npz", Path(__file__),
                    Path(__file__).with_name("make_checkerboard_geometry.py"))},
    }
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"sections": records, "velocity_range_km_s": report["velocity_range_km_s"]}, indent=2))
    print(OUTPUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
