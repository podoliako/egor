"""Publication geometry figure from the original nx6 experiment inputs."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Polygon

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/checkerboard_scaled_20261021/snapshot/projects/forward_modeling/experiments/input/checkerboard_24x12x12_nx6_98st_1200ev_grid20x10x6_minz3_noise01_pick001_seed42"
OUTPUT = Path(__file__).resolve().parent / "figures"
STEM = "checkerboard_nx6_geometry"


def coordinates(name):
    with (SOURCE / name).open() as stream:
        rows = list(csv.DictReader(stream))
    return np.array([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows]) / 1000


def map_frame(ax, interval=2.0, thickness=.09):
    """Inset alternating ring with shared, mitred corner edges and no overlaps."""
    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    outer = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    inner = np.array([(xmin + thickness, ymin + thickness),
                      (xmax - thickness, ymin + thickness),
                      (xmax - thickness, ymax - thickness),
                      (xmin + thickness, ymax - thickness)])
    segment = 0
    for side in range(4):
        a, b = outer[side], outer[(side + 1) % 4]
        c, d = inner[side], inner[(side + 1) % 4]
        length = np.linalg.norm(b - a)
        breaks = np.append(np.arange(0, length, interval), length) / length
        for start, stop in zip(breaks[:-1], breaks[1:]):
            vertices = np.array([a + start * (b - a), a + stop * (b - a),
                                 c + stop * (d - c), c + start * (d - c)])
            assert np.all(vertices >= [xmin, ymin]) and np.all(vertices <= [xmax, ymax])
            ax.add_patch(Polygon(vertices, closed=True,
                                 facecolor="black" if segment % 2 == 0 else "white",
                                 edgecolor="none", antialiased=False, clip_on=True, zorder=5))
            segment += 1
    for inset in (0, thickness):
        ax.add_patch(Rectangle((xmin + inset, ymin + inset),
                               xmax - xmin - 2 * inset, ymax - ymin - 2 * inset,
                               facecolor="none", edgecolor="black", linewidth=.25,
                               joinstyle="miter", clip_on=True, zorder=5.1))
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    with np.load(SOURCE / "model.npz") as data:
        velocity = data["velocity"] / 1000
        cell = float(data["cell_size_m"]) / 1000
        origin = data["origin_m"] / 1000
    stations, events = coordinates("stations.csv"), coordinates("events.csv")
    assert velocity.shape == (6, 3, 3)
    assert len(stations) == 98 and len(events) == 1200
    assert np.allclose(origin, 0)
    section_y = 10.0
    section_index = int(np.floor((section_y - origin[1]) / cell))
    section_x = 6.0
    cross_index = int(np.floor((section_x - origin[0]) / cell))
    extent = np.array(velocity.shape) * cell
    top_events, top_counts = np.unique(events[:, :2], axis=0, return_counts=True)
    side_events, side_counts = np.unique(events[:, [0, 2]], axis=0, return_counts=True)
    side_stations = np.unique(stations[:, [0, 2]], axis=0)
    cross_events, cross_counts = np.unique(events[:, [1, 2]], axis=0, return_counts=True)
    cross_stations = np.unique(stations[:, [1, 2]], axis=0)
    assert np.all(top_counts == 6) and np.all(side_counts == 10)
    assert np.all(cross_counts == 20)

    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.linewidth": .6, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30",
        "savefig.facecolor": "white",
    })
    colors = ["#ffffff", "#e32400"]
    cmap = LinearSegmentedColormap.from_list("initial_and_truth", colors, N=1024)
    norm = Normalize(4.75, 5.25)
    initial_path = ROOT / "runs/run_cbscale21-nx6-hard-k15-cov0-fast_v1.4_cov-0_ncand-15_20261022_084247_c78101a2744e4a4e9e15ee90c4a992d6/initial_model.npy"
    dd_path = ROOT / "runs/checkerboard_scaled_20261021/outputs/nx6-tomodd/state.npz"
    initial_model = np.load(initial_path, allow_pickle=False) / 1000
    assert initial_model.shape == (24, 12, 12) and np.all(initial_model == 5.0)
    with np.load(dd_path, allow_pickle=False) as state:
        assert np.all(state["initial_vp"] == 5.0)
    fig = plt.figure(figsize=(17.0, 3.5), layout="constrained")
    grid = fig.add_gridspec(1, 5, width_ratios=(2, 2, 1, 2, .07), wspace=.08)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    top, side, cross, initial_ax = axes
    color_ax = fig.add_subplot(grid[0, 4])
    top.imshow(velocity[:, :, 0].T, origin="lower", extent=(0, extent[0], 0, extent[1]),
               cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    side.imshow(velocity[:, section_index, :].T, origin="upper",
                extent=(0, extent[0], extent[2], 0), cmap=cmap, norm=norm,
                interpolation="nearest", aspect="equal")
    image = cross.imshow(velocity[cross_index, :, :].T, origin="upper",
                         extent=(0, extent[1], extent[2], 0), cmap=cmap, norm=norm,
                         interpolation="nearest", aspect="equal")
    event_style = dict(s=24, marker="*", facecolors="white", edgecolors="black", linewidths=.35, zorder=4)
    station_style = dict(s=28, marker="^", facecolors="#B1B1B1", edgecolors="none", linewidths=0, zorder=6)
    top.scatter(*top_events.T, **event_style)
    top.scatter(*stations[:, :2].T, **station_style)
    side.scatter(*side_events.T, **event_style)
    surface_station_style = {**station_style, "s": 112,
                             "marker": [(-.5, 0), (.5, 0), (0, 1), (-.5, 0)]}
    side.scatter(*side_stations.T, **surface_station_style, clip_on=False)
    cross.scatter(*cross_events.T, **event_style)
    cross.scatter(*cross_stations.T, **surface_station_style, clip_on=False)
    top.axhline(section_y, color="#41464b", lw=.85, zorder=3)
    top.axvline(section_x, color="#41464b", lw=.85, zorder=3)
    for tip, tail in [((0, section_y), (-.7, section_y)),
                      ((extent[0], section_y), (extent[0] + .7, section_y)),
                      ((section_x, 0), (section_x, -.7)),
                      ((section_x, extent[1]), (section_x, extent[1] + .7))]:
        top.annotate("", xy=tip, xytext=tail,
                     arrowprops=dict(arrowstyle="->", color="#41464b", lw=.95,
                                     mutation_scale=9, shrinkA=0, shrinkB=0),
                     annotation_clip=False, zorder=7)
    top.annotate("$X = 6$ km", xy=(section_x, extent[1]), xytext=(0, 17),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8,
                 annotation_clip=False)
    top.annotate("$Y = 10$ km", xy=(extent[0], section_y), xytext=(15, 0),
                 textcoords="offset points", ha="left", va="center", fontsize=7,
                 annotation_clip=False)
    top.set(title="(a)  Plan view · surface model", ylabel="$Y$ (km)", ylim=(0, extent[1]))
    top.set_title("(a)  Plan view · surface model", pad=29)
    side.set(title="(b)  Section · $Y = 10$ km", ylabel="Depth (km)", ylim=(extent[2], 0))
    for ax in axes:
        ax.set(xlabel="$X$ (km)", xlim=(0, extent[0]), xticks=np.arange(0, 25, 4), yticks=np.arange(0, 13, 4))
        ax.tick_params(direction="out", pad=3)
        ax.set_axisbelow(True)
    cross.set(title="(c)  $X = 6$ km", xlabel="$Y$ (km)", ylabel="Depth (km)",
              xlim=(0, extent[1]), ylim=(extent[2], 0), xticks=np.arange(0, 13, 4))
    initial_ax.imshow(initial_model[:, int(section_y), :].T, origin="upper",
                      extent=(0, extent[0], extent[2], 0), cmap=cmap, norm=norm,
                      interpolation="nearest", aspect="equal")
    initial_ax.set(title="(d)  Initial model · EM & DD", ylabel="Depth (km)", ylim=(extent[2], 0))
    for ax in axes:
        map_frame(ax)
    colorbar = fig.colorbar(image, cax=color_ax, ticks=[4.75, 5.0, 5.25])
    color_ax.set_title("$V$ (km/s)", fontsize=8, pad=8)
    colorbar.outline.set_linewidth(.45)
    color_ax.tick_params(labelsize=8, width=.45, length=2)
    color_ax.set_box_aspect(.65 / .07)
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {
        "source": str(SOURCE), "section_y_km": section_y, "section_x_km": section_x,
        "model_section_cell_x_bounds_km": [cross_index * cell, (cross_index + 1) * cell],
        "frame_interval_km": 2.0,
        "shared_initial_model": {"velocity_km_s": 5.0, "em_source": str(initial_path),
                                 "dd_source": str(dd_path), "em_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                                 "dd_sha256": hashlib.sha256(dd_path.read_bytes()).hexdigest()},
        "vertical_station_symbols": "Triangle base anchored at true station depth; symbol extends above surface.",
        "cross_event_markers": len(cross_events), "events_per_cross_marker": 20,
        "cross_station_markers": len(cross_stations), "stations_per_cross_marker": 14,
        "model_section_cell_y_bounds_km": [section_index * cell, (section_index + 1) * cell],
        "stations": len(stations), "events": len(events),
        "top_event_markers": len(top_events), "events_per_top_marker": 6,
        "section_event_markers": len(side_events), "events_per_section_marker": 10,
        "section_station_markers": len(side_stations), "stations_per_section_marker": 7,
        "geometry": "All events/stations orthogonally projected; duplicate positions drawn once; no jitter or depth filtering.",
        "sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                   for name in ("model.npz", "events.csv", "stations.csv", "generation.json")},
    }
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
