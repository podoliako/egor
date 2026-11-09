"""Two acquisition geometries: plan view and perpendicular vertical projections.

Run from EMTomo: python -B paper/paper_scripts/make_spherical_anomalies_geometry.py
Reads saved coordinates only, with no forward/inverse solve.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from make_checkerboard_geometry import map_frame
from make_spherical_anomalies_truth import ARTIFICIAL, GEYSERS, OUTPUT, sha256

STEM = "spherical_anomalies_geometry"
SECTION_X = 10.0
SECTION_Y = 3.5


def coordinates(source, name):
    with (source / name).open() as stream:
        rows = list(csv.DictReader(stream))
    result = np.array([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows]) / 1000
    assert result.ndim == 2 and result.shape[1] == 3 and np.isfinite(result).all()
    assert np.all(result >= 0) and np.all(result <= [20, 7, 7])
    return result


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 9, "mathtext.fontset": "stix",
        "axes.labelsize": 9, "axes.titlesize": 10, "axes.linewidth": .6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(14.5, 6.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=(20, 20, 7), height_ratios=(1, 1, .09),
                          wspace=.08, hspace=.12)
    records = []
    event_style = dict(s=13, marker="*", facecolors="white", edgecolors="#41464b",
                       linewidths=.3, zorder=4)
    station_style = dict(s=33, marker="^", facecolors="#b1b1b1", edgecolors="#41464b",
                         linewidths=.35, zorder=7)
    surface_marker = [(-.5, 0), (.5, 0), (0, 1), (-.5, 0)]
    for row, (name, source, expected) in enumerate(
            (("Artificial", ARTIFICIAL, (52, 1920)), ("Geysers", GEYSERS, (26, 3776)))):
        stations = coordinates(source, "stations.csv")
        events = coordinates(source, "events.csv")
        assert (len(stations), len(events)) == expected
        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        projection_records = []
        for col, (ax, components) in enumerate(zip(axes, ((0, 1), (0, 2), (1, 2)))):
            ep, counts = np.unique(events[:, components], axis=0, return_counts=True)
            sp, station_counts = np.unique(stations[:, components], axis=0, return_counts=True)
            assert counts.sum() == len(events) and station_counts.sum() == len(stations)
            ax.set_facecolor("#f0f0f0")
            ax.scatter(*ep.T, **event_style, alpha=.4 if name == "Geysers" else 1.)
            style = station_style if col == 0 else {
                **station_style, "marker": surface_marker, "s": 100}
            ax.scatter(*sp.T, **style, clip_on=False)
            ax.set_aspect("equal")
            ax.set(xlim=(0, 7 if col == 2 else 20), ylim=(0, 7) if col == 0 else (7, 0),
                   xlabel="$Y$ (km)" if col == 2 else "$X$ (km)",
                   ylabel="$Y$ (km)" if col == 0 else "Depth (km)",
                   xticks=(0, 2, 4, 6, 7) if col == 2 else np.arange(0, 21, 4),
                   yticks=(0, 2, 4, 6, 7))
            letter = "abcdef"[row * 3 + col]
            title = (f"{name} · plan view" if col == 0 else
                     f"Section · $Y = {SECTION_Y:g}$ km" if col == 1 else
                     f"$X = {SECTION_X:g}$ km")
            ax.set_title(f"({letter})  {title}", pad=11)
            ax.tick_params(direction="out", pad=3)
            map_frame(ax, interval=2, thickness=.06)
            projection_records.append({
                "components": list(components), "event_markers": len(ep),
                "station_markers": len(sp),
                "events_per_marker_range": [int(counts.min()), int(counts.max())],
                "stations_per_marker_range": [int(station_counts.min()), int(station_counts.max())],
            })
        top = axes[0]
        top.axhline(SECTION_Y, color="#41464b", lw=.7, zorder=2)
        top.axvline(SECTION_X, color="#41464b", lw=.7, zorder=2)
        for tip, tail in (((0, SECTION_Y), (-.6, SECTION_Y)),
                          ((20, SECTION_Y), (20.6, SECTION_Y)),
                          ((SECTION_X, 0), (SECTION_X, -.55)),
                          ((SECTION_X, 7), (SECTION_X, 7.55))):
            top.annotate("", xy=tip, xytext=tail,
                         arrowprops=dict(arrowstyle="->", color="#41464b", lw=.9,
                                         mutation_scale=8, shrinkA=0, shrinkB=0),
                         annotation_clip=False, zorder=8)
        records.append({
            "geometry": name, "source": str(source), "events": len(events),
            "stations": len(stations), "projections": projection_records,
                        "event_alpha": .4 if name == "Geysers" else 1.,
            "station_depth_range_km": [float(stations[:, 2].min()), float(stations[:, 2].max())],
            "event_depth_range_km": [float(events[:, 2].min()), float(events[:, 2].max())],
            "sha256": {f: sha256(source / f) for f in ("events.csv", "stations.csv")},
        })
    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.set_axis_off()
    legend_ax.legend(handles=[
        Line2D([], [], marker="^", color="none", markerfacecolor="#b1b1b1",
               markeredgecolor="#41464b", markeredgewidth=.35, markersize=7, label="Stations"),
        Line2D([], [], marker="*", color="none", markerfacecolor="white",
               markeredgecolor="#41464b", markeredgewidth=.5, markersize=8, label="Events"),
    ], loc="center", ncol=2, frameon=False)
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{extension}"
        fig.savefig(path, dpi=300)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    report = {
        "section_x_km": SECTION_X, "section_y_km": SECTION_Y,
        "display": "All coordinates orthogonally projected; exact duplicate projections drawn once; no filtering/jitter",
        "background": "Uniform neutral grey; not a velocity field",
        "depth": "Positive downward; station elevations retained as saved depths",
        "frame_interval_km": 2, "geometries": records,
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print(OUTPUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
