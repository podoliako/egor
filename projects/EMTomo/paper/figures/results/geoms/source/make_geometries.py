"""Draft acquisition geometries; no forward solves or inversions.

Run from any directory: python -B /path/to/this/make_geometries.py
PNG previews go only to ../png; PDFs, exported coordinates and provenance stay here.
All points are projected onto the vertical planes, not selected in thin slices.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
INPUT = ROOT.parent / "forward_modeling/experiments/input"
sys.path.insert(0, str(ROOT))
from prepare_geysers_geometry import local_xy_km


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(rows):
    return np.array([[float(row[k + "_m"]) for k in "xyz"] for row in rows])


def synthetic(key, title, sections, experiment, source=None):
    directory = source or INPUT / experiment
    stations = read_csv(directory / "stations.csv")
    events = read_csv(directory / "events.csv")
    with np.load(directory / "model.npz", allow_pickle=False) as model:
        extent = np.array(model["velocity"].shape) * float(model["cell_size_m"])
        assert np.all(model["origin_m"] == 0)
    return dict(key=key, title=title, sections=sections,
                source=str(directory), extent_m=extent,
                station_ids=[r["station_id"].removeprefix("BG.") for r in stations],
                event_ids=[r["event_id"] for r in events],
                stations_m=xyz(stations), events_m=xyz(events),
                source_sha256={name: sha(directory / name)
                               for name in ("stations.csv", "events.csv", "model.npz")})


def real_geometry():
    directory = ROOT / "data/geysers_2011/real_experiment_v1"
    geometry = read_json(directory / "geometry.json")
    catalog = {r["event_id"]: r for r in read_csv(directory / "catalog.csv")}
    with np.load(directory / "emtomo/arrivals.npz", allow_pickle=False) as data:
        event_ids = data["event_ids"].astype(str).tolist()
        station_ids = data["station_ids"].astype(str).tolist()
        stations = data["station_coordinates_m"].copy()
    events = []
    for identifier in event_ids:
        row = catalog[identifier]
        x, y = local_xy_km(float(row["latitude"]), float(row["longitude"]))
        x += geometry["geographic_reference_in_new_xy_m"][0] / 1000
        y += geometry["geographic_reference_in_new_xy_m"][1] / 1000
        z = geometry["top_elevation_m"] + 1000 * float(row["depth_km"])
        events.append([1000*x, 1000*y, z])
    return dict(key="real_picks", title="The Geysers: real picks", sections="4.7",
                source=str(directory), extent_m=np.array(geometry["domain_extent_m"]),
                station_ids=[s.removeprefix("BG.") for s in station_ids],
                event_ids=event_ids, stations_m=stations, events_m=np.array(events),
                source_sha256={name: sha(directory / name) for name in
                               ("geometry.json", "catalog.csv", "emtomo/arrivals.npz")})


def compare(a, b, offset):
    """Compare matching identifiers after translating a into b's frame, in metres."""
    result = dict(from_geometry=a["key"], to_geometry=b["key"], translation_m=offset)
    for kind in ("station", "event"):
        left = dict(zip(a[kind + "_ids"], a[kind + "s_m"]))
        right = dict(zip(b[kind + "_ids"], b[kind + "s_m"]))
        common = sorted(left.keys() & right.keys())
        delta = [np.linalg.norm(left[k] + offset - right[k]) for k in common]
        result[kind + "s"] = dict(common=len(common), only_from=len(left.keys()-right.keys()),
                                  only_to=len(right.keys()-left.keys()),
                                  max_common_coordinate_difference_m=float(max(delta)) if delta else None)
    return result


def export_coordinates(row):
    path = HERE / (row["key"] + "_coordinates.csv")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "id", "x_m", "y_m", "z_m"])
        for kind in ("station", "event"):
            for identifier, point in zip(row[kind + "_ids"], row[kind + "s_m"]):
                writer.writerow([kind, identifier, *point])


EVENT_COLOR = "#59778b"
STATION_COLOR = "#969696"
LONG_COLOR = "#326a9e"
SHORT_COLOR = "#b26d36"


def plot_row(fig, spec, row, letter):
    # Equal map distances on every axis; the short-side view has half the column width.
    grid = spec.subgridspec(1, 3, width_ratios=[2, 2, 1], wspace=.25)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    stations = row["stations_m"] / 1000
    events = row["events_m"] / 1000
    lx, ly, lz = row["extent_m"] / 1000
    section_y, section_x = ly / 2, lx / 2
    row["section_y_km"], row["section_x_km"] = section_y, section_x
    titles = ["Plan view", f"Long-side projection · Y = {section_y:g} km",
              f"Short-side projection\nX = {section_x:g} km"]
    for i, (ax, dims) in enumerate(zip(axes, [(0, 1), (0, 2), (1, 2)])):
        ev = np.unique(events[:, dims], axis=0)
        st = np.unique(stations[:, dims], axis=0)
        ax.scatter(*ev.T, s=10, marker="*", facecolors="white", edgecolors=EVENT_COLOR,
                   linewidths=.35, alpha=.8, zorder=3, rasterized=True)
        ax.scatter(*st.T, s=25, marker="^", facecolors=STATION_COLOR,
                   edgecolors="#383838", linewidths=.35, zorder=5, clip_on=False)
        width = lx if i < 2 else ly
        height = ly if i == 0 else lz
        ax.set_xlim(0, width)
        ax.set_ylim((0, height) if i == 0 else (height, 0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (km)" if i < 2 else "Y (km)", labelpad=2)
        ax.set_ylabel("Y (km)" if i == 0 else "Depth (km)", labelpad=2)
        ax.set_title(titles[i], fontsize=9, pad=7,
                     color=LONG_COLOR if i == 1 else SHORT_COLOR if i == 2 else "#222222")
        ax.xaxis.set_major_locator(MultipleLocator(4 if width >= 16 else 2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.tick_params(labelsize=8, length=3, width=.5)
        ax.grid(color="#e6e6e6", linewidth=.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        row.setdefault("projected_markers", {})[str(i)] = dict(events=len(ev), stations=len(st))
    axes[0].axhline(section_y, color=LONG_COLOR, lw=.9, ls="--", zorder=4)
    axes[0].axvline(section_x, color=SHORT_COLOR, lw=.9, ls=":", zorder=4)
    box = spec.get_position(fig)
    fig.text(box.x0, box.y1 + .019,
             f"({letter})  {row['title']}  |  §{row['sections']}  |  "
             f"{len(stations)} stations, {len(events)} events",
             fontsize=10, weight="bold", ha="left", va="bottom")


def legend(fig, y):
    handles = [Line2D([], [], marker="^", ls="", color="#383838", markerfacecolor=STATION_COLOR,
                      markersize=6, label="Stations"),
               Line2D([], [], marker="*", ls="", color=EVENT_COLOR, markerfacecolor="white",
                      markersize=7, label="Events (catalog positions for real picks)"),
               Line2D([], [], color=LONG_COLOR, ls="--", label="Long-side plane"),
               Line2D([], [], color=SHORT_COLOR, ls=":", label="Short-side plane")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=4,
               frameon=False, fontsize=8)


def main():
    assert PNG.is_dir()
    geometry_source = read_json(ROOT / "paper/figures/checkerboard_nx6_geometry.json")["source"]
    rows = [
        synthetic("full_coverage", "Full coverage", "4.1", "", source=Path(geometry_source)),
        synthetic("one_sided", "One-sided stations", "4.2",
                  "onesided_nx6_24x12x12_18st_1200ev_noise01_pick005_seed42"),
        synthetic("two_clusters", "Two station clusters", "4.3",
                  "nx6_bottom_blobs_24x8x8_22st_600ev_noise01_pick005_seed42"),
        synthetic("real_geometry_synthetic", "The Geysers: checkerboard / heterogeneous model", "4.4, 4.6",
                  "geysers_realgeom_cb5x2x2_20x8x8_26st_3465ev_noise01_pick005_seed42"),
        synthetic("fault", "The Geysers: vertical fault", "4.5", "gfsoftscreen28_26st_noise02_3776ev"),
        real_geometry(),
    ]
    heterogeneous = synthetic("heterogeneous", "", "4.6",
                              "geysers_realistic_previous_20x8x8_26st_3465ev_noise01_pick005_seed42")
    for kind in ("station", "event"):
        assert rows[3][kind+"_ids"] == heterogeneous[kind+"_ids"]
        np.testing.assert_array_equal(rows[3][kind+"s_m"], heterogeneous[kind+"s_m"])
    np.testing.assert_array_equal(rows[3]["extent_m"], heterogeneous["extent_m"])
    for row in rows:
        for kind in ("station", "event"):
            points = row[kind+"s_m"]
            assert points.shape == (len(row[kind+"_ids"]), 3)
            assert np.isfinite(points).all()
            assert np.all(points >= -1e-6) and np.all(points <= row["extent_m"]+1e-6)
        export_coordinates(row)
    comparisons = [compare(rows[3], heterogeneous, [0, 0, 0]),
                   compare(rows[3], rows[4], [1000, 0, 0]),
                   compare(rows[3], rows[5], [4000, 0, 0]),
                   compare(rows[4], rows[5], [3000, 0, 0])]
    plotted_rows = [rows[1], rows[2], rows[4]]
    plotted_rows[-1]["title"] = "The Geysers: smallest domain (fault test)"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13, 9.8))
    grid = fig.add_gridspec(3, 1, left=.065, right=.98, top=.88, bottom=.11, hspace=.57)
    fig.suptitle("Acquisition geometries — draft layout", fontsize=15, y=.992)
    fig.text(.5, .946, "No velocity background · all points projected · depth positive downward",
             ha="center", fontsize=9, color="#555555")
    for i, row in enumerate(plotted_rows):
        plot_row(fig, grid[i, 0], row, chr(97+i))
    legend(fig, .024)
    fig.text(.5, .012, "Section lines mark projection planes; no distance-to-plane filtering. "
             "Exact coincident projections are drawn once.", ha="center", fontsize=8, color="#555555")
    fig.savefig(PNG / "geometry_overview.png", dpi=200)
    fig.savefig(HERE / "all_geometries.pdf", dpi=200)
    plt.close(fig)
    for i, row in enumerate(plotted_rows):
        fig = plt.figure(figsize=(13, 3.7))
        spec = fig.add_gridspec(1, 1, left=.065, right=.98, top=.80, bottom=.20)
        plot_row(fig, spec[0, 0], row, chr(97+i))
        legend(fig, .005)
        fig.savefig(PNG / (row["key"] + ".png"), dpi=200)
        plt.close(fig)
    report = dict(
        purpose="Draft layout only; no article/PDF changes, inversions or forward solves.",
        projections="All events/stations projected orthogonally; no slice filtering, jitter or subsampling. Exact duplicate 2D positions drawn once.",
        coordinate_policy="Each geometry retains its native translated Cartesian frame, metric scale and full domain. Depth positive down; real-pick symbols use retained catalog positions, not relocated estimates.",
        shared_geometry="Sections 4.4 and 4.6 have identical ordered station and event coordinates and identical domains. Only the smallest Geysers domain (fault test, section 4.5) is plotted as a geometry illustration, not as the exact acquisition geometry of all real-geometry tests.",
                plotted_geometry_keys=[row["key"] for row in plotted_rows],
                omitted_geometry_keys=[row["key"] for row in rows if row["key"] not in {r["key"] for r in plotted_rows}],
        datum_caveat="Catalog depths and station elevations are treated as sharing a vertical datum; compatibility has not been independently established.",
        comparisons=comparisons,
        geometries=[{k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in row.items() if k not in ("events_m", "stations_m", "event_ids", "station_ids")}
                    for row in rows],
        heterogeneous_source=dict(source=heterogeneous["source"], source_sha256=heterogeneous["source_sha256"]),
        script_sha256=sha(Path(__file__)),
    )
    (HERE / "geometry_inventory.json").write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(f"{row['key']}: {len(row['station_ids'])} stations, {len(row['event_ids'])} events; "
              f"domain {row['extent_m']/1000} km")
    print(json.dumps(comparisons, indent=2))
    assert all(p.suffix == ".png" for p in PNG.iterdir())
    print("Saved combined preview + three individual rows; PNG directory contains PNG files only.")


if __name__ == "__main__":
    main()
