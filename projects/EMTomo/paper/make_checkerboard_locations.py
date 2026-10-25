"""Latest saved nx6 locations for the true Y=10.2 km event row."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from make_checkerboard_geometry import map_frame
from make_checkerboard_comparison import ROOT, STUDY, EM_RUNS, sha

OUTPUT = Path(__file__).resolve().parent / "figures"
STEM = "checkerboard_nx6_locations_y10p2km"


def read_points(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return rows, np.array([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in rows]) / 1000


def main():
    section_y = 10.2
    em_dir = ROOT / "runs" / EM_RUNS[6]
    dd_dir = STUDY / "outputs/nx6-tomodd"
    meta = json.loads((em_dir / "meta.json").read_text())
    config = json.loads((dd_dir / "config.json").read_text())
    assert meta["source_experiment"]["id"] == config["source_experiment"]
    assert meta["run_params"]["coverage_damping_power"] == 0
    source = STUDY / "snapshot/projects/forward_modeling/experiments/input" / config["source_experiment"]
    for name, digest in config["source_forward_metadata"]["input_sha256"].items():
        assert sha(source / name) == digest
    event_rows, truth = read_points(source / "events.csv")
    _, stations = read_points(source / "stations.csv")
    selected = np.flatnonzero(np.isclose(truth[:, 1], section_y, atol=1e-10, rtol=0))
    assert len(truth) == 1200 and len(selected) == 120
    assert len(np.unique(truth[selected][:, [0, 2]], axis=0)) == len(selected)
    completed = sorted(int(p.parent.name.split("_")[1]) for p in em_dir.glob("iter_*/complete.json"))
    assert completed == list(range(12))
    latest = completed[-1]
    em_estimates, weights_hashes = [], {}
    for row_index in selected:
        path = em_dir / f"iter_{latest}/event_{row_index}/weights.npz"
        with np.load(path, allow_pickle=False) as weights:
            assert len(weights["weight_values"]) == 1 and np.isclose(weights["weight_values"][0], 1)
            position = (weights["positions"][0] + .5) * meta["grid_info"]["fine_cell_size"] / 1000
            em_estimates.append(position)
        weights_hashes[str(path)] = sha(path)
    em_estimates = np.array(em_estimates)
    with np.load(dd_dir / "state.npz", allow_pickle=False) as state:
        assert np.allclose(state["true_event_xyz_km"], truth, atol=1e-10, rtol=0)
        assert np.array_equal(state["true_events"][:, 0], np.arange(1, len(truth) + 1))
        dd_ids = state["final_events"][:, 0].astype(int)
        assert len(set(dd_ids)) == len(truth)
        mapping = {event_id: row for row, event_id in enumerate(dd_ids)}
        dd_geo = state["final_events"][[mapping[int(i + 1)] for i in selected]]
        center_lat, center_lon = config["coordinate_mapping"]["center_lat"], config["coordinate_mapping"]["center_lon"]
        dd_estimates = np.column_stack(((dd_geo[:, 2] - center_lon) * 111 * np.cos(np.deg2rad(center_lat)) + 12,
                                       (dd_geo[:, 1] - center_lat) * 111 + 6, dd_geo[:, 3]))
        assert np.array_equal(state["initial_events"][:, 0], state["true_events"][:, 0])
        dd_initial = state["initial_event_xyz_km"][selected].copy()
        # Validate the geographic-to-Cartesian convention against all saved true positions.
        true_geo = state["true_events"]
        true_xyz = np.column_stack(((true_geo[:, 2] - center_lon) * 111 * np.cos(np.deg2rad(center_lat)) + 12,
                                    (true_geo[:, 1] - center_lat) * 111 + 6, true_geo[:, 3]))
        assert np.allclose(true_xyz, truth, rtol=0, atol=1e-9)
        dd_grid = config["tomodd_grid"]
        axes = (np.array(dd_grid["depths_km"]), np.array(dd_grid["y_km"]) + 6,
                np.array(dd_grid["x_km"]) + 12)
        spacing = .025
        x = (np.arange(round(24 / spacing)) + .5) * spacing
        z = (np.arange(round(12 / spacing)) + .5) * spacing
        zz, xx = np.meshgrid(z, x, indexing="ij")
        slice_points = np.stack((zz, np.full_like(xx, section_y), xx), axis=-1)
        dd_background = RegularGridInterpolator(axes, state["final_vp"])(slice_points)
        initial_background = RegularGridInterpolator(axes, state["initial_vp"])(slice_points)
    em_model = np.load(em_dir / "final_model.npy", allow_pickle=False) / 1000
    em_background = em_model[np.floor(xx).astype(int), int(section_y), np.floor(zz).astype(int)]
    selected_truth = truth[selected]
    estimates = (em_estimates, dd_estimates, dd_initial)
    errors = [1000 * np.linalg.norm(points - selected_truth, axis=1) for points in estimates]
    error_max = max(1000, np.ceil(max(float(e.max()) for e in errors) / 1000) * 1000)
    velocity_cmap = LinearSegmentedColormap.from_list("velocity", [
        (0, "#7a7a7a"), (.375, "#ffffff"), (.625, "#e32400"), (1, "#61177c")], N=1024)
    rgba = velocity_cmap(np.linspace(0, 1, 1024))
    rgba[:, :3] = .32 * rgba[:, :3] + .68
    velocity_cmap = ListedColormap(rgba, name="muted_velocity")
    velocity_norm = Normalize(4, 6)
    assert all(4 <= a.min() <= a.max() <= 6 for a in (em_background, dd_background, initial_background))
    error_norm = Normalize(0, error_max)
    error_cmap = LinearSegmentedColormap.from_list(
        "location_error", ["#ffffff", "#0085ca", "#00239c"], N=1024)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30", "savefig.facecolor": "white",
    })
    fig, panels = plt.subplots(1, 3, figsize=(16.8, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.045, right=.985, bottom=.30, top=.83, wspace=.12)
    station_projection = np.unique(stations[:, [0, 2]], axis=0)
    for ax, background, locations, distance, title in zip(
            panels, (em_background, dd_background, initial_background), estimates, errors,
            ("(a)  EMTomo · final", "(b)  tomoDD · final", "(c)  tomoDD · initial")):
        ax.imshow(background, extent=(0, 24, 12, 0), origin="upper", aspect="equal",
                  cmap=velocity_cmap, norm=velocity_norm, interpolation="nearest", rasterized=True)
        segments = np.stack((selected_truth[:, [0, 2]], locations[:, [0, 2]]), axis=1)
        ax.add_collection(LineCollection(segments, colors="#555555", linewidths=.45, alpha=.8, zorder=3))
        ax.scatter(*locations[:, [0, 2]].T, c=distance, cmap=error_cmap, norm=error_norm,
                   s=19, marker="o", edgecolors="black", linewidths=.35, zorder=4)
        ax.scatter(*selected_truth[:, [0, 2]].T, s=28, marker="*", facecolors="white",
                   edgecolors="black", linewidths=.35, zorder=5)
        ax.scatter(*station_projection.T, s=112, marker=[(-.5, 0), (.5, 0), (0, 1), (-.5, 0)],
                   facecolors="#B1B1B1", edgecolors="none", linewidths=0, clip_on=False, zorder=6)
        ax.set(xlim=(0, 24), ylim=(12, 0), xlabel="$X$ (km)", xticks=np.arange(0, 25, 4),
               yticks=np.arange(0, 13, 4), title=title)
        ax.tick_params(direction="out", pad=3)
        map_frame(ax)
    panels[0].set_ylabel("Depth (km)")
    fig.suptitle("$Y = 10.2$ km", fontsize=11, y=.96)
    for rect, norm, cmap, label, ticks in (
            ([.105, .12, .30, .025], velocity_norm, velocity_cmap, "$V$ (km/s)", np.arange(4, 6.01, .5)),
            ([.59, .12, .30, .025], error_norm, error_cmap, "3D location error (m)", np.arange(0, error_max + 1, 1000))):
        cax = fig.add_axes(rect)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
        cax.set_title(label, fontsize=9, pad=7)
        bar.outline.set_linewidth(.45)
        cax.tick_params(labelsize=8, width=.45, length=2)
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    event_records = [{"event_id": event_rows[index]["event_id"], "csv_row_index": int(index),
                      "truth_xyz_km": selected_truth[j].tolist(),
                      "em_xyz_km": em_estimates[j].tolist(), "dd_xyz_km": dd_estimates[j].tolist(),
                      "dd_initial_xyz_km": dd_initial[j].tolist(),
                      "em_error_3d_m": float(errors[0][j]), "dd_error_3d_m": float(errors[1][j]),
                      "dd_initial_error_3d_m": float(errors[2][j])}
                     for j, index in enumerate(selected)]
    provenance = {
        "section_y_km": section_y, "selected_events": len(selected),
        "selection": "True Y=10.2 km, no filtering on estimated locations; all selected IDs shown in all three panels",
        "em_run": str(em_dir), "em_velocity_updates": 12, "em_location_iteration": latest,
        "em_location_caveat": "Latest saved E-step is before the final M-step; no additional relocation was run",
        "dd_run": str(dd_dir), "dd_iterations": 3,
        "display": "X-depth projection of 3D positions; circles coloured by full Euclidean 3D error, not projected distance or weights",
        "station_projection": "All 98 stations, coincident X-depth positions drawn once (14 triangles)",
        "background_style": "Velocity RGB colours mixed with 68% white; velocity colorbar uses the identical muted palette",
        "error_scale_m": [0, error_max],
        "error_color_anchors_m": {"#ffffff": 0, "#0085ca": error_max / 2, "#00239c": error_max},
        "errors": {name: {"rms_m": float(np.sqrt(np.mean(e**2))), "median_m": float(np.median(e)), "max_m": float(e.max())}
                   for name, e in zip(("em", "dd", "dd_initial"), errors)},
        "events": event_records,
        "sha256": {**weights_hashes, **{str(p): sha(p) for p in (
            source / "events.csv", source / "stations.csv", em_dir / "meta.json", em_dir / "final_model.npy",
            em_dir / f"iter_{latest}/complete.json", dd_dir / "config.json", dd_dir / "state.npz")}},
    }
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"selected_events": len(selected), "error_scale_m": [0, error_max], "errors": provenance["errors"]}, indent=2))


if __name__ == "__main__":
    main()
