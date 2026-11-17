"""Unmasked final Hard/Soft XZ slices with catalog-to-final-MAP event links.

Run from projects/EMTomo:
    python -B paper/figures/results/real_picks/source/make_hypocentre_sections.py
Only catalog positions select the common +/-1 km slab. No relocation is run.
"""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
sys.path.insert(0, str(ROOT / "paper/paper_scripts"))
from make_geysers_real_location_shifts import load_inputs, DEFAULT_INPUT, QUEUE, sha
from make_geysers_real_published_comparison import sample_cells
from make_geysers_published_vp_sections import section_points

STEM = "real_picks_hard_soft_hypocentres_xz"
MODES = ("hard", "soft")
PLANE_Y_M = 5500.
HALF_WIDTH_M = 1000.


def select_events(catalog):
    return np.abs(catalog[:, 1] - PLANE_Y_M) <= HALF_WIDTH_M


def draw(fields, edges, starts, ends, geometry, limits, side_by_side=False):
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    fig, axes = plt.subplots(1 if side_by_side else 2, 2 if side_by_side else 1,
                             figsize=(16, 5.7) if side_by_side else (13, 10.3))
    fig.subplots_adjust(left=.065, right=.91, top=.77 if side_by_side else .85,
                        bottom=.29 if side_by_side else .20,
                        hspace=.32, wspace=.10)
    for ax, mode in zip(axes.flat, MODES):
        image = ax.pcolormesh(*edges, fields[mode], cmap="RdBu_r", norm=Normalize(*limits),
                              shading="flat", rasterized=True)
        segments = np.stack((starts[:, [0, 2]], ends[mode][:, [0, 2]]), axis=1) / 1000
        ax.add_collection(LineCollection(segments, colors="black", linewidths=.45,
                                        alpha=.45, rasterized=True, zorder=3))
        ax.scatter(starts[:, 0] / 1000, starts[:, 2] / 1000, s=12,
                   facecolors="white", edgecolors="black", linewidths=.45,
                   alpha=.85, rasterized=True, zorder=4)
        ax.scatter(ends[mode][:, 0] / 1000, ends[mode][:, 2] / 1000, s=10,
                   facecolors="#ffdf00", edgecolors="black", linewidths=.35,
                   rasterized=True, zorder=5)
        # Include every selected endpoint, even if a future catalog has out-of-box points.
        all_xz = np.concatenate((starts[:, [0, 2]], *[ends[m][:, [0, 2]] for m in MODES])) / 1000
        lower = np.minimum(all_xz.min(axis=0), [0, 0])
        upper = np.maximum(all_xz.max(axis=0), extent[[0, 2]])
        ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), aspect="equal",
               xlabel="X (km; azimuth 135°)", ylabel="Z below box top (km)",
               title=f"{mode.capitalize()} · final velocity and MAP hypocentres")
        ax.tick_params(labelsize=10)
    cax = fig.add_axes([.935, .33 if side_by_side else .27, .013, .35 if side_by_side else .51])
    fig.colorbar(image, cax=cax, label="Vp (km/s) · common scale")
    fig.suptitle("Geysers real picks · Catalog → final MAP hypocentres", fontsize=17, y=.97)
    fig.text(.5, .88 if side_by_side else .915,
             f"XZ at Y = {PLANE_Y_M / 1000:g} km · Catalog slab ±{HALF_WIDTH_M / 1000:g} km · "
             f"Same {len(starts)} events in Hard and Soft", ha="center", fontsize=12)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black",
               markersize=5, label="Catalog position"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#ffdf00", markeredgecolor="black",
               markersize=5, label="Final-model MAP position"),
        Line2D([], [], color="black", linewidth=.7, label="Same-event displacement (XZ projection)"),
    ], loc="lower center", bbox_to_anchor=(.49, .16 if side_by_side else .105), ncol=3, frameon=False)
    fig.text(.065, .035,
             "Full velocity colors: no sensitivity/resolution mask or fading. Native 500 m cells, no smoothing.\n"
             "Events selected by catalog Y only; all fitted endpoints retained, including those outside the slab. Lines are not rays.\n"
             "Soft: most probable (MAP) position, not posterior mean. Catalog is not ground truth; shifts are not location errors.",
             fontsize=9, color="0.3", linespacing=1.5)
    stem = STEM + ("_side_by_side" if side_by_side else "")
    fig.savefig(PNG / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    positions, catalog, _, _, geometry, provenance = load_inputs(DEFAULT_INPUT.resolve())
    selected = select_events(catalog)
    if not selected.any():
        raise ValueError("No catalog events in the selected slab")
    ids = np.asarray(provenance["event_ids"])[selected]
    starts = catalog[selected]
    ends = {mode: positions[mode][selected] for mode in MODES}
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    edges, _, points = section_points(extent, 0, 2, 1)
    points[..., 1] = PLANE_Y_M / 1000
    fields, model_sources = {}, {}
    hashes = dict(provenance["source_sha256"])
    for mode in MODES:
        result_path = QUEUE / "cases" / mode / "result.json"
        record = json.loads(result_path.read_text())
        model_path = Path(record["run_dir"]) / "final_model.npy"
        digest = sha(model_path)
        if digest != record["artifact_sha256"]["final_model.npy"] or hashes[str(model_path.relative_to(ROOT))] != digest:
            raise ValueError("Velocity field differs from the verified final-MAP model")
        model = np.load(model_path, allow_pickle=False)
        cell_sizes = extent / np.asarray(model.shape)
        np.testing.assert_allclose(cell_sizes, .5, rtol=0, atol=1e-12)
        if not np.isfinite(model).all() or np.any(model <= 0):
            raise ValueError("Invalid final velocity model")
        fields[mode] = sample_cells(model, points, cell_sizes[0])
        model_sources[mode] = str(model_path.relative_to(ROOT))
    values = np.concatenate([v.ravel() for v in fields.values()])
    limits = [float(np.floor(values.min() * 2) / 2), float(np.ceil(values.max() * 2) / 2)]
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    # Do not publish a plot if any input changed after validation/loading.
    for path, digest in hashes.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Input changed during plotting preparation: {path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    PNG.mkdir(parents=True, exist_ok=True)
    draw(fields, edges, starts, ends, geometry, limits)
    draw(fields, edges, starts, ends, geometry, limits, side_by_side=True)
    arrays = dict(event_ids=ids, catalog_xyz_m=starts, points_xyz_km=points,
                  x_edges_km=edges[0], z_edges_km=edges[1], selected_event_mask=selected)
    for mode in MODES:
        arrays[mode + "_map_xyz_m"] = ends[mode]
        arrays[mode + "_vp_km_s"] = fields[mode]
        arrays[mode + "_segments_xz_km"] = np.stack((starts[:, [0, 2]], ends[mode][:, [0, 2]]), axis=1) / 1000
    data_path = HERE / f"{STEM}.npz"
    np.savez_compressed(data_path, **arrays)
    csv_path = HERE / f"{STEM}.csv"
    columns = ["event_id"] + [f"{name}_{axis}_m" for name in ("catalog", *MODES) for axis in "xyz"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, event_id in enumerate(ids):
            row = {"event_id": event_id}
            for name, values in {"catalog": starts, **ends}.items():
                row.update({f"{name}_{axis}_m": value for axis, value in zip("xyz", values[i])})
            writer.writerow(row)
    report = dict(
        plane="XZ", y_m=PLANE_Y_M, slab_half_width_m=HALF_WIDTH_M,
        selection="abs(catalog_y - plane_y) <= 1000 m; same IDs in both panels",
        selected_event_count=int(selected.sum()), total_event_count=len(selected),
        projection="Orthogonal XZ projection; exact unscaled catalog-to-MAP segments; no Y displacement shown",
        endpoint_policy="Keep every selected event's fitted endpoint, regardless of fitted Y",
        fitted_outside_slab_counts={mode: int(np.sum(np.abs(ends[mode][:, 1] - PLANE_Y_M) > HALF_WIDTH_M)) for mode in MODES},
        fitting="Saved TRAIN-only 50-candidate MAP relocation in each verified final velocity model",
        soft_location="MAP, not posterior mean; not pre-final-update iteration weights",
        source_input_dir=str(DEFAULT_INPUT.relative_to(ROOT)), model_sources=model_sources,
        resolution_mask=False, sensitivity_mask=False, velocity_fading=False,
        events_subsampled=False, smoothing=False, stations_plotted=False, no_solver_calls=True,
        color_limits_km_s=limits, geometry=geometry,
        caveats=["Catalog is a diagnostic baseline, not truth; shifts are not errors",
                 "Selection is based on catalog, not final fitted proximity to plane",
                 "Catalog sea-level datum compatibility is assumed"],
        source_sha256=hashes, data_sha256=sha(data_path), csv_sha256=sha(csv_path),
    )
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("selected_event_count", "fitted_outside_slab_counts", "color_limits_km_s")}, indent=2))
    print(PNG / f"{STEM}.png")


if __name__ == "__main__":
    main()
