"""All-depth XZ event projection, at most three catalog events per 500 m square.

Run from projects/EMTomo:
    python -B paper/figures/results/real_picks/source/make_hypocentre_projection.py
No velocity background, plane/slab filter, jitter, relocation or magnitude-based selection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
sys.path.insert(0, str(ROOT / "paper/paper_scripts"))
from make_geysers_real_location_shifts import load_inputs, DEFAULT_INPUT, sha

STEM = "real_picks_hard_soft_hypocentres_projection"
CELL_M = 500.
MAX_PER_CELL = 3
MODES = ("hard", "soft")


def select_events(catalog, event_ids, cell_m=CELL_M, max_per_cell=MAX_PER_CELL):
    """Cap projected catalog cells; stable hash selection independent of fitted shifts."""
    cells = np.floor(catalog[:, [0, 2]] / cell_m).astype(int)
    buckets = {}
    for i, (cell, event_id) in enumerate(zip(cells, event_ids)):
        rank = hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()
        buckets.setdefault(tuple(cell), []).append((rank, str(event_id), i))
    selected = np.zeros(len(catalog), dtype=bool)
    for candidates in buckets.values():
        for _, _, i in sorted(candidates)[:max_per_cell]:
            selected[i] = True
    return selected, cells


def draw(catalog, positions, geometry, side_by_side=False, *, cell_m=CELL_M,
         max_per_cell=MAX_PER_CELL, stem=STEM):
    fig, axes = plt.subplots(1 if side_by_side else 2, 2 if side_by_side else 1,
                             figsize=(16, 6.2) if side_by_side else (13, 10.5))
    fig.subplots_adjust(left=.07, right=.98, top=.76 if side_by_side else .84,
                        bottom=.26 if side_by_side else .18, hspace=.32, wspace=.10)
    extent = np.asarray(geometry["domain_extent_m"])[[0, 2]] / 1000
    projected = [catalog[:, [0, 2]] / 1000] + [positions[m][:, [0, 2]] / 1000 for m in MODES]
    points = np.concatenate(projected)
    lower = np.minimum(points.min(axis=0), [0, 0])
    upper = np.maximum(points.max(axis=0), extent)
    padding = (upper - lower) * .015
    for ax, mode in zip(axes.flat, MODES):
        start, end = projected[0], positions[mode][:, [0, 2]] / 1000
        ax.add_collection(LineCollection(np.stack((start, end), axis=1), colors="0.35",
                                        linewidths=.65, alpha=.55, zorder=2))
        ax.scatter(*start.T, s=18, facecolors="white", edgecolors="0.35", linewidths=.65,
                   zorder=3)
        ax.scatter(*end.T, s=18, facecolors="#e69f00", edgecolors="black", linewidths=.45,
                   zorder=4)
        ax.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]),
               ylim=(upper[1] + padding[1], lower[1] - padding[1]), aspect="equal",
               xlabel="X (km; azimuth 135°)", ylabel="Z below box top (km)",
               title=f"{mode.capitalize()} · final MAP hypocentres")
        ax.xaxis.set_minor_locator(MultipleLocator(cell_m / 1000))
        ax.yaxis.set_minor_locator(MultipleLocator(cell_m / 1000))
        ax.grid(which="minor", color="0.9", linewidth=.4)
        ax.grid(which="major", color="0.85", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle("Geysers real picks · Catalog → final MAP hypocentres", fontsize=17, y=.97)
    fig.text(.5, .89 if side_by_side else .91,
             f"All Y positions projected onto XZ · {len(catalog)} shared events · "
             f"At most {max_per_cell} catalog events per {cell_m:g} × {cell_m:g} m square", ha="center", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="0.35", label="Catalog"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#e69f00", markeredgecolor="black", label="Final MAP"),
        Line2D([], [], color="0.35", linewidth=.8, label="Same-event displacement (XZ projection)"),
    ], loc="lower center", bbox_to_anchor=(.51, .12 if side_by_side else .09), ncol=3, frameon=False)
    fig.text(.07, .025,
             "Selection: projected catalog cells, stable event-ID hash; same events in Hard/Soft, no selection by displacement.\n"
             "No Y/slab restriction, velocity background or jitter. Fitted endpoints are not cell-limited; lines may cross.\n"
             "Soft: most probable position (MAP), not posterior mean. Catalog is not truth; shifts are not location errors.",
             fontsize=9, color="0.3", linespacing=1.5)
    stem = stem + ("_side_by_side" if side_by_side else "")
    fig.savefig(PNG / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-m", type=float, default=CELL_M)
    parser.add_argument("--max-per-cell", type=int, default=MAX_PER_CELL)
    args = parser.parse_args()
    if not np.isfinite(args.cell_m) or args.cell_m <= 0 or args.max_per_cell < 1:
        parser.error("Cell size must be finite and positive; event cap must be at least one")
    cell_m, max_per_cell = args.cell_m, args.max_per_cell
    stem = STEM if (cell_m, max_per_cell) == (CELL_M, MAX_PER_CELL) else (
        f"{STEM}_{cell_m:g}m_max{max_per_cell}")
    positions, catalog, _, _, geometry, provenance = load_inputs(DEFAULT_INPUT.resolve())
    ids = np.asarray(provenance["event_ids"])
    selected, cells = select_events(catalog, ids, cell_m, max_per_cell)
    selected_ids = ids[selected]
    starts = catalog[selected]
    ends = {m: positions[m][selected] for m in MODES}
    _, counts = np.unique(cells[selected], axis=0, return_counts=True)
    if not len(counts) or counts.max() > max_per_cell:
        raise ValueError("Invalid projected-cell selection")
    hashes = dict(provenance["source_sha256"])
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    for path, digest in hashes.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Input changed during preparation: {path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    PNG.mkdir(parents=True, exist_ok=True)
    draw(starts, ends, geometry, cell_m=cell_m, max_per_cell=max_per_cell, stem=stem)
    draw(starts, ends, geometry, side_by_side=True,
         cell_m=cell_m, max_per_cell=max_per_cell, stem=stem)
    arrays = dict(event_ids=selected_ids, selected_event_mask=selected, catalog_xyz_m=starts,
                  catalog_cell_xz=cells[selected])
    for mode in MODES:
        arrays[mode + "_map_xyz_m"] = ends[mode]
        arrays[mode + "_segments_xz_km"] = np.stack((starts[:, [0, 2]], ends[mode][:, [0, 2]]), axis=1) / 1000
    data_path = HERE / f"{stem}.npz"
    np.savez_compressed(data_path, **arrays)
    csv_path = HERE / f"{stem}.csv"
    columns = ["event_id", "catalog_cell_x", "catalog_cell_z"] + [f"{m}_{a}_m" for m in ("catalog", *MODES) for a in "xyz"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, event_id in enumerate(selected_ids):
            row = dict(event_id=event_id, catalog_cell_x=int(cells[selected][i, 0]), catalog_cell_z=int(cells[selected][i, 1]))
            for name, values in {"catalog": starts, **ends}.items():
                row.update({f"{name}_{axis}_m": value for axis, value in zip("xyz", values[i])})
            writer.writerow(row)
    report = dict(projection="XZ, all Y; no plane distance selection", cell_size_m=cell_m,
                  max_catalog_events_per_projected_cell=max_per_cell, cell_origin_xz_m=[0, 0],
                  selection=f"floor(catalog XZ / {cell_m:g}), first {max_per_cell} per cell by ascending SHA256(UTF-8 event ID); original order retained",
                  total_event_count=len(catalog), selected_event_count=int(selected.sum()),
                  occupied_selected_cells=len(counts), actual_max_per_cell=int(counts.max()),
                  same_events_both_methods=True, fitted_endpoints_cell_limited=False,
                  soft_location="Saved final-model MAP, not posterior mean", velocity_background=False,
                  no_solver_calls=True, geometry=geometry, source_sha256=hashes,
                  data_sha256=sha(data_path), csv_sha256=sha(csv_path))
    (HERE / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected {selected.sum()} / {len(catalog)} events in {len(counts)} projected cells; max {counts.max()} per cell")
    print(PNG / f"{stem}.png")


if __name__ == "__main__":
    main()
