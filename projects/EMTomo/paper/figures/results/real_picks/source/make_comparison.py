"""Real-pick Hard / Soft / Reference sections, with one reference-derived mask.

Run from projects/EMTomo:
    python -B paper/figures/results/real_picks/source/make_comparison.py
No solvers, event overlays, model smoothing or reference extrapolation.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
SCRIPTS = ROOT / "paper/paper_scripts"
sys.path.insert(0, str(SCRIPTS))
from make_geysers_real_structure_metrics import load_inputs, published_mask, THRESHOLD, QUEUE
from make_geysers_real_published_comparison import sample_cells, sha
from make_geysers_published_vp_sections import section_points, SPACING_KM

HORIZONTAL_SEA_DEPTH_KM = 1.6
MODES = ("hard", "soft", "reference")
TITLES = ("Hard", "Soft", "Reference · Guo & Thurber (2021)")


def prepare_sections(models, reference, geometry):
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    horizontal_z = HORIZONTAL_SEA_DEPTH_KM + geometry["top_elevation_m"] / 1000
    if not 0 < horizontal_z < extent[2]:
        raise ValueError("Horizontal plane lies outside the model")
    sections = []
    for name, horizontal, vertical, fixed, coordinate in [
        ("xy", 0, 1, 2, horizontal_z),
        ("xz", 0, 2, 1, extent[1] / 2),
    ]:
        edges, centres, points = section_points(extent, horizontal, vertical, fixed)
        points[..., fixed] = coordinate
        article = reference.sample(points)
        mask = published_mask(article)
        if not mask.any() or mask.all():
            raise ValueError(f"Expected a nonempty, partial reference mask in {name}")
        arrays = {mode: sample_cells(models[mode][0], points, models[mode][1])
                  for mode in MODES[:2]}
        arrays["reference"] = article[..., 0]
        for values in arrays.values():
            if not np.isfinite(values[mask]).all():
                raise ValueError("A model has missing values inside the shared mask")
        sections.append(dict(name=name, horizontal=horizontal, vertical=vertical,
                             fixed=fixed, coordinate_km=float(coordinate),
                             edges=edges, centres=centres, points=points,
                             values=arrays, mask=mask, resolvability=article[..., 1]))
    return sections


def draw(sections, geometry, limits, stem, clean=False):
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    norm = Normalize(*limits)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.93")
    combined = len(sections) == 2
    fig, axes = plt.subplots(
        len(sections), 3, squeeze=False, figsize=(14.5, 8.8 if combined else 5.5),
        gridspec_kw={"height_ratios": [extent[s["vertical"]] for s in sections]},
    )
    fig.subplots_adjust(left=.055, right=.92, top=.82 if combined else .74,
                        bottom=.19 if combined else .28, wspace=.12, hspace=.44)
    labels = ["X (km; azimuth 135°)", "Y (km; azimuth 225°)", "Z below box top (km)"]
    for row, section in enumerate(sections):
        mask, edges, centres = section["mask"], section["edges"], section["centres"]
        for col, (mode, title) in enumerate(zip(MODES, TITLES)):
            ax = axes[row, col]
            image = ax.pcolormesh(*edges, np.ma.masked_invalid(section["values"][mode]),
                                  cmap=cmap, norm=norm, shading="flat", rasterized=True)
            # The exact same reference-only mask controls all three panels.
            ax.pcolormesh(*edges, np.ma.masked_where(mask, np.ones(mask.shape)),
                          cmap=ListedColormap(["white"]), vmin=0, vmax=1,
                          shading="flat", alpha=.82, rasterized=True)
            for color, width in [("white", 2.6), ("black", 1.1)]:
                ax.contour(*centres, mask.astype(float), levels=[.5],
                           colors=[color], linewidths=width)
            if section["name"] == "xy":
                ax.axhline(extent[1] / 2, color="0.3", linestyle="--", linewidth=.8)
            else:
                ax.axhline(HORIZONTAL_SEA_DEPTH_KM + geometry["top_elevation_m"] / 1000,
                           color="0.3", linestyle="--", linewidth=.8)
            ax.set(xlim=(0, extent[section["horizontal"]]),
                   ylim=(0, extent[section["vertical"]]),
                   xlabel=labels[section["horizontal"]])
            if col == 0:
                ax.set_ylabel(labels[section["vertical"]])
            else:
                ax.tick_params(labelleft=False)
            if section["vertical"] == 2:
                ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.tick_params(labelsize=9, length=3)
            ax.set_title(title, fontsize=12, pad=8)
        caption = (f"(a) Horizontal XY · depth {HORIZONTAL_SEA_DEPTH_KM:g} km below sea level · "
                           f'Z = {section["coordinate_km"]:.3f} km below box top'
                   if section["name"] == "xy" else
                   f'(b) Longitudinal XZ · Y = {section["coordinate_km"]:g} km')
        axes[row, 0].text(0, 1.30 if combined else 1.32, caption,
                          transform=axes[row, 0].transAxes, fontsize=11, ha="left")
    cax = fig.add_axes([.94, .27 if combined else .34, .012, .47 if combined else .32])
    fig.colorbar(image, cax=cax, label="Vp (km/s) · common scale")
    fig.suptitle("Geysers real picks · Hard / Soft / Reference", fontsize=17, y=.97)
    fig.text(.5, .915 if combined else .89, "Final 500 m inversion cells · One reference-derived comparison region · No events",
             ha="center", fontsize=11, color="0.25")
    fig.legend(handles=[
        Line2D([], [], color="black", linewidth=1.2, label="Reference well-resolved region: R > 0.4"),
        Patch(facecolor="0.94", edgecolor="0.7", label="Faded: outside the shared region"),
        Line2D([], [], color="0.3", linestyle="--", linewidth=.8, label="Other section's trace"),
    ], loc="lower center", bbox_to_anchor=(.49, .092 if combined else .13),
        ncol=3, frameon=False, fontsize=10)
    fig.text(.055, .032 if combined else .035,
             "R: published scaled resolvability, interpolated within reference coverage; the identical mask is applied to all models.\n"
             "The mask does not establish Hard/Soft resolution. Reference is not ground truth. Hard/Soft cells are not smoothed.\n"
             "Horizontal view is a constant-elevation plane, not terrain-following. 100 m plot sampling is not tomographic resolution.",
             fontsize=9, color="0.3", linespacing=1.5)
    if clean:
        for text in list(fig.texts):
            text.remove()
        for legend in list(fig.legends):
            legend.remove()
        fig.subplots_adjust(top=.88, bottom=.08)
        cax.set_position([.94, .18, .012, .58])
    fig.savefig(PNG / f"{stem}.png", dpi=200, facecolor="white", bbox_inches="tight" if clean else None)
    fig.savefig(HERE / f"{stem}.pdf", dpi=200, facecolor="white", bbox_inches="tight" if clean else None)
    plt.close(fig)


def main():
    models, reference, geometry, provenance, hashes = load_inputs()
    sections = prepare_sections(models, reference, geometry)
    finite = np.concatenate([values[np.isfinite(values)]
                             for section in sections for values in section["values"].values()])
    limits = [float(np.floor(finite.min() * 2) / 2), float(np.ceil(finite.max() * 2) / 2)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.linewidth": .6})
    PNG.mkdir(parents=True, exist_ok=True)
    draw(sections, geometry, limits, "real_picks_comparison")
    draw(sections, geometry, limits, "real_picks_comparison_final", clean=True)
    for section in sections:
        draw([section], geometry, limits, "real_picks_" + section["name"])
    exports, records = {}, []
    for section in sections:
        prefix = section["name"] + "_"
        exports.update({prefix + mode + "_vp_km_s": values
                        for mode, values in section["values"].items()})
        exports.update({prefix + "shared_mask": section["mask"],
                        prefix + "reference_resolvability": section["resolvability"],
                        prefix + "points_xyz_km": section["points"],
                        prefix + "horizontal_edges_km": section["edges"][0],
                        prefix + "vertical_edges_km": section["edges"][1]})
        mask = section["mask"]
        records.append(dict(
            plane=section["name"].upper(), fixed_axis="XYZ"[section["fixed"]],
            fixed_coordinate_km=section["coordinate_km"], shared_mask_samples=int(mask.sum()),
            total_samples=int(mask.size), shared_area_km2=float(mask.sum() * SPACING_KM**2),
            shared_mask_sha256=hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest(),
        ))
    data_path = HERE / "real_picks_sections.npz"
    np.savez_compressed(data_path, **exports)
    for path in (Path(__file__), ROOT / "data/geysers_2021/Readme"):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    report = dict(
        queue=str(QUEUE.relative_to(ROOT)), models={mode: provenance[mode] for mode in MODES[:2]},
        reference_doi="10.1093/gji/ggab017", truth_available=False, events_plotted=False,
        geometry=geometry, sections=records, color_limits_km_s=limits,
        plot_spacing_km=SPACING_KM, horizontal_sea_depth_km=HORIZONTAL_SEA_DEPTH_KM,
                horizontal_selection="Largest reference-mask area in a 100 m full-domain depth scan, refined at 10 m near the peak; selected published depth 1.6 km", 
        shared_mask_policy="finite(reference Vp and R) AND R > threshold; reference-only, identical for all models",
        threshold=THRESHOLD, mask_source="data/geysers_2021/Readme, note 2",
        excluded_display="82% white overlay; reference outside coverage is missing, never extrapolated",
        contour_policy="0.5 isoline of the shared Boolean mask on 100 m plot centres",
        reference_interpolation="Linear horizontal triangles and vertical segments, no extrapolation",
        inversion_sampling="Piecewise-constant native 500 m XYZ cells, no smoothing",
        caveats=["Reference resolution mask does not assess hard/soft resolution or accuracy",
                 "Horizontal plane is not depth below terrain",
                 "Vertical-datum compatibility is assumed, not independently established",
                 "Plot sampling does not imply 100 m resolution"],
        checks_passed=["Existing loader verified final and parent hashes and completion markers",
                       "Parent-to-refined initial cells reproduced exactly",
                       "Reference source nodes, outside-support NaNs and geographic mapping validated",
                       "Both sections have nonempty shared masks and finite Vp for every model within them"],
        sections_data_sha256=sha(data_path), source_sha256=hashes,
    )
    (HERE / "real_picks_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print(PNG / "real_picks_comparison.png")


if __name__ == "__main__":
    main()
