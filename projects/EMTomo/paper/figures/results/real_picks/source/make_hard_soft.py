"""Hard/Soft only, with a raw intersection of heuristic native sensitivity.

This is NOT a resolution/recovery test. No published reference is loaded.
Run from projects/EMTomo:
    python -B paper/figures/results/real_picks/source/make_hard_soft.py
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
SCRIPTS = ROOT / "paper/paper_scripts"
sys.path.insert(0, str(SCRIPTS))
from make_geysers_real_published_comparison import completed_model, sample_cells, sha
from make_geysers_published_vp_sections import GEOMETRY, section_points, SPACING_KM


QUEUE = ROOT / "runs/geysers_real_refine500_fmm125_20261110_110138"
MODES = ("hard", "soft")
SEA_DEPTH_KM = 1.6
THRESHOLD = 1.0

STEM = "real_picks_hard_soft_sensitivity"


def load_models():
    geometry = json.loads(GEOMETRY.read_text())
    manifest_path = QUEUE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models, ratios, records = {}, {}, {}
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
        GEOMETRY, manifest_path, Path(__file__).resolve(),
    )}
    for mode in MODES:
        case = next(c for c in manifest["cases"] if c["id"] == mode)
        start = QUEUE / case["initial_model_file"]
        parent = QUEUE / case["parent_model_file"]
        np.testing.assert_array_equal(np.load(start), np.load(parent).repeat(2, 0).repeat(2, 1).repeat(2, 2))
        model, provenance, cell_km = completed_model(QUEUE, mode, start, geometry)
        result_path = QUEUE / "cases" / mode / "result.json"
        result = json.loads(result_path.read_text())
        run = Path(result["run_dir"])
        meta = json.loads((run / "meta.json").read_text())
        config = meta["run_params"]
        if config["candidate_mode"] != mode or config["coverage_damping_power"] != 0:
            raise ValueError("Expected matching mode and scalar damping")
        iteration = provenance["completed_updates"] - 1
        directory = run / f"iter_{iteration}"
        h_path = directory / "sensitivity_diagonal.npy"
        confidence_path = directory / "coverage_confidence.npy"
        state_path = directory / "model.npy"
        sensitivity = np.load(h_path, allow_pickle=False)
        if sensitivity.shape != model.shape or not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError("Invalid native sensitivity")
        positive = sensitivity[sensitivity > 0]
        if not positive.size:
            raise ValueError("No sensitivity in inversion")
        reference_scale = np.percentile(positive, config["coverage_reference_percentile"])
        np.testing.assert_allclose(np.load(confidence_path), np.clip(sensitivity / reference_scale, 0, 1))
        damping = config["lambda_reg"] * sensitivity.mean()
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("Expected positive scalar damping")
        state = np.load(state_path, allow_pickle=False)
        if state.shape != model.shape or not np.isfinite(state).all() or np.any(state <= 0):
            raise ValueError("Invalid pre-update diagnostic model")
        models[mode] = (model, cell_km)
        ratios[mode] = sensitivity / damping
        records[mode] = dict(
            final_model=str((run / "final_model.npy").relative_to(ROOT)),
            diagnostic_model=str(state_path.relative_to(ROOT)), diagnostic_iteration=iteration,
            sensitivity=str(h_path.relative_to(ROOT)), damping=float(damping),
            diagnostic_timing="Before last update, NOT recomputed in final model",
            max_final_relative_change=float(np.max(np.abs(model / state - 1))),
            cell_size_m=cell_km * 1000, completed_updates=provenance["completed_updates"],
        )
        for path, digest in provenance["source_sha256"].items():
            hashes[str(Path(path).relative_to(ROOT))] = digest
        for path in (h_path, confidence_path, state_path, start, parent):
            hashes[str(path.relative_to(ROOT))] = sha(path)
    np.testing.assert_allclose(models["hard"][1], models["soft"][1])
    return geometry, models, ratios, records, hashes


def prepare_sections(geometry, models, ratios):
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    horizontal_z = SEA_DEPTH_KM + geometry["top_elevation_m"] / 1000
    sections = []
    for name, h, v, fixed, coordinate in [("xy", 0, 1, 2, horizontal_z),
                                         ("xz", 0, 2, 1, extent[1] / 2)]:
        edges, centres, points = section_points(extent, h, v, fixed)
        points[..., fixed] = coordinate
        fields, scores = {}, {}
        for mode in MODES:
            model, cell_km = models[mode]
            fields[mode] = sample_cells(model, points, cell_km)
            # Diagnostics are dimensionless, so index native cells directly (no velocity conversion).
            indices = np.floor(points / cell_km).astype(int)
            scores[mode] = ratios[mode][tuple(np.moveaxis(indices, -1, 0))]
        mask = (scores["hard"] >= THRESHOLD) & (scores["soft"] >= THRESHOLD)
        if not mask.any() or mask.all():
            raise ValueError(f"Expected a nonempty partial sensitivity mask in {name}")
        sections.append(dict(name=name, h=h, v=v, fixed=fixed, coordinate=float(coordinate),
                             edges=edges, centres=centres, points=points,
                             fields=fields, scores=scores, mask=mask))
    return sections


def draw(sections, geometry, limits, stem):
    extent = np.asarray(geometry["domain_extent_m"]) / 1000
    combined = len(sections) == 2
    fig, axes = plt.subplots(len(sections), 2, squeeze=False,
                             figsize=(12, 9.2 if combined else 6),
                             gridspec_kw={"height_ratios": [extent[s["v"]] for s in sections]})
    fig.subplots_adjust(left=.075, right=.9, top=.79 if combined else .73,
                        bottom=.23 if combined else .32, hspace=.45, wspace=.12)
    labels = ["X (km; azimuth 135°)", "Y (km; azimuth 225°)", "Z below box top (km)"]
    for row, s in enumerate(sections):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            image = ax.pcolormesh(*s["edges"], s["fields"][mode], cmap="RdBu_r",
                                  norm=Normalize(*limits), shading="flat", rasterized=True)
            ax.pcolormesh(*s["edges"], np.ma.masked_where(s["mask"], np.ones(s["mask"].shape)),
                          cmap=ListedColormap(["white"]), vmin=0, vmax=1,
                          alpha=.82, shading="flat", rasterized=True)
            for color, width in [("white", 2.6), ("black", 1.1)]:
                ax.contour(*s["centres"], s["mask"].astype(float), levels=[.5],
                           colors=[color], linewidths=width)
            trace = extent[1] / 2 if s["name"] == "xy" else SEA_DEPTH_KM + geometry["top_elevation_m"] / 1000
            ax.axhline(trace, color="0.3", linestyle="--", linewidth=.8)
            ax.set(xlim=(0, extent[s["h"]]), ylim=(0, extent[s["v"]]),
                   xlabel=labels[s["h"]], title=mode.capitalize(), aspect="equal")
            if col == 0:
                ax.set_ylabel(labels[s["v"]])
            else:
                ax.tick_params(labelleft=False)
            if s["v"] == 2:
                ax.invert_yaxis()
        caption = (f'(a) Horizontal XY · depth {SEA_DEPTH_KM:g} km below sea level · Z = {s["coordinate"]:.3f} km'
                   if s["name"] == "xy" else f'(b) Longitudinal XZ · Y = {s["coordinate"]:g} km')
        axes[row, 0].text(0, 1.30, caption, transform=axes[row, 0].transAxes, fontsize=11)
    cax = fig.add_axes([.925, .33, .014, .38 if combined else .27])
    fig.colorbar(image, cax=cax, label="Vp (km/s) · common scale")
    fig.suptitle("Geysers real picks · Hard / Soft", fontsize=17, y=.97)
    fig.text(.5, .91 if combined else .88,
             "Shared raw native-sensitivity intersection · NOT spatial resolution",
             ha="center", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color="black", label=f"Raw intersection: Hjj / α ≥ {THRESHOLD:g} in BOTH runs"),
        Patch(facecolor="0.94", edgecolor="0.7", label="Faded: outside shared raw intersection"),
    ], loc="lower center", bbox_to_anchor=(.49, .13 if combined else .17), ncol=1, frameon=False)
    fig.text(.075, .035,
             f"Raw: Hjj / α ≥ {THRESHOLD:g} in BOTH runs; Hjj: diagonal data sensitivity; α = λ mean(Hjj), actual damping.\n"
             "Heuristic threshold, NOT spatial resolution. No hull, padding, hole filling or component removal.\n"
             "Final velocities shown; diagnostics are from before the last update (up to 2% velocity change).\n"
             "500 m velocity cells; no velocity smoothing, reference-model mask or events. Diagonal sensitivity is not recovery.",
             fontsize=9, linespacing=1.5, color="0.3")
    fig.savefig(PNG / f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    geometry, models, ratios, provenance, hashes = load_models()
    sections = prepare_sections(geometry, models, ratios)
    values = np.concatenate([v.ravel() for s in sections for v in s["fields"].values()])
    limits = [float(np.floor(values.min() * 2) / 2), float(np.ceil(values.max() * 2) / 2)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    PNG.mkdir(parents=True, exist_ok=True)
    draw(sections, geometry, limits, STEM)
    for section in sections:
        draw([section], geometry, limits, STEM + "_" + section["name"])
    arrays = {mode + "_sensitivity_over_damping_xyz": ratios[mode] for mode in MODES}
    records = []
    for s in sections:
        p = s["name"] + "_"
        np.testing.assert_array_equal(
            s["mask"], (s["scores"]["hard"] >= THRESHOLD) & (s["scores"]["soft"] >= THRESHOLD))
        arrays.update({p + "shared_mask": s["mask"], p + "points_xyz_km": s["points"],
                       p + "horizontal_edges_km": s["edges"][0], p + "vertical_edges_km": s["edges"][1]})
        for mode in MODES:
            arrays[p + mode + "_vp_km_s"] = s["fields"][mode]
            arrays[p + mode + "_sensitivity_over_damping"] = s["scores"][mode]
            arrays[p + mode + "_mask"] = s["scores"][mode] >= THRESHOLD
        areas = {str(t): float(((s["scores"]["hard"] >= t) & (s["scores"]["soft"] >= t)).sum() * SPACING_KM**2)
                 for t in (.5, 1., 2.)}
        records.append(dict(plane=s["name"].upper(), fixed_axis="XYZ"[s["fixed"]],
                            fixed_coordinate_km=s["coordinate"],
                            shared_area_km2=float(s["mask"].sum() * SPACING_KM**2),
                            shared_pixel_count=int(s["mask"].sum()),
                            shared_mask_is_exact_intersection=True,
                            threshold_sensitivity_shared_area_km2=areas))
    data_path = HERE / f"{STEM}.npz"
    np.savez_compressed(data_path, **arrays)
    report = dict(models=provenance, geometry=geometry, sections=records,
                  threshold=THRESHOLD,
                  mask_policy=f"Raw intersection: Hjj / actual scalar damping >= {THRESHOLD:g} in BOTH runs; "
                              "no hull, padding, hole filling or component removal",
                  mask_array_roles=dict(shared_mask="Exact raw intersection: hard_mask AND soft_mask; "
                                                    "same mask drawn for Hard and Soft",
                                        hard_mask="Hard-only raw threshold selection",
                                        soft_mask="Soft-only raw threshold selection"),
                  interpretation="Heuristic diagonal data-sensitivity selection, "
                                 "NOT spatial resolution, recovery or probability",
                  sensitivity_limitations="Diagonal Hjj does not capture inter-cell correlations, "
                                          "velocity-hypocentre trade-offs or the nonlinear EM trajectory",

                  diagnostic_timing="Last pre-update state; not final-model diagnostics",
                  horizontal_sea_depth_km=SEA_DEPTH_KM,
                  section_policy="Keep existing XY/XZ coordinates for comparability; no new depth optimization",
                  reference_used=False, events_plotted=False, smoothing=False,
                  mask_convex_hull=False, mask_padding=False, mask_hole_filling=False,
                  color_limits_km_s=limits, plot_spacing_km=SPACING_KM,
                  data_sha256=sha(data_path), source_sha256=hashes)
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print(PNG / f"{STEM}.png")


if __name__ == "__main__":
    main()
