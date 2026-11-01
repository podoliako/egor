"""Compare three completed Geysers tomoDD runs at an exact requested depth.

Reads saved iteration-3 fields only; never launches an inversion.
Run from projects/EMTomo: python paper/make_geysers_tomodd_start_comparison.py
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from evaluate_metrics import native_helpers, sha256

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("/home/egor/tomodd_benchmark/runs")
PREFIX = "geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_tomodd-1km-layerstart-ray005-deg4-"
RUNS = (
    (0.0, BENCHMARK / f"{PREFIX}sigma0km-seed42-3it-coordfix-nolimit"),
    (0.5, BENCHMARK / f"{PREFIX}sigma0p5km-seed42-3it-coordfix-nolimit"),
    (1.0, ROOT / "runs/matched_geometry_20261021/outputs/geysers-tomodd-sigma1"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-km", type=float, default=1.5)
    parser.add_argument("--with-references", action="store_true",
                        help="Prepend the trilinear oracle and matched artificial-geometry sigma=1 run")
    args = parser.parse_args()
    if not np.isfinite(args.depth_km) or not 0 <= args.depth_km < 7:
        parser.error("Depth must be inside [0, 7) km")
    output = Path(__file__).resolve().parent / "figures"
    stem = "geysers_tomodd_sigma0_0p5_1_z" + f"{args.depth_km:g}".replace(".", "p") + "km"
    study = ROOT / "runs/matched_geometry_20261021"
    if args.with_references:
        stem += "_with_oracle_synthetic"
    manifest = json.loads((study / "manifest.json").read_text())
    snapshot = Path(manifest["snapshot_root"])
    helper = snapshot / "benchmark/tomodd_benchmark/compare_emtomo_flat4.py"
    assert sha256(helper) == manifest["snapshot_sha256"][str(helper.relative_to(snapshot))]
    native = native_helpers(helper)

    spacing = 0.125
    x = (np.arange(160) + 0.5) * spacing
    y = (np.arange(56) + 0.5) * spacing
    xx, yy = np.meshgrid(x, y)
    points = np.stack((np.full_like(xx, args.depth_km), yy, xx), axis=-1)
    fields, records, hashes = [], [], {str(helper): sha256(helper)}
    reference, oracle_reference = None, None
    runs = [("Geysers", sigma, run) for sigma, run in RUNS]
    if args.with_references:
        runs.insert(0, ("Artificial", 1.0, study / "outputs/synthetic-tomodd-sigma1"))
    for geometry, sigma, run in runs:
        config = json.loads((run / "config.json").read_text())
        assert config["solver"]["iteration_sets"][0][0] == 3
        assert config["solver"]["iteration_sets"][0][-2] == 1, "Expected joint inversion"
        grid = config["tomodd_grid"]
        axes = (np.asarray(grid["depths_km"]), np.asarray(grid["y_km"]) + 3.5,
                np.asarray(grid["x_km"]) + 10.0)
        nodes = native.read_velocity_history(run / "work/inversion/tomoDD.vel", (9, 9, 22), 3) / 1000
        with np.load(run / "state.npz", allow_pickle=False) as state:
            np.testing.assert_allclose(nodes, state["final_vp"], rtol=0, atol=1e-11)
            truth = state["physical_velocity_km_s"]
            if reference is not None:
                np.testing.assert_array_equal(truth, reference)
            reference = truth.copy()
            oracle = RegularGridInterpolator(axes, state["oracle_vp"], bounds_error=True)(points)
            if oracle_reference is not None:
                np.testing.assert_allclose(oracle, oracle_reference, rtol=0, atol=1e-9)
            oracle_reference = oracle
            # Truth is piecewise constant: at a cell boundary take the deeper cell.
            truth_slice = truth[:, :, int(np.floor(args.depth_km / spacing))].T
            field = RegularGridInterpolator(axes, nodes, bounds_error=True)(points)
            assert field.shape == truth_slice.shape and np.isfinite(field).all()
            fields.append(field)
        results = json.loads((run / "results.json").read_text())
        records.append({"kind": "inversion", "geometry": geometry,
                        "run": str(run), "initial_sigma_km_per_coordinate": sigma,
                        "slice_rmse_m_s": float(np.sqrt(np.mean((field - truth_slice)**2)) * 1000),
                        "slice_vp_range_km_s": [float(field.min()), float(field.max())],
                        "final_location_rms_m": results["final_hypocenter_rms_km"] * 1000})
        for name in ("config.json", "state.npz", "results.json", "work/inversion/tomoDD.vel"):
            hashes[str(run / name)] = sha256(run / name)

    if args.with_references:
        fields.insert(0, oracle_reference)
        records.insert(0, {"kind": "oracle", "geometry": "Geometry-independent",
                           "slice_rmse_m_s": float(np.sqrt(np.mean((oracle_reference - truth_slice)**2)) * 1000),
                           "slice_vp_range_km_s": [float(oracle_reference.min()), float(oracle_reference.max())]})

    low = float(np.floor(min(a.min() for a in fields + [truth_slice]) * 5) / 5)
    high = float(np.ceil(max(a.max() for a in fields + [truth_slice]) * 5) / 5)

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    count = len(fields)
    fig, axs = plt.subplots(count, 1, figsize=(10, 3.5 * count + 0.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09 if args.with_references else 0.14,
                        top=0.93 if args.with_references else 0.89, hspace=0.33)
    for ax, field, record in zip(axs, fields, records):
        image = ax.imshow(field, extent=(0, 20, 0, 7), origin="lower", aspect="equal",
                          interpolation="nearest", cmap="RdBu_r", vmin=low, vmax=high)

        if record["kind"] == "oracle":
            label = "tomoDD oracle: native trilinear representation"
        else:
            sigma = record["initial_sigma_km_per_coordinate"]
            label = f"{record['geometry']} geometry: $\\sigma_0={sigma:g}$ km per coordinate"
        ax.set_title(f"{label}  |  slice RMSE {record['slice_rmse_m_s']:.1f} m/s", pad=9)
        ax.set(xlim=(0, 20), ylim=(0, 7), ylabel="$Y$ (km)",
               xticks=np.arange(0, 21, 2), yticks=[0, 2, 4, 6, 7])
    axs[-1].set_xlabel("$X$ (km)")
    fig.suptitle(f"tomoDD · shared velocity model · depth $Z={args.depth_km:g}$ km\n"
                 "All inversions: iteration 3, hypocentres free", y=0.98 if args.with_references else 0.965, fontsize=14)
    color_ax = fig.add_axes([0.24, 0.048 if args.with_references else 0.075,
                             0.58, 0.010 if args.with_references else 0.016])
    fig.colorbar(image, cax=color_ax, orientation="horizontal", label="$V_P$ (km/s)",
                 ticks=np.linspace(low, high, 6))
    fig.text(0.53, 0.014 if args.with_references else 0.022,
             "Native trilinear fields. No extra smoothing.",
             ha="center", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    report = {"depth_km": args.depth_km, "iteration": 3, "hypocentres_fixed": False,
              "sampling": "Native trilinear Vp at exact depth; 125-m XY pixel centres; no extra smoothing",
              "truth_sampling": "Piecewise-constant truth; deeper cell at a depth-cell boundary",
              "oracle_note": "Full-volume velocity-L2 trilinear fit, not a travel-time inversion or a slice-specific optimum",
              "color_limits_km_s": [low, high], "runs": records,
              "source_sha256": hashes, "script_sha256": sha256(Path(__file__))}
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figure": str(output / f"{stem}.png"), "runs": records}, indent=2))


if __name__ == "__main__":
    main()
