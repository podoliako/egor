"""Plot every predeclared artificial-geometry tomoDD sensitivity result.

python -B paper/paper_scripts/make_tomodd_sensitivity_comparison.py
Reads saved fields only. No solver is imported or launched.
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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "runs/tomodd_sensitivity_20261026_retry"


def rmse_m_s(field, truth):
    return float(np.sqrt(np.mean((field - truth) ** 2)) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--depth-km", type=float, default=1.5)
    args = parser.parse_args()
    if not np.isfinite(args.depth_km) or not 0 <= args.depth_km < 7:
        parser.error("Depth must be inside [0, 7) km")
    study = args.study_root.resolve()
    manifest = json.loads((study / "manifest.json").read_text())
    status = json.loads((study / "status.json").read_text())
    assert status["state"] == "completed", "All four predeclared cases must be complete"
    baseline = Path(manifest["baseline"])
    snapshot = Path(manifest["snapshot"])
    helper = snapshot / "benchmark/tomodd_benchmark/compare_emtomo_flat4.py"
    baseline_info = json.loads((baseline / "worker_info.json").read_text())
    assert sha256(helper) == baseline_info["snapshot_source_sha256"]["tomodd_benchmark/compare_emtomo_flat4.py"]
    native = native_helpers(helper)
    hashes = {str(p): sha256(p) for p in (study / "manifest.json", study / "status.json", helper)}
    x = (np.arange(160) + 0.5) * 0.125
    y = (np.arange(56) + 0.5) * 0.125
    centers = (x, y, (np.arange(56) + 0.5) * 0.125)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((np.full_like(xx, args.depth_km), yy, xx), axis=-1)
    cases = [{"id": "baseline", "output": str(baseline), "dd_weight": 0.1,
              "damping": 40.0, "smoothing": [5.0] * 3}] + manifest["cases"]
    fields, records = [], []
    reference = None
    reference_oracle = None
    for case in cases:
        run = Path(case["output"])
        info = json.loads((run / "worker_info.json").read_text())
        assert info["state"] == "completed", case["id"]
        config = json.loads((run / "config.json").read_text())
        schedule = config["solver"]["iteration_sets"]
        np.testing.assert_allclose(schedule, [[3, case["dd_weight"], 0.08, -9,
                                              1 / case["dd_weight"], case["damping"], 1, 0]],
                                   rtol=0, atol=1e-12)
        assert config["solver"]["smoothing_lon_lat_depth"] == case["smoothing"]
        assert config["initial_hypocenters"]["sigma_km"] == 1.0
        assert config["initial_hypocenters"]["seed"] == 42
        if case["id"] != "baseline":
            assert sha256(run / "work/inversion/tomoDD-SE.inp") == info["generated_control_sha256"]
            for name, expected in info["generated_input_sha256"].items():
                assert sha256(run / "work/inversion" / name) == expected
                assert sha256(baseline / "work/inversion" / name) == expected
        grid = config["tomodd_grid"]
        local_axes = tuple(np.asarray(grid[key]) for key in ("depths_km", "y_km", "x_km"))
        axes = (local_axes[0], local_axes[1] + 3.5, local_axes[2] + 10.0)
        nodes = native.read_velocity_history(run / "work/inversion/tomoDD.vel", (9, 9, 22), 3) / 1000
        with np.load(run / "state.npz", allow_pickle=False) as state:
            np.testing.assert_allclose(nodes, state["final_vp"], rtol=0, atol=1e-11)
            truth = state["physical_velocity_km_s"].copy()
            oracle = state["oracle_vp"].copy()
            if reference is None:
                reference = truth
                reference_oracle = oracle
            else:
                np.testing.assert_array_equal(truth, reference)
                np.testing.assert_allclose(oracle, reference_oracle, rtol=0, atol=1e-9)
        truth_slice = truth[:, :, int(np.floor(args.depth_km / 0.125))].T
        field = RegularGridInterpolator(axes, nodes, bounds_error=True)(points)
        volume = native.evaluate_native(nodes, local_axes, centers)
        truth_volume = truth.transpose(2, 1, 0)
        results = json.loads((run / "results.json").read_text())
        volume_rmse = rmse_m_s(volume, truth_volume)
        np.testing.assert_allclose(volume_rmse, results["final_field_rmse_km_s"] * 1000,
                                   rtol=0, atol=1e-6)
        assert results["final_retained_event_count"] == 1920
        assert not results["unrelocated_event_ids"]
        fields.append(field)
        records.append({"kind": "inversion", "id": case["id"], "run": str(run),
                        "absolute_weight": 1.0, "dd_weight": case["dd_weight"],
                        "damping": case["damping"], "smoothing": case["smoothing"],
                        "slice_rmse_m_s": rmse_m_s(field, truth_slice),
                        "volume_rmse_m_s": volume_rmse,
                        "final_location_rms_m": results["final_hypocenter_rms_km"] * 1000,
                        "vp_range_km_s": [float(field.min()), float(field.max())]})
        for name in ("state.npz", "config.json", "results.json", "worker_info.json",
                     "work/inversion/tomoDD.vel", "work/inversion/tomoDD-SE.inp"):
            hashes[str(run / name)] = sha256(run / name)
    oracle_slice = RegularGridInterpolator(axes, reference_oracle, bounds_error=True)(points)
    oracle_volume = native.evaluate_native(reference_oracle, local_axes, centers)
    fields.insert(0, oracle_slice)
    records.insert(0, {"kind": "oracle", "id": "oracle", "slice_rmse_m_s": rmse_m_s(oracle_slice, truth_slice),
                       "volume_rmse_m_s": rmse_m_s(oracle_volume, truth_volume)})
    low = float(np.floor(min(a.min() for a in fields + [truth_slice]) * 5) / 5)
    high = float(np.ceil(max(a.max() for a in fields + [truth_slice]) * 5) / 5)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(len(fields), 1, figsize=(10, 3.5 * len(fields) + 0.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.075, top=0.94, hspace=0.40)
    for ax, field, record in zip(axs, fields, records):
        image = ax.imshow(field, extent=(0, 20, 0, 7), origin="lower", aspect="equal",
                          interpolation="nearest", cmap="RdBu_r", vmin=low, vmax=high)
        if record["kind"] == "oracle":
            label = "3D L2 oracle · native trilinear representation"
        else:
            prefix = "Current baseline" if record["id"] == "baseline" else record["id"]
            smooth = "/".join(f"{v:g}" for v in record["smoothing"])
            label = (f"{prefix} · abs/DD = 1/{record['dd_weight']:g}"
                     f" · damping {record['damping']:g} · smoothing {smooth}")
        ax.set_title(f"{label}\nRMSE: slice {record['slice_rmse_m_s']:.1f} m/s"
                     f" · volume {record['volume_rmse_m_s']:.1f} m/s", pad=9)
        ax.set(xlim=(0, 20), ylim=(0, 7), ylabel="$Y$ (km)",
               xticks=np.arange(0, 21, 2), yticks=[0, 2, 4, 6, 7])
    axs[-1].set_xlabel("$X$ (km)")
    fig.suptitle(f"tomoDD · artificial acquisition geometry · depth $Z={args.depth_km:g}$ km\n"
                 "All inversions: iteration 3 · initial location sigma 1 km per coordinate · seed 42",
                 y=0.985, fontsize=13)
    color_ax = fig.add_axes([0.24, 0.038, 0.58, 0.008])
    fig.colorbar(image, cax=color_ax, orientation="horizontal", label="$V_P$ (km/s)",
                 ticks=np.linspace(low, high, 6))
    fig.text(0.53, 0.009, "Common color scale · no outlines or extra smoothing · all four predeclared cases shown",
             ha="center", fontsize=10)
    output = Path(__file__).resolve().parents[1] / "figures"
    stem = "tomodd_synthetic_parameter_comparison_z" + f"{args.depth_km:g}".replace(".", "p") + "km"
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    report = {"study": str(study), "depth_km": args.depth_km, "iteration": 3,
              "hypocentres_fixed": False, "initial_sigma_km_per_coordinate": 1.0, "seed": 42,
              "sampling": "Exact depth, 125-m XY pixel centres, native trilinear Vp; deeper truth cell at cell boundary",
              "oracle_note": "Full-volume velocity-L2 fit, not a slice-specific optimum",
              "color_limits_km_s": [low, high], "panels": records,
              "source_sha256": hashes, "script_sha256": sha256(Path(__file__))}
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = ["# Artificial-geometry tomoDD parameter comparison", "",
            "Exact depth: 1.5 km. All inversions: three joint iterations, initial coordinate sigma 1 km, seed 42.",
            "All four predeclared cases are shown; no checkpoint or case selection.", "",
            "| Case | Abs/DD amplitude | Damping | Smoothing | Slice RMSE, m/s | Volume RMSE, m/s | Location RMS, m |",
            "|---|---|---:|---|---:|---:|---:|"]
    for r in records:
        if r["kind"] == "oracle":
            rows.append(f"| 3D L2 oracle | — | — | — | {r['slice_rmse_m_s']:.2f} | {r['volume_rmse_m_s']:.2f} | — |")
        else:
            smooth = "/".join(f"{v:g}" for v in r["smoothing"])
            rows.append(f"| {r['id']} | 1/{r['dd_weight']:g} | {r['damping']:g} | {smooth} | "
                        f"{r['slice_rmse_m_s']:.2f} | {r['volume_rmse_m_s']:.2f} | {r['final_location_rms_m']:.2f} |")
    rows += ["", "Reproduce: `python -B paper/paper_scripts/make_tomodd_sensitivity_comparison.py`", "",
             "The 3D oracle is not a slice-specific error lower bound. No inverse/forward solver is run by this plotter."]
    (output / f"{stem}.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({"figure": str(output / f"{stem}.png"), "panels": records}, indent=2))


if __name__ == "__main__":
    main()
