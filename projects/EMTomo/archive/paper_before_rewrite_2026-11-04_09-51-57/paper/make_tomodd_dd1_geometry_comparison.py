"""Compare dd1-d40-s5 on artificial/Geysers geometry with their common 3D oracle.

python -B paper/make_tomodd_dd1_geometry_comparison.py
Only reads saved results; no inverse or forward solver is launched.
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
from make_tomodd_sensitivity_comparison import rmse_m_s

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/tomodd_dd1_geometry_20261026"
RUNS = (
    ("Artificial geometry", ROOT / "runs/tomodd_sensitivity_20261026/outputs/dd1-d40-s5", 1920),
    ("Geysers geometry", STUDY / "outputs/geysers-dd1-d40-s5", 3776),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-km", type=float, default=1.5)
    parser.add_argument("--guard-fixed", action="store_true",
                        help="Plot the corrected horizontal-guard runs without replacing old figures")
    parser.add_argument("--with-before", action="store_true",
                        help="With --guard-fixed, also show both original runs")
    args = parser.parse_args()
    if args.with_before and not args.guard_fixed:
        parser.error("--with-before requires --guard-fixed")
    if not np.isfinite(args.depth_km) or not 0 <= args.depth_km < 7:
        parser.error("Depth must be inside [0, 7) km")
    study = ROOT / "runs/tomodd_guard_fix_20261027" if args.guard_fixed else STUDY
    runs = RUNS
    if args.guard_fixed:
        corrected = (
            ("Artificial geometry · corrected guard", study / "outputs/synthetic-dd1-d40-s5", 1920),
            ("Geysers geometry · corrected guard", study / "outputs/geysers-dd1-d40-s5", 3776),
        )
        runs = ((RUNS[0], corrected[0], RUNS[1], corrected[1])
                if args.with_before else corrected)
    assert json.loads((study / "status.json").read_text())["state"] == "completed"
    snapshot = ROOT / "runs/matched_geometry_20261021/snapshot"
    helper = snapshot / "benchmark/tomodd_benchmark/compare_emtomo_flat4.py"
    native = native_helpers(helper)
    hashes = {str(p): sha256(p) for p in (helper, study / "manifest.json", study / "status.json")}
    build_manifest = None
    if args.guard_fixed:
        build_path = study / "build/manifest.json"
        build_manifest = json.loads(build_path.read_text())
        assert build_manifest["source_change_proof"]["changed_lines_1based"] == [840, 841]
        assert build_manifest["source_change_proof"]["bottom_guard_unchanged"]
        assert build_manifest["original_relink_byte_identical"]
        hashes[str(build_path)] = sha256(build_path)
        for name, digest in build_manifest["artifact_sha256"].items():
            path = study / "build" / name
            assert sha256(path) == digest
            hashes[str(path)] = digest
    x, y = (np.arange(160) + 0.5) * 0.125, (np.arange(56) + 0.5) * 0.125
    centers = (x, y, y)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((np.full_like(xx, args.depth_km), yy, xx), axis=-1)
    fields, records = [], []
    reference, reference_oracle, reference_initial = None, None, None
    for label, run, count in runs:
        info = json.loads((run / "worker_info.json").read_text())
        assert info["state"] == "completed"
        assert sha256(helper) == info["snapshot_source_sha256"]["tomodd_benchmark/compare_emtomo_flat4.py"]
        corrected_run = args.guard_fixed and run.parent == study / "outputs"
        if corrected_run:
            assert info["solver_sha256"] == build_manifest["artifact_sha256"]["tomoDD-SE"]
            assert info["solver_manifest_sha256"] == hashes[str(study / "build/manifest.json")]
            reuse = info["dd1_input_reuse"]
            assert reuse["all_inputs_byte_identical"] and reuse["control_byte_identical"]
            old_work = Path(reuse["reference"]) / "work/inversion"
            for name, digest in info["generated_input_sha256"].items():
                assert sha256(old_work / name) == digest
            assert sha256(old_work / "tomoDD-SE.inp") == info["generated_control_sha256"]
        config = json.loads((run / "config.json").read_text())
        assert config["solver"]["iteration_sets"] == [[3, 1.0, 0.08, -9.0, 1.0, 40.0, 1, 0.0]]
        assert config["solver"]["smoothing_lon_lat_depth"] == [5.0] * 3
        assert config["initial_hypocenters"]["sigma_km"] == 1.0
        assert config["initial_hypocenters"]["seed"] == 42
        assert sha256(run / "work/inversion/tomoDD-SE.inp") == info["generated_control_sha256"]
        for name, digest in info["generated_input_sha256"].items():
            assert sha256(run / "work/inversion" / name) == digest
        grid = config["tomodd_grid"]
        local_axes = tuple(np.asarray(grid[key]) for key in ("depths_km", "y_km", "x_km"))
        axes = (local_axes[0], local_axes[1] + 3.5, local_axes[2] + 10.0)
        nodes = native.read_velocity_history(run / "work/inversion/tomoDD.vel", (9, 9, 22), 3) / 1000
        with np.load(run / "state.npz", allow_pickle=False) as state:
            np.testing.assert_allclose(nodes, state["final_vp"], rtol=0, atol=1e-11)
            truth, oracle, initial = (state[key].copy() for key in
                                      ("physical_velocity_km_s", "oracle_vp", "initial_vp"))
            if reference is None:
                reference, reference_oracle, reference_initial = truth, oracle, initial
            else:
                np.testing.assert_array_equal(truth, reference)
                np.testing.assert_allclose(oracle, reference_oracle, rtol=0, atol=1e-9)
                np.testing.assert_array_equal(initial, reference_initial)
        truth_slice = truth[:, :, int(np.floor(args.depth_km / 0.125))].T
        truth_volume = truth.transpose(2, 1, 0)
        field = RegularGridInterpolator(axes, nodes, bounds_error=True)(points)
        volume_rmse = rmse_m_s(native.evaluate_native(nodes, local_axes, centers), truth_volume)
        result = json.loads((run / "results.json").read_text())
        np.testing.assert_allclose(volume_rmse, result["final_field_rmse_km_s"] * 1000, rtol=0, atol=1e-6)
        assert result["event_count"] == result["final_retained_event_count"] == count
        assert not result["unrelocated_event_ids"]
        records.append({"label": label, "run": str(run), "events": count,
                        "horizontal_guard_corrected": corrected_run,
                        "solver_sha256": info["solver_sha256"],
                        "slice_rmse_m_s": rmse_m_s(field, truth_slice),
                        "volume_rmse_m_s": volume_rmse,
                        "location_rms_m": result["final_hypocenter_rms_km"] * 1000})
        fields.append(field)
        for name in ("config.json", "state.npz", "results.json", "worker_info.json",
                     "work/inversion/tomoDD.vel", "work/inversion/tomoDD-SE.inp"):
            hashes[str(run / name)] = sha256(run / name)
    oracle_slice = RegularGridInterpolator(axes, reference_oracle, bounds_error=True)(points)
    records.insert(0, {"label": "3D L2 oracle", "slice_rmse_m_s": rmse_m_s(oracle_slice, truth_slice),
                       "volume_rmse_m_s": rmse_m_s(native.evaluate_native(reference_oracle, local_axes, centers), truth_volume)})
    fields.insert(0, oracle_slice)
    low = float(np.floor(min(a.min() for a in fields + [truth_slice]) * 5) / 5)
    high = float(np.ceil(max(a.max() for a in fields + [truth_slice]) * 5) / 5)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(len(fields), 1, figsize=(10, 2.8 * len(fields) + 2.6),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.90, hspace=0.42)
    for ax, field, record in zip(axs, fields, records):
        image = ax.imshow(field, extent=(0, 20, 0, 7), origin="lower", aspect="equal",
                          interpolation="nearest", cmap="RdBu_r", vmin=low, vmax=high)
        ax.set_title(f"{record['label']}\nRMSE: slice {record['slice_rmse_m_s']:.1f} m/s"
                     f" · volume {record['volume_rmse_m_s']:.1f} m/s", pad=9)
        ax.set(xlim=(0, 20), ylim=(0, 7), ylabel="$Y$ (km)",
               xticks=np.arange(0, 21, 2), yticks=[0, 2, 4, 6, 7])
    axs[-1].set_xlabel("$X$ (km)")
    fig.suptitle(f"tomoDD · dd1-d40-s5 · depth $Z={args.depth_km:g}$ km\n"
                 "abs/DD = 1/1 · damping 40 · smoothing 5/5/5\n"
                 "Both inversions: iteration 3 · initial location sigma 1 km per coordinate · seed 42",
                 y=0.985, fontsize=12)
    color_ax = fig.add_axes([0.24, 0.065, 0.58, 0.013])
    fig.colorbar(image, cax=color_ax, orientation="horizontal", label="$V_P$ (km/s)",
                 ticks=np.linspace(low, high, 6))
    fig.text(0.53, 0.022, "Common color scale · no outlines or extra smoothing · hypocentres free", ha="center", fontsize=10)
    output = Path(__file__).resolve().parent / "figures"
    stem = "tomodd_dd1_artificial_geysers_z" + f"{args.depth_km:g}".replace(".", "p") + "km"
    if args.guard_fixed:
        stem += "_guard_fixed" + ("_before_after" if args.with_before else "")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    report = {"depth_km": args.depth_km, "parameter_set": "dd1-d40-s5", "iteration": 3,
              "absolute_dd_weight": [1.0, 1.0], "damping": 40, "smoothing": [5, 5, 5],
              "initial_sigma_km_per_coordinate": 1, "seed": 42, "hypocentres_fixed": False,
              "guard_fixed_comparison": args.guard_fixed, "includes_before": args.with_before,
              "color_limits_km_s": [low, high], "panels": records,
              "sampling": "Native trilinear Vp at exact depth, 125-m XY centres, deeper truth cell at boundary",
              "oracle_note": "Full-volume velocity L2 fit, not a slice-specific optimum",
              "source_sha256": hashes, "script_sha256": sha256(Path(__file__))}
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figure": str(output / f"{stem}.png"), "panels": records}, indent=2))


if __name__ == "__main__":
    main()
