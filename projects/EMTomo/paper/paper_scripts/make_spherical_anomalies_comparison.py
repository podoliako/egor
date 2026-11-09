"""Native-field comparison: two vertical and one horizontal section per geometry.

python -B paper/paper_scripts/make_spherical_anomalies_comparison.py
Reads completed runs only. Oracles are velocity-L2 references, not inversions.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from make_checkerboard_geometry import map_frame
from make_spherical_anomalies_truth import ARTIFICIAL, GEYSERS, ROOT, OUTPUT, SECTIONS, sha256

STUDY = ROOT / "runs/matched_geometry_20261021"
DD_STUDY = ROOT / "runs/tomodd_guard_fix_20261027"
STEM = "spherical_anomalies_comparison"
COLUMNS = ("EMTomo oracle", "EMTomo hard", "EMTomo soft", "tomoDD", "tomoDD oracle")


def read_json(path):
    return json.loads(path.read_text())


def sample_cells(field, xx, zz, y):
    assert field.shape == (20, 7, 7) and np.isfinite(field).all()
    return field[np.floor(xx).astype(int), np.floor(y).astype(int), np.floor(zz).astype(int)]


def main():
    manifest = read_json(STUDY / "manifest.json")
    assert read_json(STUDY / "status.json")["status"] == "completed"
    assert read_json(DD_STUDY / "status.json")["state"] == "completed"
    build = read_json(DD_STUDY / "build/manifest.json")
    assert build["source_change_proof"]["changed_lines_1based"] == [840, 841]
    hashes = {}

    def record(path, expected=None):
        digest = sha256(path)
        assert expected is None or digest == expected, f"Hash mismatch: {path}"
        hashes[str(path)] = digest
        return path

    for path in (STUDY / "manifest.json", DD_STUDY / "manifest.json",
                 DD_STUDY / "build/manifest.json", Path(__file__)):
        record(path)
    with np.load(record(ARTIFICIAL / "model.npz"), allow_pickle=False) as saved:
        truth = saved["velocity"] / 1000
        assert truth.shape == (160, 56, 56) and float(saved["cell_size_m"]) == 125
        np.testing.assert_array_equal(saved["origin_m"], [0, 0, 0])
    with np.load(record(GEYSERS / "model.npz"), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["velocity"] / 1000, truth)
    # Equal-volume velocity means: the exact L2 optimum for aligned constant cells.
    em_oracle = truth.reshape(20, 8, 7, 8, 7, 8).mean(axis=(1, 3, 5))
    spacing = .025
    x = (np.arange(round(20 / spacing)) + .5) * spacing
    z = (np.arange(round(7 / spacing)) + .5) * spacing
    xx, zz = np.meshgrid(x, z)
    rows, records = [], []
    shared_dd_oracle = None
    for geometry, source in (("synthetic", ARTIFICIAL), ("geysers", GEYSERS)):
        fields, configs = [], []
        for mode, top_n in (("hard", 1), ("soft", 10)):
            case_id = f"{geometry}-em-{mode}-k15-top{top_n}-dmin1"
            case = next(c for c in manifest["cases"] if c["id"] == case_id)
            result = read_json(record(STUDY / "cases" / case_id / "result.json"))
            assert result["status"] == "completed"
            run = Path(result["run_dir"])
            meta = read_json(record(run / "meta.json"))
            assert meta["source_experiment"]["id"] == source.name == case["experiment_id"]
            assert meta["source_experiment"]["model_sha256"] == sha256(source / "model.npz")
            params = meta["run_params"]
            assert params["n_cycles"] == 12 and params["n_candidates"] == 15
            assert params["candidate_mode"] == mode and params["weights_top_n"] == top_n
            assert params["weights_min_distance"] == 1
            assert meta["grid_info"]["coarse_cell_size"] == 1000
            assert meta["grid_info"]["coarse_shape"] == [20, 7, 7]
            for iteration in range(12):
                complete = read_json(run / f"iter_{iteration}/complete.json")
                assert complete["iter"] == iteration
            final = record(run / "final_model.npy", result["final_model_sha256"])
            fields.append(np.load(final, allow_pickle=False) / 1000)
            configs.append(params)
            for name in ("arrivals.csv", "metadata.json"):
                path = source.parents[1] / "output" / source.name / name
                key = str(path.relative_to(Path(manifest["snapshot_root"])))
                record(path, result["provenance"]["snapshot_sha256"][key])
        # Only the assignment settings and run bookkeeping may differ.
        for key in ("lambda_reg", "coverage_damping_power", "max_velocity_step_fraction",
                    "temperature", "subdivision", "weight_noise_relative_sigma",
                    "weight_noise_absolute_sigma_s", "weight_model_sigma_s"):
            assert configs[0][key] == configs[1][key]
        dd = DD_STUDY / "outputs" / f"{geometry}-dd1-d40-s5"
        info = read_json(record(dd / "worker_info.json"))
        config = read_json(record(dd / "config.json"))
        assert info["state"] == "completed" and info["dd1_input_reuse"]["all_inputs_byte_identical"]
        assert info["solver_sha256"] == build["artifact_sha256"]["tomoDD-SE"]
        assert config["source_experiment"] == source.name
        assert config["solver"]["iteration_sets"] == [[3, 1., .08, -9., 1., 40., 1, 0.]]
        assert config["initial_hypocenters"]["sigma_km"] == 1
        for name, digest in config["source_forward_metadata"]["input_sha256"].items():
            record(source / name, digest)
        for name, digest in info["generated_input_sha256"].items():
            record(dd / "work/inversion" / name, digest)
        record(dd / "work/inversion/tomoDD-SE.inp", info["generated_control_sha256"])
        grid = config["tomodd_grid"]
        axes = (np.array(grid["depths_km"]), np.array(grid["y_km"]) + 3.5,
                np.array(grid["x_km"]) + 10)
        with np.load(record(dd / "state.npz"), allow_pickle=False) as state:
            np.testing.assert_array_equal(state["physical_velocity_km_s"], truth)
            nodes, oracle = state["final_vp"].copy(), state["oracle_vp"].copy()
        if shared_dd_oracle is not None:
            np.testing.assert_allclose(oracle, shared_dd_oracle, rtol=0, atol=1e-9)
        shared_dd_oracle = oracle
        interpolators = [RegularGridInterpolator(axes, a, bounds_error=True) for a in (nodes, oracle)]
        for orientation, coordinate in [("Y", y) for y, _ in SECTIONS] + [("Z", 2.5)]:
            horizontal = orientation == "Z"
            yy = zz if horizontal else np.full_like(xx, coordinate)
            depths = np.full_like(xx, coordinate) if horizontal else zz
            points = np.stack((depths, yy, xx), axis=-1)
            slices = [sample_cells(a, xx, depths, yy) for a in (em_oracle, *fields)]
            slices.extend(interp(points) for interp in interpolators)
            assert all(a.shape == xx.shape and np.isfinite(a).all() for a in slices)
            rows.append(slices)
            records.append({"geometry": geometry, "section_axis": orientation,
                                        "section_coordinate_km": coordinate, "emtomo_updates": 12,
                            "soft_top_n": 10, "n_candidates": 15, "weights_min_distance": 1,
                            "em_runs": [str(Path(read_json(STUDY / "cases" /
                                f"{geometry}-em-{mode}-k15-top{n}-dmin1/result.json")["run_dir"]))
                                for mode, n in (("hard", 1), ("soft", 10))],
                            "dd_run": str(dd), "dd_iterations": 3,
                            "slice_ranges_km_s": {name: [float(a.min()), float(a.max())]
                                                   for name, a in zip(COLUMNS, slices)}})
    anchors = ((3.5, "#454545"), (4., "#7a7a7a"), (4.75, "#ffffff"),
               (5.25, "#e32400"), (6., "#61177c"), (6.5, "#351041"), (7., "#190820"))
    norm = Normalize(3.5, 7.)
    assert all(norm.vmin <= a.min() <= a.max() <= norm.vmax for row in rows for a in row), records
    cmap = LinearSegmentedColormap.from_list(STEM, [(norm(v), c) for v, c in anchors], N=2048)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.linewidth": .6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(17, 8.2))
    panel_width, column_gap, left = .174, .011, .045
    panel_height = panel_width * 17 * (7 / 20) / 8.2
    row_bottoms = (.805, .666, .527, .335, .196, .057)
    for y, label in ((.985, "Artificial geometry"), (.49, "Geysers geometry")):
        fig.text(.5, y, label, ha="center", va="center", fontsize=12)
    for row, (slices, record_row) in enumerate(zip(rows, records)):
        horizontal = record_row["section_axis"] == "Z"
        for col, field in enumerate(slices):
            ax = fig.add_axes([left + col * (panel_width + column_gap),
                               row_bottoms[row], panel_width, panel_height])
            image = ax.imshow(field, origin="lower" if horizontal else "upper",
                              extent=(0, 20, 0, 7) if horizontal else (0, 20, 7, 0),
                              cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            ax.set(xlim=(0, 20), ylim=(0, 7) if horizontal else (7, 0),
                   xticks=(0, 5, 10, 15, 20), yticks=(0, 2, 4, 6, 7))
            if row == 0:
                ax.set_title(COLUMNS[col], pad=8)
            if col == 0:
                section = f"${record_row['section_axis']}={record_row['section_coordinate_km']:g}$ km"
                ax.set_ylabel(section + ("\n$Y$ (km)" if horizontal else "\nDepth (km)"))
            else:
                ax.tick_params(labelleft=False)
            if row == 5:
                ax.set_xlabel("$X$ (km)")
            if row % 3 != 2:
                ax.tick_params(labelbottom=False)
            ax.tick_params(direction="out", pad=2)
            map_frame(ax, interval=2, thickness=.06)
    color_ax = fig.add_axes([.975, 1 / 3, .005, 1 / 3])
    bar = fig.colorbar(image, cax=color_ax, ticks=np.arange(3.5, 7.01, .5))
    color_ax.set_title("$V$ (km/s)", fontsize=9, pad=8)
    bar.outline.set_linewidth(.45)
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{extension}"
        fig.savefig(path, dpi=300)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    report = {"columns": COLUMNS, "rows": records, "sampling_spacing_km": spacing,
              "sampling": "Exact Y=2.4/4.8 and Z=2.5 km planes; 25-m pixel centers; no added smoothing",
              "em_representation": "Native 1-km constant velocity cells",
              "em_oracle": "Arithmetic velocity mean of 8x8x8 native truth cells per 1-km cell",
              "dd_representation": "Native trilinear Vp, including fixed exterior nodes",
              "dd_oracle": "Saved whole-volume least-squares trilinear truth fit; not a slice fit",
              "color_limits_km_s": [norm.vmin, norm.vmax], "color_anchors_km_s": dict(anchors),
              "colorbar_height_fraction": 1 / 3, "colorbar_vertical_alignment": "center",
              "sha256": hashes}
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUTPUT / f"{STEM}.png")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
