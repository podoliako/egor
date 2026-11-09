"""Read-only native-field scoring of the exact runs in the comparison manifest.

python3 -B paper/evaluate_checkerboard_velocity.py
python3 -B paper/evaluate_checkerboard_velocity.py --check

Only figures/checkerboard_velocity_metrics.json can be created (exclusive write).
No solver, fitting, plotting imports, or input modifications. --check never writes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

PAPER = Path(__file__).resolve().parent
MANIFEST = PAPER / "figures/checkerboard_nx4_nx6_nx8_comparison.json"
OUTPUT = PAPER / "figures/checkerboard_velocity_metrics.json"
SPACINGS = (0.25, 0.125, 0.0625, 0.03125)
DOMAIN = np.array([24., 12., 12.])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(truth, cell, em, axes, nodes, spacing):
    shape = np.rint(DOMAIN / spacing).astype(int)
    require(np.allclose(shape * spacing, DOMAIN), "Grid must tile volume")
    centers = [(np.arange(n) + 0.5) * spacing for n in shape]
    interpolators = {k: RegularGridInterpolator(axes, v, method="linear", bounds_error=True)
                     for k, v in nodes.items()}
    sums = dict.fromkeys(("em", *nodes), 0.)
    count = 0
    # Small X slabs keep even the finest grid bounded in memory.
    for start in range(0, shape[0], 4):
        xyz = np.meshgrid(centers[0][start:start + 4], centers[1], centers[2], indexing="ij")
        indices = tuple(np.floor(a / cell).astype(int) for a in xyz)
        target = truth[indices]
        predicted = em[tuple(np.floor(a).astype(int) for a in xyz)]
        sums["em"] += float(np.sum((predicted - target)**2, dtype=np.float64))
        points = np.stack(xyz[::-1], axis=-1)
        for name, interp in interpolators.items():
            predicted = interp(points)
            require(np.isfinite(predicted).all(), "Nonfinite interpolated field")
            sums[name] += float(np.sum((predicted - target)**2, dtype=np.float64))
        count += target.size
    require(count == int(np.prod(shape)), "Incomplete volume")
    return {"spacing_km": spacing, "shape_xyz": shape.tolist(), "samples": count,
            "rmse_m_s": {k: float(np.sqrt(v / count)) for k, v in sums.items()}}


def evaluate():
    manifest = json.loads(MANIFEST.read_text())
    files = {}

    def record(path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or expected == digest, f"SHA256 mismatch: {path}")
        files[str(path)] = digest
        return path

    record(MANIFEST)
    for path, digest in manifest["sha256"].items():
        record(path, digest)
    for name in ("evaluate_metrics.py", "make_checkerboard_comparison.py"):
        record(PAPER / name)
    require([r["nx"] for r in manifest["rows"]] == [4, 6, 8], "Unexpected run selection")
    rows = []
    for row in manifest["rows"]:
        nx = row["nx"]
        em_dir, dd_dir = Path(row["em_run"]), Path(row["dd_run"])
        meta = json.loads((em_dir / "meta.json").read_text())
        config = json.loads((dd_dir / "config.json").read_text())
        eid = meta["source_experiment"]["id"]
        require(config["source_experiment"] == eid and f"_nx{nx}_" in eid, "Source mismatch")
        source = dd_dir.parents[1] / "snapshot/projects/forward_modeling/experiments/input" / eid
        for name, digest in config["source_forward_metadata"]["input_sha256"].items():
            record(source / name, digest)
        record(source / "model.npz", meta["source_experiment"]["model_sha256"])
        with np.load(source / "model.npz", allow_pickle=False) as data:
            truth = data["velocity"].astype(float)
            cell = float(data["cell_size_m"]) / 1000
            require(np.array_equal(data["origin_m"], [0, 0, 0]), "Nonzero origin")
        require(np.array_equal(np.array(truth.shape) * cell, DOMAIN), "Wrong domain")
        require(np.array_equal(np.unique(truth), [4750., 5250.]), "Unexpected truth")
        require(meta["grid_info"]["coarse_cell_size"] == 1000, "Wrong EM spacing")
        require(meta["run_params"]["n_cycles"] == row["em_updates"] == 12, "Wrong EM updates")
        require(meta["run_params"]["coverage_damping_power"] == 0, "Wrong damping")
        require(config["solver"]["iteration_sets"][0][0] == row["dd_iterations"] == 3, "Wrong DD iteration")
        for iteration in range(12):
            complete = record(em_dir / f"iter_{iteration}/complete.json")
            require(json.loads(complete.read_text())["iter"] == iteration, "Incomplete EM run")
        em = np.load(em_dir / "final_model.npy", allow_pickle=False)
        require(em.shape == (24, 12, 12) and np.isfinite(em).all(), "Invalid EM field")
        require(np.all(np.load(em_dir / "initial_model.npy", allow_pickle=False) == 5000), "EM start")
        grid = config["tomodd_grid"]
        axes = (np.array(grid["depths_km"]), np.array(grid["y_km"]) + 6,
                np.array(grid["x_km"]) + 12)
        with np.load(dd_dir / "state.npz", allow_pickle=False) as state:
            require(np.array_equal(state["physical_velocity_km_s"] * 1000, truth), "DD truth mismatch")
            require(np.array_equal(state["depths"], axes[0]), "DD depth mismatch")
            require(float(state["cell_size_km"]) == cell, "DD truth spacing mismatch")
            require(np.all(state["initial_vp"] == 5.), "DD start")
            nodes = {k: state[v] * 1000 for k, v in (("dd", "final_vp"), ("dd_oracle", "oracle_vp"))}
        existing = json.loads(record(dd_dir / "results.json").read_text())
        reproduced = score(truth, cell, em, axes, nodes, existing["evaluation_spacing_km"])
        require(reproduced["samples"] == existing["evaluation_samples"], "Existing grid count")
        for key, old_key in (("dd", "final_field_rmse_km_s"), ("dd_oracle", "oracle_trilinear_field_rmse_km_s")):
            require(abs(reproduced["rmse_m_s"][key] - 1000 * existing[old_key]) < 1e-8, "Existing score mismatch")
        quality = [json.loads(line) for line in record(em_dir / "quality.jsonl").read_text().splitlines() if line.strip()]
        evaluations = []
        for spacing in SPACINGS:
            result = score(truth, cell, em, axes, nodes, spacing)
            evaluations.append(result)
            print(f"nx={nx}: {json.dumps(result)}", flush=True)
        require(abs(evaluations[-1]["rmse_m_s"]["em"] - quality[-1]["rms_m_s"]) < 1e-8, "EM quality mismatch")
        # Every true-cell boundary is an EM-cell boundary: exact constant-cell oracle.
        require(cell.is_integer(), "Truth not representable by EM cells")
        rows.append({"nx": nx, "source_experiment": eid, "truth_model": str(source / "model.npz"),
                     "truth_cell_size_km": cell, "em_run": str(em_dir), "em_updates": 12,
                     "dd_run": str(dd_dir), "dd_iterations": 3, "dd_axes_zyx_km": [a.tolist() for a in axes],
                     "existing_dd_scores_reproduced": reproduced,
                     "existing_em_last_quality": quality[-1], "grids": evaluations,
                     "recommended_rmse_m_s": evaluations[-1]["rmse_m_s"],
                     "last_refinement_delta_m_s": {k: evaluations[-1]["rmse_m_s"][k] - evaluations[-2]["rmse_m_s"][k] for k in nodes},
                     "initial_rmse_m_s": 250., "em_constant_oracle_rmse_m_s": 0.})
    for path, digest in files.items():
        require(sha(Path(path)) == digest, f"Input changed during scoring: {path}")
    return {"schema_version": 1, "manifest": str(MANIFEST),
            "method": {"metric": "sqrt(mean((v_est_m_s - v_truth_m_s)**2))",
                       "domain_xyz_km": [[0, 24], [0, 12], [0, 12]],
                       "sampling": "Uniform cell centres (i+0.5)*h; equal volume weights; all cells; no coverage mask",
                       "truth": "Native original piecewise constant cells, XYZ array order",
                       "em": "Native piecewise constant 1 km cells, final_model.npy in m/s",
                       "dd": "RegularGridInterpolator linear Vp, state.final_vp * 1000, ZYX; X+12,Y+6 km; full native grid including boundary nodes; no extrapolation",
                       "dd_oracle": "Saved state.oracle_vp, same interpolation; not an inversion, not refitted on these grids, not asserted to be the continuous-volume optimum",
                       "em_oracle": "Exact zero: truth boundaries align with 1 km cells",
                       "recommended_spacing_km": SPACINGS[-1], "spacings_km": list(SPACINGS),
                       "legacy_checkerboard_used": False},
            "rows": rows, "files_sha256": dict(sorted(files.items())),
            "evaluator_sha256": sha(Path(__file__)),
            "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        require(not OUTPUT.exists() and not OUTPUT.is_symlink(), f"Refusing to overwrite {OUTPUT}")
    report = evaluate()
    if args.check:
        require(report == json.loads(OUTPUT.read_text()), "Report differs from recomputation")
        print("Verified: report matches recomputation; no writes.")
    else:
        with OUTPUT.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
