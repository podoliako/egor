"""Supplementary metrics for the hard/soft article tables (Sections 4.2--4.6) and Table 4.1.

python3 -B paper/paper_scripts/make_table_supplementary_metrics.py [--skip-checkerboard41]

Reads saved artifacts only (no solves). For every hard/soft table row it reports, at the
article checkpoint (L-curve V_k for 4.2--4.4, final V_N for 4.5--4.6):
  * velocity RMSE (recomputed and checked against the table JSON) and MAPE on the same
    nested common grid (coarser field repeated onto the finer grid, equal volume weights);
  * residual RMS in seconds: arrival-time residuals of the maximum-weight hypothesis, from
    iter_i/event_j/residuals.npy (pairwise differences r_a - r_b), with the per-event mean
    removed, pooled over all saved picks of all events;
  * hypocentre P90 of the maximum-weight candidate (checked against the table) and of the
    weight-averaged candidate position.
Residuals and locations come from the same E-step: iter_k in V_k (L-curve) or iter_(N-1)
in V_(N-1), before the final update (final models).
For Section 4.1 it adds whole-volume MAPE for EMTomo, tomoDD and the trilinear oracle on
the 31.25-m grid used for the RMSE table, reusing evaluate_checkerboard_velocity.py inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
OUTPUT = PAPER / "figures/table_supplementary_metrics.json"
LCURVE = PAPER / "figures/hard_soft_lcurve_comparison_metrics.json"
FINAL = PAPER / "figures/final_comparison_metrics.json"
CB41 = PAPER / "figures/checkerboard_velocity_metrics.json"
LCURVE_TABLES = ("onesided", "two_clusters", "real_geometry_noise", "real_geometry_candidates")
FINAL_TABLES = ("fault_retention", "fault_separation", "heterogeneous")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_grid(model, model_side, truth, truth_side):
    ratio = max(model_side, truth_side) / min(model_side, truth_side)
    require(np.isclose(ratio, round(ratio)), "Non-nested evaluation grids")
    f = int(round(ratio))
    if model_side >= truth_side:
        for axis in range(3):
            model = np.repeat(model, f, axis=axis)
    else:
        for axis in range(3):
            truth = np.repeat(truth, f, axis=axis)
    require(model.shape == truth.shape, "Common-grid shape mismatch")
    return model, truth


def velocity_metrics(model, model_side, truth, truth_side):
    m, t = common_grid(model, model_side, truth, truth_side)
    return float(np.sqrt(np.mean((m - t) ** 2))), float(100 * np.mean(np.abs(m - t) / t))


def event_metrics(weights_dir, truth_xyz, side):
    sq_sum, n_picks = 0.0, 0
    err_map, err_mean = [], []
    for i, true_xyz in enumerate(truth_xyz):
        event = weights_dir / f"event_{i}"
        with np.load(event / "weights.npz", allow_pickle=False) as saved:
            pos = (saved["positions"] + 0.5) * side
            w = np.asarray(saved["weight_values"], dtype=float)
        require(np.isclose(w.sum(), 1.0, atol=1e-6) and (w >= 0).all(), f"Bad weights: {event}")
        err_map.append(np.linalg.norm(pos[np.argmax(w)] - true_xyz))
        err_mean.append(np.linalg.norm((w[:, None] * pos).sum(0) / w.sum() - true_xyz))
        diff = np.load(event / "residuals.npy", allow_pickle=False)
        valid = ~np.all(np.isnan(diff), axis=1)
        sub = diff[np.ix_(valid, valid)]
        require(not np.isnan(sub).any() and sub.shape[0] >= 2, f"Bad residual matrix: {event}")
        centred = sub.mean(axis=1)            # r_a - mean_b r_b
        sq_sum += float(np.sum(centred ** 2))
        n_picks += int(centred.size)
    return dict(residual_rms_s=float(np.sqrt(sq_sum / n_picks)), n_residual_picks=n_picks,
                p90_map_m=float(np.percentile(err_map, 90)),
                p90_weighted_mean_m=float(np.percentile(err_mean, 90)))


def hard_soft_tables():
    lcurve, final = json.loads(LCURVE.read_text()), json.loads(FINAL.read_text())
    records = {(r["run_id"], "lcurve"): r for r in lcurve["records"]}
    records.update({(r["run_id"], "final"): r for r in final["records"]})
    experiments = final["experiments"]
    cache, truth_cache, out = {}, {}, {}
    for policy, source, names in (("lcurve", lcurve, LCURVE_TABLES), ("final", final, FINAL_TABLES)):
        for table in names:
            out[table] = []
            for row in source["tables"][table]:
                entry = dict(pair_id=row["pair_id"], checkpoint_policy=policy)
                if "start" in row:
                    entry["start"] = row["start"]
                for mode in ("hard", "soft"):
                    run_id = row[mode]["run_id"]
                    key = (run_id, policy)
                    if key not in cache:
                        rec = records[key]
                        exp = experiments[rec["experiment_id"]]
                        directory = Path(exp["directory"])
                        for name, digest in exp["input_sha256"].items():
                            require(sha(directory / name) == digest, f"Input changed: {directory / name}")
                        if rec["experiment_id"] not in truth_cache:
                            with np.load(directory / "model.npz", allow_pickle=False) as t:
                                truth = t["velocity"].astype(float)
                                tside = float(t["cell_size_m"])
                            with (directory / "events.csv").open(newline="") as stream:
                                ev = np.array([[float(e[a + "_m"]) for a in "xyz"] for e in csv.DictReader(stream)])
                            truth_cache[rec["experiment_id"]] = (truth, tside, ev)
                        truth, tside, ev = truth_cache[rec["experiment_id"]]
                        model = np.load(ROOT / rec["model_path"], allow_pickle=False)
                        grid = rec["grid_info"]
                        rmse, mape = velocity_metrics(model, float(grid["coarse_cell_size"]), truth, tside)
                        require(np.isclose(rmse, rec["velocity_rmse_m_s"], rtol=1e-9),
                                f"RMSE mismatch {run_id}: {rmse} vs {rec['velocity_rmse_m_s']}")
                        metrics = event_metrics(ROOT / rec["weights_directory"], ev, float(grid["fine_cell_size"]))
                        require(np.isclose(metrics["p90_map_m"], rec["hypocenter_p90_m"], rtol=1e-9),
                                f"P90 mismatch {run_id}: {metrics['p90_map_m']} vs {rec['hypocenter_p90_m']}")
                        cache[key] = dict(run_id=run_id, model_path=rec["model_path"],
                                          weights_directory=rec["weights_directory"],
                                          velocity_rmse_m_s=rmse, velocity_mape_percent=mape, **metrics)
                        print(f"{table:26s} {row['pair_id']:38s} {mode}: RMSE {rmse:7.1f} MAPE {mape:5.2f} "
                              f"res {1000 * metrics['residual_rms_s']:6.2f} ms P90 {metrics['p90_map_m']:7.1f} "
                              f"mean {metrics['p90_weighted_mean_m']:7.1f}", flush=True)
                    entry[mode] = cache[key]
                out[table].append(entry)
    return out


def checkerboard41():
    sys.path.insert(0, str(HERE))
    import evaluate_checkerboard_velocity as ev41
    report = json.loads(CB41.read_text())
    spacing = report["method"]["recommended_spacing_km"]
    rows = []
    for row in report["rows"]:
        em_dir, dd_dir = Path(row["em_run"]), Path(row["dd_run"])
        with np.load(row["truth_model"], allow_pickle=False) as data:
            truth = data["velocity"].astype(float)
            cell = float(data["cell_size_m"]) / 1000
        em = np.load(em_dir / "final_model.npy", allow_pickle=False)
        axes = tuple(np.array(a) for a in row["dd_axes_zyx_km"])
        with np.load(dd_dir / "state.npz", allow_pickle=False) as state:
            nodes = {k: state[v] * 1000 for k, v in (("dd", "final_vp"), ("dd_oracle", "oracle_vp"))}
        from scipy.interpolate import RegularGridInterpolator
        shape = np.rint(ev41.DOMAIN / spacing).astype(int)
        centers = [(np.arange(n) + 0.5) * spacing for n in shape]
        interps = {k: RegularGridInterpolator(axes, v, method="linear", bounds_error=True) for k, v in nodes.items()}
        sq = dict.fromkeys(("em", *nodes), 0.0)
        ab = dict.fromkeys(("em", *nodes), 0.0)
        count = 0
        for start in range(0, shape[0], 4):
            xyz = np.meshgrid(centers[0][start:start + 4], centers[1], centers[2], indexing="ij")
            target = truth[tuple(np.floor(a / cell).astype(int) for a in xyz)]
            preds = {"em": em[tuple(np.floor(a).astype(int) for a in xyz)]}
            pts = np.stack(xyz[::-1], axis=-1)
            for k, f in interps.items():
                preds[k] = f(pts)
            for k, p in preds.items():
                sq[k] += float(np.sum((p - target) ** 2))
                ab[k] += float(np.sum(np.abs(p - target) / target))
            count += target.size
        rmse = {k: float(np.sqrt(v / count)) for k, v in sq.items()}
        for k, v in rmse.items():
            require(np.isclose(v, row["recommended_rmse_m_s"][k], rtol=1e-9), f"4.1 RMSE mismatch nx={row['nx']} {k}")
        mape = {k: float(100 * v / count) for k, v in ab.items()}
        print(f"4.1 nx={row['nx']}: RMSE {rmse} MAPE {mape}", flush=True)
        rows.append(dict(nx=row["nx"], spacing_km=spacing, samples=count, rmse_m_s=rmse, mape_percent=mape))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checkerboard41", action="store_true")
    args = parser.parse_args()
    report = dict(
        schema_version=1,
        description=__doc__,
        sources_sha256={str(p.relative_to(PAPER)): sha(p) for p in (LCURVE, FINAL, CB41)},
        script_sha256=sha(__file__),
        residual_definition="Maximum-weight hypothesis; r_a - mean(r) per event from saved pairwise "
                            "differences; pooled RMS over all saved picks; seconds",
        location_definition="P90 of 3-D errors, equal event weights, numpy linear percentile; MAP = largest "
                            "saved weight; weighted mean = sum(w * position) / sum(w)",
        velocity_definition="Nested common grid, coarser field repeated; equal volume weights; "
                            "MAPE = 100 mean(|v - v_true| / v_true)",
        tables=hard_soft_tables())
    if not args.skip_checkerboard41:
        report["checkerboard_4_1"] = checkerboard41()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
