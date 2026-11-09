"""Read-only comparison of completed cropped-domain soft/hard noise-grid runs.

No forward solve, inversion, or relocation is executed. Reports are created
exclusively; existing result files and reports are never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


STATE = Path(__file__).resolve().parent
BASE = STATE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def velocity_scores(velocity, truth):
    difference = velocity - truth
    return {"rmse_m_s": float(np.sqrt(np.mean(difference ** 2))),
            "mae_m_s": float(np.mean(np.abs(difference))),
            "mape_pct": float(np.mean(np.abs(difference) / truth) * 100)}


def main():
    targets = [STATE / "comparison.json", STATE / "comparison.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Comparison report already exists")
    manifest = read(STATE / "manifest.json")
    if read(STATE / "status.json")["state"] != "completed":
        raise ValueError("This report requires every planned run to be completed")
    helper_path = BASE / "paper/paper_scripts/evaluate_metrics.py"
    spec = importlib.util.spec_from_file_location("saved_distribution_metrics", helper_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    root = Path(manifest["experiments_root"])
    rows = []
    first_truth = first_ids = first_xyz = first_initial = None
    for case in sorted(manifest["cases"], key=lambda c: (c["multiplier"], c["mode"])):
        result = read(STATE / "cases" / case["id"] / "result.json")
        if result["status"] != "completed":
            raise ValueError(f"Incomplete case: {case['id']}")
        rd = Path(result["run_dir"])
        meta = read(rd / "meta.json")
        for name, checksum in case["input_sha256"].items():
            if sha(root / "input" / case["experiment_id"] / name) != checksum:
                raise ValueError("Experiment input changed")
        if sha(rd / "final_model.npy") != result["final_model_sha256"]:
            raise ValueError("Final model changed since run completion")
        if meta["source_experiment"]["id"] != case["experiment_id"]:
            raise ValueError("Run references another experiment")
        for key, value in case["config"].items():
            if key not in ("runs_dir", "save_runs") and meta["run_params"][key] != value:
                raise ValueError(f"Run config mismatch: {case['id']}/{key}")
        completed = sorted(int(p.parent.name.split("_")[1]) for p in rd.glob("iter_*/complete.json"))
        if completed != list(range(12)):
            raise ValueError("Not exactly twelve completed iterations")
        input_dir = root / "input" / case["experiment_id"]
        with np.load(input_dir / "model.npz", allow_pickle=False) as saved:
            coarse_truth = saved["velocity"]
            factor = float(saved["cell_size_m"]) / meta["grid_info"]["coarse_cell_size"]
        if factor != 5:
            raise ValueError("Expected five inversion cells per 4-km truth block")
        truth = coarse_truth
        for axis in range(3):
            truth = np.repeat(truth, 5, axis=axis)
        with (input_dir / "events.csv").open(newline="") as stream:
            events = list(csv.DictReader(stream))
        ids = [event["event_id"] for event in events]
        xyz = np.array([[float(event[k]) for k in ("x_m", "y_m", "z_m")] for event in events])
        if len(events) != 3465:
            raise ValueError("Unexpected event count")
        initial = np.load(rd / "initial_model.npy", allow_pickle=False)
        final = np.load(rd / "final_model.npy", allow_pickle=False)
        if initial.shape != truth.shape or final.shape != truth.shape:
            raise ValueError("Velocity grids differ")
        if first_truth is None:
            first_truth, first_ids, first_xyz, first_initial = truth, ids, xyz, initial
        else:
            np.testing.assert_array_equal(truth, first_truth)
            np.testing.assert_array_equal(xyz, first_xyz)
            np.testing.assert_array_equal(initial, first_initial)
            if ids != first_ids:
                raise ValueError("Catalog order differs between noise levels")
        velocity = velocity_scores(final, truth)
        logged = [json.loads(line) for line in (rd / "quality.jsonl").read_text().splitlines()]
        if [row["iter"] for row in logged] != list(range(12)):
            raise ValueError("Unexpected quality checkpoints")
        np.testing.assert_allclose(velocity["rmse_m_s"], logged[-1]["rms_m_s"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(velocity["mape_pct"], logged[-1]["avg_abs_pct_dev"], rtol=0, atol=1e-9)
        fine = meta["grid_info"]["fine_cell_size"]
        statistics, horizontal_squared, vertical_squared, effective = [], [], [], []
        for event, truth_position in enumerate(xyz):
            with np.load(rd / "iter_11" / f"event_{event}" / "weights.npz", allow_pickle=False) as saved:
                positions = metrics.positions_to_metres(saved["positions"], fine)
                weights = saved["weight_values"].astype(float)
            np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)
            # Zero-mass hypotheses do not affect any distribution metric.
            active = weights > 0
            positions, weights = positions[active], weights[active]
            weights /= weights.sum()
            stat = metrics.distribution_statistics(positions, weights, truth_position)
            statistics.append(stat)
            error = positions[np.argmax(weights)] - truth_position
            horizontal_squared.append(float(error[:2] @ error[:2]))
            vertical_squared.append(float(error[2] ** 2))
            effective.append(float(1 / (weights @ weights)))
        location = metrics.summarize_locations(statistics)
        location.update(map_horizontal_rms_m=float(np.sqrt(np.mean(horizontal_squared))),
                        map_depth_rms_m=float(np.sqrt(np.mean(vertical_squared))),
                        effective_hypotheses_mean=float(np.mean(effective)),
                        effective_hypotheses_median=float(np.median(effective)))
        timing = read(rd / "timing_summary.json")
        row = dict(case_id=case["id"], noise_multiplier=case["multiplier"], mode=case["mode"],
                   run_dir=str(rd), velocity=velocity, location=location,
                   wall_time_s=timing["total_s"], mean_iteration_s=timing["mean_iter_s"],
                   final_model_sha256=result["final_model_sha256"])
        rows.append(row)
        print(f"Evaluated {case['id']}: velocity RMSE={velocity['rmse_m_s']:.3f} m/s, "
              f"MAP RMS={location['map_rms_m']:.1f} m, ES={location['energy_score_mean_m']:.1f} m", flush=True)
    pairs = []
    for factor in sorted({r["noise_multiplier"] for r in rows}):
        hard = next(r for r in rows if r["noise_multiplier"] == factor and r["mode"] == "hard")
        soft = next(r for r in rows if r["noise_multiplier"] == factor and r["mode"] == "soft")
        pairs.append(dict(noise_multiplier=factor,
                          soft_velocity_rmse_change_pct=100 * (soft["velocity"]["rmse_m_s"] /
                                                               hard["velocity"]["rmse_m_s"] - 1),
                          soft_map_rms_change_pct=100 * (soft["location"]["map_rms_m"] /
                                                         hard["location"]["map_rms_m"] - 1),
                          soft_energy_score_change_pct=100 * (soft["location"]["energy_score_mean_m"] /
                                                              hard["location"]["energy_score_mean_m"] - 1),
                          soft_wall_time_ratio=soft["wall_time_s"] / hard["wall_time_s"]))
    report = dict(checkpoints={"velocity": "after 12 M-steps: final_model.npy",
                               "locations": "iter_11 E-step, before the 12th M-step; no final relocation"},
                  velocity_region="entire physical domain, 25 x 10 x 10 equal-volume 800-m cells",
                  coordinate_conversion="(saved continuous positions + 0.5) * fine_cell_size_m",
                  energy_score="mean_events[sum(w*d_to_truth) - 0.5*sum(w_i*w_j*d_ij)]; lower is better",
                  runtime_caveat="Observed wall time during concurrent runs, not a dedicated benchmark",
                  initial_velocity=velocity_scores(first_initial, first_truth),
                  assumed_noise=manifest["assumed_noise"], events=3465, stations=26,
                  helper_source_sha256=sha(helper_path), manifest_sha256=sha(STATE / "manifest.json"),
                  rows=rows, pairs=pairs)
    lines = ["# Cropped Geysers noise grid: hard / soft comparison", "",
             "All ten cases completed 12 iterations. Lower errors are better.", "",
             "Velocity: full-domain final model after 12 updates. Locations: last saved E-step BEFORE update 12.", "",
             "## Velocity and observed runtime", "",
             "| Noise | RMSE hard (m/s) | RMSE soft (m/s) | Soft change (%) | MAE hard (m/s) | MAE soft (m/s) | Time hard / soft (h) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        factor = pair["noise_multiplier"]
        h, s = [next(r for r in rows if r["noise_multiplier"] == factor and r["mode"] == mode)
                for mode in ("hard", "soft")]
        lines.append(f"| x{factor:g} | {h['velocity']['rmse_m_s']:.1f} | {s['velocity']['rmse_m_s']:.1f} | "
                     f"{pair['soft_velocity_rmse_change_pct']:+.2f} | {h['velocity']['mae_m_s']:.1f} | "
                     f"{s['velocity']['mae_m_s']:.1f} | {h['wall_time_s']/3600:.2f} / {s['wall_time_s']/3600:.2f} |")
    lines += ["", "## Locations: values are hard / soft, km", "",
              "| Noise | MAP RMS | MAP P95 | Weighted-mean RMS | Energy score |",
              "|---|---:|---:|---:|---:|"]
    for pair in pairs:
        factor = pair["noise_multiplier"]
        h, s = [next(r for r in rows if r["noise_multiplier"] == factor and r["mode"] == mode)["location"]
                for mode in ("hard", "soft")]
        values = [f"{h[k]/1000:.3f} / {s[k]/1000:.3f}" for k in
                  ("map_rms_m", "map_p95_m", "weighted_mean_rms_m", "energy_score_mean_m")]
        lines.append(f"| x{factor:g} | " + " | ".join(values) + " |")
    with targets[0].open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with targets[1].open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
