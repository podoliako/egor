"""Markdown summary of the dip6-* EMTomo runs on one saved experiment.

Usage (from projects/EMTomo): python studies/dipping6spheres/report.py EXPERIMENT_ID
Truth is used here only for evaluation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_data import DEFAULT_EXPERIMENTS_ROOT, load_tomography_experiment  # noqa: E402
from experiment_runner import prepare_inversion  # noqa: E402
from main import CONFIG  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs"


def _rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _hypocentre_errors_km(run_dir: Path, iteration: int, truth_m: np.ndarray, fine_cell: float):
    errors = []
    for event, true in enumerate(truth_m):
        with np.load(run_dir / f"iter_{iteration}" / f"event_{event}" / "weights.npz") as saved:
            best = saved["positions"][int(np.argmax(saved["weight_values"]))]
        errors.append(np.linalg.norm((best + 0.5) * fine_cell - true) / 1000)
    return np.asarray(errors)


def _sphere_recovery(final, initial, reference, cell_m, anomalies):
    centres = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * cell_m / 1000 for n in final.shape],
                                   indexing="ij"), axis=-1)
    rows = []
    for anomaly in anomalies:
        core = np.linalg.norm(centres - np.asarray(anomaly["center_km"]), axis=-1) < 0.6 * anomaly["radius_km"]
        if not core.any():
            rows.append(np.nan)
            continue
        expected = np.mean(reference[core] - initial[core])
        rows.append(float(np.mean(final[core] - initial[core]) / expected) if expected else np.nan)
    return rows


def main(experiment_id: str, experiments_root=DEFAULT_EXPERIMENTS_ROOT, runs_root=RUNS) -> None:
    experiments_root, runs_root = Path(experiments_root), Path(runs_root)
    data = load_tomography_experiment(experiment_id, experiments_root)
    generation = json.loads((experiments_root / "input" / experiment_id / "generation.json").read_text())
    anomalies = generation["parameters"]["anomalies"]
    metadata = json.loads((experiments_root / "output" / experiment_id / "metadata.json").read_text())
    runs = sorted(d for d in runs_root.glob("run_dip6-*") if (d / "meta.json").exists()
                  and json.loads((d / "meta.json").read_text()).get("source_experiment", {}).get("id") == experiment_id)

    print(f"# EMTomo on `{experiment_id}`\n")
    print(f"{len(data.station_ids)} stations, {len(data.event_ids)} events.")
    print(f"Forward convergence (refinement r→2r): `{json.dumps(metadata.get('convergence'))}`\n")

    summary, curves, hypo_rows = [], {}, []
    for run_dir in runs:
        meta = json.loads((run_dir / "meta.json").read_text())
        params = meta["run_params"]
        name = params["run_name"].removeprefix("dip6-").removesuffix(f"_{experiment_id}")
        reference = prepare_inversion(
            experiment_id, replace(CONFIG, cell_size=params["coarse_side_m"]), experiments_root,
        ).reference_model.velocity
        initial = np.load(run_dir / "initial_model.npy")
        quality = [json.loads(line) for line in (run_dir / "quality.jsonl").read_text().splitlines() if line]
        if not (run_dir / "final_model.npy").exists():
            summary.append(f"| {name} | unfinished ({len(quality)} cycles) | | | | | |")
            continue
        final = np.load(run_dir / "final_model.npy")
        rms = [_rms(initial, reference)] + [row["rms_m_s"] for row in quality]
        curves[name] = rms
        best = int(np.argmin(rms))
        timing = json.loads((run_dir / "timing_summary.json").read_text())
        recovery = _sphere_recovery(final, initial, reference, params["coarse_side_m"], anomalies)
        fine = meta["grid_info"]["fine_cell_size"]
        first = _hypocentre_errors_km(run_dir, 0, data.reference_event_coordinates_m, fine)
        last = _hypocentre_errors_km(run_dir, len(quality) - 1, data.reference_event_coordinates_m, fine)
        summary.append(
            f"| {name} | {rms[0]:.1f} | {rms[-1]:.1f} | {rms[best]:.1f} @ {best} | "
            + " ".join(f"{value:.2f}" for value in recovery)
            + f" | {np.median(first):.2f} → {np.median(last):.2f} | {timing['mean_iter_s']:.0f} |"
        )
        hypo_rows.append(f"| {name} | {np.percentile(last, 90):.2f} | {np.mean(last > 5):.1%} |")

    print("RMS is m/s against the slowness-averaged truth on inversion cells; cycle 0 is the start model.")
    print("Sphere recovery = mean(final − start) / mean(truth − start) over inversion cells within 0.6 R.\n")
    print("| run | RMS start | RMS final | best RMS @ cycle | sphere recovery A–F | median hypo error km, first → last cycle | s / cycle |")
    print("|---|---:|---:|---:|---|---:|---:|")
    print("\n".join(summary))
    print("\n## Hypocentre errors in the last cycle (most likely hypothesis)\n")
    print("| run | 90th percentile, km | share > 5 km |\n|---|---:|---:|")
    print("\n".join(hypo_rows))
    print("\n## RMS (m/s) by cycle\n")
    if curves:
        n = max(len(c) for c in curves.values())
        print("| run | " + " | ".join(str(i) for i in range(n)) + " |")
        print("|---|" + "---:|" * n)
        for name, curve in curves.items():
            print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in curve) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:4])
