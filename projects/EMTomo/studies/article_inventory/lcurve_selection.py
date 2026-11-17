"""Training-only, geometric L-curve selection; truth metrics are read afterwards."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STATE = Path(__file__).resolve().parent
EMTOMO = STATE.parents[1]
OUTPUT = EMTOMO / "paper/exp_data"
METRIC = "avg_abs_pct_dev"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_elbow(points):
    """Maximum perpendicular chord distance in normalized log--log coordinates.

    Each point has positive residual RMS and positive relative slowness change.
    With fewer than three usable points, a degenerate axis, or a straight curve,
    choose the last available point and explicitly report the fallback.
    """
    if not points:
        raise ValueError("No positive-norm model checkpoints available")
    values = np.asarray([[p["residual_rmse_s"], p["relative_slowness_change"]] for p in points])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("L-curve coordinates must be finite and positive")
    distances = np.zeros(len(points), dtype=float)
    reason = "insufficient_points_use_last"
    selected = len(points) - 1
    if len(points) >= 3:
        logs = np.log10(values)
        span = np.ptp(logs, axis=0)
        reason = "degenerate_curve_use_last"
        if np.all(span > 1e-12):
            normalized = (logs - logs.min(axis=0)) / span
            start, end = normalized[0], normalized[-1]
            chord = end - start
            reason = "coincident_endpoints_use_last"
            if np.linalg.norm(chord) > 1e-12:
                offsets = normalized - start
                distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / np.linalg.norm(chord)
                distances[[0, -1]] = 0.
                interior = distances[1:-1]
                reason = "straight_curve_use_last"
                if interior.max() > 1e-12:
                    selected = int(np.argmax(interior)) + 1  # Earliest exact tie.
                    reason = "maximum_log_chord_distance"
    return selected, reason, distances


def select_run(directory, record):
    """Use only saved training residuals, initial model and model checkpoints."""
    directory = Path(directory)
    if record["status"] != "ok":
        raise ValueError(f"No residual trajectory: {directory.name}")
    if sha(directory / "meta.json") != record["signature"]["meta_sha256"]:
        raise ValueError(f"Residual cache metadata mismatch: {directory.name}")
    actual = sorted(int(p.parent.name[5:]) for p in directory.glob("iter_*/complete.json"))
    if actual != [p["iteration"] for p in record["trajectory"]]:
        raise ValueError(f"Residual cache iteration mismatch: {directory.name}")
    initial = np.asarray(np.load(directory / "initial_model.npy", allow_pickle=False), dtype=np.float64)
    if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError(f"Invalid initial model: {directory.name}")
    s0 = 1. / initial
    norm0 = float(np.linalg.norm(s0.ravel()))
    points = []
    skipped = []
    for residual in record["trajectory"]:
        k = residual["iteration"]
        path = directory / f"iter_{k}/model.npy"
        model = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if model.shape != initial.shape or not np.all(np.isfinite(model)) or np.any(model <= 0):
            raise ValueError(f"Invalid model checkpoint: {path}")
        relative_change = float(np.linalg.norm((1. / model - s0).ravel()) / norm0)
        # V0 has zero regularization norm and cannot be placed on a logarithmic axis.
        if relative_change <= 0 or k == 0:
            skipped.append(k)
            continue
        points.append(dict(residual_iteration_index=k, completed_velocity_updates=k,
                           residual_rmse_s=residual["rmse_s"], relative_slowness_change=relative_change,
                           model_sha256=sha(path)))
    index, reason, distances = choose_elbow(points)
    for point, distance in zip(points, distances):
        point["normalized_chord_distance"] = float(distance)
    return dict(run_id=directory.name, selected=points[index], selection_reason=reason,
                n_positive_norm_checkpoints=len(points), skipped_zero_norm_or_initial_indices=skipped,
                initial_model_sha256=sha(directory / "initial_model.npy"),
                residual_cache_signature=record["signature"]["sha256"], trajectory=points)


def selected_quality(directory, selection):
    """Align residual/model V_k with quality.jsonl entry k-1, NOT entry k."""
    update = selection["selected"]["completed_velocity_updates"]
    index = update - 1
    values = [json.loads(line) for line in (directory / "quality.jsonl").read_text().splitlines() if line.strip()]
    matches = [value for value in values if value["iter"] == index]
    if len(matches) != 1 or not np.isfinite(matches[0][METRIC]):
        raise ValueError(f"Missing/invalid selected post-update quality: {directory.name}, update={update}")
    # The next iteration's saved model must be exactly the reported selected V_k.
    return float(matches[0][METRIC]), index


def main():
    source = OUTPUT / "hard_soft_min_residual_rmse.csv"
    with source.open() as stream:
        pairs = list(csv.DictReader(stream))
    cache_path = OUTPUT / "residual_metrics_cache.json"
    cache = read(cache_path)
    selections = {}
    # Freeze ALL choices before opening any truth-derived quality.jsonl file.
    for pair in pairs:
        for mode in ("hard", "soft"):
            run_id = pair[f"{mode}_run_id"]
            if run_id not in selections:
                selections[run_id] = select_run(EMTOMO / "runs" / run_id, cache["runs"][run_id])
    output_rows = []
    excluded = []
    fields = ("pair_id", "section", "section_name", "experiment_id", "hard_control_reuse_count",
              "n_events", "n_stations", "n_candidates", "weights_top_n", "weights_min_distance",
              "soft_temperature", "hard_temperature", "soft_temperature_schedule", "hard_temperature_schedule",
              "lambda_reg", "coverage_damping_power", "initial_kind", "requested_updates", "run_version",
              "control_fraction", "study_manifest")
    quality_hashes = {}
    for pair in pairs:
        row = {field: pair[field] for field in fields}
        homogeneous_starts = []
        for mode in ("hard", "soft"):
            run_id = pair[f"{mode}_run_id"]
            directory = EMTOMO / "runs" / run_id
            selection = selections[run_id]
            value, quality_index = selected_quality(directory, selection)
            quality_hashes[run_id] = sha(directory / "quality.jsonl")
            selected = selection["selected"]
            row.update({f"{mode}_run_id": run_id,
                        f"{mode}_avg_abs_pct_dev_at_lcurve": value,
                        f"{mode}_selected_update_number": selected["completed_velocity_updates"],
                        f"{mode}_residual_iteration_index": selected["residual_iteration_index"],
                        f"{mode}_quality_iteration_index": quality_index,
                        f"{mode}_residual_rmse_s_at_lcurve": selected["residual_rmse_s"],
                        f"{mode}_relative_slowness_change_at_lcurve": selected["relative_slowness_change"],
                        f"{mode}_selection_reason": selection["selection_reason"],
                        f"{mode}_n_lcurve_checkpoints": selection["n_positive_norm_checkpoints"]})
            params = read(directory / "meta.json")["run_params"]
            initial = np.load(directory / "initial_model.npy", allow_pickle=False)
            homogeneous_starts.append(params.get("initial_gradient_m_s") is None
                                      and params.get("initial_layer_velocities_m_s") is None
                                      and np.all(initial == 5000.))
        h = row["hard_avg_abs_pct_dev_at_lcurve"]
        s = row["soft_avg_abs_pct_dev_at_lcurve"]
        if h <= 0:
            raise ValueError(f"Cannot compute relative gain: {pair['pair_id']}")
        row["soft_gain_percent"] = 100 * (h - s) / h
        row["metric"] = METRIC
        row["selection_uses_truth"] = False
        row["evaluation_uses_truth"] = True
        row["selection_rule"] = "max_distance_normalized_loglog_Lcurve"
        row["included_in_plot"] = not (pair["section"] == "6" and all(homogeneous_starts))
        row["plot_exclusion_reason"] = "complex_anomalies_homogeneous_5000_m_s_start" if not row["included_in_plot"] else ""
        if not row["included_in_plot"]:
            excluded.append(row["pair_id"])
        output_rows.append(row)
    if len(pairs) != 38 or len(excluded) != 2:
        raise ValueError(f"Unexpected comparison scope: pairs={len(pairs)}, exclusions={excluded}")
    target = OUTPUT / "hard_soft_lcurve_avg_abs_pct_dev.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(target)
    report = dict(rule="max_distance_normalized_loglog_Lcurve",
                  rule_description="Log10 residual RMS vs log10 ||s_k-s_0||_2/||s_0||_2; normalize each log axis to [0,1], maximize absolute perpendicular distance from first-to-last chord among interior checkpoints; earliest exact tie. Degenerate/straight curves or <3 checkpoints: last available checkpoint.",
                  selection_uses_truth=False, evaluation_uses_truth=True,
                  residual_timing="iter_k residual and model.npy refer to V_k (after k updates); evaluated quality index is k-1",
                  final_model_not_scored="No final relocation was computed; V_N is unavailable to the L-curve because residuals end at V_(N-1). Initial V_0 is excluded from log plot because its change norm is zero.",
                  source_pairs_sha256=sha(source), source_residual_cache_sha256=sha(cache_path),
                  csv_sha256=sha(target), excluded_plot_pair_ids=excluded,
                  quality_sha256_by_run=quality_hashes, runs=selections,
                  selection_reason_counts=dict(Counter(s["selection_reason"] for s in selections.values())))
    (OUTPUT / "lcurve_selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plotted = [r for r in output_rows if r["included_in_plot"]]
    print(json.dumps(dict(csv=str(target), csv_pairs=len(output_rows), plotted_pairs=len(plotted),
                          excluded_pair_ids=excluded, unique_runs=len(selections),
                          plotted_soft_better=sum(r["soft_gain_percent"] > 0 for r in plotted),
                          selection_reason_counts=report["selection_reason_counts"]), indent=2))


if __name__ == "__main__":
    main()
