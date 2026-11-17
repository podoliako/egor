"""Refresh the top-N comparison export without launching any inversions.

Missing case results mean pending. Completed results must provide run_id, with
artifacts under EMTomo/runs/<run_id>. Historical residuals are never extracted.
Legacy shared configuration columns describe the soft run; explicit per-mode
K/top-N columns disambiguate reused hard controls. Call export_study() or main().
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
PROJECT = STUDY.parents[1]
sys.path.insert(0, str(STUDY.parent / "article_inventory"))
import residual_metrics as residual
import lcurve_selection as lcurve

EXTRA_FIELDS = ["hard_n_candidates", "soft_n_candidates", "hard_weights_top_n",
                "soft_weights_top_n", "pair_status"]
METRICS = ["avg_abs_pct_dev_at_lcurve", "selected_update_number", "residual_iteration_index",
           "quality_iteration_index", "residual_rmse_s_at_lcurve",
           "relative_slowness_change_at_lcurve", "selection_reason", "n_lcurve_checkpoints"]


def read(path):
    return json.loads(Path(path).read_text())


def export_study(study=STUDY, project=PROJECT, output=None):
    """Return exported rows; refresh result statuses on every call.

    Writes residual_metrics_cache.json, lcurve_selection.json (frozen before
    truth access), and export_provenance.json under study, plus the NEW CSV.
    Errors in completed artifacts become pair_status=error with blank metrics
    and an explanation in provenance; running/failed pairs are never evaluated.
    """
    study, project = Path(study).resolve(), Path(project).resolve()
    data = project / "paper/exp_data"
    source = data / "hard_soft_lcurve_avg_abs_pct_dev.csv"
    output = Path(output) if output else data / "hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv"
    if output.resolve() == source.resolve():
        raise ValueError("Cannot overwrite the historical CSV")
    manifest_path = study / "manifest.json"
    manifest = read(manifest_path)
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames) + EXTRA_FIELDS
        old_rows = list(reader)
    sources = {row["pair_id"]: row for row in old_rows}
    ids = list(sources) + [p["pair_id"] for p in manifest["pairs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate pair_id")
    metas = {}

    def meta(run_id):
        if run_id not in metas:
            metas[run_id] = read(project / "runs" / run_id / "meta.json")
        return metas[run_id]

    rows = []
    for old in old_rows:
        row = dict(old)
        for mode in ("hard", "soft"):
            params = meta(row[f"{mode}_run_id"])["run_params"]
            for key in ("n_candidates", "weights_top_n"):
                row[f"{mode}_{key}"] = params[key]
        row["pair_status"] = "completed"
        rows.append(row)

    case_states = {}
    for case_id in manifest["cases"]:
        path = study / "cases" / case_id / "result.json"
        result = read(path) if path.exists() else {"status": "pending"}
        if result["status"] not in ("pending", "running", "failed", "completed"):
            raise ValueError(f"Unknown case status: {case_id}: {result['status']}")
        if result["status"] == "completed" and not result.get("run_id"):
            raise ValueError(f"Completed case missing run_id: {case_id}")
        case_states[case_id] = result

    ready = {}
    for pair in manifest["pairs"]:
        row = dict(sources[pair["source_pair_id"]])
        row.update(pair_id=pair["pair_id"], study_manifest=str(manifest_path))
        for mode in ("hard", "soft"):
            for key in METRICS:
                row[f"{mode}_{key}"] = ""
        row["soft_gain_percent"] = ""
        states = []
        for mode in ("hard", "soft"):
            case_id = pair.get(f"{mode}_case_id")
            if case_id:
                case = manifest["cases"][case_id]
                state = case_states[case_id]
                params = case["config"]
                run_id = state.get("run_id", "")
                if mode == "soft":
                    row["experiment_id"] = case["experiment_id"]
            else:
                if mode != "hard" or not pair.get("hard_run_id"):
                    raise ValueError(f"Missing {mode} case/control: {pair['pair_id']}")
                run_id = pair["hard_run_id"]
                state = {"status": "completed"}
                params = meta(run_id)["run_params"]
            row[f"{mode}_run_id"] = run_id
            states.append(state["status"])
            for key in ("n_candidates", "weights_top_n"):
                row[f"{mode}_{key}"] = params[key]
            row[f"{mode}_temperature"] = params.get("temperature", "")
            row[f"{mode}_temperature_schedule"] = json.dumps(params.get("temperature_schedule"))
            if mode == "soft":
                for key in ("n_candidates", "weights_top_n", "weights_min_distance", "lambda_reg",
                            "coverage_damping_power", "control_fraction", "run_version", "n_events"):
                    if key in params:
                        row[key] = params[key]
                if "n_cycles" in params:
                    row["requested_updates"] = params["n_cycles"]
        row["pair_status"] = ("failed" if "failed" in states else "running" if "running" in states
                              else "pending" if "pending" in states else "completed")
        if row["pair_status"] == "completed":
            for mode in ("hard", "soft"):
                ready[row[f"{mode}_run_id"]] = (row, mode)
        rows.append(row)

    # Count reuse across the expanded comparison, without changing old values.
    uses = Counter(r["hard_run_id"] for r in rows if r["hard_run_id"])
    for row in rows[len(old_rows):]:
        row["hard_control_reuse_count"] = uses[row["hard_run_id"]] if row["hard_run_id"] else ""
    old_cache_path = data / "residual_metrics_cache.json"
    historical = read(old_cache_path) if ready else {"runs": {}}
    cache_path = study / "residual_metrics_cache.json"
    cache = read(cache_path) if cache_path.exists() else {"schema_version": residual.SCHEMA_VERSION, "runs": {}}
    selections, errors, origins = {}, {}, {}
    historical_ids = {r[f"{mode}_run_id"] for r in old_rows for mode in ("hard", "soft")}
    new_ids = {state["run_id"] for state in case_states.values()
               if state["status"] == "completed"}
    for run_id, (row, mode) in ready.items():
        directory = project / "runs" / run_id
        try:
            if run_id in historical["runs"]:
                record = historical["runs"][run_id]
                origins[run_id] = "historical_cache"
            else:
                if run_id in historical_ids or run_id not in new_ids:
                    raise ValueError("Historical run missing from residual cache; refusing extraction")
                params = meta(run_id)["run_params"]
                inventory_row = dict(run_id=run_id, run_dir=str(directory), section=row["section"],
                                     mode=mode, source_id=row["experiment_id"], n_events=params["n_events"])
                record = cache["runs"].get(run_id)
                signature = residual.snapshot_signature(directory, inventory_row)
                if not record or record.get("status") != "ok" or record.get("signature") != signature:
                    record = residual.extract_run(directory, inventory_row)
                    cache["runs"][run_id] = record
                    residual.atomic_write(cache_path, cache)
                origins[run_id] = "study_cache"
            selections[run_id] = lcurve.select_run(directory, record)
        except Exception as exc:
            errors[run_id] = f"{type(exc).__name__}: {exc}"
    residual.atomic_write(cache_path, cache)
    # Persist all training-only choices before any selected_quality call.
    selection_path = study / "lcurve_selection.json"
    residual.atomic_write(selection_path, dict(selection_uses_truth=False,
        residual_timing="residual/model V_k -> quality index k-1", runs=selections, errors=errors))
    qualities, quality_hashes = {}, {}
    for run_id, selection in selections.items():
        try:
            directory = project / "runs" / run_id
            qualities[run_id] = lcurve.selected_quality(directory, selection)
            quality_hashes[run_id] = lcurve.sha(directory / "quality.jsonl")
        except Exception as exc:
            errors[run_id] = f"{type(exc).__name__}: {exc}"
    for row in rows[len(old_rows):]:
        if row["pair_status"] != "completed":
            continue
        h, s = row["hard_run_id"], row["soft_run_id"]
        if h in errors or s in errors:
            row["pair_status"] = "error"
            continue
        if qualities[h][0] <= 0:
            errors[h] = "Cannot compute relative gain with nonpositive hard quality"
            row["pair_status"] = "error"
            continue
        for mode in ("hard", "soft"):
            run_id = row[f"{mode}_run_id"]
            selection = selections[run_id]
            selected = selection["selected"]
            value, index = qualities[run_id]
            values = [value, selected["completed_velocity_updates"], selected["residual_iteration_index"],
                      index, selected["residual_rmse_s"], selected["relative_slowness_change"],
                      selection["selection_reason"], selection["n_positive_norm_checkpoints"]]
            row.update({f"{mode}_{key}": value for key, value in zip(METRICS, values)})
        row["soft_gain_percent"] = 100 * (qualities[h][0] - qualities[s][0]) / qualities[h][0]
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    residual.atomic_write(study / "export_provenance.json", dict(
        updated_utc=residual.utc_now(), source_csv_sha256=lcurve.sha(source),
        manifest_sha256=lcurve.sha(manifest_path), csv_sha256=lcurve.sha(output),
        historical_cache_sha256=lcurve.sha(old_cache_path) if ready else None,
        selection_sha256=lcurve.sha(selection_path), residual_origins=origins,
        quality_sha256_by_run=quality_hashes, case_states=case_states, errors=errors,
        pair_status_counts=dict(Counter(r["pair_status"] for r in rows))))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = export_study(args.study, args.project, args.output)
    print(json.dumps(dict(rows=len(rows), statuses=dict(Counter(r["pair_status"] for r in rows)))))
    return 1 if any(r["pair_status"] in ("error", "failed") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
