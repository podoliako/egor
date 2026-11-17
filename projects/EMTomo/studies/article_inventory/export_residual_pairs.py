"""Pair core article sections 2--6 and export independent per-run residual minima.

No FMM or inversion calls. Requires residual_metrics.py's fully extracted cache.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

STATE = Path(__file__).resolve().parent
EMTOMO = STATE.parents[1]
REPOSITORY = EMTOMO.parents[1]
OUTPUT = EMTOMO / "paper/exp_data"
SECTIONS = set("23456")

# Temperatures/schedules are intentionally allowed to differ: existing screening
# experiments explicitly reuse one hard control for multiple soft temperatures.
SCIENTIFIC_FIELDS = (
    "cell_size", "subdivision", "slowness_interpolation", "n_cycles", "n_candidates",
    "weights_top_n", "weights_min_distance", "initial_velocity_m_s",
    "initial_gradient_m_s", "initial_layer_boundaries_km", "initial_layer_velocities_m_s",
    "weight_noise_relative_sigma", "weight_noise_absolute_sigma_s", "weight_model_sigma_s",
    "lambda_reg", "coverage_damping_power", "coverage_floor", "coverage_reference_percentile",
    "max_velocity_step_fraction", "n_events", "weight_likelihood", "normal_equations",
    "candidate_selection", "travel_times", "control_fraction", "control_seed",
    "control_min_training_picks", "early_stopping_patience",
    "early_stopping_min_relative_improvement", "early_stopping", "control_prediction",
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pairing_key(row, meta, initial_hash, control_hash):
    params = meta["run_params"]
    source = meta.get("source_experiment") or {}
    return canonical(dict(
        section=row["section"], source_id=source.get("id"),
        model_sha256=source.get("model_sha256"), initial_sha256=initial_hash,
        control_mask_sha256=control_hash, stations=meta.get("station_locs"),
        grid=meta.get("grid_info"), version=meta.get("run_version", params.get("run_version")),
        # Keep the stopped pre-parallel and successful p12 attempts separate.
        cohort=source.get("study_manifest"),
        scientific={key: params.get(key) for key in SCIENTIFIC_FIELDS},
    ))


def find_pairs(rows, keys, explicit_controls):
    hard_by_key = defaultdict(list)
    by_id = {r["run_id"]: r for r in rows}
    for row in rows:
        if row["mode"] == "hard":
            hard_by_key[keys[row["run_id"]]].append(row)
    pairs = []
    for soft in rows:
        if soft["mode"] != "soft":
            continue
        controls = hard_by_key[keys[soft["run_id"]]]
        explicit = explicit_controls.get(soft["run_id"])
        if explicit is not None:
            if explicit not in by_id or keys[explicit] != keys[soft["run_id"]]:
                raise ValueError(f"Manifest control not comparable: {soft['run_id']} -> {explicit}")
            controls = [by_id[explicit]]
        if len(controls) > 1:
            raise ValueError(f"Ambiguous hard controls for {soft['run_id']}: {[r['run_id'] for r in controls]}")
        if controls:
            pairs.append((controls[0], soft))
    used = {r["run_id"] for pair in pairs for r in pair}
    return pairs, [row for row in rows if row["run_id"] not in used]


def metric_columns(prefix, row, record):
    best = record.get("best")
    trajectory = record.get("trajectory", [])
    best_viewer = min(trajectory, key=lambda p: (p["mean_event_rms_s"], p["iteration"]), default=None)
    return {
        f"{prefix}_run_id": row["run_id"],
        f"{prefix}_min_residual_rmse_s": best["rmse_s"] if best else None,
        f"{prefix}_best_iteration_index": best["iteration"] if best else None,
        f"{prefix}_best_iteration_number": best["iteration"] + 1 if best else None,
        f"{prefix}_min_mean_event_rms_s": best_viewer["mean_event_rms_s"] if best_viewer else None,
        f"{prefix}_best_mean_event_iteration_index": best_viewer["iteration"] if best_viewer else None,
        f"{prefix}_complete_iterations": record["n_complete_iterations"],
        f"{prefix}_has_final_model": row["final_model_exists"],
        f"{prefix}_n_pairs_at_minimum": best["n_pairs"] if best else None,
        f"{prefix}_metric_status": record["status"],
        f"{prefix}_metric_missing_reason": record.get("reason"),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    inventory_path = STATE / "inventory.json"
    cache = read(OUTPUT / "residual_metrics_cache.json")
    if cache["inventory_sha256"] != sha(inventory_path):
        raise ValueError("Metric cache belongs to a different inventory snapshot")
    rows = [r for r in read(inventory_path)["runs"] if r["section"] in SECTIONS]
    metas, keys = {}, {}
    for row in rows:
        directory = REPOSITORY / row["run_dir"]
        meta = read(directory / "meta.json")
        metas[row["run_id"]] = meta
        control = directory / "control_mask.npy"
        keys[row["run_id"]] = pairing_key(
            row, meta, sha(directory / "initial_model.npy"), sha(control) if control.exists() else None,
        )
        record = cache["runs"][row["run_id"]]
        if record["status"] not in ("ok", "absent"):
            raise ValueError(f"Unresolved extraction error: {row['run_id']}")
        if record["signature"]["meta_sha256"] != sha(directory / "meta.json"):
            raise ValueError(f"Metadata changed since extraction: {row['run_id']}")
        actual_iterations = sorted(int(p.parent.name[5:]) for p in directory.glob("iter_*/complete.json"))
        recorded_iterations = [p["iteration"] for p in record["signature"]["complete_markers"]]
        if actual_iterations != recorded_iterations:
            raise ValueError(f"Completed iterations changed since extraction: {row['run_id']}")
    screen = EMTOMO / "runs/geysers_faulted_layers/soft_screen_20261028"
    manifest = read(screen / "manifest.json")
    results = {c["id"]: read(screen / "cases" / c["id"] / "result.json") for c in manifest["cases"]}
    controls = {
        Path(results[c["id"]]["run_dir"]).name: Path(results[c["hard_control"]]["run_dir"]).name
        for c in manifest["cases"] if c.get("hard_control")
    }
    pairs, unpaired = find_pairs(rows, keys, controls)
    hard_uses = Counter(h["run_id"] for h, _ in pairs)
    output_rows = []
    for hard, soft in pairs:
        hp = metas[hard["run_id"]]["run_params"]
        sp = metas[soft["run_id"]]["run_params"]
        hr, sr = cache["runs"][hard["run_id"]], cache["runs"][soft["run_id"]]
        have_both = hr.get("best") is not None and sr.get("best") is not None
        status = ("complete_runs" if hard["final_model_exists"] and soft["final_model_exists"] else
                  "partial_runs" if have_both else "missing_one_or_both_metrics")
        source = metas[soft["run_id"]].get("source_experiment") or {}
        result = dict(
            section=int(soft["section"]), section_name=soft["section_name"],
            experiment_id=soft["source_id"], pair_status=status,
            hard_control_reuse_count=hard_uses[hard["run_id"]],
            **metric_columns("hard", hard, hr), **metric_columns("soft", soft, sr),
            n_events=soft["n_events"], n_stations=soft["n_stations"],
            n_candidates=sp.get("n_candidates"), weights_top_n=sp.get("weights_top_n"),
            weights_min_distance=sp.get("weights_min_distance"),
            soft_temperature=sp.get("temperature"), hard_temperature=hp.get("temperature"),
            soft_temperature_schedule=canonical(sp.get("temperature_schedule")),
            hard_temperature_schedule=canonical(hp.get("temperature_schedule")),
            lambda_reg=sp.get("lambda_reg"), coverage_damping_power=sp.get("coverage_damping_power"),
            assumed_relative_sigma=sp.get("weight_noise_relative_sigma"),
            assumed_absolute_sigma_s=sp.get("weight_noise_absolute_sigma_s"),
            assumed_model_sigma_s=sp.get("weight_model_sigma_s"),
            initial_kind=soft["initial_kind"], requested_updates=sp.get("n_cycles"),
            run_version=soft["version"], control_fraction=sp.get("control_fraction", 0.),
            study_manifest=source.get("study_manifest", ""),
            comparison_basis="independent_per_run_minimum_over_complete_E_steps",
        )
        if have_both:
            h, s = hr["best"]["rmse_s"], sr["best"]["rmse_s"]
            result["soft_gain_percent"] = 100 * (h - s) / h if h > 0 else None
        else:
            result["soft_gain_percent"] = None
        output_rows.append(result)
    output_rows.sort(key=lambda r: (r["section"], r["experiment_id"], r["soft_run_id"]))
    counters = Counter()
    for row in output_rows:
        counters[row["section"]] += 1
        row["pair_id"] = f"section{row['section']}_pair{counters[row['section']]:02d}"
    # Make the main identifier visible before the wide provenance columns.
    output_rows = [{"pair_id": r["pair_id"], **r} for r in output_rows]
    usable = [r for r in output_rows if r["hard_metric_status"] == r["soft_metric_status"] == "ok"]
    incomplete = [r for r in output_rows if r not in usable]
    write_csv(OUTPUT / "hard_soft_min_residual_rmse.csv", usable)
    write_csv(OUTPUT / "hard_soft_incomplete_pairs.csv", incomplete)
    unpaired_rows = []
    for row in unpaired:
        record = cache["runs"][row["run_id"]]
        unpaired_rows.append(dict(section=row["section"], section_name=row["section_name"],
                                  experiment_id=row["source_id"], mode=row["mode"],
                                  reason="no_opposite_mode_with_matching_data_initial_model_and_settings",
                                  **metric_columns("run", row, record)))
    write_csv(OUTPUT / "hard_soft_unpaired_runs.csv", unpaired_rows)
    used = {r["run_id"] for pair in pairs for r in pair}
    assert used | {r["run_id"] for r in unpaired} == {r["run_id"] for r in rows}
    assert not used & {r["run_id"] for r in unpaired}
    print(json.dumps(dict(pairs=len(pairs), usable_pairs=len(usable), incomplete_pairs=len(incomplete),
                          usable_by_section=dict(Counter(r["section"] for r in usable)),
                          by_section=dict(counters),
                          pair_statuses=dict(Counter(r["pair_status"] for r in output_rows)),
                          unique_paired_runs=len(used), unpaired_runs=len(unpaired),
                          all_accounted_core_runs=len(rows)), indent=2))


if __name__ == "__main__":
    main()
