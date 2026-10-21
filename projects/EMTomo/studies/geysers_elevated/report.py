"""Read-only elevated Geysers assessment; no forward, calibration, or EM solves.

API: generate_report(manifest_path, output_dir=None) -> JSON-safe summary.
CLI: python report.py --manifest /absolute/path/to/manifest.json
Frozen layout: runner/report.py beside runner/_shared_report.py (geysers2011).
Only stdlib is imported until the manifest selects production snapshot paths.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile


_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                UnicodeError, zipfile.BadZipFile)
_SETTINGS = ("candidate_mode", "n_candidates", "weights_top_n", "temperature")
_REGULARIZATION = ("lambda_reg", "coverage_damping_power")
_BASELINE = ("cell_size", "subdivision", "n_cycles", "initial_layer_boundaries_km",
             "initial_layer_velocities_m_s")
_REFERENCE_POLICY = "Latest historical saved configuration from reference_run/meta.json; not a new damping sweep"


def _object(path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_shared(manifest):
    root = Path(manifest["runtime_root"])
    if not root.is_absolute():
        raise ValueError("runtime_root must be absolute")
    # Same production precedence as run_case.load_runtime, before importing shared
    # (which imports numpy). runner is separate, never a production module root.
    paths = [str(root / "projects" / "EMTomo"), str(root), str(root / "runner")]
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    try:
        return importlib.import_module("_shared_report")
    except ModuleNotFoundError as error:
        if error.name != "_shared_report":
            raise
        return importlib.import_module("projects.EMTomo.studies.geysers2011.report")


def _temperature(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
        raise ValueError("temperature must be finite and >= 1")
    return value


def _calibration(path):
    result = {"path": str(path), "status": "unavailable", "temperature": None,
              "selection": {}, "per_k": {}, "warnings": []}
    try:
        value = _object(path)
        _temperature(value["temperature"])
        if not isinstance(value["selection"], dict) or not isinstance(value["per_k"], dict):
            raise ValueError("selection/per_k must be objects")
        if not isinstance(value.get("warnings", []), list):
            raise ValueError("warnings must be a list")
        result.update({key: value[key] for key in ("temperature", "selection", "per_k")})
        result.update(status="available", warnings=value.get("warnings", []),
                      limitations=value.get("limitations", []))
    except _READ_ERRORS as error:
        result["warnings"].append(f"calibration unavailable: {error}")
    return result


def _settings(case, calibration):
    mode, k = case["candidate_mode"], case["n_candidates"]
    if (mode not in ("hard", "soft") or type(k) is not int
            or k not in ((1, 5, 10, 15) if mode == "hard" else (5, 10, 15))):
        raise ValueError("invalid candidate mode/K")
    top_n = 1 if mode == "hard" else k
    if type(case["weights_top_n"]) is not int or case["weights_top_n"] != top_n:
        raise ValueError("hard requires top_n=1; soft requires top_n=K")
    if (type(case["lambda_reg"]) not in (int, float) or case["lambda_reg"] != 1
            or type(case["coverage_damping_power"]) not in (int, float)
            or case["coverage_damping_power"] != 2):
        raise ValueError("elevated comparison requires fixed lambda_reg=1 / coverage_damping_power=2")
    temperature = case["temperature"]
    if mode == "hard":
        if _temperature(temperature) != 1:
            raise ValueError("hard requires T=1")
    else:
        if temperature is not None:
            raise ValueError("soft manifest temperature must be null (resolved by calibration)")
        temperature = calibration["temperature"]
    return {"candidate_mode": mode, "n_candidates": k,
            "weights_top_n": top_n, "temperature": temperature}


def _check_config(actual, expected, keys):
    if not isinstance(actual, dict):
        raise ValueError("invalid effective configuration")
    for key in keys:
        if key in expected and expected[key] is not None and actual.get(key) != expected[key]:
            raise ValueError(f"run {key} differs from expected configuration")


def _validate_run(row, meta, manifest, truth):
    source = meta["source_experiment"]
    if source["id"] != manifest["experiment_id"]:
        raise ValueError("run source_experiment.id differs from manifest")
    _, stations, events = truth.source(source["id"])
    dataset = manifest["dataset"]
    if (len(stations.ids), len(events.ids), len(stations.ids) * len(events.ids)) != tuple(
        dataset[key] for key in ("station_count", "event_count", "pair_count")
    ):
        raise ValueError("source geometry counts differ from manifest.dataset")
    actual = meta["station_locs"]
    expected = stations.coordinates_m
    if (len(actual) != len(expected) or any(
        len(a) != 3 or any(not math.isclose(float(x), float(y), rel_tol=0, abs_tol=1e-7)
                          for x, y in zip(a, b)) for a, b in zip(actual, expected)
    )):
        raise ValueError("run station geometry/order differs from elevated source inputs")
    params = meta["run_params"]
    _check_config(params, manifest["baseline_config"], _BASELINE)
    # Missing calibration must not invent T=1 for soft, but a saved run can still
    # disclose its actual temperature and be assessed with an availability warning.
    if row["temperature"] is None:
        row["temperature"] = _temperature(params["temperature"])
        row["temperature_source"] = "run_meta"
    _check_config(params, row, _SETTINGS + _REGULARIZATION)
    for effective in (row.get("effective_config"), params.get("effective_config")):
        if effective is not None:
            _check_config(effective, {**manifest["baseline_config"], **row},
                          _BASELINE + _SETTINGS + _REGULARIZATION)


def _historical(shared, manifest, truth):
    old = shared._row({"id": "historical-reference", "reference_policy": _REFERENCE_POLICY})
    old.update(status="complete", run_dir=manifest["reference_run"],
               configuration={}, **{key: None for key in _SETTINGS + _REGULARIZATION})
    try:
        run = Path(old["run_dir"])
        if not run.is_absolute():
            raise ValueError("reference_run must be absolute")
        params = _object(run / "meta.json")["run_params"]
        if not isinstance(params, dict):
            raise ValueError("invalid historical run_params")
        config = params.get("effective_config", params)
        if not isinstance(config, dict):
            raise ValueError("invalid historical effective_config")
        old["configuration"] = config
        old.update({key: config.get(key, params.get(key)) for key in _SETTINGS + _REGULARIZATION})
        shared._analyze_run(old, run, manifest, truth, historical=True)
    except _READ_ERRORS as error:
        old["warnings"].append(f"historical reference unavailable: {error}")
    return old


def _text(value):
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        return "—"
    return f"{value:.4g}"


def _median(per_k, k, metric):
    try:
        value = per_k[str(k)]["at_temperature"][metric]["median"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        return value
    except (KeyError, TypeError):
        return None


def _markdown(summary):
    dataset, calibration = summary["dataset"], summary["calibration"]
    lines = [f"# Elevated Geysers — hard/soft candidate comparison — {summary['experiment_id']}", "",
             f"Snapshot: {summary['generated_at']}", "",
             "**Real catalog/station geometry; synthetic noisy picks, not real observed picks.**",
             f"{dataset['station_count']} stations, {dataset['event_count']} events, "
             f"{dataset['pair_count']} pairs; [{dataset['start_utc']}, {dataset['end_utc_exclusive']}).", "",
             "Flat model top H=1166 m: station z=H−elevation, event z=H+old catalogue depth. "
             "The velocity cube is translated unchanged; no terrain or air. "
             "The shared vertical datum is assumed, **not verified**.", "",
             "**Mode / K / top_n / T are explicit below. Fixed λ=1, coverage damping power=2; "
             "this is not a new damping sweep.** Hard: top_n=1, T=1. Soft: top_n=K, common calibrated T.",
             "An is_reference queue flag marks a queue case only, not the historical configuration.", "",
             "## Candidate comparison (all seven cases)", "",
             "| case | mode | K | top_n | T | status | final RMSE, m/s | best RMSE, m/s @cycle |",
             "|---|---|---:|---:|---:|---|---:|---|"]
    for row in summary["cases"]:
        lines.append(f"| {_text(row['id'])} | {_text(row.get('candidate_mode'))} | "
                     f"{_text(row.get('n_candidates'))} | {_text(row.get('weights_top_n'))} | "
                     f"{_fmt(row.get('temperature'))} | {_text(row['status'])} | "
                     f"{_fmt(row['final_rmse_m_s'])} | {_fmt(row['min_rmse_m_s'])} "
                     f"@{_text(row['best_cycle'])} |")
    selection = calibration["selection"]
    lines.extend(["", "## Observation-only initial-model calibration", "",
                  f"calibration.json: {_text(calibration['status'])}; common soft T={_fmt(calibration['temperature'])}. "
                  f"Selection: {_text(selection.get('reason'))}; saturated={_text(selection.get('saturated'))}.",
                  "ESS below is at the calibrated T (not hard top-1 ESS). "
                  "Label ESS is weight spread, not independent spatial modes or location accuracy.", "",
                  "| K | median label ESS | median unique-location ESS |",
                  "|---:|---:|---:|"])
    for k in (1, 5, 10, 15):
        lines.append(f"| {k} | {_fmt(_median(calibration['per_k'], k, 'ess'))} | "
                     f"{_fmt(_median(calibration['per_k'], k, 'unique_location_ess'))} |")
    lines.extend(f"- {_text(warning)}" for warning in calibration["warnings"])
    lines.extend(["", "## Assessment and availability", "",
                  "Truth is used for assessment only: volume-average source slowness, then invert to velocity. "
                  "RMSE covers the entire 3-D model. No extra solves or calibration are performed.",
                  "Cycle 0 is the initial model; quality iter i follows update i+1. "
                  "Only complete.json-published iterations enter curves/timing/rays. "
                  "Best cycle is retrospective, **not early stopping**.",
                  f"Hypocentres: {summary['hypocentre_phase']}. Maximum-weight hypothesis, all source events.",
                  "Missing metrics are unavailable, not zero. Running/failed cases are not complete assessments.", "",
                  "| case | assessed | cycles | initial RMSE, m/s | E-step median / p90, m | failed rays | mean s/cycle |",
                  "|---|---|---:|---:|---|---:|---:|"])
    for row in summary["cases"]:
        hypo = row["hypocentres"] or {}
        lines.append(f"| {_text(row['id'])} | {row['assessment_complete']} | "
                     f"{row['completed_cycles']}/{_text(row.get('expected_cycles'))} | "
                     f"{_fmt(row['initial_rmse_m_s'])} | {_fmt(hypo.get('median_m'))} / "
                     f"{_fmt(hypo.get('p90_m'))} | {_text(row['rays']['failed_total'])} | "
                     f"{_fmt(row['runtime']['mean_cycle_s'])} |")
    lines.extend(["", "![Full-model RMSE by completed update](rmse_curve.png)", "",
                  "## Latest historical configuration (not a matched elevated-geometry control)", "",
                  summary["reference_policy"] + "."])
    old = summary.get("reference_run")
    if old:
        lines.extend([f"Run: `{_text(old['run_dir'])}`. Assessed against its own source truth only.",
                      f"mode={_text(old['candidate_mode'])}, K={_text(old['n_candidates'])}, "
                      f"top_n={_text(old['weights_top_n'])}, T={_fmt(old['temperature'])}; "
                      f"λ={_text(old['lambda_reg'])}, damping={_text(old['coverage_damping_power'])}.",
                      f"Initial / final RMSE: {_fmt(old['initial_rmse_m_s'])} / "
                      f"{_fmt(old['final_rmse_m_s'])} m/s.", "", "Saved historical configuration:",
                      "```json", json.dumps(old["configuration"], indent=2, ensure_ascii=False), "```"])
    else:
        lines.append("Historical reference unavailable: no reference_run in manifest.")
    lines.extend(["", "Saved baseline_config (not live defaults):", "```json",
                  json.dumps(summary["baseline_config"], indent=2, ensure_ascii=False), "```", "",
                  "## Warnings / failures", ""])
    lines.extend(f"- {_text(warning)}" for warning in summary["warnings"])
    for row in summary["cases"] + ([old] if old else []):
        if row.get("error"):
            lines.append(f"- {_text(row['id'])}: {_text(row['error'])}")
        lines.extend(f"- {_text(row['id'])}: {_text(warning)}" for warning in row["warnings"])
    return "\n".join(lines) + "\n"


def generate_report(manifest_path, output_dir=None):
    """Snapshot all manifest cases, retaining unavailable metrics as null.

    Reads only manifests, calibration, results and existing native run artifacts.
    Writes summary.json/report.md/geometry_start.csv and an optional RMSE plot.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = _object(manifest_path)
    shared = _load_shared(manifest)
    manifest["dataset"] = shared._dataset(manifest)
    calibration = _calibration(manifest_path.parent / "calibration.json")
    truth = shared.Truth(manifest["experiments_root"])
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "manifest": str(manifest_path), "experiment_id": manifest["experiment_id"],
               "dataset": manifest["dataset"], "baseline_config": manifest["baseline_config"],
               "data_kind": "real elevated geometry, synthetic picks", "calibration": calibration,
               "comparison": "hard/soft candidate comparison; fixed lambda_reg=1, coverage_damping_power=2",
               "reference_policy": _REFERENCE_POLICY, "hypocentre_phase": shared.HYPO_PHASE,
               "best_cycle_policy": "assessment only; no early stopping",
               "warnings": list(calibration["warnings"]), "cases": []}
    for case in manifest["cases"]:
        row = shared._row(case)
        summary["cases"].append(row)
        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case["id"]):
                raise ValueError("invalid case id")
            row.update(_settings(case, calibration))
            row["temperature_source"] = "manifest" if case["candidate_mode"] == "hard" else "calibration"
            if row["temperature"] is None:
                row["warnings"].append("soft temperature unavailable until calibration or run metadata is published")
            result_path = manifest_path.parent / "cases" / case["id"] / "result.json"
            if not result_path.exists():
                continue
            result = _object(result_path)
            row["result"] = result  # Preserve old provenance and optional effective_config verbatim.
            row.update(status=result["status"], run_dir=result.get("run_dir"), error=result.get("error"))
            for key in ("provenance", "effective_config", "case_id"):
                if key in result:
                    row[key] = result[key]
            if result.get("case_id", case["id"]) != case["id"]:
                raise ValueError("result.case_id differs from manifest case")
            if row["status"] == "failed":
                continue
            if row["status"] not in ("running", "complete"):
                raise ValueError("unknown result status")
            if not row["run_dir"]:
                row["warnings"].append("run directory not ready")
                continue
            run = Path(row["run_dir"])
            if not run.is_absolute():
                raise ValueError("result.run_dir must be absolute")
            _validate_run(row, _object(run / "meta.json"), manifest, truth)
            # The old nonhistorical branch requires z=0 and knows only damping
            # cases. Validate elevated geometry/config above, then reuse its full
            # assessment with that obsolete flat-geometry branch disabled.
            shared._analyze_run(row, run, manifest, truth, historical=True)
        except _READ_ERRORS as error:
            row["warnings"].append(f"assessment unavailable: {error}")
    if manifest.get("reference_run"):
        summary["reference_run"] = _historical(shared, manifest, truth)
    else:
        summary["warnings"].append("historical reference unavailable: no reference_run in manifest")
    output = Path(output_dir) if output_dir is not None else manifest_path.parent
    output.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    columns = ["station_count", "event_count", "pair_count", "start_utc", "end_utc_exclusive"]
    writer = csv.DictWriter(buffer, fieldnames=["experiment_id", *columns, "baseline_config_json"])
    writer.writeheader()
    writer.writerow({"experiment_id": manifest["experiment_id"],
                     **{key: manifest["dataset"][key] for key in columns},
                     "baseline_config_json": json.dumps(manifest["baseline_config"], sort_keys=True)})
    shared._atomic(output / "geometry_start.csv", buffer.getvalue().encode("utf-8"))
    try:
        shared._plot(summary["cases"], output)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        summary["warnings"].append(f"RMSE plot unavailable: {error}")
    shared._atomic(output / "summary.json", (json.dumps(summary, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + "\n").encode("utf-8"))
    shared._atomic(output / "report.md", _markdown(summary).encode("utf-8"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        generate_report(args.manifest, args.output_dir)
    except _READ_ERRORS as error:
        parser.exit(2, f"report: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
