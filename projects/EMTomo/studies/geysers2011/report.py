"""Snapshot the detached Geysers queue; never run forward or time-field solves.

The runner may copy this file to runtime_root/runner/report.py. Production
imports therefore happen only after reading --manifest and bootstrapping paths.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np


_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                UnicodeError, zipfile.BadZipFile)
HYPO_PHASE = "last logged E-step, before the final velocity update; not post-final relocation"


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError("boolean is not a metric")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("expected a finite positive/nonnegative number")
    return value


def _jsonl(path, warnings):
    """A writer may currently be appending the last line, even without a newline."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    rows = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            iteration = row["iter"]
            if type(iteration) is not int or iteration < 0:
                raise ValueError("invalid iteration")
            rows[iteration] = row
        except _READ_ERRORS:
            warnings.append(f"{path.name}: ignored incomplete/invalid line")
    return rows


def _completed(run, n_cycles):
    completed = []
    # Direct paths avoid traversing the millions of per-hypothesis artifacts.
    for iteration in range(n_cycles):
        try:
            marker = _object(run / f"iter_{iteration}" / "complete.json")
            if (type(marker.get("iter")) is int and marker["iter"] == iteration
                    and type(marker.get("viewer_completion_protocol")) is int
                    and marker["viewer_completion_protocol"] == 1):
                completed.append(iteration)
        except _READ_ERRORS:
            pass
    return completed


def _load_inputs(root, experiment_id):
    from projects.forward_modeling.experiments import load_inputs
    return load_inputs(root, experiment_id)


def _dataset(manifest):
    # dataset is an optional queue convenience, not part of validate_manifest's
    # required schema. prepare.py persists the same ledger in generation.json.
    source = manifest.get("dataset")
    if source is None:
        path = Path(manifest["experiments_root"]) / "input" / manifest["experiment_id"] / "generation.json"
        source = _object(path)
    return {key: source[key] for key in ("station_count", "event_count", "pair_count",
                                        "start_utc", "end_utc_exclusive")}


class Truth:
    """Cache immutable source inputs and slowness-averaged assessment grids."""

    def __init__(self, root):
        self.root = Path(root)
        self.inputs = {}
        self.grids = {}
        self.hashes = {}

    def source(self, experiment_id):
        if experiment_id not in self.inputs:
            source = _load_inputs(self.root, experiment_id)
            if tuple(source[0].origin_m) != (0., 0., 0.):
                raise ValueError("EMTomo requires a zero-origin source model")
            self.inputs[experiment_id] = source
        return self.inputs[experiment_id]

    def grid(self, experiment_id, shape, cell):
        from velocity_model import block_average_slowness
        key = (experiment_id, tuple(shape), cell)
        if key not in self.grids:
            model, _, _ = self.source(experiment_id)
            self.grids[key] = block_average_slowness(
                model.velocity, model.cell_size_m, tuple(shape), cell)
        return self.grids[key]

    def check_hash(self, experiment_id, expected):
        if expected is None:
            return
        if experiment_id not in self.hashes:
            path = self.root / "input" / experiment_id / "model.npz"
            self.hashes[experiment_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != self.hashes[experiment_id]:
            raise ValueError("source_experiment model SHA256 differs from source inputs")


def _rmse(path, reference):
    model = np.load(path, allow_pickle=False)
    if (model.shape != reference.shape or not np.all(np.isfinite(model))
            or np.any(model <= 0)):
        raise ValueError(f"{path.name}: invalid model or shape mismatch (no broadcasting allowed)")
    return float(np.sqrt(np.mean((np.asarray(model, dtype=float) - reference) ** 2)))


def _hypocentres(run, iteration, events, fine_cell):
    errors = []
    for event, true in enumerate(events.coordinates_m):
        path = run / f"iter_{iteration}" / f"event_{event}" / "weights.npz"
        with np.load(path, allow_pickle=False) as saved:
            weights = saved["weight_values"]
            positions = saved["positions"]
        if (weights.ndim != 1 or not weights.size or positions.shape != (weights.size, 3)
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1., rtol=1e-6, atol=1e-6)
                or not np.all(np.isfinite(positions))):
            raise ValueError(f"invalid event weights: {path}")
        # Saved positions are continuous fine-grid cell-centre coordinates.
        best_m = (positions[int(np.argmax(weights))] + 0.5) * fine_cell
        errors.append(float(np.linalg.norm(best_m - true)))
    if not errors:
        raise ValueError("no truth events")
    return {"iteration": iteration, "model_cycle": iteration, "event_count": len(errors),
            "estimator": "maximum-weight hypothesis; Euclidean 3-D error",
            "median_m": float(np.median(errors)), "p90_m": float(np.percentile(errors, 90)),
            "phase": HYPO_PHASE}


def _analyze_run(row, run, manifest, truth, *, historical=False):
    meta = _object(run / "meta.json")
    source = meta.get("source_experiment")
    if not isinstance(source, dict):
        raise ValueError("missing/invalid source_experiment; refusing to guess truth")
    source_id = source.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("missing source_experiment.id; refusing to guess truth")
    if not historical and source_id != manifest["experiment_id"]:
        raise ValueError("run source_experiment.id differs from manifest")
    truth.check_hash(source_id, source.get("model_sha256"))
    model, stations, events = truth.source(source_id)
    params, grid = meta["run_params"], meta["grid_info"]
    if not isinstance(params, dict) or not isinstance(grid, dict):
        raise ValueError("invalid run_params/grid_info")
    n_cycles = params["n_cycles"]
    if type(n_cycles) is not int or n_cycles < 1:
        raise ValueError("invalid run n_cycles")
    cell = _number(grid["coarse_cell_size"], positive=True)
    fine = _number(grid["fine_cell_size"], positive=True)
    shape = tuple(grid["coarse_shape"])
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("invalid coarse_shape")
    subdivision = params["subdivision"]
    if type(subdivision) is not int or subdivision < 1:
        raise ValueError("invalid subdivision")
    if not np.isclose(cell / subdivision, fine, rtol=1e-12, atol=0):
        raise ValueError("fine grid does not match subdivision")
    if params.get("n_events", len(events.ids)) != len(events.ids):
        raise ValueError("run event count differs from source inputs")
    if not historical:
        dataset = manifest["dataset"]
        if (len(stations.ids) != dataset["station_count"]
                or len(events.ids) != dataset["event_count"]
                or len(stations.ids) * len(events.ids) != dataset["pair_count"]):
            raise ValueError("source geometry counts differ from manifest.dataset")
        if np.any(stations.coordinates_m[:, 2] != 0):
            raise ValueError("source stations are not on flat z=0")
        baseline = manifest["baseline_config"]
        for key in ("cell_size", "subdivision", "n_cycles", "initial_layer_boundaries_km",
                    "initial_layer_velocities_m_s"):
            if key in baseline and params.get(key) != baseline[key]:
                raise ValueError(f"run {key} differs from baseline_config")
        for key in ("lambda_reg", "coverage_damping_power"):
            if params.get(key) != row[key]:
                raise ValueError(f"run {key} differs from manifest case")
        actual_stations = np.asarray(meta["station_locs"], dtype=float)
        if (actual_stations.shape != stations.coordinates_m.shape
                or not np.allclose(actual_stations, stations.coordinates_m, rtol=0, atol=1e-7)):
            raise ValueError("run station geometry/order differs from source inputs")
    reference = truth.grid(source_id, shape, cell)
    completed = _completed(run, n_cycles)
    row.update(source_experiment_id=source_id, expected_cycles=n_cycles,
               completed_iterations=completed, completed_cycles=len(completed),
               geometry={"station_count": len(stations.ids), "event_count": len(events.ids),
                         "pair_count": len(stations.ids) * len(events.ids)},
               grid={"shape": list(shape), "cell_size_m": cell, "fine_cell_size_m": fine})
    try:
        row["initial_rmse_m_s"] = _rmse(run / "initial_model.npy", reference)
        row["rmse_curve"].append({"cycle": 0, "rmse_m_s": row["initial_rmse_m_s"],
                                  "source": "initial_model.npy"})
    except _READ_ERRORS as error:
        row["warnings"].append(str(error))
    quality = _jsonl(run / "quality.jsonl", row["warnings"])
    timing = _jsonl(run / "timing.jsonl", row["warnings"])
    elapsed, rays = [], []
    for iteration in completed:
        try:
            rms = _number(quality[iteration]["rms_m_s"])
            row["rmse_curve"].append({"cycle": iteration + 1, "rmse_m_s": rms,
                                      "source": "quality.jsonl"})
        except _READ_ERRORS:
            row["warnings"].append(f"completed iter_{iteration}: quality unavailable/invalid")
        try:
            elapsed.append(_number(timing[iteration]["elapsed_s"]))
        except _READ_ERRORS:
            pass
        try:
            status = _object(run / f"iter_{iteration}" / "ray_status.json")
            counts = {key: status[key] for key in ("expected", "reached", "failed", "used")}
            if (any(type(v) is not int or v < 0 for v in counts.values())
                    or counts["expected"] != counts["reached"] + counts["failed"]
                    or counts["used"] > counts["reached"]):
                raise ValueError("invalid ray accounting")
            rays.append({"iteration": iteration, **counts})
        except _READ_ERRORS:
            pass
    row["runtime"] = {"mean_cycle_s": float(np.mean(elapsed)) if elapsed else None,
                      "timed_cycles": len(elapsed)}
    row["rays"] = {"iterations": rays, "available_cycles": len(rays),
                   "completed_cycles": len(completed),
                   "failed_total": sum(r["failed"] for r in rays) if rays else None,
                   "counts_complete": len(rays) == len(completed) and bool(completed)}
    fully_published = completed == list(range(n_cycles))
    if row["status"] == "complete":
        # final_model is published after the last cycle, independently of the next E-step.
        try:
            row["final_rmse_m_s"] = _rmse(run / "final_model.npy", reference)
            if fully_published:
                logged = next((p for p in row["rmse_curve"] if p["cycle"] == n_cycles), None)
                if logged and not np.isclose(logged["rmse_m_s"], row["final_rmse_m_s"], rtol=1e-8):
                    row["warnings"].append("final model RMSE differs from last logged quality")
            else:
                row["warnings"].append("complete result but missing/invalid completion markers")
        except _READ_ERRORS as error:
            row["warnings"].append(f"final model unavailable: {error}")
        if fully_published:
            try:
                row["hypocentres"] = _hypocentres(run, n_cycles - 1, events, fine)
            except _READ_ERRORS as error:
                row["warnings"].append(f"last E-step hypocentres unavailable: {error}")
    if row["rmse_curve"]:
        best = min(row["rmse_curve"], key=lambda p: p["rmse_m_s"])
        row["best_cycle"] = best["cycle"]
        row["min_rmse_m_s"] = best["rmse_m_s"]
    row["assessment_complete"] = (fully_published and row["status"] == "complete"
                                  and len(row["rmse_curve"]) == n_cycles + 1
                                  and row["final_rmse_m_s"] is not None
                                  and row["hypocentres"] is not None)


def _row(case):
    return {**case, "status": "pending", "run_dir": None, "warnings": [],
            "completed_iterations": [], "completed_cycles": 0, "rmse_curve": [],
            "initial_rmse_m_s": None, "final_rmse_m_s": None,
            "best_cycle": None, "min_rmse_m_s": None, "hypocentres": None,
            "runtime": {"mean_cycle_s": None, "timed_cycles": 0},
            "rays": {"failed_total": None, "available_cycles": 0, "counts_complete": False},
            "assessment_complete": False}


def _atomic(path, content):
    """Do not expose partly written reports to a live reader."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _plot(rows, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(7, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for row in rows:
        points = row["rmse_curve"]
        if points:
            # NaNs break the line at any missing cycle: never interpolate unlogged updates.
            values = {p["cycle"]: p["rmse_m_s"] for p in points}
            cycles = range(max(values) + 1)
            ax.plot(list(cycles), [values.get(c, np.nan) for c in cycles], marker=".",
                    linewidth=1, label=f"{row['id']} ({row['status']})")
    ax.set(xlabel="Completed model updates (0 = initial)", ylabel="Full-model RMSE (m/s)")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=7)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130)
    _atomic(output / "rmse_curve.png", buffer.getvalue())


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(value):
    return "—" if value is None else f"{value:.2f}"


def _markdown(summary):
    dataset = summary["dataset"]
    lines = [f"# Geysers 2011 — {summary['experiment_id']}", "",
             f"Snapshot: {summary['generated_at']}", "",
             "**Real catalog/station geometry; synthetic noisy picks, not real observed picks.**",
             f"{dataset['station_count']} stations, {dataset['event_count']} events, "
             f"{dataset['pair_count']} station–event pairs; "
             f"[{dataset['start_utc']}, {dataset['end_utc_exclusive']}).", "",
             "Stations remain at flat z=0; original elevation_m is preserved in the source "
             "generation.station_metadata, not applied as topography. Catalog depth is used "
             "positive-down from the flat model top (datum approximation).", "",
             "## Method and interpretation", "",
             "- Truth is used for assessment only: volume-average source slowness onto inversion cells, "
             "then invert to velocity. RMSE covers the **entire 3-D model**, not an upper-layer focus.",
             "- Cycle 0 is initial_model.npy; quality iter i is the model after update i+1. "
             "Only valid complete.json iterations enter quality/timing/ray totals.",
             "- Minimum/best cycle is retrospective assessment only, **not early stopping**. "
             "The queue keeps its configured cycle budget.",
             f"- Hypocentres: {HYPO_PHASE}. Maximum-weight hypothesis; all source events, "
             "no subsampling. Weights are read only for complete cases with all cycles published.",
             "- Missing ray counts are unavailable, not zero; runtime is mean elapsed_s of "
             "published cycles (not total queue wall time). No forward or time-field solves.",
             "", "Saved baseline_config (no main.py defaults):", "", "```json",
             json.dumps(summary["baseline_config"], indent=2, ensure_ascii=False), "```", "",
             "## Queue cases", "",
             "| case | status | λ / damping | cycles | RMSE initial / min @cycle / final, m/s | "
             "last E-step median / p90, m | failed rays (available cycles) | mean s/cycle |",
             "|---|---|---|---:|---|---|---|---:|"]
    for row in summary["cases"]:
        hypo, rays = row["hypocentres"], row["rays"]
        lines.append(f"| {_text(row['id'])}{' (reference)' if row.get('is_reference') else ''} | "
                     f"{row['status']} | {row.get('lambda_reg')} / {row.get('coverage_damping_power')} | "
                     f"{row['completed_cycles']}/{row.get('expected_cycles', '—')} | "
                     f"{_fmt(row['initial_rmse_m_s'])} / {_fmt(row['min_rmse_m_s'])} "
                     f"@{row['best_cycle'] if row['best_cycle'] is not None else '—'} / "
                     f"{_fmt(row['final_rmse_m_s'])} | "
                     f"{_fmt(hypo['median_m'] if hypo else None)} / "
                     f"{_fmt(hypo['p90_m'] if hypo else None)} | "
                     f"{rays['failed_total'] if rays['failed_total'] is not None else '—'} "
                     f"({rays['available_cycles']}) | {_fmt(row['runtime']['mean_cycle_s'])} |")
    lines.extend(["", "![Full-model RMSE by completed update](rmse_curve.png)", "",
                  "## RMSE by model cycle (m/s)", ""])
    cycles = sorted({p["cycle"] for r in summary["cases"] for p in r["rmse_curve"]})
    if cycles:
        lines.extend(["| case | " + " | ".join(map(str, cycles)) + " |",
                      "|---|" + "---:|" * len(cycles)])
        for row in summary["cases"]:
            values = {p["cycle"]: p["rmse_m_s"] for p in row["rmse_curve"]}
            lines.append(f"| {_text(row['id'])} | " + " | ".join(_fmt(values.get(c)) for c in cycles) + " |")
    old = summary.get("reference_run")
    if old:
        lines.extend(["", "## Historical full-synthetic reference (not a matched geometry control)", "",
                      f"Run: `{_text(old['run_dir'])}`; source experiment: "
                      f"`{_text(old.get('source_experiment_id', 'unavailable'))}`. "
                      "Evaluated only against its own source truth, never against the Geysers truth.",
                      f"Initial / final full-model RMSE: {_fmt(old['initial_rmse_m_s'])} / "
                      f"{_fmt(old['final_rmse_m_s'])} m/s. Legacy logs without complete.json "
                      "do not publish quality curve points."])
    lines.extend(["", "## Availability / errors", ""])
    for warning in summary["warnings"]:
        lines.append(f"- {_text(warning)}")
    for row in summary["cases"] + ([old] if old else []):
        if row.get("error"):
            lines.append(f"- {_text(row['id'])}: {_text(row['error'])}")
        lines.extend(f"- {_text(row['id'])}: {_text(w)}" for w in row["warnings"])
    return "\n".join(lines) + "\n"


def generate_report(manifest_path, output_dir=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = _object(manifest_path)
    manifest["dataset"] = _dataset(manifest)
    runtime = Path(manifest["runtime_root"])
    if not runtime.is_absolute():
        raise ValueError("runtime_root must be absolute")
    for path in (runtime, runtime / "projects" / "EMTomo"):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    output = Path(output_dir) if output_dir is not None else manifest_path.parent
    output.mkdir(parents=True, exist_ok=True)
    truth = Truth(manifest["experiments_root"])
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "manifest": str(manifest_path), "experiment_id": manifest["experiment_id"],
               "dataset": manifest["dataset"], "baseline_config": manifest["baseline_config"],
               "data_kind": "real geometry, synthetic picks", "hypocentre_phase": HYPO_PHASE,
               "best_cycle_policy": "assessment only; no early stopping", "warnings": [], "cases": []}
    for case in manifest["cases"]:
        row = _row(case)
        summary["cases"].append(row)
        try:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case["id"]):
                raise ValueError("invalid case id")
            result_path = manifest_path.parent / "cases" / case["id"] / "result.json"
            if not result_path.exists():
                continue
            result = _object(result_path)
            row["status"] = result["status"]
            row["error"] = result.get("error")
            row["run_dir"] = result.get("run_dir")
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
            _analyze_run(row, run, manifest, truth)
        except _READ_ERRORS as error:
            row["warnings"].append(f"assessment unavailable: {error}")
    old_path = manifest.get("reference_run")
    if old_path:
        old = _row({"id": "historical-full-synthetic"})
        old.update(status="complete", run_dir=str(old_path))
        summary["reference_run"] = old
        try:
            run = Path(old_path)
            if not run.is_absolute():
                raise ValueError("reference_run must be absolute")
            _analyze_run(old, run, manifest, truth, historical=True)
        except _READ_ERRORS as error:
            old["warnings"].append(f"historical reference unavailable: {error}")
    # A compact geometry/start ledger is tied to the exact manifest metadata,
    # rather than stale hard-coded counts from the old fully synthetic study.
    buffer = io.StringIO()
    columns = ["station_count", "event_count", "pair_count", "start_utc", "end_utc_exclusive"]
    writer = csv.DictWriter(buffer, fieldnames=["experiment_id", *columns, "baseline_config_json"])
    writer.writeheader()
    writer.writerow({"experiment_id": manifest["experiment_id"],
                     **{k: manifest["dataset"][k] for k in columns},
                     "baseline_config_json": json.dumps(manifest["baseline_config"], sort_keys=True)})
    _atomic(output / "geometry_start.csv", buffer.getvalue().encode("utf-8"))
    try:
        _plot(summary["cases"], output)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        summary["warnings"].append(f"RMSE plot unavailable: {error}")
    _atomic(output / "summary.json", (json.dumps(summary, indent=2, ensure_ascii=False,
                                                 allow_nan=False) + "\n").encode("utf-8"))
    _atomic(output / "report.md", _markdown(summary).encode("utf-8"))
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
