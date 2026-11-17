"""Residual-difference RMS extraction, without rays or inversion dependencies.

Artifacts describe the first active (MAP) E-step hypothesis BEFORE the M-update.
Cache reuse assumes historical residual artifacts are immutable: metadata and all
complete markers are revalidated, but residual bytes are not re-read on resume.
Use --refresh after any in-place residual edits. All expected events are checked
on initial extraction; an invalid iteration fails the entire run, never drops it.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SCHEMA_VERSION = 1
CORE_SECTIONS = frozenset(("2", "3", "4", "5", "6"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def complete_iterations(run_dir):
    return sorted((int(p.name[5:]), p) for p in Path(run_dir).glob("iter_*")
                  if p.is_dir() and p.name[5:].isdigit() and (p / "complete.json").is_file())


def snapshot_signature(run_dir, row):
    run_dir = Path(run_dir)
    markers = []
    for iteration, directory in complete_iterations(run_dir):
        p = directory / "complete.json"
        stat = p.stat()
        markers.append({"iteration": iteration, "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    data = {"meta_sha256": hashlib.sha256((run_dir / "meta.json").read_bytes()).hexdigest(),
            "complete_markers": markers, "n_events": row["n_events"],
            "run_dir": row["run_dir"], "schema_version": SCHEMA_VERSION}
    data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def compute_iteration(iteration_dir, n_events, iteration=None):
    """Pooled finite upper-triangle RMS and viewer mean-event RMS (seconds).

    NaNs are masked; infinities, missing events and events with no pairs fail.
    Events and reductions are always processed in ascending numerical order.
    """
    directory = Path(iteration_dir)
    if not (directory / "complete.json").is_file():
        raise ValueError(f"Incomplete iteration: {directory}")
    if not isinstance(n_events, int) or n_events <= 0:
        raise ValueError("n_events must be a positive integer")
    sums, event_rms, count = [], [], 0
    for event in range(n_events):
        path = directory / f"event_{event}" / "residuals.npy"
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square residual matrix: {path}: {matrix.shape}")
        if np.isinf(matrix).any():
            raise ValueError(f"Infinite residual: {path}")
        values = np.asarray(matrix[np.triu_indices(matrix.shape[0], k=1)], dtype=np.float64)
        values = values[~np.isnan(values)]
        if not values.size:
            raise ValueError(f"No finite residual pairs: {path}")
        square_sum = float(np.dot(values, values))
        if not math.isfinite(square_sum):
            raise ValueError(f"Residual square sum overflow: {path}")
        sums.append(square_sum)
        event_rms.append(math.sqrt(square_sum / values.size))
        count += int(values.size)
    return {"iteration": int(directory.name[5:]) if iteration is None else iteration,
            "rmse_s": math.sqrt(math.fsum(sums) / count),
            "mean_event_rms_s": math.fsum(event_rms) / n_events,
            "n_events": n_events, "n_pairs": count}


def select_best(trajectory):
    """Select minimum GLOBAL RMS independently per run; earliest exact tie."""
    return min(trajectory, key=lambda point: (point["rmse_s"], point["iteration"]), default=None)


def extract_run(run_dir, row):
    """Return a JSON-serializable run record; errors propagate to the caller."""
    signature = snapshot_signature(run_dir, row)
    iterations = complete_iterations(run_dir)
    trajectory = [compute_iteration(directory, row["n_events"], i) for i, directory in iterations]
    if snapshot_signature(run_dir, row) != signature:
        raise RuntimeError(f"Run changed during extraction: {run_dir}")
    return {**{k: row.get(k) for k in ("run_id", "run_dir", "section", "mode", "source_id")},
            "status": "ok" if trajectory else "absent",
            "reason": None if trajectory else "zero_complete_iterations",
            "n_complete_iterations": len(iterations), "n_extracted_iterations": len(trajectory),
            "expected_n_events": row["n_events"], "trajectory": trajectory,
            "best": select_best(trajectory), "signature": signature,
            "residual_completeness_verified": bool(trajectory), "extracted_utc": utc_now()}


def atomic_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def extract_inventory(inventory_path, repo_root, cache_path, workers=16, refresh=False, budget_seconds=150):
    """Extract core runs with bounded workers and atomic per-run checkpoints.

    budget_seconds bounds task submission batches, not an individual run's I/O.
    Reinvoke to resume. Returns cache, including pending/error/absent records.
    """
    inventory_path, repo_root, cache_path = map(Path, (inventory_path, repo_root, cache_path))
    raw = inventory_path.read_bytes()
    rows = [r for r in json.loads(raw)["runs"] if str(r["section"]) in CORE_SECTIONS]
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh else {}
    previous = cache.get("runs", {}) if cache.get("schema_version") == SCHEMA_VERSION else {}
    cache = {"schema_version": SCHEMA_VERSION, "inventory_sha256": hashlib.sha256(raw).hexdigest(),
             "provenance": {"artifact": "residual differences r_i-r_j, first active MAP hypothesis",
                            "phase": "E-step before iteration M-update", "units": "seconds",
                            "selection": "minimum pooled rmse_s; earliest iteration on exact tie",
                            "residual_policy": "historical outputs immutable at snapshot; --refresh required after residual edits",
                            "completeness": "every expected event loaded; NaNs masked; infinities rejected",
                            "reduction": "ascending event ID; float64 dot per event; math.fsum across events"},
             "runs": {}}
    pending = []
    for row in rows:
        key = row["run_id"]
        old = previous.get(key)
        if old and old.get("status") in ("ok", "absent") and old.get("signature") == snapshot_signature(repo_root / row["run_dir"], row):
            cache["runs"][key] = old
        else:
            cache["runs"][key] = {"status": "pending", "run_id": key, "section": row["section"]}
            pending.append(row)
    started = time.monotonic()
    # Batches avoid an unbounded queue and make resumable wall-time chunks practical.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending), workers):
            if time.monotonic() - started >= budget_seconds:
                break
            futures = {pool.submit(extract_run, repo_root / r["run_dir"], r): r for r in pending[offset:offset + workers]}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"run_id": row["run_id"], "section": row["section"], "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                cache["runs"][row["run_id"]] = result
                cache["updated_utc"] = utc_now()
                atomic_write(cache_path, cache)
                print(f"{result['status']}: {row['run_id']} ({result.get('n_extracted_iterations', 0)} iterations)", flush=True)
    cache["counts"] = {status: sum(r["status"] == status for r in cache["runs"].values())
                       for status in ("ok", "absent", "error", "pending")}
    atomic_write(cache_path, cache)
    print(cache["counts"], flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--inventory", type=Path, default=project / "studies/article_inventory/inventory.json")
    parser.add_argument("--repo-root", type=Path, default=project.parents[1])
    parser.add_argument("--cache", type=Path, default=project / "paper/exp_data/residual_metrics_cache.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=140)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    extract_inventory(args.inventory, args.repo_root, args.cache, args.workers, args.refresh, args.budget_seconds)


if __name__ == "__main__":
    main()
