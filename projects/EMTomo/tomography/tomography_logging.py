from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def _ray_status_rows(log, n_stations=None):
    """Extract available active masks; missing legacy masks remain unavailable."""
    statuses = getattr(log, "ray_reached_per_weight", {})
    indices, rows = [], []
    for index in np.flatnonzero(np.asarray(log.weights) > 0):
        if index not in statuses:
            continue
        reached = np.asarray(statuses[index], dtype=bool)
        if n_stations is None and reached.ndim == 1:
            n_stations = reached.size
        if reached.shape != (n_stations,):
            raise ValueError(
                f"Ray reached mask for hypothesis {index} must have shape "
                f"({n_stations},), got {reached.shape}"
            )
        indices.append(int(index))
        rows.append(reached)
    return np.asarray(indices, dtype=np.int64), np.asarray(rows, dtype=bool)


def summarize_ray_status(event_logs, n_stations: int) -> Optional[dict]:
    """Count primary traces only, requiring masks for every positive hypothesis."""
    summary = {
        "schema_version": 1,
        "expected": 0,
        "reached": 0,
        "failed": 0,
        "used": 0,
        "n_events": 0,
        "n_hypotheses": 0,
        "events_with_failed_rays": 0,
        "hypotheses_with_insufficient_stations": 0,
        "failures": [],
    }
    for event, log in event_logs:
        indices, rows = _ray_status_rows(log, n_stations)
        if len(indices) != np.count_nonzero(np.asarray(log.weights) > 0):
            return None
        summary["n_events"] += 1
        event_failed = False
        for index, reached in zip(indices, rows):
            count = int(np.count_nonzero(reached))
            summary["n_hypotheses"] += 1
            summary["expected"] += n_stations
            summary["reached"] += count
            if count >= 2:
                summary["used"] += count
            else:
                summary["hypotheses_with_insufficient_stations"] += 1
            failed = np.flatnonzero(~reached).tolist()
            if failed:
                event_failed = True
                summary["failures"].append({
                    "event": int(event), "hypothesis": int(index), "stations": failed,
                })
        summary["events_with_failed_rays"] += int(event_failed)
    summary["failed"] = summary["expected"] - summary["reached"]
    return summary


class TomographyLogger:
    """
    Directory layout:
        runs/run_<method>_v<version>_<tags>_<timestamp>/
          meta.json
          initial_model.npy / final_model.npy
          timing.jsonl / timing_summary.json / quality.jsonl
          iter_<i>/
            complete.json            ← published only after all cycle artifacts
            model.npy / delta_s.npy
            sensitivity_diagonal.npy / coverage_confidence.npy
            ray_status.json           ← aggregate primary-trace accounting
            event_<j>/
              weights.npz / residuals.npy
              ray_status.npz          ← reached station masks per active hypothesis
              weight_<w>/
                G_stations_sparse.npz  ← compact fine-grid ray paths
                ray_count.npy          ← station ray count (coarse grid)
    """

    def __init__(
        self,
        base_dir: str = "runs",
        save_misfit: bool = False,
        save_timefields: bool = False,
        run_name: str = "em",
        run_version: str = "1.0",
        run_tags: Optional[Mapping[str, object]] = None,
    ):
        self.run_name = self._slug(run_name, "run_name")
        self.run_version = self._slug(run_version, "run_version")
        self.run_tags = {
            self._slug(key, "run tag key"): self._slug(str(value), "run tag value")
            for key, value in (run_tags or {}).items()
        }
        self.started_at = datetime.now(timezone.utc)
        timestamp = self.started_at.astimezone().strftime("%Y%m%d_%H%M%S")
        tag_part = "_".join(f"{key}-{value}" for key, value in self.run_tags.items())
        parts = ["run", self.run_name, f"v{self.run_version}", tag_part, timestamp]
        self.run_id = "_".join(part for part in parts if part)
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._iter_start: float = 0.0
        self._run_start: float = time.perf_counter()
        self.timing: dict = {}
        self.save_misfit = save_misfit
        self.save_timefields = save_timefields

    @staticmethod
    def _slug(value: str, field_name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", value.strip()).strip("-.").lower()
        if not slug:
            raise ValueError(f"{field_name} must contain at least one letter or digit")
        return slug

    @staticmethod
    def _save_npy(path: Path, values) -> None:
        """Write an array atomically so the live viewer never sees a partial file."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values)
        temporary.replace(path)

    @staticmethod
    def _save_npz(path: Path, **values) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(path)

    @staticmethod
    def _save_json(path: Path, values) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(values, stream, indent=2)
        temporary.replace(path)

    def save_meta(self, run_params, station_locs, event_locs, grid_info=None, source_experiment=None):
        meta = {
            "viewer_completion_protocol": 1,
            "viewer_ray_status_schema": 1,
            "viewer_saved_artifacts": {
                "save_misfit": self.save_misfit,
                "save_timefields": self.save_timefields,
            },
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "run_name": self.run_name,
            "run_version": self.run_version,
            "run_tags": self.run_tags,
            "run_params": run_params,
            "station_locs": [list(s) for s in station_locs],
            "event_locs": [list(e) for e in event_locs],
            "grid_info": grid_info or {},
            "source_experiment": source_experiment,
        }
        self._save_json(self.run_dir / "meta.json", meta)

    def save_initial_model(self, velocity: np.ndarray):
        self._save_npy(self.run_dir / "initial_model.npy", np.asarray(velocity))

    def save_final_model(self, velocity: np.ndarray):
        """Model after the last update; iter_<i>/model.npy holds models before each update."""
        self._save_npy(self.run_dir / "final_model.npy", np.asarray(velocity))

    def iter_dir(self, iteration: int) -> Path:
        d = self.run_dir / f"iter_{iteration}"
        d.mkdir(exist_ok=True)
        return d

    def save_iteration_model(self, iteration: int, velocity: np.ndarray):
        self._save_npy(self.iter_dir(iteration) / "model.npy", np.asarray(velocity))

    def save_delta_s(self, iteration: int, delta_s: np.ndarray):
        self._save_npy(self.iter_dir(iteration) / "delta_s.npy", delta_s)

    def save_inversion_diagnostics(
        self,
        iteration: int,
        sensitivity_diagonal: np.ndarray,
        coverage_confidence: np.ndarray,
    ) -> None:
        directory = self.iter_dir(iteration)
        self._save_npy(directory / "sensitivity_diagonal.npy", sensitivity_diagonal)
        self._save_npy(directory / "coverage_confidence.npy", coverage_confidence)

    def save_station_fields(self, iteration: int, station_fields: np.ndarray):
        if not self.save_timefields:
            return
        self._save_npy(self.iter_dir(iteration) / "station_fields.npy", np.asarray(station_fields))

    def save_ray_count(self, iteration: int, ray_count: np.ndarray):
        self._save_npy(self.iter_dir(iteration) / "ray_count.npy", np.asarray(ray_count))

    def save_ray_status(self, iteration: int, event_logs, n_stations: int) -> Optional[dict]:
        """Save trustworthy cycle totals, or skip legacy logs lacking active masks."""
        summary = summarize_ray_status(event_logs, n_stations)
        if summary is not None:
            self._save_json(self.iter_dir(iteration) / "ray_status.json", summary)
        return summary

    def save_event_data(self, iteration: int, event_idx: int, log):
        """Store one event's hypotheses (a ``tomography_events.EventLog``)."""
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)
        self._save_npz(
            event_dir / "weights.npz",
            weight_shape=np.asarray(log.misfit_shape, dtype=np.int32),
            weight_indices=np.asarray(log.candidate_cells, dtype=np.int32),
            positions=np.asarray(log.positions, dtype=np.float64),
            weight_values=np.asarray(log.weights, dtype=np.float64),
        )
        if log.misfit is not None and self.save_misfit:
            self._save_npy(event_dir / "misfit.npy", log.misfit)
        self._save_npy(event_dir / "residuals.npy", log.residuals)
        indices, reached = _ray_status_rows(log)
        if len(indices):
            self._save_npz(
                event_dir / "ray_status.npz", hypothesis_indices=indices, reached=reached,
            )

        for w_idx, ray_count in log.ray_count_per_weight.items():
            w_dir = event_dir / f"weight_{w_idx}"
            w_dir.mkdir(exist_ok=True)
            self._save_npy(w_dir / "ray_count.npy", ray_count)
        # Fine-grid G is optional because it is substantially larger than coverage.
        for w_idx, sparse_g in (log.G_per_weight or {}).items():
            w_dir = event_dir / f"weight_{w_idx}"
            w_dir.mkdir(exist_ok=True)
            self._save_npz(w_dir / "G_stations_sparse.npz", **sparse_g)

    def start_iteration(self, iteration: int):
        self._iter_start = time.perf_counter()

    def end_iteration(self, iteration: int):
        elapsed = time.perf_counter() - self._iter_start
        self.timing[iteration] = elapsed
        with open(self.run_dir / "timing.jsonl", "a") as f:
            json.dump({"iter": iteration, "elapsed_s": elapsed}, f)
            f.write("\n")

    def complete_iteration(self, iteration: int) -> None:
        """Publish only after the update and every configured save have succeeded."""
        self._save_json(
            self.iter_dir(iteration) / "complete.json",
            {"iter": int(iteration), "viewer_completion_protocol": 1},
        )

    def save_timing_summary(self):
        total = time.perf_counter() - self._run_start
        summary = {
            "total_s": round(total, 3),
            "per_iter": {str(k): round(v, 3) for k, v in self.timing.items()},
            "mean_iter_s": round(sum(self.timing.values()) / len(self.timing), 3)
            if self.timing
            else None,
        }
        self._save_json(self.run_dir / "timing_summary.json", summary)
        return summary

    def save_quality(self, iteration: int, avg_abs_pct_dev: float, rms_m_s: float):
        """Errors of the model after the update of ``iteration`` against the reference."""
        row = {"iter": int(iteration), "avg_abs_pct_dev": float(avg_abs_pct_dev), "rms_m_s": float(rms_m_s)}
        with open(self.run_dir / "quality.jsonl", "a") as f:
            json.dump(row, f)
            f.write("\n")