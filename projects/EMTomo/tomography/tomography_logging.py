from __future__ import annotations

import io
import json
import pstats
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class TomographyLogger:
    """
    Directory layout:
        runs/run_<method>_v<version>_<tags>_<timestamp>/
          meta.json
          initial_model.npy / true_model.npy
          timing.jsonl / timing_summary.json
          profile.txt / profile_top30.json
          iter_<i>/
            model.npy / delta_s.npy
            sensitivity_diagonal.npy / coverage_confidence.npy
            event_<j>/
              weights.npz / residuals.npy
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

    def save_meta(self, run_params, station_locs, event_locs, grid_info=None):
        meta = {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "run_name": self.run_name,
            "run_version": self.run_version,
            "run_tags": self.run_tags,
            "run_params": run_params,
            "station_locs": [list(s) for s in station_locs],
            "event_locs": [list(e) for e in event_locs],
            "grid_info": grid_info or {},
        }
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def save_initial_model(self, model):
        np.save(self.run_dir / "initial_model.npy", model.get_geo_grid(subdivision=1).vp)

    def save_true_model(self, model):
        if model is not None:
            np.save(self.run_dir / "true_model.npy", model.get_geo_grid(subdivision=1).vp)

    def iter_dir(self, iteration: int) -> Path:
        d = self.run_dir / f"iter_{iteration}"
        d.mkdir(exist_ok=True)
        return d

    def save_iteration_model(self, iteration: int, model):
        np.save(self.iter_dir(iteration) / "model.npy", model.get_geo_grid(subdivision=1).vp)

    def save_delta_s(self, iteration: int, delta_s: np.ndarray):
        np.save(self.iter_dir(iteration) / "delta_s.npy", delta_s)

    def save_inversion_diagnostics(
        self,
        iteration: int,
        sensitivity_diagonal: np.ndarray,
        coverage_confidence: np.ndarray,
    ) -> None:
        directory = self.iter_dir(iteration)
        np.save(directory / "sensitivity_diagonal.npy", sensitivity_diagonal)
        np.save(directory / "coverage_confidence.npy", coverage_confidence)

    def save_station_fields(self, iteration: int, station_fields: np.ndarray):
        if not self.save_timefields:
            return
        np.save(self.iter_dir(iteration) / "station_fields.npy", np.asarray(station_fields))

    def save_event_data(
        self,
        iteration: int,
        event_idx: int,
        weights: np.ndarray,
        positions: Optional[np.ndarray] = None,
        weight_values: Optional[np.ndarray] = None,
        misfit: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        G_per_weight: Optional[
            Dict[int, Dict[str, np.ndarray] | List[np.ndarray]]
        ] = None,
        ray_count_per_weight: Optional[Dict[int, np.ndarray]] = None,
    ):
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)

        payload = {"weights": weights}
        if positions is not None:
            payload["positions"] = np.asarray(positions, dtype=np.float64)
        if weight_values is not None:
            payload["weight_values"] = np.asarray(weight_values, dtype=np.float64)
        np.savez_compressed(event_dir / "weights.npz", **payload)

        if misfit is not None and self.save_misfit:
            np.save(event_dir / "misfit.npy", misfit)
        if residuals is not None:
            np.save(event_dir / "residuals.npy", residuals)

        if ray_count_per_weight is not None:
            for w_idx, ray_count in ray_count_per_weight.items():
                w_dir = event_dir / f"weight_{w_idx}"
                w_dir.mkdir(exist_ok=True)
                np.save(w_dir / "ray_count.npy", ray_count)

        # Fine-grid G is optional because it is substantially larger than coverage.
        if G_per_weight is not None:
            for w_idx, sparse_g in G_per_weight.items():
                w_dir = event_dir / f"weight_{w_idx}"
                w_dir.mkdir(exist_ok=True)
                if isinstance(sparse_g, dict):
                    np.savez_compressed(
                        w_dir / "G_stations_sparse.npz",
                        **sparse_g,
                    )
                else:
                    # Backward compatibility for callers using the old logger API.
                    for station_idx, g in enumerate(sparse_g):
                        np.savez_compressed(
                            w_dir / f"G_station_{station_idx}.npz",
                            G=np.asarray(g, dtype=np.float32),
                        )

    def start_iteration(self, iteration: int):
        self._iter_start = time.perf_counter()

    def end_iteration(self, iteration: int):
        elapsed = time.perf_counter() - self._iter_start
        self.timing[iteration] = elapsed
        with open(self.run_dir / "timing.jsonl", "a") as f:
            json.dump({"iter": iteration, "elapsed_s": elapsed}, f)
            f.write("\n")

    def save_profiling(self, profiler):
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative")
        stats.print_stats(50)
        (self.run_dir / "profile.txt").write_text(buf.getvalue())
        rows = []
        for func, (cc, nc, tt, ct, _) in list(stats.stats.items())[:30]:
            rows.append(
                {
                    "func": f"{func[0]}:{func[1]}:{func[2]}",
                    "n_calls": nc,
                    "tottime_s": round(tt, 6),
                    "cumtime_s": round(ct, 6),
                }
            )
        with open(self.run_dir / "profile_top30.json", "w") as f:
            json.dump(rows, f, indent=2)

    def save_timing_summary(self):
        total = time.perf_counter() - self._run_start
        summary = {
            "total_s": round(total, 3),
            "per_iter": {str(k): round(v, 3) for k, v in self.timing.items()},
            "mean_iter_s": round(sum(self.timing.values()) / len(self.timing), 3)
            if self.timing
            else None,
        }
        with open(self.run_dir / "timing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def save_quality(self, iteration: int, avg_abs_pct_dev: float):
        with open(self.run_dir / "quality.jsonl", "a") as f:
            json.dump({"iter": int(iteration), "avg_abs_pct_dev": float(avg_abs_pct_dev)}, f)
            f.write("\n")