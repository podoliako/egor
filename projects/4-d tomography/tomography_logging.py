from __future__ import annotations

import io
import json
import pstats
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class TomographyLogger:
    """
    Directory layout:
        runs/run_<timestamp>/
          meta.json
          initial_model.npy / true_model.npy
          timing.jsonl / timing_summary.json
          profile.txt / profile_top30.json
          iter_<i>/
            model.npy / delta_s.npy / station_fields.npy
            event_<j>/
              weights.npy / misfit.npy / residuals.npy
              weight_<w>/G_station_<k>.npy
    """

    def __init__(self, base_dir: str = "runs"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._iter_start: float = 0.0
        self._run_start: float = time.perf_counter()
        self.timing: dict = {}

    def save_meta(self, run_params, station_locs, event_locs, grid_info=None):
        meta = {
            "run_id": self.run_id,
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

    def save_station_fields(self, iteration: int, station_fields: np.ndarray):
        np.save(self.iter_dir(iteration) / "station_fields.npy", np.asarray(station_fields))

    def save_event_data(
        self,
        iteration: int,
        event_idx: int,
        weights: np.ndarray,
        misfit: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        G_per_weight: Optional[Dict[int, List[np.ndarray]]] = None,
    ):
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)
        np.save(event_dir / "weights.npy", weights)
        if misfit is not None:
            np.save(event_dir / "misfit.npy", misfit)
        if residuals is not None:
            np.save(event_dir / "residuals.npy", residuals)
        if G_per_weight is not None:
            for w_idx, g_list in G_per_weight.items():
                w_dir = event_dir / f"weight_{w_idx}"
                w_dir.mkdir(exist_ok=True)
                for si, g in enumerate(g_list):
                    np.save(w_dir / f"G_station_{si}.npy", g)

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
