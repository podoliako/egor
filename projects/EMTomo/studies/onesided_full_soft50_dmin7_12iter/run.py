"""Run a new twelve-cycle inversion with the validated one-cycle pilot settings.

Launch from EMTomo. Outputs go to the shared runs directory; progress and
failures are recorded beside this script. Existing runs are never resumed or
modified.
"""
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback


BASE = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
SOURCE_RUN = (
    "run_onesided-full-soft50-dmin7-onesided-nx6-24x12x12-18st-1200ev-noise01-pick005-seed42"
    "_v1.4_ncand-50_topn-50_mode-soft_dmin-7_lam-1.0_temp-1.0_sub-8_cov-2.0"
    "_20261105_093031_86db9cfed9a741669d8063a7e4ecf839"
)
sys.path.insert(0, str(BASE))

from config import InversionConfig
from experiment_runner import run_saved_experiment


def save_status(status):
    path = STUDY / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    if (STUDY / "status.json").exists():
        raise FileExistsError("Study already launched; inspect status.json rather than starting a duplicate")
    meta = json.loads((BASE / "runs" / SOURCE_RUN / "meta.json").read_text())
    params = meta["run_params"]
    config = InversionConfig(**{f.name: params[f.name] for f in fields(InversionConfig)
                                if f.name in params})
    config = replace(config, n_cycles=12, run_name="onesided-full-soft50-dmin7-12iter",
                     runs_dir=str(BASE / "runs"), save_runs=True)
    status = {"state": "running", "pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_run": SOURCE_RUN, "experiment_id": meta["source_experiment"]["id"],
              "config": asdict(config)}
    save_status(status)
    print(json.dumps(status, indent=2), flush=True)
    try:
        logger = run_saved_experiment(status["experiment_id"], config)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}",
                      finished_at=datetime.now(timezone.utc).isoformat())
        save_status(status)
        traceback.print_exc()
        raise
    status.update(state="completed", run_id=logger.run_id, run_dir=str(logger.run_dir),
                  finished_at=datetime.now(timezone.utc).isoformat())
    save_status(status)
    print("SAVED_RUN_ID=" + logger.run_id, flush=True)


if __name__ == "__main__":
    main()
