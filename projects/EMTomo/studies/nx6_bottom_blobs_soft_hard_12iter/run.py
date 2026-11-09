"""Run a matched twelve-cycle soft/hard inversion of the bottom-and-blobs catalog.

Launch from EMTomo. Outputs go to the shared runs directory; progress and
failures are recorded beside this script. Existing runs are never resumed or
modified.
"""
import argparse
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
    "run_onesided-full-soft50-dmin7-12iter-onesided-nx6-24x12x12-18st-1200ev-noise01-pick005-seed42"
    "_v1.4_ncand-50_topn-50_mode-soft_dmin-7_lam-1.0_temp-1.0_sub-8_cov-2.0"
    "_20261105_094430_ee66281e11c44fc8aa6262245b4d3065"
)
sys.path.insert(0, str(BASE))

from config import InversionConfig
from experiment_runner import run_saved_experiment


def save_status(status):
    path = STUDY / f"{status['mode']}_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("soft", "hard"), required=True)
    args = parser.parse_args()
    if (STUDY / f"{args.mode}_status.json").exists():
        raise FileExistsError(f"{args.mode} already launched; inspect its status rather than starting a duplicate")
    meta = json.loads((BASE / "runs" / SOURCE_RUN / "meta.json").read_text())
    params = meta["run_params"]
    config = InversionConfig(**{f.name: params[f.name] for f in fields(InversionConfig)
                                if f.name in params})
    config = replace(config, n_cycles=12, candidate_mode=args.mode,
                     run_name=f"nx6-bottom-blobs-{args.mode}50-dmin7-12iter",
                     runs_dir=str(BASE / "runs"), save_runs=True)
    status = {"state": "running", "mode": args.mode, "pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_run": SOURCE_RUN,
              "experiment_id": "nx6_bottom_blobs_24x8x8_22st_600ev_noise01_pick005_seed42",
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
