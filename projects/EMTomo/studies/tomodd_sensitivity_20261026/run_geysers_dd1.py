"""Run only dd1-d40-s5 on frozen Geysers inputs; preserve all existing runs.

python -B studies/tomodd_sensitivity_20261026/run_geysers_dd1.py --validate-only
python -B studies/tomodd_sensitivity_20261026/run_geysers_dd1.py
"""
from pathlib import Path
import argparse
import json
import os
import tempfile
import time

import run_study as shared

ROOT = shared.PROJECT / "runs/tomodd_dd1_geometry_20261026"
BASELINE = shared.OLD / "outputs/geysers-tomodd-sigma1"
ARTIFICIAL = shared.PROJECT / "runs/tomodd_sensitivity_20261026/outputs/dd1-d40-s5"
CASE = next(c for c in shared.CASES if c["id"] == "dd1-d40-s5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    shared.check_root(ROOT)
    output = ROOT / "outputs/geysers-dd1-d40-s5"
    # The shared input/control guards must compare against the Geysers baseline,
    # not against synthetic observations. This changes only the current process.
    shared.BASELINE = BASELINE
    with tempfile.TemporaryDirectory(prefix="geysers-dd1-") as temporary:
        os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "mpl")
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["GFORTRAN_UNBUFFERED_ALL"] = "y"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        generated = Path(temporary) / "validation" if args.validate_only else output
        worker_args = argparse.Namespace(
            geometry="geysers", output=generated, benchmark_root=shared.SNAPSHOT / "benchmark",
            forward_root=shared.SNAPSHOT / "projects/forward_modeling",
            solver_build=shared.SNAPSHOT / "solver", support_root=shared.SNAPSHOT / "support")
        adapter = shared.worker.load_adapter(worker_args.benchmark_root)
        arguments, info = shared.worker.preflight(worker_args, adapter)
        old = json.loads((BASELINE / "worker_info.json").read_text())
        shared.require(old["state"] == "completed", "Geysers baseline incomplete")
        for key in shared.PROVENANCE_KEYS:
            shared.require(info[key] == old[key], f"Geysers frozen provenance differs: {key}")
        artificial = json.loads((ARTIFICIAL / "worker_info.json").read_text())
        artificial_config = json.loads((ARTIFICIAL / "config.json").read_text())
        shared.require(artificial["state"] == "completed", "Artificial reference incomplete")
        shared.require(artificial["solver_sha256"] == info["solver_sha256"], "Binary mismatch")
        shared.require(artificial_config["solver"]["iteration_sets"] ==
                       json.loads(json.dumps(shared.solver_schedule(CASE))), "Artificial schedule mismatch")
        shared.require(artificial_config["solver"]["smoothing_lon_lat_depth"] == CASE["smoothing"],
                       "Artificial smoothing mismatch")
        arguments["timeout_s"] = shared.SOLVER_TIMEOUT_S
        info.update(id="geysers-dd1-d40-s5", parameter_set=CASE["id"], baseline=str(BASELINE),
                    artificial_reference=str(ARTIFICIAL), output=str(output),
                    absolute_dd_weight=[1.0, 1.0], solver_schedule=shared.solver_schedule(CASE),
                    smoothing_lon_lat_depth=CASE["smoothing"], timeout_s=shared.SOLVER_TIMEOUT_S,
                    script_sha256=shared.worker.digest(Path(__file__)),
                    shared_script_sha256=shared.worker.digest(Path(shared.__file__)),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()})
        info["arguments"]["run_dir"] = str(output)
        shared.configure_adapter(adapter, CASE, generated, info, args.validate_only)
        if args.validate_only:
            try:
                adapter.run(**arguments)
            except shared.InputsValidated:
                pass
            else:
                raise RuntimeError("Validation did not stop before the solver")
            print(json.dumps({"state": "validated", "solver_launched": False,
                              "geometry": "geysers", "parameter_set": CASE["id"],
                              "events": info["events"], "stations": info["stations"],
                              "generated_input_sha256": info["generated_input_sha256"]}, indent=2))
            return 0
        shared.check_root(ROOT)
        ROOT.mkdir(exist_ok=True)
        (ROOT / "outputs").mkdir()
        shared.worker.new_output(output)
        output.mkdir()
        manifest = dict(info, created_utc=shared.worker.now(),
                        baseline_sha256={name: shared.worker.digest(BASELINE / name)
                                         for name in ("config.json", "state.npz", "worker_info.json")},
                        artificial_sha256={name: shared.worker.digest(ARTIFICIAL / name)
                                           for name in ("config.json", "state.npz", "worker_info.json")})
        shared.write_json(ROOT / "manifest.json", manifest, exclusive=True)
        started = time.monotonic()
        info.update(state="running", started_utc=shared.worker.now(), worker_pid=os.getpid())
        shared.write_json(ROOT / "status.json", {k: info[k] for k in ("state", "started_utc", "worker_pid")})
        try:
            metrics = adapter.run(**arguments)
            info["iteration_artifacts"] = shared.worker.check_completed(output, adapter, metrics)
            shared.require(metrics["event_count"] == 3776, "Unexpected event count")
            info["state"] = "completed"
        except BaseException as error:
            info.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            info.update(finished_utc=shared.worker.now(), elapsed_s=time.monotonic() - started)
            shared.write_json(output / "worker_info.json", info)
            shared.write_json(ROOT / "status.json", {k: info[k] for k in
                              ("state", "started_utc", "finished_utc", "elapsed_s", "worker_pid")}
                              | ({"error": info["error"]} if "error" in info else {}))
        print(json.dumps({"state": info["state"], "output": str(output),
                          "elapsed_s": info["elapsed_s"], "results": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
