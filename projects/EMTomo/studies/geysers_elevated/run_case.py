"""One elevated-geometry case, using the exact last saved EMTomo method."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

try:
    import _shared_case as shared
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from projects.EMTomo.studies.geysers2011 import run_case as shared

EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001"


def validate_manifest(manifest):
    if manifest.get("schema_version") != 2 or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Expected elevated-geometry manifest v2")
    if manifest.get("resources") != shared.RESOURCE_VALUES:
        raise ValueError("Unexpected resources")
    for key in ("runtime_root", "experiments_root", "runs_dir", "reference_run"):
        if not Path(manifest[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    for key in ("input_sha256", "code_sha256", "reference_sha256"):
        shared._hash_mapping(manifest.get(key), key)
    if set(manifest["input_sha256"]) != shared.INPUT_FILES:
        raise ValueError("Expected all native input hashes")
    if set(manifest["reference_sha256"]) != {"meta.json", "initial_model.npy"}:
        raise ValueError("Expected both reference hashes")
    if "projects/EMTomo/eikonal.py" not in manifest["code_sha256"]:
        raise ValueError("Missing frozen eikonal hash")
    for name in manifest["code_sha256"]:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe runtime relative path")
    cases = manifest["cases"]
    identities = [(case["candidate_mode"], case["n_candidates"]) for case in cases]
    expected = {("hard", k) for k in (1, 5, 10, 15)} | {("soft", k) for k in (5, 10, 15)}
    if len(cases) != 7 or set(identities) != expected:
        raise ValueError("Expected exactly seven hard/soft cases")
    if len({case["id"] for case in cases}) != 7 or len({case["run_name"] for case in cases}) != 7:
        raise ValueError("Duplicate case identity")
    if sum(case["is_reference"] for case in cases) != 1:
        raise ValueError("Expected one reference case")
    for case in cases:
        shared._identifier(case["id"], "case")
        shared._identifier(case["run_name"], "run name")
        soft = case["candidate_mode"] == "soft"
        if case["weights_top_n"] != (case["n_candidates"] if soft else 1):
            raise ValueError("Soft must keep all K; hard keeps top1")
        if case["temperature"] != (None if soft else 1.):
            raise ValueError("Soft temperature is resolved only from calibration; hard uses T1")
        if case["lambda_reg"] != 1. or case["coverage_damping_power"] != 2.:
            raise ValueError("Damping must match latest saved reference")
        if case["is_reference"] and (soft or case["n_candidates"] != 5):
            raise ValueError("Reference must be hard K5")
    shared._json(manifest)


def verify_hashes(manifest):
    validate_manifest(manifest)
    shared.verify_hashes(manifest)


def build_config(manifest, case, runtime, calibration=None):
    allowed = {field.name for field in fields(runtime.InversionConfig)}
    meta = shared._read_json(Path(manifest["reference_run"]) / "meta.json")
    saved = {key: value for key, value in meta["run_params"].items() if key in allowed}
    if saved != manifest["baseline_config"]:
        raise ValueError("Baseline differs from last saved reference")
    required = {"cell_size": 1000., "subdivision": 8, "n_cycles": 12,
                "lambda_reg": 1., "coverage_damping_power": 2., "candidate_mode": "hard", "n_candidates": 5}
    if any(saved.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected latest reference configuration")
    if meta["run_params"]["travel_times"] != shared.SAVED_TRAVEL_TIMES:
        raise ValueError("Reference solver does not match frozen legacy method")
    if case["candidate_mode"] == "soft":
        if calibration is None or not 1. <= calibration["temperature"] <= 16.:
            raise ValueError("Soft requires validated calibration T in [1,16]")
        temperature = float(calibration["temperature"])
    else:
        temperature = 1.
    return runtime.InversionConfig(**{
        **saved, "candidate_mode": case["candidate_mode"], "n_candidates": case["n_candidates"],
        "weights_top_n": case["weights_top_n"], "temperature": temperature,
        "n_workers": manifest["resources"]["inversion_workers_per_run"],
        "run_name": case["run_name"], "runs_dir": manifest["runs_dir"], "save_runs": True,
    })


def verify_observations(state, manifest):
    ledger = shared._read_json(state / "forward_complete.json")
    output = Path(manifest["experiments_root"]) / "output" / manifest["experiment_id"]
    if ledger.get("status") != "complete":
        raise ValueError("Forward not complete")
    shared._hash_mapping(ledger.get("output_sha256"), "output_sha256")
    if set(ledger["output_sha256"]) != {"arrivals.csv", "metadata.json"}:
        raise ValueError("Expected both native forward output hashes")
    for name, digest in ledger["output_sha256"].items():
        if shared.sha256(output / name) != digest:
            raise ValueError(f"Forward output changed: {name}")
    if shared.sha256(state / "calibration.json") != ledger["calibration_sha256"]:
        raise ValueError("Temperature calibration changed")
    calibration = shared._read_json(state / "calibration.json")
    return ledger, calibration


def run_case(manifest_path, case_id):
    manifest_path = Path(manifest_path).resolve()
    state = manifest_path.parent
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    validate_manifest(manifest)
    case = next(case for case in manifest["cases"] if case["id"] == case_id)
    result_path = state / "cases" / case_id / "result.json"
    previous = shared._read_json(result_path) if result_path.exists() else None
    result = {"case_id": case_id, "status": "running", "started_utc": shared._utc(),
              "run_dir": None, "attempts": []}
    if previous:
        result["attempts"] = previous.get("attempts", []) + [
            {key: value for key, value in previous.items() if key != "attempts"}]
    started = time.perf_counter()
    try:
        verify_hashes(manifest)
        ledger, calibration = verify_observations(state, manifest)
        runtime = shared.load_runtime(manifest)
        config = build_config(manifest, case, runtime, calibration)
        origin = shared.provenance(manifest, manifest_path, hashlib.sha256(payload).hexdigest(), case)
        origin.update(output_sha256=ledger["output_sha256"], calibration_sha256=ledger["calibration_sha256"])
        result.update(provenance=origin, effective_config=shared._json(asdict(config)))
        if previous and (previous.get("status") == "complete" or previous.get("resume_rejected")):
            if previous.get("resume_rejected"):
                raise ValueError("Previously rejected complete result; inspect before retry")
            if previous.get("effective_config") != shared._json(asdict(config)):
                raise ValueError("Completed result effective_config mismatch")
            shared.check_completed_result(previous, manifest, case, config, origin, runtime.np)
            print(f"Validated completed {case_id}: {previous['run_dir']}", flush=True)
            return previous
        shared.atomic_json(result_path, result)
        logger = shared.make_logger(runtime.TomographyLogger, config, manifest, case, origin)
        result["run_dir"] = str(logger.run_dir.resolve())
        shared.atomic_json(result_path, result)
        print(f"{case_id}: K={config.n_candidates}, top_n={config.weights_top_n}, T={config.temperature:g}", flush=True)
        print(f"Run directory: {logger.run_dir}", flush=True)
        prepared = runtime.prepare_inversion(manifest["experiment_id"], config, experiments_root=Path(manifest["experiments_root"]))
        shared.verify_initial(prepared.initial_model.velocity, Path(manifest["reference_run"]) / "initial_model.npy", runtime.np)
        if len(prepared.event_ids) != 3776 or len(prepared.station_ids) != 26 or tuple(prepared.noise_sigmas) != (.01, .01):
            raise ValueError("Unexpected geometry/observation count or noise")
        old_description = runtime.em.METHOD_DESCRIPTION
        try:
            runtime.em.METHOD_DESCRIPTION = {**old_description, "travel_times": shared.SAVED_TRAVEL_TIMES}
            runtime.em.warm_up_jit()
            runtime.em.run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
                runtime.resolve_pick_noise(config, prepared.noise_sigmas), reference_model=prepared.reference_model,
                source_experiment={"id": manifest["experiment_id"], "model_sha256": manifest["input_sha256"]["model.npz"]},
                run_name=f"{case['run_name']}_{manifest['experiment_id']}", logger=logger)
        finally:
            runtime.em.METHOD_DESCRIPTION = old_description
        result.update(status="complete", finished_utc=shared._utc(), elapsed_s=time.perf_counter()-started,
                      final_rms_m_s=shared._quality(logger.run_dir, config.n_cycles))
        verify_hashes(manifest)
        final_ledger, _ = verify_observations(state, manifest)
        if final_ledger != ledger:
            raise ValueError("Observation/calibration ledger changed during case")
        shared.check_completed_result(result, manifest, case, config, origin, runtime.np)
        shared.atomic_json(result_path, result)
        return result
    except (Exception, KeyboardInterrupt) as error:
        if previous and (previous.get("status") == "complete" or previous.get("resume_rejected")):
            result.update(resume_rejected=True, run_dir=previous.get("run_dir"))
        result.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                      finished_utc=shared._utc(), elapsed_s=time.perf_counter()-started)
        shared.atomic_json(result_path, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    run_case(args.manifest, args.case_id)


if __name__ == "__main__":
    main()
