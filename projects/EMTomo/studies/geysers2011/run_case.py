"""Run one immutable Geysers hard1000 case from the queue's runtime snapshot.

Only stdlib imports are allowed until the manifest has selected the runtime.
This file is copied to <runtime_root>/runner/run_case.py by the queue.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback
from types import SimpleNamespace
import uuid


EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_26st_3780ev_noise01_pick001"
FORWARD_IMPLEMENTATION = "saved_hard1000_legacy_sphere"
SAVED_TRAVEL_TIMES = "skfmm_order2_exact_station_homogeneous_sphere"
INPUT_FILES = {"model.npz", "stations.csv", "events.csv", "generation.json"}
RESOURCE_VALUES = {"forward_workers": 32, "inversion_workers_per_run": 12,
                   "blas_threads_per_run": 12, "max_parallel_runs": 2}
CASE_FIELDS = {"id", "lambda_reg", "coverage_damping_power", "run_name", "is_reference"}
RUNTIME_FIELDS = {"n_workers", "run_name", "runs_dir", "save_runs"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    """Normalize tuples as well as reject non-JSON/nonfinite configuration values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {name}: {value!r}")


def _hash_mapping(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty hash mapping")
    for key, digest in value.items():
        if not isinstance(key, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid {name} SHA256 entry: {key!r}")


def validate_manifest(manifest: dict) -> None:
    """Structural validation without importing anything from the working checkout."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected manifest schema_version 1")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected experiment_id")
    if manifest.get("forward_implementation") != FORWARD_IMPLEMENTATION:
        raise ValueError("Expected saved_hard1000_legacy_sphere forward_implementation")
    for key in ("experiments_root", "runs_dir", "runtime_root", "reference_run"):
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    if not isinstance(manifest.get("baseline_config"), dict) or not manifest["baseline_config"]:
        raise ValueError("Missing baseline_config")
    if manifest.get("resources") != RESOURCE_VALUES or any(
        type(value) is not int for value in manifest["resources"].values()
    ):
        raise ValueError(f"Expected resources {RESOURCE_VALUES}")
    created = manifest.get("created_utc")
    try:
        if not isinstance(created, str):
            raise ValueError("not a timestamp")
        parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
        offset = parsed.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("not UTC")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("created_utc must be a UTC timestamp") from error
    _hash_mapping(manifest.get("input_sha256"), "input_sha256")
    if set(manifest["input_sha256"]) != INPUT_FILES:
        raise ValueError("input_sha256 must name the four native input files")
    _hash_mapping(manifest.get("code_sha256"), "code_sha256")
    for name in manifest["code_sha256"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise ValueError(f"Unsafe code_sha256 relative path: {name}")
    if "projects/EMTomo/eikonal.py" not in manifest["code_sha256"]:
        raise ValueError("code_sha256 must include the legacy eikonal.py")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen, names, references = set(), set(), []
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError(f"Case must contain only {sorted(CASE_FIELDS)}; no extra config overrides")
        _identifier(case["id"], "case id")
        _identifier(case["run_name"], "run_name")
        if case["id"] in seen or case["run_name"] in names:
            raise ValueError("Duplicate case id or run_name")
        seen.add(case["id"])
        names.add(case["run_name"])
        for key in ("lambda_reg", "coverage_damping_power"):
            value = case[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Case {key} must be finite and nonnegative")
        if type(case["is_reference"]) is not bool:
            raise ValueError("is_reference must be boolean")
        if case["is_reference"]:
            references.append(case)
            if case["lambda_reg"] != 0.5 or case["coverage_damping_power"] != 1.0:
                raise ValueError("Reference case must use lambda_reg=0.5 and coverage_damping_power=1")
    if len(references) != 1:
        raise ValueError("Expected exactly one is_reference case")
    _json(manifest)


def verify_hashes(manifest: dict) -> None:
    input_dir = Path(manifest["experiments_root"]) / "input" / manifest["experiment_id"]
    root = Path(manifest["runtime_root"]).resolve()
    for name, digest in manifest["input_sha256"].items():
        if sha256(input_dir / name) != digest:
            raise ValueError(f"Input SHA256 mismatch: {name}")
    for name, digest in manifest["code_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Runtime SHA256 mismatch: {name}")
    for name, digest in manifest.get("reference_sha256", {}).items():
        if sha256(Path(manifest["reference_run"]) / name) != digest:
            raise ValueError(f"Reference SHA256 mismatch: {name}")


def load_runtime(manifest: dict):
    """Set paths and BLAS limits BEFORE the first production/numpy import."""
    root = Path(manifest["runtime_root"]).resolve()
    emtomo = root / "projects" / "EMTomo"
    paths = [str(emtomo), str(root)]
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    threads = str(manifest["resources"]["blas_threads_per_run"])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = threads
    # All production dependencies are deliberately local to this function.
    config = importlib.import_module("config")
    runner = importlib.import_module("experiment_runner")
    em = importlib.import_module("tomography.tomography_em")
    logging = importlib.import_module("tomography.tomography_logging")
    for name, module in list(sys.modules.items()):
        if (name in {"config", "eikonal", "raytracing", "velocity_model", "experiment_data", "experiment_runner"}
                or name == "instruments" or name.startswith("instruments.")
                or name == "tomography" or name.startswith("tomography.")
                or name == "projects" or name.startswith("projects.forward_modeling")):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(root):
                raise ValueError(f"Production module imported outside runtime snapshot: {name}: {file}")
    return SimpleNamespace(InversionConfig=config.InversionConfig, prepare_inversion=runner.prepare_inversion,
                           resolve_pick_noise=runner.resolve_pick_noise, em=em,
                           TomographyLogger=logging.TomographyLogger,
                           np=importlib.import_module("numpy"))


def build_config(manifest, case, config_type, reference_meta):
    """Allow only lambda/coverage and explicit runtime differences from saved meta."""
    allowed = {field.name for field in fields(config_type)}
    baseline = manifest["baseline_config"]
    unknown = set(baseline) - allowed
    if unknown:
        raise ValueError(f"Unknown baseline_config fields (no silent overrides): {sorted(unknown)}")
    saved = {key: value for key, value in reference_meta["run_params"].items() if key in allowed}
    if _json(baseline) != _json(saved):
        raise ValueError("baseline_config differs from saved reference InversionConfig fields")
    if reference_meta["run_params"].get("travel_times") != SAVED_TRAVEL_TIMES:
        raise ValueError("Reference run does not use saved legacy sphere travel_times")
    expected = {"cell_size": 1000.0, "subdivision": 8, "n_cycles": 12,
                "candidate_mode": "hard", "n_candidates": 5, "weights_top_n": 1,
                "lambda_reg": 0.5, "coverage_damping_power": 1.0}
    if any(baseline.get(key) != value for key, value in expected.items()):
        raise ValueError("Reference baseline must be ordinary hard1000/sub8/K5/top1/12cycles, lambda=.5/cov=1")
    values = {**baseline, "lambda_reg": case["lambda_reg"],
              "coverage_damping_power": case["coverage_damping_power"],
              "n_workers": manifest["resources"]["inversion_workers_per_run"],
              "run_name": case["run_name"], "runs_dir": manifest["runs_dir"], "save_runs": True}
    return config_type(**values)


def provenance(manifest, manifest_path, manifest_sha, case):
    initial = Path(manifest["reference_run"]) / "initial_model.npy"
    return {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
            "case_id": case["id"], "runtime_root": manifest["runtime_root"],
            "runtime_hashes": manifest["code_sha256"], "input_sha256": manifest["input_sha256"],
            "forward_implementation": manifest["forward_implementation"],
            "reference_initial_model": str(initial), "reference_initial_sha256": sha256(initial)}


def make_logger(logger_type, config, manifest, case, origin):
    class StudyLogger(logger_type):
        def save_meta(self, run_params, *args, **kwargs):
            params = {**run_params, "initialization": "baseline_config_bitwise_verified",
                      "initial_model_source": {"path": origin["reference_initial_model"],
                                               "sha256": origin["reference_initial_sha256"],
                                               "used_as_checkpoint": False},
                      "provenance": origin, "effective_config": _json(asdict(config))}
            return super().save_meta(params, *args, **kwargs)

    return StudyLogger(base_dir=config.runs_dir,
                       run_name=f"{case['run_name']}_{manifest['experiment_id']}",
                       run_version=config.run_version,
                       run_tags={"lam": config.lambda_reg, "cov": config.coverage_damping_power,
                                 "attempt": uuid.uuid4().hex[:8]})


def verify_initial(initial, reference_path, np):
    reference = np.load(reference_path, allow_pickle=False)
    actual = np.asarray(initial)
    if (actual.shape != reference.shape or actual.dtype != reference.dtype
            or actual.tobytes() != reference.tobytes()):
        raise ValueError("Prepared initial_model is not BITWISE identical to reference initial_model.npy")
    if actual.shape != (20, 7, 7) or not np.all(np.isfinite(actual)):
        raise ValueError("Expected a finite hard1000 initial model with shape (20,7,7)")


def _quality(run_dir, n_cycles):
    rows = [json.loads(line) for line in (run_dir / "quality.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != n_cycles or [row.get("iter") for row in rows] != list(range(n_cycles)):
        raise ValueError("quality.jsonl must contain exactly all 12 completed updates")
    for row in rows:
        for key in ("rms_m_s", "avg_abs_pct_dev"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid quality {key}")
    return float(rows[-1]["rms_m_s"])


def check_completed_result(result, manifest, case, config, origin, np):
    """Do not trust a 'complete' string: validate persisted artifacts and identity."""
    if result.get("status") != "complete":
        raise ValueError("Existing result is not complete")
    if result.get("case_id") != case["id"] or result.get("provenance") != origin:
        raise ValueError("Existing result case/provenance mismatch")
    directory = result.get("run_dir")
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise ValueError("Existing result has no absolute run_dir")
    run_dir = Path(directory).resolve()
    if run_dir.parent != Path(manifest["runs_dir"]).resolve():
        raise ValueError("Existing run_dir is outside runs_dir")
    meta = _read_json(run_dir / "meta.json")
    params = meta["run_params"]
    if params.get("provenance") != origin or params.get("effective_config") != _json(asdict(config)):
        raise ValueError("Existing run config/provenance mismatch")
    if params.get("initialization") != "baseline_config_bitwise_verified":
        raise ValueError("Existing initialization mismatch")
    if meta.get("source_experiment") != {"id": manifest["experiment_id"],
                                         "model_sha256": manifest["input_sha256"]["model.npz"]}:
        raise ValueError("Existing source input mismatch")
    expected = _json(asdict(config))
    # run_em records resolved noise, not the optional config overrides. This
    # study verifies observation sigmas (.01, .01) before entering run_em.
    for key in ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s"):
        if expected[key] is None:
            expected[key] = 0.01
    for key, value in expected.items():
        if key not in {"runs_dir", "save_runs", "run_name"} and params.get(key) != value:
            raise ValueError(f"Existing run_params config mismatch: {key}")
    name = f"{case['run_name']}_{manifest['experiment_id']}"
    if params.get("run_name") != name or params.get("travel_times") != SAVED_TRAVEL_TIMES:
        raise ValueError("Existing run label/travel_times mismatch")
    if meta.get("viewer_completion_protocol") != 1:
        raise ValueError("Existing run lacks viewer completion protocol")
    verify_initial(np.load(run_dir / "initial_model.npy", allow_pickle=False),
                   Path(origin["reference_initial_model"]), np)
    final = np.load(run_dir / "final_model.npy", allow_pickle=False)
    if final.shape != (20, 7, 7) or not np.all(np.isfinite(final)) or np.any(final <= 0):
        raise ValueError("Invalid final_model.npy")
    expected_iterations = {f"iter_{iteration}" for iteration in range(config.n_cycles)}
    if {path.name for path in run_dir.glob("iter_*") if path.is_dir()} != expected_iterations:
        raise ValueError("Expected exactly 12 iteration directories")
    for iteration in range(config.n_cycles):
        directory = run_dir / f"iter_{iteration}"
        if _read_json(directory / "complete.json") != {"iter": iteration, "viewer_completion_protocol": 1}:
            raise ValueError(f"Invalid complete marker for iter_{iteration}")
        for name in ("model.npy", "delta_s.npy", "sensitivity_diagonal.npy", "coverage_confidence.npy"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing completed iteration artifact: {directory / name}")
    rms = _quality(run_dir, config.n_cycles)
    if result.get("final_rms_m_s") != rms:
        raise ValueError("Existing result final RMS differs from quality.jsonl")
    return rms


def run_case(manifest_path: Path, case_id: str):
    manifest_path = Path(manifest_path).resolve()
    # Read once: the hash identifies exactly the bytes parsed, not a later reread.
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    validate_manifest(manifest)
    case = next((case for case in manifest["cases"] if case["id"] == case_id), None)
    if case is None:
        raise ValueError(f"Unknown case-id: {case_id}")
    case_dir = manifest_path.parent / "cases" / case_id
    if not case_dir.is_dir():
        raise ValueError(f"Queue must create case artifacts directory first: {case_dir}")
    result_path = case_dir / "result.json"
    started = time.perf_counter()
    result = {"case_id": case_id, "status": "running", "run_dir": None,
              "started_utc": _utc(), "attempts": []}
    previous = None
    resume_check = False
    try:
        if result_path.exists():
            previous = _read_json(result_path)
            resume_check = previous.get("status") == "complete" or previous.get("resume_rejected", False)
            result["attempts"] = previous.get("attempts", []) + [
                {key: value for key, value in previous.items() if key != "attempts"}]
        if not resume_check:
            atomic_json(result_path, result)
        verify_hashes(manifest)
        runtime = load_runtime(manifest)
        reference_meta = _read_json(Path(manifest["reference_run"]) / "meta.json")
        config = build_config(manifest, case, runtime.InversionConfig, reference_meta)
        origin = provenance(manifest, manifest_path, hashlib.sha256(payload).hexdigest(), case)
        result["provenance"] = origin
        if resume_check and previous is not None:
            if previous.get("resume_rejected"):
                raise ValueError("Previously rejected completed result; inspect artifacts before retrying")
            check_completed_result(previous, manifest, case, config, origin, runtime.np)
            print(f"Validated completed case {case_id}: {previous['run_dir']}", flush=True)
            return previous
        logger = make_logger(runtime.TomographyLogger, config, manifest, case, origin)
        result["run_dir"] = str(logger.run_dir.resolve())
        atomic_json(result_path, result)
        print(f"Run directory: {result['run_dir']}", flush=True)
        prepared = runtime.prepare_inversion(manifest["experiment_id"], config,
                                             experiments_root=Path(manifest["experiments_root"]))
        verify_initial(prepared.initial_model.velocity, Path(origin["reference_initial_model"]), runtime.np)
        if sha256(Path(origin["reference_initial_model"])) != origin["reference_initial_sha256"]:
            raise ValueError("Reference initial model changed during preparation")
        if len(prepared.station_ids) != 26 or len(prepared.event_ids) != 3780:
            raise ValueError("Expected all 26 stations and 3780 events; no thinning")
        if tuple(prepared.noise_sigmas) != (0.01, 0.01):
            raise ValueError("Expected synthetic observation noise relative=.01, absolute=.01 s")
        old_description = runtime.em.METHOD_DESCRIPTION
        try:
            # eikonal.py itself is the immutable c381cfd snapshot, NOT patched here.
            runtime.em.METHOD_DESCRIPTION = {**old_description, "travel_times": SAVED_TRAVEL_TIMES}
            runtime.em.warm_up_jit()
            runtime.em.run_em(
                config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
                runtime.resolve_pick_noise(config, prepared.noise_sigmas),
                reference_model=prepared.reference_model,
                source_experiment={"id": manifest["experiment_id"],
                                   "model_sha256": manifest["input_sha256"]["model.npz"]},
                run_name=f"{case['run_name']}_{manifest['experiment_id']}", logger=logger,
            )
        finally:
            runtime.em.METHOD_DESCRIPTION = old_description
        result.update(status="complete", finished_utc=_utc(), elapsed_s=time.perf_counter() - started,
                      final_rms_m_s=_quality(logger.run_dir, config.n_cycles))
        # The same strict validation gates both initial publication and later reuse.
        verify_hashes(manifest)
        check_completed_result(result, manifest, case, config, origin, runtime.np)
        atomic_json(result_path, result)
        return result
    except (Exception, KeyboardInterrupt) as error:
        if resume_check and previous:
            result["run_dir"] = previous.get("run_dir")
            result["resume_rejected"] = True
        result.update(status="failed", finished_utc=_utc(), elapsed_s=time.perf_counter() - started,
                      error=str(error), traceback=traceback.format_exc())
        atomic_json(result_path, result)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)
    try:
        run_case(args.manifest, args.case_id)
    except KeyboardInterrupt:
        traceback.print_exc()
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
