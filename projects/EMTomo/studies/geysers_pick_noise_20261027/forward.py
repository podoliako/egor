"""One clean absolute-time solve, baseline reproduction gate, five paired noise levels.

Run in a fresh process: python -B forward.py --manifest /absolute/manifest.json.
No inversion or convergence solve is launched here. A failed/interrupted solve is
never retried automatically; validated clean times survive downstream failures.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


STUDY_ID = "geysers_pick_noise_20261027"
SOURCE_EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001"
EXPECTED_SHAPE = (3776, 26)
INPUT_NAMES = ("model.npz", "stations.csv", "events.csv", "generation.json")
OUTPUT_NAMES = ("arrivals.csv", "metadata.json")
CACHE_KEYS = {"absolute_travel_times_s", "event_ids", "station_ids", "event_xyz_m", "station_xyz_m"}
LEVELS = {"x0p5": .5, "x2": 2., "x3": 3., "x5": 5., "x10": 10.}
BASELINE_ATOL_S = 1e-8


def _now():
    return datetime.now(UTC).isoformat()


def _load_api(manifest, common):
    # This establishes frozen import paths before any numerical import.
    common.load_runtime(manifest)
    modules = {name: importlib.import_module("projects.forward_modeling." + name)
               for name in ("experiments", "model", "noise", "solver")}
    data = importlib.import_module("experiment_data")
    root = Path(manifest["runtime_root"]).resolve()
    for module in (*modules.values(), data):
        filename = module.__file__
        if filename is None or not Path(filename).resolve().is_relative_to(root):
            raise ValueError(f"Non-frozen forward import: {filename}")
    return SimpleNamespace(
        np=importlib.import_module("numpy"), load_inputs=modules["experiments"].load_inputs,
        load_tomography_experiment=data.load_tomography_experiment,
        ForwardConfig=modules["model"].ForwardConfig, NoiseConfig=modules["noise"].NoiseConfig,
        add_arrival_noise=modules["noise"].add_arrival_noise,
        compute_travel_times=modules["solver"].compute_travel_times,
    )


def _manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _validate_plan(manifest):
    if manifest.get("study_id", STUDY_ID) != STUDY_ID or manifest.get("schema_version", 1) != 1:
        raise ValueError("Unsupported forward manifest")
    for name in ("state_dir", "runtime_root", "source_experiments_root", "experiments_root"):
        if not Path(manifest[name]).is_absolute():
            raise ValueError(f"{name} must be absolute")
    if Path(manifest["experiments_root"]).resolve() != Path(manifest["state_dir"]).resolve() / "experiments":
        raise ValueError("experiments_root must be state_dir/experiments")
    if manifest["resources"]["forward_workers"] != 32:
        raise ValueError("Expected the approved 32 forward workers")
    if set(manifest["source_input_sha256"]) != set(INPUT_NAMES):
        raise ValueError("Expected exactly the four original input files")
    if set(manifest["source_output_sha256"]) != set(OUTPUT_NAMES):
        raise ValueError("Expected original arrivals.csv and metadata.json hashes")
    for mapping in (manifest["runtime_sha256"], manifest["source_input_sha256"],
                    manifest["source_output_sha256"]):
        if not mapping or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                              for v in mapping.values()):
            raise ValueError("Invalid SHA256 mapping")
    if manifest["source_experiment_id"] != SOURCE_EXPERIMENT_ID:
        raise ValueError("Unexpected source experiment ID")
    levels = manifest["levels"]
    if len(levels) != 5 or {level["id"] for level in levels} != set(LEVELS):
        raise ValueError("Expected exactly the five approved noise levels")
    for level in levels:
        multiplier = LEVELS[level["id"]]
        if (level["multiplier"] != multiplier or level["relative_sigma"] != .01 * multiplier
                or level["absolute_sigma_s"] != .01 * multiplier or level["seed"] != 42):
            raise ValueError(f"Unapproved noise configuration: {level['id']}")
        if level["experiment_id"] != "geysers_spheres_picknoise_20261027_" + level["id"]:
            raise ValueError(f"Unexpected experiment ID: {level['experiment_id']}")


@contextmanager
def _exclusive(path):
    # Keep the inode stable; OS-released locks permit safe resume after SIGKILL.
    # Deleting a flock file would let two writers lock different inodes.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@contextmanager
def _staged_directory(destination):
    """Publish a whole directory, without replacing existing experiment data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(destination.parent / ("." + destination.name + ".lock")):
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
        try:
            yield temporary
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _publish_file(path, writer):
    """An atomic no-replace link publishes only closed, flushed file contents."""
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _publish_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _publish_file(path, lambda stream: stream.write(payload))


def noisy_relative(api, clean, event_ids, station_ids, config):
    """Perturb absolute clean times; allow negative picks BEFORE normalization."""
    np = api.np
    noisy = np.array(api.add_arrival_noise(clean, event_ids, station_ids, config),
                     dtype=np.float64, copy=True)
    if noisy.shape != clean.shape or not np.isfinite(noisy).all():
        raise ValueError("Invalid noisy absolute times")
    return noisy - noisy.min(axis=1, keepdims=True)


def _validate_clean(np, clean, events, stations):
    if (clean.shape != EXPECTED_SHAPE or clean.shape != (len(events.ids), len(stations.ids))
            or not np.isfinite(clean).all() or np.any(clean < 0)):
        raise ValueError("Invalid clean absolute travel-time matrix")
    for points in (events, stations):
        if (len(set(points.ids)) != len(points.ids)
                or any(not isinstance(i, str) or not i.strip() for i in points.ids)
                or np.shape(points.coordinates_m) != (len(points.ids), 3)
                or not np.isfinite(points.coordinates_m).all()):
            raise ValueError("Invalid geometry IDs or coordinates")


def _read_cache(api, common, forward, binding, events, stations):
    ledger = common.read(forward / "clean_times.json")
    cache = forward / "clean_times.npz"
    if (ledger.get("status") != "completed" or ledger.get("manifest_digest") != binding
            or ledger.get("shape") != list(EXPECTED_SHAPE)
            or ledger.get("clean_sha256") != common.sha(cache)):
        raise ValueError("Clean cache ledger/hash/manifest mismatch")
    np = api.np
    with np.load(cache, allow_pickle=False) as saved:
        if set(saved.files) != CACHE_KEYS:
            raise ValueError("Unexpected clean cache arrays")
        for key, expected in (("event_ids", events.ids), ("station_ids", stations.ids)):
            if saved[key].dtype.kind != "U" or not np.array_equal(saved[key], np.asarray(expected)):
                raise ValueError(f"Clean cache {key} mismatch")
        for key, expected in (("event_xyz_m", events.coordinates_m),
                              ("station_xyz_m", stations.coordinates_m)):
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Clean cache {key} mismatch")
        clean = saved["absolute_travel_times_s"]
        if clean.dtype != np.dtype("float64"):
            raise ValueError("Clean cache must be float64")
    _validate_clean(np, clean, events, stations)
    clean.flags.writeable = False
    return clean, ledger


def _baseline_control(api, manifest, clean, events, stations):
    np = api.np
    baseline = api.load_tomography_experiment(
        manifest["source_experiment_id"], experiments_root=Path(manifest["source_experiments_root"]))
    if (len(baseline.event_ids) != len(events.ids) or len(baseline.station_ids) != len(stations.ids)
            or set(baseline.event_ids) != set(events.ids)
            or set(baseline.station_ids) != set(stations.ids)
            or tuple(baseline.noise_sigmas) != (.01, .01)):
        raise ValueError("Baseline geometry IDs or noise sigmas differ")
    reproduced = noisy_relative(api, clean, events.ids, stations.ids,
                                api.NoiseConfig(relative_sigma=.01, absolute_sigma_s=.01, seed=42))
    # CSV order is irrelevant, including numeric-looking IDs with leading zeros.
    event_index = {identifier: i for i, identifier in enumerate(baseline.event_ids)}
    station_index = {identifier: i for i, identifier in enumerate(baseline.station_ids)}
    reference = baseline.arrival_times_s[np.ix_(
        [event_index[i] for i in events.ids], [station_index[i] for i in stations.ids])]
    difference = reproduced - reference
    if difference.shape != clean.shape or not np.isfinite(difference).all():
        raise ValueError("Invalid baseline comparison matrix")
    worst = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    maximum = float(np.max(np.abs(difference)))
    return {
        "status": "completed" if maximum <= BASELINE_ATOL_S else "failed",
        "max_abs_difference_s": maximum,
        "rms_difference_s": float(np.sqrt(np.mean(difference ** 2))),
        "allclose": bool(np.allclose(reproduced, reference, rtol=0., atol=BASELINE_ATOL_S)),
        "exact": bool(np.array_equal(reproduced, reference)),
        "atol_s": BASELINE_ATOL_S, "rtol": 0., "pairs": int(difference.size),
        "worst_pair": {"event_id": events.ids[worst[0]], "station_id": stations.ids[worst[1]]},
        "noise_config": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42},
    }


def _write_arrivals(path, times, events, stations):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "event_id", "arrival_time_s"))
        writer.writerows((station_id, event_id, float(times[e, s]))
                         for e, event_id in enumerate(events.ids)
                         for s, station_id in enumerate(stations.ids))


def _validate_level(api, manifest, level, times, events, stations):
    np = api.np
    loaded = api.load_tomography_experiment(
        level["experiment_id"], experiments_root=Path(manifest["experiments_root"]))
    if (loaded.arrival_times_s.shape != EXPECTED_SHAPE
            or tuple(loaded.event_ids) != tuple(events.ids)
            or tuple(loaded.station_ids) != tuple(stations.ids)
            or not np.array_equal(loaded.station_coordinates_m, stations.coordinates_m)
            or not np.array_equal(loaded.reference_event_coordinates_m, events.coordinates_m)):
        raise ValueError(f"Derived geometry mismatch: {level['id']}")
    if (not np.isfinite(loaded.arrival_times_s).all()
            or not np.array_equal(loaded.arrival_times_s.min(axis=1), np.zeros(len(events.ids)))
            or not np.array_equal(loaded.arrival_times_s, times)
            or tuple(loaded.noise_sigmas) != (level["relative_sigma"], level["absolute_sigma_s"])):
        raise ValueError(f"Derived arrivals/noise mismatch: {level['id']}")


def _hashes(common, directory, names):
    return {name: common.sha(directory / name) for name in names}


def _verify_record(common, manifest, level, record, binding):
    if (record.get("status") != "completed" or record.get("manifest_digest") != binding
            or record.get("experiment_id") != level["experiment_id"]
            or record.get("input_sha256") != manifest["source_input_sha256"]
            or set(record.get("output_sha256", {})) != set(OUTPUT_NAMES)):
        raise ValueError(f"Invalid level ledger: {level['id']}")
    root = Path(manifest["experiments_root"])
    for category, names in (("input", INPUT_NAMES), ("output", OUTPUT_NAMES)):
        if _hashes(common, root / category / level["experiment_id"], names) != record[category + "_sha256"]:
            raise ValueError(f"Derived {category} hash mismatch: {level['id']}")


def _verify_complete(common, manifest, ledger, binding, cache_ledger):
    baseline = ledger.get("baseline_reproduction", {})
    if (ledger.get("status") != "completed" or ledger.get("manifest_digest") != binding
            or ledger.get("clean_sha256") != cache_ledger["clean_sha256"]
            or set(ledger.get("levels", {})) != set(LEVELS)
            or baseline.get("status") != "completed" or not baseline.get("allclose")
            or baseline.get("pairs") != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]
            or not 0 <= baseline.get("max_abs_difference_s", float("inf")) <= BASELINE_ATOL_S):
        raise ValueError("Invalid forward completion ledger")
    for level in manifest["levels"]:
        _verify_record(common, manifest, level, ledger["levels"][level["id"]], binding)
    return ledger


def _level_metadata(source_metadata, api, level, provenance, baseline, elapsed):
    metadata = deepcopy(source_metadata)
    # A source convergence result, if present, is not a new convergence solve.
    for key in ("convergence", "max_difference_s"):
        metadata.pop(key, None)
    noise = api.NoiseConfig(relative_sigma=level["relative_sigma"],
                            absolute_sigma_s=level["absolute_sigma_s"], seed=level["seed"])
    metadata.update({
        "time_reference": "earliest_station_arrival_per_event", "elapsed_seconds": elapsed,
        "input_sha256": provenance["source_input_sha256"],
        "noise": {
            "enabled": True, "model": "independent_gaussian_relative_plus_absolute",
            "config": asdict(noise), "sigma_reference": "noiseless_absolute_travel_time_s",
            "application": "before_subtracting_earliest_noisy_arrival_per_event",
            "negative_picks": "not_clipped_before_normalization", "rng": "PCG64_pair_sha256_v1",
            "numpy_version": api.np.__version__, "convergence": "noiseless",
        },
        "provenance": {**provenance, "level_id": level["id"], "noise_multiplier": level["multiplier"],
                       "same_pair_keyed_draws_across_levels": True,
                       "baseline_reproduction": baseline},
    })
    return metadata


def run(manifest, common, *, manifest_path=None):
    """Execute or hash-verify/resume the stage; tests inject a solve-free API."""
    common.verify_manifest(manifest)
    _validate_plan(manifest)
    state = Path(manifest["state_dir"])
    forward = state / "forward"
    forward.mkdir(parents=True, exist_ok=True)
    binding = _manifest_digest(manifest)
    started = time.monotonic()
    progress = {"study_id": STUDY_ID, "status": "running", "started_utc": _now(),
                "worker_pid": os.getpid(), "manifest_digest": binding}

    def stage(name, **fields):
        progress.update(stage=name, updated_utc=_now(), elapsed_s=time.monotonic() - started, **fields)
        common.atomic(forward / "progress.json", progress)
        print(f"[{progress['updated_utc']}] forward: {name}", flush=True)

    with _exclusive(forward / ".run.lock"):
        try:
            api = _load_api(manifest, common)
            np = api.np
            source_root = Path(manifest["source_experiments_root"])
            source_id = manifest["source_experiment_id"]
            source_input = source_root / "input" / source_id
            source_metadata = common.read(source_root / "output" / source_id / "metadata.json")
            config = api.ForwardConfig(refinement=2, source_radius_cells=2)
            if source_metadata.get("config") != asdict(config):
                raise ValueError("Baseline forward configuration differs from the approved configuration")
            if source_metadata.get("noise", {}).get("config") != {
                    "relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}:
                raise ValueError("Baseline noise configuration must be .01 relative + .01 s, seed 42")
            model, stations, events = api.load_inputs(source_root, source_id)
            if (len(events.ids), len(stations.ids)) != EXPECTED_SHAPE:
                raise ValueError(f"Expected geometry {EXPECTED_SHAPE}")
            cache = forward / "clean_times.npz"
            cache_ledger_path = forward / "clean_times.json"
            complete = state / "forward_complete.json"
            solve_started = forward / "solve_started.json"
            if cache_ledger_path.exists():
                clean, cache_ledger = _read_cache(api, common, forward, binding, events, stations)
            else:
                if any(os.path.lexists(p) for p in (cache, solve_started, complete)):
                    raise RuntimeError("A solve was already attempted or cache is incomplete; refusing a second solve")
                attempt = {**progress, "config": asdict(config), "forward_workers": 32}
                _publish_json(solve_started, attempt)
                stage("computing_clean_times")
                solve_clock = time.monotonic()
                try:
                    # The ONLY solver call. Never call compute_arrivals/check_convergence.
                    clean = np.array(api.compute_travel_times(
                        model=model, stations=stations, events=events, config=config, workers=32),
                        dtype=np.float64, copy=True)
                    _validate_clean(np, clean, events, stations)
                except BaseException as error:
                    _publish_json(forward / "solve_failed.json", {
                        **attempt, "status": "failed", "finished_utc": _now(),
                        "elapsed_s": time.monotonic() - solve_clock, "error": str(error),
                        "traceback": traceback.format_exc()})
                    raise
                _publish_file(cache, lambda stream: np.savez(
                    stream, absolute_travel_times_s=clean,
                    event_ids=np.asarray(events.ids, dtype=str), station_ids=np.asarray(stations.ids, dtype=str),
                    event_xyz_m=events.coordinates_m, station_xyz_m=stations.coordinates_m))
                cache_ledger = {
                    **attempt, "status": "completed", "created_utc": _now(),
                    "elapsed_s": time.monotonic() - solve_clock, "clean_sha256": common.sha(cache),
                    "shape": list(clean.shape), "dtype": str(clean.dtype),
                    "time_reference": "absolute_propagation_time_s", "clean_times_path": str(cache),
                    "source_input_sha256": manifest["source_input_sha256"],
                    "runtime_sha256": manifest["runtime_sha256"],
                }
                _publish_json(cache_ledger_path, cache_ledger)
                clean.flags.writeable = False
            if complete.exists():
                return _verify_complete(common, manifest, common.read(complete), binding, cache_ledger)

            provenance = {
                "study_id": STUDY_ID, "manifest_digest": binding,
                "runtime_root": manifest["runtime_root"], "runtime_sha256": manifest["runtime_sha256"],
                "source_experiments_root": manifest["source_experiments_root"],
                "source_experiment_id": source_id, "source_input_sha256": manifest["source_input_sha256"],
                "source_output_sha256": manifest["source_output_sha256"],
                "clean_times_path": str(cache), "clean_sha256": cache_ledger["clean_sha256"],
                "clean_time_reference": "absolute_propagation_time_s",
                "forward_config": asdict(config), "forward_workers": 32,
            }
            if manifest_path is not None:
                provenance["manifest_path"] = str(manifest_path)
            stage("baseline_reproduction")
            baseline = _baseline_control(api, manifest, clean, events, stations)
            common.atomic(forward / "baseline_reproduction.json", {
                **baseline, "checked_utc": _now(), "manifest_digest": binding,
                "clean_sha256": cache_ledger["clean_sha256"]})
            if not baseline["allclose"] or baseline["max_abs_difference_s"] > BASELINE_ATOL_S:
                raise ValueError("Baseline reproduction failed closed: "
                                 f"max absolute difference {baseline['max_abs_difference_s']:.17g} s")

            ledger_levels = {}
            root = Path(manifest["experiments_root"])
            for level in manifest["levels"]:
                level_clock = time.monotonic()
                level_started = _now()
                stage("generating_noise_level", level_id=level["id"])
                noise = api.NoiseConfig(relative_sigma=level["relative_sigma"],
                                        absolute_sigma_s=level["absolute_sigma_s"], seed=level["seed"])
                times = noisy_relative(api, clean, events.ids, stations.ids, noise)
                record_path = forward / (level["id"] + ".json")
                if record_path.exists():
                    record = common.read(record_path)
                    _verify_record(common, manifest, level, record, binding)
                    _validate_level(api, manifest, level, times, events, stations)
                    ledger_levels[level["id"]] = record
                    continue
                input_dir = root / "input" / level["experiment_id"]
                if not os.path.lexists(input_dir):
                    with _staged_directory(input_dir) as temporary:
                        for name in INPUT_NAMES:
                            shutil.copyfile(source_input / name, temporary / name)
                        if _hashes(common, temporary, INPUT_NAMES) != manifest["source_input_sha256"]:
                            raise ValueError("Copied inputs differ from the frozen source")
                input_hashes = _hashes(common, input_dir, INPUT_NAMES)
                if input_hashes != manifest["source_input_sha256"]:
                    raise ValueError("Existing derived inputs differ from the frozen source")
                metadata = _level_metadata(source_metadata, api, level, provenance, baseline,
                                           time.monotonic() - level_clock)
                output_dir = root / "output" / level["experiment_id"]
                if not os.path.lexists(output_dir):
                    with _staged_directory(output_dir) as temporary:
                        _write_arrivals(temporary / "arrivals.csv", times, events, stations)
                        common.atomic(temporary / "metadata.json", metadata)
                else:
                    # Recover a directory published just before a crash, never replace it.
                    existing = common.read(output_dir / "metadata.json")
                    expected = dict(metadata)
                    existing = dict(existing)
                    expected.pop("elapsed_seconds", None)
                    elapsed = existing.pop("elapsed_seconds", None)
                    if existing != expected or not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
                        raise ValueError(f"Existing metadata mismatch: {level['id']}")
                _validate_level(api, manifest, level, times, events, stations)
                record = {
                    "status": "completed", "manifest_digest": binding, "level_id": level["id"],
                    "experiment_id": level["experiment_id"], "input_sha256": input_hashes,
                    "output_sha256": _hashes(common, output_dir, OUTPUT_NAMES),
                    "started_utc": level_started, "finished_utc": _now(),
                    "elapsed_s": time.monotonic() - level_clock, "pairs": int(times.size),
                }
                _publish_json(record_path, record)
                ledger_levels[level["id"]] = record
            stage("verifying_final_hashes")
            common.verify_manifest(manifest)
            if common.sha(cache) != cache_ledger["clean_sha256"]:
                raise ValueError("Clean cache changed during forward stage")
            ledger = {
                "schema_version": 1, "study_id": STUDY_ID, "status": "completed", "manifest_digest": binding,
                "started_utc": progress["started_utc"], "finished_utc": _now(),
                "elapsed_s": time.monotonic() - started, "clean_sha256": cache_ledger["clean_sha256"],
                "clean_times_sha256": cache_ledger["clean_sha256"], "clean_times_path": str(cache),
                "baseline_reproduction": baseline, "levels": ledger_levels, "provenance": provenance,
            }
            _verify_complete(common, manifest, ledger, binding, cache_ledger)
            _publish_json(complete, ledger)
            stage("completed", status="completed", level_id=None)
            return ledger
        except BaseException as error:
            failure = {**progress, "status": "failed", "finished_utc": _now(),
                       "elapsed_s": time.monotonic() - started, "error": str(error),
                       "traceback": traceback.format_exc()}
            # Preserve every failure, even if a later resume succeeds.
            fd, name = tempfile.mkstemp(prefix="failure-", suffix=".json", dir=forward)
            os.close(fd)
            common.atomic(Path(name), failure)
            stage("failed", status="failed", error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.manifest.is_absolute():
        parser.error("--manifest must be an absolute path")
    sys.dont_write_bytecode = True
    common = importlib.import_module("common")
    run(common.read(args.manifest), common, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
