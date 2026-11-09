"""Derive actual-observation-noise levels from a verified, cached forward solve.

No forward solver or inversion is run. ``prepare(root, baseline_id, state)``
publishes four immutable experiments and returns the manifest written exclusively
to ``state/data_manifest.json``. The x1 level reuses the untouched baseline.
Existing manifests or target input/output directories are errors, not resumptions.
After an interrupted publication, inspect partial experiments rather than overwrite.
"""

from contextlib import contextmanager
from copy import deepcopy
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

import numpy as np

# Direct execution from EMTomo (or any working directory), without study pipelines.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EMTOMO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _EMTOMO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from projects.EMTomo.experiment_data import load_tomography_experiment
from projects.forward_modeling.experiments import (
    _input_file_names,
    _sha256,
    _staged_directory,
    load_inputs,
    save_inputs,
)
from projects.forward_modeling.noise import NoiseConfig, add_arrival_noise


BASELINE_ID = "geysers_realgeom_cb5x2x2_20x8x8_26st_3465ev_noise01_pick005_seed42"
DEFAULT_EXPERIMENTS_ROOT = _REPO_ROOT / "projects/forward_modeling/experiments"
DEFAULT_STATE = Path(__file__).resolve().parent
LEVELS = (("x0p5", 0.5), ("x0p75", 0.75), ("x1", 1.0),
          ("x1p5", 1.5), ("x2", 2.0))
_OUTPUT_FILES = ("arrivals.csv", "metadata.json", "clean_times.npz")
_BASE_NOISE = NoiseConfig(relative_sigma=0.01, absolute_sigma_s=0.05, seed=42)


def _experiment_id(level, baseline_id):
    return (baseline_id if level == "x1"
            else f"geysers_cb5x2x2_real26_noisegrid_{level}_seed42")


def _hashes(directory, names):
    return {name: _sha256(directory / name) for name in names}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _relative_times(absolute_times, events, stations, noise):
    noisy = add_arrival_noise(absolute_times, events.ids, stations.ids, noise)
    return noisy - noisy.min(axis=1, keepdims=True)


def _load_cache(path, descriptor, events, stations):
    expected = {
        "file": "clean_times.npz", "array": "absolute_times_s",
        "axes": ["event", "station"], "time_reference": "event_origin",
        "units": "s", "noiseless": True,
    }
    if not isinstance(descriptor, dict) or any(descriptor.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported absolute_times_cache metadata")
    if descriptor.get("sha256") != _sha256(path):
        raise ValueError("Source clean_times.npz SHA256 mismatch")
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"absolute_times_s", "event_ids", "station_ids",
                                "event_coordinates_m", "station_coordinates_m"}:
            raise ValueError("Unexpected clean_times.npz fields")
        for name, points in (("event", events), ("station", stations)):
            ids = cache[f"{name}_ids"]
            if ids.dtype.kind != "U" or not np.array_equal(ids, np.asarray(points.ids)):
                raise ValueError(f"Cache {name} IDs mismatch")
            if not np.array_equal(cache[f"{name}_coordinates_m"], points.coordinates_m):
                raise ValueError(f"Cache {name} coordinates mismatch")
        times = cache["absolute_times_s"]
    if (times.shape != (len(events.ids), len(stations.ids))
            or times.dtype.kind != "f" or not np.isfinite(times).all() or np.any(times < 0)):
        raise ValueError("Invalid cache absolute_times_s")
    return times


def _verify_experiment(root, experiment_id, model, stations, events, expected, noise):
    loaded = load_tomography_experiment(experiment_id, root)
    if (loaded.event_ids != events.ids or loaded.station_ids != stations.ids
            or not np.array_equal(loaded.reference_event_coordinates_m, events.coordinates_m)
            or not np.array_equal(loaded.station_coordinates_m, stations.coordinates_m)
            or not np.array_equal(loaded.source_model.velocity, model.velocity)
            or loaded.source_model.cell_size_m != model.cell_size_m
            or loaded.source_model.origin_m != model.origin_m):
        raise ValueError(f"Input geometry/model mismatch for {experiment_id}")
    if loaded.noise_sigmas != (noise.relative_sigma, noise.absolute_sigma_s):
        raise ValueError(f"Actual noise mismatch for {experiment_id}")
    if not np.array_equal(loaded.arrival_times_s, expected):
        raise ValueError(f"Regenerated noisy matrix does not match exactly: {experiment_id}")


@contextmanager
def _preparation_lock(state):
    state.mkdir(parents=True, exist_ok=True)
    lock = state / ".prepare.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(fd)
        if os.path.lexists(state / "data_manifest.json"):
            raise FileExistsError(f"Preparation already exists: {state / 'data_manifest.json'}")
        yield
    finally:
        lock.unlink()


def _publish_manifest(state, manifest):
    # A hard link atomically publishes the complete file and never replaces even
    # a dangling symlink. Temporary state/lock files are removed on exit.
    fd, name = tempfile.mkstemp(prefix=".data_manifest-", dir=state)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.link(temporary, state / "data_manifest.json")
    finally:
        temporary.unlink()


def prepare(experiments_root, baseline_id, state):
    """Verify the source, publish cached-noise variants, and write one manifest.

    ``state`` is a directory, independent of the standard experiment root. All
    levels are checked for collisions before any experiment is written. Only
    observation-noise metadata is set here; inference/likelihood is out of scope.
    """
    root = Path(experiments_root).expanduser().resolve()
    state = Path(state).expanduser().resolve()
    with _preparation_lock(state):
        # load_inputs validates baseline_id before it can be used as a path.
        model, stations, events = load_inputs(root, baseline_id)
        targets = [_experiment_id(level, baseline_id) for level, _ in LEVELS if level != "x1"]
        if baseline_id in targets:
            raise ValueError("baseline_id collides with a derived experiment ID")
        for experiment_id in targets:
            for category in ("input", "output"):
                target = root / category / experiment_id
                if os.path.lexists(target):
                    raise FileExistsError(f"Target level already exists: {target}")

        source_input = root / "input" / baseline_id
        source_output = root / "output" / baseline_id
        source_inputs = _hashes(source_input, _input_file_names(source_input))
        source_outputs = _hashes(source_output, _OUTPUT_FILES)
        metadata = json.loads((source_output / "metadata.json").read_text(encoding="utf-8"))
        noise_metadata = metadata.get("noise", {})
        if (noise_metadata.get("enabled") is not True
                or noise_metadata.get("model") != "independent_gaussian_relative_plus_absolute"
                or noise_metadata.get("config") != asdict(_BASE_NOISE)
                or noise_metadata.get("rng") != "PCG64_pair_sha256_v1"
                or noise_metadata.get("sigma_reference") != "noiseless_absolute_travel_time_s"
                or noise_metadata.get("application") != "before_subtracting_earliest_noisy_arrival_per_event"):
            raise ValueError("Baseline noise must be relative .01 / absolute .05 s / seed 42 before normalization")
        absolute_times = _load_cache(source_output / "clean_times.npz",
                                     metadata.get("absolute_times_cache"), events, stations)
        baseline_times = _relative_times(absolute_times, events, stations, _BASE_NOISE)
        _verify_experiment(root, baseline_id, model, stations, events, baseline_times, _BASE_NOISE)
        if metadata.get("input_sha256") != source_inputs:
            raise ValueError("Source input SHA256 mismatch")
        generation = (json.loads((source_input / "generation.json").read_text(encoding="utf-8"))
                      if "generation.json" in source_inputs else None)
        source_forward = {
            "experiment_id": baseline_id,
            "input_sha256": source_inputs,
            "output_sha256": source_outputs,
            "metadata": metadata,
        }
        records = []
        for level, multiplier in LEVELS:
            experiment_id = _experiment_id(level, baseline_id)
            noise = NoiseConfig(_BASE_NOISE.relative_sigma * multiplier,
                                _BASE_NOISE.absolute_sigma_s * multiplier, _BASE_NOISE.seed)
            if level != "x1":
                started = perf_counter()
                expected = _relative_times(absolute_times, events, stations, noise)
                input_dir = save_inputs(root, experiment_id, model, stations, events, generation=generation)
                with _staged_directory(root / "output" / experiment_id) as temporary:
                    shutil.copyfile(source_output / "clean_times.npz", temporary / "clean_times.npz")
                    if _sha256(temporary / "clean_times.npz") != source_outputs["clean_times.npz"]:
                        raise ValueError("Source cache changed during preparation")
                    with (temporary / "arrivals.csv").open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("station_id", "event_id", "arrival_time_s"))
                        writer.writerows((station_id, event_id, expected[e, s])
                                         for e, event_id in enumerate(events.ids)
                                         for s, station_id in enumerate(stations.ids))
                    actual_noise = deepcopy(noise_metadata)
                    actual_noise.update(config=asdict(noise), numpy_version=np.__version__)
                    derived = {
                        "preparation_method": "cached_absolute_times_noise_grid",
                        "forward_solver_executed": False,
                        "config": deepcopy(metadata["config"]),
                        "time_reference": metadata["time_reference"],
                        "units": deepcopy(metadata["units"]),
                        "input_sha256": _hashes(input_dir, _input_file_names(input_dir)),
                        "noise": actual_noise,
                        "absolute_times_cache": deepcopy(metadata["absolute_times_cache"]),
                        "source_forward": source_forward,
                        "elapsed_seconds": perf_counter() - started,
                        "elapsed_seconds_scope": "noise_generation_and_serialization_excluding_validation_no_forward_solve",
                    }
                    _write_json(temporary / "metadata.json", derived)
                _verify_experiment(root, experiment_id, model, stations, events, expected, noise)
            input_dir = root / "input" / experiment_id
            records.append({
                "id": level, "multiplier": multiplier, "experiment_id": experiment_id,
                "actual_noise": asdict(noise),
                "input_sha256": _hashes(input_dir, _input_file_names(input_dir)),
                "output_sha256": _hashes(root / "output" / experiment_id, _OUTPUT_FILES),
            })

        # Do not certify a source that changed while derived files were written.
        if (_hashes(source_input, _input_file_names(source_input)) != source_inputs
                or _hashes(source_output, _OUTPUT_FILES) != source_outputs):
            raise ValueError("Source experiment changed during preparation")
        manifest = {
            "baseline_experiment_id": baseline_id,
            "experiments_root": str(root),
            "levels": records,
            "source_forward": source_forward,
        }
        _publish_manifest(state, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--baseline-id", default=BASELINE_ID)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)
    manifest = prepare(args.experiments_root, args.baseline_id, args.state)
    print(f"Prepared {len(manifest['levels'])} verified levels: {args.state / 'data_manifest.json'}")
    return manifest


if __name__ == "__main__":
    main()
