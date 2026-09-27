"""File-based inputs and atomic results for forward-modeling experiments."""

from contextlib import contextmanager
from dataclasses import asdict
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter

import numpy as np

from .model import Arrival, ForwardConfig, PointSet, VelocityGrid
from . import solver


_POINT_COLUMNS = ("x_m", "y_m", "z_m")
_ARRIVAL_COLUMNS = ("station_id", "event_id", "arrival_time_s")
_INPUT_FILES = ("model.npz", "stations.csv", "events.csv")


def _experiment_path(root, category, experiment_id):
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id
    ):
        raise ValueError("experiment_id must start with an ASCII letter or digit and "
                         "contain only letters, digits, underscores, dots or hyphens")
    return Path(root) / category / experiment_id


@contextmanager
def _staged_directory(destination):
    """Serialize cooperating writers and publish only a complete directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.lock"
    # lexists also rejects dangling symlinks, which must not be replaced.
    if os.path.lexists(destination):
        raise FileExistsError(f"Experiment already exists: {destination}")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(fd)
        if os.path.lexists(destination):
            raise FileExistsError(f"Experiment already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        ))
        yield temporary
        if os.path.lexists(destination):
            raise FileExistsError(f"Experiment already exists: {destination}")
        temporary.rename(destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink()


def _write_points(path, id_column, points):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((id_column, *_POINT_COLUMNS))
        writer.writerows((identifier, *coordinates)
                         for identifier, coordinates in zip(points.ids, points.coordinates_m))


def _read_rows(path, columns):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != list(columns):
            raise ValueError(f"Invalid CSV header in {path}; expected {columns}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(columns) or any(value == "" for value in row):
                raise ValueError(f"Invalid CSV row in {path} at line {line_number}")
            yield row


def _load_points(path, id_column):
    rows = list(_read_rows(path, (id_column, *_POINT_COLUMNS)))
    return PointSet(
        ids=tuple(row[0] for row in rows),
        coordinates_m=np.asarray([row[1:] for row in rows], dtype=float).reshape(-1, 3),
    )


def save_inputs(root, experiment_id, model, stations, events, generation=None) -> Path:
    """Save immutable experiment inputs; optionally include generator provenance."""
    destination = _experiment_path(root, "input", experiment_id)
    if generation is not None:
        # Validate before writing the model. Reject NaN and non-JSON types.
        generation_json = json.dumps(generation, indent=2, allow_nan=False) + "\n"
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    with _staged_directory(destination) as temporary:
        np.savez(
            temporary / "model.npz",
            velocity=np.asarray(model.velocity, dtype=float),
            cell_size_m=np.asarray(model.cell_size_m, dtype=float),
            origin_m=np.asarray(model.origin_m, dtype=float),
        )
        _write_points(temporary / "stations.csv", "station_id", stations)
        _write_points(temporary / "events.csv", "event_id", events)
        if generation is not None:
            (temporary / "generation.json").write_text(generation_json, encoding="utf-8")
    return destination


def load_inputs(root, experiment_id):
    """Read inputs without pickle support; model constructors validate values."""
    directory = _experiment_path(root, "input", experiment_id)
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"velocity", "cell_size_m", "origin_m"}:
            raise ValueError("model.npz must contain velocity, cell_size_m and origin_m")
        cell_size = archive["cell_size_m"]
        origin = archive["origin_m"]
        if cell_size.shape != () or origin.shape != (3,):
            raise ValueError("Invalid cell_size_m or origin_m shape in model.npz")
        model = VelocityGrid(
            velocity=archive["velocity"],
            cell_size_m=float(cell_size),
            origin_m=tuple(float(value) for value in origin),
        )
    stations = _load_points(directory / "stations.csv", "station_id")
    events = _load_points(directory / "events.csv", "event_id")
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    return model, stations, events


def _validated_arrivals(rows):
    arrivals = []
    seen = set()
    for station_id, event_id, time in rows:
        station_id, event_id = str(station_id), str(event_id)
        time = float(time)
        if not station_id or not event_id or not np.isfinite(time) or time < 0:
            raise ValueError("Arrivals require nonempty IDs and finite nonnegative times")
        pair = (station_id, event_id)
        if pair in seen:
            raise ValueError(f"Duplicate arrival pair: {pair}")
        seen.add(pair)
        arrivals.append(Arrival(station_id=station_id, event_id=event_id, arrival_time_s=time))
    return arrivals


def load_arrivals(root, experiment_id) -> list[Arrival]:
    """Read validated relative arrivals, preserving CSV IDs as strings."""
    directory = _experiment_path(root, "output", experiment_id)
    return _validated_arrivals(_read_rows(directory / "arrivals.csv", _ARRIVAL_COLUMNS))


def _input_file_names(input_directory):
    return _INPUT_FILES + (("generation.json",) if (input_directory / "generation.json").exists() else ())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_experiment(root, experiment_id, config=ForwardConfig(), check_accuracy=False,
                   max_difference_s=None) -> Path:
    """Compute existing inputs and atomically publish results exactly once.

    Times are relative to the earliest station arrival for each event, not
    event origin times. Elapsed time includes the optional convergence check.
    A threshold requires relative and absolute differences at r and 2r to be
    within the limit; this convergence indicator is not a guaranteed error bound.
    """
    if max_difference_s is not None:
        try:
            max_difference_s = float(max_difference_s)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("max_difference_s must be finite and positive") from error
        if not np.isfinite(max_difference_s) or max_difference_s <= 0:
            raise ValueError("max_difference_s must be finite and positive")
    destination = _experiment_path(root, "output", experiment_id)
    with _staged_directory(destination) as temporary:
        input_directory = _experiment_path(root, "input", experiment_id)
        input_files = _input_file_names(input_directory)
        hashes = {name: _sha256(input_directory / name) for name in input_files}
        model, stations, events = load_inputs(root, experiment_id)
        started = perf_counter()
        arrivals = _validated_arrivals(
            (arrival.station_id, arrival.event_id, arrival.arrival_time_s)
            for arrival in solver.compute_arrivals(
                model=model, stations=stations, events=events, config=config
            )
        )
        convergence = None
        if check_accuracy or max_difference_s is not None:
            convergence = solver.check_convergence(
                model=model, stations=stations, events=events, config=config
            )
        if max_difference_s is not None:
            differences = (convergence["max_abs_difference_s"],
                           convergence["max_absolute_time_difference_s"])
            if not all(np.isfinite(value) and value >= 0 for value in differences):
                raise RuntimeError("Invalid convergence differences: expected finite nonnegative values")
            difference = max(differences)
            if difference > max_difference_s:
                raise RuntimeError(
                    f"Convergence difference {difference:g} s exceeds "
                    f"max_difference_s={max_difference_s:g} s (r versus 2r)"
                )
        elapsed = perf_counter() - started
        with (temporary / "arrivals.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_ARRIVAL_COLUMNS)
            writer.writerows((a.station_id, a.event_id, a.arrival_time_s) for a in arrivals)
        metadata = {
            "solver_name": "pykonal.EikonalSolver",
            "pykonal_version": version("pykonal"),
            "solver_version": solver.__version__,
            "config": asdict(config),
            "source_seed": {
                "method": "local_off_grid_straight_ray_slowness_integral",
                "source_radius_cells": config.source_radius_cells,
                "radius_cell_size_m": model.cell_size_m / config.refinement,
                "approximation": "local_straight_ray_upper_bounds_not_bent_rays",
            },
            "receiver_interpolation": {
                "method": "factored_trilinear",
                "factor": "T/d",
                "distance": "euclidean_distance_to_exact_source",
                "zero_distance_factor": "source_voxel_slowness",
            },
            "velocity_representation": {
                "input": "cell_centered_piecewise_constant_voxels",
                "origin": "lower_domain_corner",
                "numerical_grid": "nodes_with_adjacent_slowness_averaging_at_interfaces",
            },
            "time_reference": "earliest_station_arrival_per_event",
            "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
                      "arrival_time": "s", "elapsed_time": "s"},
            "input_sha256": hashes,
            "elapsed_seconds": elapsed,
        }
        if max_difference_s is not None:
            metadata["max_difference_s"] = max_difference_s
        if convergence is not None:
            metadata["convergence"] = convergence
        with (temporary / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            final_hashes = {name: _sha256(input_directory / name) for name in _input_file_names(input_directory)}
        except OSError as error:
            raise RuntimeError("Input files changed or became unreadable during computation") from error
        if final_hashes != hashes:
            raise RuntimeError("Input files changed during computation; output was not published")
    return destination
