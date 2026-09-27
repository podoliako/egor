"""Read saved forward experiments for tomography without constructing event priors.

``reference_event_coordinates_m`` and ``source_model`` are ground truth for
assessment only. An inversion should use station geometry and relative arrivals,
not the reference event coordinates as its initial event locations.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

try:
    from projects.forward_modeling.experiments import load_arrivals, load_inputs
    from projects.forward_modeling.model import VelocityGrid
except ModuleNotFoundError as error:
    if error.name != "projects":
        raise
    # Also support importing experiment_data directly from the EMTomo directory.
    _repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, _repo_root)
    try:
        from projects.forward_modeling.experiments import load_arrivals, load_inputs
        from projects.forward_modeling.model import VelocityGrid
    finally:
        sys.path.remove(_repo_root)


DEFAULT_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "forward_modeling" / "experiments"
_UNITS = {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
          "arrival_time": "s", "elapsed_time": "s"}


@dataclass(frozen=True)
class TomographyExperiment:
    """Arrivals shape (n_events, n_stations), in event_ids/station_ids order.

    source_model and reference_event_coordinates_m are truth for metrics only;
    neither supplies event-location priors to an inversion.
    """

    source_model: VelocityGrid
    station_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    station_coordinates_m: np.ndarray
    reference_event_coordinates_m: np.ndarray
    arrival_times_s: np.ndarray


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tomography_experiment(experiment_id, experiments_root=DEFAULT_EXPERIMENTS_ROOT) -> TomographyExperiment:
    """Load a complete SI-unit experiment with earliest-station-relative times.

    Rejects stale inputs, unsupported conventions, missing/unknown arrival pairs,
    and events whose first observed arrival is not zero (within 1e-8 s).
    """
    # The forward loader validates experiment_id before accessing any paths.
    model, stations, events = load_inputs(experiments_root, experiment_id)
    root = Path(experiments_root)
    metadata_path = root / "output" / experiment_id / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing experiment metadata: {metadata_path}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed experiment metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Malformed experiment metadata: {metadata_path} must be an object")
    if metadata.get("time_reference") != "earliest_station_arrival_per_event":
        raise ValueError("Unsupported metadata time_reference: expected earliest_station_arrival_per_event")
    units = metadata.get("units")
    if not isinstance(units, dict) or any(units.get(k) != v for k, v in _UNITS.items()):
        raise ValueError("Unsupported or missing metadata units (expected SI units)")
    input_dir = root / "input" / experiment_id
    names = ("model.npz", "stations.csv", "events.csv")
    if (input_dir / "generation.json").exists():
        names += ("generation.json",)
    hashes = metadata.get("input_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(names):
        raise ValueError("Missing or mismatched metadata input_sha256 file list")
    for name in names:
        if hashes[name] != _sha256(input_dir / name):
            raise ValueError(f"Input SHA256 mismatch for {name}")
    if model.origin_m != (0.0, 0.0, 0.0):
        raise ValueError("Only zero-origin velocity grids are supported by EMTomo")
    if len(stations.ids) < 2:
        raise ValueError("Tomography requires at least two stations")

    station_index = {identifier: i for i, identifier in enumerate(stations.ids)}
    event_index = {identifier: i for i, identifier in enumerate(events.ids)}
    times = np.full((len(events.ids), len(stations.ids)), np.nan, dtype=np.float64)
    for arrival in load_arrivals(experiments_root, experiment_id):
        if arrival.station_id not in station_index or arrival.event_id not in event_index:
            raise ValueError(f"Unknown arrival ID pair: ({arrival.station_id!r}, {arrival.event_id!r})")
        times[event_index[arrival.event_id], station_index[arrival.station_id]] = arrival.arrival_time_s
    if np.isnan(times).any():
        raise ValueError("Incomplete arrivals: expected exactly one row for every event/station ID pair")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("Arrival times must be finite and nonnegative")
    if np.any(np.min(times, axis=1) > 1e-8):
        raise ValueError("Each event must have a zero earliest-station arrival (within 1e-8 s)")
    times.flags.writeable = False
    return TomographyExperiment(model, stations.ids, events.ids, stations.coordinates_m,
                                events.coordinates_m, times)
