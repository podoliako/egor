"""Small native fixtures; no forward/inversion, plus read-only real validation."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projects.EMTomo.studies.geysers_elevated import prepare
from projects.forward_modeling import solver
from projects.forward_modeling.experiments import load_inputs, save_inputs
from projects.forward_modeling.model import PointSet, VelocityGrid


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.stat().st_mtime_ns, path.read_bytes())
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(autouse=True)
def no_solves_or_resampling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not solve, generate arrivals, or resample data")

    monkeypatch.setattr(solver, "compute_arrivals", forbidden)
    monkeypatch.setattr(solver, "check_convergence", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    # Coarse, heterogeneous cube: same physical bounds, only 980 voxels.
    model = VelocityGrid(4000.0 + np.arange(20 * 7 * 7).reshape(20, 7, 7) / 8, 1000.0)
    stations = PointSet(("S-low", "S-high", "S-mid"),
                        np.asarray([[0, 7000, 0], [20000, 0, 0], [10000, 3500, 0]]))
    # Deliberately not ID-sorted: preserve the native source order.
    events = PointSet(("z-first", "drop-near", "b-bottom", "drop2", "a-last", "drop3", "drop4"),
                      np.asarray([[0, 0, 0], [1000, 1000, np.nextafter(5834.0, np.inf)],
                                  [20000, 7000, 5834], [3000, 2000, 6000],
                                  [9000, 4500, 1234.5], [7000, 1000, 6500], [2000, 3000, 7000]]))
    rows = [{"station_id": identifier, "elevation_m": elevation, "network": "fixture",
             "site_name": identifier, "start_utc": "2000-01-01T00:00:00Z", "end_utc": None,
             **dict(zip(("x_m", "y_m", "z_m"), xyz.tolist()))}
            for identifier, elevation, xyz in zip(stations.ids, (327.0, 1166.0, 803.0),
                                                   stations.coordinates_m)]
    generation = {
        "experiment_id": prepare.SOURCE_EXPERIMENT_ID, "generator": "immutable-fixture",
        "start_utc": prepare.START_UTC, "end_utc_exclusive": prepare.END_UTC_EXCLUSIVE,
        "station_count": 3, "event_count": 7, "pair_count": 21,
        "station_ids": list(stations.ids), "event_ids": list(events.ids),
        "station_metadata": rows[::-1],
        "event_origin_times_utc": {identifier: f"2011-04-01T00:00:0{i}.123000Z"
                                   for i, identifier in enumerate(events.ids)},
        "true_velocity_sha256": hashlib.sha256(model.velocity.tobytes()).hexdigest(),
        "geometry": {"station_z_m": 0.0, "fixture": "original geometry metadata"},
        "source_hashes": {"raw_data_not_to_be_read.csv": "a" * 64},
        "selection": "fixed native IDs, not resampled",
    }
    save_inputs(root, prepare.SOURCE_EXPERIMENT_ID, model, stations, events, generation)
    monkeypatch.setattr(prepare, "EXPECTED_STATION_COUNT", 3)
    monkeypatch.setattr(prepare, "EXPECTED_SOURCE_EVENT_COUNT", 7)
    monkeypatch.setattr(prepare, "EXPECTED_EVENT_COUNT", 3)
    return root, model, stations, events, generation


def _rewrite_source(sources, generation):
    path = sources[0] / "input" / prepare.SOURCE_EXPERIMENT_ID / "generation.json"
    path.write_text(json.dumps(generation), encoding="utf-8")


def test_transform_exclusion_closed_boundary_order_and_metadata(sources):
    root, original_model, original_stations, original_events, source = sources
    before = _snapshot(root / "input" / prepare.SOURCE_EXPERIMENT_ID)
    result = prepare.prepare_experiment(root)
    model, stations, events = load_inputs(root, prepare.EXPERIMENT_ID)
    assert result["status"] == "created"
    assert result["station_count"] == 3
    assert result["event_count"] == 3
    assert result["pair_count"] == 9
    assert result["dropped_event_count"] == 4
    assert stations.ids == original_stations.ids
    assert events.ids == ("z-first", "b-bottom", "a-last")
    np.testing.assert_array_equal(stations.coordinates_m[:, :2], original_stations.coordinates_m[:, :2])
    np.testing.assert_array_equal(stations.coordinates_m[:, 2], [839.0, 0.0, 363.0])
    retained = [0, 2, 4]
    np.testing.assert_array_equal(events.coordinates_m[:, :2], original_events.coordinates_m[retained, :2])
    np.testing.assert_array_equal(events.coordinates_m[:, 2], [1166.0, 7000.0, 2400.5])
    assert prepare._same_model(model, original_model)
    g = result["generation"]
    assert g["terrain"] is False and g["air"] is False
    assert g["shared_datum_assumption"]["verified"] is False
    assert g["top_elevation_m"] == 1166.0
    assert g["bottom_elevation_m"] == -5834.0
    assert set(g["event_origin_times_utc"]) == set(events.ids)
    assert g["event_origin_times_utc"] == {identifier: source["event_origin_times_utc"][identifier]
                                            for identifier in events.ids}
    original_rows = {row["station_id"]: row for row in source["station_metadata"]}
    assert g["source_station_metadata"] == [original_rows[identifier] for identifier in stations.ids]
    for row, xyz in zip(g["station_metadata"], stations.coordinates_m):
        assert row == {**original_rows[row["station_id"]], "z_m": xyz[2]}
    assert g["source_generation_metadata"]["geometry"] == source["geometry"]
    assert g["source_generation_metadata"]["source_hashes"] == source["source_hashes"]
    assert "event_origin_times_utc" not in g["source_generation_metadata"]
    assert g["dropped_event_ids"] == ["drop-near", "drop2", "drop3", "drop4"]
    for row in g["dropped_events"]:
        assert row["transformed_coordinates_m"][2] > 7000
        assert "outside_closed_model_domain" in row["reason"]
        assert row["origin_time_utc"] == source["event_origin_times_utc"][row["event_id"]]
    directory = Path(result["input_dir"])
    assert {path.name for path in directory.iterdir()} == set(prepare._SOURCE_FILES)
    assert json.loads((directory / "generation.json").read_text()) == g
    assert before == _snapshot(root / "input" / prepare.SOURCE_EXPERIMENT_ID)
    assert not (root / "output").exists()
    for name, digest in result["source_hashes"].items():
        assert digest == hashlib.sha256((root / "input" / prepare.SOURCE_EXPERIMENT_ID / name).read_bytes()).hexdigest()
    json.dumps(result, allow_nan=False)


def test_bitwise_velocity_grid_invariance_and_physical_xyz_distances(sources):
    root, original_model, original_stations, original_events, source = sources
    prepare.prepare_experiment(root)
    model, stations, events = load_inputs(root, prepare.EXPERIMENT_ID)
    assert model.velocity.dtype == original_model.velocity.dtype
    assert model.velocity.shape == original_model.velocity.shape
    assert model.velocity.tobytes() == original_model.velocity.tobytes()
    assert model.cell_size_m == original_model.cell_size_m
    assert model.origin_m == original_model.origin_m
    physical_stations = original_stations.coordinates_m.copy()
    metadata = {row["station_id"]: row for row in source["station_metadata"]}
    physical_stations[:, 2] = [-metadata[identifier]["elevation_m"] for identifier in stations.ids]
    physical_events = original_events.coordinates_m[[0, 2, 4]]
    physical_xyz = np.concatenate([physical_stations, physical_events])
    native_xyz = np.concatenate([stations.coordinates_m, events.coordinates_m])
    np.testing.assert_array_equal(native_xyz, physical_xyz + [0, 0, 1166])
    # Every station/station, event/event and station/event 3-D distance.
    np.testing.assert_allclose(np.linalg.norm(native_xyz[:, None] - native_xyz[None, :], axis=2),
                               np.linalg.norm(physical_xyz[:, None] - physical_xyz[None, :], axis=2),
                               rtol=0, atol=1e-11)
    # Not invariance relative to the old artificially flattened stations.
    assert not np.array_equal(stations.coordinates_m[:, 2], original_stations.coordinates_m[:, 2])


def test_validate_only_never_writes_and_reads_only_native_source(sources, monkeypatch):
    root = sources[0]
    before = _snapshot(root)
    calls = []
    native_load = prepare.load_inputs

    def track_load(root, experiment_id):
        calls.append(experiment_id)
        return native_load(root, experiment_id)

    def forbidden_save(*args, **kwargs):
        pytest.fail("validate_only must never save")

    monkeypatch.setattr(prepare, "load_inputs", track_load)
    monkeypatch.setattr(prepare, "save_inputs", forbidden_save)
    result = prepare.prepare_experiment(root, validate_only=True)
    assert result["status"] == "validated"
    assert calls == [prepare.SOURCE_EXPERIMENT_ID]
    assert not Path(result["input_dir"]).exists()
    assert before == _snapshot(root)


def test_exact_dataset_reuse_is_read_only(sources):
    first = prepare.prepare_experiment(sources[0])
    before = _snapshot(sources[0])
    second = prepare.prepare_experiment(sources[0])
    assert second == {**first, "status": "reused"}
    assert prepare.prepare_experiment(sources[0], validate_only=True) == {**first, "status": "validated"}
    assert before == _snapshot(sources[0])


@pytest.mark.parametrize("kind", ["velocity", "grid", "stations", "events", "generation",
                                  "missing_generation", "invalid_json", "broken_archive",
                                  "source_hash", "source_archive"])
def test_refuse_corrupt_or_different_existing_dataset_without_overwrite(sources, kind):
    root, model, _, _, _ = sources
    result = prepare.prepare_experiment(root)
    directory = Path(result["input_dir"])
    if kind in ("velocity", "grid"):
        velocity = model.velocity.copy()
        if kind == "velocity":
            velocity[0, 0, 0] = np.nextafter(velocity[0, 0, 0], np.inf)
        np.savez(directory / "model.npz", velocity=velocity,
                 cell_size_m=model.cell_size_m + (1 if kind == "grid" else 0), origin_m=model.origin_m)
    elif kind in ("stations", "events"):
        file = directory / f"{kind}.csv"
        # Valid CSV with changed semantics, not merely a broken file.
        text = file.read_text()
        file.write_text(text.replace(",0.0,", ",1.0,", 1))
    elif kind == "generation":
        file = directory / "generation.json"
        g = json.loads(file.read_text())
        g["shared_datum_assumption"]["verified"] = True
        file.write_text(json.dumps(g))
    elif kind == "missing_generation":
        (directory / "generation.json").unlink()
    elif kind == "invalid_json":
        (directory / "generation.json").write_text('{"x": NaN}')
    elif kind == "broken_archive":
        (directory / "model.npz").write_bytes(b"corrupt model")
    else:
        source_dir = root / "input" / prepare.SOURCE_EXPERIMENT_ID
        file = source_dir / ("generation.json" if kind == "source_hash" else "model.npz")
        # Semantics unchanged, but source byte provenance no longer matches.
        with file.open("ab") as stream:
            stream.write(b"\n")
    before = _snapshot(root)
    for validate_only in (False, True):
        with pytest.raises(ValueError, match="refusing overwrite"):
            prepare.prepare_experiment(root, validate_only=validate_only)
        assert before == _snapshot(root)


@pytest.mark.parametrize("kind,match", [
    ("duplicate_station", "station_metadata IDs"), ("missing_station", "station_metadata IDs"),
    ("nonfinite_elevation", "finite"), ("wrong_highest", "highest station"),
    ("station_xyz", "XYZ mismatch"), ("missing_time", "UTC time IDs"),
    ("extra_time", "UTC time IDs"), ("naive_time", "explicit UTC"),
    ("outside_time", "outside the original"), ("wrong_interval", "end_utc_exclusive mismatch"),
    ("wrong_count", "event_count mismatch"), ("wrong_ids", "station_ids mismatch"),
    ("velocity_hash", "true_velocity_sha256 mismatch"),
])
def test_invalid_source_metadata_is_rejected_without_writes(sources, kind, match):
    g = copy.deepcopy(sources[4])
    if kind == "duplicate_station":
        g["station_metadata"].append(copy.deepcopy(g["station_metadata"][0]))
    elif kind == "missing_station":
        g["station_metadata"].pop()
    elif kind == "nonfinite_elevation":
        g["station_metadata"][0]["elevation_m"] = "nan"
    elif kind == "wrong_highest":
        g["station_metadata"][1]["elevation_m"] = 1167
    elif kind == "station_xyz":
        g["station_metadata"][0]["x_m"] += 1
    elif kind == "missing_time":
        g["event_origin_times_utc"].pop("z-first")
    elif kind == "extra_time":
        g["event_origin_times_utc"]["extra"] = prepare.START_UTC
    elif kind == "naive_time":
        g["event_origin_times_utc"]["z-first"] = "2011-04-01T00:00:00"
    elif kind == "outside_time":
        g["event_origin_times_utc"]["z-first"] = prepare.END_UTC_EXCLUSIVE
    elif kind == "wrong_interval":
        g["end_utc_exclusive"] = "2011-04-20T00:00:00Z"
    elif kind == "wrong_count":
        g["event_count"] = 8
    elif kind == "wrong_ids":
        g["station_ids"].reverse()
    else:
        g["true_velocity_sha256"] = "0" * 64
    _rewrite_source(sources, g)
    before = _snapshot(sources[0])
    with pytest.raises(ValueError, match=match):
        prepare.prepare_experiment(sources[0])
    assert before == _snapshot(sources[0])


def test_bad_station_not_clipped_or_dropped(sources):
    g = copy.deepcopy(sources[4])
    g["station_metadata"][0]["elevation_m"] = -6000
    _rewrite_source(sources, g)
    with pytest.raises(ValueError, match="elevated stations outside"):
        prepare.prepare_experiment(sources[0])
    assert not (sources[0] / "input" / prepare.EXPERIMENT_ID).exists()


@pytest.mark.parametrize("constant", ["EXPECTED_EVENT_COUNT", "EXPECTED_DROPPED_COUNT",
                                     "EXPECTED_SOURCE_EVENT_COUNT"])
def test_count_mismatch_never_satisfied_by_sampling(sources, monkeypatch, constant):
    monkeypatch.setattr(prepare, constant, getattr(prepare, constant) + 1)
    with pytest.raises(ValueError, match="Expected|Unexpected source"):
        prepare.prepare_experiment(sources[0])
    assert not (sources[0] / "input" / prepare.EXPERIMENT_ID).exists()


def test_source_not_regridded_to_fit(sources):
    root, model, _, _, _ = sources
    file = root / "input" / prepare.SOURCE_EXPERIMENT_ID / "model.npz"
    np.savez(file, velocity=model.velocity, cell_size_m=1001.0, origin_m=model.origin_m)
    with pytest.raises(ValueError, match="Source model must cover"):
        prepare.prepare_experiment(root)


def test_source_changed_during_preparation_is_rejected(sources, monkeypatch):
    native_hashes = prepare._source_hashes
    calls = []

    def changed_hashes(directory):
        hashes = native_hashes(directory)
        if calls:
            hashes["generation.json"] = "0" * 64
        calls.append(directory)
        return hashes

    monkeypatch.setattr(prepare, "_source_hashes", changed_hashes)
    with pytest.raises(ValueError, match="Source files changed"):
        prepare.prepare_experiment(sources[0])
    assert not (sources[0] / "input" / prepare.EXPERIMENT_ID).exists()


def test_concurrent_identical_publication_is_verified_and_reused(sources, monkeypatch):
    def concurrent_save(*args, **kwargs):
        save_inputs(*args, **kwargs)
        raise FileExistsError("A cooperating preparer published first")

    monkeypatch.setattr(prepare, "save_inputs", concurrent_save)
    assert prepare.prepare_experiment(sources[0])["status"] == "reused"


def test_cli_json_validate_only(sources, capsys):
    before = _snapshot(sources[0])
    assert prepare.main(["--experiments-root", str(sources[0]), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert before == _snapshot(sources[0])


def test_direct_cli_help_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(prepare.__file__)), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert "--validate-only" in result.stdout


def test_real_native_source_validate_only():
    source_dir = prepare.DEFAULT_EXPERIMENTS_ROOT / "input" / prepare.SOURCE_EXPERIMENT_ID
    if not (source_dir / "generation.json").is_file():
        pytest.skip("Real immutable native source is not installed")
    target = prepare.DEFAULT_EXPERIMENTS_ROOT / "input" / prepare.EXPERIMENT_ID
    existed = target.exists()
    before_source, before_target = _snapshot(source_dir), _snapshot(target)
    result = prepare.prepare_experiment(validate_only=True)
    assert result["status"] == "validated"
    assert (result["event_count"], result["station_count"], result["pair_count"]) == (3776, 26, 98176)
    g = result["generation"]
    assert g["true_grid"] == {"shape": [160, 56, 56], "cell_size_m": 125.0, "origin_m": [0, 0, 0]}
    assert g["dropped_event_ids"] == ["3156222", "3260211", "3311973", "3159245"]
    assert set(g["event_origin_times_utc"]) == set(g["event_ids"])
    assert not set(g["dropped_event_ids"]) & set(g["event_ids"])
    assert len(g["event_origin_times_utc"]) == 3776
    assert len(g["station_metadata"]) == 26
    assert before_source == _snapshot(source_dir)
    assert before_target == _snapshot(target)
    assert target.exists() == existed
