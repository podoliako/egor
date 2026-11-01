"""Solve-free tests: all forward results are tiny synthetic absolute matrices."""
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
forward = _module("pick_noise_test_forward_impl", HERE / "forward.py")
# Import only the pure model/noise modules, not the FMM implementation.
MODEL = _module("pick_noise_test_model", HERE.parents[2] / "forward_modeling" / "model.py")
NOISE = _module("pick_noise_test_noise", HERE.parents[2] / "forward_modeling" / "noise.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".test-tmp")
    temporary.write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "EXPECTED_SHAPE", (2, 3))
    events = MODEL.PointSet(("001", "ev2"), np.array([[1., 2., 3.], [4., 5., 6.]]))
    stations = MODEL.PointSet(("01", "s2", "s3"), np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]]))
    clean = np.array([[.03, .2, .8], [.6, .1, .4]])
    source = tmp_path / "snapshot" / "projects" / "forward_modeling" / "experiments"
    source_id = forward.SOURCE_EXPERIMENT_ID
    source_input = source / "input" / source_id
    source_output = source / "output" / source_id
    source_input.mkdir(parents=True)
    source_output.mkdir(parents=True)
    for name in forward.INPUT_NAMES:
        (source_input / name).write_bytes((name + "\nunchanged frozen input\n").encode())
    input_hashes = {name: sha(source_input / name) for name in forward.INPUT_NAMES}
    units = {"coordinates": "m", "cell_size": "m", "velocity": "m/s", "arrival_time": "s", "elapsed_time": "s"}
    source_metadata = {
        "config": asdict(MODEL.ForwardConfig()), "input_sha256": input_hashes,
        "time_reference": "earliest_station_arrival_per_event", "units": units,
        "noise": {"enabled": True, "model": "independent_gaussian_relative_plus_absolute",
                  "config": asdict(NOISE.NoiseConfig(.01, .01, 42))},
        "solver_name": "synthetic_test_only", "source_seed": {"preserve": "all original metadata"},
        "receiver_interpolation": {"method": "factored_trilinear"},
        "elapsed_seconds": 123., "convergence": {"old": "not a new solve"},
    }
    atomic(source_output / "metadata.json", source_metadata)
    baseline = NOISE.add_arrival_noise(clean, events.ids, stations.ids, NOISE.NoiseConfig(.01, .01, 42))
    baseline -= baseline.min(axis=1, keepdims=True)
    forward._write_arrivals(source_output / "arrivals.csv", baseline, events, stations)
    state = tmp_path / "state"
    manifest = {
        "schema_version": 1, "study_id": forward.STUDY_ID, "state_dir": str(state),
        "runtime_root": str(tmp_path / "snapshot"), "runtime_sha256": {"numerical.py": "a" * 64},
        "source_experiments_root": str(source), "source_experiment_id": source_id,
        "source_input_sha256": input_hashes,
        "source_output_sha256": {name: sha(source_output / name) for name in forward.OUTPUT_NAMES},
        "experiments_root": str(state / "experiments"),
        "resources": {"forward_workers": 32, "max_parallel_runs": 3, "inversion_workers_per_run": 12},
        "levels": [{"id": identifier, "multiplier": multiplier, "relative_sigma": .01 * multiplier,
                    "absolute_sigma_s": .01 * multiplier, "seed": 42,
                    "experiment_id": "geysers_spheres_picknoise_20261027_" + identifier}
                   for identifier, multiplier in forward.LEVELS.items()],
        "cases": [{"id": str(i)} for i in range(15)],
    }

    def verify_manifest(value):
        for category, names in (("input", forward.INPUT_NAMES), ("output", forward.OUTPUT_NAMES)):
            directory = Path(value["source_experiments_root"]) / category / value["source_experiment_id"]
            assert {name: sha(directory / name) for name in names} == value["source_" + category + "_sha256"]

    common = SimpleNamespace(read=read, atomic=atomic, sha=sha, verify_manifest=Mock(side_effect=verify_manifest))
    model = object()
    api = SimpleNamespace(np=np, ForwardConfig=MODEL.ForwardConfig, NoiseConfig=NOISE.NoiseConfig,
                          add_arrival_noise=NOISE.add_arrival_noise,
                          load_inputs=Mock(return_value=(model, stations, events)),
                          compute_travel_times=Mock(return_value=clean.copy()))

    def load_experiment(experiment_id, experiments_root):
        root = Path(experiments_root)
        metadata = read(root / "output" / experiment_id / "metadata.json")
        assert metadata["time_reference"] == "earliest_station_arrival_per_event"
        assert metadata["units"] == units
        assert metadata["input_sha256"] == input_hashes
        assert metadata["noise"]["enabled"] is True
        assert metadata["noise"]["model"] == "independent_gaussian_relative_plus_absolute"
        for name, digest in input_hashes.items():
            assert sha(root / "input" / experiment_id / name) == digest
        times = np.full(clean.shape, np.nan)
        pairs = set()
        with (root / "output" / experiment_id / "arrivals.csv").open(newline="") as stream:
            rows = csv.DictReader(stream)
            assert rows.fieldnames == ["station_id", "event_id", "arrival_time_s"]
            for row in rows:
                pair = (row["event_id"], row["station_id"])
                assert pair not in pairs
                pairs.add(pair)
                times[events.ids.index(pair[0]), stations.ids.index(pair[1])] = float(row["arrival_time_s"])
        assert len(pairs) == clean.size
        assert np.isfinite(times).all() and (times >= 0).all()
        np.testing.assert_array_equal(times.min(axis=1), np.zeros(2))
        sigmas = metadata["noise"]["config"]
        # Deliberately reorder the baseline loader to exercise the ID-based gate.
        ei = [1, 0] if experiment_id == source_id else [0, 1]
        si = [2, 0, 1] if experiment_id == source_id else [0, 1, 2]
        return SimpleNamespace(
            arrival_times_s=times[np.ix_(ei, si)], event_ids=tuple(events.ids[i] for i in ei),
            station_ids=tuple(stations.ids[i] for i in si),
            station_coordinates_m=stations.coordinates_m[si],
            reference_event_coordinates_m=events.coordinates_m[ei],
            noise_sigmas=(sigmas["relative_sigma"], sigmas["absolute_sigma_s"]),
        )

    api.load_tomography_experiment = Mock(side_effect=load_experiment)
    monkeypatch.setattr(forward, "_load_api", Mock(return_value=api))
    return SimpleNamespace(manifest=manifest, common=common, api=api, state=state, clean=clean,
                           events=events, stations=stations, source_metadata=source_metadata,
                           source_input=source_input, source_output=source_output, model=model)


def run(study):
    return forward.run(study.manifest, study.common)


def test_normalize_after_noise_without_clipping():
    api = SimpleNamespace(np=np, add_arrival_noise=Mock(return_value=np.array([[-3., -1., 2.], [4., -2., 0.]])))
    clean = np.ones((2, 3))
    result = forward.noisy_relative(api, clean, ("e1", "e2"), ("s1", "s2", "s3"), object())
    np.testing.assert_array_equal(result, [[0., 2., 5.], [6., 0., 2.]])
    np.testing.assert_array_equal(clean, np.ones((2, 3)))


def test_pair_draws_deterministic_reordered_and_scaled():
    clean = np.array([[.1, .4, .8], [.3, 1., .7]])
    events, stations = ("001", "event"), ("01", "s2", "s3")
    base = NOISE.add_arrival_noise(clean, events, stations, NOISE.NoiseConfig(.01, .01, 42))
    repeated = NOISE.add_arrival_noise(clean, events, stations, NOISE.NoiseConfig(.01, .01, 42))
    np.testing.assert_array_equal(base, repeated)
    reordered = NOISE.add_arrival_noise(clean[::-1, ::-1], events[::-1], stations[::-1], NOISE.NoiseConfig(.01, .01, 42))
    np.testing.assert_array_equal(reordered[::-1, ::-1], base)
    for multiplier in forward.LEVELS.values():
        noisy = NOISE.add_arrival_noise(clean, events, stations,
                                       NOISE.NoiseConfig(.01 * multiplier, .01 * multiplier, 42))
        np.testing.assert_allclose(noisy - clean, multiplier * (base - clean), rtol=0., atol=5e-16)


def test_one_solve_cache_contract_metadata_and_hashes(study):
    ledger = run(study)
    study.api.compute_travel_times.assert_called_once_with(
        model=study.model, stations=study.stations, events=study.events,
        config=MODEL.ForwardConfig(refinement=2, source_radius_cells=2), workers=32)
    assert ledger["status"] == "completed"
    assert ledger["baseline_reproduction"]["pairs"] == 6
    assert ledger["baseline_reproduction"]["max_abs_difference_s"] == 0
    assert ledger["baseline_reproduction"]["rms_difference_s"] == 0
    cache = study.state / "forward" / "clean_times.npz"
    assert ledger["clean_sha256"] == sha(cache)
    with np.load(cache, allow_pickle=False) as saved:
        assert set(saved.files) == forward.CACHE_KEYS
        np.testing.assert_array_equal(saved["absolute_travel_times_s"], study.clean)
        np.testing.assert_array_equal(saved["event_ids"], study.events.ids)
        np.testing.assert_array_equal(saved["station_ids"], study.stations.ids)
        np.testing.assert_array_equal(saved["event_xyz_m"], study.events.coordinates_m)
        np.testing.assert_array_equal(saved["station_xyz_m"], study.stations.coordinates_m)
    for level in study.manifest["levels"]:
        record = ledger["levels"][level["id"]]
        assert record == read(study.state / "forward" / (level["id"] + ".json"))
        output = Path(study.manifest["experiments_root"]) / "output" / level["experiment_id"]
        assert record["output_sha256"] == {name: sha(output / name) for name in forward.OUTPUT_NAMES}
        assert record["input_sha256"] == study.manifest["source_input_sha256"]
        metadata = read(output / "metadata.json")
        assert metadata["source_seed"] == study.source_metadata["source_seed"]
        assert "convergence" not in metadata
        assert metadata["elapsed_seconds"] != 123.
        assert metadata["noise"]["config"] == {
            "relative_sigma": level["relative_sigma"], "absolute_sigma_s": level["absolute_sigma_s"], "seed": 42}
        assert metadata["noise"]["negative_picks"] == "not_clipped_before_normalization"
        assert metadata["provenance"]["clean_sha256"] == ledger["clean_sha256"]


def test_completed_resume_is_hash_verified_without_solve_or_writes(study):
    first = run(study)
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in study.state.rglob("*") if p.is_file()}
    study.api.compute_travel_times.side_effect = AssertionError("must never solve again")
    assert run(study) == first
    assert before == {p: (sha(p), p.stat().st_mtime_ns) for p in study.state.rglob("*") if p.is_file()}
    assert study.api.compute_travel_times.call_count == 1


def test_clean_cache_resume_after_downstream_failure_never_solves(study, monkeypatch):
    original = forward._baseline_control
    monkeypatch.setattr(forward, "_baseline_control", Mock(side_effect=RuntimeError("downstream failed")))
    with pytest.raises(RuntimeError, match="downstream failed"):
        run(study)
    cache = study.state / "forward" / "clean_times.npz"
    digest = sha(cache)
    assert read(study.state / "forward" / "clean_times.json")["status"] == "completed"
    assert list((study.state / "forward").glob("failure-*.json"))
    monkeypatch.setattr(forward, "_baseline_control", original)
    study.api.compute_travel_times.side_effect = AssertionError("must reuse clean cache")
    assert run(study)["clean_sha256"] == digest
    assert study.api.compute_travel_times.call_count == 1


def test_partial_level_resume_keeps_completed_bytes(study, monkeypatch):
    original = forward._validate_level

    def fail_second(api, manifest, level, *args):
        if level["id"] == "x2":
            raise RuntimeError("second level failed after publication")
        return original(api, manifest, level, *args)

    monkeypatch.setattr(forward, "_validate_level", fail_second)
    with pytest.raises(RuntimeError, match="second level"):
        run(study)
    root = Path(study.manifest["experiments_root"])
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(forward, "_validate_level", original)
    ledger = run(study)
    for path, expected in before.items():
        assert (sha(path), path.stat().st_mtime_ns) == expected
    assert len(ledger["levels"]) == 5
    assert study.api.compute_travel_times.call_count == 1


def test_baseline_mismatch_blocks_all_levels_and_keeps_cache(study):
    loader = study.api.load_tomography_experiment.side_effect

    def mismatch(*args, **kwargs):
        loaded = loader(*args, **kwargs)
        loaded.arrival_times_s[0, 0] += 1e-5
        return loaded

    study.api.load_tomography_experiment.side_effect = mismatch
    with pytest.raises(ValueError, match="Baseline reproduction failed closed"):
        run(study)
    assert not (study.state / "forward_complete.json").exists()
    assert not Path(study.manifest["experiments_root"]).exists()
    assert (study.state / "forward" / "clean_times.npz").is_file()
    baseline = read(study.state / "forward" / "baseline_reproduction.json")
    assert baseline["status"] == "failed" and baseline["pairs"] == 6
    assert baseline["max_abs_difference_s"] > 1e-8
    assert baseline["rms_difference_s"] > 0


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.array([[1., -1., 1.], [1., 1., 1.]]),
                                  np.full((2, 3), np.nan)])
def test_invalid_solver_result_is_preserved_and_not_retried(study, bad):
    study.api.compute_travel_times.return_value = bad
    with pytest.raises(ValueError, match="Invalid clean"):
        run(study)
    failure = study.state / "forward" / "solve_failed.json"
    assert read(failure)["status"] == "failed"
    before = sha(failure)
    with pytest.raises(RuntimeError, match="refusing a second solve"):
        run(study)
    assert sha(failure) == before
    assert study.api.compute_travel_times.call_count == 1
    assert not (study.state / "forward" / "clean_times.npz").exists()


def test_compute_exception_never_retried(study):
    study.api.compute_travel_times.side_effect = RuntimeError("synthetic FMM failure")
    with pytest.raises(RuntimeError, match="synthetic FMM failure"):
        run(study)
    with pytest.raises(RuntimeError, match="refusing a second solve"):
        run(study)
    assert study.api.compute_travel_times.call_count == 1


@pytest.mark.parametrize("target", ["clean", "input", "arrivals", "metadata"])
def test_completed_resume_rejects_changed_hashes(study, target):
    run(study)
    level = study.manifest["levels"][0]
    root = Path(study.manifest["experiments_root"])
    path = {
        "clean": study.state / "forward" / "clean_times.npz",
        "input": root / "input" / level["experiment_id"] / "events.csv",
        "arrivals": root / "output" / level["experiment_id"] / "arrivals.csv",
        "metadata": root / "output" / level["experiment_id"] / "metadata.json",
    }[target]
    path.write_bytes(path.read_bytes() + b"changed")
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        run(study)
    assert path.read_bytes() == changed
    assert study.api.compute_travel_times.call_count == 1


@pytest.mark.parametrize("key", ["event_ids", "station_ids", "event_xyz_m", "station_xyz_m", "absolute_travel_times_s"])
def test_cache_structure_ids_coordinates_and_shape_validated_even_with_matching_hash(study, key):
    run(study)
    cache = study.state / "forward" / "clean_times.npz"
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if key.endswith("ids"):
        arrays[key] = arrays[key][::-1]
    elif key.endswith("xyz_m"):
        arrays[key] = arrays[key] + 1
    else:
        arrays[key] = arrays[key][:, :1]
    np.savez(cache, **arrays)
    path = study.state / "forward" / "clean_times.json"
    ledger = read(path)
    ledger["clean_sha256"] = sha(cache)
    atomic(path, ledger)
    with pytest.raises(ValueError, match="mismatch|Invalid clean"):
        run(study)
    assert study.api.compute_travel_times.call_count == 1


def test_manifest_change_rejects_cache_without_solve(study):
    run(study)
    study.manifest["runtime_sha256"]["numerical.py"] = "b" * 64
    with pytest.raises(ValueError, match="manifest mismatch"):
        run(study)
    assert study.api.compute_travel_times.call_count == 1


def test_cache_without_valid_ledger_refuses_second_solve(study):
    run(study)
    (study.state / "forward" / "clean_times.json").unlink()
    with pytest.raises(RuntimeError, match="refusing a second solve"):
        run(study)
    assert study.api.compute_travel_times.call_count == 1


@pytest.mark.parametrize("change", ["seed", "sigma", "levels", "path", "workers", "source", "hashes"])
def test_reject_unapproved_plan_before_solve(study, change):
    if change == "seed":
        study.manifest["levels"][0]["seed"] = 43
    elif change == "sigma":
        study.manifest["levels"][0]["relative_sigma"] = .01
    elif change == "levels":
        study.manifest["levels"].pop()
    elif change == "path":
        study.manifest["experiments_root"] = "relative/path"
    elif change == "workers":
        study.manifest["resources"]["forward_workers"] = 64
    elif change == "source":
        study.manifest["source_experiment_id"] = "../unsafe"
    else:
        study.manifest["source_input_sha256"].pop("generation.json")
    # Isolate this function's plan checks from the parent's provenance checks.
    study.common.verify_manifest.side_effect = None
    with pytest.raises(ValueError):
        run(study)
    study.api.compute_travel_times.assert_not_called()


def test_exclusive_publication_never_overwrites_complete_bytes(tmp_path):
    path = tmp_path / "result.json"
    forward._publish_json(path, {"immutable": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        forward._publish_json(path, {"immutable": False})
    assert path.read_bytes() == before
    directory = tmp_path / "experiment"
    directory.mkdir()
    (directory / "existing").write_text("preserve")
    with pytest.raises(FileExistsError), forward._staged_directory(directory):
        pytest.fail("must not enter existing destination")
    assert (directory / "existing").read_text() == "preserve"


def test_exclusive_lock_rejects_concurrent_writer_and_allows_resume(tmp_path):
    path = tmp_path / ".lock"
    with forward._exclusive(path):
        with pytest.raises(BlockingIOError), forward._exclusive(path):
            pytest.fail("concurrent writer must not acquire the lock")
    with forward._exclusive(path):
        assert path.is_file()


@pytest.mark.parametrize("foreign", [None, "solver", "experiment_data"])
def test_load_api_uses_common_runtime_without_requiring_numpy_attribute(tmp_path, monkeypatch, foreign):
    root = tmp_path / "frozen"
    manifest = {"runtime_root": str(root)}
    modules = {}
    for name in ("experiments", "model", "noise", "solver"):
        module = SimpleNamespace(__file__=str(root / (name + ".py")))
        modules["projects.forward_modeling." + name] = module
    modules["projects.forward_modeling.experiments"].load_inputs = Mock()
    modules["projects.forward_modeling.model"].ForwardConfig = MODEL.ForwardConfig
    modules["projects.forward_modeling.noise"].NoiseConfig = NOISE.NoiseConfig
    modules["projects.forward_modeling.noise"].add_arrival_noise = NOISE.add_arrival_noise
    modules["projects.forward_modeling.solver"].compute_travel_times = Mock()
    modules["experiment_data"] = SimpleNamespace(__file__=str(root / "experiment_data.py"),
                                                load_tomography_experiment=Mock())
    modules["numpy"] = np
    if foreign is not None:
        key = "experiment_data" if foreign == "experiment_data" else "projects.forward_modeling." + foreign
        modules[key].__file__ = str(tmp_path / "mutable" / "foreign.py")
    common = SimpleNamespace(load_runtime=Mock(return_value=SimpleNamespace(config_module=None, runner=None, em=None)))
    monkeypatch.setattr(forward.importlib, "import_module", lambda name: modules[name])
    if foreign is not None:
        with pytest.raises(ValueError, match="Non-frozen forward import"):
            forward._load_api(manifest, common)
    else:
        api = forward._load_api(manifest, common)
        assert api.np is np
        assert api.add_arrival_noise is NOISE.add_arrival_noise
        api.compute_travel_times.assert_not_called()
    common.load_runtime.assert_called_once_with(manifest)


def test_frozen_metadata_loader_accepts_derived_metadata_with_mocked_input_io(study, monkeypatch):
    ledger = run(study)
    # Run the real EMTomo metadata ingestion path, but mock geometry/arrival IO.
    # This imports neither solver nor pykonal and never performs an FMM call.
    experiments = ModuleType("projects.forward_modeling.experiments")
    monkeypatch.setattr(experiments, "load_inputs", study.api.load_inputs, raising=False)

    def load_arrivals(root, experiment_id):
        with (Path(root) / "output" / experiment_id / "arrivals.csv").open(newline="") as stream:
            return [SimpleNamespace(station_id=row["station_id"], event_id=row["event_id"],
                                    arrival_time_s=float(row["arrival_time_s"])) for row in csv.DictReader(stream)]

    monkeypatch.setattr(experiments, "load_arrivals", load_arrivals, raising=False)
    monkeypatch.setitem(sys.modules, "projects.forward_modeling.experiments", experiments)
    monkeypatch.setitem(sys.modules, "projects.forward_modeling.model", MODEL)
    # The loader only inspects origin_m on its model in this path.
    study.api.load_inputs.return_value = (SimpleNamespace(origin_m=(0., 0., 0.)), study.stations, study.events)
    data = _module("pick_noise_test_experiment_data", HERE.parents[1] / "experiment_data.py")
    for level in study.manifest["levels"]:
        loaded = data.load_tomography_experiment(level["experiment_id"], study.manifest["experiments_root"])
        assert loaded.noise_sigmas == (level["relative_sigma"], level["absolute_sigma_s"])
        assert loaded.event_ids == study.events.ids
        assert loaded.station_ids == study.stations.ids
        assert loaded.arrival_times_s.shape == (2, 3)
        np.testing.assert_array_equal(loaded.arrival_times_s.min(axis=1), [0., 0.])
        assert ledger["levels"][level["id"]]["input_sha256"] == study.manifest["source_input_sha256"]
    assert study.api.compute_travel_times.call_count == 1
