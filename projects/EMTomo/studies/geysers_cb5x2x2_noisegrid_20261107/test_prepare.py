"""Tiny cached fixtures only: no FMM, real preparation, or inversions."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from projects.EMTomo.studies.geysers_cb5x2x2_noisegrid_20261107 import prepare as study
from projects.EMTomo.experiment_data import load_tomography_experiment
from projects.forward_modeling import experiments, solver
from projects.forward_modeling.model import PointSet, VelocityGrid
from projects.forward_modeling.noise import NoiseConfig, add_arrival_noise


BASELINE = "tiny_cached_baseline"
NOISE = NoiseConfig(0.01, 0.05, 42)
OUTPUT_FILES = ("arrivals.csv", "metadata.json", "clean_times.npz")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture(autouse=True)
def forbid_forward_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must never invoke a forward solver")

    for name in ("compute_travel_times", "compute_arrivals", "check_convergence"):
        monkeypatch.setattr(solver, name, forbidden)
    monkeypatch.setattr(experiments, "run_experiment", forbidden)


@pytest.fixture
def cached(tmp_path):
    root = tmp_path / "experiments"
    state = tmp_path / "state"
    model = VelocityGrid(np.array([3200., 5100.]).reshape(2, 1, 1), 1000.)
    stations = PointSet(("S2", "001", "S1"),
                        [[0., 0., 0.], [1200., 100., 0.], [2000., 1000., 0.]])
    events = PointSet(("E9", "E01", "E-3"),
                      [[100., 200., 700.], [1400., 900., 500.], [900., 500., 950.]])
    source_input = experiments.save_inputs(root, BASELINE, model, stations, events,
                                           generation={"fixture": "cached_not_solved", "seed": 17})
    output = root / "output" / BASELINE
    output.mkdir(parents=True)
    # Near-ties and very short absolute times expose clipping, wrong noise
    # reference, and scaling of already-normalized observations.
    absolute = np.array([[0.01, 0.012, 0.02], [1.7, 1.71, 2.4], [4., 2., 3.]])
    np.savez(output / "clean_times.npz", absolute_times_s=absolute,
             event_ids=np.asarray(events.ids), station_ids=np.asarray(stations.ids),
             event_coordinates_m=events.coordinates_m, station_coordinates_m=stations.coordinates_m)
    noisy = add_arrival_noise(absolute, events.ids, stations.ids, NOISE)
    relative = noisy - noisy.min(axis=1, keepdims=True)
    with (output / "arrivals.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "event_id", "arrival_time_s"))
        # The adapter must use IDs, not assume CSV row ordering.
        writer.writerows(reversed([(station, event, relative[e, s])
                                   for e, event in enumerate(events.ids)
                                   for s, station in enumerate(stations.ids)]))
    metadata = {
        "solver_name": "fixture_original_forward_solver",
        "solver_version": "fixture",
        "pykonal_version": "not_used_in_test",
        "config": {"refinement": 32, "source_radius_cells": 2},
        "source_seed": {"source_radius_cells": 2, "radius_cell_size_m": 31.25},
        "time_reference": "earliest_station_arrival_per_event",
        "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
                  "arrival_time": "s", "elapsed_time": "s"},
        "input_sha256": {p.name: sha256(p) for p in source_input.iterdir()},
        "elapsed_seconds": 571.25,
        "noise": {
            "enabled": True, "model": "independent_gaussian_relative_plus_absolute",
            "config": asdict(NOISE), "sigma_reference": "noiseless_absolute_travel_time_s",
            "application": "before_subtracting_earliest_noisy_arrival_per_event",
            "negative_picks": "not_clipped_before_normalization",
            "rng": "PCG64_pair_sha256_v1", "numpy_version": np.__version__,
            "convergence": "noiseless",
        },
        "absolute_times_cache": {
            "file": "clean_times.npz", "sha256": sha256(output / "clean_times.npz"),
            "array": "absolute_times_s", "axes": ["event", "station"],
            "time_reference": "event_origin", "units": "s", "noiseless": True,
        },
    }
    write_json(output / "metadata.json", metadata)
    return root, state, model, stations, events, absolute


def test_manifest_hashes_metadata_loaders_and_untouched_baseline(cached):
    root, state, model, stations, events, absolute = cached
    before = snapshot(root)
    manifest = study.prepare(root, BASELINE, state)
    assert manifest == json.loads((state / "data_manifest.json").read_text())
    assert {p.name for p in state.iterdir()} == {"data_manifest.json"}
    assert manifest["baseline_experiment_id"] == BASELINE
    assert manifest["experiments_root"] == str(root.resolve())
    assert [(r["id"], r["multiplier"]) for r in manifest["levels"]] == list(study.LEVELS)
    assert manifest["source_forward"]["metadata"]["elapsed_seconds"] == 571.25
    source_output = root / "output" / BASELINE
    for name, digest in manifest["source_forward"]["output_sha256"].items():
        assert digest == sha256(source_output / name)
    for record in manifest["levels"]:
        level, factor, identifier = record["id"], record["multiplier"], record["experiment_id"]
        expected_id = BASELINE if level == "x1" else f"geysers_cb5x2x2_real26_noisegrid_{level}_seed42"
        assert identifier == expected_id
        config = NoiseConfig(0.01 * factor, 0.05 * factor, 42)
        assert record["actual_noise"] == asdict(config)
        noisy = add_arrival_noise(absolute, events.ids, stations.ids, config)
        expected = noisy - noisy.min(axis=1, keepdims=True)
        loaded = load_tomography_experiment(identifier, root)
        np.testing.assert_array_equal(loaded.arrival_times_s, expected)
        np.testing.assert_array_equal(loaded.arrival_times_s.min(axis=1), np.zeros(len(events.ids)))
        assert loaded.noise_sigmas == (config.relative_sigma, config.absolute_sigma_s)
        assert loaded.station_ids == stations.ids
        assert loaded.event_ids == events.ids
        np.testing.assert_array_equal(loaded.station_coordinates_m, stations.coordinates_m)
        np.testing.assert_array_equal(loaded.reference_event_coordinates_m, events.coordinates_m)
        np.testing.assert_array_equal(loaded.source_model.velocity, model.velocity)
        assert loaded.source_model.cell_size_m == model.cell_size_m
        assert loaded.source_model.origin_m == model.origin_m
        input_dir, output_dir = root / "input" / identifier, root / "output" / identifier
        assert set(record["input_sha256"]) == {p.name for p in input_dir.iterdir()}
        assert set(record["output_sha256"]) == set(OUTPUT_FILES)
        for category in ("input", "output"):
            for name, digest in record[f"{category}_sha256"].items():
                assert digest == sha256(root / category / identifier / name)
        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["input_sha256"] == record["input_sha256"]
        assert metadata["noise"]["config"] == asdict(config)
        assert metadata["absolute_times_cache"]["sha256"] == sha256(output_dir / "clean_times.npz")
        assert (output_dir / "clean_times.npz").read_bytes() == (source_output / "clean_times.npz").read_bytes()
        assert json.loads((input_dir / "generation.json").read_text()) == {
            "fixture": "cached_not_solved", "seed": 17}
        if level != "x1":
            assert metadata["forward_solver_executed"] is False
            assert metadata["preparation_method"] == "cached_absolute_times_noise_grid"
            assert "solver_name" not in metadata
            assert 0 <= metadata["elapsed_seconds"] < 571.25
            assert "no_forward_solve" in metadata["elapsed_seconds_scope"]
            assert metadata["source_forward"] == manifest["source_forward"]
            assert metadata["config"] == {"refinement": 32, "source_radius_cells": 2}
    after = snapshot(root)
    assert all(after[path] == content for path, content in before.items())
    assert {p.name for p in (root / "input").iterdir()} == {
        r["experiment_id"] for r in manifest["levels"]}


def test_new_prefix_avoids_existing_nx6_experiments(cached):
    root, state, *_ = cached
    assert study.BASELINE_ID == "geysers_realgeom_cb5x2x2_20x8x8_26st_3465ev_noise01_pick005_seed42"
    assert study._experiment_id("x1", study.BASELINE_ID) == study.BASELINE_ID
    old_ids = {f"geysers_nx6_real31_noisegrid_{level}_seed42"
               for level, _ in study.LEVELS if level != "x1"}
    for identifier in old_ids:
        for section in ("input", "output"):
            target = root / section / identifier
            target.mkdir(parents=True, exist_ok=True)
            (target / "sentinel").write_text("old study must remain untouched")
    before = snapshot(root)
    manifest = study.prepare(root, BASELINE, state)
    new_ids = {record["experiment_id"] for record in manifest["levels"]}
    assert new_ids.isdisjoint(old_ids)
    assert new_ids == {BASELINE} | {
        f"geysers_cb5x2x2_real26_noisegrid_{level}_seed42"
        for level, _ in study.LEVELS if level != "x1"}
    after = snapshot(root)
    assert all(after[path] == content for path, content in before.items())


def test_common_random_scaling_on_absolute_times_before_normalization(cached, monkeypatch):
    root, state, _, stations, events, absolute = cached
    calls = []

    def capture(times, event_ids, station_ids, config):
        np.testing.assert_array_equal(times, absolute)
        assert tuple(event_ids) == events.ids
        assert tuple(station_ids) == stations.ids
        result = add_arrival_noise(times, event_ids, station_ids, config)
        calls.append((config, result.copy()))
        return result

    monkeypatch.setattr(study, "add_arrival_noise", capture)
    manifest = study.prepare(root, BASELINE, state)
    assert len(calls) == 5  # Baseline exactness check plus four genuinely new levels.
    unit_noisy = calls[0][1]
    assert np.any(unit_noisy < 0)  # Ensure this fixture actually exercises no clipping.
    unit_relative = unit_noisy - unit_noisy.min(axis=1, keepdims=True)
    for config, noisy in calls:
        factor = config.relative_sigma / NOISE.relative_sigma
        assert config.absolute_sigma_s == NOISE.absolute_sigma_s * factor
        assert config.seed == 42
        np.testing.assert_allclose(noisy - absolute, factor * (unit_noisy - absolute),
                                   rtol=0, atol=3e-15)
        expected = noisy - noisy.min(axis=1, keepdims=True)
        identifier = next(r["experiment_id"] for r in manifest["levels"] if r["multiplier"] == factor)
        np.testing.assert_array_equal(load_tomography_experiment(identifier, root).arrival_times_s, expected)
        if factor != 1:
            assert not np.allclose(expected, unit_relative * factor)


def test_baseline_must_regenerate_exactly_before_any_publication(cached):
    root, state, *_ = cached
    arrivals = root / "output" / BASELINE / "arrivals.csv"
    with arrivals.open(newline="") as stream:
        rows = list(csv.reader(stream))
    row = next(row for row in rows[1:] if float(row[2]) > 0)
    row[2] = str(float(row[2]) + 0.001)
    with arrivals.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    before = snapshot(root)
    with pytest.raises(ValueError, match="does not match exactly"):
        study.prepare(root, BASELINE, state)
    assert snapshot(root) == before
    assert not list(state.iterdir())


@pytest.mark.parametrize("corruption", ["hash", "event_ids", "station_ids", "event_coordinates_m",
                                        "station_coordinates_m", "nan", "negative", "shape", "fields"])
def test_reject_corrupt_cache_even_with_updated_hash(cached, corruption):
    root, state, *_ = cached
    output = root / "output" / BASELINE
    path = output / "clean_times.npz"
    metadata_path = output / "metadata.json"
    if corruption == "hash":
        with path.open("ab") as stream:
            stream.write(b"changed")
        error = "SHA256 mismatch"
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if corruption.endswith("ids"):
            arrays[corruption] = arrays[corruption][::-1]
            error = "IDs mismatch"
        elif corruption.endswith("coordinates_m"):
            arrays[corruption][0, 0] += 1
            error = "coordinates mismatch"
        elif corruption == "fields":
            arrays["unexpected"] = np.zeros(1)
            error = "Unexpected clean_times"
        else:
            if corruption == "shape":
                arrays["absolute_times_s"] = arrays["absolute_times_s"][:, :-1]
            else:
                arrays["absolute_times_s"][0, 0] = np.nan if corruption == "nan" else -1
            error = "Invalid cache"
        np.savez(path, **arrays)
        metadata = json.loads(metadata_path.read_text())
        metadata["absolute_times_cache"]["sha256"] = sha256(path)
        write_json(metadata_path, metadata)
    before = snapshot(root)
    with pytest.raises(ValueError, match=error):
        study.prepare(root, BASELINE, state)
    assert snapshot(root) == before
    assert not list(state.iterdir())


def test_reject_stale_input_hash(cached):
    root, state, *_ = cached
    path = root / "input" / BASELINE / "generation.json"
    write_json(path, {"tampered": True})
    before = snapshot(root)
    with pytest.raises(ValueError, match="Input SHA256 mismatch"):
        study.prepare(root, BASELINE, state)
    assert snapshot(root) == before
    assert not list(state.iterdir())


@pytest.mark.parametrize("section", ["input", "output"])
def test_preflight_rejects_existing_last_target_without_partial_publication(cached, section):
    root, state, *_ = cached
    target = root / section / "geysers_cb5x2x2_real26_noisegrid_x2_seed42"
    target.mkdir(parents=True)
    (target / "sentinel").write_text("must not replace")
    before = snapshot(root)
    with pytest.raises(FileExistsError, match="Target level already exists"):
        study.prepare(root, BASELINE, state)
    assert snapshot(root) == before
    assert not list(state.iterdir())


def test_reject_dangling_target_symlink(cached):
    root, state, *_ = cached
    target = root / "input" / "geysers_cb5x2x2_real26_noisegrid_x2_seed42"
    target.symlink_to(root / "missing")
    with pytest.raises(FileExistsError, match="Target level already exists"):
        study.prepare(root, BASELINE, state)
    assert target.is_symlink()
    assert len(list((root / "output").iterdir())) == 1


def test_reject_duplicate_preparation_and_existing_levels_in_another_state(cached):
    root, state, *_ = cached
    study.prepare(root, BASELINE, state)
    before, manifest_bytes = snapshot(root), (state / "data_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="Preparation already exists"):
        study.prepare(root, BASELINE, state)
    with pytest.raises(FileExistsError, match="Target level already exists"):
        study.prepare(root, BASELINE, state.parent / "other_state")
    assert snapshot(root) == before
    assert (state / "data_manifest.json").read_bytes() == manifest_bytes
    assert {p.name for p in state.iterdir()} == {"data_manifest.json"}


@pytest.mark.parametrize("name", ["data_manifest.json", ".prepare.lock"])
def test_existing_state_artifact_is_never_overwritten(cached, name):
    root, state, *_ = cached
    state.mkdir()
    (state / name).write_text("existing owner")
    before = snapshot(root)
    with pytest.raises(FileExistsError):
        study.prepare(root, BASELINE, state)
    assert snapshot(root) == before
    assert {p.name for p in state.iterdir()} == {name}
    assert (state / name).read_text() == "existing owner"


def test_inputs_without_optional_generation(cached):
    root, state, *_ = cached
    source = root / "input" / BASELINE
    (source / "generation.json").unlink()
    path = root / "output" / BASELINE / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["input_sha256"]["generation.json"]
    write_json(path, metadata)
    manifest = study.prepare(root, BASELINE, state)
    for record in manifest["levels"]:
        assert set(record["input_sha256"]) == {"model.npz", "events.csv", "stations.csv"}
        load_tomography_experiment(record["experiment_id"], root)


def test_public_entry_point_overrides_and_defaults(cached, monkeypatch, capsys):
    root, state, *_ = cached
    manifest = study.main(["--experiments-root", str(root), "--baseline-id", BASELINE,
                           "--state", str(state)])
    assert manifest == json.loads((state / "data_manifest.json").read_text())
    assert "Prepared 5 verified levels" in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(study, "prepare", lambda *args: calls.append(args) or {"levels": []})
    study.main([])
    assert calls == [(study.DEFAULT_EXPERIMENTS_ROOT, study.BASELINE_ID, study.DEFAULT_STATE)]
    assert study.DEFAULT_STATE == Path(study.__file__).resolve().parent
    assert study.DEFAULT_EXPERIMENTS_ROOT == Path(__file__).resolve().parents[4] / "projects/forward_modeling/experiments"
    # Direct script imports must work outside both repository and EMTomo roots.
    result = subprocess.run([sys.executable, str(Path(study.__file__).resolve()), "--help"],
                            cwd=state, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--experiments-root" in result.stdout
    assert "--baseline-id" in result.stdout
    assert "--state" in result.stdout
