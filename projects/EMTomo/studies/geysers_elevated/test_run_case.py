"""Elevated runner contracts and real production wiring, without numerical solves.

Failing regression assertions deliberately expose runner bugs rather than patching
production/shared helpers or silently marking unsafe behavior as expected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.EMTomo.config import InversionConfig
from projects.EMTomo.studies.geysers2011 import run_case as oldrunner
from projects.EMTomo.studies.geysers_elevated import run_case as study
from projects.forward_modeling.experiments import save_inputs
from projects.forward_modeling.model import PointSet, VelocityGrid

CASE_IDENTITIES = [(mode, k) for mode, ks in (("hard", (1, 5, 10, 15)), ("soft", (5, 10, 15)))
                   for k in ks]
CASES = [{"id": f"{mode}-k{k}", "run_name": f"gy-elev-{mode}-k{k}",
          "candidate_mode": mode, "n_candidates": k, "weights_top_n": 1 if mode == "hard" else k,
          "temperature": 1.0 if mode == "hard" else None,
          "lambda_reg": 1.0, "coverage_damping_power": 2.0,
          "is_reference": mode == "hard" and k == 5}
         for mode, k in CASE_IDENTITIES]
CASE_RUNTIME_FIELDS = {"candidate_mode", "n_candidates", "weights_top_n", "temperature",
                       "n_workers", "run_name", "runs_dir", "save_runs"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    oldrunner.atomic_json(path, value)


def snapshot(path):
    return {str(file.relative_to(path)): (file.stat().st_mtime_ns, file.read_bytes())
            for file in path.rglob("*") if file.is_file()}


def baseline_config():
    config = InversionConfig(
        cell_size=1000.0, subdivision=8, n_cycles=12, candidate_mode="hard",
        n_candidates=5, weights_top_n=1, temperature=1.0,
        initial_layer_boundaries_km=(0.7, 1.9, 3.4),
        initial_layer_velocities_m_s=(3800.0, 4600.0, 5200.0, 5700.0),
        lambda_reg=1.0, coverage_damping_power=2.0, max_velocity_step_fraction=0.02,
        weight_noise_relative_sigma=0.01, weight_noise_absolute_sigma_s=0.01,
        weight_model_sigma_s=0.02, n_workers=12,
    )
    return oldrunner._json({key: value for key, value in asdict(config).items()
                            if key not in {"runs_dir", "save_runs"}})


def manifest_for(tmp_path, runtime=None):
    return {
        "schema_version": 2, "experiment_id": study.EXPERIMENT_ID,
        "forward_implementation": oldrunner.FORWARD_IMPLEMENTATION,
        "created_utc": "2026-10-18T00:00:00Z",
        "experiments_root": str(tmp_path / "experiments"), "runs_dir": str(tmp_path / "runs"),
        "runtime_root": str(runtime or tmp_path / "runtime"),
        "reference_run": str(tmp_path / "reference-latest-lam1-cov2-legacy"),
        "resources": dict(oldrunner.RESOURCE_VALUES), "baseline_config": baseline_config(),
        "input_sha256": {name: "0" * 64 for name in oldrunner.INPUT_FILES},
        "code_sha256": {"projects/EMTomo/eikonal.py": "0" * 64},
        "reference_sha256": {name: "0" * 64 for name in ("meta.json", "initial_model.npy")},
        "cases": copy.deepcopy(CASES),
    }


def save_reference(manifest):
    reference = Path(manifest["reference_run"])
    write_json(reference / "meta.json", {"run_params": {
        **manifest["baseline_config"], "travel_times": oldrunner.SAVED_TRAVEL_TIMES}})
    profile = np.asarray([3800.0, 4600.0, 5200.0, 5700.0, 5700.0, 5700.0, 5700.0])
    np.save(reference / "initial_model.npy", np.broadcast_to(profile, (20, 7, 7)))
    manifest["reference_sha256"] = {name: oldrunner.sha256(reference / name)
                                     for name in ("meta.json", "initial_model.npy")}


@pytest.fixture
def config_study(tmp_path):
    manifest = manifest_for(tmp_path)
    save_reference(manifest)
    return manifest, SimpleNamespace(InversionConfig=InversionConfig)


@pytest.fixture(scope="module")
def runtime_snapshot(tmp_path_factory):
    """Same frozen legacy sphere implementation used by the old runner tests."""
    runtime = tmp_path_factory.mktemp("elevated-runtime")
    for relative in ("projects/EMTomo", "projects/EMTomo/instruments",
                     "projects/EMTomo/tomography", "projects/forward_modeling"):
        destination = runtime / relative
        destination.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / relative).glob("*.py"):
            shutil.copy2(source, destination / source.name)
    legacy = subprocess.run(
        ["git", "--no-pager", "show", "c381cfd:projects/EMTomo/eikonal.py"],
        cwd=ROOT, capture_output=True, check=True, timeout=10,
    ).stdout
    (runtime / "projects/EMTomo/eikonal.py").write_bytes(legacy)
    runner = runtime / "runner"
    runner.mkdir()
    shutil.copy2(study.__file__, runner / "run_case.py")
    shutil.copy2(oldrunner.__file__, runner / "_shared_case.py")
    return runtime


@pytest.fixture
def saved_study(tmp_path, runtime_snapshot):
    """All 98,176 pairs, coarse native truth, hand-written observed arrivals."""
    manifest = manifest_for(tmp_path, runtime_snapshot)
    root = Path(manifest["experiments_root"])
    station_xyz = np.column_stack((np.linspace(100.0, 19900.0, 26), np.full(26, 3500.0),
                                   np.linspace(0.0, 839.0, 26)))
    stations = PointSet(tuple(f"s{i:02}" for i in range(26)), station_xyz)
    events = PointSet(tuple(f"e{i:04}" for i in range(3776)), np.tile([9000.0, 3000.0, 3166.0], (3776, 1)))
    generation = {"start_utc": "2011-04-01T00:00:00Z", "end_utc_exclusive": "2011-04-21T00:00:00Z",
                  "station_count": 26, "event_count": 3776, "pair_count": 98176,
                  "top_elevation_m": 1166.0, "terrain": False, "air": False}
    directory = save_inputs(root, manifest["experiment_id"],
                            VelocityGrid(np.full((20, 7, 7), 5000.0), 1000.0), stations, events, generation)
    manifest["input_sha256"] = {name: oldrunner.sha256(directory / name) for name in oldrunner.INPUT_FILES}
    manifest["code_sha256"] = {path.relative_to(runtime_snapshot).as_posix(): oldrunner.sha256(path)
                                for path in runtime_snapshot.rglob("*.py")}
    output = root / "output" / manifest["experiment_id"]
    output.mkdir(parents=True)
    with (output / "arrivals.csv").open("w", encoding="utf-8") as stream:
        stream.write("station_id,event_id,arrival_time_s\n")
        for event in events.ids:
            stream.writelines(f"{station},{event},{i / 100}\n" for i, station in enumerate(stations.ids))
    write_json(output / "metadata.json", {
        "time_reference": "earliest_station_arrival_per_event",
        "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
                  "arrival_time": "s", "elapsed_time": "s"},
        "input_sha256": manifest["input_sha256"],
        "noise": {"enabled": True, "model": "independent_gaussian_relative_plus_absolute",
                  "config": {"relative_sigma": 0.01, "absolute_sigma_s": 0.01, "seed": 42}},
    })
    # Calibration is already computed/published by the pipeline, not by a case.
    calibration = {"schema_version": 1, "temperature": 4.25, "sample_count": 128,
                   "event_count": 3776, "station_count": 26,
                   "cases": [{key: case[key] for key in ("candidate_mode", "n_candidates", "weights_top_n")}
                             | {"temperature": 1.0 if case["candidate_mode"] == "hard" else 4.25}
                             for case in manifest["cases"]]}
    write_json(tmp_path / "calibration.json", calibration)
    ledger = {"schema_version": 2, "experiment_id": manifest["experiment_id"], "status": "complete",
              "output_sha256": {name: oldrunner.sha256(output / name)
                                  for name in ("arrivals.csv", "metadata.json")},
              "calibration_sha256": oldrunner.sha256(tmp_path / "calibration.json")}
    write_json(tmp_path / "forward_complete.json", ledger)
    save_reference(manifest)
    for case in manifest["cases"]:
        (tmp_path / "cases" / case["id"]).mkdir(parents=True)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return SimpleNamespace(path=path, manifest=manifest, state=tmp_path, output=output,
                           ledger=ledger, calibration=calibration)


# Child processes isolate bare production imports, BLAS configuration and mocks.
# prepare_inversion, logger, run_em, updates and completion publication remain real;
# only the expensive numerical step and JIT warm-up are replaced.
CHILD_SETUP = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_case as study
shared = study.shared
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
study.verify_hashes(manifest)
assert 'numpy' not in sys.modules, 'production imported before snapshot selection'
runtime = shared.load_runtime(manifest)
import eikonal, experiment_runner, raytracing
assert Path(eikonal.__file__).resolve().is_relative_to(Path(manifest['runtime_root']))
assert Path(shared.__file__).name == '_shared_case.py'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    assert os.environ[key] == '12'
assert runtime.em.run_em.__module__ == 'tomography.tomography_em'
assert runtime.TomographyLogger.__module__ == 'tomography.tomography_logging'
old_description = runtime.em.METHOD_DESCRIPTION
np = runtime.np
calls, preparations, warms = [], [], []
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected forward/ray/likelihood/linear solve')
for module, names in ((eikonal, ('station_travel_time_fields', 'travel_time_field')),
                      (runtime.em, ('station_travel_time_fields', 'run_events', 'solve_slowness_update',
                                    'compute_G_all_stations', 'compute_G_all_stations_serial')),
                      (raytracing, ('compute_G_all_stations', 'compute_G_all_stations_serial'))):
    for name in names:
        setattr(module, name, forbidden)
runtime.em.warm_up_jit = lambda: warms.append(True)
real_prepare = experiment_runner.prepare_inversion
def prepare(*args, **kwargs):
    assert args[0] == manifest['experiment_id']
    assert kwargs['experiments_root'] == Path(manifest['experiments_root'])
    prepared = real_prepare(*args, **kwargs)
    assert prepared.arrivals_table.shape == (3776, 26)
    assert prepared.station_locs.shape == (26, 3)
    assert prepared.station_locs[0, 2] == 0 and prepared.station_locs[-1, 2] == 839
    assert prepared.noise_sigmas == (.01, .01)
    preparations.append(prepared)
    return prepared
experiment_runner.prepare_inversion = prepare
def step(model, arrivals, stations, noise, config, *, iteration, logger):
    assert arrivals.shape == (3776, 26) and np.all(arrivals[:, 0] == 0)
    assert stations.shape == (26, 3) and np.count_nonzero(stations[:, 2]) == 25
    assert config.n_workers == 12 and config.n_cycles == 12
    assert config.lambda_reg == 1 and config.coverage_damping_power == 2
    assert noise.relative_sigma == .01 and noise.absolute_sigma_s == .01 and noise.model_sigma_s == .02
    assert runtime.em.METHOD_DESCRIPTION['travel_times'] == shared.SAVED_TRAVEL_TIMES
    assert isinstance(logger, runtime.TomographyLogger)
    calls.append((config.candidate_mode, config.n_candidates, config.weights_top_n, config.temperature, iteration))
    return np.zeros(model.shape), np.ones(model.shape), np.ones(model.shape)
runtime.em.make_tomography_step = step
'''

CHILD_RUN = CHILD_SETUP + r'''
for case_id in sys.argv[3:]:
    case = next(case for case in manifest['cases'] if case['id'] == case_id)
    result = study.run_case(manifest_path, case_id)
    assert result['status'] == 'complete'
    assert runtime.em.METHOD_DESCRIPTION is old_description
    expected = (case['candidate_mode'], case['n_candidates'], case['weights_top_n'],
                1. if case['candidate_mode'] == 'hard' else 4.25)
    assert calls[-12:] == [(*expected, i) for i in range(12)]
    run = Path(result['run_dir'])
    def snapshot(root):
        return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes())
                for p in root.rglob('*') if p.is_file()}
    result_path = manifest_path.parent / 'cases' / case_id / 'result.json'
    before = (snapshot(run), result_path.read_bytes(), result_path.stat().st_mtime_ns)
    count = len(calls), len(preparations), len(warms)
    assert study.run_case(manifest_path, case_id) == result
    assert count == (len(calls), len(preparations), len(warms)), 'complete result was recomputed'
    assert before == (snapshot(run), result_path.read_bytes(), result_path.stat().st_mtime_ns)
assert len(calls) == 12 * len(sys.argv[3:])
assert len(preparations) == len(warms) == len(sys.argv[3:])
'''


def run_child(saved, cases=(), code=CHILD_RUN):
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(saved.manifest["runtime_root"]) / "runner"),
         str(saved.path), *cases],
        cwd=saved.state, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def read_result(saved, case_id="hard-k5"):
    return oldrunner._read_json(saved.state / "cases" / case_id / "result.json")


@pytest.fixture
def completed_study(saved_study):
    run_child(saved_study, ["hard-k5"])
    return saved_study


def test_shared_helpers_are_oldrunner_without_production_imports():
    assert study.shared is oldrunner
    for name in ("verify_hashes", "load_runtime", "provenance", "make_logger", "verify_initial",
                 "check_completed_result", "atomic_json", "_quality"):
        assert getattr(study.shared, name) is getattr(oldrunner, name)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_build_config_exact_baseline_except_mode_k_topn_temperature_runtime(config_study, case):
    manifest, runtime = config_study
    before = copy.deepcopy(manifest)
    reference_before = snapshot(Path(manifest["reference_run"]))
    calibration = {"temperature": 4.25}
    config = study.build_config(manifest, case, runtime, calibration)
    actual = oldrunner._json(asdict(config))
    for key, value in manifest["baseline_config"].items():
        if key not in CASE_RUNTIME_FIELDS:
            assert actual[key] == value, f"unexpected baseline override: {key}"
    assert config.candidate_mode == case["candidate_mode"]
    assert config.n_candidates == case["n_candidates"]
    assert config.weights_top_n == (case["n_candidates"] if case["candidate_mode"] == "soft" else 1)
    assert config.temperature == (4.25 if case["candidate_mode"] == "soft" else 1.0)
    assert config.n_workers == 12 and config.run_name == case["run_name"]
    assert config.runs_dir == manifest["runs_dir"] and config.save_runs is True
    assert manifest == before and calibration == {"temperature": 4.25}
    assert reference_before == snapshot(Path(manifest["reference_run"]))


@pytest.mark.parametrize("temperature", [None, 0.99, 16.01, float("nan"), float("inf")])
def test_soft_requires_validated_bounded_temperature(config_study, temperature):
    manifest, runtime = config_study
    calibration = None if temperature is None else {"temperature": temperature}
    with pytest.raises(ValueError, match="calibration"):
        study.build_config(manifest, CASES[-1], runtime, calibration)


@pytest.mark.parametrize("temperature", [1.0, 16.0])
def test_soft_temperature_closed_bounds(config_study, temperature):
    manifest, runtime = config_study
    assert study.build_config(manifest, CASES[-1], runtime, {"temperature": temperature}).temperature == temperature


def test_hard_ignores_soft_calibration_temperature(config_study):
    manifest, runtime = config_study
    assert study.build_config(manifest, CASES[0], runtime).temperature == 1.0


@pytest.mark.parametrize("kind", ["baseline_changed", "unknown_baseline_field", "old_lam05_cov1", "nonlegacy_solver"])
def test_build_config_rejects_wrong_saved_reference(config_study, kind):
    manifest, runtime = config_study
    if kind == "baseline_changed":
        manifest["baseline_config"]["coverage_floor"] = 0.2
    elif kind == "unknown_baseline_field":
        manifest["baseline_config"]["not_a_production_field"] = True
    else:
        meta_path = Path(manifest["reference_run"]) / "meta.json"
        meta = oldrunner._read_json(meta_path)
        if kind == "old_lam05_cov1":
            for mapping in (manifest["baseline_config"], meta["run_params"]):
                mapping.update(lambda_reg=0.5, coverage_damping_power=1.0)
        else:
            meta["run_params"]["travel_times"] += "_v2"
        write_json(meta_path, meta)
    with pytest.raises(ValueError, match="Baseline|reference|legacy"):
        study.build_config(manifest, CASES[1], runtime, {"temperature": 4.25})


def test_schema2_exactly_seven_cases_and_reference(config_study):
    manifest, _ = config_study
    study.validate_manifest(manifest)
    assert [(case["candidate_mode"], case["n_candidates"]) for case in manifest["cases"]] == CASE_IDENTITIES
    assert [case["id"] for case in manifest["cases"] if case["is_reference"]] == ["hard-k5"]


@pytest.mark.parametrize("kind", ["schema1", "wrong_dataset", "resources", "relative_root", "missing_case",
                                  "soft_k1", "duplicate_id", "duplicate_run", "no_reference",
                                  "wrong_reference", "soft_top1", "hard_topk", "soft_fixed_t", "hard_t2",
                                  "different_lambda", "different_coverage"])
def test_manifest_rejects_changed_contract(config_study, kind):
    m = copy.deepcopy(config_study[0])
    if kind == "schema1":
        m["schema_version"] = 1
    elif kind == "wrong_dataset":
        m["experiment_id"] = oldrunner.EXPERIMENT_ID
    elif kind == "resources":
        m["resources"]["forward_workers"] = 31
    elif kind == "relative_root":
        m["runtime_root"] = "relative"
    elif kind == "missing_case":
        m["cases"].pop()
    elif kind == "soft_k1":
        m["cases"][-1]["n_candidates"] = 1
    elif kind in ("duplicate_id", "duplicate_run"):
        key = "id" if kind == "duplicate_id" else "run_name"
        m["cases"][-1][key] = m["cases"][0][key]
    elif kind == "no_reference":
        m["cases"][1]["is_reference"] = False
    elif kind == "wrong_reference":
        m["cases"][1]["is_reference"] = False
        m["cases"][0]["is_reference"] = True
    else:
        index, key, value = {
            "soft_top1": (-1, "weights_top_n", 1), "hard_topk": (1, "weights_top_n", 5),
            "soft_fixed_t": (-1, "temperature", 4.25), "hard_t2": (0, "temperature", 2.0),
            "different_lambda": (0, "lambda_reg", 0.5), "different_coverage": (0, "coverage_damping_power", 1.0),
        }[kind]
        m["cases"][index][key] = value
    with pytest.raises(ValueError):
        study.validate_manifest(m)


def test_all_seven_cases_real_prepare_logger_em_and_read_only_complete_reuse(saved_study):
    s = saved_study
    immutable_before = (snapshot(Path(s.manifest["experiments_root"])), snapshot(Path(s.manifest["reference_run"])))
    run_child(s, [case["id"] for case in s.manifest["cases"]])
    for case in s.manifest["cases"]:
        result = read_result(s, case["id"])
        run = Path(result["run_dir"])
        meta = oldrunner._read_json(run / "meta.json")
        params = meta["run_params"]
        assert result["status"] == "complete" and result["attempts"] == []
        assert result["effective_config"] == params["effective_config"]
        assert params["travel_times"] == oldrunner.SAVED_TRAVEL_TIMES
        assert params["lambda_reg"] == 1.0 and params["coverage_damping_power"] == 2.0
        assert params["initial_model_source"]["used_as_checkpoint"] is False
        assert meta["station_locs"][0][2] == 0.0 and meta["station_locs"][-1][2] == 839.0
        origin = result["provenance"]
        assert origin == params["provenance"]
        assert origin["manifest_sha256"] == hashlib.sha256(s.path.read_bytes()).hexdigest()
        assert origin["output_sha256"] == s.ledger["output_sha256"]
        assert origin["calibration_sha256"] == s.ledger["calibration_sha256"]
        assert result["final_rms_m_s"] == oldrunner._quality(run, 12)
        assert len(list(run.glob("iter_*/complete.json"))) == 12
        np.testing.assert_array_equal(np.load(run / "final_model.npy"), np.load(run / "initial_model.npy"))
        json.dumps(result, allow_nan=False)
    assert immutable_before == (snapshot(Path(s.manifest["experiments_root"])), snapshot(Path(s.manifest["reference_run"])))
    assert not list(s.state.rglob("*.tmp"))


@pytest.mark.parametrize("kind", ["input", "runtime", "reference"])
def test_verify_hashes_real_files_without_loading_production(saved_study, monkeypatch, kind):
    s = saved_study
    study.verify_hashes(s.manifest)
    if kind == "input":
        file = Path(s.manifest["experiments_root"]) / "input" / study.EXPERIMENT_ID / "generation.json"
        with file.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif kind == "runtime":
        s.manifest["code_sha256"]["projects/EMTomo/eikonal.py"] = "0" * 64
    else:
        file = Path(s.manifest["reference_run"]) / "meta.json"
        with file.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: pytest.fail("runtime imported before hash validation"))
    with pytest.raises(ValueError, match="SHA256"):
        study.verify_hashes(s.manifest)


@pytest.mark.parametrize("kind", ["input", "runtime", "reference"])
def test_early_hash_failure_must_persist_failed_result_like_oldrunner(saved_study, monkeypatch, kind):
    s = saved_study
    mapping, name = {"input": ("input_sha256", "model.npz"),
                     "runtime": ("code_sha256", "projects/EMTomo/eikonal.py"),
                     "reference": ("reference_sha256", "initial_model.npy")}[kind]
    s.manifest[mapping][name] = "0" * 64
    write_json(s.path, s.manifest)
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: pytest.fail("production imported before hash check"))
    with pytest.raises(ValueError, match="SHA256"):
        study.run_case(s.path, "hard-k5")
    path = s.state / "cases" / "hard-k5" / "result.json"
    assert path.is_file(), "early hash failure escaped result.json publication"
    result = oldrunner._read_json(path)
    assert result["status"] == "failed" and result["run_dir"] is None
    assert "SHA256" in result["error"]


@pytest.mark.parametrize("mapping", ["input_sha256", "code_sha256", "reference_sha256"])
def test_verify_hashes_must_not_allow_empty_required_hash_mapping(saved_study, mapping):
    manifest = copy.deepcopy(saved_study.manifest)
    manifest[mapping] = {}
    with pytest.raises(ValueError):
        study.verify_hashes(manifest)


def test_observed_ledger_reuse_is_read_only(saved_study):
    s = saved_study
    before = snapshot(s.state)
    assert study.verify_observations(s.state, s.manifest) == (s.ledger, s.calibration)
    assert before == snapshot(s.state)


@pytest.mark.parametrize("kind", ["arrivals", "metadata", "calibration", "incomplete", "missing_calibration_hash"])
def test_observed_ledger_corruption_blocks_production_and_persists_failure(saved_study, monkeypatch, kind):
    s = saved_study
    if kind in ("arrivals", "metadata", "calibration"):
        file = {"arrivals": s.output / "arrivals.csv", "metadata": s.output / "metadata.json",
                "calibration": s.state / "calibration.json"}[kind]
        with file.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        ledger = copy.deepcopy(s.ledger)
        if kind == "incomplete":
            ledger["status"] = "running"
        else:
            del ledger["calibration_sha256"]
        write_json(s.state / "forward_complete.json", ledger)
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: pytest.fail("corrupt observations reached production"))
    with pytest.raises((ValueError, KeyError)):
        study.run_case(s.path, "hard-k5")
    result = read_result(s)
    assert result["status"] == "failed" and result["run_dir"] is None
    assert not Path(s.manifest["runs_dir"]).exists()


@pytest.mark.parametrize("missing", ["arrivals.csv", "metadata.json", "both"])
def test_observed_ledger_requires_hashes_for_both_native_output_files(saved_study, missing):
    s = saved_study
    ledger = copy.deepcopy(s.ledger)
    if missing == "both":
        ledger["output_sha256"] = {}
    else:
        del ledger["output_sha256"][missing]
    write_json(s.state / "forward_complete.json", ledger)
    with pytest.raises(ValueError):
        study.verify_observations(s.state, s.manifest)


@pytest.mark.parametrize("tamper", ["marker", "final", "quality", "effective_config", "run_params", "provenance",
                                    "initial", "missing_diagnostic", "result_identity", "result_provenance"])
def test_complete_artifact_corruption_is_rejected_without_numerical_retry(completed_study, tamper, monkeypatch):
    s = completed_study
    previous = read_result(s)
    run = Path(previous["run_dir"])
    result_path = s.state / "cases" / "hard-k5" / "result.json"
    if tamper == "marker":
        (run / "iter_11" / "complete.json").unlink()
    elif tamper == "final":
        np.save(run / "final_model.npy", np.full((20, 7, 7), np.nan))
    elif tamper == "quality":
        (run / "quality.jsonl").write_text('{"iter":0,"rms_m_s":1,"avg_abs_pct_dev":1}\n', encoding="utf-8")
    elif tamper == "initial":
        np.save(run / "initial_model.npy", np.full((20, 7, 7), 5000.0))
    elif tamper == "missing_diagnostic":
        (run / "iter_11" / "coverage_confidence.npy").unlink()
    elif tamper in ("result_identity", "result_provenance"):
        result = copy.deepcopy(previous)
        if tamper == "result_identity":
            result["case_id"] = "soft-k15"
        else:
            result["provenance"]["calibration_sha256"] = "0" * 64
        write_json(result_path, result)
    else:
        meta_path = run / "meta.json"
        meta = oldrunner._read_json(meta_path)
        if tamper == "effective_config":
            meta["run_params"]["effective_config"]["n_workers"] = 1
        elif tamper == "run_params":
            meta["run_params"]["temperature"] = 0.25
        else:
            meta["run_params"]["provenance"]["output_sha256"]["arrivals.csv"] = "0" * 64
        write_json(meta_path, meta)
    before = snapshot(run)
    case = s.manifest["cases"][1]
    config = study.build_config(s.manifest, case, SimpleNamespace(InversionConfig=InversionConfig), s.calibration)
    with pytest.raises((ValueError, OSError)):
        oldrunner.check_completed_result(oldrunner._read_json(result_path), s.manifest, case, config,
                                        previous["provenance"], np)
    # Avoid importing production into the pytest parent: a complete result must
    # be verified using only config + np, never prepare/logger/run_em.
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: SimpleNamespace(InversionConfig=InversionConfig, np=np))
    with pytest.raises((ValueError, OSError)):
        study.run_case(s.path, "hard-k5")
    rejected = read_result(s)
    assert rejected["status"] == "failed" and rejected["resume_rejected"] is True
    assert rejected["run_dir"] == previous["run_dir"]
    with pytest.raises(ValueError, match="Previously rejected"):
        study.run_case(s.path, "hard-k5")
    assert before == snapshot(run)


def test_rejected_complete_result_retains_original_attempt_history(completed_study, monkeypatch):
    s = completed_study
    previous = read_result(s)
    (Path(previous["run_dir"]) / "iter_11" / "complete.json").unlink()
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: SimpleNamespace(InversionConfig=InversionConfig, np=np))
    with pytest.raises(OSError):
        study.run_case(s.path, "hard-k5")
    rejected = read_result(s)
    assert rejected["attempts"] and rejected["attempts"][-1]["status"] == "complete", (
        "rejecting a completed result discarded its persisted history")
    assert rejected["attempts"][-1]["run_dir"] == previous["run_dir"]


def test_complete_result_effective_config_must_match_persisted_run(completed_study, monkeypatch):
    s = completed_study
    result = read_result(s)
    result["effective_config"]["temperature"] = 9.0
    write_json(s.state / "cases" / "hard-k5" / "result.json", result)
    monkeypatch.setattr(study.shared, "load_runtime", lambda _: SimpleNamespace(InversionConfig=InversionConfig, np=np))
    with pytest.raises(ValueError):
        study.run_case(s.path, "hard-k5")


def test_failed_em_restores_method_and_retry_retains_history(saved_study):
    s = saved_study
    code = CHILD_SETUP + r'''
def fail_step(*args, **kwargs):
    raise RuntimeError('fixture numerical failure')
runtime.em.make_tomography_step = fail_step
try:
    study.run_case(manifest_path, sys.argv[3])
except RuntimeError as error:
    assert str(error) == 'fixture numerical failure'
else:
    raise AssertionError('numerical failure swallowed')
assert runtime.em.METHOD_DESCRIPTION is old_description
'''
    run_child(s, ["hard-k5"], code)
    failed = read_result(s)
    assert failed["status"] == "failed" and "fixture numerical failure" in failed["error"]
    run_child(s, ["hard-k5"])
    retried = read_result(s)
    assert retried["status"] == "complete" and retried["run_dir"] != failed["run_dir"]
    assert retried["attempts"][-1]["run_dir"] == failed["run_dir"]
    assert Path(failed["run_dir"]).is_dir()


@pytest.mark.parametrize("kind", ["input", "output", "calibration"])
def test_end_of_run_rechecks_input_observations_and_calibration(saved_study, kind):
    code = CHILD_SETUP + r'''
real_step = step
def tampering_step(*args, **kwargs):
    values = real_step(*args, **kwargs)
    if kwargs['iteration'] == 11:
        kind = sys.argv[4]
        if kind == 'input':
            file = Path(manifest['experiments_root']) / 'input' / manifest['experiment_id'] / 'generation.json'
        elif kind == 'output':
            file = Path(manifest['experiments_root']) / 'output' / manifest['experiment_id'] / 'arrivals.csv'
        else:
            file = manifest_path.parent / 'calibration.json'
        with file.open('a') as stream:
            stream.write('\n')
    return values
runtime.em.make_tomography_step = tampering_step
try:
    study.run_case(manifest_path, sys.argv[3])
except ValueError as error:
    assert 'changed' in str(error) or 'SHA256' in str(error)
else:
    raise AssertionError('changed inputs or observations published as complete')
assert len(calls) == 12 and runtime.em.METHOD_DESCRIPTION is old_description
'''
    run_child(saved_study, ["hard-k5", kind], code)
    assert read_result(saved_study)["status"] == "failed"


@pytest.mark.parametrize("kind", ["output", "calibration"])
def test_atomic_ledger_republication_must_not_publish_stale_provenance(saved_study, kind):
    code = CHILD_SETUP + r'''
real_step = step
def republish_step(*args, **kwargs):
    values = real_step(*args, **kwargs)
    if kwargs['iteration'] == 11:
        state = manifest_path.parent
        ledger = shared._read_json(state / 'forward_complete.json')
        if sys.argv[4] == 'output':
            file = Path(manifest['experiments_root']) / 'output' / manifest['experiment_id'] / 'arrivals.csv'
            with file.open('a') as stream:
                stream.write('\n')
            ledger['output_sha256']['arrivals.csv'] = shared.sha256(file)
        else:
            file = state / 'calibration.json'
            calibration = shared._read_json(file)
            calibration['temperature'] = 5.0
            shared.atomic_json(file, calibration)
            ledger['calibration_sha256'] = shared.sha256(file)
        shared.atomic_json(state / 'forward_complete.json', ledger)
    return values
runtime.em.make_tomography_step = republish_step
rejected = False
try:
    study.run_case(manifest_path, sys.argv[3])
except ValueError:
    rejected = True
assert len(calls) == 12 and runtime.em.METHOD_DESCRIPTION is old_description
print('LEDGER_REPUBLICATION_REJECTED=' + json.dumps(rejected))
'''
    result = run_child(saved_study, ["soft-k15", kind], code)
    assert "LEDGER_REPUBLICATION_REJECTED=true" in result.stdout, (
        "atomic ledger changed during the case but stale observation/calibration provenance was published")
    assert read_result(saved_study, "soft-k15")["status"] == "failed"


def test_optional_noise_is_resolved_by_real_production_metadata(saved_study):
    s = saved_study
    for key in ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s"):
        s.manifest["baseline_config"][key] = None
    save_reference(s.manifest)
    write_json(s.path, s.manifest)
    run_child(s, ["soft-k15"])
    result = read_result(s, "soft-k15")
    params = oldrunner._read_json(Path(result["run_dir"]) / "meta.json")["run_params"]
    assert params["effective_config"]["weight_noise_relative_sigma"] is None
    assert params["weight_noise_relative_sigma"] == 0.01
    assert params["weight_noise_absolute_sigma_s"] == 0.01


def test_cached_checkout_production_is_refused_by_shared_runtime(saved_study):
    code = f'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_case as study
sys.path.insert(0, {str(ROOT / "projects/EMTomo")!r})
import config
manifest = json.loads(Path(sys.argv[2]).read_text())
try:
    study.shared.load_runtime(manifest)
except ValueError as error:
    assert 'outside runtime snapshot' in str(error)
else:
    raise AssertionError('working-checkout production was accepted')
'''
    run_child(saved_study, code=code)


def test_latest_saved_lam1_cov2_legacy_reference_read_only(tmp_path):
    references = []
    for path in (ROOT / "projects/EMTomo/runs").glob("*/meta.json"):
        try:
            params = oldrunner._read_json(path)["run_params"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        required = {"cell_size": 1000.0, "subdivision": 8, "n_cycles": 12,
                    "lambda_reg": 1.0, "coverage_damping_power": 2.0, "candidate_mode": "hard",
                    "n_candidates": 5, "weights_top_n": 1, "travel_times": oldrunner.SAVED_TRAVEL_TIMES}
        if all(params.get(key) == value for key, value in required.items()):
            references.append(path.parent)
    if not references:
        pytest.skip("Latest saved lam1/cov2 legacy reference is not installed")
    reference = max(references, key=lambda path: path.name.rsplit("_", 2)[-2:])
    meta_path = reference / "meta.json"
    initial_path = reference / "initial_model.npy"
    before = [(file.stat().st_mtime_ns, file.read_bytes()) for file in (meta_path, initial_path)]
    manifest = manifest_for(tmp_path)
    manifest["reference_run"] = str(reference.resolve())
    allowed = set(asdict(InversionConfig()))
    manifest["baseline_config"] = {key: value for key, value in oldrunner._read_json(meta_path)["run_params"].items()
                                    if key in allowed}
    for case in manifest["cases"]:
        config = study.build_config(manifest, case, SimpleNamespace(InversionConfig=InversionConfig), {"temperature": 4.25})
        expected = oldrunner._json(asdict(config))
        assert all(expected[key] == value for key, value in manifest["baseline_config"].items()
                   if key not in CASE_RUNTIME_FIELDS)
    # Initializer only: no observations, JIT, forward or inversion are run.
    code = f'''
import sys
from pathlib import Path
sys.path[:0] = [{str(ROOT / "projects/EMTomo")!r}, {str(ROOT)!r}]
import numpy as np
from config import InversionConfig
from experiment_runner import initial_velocity
from projects.EMTomo.studies.geysers2011.run_case import verify_initial
config = InversionConfig(**{manifest["baseline_config"]!r})
verify_initial(initial_velocity(config, (20, 7, 7)), Path({str(initial_path)!r}), np)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == [(file.stat().st_mtime_ns, file.read_bytes()) for file in (meta_path, initial_path)]


def test_direct_cli_help_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(study.__file__)), "--help"], cwd=tmp_path,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout and "--case-id" in result.stdout
