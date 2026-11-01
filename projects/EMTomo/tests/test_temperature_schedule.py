"""Annealing configuration, CLI and publication checks without real FMM/EM steps."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from config import InversionConfig
from instruments.likelihood import PickNoise
from main import CONFIG, _parser
from tomography import tomography_em as em
from tomography.tomography_logging import TomographyLogger
from velocity_model import VelocityModel


SCHEDULE = (4.,) * 4 + (2.,) * 4 + (1.,) * 4


def _parse_config(*options):
    args = vars(_parser().parse_args(["test_experiment", *options]))
    for name in ("experiment_id", "experiments_root", "validate_only"):
        args.pop(name)
    return replace(CONFIG, **{
        name: tuple(value) if isinstance(value, list) else value
        for name, value in args.items()
    })


@pytest.mark.parametrize("schedule", [(), (4.,), (4., 2.), (4., 2., 1., 1.)])
def test_schedule_requires_exactly_n_cycles_values(schedule):
    with pytest.raises(ValueError, match="temperature_schedule.*n_cycles"):
        InversionConfig(n_cycles=3, temperature_schedule=schedule)


@pytest.mark.parametrize("value", [
    0., -1., np.nan, np.inf, -np.inf, True, False, np.bool_(True), np.bool_(False),
    "2", None, complex(2., 0.),
])
def test_schedule_rejects_nonfinite_nonpositive_boolean_or_nonnumeric_values(value):
    with pytest.raises(ValueError, match="temperature_schedule.*finite and positive"):
        InversionConfig(n_cycles=3, temperature_schedule=(4., value, 1.))


@pytest.mark.parametrize("value", [1, 0.5, np.int64(2), np.float32(4.)])
def test_schedule_accepts_finite_positive_numbers(value):
    config = InversionConfig(n_cycles=1, temperature_schedule=(value,))
    assert config.temperature_schedule == (value,)
    assert config.temperature == 1.


def test_schedule_defaults_to_none():
    assert InversionConfig().temperature_schedule is None
    assert _parse_config() == CONFIG
    assert asdict(CONFIG)["temperature_schedule"] is None


def test_cli_schedule_roundtrip():
    config = _parse_config(
        "--cycles", "12", "--temperature", "3.5", "--n-candidates", "7",
        "--temperature-schedule", *map(str, SCHEDULE),
    )
    assert config.temperature_schedule == SCHEDULE
    assert config.temperature == 3.5
    assert config.n_candidates == 7
    metadata = json.loads(json.dumps(asdict(config)))
    assert metadata["temperature_schedule"] == list(SCHEDULE)
    assert InversionConfig(**{
        **metadata, "temperature_schedule": tuple(metadata["temperature_schedule"]),
    }) == config


@pytest.mark.parametrize("options", [[], ["not-a-float"]])
def test_cli_schedule_requires_float_values(options):
    with pytest.raises(SystemExit) as error:
        _parse_config("--temperature-schedule", *options)
    assert error.value.code == 2


@pytest.mark.parametrize("values", [("4", "2"), ("4", "nan", "1"), ("4", "0", "1")])
def test_cli_schedule_is_validated(values):
    with pytest.raises(ValueError, match="temperature_schedule"):
        _parse_config("--cycles", "3", "--temperature-schedule", *values)


@pytest.fixture
def mocked_problem(monkeypatch):
    model = VelocityModel(np.full((1, 1, 1), 5000.), 1000.)
    arrivals = np.array([[0., 1.]])
    stations = np.array([[0., 0., 0.], [1000., 0., 0.]])
    noise = PickNoise(.01, .05, .1)
    calls = []

    def step(current_model, current_arrivals, current_stations, current_noise, config,
             *, iteration, logger):
        assert current_arrivals is arrivals
        assert current_stations is stations
        assert current_noise is noise
        np.testing.assert_array_equal(current_model.velocity, model.velocity)
        calls.append((iteration, config))
        return np.zeros(model.shape), np.ones(model.shape), np.ones(model.shape)

    def unexpected(*args, **kwargs):
        pytest.fail("Mocked run must not execute FMM or EM kernels")

    monkeypatch.setattr(em, "make_tomography_step", step)
    monkeypatch.setattr(em, "station_travel_time_fields", unexpected)
    monkeypatch.setattr(em, "run_events", unexpected)
    return model, arrivals, stations, noise, calls


@pytest.mark.parametrize("schedule", [SCHEDULE, None])
@pytest.mark.parametrize("save_runs", [False, True])
@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_run_em_uses_only_scheduled_temperature_and_publishes_it(
    mocked_problem, tmp_path, monkeypatch, schedule, save_runs, mode,
):
    model, arrivals, stations, noise, calls = mocked_problem
    original_arrivals, original_stations = arrivals.copy(), stations.copy()
    config = InversionConfig(
        n_cycles=12, temperature=3.5, temperature_schedule=schedule,
        n_candidates=7, weights_top_n=3, candidate_mode=mode,
        save_runs=save_runs, runs_dir=str(tmp_path),
    )
    original_config = asdict(config)
    publications = []
    save_json = TomographyLogger._save_json

    def record_json(path, values):
        if path.name == "complete.json":
            temperature_path = path.with_name("temperature.json")
            assert temperature_path.exists() == (schedule is not None)
            if schedule is not None:
                assert json.loads(temperature_path.read_text()) == {
                    "iter": values["iter"], "temperature": schedule[values["iter"]],
                }
            publications.append(values["iter"])
        save_json(path, values)

    monkeypatch.setattr(TomographyLogger, "_save_json", staticmethod(record_json))
    logger = em.run_em(config, model, arrivals, stations, noise)
    assert [iteration for iteration, _ in calls] == list(range(12))
    assert [step.temperature for _, step in calls] == (
        list(schedule) if schedule is not None else [3.5] * 12
    )
    for _, step in calls:
        assert asdict(step) == {**original_config, "temperature": step.temperature}
        assert (step is config) == (schedule is None)
    assert asdict(config) == original_config
    np.testing.assert_array_equal(arrivals, original_arrivals)
    np.testing.assert_array_equal(stations, original_stations)
    if save_runs:
        assert publications == list(range(12))
        params = json.loads((logger.run_dir / "meta.json").read_text())["run_params"]
        assert params["temperature_schedule"] == (list(schedule) if schedule else None)
        assert params["temperature"] == 3.5
        assert params["n_candidates"] == 7
        assert (logger.run_dir / "final_model.npy").exists()
    else:
        assert logger is None
        assert publications == []
        assert list(tmp_path.iterdir()) == []


def test_temperature_write_failure_does_not_publish(mocked_problem, tmp_path, monkeypatch):
    model, arrivals, stations, noise, _ = mocked_problem
    logger = TomographyLogger(base_dir=tmp_path)
    save_json = logger._save_json

    def fail_temperature(path, values):
        if path.name == "temperature.json":
            raise OSError("temperature write failed")
        save_json(path, values)

    monkeypatch.setattr(logger, "_save_json", fail_temperature)
    with pytest.raises(OSError, match="temperature write failed"):
        em.run_em(
            InversionConfig(n_cycles=1, temperature_schedule=(4.,)),
            model, arrivals, stations, noise, logger=logger,
        )
    assert not (logger.run_dir / "iter_0/complete.json").exists()
