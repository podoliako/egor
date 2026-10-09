"""Fast experiment I/O tests; numerical solver behavior is tested separately."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from projects.forward_modeling import experiments as io
from projects.forward_modeling import __main__ as cli
from projects.forward_modeling.model import Arrival, ForwardConfig, PointSet, VelocityGrid


@pytest.fixture
def inputs():
    return (
        VelocityGrid(velocity=np.full((3, 3, 3), 2000.0), cell_size_m=10.0,
                     origin_m=(100.0, -20.0, 30.0)),
        PointSet(ids=("001", "station,quoted"),
                 coordinates_m=np.array([[100., -20., 30.], [110., -10., 40.]])),
        PointSet(ids=(7, "0008"),
                 coordinates_m=np.array([[105., -15., 35.], [115., -5., 45.]])),
    )


@pytest.fixture
def solver_mock(monkeypatch):
    compute = Mock(return_value=[Arrival(station_id="001", event_id="7", arrival_time_s=0.),
                                 Arrival(station_id="station,quoted", event_id="0008",
                                         arrival_time_s=0.125)])
    convergence = Mock(return_value={"coarse_refinement": 2, "fine_refinement": 4,
                                    "max_abs_difference_s": 0.001,
                                    "rms_difference_s": 0.0005,
                                    "max_absolute_time_difference_s": 0.002,
                                    "rms_absolute_time_difference_s": 0.001})
    monkeypatch.setattr(io, "version", Mock(return_value="0.4.1"))
    monkeypatch.setattr(io.solver, "compute_arrivals", compute)
    monkeypatch.setattr(io.solver, "check_convergence", convergence)
    return compute, convergence


def test_noise_metadata_and_clean_convergence(tmp_path, inputs, solver_mock):
    from projects.forward_modeling import NoiseConfig

    io.save_inputs(tmp_path, "noisy", *inputs)
    noise = NoiseConfig(0.01, 0.05, 123)
    destination = io.run_experiment(tmp_path, "noisy", noise=noise, check_accuracy=True)
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["noise"]["enabled"] is True
    assert metadata["noise"]["config"] == asdict(noise)
    assert metadata["noise"]["sigma_reference"] == "noiseless_absolute_travel_time_s"
    assert metadata["noise"]["convergence"] == "noiseless"
    compute, convergence = solver_mock
    assert compute.call_args.kwargs["noise"] == noise
    assert "noise" not in convergence.call_args.kwargs


@pytest.mark.parametrize("extra,expected", [
    (["--noise"], {"relative_sigma": 0.01, "absolute_sigma_s": 0.05, "seed": 42}),
    (["--noise", "--noise-relative-sigma", "0", "--noise-absolute-sigma-s", "0.1", "--noise-seed", "7"],
     {"relative_sigma": 0., "absolute_sigma_s": 0.1, "seed": 7}),
])
def test_cli_noise(monkeypatch, extra, expected):
    run = Mock(return_value=Path("output/noisy"))
    monkeypatch.setattr(cli, "run_experiment", run)
    assert cli.main(["noisy", *extra]) == 0
    assert asdict(run.call_args.kwargs["noise"]) == expected


@pytest.mark.parametrize("extra", [
    ["--noise-relative-sigma", "0.01"],
    ["--noise", "--noise-relative-sigma", "nan"],
    ["--noise", "--noise-absolute-sigma-s", "-0.1"],
    ["--noise", "--noise-seed", "-1"],
])
def test_cli_rejects_invalid_noise(monkeypatch, extra):
    run = Mock()
    monkeypatch.setattr(cli, "run_experiment", run)
    with pytest.raises(SystemExit) as error:
        cli.main(["noisy", *extra])
    assert error.value.code == 1
    run.assert_not_called()


def test_inputs_roundtrip(tmp_path, inputs):
    directory = io.save_inputs(tmp_path, "case-1", *inputs)
    assert directory == tmp_path / "input" / "case-1"
    assert not (tmp_path / "output").exists()
    assert {p.name for p in directory.iterdir()} == {"model.npz", "stations.csv", "events.csv"}
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"velocity", "cell_size_m", "origin_m"}
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    model, stations, events = io.load_inputs(tmp_path, "case-1")
    np.testing.assert_array_equal(model.velocity, inputs[0].velocity)
    np.testing.assert_array_equal(model.origin_m, inputs[0].origin_m)
    assert model.cell_size_m == 10.
    assert stations.ids == ("001", "station,quoted")
    assert events.ids == ("7", "0008")
    np.testing.assert_array_equal(stations.coordinates_m, inputs[1].coordinates_m)
    np.testing.assert_array_equal(events.coordinates_m, inputs[2].coordinates_m)
    for name, identifier in (("stations.csv", "station_id"), ("events.csv", "event_id")):
        with (directory / name).open(newline="") as stream:
            assert next(csv.reader(stream)) == [identifier, "x_m", "y_m", "z_m"]


@pytest.mark.parametrize("check_accuracy", [False, True])
def test_run_roundtrip_metadata(tmp_path, inputs, solver_mock, check_accuracy):
    source = io.save_inputs(tmp_path, "case", *inputs)
    config = ForwardConfig(refinement=2, source_radius_cells=3)
    destination = io.run_experiment(tmp_path, "case", config=config,
                                    check_accuracy=check_accuracy)
    assert destination == tmp_path / "output" / "case"
    assert {p.name for p in destination.iterdir()} == {"arrivals.csv", "metadata.json"}
    with (destination / "arrivals.csv").open(newline="") as stream:
        assert next(csv.reader(stream)) == ["station_id", "event_id", "arrival_time_s"]
    arrivals = io.load_arrivals(tmp_path, "case")
    assert [(a.station_id, a.event_id, a.arrival_time_s) for a in arrivals] == [
        ("001", "7", 0.), ("station,quoted", "0008", 0.125)]
    metadata = json.loads((destination / "metadata.json").read_text())
    assert "max_difference_s" not in metadata
    assert metadata["config"] == asdict(config)
    assert metadata["config"] == {"refinement": 2, "source_radius_cells": 3}
    assert metadata["solver_name"] == "pykonal.EikonalSolver"
    assert metadata["pykonal_version"] == "0.4.1"
    io.version.assert_called_once_with("pykonal")
    assert metadata["solver_version"] == io.solver.__version__
    assert metadata["source_seed"] == {
        "method": "local_off_grid_straight_ray_slowness_integral",
        "source_radius_cells": 3,
        "radius_cell_size_m": 5.0,
        "approximation": "local_straight_ray_upper_bounds_not_bent_rays",
    }
    assert metadata["receiver_interpolation"] == {
        "method": "factored_trilinear",
        "factor": "T/d",
        "distance": "euclidean_distance_to_exact_source",
        "zero_distance_factor": "source_voxel_slowness",
    }
    assert metadata["velocity_representation"] == {
        "input": "cell_centered_piecewise_constant_voxels",
        "origin": "lower_domain_corner",
        "numerical_grid": "nodes_with_adjacent_slowness_averaging_at_interfaces",
    }
    assert metadata["time_reference"] == "earliest_station_arrival_per_event"
    assert metadata["units"]["arrival_time"] == "s"
    assert metadata["elapsed_seconds"] >= 0
    assert metadata["input_sha256"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    assert "coordinates_m" not in json.dumps(metadata)
    compute, convergence = solver_mock
    compute.assert_called_once()
    assert set(compute.call_args.kwargs) == {"model", "stations", "events", "config", "workers"}
    assert compute.call_args.kwargs["config"] is config
    np.testing.assert_array_equal(compute.call_args.kwargs["model"].origin_m, inputs[0].origin_m)
    if check_accuracy:
        convergence.assert_called_once_with(**compute.call_args.kwargs)
        assert metadata["convergence"] == convergence.return_value
    else:
        convergence.assert_not_called()
        assert "convergence" not in metadata


def test_no_overwrite(tmp_path, inputs, solver_mock):
    source = io.save_inputs(tmp_path, "case", *inputs)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        io.save_inputs(tmp_path, "case", *inputs)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    destination = io.run_experiment(tmp_path, "case")
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        io.run_experiment(tmp_path, "case")
    solver_mock[0].assert_called_once()
    assert before == {p.name: p.read_bytes() for p in destination.iterdir()}


@pytest.mark.parametrize("identifier", ["", ".", "..", "../escape", "a/../../b", "/absolute",
                                        "a/b", "a\\b", "a\x00b", "a\nb", 123])
def test_unsafe_ids(tmp_path, inputs, identifier):
    for operation, args in ((io.save_inputs, inputs), (io.load_inputs, ()),
                            (io.run_experiment, ()), (io.load_arrivals, ())):
        with pytest.raises(ValueError, match="experiment_id"):
            operation(tmp_path, identifier, *args)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ["compute", "convergence", "metadata"])
def test_failed_run_not_published(tmp_path, inputs, solver_mock, monkeypatch, stage):
    io.save_inputs(tmp_path, "case", *inputs)
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / ".case.tmp-someone-else"
    unrelated.mkdir()
    sentinel = unrelated / "keep"
    sentinel.write_text("untouched")
    def fail(*args, **kwargs):
        assert not (output / "case").exists()
        raise RuntimeError("interrupted")
    if stage == "compute":
        solver_mock[0].side_effect = fail
    elif stage == "convergence":
        solver_mock[1].side_effect = fail
    else:
        monkeypatch.setattr(io.json, "dump", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        io.run_experiment(tmp_path, "case", check_accuracy=True)
    assert set(output.iterdir()) == {unrelated}
    assert sentinel.read_text() == "untouched"


def test_failed_input_write_not_published(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(io, "_write_points", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        io.save_inputs(tmp_path, "case", *inputs)
    assert list((tmp_path / "input").iterdir()) == []


def test_existing_lock_not_removed(tmp_path, inputs, solver_mock):
    io.save_inputs(tmp_path, "case", *inputs)
    output = tmp_path / "output"
    output.mkdir()
    lock = output / ".case.lock"
    lock.write_text("another writer")
    with pytest.raises(FileExistsError):
        io.run_experiment(tmp_path, "case")
    assert lock.read_text() == "another writer"
    solver_mock[0].assert_not_called()


@pytest.mark.parametrize("time", ["nan", "inf", "-inf", "-0.1", "bad", ""])
def test_invalid_arrival_time(tmp_path, time):
    directory = tmp_path / "output" / "case"
    directory.mkdir(parents=True)
    (directory / "arrivals.csv").write_text(
        f"station_id,event_id,arrival_time_s\n001,007,{time}\n")
    with pytest.raises(ValueError):
        io.load_arrivals(tmp_path, "case")


@pytest.mark.parametrize("body", [
    "station_id,event_id,arrival_time_s\n001,007,1\n001,007,2\n",
    "station_id,event_id,arrival_time_s\n,007,1\n",
    "station_id,event_id,arrival_time_s\n001,007,1,extra\n",
    "event_id,station_id,arrival_time_s\n007,001,1\n",
])
def test_invalid_arrival_csv(tmp_path, body):
    directory = tmp_path / "output" / "case"
    directory.mkdir(parents=True)
    (directory / "arrivals.csv").write_text(body)
    with pytest.raises(ValueError):
        io.load_arrivals(tmp_path, "case")


@pytest.mark.parametrize("body", [
    "station_id,x_m,y_m,z_m\n001,100,-20,nan\n",
    "station_id,x_m,y_m,z_m\n001,100,-20,30\n001,110,-10,40\n",
    "station_id,x_m,y_m,z_m\n",
    "station_id,x_m,y_m,z_m\n001,100,-20\n",
])
def test_invalid_input_points(tmp_path, inputs, body):
    directory = io.save_inputs(tmp_path, "case", *inputs)
    (directory / "stations.csv").write_text(body)
    with pytest.raises(ValueError):
        io.load_inputs(tmp_path, "case")


def test_duplicate_solver_results_not_published(tmp_path, inputs, solver_mock):
    io.save_inputs(tmp_path, "case", *inputs)
    arrival = solver_mock[0].return_value[0]
    solver_mock[0].return_value = [arrival, arrival]
    with pytest.raises(ValueError, match="Duplicate arrival"):
        io.run_experiment(tmp_path, "case")
    assert list((tmp_path / "output").iterdir()) == []


def test_empty_existing_output_not_replaced(tmp_path, solver_mock):
    destination = tmp_path / "output" / "case"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        io.run_experiment(tmp_path, "case")
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    solver_mock[0].assert_not_called()


def test_missing_inputs(tmp_path, solver_mock):
    with pytest.raises(FileNotFoundError):
        io.run_experiment(tmp_path, "missing")
    assert not (tmp_path / "input").exists()
    assert list((tmp_path / "output").iterdir()) == []
    solver_mock[0].assert_not_called()


def test_pickle_rejected(tmp_path, inputs):
    directory = io.save_inputs(tmp_path, "case", *inputs)
    np.savez(directory / "model.npz", velocity=np.array([object()], dtype=object),
             cell_size_m=10., origin_m=np.array([100., -20., 30.]))
    with pytest.raises(ValueError):
        io.load_inputs(tmp_path, "case")


def test_cli_options(tmp_path, monkeypatch, capsys):
    run = Mock(return_value=tmp_path / "output" / "case")
    monkeypatch.setattr(cli, "run_experiment", run)
    assert cli.main(["case", "--root", str(tmp_path), "--refinement", "3",
                     "--source-radius-cells", "4",
                     "--check-convergence"]) == 0
    kwargs = run.call_args.kwargs
    assert kwargs["root"] == tmp_path
    assert kwargs["experiment_id"] == "case"
    assert kwargs["check_accuracy"] is True
    assert asdict(kwargs["config"]) == {
        "refinement": 3, "source_radius_cells": 4}
    assert str(run.return_value) in capsys.readouterr().out


def test_cli_default_root(monkeypatch):
    run = Mock(return_value=Path("output/case"))
    monkeypatch.setattr(cli, "run_experiment", run)
    cli.main(["case"])
    assert run.call_args.kwargs["root"] == Path(cli.__file__).parent / "experiments"
    assert asdict(run.call_args.kwargs["config"]) == asdict(ForwardConfig())
    assert run.call_args.kwargs["check_accuracy"] is False
    assert "max_difference_s" not in run.call_args.kwargs


@pytest.mark.parametrize("option", ["--refinement", "--source-radius-cells"])
@pytest.mark.parametrize("value,exit_code", [("0", 1), ("-1", 1), ("1.5", 2)])
def test_cli_invalid_config(monkeypatch, option, value, exit_code):
    run = Mock()
    monkeypatch.setattr(cli, "run_experiment", run)
    with pytest.raises(SystemExit) as error:
        cli.main(["case", option, value])
    assert error.value.code == exit_code
    run.assert_not_called()


@pytest.mark.parametrize("option", ["--tolerance", "--max-iterations"])
def test_cli_rejects_removed_options(monkeypatch, option):
    run = Mock()
    monkeypatch.setattr(cli, "run_experiment", run)
    with pytest.raises(SystemExit) as error:
        cli.main(["case", option, "1"])
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), -float("inf"), "bad"])
def test_invalid_threshold_before_staging(tmp_path, solver_mock, threshold):
    with pytest.raises(ValueError, match="max_difference_s"):
        io.run_experiment(tmp_path, "case", max_difference_s=threshold)
    assert list(tmp_path.iterdir()) == []
    solver_mock[0].assert_not_called()
    solver_mock[1].assert_not_called()


@pytest.mark.parametrize("threshold", [0.002, 0.003])
def test_threshold_enables_convergence(tmp_path, inputs, solver_mock, threshold):
    io.save_inputs(tmp_path, "case", *inputs)
    output = io.run_experiment(tmp_path, "case", max_difference_s=threshold)
    solver_mock[1].assert_called_once_with(**solver_mock[0].call_args.kwargs)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["max_difference_s"] == threshold
    assert metadata["convergence"] == solver_mock[1].return_value


@pytest.mark.parametrize("field", ["max_abs_difference_s", "max_absolute_time_difference_s"])
def test_threshold_failure_not_published(tmp_path, inputs, solver_mock, field):
    io.save_inputs(tmp_path, "case", *inputs)
    solver_mock[1].return_value[field] = 0.004
    with pytest.raises(RuntimeError, match="exceeds"):
        io.run_experiment(tmp_path, "case", max_difference_s=0.003)
    assert list((tmp_path / "output").iterdir()) == []
    # The failed attempt does not prevent a subsequent successful run.
    io.run_experiment(tmp_path, "case", max_difference_s=0.005)


@pytest.mark.parametrize("field", ["max_abs_difference_s", "max_absolute_time_difference_s"])
def test_nonfinite_convergence_not_published(tmp_path, inputs, solver_mock, field):
    io.save_inputs(tmp_path, "case", *inputs)
    solver_mock[1].return_value[field] = float("nan")
    with pytest.raises(RuntimeError, match="Invalid convergence"):
        io.run_experiment(tmp_path, "case", max_difference_s=0.003)
    assert list((tmp_path / "output").iterdir()) == []


@pytest.mark.parametrize("label,index", [("stations", 1), ("events", 2)])
def test_save_outside_points_rejected(tmp_path, inputs, label, index):
    invalid = list(inputs)
    invalid[index] = PointSet(ids=("outside",), coordinates_m=np.array([[99., -20., 30.]]))
    with pytest.raises(ValueError, match=label + " outside"):
        io.save_inputs(tmp_path, "case", *invalid)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("label,id_column", [("stations", "station_id"), ("events", "event_id")])
def test_load_outside_points_rejected(tmp_path, inputs, label, id_column):
    source = io.save_inputs(tmp_path, "case", *inputs)
    (source / f"{label}.csv").write_text(f"{id_column},x_m,y_m,z_m\noutside,131,-20,30\n")
    with pytest.raises(ValueError, match=label + " outside"):
        io.load_inputs(tmp_path, "case")


@pytest.mark.parametrize("filename", ["model.npz", "stations.csv", "events.csv"])
@pytest.mark.parametrize("delete", [False, True])
def test_inputs_changed_during_computation(tmp_path, inputs, solver_mock, filename, delete):
    source = io.save_inputs(tmp_path, "case", *inputs)
    def mutate(**kwargs):
        path = source / filename
        if delete:
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"\n")
        return solver_mock[0].return_value
    solver_mock[0].side_effect = mutate
    with pytest.raises(RuntimeError, match="Input files changed"):
        io.run_experiment(tmp_path, "case")
    assert list((tmp_path / "output").iterdir()) == []


def test_hashes_taken_before_load(tmp_path, inputs, solver_mock, monkeypatch):
    source = io.save_inputs(tmp_path, "case", *inputs)
    original_load = io.load_inputs
    def mutate_then_load(*args, **kwargs):
        path = source / "stations.csv"
        path.write_text(path.read_text().replace("001", "002"))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(io, "load_inputs", mutate_then_load)
    with pytest.raises(RuntimeError, match="Input files changed"):
        io.run_experiment(tmp_path, "case")
    assert list((tmp_path / "output").iterdir()) == []


def test_cli_threshold(tmp_path, monkeypatch):
    run = Mock(return_value=tmp_path / "output" / "case")
    monkeypatch.setattr(cli, "run_experiment", run)
    cli.main(["case", "--root", str(tmp_path), "--max-difference-s", "0.003"])
    assert run.call_args.kwargs["max_difference_s"] == 0.003
    assert run.call_args.kwargs["check_accuracy"] is False


def test_cli_invalid_threshold(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["case", "--root", str(tmp_path), "--max-difference-s", "nan"])
    assert error.value.code == 1
    assert "max_difference_s" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_threshold_help(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "r and 2r" in help_text
    assert "not a guaranteed error bound" in help_text


def test_cli_missing_inputs(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["missing", "--root", str(tmp_path)])
    assert error.value.code == 1
    assert "model.npz" in capsys.readouterr().err
    assert not (tmp_path / "input").exists()
    assert not (tmp_path / "output" / "missing").exists()
