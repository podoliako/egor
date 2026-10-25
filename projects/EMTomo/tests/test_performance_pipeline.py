"""End-to-end equivalence and resource contracts of the accelerated execution paths."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import InversionConfig
from eikonal import station_travel_time_fields
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise
from tomography import tomography_events as events
from tomography.tomography_em import _normal_solver, run_em, warm_up_jit
from tomography.tomography_logging import TomographyLogger
from tomography.tomography_math import MatrixFreeNormalEquations, solve_slowness_update
from velocity_model import VelocityModel


@pytest.fixture(scope="module")
def problem():
    warm_up_jit()
    rng = np.random.default_rng(17)
    truth = VelocityModel(rng.uniform(4600., 5400., (4, 3, 3)), 1000.)
    initial = VelocityModel(np.full(truth.shape, 5000.), truth.cell_size)
    stations = np.array([[200., 300., 0.], [3700., 400., 0.], [1900., 2800., 0.],
                         [600., 2500., 0.], [3300., 2600., 0.]])
    fine = truth.refined(2)
    times = station_travel_time_fields(fine, stations)
    positions = rng.uniform([300., 300., 800.], [3700., 2700., 2700.], size=(9, 3))
    arrivals = np.stack([sample_cell_centered_trilinear_batch(
        times, metric_to_cell_coord(position, fine.cell_size)) for position in positions])
    arrivals -= arrivals.min(axis=1, keepdims=True)
    return initial, stations, arrivals, PickNoise(.01, .05, .1)


def fields_and_settings(problem, interpolation="nearest", mode="soft"):
    model, stations, arrivals, noise = problem
    fine = model.refined(2, interpolation)
    times = station_travel_time_fields(fine, stations)
    fields = events.IterationFields.from_times(times, metric_to_cell_coord(stations, fine.cell_size),
                                               fine.cell_size, 2, noise, interpolation)
    settings = events.EventSettings(2, interpolation, 3, 2, 1, mode, 1., noise,
                                    log_g_per_weight=True, log_misfit=True, likelihood_batch_size=4)
    return fields, settings


def assert_logs_equal(reference, actual):
    assert [i for i, _ in reference] == [i for i, _ in actual]
    for (_, a), (_, b) in zip(reference, actual):
        np.testing.assert_array_equal(a.candidate_cells, b.candidate_cells)
        for field in ("positions", "weights", "residuals", "misfit"):
            np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=1e-12, atol=1e-12)
        for field in ("ray_reached_per_weight", "ray_count_per_weight"):
            assert getattr(a, field).keys() == getattr(b, field).keys()
            for key in getattr(a, field):
                np.testing.assert_array_equal(getattr(a, field)[key], getattr(b, field)[key])
        assert a.G_per_weight.keys() == b.G_per_weight.keys()
        for key in a.G_per_weight:
            for field in ("shape", "offsets", "coords", "values"):
                np.testing.assert_array_equal(a.G_per_weight[key][field], b.G_per_weight[key][field])


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("interpolation", ["nearest", "trilinear"])
@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_batched_compact_pipeline_matches_reference(problem, workers, interpolation, mode):
    fields, settings = fields_and_settings(problem, interpolation, mode)
    reference = replace(settings, likelihood_batch_size=1, compact_rays=False, fast_refinement=False)
    old_h, old_b, old_logs = events.run_events(problem[2], fields, reference, workers)
    diagnostics = {}
    new_h, new_b, new_logs = events.run_events(problem[2], fields, settings, workers,
                                              diagnostics=diagnostics)
    np.testing.assert_allclose(new_h, old_h, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(new_b, old_b, rtol=1e-11, atol=1e-10)
    assert_logs_equal(old_logs, new_logs)
    assert diagnostics["workers"] == workers
    assert diagnostics["likelihood_batch_size"] == 4
    assert len(diagnostics["chunk_elapsed_s"]) == workers
    assert set(diagnostics["worker_time_s"]) == {
        "likelihood_s", "refinement_s", "rays_s", "assembly_s", "diagnostics_s",
    }
    assert all(value >= 0 for value in diagnostics["worker_time_s"].values())


@pytest.mark.parametrize("workers", [1, 2])
def test_matrix_free_event_assembly_and_solve(problem, workers):
    fields, settings = fields_and_settings(problem)
    dense, rhs, _ = events.run_events(problem[2], fields, settings, workers)
    operator, operator_rhs, _ = events.run_events(
        problem[2], fields, replace(settings, normal_solver="matrix_free"), workers)
    assert isinstance(operator, MatrixFreeNormalEquations)
    vector = np.linspace(-1, 1, fields.n_vox)
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector, rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(operator.diagonal(), np.diag(dense), rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(operator_rhs, rhs, rtol=1e-11, atol=1e-10)
    expected = solve_slowness_update(dense, rhs, fields.coarse_shape, .1)
    actual = solve_slowness_update(operator, operator_rhs, fields.coarse_shape, .1, rtol=1e-11)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)


def test_batch_memory_budget_and_minimum_one_event():
    fields = SimpleNamespace(times=SimpleNamespace(shape=(26, 128, 128, 128)))
    settings = SimpleNamespace(likelihood_batch_memory_mb=32, likelihood_batch_size=8)
    assert events._effective_batch_size(fields, settings) == 2
    settings.likelihood_batch_memory_mb = 1
    assert events._effective_batch_size(fields, settings) == 1
    settings.likelihood_batch_memory_mb = 256
    assert events._effective_batch_size(fields, settings) == 8


@pytest.mark.parametrize("solver,voxels,damping,expected", [
    ("auto", 8191, .1, "dense"), ("auto", 8192, .1, "matrix_free"),
    ("auto", 8192, 0., "dense"), ("dense", 10000, .1, "dense"),
    ("legacy", 10000, .1, "dense"), ("matrix_free", 8, .1, "matrix_free"),
])
def test_solver_selected_before_allocation(solver, voxels, damping, expected):
    config = InversionConfig(linear_solver=solver, lambda_reg=damping)
    assert _normal_solver(config, voxels) == expected


@pytest.mark.parametrize("mode", ["dense", "matrix_free"])
def test_dense_worker_results_do_not_send_normal_matrices_through_pipe(problem, monkeypatch, mode):
    fields, settings = fields_and_settings(problem)
    settings = replace(settings, normal_solver=mode, log_g_per_weight=False, log_misfit=False)
    context = events.mp.get_context("fork")
    returned = []

    class RecordingPool:
        def __init__(self, *args, **kwargs):
            self.pool = context.Pool(*args, **kwargs)

        def __enter__(self):
            self.pool.__enter__()
            return self

        def __exit__(self, *args):
            return self.pool.__exit__(*args)

        def imap(self, function, tasks):
            for result in self.pool.imap(function, tasks):
                returned.append((result[1] is None, result[2] is None))
                yield result

    proxy = SimpleNamespace(RawArray=context.RawArray, Pool=RecordingPool)
    monkeypatch.setattr(events.mp, "get_context", lambda name: proxy)
    events.run_events(problem[2], fields, settings, 2)
    assert returned == [(mode == "dense", mode == "dense")] * 2


def test_worker_failure_is_not_returned_as_partial_results(problem):
    fields, settings = fields_and_settings(problem)
    with pytest.raises(RuntimeError, match="Failed to process event"):
        events.run_events(problem[2], fields, replace(settings, candidate_mode="invalid"), 2)


def test_no_output_diagnostics_can_be_skipped_without_changing_system(problem):
    fields, settings = fields_and_settings(problem)
    settings = replace(settings, log_g_per_weight=False, log_misfit=False)
    h, b, _ = events.run_events(problem[2], fields, settings)
    fast_h, fast_b, logs = events.run_events(problem[2], fields, replace(settings, log_diagnostics=False))
    np.testing.assert_array_equal(h, fast_h)
    np.testing.assert_array_equal(b, fast_b)
    assert all(not log.ray_count_per_weight and log.residuals.size == 0 for _, log in logs)
    assert all(log.ray_reached_per_weight for _, log in logs)


def test_two_em_cycles_and_publication_match_reference(problem, tmp_path):
    initial, stations, arrivals, noise = problem
    base = InversionConfig(n_cycles=2, subdivision=2, n_workers=2, n_candidates=3, weights_top_n=2,
                            lambda_reg=.1, runs_dir=str(tmp_path), log_g_per_weight=True)
    configurations = [
        replace(base, run_name="reference", likelihood_batch_size=1, compact_rays=False,
                fast_refinement=False, linear_solver="legacy"),
        replace(base, run_name="optimized"),
        replace(base, run_name="matrix-free", linear_solver="matrix_free", solver_rtol=1e-12),
    ]
    runs = [run_em(config, initial, arrivals, stations, noise, reference_model=initial)
            for config in configurations]
    for logger in runs:
        for iteration in range(2):
            folder = logger.run_dir / f"iter_{iteration}"
            assert (folder / "complete.json").exists()
            timing = json.loads((folder / "performance.json").read_text())
            assert set(timing["wall_s"]) == {"refine_model_s", "station_fields_s", "save_fields_s",
                                              "prepare_fields_s", "events_s", "save_events_s", "solve_s"}
            assert timing["solver"]["relative_residual"] < 1e-10
            assert all(value >= 0 for value in timing["wall_s"].values())
    assert json.loads((runs[1].run_dir / "iter_0/performance.json").read_text())["solver"]["solver"] == "cholesky"
    assert json.loads((runs[2].run_dir / "iter_0/performance.json").read_text())["solver"]["solver"] == "cg"
    expected = np.load(runs[0].run_dir / "final_model.npy")
    for logger in runs[1:]:
        np.testing.assert_allclose(np.load(logger.run_dir / "final_model.npy"), expected, rtol=1e-8, atol=1e-5)
        for iteration in range(2):
            for artifact in ("delta_s.npy", "sensitivity_diagonal.npy", "coverage_confidence.npy"):
                a = np.load(logger.run_dir / f"iter_{iteration}" / artifact)
                b = np.load(runs[0].run_dir / f"iter_{iteration}" / artifact)
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-10)


def test_performance_write_failure_does_not_publish(problem, tmp_path, monkeypatch):
    initial, stations, arrivals, noise = problem
    logger = TomographyLogger(base_dir=tmp_path)
    def fail(*args):
        raise OSError("performance write failed")
    monkeypatch.setattr(logger, "save_performance", fail)
    with pytest.raises(OSError, match="performance write"):
        run_em(InversionConfig(n_cycles=1, n_workers=1, subdivision=2), initial,
               arrivals[:1], stations, noise, logger=logger)
    assert not (logger.run_dir / "iter_0/complete.json").exists()
