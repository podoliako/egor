"""Performance configuration and CLI checks; never launch an inversion."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from config import InversionConfig
from main import CONFIG, _parser


DEFAULTS = {
    "likelihood_batch_size": 8,
    "likelihood_batch_memory_mb": 128,
    "compact_rays": True,
    "fast_refinement": True,
    "linear_solver": "auto",
    "matrix_free_min_voxels": 8192,
    "solver_rtol": 1e-10,
    "solver_maxiter": None,
}


def _parse_config(*options):
    args = vars(_parser().parse_args(["test_experiment", *options]))
    for name in ("experiment_id", "experiments_root", "validate_only"):
        args.pop(name)
    return replace(CONFIG, **{
        name: tuple(value) if isinstance(value, list) else value
        for name, value in args.items()
    })


def test_defaults_and_metadata_include_all_performance_controls():
    config = InversionConfig()
    metadata = json.loads(json.dumps(asdict(config)))
    for name, value in DEFAULTS.items():
        assert getattr(config, name) == value
        assert metadata[name] == value
    assert config.run_version == "1.4"
    assert config.n_workers == 25
    assert config.lambda_reg == 0.01
    assert config.cell_size == 10000.0
    assert config.subdivision == 3
    assert config.n_cycles == 7
    assert config.n_candidates == 5
    assert config.temperature == 1.0
    assert config.weight_model_sigma_s == 0.2
    assert config.max_velocity_step_fraction == 0.03


@pytest.mark.parametrize("name", ["likelihood_batch_size", "likelihood_batch_memory_mb",
                                  "matrix_free_min_voxels", "solver_maxiter"])
@pytest.mark.parametrize("value", [0, -1, 1.5, 8.0, True, False, np.bool_(True), "8", np.nan, np.inf])
def test_integer_controls_reject_nonpositive_or_noninteger_values(name, value):
    with pytest.raises(ValueError, match=name):
        InversionConfig(**{name: value})


@pytest.mark.parametrize("name", ["likelihood_batch_size", "likelihood_batch_memory_mb",
                                  "matrix_free_min_voxels", "solver_maxiter"])
@pytest.mark.parametrize("value", [1, 8, np.int64(16)])
def test_integer_controls_accept_positive_integers(name, value):
    assert getattr(InversionConfig(**{name: value}), name) == value


@pytest.mark.parametrize("name", ["likelihood_batch_size", "likelihood_batch_memory_mb",
                                  "matrix_free_min_voxels"])
def test_required_integer_controls_reject_none(name):
    with pytest.raises(ValueError, match=name):
        InversionConfig(**{name: None})


def test_solver_maxiter_accepts_none():
    assert InversionConfig(solver_maxiter=None).solver_maxiter is None


@pytest.mark.parametrize("name", ["compact_rays", "fast_refinement"])
@pytest.mark.parametrize("value", [0, 1, "true", "false", None, np.bool_(True)])
def test_boolean_controls_require_actual_bool(name, value):
    with pytest.raises(ValueError, match=name):
        InversionConfig(**{name: value})


@pytest.mark.parametrize("name", ["compact_rays", "fast_refinement"])
@pytest.mark.parametrize("value", [False, True])
def test_boolean_controls_accept_bool(name, value):
    assert getattr(InversionConfig(**{name: value}), name) is value


@pytest.mark.parametrize("value", [0.0, 1.0, -1e-10, 1.1, np.nan, np.inf, -np.inf,
                                   True, False, None, "1e-10", complex(0.1, 0.0)])
def test_rtol_requires_finite_open_unit_interval(value):
    with pytest.raises(ValueError, match="solver_rtol"):
        InversionConfig(solver_rtol=value)


@pytest.mark.parametrize("value", [1e-10, 0.5, np.float32(0.01), np.nextafter(1.0, 0.0)])
def test_rtol_accepts_valid_values(value):
    assert InversionConfig(solver_rtol=value).solver_rtol == value


@pytest.mark.parametrize("solver", ["auto", "dense", "matrix_free", "legacy"])
def test_solver_choices(solver):
    assert InversionConfig(linear_solver=solver).linear_solver == solver


@pytest.mark.parametrize("solver", ["cg", "lu", "AUTO", "", None, True, 1])
def test_invalid_solver_choices(solver):
    with pytest.raises(ValueError, match="linear_solver"):
        InversionConfig(linear_solver=solver)


@pytest.mark.parametrize("damping", [0.0, None])
def test_forced_matrix_free_requires_positive_damping(damping):
    with pytest.raises(ValueError, match="matrix_free.*lambda_reg > 0"):
        InversionConfig(linear_solver="matrix_free", lambda_reg=damping)


@pytest.mark.parametrize("solver", ["auto", "dense", "legacy"])
def test_undamped_direct_or_auto_configuration_remains_valid(solver):
    config = InversionConfig(linear_solver=solver, lambda_reg=0.0, matrix_free_min_voxels=1)
    assert config.lambda_reg == 0.0


def test_legacy_solver_does_not_toggle_kernels():
    config = InversionConfig(linear_solver="legacy")
    assert config.compact_rays is True
    assert config.fast_refinement is True
    assert config.likelihood_batch_size == 8


def test_parser_defaults_match_config():
    assert _parse_config() == CONFIG


def test_parser_all_performance_overrides():
    config = _parse_config(
        "--likelihood-batch-size", "3", "--likelihood-batch-memory-mb", "64",
        "--no-compact-rays", "--no-fast-refinement", "--linear-solver", "matrix_free",
        "--matrix-free-min-voxels", "4096", "--solver-rtol", "1e-9", "--solver-maxiter", "500",
    )
    assert {name: getattr(config, name) for name in DEFAULTS} == {
        "likelihood_batch_size": 3, "likelihood_batch_memory_mb": 64,
        "compact_rays": False, "fast_refinement": False, "linear_solver": "matrix_free",
        "matrix_free_min_voxels": 4096, "solver_rtol": 1e-9, "solver_maxiter": 500,
    }
    assert config.lambda_reg == CONFIG.lambda_reg
    assert config.n_workers == CONFIG.n_workers


def test_parser_reference_flags():
    config = _parse_config("--likelihood-batch-size", "1", "--no-compact-rays",
                           "--no-fast-refinement", "--linear-solver", "legacy")
    assert config.likelihood_batch_size == 1
    assert not config.compact_rays
    assert not config.fast_refinement
    assert config.linear_solver == "legacy"


def test_parser_boolean_optional_actions_can_reenable_controls():
    config = _parse_config("--no-compact-rays", "--compact-rays",
                           "--no-fast-refinement", "--fast-refinement")
    assert config.compact_rays is True
    assert config.fast_refinement is True


@pytest.mark.parametrize("value", ["none", "None", "NONE"])
def test_parser_optional_iteration_limit(value):
    assert _parse_config("--solver-maxiter", value).solver_maxiter is None


@pytest.mark.parametrize("solver", ["auto", "dense", "matrix_free", "legacy"])
def test_parser_solver_choices(solver):
    assert _parse_config("--linear-solver", solver).linear_solver == solver


@pytest.mark.parametrize("options", [
    ["--linear-solver", "cg"], ["--likelihood-batch-size", "1.5"],
    ["--likelihood-batch-memory-mb", "true"], ["--matrix-free-min-voxels", "2.5"],
    ["--solver-maxiter", "1.5"], ["--solver-rtol", "invalid"], ["--compact-rays", "false"],
])
def test_parser_rejects_invalid_syntax(options):
    with pytest.raises(SystemExit) as exc:
        _parse_config(*options)
    assert exc.value.code == 2


@pytest.mark.parametrize("options, field", [
    (["--likelihood-batch-size", "0"], "likelihood_batch_size"),
    (["--likelihood-batch-memory-mb", "-1"], "likelihood_batch_memory_mb"),
    (["--matrix-free-min-voxels", "0"], "matrix_free_min_voxels"),
    (["--solver-rtol", "nan"], "solver_rtol"),
    (["--solver-rtol", "1"], "solver_rtol"),
    (["--solver-maxiter", "0"], "solver_maxiter"),
    (["--linear-solver", "matrix_free", "--lambda-reg", "0"], "lambda_reg"),
])
def test_cli_values_are_validated_when_building_config(options, field):
    with pytest.raises(ValueError, match=field):
        _parse_config(*options)
