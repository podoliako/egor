import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExampleConfig:
    """Saved-experiment inversion options, with legacy synthetic scenario fields."""

    # Model geometry and geographic reference: 350 x 150 x 70 km.
    cell_size: float = 10_000.0
    grid_shape: tuple[int, int, int] = (35, 15, 7)
    lon: float = 37.6173
    lat: float = 55.7558
    height: float = 50.0
    azimuth: float = 45.0


    # Legacy in-process synthetic scenarios (archive/legacy_synthetics).
    station_grid_shape: tuple[int, int] = (7, 5)
    station_locations_csv: str | None = None
    event_locations_csv: str | None = None
    background_vp: float = 5000.0
    initial_gradient_m_s: tuple[float, float] | None = None
    initial_layer_boundaries_km: tuple[float, ...] | None = None
    initial_layer_velocities_m_s: tuple[float, ...] | None = None
    checkerboard_anomaly_fraction: float = 0.05
    checkerboard_cell_size: float = 20_000.0
    checkerboard_block_shape: tuple[int, int, int] | None = None
    checkerboard_pattern_shape: tuple[int, int, int] | None = None
    checkerboard_rotation_degrees: float = 45.0

    # Synthetic arrival generation: events are placed on a uniform volume grid.
    subdivision: int = 3
    event_grid_shape: tuple[int, int, int] = (35, 15, 7)  # 3,675 hypocentres.
    random_seed: int = 7
    event_depth_bias: float = 0.0
    event_z_offset: float = 250.0
    slowness_interpolation: str = "nearest"
    arrival_noise_std: float = 0.01  # Gaussian pick noise: 10 ms per station.
    synthetic_arrivals_cache: str | None = None

    # EM inversion. Change the version for every method release.
    run_name: str = "em"
    run_version: str = "1.3"
    n_cycles: int = 7
    weights_top_n: int = 1
    weights_min_distance: int = 1
    candidate_mode: str = "soft"
    temperature: float = 1
    weight_noise_relative_sigma: float | None = None
    weight_noise_absolute_sigma_s: float | None = None
    weight_model_sigma_s: float = 0.2
    lambda_reg: float = 0.01
    smoothness_reg: float = 0.0
    coverage_damping_power: float = 1
    coverage_floor: float = 0.05
    coverage_reference_percentile: float = 75.0
    max_velocity_step_fraction: float = 0.03
    v_bounds: tuple[float, float] = (4000.0, 6000.0)
    v_reg_strength: float = 0.0
    v_left_mode: str = "lin"
    v_right_mode: str = "lin"
    v_left_rate: float = 0.0
    v_right_rate: float = 0.0
    v_left_power: float = 0.0
    v_right_power: float = 0.0

    # Runtime and output.
    n_workers: int = 25
    save_runs: bool = True
    runs_dir: str = "runs"
    log_g_per_weight: bool = False
    profiling_stats_limit: int = 30


CONFIG = ExampleConfig()


def main(config: ExampleConfig = CONFIG, *, experiment_id: str | None = None,
         experiments_root: str | Path | None = None, validate_only: bool = False):
    if experiment_id is not None:
        from experiment_runner import run_saved_experiment

        kwargs = {"validate_only": validate_only}
        if experiments_root is not None:
            kwargs["experiments_root"] = experiments_root
        return run_saved_experiment(experiment_id, config, **kwargs)
    raise ValueError(
        "main(config) requires an experiment_id; use "
        "archive.legacy_synthetics.runner.main(config) for legacy synthetics"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EMTomo on a saved forward experiment")
    parser.add_argument("experiment_id", help="Name of matching input/ and output/ experiment directories")
    parser.add_argument("--experiments-root", type=Path, default=None)
    parser.add_argument("--cell-size-m", type=float, default=CONFIG.cell_size,
                        help="Inversion cell side in metres (must tile the forward domain)")
    parser.add_argument("--initial-gradient-m-s", type=float, nargs=2, metavar=("TOP", "BOTTOM"),
                        help="Approximate 1-D initial speed at the domain top and bottom; default: homogeneous 5000 m/s")
    parser.add_argument("--initial-layer-boundaries-km", type=float, nargs="+", metavar="Z",
                        help="Horizontal depth boundaries for an approximate layered initial model")
    parser.add_argument("--initial-layer-velocities-m-s", type=float, nargs="+", metavar="V",
                        help="Initial layer speeds from top to bottom (one more than boundaries)")
    parser.add_argument("--subdivision", type=int, default=CONFIG.subdivision)
    parser.add_argument("--cycles", type=int, default=CONFIG.n_cycles)
    parser.add_argument("--workers", type=int, default=CONFIG.n_workers)
    parser.add_argument("--weights-top-n", type=int, default=CONFIG.weights_top_n,
                        help="Number of candidate hypocentres per event (default: 1)")
    parser.add_argument("--weights-min-distance", type=int, default=CONFIG.weights_min_distance,
                        help="Chebyshev separation of candidate hypocentres in fine-grid cells")
    parser.add_argument("--candidate-mode", choices=("soft", "hard"), default=CONFIG.candidate_mode,
                        help="Soft: all shortlisted hypotheses; hard: most likely of the same refined shortlist")
    parser.add_argument("--temperature", type=float, default=CONFIG.temperature,
                        help="Candidate likelihood temperature: 1 for untempered, >1 for softer weights")
    parser.add_argument("--weight-noise-relative-sigma", type=float, default=None,
                        help="Override the relative pick-noise sigma in saved metadata")
    parser.add_argument("--weight-noise-absolute-sigma-s", type=float, default=None,
                        help="Override the absolute pick-noise sigma in saved metadata (seconds)")
    parser.add_argument("--weight-model-sigma-s", type=float, default=CONFIG.weight_model_sigma_s,
                        help="Model/numerical mismatch sigma in seconds (default: 0.2)")
    parser.add_argument("--lambda-reg", type=float, default=CONFIG.lambda_reg)
    parser.add_argument("--smoothness-reg", type=float, default=CONFIG.smoothness_reg,
                        help="Dimensionless six-neighbor penalty on slowness updates (default: off)")
    parser.add_argument("--coverage-damping-power", type=float, default=CONFIG.coverage_damping_power)
    parser.add_argument("--max-velocity-step-fraction", type=float, default=CONFIG.max_velocity_step_fraction)
    parser.add_argument("--run-name", type=str, default=CONFIG.run_name)
    parser.add_argument("--runs-dir", type=str, default=CONFIG.runs_dir)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the experiment and inversion setup without running EM")
    args = parser.parse_args()
    if args.subdivision < 1 or args.cycles < 1 or args.workers < 1 or args.weights_top_n < 1 or args.weights_min_distance < 1:
        parser.error("--subdivision, --cycles, --workers, --weights-top-n and --weights-min-distance must be positive")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    for name in ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s", "weight_model_sigma_s"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.lambda_reg < 0 or not np.isfinite(args.smoothness_reg) or args.smoothness_reg < 0 or args.coverage_damping_power < 0 or not 0 < args.max_velocity_step_fraction <= 1:
        parser.error("--lambda-reg, --smoothness-reg and --coverage-damping-power must be nonnegative; --max-velocity-step-fraction must be in (0, 1]")
    config = replace(CONFIG, cell_size=args.cell_size_m, subdivision=args.subdivision,
                     initial_gradient_m_s=(tuple(args.initial_gradient_m_s) if args.initial_gradient_m_s is not None else None),
                     initial_layer_boundaries_km=(tuple(args.initial_layer_boundaries_km) if args.initial_layer_boundaries_km is not None else None),
                     initial_layer_velocities_m_s=(tuple(args.initial_layer_velocities_m_s) if args.initial_layer_velocities_m_s is not None else None),
                     n_cycles=args.cycles, n_workers=args.workers, weights_top_n=args.weights_top_n,
                     weights_min_distance=args.weights_min_distance, candidate_mode=args.candidate_mode,
                     temperature=args.temperature,
                     weight_noise_relative_sigma=args.weight_noise_relative_sigma,
                     weight_noise_absolute_sigma_s=args.weight_noise_absolute_sigma_s,
                     weight_model_sigma_s=args.weight_model_sigma_s,
                     runs_dir=args.runs_dir,
                     lambda_reg=args.lambda_reg, smoothness_reg=args.smoothness_reg,
                     coverage_damping_power=args.coverage_damping_power,
                     max_velocity_step_fraction=args.max_velocity_step_fraction, run_name=args.run_name)
    try:
        main(config, experiment_id=args.experiment_id,
             experiments_root=args.experiments_root, validate_only=args.validate_only)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
