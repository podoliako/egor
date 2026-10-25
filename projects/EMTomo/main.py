"""Run EMTomo on a saved forward experiment: ``python main.py EXPERIMENT_ID [options]``."""
import argparse
from dataclasses import replace
from pathlib import Path

from config import InversionConfig

CONFIG = InversionConfig()


def main(config: InversionConfig = CONFIG, *, experiment_id: str,
         experiments_root: str | Path | None = None, validate_only: bool = False):
    from experiment_runner import run_saved_experiment

    kwargs = {"validate_only": validate_only}
    if experiments_root is not None:
        kwargs["experiments_root"] = experiments_root
    return run_saved_experiment(experiment_id, config, **kwargs)


_CLI_FIELDS = {"cell_size_m": "cell_size", "cycles": "n_cycles", "workers": "n_workers"}


def _optional_fraction(text: str):
    return None if text.lower() == "none" else float(text)


def _optional_int(text: str):
    return None if text.lower() == "none" else int(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="Name of matching input/ and output/ experiment directories")
    parser.add_argument("--experiments-root", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the experiment and inversion setup without running EM")

    def option(name, **kwargs):
        field = name.replace("-", "_")
        field = _CLI_FIELDS.get(field, field)
        parser.add_argument(f"--{name}", dest=field, default=getattr(CONFIG, field), **kwargs)

    option("cell-size-m", type=float, help="Inversion cell side in metres (must tile the domain)")
    option("subdivision", type=int)
    option("slowness-interpolation", choices=("nearest", "trilinear"))
    option("initial-velocity-m-s", type=float, help="Homogeneous starting velocity")
    option("initial-gradient-m-s", type=float, nargs=2, metavar=("TOP", "BOTTOM"),
           help="Linear starting velocity between the domain top and bottom")
    option("initial-layer-boundaries-km", type=float, nargs="+", metavar="Z",
           help="Depths of horizontal starting-layer boundaries")
    option("initial-layer-velocities-m-s", type=float, nargs="+", metavar="V",
           help="Starting-layer velocities from top to bottom (one more than boundaries)")
    option("cycles", type=int)
    option("n-candidates", type=int,
           help="Cells per event refined within the cell before choosing hypotheses")
    option("weights-top-n", type=int,
           help="Best refined candidates kept as weighted hypotheses (<= n-candidates)")
    option("weights-min-distance", type=int,
           help="Chebyshev separation of hypotheses in fine-grid cells")
    option("candidate-mode", choices=("soft", "hard"),
           help="soft: all weighted hypotheses; hard: the most likely of the same shortlist")
    option("temperature", type=float, help="Likelihood temperature: 1 is untempered")
    option("weight-noise-relative-sigma", type=float,
           help="Override the relative pick-noise sigma from the experiment metadata")
    option("weight-noise-absolute-sigma-s", type=float,
           help="Override the absolute pick-noise sigma from the experiment metadata")
    option("weight-model-sigma-s", type=float, help="Modelling-error sigma in seconds")
    option("lambda-reg", type=float)
    option("coverage-damping-power", type=float)
    option("max-velocity-step-fraction", type=_optional_fraction,
           help="Per-cycle relative velocity limit per cell, or 'none'")
    option("likelihood-batch-size", type=int,
           help="Maximum events per likelihood batch (>= 1)")
    option("likelihood-batch-memory-mb", type=int,
           help="Per-event-worker cost-cube budget in MB; always allow at least one event")
    option("compact-rays", action=argparse.BooleanOptionalAction,
           help="Use compact ray sensitivities instead of dense station/voxel arrays")
    option("fast-refinement", action=argparse.BooleanOptionalAction,
           help="Use fused compiled interpolation/likelihood with unchanged Powell settings")
    option("linear-solver", choices=("auto", "dense", "matrix_free", "legacy"),
           help="auto: dense below threshold or without damping; matrix_free requires lambda-reg > 0")
    option("matrix-free-min-voxels", type=int,
           help="Minimum inversion voxel count for auto matrix-free solving (>= 1)")
    option("solver-rtol", type=float,
           help="Matrix-free solver relative residual tolerance, strictly between 0 and 1")
    option("solver-maxiter", type=_optional_int,
           help="Matrix-free solver iteration limit (positive integer), or 'none' for solver default")
    option("workers", type=int)
    option("run-name")
    option("runs-dir")
    return parser


if __name__ == "__main__":
    parser = _parser()
    args = vars(parser.parse_args())
    experiment_id = args.pop("experiment_id")
    experiments_root = args.pop("experiments_root")
    validate_only = args.pop("validate_only")
    overrides = {key: tuple(value) if isinstance(value, list) else value
                 for key, value in args.items()}
    try:
        config = replace(CONFIG, **overrides)
        main(config, experiment_id=experiment_id, experiments_root=experiments_root,
             validate_only=validate_only)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
