"""Run previously saved forward-modeling inputs from the command line."""

import argparse
from pathlib import Path

from .experiments import run_experiment
from .model import ForwardConfig
from .noise import NoiseConfig


def main(argv=None):
    defaults = ForwardConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--refinement", type=int, default=defaults.refinement)
    parser.add_argument("--source-radius-cells", type=int, default=defaults.source_radius_cells)
    parser.add_argument("--check-convergence", action="store_true")
    parser.add_argument("--noise", action="store_true", help="Add independent propagation and picking noise")
    parser.add_argument("--noise-relative-sigma", type=float, default=None,
                        help="Sigma / absolute travel time (default with --noise: 0.01)")
    parser.add_argument("--noise-absolute-sigma-s", type=float, default=None,
                        help="Distance-independent picking sigma in seconds (default with --noise: 0.05)")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Reproducible noise seed (default with --noise: 42)")
    parser.add_argument(
        "--max-difference-s", type=float,
        help="Require relative and absolute time differences between r and 2r to be "
             "within this positive threshold in seconds; enables convergence checking, "
             "not a guaranteed error bound",
    )
    args = parser.parse_args(argv)
    try:
        config = ForwardConfig(refinement=args.refinement,
                               source_radius_cells=args.source_radius_cells)
        options = {}
        noise_options = {name: value for name, value in (
            ("relative_sigma", args.noise_relative_sigma),
            ("absolute_sigma_s", args.noise_absolute_sigma_s),
            ("seed", args.noise_seed),
        ) if value is not None}
        if noise_options and not args.noise:
            raise ValueError("Noise parameters require --noise")
        if args.noise:
            options["noise"] = NoiseConfig(**noise_options)
        if args.max_difference_s is not None:
            options["max_difference_s"] = args.max_difference_s
        destination = run_experiment(
            root=args.root, experiment_id=args.experiment_id, config=config,
            check_accuracy=args.check_convergence, **options,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
