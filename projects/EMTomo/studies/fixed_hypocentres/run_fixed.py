"""Diagnostic oracle: EMTomo with every hypocentre fixed at its true position.

Takes main.py's options; forces one hard hypothesis per event. The true
coordinates come from the forward experiment and replace the hypocentre search
in ``tomography.tomography_events`` for this process only (fork workers inherit
it); the package itself is unchanged. Not a method run: truth enters the
inversion here on purpose, to separate velocity resolution from the
hypocentre-velocity trade-off.

    python studies/fixed_hypocentres/run_fixed.py EXPERIMENT_ID [main.py options]
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tomography.tomography_events as events  # noqa: E402
from experiment_data import DEFAULT_EXPERIMENTS_ROOT, load_tomography_experiment  # noqa: E402
from instruments.instruments_coords import sample_cell_centered_trilinear_batch  # noqa: E402
from instruments.likelihood import negative_log_likelihood  # noqa: E402
from main import CONFIG, _parser, main  # noqa: E402


def install_true_hypocentres(experiment_id: str, experiments_root=DEFAULT_EXPERIMENTS_ROOT) -> None:
    """Make ``events.process_event`` use the true hypocentre of each event."""
    data = load_tomography_experiment(experiment_id, experiments_root)
    extent_m = np.asarray(data.source_model.velocity.shape) * data.source_model.cell_size_m
    truth = {
        np.asarray(row, dtype=np.float64).tobytes(): np.asarray(xyz, dtype=np.float64)
        for row, xyz in zip(data.arrival_times_s, data.reference_event_coordinates_m)
    }
    if len(truth) != len(data.event_ids):
        raise ValueError("Arrival rows are not unique; events cannot be identified")
    current = {}

    def locate(times, observed, noise, normalization=None):
        # Called first in process_event with the event's arrivals; skips the grid search.
        fine_shape = np.asarray(times.shape[1:])
        cell_size = extent_m[0] / fine_shape[0]
        position = truth[np.asarray(observed, dtype=np.float64).tobytes()] / cell_size - 0.5
        current["position"] = np.clip(position, 0.0, fine_shape - 1.0)
        return np.zeros(times.shape[1:])

    def select(cost, n, min_distance=1):
        return np.rint(current["position"]).astype(np.int64)[None, :]

    def refine(times, observed, cell, noise):
        position = current["position"].copy()
        predicted = sample_cell_centered_trilinear_batch(times, position)
        return position, float(negative_log_likelihood(predicted, observed, noise))

    events.negative_log_likelihood = locate
    events.select_candidate_cells = select
    events.refine_hypocentre_in_cell = refine


if __name__ == "__main__":
    parser = _parser()
    args = vars(parser.parse_args())
    experiment_id = args.pop("experiment_id")
    experiments_root = args.pop("experiments_root") or DEFAULT_EXPERIMENTS_ROOT
    validate_only = args.pop("validate_only")
    overrides = {key: tuple(value) if isinstance(value, list) else value for key, value in args.items()}
    overrides.update(candidate_mode="hard", n_candidates=1, weights_top_n=1)
    try:
        config = replace(CONFIG, **overrides)
        install_true_hypocentres(experiment_id, experiments_root)
        main(config, experiment_id=experiment_id, experiments_root=experiments_root,
             validate_only=validate_only)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
