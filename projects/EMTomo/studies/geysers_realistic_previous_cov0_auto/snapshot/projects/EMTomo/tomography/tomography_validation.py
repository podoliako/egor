"""Held-out absolute-pick validation, independent of the EM normal equations."""
from __future__ import annotations

import numpy as np

from eikonal import station_travel_time_fields
from instruments.instruments_coords import sample_cell_centered_trilinear_batch
from instruments.likelihood import negative_log_likelihood
from .tomography_math import refine_hypocentre_in_cell, select_candidate_cells


def split_control_arrivals(arrivals, *, fraction, seed, min_training_picks):
    """Hide a seeded, capacity-constrained random sample of finite picks with NaN.

    The target is ``round(fraction * n_finite)``, capped by the sum of event
    capacities. A random permutation of finite picks is accepted until that
    target is reached, skipping events already at their training minimum.
    Events with at most ``min_training_picks`` observations are unchanged.
    NaN denotes a missing pick; infinities are rejected, as in the inversion.
    The input is never mutated and is returned unchanged when no picks are hidden.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and in [0, 1]")
    if (isinstance(min_training_picks, (bool, np.bool_))
            or not isinstance(min_training_picks, (int, np.integer))
            or min_training_picks < 2):
        raise ValueError("min_training_picks must be an integer >= 2")
    values = np.asarray(arrivals, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("arrivals must have shape (events, global stations)")
    if np.any(np.isinf(values)):
        raise ValueError("arrivals must not contain inf; use NaN for missing picks")
    finite = np.isfinite(values)
    capacity = np.maximum(finite.sum(axis=1) - min_training_picks, 0)
    target = min(round(float(fraction) * int(finite.sum())), int(capacity.sum()))
    control_mask = np.zeros(values.shape, dtype=bool)
    if target == 0:
        return arrivals, control_mask

    rng = np.random.default_rng(seed)
    n_stations = values.shape[1]
    selected = 0
    for index in rng.permutation(np.flatnonzero(finite)):
        event = index // n_stations
        if capacity[event] == 0:
            continue
        control_mask.flat[index] = True
        capacity[event] -= 1
        selected += 1
        if selected == target:
            break
    training = values.copy()
    training[control_mask] = np.nan
    return training, control_mask


def evaluate_control_rmse(model, training_arrivals, arrivals, control_mask, stations_m, noise, config) -> float:
    """Global held-out absolute-arrival RMSE in seconds for the supplied model.

    Recompute FMM fields (using ``config.n_workers``) after refining this model.
    Only events with control picks are localized, serially, using their training
    picks and the same grid shortlist/refine/best-likelihood rule as process_event.
    Temperature, candidate weights and ray tracing play no role. Fit origin time
    from training residuals with precision ``1 / noise.sigmas(predicted)**2``.
    Station-indexed APIs share the full field volume without subset copies.

    Control entries must be finite in ``arrivals`` and NaN in ``training_arrivals``.
    At least two training picks are required for each scored event. An empty
    control set or nonfinite predictions/scores raise ValueError rather than
    returning an undefined metric or silently dropping picks.
    """
    training = np.asarray(training_arrivals, dtype=np.float64)
    observed = np.asarray(arrivals, dtype=np.float64)
    mask = np.asarray(control_mask)
    stations = np.asarray(stations_m, dtype=np.float64)
    if training.ndim != 2 or observed.shape != training.shape:
        raise ValueError("training_arrivals and arrivals must have matching (events, stations) shapes")
    if mask.shape != training.shape or mask.dtype != np.bool_:
        raise ValueError("control_mask must be boolean with the arrivals shape")
    if stations.shape != (training.shape[1], 3) or not np.all(np.isfinite(stations)):
        raise ValueError("stations_m must have shape (global stations, 3) and be finite")
    if np.any(np.isinf(training)) or np.any(np.isinf(observed)):
        raise ValueError("arrivals must not contain inf; use NaN for missing picks")
    n_control = int(mask.sum())
    if n_control == 0:
        raise ValueError("control_mask must select at least one held-out pick")
    if not np.all(np.isfinite(observed[mask])):
        raise ValueError("Control arrivals must be finite")
    if not np.all(np.isnan(training[mask])):
        raise ValueError("Control picks must be excluded from training_arrivals (NaN)")
    event_indices = np.flatnonzero(mask.any(axis=1))
    for event in event_indices:
        if np.count_nonzero(np.isfinite(training[event])) < 2:
            raise ValueError(f"Control event {event} requires at least two training picks")

    fine = model.refined(config.subdivision, config.slowness_interpolation)
    times = station_travel_time_fields(fine, stations, config.n_workers)
    # Check one station at a time, avoiding a full-volume boolean temporary.
    for station, field in enumerate(times):
        if not np.all(np.isfinite(field)):
            raise ValueError(f"Nonfinite travel-time field for station {station}")

    squared_error = 0.0
    for event in event_indices:
        training_indices = np.flatnonzero(np.isfinite(training[event]))
        picks = training[event, training_indices]
        cost = negative_log_likelihood(times, picks, noise, station_indices=training_indices)
        if not np.all(np.isfinite(cost)):
            raise ValueError(f"Nonfinite training likelihood for control event {event}")
        cells = select_candidate_cells(cost, config.n_candidates, config.weights_min_distance)
        # Match process_event: refine the entire shortlist before ranking it.
        refined = [refine_hypocentre_in_cell(
            times, picks, cell, noise, accelerated=config.fast_refinement,
            station_indices=training_indices,
        ) for cell in cells]
        positions = np.asarray([position for position, _ in refined], dtype=np.float64)
        costs = np.asarray([value for _, value in refined], dtype=np.float64)
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(costs)):
            raise ValueError(f"Nonfinite refined candidate for control event {event}")
        position = positions[int(np.argmin(costs))]
        predicted_training = np.asarray(sample_cell_centered_trilinear_batch(
            times, position, station_indices=training_indices,
        ), dtype=np.float64)
        control_indices = np.flatnonzero(mask[event])
        predicted_control = np.asarray(sample_cell_centered_trilinear_batch(
            times, position, station_indices=control_indices,
        ), dtype=np.float64)
        if not (np.all(np.isfinite(predicted_training)) and np.all(np.isfinite(predicted_control))):
            raise ValueError(f"Nonfinite predicted arrivals for control event {event}")
        precision = 1.0 / noise.sigmas(predicted_training) ** 2
        if not np.all(np.isfinite(precision)) or np.any(precision <= 0):
            raise ValueError(f"Invalid training precision for control event {event}")
        origin = np.sum(precision * (picks - predicted_training)) / np.sum(precision)
        residual = observed[event, control_indices] - (predicted_control + origin)
        if not np.isfinite(origin) or not np.all(np.isfinite(residual)):
            raise ValueError(f"Nonfinite control residual for event {event}")
        squared_error += float(np.dot(residual, residual))

    rmse = float(np.sqrt(squared_error / n_control))
    if not np.isfinite(rmse):
        raise ValueError("Nonfinite control RMSE")
    return rmse
