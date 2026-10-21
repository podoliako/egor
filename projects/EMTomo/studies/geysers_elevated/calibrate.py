"""Observation-only, initial-model calibration of one common soft temperature.

Public API: calibrate(prepared, config, runtime, state_dir) -> JSON-safe dict.
``prepared`` is PreparedInversion; ``runtime`` is the namespace returned by
geysers2011.run_case.load_runtime (np and resolve_pick_noise are used here).
Production imports are delayed until the caller has installed snapshot paths.
Writes state_dir/calibration.json non-atomically; the pipeline owns publication.
No event truth, reference model, rays, normal equations, or model updates are used.
"""

from __future__ import annotations

import json
from pathlib import Path


_KS = (1, 5, 10, 15)
_TARGET_ESS = 3.0
_MAX_TEMPERATURE = 16.0


def _summary(values, np):
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "median": float(median), "p90": float(p90)}


def _choose_temperature(costs, np, candidate_weights):
    """Find the smallest T >= 1 attaining median ESS 3, or report saturation."""
    trace = []

    def evaluate(temperature):
        ess = []
        for values in costs:
            weights = candidate_weights(values, temperature)
            ess.append(1.0 / float(weights @ weights))
        median = float(np.median(ess))
        trace.append({"temperature": float(temperature), "median_ess": median})
        return median

    baseline = evaluate(1.0)
    saturated = False
    if baseline >= _TARGET_ESS:
        temperature, achieved, reason = 1.0, baseline, "proper_likelihood_already_sufficient"
    else:
        upper_ess = evaluate(_MAX_TEMPERATURE)
        if upper_ess < _TARGET_ESS:
            temperature, achieved, reason = _MAX_TEMPERATURE, upper_ess, "upper_bound_saturated"
            saturated = True
        else:
            lower, upper = 1.0, _MAX_TEMPERATURE
            for _ in range(40):
                middle = (lower + upper) / 2.0
                if evaluate(middle) < _TARGET_ESS:
                    lower = middle
                else:
                    upper = middle
            temperature = upper
            achieved = evaluate(temperature)
            reason = "target_reached"
    return float(temperature), {
        "target_k": 15, "target_median_ess": _TARGET_ESS,
        "bounds": [1.0, _MAX_TEMPERATURE], "median_ess_at_t1": baseline,
        "median_ess_at_selected_temperature": achieved,
        "saturated": saturated, "reason": reason, "search_ledger": trace,
    }


def _location_groups(positions, cell_size, np):
    """Connected components of pair distances strictly < 1 m (transitive)."""
    labels = list(range(len(positions)))
    for i in range(len(positions)):
        for j in range(i):
            if float(np.linalg.norm(positions[i] - positions[j]) * cell_size) < 1.0:
                old, new = labels[i], labels[j]
                labels = [new if label == old else label for label in labels]
    return labels


def _weight_diagnostics(costs, positions, temperature, cell_size, np, candidate_weights):
    weights = candidate_weights(costs, temperature)
    labels = _location_groups(positions, cell_size, np)
    masses = [float(weights[np.asarray(labels) == label].sum()) for label in sorted(set(labels))]
    return {
        "weights": weights.tolist(), "ess": 1.0 / float(weights @ weights),
        "max_weight": float(weights.max()), "positive_weight_count": int((weights > 0).sum()),
        "unique_hypotheses": len(masses),
        "unique_location_ess": 1.0 / sum(mass * mass for mass in masses),
        "unique_location_max_weight": max(masses),
    }


def calibrate(prepared, config, runtime, state_dir) -> dict:
    """Calibrate common soft T on <=128 evenly spaced observation-row indices.

    Uses production grid NLL, shortlist, cell refinement, stable cost ranking,
    and candidate_weights, but stops before ray tracing. Each smaller K is
    selected independently to check the K15 prefix assumption (ties can break
    it); refinement is cached by cell, not by globally refined rank.

    The result contains ``temperature``, ``selection``, ``per_k`` summaries,
    ``events`` (native grid/refined costs, positions and weights), ``cases``
    (pipeline config overrides), and provenance/limitations. Hard overrides
    always use T=1/top_n=1; soft overrides retain top_n=K with the common T.
    Neither prepared nor config is changed. Only calibration.json is written.
    """
    # Must resolve from the frozen runtime, never from paths set at module import.
    from eikonal import station_travel_time_fields
    from instruments.likelihood import candidate_weights, likelihood_normalization, negative_log_likelihood
    from tomography.tomography_events import refine_hypocentre_in_cell, select_candidate_cells

    np = runtime.np
    observations = np.asarray(prepared.arrivals_table, dtype=np.float64)
    stations = np.asarray(prepared.station_locs, dtype=np.float64)
    if (observations.ndim != 2 or observations.shape[0] == 0
            or observations.shape[1] < 2 or not np.all(np.isfinite(observations))):
        raise ValueError("Expected nonempty finite observations (events, stations >= 2)")
    if stations.shape != (observations.shape[1], 3) or not np.all(np.isfinite(stations)):
        raise ValueError("Station locations must be finite and match observation columns")
    indices = np.linspace(0, len(observations) - 1, min(128, len(observations)), dtype=int)
    noise = runtime.resolve_pick_noise(config, prepared.noise_sigmas)
    fine = prepared.initial_model.refined(config.subdivision, config.slowness_interpolation)
    times = station_travel_time_fields(fine, prepared.station_locs, n_workers=config.n_workers)
    normalization = likelihood_normalization(times, noise)
    events = []
    prefix_failures = []
    costs_by_k = {k: [] for k in _KS}
    for event_index in indices:
        observed = np.asarray(observations[event_index], dtype=np.float64)
        grid_cost = negative_log_likelihood(times, observed, noise, normalization)
        largest = select_candidate_cells(grid_cost, 15, config.weights_min_distance)
        selections = {15: largest}
        for k in _KS[:-1]:
            selections[k] = select_candidate_cells(grid_cost, k, config.weights_min_distance)
        cache = {}
        per_k = {}
        for k in (15, 1, 5, 10):
            cells = selections[k]
            prefix_matches = bool(np.array_equal(cells, largest[:k]))
            if not prefix_matches:
                prefix_failures.append({"event_index": int(event_index), "k": k})
            refined = []
            for cell in cells:
                key = tuple(int(v) for v in cell)
                if key not in cache:
                    position, value = refine_hypocentre_in_cell(times, observed, cell, noise)
                    position = np.asarray(position, dtype=np.float64)
                    if position.shape != (3,) or not np.all(np.isfinite(position)) or not np.isfinite(value):
                        raise ValueError(f"Nonfinite refinement for event {event_index}, cell {key}")
                    cache[key] = (position, float(value))
                refined.append(cache[key])
            order = np.argsort([value for _, value in refined], kind="stable")
            costs = np.asarray([refined[i][1] for i in order], dtype=np.float64)
            positions = np.asarray([refined[i][0] for i in order], dtype=np.float64)
            costs_by_k[k].append(costs)
            per_k[str(k)] = {
                "shortlist_cells": cells.tolist(),
                "shortlist_grid_costs": [float(grid_cost[tuple(cell)]) for cell in cells],
                "prefix_matches_k15": prefix_matches,
                "refined_order": order.tolist(), "ranked_cells": cells[order].tolist(),
                "costs": costs.tolist(), "positions_fine_index": positions.tolist(),
                "best_cost": float(costs[0]), "retained_count": len(costs),
            }
        events.append({"event_index": int(event_index), "refinements": len(cache), "per_k": per_k})

    temperature, selection = _choose_temperature(costs_by_k[15], np, candidate_weights)
    summaries = {}
    warnings = []
    for k in _KS:
        metrics = {"at_t1": [], "at_temperature": []}
        for event in events:
            detail = event["per_k"][str(k)]
            for key, t in (("at_t1", 1.0), ("at_temperature", temperature)):
                detail[key] = _weight_diagnostics(
                    detail["costs"], np.asarray(detail["positions_fine_index"]), t,
                    fine.cell_size, np, candidate_weights,
                )
                metrics[key].append(detail[key])
        summaries[str(k)] = {
            key: {name: _summary([row[name] for row in rows], np)
                  for name in ("ess", "max_weight", "unique_hypotheses", "unique_location_ess",
                               "unique_location_max_weight", "positive_weight_count")}
            for key, rows in metrics.items()
        }
        summaries[str(k)]["best_cost"] = _summary([event["per_k"][str(k)]["best_cost"] for event in events], np)
        summaries[str(k)]["shortlist_shortfall_events"] = sum(
            event["per_k"][str(k)]["retained_count"] < k for event in events)
        if summaries[str(k)]["shortlist_shortfall_events"]:
            warnings.append(f"K{k}: fewer than K native candidates for some events; all available retained")
        if k > 1 and any(row["unique_hypotheses"] < event["per_k"][str(k)]["retained_count"]
                         for row, event in zip(metrics["at_temperature"], events)):
            warnings.append(f"K{k}: hypotheses within <1 m merge; label ESS is not spatial diversity")
        if k > 1 and any(row["positive_weight_count"] < event["per_k"][str(k)]["retained_count"]
                         for row, event in zip(metrics["at_temperature"], events)):
            warnings.append(f"K{k}: native exponential weights underflow to zero for some retained hypotheses")
    if prefix_failures:
        warnings.append("Native K shortlists are not all K15 prefixes; independently selected cells were refined")
    if selection["saturated"]:
        warnings.append("T=16 upper bound reached without attaining median K15 ESS=3")
    cases = [{"candidate_mode": "hard", "n_candidates": k, "weights_top_n": 1, "temperature": 1.0}
             for k in _KS]
    cases += [{"candidate_mode": "soft", "n_candidates": k, "weights_top_n": k, "temperature": temperature}
              for k in (5, 10, 15)]
    result = {
        "schema_version": 1, "temperature": temperature, "selection": selection,
        "event_count": len(observations), "station_count": observations.shape[1],
        "sample_count": len(indices), "event_indices": indices.tolist(),
        "sampling": "evenly spaced observation-row indices, inclusive endpoints, at most 128",
        "model_source": "prepared.initial_model only; no updates",
        "subdivision": int(config.subdivision), "slowness_interpolation": config.slowness_interpolation,
        "weights_min_distance": int(config.weights_min_distance),
        "fine_cell_size_m": float(fine.cell_size), "field_shape": list(times.shape),
        "noise": {name: float(getattr(noise, name)) for name in
                  ("relative_sigma", "absolute_sigma_s", "model_sigma_s")},
        "shortlist_prefix_verified": not prefix_failures, "prefix_failures": prefix_failures,
        "per_k": summaries, "events": events, "cases": cases, "warnings": warnings,
        "unique_hypotheses_definition": "connected components of refined positions with pair distance strictly <1 m; transitive, weights summed per component",
        "limitations": [
            "T=1 is the proper likelihood; calibration only flattens, never sharpens",
            "ESS measures weight spread, not location accuracy or independent spatial modes",
            "Initial-model observation-only heuristic; no true events/reference model or regularization tuning",
            "Soft retains all K candidates (or all available); native floating-point weights may underflow",
        ],
    }
    # Reject NaN/Infinity and numpy scalars before either returning or writing.
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    result = json.loads(text)
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calibration.json").write_text(text, encoding="utf-8")
    return result
