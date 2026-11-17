from copy import deepcopy

import pytest

from export_residual_pairs import find_pairs, metric_columns, pairing_key


def test_hard_control_can_be_reused_and_unpaired_runs_stay_visible():
    rows = [{"run_id": r, "mode": mode} for r, mode in
            (("h", "hard"), ("s1", "soft"), ("s2", "soft"), ("orphan", "soft"))]
    keys = {"h": "same", "s1": "same", "s2": "same", "orphan": "different"}
    pairs, unpaired = find_pairs(rows, keys, {"s2": "h"})
    assert [(h["run_id"], s["run_id"]) for h, s in pairs] == [("h", "s1"), ("h", "s2")]
    assert [r["run_id"] for r in unpaired] == ["orphan"]


def test_ambiguous_or_incompatible_controls_are_not_guessed():
    rows = [{"run_id": r, "mode": "soft" if r == "s" else "hard"} for r in ("h1", "h2", "s")]
    keys = dict.fromkeys(("h1", "h2", "s"), "same")
    with pytest.raises(ValueError, match="Ambiguous"):
        find_pairs(rows, keys, {})
    pairs, _ = find_pairs(rows, keys, {"s": "h2"})
    assert pairs[0][0]["run_id"] == "h2"
    keys["h2"] = "different"
    with pytest.raises(ValueError, match="not comparable"):
        find_pairs(rows, keys, {"s": "h2"})


def test_pairing_validates_science_and_inputs_but_allows_soft_temperature():
    row = {"section": "5"}
    meta = {"run_params": {"n_cycles": 12, "n_candidates": 50, "temperature": 1.,
                           "candidate_mode": "hard", "run_name": "name", "n_workers": 12},
            "run_version": "1.4", "source_experiment": {"id": "dataset", "study_manifest": "cohort"},
            "station_locs": [[1, 2, 3]]}
    expected = pairing_key(row, meta, "initial", None)
    other = deepcopy(meta)
    other["run_params"].update(temperature=8., temperature_schedule=[8., 1.],
                                candidate_mode="soft", run_name="other", n_workers=24)
    assert pairing_key(row, other, "initial", None) == expected
    for field, value in (("n_cycles", 24), ("n_candidates", 60), ("control_fraction", .1)):
        changed = deepcopy(other)
        changed["run_params"][field] = value
        assert pairing_key(row, changed, "initial", None) != expected
    assert pairing_key(row, other, "different-initial", None) != expected
    assert pairing_key(row, other, "initial", "different-mask") != expected
    other["source_experiment"]["study_manifest"] = "other-cohort"
    assert pairing_key(row, other, "initial", None) != expected


def test_independent_pooled_and_viewer_minima_and_missing_values():
    row = {"run_id": "soft", "final_model_exists": True}
    points = [{"iteration": 0, "rmse_s": .1, "mean_event_rms_s": .08, "n_pairs": 12},
              {"iteration": 1, "rmse_s": .09, "mean_event_rms_s": .085, "n_pairs": 12}]
    record = {"best": points[1], "trajectory": points, "status": "ok", "n_complete_iterations": 2}
    columns = metric_columns("soft", row, record)
    assert columns["soft_min_residual_rmse_s"] == .09
    assert columns["soft_best_iteration_index"] == 1
    assert columns["soft_best_iteration_number"] == 2
    assert columns["soft_min_mean_event_rms_s"] == .08
    assert columns["soft_best_mean_event_iteration_index"] == 0
    record.update(best=None, trajectory=[], status="absent", reason="zero_complete_iterations",
                  n_complete_iterations=0)
    missing = metric_columns("soft", row, record)
    assert missing["soft_min_residual_rmse_s"] is None
    assert missing["soft_best_iteration_number"] is None
    assert missing["soft_metric_missing_reason"] == "zero_complete_iterations"
