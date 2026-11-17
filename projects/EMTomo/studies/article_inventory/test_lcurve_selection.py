import hashlib
import json

import numpy as np
import pytest

from lcurve_selection import choose_elbow, select_run, selected_quality


def _points(residual, norm):
    return [dict(residual_rmse_s=r, relative_slowness_change=n)
            for r, n in zip(residual, norm)]


def test_maximum_geometric_distance_selects_interior_corner():
    points = _points([100., 10., 1.], [1., 1.1, 100.])
    index, reason, distances = choose_elbow(points)
    assert index == 1
    assert reason == "maximum_log_chord_distance"
    assert distances[1] > 0 and distances[0] == distances[-1] == 0


def test_log_normalization_is_invariant_to_units():
    points = _points([100., 20., 5., 1.], [1., 1.1, 2., 100.])
    changed = [{**p, "residual_rmse_s": p["residual_rmse_s"] * .001,
                "relative_slowness_change": p["relative_slowness_change"] * 10} for p in points]
    before, reason, distances = choose_elbow(points)
    after, other_reason, other_distances = choose_elbow(changed)
    assert before == after and reason == other_reason
    np.testing.assert_allclose(distances, other_distances, atol=1e-15)


@pytest.mark.parametrize("residual,norm,reason", [
    ([10., 1.], [1., 10.], "insufficient_points_use_last"),
    ([1., 1., 1.], [1., 2., 3.], "degenerate_curve_use_last"),
    ([100., 10., 1.], [1., 10., 100.], "straight_curve_use_last"),
])
def test_explicit_fallbacks(residual, norm, reason):
    index, actual_reason, _ = choose_elbow(_points(residual, norm))
    assert index == len(residual) - 1 and actual_reason == reason


@pytest.mark.parametrize("points", [[], _points([0., 1.], [1., 2.]), _points([1.], [np.nan])])
def test_invalid_coordinates_rejected(points):
    with pytest.raises(ValueError):
        choose_elbow(points)


def test_model_and_quality_timing_and_truth_independence(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    np.save(tmp_path / "initial_model.npy", np.full((2, 2, 2), 5000.))
    trajectory = []
    for k in range(3):
        directory = tmp_path / f"iter_{k}"
        directory.mkdir()
        (directory / "complete.json").write_text(json.dumps({"iter": k}))
        np.save(directory / "model.npy", np.full((2, 2, 2), 5000. - 100. * k))
        trajectory.append(dict(iteration=k, rmse_s=.1 / (k + 1)))
    record = dict(status="ok", trajectory=trajectory,
                  signature=dict(meta_sha256=hashlib.sha256(b"{}").hexdigest(), sha256="test"))
    quality_path = tmp_path / "quality.jsonl"
    quality_path.write_text('\n'.join(json.dumps(p) for p in [
        {"iter": 0, "avg_abs_pct_dev": .001}, {"iter": 1, "avg_abs_pct_dev": 42.},
        {"iter": 2, "avg_abs_pct_dev": 999.},
    ]))
    selection = select_run(tmp_path, record)
    assert selection["selected"]["residual_iteration_index"] == 2
    assert selection["selected"]["completed_velocity_updates"] == 2
    assert selection["skipped_zero_norm_or_initial_indices"] == [0]
    assert selected_quality(tmp_path, selection) == (42., 1)  # Not quality iter_2 or its minimum.
    quality_path.unlink()
    assert select_run(tmp_path, record) == selection  # Selection never needs truth quality.
