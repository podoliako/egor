"""Hard and soft updates must use the same refined shortlist and noise model."""

import numpy as np
import pytest

import tomography.tomography_events as events


def test_hard_and_soft_modes_use_identical_refined_candidates(monkeypatch):
    candidates = np.array([[0, 0, 0], [1, 0, 0]])
    monkeypatch.setattr(events, "_select_top_n_cells_by_misfit", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        events, "_refine_epicenter_in_cell",
        lambda sf, observed, cell: (np.asarray(cell, dtype=float), 0.0),
    )
    calls = []

    def rays(gx, gy, gz, sl, epic, *args):
        calls.append(tuple(epic))
        G = np.zeros((2, 2, 1, 1))
        G[0, int(epic[0]), 0, 0] = 1.0
        return G, np.array([True, True])

    sf = np.array([[[[1.]], [[1.]]], [[[2.]], [[2.2]]]])
    common = dict(
        observed=np.array([0., 1.]), sf=sf,
        gx=np.zeros_like(sf), gy=np.zeros_like(sf), gz=np.zeros_like(sf),
        sl=np.zeros((2, 3)), x_lo=np.zeros(3), x_hi=np.array([1., 0., 0.]),
        fine_cell_size=1., subdivision=1, slowness_interpolation="nearest",
        temperature=1., weights_top_n=2, weights_min_distance=1,
        weight_noise_relative_sigma=0., weight_noise_absolute_sigma_s=0.1,
        weight_model_sigma_s=0., compute_G=rays, log_G_per_weight=False,
        log_misfit=False,
    )
    soft_h, soft_b, soft_log = events._process_event(**common, candidate_mode="soft")
    assert calls == [(0., 0., 0.), (1., 0., 0.)]
    assert np.all(soft_log[2] > 0)
    assert sum(soft_log[2]) == pytest.approx(1.)
    calls.clear()
    hard_h, hard_b, hard_log = events._process_event(**common, candidate_mode="hard")
    assert calls == [(0., 0., 0.)]
    np.testing.assert_array_equal(hard_log[2], [1., 0.])
    np.testing.assert_array_equal(hard_log[1], soft_log[1])
    assert hard_h[0, 0] > soft_h[0, 0]
    assert hard_b.shape == soft_b.shape


def test_invalid_candidate_mode_rejected_before_tracing():
    with pytest.raises(ValueError, match="candidate_mode"):
        events._process_event(candidate_mode="invalid", **{
            "observed": np.array([0., 1.]), "sf": np.zeros((2, 1, 1, 1)),
            "gx": None, "gy": None, "gz": None, "sl": None,
            "x_lo": None, "x_hi": None, "fine_cell_size": 1.,
            "subdivision": 1, "slowness_interpolation": "nearest",
            "temperature": 1., "weights_top_n": 1, "weights_min_distance": 1,
            "weight_noise_relative_sigma": 0., "weight_noise_absolute_sigma_s": .1,
            "weight_model_sigma_s": .2, "compute_G": None,
            "log_G_per_weight": False, "log_misfit": False,
        })
