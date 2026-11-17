"""One-model, training-only localization reuse between control and the next E-step."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def _fingerprint(model, training, stations, noise, config):
    digest = hashlib.sha256()
    for value in (model.velocity, training, stations):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    # Weights, temperature and worker count do not affect the candidate search.
    digest.update(repr((
        model.cell_size, config.subdivision, config.slowness_interpolation,
        config.n_candidates, config.weights_min_distance, config.fast_refinement,
        noise.relative_sigma, noise.absolute_sigma_s, noise.model_sigma_s,
    )).encode())
    return digest.digest()


@dataclass(frozen=True)
class RefinedCandidates:
    """All candidates in original shortlist order, before top-N or weighting."""

    cells: np.ndarray
    positions: np.ndarray
    costs: np.ndarray

    def __post_init__(self):
        for value in (self.cells, self.positions, self.costs):
            value.flags.writeable = False


@dataclass(frozen=True)
class LocalizationCache:
    """Ephemeral cache; never reuse across a model, geometry, noise or pick change."""

    fingerprint: bytes
    times: np.ndarray
    candidates: dict[int, RefinedCandidates]

    @classmethod
    def create(cls, model, training, stations, noise, config, times, candidates):
        times.flags.writeable = False
        return cls(_fingerprint(model, training, stations, noise, config), times, candidates)

    def validate(self, model, training, stations, noise, config):
        if self.fingerprint != _fingerprint(model, training, stations, noise, config):
            raise ValueError("Localization cache does not match model/training/geometry/search settings")
