"""Reproducible observation errors, separate from the noiseless forward solver."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    relative_sigma: float = 0.01
    absolute_sigma_s: float = 0.05
    seed: int = 42

    def __post_init__(self):
        for name in ("relative_sigma", "absolute_sigma_s"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and nonnegative")
            try:
                value = float(value)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{name} must be finite and nonnegative") from error
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))


def add_arrival_noise(travel_times, event_ids, station_ids, config=NoiseConfig()):
    """Add two independent Gaussian errors to absolute propagation times.

    Each pair has sigma sqrt((relative_sigma*T)**2 + absolute_sigma_s**2).
    Pair-ID keyed PCG64 streams preserve a realization under reordering/subsetting.
    Negative noisy picks are allowed: these are timing errors, not propagation
    times. Do not clip them; normalize to the earliest noisy pick afterwards.
    """
    times = np.array(travel_times, dtype=np.float64, copy=True)
    event_ids, station_ids = tuple(map(str, event_ids)), tuple(map(str, station_ids))
    if (not event_ids or not station_ids or len(set(event_ids)) != len(event_ids)
            or len(set(station_ids)) != len(station_ids)
            or any(not identifier.strip() for identifier in (*event_ids, *station_ids))):
        raise ValueError("Event and station IDs must be nonempty and unique")
    if times.shape != (len(event_ids), len(station_ids)):
        raise ValueError("travel_times shape must match event and station IDs")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("travel_times must be finite and nonnegative")
    if config.relative_sigma == 0 and config.absolute_sigma_s == 0:
        return times
    for e, event_id in enumerate(event_ids):
        for s, station_id in enumerate(station_ids):
            key = json.dumps([config.seed, event_id, station_id], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(key).digest(), "little")
            rng = np.random.Generator(np.random.PCG64(seed))
            propagation, picking = rng.standard_normal(2)
            times[e, s] += config.relative_sigma * times[e, s] * propagation + config.absolute_sigma_s * picking
    if not np.all(np.isfinite(times)):
        raise ValueError("Noise produced nonfinite arrival times")
    return times
