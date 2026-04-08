"""
tomography.py — public API facade.

Implementation is split across:
  - tomography_em.py        (EM loop)
  - tomography_events.py    (event processing + multiprocessing)
  - tomography_logging.py   (run logger)
  - tomography_math.py      (math helpers)
"""
from __future__ import annotations

from tomography_em import make_tomography_step, run_em, warm_up_jit
from tomography_logging import TomographyLogger
from tomography_math import _calculate_G

__all__ = [
    "TomographyLogger",
    "make_tomography_step",
    "run_em",
    "warm_up_jit",
    "_calculate_G",
]
