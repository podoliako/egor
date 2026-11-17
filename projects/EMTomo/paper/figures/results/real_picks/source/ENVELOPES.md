# 4.7 — Expanded-envelope change reverted

Expanded masks are intended for **4.5 and 4.6, not 4.7**. The temporary
expanded-envelope change in `make_hard_soft.py` has been reverted. The current
`real_picks_hard_soft_sensitivity*` outputs display the exact raw intersection:

```python
mask = (hard_Hjj / hard_alpha >= 1) & (soft_Hjj / soft_alpha >= 1)
```

Here `alpha = lambda_reg * mean(Hjj)` is each run's actual scalar damping.
There is **no convex hull, padding, hole filling or component removal**. Both
panels use this same raw mask; NPZ `xy_shared_mask` / `xz_shared_mask` are the
exact intersections, not envelope aliases. Obsolete envelope/display/added-mask
metadata and the envelope-helper dependency/hash have been removed.

Shared areas are **57.75 km² (XY)** and **39.25 km² (XZ)** on the 100 m plotting
grid. This heuristic diagonal-sensitivity selection is **not spatial resolution,
recovery or probability**. Diagnostics remain from `iter_2/model.npy`, before
the last update, whereas displayed velocities are final; that update changes
velocities by up to 2%. Sensitivity has not been recomputed in the final model.

See [README.md](README.md) for the current construction, threshold-dependence
areas, limitations, outputs and rebuild command. Only the Hard/Soft sensitivity
output family is regenerated; reference comparisons, events/hypocentre outputs,
and 4.5/4.6 code and outputs are unchanged by this rollback.
