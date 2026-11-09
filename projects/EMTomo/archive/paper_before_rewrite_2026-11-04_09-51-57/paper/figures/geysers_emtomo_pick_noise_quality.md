# Geysers EMTomo pick-noise sensitivity

Two plots with three lines each, all six noise levels including the original baseline.
Final update 12 only; native piecewise-constant reconstruction on 125-m truth cells.
Volume: RMSE over the entire physical domain. Anomalies: compute RMSE separately inside
each of the six sphere supports, then take their arithmetic mean (not volume-weighted).
Lower is better for both plots. Relative and absolute noise increase together; the
top and bottom axes describe the same noise levels, not independent experiments.
All inversion settings, including assumed noise, remain fixed. One paired seed,
no replicate-based uncertainty estimates. See CSV/JSON for all scores and provenance.

Rebuild: `python -B paper/make_geysers_noise_quality.py`. No solves are launched.
