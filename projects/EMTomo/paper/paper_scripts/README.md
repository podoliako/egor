# Paper scripts

The Python files live in `projects/EMTomo/paper/paper_scripts`. Run commands
from `projects/EMTomo`, for example:

```sh
python3 -B -m pytest -p no:cacheprovider paper/paper_scripts/test_evaluate_metrics.py
python3 -B paper/paper_scripts/evaluate_metrics.py --check
python3 -B paper/paper_scripts/evaluate_spherical_anomalies.py --check
```

The last two commands recompute metrics from saved artifacts; they are not unit
tests and require the original data and benchmark environment.

Paths are anchored to `__file__`, independently of the working directory:

- `Path(__file__).resolve().parents[2]` is the EMTomo project (`ROOT`).
- `Path(__file__).resolve().parents[1]` is `EMTomo/paper` (`PAPER`).
- Figure outputs remain in `EMTomo/paper/figures`; supplemental metrics remain
  in `EMTomo/paper/supplemental_metrics.json`.
- Sibling imports remain local to this directory. The noise-quality scripts
  also use `EMTomo/studies/geysers_pick_noise_20261027/common.py`; the faulted-layer
  oracle resolves project modules through `ROOT` and the repository root.

## Provenance after relocation

Evaluator source hashes must use files in this directory, not `PAPER / name`.
`evaluate_spherical_anomalies.py` locates its sibling sources with `with_name`,
and records their paths relative to EMTomo as `paper/paper_scripts/<name>.py`.

Existing figure/metrics JSON reports retain historical source paths and hashes.
For example, `spherical_anomalies_comparison.json` refers to the former
`paper/make_spherical_anomalies_comparison.py`. Moving and editing a source file
makes that recorded path/hash stale: it must not be silently redirected or
accepted without validation. Existing provenance artifacts were not rewritten,
and figure/metric generation was not run as part of the relocation fix.
Revalidation of those artifacts requires a separate, explicit regeneration and
provenance review; `--check` may fail on the historical source path/hash until then.

## Results tables and real-data structure split

```sh
python3 -B paper/paper_scripts/make_table_supplementary_metrics.py   # ~10 min, mostly 4.1 MAPE
python3 -B paper/paper_scripts/make_results_tables.py                # rewrites the 8 tables in results.tex
python3 -B paper/paper_scripts/make_geysers_real_structure_by_x.py
```

- `make_table_supplementary_metrics.py` writes `paper/figures/table_supplementary_metrics.json`:
  for every hard/soft table row (4.2--4.6, article checkpoints) it recomputes velocity RMSE and
  hypocentre P90 and fails if they differ from `hard_soft_lcurve_comparison_metrics.json` /
  `final_comparison_metrics.json`, then adds MAPE on the same nested grid, residual RMS in seconds
  (largest-weight hypothesis, per-event mean removed, from saved `residuals.npy`) and the P90 of the
  weighted-mean candidate. It also adds 4.1 MAPE for EMTomo, tomoDD and the trilinear oracle on the
  31.25-m grid, checking RMSE against `checkerboard_velocity_metrics.json`.
- `make_results_tables.py` regenerates all metric tables of `results.tex` (matched by `\label`) from
  that JSON; 4.1 velocity and location tables are merged into `tab:checkerboard_metrics`.
- `make_geysers_real_structure_by_x.py` writes `paper/exp_data/geysers_real_structure_by_x.json`:
  the Guo--Thurber (2021) comparison of `make_geysers_real_structure_metrics.py`, split at the
  midpoint of the masked X range and in 2-km X bins, with lateral-anomaly (per-depth demeaned)
  correlations. Used in the Discussion.
