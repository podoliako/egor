# Discussion: soft gain versus top-N, article-table pairs only

Rebuild from the EMTomo root:

```sh
python -B paper/figures/results/discussion/source/make_soft_gain_vs_top_n.py
```

`../png/soft_gain_vs_top_n_lcurve.png` is the PNG preview; this directory holds the matching
PDF, the script and the provenance JSON (one record per point, including run IDs, updates,
`plotted_x` and the table-RMSE gain for cross-checking).

Points: exactly the 28 unique hard/soft pairs of the article tables in Sections 4.2--4.6
(`paper/figures/final_comparison_metrics.json -> tables`; the shared nominal-noise pair in 4.4
is drawn once). This includes the homogeneous-start complex-anomaly pair from the 4.6 table;
both complex-anomaly pairs are drawn as ordinary filled points of one series.
The real-geometry checkerboard series (4.4) uses a muted green (#6BA56F); its noise level is not encoded.

Metric: viewer `avg_abs_pct_dev`, gain = 100 (E_hard - E_soft) / E_hard; positive favours soft.
Checkpoints follow the article policy:

- 4.2--4.4 (one-sided, two clusters, real-geometry checkerboard): L-curve-selected V_k from
  `paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv` (all k = 3);
- 4.5--4.6 (fault, complex anomalies): final model V_N from each run's `quality.jsonl`
  (fault 12 updates; complex 9 for the 1-D start and 20 for the homogeneous start).

For every pair, the gain sign matches the sign of the velocity-RMSE gain in the article tables
(`sign_differs_from_table_rmse_gain` is empty in the JSON).

Layout: no title, subtitle or footer; no noise colorbar or dmin call-outs; log x axis; series that
share a top-N value are dodged horizontally by a factor of 1.045. Points within one series are not
jittered, so identical values coincide (e.g. the five noise levels at top-N = 50).

### Companion: hypocentre P90 gain (`../png/soft_p90_gain_vs_top_n.png`)

Written by the same script, with the same 28 pairs, x positions, colours and layout. The metric is
the hypocentre P90 from the article tables (largest-weight candidate in both modes), with
gain = 100 (P90_hard - P90_soft) / P90_hard; positive favours soft. Timing follows the tables:
for 4.2--4.4, the E-step in the L-curve-selected V_k; for 4.5--4.6, the last E-step in V_(N-1),
before the final velocity update. Per-point values are the `hard_p90_m`, `soft_p90_m` and
`p90_gain_percent` fields of `soft_gain_vs_top_n_lcurve.json`. The PDF is `soft_p90_gain_vs_top_n.pdf`.

## Velocity gain versus hypocentre gain

Rebuild: `python3 -B paper/figures/results/discussion/source/make_soft_velocity_gain_vs_hypocenter_gain.py`.

Outputs: `../png/soft_velocity_gain_vs_hypocenter_gain.png`, matching PDF and JSON here.
Uses all 28 unchanged points from `soft_gain_vs_top_n_lcurve.json`: X is relative
reduction in hypocentre P90, Y is relative reduction in velocity `avg_abs_pct_dev`.
Both gains are `100 * (hard - soft) / hard`; positive favours Soft. Series colours
and checkpoint policies are unchanged. Linear axes, no jitter, fitted trend or
coordinate offsets. Zero lines separate improvement/degradation quadrants.
The JSON preserves source hash, metrics, checkpoint timing and all pair records.
For 4.5--4.6, locations precede the final velocity update, as in the source plots.

## Weighted-mean counterpart: velocity gain versus hypocentre gain

```sh
OPENBLAS_NUM_THREADS=1 python3 -B paper/figures/results/discussion/source/make_soft_velocity_gain_vs_weighted_mean_hypocenter_gain.py
```

New outputs (original figures and article unchanged):

- `../png/soft_velocity_gain_vs_weighted_mean_hypocenter_gain.png`
- `soft_velocity_gain_vs_weighted_mean_hypocenter_gain.pdf`
- `soft_velocity_gain_vs_weighted_mean_hypocenter_gain.json`

All 28 original pair IDs, their order, and velocity Y values are preserved exactly.
For each event, convert saved refined candidate positions to metres as
`x_j = (positions_j + 0.5) * fine_cell_size`, then compute
`mean = sum(weight_values_j * x_j) / sum(weight_values_j)` using **all** saved
weights, without thresholding, new top-N selection, re-tempering, or recomputing
likelihoods. P90 is NumPy's linear 90th percentile of the equal-event-weight 3-D
Euclidean errors against the synthetic truth. X is `100 * (P90_hard - P90_soft) / P90_hard`.
The JSON retains original maximum-weight P90 fields and adds explicitly named
weighted-mean fields; `plotted_x` is replaced with the new gain and `plotted_y`
is the unchanged velocity gain. Colours/layout follow the original scatter plot.

**Hard semantics:** `tomography/tomography_events.py::process_event` selects top-N,
calculates candidate weights, then replaces them with a one-hot distribution in
hard mode. `tomography_logging.py` saves those post-mode weights alongside candidate
positions, including zero-weight candidates. Every hard event was verified to have
exactly one positive weight; its weighted mean and maximum-weight location agree
exactly. Averaging all hard candidate positions uniformly would be incorrect.
Soft uses the full original retained distribution, not discarded search candidates.

**Checkpoint and truth validation:** Sections 4.2–4.4 use `iter_3` relocation in
selected V3 (frozen CSV and original metric records checked). Section 4.5 uses
`iter_11` in V11; Section 4.6 uses `iter_8` in V8 for the 1-D start and `iter_19`
in V19 for the homogeneous start. These last E-steps precede the final velocity
updates to V12, V9 and V20 respectively. No post-final relocation is substituted.
The extraction verifies complete checkpoints, quality-log velocity values, exact
event-directory sets, source/forward input hashes, unique event/station IDs, full
arrival ID cross-products, station geometry, and original truth hashes. Event_i
maps to **CSV row i**, not sorted event ID, following `experiment_data.py`.
Legacy run metadata does not independently store event IDs: this remains a
provenance caveat, not an independently recoverable ID match. The ordered
`row_index / event_id / weights-file-SHA256` digest must match the original metric
report, and every original maximum-weight P90 must reproduce before plotting.

Validation passed for all 28 pairs, 48 unique run/checkpoints and 139,894 event
records. All weights are finite/nonnegative; the largest sum deviation from one
is 8.88e-16. Candidate counts and grids are checked. JSON includes input/code
SHA256 hashes, per-run checkpoint paths, ordered weight digests, count/sum ranges,
old/new P90 values, gain changes and validation summary. Missing or inconsistent
inputs raise an error rather than trigger a fallback. No solver code is executed.

Observed changes: soft has lower P90 in **8/28** pairs versus **11/28** with the
maximum-weight estimator. Seven signs change: `section2_pair01` and
`section2_topnsweep_k100_top100` now favour soft; `section5_pair22` and the four
`section5_topnsweep_k128_top{50,75,100,128}` pairs now favour hard. All four
one-sided pairs favour soft (8.91–15.09%); the two-cluster gains remain negative
but improve to −5.67 to −5.39%. Overall weighted-mean gains span −24.71 to +15.09%.
Hard P90 is unchanged everywhere; velocity gains are unchanged everywhere.

Interpretation caveats: means can fall between separated modes; this is a point
estimator comparison, not posterior uncertainty. Shared hard controls are not
independent replicates; the experiments are single synthetic realizations.
Synthetic catalogue truth and the original fault vertical-datum caveat are
inherited from the metric reports and recorded in the JSON.

## Section 4.3: initial and final saved hypocenter-error densities

```sh
python -B paper/figures/results/discussion/source/make_section43_location_errors.py
```

Output: `../png/section43_hypocenter_error_initial_final.png`, plus matching PDF, per-event CSV, density arrays and provenance JSON in this directory.

The fixed comparison is `section3_pair01`: 50 tested candidates, hard retains one and soft retains all 50, seed separation seven fine cells. Standard noise is 1% relative plus 0.05 s absolute. All 600 events have equal weight. Both modes use the maximum-weight candidate as their point location.

Two panels show `iter_0` and `iter_11`: the first location step in the initial V0 model and the last saved location step in V11, **before the twelfth velocity update**. This is not post-final relocation in V12, and not the V3 checkpoint used in the 4.3 article table. Initial hard/soft point locations are exactly identical; the initial curves therefore overlap exactly.

Densities use Gaussian KDE with the same fixed 150-m bandwidth for all four curves and reflection at zero. Numerical integration over the common plotted range is checked against one. The figure displays smoothed densities of point-location errors, not hypocenter posterior distributions. P90 labels are computed from the unsmoothed event errors. No solves or article edits are performed.

## Section 4.5: 128 candidates, min_dist = 7

```sh
OPENBLAS_NUM_THREADS=1 python -B paper/figures/results/discussion/source/make_section45_location_errors.py
```

Output: `../png/section45_k128_dmin7_hypocenter_error_initial_final.png`, plus matching PDF, per-event CSV, density arrays and provenance JSON here. The wrapper uses the shared renderer in `make_section43_location_errors.py`.

Pair: `section5_dminsweep_k128_top128_dmin7`; 3776 events, 26 stations, 128 tested candidates, soft retains all 128, and minimum seed separation seven fine cells (875 m). This run uses 2% relative plus 0.02 s absolute pick noise, unlike the Section 4.3 example.

The estimator, KDE and checkpoints are the same as above: maximum-weight point location, reflected 150-m-bandwidth densities, and initial `iter_0` versus last saved `iter_11` E-step before velocity update 12. Initial hard/soft locations coincide exactly. Raw P90 is initially 468.988 m in both modes; finally 463.818 m for hard and 446.624 m for soft. All density normalization checks pass. No article edits or new solves.
