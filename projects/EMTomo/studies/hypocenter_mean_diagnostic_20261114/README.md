# Fixed-weight hypocenter mean diagnostic — 2026-11-14

## Scope and verification

Read-only analysis of the current article's saved artifacts: sections 4.2–4.5 from `paper/figures/hard_soft_lcurve_comparison_metrics.json`, and **PRIMARY section6_pair01 / section6_pair05 only** from `heterogeneous_lcurve_comparison_metrics.json` for section 4.6. No article files, existing outputs, velocities, temperatures, or selections were changed. No FMM, inversion, or forward solve was run.

- 28 distinct pairs, 48 unique selected runs, 139,894 run-event records; extraction **76.70 seconds**, protected by a 300-second alarm.
- Table memberships: onesided 4, two clusters 4, noise 5, candidates 5, fault retention 5, fault separation 4, heterogeneous 2. `section4_pair07` is shared between noise/candidates, counted once in the total (29 memberships, 28 pairs).
- Every selected run's MAP P90 reproduced with **absolute difference 0.0 m** (required <= 1e-8 m); table P90 and selected k cross-checked too.
- All 48 `row_id_weights_sha256` digests matched the published reports **exactly**. Protocol: SHA256 of UTF-8 `row_index TAB event_id TAB SHA256(weights.npz) NEWLINE`, in source row order.
- All source input hashes listed in report forward metadata matched; event IDs were unique, event counts matched, and event order was verified against published order digests or heterogeneous explicit ID list. `event_i` means source `events.csv` row i, never sorted ID. Legacy weights do not independently store event IDs: this retains the published loader-protocol caveat.
- All weights finite/nonnegative with sum within absolute 1e-6 of one; mean explicitly divides by the saved sum. Grid/count/bounds checked. **Every hard weighted mean equals its MAP exactly**, event by event.

Each mode keeps its own published L-curve k. Coordinates are `(positions + 0.5) * fine_cell_size`, in metres, from **iter_k**, not iter_(k-1). MAP uses the first maximum on ties. All-event metrics use equal event weights and linear percentiles. Spatial RMSE is `sqrt(mean(||estimate-truth||^2))`, not mean distance and not a componentwise RMSE.

## Main effects by series

Here Δ means soft mean minus soft MAP, averaged equally over selected pairs (m); negative is better. Counts compare soft to its matched **hard MAP**. Series averages are descriptive, not pooled-event effects or independent-replicate estimates.

| Series | Pairs | Average ΔP90 | Average Δspatial RMSE | Soft better hard: P90 MAP→mean | Soft better hard: RMSE MAP→mean |
|---|---:|---:|---:|---:|---:|
| One-sided | 4 | -815.17 | -438.00 | 2→4 | 1→4 |
| Two clusters | 4 | -53.83 | -14.79 | 0→0 | 0→0 |
| Real-geometry noise | 5 | +0.89 | -3.31 | 0→0 | 0→0 |
| Real-geometry candidates | 5 | +9.48 | -5.43 | 0→0 | 0→0 |
| Fault retention, dmin2 | 5 | -4.16 | -3.52 | 5→5 | 1→5 |
| Fault separation, dmin4/7 | 4 | -5.71 | -3.53 | 4→4 | 4→4 |
| Heterogeneous PRIMARY | 2 | -61.00 | -26.25 | 0→0 | 0→1 |
| **All, shared pair deduplicated** | **28** | **-128.55** | **-69.05** | **11→13** | **6→14** |

Mean improves soft MAP P90 in **20/28** pairs and spatial RMSE in **27/28**. Average pairwise soft gain relative to hard, `100*(hard-soft)/hard`, changes from **-5.58% to -3.10% for P90** and **-4.96% to -2.38% for spatial RMSE**. These across-series averages should not be interpreted as a universal or statistically estimated effect.

## Representative triples

All errors/shifts in metres. Each triple is **hard MAP / soft MAP / soft mean**. All tested/retained counts and frozen iterations are in `pairs.csv`.

| Case | P90 triple | Spatial RMSE triple | Events better/worse, mean vs MAP | Average shift |
|---|---|---|---:|---:|
| One-sided top50 | 6160.84 / 6222.32 / 5247.99 | 3738.91 / 3795.53 / 3302.98 | 60.8% / 39.2% | 1354.33 |
| One-sided top100 | 6160.84 / 6390.29 / 5231.28 | 3738.92 / 3880.54 / 3241.52 | 62.0% / 38.0% | 1642.33 |
| Two clusters top50 | 1340.36 / 1469.07 / 1414.66 | 911.68 / 981.34 / 966.58 | 53.2% / 46.8% | 227.14 |
| Real-geometry nominal noise/top50 (shared pair07) | 665.28 / 755.91 / 765.64 | 434.24 / 483.85 / 478.23 | 48.9% / 51.1% | 61.98 |
| Real-geometry low-noise pair01 | 378.64 / 448.54 / 472.18 | 245.57 / 280.70 / 288.81 | 44.7% / 55.3% | 54.38 |
| Fault dmin2, top50 | 460.37 / 453.57 / 448.69 | 297.50 / 297.93 / 294.12 | 56.1% / 43.9% | 40.91 |
| Fault dmin2, top128 | 460.37 / 454.63 / 452.05 | 297.50 / 298.52 / 295.51 | 56.0% / 44.0% | 41.99 |
| Fault dmin4, top50 | 459.03 / 449.43 / 440.18 | 296.28 / 293.14 / 287.37 | 58.2% / 41.8% | 33.71 |
| Fault dmin4, top128 | 459.03 / 449.43 / 440.04 | 296.28 / 293.17 / 287.37 | 58.2% / 41.8% | 33.72 |
| Fault dmin7, top50 | 457.94 / 450.83 / 448.73 | 297.37 / 292.91 / 291.63 | 67.5% / 32.5% | 7.07 |
| Fault dmin7, top128 | 457.94 / 450.83 / 448.73 | 297.37 / 292.91 / 291.63 | 67.5% / 32.5% | 7.07 |
| Heterogeneous supplied 1D, pair01 | 1698.62 / 1706.76 / 1734.51 | 998.42 / 1000.77 / 998.19 | 42.0% / 58.0% | 42.12 |
| Heterogeneous homogeneous, pair05 | 2883.64 / 3121.93 / 2972.18 | 1980.60 / 2096.17 / 2046.24 | 65.3% / 34.7% | 152.45 |

## Interpretation: squared loss is not high-quantile risk

Replacing MAP with the mean **can materially change hypocenter conclusions**, especially for the broad one-sided distributions: all four pairs then beat hard in both P90 and spatial RMSE. It does **not** generally overturn soft-versus-hard rankings: clusters, real-geometry noise/candidate sweeps, and heterogeneous starts still do not beat hard in P90. The fault pairs retain their P90 advantage; all five dmin2 retention pairs gain a spatial-RMSE advantage after averaging.

The candidate-count series is a direct counterexample to treating the estimators as interchangeable: **all five means lower spatial RMSE while all five worsen P90**. The heterogeneous supplied-1D case has the same conflict: P90 worsens by 27.75 m, while spatial RMSE improves by 2.57 m, enough to narrowly beat hard by about 0.22 m. That marginal advantage is not evidence of statistical significance. The lowest-noise real-geometry case worsens in both metrics.

The coordinate mean minimizes **posterior expected squared coordinate error under the saved discrete weights**. It does not guarantee lower realized error against synthetic truth, a lower P90, a lower mean Euclidean distance, or improvement for most events. Here it is an approximation on retained candidates, not a calibrated full posterior. In particular, multimodal means may fall between modes in low-probability or physically implausible regions. No new forward solve tests those locations.

Average soft ESS is approximately 8.9–58.3 for one-sided cases, 2.58 for clusters, 1.86–1.99 for real geometry, 8.7–9.1 for fault dmin2, 1.64 for dmin4, 1.02 for dmin7, and 1.25/1.61 for heterogeneous 1D/homogeneous. Thus the near-MAP fault dmin7 result is consistent with very concentrated saved weights. ESS is not a modality test; no unimodality claim is made.

These are single synthetic realizations, with reused controls, not independent replicates; catalogue coordinates are synthetic truth rather than independently established earthquake locations. All original report caveats still apply. **Velocity results cannot change in this postprocessing diagnostic.**

## Files and reproduction

- `diagnose.py`: NumPy/standard-library extractor; only writes beside itself, exclusive-create mode to avoid overwriting results. Fixed ROOT `/mnt/disk01/egor`; 300-second alarm. To rerun, first make a new study-directory copy containing only the script (do not overwrite these outputs).
- `diagnostic.json`: complete run/pair metrics, source report hashes, input hashes/order digests, validation, mean/median 3D error, P90/P95/max, spatial RMSE, fractions, shifts, spread, ESS, caveats.
- `pairs.csv`: all 28 distinct matched comparisons, including P90 and spatial-RMSE gains for both estimators relative to hard.
- `series.csv`: descriptive series effects and before/after counts; shared pair deduplicated in total.
- `per_event.csv`: 139,894 rows (about 52 MB), each unique run-event once rather than duplicating reused hard controls; truth/MAP/mean coordinates, errors, shift, posterior RMS spread, ESS. Pair membership is joined via run IDs in `pairs.csv`.
- `test_diagnose.py`: focused tests of cell centres, mean normalization, hard equality, MAP ties, invalid weights, spatial RMSE and linear percentile.

Run tests from this directory with `PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_diagnose.py`. The extraction completed successfully, and all five focused unit tests passed in 0.032 seconds.
