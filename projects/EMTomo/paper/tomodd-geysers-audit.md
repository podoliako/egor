# Read-only audit: tomoDD flat4 / Geysers comparison

Date: 2026-10-21. No inversions, forward solves, or builds were run. No original run artifacts or paper claims were changed. Inputs, saved solver sources, logs, hashes, observation rows, pair graphs, and selected iteration-3 fields were inspected. New slice scores were recomputed from saved fields using the native interpolation/history helpers exposed by `paper/evaluate_metrics.py`.

## Main finding

No Geysers-specific input-coordinate, elevation-sign, observation-assembly, model-initialization, capacity, or solver-termination failure was found. This does **not** demonstrate that the shared tomoDD settings are appropriate for both geometries.

The premise of a large overall deterioration is not supported by the saved metrics. Recovery of particular shallow slow anomalies deteriorates, especially with noisy initialization, but whole-volume and location scores do not uniformly deteriorate. The current runs cannot separate acquisition geometry from regularization, source/velocity trade-off, pair selection, and checkpoint effects.

## Artifact locations

Benchmark root `B = /home/egor/tomodd_benchmark`.

Comparison directories under `B/runs/`:

- `comparison_flat4_synthetic_em8_tomodd3_zero_noise`
- `comparison_flat4_geysers_em12_tomodd3_zero_noise`

Their `provenance.json` files identify all inputs and checkpoints. Source run aliases below denote directories under `B/runs/`:

- **S0:** `flat4_20x7x7km_52st_1920ev_tomodd-1km-layerstart-ray005-deg4-sigma0km-seed42-10it-coordfix-nolimit`
- **SN:** `flat4_20x7x7km_52st_1920ev_tomodd-1km-layerstart-ray005-deg4-sigma0p5km-seed42-3it-coordfix-nolimit`
- **G0:** `geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_tomodd-1km-layerstart-ray005-deg4-sigma0km-seed42-3it-coordfix-nolimit`
- **GN:** `geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_tomodd-1km-layerstart-ray005-deg4-sigma0p5km-seed42-3it-coordfix-nolimit`

Zero means zero initial hypocenter perturbation, not noiseless arrivals. All tomoDD comparisons here use iteration 3. S0 continued to iteration 10; its final residual file/model must not be confused with iteration 3.

## Quantitative comparison

Velocity errors are m/s; location errors are metres. The slice is the actual paper plane, z=1.4375 km, fine-grid index 11. Common-grid interpolation is native trilinear Vp for tomoDD and constant cells for EMTomo.

| Metric | Artificial geometry | Geysers geometry |
|---|---:|---:|
| tomoDD zero: slice RMSE | 141.94 | 143.21 |
| tomoDD noisy: slice RMSE | 115.97 | 166.96 |
| EMTomo hard: slice RMSE | 63.97 | 103.81 |
| tomoDD zero: volume RMSE | 251.62 | 188.75 |
| tomoDD noisy: volume RMSE | 208.12 | 214.69 |
| EMTomo hard: volume RMSE | 202.86 | 208.32 |
| tomoDD zero: all-sphere RMSE | 287.36 | 276.76 |
| tomoDD noisy: all-sphere RMSE | 273.49 | 281.31 |
| EMTomo hard: all-sphere RMSE | 327.62 | 329.69 |
| tomoDD zero: location RMS | 258.58 | 141.90 |
| tomoDD noisy: location RMS | 222.08 | 180.58 |

The whole-volume initial tomoDD RMSE is 173.25 m/s in both geometries: all four selected reconstructions improve sphere RMSE but worsen whole-volume RMSE relative to that initial field. The initial model is not a representation oracle.

The two shallow slow spheres visible in the paper lose template amplitude. Here amplitude is the projection of reconstructed contrast against the true layered background onto the true 3-D sphere template; 1 denotes matching template amplitude, not necessarily perfect shape or localization.

| Sphere / reconstruction | Artificial | Geysers |
|---|---:|---:|
| Sphere 2, center (8,4.8,2) km, tomoDD zero | 0.610 | 0.448 |
| Sphere 2, tomoDD noisy | 0.551 | 0.373 |
| Sphere 4, center (16.5,4.5,1.6) km, tomoDD zero | 0.815 | 0.476 |
| Sphere 4, tomoDD noisy | 0.722 | 0.341 |
| Sphere 4, EMTomo hard | 0.784 | 0.547 |

Thus the visible loss is real for these structures, but is not specific to tomoDD in every metric. These amplitudes include background leakage; they are not isolated peak amplitudes.

## Geometry and observation checks

- Geysers preparation applies `station depth = 1166 m - station elevation` and `event depth = 1166 m + catalog depth`. Independent reconstruction matched source CSV xyz to numerical precision. Four events below 7 km were excluded consistently.
- Actual tomoDD station elevations are negative local depths; the solver converts them back to positive-down km. Example: ACR elevation 803 m becomes source depth 363 m, input elevation -363 m, receiver depth +0.363 km. No double elevation correction was found.
- All actual event IDs/station IDs and saved truth coordinates match source data. Event serialization/coordinate roundtrip errors are sub-metre, not a hundred-metre shift.
- Every absolute observation matches source relative arrivals plus the intended per-event gauge within 5e-9 s. Every serialized DD component equals its corresponding absolute pick exactly; no reversed DD sign, duplicated rows, or missing source arrivals were found.
- Source input/forward metadata hashes and audited run config/model/control/log hashes match provenance. Build binary hashes match manifests.
- The common catalog/station vertical datum remains an unverified physical assumption. Both synthetic forwarding and inversion use the same transformed geometry, so this is not an internal sign mismatch.
- There is a shared Cartesian-to-spherical embedding approximation: pairwise distance RMS differences are approximately 12.5 m artificial versus 9.8 m Geysers. This is not larger for Geysers and is not a measurement of heterogeneous travel-time error.

Evidence: `B/tomodd_benchmark/external_checkerboard.py:185–187,229–244,311–341,618–630`; `projects/EMTomo/studies/geysers_elevated/prepare.py:123–144,210–215`; saved build `getdata_SPDR.f:222–224,534–551`, `dtres_FDD_lm5.f:58–90`.

## Solver settings and runtime

All four actual `MOD` files are byte-identical (SHA256 `d9a1b9c548d338bef940012b0e03ec8875c655569861394938d598ac3fa432df`): 22x9x9 total nodes, 980 interior inversion parameters, 1-km spacing, identical layered initial Vp and padding.

Actual controls (`work/inversion/tomoDD-SE.inp:33–51`) are shared:

- catalog P observations only; no waveform cross-correlation observations;
- joint source/velocity inversion from the first iteration;
- smoothing 5/5/5, damping 40, `THRE=0`;
- catalog P weight 0.1, absolute multiplier 10;
- residual/distance rejection disabled, station distance limit 1000 km;
- ray control `stepl=0.05` km; patched ray minimum-segment threshold 0.05 km;
- no initial relocation-only stage or later reweighting schedule.

The wrapper chooses smoothing/damping by grid spacing (`external_checkerboard.py:386–400`), not by an independently validated Geysers tuning procedure.

### Absolute-dominated weighting

The solver multiplies both row coefficients and residuals by weights. A DD P row has amplitude weight 0.1; an absolute P row has weight 1.0. Thus equal residuals have squared-objective ratio 0.01, not 0.1. Roughly twice as many DD rows gives a nominal DD/absolute contribution ratio of about 2% **only if residual magnitudes are equal**; this is not an effective-information calculation. DD observations are derived from the same picks and are not independent extra measurements.

Evidence in Geysers build: `weighting_FDD.f:96–104`; `tomoDD-SE.f:1375–1386,1517–1537`. This weighting is not inherently a bug: absolute and DD rows have different roles and correlated errors. It is nevertheless an important sensitivity parameter for the current comparison.

### Runtime checks

- Every selected iteration retains all events/observations and 980 velocity nodes; no airquakes or failed LSQR exits.
- LSQR reports `istop=2` (least-squares stopping tolerance), not iteration or condition-limit failure.
- Logged weighted catalog residual statistics, iterations 1/2/3 in ms: S0 34.3/33.4/33.3; SN 36.9/33.4/33.4; G0 27.2/27.0/26.9; GN 28.8/27.0/27.0.
- These are mixed catalog-row statistics, not absolute-only RMS, and are computed from linearized residual corrections before velocity clipping. They are not an independent final nonlinear forward check.
- `nolimit` does not remove the solver's hard Vp update cap of +/-0.4 km/s per iteration. Counts of printed cap-sized first updates: S0 10, SN 6, G0 14, GN 27 (output rounding prevents exact clipping counts). No absolute Vp bounds were reached.
- S0 continuation from iteration 3 to 10 barely changes the residual statistic while increasing mean depth-centroid shift. More iterations alone are not demonstrated to improve reconstruction.

Evidence: each `tomoDD-SE.log:53–194`, `tomodd.log:48–81`; saved `tomoDD-SE.f:2101–2106,2165–2188,2485–2494`.

The two saved builds under `B/data/small_domain_solver/` are `mins_0p05_e73254583560` and `mins_0p05_2d6bca5eb5b1`. Compared saved solver source files differ only in event capacity, 2000 versus 4000. Coordinate/ray/observation/residual implementations are identical.

## Geometry and regularization change the inverse problem

| Quantity | Artificial | Geysers |
|---|---:|---:|
| Events | 1920 | 3776 |
| Stations per event | 52 | 26 |
| Absolute picks | 99840 | 98176 |
| Joint unknowns, 4E+980 | 8660 | 16084 |
| Occupied 1-km event cells | 840 | 385 |
| Events above z=1.4375 km | 384 (20%) | 101 (2.7%) |
| Events with x>15 km | 480 (25%) | 150 (4.0%) |

Event occupancy is a geometry diagnostic, **not a ray-coverage or resolution measurement**. More Geysers events do not compensate automatically for half as many station observations per event, concentrated sources, and nearly twice as many source unknowns.

Pairs are selected greedily by shortest initial-event distance, at maximum degree 4 and distance 3 km. Zero-start median edge length falls from 0.833 km to 0.221 km. Graph components: S0 1, SN 2, G0 4, GN 2; G0's giant component contains 3761/3776 events, so small isolated groups alone are unlikely to explain the broad slice.

Short DD baselines reduce velocity-derivative differences. The solver also drops individual model derivatives below 0.05 and DD derivative differences below 0.01 (`tomoDD-SE.f:965–973,1516–1522`). Their actual influence was not quantified from saved Jacobians.

All nodes remain eligible under `THRE=0`; smoothing and column normalization affect the physical meaning of damping. The same numerical damping does not imply equal effective regularization across geometries.

### EMTomo was not held at the same settings either

| EMTomo setting | Artificial selected run | Geysers selected run |
|---|---:|---:|
| Updates scored | 8 | 12 |
| Damping | 0.5 | 1.0 |
| Coverage damping exponent | 1 | 2 |
| Candidate cells searched | 5 | 1 |
| Retained candidates / mode | 1 / hard | 1 / hard |
| Maximum fractional velocity step | 0.02 | 0.02 |
| Model sigma, seconds | 0.02 | 0.02 |

Settings verified against actual `meta.json`. These differences prevent attributing relative visual robustness solely to the methods' intrinsic properties.

## Recommended next diagnostic experiments (not executed)

1. **Separate source/velocity trade-off:** same Geysers data and iteration protocol, but true locations held fixed (and origin-time handling explicitly controlled). Merely initializing true locations does not fix them. Compare with the joint run.
2. **Small predeclared regularization/weighting sensitivity study:** vary absolute/DD balance and smoothing/damping, not just iteration count. Keep absolute observation scale and other settings controlled; increasing DD weight is a test, not a guaranteed improvement. Account for DD dependence on the same noisy picks.
3. **Pair-graph sensitivity:** compare degree 4 with a larger and/or spatially balanced graph, reporting edge lengths and connectivity. More nearly coincident pairs may add little velocity sensitivity.
4. **Actual ray/Jacobian coverage:** inspect the two shallow slow spheres and shallow background; event density alone cannot establish resolution.
5. **Independent final forward residual check and checkpoint curves:** avoid interpreting the linearized logged residual as a nonlinear goodness-of-fit certificate. Select checkpoints by a declared rule, not retrospectively by the most attractive slice.

Until these tests, the defensible conclusion is: this Geysers run loses some shallow anomaly contrast under the chosen tomoDD protocol; no obvious launch/input failure was found, and the existing evidence does not establish that tomoDD generally performs worse on real acquisition geometry.
