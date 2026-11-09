# Fixed-source homogeneous tomoDD forward diagnostic

Date: 2026-10-26. This follow-up supersedes any inference of a clean forward path from the read-only 2026-10-21 audit. It does **not** replace existing inversion results or demonstrate the cause of their velocity errors. No production solver or old inversion artifact was changed, and no inversion was run in this diagnostic.

> **2026-10-27 inversion follow-up:** a separate corrected production build has now completed matched `dd1-d40-s5` inversions on both geometries. It did not improve velocity RMSE in either geometry. See [the controlled inversion report](tomodd-guard-fix-comparison.md). The fixed-source diagnostic below remains unchanged in scope; the original solver and old runs remain intact.

## Result relevant to the paper

No station-file coordinate/order/unit/elevation-sign mismatch was found on the exercised path. However, the unchanged spherical ray routine can use AK135 outside an artificially narrowed regional lookup area, even when the regional MOD and its padding are homogeneous. Removing only four 0.1-degree horizontal exclusion offsets in a private forward-only build eliminates the observed discrepancies for every tested event–station pair, with the original AK135 support unchanged.

The fixed-source evidence alone does not establish how much of the existing reconstruction error it explains. At the time of this diagnostic, a controlled heterogeneous-inversion check was required before finalizing an intrinsic method/geometry comparison; the dated follow-up above links that check. No manuscript result or PDF was changed on the assumption that the diagnostic found nothing.

## Protocol and controls

- Full artificial catalog: 1920 events × 52 stations = **99,840 pairs**.
- Full Geysers catalog: 3776 events × 26 stations = **98,176 pairs**.
- True source positions fixed; no origin-time estimation, inversion, observation noise, sampling, or checkpoint selection.
- Source IDs and original Cartesian XYZ checked against saved truth/state arrays. Actual station files checked byte-for-byte against frozen-adapter regeneration. Event/station serialization, REAL*4 reader conversions and station elevation sign conversion reproduced.
- Regional grid: original padded 22×9×9 nodes, **5 km/s at every node including padding**.
- `t_cart`: original Cartesian straight distance / 5 km/s.
- `t_chord`: spherical straight chord / 5 km/s on the serialized endpoints, radius 6371 km.
- `t_probe`: the frozen production PBR object, called by a forward-only probe in production endpoint order.
- The probe bypasses the main catalog reader; the endpoint conversion contract is separately checked against the frozen reader source. This is not a full production-main execution test.

Three settings were evaluated, without changing the declared tolerances:

1. **Stock:** regional MOD=5, original AK135 support, original horizontal guard.
2. **Global P=5 control:** stock ray object and geometry; only the copied AK135 P-speed column changed to 5 km/s. This is the true globally homogeneous test.
3. **Guard-only control:** original MOD and original AK135 support; a private ray object changes only the four horizontal guard offsets to the full node bounds. The bottom guard is unchanged.

Predeclared tolerances: probe−chord RMS ≤1 ms, maximum ≤5 ms; probe−Cartesian RMS ≤5 ms, maximum ≤10 ms. The existing absolute pick-noise reference is 10 ms. They are diagnostic tolerances, not an inversion-validity certificate.

## Full-pair results

Errors are milliseconds. The `probe−chord` label is deliberately not called pure numerical error for the stock setting: the stock support introduces nonhomogeneous speeds.

| Setting | Geometry | Probe−chord RMS | Probe−chord max abs | Probe−Cartesian RMS | Probe−Cartesian max abs |
|---|---|---:|---:|---:|---:|
| Stock | Artificial | 3.965629 | 12.199584 | 6.033565 | 19.240779 |
| Stock | Geysers | 3.247847 | 371.389735 | 4.707885 | 365.394219 |
| Global P=5 | Artificial | 0.000070311 | 0.000488938 | 2.504945 | 7.475775 |
| Global P=5 | Geysers | 0.000062790 | 0.000463426 | 1.953110 | 6.557972 |
| Guard-only | Artificial | 0.000070327 | 0.000488938 | 2.504945 | 7.475775 |
| Guard-only | Geysers | 0.000062782 | 0.000463426 | 1.953110 | 6.557972 |

The true globally homogeneous and guard-only controls both pass all tolerances on **all 198,016 pairs**. Their maximum probe−chord error is below 0.5 microseconds. Remaining approximately 2 ms RMS differences from original Cartesian times are dominated by spherical embedding/serialization; they are smaller for Geysers, not larger.

Compared with stock, the guard-only control changes times by more than 1 ms for **26,226 artificial pairs** and **16,598 Geysers pairs**. Thus the support influence is not confined to one isolated outlier, although the largest discrepancy is isolated in the homogeneous true-source catalog.

## Reproduced outlier and path evidence

Geysers event **3157429**, station **RGP**:

| Quantity | Value |
|---|---:|
| Original Cartesian time | 3.656960352219 s |
| Serialized spherical chord time | 3.662955868408 s |
| Stock probe time | 3.291566133499 s |
| Guard-only probe time | 3.662955999374 s |
| Stock probe−chord | −371.389735 ms |
| Stock / guard-only segments | 32 / 4 |

The stock value repeats exactly in fresh processes, with endpoints reversed, with repeated calls, and with the neighboring pair before/after it. A private extended driver linked against the **exact unchanged** ray/transform objects reproduces all three inspected target times and segment counts before exposing their paths.

For the outlier, 28 of 33 returned nodes cross the southern regional guard and use **5.8 km/s AK135**. The path departs from the direct chord by as much as 1.18 km. The dumped-node trapezoidal integral rounds exactly to the stock output. Its faster support material saves 451.90 ms relative to 5 km/s on that polyline, while the detour costs about 80.51 ms, explaining the net −371.39 ms discrepancy.

Two inspected positive-error paths finish inside the regional guard, yet their approximately +12.1/+12.2 ms discrepancies disappear with globally P=5 support and with the guard-only change. This establishes support influence during path construction; the post-PBR path dump does not isolate the individual gradient-stencil evaluation responsible.

## Production call path and source evidence

Frozen sources are under `runs/matched_geometry_20261021/snapshot/solver/`:

- `RaySPDR2.f:840–841`: the original regional test uses `xn(1)+0.1`, `xn(nx)-0.1`, `yn(1)+0.1`, `yn(ny)-0.1` (degrees).
- `RaySPDR2.f:918`: points outside that test use AK135 P speed.
- `RaySPDR2.f:426–508`: path construction evaluates velocities at surrounding finite-difference points as well as along the ray.
- `tomoDD-SE.f:928`: PBR's returned time is stored in `tmp_ttp`.
- `tomoDD-SE.f:951`: `ttmder` is separately called for derivatives; it does not replace the stored PBR time.
- `dtres_FDD_lm5.f:58–60`: residual calculation consumes `tmp_ttp`.
- `RaySPDR2.f:1161–1170`: derivative integration uses full model bounds and regional nodal speeds, rather than the same 0.1-degree exclusion. The forward/derivative behavior warrants a separate consistency check; it was not quantified here.

The causal control changes only four offsets on two source lines. The private diff is retained at:
`runs/tomodd_homogeneous_20261026/regional_guard_control/build/horizontal_guard_only.diff`.
It builds only a forward probe, **not a production inversion executable**.

## Reproducibility and validation

Implementation: `studies/tomodd_homogeneous_20261026/`:

- `run_check.py`, `test_run_check.py`: initial full-pair test and 14 solver-free tests.
- `reproduce_pairs.py`: fresh-process/repetition/direction/order checks.
- `inspect_ray_paths.py`: exact-object private driver and target support controls.
- `check_global_homogeneous.py`: full globally P=5 control; seven support-transformation tests.
- `check_regional_guard.py`: guard-only causal control; seven source-diff tests.

Each script exclusively creates its named artifact directory and must not be rerun over existing output. Probe/compiler commands are bounded. Original source/object/binary/input hashes and old sealed outputs were checked unchanged. The initial pre-probe harness import failure is retained separately; it launched no ray probe.

Detailed arrays, raw requests/results, metrics, logs and provenance:

- `runs/tomodd_homogeneous_20261026/report.md` — stock regional-homogeneity test, not the final global control.
- `runs/tomodd_homogeneous_20261026/pair_reproduction/report.md`.
- `runs/tomodd_homogeneous_20261026/path_inspection/report.md`.
- `runs/tomodd_homogeneous_20261026/global_homogeneous/report.md`.
- `runs/tomodd_homogeneous_20261026/regional_guard_control/report.md`.
