# tomoDD horizontal regional guard: matched inversion follow-up

Date: 2026-10-27. This is the controlled heterogeneous-inversion follow-up to [the homogeneous forward diagnostic](tomodd-homogeneous-diagnostic.md). No manuscript TeX or article PDF was changed.

## Intervention and matched protocol

The new production inversion binary removes only the four horizontal 0.1-degree exclusion offsets at frozen `RaySPDR2.f:840–841`. The lookup now uses the full horizontal regional node bounds. AK135 support, the bottom guard, all other ray code, all other linked objects, compiler and flags are unchanged. An original-object relink reproduces the frozen inversion binary byte-for-byte before the corrected Ray object is substituted.

This is an explicit corrected build under `runs/tomodd_guard_fix_20261027/build/`, not an overwrite of the frozen solver or an external live installation. Reusable build/launch code, regression tests and the source patch are retained under `studies/tomodd_guard_fix_20261027/`.

Both geometries use `dd1-d40-s5`: absolute/DD amplitude 1/1, damping 40, smoothing 5/5/5, three joint source/velocity iterations, 1-km nodes, initial source sigma 1 km per coordinate, seed 42, pair degree 4 / radius 3 km, ray step 0.05 km. Hypocentres are **free**, unlike the preceding fixed-source forward diagnostic.

All nine generated input/support files and the complete solver control are **byte-identical** to their geometry-specific original dd1 reference. Truth arrays and initial states are identical. No observations were regenerated and no old result was reused as a corrected result.

- Original inversion SHA-256: `33f41e0eddb2d6f82b860331f2f1803cb4b240f4671826610ceb948fef58c3f4`.
- Corrected inversion SHA-256: `0c686a17b8f519adaa52281d6950cb45782e5fd5996a334ae65d986e272fa7a4`.

## Results at iteration 3

Velocity errors are m/s; source errors are metres. Slices are native trilinear Vp at exact **z=1.5 km**, sampled on 125-m XY centres; volume errors use the full 125-m physical grid. The reference truth cell at the slice boundary is the deeper cell, consistent with earlier plots.

| Geometry / solver | Slice RMSE | Volume RMSE | Hypocentre RMS |
|---|---:|---:|---:|
| Common 3D L2 oracle | 183.993 | 142.859 | — |
| Artificial / original | 250.337 | 404.072 | 277.878 |
| Artificial / corrected guard | 360.699 | 506.354 | 349.576 |
| Geysers / original | 476.034 | 496.070 | 573.763 |
| Geysers / corrected guard | 489.549 | 500.981 | 568.637 |

The oracle is a full-volume L2 projection into the model space, not a slice-specific lower bound.

**The guard correction does not improve velocity recovery in either geometry at this fixed protocol/checkpoint.** Artificial recovery worsens; Geysers velocity scores worsen slightly while source RMS improves slightly. Thus the preceding homogeneous forward discrepancy is real, but its correction does **not** explain or cure the poor Geysers reconstruction in this matched dd1 comparison. This does not establish that all remaining error is intrinsically due to acquisition geometry, nor rule out sensitivity at other settings/checkpoints. Forward/Jacobian consistency has not been independently quantified.

## Completion and validation

- Artificial: **1920/1920** events retained, 52 stations, three completed joint iterations.
- Geysers: **3776/3776** events retained, 26 stations, three completed joint iterations.
- Both solver exit codes zero. Solver runtimes approximately 195 and 184 s, concurrently; this is not a controlled speed comparison.
- 16 new regression tests and 19 shared tests passed. The 16 new tests were additionally rerun independently after completion.
- Independent read-only review checked invocation code, actual input/control bytes, active build hashes, finite positive velocity snapshots, complete relocation snapshots, and agreement of final snapshots with retained state.
- 389 protected files and output/validation artifact ledgers verified unchanged. Two initial solver-free validation failures and their exact source/log histories remain archived; neither launched an inversion.

## Artifacts and figures

Run root: `runs/tomodd_guard_fix_20261027/`:

- `manifest.json`, `status.json`, `final_audit.json`.
- `build/manifest.json`, `build/commands.json`, corrected `build/tomoDD-SE` and source/patch/object.
- `outputs/synthetic-dd1-d40-s5/` and `outputs/geysers-dd1-d40-s5/`, each with config, results, state, truth and all solver iteration artifacts.
- `validation/`, `tests/`, and preserved `history/attempt1/`, `history/attempt2/`.

Figures (PNG/PDF/JSON, shared colour scale, no outlines or added smoothing):

- `figures/tomodd_dd1_artificial_geysers_z1p5km_guard_fixed.*`: oracle, corrected artificial, corrected Geysers.
- `figures/tomodd_dd1_artificial_geysers_z1p5km_guard_fixed_before_after.*`: oracle, artificial before/after, Geysers before/after.

Rebuild the figures without running any solver:

```sh
python -B paper/paper_scripts/make_tomodd_dd1_geometry_comparison.py --guard-fixed
python -B paper/paper_scripts/make_tomodd_dd1_geometry_comparison.py --guard-fixed --with-before
```

The original no-flag figure path and old run artifacts remain intact. The plotting script validates the corrected build ledger and matched-input provenance before plotting, and recomputes volume RMSE against saved results.
