# Real picks: Hard / Soft / Reference

## Separate Hard / Soft figure: native sensitivity, not resolution

**Current display policy (4.7):** the exact raw intersection `Hjj / alpha >= 1` in both runs, with no convex hull, padding or hole filling. The temporary expanded-envelope change has been reverted; see [ENVELOPES.md](ENVELOPES.md). Expanded masks belong to 4.5/4.6, not this figure.

Rebuild from `projects/EMTomo`:

```sh
python -B paper/figures/results/real_picks/source/make_hard_soft.py
```

Outputs: `../png/real_picks_hard_soft_sensitivity.png`, plus separate `_xy.png` and `_xz.png` views. Matching PDFs, JSON provenance and NPZ arrays are stored here. The original reference comparison is unchanged.

This figure uses final Hard/Soft velocities and a **heuristic data-sensitivity mask**, not a claim of good spatial resolution. No published reference velocities or resolvability values are loaded. Section coordinates remain the same as the earlier figure (XY: 1.6 km below sea level; XZ: Y = 5.5 km); the XY depth was originally chosen using reference coverage, and is retained for comparability rather than re-optimized here. The new mask and color limits are calculated from our runs only.

For each method, `Hjj` is its saved diagonal of the weighted data normal matrix, after eliminating event origin time. For these runs, coverage-dependent damping is disabled and the actual scalar regularization is `alpha = lambda_reg * mean(Hjj)`. The mask is:

```python
(hard_Hjj / hard_alpha >= 1) & (soft_Hjj / soft_alpha >= 1)
```

This requires diagonal data sensitivity at least as large as damping **in both methods**, with the same intersection drawn on both panels. It avoids treating the independent per-run percentile normalization in `coverage_confidence` as an absolute common scale. The threshold is an explicit heuristic, not a validated resolution threshold. Inter-cell correlations, spatial velocity–hypocentre trade-offs and the nonlinear EM trajectory are not captured by this diagonal metric. No smoothing, convex hull, padding, hole filling or removal of disconnected selected cells is applied.

**Timing limitation:** `iter_2/sensitivity_diagonal.npy` belongs to `iter_2/model.npy`, before the third update. The figure displays `final_model.npy`, after that update, which changes velocities by up to 2%. Thus the mask is not a diagnostic recomputed in the final model. Saved coverage confidence is checked against the sensitivity formula; final artifacts and completion markers are verified by the existing loader. New hashes of diagnostic files are recorded for reproducibility, not presented as previously frozen integrity attestations.

Shared areas at the plotted threshold are **57.75 km² (XY)** and **39.25 km² (XZ)**. To expose dependence on the heuristic threshold, JSON also records areas at thresholds 0.5 and 2.0:

| Hjj / alpha threshold | XY area, km² | XZ area, km² |
|---|---:|---:|
| 0.5 | 75.00 | 46.75 |
| 1.0 | 57.75 | 39.25 |
| 2.0 | 40.00 | 28.25 |

The NPZ includes full native 3-D sensitivity/damping fields, sampled fields, individual masks (`xy_hard_mask`, `xy_soft_mask`, and corresponding `xz_` arrays), their exact raw intersection (`xy_shared_mask`, `xz_shared_mask`), and exact plot coordinates. The same shared mask controls fading and contours in both panels. JSON records threshold 1.0, shared pixel counts/areas, exact-intersection checks, timing limitations and reproducibility hashes; obsolete envelope/display/added-mask fields and the envelope-helper hash are removed. Events and stations are not plotted.

A genuine resolution region requires a separate matched recovery/point-spread diagnostic. A frozen linear M-step operator would be `R = (H + alpha I)^(-1) H`, with limitations from fixed rays, hypocentral hypotheses and weights. It cannot be recovered from `diag(H)` alone. No such test is claimed or performed by this plotting script.

## Original three-column comparison

Rebuild from `projects/EMTomo`:

```sh
python -B paper/figures/results/real_picks/source/make_comparison.py
```

## Outputs

- `../png/real_picks_comparison.png`: combined two-row, three-column comparison.
- `../png/real_picks_xy.png`: horizontal plan view.
- `../png/real_picks_xz.png`: longitudinal vertical section.
- Matching PDFs, this script, provenance JSON and `real_picks_sections.npz` remain in `source/`. The `png/` directory contains only PNGs.

No events or stations are plotted. No inversions are run, and existing results and article files are not changed.

## Models and sections

Columns are Hard, Soft, Reference. Hard/Soft are the verified final **500 m** models from `runs/geysers_real_refine500_fmm125_20261110_110138`, each after three continuation updates. They continue their own earlier 1 km final models (hard: four updates, soft: five). Loading reuses the existing structural-comparison checks for artifact hashes, completion markers, domain registration and unsmoothed parent-to-child initialization.

The reference is **Guo & Thurber, GJI 2021**, DOI `10.1093/gji/ggab017`, from `data/geysers_2021/Geysers_Vpmodel_Guo-Thurber2021GJI.CSV`, not the JGR 2022 temporal model and not ground truth.

- **XY:** published source depth **1.6 km below sea level**, or **Z = 2.766 km below the box top**. This is a constant-elevation horizontal slice, **not a depth below local terrain**. Selected for a larger reference-resolved area: a full-domain depth scan at 100 m spacing peaked near Z = 2.75 km; refinement at 10 m spacing in sea-depth 1.50–1.70 km found the largest sampled area at the published 1.6 km layer. Selection uses only reference resolvability, never agreement with Hard/Soft.
- **XZ:** **Y = 5.5 km**, through the domain centre along its long X axis (azimuth 135°). Z increases downward.
- Dashed lines show the intersection with the other plane.
- Full domain: 23 × 11 × 7 km. All panels use equal metric aspect and one common Vp scale, also shared by the separate images.

## Identical comparison region

The author's `data/geysers_2021/Readme`, note 2, explicitly states:

> The regions with scaled resolvabilities larger than 0.4 are well resolved.

The indicator measures semblance of recovered and true checkerboard models and is rescaled to [0, 1]; it is not a probability or a Vp error estimate.

For each plane the shared region is computed **once, only from the reference**:

```python
finite(reference_vp) & finite(reference_resolvability) & (reference_resolvability > 0.4)
```

Vp and resolvability use the existing piecewise-linear horizontal triangulation and vertical interpolation, without extrapolation. The exact same Boolean mask and contour are drawn on all three columns. All selected components are retained; there is no manual polygon, model-dependent mask, or selection by agreement with Hard/Soft. Outside the region, a white overlay with alpha 0.82 fades the data; missing reference coverage remains blank. The boundary is the 0.5 isoline of the Boolean mask sampled at plot centres.

The XY region occupies approximately **96.66 km²** (previous near-surface slice: 10.52 km², a 9.19-fold increase), and the XZ region **56.06 km²**, at the 100 m plotting step. The NPZ stores one shared mask per plane, reference resolvability, all three Vp fields, XYZ coordinates and plotting edges. JSON records mask hashes and data/source hashes.

Hard/Soft are sampled as actual piecewise-constant 500 m cells, without smoothing. A 100 m plotting step does **not** imply 100 m tomographic resolution. Applying the reference mask does **not** establish the resolution or accuracy of Hard/Soft. Exact vertical-datum compatibility with the published model is assumed, not independently established.
