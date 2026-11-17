# Synthetic tomography comparison

Rebuild from the EMTomo root:

```sh
python -B paper/figures/results/tomography/source/make_comparison.py
```

## Outputs

- `../png/synthetic_tomography_comparison.png`: all five sections. Each section is a **3 × 3 table**: rows **True / Soft / Hard** from top to bottom; columns **horizontal XY / transverse YZ / longitudinal XZ** from left to right. The short transverse view uses a narrower column while all axes retain equal metric scale. Slice captions sit just above the images. `synthetic_tomography_comparison_three_planes.png` is an identical preview.
- `../png/section_4_2.png` through `section_4_6.png`: individual blocks.
- PDF versions, numerical slice arrays (`*_slices.npz`), and `selection_and_sections.json` stay in this source directory.
- The `png` directory contains PNGs only. No article insertion or PDF rebuild is performed.

## Fixed model pairs

| Section | Pair ID | Tested | Soft retained | Seed separation (m) | Updates hard / soft |
|---|---|---:|---:|---:|---|
| 4.2 one-sided | `section2_pair01` | 50 | 50 | 875 | 3 / 3 |
| 4.3 two clusters | `section3_pair01` | 50 | 50 | 875 | 3 / 3 |
| 4.4 real-geometry checkerboard, nominal noise | `section4_pair07` | 50 | 50 | 700 | 3 / 3 |
| 4.5 fault | `section5_topnsweep_k128_top50` | 128 | 50 | 250 | 12 / 12 |
| 4.6 heterogeneous, supplied 1-D start | `section6_pair01` | 50 | 50 | 700 | 9 / 9 |

Hard retains one candidate. Sections **4.2–4.4 retain the prior selected models after three updates**. Sections **4.5–4.6 use final_model.npy**. The script reads the combined policy from `paper/figures/article_comparison_metrics.json`. Rebuild that report and numeric tables with `python -B paper/paper_scripts/make_article_comparison_metrics.py --update-tables`. It does not select checkpoints or pairs using their errors. The original source truth is shown, not a representation oracle or an inversion-cell-averaged reference.

The fault comparison now uses twelve updates in both modes. The heterogeneous model uses a favorable 1-D start constructed from the known truth; this is explicitly labeled. All pairs share identical initial velocity arrays, verified before drawing. Whole-volume RMSE is independently recomputed from the displayed arrays and original truth on their exact nested common grid, and checked against the matching table metrics.

For 4.2–4.4, P90 uses locations in the selected V_3 model. For 4.5–4.6, location P90 uses the **last saved E-step**, before the final velocity update: V_(N−1), not a new relocation in V_N. This distinction is explicit in `article_comparison_metrics.json`. No forward/inverse solves are rerun. Article prose and captions were deliberately left unchanged when updating the numeric table bodies, so their old iteration labels and conclusions require a separate editorial pass.

## Physical section planes

Each block shows an XY slice at **Z = 1.6 km below the native model top**, an X–depth slice at fixed Y, and a Y–depth slice at fixed X. Exception: **4.3 uses Z = 4.5 km**. Each series uses identical planes for Hard, Soft and True; Z is not sea-level depth. All nine slice arrays per series are exported to NPZ.

- 4.2: Y = 6 km, X = 12 km.
- 4.3: **Y = 3.5 km, X = 6.5 km, Z = 4.5 km**.
- 4.4, 4.5 and 4.6: **the same physical planes**, Y = 3.5 km and X = 10 km in the fault-test Cartesian frame.

For 4.4 and 4.6, native X coordinates must be shifted by **+1 km** to obtain fault-frame coordinates. Their native X section is therefore 9 km, not 10 km. Their displayed X domain is 1–21 km; the fault domain is 0–20 km. Native full domains and depths are retained, with no rescaling or cropping to force a visual match. Matching event/station coordinates are checked by identifier to sub-micrometer tolerance after translation.

## Rendering

### 4.5 and 4.6 shared, native-cell sensitivity envelopes

Hard/Soft XY, XZ and YZ use the same display envelope within each plane: intersect the native `Hjj / alpha >= 1` selections, take their convex hull **on the native cell-centre grid**, expand by **one native cell** with Euclidean distance, fill holes and retain whole cells. The raw sensitivity threshold remains 1; the displayed spatial criterion is relaxed to include the neighbouring cell layer. This gives a controlled one-cell widening rather than an unpredictable change from lowering the numerical threshold. Boundaries are exact axis-aligned cell edges: **1 km in 4.5**, **800 m in 4.6**, with no interpolated diagonal contour. The native mask is repeated onto the plot grid without interpolation.

`alpha = lambda_reg * mean(Hjj)` is the actual scalar damping in these runs (`coverage_damping_power=0`). Outside the envelope, velocities are faded by an 82% white overlay; True is not masked. No velocities are smoothed or replaced. The raw threshold is heuristic; this is **not a spatial-resolution test**. Hull/padding/fill pixels may fail the raw criterion. Diagnostics are from `iter_11` in 4.5 and `iter_8` in 4.6, before the final update, not recomputed at the shown final velocities. Diagnostic hashes and maximum final update are recorded in JSON; the shared helper is also hashed.

| Series | Plane | Raw area, km² | Display area, km² |
|---|---|---:|---:|
| 4.5 | XY | 75.00 | 122.00 |
| 4.5 | XZ | 42.00 | 80.00 |
| 4.5 | YZ | 18.00 | 31.00 |
| 4.6 | XY | 68.48 | 114.56 |
| 4.6 | XZ | 51.20 | 95.36 |
| 4.6 | YZ | 18.56 | 40.32 |

`section_4_5_slices.npz` and `section_4_6_slices.npz` retain all nine velocity slices plus raw/display/added masks, native raw/display masks and Hard/Soft dimensionless sensitivity scores. **4.7 was excluded after the numbering correction**: its original raw threshold-1 mask was restored, without hull or padding.

### 4.5 source fault and layer interfaces

All **True / Soft / Hard** rows in 4.5 carry the same original-source geometry: solid black fault at **X = 6.5 km** (0.85 pt), dashed black layer interfaces (0.7 pt). Interfaces are extracted from the actual source truth array, not the proposed analytical preview or inferred inversion features. Left depths: **0.750 and 1.875 km**; right depths: **1.125 and 2.875 km**. The 125 m source discretization determines these exact boundaries.

XZ shows the fault and separate left/right layer segments. XY shows the vertical fault trace; horizontal layer interfaces do not intersect this constant-depth plane. YZ at X = 10 km lies entirely on the right side and therefore shows only right-side layer boundaries, not a fictitious fault trace. Geometry and its source interpretation are exported in `selection_and_sections.json` under the 4.5 block's `fault_geometry`. Other series are unaffected.

### Checkerboard grids and native fields

Hard and Soft panels in **4.2–4.4** have a thin black checkerboard overlay on all XY/XZ/YZ views (0.45 pt, alpha 0.8). Lines follow interfaces detected in the original True slice, with native-to-common coordinate offsets applied; they are not the finer inversion-cell grid. True panels and sections 4.5–4.6 have no checkerboard grid. Sections 4.5 and 4.6 have separate display envelopes described above.

Native piecewise-constant velocities are sampled at 25-m pixel centers. Sampling is exact cell lookup, with no smoothing or trilinear velocity interpolation. On exact cell boundaries the positive-side cell is used consistently. All axes use equal physical aspect ratio, so short-side sections are narrower.

Colorbars sit immediately to the right of the actual slice extent, are vertically centred, and use **half the height** of the three-row block. Saved PNG/PDF canvases are tightly cropped with a small outer margin, removing excess right whitespace. The same color scale covers all three planes and all three models within each block. The three checkerboards share 4.6–5.4 km/s; the fault uses 3.4–5.8 km/s; the heterogeneous model uses 2.8–7.2 km/s. Assertions reject any displayed value outside the chosen range. Different colorbar ranges must not be interpreted as a common cross-series anomaly scale.

The source JSON records run/checkpoint paths, hashes, displayed model update counts, original truth inputs, physical-plane translations, and sampled ranges. These are illustrative sections; whole-volume errors are not measured from the pictured planes.
