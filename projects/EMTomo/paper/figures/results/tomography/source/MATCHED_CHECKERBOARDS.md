# Separate 4.2–4.4 view with exactly matching True fields

Rebuild from `projects/EMTomo`:

```sh
python -B paper/figures/results/tomography/source/make_matched_checkerboards.py
```

Outputs: `../png/checkerboards_matched_true.png`, plus `checkerboards_matched_true.pdf`, `.npz` and `.json` here. The main comparison, individual section figures and their data are unchanged.

## Final clean figures

```sh
python -B paper/figures/results/tomography/source/make_checkerboards_final.py
```

Final output stems are **`checkerboards_4_2-4_4_results_final`** and **`checkerboards_4_2-4_4_true_final`**. PNGs are in `../png`, PDFs here. The final results NPZ/JSON retain all original/display arrays and provenance.

These use the aligned-cut/sign-reversed data below, unchanged. Overall titles and bottom explanatory text are removed; plane-group headings, Hard/Soft labels, section labels, axes and cut coordinates remain. Every 4.4 results panel retains `−ΔVp` and colorbars say `Display Vp`, making the sign convention visible even without bottom notes. Cropping uses tight figure bounds. Earlier annotated versions remain separate.

## Common X = Y = 4.5 km and Z = 1.5 km, sign reversal throughout 4.4

```sh
python -B paper/figures/results/tomography/source/make_aligned_phase_checkerboards.py
```

New stems: `checkerboards_aligned45_phase_flipped_results` and `checkerboards_aligned45_phase_flipped_true` (PNG in `../png`, matching PDF here, results NPZ/JSON). Every series uses exactly the same native cuts: **XY Z = 1.5 km; XZ Y = 4.5 km; YZ X = 4.5 km**. The same right-edge crops as before preserve X = 0–20 km and Y,Z = 0–8 km.

**Sign reversal in 4.4 alone suffices**, now on **all three planes** and consistently on True/Hard/Soft: `display_Vp = 10 - raw_Vp` in km/s. 4.2/4.3 keep their original signs. All three displayed True patterns are exactly equal across all series, without any reflection or other phase manipulation. Plot annotations, notes and colorbar identify the values as display-transformed. Original 4.4 physical velocities are not changed. NPZ stores the 27 raw and 27 displayed sections separately. The previous XY-only phase variant is preserved.

## Display-only phase convention: all XY cuts Z = 1.5 km

```sh
python -B paper/figures/results/tomography/source/make_phase_flipped_checkerboards.py
```

New outputs are `checkerboards_z15_phase_flipped_results` and `checkerboards_z15_phase_flipped_true` (PNG in `../png`, PDF here), plus results NPZ/JSON. Right-edge cropping and all XZ/YZ cuts remain unchanged. Every horizontal XY cut is **Z = 1.5 km**. In **4.4 XY only**, all True/Hard/Soft values undergo the same display transformation: `display_Vp = 10 - raw_Vp` in km/s, i.e. `delta_Vp -> -delta_Vp` about a 5 km/s baseline. Both signs are reversed; this is not absolute-valuing negative anomalies.

This yields exactly equal **displayed** True slices while preserving every other plane. The raw XY truth in 4.4 is still opposite to 4.2/4.3. The plot explicitly labels the phase-reversed XY panels, title and colorbar as display values. It must not be presented as an unchanged physical velocity field or as a consistently sign-reversed 3-D model: only the selected horizontal view is transformed. Source arrays, inversion results and previous figures remain untouched. NPZ stores raw and displayed slices separately; JSON records baseline, formula, scope and provenance.

## Alternative: crop right edge, preserve X = 0

```sh
python -B paper/figures/results/tomography/source/make_right_cropped_checkerboards.py
```

Outputs: `../png/checkerboards_right_crop_results_pairs.png` and `../png/checkerboards_right_crop_true_only.png`, plus matching PDFs and results NPZ/JSON here. Existing left-cropped and original figures remain untouched.

Keep **X = 0–20 km in every series**: crop the last 4 km of 4.2/4.3, not their first block. As before, 4.2 retains Y,Z = 0–8 km. The following native cuts yield **exactly identical XY/XZ/YZ True arrays**:

| Series | X cut, km | Y cut, km | Z cut, km |
|---|---:|---:|---:|
| 4.2 | 12 | 6 | 4.5 |
| 4.3 | 6.5 | 4.5 | 4.5 |
| 4.4 | 9 | 3.5 | 1.6 |

Relative to the original full-domain figure, only 4.2 Z (1.6→4.5 km) and 4.3 Y (3.5→4.5 km) change. All displayed axes coincide with native coordinates; the +1 km fault-frame translation of 4.4 is not applied.

**Unlike the earlier left-cropped case, the full 3-D True volumes are not equal:** 4.2/4.3 have the opposite checkerboard phase to 4.4. Choosing the other checkerboard slab along each fixed axis makes the three displayed 2-D patterns equal without reflection, color reversal or velocity modification. This distinction is explicit on the separate True figure and in JSON. Inversion backgrounds are evaluated at their actual stated cuts; no relocation or inversion is rerun.

### Right crop while keeping 4.2 Z = 1.6 km

```sh
python -B paper/figures/results/tomography/source/make_right_cropped_checkerboards.py --keep-42-z16
```

This variant adds `_z16_42` to both output stems and leaves earlier images untouched. Native X/Y cuts and crop bounds are unchanged from the table above. Only horizontal cuts change: **4.2 Z = 1.6 km; 4.3 Z = 1.6 km; 4.4 Z = 4.5 km**. All corresponding True XY/XZ/YZ arrays are still exactly equal. Vertical Hard/Soft slices are identical to the earlier right-crop variant. Full 3-D True phases remain different. Keeping both 4.2 and 4.4 at Z = 1.6 km would not match XY with these same crop bounds and no reflection/color reversal.

## Split True and inversion-results figures

```sh
python -B paper/figures/results/tomography/source/make_checkerboard_pairs.py
```

This additional layout reuses exactly the crops and cuts below, without overwriting any previous figures:

- `../png/checkerboards_matched_true_only.png`: shared True slices shown once, in **XY / XZ / YZ** order.
- `../png/checkerboards_matched_results_pairs.png`: **three rows (4.2, 4.3, 4.4)** and **six columns (XY Hard, XY Soft, XZ Hard, XZ Soft, YZ Hard, YZ Soft)**.
- Matching PDFs are in this directory. `checkerboards_matched_results_pairs.npz` stores all 27 source section arrays; its JSON records layout, crop/cut metadata, checkpoint provenance and hashes.

Both figures use the same 4.6–5.4 km/s scale. All panels retain equal metric scale; YZ columns are narrower. Actual crop-local cut coordinates are printed above the inversion panels, not above the shared True view, because corresponding True patterns are exactly equal despite different fixed coordinates. Thin original-checkerboard boundary lines remain on Hard and Soft. Every displayed inversion is the same selected V3 checkpoint as before.

## Feasibility and required crops

All three original True fields use 4 km cubic checkerboard blocks and the same velocities, 4750 and 5250 m/s. Their grids are 6×3×3 (4.2), 6×2×2 (4.3) and 5×2×2 (4.4). The corner phase in 4.2/4.3 is opposite to 4.4.

**Cropping only 4.2 is insufficient**: 4.3 remains one block longer than 4.4 along X. Removing the first 4 km X block from both 4.2 and 4.3 matches both length and phase. Removing the final Y and Z blocks from 4.2 then yields three **exactly identical 5×2×2 True volumes**, without changing or rescaling any velocities.

| Series | Native X range, km | Native Y range, km | Native Z range, km |
|---|---|---|---|
| 4.2 | 4–24 | 0–8 | 0–8 |
| 4.3 | 4–24 | 0–8 (unchanged) | 0–8 (unchanged) |
| 4.4 | 0–20 (unchanged) | 0–8 (unchanged) | 0–8 (unchanged) |

Hard and Soft are cropped identically to their corresponding True field. Crop boundaries coincide with native cell faces in every array. The previous selected V3 checkpoints are retained; no inversion is rerun.

## Small plane changes

Only these plane levels change relative to the main figure:

- **4.2: Y = 6 → 3.5 km**.
- **4.3: Z = 4.5 → 3.5 km**.

Other native planes remain unchanged. Due to periodic truth, distinct fixed coordinates inside equivalent checkerboard phases can produce identical slices.

| Series | Native cuts (X, Y, Z), km | Display crop-local cuts (X, Y, Z), km |
|---|---|---|
| 4.2 | 12, 3.5, 1.6 | 8, 3.5, 1.6 |
| 4.3 | 6.5, 3.5, 3.5 | 2.5, 3.5, 3.5 |
| 4.4 | 9, 3.5, 1.6 | 9, 3.5, 1.6 |

Plot axes are reset to **crop-local** coordinates, giving 20×8×8 km domains. In particular, 4.4 is no longer displayed with the main figure's +1 km fault-frame X offset. These are registered periodic model patterns, **not a common geographic/acquisition frame**.

Rows remain True / Soft / Hard; columns XY / YZ / XZ. Checkerboard lines and the shared 4.6–5.4 km/s scale are retained. There is no smoothing, interpolation, stretching, velocity modification, or error-based selection.

## Validation and interpretation

The script checks exact equality of all three cropped native True volumes **and each corresponding True section array**. Original model hashes, common starts and original full-volume RMSE are verified by the existing loader before cropping. Stored checkpoint RMSE values continue to refer to the original full domains, not these crops.

The NPZ contains 27 section arrays and the three cropped native True volumes. JSON stores native crop bounds, both native and displayed cut coordinates, source hashes and validation results.

Equal True pictures do not make station/event geometry, observations, initial models or inversion resolution equivalent. Cropping removes some portions of the inversion results and is only a separate controlled visualization.
