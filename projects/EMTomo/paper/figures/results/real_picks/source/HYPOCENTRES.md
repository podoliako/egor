# Hard / Soft projected hypocentre shifts

## Sparse variant: one event per 1 km projected square

```sh
python -B paper/figures/results/real_picks/source/make_hypocentre_projection.py --cell-m 1000 --max-per-cell 1
```

Selects **97 events in 97 occupied XZ squares**. Outputs use the suffix `_1000m_max1`, with optional `_side_by_side`, leaving the earlier 500 m / three-event files unchanged. The cap applies to catalog positions; final MAP endpoints are all retained. Selection uses the same stable event-ID hash rule, independently of displacement and method. Grid spacing is 1 km.

## All-Y projection without velocity background

```sh
python -B paper/figures/results/real_picks/source/make_hypocentre_projection.py
```

New outputs: `../png/real_picks_hard_soft_hypocentres_projection.png` and `_side_by_side.png`, with corresponding PDFs, NPZ, CSV and JSON in `source/`. Earlier section figures are preserved.

All 3471 retained catalog events are eligible, with **no Y/slab filter**. Catalog XZ coordinates are binned into **500 × 500 m projected squares**, anchored at X = Z = 0 using floor division. Up to **three events per projected catalog square** are selected, giving **726 events in 312 occupied squares**. This is a 2-D projection cap, not a 3-D voxel cap: multiple Y depths cannot overpopulate one projected catalog square.

Selection ranks event IDs by SHA-256 and takes the first three within each square. It is deterministic, independent of input row order, event displacement and method. Both panels use the same selected IDs. It is a spatially thinned visualization, not a representative statistical sample of all events; long-shift outliers are not trimmed. Fitted endpoints are not independently capped and may cluster in the same square; displacement lines may cross. No coordinates are jittered or moved for presentation.

White circles are catalog positions; orange circles are final-model MAP locations, including for Soft. Straight connecting lines show exact XZ projected displacements. No background velocities, sensitivity masks, reference model or smoothing are used. Coordinates, selected IDs, catalog cell indices and segment endpoints are exported for validation. The same verified final-model MAP loader described below is used.

## Earlier vertical sections with velocity background

From `projects/EMTomo`:

```sh
python -B paper/figures/results/real_picks/source/make_hypocentre_sections.py
```

## Files

- `../png/real_picks_hard_soft_hypocentres_xz.png`: stacked Hard/Soft panels.
- `../png/real_picks_hard_soft_hypocentres_xz_side_by_side.png`: side-by-side panels.
- Matching PDFs, NPZ section/coordinate/segment data, CSV selected-event coordinates, and JSON provenance are in this `source/` directory.

Other figures are unchanged. There are **no sensitivity/resolution masks, no faded velocity regions, no smoothing and no stations**. Velocities are the final native 500 m Hard/Soft cells, with a common scale of 3–7 km/s.

## Selection and projection

The plane is XZ at **Y = 5.5 km**. Select by **catalog position only**, using the inclusive criterion `abs(catalog_y - 5500) <= 1000` in metres. This yields **938 of 3471** events. Both panels use exactly the same ordered event IDs; no random sampling or outlier removal is performed.

White points are catalog locations. Yellow points are saved **MAP locations in the final velocity model**, including for Soft: not the posterior mean and not hypotheses from the last pre-update inversion iteration. Thin straight lines connect the two positions of each event. Both endpoints are projected orthogonally onto XZ, with no displacement rescaling. These lines are displacements, not seismic rays; displacement normal to the section is not visible.

All fitted endpoints of the selected catalog events are retained, even if fitted Y is outside the slab: **61 Hard** and **81 Soft** endpoints. Model colors remain the values exactly on the Y = 5.5 km plane, not sampled at each event's Y.

## Provenance and checks

The existing `make_geysers_real_location_shifts.load_inputs` validates training-only MAP prediction hashes, binds predictions to final velocity model hashes, verifies the original actual38676 training mask, and aligns predictions and catalog by unique event IDs. Predictions come from `paper/exp_data/geysers_real_holdout_latest/{hard,soft}_predictions.npz`. The loader also checks the existing initial-model prediction files, but they are not plotted.

The MAP protocol uses 50 candidates and train-only origin-time fitting with the same likelihood settings for both velocity models. No relocation, inversion, FMM calculation or holdout scoring is run here. The background velocity array is checked against the exact final model used for its predictions. Consumed-file hashes are rechecked before outputs are written.

Catalog coordinates are a diagnostic baseline, **not ground truth**; displacement lengths are not location-error estimates. Depth is measured positive downward from the model box top; catalog sea-level depth is converted with `Z_m = 1166 + 1000 * depth_km`. Exact vertical-datum compatibility is assumed.
