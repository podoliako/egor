# Accepted checkerboard illustrations

Three illustrations are currently accepted. Each has a canonical PNG preview,
PDF/SVG for publication, a Markdown caption, and JSON source/checkpoint provenance.

| Illustration | Canonical stem | Rebuild script (relative to `paper/`) |
|---|---|---|
| nx6 model and acquisition geometry | `checkerboard_nx6_geometry` | `make_checkerboard_geometry.py` |
| nx4/nx6/nx8 velocity comparison: vertical Y=10 km and horizontal Z=5 km | `checkerboard_nx4_nx6_nx8_comparison` | `make_checkerboard_comparison.py` |
| nx6 hypocentres: final EM, final DD, initial DD at true Y=10.2 km | `checkerboard_nx6_locations_y10p2km` | `make_checkerboard_locations.py` |

A fourth figure is under review: full-catalogue 3D location-error ECDFs for all
three grids, `checkerboard_location_error_ecdf.*`, rebuilt with
`python paper/paper_scripts/make_checkerboard_location_ecdf.py`. Its CSV contains all per-event
errors, and its Markdown/JSON files describe statistics and validation.

From the EMTomo project directory:

```sh
python paper/paper_scripts/make_checkerboard_geometry.py
python paper/paper_scripts/make_checkerboard_comparison.py
python paper/paper_scripts/make_checkerboard_locations.py
```

Dependencies are NumPy, SciPy and Matplotlib from the project environment.
The scripts read existing saved experiment inputs and run outputs; they do not
rerun inversions or generate replacement observations. Exact run IDs and data
paths are in the scripts and JSON files. The source data under ignored `runs/`
must be available: artwork can be viewed without those data, but cannot be
rebuilt from a source-only checkout. Output files are overwritten in place.

Style is STIX text/math, inward black/white mitred frames, native piecewise
constant EM cells and trilinear DD interpolation. Velocity colours are white at
4.75 km/s and red at 5.25 km/s, with grey at 4 km/s and purple at 6 km/s in the
comparison. The geometry uses the white-to-red subsection. The location figure
mixes velocity colours with 68% white and uses a separate white/blue/dark-blue
0–4000 m full 3D location-error scale. See the individual captions for projection
and checkpoint caveats.

The following retained aliases are exact PNG copies of the accepted figures,
not additional variants. They preserve image links used during review:

- `checkerboard_geometry_with_initial.png`
- `checkerboard_comparison_sections_restored.png`
- `checkerboard_nx6_locations_blue_preview.png`

Rebuild scripts update canonical files, not the aliases. If an accepted figure
is revised, refresh its alias from the matching canonical PNG.

Older `checkerboard_y25km.png` and `flat4_*.png` are separate figures still
referenced in the article, not intermediate versions of these illustrations.

## Shared spherical-anomaly truth

`spherical_anomalies_truth.{png,pdf,svg,json}` shows the native saved 125-m
truth shared by the artificial and Geysers geometries. Rebuild with
`python -B paper/paper_scripts/make_spherical_anomalies_truth.py`. The script verifies that
both source velocity arrays agree and never invokes a forward/inverse solver.

The surface is homogeneous (3.8 km/s). The Y=2.4 km section intersects spheres
1, 3, and 5; Y=4.8 km intersects spheres 2, 4, and 6, all within half a radius
of their center planes. Slices use native piecewise-constant cells, without
smoothing or slab averaging; their cell bounds are recorded in JSON. There
are no station/event overlays. STIX typography, 2-km black/white frames and
the checkerboard velocity-color anchors are retained, extending the scale
to 3.5–6.5 km/s without clipping. JSON records source and script hashes.

## Spherical-anomaly acquisition geometries

`spherical_anomalies_geometry.{png,pdf,svg,json}` compares artificial (52 stations,
1920 events) and Geysers (26 stations, 3776 events) geometry in two rows. Columns
show XY, X-depth and Y-depth projections. Central planes Y=3.5 km and X=10 km
are marked in the plan views; all saved positions are projected, not filtered
by distance to these planes. Exact coincident projections are drawn once.
The uniform grey background is not a velocity field. Station elevations are
retained. Fonts, frame and markers follow the checkerboard geometry style.

Rebuild with `python -B paper/paper_scripts/make_spherical_anomalies_geometry.py`. The script
checks counts, coordinate bounds and projection multiplicities. Source hashes,
marker counts and depth ranges are recorded in JSON. No solvers are invoked.

## Spherical-anomaly reconstruction comparison

`spherical_anomalies_comparison.{png,pdf,svg,json}` has six rows: artificial
Y=2.4/4.8 km and Z=2.5 km, then the same three sections for Geysers.
The five columns are EMTomo oracle, hard,
soft, tomoDD, tomoDD oracle. Rebuild with
`python -B paper/paper_scripts/make_spherical_anomalies_comparison.py`.

EMTomo uses the matched-geometry study's 12-update hard top-1 and soft top-10
runs, both with 15 candidates and dmin=1. tomoDD uses the completed corrected
horizontal-guard dd1-d40-s5 runs (three joint iterations, 1-km initial-coordinate
noise). No forward/inverse solver is invoked. Input/checkpoint hashes,
completion markers, native grids, common truth and common DD oracles are checked.

The EM oracle is the arithmetic velocity mean of each 8x8x8 block of 125-m
truth cells, not inverse mean slowness. DD retains its saved native trilinear
velocity oracle. Both oracles are geometry-independent, not per-slice fits.
Exact constant-Y and constant-Z planes are sampled at 25-m pixel centers, without
added smoothing or station/event overlays. The vertical shared scale extends
to 7 km/s to avoid clipping the 6.62-km/s DD maximum. Existing velocity-color
anchors remain unchanged. Panels use compact explicit positioning, with shared
column headings and X labels only along the bottom; the vertical colorbar is
centered at one third of the figure height. JSON records all sources, settings
and slice ranges.

Velocity and location tables are generated from the same selected runs by
`python -B paper/paper_scripts/evaluate_spherical_anomalies.py`. The report is
`spherical_anomalies_metrics.json`; `--check` fully recomputes and verifies it.
Velocity RMSE uses all 501,760 native 125-m truth-cell centers and each of six
nonzero sphere supports. Location P90 uses full-catalog 3D errors, maximum-weight
hypotheses for soft EMTomo, and saved pre-last-update E-step positions.

## tomoDD regional-guard follow-up (2026-10-27)

Separate diagnostic figures, not replacements for the accepted checkerboard
illustrations or the earlier tomoDD images:

- `tomodd_dd1_artificial_geysers_z1p5km_guard_fixed.{png,pdf,json}`:
  oracle, corrected artificial geometry, corrected Geysers geometry.
- `tomodd_dd1_artificial_geysers_z1p5km_guard_fixed_before_after.{png,pdf,json}`:
  oracle, artificial before/after, Geysers before/after.

Rebuild from the retained runs with
`python -B paper/paper_scripts/make_tomodd_dd1_geometry_comparison.py --guard-fixed`, adding
`--with-before` for the five-panel comparison. Exact z=1.5 km, native trilinear
Vp, shared colour scale, no dashed outlines or added smoothing. The JSON files
retain RMSE, source hashes and solver identity. See
`paper/tomodd-guard-fix-comparison.md` for matched protocol and interpretation.
