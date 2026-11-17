# Acquisition geometry drafts

Rebuild from the EMTomo root:

```sh
python -B paper/figures/results/geoms/source/make_geometries.py
```

The sibling `../png/` directory contains **only PNG files**: one combined three-row figure and three individual row previews. The script, PDF, exported coordinate CSVs, and SHA-256 provenance remain in this directory. No article sources or inversion artifacts are changed.

## Plotted geometries

| Row | Results section | Stations | Events | Domain (km) |
|---|---|---:|---:|---|
| a | 4.2 one-sided | 18 | 1200 | 24 × 12 × 12 |
| b | 4.3 two station clusters | 22 | 600 | 24 × 8 × 8 |
| c | 4.5 vertical fault: smallest Geysers domain | 26 | 3776 | 20 × 7 × 7 |

Full coverage (4.1) is omitted because it already has a separate article figure. Only the smallest Geysers domain is plotted. This is an illustration of the real-derived geometry, **not** a claim that all Geysers tests use identical event selections and boundaries.

## Layout

Three columns: plan view with two plane traces; long-side X–depth projection; short-side Y–depth projection. All stations and events in each plotted experiment are shown, with exact coincident 2-D projections drawn once. There is no distance-to-plane filtering, random jitter, event subsampling, or velocity background. The traces pass through the domain center. Axes retain equal horizontal/vertical metric scale, native domain limits, and positive-down depth.

## Geometry audit retained in source files

Sections 4.4 and 4.6 have exactly identical ordered station/event coordinates and domain dimensions: 26 stations, 3465 events, 20 × 8 × 8 km. Real picks (4.7) use 31 stations, 3471 events, 23 × 11 × 7 km. These variants are no longer drawn.

Matching coordinates agree to floating-point precision after these X translations:

- Sections 4.4/4.6 to 4.5: +1 km. The same 26 stations; 4.5 adds 311 events.
- Sections 4.4/4.6 to 4.7: +4 km. Real picks add five stations and six events.
- Section 4.5 to 4.7: +3 km. Real picks have five more stations and 305 fewer events.

Station elevations are preserved. Catalog/station vertical-datum compatibility has not been independently established. Catalog positions are not independently verified earthquake truth.

`geometry_inventory.json` retains all six source geometries and ID-matched coordinate comparisons, and separately lists plotted and omitted geometry keys. Exported coordinate CSVs for omitted geometries are retained as audit data. The real-pick event selection comes from `emtomo/arrivals.npz`, not from the entire unfiltered catalog.
