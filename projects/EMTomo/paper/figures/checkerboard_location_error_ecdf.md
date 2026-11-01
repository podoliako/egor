# Full-catalogue checkerboard location-error ECDF

Suggested caption:

**Cumulative distributions of three-dimensional hypocentre error.** Each curve shows the fraction of all 1,200 events whose Euclidean 3D location error is at most the horizontal-axis value. Colours identify nx = 4, 6, and 8; solid curves denote EMTomo and dashed curves denote tomoDD. EMTomo uses the latest saved hard top-1 E-step locations from iter_11 in the coverage-damping-power-zero runs; these precede the final twelfth velocity update and are not a new relocation in the final model. tomoDD uses the saved final locations after three joint iterations. All events have equal weight. Curves are exact right-continuous empirical step functions, with no smoothing, spatial selection, or tail clipping. A higher curve at a fixed error indicates a larger fraction of events within that error.

The linear horizontal axis spans 0–2250 m and includes every saved error. Quantile summaries below use NumPy's linear interpolation and are descriptive values, rather than additional smoothing of the ECDF itself.

| nx | Method | Median (m) | 90th percentile (m) | 95th percentile (m) | RMS (m) |
|---:|---|---:|---:|---:|---:|
| 4 | EMTomo | 190.8 | 505.9 | 639.7 | 315.0 |
| 4 | tomoDD | 232.5 | 599.2 | 811.4 | 379.9 |
| 6 | EMTomo | 130.3 | 373.4 | 485.1 | 237.6 |
| 6 | tomoDD | 240.0 | 588.1 | 762.5 | 372.0 |
| 8 | EMTomo | 137.0 | 402.5 | 504.2 | 254.1 |
| 8 | tomoDD | 240.2 | 609.1 | 755.4 | 380.5 |

Rebuild from the EMTomo directory with `python paper/make_checkerboard_location_ecdf.py`. PNG is 300 dpi; PDF and SVG are vector graphics. The CSV contains all 3,600 nx/event pairs with one EM and one DD error per row. JSON contains exact summary values, source/checkpoint hashes, CSV hash, and a row-index/event-ID/weights-file checksum for each EM run.

Validation checks all event identities and catalogue coordinates, all 12 EM completion markers per run, hard top-1 weights, saved grid dimensions, coordinate conversions, and finite values. DD catalogue RMS values are independently checked against the saved run results. ECDF tie handling and inclusive error thresholds were checked on small known arrays.
