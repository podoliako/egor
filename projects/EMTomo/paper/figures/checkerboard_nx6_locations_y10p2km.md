# nx6 event-location comparison at Y = 10.2 km

Suggested caption:

**Saved event-location estimates for the nx6 checkerboard.** (a) Final EMTomo, (b) final tomoDD, and (c) initial tomoDD hypocentres. Backgrounds show the respective final velocity fields in (a,b) and the saved homogeneous initial velocity field in (c), at Y = 10.2 km. Velocity colours are mixed with 68% white to emphasize location markers; the velocity colorbar uses the identical muted palette. The same 120 events with true Y = 10.2 km are displayed in all three panels. White stars indicate true positions; circles indicate saved positions; thin grey segments connect each circle to its corresponding star. Positions and connecting segments are projected onto the X–depth plane. Circle colours encode full three-dimensional Euclidean location error, including the omitted Y component, on one shared 0–4000 m scale. They do not represent candidate weights or projected segment lengths. All 98 surface stations are projected onto the section and drawn as 14 distinct light-grey triangles, with bases at zero depth. No filtering is applied to estimated or initial positions. Model fields retain their native piecewise-constant (EMTomo) and trilinear (tomoDD) representations.

The EMTomo velocity field is the final model after 12 updates with coverage-damping power zero. Its locations are the saved E-step locations from iter_11, immediately before the twelfth velocity update, not a new relocation in the final post-update model. Hard top-1 assignment gives one saved position per event with weight one. tomoDD uses three joint iterations; initial positions are the actual saved perturbed starting hypocentres, not newly generated noise. Event identity is matched by source CSV row for EMTomo and by numeric event ID for tomoDD; the geographic-to-Cartesian conversion is validated against every saved true coordinate.

Full 3D errors for this selected 120-event row:

| Method / checkpoint | RMS (m) | Median (m) | Maximum (m) |
|---|---:|---:|---:|
| EMTomo final saved E-step | 255.4 | 150.9 | 922.8 |
| tomoDD final | 350.9 | 257.9 | 844.7 |
| tomoDD initial | 1630.9 | 1372.6 | 3865.5 |

These are subset location statistics, not full-catalog scores. The shared error gradient is white (#ffffff) at 0 m, blue (#0085ca) at 2000 m, and dark blue (#00239c) at 4000 m, with linear interpolation. The common scale includes all initial errors without clipping; consequently final-error colour differences are less pronounced than with the former 0–1000 m scale.

Rebuild from the EMTomo directory with `python paper/make_checkerboard_locations.py`. PDF/SVG retain vector markers, lines and frame, with rasterized background fields. PNG is 300 dpi. Exact coordinates, errors in metres, source/checkpoint paths and hashes are stored in the accompanying JSON.
