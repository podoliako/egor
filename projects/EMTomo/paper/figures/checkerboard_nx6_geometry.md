# Checkerboard nx6: model and acquisition geometry

Suggested caption:

**Checkerboard model and acquisition geometry.** (a) Plan view of the surface velocity model, showing 98 stations (light-grey triangles) and horizontal projections of 1,200 events (white stars). Solid grey lines mark the vertical sections at Y = 10 km and X = 6 km. (b) X–depth section at Y = 10 km. (c) Y–depth section at X = 6 km. (d) Common homogeneous initial velocity model (5 km/s) for EMTomo and tomoDD, shown as an X–depth section at Y = 10 km without geometry markers. This initial field is also shared by nx4 and nx8. Stations and events are orthogonally projected onto each panel; coincident positions are drawn once. An event marker represents six events in (a), ten in (b), and twenty in (c). A station marker represents seven stations in (b) and fourteen in (c). The 24 × 12 × 12 km domain comprises 6 × 3 × 3 alternating 4 km blocks with P-wave velocities of 4.75 and 5.25 km/s. White denotes 4.75 km/s and red (#e32400) denotes 5.25 km/s; intermediate velocities use a continuous white-to-red scale. Depth is positive downwards. The alternating black and white frame lies entirely inside the model, with mitred corners and nominal 2 km segments along its outer edge. On vertical sections, station triangle bases are anchored at zero depth.

The geometry is projected, not restricted to a slab around either section. No coordinate jitter, depth-dependent fading, or hidden event selection is applied.

- PDF/SVG: vector artwork for publication.
- PNG: 300 dpi preview.
- JSON: source hashes, counts, and projection conventions.
- Rebuild from the EMTomo project directory: `python paper/make_checkerboard_geometry.py`.
