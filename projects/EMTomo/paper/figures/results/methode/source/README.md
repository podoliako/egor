# 2-D method illustration

Rebuild from the EMTomo root:

```sh
python -B paper/figures/results/methode/source/make_method_2d.py
```

- `../png/method_2d.png`: preview; the PNG directory contains only PNG files.
- `method_2d.pdf`: version for possible article inclusion.
- `method_2d_data.npz`: coarse/fine velocities, computed travel-time field, station, candidate locations, traced paths, and coarse-cell ray lengths.
- `method_2d.json`: illustrative parameters and calculation provenance.

## Panels

1. **Velocity and computational grids.** Thick boundaries delimit homogeneous velocity cells. Thin lines subdivide each cell into q × q subcells in this 2-D illustration. Every subcell inherits its parent velocity; h and h/q mark the two spacings.
2. **Station travel-time field.** An actual second-order scikit-fmm calculation in the displayed heterogeneous medium. The off-grid station is seeded by a local homogeneous circle (the 2-D analogue of the source sphere in `eikonal.py`). Contours are first-arrival times in seconds. One station field serves all three candidate locations.
3. **Ray tracing.** Each off-grid candidate is traced toward the station along the normalized negative gradient of that same field. Arrows point toward the station. The same velocity-cell boundaries are overlaid, and a gold path segment shows a candidate–station ray's contribution to one cell's path length coefficient G.

This is a **2-D analogue**, not a slice of a production 3-D experiment and not a reconstruction result. Bilinear gradient interpolation replaces 3-D trilinear interpolation. All three rays must reach the station, travel times must decrease along the traced steps, and cellwise accumulated lengths must sum to the total polyline length. The script computes the rays; it does not draw arbitrary curved connectors.

The example parameters are chosen for illustration only. The three stars are alternative candidate locations, not three independently observed events. No candidate probabilities are assigned in this figure. The direct ray-integral times need not exactly equal the fast-marching field samples because tracing and integration are numerical approximations.

No article sources, saved experiments, inversion results, or published PDF are changed by this script.
