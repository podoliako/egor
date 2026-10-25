# Checkerboard reconstruction comparison

Suggested caption:

**Checkerboard velocity reconstructions.** Columns show the true model, EMTomo reconstruction, tomoDD reconstruction, and the saved tomoDD representation oracle. The upper three rows show vertical sections at Y = 10 km for nx = 4, 6, and 8; the lower three rows show horizontal sections at depth Z = 5 km in the same nx order. The physical domain is 24 × 12 × 12 km. EMTomo panels use the final models after 12 EM updates with coverage_damping_power = 0, lambda_reg = 1, hard candidate selection, 15 candidates, and top-1 weights. tomoDD panels use the saved final models after three joint source/velocity iterations. Both methods and all three nx values share a homogeneous 5 km/s initial velocity model, shown in the geometry figure; this is checked against all six saved initial arrays. tomoDD starts from hypocentres independently perturbed with a 1 km per-coordinate Gaussian standard deviation (seed 42); EMTomo does not use those initial hypocentres. The oracle is a saved model-space reference, not an inversion result. True and EMTomo fields are piecewise constant; tomoDD and its oracle use native trilinear velocity interpolation. No stations or events are overlaid. Depth is positive downwards on vertical sections; Y increases upwards on horizontal sections. The inset black-and-white frames have nominal 2 km segments along their outer edges.

## Colour scale

A single shared linear velocity normalization from 4 to 6 km/s is used, without clipping or panel-specific rescaling. The horizontal colourbar between the two section groups has uniformly spaced ticks every 0.5 km/s. All displayed values lie within this range, checked before plotting. Colour anchors are positioned at:

| V (km/s) | Colour |
|---:|---|
| 4.00 | #7a7a7a |
| 4.75 | #ffffff |
| 5.25 | #e32400 |
| 6.00 | #61177c |

The unequal spacing of the colour anchors is intentional: the true velocities remain white and red, matching the geometry figure. The mapping is not a quantitative perceptually uniform colour scale.

## Reproduction

Run `python paper/make_checkerboard_comparison.py` from the EMTomo project directory. The figure is sampled at 25 m pixel centres in X and depth (vertical sections at exactly Y = 10 km), and in X and Y (horizontal sections at exactly Z = 5 km). There is no averaging or slab selection. For the piecewise-constant EM field, a section exactly on a cell boundary uses the positive-coordinate cell. DD X/Y coordinates are translated from the centred coordinate system using the saved grid metadata. PDF and SVG contain rasterized model fields and vector annotations/frames. PNG is 300 dpi.

Exact run directories, model/checkpoint hashes, sampling conventions and slice ranges are recorded in `checkerboard_nx4_nx6_nx8_comparison.json`. The script checks original input hashes, EM completion markers, experiment identity, DD truth equality, model dimensions and finite values before writing the figure.
