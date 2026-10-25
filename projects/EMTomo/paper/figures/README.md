# Accepted checkerboard illustrations

Three illustrations are currently accepted. Each has a canonical PNG preview,
PDF/SVG for publication, a Markdown caption, and JSON source/checkpoint provenance.

| Illustration | Canonical stem | Rebuild script (relative to `paper/`) |
|---|---|---|
| nx6 geometry and common homogeneous initial model | `checkerboard_nx6_geometry` | `make_checkerboard_geometry.py` |
| nx4/nx6/nx8 velocity comparison: vertical Y=10 km and horizontal Z=5 km | `checkerboard_nx4_nx6_nx8_comparison` | `make_checkerboard_comparison.py` |
| nx6 hypocentres: final EM, final DD, initial DD at true Y=10.2 km | `checkerboard_nx6_locations_y10p2km` | `make_checkerboard_locations.py` |

From the EMTomo project directory:

```sh
python paper/make_checkerboard_geometry.py
python paper/make_checkerboard_comparison.py
python paper/make_checkerboard_locations.py
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
