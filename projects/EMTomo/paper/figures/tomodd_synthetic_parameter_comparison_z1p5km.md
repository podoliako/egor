# Artificial-geometry tomoDD parameter comparison

Exact depth: 1.5 km. All inversions: three joint iterations, initial coordinate sigma 1 km, seed 42.
All four predeclared cases are shown; no checkpoint or case selection.

| Case | Abs/DD amplitude | Damping | Smoothing | Slice RMSE, m/s | Volume RMSE, m/s | Location RMS, m |
|---|---|---:|---|---:|---:|---:|
| 3D L2 oracle | — | — | — | 183.99 | 142.86 | — |
| baseline | 1/0.1 | 40 | 5/5/5 | 144.62 | 240.22 | 207.95 |
| dd03-d40-s5 | 1/0.3 | 40 | 5/5/5 | 159.17 | 264.64 | 205.43 |
| dd1-d40-s5 | 1/1 | 40 | 5/5/5 | 250.34 | 404.07 | 277.88 |
| dd03-d80-s10 | 1/0.3 | 80 | 10/10/10 | 209.71 | 257.53 | 276.62 |
| dd03-d20-s2 | 1/0.3 | 20 | 2/2/2 | 169.23 | 296.34 | 216.23 |

Reproduce: `python -B paper/make_tomodd_sensitivity_comparison.py`

The 3D oracle is not a slice-specific error lower bound. No inverse/forward solver is run by this plotter.
