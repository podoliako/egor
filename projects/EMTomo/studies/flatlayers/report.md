# EMTomo on `flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001`

52 stations, 1920 events. Forward convergence r→2r: `{"coarse_refinement": 2, "fine_refinement": 4, "max_abs_difference_s": 0.007395373304608377, "rms_difference_s": 0.0015582024740264542, "max_absolute_time_difference_s": 0.006678055480201062, "rms_absolute_time_difference_s": 0.0015864292543260805}`

- RMS: m/s against the slowness-averaged truth on inversion cells; cycle 0 = start model.
- Spheres A–F: mean(final − start)/mean(truth − start) over cells within 0.6 R.
- Wedge: cells whose layer differs between the flat start interface and the true dipping one (spheres excluded). gain = projection of the model change on the needed change (1 = fully corrected, 0 = untouched); corr = correlation of the two.
- Background: RMS outside spheres and wedge, above the deepest event.
- Boundary depth: where the y-median profile crosses the mid velocity of the last two layers.

| run | RMS start → best @cycle → final | spheres A–F (final) | wedge gain / corr @best | wedge gain / corr final | background start → final | hypo error m: median first → last, p90 | s/cycle |
|---|---|---|---|---|---|---|---:|
| fixed-c1000 | 111 → 66 @12 → 66 | 1.49 nan 1.26 0.86 1.23 nan | 0.70 / nan | 0.70 / nan | 98 → 73 | 0 → 0, 0 | 10 |
| hard-c1000 | 111 → 84 @8 → 85 | 1.27 nan 1.17 0.98 1.08 nan | 0.50 / nan | 0.59 / nan | 98 → 100 | 151 → 110, 235 | 54 |
| fixed-c500 | 140 → 126 @2 → 159 | 1.12 1.08 1.01 1.02 1.25 1.00 | 0.31 / 0.00 | 1.03 / 0.00 | 155 → 188 | 0 → 0, 0 | 43 |
| hard-c500 | 140 → 128 @3 → 158 | 1.11 1.13 0.95 1.11 1.13 0.86 | 0.06 / -0.00 | 0.41 / -0.00 | 155 → 184 | 143 → 96, 210 | 72 |

## Dipping boundary depth (km) by x, final model

| run | x=0.5 | x=2.5 | x=4.5 | x=6.5 | x=8.5 | x=10.5 | x=12.5 | x=14.5 | x=16.5 | x=18.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **truth (cell average)** | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 |
| true interface | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 |
| fixed-c1000 | 2.99 | 3.24 | 3.38 | 3.19 | 3.31 | 3.36 | 3.18 | 3.28 | 3.26 | 3.13 |
| hard-c1000 | 3.00 | 3.17 | 3.37 | 3.16 | 3.26 | 3.38 | 3.31 | 3.38 | 3.21 | 3.07 |
| fixed-c500 | 3.49 | 3.47 | 3.42 | 3.36 | 3.41 | 3.34 | 3.32 | 3.40 | 3.47 | 3.45 | 3.53 | 3.50 | 3.38 | 3.46 | 3.43 | 3.44 | 3.43 | 3.40 | 3.38 | 3.47 |
| hard-c500 | 3.50 | 3.47 | 3.50 | 3.48 | 3.51 | 3.47 | 3.38 | 3.45 | 3.46 | 3.44 | 3.54 | 3.59 | 3.57 | 3.52 | 3.43 | 3.44 | 3.47 | 3.45 | 3.50 | 3.48 |

## RMS (m/s) by cycle

| run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-c1000 | 111 | 90 | 81 | 77 | 74 | 73 | 71 | 70 | 69 | 68 | 67 | 67 | 66 |
| hard-c1000 | 111 | 99 | 92 | 88 | 86 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 85 |
| fixed-c500 | 140 | 131 | 126 | 126 | 129 | 133 | 137 | 142 | 146 | 149 | 153 | 156 | 159 |
| hard-c500 | 140 | 133 | 129 | 128 | 130 | 133 | 136 | 140 | 143 | 147 | 151 | 155 | 158 |

## Data misfit (s): RMS of centred residuals, every 4th event

Columns are the models entering each cycle (0 = start). 'truth' = the cell-averaged true model on the same grid at the true hypocentres; going below it means fitting noise. For EM runs relocation also absorbs residuals, so there it is only a reference level.

| run | truth | c0 | c1 | c2 | c4 | c8 | c11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-c1000 | 0.0235 | 0.0278 | 0.0247 | 0.0237 | 0.0231 | 0.0228 | 0.0228 |
| hard-c1000 | 0.0235 | 0.0252 | 0.0233 | 0.0226 | 0.0221 | 0.0218 | 0.0218 |
| fixed-c500 | 0.0223 | 0.0275 | 0.0238 | 0.0226 | 0.0217 | 0.0214 | 0.0213 |
| hard-c500 | 0.0223 | 0.0248 | 0.0227 | 0.0218 | 0.0211 | 0.0209 | 0.0209 |
