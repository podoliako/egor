# EMTomo on `dipping6spheres_240x120x120_72st_1000ev_sobol_minz5_noise01_pick005_seed42`

72 stations, 1000 events.
Forward convergence (refinement r→2r): `{"coarse_refinement": 2, "fine_refinement": 4, "max_abs_difference_s": 0.15109853015792396, "rms_difference_s": 0.028762186444758805, "max_absolute_time_difference_s": 0.11702567245339424, "rms_absolute_time_difference_s": 0.022628346634128473}`

RMS is m/s against the slowness-averaged truth on inversion cells; cycle 0 is the start model.
Sphere recovery = mean(final − start) / mean(truth − start) over inversion cells within 0.6 R.

| run | RMS start | RMS final | best RMS @ cycle | sphere recovery A–F | median hypo error km, first → last cycle | s / cycle |
|---|---:|---:|---:|---|---:|---:|
| hard-ncand1 | 140.9 | 142.4 | 136.3 @ 2 | 1.06 1.18 1.07 0.36 0.48 0.42 | 2.28 → 1.26 | 164 |
| hard-top1 | 140.9 | 142.7 | 136.4 @ 2 | 1.07 1.17 1.07 0.36 0.50 0.42 | 2.28 → 1.31 | 166 |
| hard-top1-gradstart | 190.6 | 175.9 | 175.9 @ 12 | 1.17 1.16 0.97 0.40 0.57 0.54 | 3.62 → 1.41 | 165 |
| hard-top1-lam02 | 140.9 | 128.1 | 127.0 @ 7 | 1.07 1.09 1.09 0.25 0.54 0.30 | 2.28 → 1.19 | 180 |
| hard-top1-step01 | 140.9 | 151.0 | 138.6 @ 1 | 0.87 0.93 0.93 0.34 0.45 0.41 | 2.28 → 1.55 | 181 |
| hard-top1-sub6 | 140.9 | 147.1 | 136.9 @ 1 | 1.09 1.20 1.10 0.37 0.58 0.47 | 2.22 → 1.32 | 35 |
| soft-top3 | 140.9 | 142.6 | 136.4 @ 2 | 1.06 1.18 1.07 0.36 0.50 0.42 | 2.28 → 1.29 | 171 |
| soft-top3-gradstart | 190.6 | 175.6 | 175.6 @ 12 | 1.16 1.16 0.97 0.41 0.60 0.53 | 3.62 → 1.40 | 171 |
| soft-top5-d3 | 140.9 | 141.5 | 136.2 @ 2 | 1.06 1.18 1.07 0.37 0.49 0.43 | 2.28 → 1.25 | 172 |

## Hypocentre errors in the last cycle (most likely hypothesis)

| run | 90th percentile, km | share > 5 km |
|---|---:|---:|
| hard-ncand1 | 4.51 | 8.5% |
| hard-top1 | 4.60 | 8.4% |
| hard-top1-gradstart | 4.58 | 6.9% |
| hard-top1-lam02 | 4.09 | 6.1% |
| hard-top1-step01 | 5.60 | 12.6% |
| hard-top1-sub6 | 4.43 | 8.0% |
| soft-top3 | 4.58 | 8.1% |
| soft-top3-gradstart | 4.59 | 7.0% |
| soft-top5-d3 | 4.37 | 8.1% |

## RMS (m/s) by cycle

| run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard-ncand1 | 140.9 | 136.8 | 136.3 | 137.0 | 138.0 | 138.7 | 139.5 | 139.8 | 140.2 | 140.8 | 141.4 | 141.8 | 142.4 |
| hard-top1 | 140.9 | 136.9 | 136.4 | 137.0 | 137.8 | 138.5 | 139.3 | 139.9 | 140.6 | 141.2 | 141.8 | 142.3 | 142.7 |
| hard-top1-gradstart | 190.6 | 186.2 | 182.8 | 181.5 | 180.0 | 178.8 | 177.9 | 177.4 | 176.9 | 176.5 | 176.1 | 175.9 | 175.9 |
| hard-top1-lam02 | 140.9 | 135.1 | 131.9 | 129.7 | 128.3 | 127.5 | 127.1 | 127.0 | 127.1 | 127.3 | 127.5 | 127.7 | 128.1 |
| hard-top1-step01 | 140.9 | 138.6 | 139.1 | 141.2 | 143.4 | 145.7 | 147.6 | 149.0 | 150.1 | 150.7 | 150.9 | 151.0 | 151.0 |
| hard-top1-sub6 | 140.9 | 136.9 | 137.1 | 138.4 | 140.1 | 141.6 | 142.8 | 143.5 | 144.3 | 144.9 | 145.7 | 146.3 | 147.1 |
| soft-top3 | 140.9 | 136.8 | 136.4 | 137.0 | 138.0 | 138.6 | 139.4 | 139.8 | 140.5 | 141.0 | 141.4 | 142.0 | 142.6 |
| soft-top3-gradstart | 190.6 | 186.1 | 182.8 | 181.4 | 180.0 | 178.8 | 178.0 | 177.4 | 176.9 | 176.4 | 176.0 | 175.8 | 175.6 |
| soft-top5-d3 | 140.9 | 136.8 | 136.2 | 136.9 | 137.8 | 138.5 | 139.2 | 139.5 | 140.0 | 140.3 | 140.7 | 141.1 | 141.5 |
