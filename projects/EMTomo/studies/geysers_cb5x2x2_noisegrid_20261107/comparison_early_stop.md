# Early-stop comparison: last strictly decreasing RMSE checkpoint

Post-hoc selection using known synthetic truth. First local minima equal global minima across all 12 saved updates for all ten runs. Iteration numbers count completed M-steps.

| Noise | Hard iteration | Hard RMSE (m/s) | Soft iteration | Soft RMSE (m/s) | Soft RMSE gain (%) | MAE hard / soft (m/s) | Time hard / soft (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| x0.5 | 7 | 229.121 | 3 | 228.167 | 0.416 | 216.86 / 214.14 | 1.26 / 1.23 |
| x0.75 | 5 | 230.499 | 3 | 228.665 | 0.796 | 218.89 / 214.56 | 0.90 / 1.24 |
| x1 | 3 | 232.045 | 3 | 229.618 | 1.046 | 220.38 / 215.24 | 0.54 / 1.29 |
| x1.5 | 2 | 234.868 | 2 | 232.410 | 1.047 | 225.47 / 221.19 | 0.36 / 0.83 |
| x2 | 2 | 236.474 | 2 | 234.879 | 0.675 | 225.90 / 222.83 | 0.36 / 0.84 |
