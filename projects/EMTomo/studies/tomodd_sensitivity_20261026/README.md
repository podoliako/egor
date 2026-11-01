# tomoDD sensitivity study — 2026-10-26

Four predeclared **artificial-event** cases, for comparison at **z = 1.5 km** with the existing matched-geometry baseline. This runner does not select a visually favorable iteration or make plots: compare the final, third joint iteration. Plotting is a separate parent task.

Baseline (read-only):

`/mnt/disk01/egor/projects/EMTomo/runs/matched_geometry_20261021/outputs/synthetic-tomodd-sigma1`

All benchmark Python sources, saved input models/catalogs/picks, solver build/binary, and support data come from `runs/matched_geometry_20261021/snapshot/`. Nothing is built, repicked, or changed in the old study, snapshot, live benchmark, or baseline. The existing `studies/matched_geometry_20261021/tomodd_worker.py` supplies `load_adapter`, default `preflight` (including `verify_build`), `run_arguments`, and `check_completed`.

## Exact matrix

Weights are **row amplitudes**, not variance weights. The baseline uses absolute/DD amplitudes **1 / 0.1**, damping 40, and smoothing 5/5/5.

| Case ID | Absolute amplitude | DD amplitude `q` | Damping | Smoothing lon/lat/depth |
|---|---:|---:|---:|---|
| `dd03-d40-s5` | 1 | 0.3 | 40 | 5 / 5 / 5 |
| `dd1-d40-s5` | 1 | 1 | 40 | 5 / 5 / 5 |
| `dd03-d80-s10` | 1 | 0.3 | 80 | 10 / 10 / 10 |
| `dd03-d20-s2` | 1 | 0.3 | 20 | 2 / 2 / 2 |

Common protocol: 1920 artificial events, 52 stations, the identical saved noisy P picks and physical truth model, initial location sigma **1 km per coordinate**, seed 42, coordinate-wise boundary rejection, initial layer boundaries 0.7/1.9/3.4 km and velocities 3800/4600/5200/5700 m/s, 1-km interior nodes, no free buffer layers, ray step 0.05 km, pair degree 4 within 3 km, and **three joint iterations**. The Gaussian sigma describes initial hypocenters, not additional pick noise.

After the original default-protocol preflight passes, each fresh worker installs a **process-local** `adapter._solver_config` override returning `SolverConfig` with smoothing from the table and exactly:

```python
((3, q, 0.08, -9.0, 1.0 / q, damping, 1, 0.0),)
```

Thus `WTCTP=q`, `WTDD=1/q`, and the absolute amplitude is `WTCTP * WTDD = 1`. `IPHA=1` and the saved source observations are P-only; the retained S weight `0.08` is unused. Residual clipping remains disabled. Other controls are checked against the baseline.

### Precision and the attempt-one 80-column failure

The snapshot renderer uses `WTCTP:.2f` and `WTDD:g` (six significant digits). For `q=0.3`, its unmodified output would be `0.30 * 3.33333 = 0.999999`, an absolute-amplitude error of about **−1e−6**. The process-local wrapper retains `.17g` for `WTDD`: `3.3333333333333335`, with `WTCTP=0.30`. For `q=1`, these tokens are `1.00` and `1`.

The parent launched attempt one in `runs/tomodd_sensitivity_20261026`. **`dd1-d40-s5` completed in about 302 seconds. All three `q=0.3` cases failed before iteration one**, with EOF at `getinpSPDR.f:138`. The precision patch had expanded their schedule data line to **88 characters**, overflowing the frozen solver's **`character line*80`** reader. The earlier solver-free checks did not catch this parser-width constraint.

Version `retry-compact80-v2` now compacts schedule whitespace **after** setting the precision token, preserving all **13 fields**. The three `q=0.3` schedule data lines are **55 characters**. Every non-comment data line is explicitly checked to be at most 80 columns. Other controls remain frozen. No solver source or binary is changed. The existing completed `q=1` control remains untouched (its longest data line is 71 characters).

Validation reports the serialized tokens, their product/error, a float32 product, and the unmodified renderer's product in `serialized_weights`. In the validated environment both the Python product and float32 product are `1.0` for every case. This avoids the six-digit decimal truncation; it does not claim arbitrary exact arithmetic inside the Fortran solver.

## Validate without running any inversion

From `/mnt/disk01/egor/projects/EMTomo` with the existing Python environment (Python >= 3.11, NumPy, SciPy, Matplotlib; verified Linux x86_64 solver build):

```sh
python3 -B -m unittest discover -s studies/tomodd_sensitivity_20261026 -p test_run_study.py -v
python3 -B studies/tomodd_sensitivity_20261026/run_study.py \
  --study-root runs/tomodd_sensitivity_20261026_retry \
  --reuse-completed-from runs/tomodd_sensitivity_20261026 \
  --validate-only --jobs 4
```

Validation starts each case in its own fresh Python subprocess to respect the adapter import guard. It runs the original preflight, checks source/input/support/binary provenance against the completed baseline, and exercises **the actual adapter input writers** in an automatically removed temporary directory. At `_input_text`, after the model/grid/observations have been written, it checks bytewise SHA-256 equality to baseline for:

- `MOD`, `grid.dat`, `event.dat`, `station.dat`;
- `dt.ct`, `absolute.dat`, and empty `dt.cc`;
- `ak135.15.SKS`, `layer-16.dat`.

It also verifies the rendered schedule/smoothing and equality of every other control to baseline. Validation then raises an internal sentinel **before the adapter can call the solver**. These same input/control checks run again on the retained files during normal execution, before the solver call. Temporary Matplotlib caches are removed; bytecode writes are disabled. Validation neither claims nor writes the requested study root. The retry root has not been created by this implementation; the parent may create it empty before launch.

Revision validation: **19 focused unit tests passed**, including the original 88-character regression, all-case field counts/line lengths, reuse rejection checks, original-artifact preservation, manifest references, and queue skipping. All four fresh frozen-input preflights passed against the retry root with reuse enabled, matching all nine constant files per case. The completed original `dd1` also passed read-only completion verification. **No new inversion was launched during this revision.**

## Parent retry launch (three real inversions — not executed during this revision)

```sh
python3 -B studies/tomodd_sensitivity_20261026/run_study.py \
  --study-root runs/tomodd_sensitivity_20261026_retry \
  --reuse-completed-from runs/tomodd_sensitivity_20261026 \
  --jobs 4
```

Use this explicit retry root: the historical default `runs/tomodd_sensitivity_20261026` is now nonempty and will be rejected. `--study-root PATH` accepts an empty/new root, never a root overlapping the imported original study. `--jobs` accepts 1–4. Each rerun uses a separate subprocess and normal `adapter.run`, with single-threaded BLAS/OpenMP settings. `--case ID` is an internal worker entry point, not a separate public launch recipe.

`--reuse-completed-from` permits **only `dd1-d40-s5`** to be imported; it is not a general resume mechanism. All four cases still get fresh input preflights. The imported case is checked against the original manifest and saved `run_study_attempt1.py` hash, successful original status/worker completion, the expected schedule/smoothing, sigma/seed, arguments, binary/build/support/source/input provenance, geometry/ray/pair settings, all nine actual constant files, and the actual original control values and manifest control hash. Normal retained artifacts and all three iteration histories/catalogs are verified without running a binary.

The retry manifest's `dd1` output points to the **original absolute output path**, not a copy or a new directory. Its status starts `completed` with `reused_from` and `reuse_validated`. Only this validated reused case is omitted from the normal solver queue; the three `q=0.3` cases rerun with their unchanged predeclared parameters. The internal worker entry point also refuses to rerun a reused case.

The command stays in the foreground and waits for every child; it does not detach. Each solver has a **1200-second timeout**. A worker is also limited to **1500 seconds** including preparation, solver, and postprocessing; each validation worker is limited to 240 seconds. Both queues have finite overall deadlines based on the number of job batches. Timeout/interruption kills the entire affected child process group, including its solver, and reaps the worker. Independent case failures do not prevent the remaining cases from running.

For `--jobs 4`, budget roughly **31 minutes plus cleanup margin** for the conservative combined validation/worker queue deadlines (normally much less). An enclosing tool timeout of only 900000 ms is shorter than the configured worst case; use a longer bounded wait (for example 2100000 ms) if the parent tool supports it. Lower concurrency needs a longer parent allowance. SIGINT/SIGTERM also cleans up active child groups.

## Retained output and safety

Normal launch repeats validation first, then exclusively writes the **new root's** `manifest.json` and `status.json` before launching any real inversion. The manifest has an absolute `baseline`, a top-level four-entry `cases` list with `id`, `dd_weight`, `damping`, `smoothing`, `output`, exact schedules/arguments and validation hashes, plus script version/hash and helper/snapshot-source/input/build-manifest/binary/support provenance. Initial frozen provenance is checked and retained. The retry also records `original_manifest_sha256`, `original_script_sha256`, and per-artifact hashes under the reused case's `reuse_provenance`. Its `result_script_sha256` identifies the original script that produced the reused result; `preflight_control_sha256` distinguishes the newly compacted validation control from the unchanged original `generated_control_sha256`. `comparison_depth_km` remains 1.5; no plotting is performed.

Script/provenance identities for this revision:

| Artifact | SHA-256 |
|---|---|
| Attempt-one script (`runs/tomodd_sensitivity_20261026/run_study_attempt1.py`) | `bb180e52534269f83ed814b323031efa21c72bc9440886067b8d35a0c0c673fa` |
| Original manifest | `1ee38b7544477fa5b4f9366275710abd85c7b7dcf774f701e51fac2db5f02764` |
| Current `run_study.py` (`retry-compact80-v2`) | `37117bc8586c890c75490dcc8ccf185e7384069c2be5f2454ba429fd3a78316b` |

```text
runs/tomodd_sensitivity_20261026_retry/
  manifest.json                       # dd1 points to original completed output
  status.json                         # dd1 starts completed/reused
  logs/<q=0.3-case-id>.log
  outputs/<q=0.3-case-id>/
    config.json, results.json, state.npz, truth.npz
    worker_info.json
    work/inversion/...
    figures/...
```

The adapter retains its usual artifacts, including all three relocation snapshots and velocity history. A worker is marked `completed` only after `adapter.run` succeeds, normal retained files exist, all three iterations have finished, each velocity snapshot exists exactly once, and the final and intermediate event catalogs are complete. `worker_info.json` records `state`, `elapsed_s`, actual controls, and generated-input hashes without duplicating full results/event-ID catalogs; results remain in the normal `results.json`.

The parent updates status atomically and marks the study `completed` only when the verified reused case and **all three new workers** pass; otherwise it exits nonzero and retains failed outputs/logs. Without the reuse option, all four cases are run in a fresh root. It rejects a nonempty destination root and every existing new case output (even an empty directory or dangling final symlink). There is **no overwrite, cleanup, or automatic retry**. Original attempt-one artifacts, status, manifest, and saved script are read-only and preserved in place, including the three failed cases. An existing manifest cannot be overwritten.

## Completed results

The retry series completed on 2026-10-26 at 20:58:16 UTC. All four results contain three completed joint iterations and all 1920 relocated events. The `q=1` result was reused from the successful first attempt; only the three cases that failed before iteration 1 were relaunched after correcting control-line formatting. No parameter values were changed after seeing results.

| Case | Abs/DD amplitude | Damping | Smoothing | Slice RMSE at z=1.5 km, m/s | Volume RMSE, m/s | Location RMS, m |
|---|---|---:|---|---:|---:|---:|
| 3D L2 oracle | — | — | — | 183.99 | 142.86 | — |
| Existing baseline | 1/0.1 | 40 | 5/5/5 | 144.62 | 240.22 | 207.95 |
| `dd03-d40-s5` | 1/0.3 | 40 | 5/5/5 | 159.17 | 264.64 | 205.43 |
| `dd1-d40-s5` | 1/1 | 40 | 5/5/5 | 250.34 | 404.07 | 277.88 |
| `dd03-d80-s10` | 1/0.3 | 80 | 10/10/10 | 209.71 | 257.53 | 276.62 |
| `dd03-d20-s2` | 1/0.3 | 20 | 2/2/2 | 169.23 | 296.34 | 216.23 |

None of these four alternatives improves velocity RMSE over the baseline on either this slice or the full volume. This is a bounded sensitivity result, not evidence that the baseline is globally optimal or a diagnosis of the Geysers deterioration. The oracle is a full-volume fit, not a slice-specific lower bound.

The plotter independently checks the iteration-3 history against saved final nodes, equality of physical truth/oracles, actual input/control hashes, and agreement of recomputed volume RMSE with each saved result. All panels use exact depth 1.5 km, native trilinear Vp, 125-m XY pixel centres, a common color scale, no outlines, and no additional smoothing.

Reproduce the figure without launching a solver:

```sh
python -B paper/make_tomodd_sensitivity_comparison.py
```

Outputs: `paper/figures/tomodd_synthetic_parameter_comparison_z1p5km.{png,pdf,json,md}`. The plotter defaults to the completed retry manifest, including its reference to the original successful `q=1` output.

Validation: 19 focused runner tests passed; all four input preflights passed; the completed comparison plot and metric checks passed. No new forward arrivals were generated.

## Follow-up: dd1-d40-s5 on Geysers

The user requested the same `dd1-d40-s5` setting on Geysers. Completed output:
`runs/tomodd_dd1_geometry_20261026/outputs/geysers-dd1-d40-s5`.
The original artificial-geometry result was reused, not rerun.

Runner: `studies/tomodd_sensitivity_20261026/run_geysers_dd1.py` (supports `--validate-only`). It reuses the frozen matched-series data/binary and verifies the nine generated constant input files against the existing Geysers sigma=1 baseline before solver execution. The only numerical change from that Geysers baseline is absolute/DD row amplitudes 1/1 instead of 1/0.1. Damping 40, smoothing 5/5/5, three joint iterations, initial sigma 1 km per coordinate and seed 42 are unchanged. All 3776 events and three completed iterations were verified. Runtime was 231.1 seconds. Input/control/source hashes and copied launch scripts are retained with the new manifest/status; older runs remain untouched.

| dd1-d40-s5 comparison | Slice RMSE at z=1.5 km, m/s | Volume RMSE, m/s | Location RMS, m |
|---|---:|---:|---:|
| Common 3D L2 oracle | 183.99 | 142.86 | — |
| Artificial geometry | 250.34 | 404.07 | 277.88 |
| Geysers geometry | 476.03 | 496.07 | 573.76 |

Reproduce the three-panel figure without rerunning inversions:

```sh
python -B paper/make_tomodd_dd1_geometry_comparison.py
```

Outputs: `paper/figures/tomodd_dd1_artificial_geysers_z1p5km.{png,pdf,json}`. The plotter checks equality of truth, initial velocity fields and oracle fields; actual controls/input hashes; iteration-3 history against final nodes; and recomputed volume RMSE against saved metrics. Common color scale, exact depth, no outlines or extra smoothing. Preflight, actual inversion, plot checks, and 19 shared runner tests passed.
