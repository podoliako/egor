# Matched geometry series, 2026-10-21

Eight user-approved runs on the existing flat4 synthetic picks. No new arrivals are generated. The two geometries are the artificial 1920-event/52-station catalog and the elevated Geysers 3776-event/26-station catalog. Pick noise remains 1% relative plus 0.01 s absolute (seed 42).

## Fixed experiment matrix

Each row is run on **both** geometries:

| Method | Candidates refined | Retained | Minimum cell separation | Updates |
|---|---:|---:|---:|---:|
| EMTomo hard | 15 | 1 | 1 | 12 |
| EMTomo soft | 15 | 10 | 1 | 12 |
| EMTomo soft | 15 | 10 | 2 | 12 |
| tomoDD | — | — | — | 3 |

Minimum separation is Chebyshev distance between shortlisted 125-m fine cells, before continuous refinement. Hard uses one candidate for the update even though it refines 15.

EMTomo common configuration: 1-km constant cells, subdivision 8, nearest slowness interpolation, initial layer boundaries 0.7/1.9/3.4 km and velocities 3800/4600/5200/5700 m/s, damping 1, coverage exponent 2, floor 0.05 and reference percentile 75, maximum velocity step 2%, temperature 1, alpha 0.01, sigma_pick 0.01 s, sigma_model 0.02 s. 12 event workers per run. All six use the **same frozen current sphere_v2 implementation**, not the previous paper-run sphere implementation. Core numerical code was not changed for this study.

tomoDD common configuration: Gaussian **initial hypocenter** perturbations with sigma **1 km per coordinate**, seed 42, with boundary rejection; not kilometre-valued pick noise. Identical 4000-event-capacity binary for both geometries; 1-km interior nodes, identical initial layers, no free buffer layers, ray controls 0.05 km, pair degree 4/distance 3 km, smoothing 5/5/5, damping 40, absolute/DD amplitude weights 1/0.1, three joint iterations and no imposed solver time limit. Native hard velocity caps remain in the solver.

## Frozen artifacts and monitoring

Study state:

`/mnt/disk01/egor/projects/EMTomo/runs/matched_geometry_20261021/`

- `manifest.json`: all eight exact commands, complete EMTomo configurations, input/code/binary hashes, versions, protocol notes.
- `snapshot/`: copied runtime sources, input models/catalogs, saved arrivals, tomoDD executable/build sources and support data, plus the runner scripts actually used. Subsequent edits to live code/data do not alter this series. Do not modify the snapshot or manifest during the study.
- `status.json`: queue and per-case status, PIDs, start/finish times and exit codes.
- `supervisor.pid`: detached supervisor PID (initially 279230; always inspect status/process identity before sending a signal).
- `logs/supervisor.log`: launch/completion events.
- `logs/<case-id>.log`: worker stdout/stderr.
- `logs/<case-id>.preflight.log`: successful validation before launch.
- `cases/<EM-case-id>/result.json`: current status and actual EMTomo run directory, published before iteration 1. Final verification records the final model hash.
- `outputs/<geometry>-tomodd-sigma1/`: tomoDD config, state/results, solver logs and per-iteration velocity/relocation history. `worker_info.json` is written on exit.
- `source_git_status.txt`, `source_git_diff.patch`: record the live tree's pre-existing modifications at preparation time. No commit or branch was created.

EMTomo output directories are directly under `EMTomo/runs/`, named `run_mg21-*`, so the existing viewer can discover them. Every iteration saves its starting model, update, diagnostics and event hypotheses, with `final_model.npy` after update 12. As in existing code, saved E-step locations precede the corresponding M-step; a final location-only pass was not added.

The detached queue runs at most **three EMTomo jobs (12 workers each) and two tomoDD jobs** concurrently. Remaining cases start automatically. The initial launch started both tomoDD jobs, both hard jobs, and artificial-geometry soft/dmin1. No automatic rerun or overwrite is performed. Failed cases are marked failed; independent cases continue. Each child has its own process group; terminating the supervisor with SIGTERM terminates its active child groups and cancels its pending queue.

From `projects/EMTomo`, inspect status without starting anything:

```sh
python3 -m json.tool runs/matched_geometry_20261021/status.json
```

Do not rerun `supervise.py` against this manifest: it deliberately rejects an existing status file. Original benchmark and EMTomo runs remain untouched.

## Validation before launch

- `python3 -B -m pytest -q tests/test_tomography_logic.py tests/test_station_seed.py tests/test_tomography_logging.py studies/matched_geometry_20261021/test_tomodd_worker.py`
- Result: **76 passed, 1 xfailed, 2 subtests passed**.
- All eight frozen commands passed `--validate-only`; inputs, counts, grids, noise, imports, build signature and capacities were checked without running an inversion.
- Copied observation/input hashes were checked against originals and forward metadata.

No new plots or paper replacements are produced by this series. Use the predeclared final checkpoints for subsequent comparisons rather than choosing the visually best iteration.
