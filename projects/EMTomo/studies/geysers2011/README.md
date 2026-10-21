# Geysers 2011: hard-1000 damping sweep

## What this experiment measures

This is **real station/catalogue geometry with synthetic picks**, not inversion of
measured arrival picks. The local Geysers files contain no such measured picks.
The original flat4 true velocity array, origin, cell size and anomaly locations
are copied without changes. Each of the nine runs starts from the original
layered hard-1000 initial model, verified bitwise against the saved run.

- Catalogue interval: UTC **[2011-04-01, 2011-04-21)**, all **3780** events.
- **26 stations**, all active throughout that interval; real projected XY.
- **98,280 pairs**, versus 99,840 in flat4: **−1.5625%**. No thinning.
- Stations lie at **z=0**. Real elevations are preserved in generation provenance,
  but not modelled. Catalogue depths are used unchanged relative to the flat top;
  this is a depth-datum/topography approximation, not an elevation correction.
- All station/event pairs are synthesized, not a reconstruction of actual pick
  availability, completeness or detection thresholds.
- True grid: 160×56×56, 125 m; domain 20×7×7 km.
- Inversion grid: 20×7×7, 1000 m; subdivision 8, nearest interpolation.
- Initial layers: boundaries 0.7/1.9/3.4 km, speeds 3800/4600/5200/5700 m/s.
- Forward: independent PyKonal, refinement 2, source radius 2 cells; Gaussian
  relative sigma .01, absolute sigma .01 s, seed 42, old normalization convention.
- Before full synthesis: noiseless r=2/r=4 convergence diagnostics for 16 evenly
  spaced catalogue events plus the shallowest/deepest (duplicates removed).
  This subset diagnostic is **not** a full-data convergence check or guaranteed
  error bound, and has no new acceptance/early-stopping threshold.

## Cases and method equivalence

Cartesian sweep: `lambda_reg ∈ {0.1,0.5,1.0}` ×
`coverage_damping_power ∈ {0,1,2}`. Reference **λ=.5, coverage=1** goes first.

All other inversion parameters are taken directly from the saved
`run_fl-hard-c1000-*20261013_200119/meta.json` and validated against it: hard,
K=5, top1, 12 updates, temperature 1, coverage floor .05/reference percentile75,
max velocity step .02, pick sigmas .01/.01 s, model sigma .02 s.
No initial-model prior, new smoothing or stopping changes are introduced.
Catalogue hypocentres supply forward sources and assessment only; inversion
searches locations normally, without catalogue starting locations/fixing.

For a genuine old-method comparison, **all nine cases replay the old station
sphere FMM formulation**: immutable `eikonal.py` from commit `c381cfd`, labelled
`skfmm_order2_exact_station_homogeneous_sphere`. The newer v2 station-sphere fix
in the working tree is deliberately **not** included. No production source is
reverted or modified by this study. Other required source files and study
runners are copied into a small immutable-by-convention runtime snapshot and
SHA256-checked before and after cases. Installed Python dependencies are shared
with the venv; do not upgrade them while the queue runs.

## Launch, inspect, stop, resume

Run from repository root (`/mnt/disk01/egor`), using its venv:

```sh
venv/bin/python projects/EMTomo/studies/geysers2011/pipeline.py launch
venv/bin/python projects/EMTomo/studies/geysers2011/pipeline.py status
venv/bin/python projects/EMTomo/studies/geysers2011/pipeline.py stop
```

`launch` returns once a detached supervisor has published its child heartbeat.
The supervisor has its own session, ignores SIGHUP, has no terminal stdin and
writes persistent logs; an editor/SSH disconnect does not intentionally stop it.
The machine must remain running. Forward uses 32 workers (convergence subset16);
then at most two inversions run concurrently, each with 12 workers and BLAS12.
Each child has a 72-hour emergency runtime bound, not a scientific stopping
criterion.

State lives in the ignored directory:

`projects/EMTomo/runs/_geysers2011_queue/`

- `manifest.json`: exact dataset, cases, reference, resources, source hashes.
- `status.json`: supervisor/active child PIDs, birth tokens, stage, exit codes.
- `queue.log`: orchestration and failures.
- `forward.log`, `forward_convergence.json`, `forward_complete.json`.
- `cases/<case>/run.log`, `cases/<case>/result.json`: case status, real run path,
  provenance, elapsed time, final quality or traceback; failed attempts retained.
- `runtime/`: frozen source-only snapshot; **do not edit**.
- Automatic final `report.md`, `summary.json`, `geometry_start.csv`,
  `rmse_curve.png` if plotting is available.

Normal inversion directories are under `projects/EMTomo/runs/`, prefixed
`run_gy11-hard-c1000-...`, and use the existing viewer completion protocol.
Report can also be generated during calculations without new solves:

```sh
venv/bin/python projects/EMTomo/studies/geysers2011/report.py \
  --manifest projects/EMTomo/runs/_geysers2011_queue/manifest.json
```

Resume with the same `launch` command after a graceful stop/failure. OS `flock`
prevents simultaneous queues; a live previously recorded child blocks a duplicate
launch even if its supervisor died. Do not remove locks or start another state
for the same inputs blindly. A stale lock filename is harmless: the OS owns the
actual lock. Hard-killed native forward stages may leave their own publication
lock/staging area: inspect and recover explicitly rather than deleting it blindly.

On resume, complete forward outputs are validated against input hashes,
configuration/noise and persisted output hashes. Completed cases are reused only
if provenance, effective configuration, initial/final arrays, all 12 iteration
completion markers and quality logs validate. Interrupted/failed inversions start
a new run (not a mid-cycle checkpoint); their old directories remain untouched.
A corrupt previously completed result is rejected, not silently overwritten.
A forward failure prevents all inversions. An individual inversion failure is
recorded and the remaining cases/report still run. Final status distinguishes
`complete`, `complete_with_failures`, `failed`, `interrupted`.

## Assessment and tests

Report assesses the initial and updated velocity models against slowness-averaged
truth, RMSE by depth, coverage, logged residuals/ray status and hypocentre error.
Hypocentre statistics refer to the **last saved E-step before the final velocity
update**, not an unperformed post-final relocation. Any best-cycle ranking is
assessment only. Historical full-synthetic metrics are not a controlled geometry
comparison at identical event distribution.

```sh
venv/bin/python -m pytest projects/EMTomo/studies/geysers2011 -q
```

Tests cover dataset/provenance invariance, baseline/config equality, production
API preparation/logging wiring, report failures/partial data, duplicate locks,
detached launch/stop, finite scheduling, failures, timeouts and reuse. Heavy
numerical solves are replaced in tests; actual forward/inversions are the queued
experiment, not claimed to have passed by unit tests.
