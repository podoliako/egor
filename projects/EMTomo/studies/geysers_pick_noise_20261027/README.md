# Geysers spheres: five additional pick-noise levels

User-approved 2026-10-27: **one** clean forward calculation, then **15** inversions (five noise levels × three unchanged EMTomo configurations). Real acquisition geometry, synthetic arrivals; 3776 events × 26 stations. Original inputs, numerical snapshots, observations and prior runs remain untouched.

## Protocol

Baseline is the completed `matched_geometry_20261021` series, not the legacy Geysers candidate-count series. Original configurations:

| Variant | Refined shortlist | Retained | Minimum fine-cell separation |
|---|---:|---:|---:|
| hard-k15-top1-dmin1 | 15 | 1 | 1 |
| soft-k15-top10-dmin1 | 15 | 10 | 1 |
| soft-k15-top10-dmin2 | 15 | 10 | 2 |

All scientific `InversionConfig` fields are exactly preserved, including 12 updates, 1-km cells/subdivision 8, initial layers, damping, coverage, temperature, 12 event workers, and **assumed** noise `.01` relative / `.01 s` absolute / `.02 s` model. Only logging run names and output directories may differ. No temperature calibration or retuning is performed. This evaluates robustness of **fixed** configurations, including noise misspecification at nonbaseline levels.

| Multiplier | Actual relative sigma | Actual absolute sigma |
|---|---:|---:|
| 0.5 | 0.5% | 5 ms |
| 1 (existing runs reused) | 1% | 10 ms |
| 2 | 2% | 20 ms |
| 3 | 3% | 30 ms |
| 5 | 5% | 50 ms |
| 10 | 10% | 100 ms |

The forward worker calls the original frozen PyKonal `compute_travel_times` exactly once, refinement 2 / source radius 2, using 32 workers. It saves **absolute noiseless** times and IDs/coordinates in `forward/clean_times.npz`. No extra convergence calculations are performed. Applying the original pair-ID-seeded noise (seed 42) must reproduce all 98,176 old normalized picks within `1e-8 s` before any inversions are allowed.

Each new dataset is made from these same clean absolute times, with common pair-ID Gaussian draws scaled by the multiplier, followed by subtraction of each event's earliest noisy pick. Negative noisy absolute picks are not clipped. The three variants at each noise level read exactly the same observations. New input copies are byte-identical to the old four input files; metadata records the actual noise rather than the fixed likelihood assumptions.

The numerical runtime remains the original immutable matched-study snapshot. The new queue wrappers are copied into `runs/geysers_pick_noise_20261027/runner/` and hashed before launch; future working-tree edits do not alter this queue. The Python environment must not be upgraded while it runs.

## Launch, status and stop

From `/mnt/disk01/egor/projects/EMTomo` using the project environment:

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_pick_noise_20261027/pipeline.py launch
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_pick_noise_20261027/pipeline.py status
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_pick_noise_20261027/pipeline.py stop
```

Launch is detached, has no terminal input/output dependency, ignores SIGHUP, and returns after verifying a live forward child and fresh persistent heartbeat. It uses an inherited exclusive OS flock and process-birth tokens to prevent duplicate runs and unsafe signals. Launch from an ordinary terminal with standard input/output/error open; the supervisor and all workers redirect them explicitly.

At most **three** EM inversions execute concurrently (12 event workers each, numerical library threads 1). The finite supervisor automatically transitions from the validated forward stage to all 15 cases. Forward failure blocks all inversions; individual inversion failures are recorded and independent cases continue. Emergency limits: forward 3 h, each inversion 4 h, whole queue 24 h. Stop/timeout cleans up owned process groups. Do not power off/reboot the server.

This queue is **initial-only**: another launch refuses any existing queue attempt rather than silently resuming or overwriting it. Inspect failed artifacts before planning an explicit recovery. A validated clean cache can be reused by the forward worker without a second solve, but the supervisor does not automatically retry failures.

## Persistent artifacts

State root: `runs/geysers_pick_noise_20261027/`.

- `manifest.json`: approved matrix, complete old/new configs, input/runtime/runner/environment hashes, resource limits.
- `status.json`: persistent heartbeat, current phase, supervisor/child birth tokens, all case statuses and run directories.
- `queue.log`, `forward.log`, `cases/<id>/run.log`: durable logs.
- `forward/clean_times.npz`, `clean_times.json`: clean absolute-time cache with SHA-256 and provenance. Failed/interrupted solve history is retained.
- `forward_complete.json`: baseline reproduction and all five derived-observation ledgers; inversion gate.
- `experiments/{input,output}/<level-experiment>/`: immutable derived datasets.
- `cases/<id>/result.json`: actual run path, config/noise provenance, success/failure, all-cycle validation and model hashes.
- `summary.json`, `summary.csv`, `report.md`: automatic final/partial evaluation of all new cases plus the three old baselines.

Detailed EM results are under `runs/run_gpnoise27-*`. The viewer also requires each source experiment in its mounted global `forward_modeling/experiments/{input,output}` registry; a run directory alone is insufficient. After the forward stage, byte-identical copies of all five derived datasets were published there without changing the queue. `viewer_publication.json` records the publication; all copied files were checked against the forward ledger. The live API was checked for run listing, published iterations, all 3776 reference events and the true-model slice. Future cases at these five levels become visible as soon as they create metadata. All 12 iterations are retained. Source coordinates are assessment-only, not initial location priors; saved final event hypotheses are the last E-step **before** the last velocity update, matching the original protocol.

## Validation

```sh
/mnt/disk01/egor/venv/bin/python3 -B -m pytest -q -p no:cacheprovider studies/geysers_pick_noise_20261027
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_pick_noise_20261027/prepare.py --check-baselines
```

Before the real launch, 95 solve-free tests passed and all three actual frozen baseline configurations/initial models and fixed-noise resolution were checked. Unit tests do not substitute for the real baseline-reproduction gate or completion checks of the queued numerical jobs.
