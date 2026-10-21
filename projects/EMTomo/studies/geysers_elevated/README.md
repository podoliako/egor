# Elevated Geysers: latest configuration, hard/soft K sweep

No terrain, air mask, new regularization or scientific stopping changes.
This is **real catalogue/station geometry with synthetic picks**, not measured picks.

## Dataset

Native experiment:
`geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001`

Derived from the immutable previous Geysers input, all events on April 1–20 UTC.
The unchanged 20×7×7 km flat4 velocity cube is physically translated upward:

- Flat top elevation H=1166 m, highest station LCK at z=0.
- Station z=H−elevation: depths 0–839 m, original XY and station order.
- Event z=H+catalogue_depth: original XY, times and retained ordering.
- Four events now below the 7 km model bottom are **excluded, never clipped**:
  `3156222`, `3260211`, `3311973`, `3159245`.
- **3776 events × 26 stations = 98,176 pairs**.
- Catalogue depths and station elevations are assumed to share a vertical datum;
  that datum compatibility has **not** been independently established.
- The velocity array, shape (160,56,56), origin and 125 m cell size are unchanged.
  This is not an extension or resampling of the true model. Generation provenance
  records all source hashes, coordinate transformations and exclusion reasons.

Existing source inputs, outputs and runs are never changed.

## Latest saved configuration

Select the chronologically last **completed** prior `run_gy11-hard-c1000-*` run,
not the smallest RMSE. At preparation this is **lambda_reg=1, coverage power=2**,
hard K5, 1000 m/subdivision8, nearest, 12 updates, layered initialization
(boundaries .7/1.9/3.4 km; 3800/4600/5200/5700 m/s), max velocity step .02,
noise sigmas .01/.01 s and model sigma .02 s.

The initial model is checked bitwise against this run. No velocities or event
positions transfer from its final reconstruction. Catalogue event coordinates
are assessment/forward truth only, never localization priors.

All production Python sources are copied from that run's recorded immutable
runtime, checked against its recorded hashes. This deliberately retains the
**legacy station-sphere FMM** used by the selected run, not the working-tree v2
change: changing solver and geometry simultaneously would add another factor.
Study wrappers and queue helpers are separately frozen and hashed. The shared
venv remains live; do not upgrade dependencies during this queue.

Seven cases, first hard K5 as the same-configuration geometry baseline:

| Mode | K | Retained hypotheses | Temperature |
|---|---:|---:|---|
| hard | 5 | 1 | 1 |
| soft | 5 | 5 | calibrated common T |
| hard | 10 | 1 | 1 |
| soft | 10 | 10 | calibrated common T |
| hard | 15 | 1 | 1 |
| soft | 15 | 15 | calibrated common T |
| hard | 1 | 1 | 1 |

K is the grid shortlist refined within each candidate cell. Hard chooses the
best refined hypothesis; it is not a fixed-catalogue-hypocentre run. Soft really
keeps all K rather than the old top1, which would make it identical to hard.

## Temperature calibration

Once the new PyKonal observations exist, compute station fields of the **initial
model**, score <=128 deterministic evenly spaced observation rows with production
NLL, shortlist and refinement. No true positions, reference velocity or model
updates are used. Select one common soft T >=1, targeting median effective sample
size ESS=3 for K15, bounded above by16. If T1 already suffices, use the proper
untempered likelihood T1. If T16 still falls short, record saturation rather
than silently claim the target was achieved. Report ESS/max weight for each K,
plus duplicate-location/unique-location diagnostics. Calibration is initial-only,
not cycle-adaptive, and its data-dependent temperature is an experimental choice.
The detailed ledger, warnings and actual T are saved in `calibration.json`.

Forward uses PyKonal refinement2/source radius2, relative noise .01, absolute
noise .01 s, seed42. A deterministic subset r2/r4 convergence diagnostic is
recorded first; it is not a full-data guaranteed error bound.

## Detached launch / inspect / stop / resume

From the repository root:

```sh
venv/bin/python projects/EMTomo/studies/geysers_elevated/pipeline.py launch
venv/bin/python projects/EMTomo/studies/geysers_elevated/pipeline.py status
venv/bin/python projects/EMTomo/studies/geysers_elevated/pipeline.py stop
```

Launch detaches from terminal/SSH and returns after a child heartbeat. The machine
must remain on. Reuse tested queue helpers: exclusive OS flock, recorded PID
birth tokens, child sessions, finite stages, cleanup of process groups, max two
concurrent inversions (12 workers and BLAS12 each), forward32 workers, 72-hour
emergency bound per child. A forward/calibration failure blocks inversions;
individual inversion failures are recorded and other cases/report continue.

Persistent state: `projects/EMTomo/runs/_geysers_elevated_queue/`

- `manifest.json`, `status.json`, `queue.log`, `runtime/` source-only snapshot;
- `forward.log`, `forward_convergence.json`, `forward_complete.json`;
- `calibration.json` (atomically published after staging);
- `cases/<id>/run.log`, `cases/<id>/result.json`, preserving attempt history;
- automatic `report.md`, `summary.json`, `geometry_start.csv`, `rmse_curve.png`.

Normal viewer runs are under `projects/EMTomo/runs/run_gyelev-*`.
Resume with the same launch command: validate/reuse complete forward and cases;
restart an interrupted inversion into a new run rather than a partial-cycle
checkpoint. Never delete locks or overwrite a completed result blindly.

The report labels soft/hard, K/topN/T, initial-calibration ESS and final/best
velocity errors, retaining shared depth/coverage/hypocentre assessment. Saved
hypocentres are the last E-step **before** the last velocity update, not a new
post-final relocation. Best-cycle assessment does not implement early stopping.

Generate a read-only interim report if needed:

```sh
venv/bin/python projects/EMTomo/studies/geysers_elevated/report.py \
  --manifest projects/EMTomo/runs/_geysers_elevated_queue/manifest.json
```

Validation:

```sh
venv/bin/python -m pytest projects/EMTomo/studies/geysers_elevated -q
```

Tests include exact production wiring with numerical solves mocked; numerical
success is established only by the actual queued calculations, not unit tests.
