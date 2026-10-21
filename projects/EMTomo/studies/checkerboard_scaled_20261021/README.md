# Scaled nx4/nx6/nx8 checkerboards, 2026-10-21

User-approved series: three NEW forward solves followed by six inversions. No old picks are rescaled/reused; old physical models and event/station positions supply the scaled geometry.

## Physical problem and noise

- Domain 24 x 12 x 12 km (all old coordinates and model cell sizes divided by 10).
- Same 1200 event identities and 98 station identities; shared byte-identical scaled catalogs across nx4/6/8.
- Checker cubes: 6 / 4 / 3 km (4x2x2, 6x3x3, 8x4x4 blocks).
- Physical velocities unchanged: 4750 and 5250 m/s. Both methods start at homogeneous 5000 m/s.
- Inversion spacing 1 km, 24x12x12 = 3456 working parameters in both representations; tomoDD also has fixed remote exterior nodes.
- Forward solver: frozen PyKonal implementation, fresh solve at 125-m numerical spacing. Refinements relative to coarse physical blocks: 48 / 32 / 24. Source-radius control 2 cells; 12 event workers per forward job.
- New Gaussian pick noise: relative sigma .01 plus absolute sigma .01 s, seed 42; added before per-event earliest-arrival subtraction.
- tomoDD initial hypocenters: independent sigma 1 km per coordinate, seed 42, boundary-rejection handling as in the sphere series.

This is **not** a scale-equivalent rerun of the old experiment: absolute pick noise and hypocenter perturbations were deliberately chosen to match the sphere protocol, rather than divided by 10.

## Fixed inversion protocols

EMTomo uses the **exact frozen numerical runtime from `matched_geometry_20261021`**, including sphere_v2 source-field initialization:

- hard; 15 refined candidate cells, top1, dmin1;
- nearest slowness prolongation, subdivision8 (125-m fine cells);
- 12 updates, temperature1, damping1, coverage exponent2;
- coverage floor .05 and reference percentile75;
- max fractional velocity step .02;
- alpha .01, sigma_pick .01 s, sigma_model .02 s;
- 12 event workers per run; full cycle artifacts and event weights retained.

TomoDD uses the same sphere-series adapter and solver algorithm/patches:

- 3 joint iterations, 1-km nodes, no free buffer layers;
- smoothing5/5/5, damping40, absolute/DD amplitude weights1/.1;
- pair radius3km, max degree4, ray derivative step .05km, min-segment threshold .05km;
- coordinate-roundoff patch retained; no imposed solver wall-time limit;
- solver's original hard Vp update caps still apply.

### Two necessary technical adaptations

1. **Array capacity only:** sphere build's MAXOBS60 cannot hold 98 stations per event. A new private build increases MAXOBS to120, retaining MAXEVE4000 and all numerical source files/settings. Preparation compared saved sources against the actual sphere build: only `include/tomoFDD.inc` differs, and only MAXOBS60→120. Build log, recipe, source hashes and executable hashes are preserved under `builds/`.
2. **One absolute-only event:** with the fixed sigma1km/seed42 initial positions, the unmodified shortest-first degree4/radius3km graph leaves EVT_0241 without DD edges. It still has all 98 absolute picks. The worker reproduces the adapter's exact sorted greedy edge list but does not reject absolute-only events. No radius/degree increase, rewiring or event removal is performed. This policy is explicit in the manifest; successful tomoDD completion requires all 1200 events retained. Final per-case results report absolute-only event IDs and pair count.

## State and monitoring

`/mnt/disk01/egor/projects/EMTomo/runs/checkerboard_scaled_20261021/`

- `manifest.json`: nine jobs, exact frozen commands, full EMTomo config, approved protocol, original-source hashes, snapshot hashes, versions and build hashes.
- `snapshot/projects/forward_modeling/experiments/input/`: new scaled physical inputs.
- `snapshot/projects/forward_modeling/experiments/output/`: fresh arrivals and metadata created by the forward jobs. Each forward completion publishes hashes; inverse workers verify these before ingestion.
- `snapshot/`: frozen EMTomo, forward and benchmark sources, support files, upstream solver source copy, and actual worker/supervisor scripts.
- `builds/mins_0p05_9e81584183c5/`: completed private solver build, including capacity-only change.
- `status.json`: per-job queued/running/completed/failed/blocked state and PIDs.
- `logs/supervisor.log`, `logs/<case-id>.log`: controller and worker logs.
- `cases/<case-id>/result.json`: worker state, errors, output hashes; EMTomo run directory is published before iteration1.
- `outputs/nx4-tomodd/`, `nx6-tomodd/`, `nx8-tomodd/`: tomoDD inputs, logs, model and relocation histories.
- EMTomo run directories are under the normal `EMTomo/runs/` root with prefix `run_cbscale21-*`, discoverable by the viewer. The source experiment itself lives in the study snapshot, not the old global input tree.

Supervisor launched detached with initial PID291281. Inspect current status/process identity rather than assuming that PID remains valid. The three forward jobs run concurrently (12 workers each). A barrier requires all three to finish successfully before inversion starts. Up to three EMTomo jobs and two tomoDD jobs then run concurrently; the third tomoDD starts when a slot frees.

The queue never silently restarts or overwrites a case. A failed upstream forward solve blocks dependent inversions. SIGTERM to the supervisor stops its active child process groups and cancels pending work. The queue survives the editor/chat connection.

```sh
python3 -m json.tool runs/checkerboard_scaled_20261021/status.json
```

## Validation and limitations

- New private build completed successfully.
- Four read-only study tests passed: exact spatial scaling/model preservation; snapshot/build hashes; tomoDD grid, schedule and graph capacity; dependency waiting/blocking.
- Three forward `--validate-only` checks passed; inverse workers validate newly produced picks when their prerequisites complete.
- Initial catalogs are byte-identical across cases. Physical truths differ, so arrivals differ.
- Numerical runtime matches the new sphere runs, not legacy checkerboard v1.1.
- Fixed terminal checkpoints: EMTomo12 and tomoDD3; no truth-based iteration selection.
- Native velocity representations remain different. A later comparison should retain both representation oracles.
- Saved EMTomo final E-step locations still precede the last M-step velocity update; no final location-only pass is added.
- No figures or paper edits are produced by this pipeline.
