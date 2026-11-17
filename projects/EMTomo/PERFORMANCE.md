# Performance controls

These controls change execution and storage, not the inversion model, pick-noise
parameters, candidate selection, or Powell method/bounds/options. Small floating
point rounding differences are possible. These execution changes retain the current
method version (**1.5**, including held-out stopping).
Configuration metadata uses `dataclasses.asdict`, so it records all controls below.

## Configuration and CLI

Every field has a matching hyphenated CLI flag:

| Config field | Default | Meaning |
| --- | --- | --- |
| `likelihood_batch_size` | `8` | Maximum events evaluated together per event worker. |
| `likelihood_batch_memory_mb` | `128` | Per-event-worker budget in MiB for the batch likelihood cost cube. |
| `compact_rays` | `True` | Compact ray sensitivities rather than dense station/voxel arrays. |
| `fast_refinement` | `True` | Fused compiled interpolation and likelihood callback. |
| `linear_solver` | `auto` | `auto`, `dense`, `matrix_free`, or `legacy`. |
| `matrix_free_min_voxels` | `8192` | Inversion voxel threshold for automatic matrix-free selection. |
| `solver_rtol` | `1e-10` | Matrix-free relative residual tolerance. |
| `solver_maxiter` | `None` | Matrix-free iteration limit; `None` uses the solver default. |

Batch sizes, memory budgets, thresholds, and explicit iteration limits must be
positive integers, not booleans. `solver_rtol` must be finite and strictly between
zero and one. The two toggles require actual booleans in Python configuration.
Use `--compact-rays` / `--no-compact-rays` and
`--fast-refinement` / `--no-fast-refinement` on the CLI.
`--solver-maxiter none` restores the default iteration limit.

The event batch is reduced to fit the configured cost-cube budget, but always
contains at least one event. Each float64 cost grid costs `8 * n_grid` bytes;
the batch ceiling is conceptually
`max(1, min(likelihood_batch_size, budget_bytes // (8 * n_grid)))`.
This is **not a total process-memory cap**: fields, ray data, normal equations,
scratch buffers, and other allocations are separate. A single event grid may
exceed the budget. Multiple event workers each have their own batch budget.

The tiled likelihood shares station precisions across the caller-sized batch;
it does not build a full catalog-by-station-by-grid array. The refinement callback
preserves clipped, staged trilinear interpolation and its intermediate dtype
promotion, including the relative-noise likelihood normalization.

## Solver selection

- `auto` selects matrix-free only when the **inversion** voxel count is at least
  `matrix_free_min_voxels` **and** `lambda_reg > 0`. Otherwise it selects dense.
  Fine-grid subdivision does not itself increase the inversion voxel count.
- `dense` uses the direct path: try Cholesky, falling back to LU and then
  least squares as needed. It allocates the dense normal matrix.
- `matrix_free` forces the iterative CG path. Production configuration requires
  `lambda_reg > 0`: an undamped singular solve could otherwise change the
  minimum-norm solution. CG failure is explicit; there is **no silent fallback
  that allocates a dense matrix**.
- `legacy` forces the previous dense solve path. It does not disable batching,
  compact rays, or fused refinement; those controls are independent.

The default threshold keeps current small grids on the direct path: for example,
`24 * 12 * 12 = 3456` inversion voxels is below 8192. Matrix-free reduces dense
matrix storage, but is **not guaranteed faster**. Its time depends on conditioning,
ray data, tolerance, and the required number of iterations. Tolerance and iteration
limit settings apply to the iterative path, not to the direct solver.

## Examples

Run these from `projects/EMTomo`, replacing `EXPERIMENT_ID` with an existing saved
experiment. These examples are configuration checks only; remove `--validate-only`
deliberately to run EM.

```sh
# Default performance controls, conservative worker count.
python main.py EXPERIMENT_ID \
  --workers 1 --validate-only

# Reference-style comparison: switch each execution control explicitly.
python main.py EXPERIMENT_ID \
  --likelihood-batch-size 1 --no-compact-rays --no-fast-refinement \
  --linear-solver legacy --workers 1 --validate-only

# Explicit matrix-free configuration (positive damping is required).
python main.py EXPERIMENT_ID \
  --linear-solver matrix_free --lambda-reg 0.01 \
  --solver-rtol 1e-10 --solver-maxiter 1000 --workers 1 --validate-only
```

The existing `n_workers` default is unchanged. Do not raise it arbitrarily:
additional workers multiply private memory and can compete for memory bandwidth.
Keep worker/thread counts fixed when comparing implementations. For controlled
single-thread kernel measurements, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `NUMBA_NUM_THREADS=1` before starting Python.

## Measurements and timing interpretation

A small warmed, single-thread measurement with float32 fields and nonzero relative
noise gave the following results; these are **kernel measurements, not a whole-run
speedup guarantee**:

| Workload | Existing | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| 26 stations, 32³ grid, 8 events, tile 1024, precomputed normalization | 21.369 ms | 5.739 ms | 3.72× |
| 26-station interpolation + likelihood callback | 42.131 µs | 1.724 µs | 24.44× |

Both measured outputs matched exactly. Compilation was warmed before timing.
Hardware, grid dimensions, batch size, cache behavior, and NumPy/Numba versions
matter. No production-sized run was used to establish these numbers. FMM, ray
tracing, output, solver work, and process startup can limit whole-run gains.

Each saved iteration writes `iter_N/performance.json` with stage wall times,
event-worker time sums, and solver residual/iteration diagnostics. The JSON is
written atomically before the iteration's completion marker.
Worker sums describe accumulated work across workers: they overlap in time and
are **not additive wall-clock stages**. Do not add them to stage wall times to
estimate elapsed time. Compare actual iteration wall time separately; inspect
solver residuals and iterations alongside timing, and distinguish JIT/startup
costs from warmed work. `wall_s` covers model refinement, station fields, optional
field output, gradient/normalization preparation, event processing, event output,
and the solve. It excludes the velocity update and final diagnostic publication.
`events.worker_time_s` separates likelihood, candidate refinement, rays, assembly,
and event diagnostics; the times overlap between workers.

## Reproducible complete-step benchmark

`benchmark_performance.py` compares reference-style kernels, optimized dense, and
matrix-free execution on the same synthetic observations and initial model. It
warms each path, rotates their order, checks numerical equivalence, and reports
median times and maximum model differences. It does not write or modify runs.
The reference uses serial station-ray kernels within each event process as well;
this avoids nested parallelism and is not a byte-for-byte replay of an old build.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python benchmark_performance.py --repeats 3 --workers 1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python benchmark_performance.py --shape 20 7 7 --subdivision 4 \
  --events 32 --stations 26 --workers 2 --repeats 3
```

A measurement of the second command (20×7×7 model, subdivision 4, 32 events,
26 stations, 5 candidates, 2 workers, three warmed repetitions) gave:

| Mode | Median step | Speedup | Maximum updated-velocity difference from reference |
| --- | ---: | ---: | ---: |
| Reference-style | 2.303 s | 1× | 0 |
| Optimized dense | 1.038 s | 2.22× | 3.64e-12 m/s |
| Matrix-free | 1.008 s | 2.28× | 4.25e-7 m/s |

CG took 88 iterations and its actual relative residual was 9.79e-11. These
measurements were made on the existing dual-Xeon host, not an isolated machine.

The benchmark omits result-file I/O. Its small synthetic catalog is not a forecast
for an entire production run; FMM and solver fractions depend on grid size and
catalog size. Use saved per-stage timings to assess production behavior.

## Memory and compatibility

- Dense partial normal systems and parallel station-field outputs use disjoint
  shared `RawArray` slices, avoiding large result arrays in multiprocessing pipes.
  Reduction order remains chunk-ordered; changing worker counts can still change
  rounding. Input fields retain fork's copy-on-write sharing.
- The compact ray path uses the original tracing/deposition kernels, one touched
  row workspace, and CSR output. It does not allocate a dense station-by-voxel
  array first. Optional fine-grid viewer artifacts retain their sparse array
  representation; new runs pack them into event containers as described below.
- Matrix-free mode stores compact hypotheses and packs a global-index CSR payload
  once for compiled CG matvecs. This adds linear ray storage and is not necessarily
  cheaper than a small dense Hessian. No large dense fallback occurs on CG failure.
- Candidate counts, Powell tolerances, source seeding, FMM order, ray step, and
  velocity-update clipping are unchanged. Existing runs are neither rewritten nor
  migrated. Only future processes use the new code; already running studies may
  also use frozen code snapshots.

## Reusing control localizations (enabled by default)

After update k, held-out scoring builds fields in the updated model and localizes
only events with control picks, using **training picks only**. Previously, E-step
k+1 repeated this exact search. It now reuses:

- The station travel-time fields of that same model.
- Every shortlisted cell, refined position and refined cost, in original order
  (not just the MAP candidate and not the previous top-N subset).

The next E-step still applies its own stable ranking, top-N, temperature and
soft/hard weights. Events without control picks follow the original search path.
Control scoring remains post-update on every cycle; the split, origin fit,
RMSE reduction order, stopping rule, rays and normal equations are unchanged.

`tomography_cache.py` fingerprints the model, cell size, training arrivals including
missing-pick masks, station coordinates, noise and search settings. Reusing a
stale cache raises an error. Temperature, top-N, candidate mode and worker count
are deliberately not part of the search key. No held-out observations or full
per-event cost volumes are stored. With optional misfit-volume logging, the
likelihood volume is regenerated, but candidate refinement is still reused.

The cache is in memory only, handed off for one model at a time; the EM loop
releases its field references before the following M-step solve. It neither
persists across restarts nor changes historical runs. No cache is retained on the
last allowed update. `performance.json` reports `events.reused_localizations`;
`station_fields_s` is zero for a reused field volume, and the console prints a
`[Reuse]` line. `run_em(..., reuse_control_localization=False)` provides an
uncached reference path for tests/benchmarks (not a change to the scientific
configuration). This execution choice is recorded in run metadata.

Validation covers genuine serial/fork calculations, soft/hard weighting, changing
temperature/top-N, sparse picks, optional misfit and fine-ray output, dense and
matrix-free updates, stale-cache rejection, and complete multi-update runs with
identical stopping and saved outputs. An additional check on 8 events from the
current realistic Geysers experiment, 26 stations, 100 m fine grid, 50 candidates
and both initial models produced **exactly equal updates** with and without reuse
for both methods. This is a correctness check, not a full-catalog speedup estimate.

## Packed event ray diagnostics

New output uses one `iter_N/event_M/rays.npz` instead of separate
`weight_K/ray_count.npy` and `weight_K/G_stations_sparse.npz` files. All hypotheses,
ray counts and fine-grid sparse ray arrays are retained **losslessly**. Weights,
positions, residuals, station masks, aggregate counts and completion markers
retain their existing roles. Fine-grid diagnostic tracing is unchanged.

Event archive schema 1:

- `schema_version`: integer 1.
- `ray_count_indices`, `g_indices`: original hypothesis IDs for each payload type.
- `ray_count_<id>`: original coarse-grid array.
- `g_<id>_shape`, `g_<id>_offsets`, `g_<id>_coords`, `g_<id>_values`:
  original sparse station-ray arrays.

Each array is a separate NPY member in the NPZ, compressed with level-1 DEFLATE.
Writing streams arrays rather than concatenating them and atomically publishes
only the complete archive. The viewer reads just the requested hypothesis's G
members. No parallel writer pool is needed for this first optimization.

The server supports both this format and legacy per-weight sparse/dense files,
with unchanged endpoint responses. Metadata declares `viewer_event_rays_schema=1`.
**The Docker viewer must be rebuilt/recreated with the updated `server.py` before
viewing new-format rays**; its source is baked into the image, not bind-mounted.
Existing runs need no migration. Frozen inversion snapshots must include the new
`tomography/tomography_cache.py` when preparing future studies.

A temporary-directory synthetic benchmark of 4 events × 50 hypotheses (three
repetitions, ray artifacts only) measured:

| Output | Median save time | Files | Size |
| --- | ---: | ---: | ---: |
| Previous per-weight files | 3.044 s | 400 | 19.56 MB |
| Packed event archives | 0.715 s | 4 | 19.71 MB |

This is 4.25× faster artifact writing and 99% fewer ray files on that fixture,
with 0.75% size growth. It is **not** a measured whole-iteration speedup; the
benchmark did not call `fsync` or reproduce the full production I/O workload.

Focused regression commands, from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  PYTHONPATH=projects/EMTomo venv/bin/python -m pytest \
  projects/EMTomo/tests/test_localization_cache.py \
  projects/EMTomo/tests/test_event_ray_container.py \
  projects/EMTomo/tests/test_early_stopping.py \
  projects/EMTomo/tests/test_viewer_api.py -q
```
