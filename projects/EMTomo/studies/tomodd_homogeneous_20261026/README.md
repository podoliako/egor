# Forward-only homogeneous diagnostic — 2026-10-26

> **Final interpretation:** the original test below homogenizes the regional MOD but retains AK135 outside the regional guard. Follow-up controls found unintended support influence, including a −371 ms Geysers outlier. Both a true globally P=5 control and a private guard-only control pass all criteria on all 198,016 pairs. The complete synthesis is `paper/tomodd-homogeneous-diagnostic.md`. The initial protocol's no-build restriction applies to `run_check.py`; the later path driver and guard-only causal control use private diagnostic builds, never a production inversion executable.

This diagnostic tests station coordinate transfer and the production spherical
ray routine at fixed TRUE source positions, not an inversion. It writes only
`runs/tomodd_homogeneous_20261026/`; implementation is confined to this README,
`run_check.py`, and `test_run_check.py`. It does not edit the article, old runs,
snapshot, or solver; does not build; and never executes `tomoDD-SE`.

## Inputs and protocol

- Artificial: `runs/tomodd_sensitivity_20261026/outputs/dd1-d40-s5`, all
  **1920 × 52 = 99,840** pairs, 600-second probe timeout.
- Geysers: `runs/tomodd_dd1_geometry_20261026/outputs/geysers-dd1-d40-s5`, all
  **3776 × 26 = 98,176** pairs, 1200-second probe timeout.
- Frozen adapter package: `runs/matched_geometry_20261021/snapshot/benchmark/`;
  imported with the existing matched worker's `load_adapter` guard, once for both
  geometries. Frozen forward inputs come from that snapshot's
  `projects/forward_modeling/`.
- Existing `snapshot/solver/forward_probe` only, through frozen
  `small_domain_solver.forward_times`. It is linked with the same recorded
  `RaySPDR2.o` used by production inversion. No relink or new solver is allowed.
- Build recipe signature, all recorded artifacts, patched sources, frozen
  snapshot hashes, original input ledgers, actual generated station-file bytes,
  and support hashes are checked before running and again afterward.
- `adapter._write_external_mod` writes the original padded **22 × 9 × 9** grid
  with **5 km/s at every node**, including padding. Support tables are copied
  into the new work directory; the frozen helper copies these and MOD into its
  own retained child directory.
- TRUE original Cartesian source arrays must exactly match both saved truth and
  state arrays, including event IDs. No noisy initial source positions are used.
- Frozen `synthetic.write_events` writes a TRUE event catalog: latitude/longitude
  five decimal places and depth three decimal places in km. Parsing reproduces
  REAL*4 reads then REAL*8 promotion. Receivers are parsed from the **actual**
  prior inversion `station.dat`: float32 latitude/longitude and
  `float32(-float32(height)/float32(1000))` depth, then promoted to float64.
- A separately regenerated station catalog must be byte-identical to the actual
  one. ID order is verified rather than assumed. Both geometries retain model
  centering at x=10 km/y=3.5 km, mapped to 30.3°N/103.3°E, with 111 km/latitude
  degree and cosine-scaled longitude. No original geographic coordinates go to
  PBR.
- No picks are passed to the probe, no noise is generated, and no relocation,
  inverse solve, sampling, or truth-dependent parameter selection occurs. The
  frozen data loader reads the pre-existing arrivals solely as part of loading
  its data structure; those values are unused in this diagnostic.

## Three distinct times and predeclared criteria

All arrays are source-major `(events, stations)`:

1. `t_cart`: Euclidean distance of **original supplied Cartesian XYZ** / 5 km/s.
2. `t_chord`: spherical-Earth straight chord / 5 km/s, radius **6371 km**, standard
   latitude convention, evaluated on the **serialized float32-promoted endpoints**.
3. `t_probe`: actual frozen forward probe, receiver first, fresh scalars per pair.

Errors: `ray_only = probe − chord`, `embedding_serialization = chord − cart`,
`total = probe − cart`. Their decomposition must be exactly equal in float64.
For each error the report includes signed bias, RMS, P95 absolute error, maximum
absolute error and per-event-demeaned RMS, all in ms. Lengths, segment counts,
chord length per segment (not actual segment samples), worst pair IDs, XYZ and
serialized endpoints are retained.

**Unchanged predeclared success criteria:**

- Ray-only RMS ≤ **1 ms** and maximum absolute error ≤ **5 ms** (prior published
  ray-numeric criteria).
- Total RMS ≤ **5 ms** and maximum absolute error ≤ **10 ms**, relative to the
  existing **10 ms absolute pick-noise** scale.

Failure of a criterion is a result, not a reason to tune thresholds or sample.
Analytic-only detectability checks flip receiver-depth sign and reverse receiver
association, comparing each with the correct spherical chord. Neither wrong
setting is passed to the probe. Zero-elevation artificial receivers can make the
height-sign check a null test; that is explicitly reported.

## Running (from the EMTomo project directory)

Use the existing environment `/mnt/disk01/egor/venv/bin/python3`. Disable bytecode
writes. The precreated run root must be empty except for a `validation/` directory
containing test evidence. Tests use temporary directories only inside that tree.

```sh
mkdir runs/tomodd_homogeneous_20261026/validation
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_homogeneous_20261026/test_run_check.py
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_homogeneous_20261026/run_check.py
```

Run the full diagnostic only after the solver-free tests pass. Run commands are
bounded externally as well as by each probe's timeout. `--validate-only` verifies
hashes and the two geometries without executing the probe, but **also claims the
run root**; it cannot be followed by a normal run in the same root. No overwrite
or resume option is provided. Do not remove a failed run to hide evidence.

The first real attempt failed during adapter import, before either probe:
Matplotlib tried optional `fc-list` discovery, which the subprocess guard denied.
Its complete claim, hashes, cache and implementation were preserved in
`failed_preprobe_001/` (0.201306 s). The guard now raises `PermissionError`, an
`OSError` handled by Matplotlib's existing no-fontconfig fallback, without
allowing font-discovery subprocesses. This one explicitly named archived
pre-probe failure is permitted alongside `validation/` before a new exclusive
root claim; completed or partial probe runs are never overwritten or retried.

The exclusive root `manifest.json` records running/completed/failed status,
criteria, command, versions, timestamps and total wall time. Geometry status
files and raw probe artifacts survive failures/timeouts; a geometry failure does
not silently sample or prevent attempting the other geometry. A Python audit
hook confines cache/library writes to the new run root and allows only the
existing probe executable as a subprocess. Matplotlib/cache/temp locations are
confined to the same root. The Fortran process runs only inside the new child
work directory.

Outputs: root and per-geometry `metrics.json`, `report.md`,
`input_sha256.json`, `run_sha256.json`, original/serialized endpoints and all
analytic/probe/error arrays in compressed NPZ, TRUE catalog, actual and
regenerated station catalogs, homogeneous MOD, raw probe requests/results,
logs and helper manifests. `run_sha256.json` hashes final run files except itself
and mutable root `manifest.json` (child manifests are included). Console output
is intentionally compact; pair metadata stays on disk.

## Interpretation limits

**The probe bypasses the main input reader**, but serialized endpoint values and
scalar types are reproduced from actual files and verified against frozen
`getdata_SPDR.f`, `trialsrc_FDD_shot.f` and `tomoDD-SE.f` source. Recorded build
linkage and artifact hashes support provenance without rebuilding. This limited
homogeneous harness cannot prove a general absence of bugs, validate every
input-reader behavior, or establish the cause of heterogeneous inversion
performance. Findings go to the parent task; this study does not write article
content or modify production code.

## Additive follow-up scripts and outputs

The original completed output is sealed and unchanged. These additional scripts use exclusive new subdirectories under the same run root:

| Script | Output subdirectory | Purpose |
|---|---|---|
| `reproduce_pairs.py` | `pair_reproduction/` | Fresh-process, endpoint-direction, repetition and pair-order checks |
| `inspect_ray_paths.py` | `path_inspection/` | A private driver linked with unchanged ray/transform objects; returned paths and speeds; target support controls |
| `check_global_homogeneous.py` | `global_homogeneous/` | All pairs with the unchanged probe and copied AK135 P-speed column set to 5 km/s; seven support tests |
| `check_regional_guard.py` | `regional_guard_control/` | All pairs with original AK135 and a private ray object changing only four 0.1-degree horizontal offsets; seven source-diff tests |

The scripts were executed successfully. Do not rerun them over their existing output directories. Raw requests, arrays, reports, diffs and provenance are retained. Neither full-pair control changes production source, executable or inversion artifacts. No heterogeneous inversion has been rerun with the private guard change; its effect on reconstruction quality remains unmeasured.
