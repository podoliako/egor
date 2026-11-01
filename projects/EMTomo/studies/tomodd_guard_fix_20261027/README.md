# Horizontal regional guard: corrected production inversion

Write scope: this study directory and `runs/tomodd_guard_fix_20261027/` only.
No existing entrypoint is executed, no original artifact is replaced, and no
paper file is edited. The earlier causal control is read/hash-checked, not rerun.

## Reproduction (foreground, exclusive phases)

From the EMTomo project root, using Linux x86_64 and the frozen compiler:

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_guard_fix_20261027/run_study.py --build-only
/mnt/disk01/egor/venv/bin/python3 -B -m unittest discover -s studies/tomodd_guard_fix_20261027 -p 'test_*.py' -v
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_guard_fix_20261027/run_study.py --validate-only
/mnt/disk01/egor/venv/bin/python3 -B studies/tomodd_guard_fix_20261027/run_study.py --run
```

The fixed run root must initially be empty. Each build/validation/run phase
exclusively claims its own paths. Never delete/overwrite failed history. The
script intentionally refuses to overwrite an existing phase. A driver-only retry
requires first archiving the failed phase, source version, root manifest, and
status, and explicitly recording old/current source provenance in the manifest.

The first validation attempt stopped before either solver launched: live tuples
for layer boundaries/velocities were compared against JSON lists. This driver
normalizes that metadata comparison without weakening value checks; a regression
test covers live preflight tuples. The failed validation and exact prior source,
manifest, and status are retained under `history/attempt1/`. The original build
manifest is unchanged; its historical study-source hashes are verified against
that archive. Current driver hashes are checked separately. No binary rebuild,
input alteration, or old-run change was needed.

A second solver-free validation exposed the old synthetic reference's original
71-column schedule spacing (Geysers uses the compact 38-column layout). Its
failed synthetic validation and passed Geysers validation are retained under
`history/attempt2/`. The driver now restores only the reference schedule line's
whitespace after the shared renderer checks, requiring identical tokens and
byte-identical complete controls. It records both shared-rendered and actual
control hashes. Tests reject changed tokens and non-schedule bytes.

## Exact intervention and build

`horizontal_guard.patch` removes only the four horizontal 0.1-degree offsets
from frozen `RaySPDR2.f` lines 840–841, matching the prior causal diagnostic's
`OLD_GUARD`/`NEW_GUARD`. Source reversal is byte-exact. The bottom guard
`if(dp.GE.zn(nz)-1)then`, all other numerical code, includes, flags, and objects
remain unchanged.

The frozen `snapshot/solver/manifest.json` verifies compiler and original
artifacts. A read-only Makefile dry run obtains the exact inversion link object
order. Only the corrected Ray object is compiled; all other objects are linked
directly from the snapshot. An original-object relink must reproduce the frozen
`tomoDD-SE` SHA-256 exactly before accepting the corrected production inversion.
This is not a probe-only build. Source, patch, command logs, commands, compiler
identity, object hashes, binary hashes, original-relink proof, and timing are
retained in `build/manifest.json` and `build/commands.json`.

## Input guards and provenance

Each geometry runs in a fresh process. The unchanged shared worker first
preflights the **original frozen binary** and all original input/support/control
checks. Its provenance is checked against the geometry's matched sigma1 baseline
and old dd1 run. Only after that preflight is the independently verified corrected
binary substituted. Worker metadata contains both original and corrected solver
provenance; the active `solver_sha256`, `solver_manifest_sha256`,
`solver_build_manifest`, and adapter `config.json` refer to the corrected build.

`shared.BASELINE` is set separately for synthetic and Geysers. The existing
`configure_adapter` checks are retained. An additional hook requires every
constant generated input and the complete control to be byte-identical to the
old geometry-specific dd1 reference. This is checked before solver launch and
after completion. Validation retains generated inputs/control but stops before
executing a solver. Full runs additionally verify the actual invocation, binary
hash, timeout, config provenance, full event retention, and all three completed
joint iterations through `worker.check_completed`.

Protocol: q=1, damping=40, smoothing=(5,5,5), three joint iterations, sigma=1 km,
seed=42, degree=4/distance=3 km pair graph, 0.05-km ray step, free hypocenters.

## Supervision and retained artifacts

Both geometries execute concurrently in distinct foreground worker process
groups. The solver has a 1200-s timeout; each worker has a 1500-s timeout (240 s
for validation), and the finite supervisor has an additional 20-s bound. Process
groups are terminated/reaped on errors or interruption. No daemon is started.
Caches, compiler temporaries, and test fixtures stay in the two allowed trees.

- `manifest.json`: root protocol/provenance and phase-summary checksums.
- `status.json`, `status_history.jsonl`: latest state plus retained transitions;
  `completed` is published only after **both** inversions pass completion checks.
- `protected_sha256.json`: before/after checks of frozen solver, benchmark,
  support, forward inputs, old baselines/references, helper scripts, and the
  immutable prior guard-control artifacts.
- `validation/`: two retained preflight/input checks and integrity summary.
- `outputs/{synthetic,geysers}-dd1-d40-s5/`: `worker_info.json`, adapter config,
  results, truth/state, generated inputs, solver logs, and every iteration
  relocation/velocity snapshot. No old run is reused as a new result.
- `outputs/summary.json`: final metrics, per-worker and solver wall times,
  provenance, and SHA-256 ledger of every retained output artifact.

The source-only and provenance/input-reuse tests include negative mutations for
bottom/source changes, binary/object/source checksums, compiler provenance,
generated event input, control bytes, old provenance, and perturbation seed.
