#!/bin/bash
# Unattended pipeline for the dipping-layers + six spheres experiment:
#   1) wait for (or restart) the forward solve, 2) run the EMTomo matrix two at a
#   time, 3) write report.md. Start detached:
#   setsid nohup bash studies/dipping6spheres/pipeline.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/disk01/egor
ID=dipping6spheres_240x120x120_72st_1000ev_sobol_minz5_noise01_pick005_seed42
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/dipping6spheres
OUTPUT_DIR=$ROOT/projects/forward_modeling/experiments/output
LOGS=$HERE/logs
mkdir -p "$LOGS"
log() { echo "$(date '+%F %T') $*" >> "$HERE/pipeline.log"; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ── 1. Forward problem ────────────────────────────────────────────────────────
log "waiting for forward output"
while [ ! -f "$OUTPUT_DIR/$ID/metadata.json" ]; do
    if ! pgrep -f "projects.forward_modeling $ID" > /dev/null; then
        log "forward solver not running and no output: cleaning stale lock and restarting"
        rm -f "$OUTPUT_DIR/.$ID.lock"
        rm -rf "$OUTPUT_DIR/.$ID.tmp-"*
        (cd "$ROOT" && "$PY" -m projects.forward_modeling "$ID" --refinement 2 \
            --check-convergence --workers 24 --noise --noise-relative-sigma 0.01 \
            --noise-absolute-sigma-s 0.05 --noise-seed 42) >> "$LOGS/forward.log" 2>&1
        log "forward solver exited with $?"
        [ -f "$OUTPUT_DIR/$ID/metadata.json" ] || { log "forward failed; stopping"; exit 1; }
    fi
    sleep 60
done
log "forward output ready"

# ── 2. Inversions ─────────────────────────────────────────────────────────────
COMMON=(--cell-size-m 10000 --subdivision 9 --cycles 12 --workers 24
        --lambda-reg 0.05 --coverage-damping-power 1.5 --max-velocity-step-fraction 0.02
        --weight-model-sigma-s 0.1 --runs-dir "$ROOT/projects/EMTomo/runs")
LAYERS=(--initial-layer-boundaries-km 20 45 80 --initial-layer-velocities-m-s 4600 4900 5200 5600)
GRADIENT=(--initial-gradient-m-s 4700 5500)
HARD=(--candidate-mode hard --weights-top-n 1 --n-candidates 5 --weights-min-distance 1)
SOFT3=(--candidate-mode soft --weights-top-n 3 --n-candidates 5 --weights-min-distance 1)

run() {
    local name=$1; shift
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && "$PY" main.py "$ID" --run-name "dip6-$name" \
        "${COMMON[@]}" "$@") > "$LOGS/$name.log" 2>&1
    log "end $name (exit $?)"
}

(cd "$ROOT/projects/EMTomo" && "$PY" main.py "$ID" --validate-only "${COMMON[@]}" "${LAYERS[@]}") \
    >> "$LOGS/validate.log" 2>&1 || { log "validation failed"; exit 1; }

run hard-top1 "${HARD[@]}" "${LAYERS[@]}" &
run soft-top3 "${SOFT3[@]}" "${LAYERS[@]}" &
wait
run hard-ncand1 --candidate-mode hard --weights-top-n 1 --n-candidates 1 "${LAYERS[@]}" &
run soft-top5-d3 --candidate-mode soft --weights-top-n 5 --n-candidates 5 \
    --weights-min-distance 3 "${LAYERS[@]}" &
wait
run hard-top1-step01 "${HARD[@]}" "${LAYERS[@]}" --max-velocity-step-fraction 0.01 &
run hard-top1-lam02 "${HARD[@]}" "${LAYERS[@]}" --lambda-reg 0.2 &
wait
run hard-top1-gradstart "${HARD[@]}" "${GRADIENT[@]}" &
run soft-top3-gradstart "${SOFT3[@]}" "${GRADIENT[@]}" &
wait
run hard-top1-sub6 "${HARD[@]}" "${LAYERS[@]}" --subdivision 6 &
wait

# ── 3. Report ─────────────────────────────────────────────────────────────────
log "writing report"
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipping6spheres/report.py "$ID") \
    > "$HERE/report.md" 2>> "$LOGS/report.log"
log "done (report exit $?)"
