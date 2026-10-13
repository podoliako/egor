#!/bin/bash
# Four runs on flat4 after the forward solve: cells 1 km (sub 8) and 0.5 km (sub 4),
# true hypocentres and EM hard; same regularization, nearest interpolation.
#   setsid nohup bash studies/flatlayers/runs.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/disk01/egor
ID=flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/flatlayers
OUT=$ROOT/projects/forward_modeling/experiments/output
log() { echo "$(date '+%F %T') $*" >> "$HERE/pipeline.log"; }
log "runs.sh waiting for forward output"
while [ ! -f "$OUT/$ID/metadata.json" ]; do
    if ! pgrep -f "projects.forward_modeling $ID" > /dev/null; then
        log "forward not running and no output: restarting it"
        rm -f "$OUT/.$ID.lock"; rm -rf "$OUT/.$ID.tmp-"*
        (cd "$ROOT" && OPENBLAS_NUM_THREADS=1 "$PY" -m projects.forward_modeling "$ID" --refinement 2 \
            --check-convergence --workers 46 --noise --noise-relative-sigma 0.01 \
            --noise-absolute-sigma-s 0.01 --noise-seed 42) >> "$HERE/logs/forward.log" 2>&1
        log "forward exit $?"
        [ -f "$OUT/$ID/metadata.json" ] || { log "forward failed; stopping"; exit 1; }
    fi
    sleep 60
done
log "forward output ready"
COMMON=(--cycles 12 --lambda-reg 0.5 --coverage-damping-power 1.0 --workers 12
        --max-velocity-step-fraction 0.02 --weight-model-sigma-s 0.02 --slowness-interpolation nearest
        --initial-layer-boundaries-km 0.7 1.9 3.4 --initial-layer-velocities-m-s 3800 4600 5200 5700
        --runs-dir "$ROOT/projects/EMTomo/runs")
run() {  # name, script, options...
    local name=$1 script=$2; shift 2
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 \
        "$PY" "$script" "$ID" --run-name "fl-$name" "${COMMON[@]}" "$@") > "$HERE/logs/$name.log" 2>&1
    log "end $name (exit $?)"
}
HARD=(--candidate-mode hard --weights-top-n 1 --n-candidates 5)
FIXED=studies/fixed_hypocentres/run_fixed.py
run fixed-c1000 "$FIXED" --cell-size-m 1000 --subdivision 8 &
run hard-c1000 main.py "${HARD[@]}" --cell-size-m 1000 --subdivision 8 &
run fixed-c500 "$FIXED" --cell-size-m 500 --subdivision 4 &
run hard-c500 main.py "${HARD[@]}" --cell-size-m 500 --subdivision 4 &
wait
log "writing report"
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipwedge/report.py "$ID" fl- --misfit) \
    > "$HERE/report.md" 2>> "$HERE/logs/report.log"
log "done (report exit $?)"
