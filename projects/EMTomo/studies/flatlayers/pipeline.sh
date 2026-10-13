#!/bin/bash
# Four flat layers + spheres, dense uniform stations, hypocentre lattice:
# velocity-parameterization series with fixed (true) and EM-hard hypocentres.
#   setsid nohup bash studies/flatlayers/pipeline.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/disk01/egor
ID=flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/flatlayers
OUT=$ROOT/projects/forward_modeling/experiments/output
mkdir -p "$HERE/logs"
log() { echo "$(date '+%F %T') $*" >> "$HERE/pipeline.log"; }

if [ ! -f "$OUT/$ID/metadata.json" ]; then
    log "forward start"
    rm -f "$OUT/.$ID.lock"; rm -rf "$OUT/.$ID.tmp-"*
    (cd "$ROOT" && OPENBLAS_NUM_THREADS=1 "$PY" -m projects.forward_modeling "$ID" --refinement 2 \
        --check-convergence --workers 46 --noise --noise-relative-sigma 0.01 \
        --noise-absolute-sigma-s 0.01 --noise-seed 42) >> "$HERE/logs/forward.log" 2>&1
    log "forward exit $?"
fi
[ -f "$OUT/$ID/metadata.json" ] || { log "no forward output; stopping"; exit 1; }

COMMON=(--cycles 12 --lambda-reg 0.5 --coverage-damping-power 1.0
        --max-velocity-step-fraction 0.02 --weight-model-sigma-s 0.02
        --initial-layer-boundaries-km 0.7 1.9 3.4 --initial-layer-velocities-m-s 3800 4600 5200 5700
        --runs-dir "$ROOT/projects/EMTomo/runs")
run() {  # name, script, workers, blas threads, options...
    local name=$1 script=$2 workers=$3 threads=$4; shift 4
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && OPENBLAS_NUM_THREADS=$threads OMP_NUM_THREADS=$threads \
        "$PY" "$script" "$ID" --run-name "fl-$name" --workers "$workers" "${COMMON[@]}" "$@") \
        > "$HERE/logs/$name.log" 2>&1
    log "end $name (exit $?)"
}
HARD=(--candidate-mode hard --weights-top-n 1 --n-candidates 5)
FIXED=studies/fixed_hypocentres/run_fixed.py
# Cell 1 km (sub 8) and 0.5 km (sub 4): fine cells 125 m.
for cell in 1000 500; do
    sub=$((1000 * 8 / cell / 1)); [ "$cell" = 500 ] && sub=4
    for interp in nearest trilinear; do
        tag="c$cell-$interp"
        run "fixed-$tag" "$FIXED" 12 12 --cell-size-m "$cell" --subdivision "$sub" --slowness-interpolation "$interp" &
        run "hard-$tag" main.py 12 12 "${HARD[@]}" --cell-size-m "$cell" --subdivision "$sub" --slowness-interpolation "$interp" &
        wait
    done
done
# Cell 1/3 km (sub 3, fine cells 111 m): 26 460 unknowns, dense normal matrix 5.6 GB per worker.
third=333.3333333333333
run fixed-c333-nearest "$FIXED" 6 20 --cell-size-m $third --subdivision 3 &
run hard-c333-nearest main.py 6 20 "${HARD[@]}" --cell-size-m $third --subdivision 3 &
wait
log "writing report"
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipwedge/report.py "$ID" fl- --misfit) \
    > "$HERE/report.md" 2>> "$HERE/logs/report.log"
log "done (report exit $?)"
