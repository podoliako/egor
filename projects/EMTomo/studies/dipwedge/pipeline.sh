#!/bin/bash
# Geysers-scale dipping-wedge experiment: forward solve, large EMTomo grid, report.
#   setsid nohup bash studies/dipwedge/pipeline.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/disk01/egor
ID=dipwedge_20x7x7km_27st_1000ev_sobol_z0.5-5.5_noise01_pick001_seed42
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/dipwedge
OUT=$ROOT/projects/forward_modeling/experiments/output
mkdir -p "$HERE/logs"
log() { echo "$(date '+%F %T') $*" >> "$HERE/pipeline.log"; }

# ── 1. Forward problem (refinement 2 = 62.5 m nodes; full-table check against 4) ──
if [ ! -f "$OUT/$ID/metadata.json" ]; then
    log "forward start"
    rm -f "$OUT/.$ID.lock"; rm -rf "$OUT/.$ID.tmp-"*
    (cd "$ROOT" && OPENBLAS_NUM_THREADS=1 "$PY" -m projects.forward_modeling "$ID" --refinement 2 \
        --check-convergence --workers 46 --noise --noise-relative-sigma 0.01 \
        --noise-absolute-sigma-s 0.01 --noise-seed 42) >> "$HERE/logs/forward.log" 2>&1
    log "forward exit $?"
fi
[ -f "$OUT/$ID/metadata.json" ] || { log "no forward output; stopping"; exit 1; }

# ── 2. Inversions: 4 at a time, 12 event workers each ────────────────────────
# Workers limit themselves to one BLAS thread; the parent solve may use 12.
export OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
COMMON=(--cell-size-m 500 --subdivision 4 --cycles 20 --workers 12
        --max-velocity-step-fraction 0.02 --weight-model-sigma-s 0.02
        --runs-dir "$ROOT/projects/EMTomo/runs")
LAYERS=(--initial-layer-boundaries-km 0.6 1.3 2.75 --initial-layer-velocities-m-s 4000 4500 5000 5500)
HARD=(--candidate-mode hard --weights-top-n 1 --n-candidates 5)
run() {  # name, script, options...
    local name=$1 script=$2; shift 2
    local start=("${LAYERS[@]}")
    [[ " $* " == *" --initial-gradient-m-s "* ]] && start=()
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && "$PY" "$script" "$ID" --run-name "dw-$name" "${COMMON[@]}" "${start[@]}" "$@") \
        > "$HERE/logs/$name.log" 2>&1
    log "end $name (exit $?)"
}
jobs_file=$HERE/jobs.txt
: > "$jobs_file"
for lam in 0.05 0.1 0.2 0.5 1.0; do for cov in 0 0.5 1.0 1.5; do
    echo "hard-lam$lam-cov$cov main.py ${HARD[*]} --lambda-reg $lam --coverage-damping-power $cov" >> "$jobs_file"
done; done
for lam in 0.2 0.5; do for cov in 0.5 1.0; do
    echo "soft3-lam$lam-cov$cov main.py --candidate-mode soft --weights-top-n 3 --n-candidates 5 --weights-min-distance 2 --lambda-reg $lam --coverage-damping-power $cov" >> "$jobs_file"
done; done
for lam in 0.1 0.2 0.5; do for cov in 0 0.5 1.0; do
    echo "fixedhypo-lam$lam-cov$cov studies/fixed_hypocentres/run_fixed.py --lambda-reg $lam --coverage-damping-power $cov" >> "$jobs_file"
done; done
BASE="--lambda-reg 0.5 --coverage-damping-power 1.0"
cat >> "$jobs_file" <<JOBS
soft5d3-lam0.5-cov1.0 main.py --candidate-mode soft --weights-top-n 5 --n-candidates 5 --weights-min-distance 3 $BASE
hard-sub6-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --subdivision 6
hard-msig0.005-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --weight-model-sigma-s 0.005
hard-msig0.05-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --weight-model-sigma-s 0.05
hard-step0.04-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --max-velocity-step-fraction 0.04
hard-trilinear-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --slowness-interpolation trilinear
hard-gradstart-lam0.5-cov1.0 main.py ${HARD[*]} $BASE --initial-gradient-m-s 4000 5500
hard-ncand1-lam0.5-cov1.0 main.py --candidate-mode hard --weights-top-n 1 --n-candidates 1 $BASE
JOBS
log "$(wc -l < "$jobs_file") jobs queued"
while read -r name script options; do
    while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
    # shellcheck disable=SC2086
    run "$name" "$script" $options &
done < "$jobs_file"
wait

# ── 3. Report ────────────────────────────────────────────────────────────────
log "writing report"
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipwedge/report.py "$ID" dw-) \
    > "$HERE/report.md" 2>> "$HERE/logs/report.log"
log "done (report exit $?)"
