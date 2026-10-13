#!/bin/bash
# Fixed-true-hypocentre runs matching two grid runs of studies/dipping6spheres/grid.sh.
set -u
ROOT=/mnt/disk01/egor
ID=dipping6spheres_240x120x120_72st_1000ev_sobol_minz5_noise01_pick005_seed42
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/fixed_hypocentres
mkdir -p "$HERE/logs"
log() { echo "$(date '+%F %T') $*" >> "$HERE/oracle.log"; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON=(--cell-size-m 10000 --subdivision 6 --cycles 16 --workers 24
        --max-velocity-step-fraction 0.02 --weight-model-sigma-s 0.1
        --initial-layer-boundaries-km 20 45 80 --initial-layer-velocities-m-s 4600 4900 5200 5600
        --runs-dir "$ROOT/projects/EMTomo/runs")
run() {
    local name="fixedhypo-lam$1-cov$2"
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && "$PY" studies/fixed_hypocentres/run_fixed.py "$ID" \
        --run-name "dip6-$name" "${COMMON[@]}" --lambda-reg "$1" --coverage-damping-power "$2") \
        > "$HERE/logs/$name.log" 2>&1
    log "end $name (exit $?)"
}
run 0.5 1.0 &
run 0.1 0 &
wait
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipping6spheres/report.py "$ID") > "$HERE/report.md" 2>> "$HERE/logs/report.log"
log "done (report exit $?)"
