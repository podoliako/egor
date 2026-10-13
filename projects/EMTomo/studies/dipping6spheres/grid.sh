#!/bin/bash
# Regularization grid on the dipping-layers experiment (subdivision 6, 16 cycles).
#   setsid nohup bash studies/dipping6spheres/grid.sh > /dev/null 2>&1 &
set -u
ROOT=/mnt/disk01/egor
ID=dipping6spheres_240x120x120_72st_1000ev_sobol_minz5_noise01_pick005_seed42
PY=$ROOT/venv/bin/python
HERE=$ROOT/projects/EMTomo/studies/dipping6spheres
mkdir -p "$HERE/logs"
log() { echo "$(date '+%F %T') $*" >> "$HERE/grid.log"; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON=(--cell-size-m 10000 --subdivision 6 --cycles 16 --workers 24
        --max-velocity-step-fraction 0.02 --weight-model-sigma-s 0.1
        --candidate-mode hard --weights-top-n 1 --n-candidates 5
        --initial-layer-boundaries-km 20 45 80 --initial-layer-velocities-m-s 4600 4900 5200 5600
        --runs-dir "$ROOT/projects/EMTomo/runs")
run() {
    local lam=$1 cov=$2 name="grid-lam$1-cov$2"
    log "start $name"
    (cd "$ROOT/projects/EMTomo" && "$PY" main.py "$ID" --run-name "dip6-$name" "${COMMON[@]}" \
        --lambda-reg "$lam" --coverage-damping-power "$cov") > "$HERE/logs/$name.log" 2>&1
    log "end $name (exit $?)"
}
pairs=("0.1 0" "0.1 0.5" "0.1 1.0" "0.2 0" "0.2 0.5" "0.2 1.0" "0.5 0" "0.5 0.5" "0.5 1.0" "0.2 1.5")
for ((i = 0; i < ${#pairs[@]}; i += 2)); do
    run ${pairs[i]} &
    [ $((i + 1)) -lt ${#pairs[@]} ] && run ${pairs[i + 1]} &
    wait
done
log "writing report"
(cd "$ROOT/projects/EMTomo" && "$PY" studies/dipping6spheres/report.py "$ID") \
    > "$HERE/report_grid.md" 2>> "$HERE/logs/report.log"
log "done (report exit $?)"
