#!/bin/bash
# 학습 데이터 가속의 GPU 검증·측정을 차례로 (WSL). 각 단계 전에 GPU 가 비었는지 본다 — 다른 작업(Isaac Sim 등)이 GPU 를
# 쓰고 있으면(사용 중 메모리 > FT_GPU_BUSY_MB, 기본 5000) 기다렸다가 한다. 결과는 $FT_WORK/logs/ 와 저장소 logs/fasttrain/.
#   bash /mnt/c/behavior-2026/tools/ft_gpu_suite.sh [단계 ...]     (기본: 전부)
# 단계: stage ref same check neg loader cbench native orig(원래 로더 처리량, 같은 조건 비교용)
set -u
REPO=/mnt/c/behavior-2026
RUN="bash $REPO/tools/ft_run.sh"
FT_WORK=${FT_WORK:-$HOME/fasttrain_work}
LOG=$FT_WORK/logs
BUSY=${FT_GPU_BUSY_MB:-5000}
mkdir -p "$LOG" "$REPO/logs/fasttrain"
steps=${*:-stage ref same check neg loader cbench native}

# GPU 잠금: 공용 도구(선착순 대기열, src/engine/scripts/gpu_lock.sh), owner=fasttrain. 단계마다 잡고 끝나면 푼다.
# 이미 줄 선 자리를 이어받으려면 GPU_QUEUE_TS=<20자리> 를 준다(첫 단계에만 쓰인다).
source "$REPO/src/engine/scripts/gpu_lock.sh"
ME=fasttrain
take_lock() { gpu_lock_acquire "$ME" "$1" "$2" 3; unset GPU_QUEUE_TS; }
drop_lock() { gpu_lock_release "$ME" >/dev/null 2>&1; return 0; }
trap 'drop_lock; gpu_queue_leave "$ME"' EXIT INT TERM

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt "$BUSY" ] && return 0
    echo "[$(date +%T)] GPU 사용 중 ${used} MiB — 기다린다"
    sleep 30
  done
}

run() {  # run <이름> <명령...>  — 단계마다 잠금을 잡고 끝나면 푼다 (한 번에 30 분 넘지 않게)
  local name=$1; shift
  take_lock "$name" 20
  wait_gpu
  echo "[$(date +%T)] === $name ==="
  "$@" 2>&1 | grep -v -i -E "warn|^\s*$" | tee "$LOG/$name.txt"
  local rc=${PIPESTATUS[0]}
  cp "$LOG/$name.txt" "$REPO/logs/fasttrain/$name.txt"
  echo "[$(date +%T)] === $name 끝 (exit $rc) ==="
  drop_lock
}

for s in $steps; do
  case $s in
    stage)  run verify_stage  $RUN tools/ft_verify.py stage ;;
    ref)    run verify_ref_a  $RUN tools/ft_verify.py ref --tag a
            run verify_ref_b  $RUN tools/ft_verify.py ref --tag b ;;
    same)   run verify_same   $RUN tools/ft_verify.py same a b ;;
    check)  run verify_check  $RUN tools/ft_verify.py check --tag a ;;
    neg)    for n in lut resize frame; do run verify_neg_$n $RUN tools/ft_verify.py check --tag a --negative $n; done ;;
    loader) run verify_loader $RUN tools/ft_verify.py loader --batches 4 --workers 2 ;;
    cbench) run bench_cbench  $RUN tools/ft_bench.py cbench --threads 2,4,6,8,12 --batches 150 --json $FT_WORK/bench_cbench.json
            run bench_cbench_step $RUN tools/ft_bench.py cbench --threads 6 --batches 100 --step-ms 100 ;;
    native) run bench_native  $RUN tools/ft_bench.py native --threads 6 --batches 80 --json $FT_WORK/bench_native.json
            run bench_native_step $RUN tools/ft_bench.py native --threads 6 --batches 60 --step-ms 100 --json $FT_WORK/bench_native_step.json ;;
    orig)   run bench_orig    $RUN tools/ft_bench.py loader --workers 8,16 --batches 16 --warm 1 --json $FT_WORK/bench_loader_orig_quiet.json ;;
    *) echo "모르는 단계 $s" ;;
  esac
done
cp $FT_WORK/bench_*.json "$REPO/logs/fasttrain/" 2>/dev/null
echo "끝"
