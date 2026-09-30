#!/bin/bash
# One GPU-lock session for the native trainer: GEMM/kernel tests, JAX training reference dumps, native verification.
#   session.sh [steps...]   steps: gemm kern ref_gpu ref_cpu verify   (PI05_LOCK_MIN minutes, default 15)
T=/mnt/c/behavior-2026/src/pi05_train/tools
NT=/mnt/c/behavior-2026/src/pi05_native/tools
B=~/pi05_train_build
R=/mnt/c/behavior-2026/data/pi05_train/ref
MODE=${PI05_TRAIN_MODE:-expert}
DEP=${PI05_TRAIN_DEPTH:-"--img-depth 2 --llm-depth 2"}
STEPS=${@:-gemm}
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire "pi05_train" "pi05 trainer: ${STEPS}" ${PI05_LOCK_MIN:-15} 8
trap "gpu_lock_release pi05_train" EXIT
for s in $STEPS; do
  echo "=== $s $(date +%T)"
  case $s in
    gemm) $B/tgemm_test ;;
    kern) $B/tkern_test ;;
    ref_gpu) JAX_PLATFORMS=cuda bash $NT/wsl_py.sh $T/train_ref.py --mode $MODE $DEP --tag ${MODE}_gpu 2>&1 | grep -v -i -E "warn|^\s*$" ;;
    ref_cpu) JAX_PLATFORMS=cpu taskset -c 0-15 bash $NT/wsl_py.sh $T/train_ref.py --mode $MODE $DEP --tag ${MODE}_cpu 2>&1 | grep -v -i -E "warn|^\s*$" ;;
    verify) $B/pi05_train_verify --ref $R --tag ${MODE}_gpu --floor ${MODE}_cpu ;;
  esac
done
echo "=== done $(date +%T)"
