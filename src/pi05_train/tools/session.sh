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
    verifyl) PI05_VERIFY_LAYERS=1 $B/pi05_train_verify --ref $R --tag ${MODE}_gpu --floor ${MODE}_cpu | grep -E "^(loss|grad|  .[0-9])" ;;
    state) JAX_PLATFORMS=cpu bash $NT/wsl_py.sh $T/make_state.py --ckpt ~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio --mode $MODE --out /mnt/c/behavior-2026/data/pi05_train/state_${MODE}_radio.pi05d 2>&1 | grep -v -i warn ;;
    bench) for o in 2 1; do $B/pi05_train_bench --state /mnt/c/behavior-2026/data/pi05_train/state_${MODE}_radio.pi05d --model /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref --batch 32 --steps 2 --offload $o; done ;;
    aug) JAX_PLATFORMS=cuda bash $NT/wsl_py.sh $T/aug_ref.py --tag gpu 2>&1 | grep -v -i warn; JAX_PLATFORMS=cpu bash $NT/wsl_py.sh $T/aug_ref.py --tag cpu 2>&1 | grep -v -i warn; $B/pi05_aug_test --ref $R --tag gpu --floor cpu ;;
    augt) $B/pi05_aug_test --ref $R --tag gpu --floor cpu ;;
    quick) ~/pi05_native_build/pi05_verify --weights /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref --tag gpu --floor cpu --floor-single cpu_planted --samples 0-3 --time | grep -E "^(actions|graph|PASS|FAIL)" ;;
    drive) $B/pi05_train --state /mnt/c/behavior-2026/data/pi05_train/state_${MODE}_radio.pi05d --model /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w --table $(ls -d ~/fasttrain_work/cache/pi05_b1k-turning_on_radio-*) --lut ~/fasttrain_work/lut/lut_w720.bin --steps ${PI05_DRIVE_STEPS:-3} --log-every 1 --save-every 100000 --out /tmp ;;
    verify2) $B/pi05_train_verify --ref $R --tag ${MODE}_gpu --floor ${MODE}_cpu --offload 2 | grep -E "FAIL|PASS|worst|forward" ;;
  esac
done
echo "=== done $(date +%T)"
