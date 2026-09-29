#!/bin/bash
# 레지스터 상한별 처리량 (R1Pro 1 개/판, 판 16384, 검증은 판 4 개) — GPU 잠금 안에서
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-articulation "관절체 CUDA 레지스터 상한별 처리량" 12 3
trap 'gpu_lock_release engine-articulation' EXIT
R=/mnt/c/behavior-2026/src/engine/tests/articulation/run.sh
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
for t in test_articulation_gpu_r1 test_articulation_gpu_r1_m128 test_articulation_gpu_r1_m96 test_articulation_gpu_r1_m64; do
  echo "== $t"
  bash $R $t --arts 0 --r1pro $U --r1copies 1 --pos 32 --vel 1 --steps 120 --envs 4 --check 4 --benchenvs 16384 2>&1 | grep -E "처리량|대조|결과|비트 다름 [1-9]"
done
