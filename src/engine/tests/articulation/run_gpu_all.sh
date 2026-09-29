#!/bin/bash
# 층 2 시험 전부를 GPU 잠금 한 번 안에서 (WSL): 검증(무작위·R1Pro·구면) + R1Pro 처리량(레지스터 상한별)
ulimit -c 0
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-articulation "관절체 CUDA 판 비트·처리량 시험 (무작위 + R1Pro + 레지스터)" 15 3
trap 'gpu_lock_release engine-articulation' EXIT
R=/mnt/c/behavior-2026/src/engine/tests/articulation/run.sh
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
echo "== 무작위 관절체 8 개/판 (구면 끔)"
bash $R test_articulation_gpu --arts 8 --steps 600 --envs 64 --check 8 --benchenvs 4096 --seed 1 --spherical 0
echo "== 무작위 관절체 8 개/판, 구면 관절 포함, seed 5"
bash $R test_articulation_gpu --arts 8 --steps 600 --envs 64 --check 8 --benchenvs 0 --seed 5 --spherical 1
echo "== R1Pro 1 개 + 무작위 3 개/판 (위치반복 32)"
bash $R test_articulation_gpu_r1 --arts 3 --r1pro $U --r1copies 1 --pos 32 --vel 1 --steps 600 --envs 32 --check 4 --benchenvs 0 --seed 11
for t in test_articulation_gpu_r1 test_articulation_gpu_r1_m128 test_articulation_gpu_r1_m96 test_articulation_gpu_r1_m64; do
  echo "== R1Pro 1 개/판 처리량: $t (판 16384, 120 스텝)"
  bash $R $t --arts 0 --r1pro $U --r1copies 1 --pos 32 --vel 1 --steps 120 --envs 4 --check 4 --benchenvs 16384
done
