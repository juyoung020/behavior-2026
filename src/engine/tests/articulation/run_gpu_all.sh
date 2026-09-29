#!/bin/bash
# 층 2 시험 두 가지를 GPU 잠금 한 번 안에서 (WSL)
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-articulation "관절체 CUDA 판 비트·처리량 시험 (무작위 + R1Pro)" 15 3
trap 'gpu_lock_release engine-articulation' EXIT
R=/mnt/c/behavior-2026/src/engine/tests/articulation/run.sh
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
echo "== 무작위 관절체 8 개/판"
bash $R test_articulation_gpu --arts 8 --steps 600 --envs 64 --check 8 --benchenvs 4096 --seed 1 --spherical 0
echo "== R1Pro 1 개 + 무작위 3 개/판 (위치반복 32)"
bash $R test_articulation_gpu_r1 --arts 3 --r1pro $U --r1copies 1 --pos 32 --vel 1 --steps 600 --envs 32 --check 4 --benchenvs 2048 --seed 11
echo "== R1Pro 1 개/판 (위치반복 32)"
bash $R test_articulation_gpu_r1 --arts 0 --r1pro $U --r1copies 1 --pos 32 --vel 1 --steps 600 --envs 32 --check 4 --benchenvs 16384
echo "== 무작위 관절체 8 개/판, 구면 관절 포함, seed 5"
bash $R test_articulation_gpu --arts 8 --steps 600 --envs 64 --check 8 --benchenvs 0 --seed 5 --spherical 1
