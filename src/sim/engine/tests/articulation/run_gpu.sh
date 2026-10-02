#!/bin/bash
# 층 2 시험을 GPU 잠금 안에서 (WSL): bash run_gpu.sh [test_articulation_gpu 인자...]
source /mnt/c/behavior-2026/src/sim/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-articulation "관절체 CUDA 판 비트·처리량 시험" 10 3
trap 'gpu_lock_release engine-articulation' EXIT
bash /mnt/c/behavior-2026/src/sim/engine/tests/articulation/run.sh test_articulation_gpu "$@"
