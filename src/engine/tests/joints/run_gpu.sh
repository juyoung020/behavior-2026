#!/bin/bash
# joints 층 2 시험 실행 (GPU 잠금 안에서). 사용: bash run_gpu.sh [test_joints_gpu 인자...]
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
B=~/engine-build/joints/joints
gpu_lock_acquire engine-joints "joints 층2 시험(조인트 섬 판 N·4묶음 판·libm 전수)" 10 2
trap 'gpu_lock_release engine-joints' EXIT
cd $B
./test_joints_gpu "$@"
./test_joints_gpu --envs 262144 --seed 5 --reps 5 | tail -6
./test_joints_block4_gpu --envs 32768 --seed 3 --reps 5 | tail -7
./test_libm_joints_gpu
