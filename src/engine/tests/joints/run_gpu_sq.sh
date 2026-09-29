#!/bin/bash
# joints 층 2 장면 광선 시험만 (GPU 잠금 안에서, 1~2 분).
ulimit -c 0
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-joints "joints 층2 장면 광선 시험" 5 1
trap 'gpu_lock_release engine-joints' EXIT
cd ~/engine-build/joints/joints
./test_scene_raycast_gpu --envs 4096 --rays 64 --seed 1 | tail -5
./test_scene_raycast_gpu --envs 16384 --rays 64 --seed 2 | tail -5
