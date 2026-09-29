#!/bin/bash
# 층 1 전부 (WSL, CPU): 빌드 -> 11 조건 스윕 -> 결합점 함수 시험
ulimit -c 0
T=/mnt/c/behavior-2026/src/engine/tests/articulation
bash $T/build.sh test_articulation test_articulation_coupling test_articulation_gpu test_articulation_gpu_r1 || exit 1
bash $T/sweep.sh
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
echo "== coupling"
bash $T/run.sh test_articulation_coupling 2>&1 | tail -20
echo "== coupling r1pro"
bash $T/run.sh test_articulation_coupling --r1pro $U 2>&1 | tail -20
