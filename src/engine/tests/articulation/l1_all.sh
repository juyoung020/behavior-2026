#!/bin/bash
# 층 1 전부 (WSL, CPU): 빌드 -> 11 조건 스윕 -> 결합점 함수 -> 옮겨 담기
ulimit -c 0
T=/mnt/c/behavior-2026/src/engine/tests/articulation
B=~/engine-build/articulation
bash $T/build.sh test_articulation test_articulation_coupling test_articulation_snapshot test_articulation_gpu test_articulation_gpu_r1 > $B/l1_build.log 2>&1
grep -E "error" $B/l1_build.log | head -20
bash $T/sweep.sh > $B/sweep.log 2>&1
grep -E "^==|비교 [0-9]+,|결과" $B/sweep.log
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
echo "== coupling";        bash $T/run.sh test_articulation_coupling 2>&1 | grep -E "비트 다름 +[1-9]|결과"
echo "== coupling r1pro";  bash $T/run.sh test_articulation_coupling --r1pro $U 2>&1 | grep -E "비트 다름 +[1-9]|결과"
echo "== snapshot";        bash $T/run.sh test_articulation_snapshot --seed 1 2>&1 | tail -3
echo "== snapshot r1pro";  bash $T/run.sh test_articulation_snapshot --seed 11 --arts 4 --pos 32 --vel 1 --r1pro $U --r1copies 2 2>&1 | tail -3
