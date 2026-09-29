#!/bin/bash
# 층 1 여러 조건 (WSL). 결과 줄만 모아 출력
R=/mnt/c/behavior-2026/src/engine/tests/articulation/run.sh
run() { echo "== $*"; bash $R test_articulation "$@" 2>&1 | grep -E "비교 [0-9]+,|비트 다름 +[1-9]|결과|첫 다름|\[다름\]" | head -12; }
run --arts 16 --steps 600 --seed 1
run --arts 16 --steps 600 --seed 2 --pos 32 --vel 1
run --arts 16 --steps 600 --seed 3 --pos 8 --vel 4
run --arts 16 --steps 600 --seed 4 --extevery 1
run --arts 16 --steps 600 --seed 5 --contactlast 1
run --arts 16 --steps 600 --seed 6 --links 20 --threads 4
run --arts 16 --steps 600 --seed 7 --floating 1
run --arts 16 --steps 600 --seed 8 --pos 4 --vel 0 --extevery 1 --contactlast 1
run --arts 24 --steps 600 --seed 9 --links 30
run --arts 24 --steps 600 --seed 10 --links 12 --pos 32 --vel 1 --threads 8
run --arts 4 --steps 600 --seed 11 --pos 32 --vel 1 --r1pro /mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf --r1copies 2
