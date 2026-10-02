#!/bin/bash
ulimit -c 0
T=/mnt/c/behavior-2026/src/sim/engine/tests/articulation
cmake --build ~/engine-build/articulation --target test_articulation_snapshot -j 16 2>&1 | grep -B2 -A6 "error" | head -30
for k in 0 2 4; do echo "== every 2, neg $k"; bash $T/run.sh test_articulation_snapshot --seed 1 --arts 8 --every 2 --neg $k 2>&1 | tail -5; done
