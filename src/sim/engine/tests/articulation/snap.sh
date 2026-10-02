#!/bin/bash
ulimit -c 0
T=/mnt/c/behavior-2026/src/sim/engine/tests/articulation
bash $T/build.sh test_articulation_snapshot || exit 1
U=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf
for a in "--seed 1" "--seed 2 --pos 32 --vel 1" "--seed 7 --floating 1" "--seed 11 --arts 4 --pos 32 --vel 1 --r1pro $U --r1copies 2" "--seed 6 --links 20"; do
  echo "== $a"; bash $T/run.sh test_articulation_snapshot $a 2>&1 | tail -16
done
