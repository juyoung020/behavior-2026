#!/bin/bash
ulimit -c 0
bash /mnt/c/behavior-2026/src/engine/tests/articulation/sweep.sh > ~/engine-build/articulation/sweep.log 2>&1
grep -E "^==|비교 [0-9]+,|결과" ~/engine-build/articulation/sweep.log
