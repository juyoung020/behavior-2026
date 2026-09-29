#!/usr/bin/env bash
# 층 0 도구(ovd_dump / ovd_replay / ovd_selftest)를 WSL 에서 빌드한다. 빌드 폴더는 ~/engine-build/replay (저장소 밖).
#   bash /mnt/c/behavior-2026/src/engine/scripts/build_replay.sh [checked|release]
set -euo pipefail
CFG=${1:-checked}
SRC=/mnt/c/behavior-2026/src/engine/replay
OUT=~/engine-build/replay-$CFG
cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=Release -DPHYSX_CONFIG="$CFG" \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang > /dev/null
cmake --build "$OUT" -j"$(nproc)" 2>&1 | grep -E "error|warning: unused|Error|Built target" || true
ls -la "$OUT"/ovd_* 2>/dev/null | awk '{print $NF}'
