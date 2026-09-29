#!/usr/bin/env bash
# 렌더 비교 도구 빌드 (WSL). 결과: ~/engine-build/render-tools/render_compare
set -euo pipefail
SRC=/mnt/c/behavior-2026/src/engine
OUT=~/engine-build/render-tools
mkdir -p "$OUT"
clang++ -std=c++17 -O2 -I"$SRC" -o "$OUT/render_compare" "$SRC/tests/render/compare/render_compare.cpp"
echo "built $OUT/render_compare"
clang++ -std=c++17 -O2 -I"$SRC" -o "$OUT/rsc_check" "$SRC/tests/render/io/rsc_check.cpp"
echo "built $OUT/rsc_check"
clang++ -std=c++17 -O2 -DRENDER_STATS -I"$SRC" -o "$OUT/render_stats" "$SRC/tests/render/io/render_stats.cpp" -lpthread
echo "built $OUT/render_stats"
