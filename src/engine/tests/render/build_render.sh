#!/usr/bin/env bash
# render 시험 빌드 (WSL Ubuntu 22.04, clang 14 + CUDA 12.8). 산출물: ~/engine-build/render
#   bash /mnt/c/behavior-2026/src/engine/tests/render/build_render.sh [대상...]   (기본: test_render_synth test_render_scene)
set -euo pipefail
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
export PATH=/usr/local/cuda-13.2/bin:$PATH
B=${RENDER_BUILD:-~/engine-build/render}
cmake -S /mnt/c/behavior-2026/src/engine/tests -B "$B" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release > "$B.cmake.log" 2>&1 || { cat "$B.cmake.log"; exit 1; }
T=("$@")
[ ${#T[@]} -eq 0 ] && T=(test_render_synth test_render_scene)
for t in "${T[@]}"; do
  if grep -q "$t" "$B/Makefile" 2>/dev/null || [ -f "$B/build.ninja" ]; then
    cmake --build "$B" --target "$t" -j 16
  fi
done
