#!/bin/bash
# articulation 시험만 빌드 (WSL). 산출물: ~/engine-build/articulation
#   bash /mnt/c/behavior-2026/src/sim/engine/tests/articulation/build.sh [대상...]
set -e
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
B=~/engine-build/articulation
cmake -S /mnt/c/behavior-2026/src/sim/engine/tests -B $B -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_BUILD_TYPE= > $B.cmake.log 2>&1 || { tail -30 $B.cmake.log; exit 1; }
T=${@:-test_articulation}
for t in $T; do cmake --build $B --target $t -j 16 2>&1 | grep -E "error|warning: unused|Error|Built target" | head -60; done
