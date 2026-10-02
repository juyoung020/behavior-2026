#!/bin/bash
# omni 시험 빌드 (WSL). 산출물: ~/engine-build/omni
#   bash /mnt/c/behavior-2026/src/engine/tests/omni/build_omni.sh [target...]
set -e
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
B=~/engine-build/omni
if [ ! -f $B/CMakeCache.txt ]; then
  cmake -S /mnt/c/behavior-2026/src/engine/tests -B $B -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release > $B.cmake.log 2>&1 || { tail -30 $B.cmake.log; exit 1; }
fi
T=${@:-test_bddl}
for t in $T; do cmake --build $B --target $t -j 16 2>&1 | grep -E "error|warning: unused|Error|Built target" | head -40; done
