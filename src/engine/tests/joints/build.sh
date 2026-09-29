#!/bin/bash
# joints 시험 빌드·실행 (WSL). 산출물: ~/engine-build/joints
#   bash /mnt/c/behavior-2026/src/engine/tests/joints/build.sh [대상...]      (대상 없으면 joints 시험 전부)
#   실행: LD_LIBRARY_PATH=~/engine-deps/physx-107.3-omni/physx/bin/linux.x86_64/checked ~/engine-build/joints/joints/test_joints_prep
set -e
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
B=~/engine-build/joints
if [ ! -f $B/CMakeCache.txt ]; then
  mkdir -p $B
  cmake -S /mnt/c/behavior-2026/src/engine/tests -B $B -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++ > $B/cmake.log 2>&1 || { cat $B/cmake.log; exit 1; }
fi
T="$@"
[ -z "$T" ] && T="test_joints_prep test_libm_joints test_rigid_api test_joints_gpu test_libm_joints_gpu"
cmake --build $B -j 16 --target $T 2>&1 | grep -E 'error|Error|Built target|FAILED' | head -60
