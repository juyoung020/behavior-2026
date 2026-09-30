#!/usr/bin/env bash
# 관측 가공 C++(core/omni/obs.h) vs 공식 trace 비트 대조 (CPU)
set -e
B=~/engine-build/omni-obs
mkdir -p $B
g++ -O2 -ffp-contract=off -fno-fast-math -std=c++17 -fPIC -shared -I/mnt/c/behavior-2026/src/engine \
  /mnt/c/behavior-2026/src/engine/tests/omni/obs_capi.cpp -o $B/libobs.so -ldl
export ENGINE_MKL_LIB=$(~/behavior-linux/.venv/bin/python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\",\"libtorch_cpu.so\"))")
~/behavior-linux/.venv/bin/python /mnt/c/behavior-2026/src/engine/tests/omni/test_obs.py $B/libobs.so "$@"
