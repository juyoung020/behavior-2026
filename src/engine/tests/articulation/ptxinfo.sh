#!/bin/bash
# 커널 레지스터·스택 정보 (GPU 없이 컴파일만)
cd ~/engine-build/articulation
F=$(grep -l "test_articulation_gpu.cu" articulation/CMakeFiles/test_articulation_gpu.dir/build.make >/dev/null && echo ok)
/usr/local/cuda-13.2/bin/nvcc -std=c++17 -O2 -arch=sm_120 -fmad=false -prec-div=true -prec-sqrt=true -ftz=true --expt-relaxed-constexpr \
  -Xcompiler=-ffp-contract=off -Xptxas -v -I/mnt/c/behavior-2026/src/engine -I/mnt/c/behavior-2026/src/engine/tests/articulation \
  -I$HOME/engine-deps/physx-107.3-omni/physx/include -I$HOME/engine-deps/physx-107.3-omni/physx/pvdruntime/include \
  -DNDEBUG -DPX_CHECKED=1 -DPX_SUPPORT_PVD=1 -DPX_SUPPORT_OMNI_PVD=1 $@ \
  -c /mnt/c/behavior-2026/src/engine/tests/articulation/test_articulation_gpu.cu -o /tmp/tag.o 2>&1 | grep -E "kStep|kExtract|registers|stack" | head -20
