#!/bin/bash
# particles 시험 빌드 (WSL). 산출물: ~/engine-build/particles
#   bash /mnt/c/behavior-2026/src/sim/engine/tests/particles/build_particles.sh [test_visual] [test_visual_gpu]
# tests/CMakeLists.txt(리드 관리)의 모듈 목록에 particles 가 들어가기 전까지 직접 짓는다. 깃발은 ENGINE_*_FLAGS 와 같다
# (단 GPU 는 omni 처럼 -ftz=false: torch 는 비정규수를 유지).
set -e
ulimit -c 0
E=/mnt/c/behavior-2026/src/sim/engine
B=~/engine-build/particles
mkdir -p $B
T=${@:-test_visual}
for t in $T; do
  case $t in
    *_gpu) /usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_120 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
             --expt-relaxed-constexpr --extended-lambda -Xcompiler=-ffp-contract=off -I $E $E/tests/particles/$t.cu -o $B/$t ;;
    *) clang++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -I $E $E/tests/particles/$t.cpp -o $B/$t ;;
  esac
  echo "built $B/$t"
done
