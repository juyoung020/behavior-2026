#!/usr/bin/env bash
# PhysX 5.6.1 (Isaac Sim 5.1.0 짝 태그 107.3-omni-and-physx-5.6.1) 를 WSL 에서 빌드한다.
#   bash /mnt/c/behavior-2026/src/engine/scripts/build_physx.sh [preset] [config]
#   preset 기본 linux-carbonite (= Omniverse/Isaac Sim 리눅스 배포판이 쓰는 프리셋, clang)
#   config 기본 release  (checked | profile | release)
# 소스·빌드는 ~/engine-deps/ 에만 둔다(우리 저장소에 넣지 않음).
set -euo pipefail
PRESET=${1:-linux-carbonite}
CONFIG=${2:-release}
ROOT=~/engine-deps/physx-107.3-omni
TAG=107.3-omni-and-physx-5.6.1

if [ ! -d "$ROOT" ]; then
  mkdir -p ~/engine-deps
  git clone --depth 1 --branch "$TAG" https://github.com/NVIDIA-Omniverse/PhysX.git "$ROOT"
fi
cd "$ROOT/physx"

# CUDA: 13.2 하나로 통일 (/usr/local/cuda 링크 대신 직접 지정)
export CUDA_PATH=/usr/local/cuda-13.2
export CUDACXX=$CUDA_PATH/bin/nvcc
export PATH=$CUDA_PATH/bin:$PATH

# GPU 커널을 이 PC 아키텍처(sm_120)만 SASS 로 만든다 (공식은 70~120 전부 — 빌드 시간만 늘어남, 결과 코드는 같음).
# 원본 줄을 .orig 로 남기고 로컬 사본만 바꾼다.
GPU_CMAKE=source/compiler/cmakegpu/CMakeLists.txt
if ! grep -q 'ENGINE_LOCAL_ARCH' "$GPU_CMAKE"; then
  cp -n "$GPU_CMAKE" "$GPU_CMAKE.orig"
  sed -i 's/GENERATE_ARCH_CODE_LIST(SASS "70,80,86,89,90,100,120" PTX "120")/GENERATE_ARCH_CODE_LIST(SASS "120" PTX "120") # ENGINE_LOCAL_ARCH/' "$GPU_CMAKE"
fi
grep -n 'ENGINE_LOCAL_ARCH' "$GPU_CMAKE"

# CUDA 13: cuCtxCreate 가 cuCtxCreate_v4(ctx, params, flags, dev) 로 바뀜 — params 는 NULL(기본, 이전 동작과 같음). CPU 물리 결과와는 무관(GPU 문맥 생성만).
CCM=source/cudamanager/src/CudaContextManager.cpp
if ! grep -q 'ENGINE_CUDA13' "$CCM"; then
  cp -n "$CCM" "$CCM.orig"
  sed -i 's/status = cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);/status = cuCtxCreate(\&mCtx, NULL, (unsigned int)flags, mDevHandle); \/\/ ENGINE_CUDA13/' "$CCM"
fi
grep -n 'ENGINE_CUDA13' "$CCM"

./generate_projects.sh "$PRESET"
cd "compiler/$PRESET-$CONFIG"
time make -j"$(nproc)"
make install >/dev/null
echo "완료: $ROOT/physx/install/$PRESET/PhysX"
