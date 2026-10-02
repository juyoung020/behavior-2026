#!/usr/bin/env bash
# llama.cpp 를 CUDA 12.8(Blackwell sm_120)로 빌드한다 -> ~/llama.cpp/build/bin/. Qwen3.5-9B 계획기 서버(tools/run_qwen_server.sh)용.
# (리눅스판; 옛 WSL 판 archive/tools/setup/setup_llamacpp_cuda_wsl.sh 와 같은 위치·빌드 설정·대상)
# - CUDA 는 저장소 기준 12.8(/usr/local/cuda-12.8). 13.2 는 쓰지 않는다. sm_120(RTX 50 계열)은 12.8 이상이 필요.
# - 12.8 이 없으면 NVIDIA ubuntu2204 저장소에서 컴파일러·런타임·cuBLAS 개발 패키지만 받는다(sudo, 드라이버는 안 건드림).
#   bash tools/setup/setup_llamacpp_cuda.sh [--dry-run]     (로그: logs/setup_llamacpp_cuda_<시각>.log)
#   LLAMA_DIR(기본 ~/llama.cpp), LLAMA_REF(기본 없음 = 최신 master 얕은 클론; 이미 있으면 그대로 씀), JOBS(기본 nproc)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DRY=0
case ${1:-} in --dry-run) DRY=1 ;; -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;; '') ;; *) echo "모르는 인자: $1" >&2; exit 2 ;; esac
CUDA_VER=12-8
CUDA_HOME=/usr/local/cuda-12.8
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
JOBS=${JOBS:-$(nproc)}
CMAKE_ARGS=(-B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
            -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release)
TARGETS=(llama-server llama-mtmd-cli llama-bench)
if [ $DRY = 1 ]; then
  echo "(dry-run) CUDA: $CUDA_HOME ($([ -x $CUDA_HOME/bin/nvcc ] && echo 있음 || echo '없음 -> apt 로 설치'))"
  echo "(dry-run) 소스: $LLAMA_DIR ($([ -d "$LLAMA_DIR" ] && echo 있음 || echo '없음 -> git clone --depth 1'))"
  echo "(dry-run) cmake ${CMAKE_ARGS[*]} && cmake --build build -j $JOBS --target ${TARGETS[*]}"
  exit 0
fi
mkdir -p "$REPO/logs"
LOG=$REPO/logs/setup_llamacpp_cuda_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) 시작"

if [ ! -x $CUDA_HOME/bin/nvcc ]; then
  TMP=$(mktemp -d)
  wget -q -O "$TMP/cuda-keyring.deb" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "$TMP/cuda-keyring.deb"
  rm -rf "$TMP"
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      cuda-nvcc-$CUDA_VER cuda-cudart-dev-$CUDA_VER libcublas-dev-$CUDA_VER cuda-nvml-dev-$CUDA_VER
fi
export PATH=$CUDA_HOME/bin:$PATH CUDACXX=$CUDA_HOME/bin/nvcc
nvcc --version | tail -2

if [ ! -d "$LLAMA_DIR" ]; then
  git clone --depth 1 ${LLAMA_REF:+--branch "$LLAMA_REF"} https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
echo "llama.cpp 커밋: $(git log --oneline -1)"
cmake "${CMAKE_ARGS[@]}"
cmake --build build --config Release -j "$JOBS" --target "${TARGETS[@]}"
ls -la build/bin/
echo "== $(date) 끝"
