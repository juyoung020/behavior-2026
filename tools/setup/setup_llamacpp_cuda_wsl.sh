#!/usr/bin/env bash
# WSL(Ubuntu 22.04)에 llama.cpp 를 CUDA(Blackwell sm_120)로 빌드한다. Qwen3.5-9B 계획기 서버용.
# - CUDA 는 NVIDIA wsl-ubuntu 저장소에서 컴파일러·런타임·cuBLAS 개발 패키지만 받는다(드라이버는 Windows 쪽 것을 씀).
# - sm_120(RTX 50 계열)은 CUDA 12.8 이상이 필요.
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/setup/setup_llamacpp_cuda_wsl.sh
set -euo pipefail
CUDA_VER=12-8
LOG=/mnt/c/behavior-2026/logs/setup_llamacpp_cuda_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) 시작"

if [ ! -x /usr/local/cuda-13.2/bin/nvcc ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      cuda-nvcc-$CUDA_VER cuda-cudart-dev-$CUDA_VER libcublas-dev-$CUDA_VER cuda-nvml-dev-$CUDA_VER
fi
export PATH=/usr/local/cuda-13.2/bin:$PATH
nvcc --version | tail -2

if [ ! -d ~/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
fi
cd ~/llama.cpp
echo "llama.cpp 커밋: $(git log --oneline -1)"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 24 --target llama-server llama-mtmd-cli llama-bench
ls -la build/bin/
echo "== $(date) 끝"
