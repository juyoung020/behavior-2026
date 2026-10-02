#!/bin/bash
# Build the pi0.5 native engine on Linux / WSL (CUDA 12.8+, sm_120 = RTX 50xx; add archs for other GPUs).
#   bash build_linux.sh [build_dir]          -> libpi05.a, pi05_verify, pi05_bench, tok_test
# No CMake, no third-party libraries: nvcc + the CUDA runtime only.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
B=${1:-$HOME/pi05_native_build}
CUDA=${CUDA_HOME:-/usr/local/cuda-13.2}
ARCH=${PI05_ARCH:-"-gencode arch=compute_120,code=sm_120"}
NVCC="$CUDA/bin/nvcc -std=c++20 -O3 $ARCH -Xcompiler -fPIC,-O3 -lineinfo ${PI05_DEFS}"
mkdir -p $B/obj
cd $HERE/src
for f in kernels model pb_kernels batch_kernels image_gpu; do $NVCC -c $f.cu -o $B/obj/$f.o & done
for f in tokenizer weights host_io image pb_host engine_api; do
  [ -f $f.cpp ] && $NVCC -x cu -c $f.cpp -o $B/obj/$f.o &
done
wait
ar rcs $B/libpi05.a $B/obj/*.o
cd $HERE/tools
$NVCC -o $B/pi05_verify verify.cpp $B/libpi05.a
$NVCC -o $B/pi05_verify_pb verify_pb.cpp $B/libpi05.a
$NVCC -o $B/pi05_batch_test batch_test.cpp $B/libpi05.a
$NVCC -o $B/pi05_server_ref server_ref.cpp $B/libpi05.a
[ -f bench.cpp ] && $NVCC -o $B/pi05_bench bench.cpp $B/libpi05.a
g++ -O2 -std=c++20 -o $B/tok_test tok_test.cpp ../src/tokenizer.cpp ../src/weights.cpp
echo "built into $B"
# native websocket policy server (submission side)
g++ -O2 -std=c++20 -pthread -o $B/pi05_server $HERE/server/pi05_server.cpp $B/libpi05.a -L$CUDA/lib64 -lcudart_static -ldl -lrt
echo "built $B/pi05_server"
