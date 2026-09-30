#!/bin/bash
# Build the native pi0.5 trainer on Linux / WSL (CUDA 12.8+, sm_120). Links the inference engine library for the
# frozen prefix (build that first: src/pi05_native/build_linux.sh).
#   bash build_linux.sh [build_dir]    -> tgemm_test, tkern_test, pi05_train_verify, pi05_train
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
B=${1:-$HOME/pi05_train_build}
NB=${PI05_NATIVE_BUILD:-$HOME/pi05_native_build}
CUDA=${CUDA_HOME:-/usr/local/cuda-12.8}
ARCH=${PI05_ARCH:-"-gencode arch=compute_120,code=sm_120"}
NVCC="$CUDA/bin/nvcc -std=c++20 -O3 $ARCH -Xcompiler -fPIC,-O3 -lineinfo ${PI05_DEFS}"
mkdir -p $B/obj
cd $HERE/tools
$NVCC -o $B/tgemm_test tgemm_test.cu &
if [ -f ../src/tkern.cu ]; then
  cd $HERE/src
  for f in tkern tkern2 tparams trainer lora augment; do $NVCC -c $f.cu -o $B/obj/$f.o & done
  wait
  ar rcs $B/libpi05train.a $B/obj/*.o
  cd $HERE/tools
  $NVCC -o $B/pi05_train_verify train_verify.cpp $B/libpi05train.a $NB/libpi05.a &
  $NVCC -o $B/pi05_train_bench train_bench.cpp $B/libpi05train.a $NB/libpi05.a &
  $NVCC -o $B/pi05_aug_test aug_test.cu $B/libpi05train.a $NB/libpi05.a &
  FTL=${FT_WORK:-$HOME/fasttrain_work}/build/native
  [ -f $FTL/libftcore.so ] && $NVCC -o $B/pi05_train train_main.cu $B/libpi05train.a $NB/libpi05.a -L$FTL -lftcore -Xlinker -rpath=$FTL &
fi
wait
echo "built into $B"
