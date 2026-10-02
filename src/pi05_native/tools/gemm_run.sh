#!/bin/bash
# Build + run the GEMM unit test (small GPU memory, a few seconds).  arg: --bench for throughput
set -e
B=~/pi05_native_build; mkdir -p $B
N=/usr/local/cuda-13.2/bin/nvcc
cd /mnt/c/behavior-2026/src/pi05_native/tools
$N -std=c++20 -O3 -gencode arch=compute_120,code=sm_120 -o $B/gemm_test gemm_test.cu
$N -std=c++20 -O3 -gencode arch=compute_120,code=sm_120 -DWITH_CUBLAS -o $B/gemm_test_cublas gemm_test.cu -lcublasLt
if [ "$1" = "--bench" ]; then $B/gemm_test_cublas --bench; else $B/gemm_test; fi
