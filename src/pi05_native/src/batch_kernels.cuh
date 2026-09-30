// Batched copies of per-inference kernels (batched suffix of pi05_infer_batch). Each keeps the single-inference
// kernel's arithmetic exactly (same thread mapping and reductions), only indexing an episode by blockIdx.y / .x.
#pragma once
#include "common.cuh"

namespace pi05 {
// softmax_f32_kernel per episode e = blockIdx.y: rows of episode e start at lg + e*lg_stride, keys [0, cols[e]),
// zeros written up to `fill` (the batched P.V GEMM's K extent)
void launch_softmax_f32_rows_b(const float* lg, int ld_in, long long lg_stride, bf16* p, int ld_out, long long p_stride,
                               int rows, int n, const int* d_cols, int fill, cudaStream_t st);
// inpaint_kernel per episode e = blockIdx.x (x, x0O, zO, ints offset by their per-episode strides)
void launch_inpaint_b(float* x, long long x_stride, const float* x0O, const float* zO, long long o_stride,
                      const float* C, int nO, int nU, float t_new, const int* ints, int n, cudaStream_t st);
}  // namespace pi05
