#include "batch_kernels.cuh"

namespace pi05 {
namespace {
__device__ __forceinline__ float wsum_b(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax_b(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <int NT>
__device__ float bsum_b(float v, float* sh) {  // == kernels.cu block_sum
  v = wsum_b(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) r += sh[i];
  return r;
}
template <int NT>
__device__ float bmax_b(float v, float* sh) {  // == kernels.cu block_max
  v = wmax_b(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = sh[0];
#pragma unroll
  for (int i = 1; i < NT / 32; ++i) r = fmaxf(r, sh[i]);
  return r;
}
}  // namespace

__global__ void __launch_bounds__(256) softmax_f32_b_kernel(const float* __restrict__ lg, int ld_in, long long lg_stride,
                                                            bf16* __restrict__ p, int ld_out, long long p_stride,
                                                            const int* d_cols, int fill) {
  pdl_entry();
  __shared__ float sh[8];
  const int e = blockIdx.y;
  const int cols = d_cols[e];
  const float* row = lg + e * lg_stride + (size_t)blockIdx.x * ld_in;
  float m = __int_as_float(0xff800000);
  for (int i = threadIdx.x; i < cols; i += 256) m = fmaxf(m, row[i]);
  m = bmax_b<256>(m, sh);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += expf(row[i] - m);
  const float tot = bsum_b<256>(s, sh);
  bf16* pr = p + e * p_stride + (size_t)blockIdx.x * ld_out;
  for (int i = threadIdx.x; i < fill; i += 256) pr[i] = i < cols ? f2b(__fdiv_rn(expf(row[i] - m), tot)) : f2b(0.f);
}
void launch_softmax_f32_rows_b(const float* lg, int ld_in, long long lg_stride, bf16* p, int ld_out, long long p_stride,
                               int rows, int n, const int* d_cols, int fill, cudaStream_t st) {
  launch_k(softmax_f32_b_kernel, dim3(rows, n), 256, 0, st, lg, ld_in, lg_stride, p, ld_out, p_stride, d_cols, fill);
}

__global__ void __launch_bounds__(1024) inpaint_b_kernel(float* xb, long long xs, const float* x0b, const float* zb,
                                                         long long os, const float* C, int nO, int nU, float t_new,
                                                         const int* intsb) {
  pdl_entry();
  const int e = blockIdx.x;
  const int* ints = intsb + e * 4;
  if (!ints[2]) return;
  float* x = xb + e * xs;
  const float* x0O = x0b + e * os;
  const float* zO = zb + e * os;
  __shared__ float dO[512];
  __shared__ float sh[32];
  for (int i = threadIdx.x; i < nO; i += blockDim.x) {
    const float want = (1.0f - t_new) * x0O[i] + t_new * zO[i];
    dO[i] = want - x[i];
  }
  __syncthreads();
  float mx = 0.f, du = 0.f;
  const int u = threadIdx.x;
  if (u < nU) {
    for (int k = 0; k < nO; ++k) du = fmaf(dO[k], C[(size_t)u * nO + k], du);
    mx = fabsf(du);
  }
  mx = wmax_b(mx);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
  __syncthreads();
  float gmax = 0.f;
  for (int i = 0; i < (int)(blockDim.x / 32); ++i) gmax = fmaxf(gmax, sh[i]);
  for (int i = threadIdx.x; i < nO; i += blockDim.x) x[i] = (1.0f - t_new) * x0O[i] + t_new * zO[i];
  if (gmax <= 1.0f && u < nU) x[nO + u] += du;
}
void launch_inpaint_b(float* x, long long x_stride, const float* x0O, const float* zO, long long o_stride,
                      const float* C, int nO, int nU, float t_new, const int* ints, int n, cudaStream_t st) {
  launch_k(inpaint_b_kernel, n, 1024, 0, st, x, x_stride, x0O, zO, o_stride, C, nO, nU, t_new, ints);
}
}  // namespace pi05
