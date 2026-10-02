// LoRA-mode kernels: SigLIP pieces (LayerNorm with f32 params, bf16 softmax, gelu), f32 stem, helpers.
#include "tkern.cuh"

namespace pi05t {
namespace {
__device__ __forceinline__ float wsum2(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax2(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <int NT>
__device__ float bsum2(float v, float* sh) {
  v = wsum2(v);
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
__device__ float bmax2(float v, float* sh) {
  v = wmax2(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = sh[0];
#pragma unroll
  for (int i = 1; i < NT / 32; ++i) r = fmaxf(r, sh[i]);
  return r;
}
inline int nb(long long n) { return (int)std::min<long long>((n + 255) / 256, 65535LL * 8); }
}  // namespace

// ---- LayerNorm -------------------------------------------------------------------------------------------------------------
template <int NT>
__global__ void __launch_bounds__(NT) ln_fwd_k(const bf16* x, const float* sc, const float* bi, bf16* y, float* mu_o,
                                               float* rs_o, int dim) {
  __shared__ float sh[NT / 32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float s = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = b2f(xr[i]);
    s += v;
    s2 += v * v;
  }
  const float mu = bsum2<NT>(s, sh) / (float)dim;
  const float mu2 = bsum2<NT>(s2, sh) / (float)dim;
  const float var = fmaxf(0.f, mu2 - mu * mu);
  const float r = rsqrtf(var + 1e-6f);
  if (threadIdx.x == 0) { mu_o[blockIdx.x] = mu; rs_o[blockIdx.x] = r; }
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += NT) yr[i] = f2b((b2f(xr[i]) - mu) * (r * sc[i]) + bi[i]);
}
void layernorm_fwd(const bf16* x, const float* scale, const float* bias, bf16* y, float* mu, float* rs, int rows,
                   int dim, cudaStream_t st) {
  ln_fwd_k<128><<<rows, 128, 0, st>>>(x, scale, bias, y, mu, rs, dim);
}
template <int NT>
__global__ void __launch_bounds__(NT) ln_dx_k(const bf16* x, const float* sc, const float* mu_i, const float* rs_i,
                                              const bf16* dy, bf16* dx, int dim) {
  __shared__ float sh[NT / 32];
  const size_t row = (size_t)blockIdx.x * dim;
  const float mu = mu_i[blockIdx.x], r = rs_i[blockIdx.x];
  float a = 0.f, b = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float g = b2f(dy[row + i]) * sc[i];
    const float xh = (b2f(x[row + i]) - mu) * r;
    a += g;
    b += g * xh;
  }
  const float mg = bsum2<NT>(a, sh) / (float)dim;
  const float mgx = bsum2<NT>(b, sh) / (float)dim;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float g = b2f(dy[row + i]) * sc[i];
    const float xh = (b2f(x[row + i]) - mu) * r;
    dx[row + i] = f2b(r * (g - mg - xh * mgx));
  }
}
__global__ void ln_dp_k(const bf16* x, const float* mu_i, const float* rs_i, const bf16* dy, float* dsc, float* dbi,
                        int rows, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  float a = 0.f, b = 0.f;
  for (int t = 0; t < rows; ++t) {
    const float g = b2f(dy[(size_t)t * dim + c]);
    a = fmaf(g, (b2f(x[(size_t)t * dim + c]) - mu_i[t]) * rs_i[t], a);
    b += g;
  }
  dsc[c] += a;
  dbi[c] += b;
}
void layernorm_bwd(const bf16* x, const float* scale, const float* mu, const float* rs, const bf16* dy, bf16* dx,
                   float* dscale, float* dbias, int rows, int dim, cudaStream_t st) {
  if (dx) ln_dx_k<128><<<rows, 128, 0, st>>>(x, scale, mu, rs, dy, dx, dim);
  if (dscale) ln_dp_k<<<cdiv(dim, 128), 128, 0, st>>>(x, mu, rs, dy, dscale, dbias, rows, dim);
}

// ---- bf16 softmax ------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) smb_fwd_k(const bf16* x, bf16* y, int cols) {
  __shared__ float sh[8];
  const bf16* row = x + (size_t)blockIdx.x * cols;
  float m = __int_as_float(0xff800000);
  for (int i = threadIdx.x; i < cols; i += 256) m = fmaxf(m, b2f(row[i]));
  m = bmax2<256>(m, sh);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += bfr(expf(bfr(b2f(row[i]) - m)));
  const float tot = bfr(bsum2<256>(s, sh));
  bf16* yr = y + (size_t)blockIdx.x * cols;
  for (int i = threadIdx.x; i < cols; i += 256) yr[i] = f2b(__fdiv_rn(bfr(expf(bfr(b2f(row[i]) - m))), tot));
}
void softmax_bf16_fwd(const bf16* x, bf16* y, int rows, int cols, cudaStream_t st) {
  smb_fwd_k<<<rows, 256, 0, st>>>(x, y, cols);
}
// transpose of the linear jvp y*(t - sum(y*t)) with bf16 ops: di = bf16(y*g); s = bf16(sum di); dx = bf16(di + bf16(y*(-s)))
__global__ void __launch_bounds__(256) smb_bwd_k(const bf16* y, const bf16* dy, bf16* dx, int cols) {
  __shared__ float sh[8];
  const size_t row = (size_t)blockIdx.x * cols;
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += bfr(b2f(y[row + i]) * b2f(dy[row + i]));
  const float tot = bfr(bsum2<256>(s, sh));
  for (int i = threadIdx.x; i < cols; i += 256) {
    const float yi = b2f(y[row + i]);
    dx[row + i] = f2b(bfr(yi * b2f(dy[row + i])) + bfr(yi * -tot));
  }
}
void softmax_bf16_bwd(const bf16* y, const bf16* dy, bf16* dx, int rows, int cols, cudaStream_t st) {
  smb_bwd_k<<<rows, 256, 0, st>>>(y, dy, dx, cols);
}

// ---- gelu -------------------------------------------------------------------------------------------------------------------
__global__ void gelu_fwd_k(const bf16* h, bf16* a, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    a[i] = f2b(gelu_bf16(b2f(h[i])));
}
void gelu_fwd(const bf16* h, bf16* a, long long n, cudaStream_t st) { gelu_fwd_k<<<nb(n), 256, 0, st>>>(h, a, n); }
__global__ void gelu_bwd_k(const bf16* h, const bf16* da, bf16* dh, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    dh[i] = f2b(gelu_bwd_bf16(b2f(h[i]), b2f(da[i])));
}
void gelu_bwd(const bf16* h, const bf16* da, bf16* dh, long long n, cudaStream_t st) {
  gelu_bwd_k<<<nb(n), 256, 0, st>>>(h, da, dh, n);
}

// ---- misc --------------------------------------------------------------------------------------------------------------------
__global__ void colsum_k(const bf16* x, float* acc, int rows, int cols, long long ld) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float s = 0.f;
  for (int r = 0; r < rows; ++r) s += b2f(x[r * ld + c]);
  acc[c] += s;
}
void colsum_bf16(const bf16* x, float* acc, int rows, int cols, long long ld, cudaStream_t st) {
  colsum_k<<<cdiv(cols, 256), 256, 0, st>>>(x, acc, rows, cols, ld);
}
__global__ void div_k(bf16* x, long long n, float c) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    x[i] = f2b(__fdiv_rn(b2f(x[i]), c));
}
void div_bf16(bf16* x, long long n, float c, cudaStream_t st) { div_k<<<nb(n), 256, 0, st>>>(x, n, c); }

template <bool AK, bool BK>
__global__ void __launch_bounds__(256) sgemm_k(int M, int N, int K, const float* A, long long lda, const float* B,
                                               long long ldb, float* C, long long ldc, bool acc) {
  __shared__ float As[16][64 + 1], Bs[16][64 + 1];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float c[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int i = threadIdx.x; i < 16 * 64; i += 256) {
      const int kk = i / 64, mm = i % 64;
      const int m = m0 + mm, k = k0 + kk;
      As[kk][mm] = (m < M && k < K) ? (AK ? A[(long long)m * lda + k] : A[(long long)k * lda + m]) : 0.f;
      const int n = n0 + mm;
      Bs[kk][mm] = (n < N && k < K) ? (BK ? B[(long long)n * ldb + k] : B[(long long)k * ldb + n]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[i][j] = fmaf(As[kk][ty * 4 + i], Bs[kk][tx * 4 + j], c[i][j]);
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      const int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
      if (m < M && n < N) {
        float* p = C + (long long)m * ldc + n;
        *p = acc ? *p + c[i][j] : c[i][j];
      }
    }
}
void sgemm(bool a_km, bool b_km, int M, int N, int K, const float* A, long long lda, const float* B, long long ldb,
           float* C, long long ldc, bool accumulate, cudaStream_t st) {
  dim3 g(cdiv(N, 64), cdiv(M, 64));
  if (a_km && !b_km) sgemm_k<true, false><<<g, 256, 0, st>>>(M, N, K, A, lda, B, ldb, C, ldc, accumulate);
  else if (a_km && b_km) sgemm_k<true, true><<<g, 256, 0, st>>>(M, N, K, A, lda, B, ldb, C, ldc, accumulate);
  else if (!a_km && !b_km) sgemm_k<false, false><<<g, 256, 0, st>>>(M, N, K, A, lda, B, ldb, C, ldc, accumulate);
  else sgemm_k<false, true><<<g, 256, 0, st>>>(M, N, K, A, lda, B, ldb, C, ldc, accumulate);
}

__global__ void im2col_k(const uint8_t* u8, const float* f32, float* P, int n_img) {
  const long long total = (long long)n_img * 256 * 588;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int col = (int)(i % 588);
    const long long row = i / 588;
    const int im = (int)(row / 256), p = (int)(row % 256), py = p / 16, px = p % 16;
    const int c = col % 3, kw = (col / 3) % 14, kh = col / 42;
    const long long src = (((long long)im * 224 + py * 14 + kh) * 224 + px * 14 + kw) * 3 + c;
    // openpi model.py:118  uint8 / 255.0 * 2.0 - 1.0 (f32)
    P[i] = u8 ? __fsub_rn(__fmul_rn(__fdiv_rn((float)u8[src], 255.0f), 2.0f), 1.0f) : f32[src];
  }
}
void im2col_patches(const uint8_t* img_u8, const float* img_f32, float* patches, int n_img, cudaStream_t st) {
  im2col_k<<<nb((long long)n_img * 256 * 588), 256, 0, st>>>(img_u8, img_f32, patches, n_img);
}
__global__ void stem_fin_k(const float* s, const float* b, const float* pos, bf16* x, int rows) {
  const long long n = (long long)rows * 1152;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const int c = (int)(i % 1152), tok = (int)((i / 1152) % 256);
    x[i] = f2b(__fadd_rn(__fadd_rn(s[i], b[c]), pos[(size_t)tok * 1152 + c]));
  }
}
void stem_finish(const float* stem, const float* bias, const float* pos, bf16* x, int rows, cudaStream_t st) {
  stem_fin_k<<<nb((long long)rows * 1152), 256, 0, st>>>(stem, bias, pos, x, rows);
}
__global__ void sumh_k(const bf16* B, bf16* S, int N, long long rd) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < rd; i += (long long)gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int n = 0; n < N; ++n) s += b2f(B[n * rd + i]);
    S[i] = f2b(s);
  }
}
void sum_heads_bf16(const bf16* B, bf16* Bsum, int N, long long rd, cudaStream_t st) {
  sumh_k<<<nb(rd), 256, 0, st>>>(B, Bsum, N, rd);
}
__global__ void bcast_k(const float* s, float* d, int copies, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    for (int k = 0; k < copies; ++k) d[k * n + i] += s[i];
}
void bcast_add_f32(const float* src, float* dst, int copies, long long n, cudaStream_t st) {
  bcast_k<<<nb(n), 256, 0, st>>>(src, dst, copies, n);
}

}  // namespace pi05t

namespace pi05t {
__global__ void stem_bp_k(const float* d, float* db, float* dpos, int n_img) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 1152) return;
  float sb = 0.f;
  for (int tok = 0; tok < 256; ++tok) {
    float sp = 0.f;
    for (int im = 0; im < n_img; ++im) sp += d[((size_t)im * 256 + tok) * 1152 + c];
    dpos[(size_t)tok * 1152 + c] += sp;
    sb += sp;
  }
  db[c] += sb;
}
void stem_bias_pos_grads(const float* dstem, float* dbias, float* dpos, int n_img, cudaStream_t st) {
  stem_bp_k<<<cdiv(1152, 128), 128, 0, st>>>(dstem, dbias, dpos, n_img);
}
}  // namespace pi05t
