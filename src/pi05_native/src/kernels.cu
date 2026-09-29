#include "kernels.cuh"

namespace pi05 {

// ---- block reductions -------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <int NT>
__device__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
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
__device__ float block_max(float v, float* sh) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = sh[0];
#pragma unroll
  for (int i = 1; i < NT / 32; ++i) r = fmaxf(r, sh[i]);
  return r;
}

// ---- SigLIP stem ------------------------------------------------------------------------------------
// openpi models/model.py:118  image = uint8 / 255.0 * 2.0 - 1.0          (f32)
// siglip.py:213-239  x = Conv14x14/14(image) + bias (f32) ; x = x + pos_embedding (f32) ; x.astype(bf16)
// f32 SIMT GEMM: patches [n_img*256, 588] x W[1152, 588]^T. Tile 64x64, 256 threads, 4x4 per thread.
__global__ void __launch_bounds__(256) stem_kernel(const uint8_t* __restrict__ img, const bf16* __restrict__ w,
                                                   const bf16* __restrict__ b, const bf16* __restrict__ pos,
                                                   float* __restrict__ stem_out, bf16* __restrict__ x, int M) {
  pdl_entry();
  constexpr int K = 588, N = 1152, BK = 16;
  __shared__ float sa[BK][64 + 4];
  __shared__ float sb[BK][64 + 4];
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < 64 * BK; i += 256) {
      const int r = i / BK, kk = i % BK;
      const int k = k0 + kk, p = m0 + r;
      float av = 0.f, bv = 0.f;
      if (k < K && p < M) {
        const int im = p >> 8, py = (p >> 4) & 15, px = p & 15;
        const int kh = k / 42, kw = (k / 3) % 14, c = k % 3;
        const uint8_t u = img[(((size_t)im * 224 + py * 14 + kh) * 224 + px * 14 + kw) * 3 + c];
        av = __fdiv_rn(__uint2float_rn(u), 255.0f) * 2.0f - 1.0f;
      }
      if (k < K) bv = b2f(w[(size_t)(n0 + r) * K + k]);
      sa[kk][r] = av;
      sb[kk][r] = bv;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a4[4], b4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a4[i] = sa[kk][ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b4[j] = sb[kk][tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a4[i], b4[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int p = m0 + ty * 4 + i;
    if (p >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = n0 + tx * 4 + j;
      const float conv = __fadd_rn(acc[i][j], b2f(b[n]));
      if (stem_out) stem_out[(size_t)p * N + n] = conv;
      x[(size_t)p * N + n] = f2b(__fadd_rn(conv, b2f(pos[(size_t)(p & 255) * N + n])));
    }
  }
}

void launch_stem(const uint8_t* img, const bf16* w, const bf16* b, const bf16* pos, float* stem_out, bf16* x,
                 int n_img, cudaStream_t st) {
  const int M = n_img * 256;
  launch_k(stem_kernel, dim3(1152 / 64, cdiv(M, 64)), 256, 0, st, img, w, b, pos, stem_out, x, M);
}

// ---- LayerNorm (flax nn.LayerNorm(dtype=bf16), use_fast_variance=True) -----------------------------------
// flax/linen/normalization.py:107-137,190-222:
//   x32 = x.astype(f32); mu = mean(x32); var = max(0, mean(x32^2) - mu^2)
//   y = (x32 - mu) * (rsqrt(var + 1e-6) * scale) + bias ; y.astype(bf16)
template <int NT>
__global__ void __launch_bounds__(NT) layernorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ scale,
                                                       const bf16* __restrict__ bias, bf16* __restrict__ y, int dim) {
  pdl_entry();
  __shared__ float sh[NT / 32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float s = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = b2f(xr[i]);
    s += v;
    s2 += v * v;
  }
  const float mu = block_sum<NT>(s, sh) / (float)dim;
  const float mu2 = block_sum<NT>(s2, sh) / (float)dim;
  const float var = fmaxf(0.f, mu2 - mu * mu);
  const float r = rsqrtf(var + 1e-6f);
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float mul = r * b2f(scale[i]);
    yr[i] = f2b((b2f(xr[i]) - mu) * mul + b2f(bias[i]));
  }
}

void launch_layernorm(const bf16* x, const bf16* scale, const bf16* bias, bf16* y, int rows, int dim, cudaStream_t st) {
  launch_k(layernorm_kernel<128>, rows, 128, 0, st, x, scale, bias, y, dim);
}

// ---- SigLIP attention pieces ------------------------------------------------------------------------------
// flax dot_product_attention_weights (attention.py:129-145), dtype bf16:
//   query = query / bf16(sqrt(72))   -> bf16 division by 8.5
//   w = einsum(q, k) -> bf16 ; softmax(w) in bf16 (sum upcast to f32) ; out = einsum(w, v) -> bf16
// prep: qs[r, h*72+d] = bf16(q / 8.5); vt[img][h][d][tok] = v  (so P.V is an NT GEMM)
__global__ void siglip_attn_prep_kernel(const bf16* __restrict__ qkv, bf16* __restrict__ qs, bf16* __restrict__ vt,
                                        int rows) {
  pdl_entry();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 1152) return;
  const int r = i / 1152, c = i % 1152;
  qs[i] = f2b(__fdiv_rn(b2f(qkv[(size_t)r * 3456 + c]), 8.5f));
  const int im = r >> 8, tok = r & 255, h = c / 72, d = c % 72;
  vt[(((size_t)im * 16 + h) * 72 + d) * 256 + tok] = qkv[(size_t)r * 3456 + 2304 + c];
}

void launch_siglip_attn_prep(const bf16* qkv, bf16* qs, bf16* vt, int n_img, cudaStream_t st) {
  const int n = n_img * 256 * 1152;
  launch_k(siglip_attn_prep_kernel, cdiv(n, 256), 256, 0, st, qkv, qs, vt, n_img * 256);
}

// jax.nn.softmax on a bf16 row: m = max; e = bf16(exp(bf16(x - m))); s = bf16(sum_f32(e)); p = bf16(e / s)
__global__ void __launch_bounds__(256) softmax_bf16_kernel(bf16* __restrict__ s, int cols) {
  pdl_entry();
  __shared__ float sh[8];
  bf16* row = s + (size_t)blockIdx.x * cols;
  float m = __int_as_float(0xff800000);  // -inf
  for (int i = threadIdx.x; i < cols; i += 256) m = fmaxf(m, b2f(row[i]));
  m = block_max<256>(m, sh);
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) sum += bfr(expf(bfr(b2f(row[i]) - m)));
  const float tot = bfr(block_sum<256>(sum, sh));
  for (int i = threadIdx.x; i < cols; i += 256) {
    const float e = bfr(expf(bfr(b2f(row[i]) - m)));
    row[i] = f2b(__fdiv_rn(e, tot));
  }
}

void launch_softmax_bf16_rows(bf16* s, int rows, int cols, cudaStream_t st) {
  launch_k(softmax_bf16_kernel, rows, 256, 0, st, s, cols);
}

// ---- Gemma RMSNorm (gemma.py:113-131) ---------------------------------------------------------------------
//   var = mean(x32^2); normed = x * reciprocal(sqrt(var + 1e-6))      (f32)
//   plain:    y = normed * bf16(1 + scale)                      -> bf16
//   adaptive: y = normed * bf16(1 + scale_c) + shift_c           -> bf16   (scale/shift/gate = Dense(cond), bf16)
__device__ __forceinline__ float rms_inv(float var) {
#ifdef PI05_RMS_DIVSQRT
  return __fdiv_rn(1.0f, __fsqrt_rn(var + 1e-6f));
#else
  return rsqrtf(var + 1e-6f);
#endif
}

template <int NT>
__global__ void __launch_bounds__(NT) rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ scale,
                                                     bf16* __restrict__ y, int dim, const int* d_rows) {
  pdl_entry();
  if (d_rows && (int)blockIdx.x >= *d_rows) return;
  __shared__ float sh[NT / 32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float s2 = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = b2f(xr[i]);
    s2 += v * v;
  }
  const float r = rms_inv(block_sum<NT>(s2, sh) / (float)dim);
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += NT) yr[i] = f2b(b2f(xr[i]) * r * bfr(1.0f + b2f(scale[i])));
}

void launch_rmsnorm(const bf16* x, const bf16* scale, bf16* y, int rows, int dim, const int* d_rows, cudaStream_t st) {
  launch_k(rmsnorm_kernel<256>, rows, 256, 0, st, x, scale, y, dim, d_rows);
}

// mod = [scale | shift | gate] (bf16, 3*dim), shared by every row (cond is per batch, gemma.py:128-130)
template <int NT>
__global__ void __launch_bounds__(NT) adarms_kernel(const bf16* __restrict__ x, const bf16* __restrict__ mod,
                                                    bf16* __restrict__ y, int dim) {
  pdl_entry();
  __shared__ float sh[NT / 32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float s2 = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = b2f(xr[i]);
    s2 += v * v;
  }
  const float r = rms_inv(block_sum<NT>(s2, sh) / (float)dim);
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += NT)
    yr[i] = f2b(b2f(xr[i]) * r * bfr(1.0f + b2f(mod[i])) + b2f(mod[dim + i]));
}

void launch_adarms(const bf16* x, const bf16* mod, bf16* y, int rows, int dim, cudaStream_t st) {
  launch_k(adarms_kernel<256>, rows, 256, 0, st, x, mod, y, dim);
}

// ---- RoPE (gemma.py:424-440) ----------------------------------------------------------------------------------
//   freq_exp = (2/256) * arange(128) ; timescale = 10000 ** freq_exp ; rad = pos / timescale  (f32)
//   out = [x1*cos - x2*sin, x2*cos + x1*sin] in f32, then .astype(bf16); q additionally *= 256**-0.5 (bf16)
__global__ void rope_table_kernel(float2* table, int max_pos, int half) {
  pdl_entry();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_pos * half) return;
  const int pos = i / half, j = i % half;
  const float fe = (2.0f / (2 * half)) * (float)j;
  const float ts = powf(10000.0f, fe);
  const float rad = __fdiv_rn((float)pos, ts);
  table[i] = make_float2(cosf(rad), sinf(rad));
}

void launch_rope_tables(float2* table, int max_pos, int half, cudaStream_t st) {
  launch_k(rope_table_kernel, cdiv(max_pos * half, 256), 256, 0, st, table, max_pos, half);
}

// one block per token row, 128 threads: each thread owns rotation pair j (x[j], x[j+128])
__global__ void __launch_bounds__(128) rope_split_kernel(const bf16* __restrict__ qkv, const float2* __restrict__ rope,
                                                         bf16* __restrict__ q_out, bf16* __restrict__ kc,
                                                         bf16* __restrict__ vt, const int* d_rows, int heads,
                                                         int vt_ld, int pos0, const int* d_pos0) {
  pdl_entry();
  const int r = blockIdx.x;
  if (d_rows && r >= *d_rows) return;
  const int p0 = d_pos0 ? *d_pos0 : pos0;
  const int pos = p0 + r;
  const int j = threadIdx.x;
  const float2 cs = rope[pos * 128 + j];
  const int width = (heads + 2) * 256;
  const bf16* row = qkv + (size_t)r * width;
  for (int h = 0; h < heads; ++h) {
    const float x1 = b2f(row[h * 256 + j]), x2 = b2f(row[h * 256 + j + 128]);
    const float o1 = bfr(x1 * cs.x - x2 * cs.y), o2 = bfr(x2 * cs.x + x1 * cs.y);
    bf16* qo = q_out + ((size_t)r * heads + h) * 256;
    qo[j] = f2b(o1 * 0.0625f);
    qo[j + 128] = f2b(o2 * 0.0625f);
  }
  {
    const float x1 = b2f(row[heads * 256 + j]), x2 = b2f(row[heads * 256 + j + 128]);
    kc[(size_t)pos * 256 + j] = f2b(x1 * cs.x - x2 * cs.y);
    kc[(size_t)pos * 256 + j + 128] = f2b(x2 * cs.x + x1 * cs.y);
  }
  vt[(size_t)j * vt_ld + pos] = row[(heads + 1) * 256 + j];
  vt[(size_t)(j + 128) * vt_ld + pos] = row[(heads + 1) * 256 + j + 128];
}

void launch_rope_split(const bf16* qkv, const float2* rope, bf16* q_out, bf16* kc, bf16* vt, int rows,
                       const int* d_rows, int heads, int vt_ld, int pos0, const int* d_pos0, cudaStream_t st) {
  launch_k(rope_split_kernel, rows, 128, 0, st, qkv, rope, q_out, kc, vt, d_rows, heads, vt_ld, pos0, d_pos0);
}

// ---- f32 attention softmax (gemma.py:217-228) --------------------------------------------------------------------
//   logits f32 ; probs = softmax(logits).astype(bf16). Columns [cols, cols8) are written as exact zeros so the
//   P.V GEMM can run its K loop to a multiple of 8.
__global__ void __launch_bounds__(256) softmax_f32_kernel(const float* __restrict__ lg, int ld_in, bf16* __restrict__ p,
                                                          int ld_out, const int* d_rows, const int* d_cols,
                                                          const int* d_cols8, int g0, int heads) {
  pdl_entry();
  if (d_rows && (int)blockIdx.x >= *d_rows) return;
  __shared__ float sh[8];
  // g0 > 0: two attention groups (PiBehavior prefix, pi_behavior.py:590-609): query tokens < g0 see keys < g0 only
  const int cols = (g0 > 0 && (int)blockIdx.x / heads < g0) ? g0 : *d_cols, cols8 = *d_cols8;
  const float* row = lg + (size_t)blockIdx.x * ld_in;
  float m = __int_as_float(0xff800000);  // -inf
  for (int i = threadIdx.x; i < cols; i += 256) m = fmaxf(m, row[i]);
  m = block_max<256>(m, sh);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += expf(row[i] - m);
  const float tot = block_sum<256>(s, sh);
  bf16* pr = p + (size_t)blockIdx.x * ld_out;
  for (int i = threadIdx.x; i < cols8; i += 256)
    pr[i] = i < cols ? f2b(__fdiv_rn(expf(row[i] - m), tot)) : f2b(0.f);
}

void launch_softmax_f32_rows(const float* logits, int ld_in, bf16* p, int ld_out, int rows, const int* d_rows,
                             const int* d_cols, const int* d_cols8, cudaStream_t st, int g0, int heads) {
  launch_k(softmax_f32_kernel, rows, 256, 0, st, logits, ld_in, p, ld_out, d_rows, d_cols, d_cols8, g0, heads);
}

// ---- action expert input / flow update ------------------------------------------------------------------------------
// pi0.py:159  action_in_proj(x_t): nnx.Linear, f32 input x bf16 kernel -> f32 ; later .astype(bf16) (gemma.py:400)
__global__ void action_in_kernel(const float* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                 bf16* __restrict__ h, int in_dim, int out_dim) {
  pdl_entry();
  const int r = blockIdx.y;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_dim) return;
  float acc = 0.f;
  for (int k = 0; k < in_dim; ++k) acc = fmaf(x[r * in_dim + k], b2f(w[(size_t)o * in_dim + k]), acc);
  h[(size_t)r * out_dim + o] = f2b(acc + b2f(b[o]));
}

void launch_action_in(const float* x, const bf16* w, const bf16* b, bf16* h, int rows, int in_dim, int out_dim,
                      cudaStream_t st) {
  launch_k(action_in_kernel, dim3(cdiv(out_dim, 128), rows), 128, 0, st, x, w, b, h, in_dim, out_dim);
}

// pi0.py:271  x_t + dt * v_t : dt is a Python float, so it becomes bf16(dt) (bf16(-0.1) = -0.10009765625) and dt*v_t
// is a bf16 product; x_t is f32 (bit-checked against the JAX dump). dtb = bf16(-1/num_steps) as a float.
__global__ void flow_update_kernel(float* x, const bf16* v, int n, float dtb) {
  pdl_entry();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = x[i] + bfr(dtb * b2f(v[i]));
}

void launch_flow_update(float* x, const bf16* v, int n, float dtb, cudaStream_t st) {
  launch_k(flow_update_kernel, cdiv(n, 256), 256, 0, st, x, v, n, dtb);
}

// ---- time conditioning (pi0.py:47-63, 161-167), computed once per denoising step at load time ----------------------
//   fraction = linspace(0, 1, dim/2); period = 4e-3 * (4.0/4e-3) ** fraction
//   emb = [sin(t * (1/period * 2 * pi)), cos(...)]  (f32)
__global__ void time_embed_kernel(const float* times, int dim, float* out) {
  pdl_entry();
  const int s = blockIdx.x;
  const int half = dim / 2;
  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    const float frac = __fdiv_rn((float)j, (float)(half - 1));
    const float period = 4e-3f * powf(1000.0f, frac);
    const float f = __fmul_rn(__fmul_rn(__fdiv_rn(1.0f, period), 2.0f), 3.14159274101257324f);
    const float a = __fmul_rn(times[s], f);
    out[(size_t)s * dim + j] = sinf(a);
    out[(size_t)s * dim + half + j] = cosf(a);
  }
}

void launch_time_embed(const float* times, int n_steps, int dim, float* out, cudaStream_t st) {
  launch_k(time_embed_kernel, n_steps, 256, 0, st, times, dim, out);
}

// y = x . W^T + b in f32 (nnx.Linear with f32 input, bf16 kernel); optional swish (jax.nn.swish = x * sigmoid(x))
__global__ void linear_f32_kernel(const float* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                  float* __restrict__ y, int in_dim, int out_dim, int swish) {
  pdl_entry();
  const int r = blockIdx.y;
  const int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (o >= out_dim) return;
  float acc = 0.f;
  for (int k = lane; k < in_dim; k += 32) acc = fmaf(x[(size_t)r * in_dim + k], b2f(w[(size_t)o * in_dim + k]), acc);
  acc = warp_sum(acc);
  if (lane == 0) {
    float v = acc + b2f(b[o]);
    if (swish == 1) v = v * __fdiv_rn(1.0f, 1.0f + expf(-v));
    else if (swish == 2) v = __fdiv_rn(1.0f, 1.0f + expf(-v));  // jax.nn.sigmoid
    else if (swish == 3) v = fmaxf(v, 0.0f);                    // relu
    y[(size_t)r * out_dim + o] = v;
  }
}

void launch_linear_f32(const float* x, const bf16* w, const bf16* b, float* y, int rows, int in_dim, int out_dim,
                       int swish, cudaStream_t st) {
  launch_k(linear_f32_kernel, dim3(cdiv(out_dim, 8), rows), 256, 0, st, x, w, b, y, in_dim, out_dim, swish);
}

// flax nn.Dense(dtype=bf16) on an f32 input: x -> bf16, dot -> bf16, + bias (bf16)
__global__ void linear_bf16_vec_kernel(const float* __restrict__ x, const bf16* __restrict__ w,
                                       const bf16* __restrict__ b, bf16* __restrict__ y, int in_dim, int out_dim) {
  pdl_entry();
  const int r = blockIdx.y;
  const int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (o >= out_dim) return;
  float acc = 0.f;
  for (int k = lane; k < in_dim; k += 32) acc = fmaf(bfr(x[(size_t)r * in_dim + k]), b2f(w[(size_t)o * in_dim + k]), acc);
  acc = warp_sum(acc);
  if (lane == 0) y[(size_t)r * out_dim + o] = f2b(bfr(acc) + b2f(b[o]));
}

void launch_linear_bf16_vec(const float* x, const bf16* w, const bf16* b, bf16* y, int rows, int in_dim, int out_dim,
                            cudaStream_t st) {
  launch_k(linear_bf16_vec_kernel, dim3(cdiv(out_dim, 8), rows), 256, 0, st, x, w, b, y, in_dim, out_dim);
}

}  // namespace pi05
