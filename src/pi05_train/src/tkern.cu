#include "tkern.cuh"

namespace pi05t {
namespace {

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <int NT>
__device__ float bsum(float v, float* sh) {
  v = wsum(v);
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
__device__ float bmax(float v, float* sh) {
  v = wmax(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = sh[0];
#pragma unroll
  for (int i = 1; i < NT / 32; ++i) r = fmaxf(r, sh[i]);
  return r;
}
__device__ __forceinline__ float sigm(float z) { return __fdiv_rn(1.0f, 1.0f + expf(-z)); }
inline int nblk(long long n, int t = 256) { return (int)std::min<long long>((n + t - 1) / t, 65535LL * 8); }

}  // namespace

// ---- f32 dense ---------------------------------------------------------------------------------------------------------
__global__ void lin_f32_fwd_k(const float* xf, const bf16* xb, const float* W, const float* b, float* z, float* y,
                              int in, int out, int act) {
  const int r = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  float acc = 0.f;
  for (int i = 0; i < in; ++i) {
    const float xv = xf ? xf[(size_t)r * in + i] : b2f(xb[(size_t)r * in + i]);
    acc = fmaf(xv, W[(size_t)i * out + o], acc);
  }
  const float v = acc + b[o];
  if (z) z[(size_t)r * out + o] = v;
  y[(size_t)r * out + o] = act == 1 ? v * sigm(v) : v;
}
void lin_f32_fwd(const float* xf, const bf16* xb, const float* W, const float* b, float* z, float* y, int rows,
                 int in, int out, int act, cudaStream_t st) {
  lin_f32_fwd_k<<<dim3(cdiv(out, 128), rows), 128, 0, st>>>(xf, xb, W, b, z, y, in, out, act);
}

__global__ void lin_act_bwd_k(const float* z, const float* dy, float* dz, int n, int act) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = dy[i];
  if (act == 1) {  // d/dz z*s(z) = s + z s (1 - s)
    const float s = sigm(z[i]);
    g = g * s + g * z[i] * s * (1.0f - s);
  }
  dz[i] = g;
}
// dx[r][i] = sum_o dz[r][o] W[i][o]   (warp per (r, i), W row i contiguous)
__global__ void lin_dx_k(const float* dz, const float* W, float* dx, int in, int out) {
  const int r = blockIdx.y;
  const int i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (i >= in) return;
  float acc = 0.f;
  for (int o = lane; o < out; o += 32) acc = fmaf(dz[(size_t)r * out + o], W[(size_t)i * out + o], acc);
  acc = wsum(acc);
  if (lane == 0) dx[(size_t)r * in + i] = acc;
}
// dW[i][o] += sum_r x[r][i] dz[r][o]; db[o] += sum_r dz[r][o]
__global__ void lin_dw_k(const float* xf, const bf16* xb, const float* dz, float* dW, float* db, int rows, int in,
                         int out) {
  const int i = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  float acc = 0.f, accb = 0.f;
  for (int r = 0; r < rows; ++r) {
    const float xv = xf ? xf[(size_t)r * in + i] : b2f(xb[(size_t)r * in + i]);
    const float g = dz[(size_t)r * out + o];
    acc = fmaf(xv, g, acc);
    accb += g;
  }
  dW[(size_t)i * out + o] += acc;
  if (i == 0 && db) db[o] += accb;
}
void lin_f32_bwd(const float* xf, const bf16* xb, const float* W, const float* z, const float* dy, int act, float* dx,
                 float* dW, float* db, int rows, int in, int out, float* dz_tmp, cudaStream_t st) {
  const float* dz = dy;
  if (act) {
    lin_act_bwd_k<<<cdiv(rows * out, 256), 256, 0, st>>>(z, dy, dz_tmp, rows * out, act);
    dz = dz_tmp;
  }
  if (dx) lin_dx_k<<<dim3(cdiv(in, 8), rows), 256, 0, st>>>(dz, W, dx, in, out);
  if (dW) lin_dw_k<<<dim3(cdiv(out, 128), in), 128, 0, st>>>(xf, xb, dz, dW, db, rows, in, out);
}

// ---- bf16 Dense on the condition vector ----------------------------------------------------------------------------------
__global__ void mod_fwd_k(const float* cond, const bf16* Wb, const bf16* bb, bf16* mod, int in, int out) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  float acc = 0.f;
  for (int i = 0; i < in; ++i) acc = fmaf(bfr(cond[i]), b2f(Wb[(size_t)i * out + o]), acc);
  mod[o] = f2b(bfr(acc) + b2f(bb[o]));
}
void mod_fwd(const float* cond, const bf16* Wb, const bf16* bb, bf16* mod, int in, int out, cudaStream_t st) {
  mod_fwd_k<<<cdiv(out, 128), 128, 0, st>>>(cond, Wb, bb, mod, in, out);
}
__global__ void mod_dw_k(const float* cond, const bf16* dmod, float* dW, float* db, int in, int out) {
  const int i = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  const float g = b2f(dmod[o]);
  dW[(size_t)i * out + o] += bfr(cond[i]) * g;
  if (i == 0) db[o] += g;
}
__global__ void mod_dcond_k(const bf16* Wb, const bf16* dmod, float* dcond, int in, int out) {
  const int i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (i >= in) return;
  float acc = 0.f;
  for (int o = lane; o < out; o += 32) acc = fmaf(b2f(dmod[o]), b2f(Wb[(size_t)i * out + o]), acc);
  acc = wsum(acc);
  if (lane == 0) dcond[i] += bfr(acc);
}
void mod_bwd(const float* cond, const bf16* Wb, const bf16* dmod, float* dW, float* db, float* dcond, int in, int out,
             cudaStream_t st) {
  if (dW) mod_dw_k<<<dim3(cdiv(out, 256), in), 256, 0, st>>>(cond, dmod, dW, db, in, out);
  if (dcond) mod_dcond_k<<<cdiv(in, 8), 256, 0, st>>>(Wb, dmod, dcond, in, out);
}

// ---- adaRMS ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float rinv(float var) { return rsqrtf(var + 1e-6f); }

template <int NT>
__global__ void __launch_bounds__(NT) adarms_fwd_k(const bf16* x, const bf16* mod, bf16* y, float* rr, int dim) {
  __shared__ float sh[NT / 32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float s2 = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float v = b2f(xr[i]);
    s2 += v * v;
  }
  const float r = rinv(bsum<NT>(s2, sh) / (float)dim);
  if (threadIdx.x == 0) rr[blockIdx.x] = r;
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += NT)
    yr[i] = f2b(b2f(xr[i]) * r * bfr(1.0f + b2f(mod[i])) + b2f(mod[dim + i]));
}
void adarms_fwd(const bf16* x, const bf16* mod, bf16* y, float* r, int rows, int dim, cudaStream_t st) {
  adarms_fwd_k<256><<<rows, 256, 0, st>>>(x, mod, y, r, dim);
}
template <int NT>
__global__ void __launch_bounds__(NT) adarms_dx_k(const bf16* x, const bf16* mod, const float* rr, const bf16* dy,
                                                  bf16* dx, int dim) {
  __shared__ float sh[NT / 32];
  const size_t row = (size_t)blockIdx.x * dim;
  const float r = rr[blockIdx.x];
  float s = 0.f;
  for (int i = threadIdx.x; i < dim; i += NT) s += b2f(dy[row + i]) * bfr(1.0f + b2f(mod[i])) * b2f(x[row + i]);
  const float tot = bsum<NT>(s, sh);
  const float c = r * r * r * tot / (float)dim;
  for (int i = threadIdx.x; i < dim; i += NT) {
    const float dn = b2f(dy[row + i]) * bfr(1.0f + b2f(mod[i]));
    dx[row + i] = f2b(r * dn - b2f(x[row + i]) * c);
  }
}
// dss[c] = sum_t dy*x*r (scale), dss[dim + c] = sum_t dy (shift)
__global__ void adarms_dmod_k(const bf16* x, const float* rr, const bf16* dy, float* dss, int rows, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  float a = 0.f, b = 0.f;
  for (int t = 0; t < rows; ++t) {
    const float g = b2f(dy[(size_t)t * dim + c]);
    a = fmaf(g, b2f(x[(size_t)t * dim + c]) * rr[t], a);
    b += g;
  }
  dss[c] = a;
  dss[dim + c] = b;
}
void adarms_bwd(const bf16* x, const bf16* mod, const float* r, const bf16* dy, bf16* dx, float* dss, int rows,
                int dim, cudaStream_t st) {
  adarms_dx_k<256><<<rows, 256, 0, st>>>(x, mod, r, dy, dx, dim);
  adarms_dmod_k<<<cdiv(dim, 256), 256, 0, st>>>(x, r, dy, dss, rows, dim);
}

// ---- RoPE -------------------------------------------------------------------------------------------------------------
__global__ void rope_fwd_k(bf16* x, const float2* tab, int heads, int rs, int pos0, bool sq) {
  const int r = blockIdx.x, j = threadIdx.x;  // 128 threads
  const float2 cs = tab[(size_t)(pos0 + r) * 128 + j];
  for (int h = 0; h < heads; ++h) {
    bf16* p = x + (size_t)r * rs + h * 256;
    const float x1 = b2f(p[j]), x2 = b2f(p[j + 128]);
    float o1 = bfr(x1 * cs.x - x2 * cs.y), o2 = bfr(x2 * cs.x + x1 * cs.y);
    if (sq) { o1 *= 0.0625f; o2 *= 0.0625f; }
    p[j] = f2b(o1);
    p[j + 128] = f2b(o2);
  }
}
void rope_fwd(bf16* x, const float2* table, int rows, int heads, int row_stride, int pos0, bool scale_q, cudaStream_t st) {
  rope_fwd_k<<<rows, 128, 0, st>>>(x, table, heads, row_stride, pos0, scale_q);
}
// VJP: g1, g2 = cotangents of (o1, o2) (after the exact *0.0625): dx1 = g1 c + g2 s, dx2 = -g1 s + g2 c (f32 -> bf16)
__global__ void rope_bwd_k(bf16* dx, const float2* tab, int heads, int rs, int pos0, bool sq) {
  const int r = blockIdx.x, j = threadIdx.x;
  const float2 cs = tab[(size_t)(pos0 + r) * 128 + j];
  for (int h = 0; h < heads; ++h) {
    bf16* p = dx + (size_t)r * rs + h * 256;
    float g1 = b2f(p[j]), g2 = b2f(p[j + 128]);
    if (sq) { g1 = bfr(g1 * 0.0625f); g2 = bfr(g2 * 0.0625f); }
    p[j] = f2b(g1 * cs.x + g2 * cs.y);
    p[j + 128] = f2b(g2 * cs.x - g1 * cs.y);
  }
}
void rope_bwd(bf16* dx, const float2* table, int rows, int heads, int row_stride, int pos0, bool scale_q,
              cudaStream_t st) {
  rope_bwd_k<<<rows, 128, 0, st>>>(dx, table, heads, row_stride, pos0, scale_q);
}
__global__ void pack_dmod_k(const float* dss, const float* dg, bf16* dmod, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  dmod[c] = f2b(dss[c]);
  dmod[dim + c] = f2b(dss[dim + c]);
  dmod[2 * dim + c] = f2b(dg ? dg[c] : 0.f);
}
void pack_dmod(const float* dss, const float* dgate, bf16* dmod, int dim, cudaStream_t st) {
  pack_dmod_k<<<cdiv(dim, 256), 256, 0, st>>>(dss, dgate, dmod, dim);
}

// ---- softmax ------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) softmax_fwd_k(const float* lg, float* p32, bf16* pb, int cols, int ld) {
  __shared__ float sh[8];
  const float* row = lg + (size_t)blockIdx.x * ld;
  float m = __int_as_float(0xff800000);
  for (int i = threadIdx.x; i < cols; i += 256) m = fmaxf(m, row[i]);
  m = bmax<256>(m, sh);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += expf(row[i] - m);
  const float tot = bsum<256>(s, sh);
  for (int i = threadIdx.x; i < ld; i += 256) {
    const float v = i < cols ? __fdiv_rn(expf(row[i] - m), tot) : 0.f;
    p32[(size_t)blockIdx.x * ld + i] = v;
    pb[(size_t)blockIdx.x * ld + i] = f2b(v);
  }
}
void softmax_fwd(const float* logits, float* p32, bf16* pb, int rows, int cols, int ld, cudaStream_t st) {
  softmax_fwd_k<<<rows, 256, 0, st>>>(logits, p32, pb, cols, ld);
}
// d3 layout per row: [hi (ld) | mid (ld) | lo (ld)]
__global__ void __launch_bounds__(256) softmax_bwd_k(const float* p32, const bf16* dpb, bf16* d3, int cols, int ld) {
  __shared__ float sh[8];
  const size_t row = (size_t)blockIdx.x * ld;
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += 256) s += p32[row + i] * b2f(dpb[row + i]);
  const float tot = bsum<256>(s, sh);
  bf16* o = d3 + (size_t)blockIdx.x * 3 * ld;
  for (int i = threadIdx.x; i < ld; i += 256) {
    float d = 0.f;
    if (i < cols) {
      const float y = p32[row + i];
      d = y * b2f(dpb[row + i]) - y * tot;
    }
    const bf16 hi = f2b(d);
    const float r1 = d - b2f(hi);
    const bf16 mid = f2b(r1);
    o[i] = hi;
    o[ld + i] = mid;
    o[2 * ld + i] = f2b(r1 - b2f(mid));
  }
}
void softmax_bwd_split(const float* p32, const bf16* dpb, bf16* d3, int rows, int cols, int ld, cudaStream_t st) {
  softmax_bwd_k<<<rows, 256, 0, st>>>(p32, dpb, d3, cols, ld);
}

// ---- gated residual -------------------------------------------------------------------------------------------------------
__global__ void gres_fwd_k(const bf16* x, const bf16* y, const bf16* gate, bf16* out, int n, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f2b(b2f(x[i]) + bfr(b2f(y[i]) * b2f(gate[i % dim])));
}
void gated_res_fwd(const bf16* x, const bf16* y, const bf16* gate, bf16* out, int rows, int dim, cudaStream_t st) {
  gres_fwd_k<<<cdiv(rows * dim, 256), 256, 0, st>>>(x, y, gate, out, rows * dim, dim);
}
__global__ void gres_bwd_k(const bf16* dout, const bf16* y, const bf16* gate, bf16* dy, float* dg, int rows, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  const float gt = b2f(gate[c]);
  float acc = 0.f;
  for (int t = 0; t < rows; ++t) {
    const float d = b2f(dout[(size_t)t * dim + c]);
    acc += bfr(d * b2f(y[(size_t)t * dim + c]));
    dy[(size_t)t * dim + c] = f2b(d * gt);
  }
  dg[c] = acc;
}
void gated_res_bwd(const bf16* dout, const bf16* y, const bf16* gate, bf16* dy, float* dgate32, int rows, int dim,
                   cudaStream_t st) {
  gres_bwd_k<<<cdiv(dim, 256), 256, 0, st>>>(dout, y, gate, dy, dgate32, rows, dim);
}

// ---- gelu * up ----------------------------------------------------------------------------------------------------------
__global__ void gelu_mul_fwd_k(const bf16* g, const bf16* u, bf16* a, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = f2b(gelu_bf16(b2f(g[i])) * b2f(u[i]));
}
void gelu_mul_fwd(const bf16* g, const bf16* u, bf16* a, int n, cudaStream_t st) {
  gelu_mul_fwd_k<<<cdiv(n, 256), 256, 0, st>>>(g, u, a, n);
}
__global__ void gelu_mul_bwd_k(const bf16* g, const bf16* u, const bf16* da, bf16* dg, bf16* du, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = b2f(g[i]), d = b2f(da[i]);
  du[i] = f2b(d * gelu_bf16(x));
  dg[i] = f2b(gelu_bwd_bf16(x, bfr(d * b2f(u[i]))));
}
void gelu_mul_bwd(const bf16* g, const bf16* u, const bf16* da, bf16* dg, bf16* du, int n, cudaStream_t st) {
  gelu_mul_bwd_k<<<cdiv(n, 256), 256, 0, st>>>(g, u, da, dg, du, n);
}

// ---- elementwise ------------------------------------------------------------------------------------------------------------
__global__ void add_bf16_k(const bf16* a, const bf16* b, bf16* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = f2b(b2f(a[i]) + b2f(b[i]));
}
void add_bf16(const bf16* a, const bf16* b, bf16* out, int n, cudaStream_t st) {
  add_bf16_k<<<cdiv(n, 256), 256, 0, st>>>(a, b, out, n);
}
__global__ void f2b_k(const float* a, bf16* o, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    o[i] = f2b(a[i]);
}
void f32_to_bf16(const float* a, bf16* out, long long n, cudaStream_t st) { f2b_k<<<nblk(n), 256, 0, st>>>(a, out, n); }
__global__ void b2f_k(const bf16* a, float* o, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    o[i] = b2f(a[i]);
}
void bf16_to_f32(const bf16* a, float* out, long long n, cudaStream_t st) { b2f_k<<<nblk(n), 256, 0, st>>>(a, out, n); }
__global__ void rnd_k(float* a, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    a[i] = bfr(a[i]);
}
void round_bf16_inplace(float* a, long long n, cudaStream_t st) { rnd_k<<<nblk(n), 256, 0, st>>>(a, n); }
__global__ void addbf_k(const bf16* a, float* acc, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    acc[i] += b2f(a[i]);
}
void add_bf16_to_f32(const bf16* a, float* acc, long long n, cudaStream_t st) {
  addbf_k<<<nblk(n), 256, 0, st>>>(a, acc, n);
}
__global__ void split3_k(const float* a, bf16* hi, bf16* mid, bf16* lo, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const float d = a[i];
    const bf16 h = f2b(d);
    const float r1 = d - b2f(h);
    const bf16 m = f2b(r1);
    hi[i] = h;
    mid[i] = m;
    lo[i] = f2b(r1 - b2f(m));
  }
}
void split3(const float* a, bf16* hi, bf16* mid, bf16* lo, long long n, cudaStream_t st) {
  split3_k<<<nblk(n), 256, 0, st>>>(a, hi, mid, lo, n);
}
__global__ void copy_cols_k(const bf16* s, long long lds, bf16* d, long long ldd, int ncols) {
  const int r = blockIdx.y;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ncols; j += gridDim.x * blockDim.x) d[r * ldd + j] = s[r * lds + j];
}
void copy_cols(const bf16* src, long long lds, bf16* dst, long long ldd, int rows, int ncols, cudaStream_t st) {
  copy_cols_k<<<dim3(cdiv(ncols, 256), rows), 256, 0, st>>>(src, lds, dst, ldd, ncols);
}
__global__ void transpose_k(const bf16* s, bf16* d, int rows, int cols, int lds, int ldd) {
  __shared__ bf16 tile[32][33];
  const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int r = by + j, c = bx + threadIdx.x;
    if (r < rows && c < cols) tile[j][threadIdx.x] = s[(size_t)r * lds + c];
  }
  __syncthreads();
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int c = bx + j, r = by + threadIdx.x;
    if (r < rows && c < cols) d[(size_t)c * ldd + r] = tile[threadIdx.x][j];
  }
}
void transpose_bf16(const bf16* src, bf16* dst, int rows, int cols, int ld_src, int ld_dst, cudaStream_t st) {
  transpose_k<<<dim3(cdiv(cols, 32), cdiv(rows, 32)), dim3(32, 8), 0, st>>>(src, dst, rows, cols, ld_src, ld_dst);
}

// ---- flow matching ------------------------------------------------------------------------------------------------------------
__global__ void flow_inputs_k(const float* noise, const float* a, float t, float* xt, float* u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  xt[i] = t * noise[i] + (1.0f - t) * a[i];
  u[i] = noise[i] - a[i];
}
void flow_inputs(const float* noise, const float* act, float t, float* xt, float* u, int n, cudaStream_t st) {
  flow_inputs_k<<<cdiv(n, 256), 256, 0, st>>>(noise, act, t, xt, u, n);
}
__global__ void flow_loss_k(const float* v, const float* u, float* lr, float* dv, int d, float scale) {
  const int r = blockIdx.x, j = threadIdx.x;  // d <= 32 threads
  const float e = j < d ? v[r * d + j] - u[r * d + j] : 0.f;
  const float s = wsum(e * e);
  if (j == 0) lr[r] = s / (float)d;
  if (j < d) dv[r * d + j] = 2.0f * e * (scale / (float)d);
}
void flow_loss(const float* v, const float* u, float* loss_row, float* dv, int rows, int d, float scale,
               cudaStream_t st) {
  flow_loss_k<<<rows, 32, 0, st>>>(v, u, loss_row, dv, d, scale);
}

// ---- optimizer ------------------------------------------------------------------------------------------------------------------
__global__ void sumsq_k(const float* g, long long n, double* acc) {
  __shared__ float sh[8];
  float s = 0.f;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    s += g[i] * g[i];
  s = bsum<256>(s, sh);
  if (threadIdx.x == 0) atomicAdd(acc, (double)s);
}
void sumsq_f32(const float* g, long long n, double* acc, cudaStream_t st) {
  sumsq_k<<<std::min(nblk(n), 1024), 256, 0, st>>>(g, n, acc);
}
// optax: g' = clip ; mu = (1-b1) g' + b1 mu ; nu = (1-b2) g'^2 + b2 nu ; u = (mu/bc1) / (sqrt(nu/bc2) + eps)
//        u += wd * p ; p = p + (-lr) * u
__global__ void adamw_k(float* p, const float* g, float* m, float* v, long long n, float gnorm, float maxn, float b1,
                        float ob1, float b2, float ob2, float eps, float wd, float lr, float bc1, float bc2) {
  const bool clip = !(gnorm < maxn);  // optax clip_by_global_norm: select(norm < max, g, (g / norm) * max)
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    float gi = g[i];
    if (clip) gi = __fmul_rn(__fdiv_rn(gi, gnorm), maxn);
    // separate roundings (no FMA contraction), as XLA evaluates these elementwise expressions
    const float mu = __fadd_rn(__fmul_rn(ob1, gi), __fmul_rn(b1, m[i]));
    const float nu = __fadd_rn(__fmul_rn(ob2, __fmul_rn(gi, gi)), __fmul_rn(b2, v[i]));
    m[i] = mu;
    v[i] = nu;
    float u = __fdiv_rn(__fdiv_rn(mu, bc1), __fsqrt_rn(__fdiv_rn(nu, bc2)) + eps);
    u = __fadd_rn(u, __fmul_rn(wd, p[i]));
    p[i] = __fadd_rn(p[i], __fmul_rn(-lr, u));
  }
}
void adamw_step(float* p, const float* g, float* m, float* v, long long n, float gnorm, float maxn, float b1,
                float ob1, float b2, float ob2, float eps, float wd, float lr, float bc1, float bc2, cudaStream_t st) {
  adamw_k<<<nblk(n), 256, 0, st>>>(p, g, m, v, n, gnorm, maxn, b1, ob1, b2, ob2, eps, wd, lr, bc1, bc2);
}
__global__ void ema_k(float* e, const float* p, long long n, float d, float od) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    e[i] = __fadd_rn(__fmul_rn(d, e[i]), __fmul_rn(od, p[i]));
}
void ema_step(float* e, const float* p, long long n, float decay, float one_minus, cudaStream_t st) {
  ema_k<<<nblk(n), 256, 0, st>>>(e, p, n, decay, one_minus);
}

}  // namespace pi05t
