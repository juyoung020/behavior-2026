// Training kernels (forward pieces that must keep what backward needs, and every backward piece).
// Each kernel names the JAX expression whose forward/VJP it reproduces and where JAX rounds to bf16.
#pragma once
#include "../../pi05_native/src/common.cuh"

namespace pi05t {
using namespace pi05;

// ---- f32 dense layers (nnx.Linear, f32 params: action_in_proj, time_mlp_in/out, action_out_proj) -------------------
// y[r][o] = sum_i x[r][i] W[i][o] + b[o]  (W openpi layout [in][out]); act 1 = swish. x may be bf16 (xb) or f32 (xf).
void lin_f32_fwd(const float* xf, const bf16* xb, const float* W, const float* b, float* z, float* y, int rows,
                 int in, int out, int act, cudaStream_t st);  // z: pre-activation (may be null)
// backward: dz = dy * act'(z) (z = pre-activation, saved), dx = dz W^T (f32), dW += x^T dz, db += sum_r dz
void lin_f32_bwd(const float* xf, const bf16* xb, const float* W, const float* z, const float* dy, int act, float* dx,
                 float* dW, float* db, int rows, int in, int out, float* dz_tmp, cudaStream_t st);

// ---- flax Dense(dtype=bf16) on the time condition (adaRMS modulation, gemma.py:128) -------------------------------
// mod[o] = bf16(bf16(sum_i bf16(cond[i]) Wb[i][o]) + bf16(b[o]))      Wb = bf16 copy of the f32 kernel [in][out]
void mod_fwd(const float* cond, const bf16* Wb, const bf16* bb, bf16* mod, int in, int out, cudaStream_t st);
// dW[i][o] += bf16(cond[i]) * dmod[o] (exact product, rounded once per batch in finalize), db[o] += dmod[o],
// dcond[i] += f32(bf16(sum_o dmod[o] Wb[i][o]))
void mod_bwd(const float* cond, const bf16* Wb, const bf16* dmod, float* dW, float* db, float* dcond, int in, int out,
             cudaStream_t st);

// ---- adaptive RMSNorm (gemma.py:113-131) --------------------------------------------------------------------------
// y = bf16(x * r * bf16(1 + scale) + shift), r = rsqrt(mean(x^2) + 1e-6) saved per row. mod = [scale|shift|gate].
void adarms_fwd(const bf16* x, const bf16* mod, bf16* y, float* r, int rows, int dim, cudaStream_t st);
// dx = bf16(r*dn - x r^3 mean(dn x)) with dn = dy bf16(1+scale); dmod[scale] += sum_t dy*x*r, dmod[shift] += sum_t dy
// (f32 partial sums over rows, written as bf16 by adarms_bwd_mod_finish)
void adarms_bwd(const bf16* x, const bf16* mod, const float* r, const bf16* dy, bf16* dx, float* dss, int rows,
                int dim, cudaStream_t st);

// ---- RoPE + scaling (gemma.py:189-206, 424-440) ---------------------------------------------------------------------
// in place on [rows][heads][256] bf16: x = bf16(rope(x)) then (q only) x = bf16(x * 0.0625). pos = pos0 + row
void rope_fwd(bf16* x, const float2* table, int rows, int heads, int row_stride, int pos0, bool scale_q, cudaStream_t st);
void rope_bwd(bf16* dx, const float2* table, int rows, int heads, int row_stride, int pos0, bool scale_q,
              cudaStream_t st);
// dmod = [bf16(dss[0:dim]) | bf16(dss[dim:2dim]) | bf16(dgate) or 0]
void pack_dmod(const float* dss, const float* dgate, bf16* dmod, int dim, cudaStream_t st);

// ---- attention softmax (gemma.py:217-228), f32 logits, keys [0, cols) valid ----------------------------------------
// p32 = softmax(logits) (f32, kept for backward), pb = bf16(p32); columns [cols, ld) zeroed
void softmax_fwd(const float* logits, float* p32, bf16* pb, int rows, int cols, int ld, cudaStream_t st);
// dlogits = p * (dp - sum(p * dp)) (f32) with dp = f32(dpb); split into three bf16 terms hi+mid+lo (exact f32)
void softmax_bwd_split(const float* p32, const bf16* dpb, bf16* d3, int rows, int cols, int ld, cudaStream_t st);

// ---- gated residual (gemma.py:462-468): out = bf16(x + bf16(y * gate)) ----------------------------------------------
void gated_res_fwd(const bf16* x, const bf16* y, const bf16* gate, bf16* out, int rows, int dim, cudaStream_t st);
// dy = bf16(dout * gate); dgate partial f32 sums of bf16(dout * y) over rows (into dgate32[dim])
void gated_res_bwd(const bf16* dout, const bf16* y, const bf16* gate, bf16* dy, float* dgate32, int rows, int dim,
                   cudaStream_t st);

// ---- Gemma MLP activation (gemma.py:344-352): a = bf16(gelu(g) * u) -------------------------------------------------
void gelu_mul_fwd(const bf16* g, const bf16* u, bf16* a, int n, cudaStream_t st);
void gelu_mul_bwd(const bf16* g, const bf16* u, const bf16* da, bf16* dg, bf16* du, int n, cudaStream_t st);

// ---- elementwise helpers ----------------------------------------------------------------------------------------------
void add_bf16(const bf16* a, const bf16* b, bf16* out, int n, cudaStream_t st);        // out = bf16(a + b)
void f32_to_bf16(const float* a, bf16* out, long long n, cudaStream_t st);
void bf16_to_f32(const bf16* a, float* out, long long n, cudaStream_t st);
void round_bf16_inplace(float* a, long long n, cudaStream_t st);                        // a = f32(bf16(a))
void add_bf16_to_f32(const bf16* a, float* acc, long long n, cudaStream_t st);         // acc += f32(a)
void split3(const float* a, bf16* hi, bf16* mid, bf16* lo, long long n, cudaStream_t st);
// columns [c0, c0+n) of rows: dst[r][j] = src[r][c0 + j]  (bf16)
void copy_cols(const bf16* src, long long lds, bf16* dst, long long ldd, int rows, int ncols, cudaStream_t st);
void transpose_bf16(const bf16* src, bf16* dst, int rows, int cols, int ld_src, int ld_dst, cudaStream_t st);

// ---- flow matching (pi0.py:195-214) -----------------------------------------------------------------------------------
// x_t = t*noise + (1-t)*a ; u = noise - a   (f32)
void flow_inputs(const float* noise, const float* act, float t, float* xt, float* u, int n, cudaStream_t st);
// loss_row[r] = mean_d (v - u)^2 ; dv = 2 (v - u) / d * scale   (scale = 1 / (batch * rows))
void flow_loss(const float* v, const float* u, float* loss_row, float* dv, int rows, int d, float scale,
               cudaStream_t st);

// ---- optimizer (optax: clip_by_global_norm -> adamw; EMA) -----------------------------------------------------------
void sumsq_f32(const float* g, long long n, double* acc, cudaStream_t st);  // acc += sum g^2 (f64 atomics per block)
// one AdamW step (optax.scale_by_adam + add_decayed_weights + scale_by_learning_rate) on f32 params with the global
// clip factor applied to g first; bias corrections bc1 = 1 - b1^t, bc2 = 1 - b2^t (computed by the caller in f32)
// gnorm = global grad norm, maxn = clip threshold; ob1 = f32(1 - b1) and ob2, one_minus computed in double like Python
void adamw_step(float* p, const float* g, float* m, float* v, long long n, float gnorm, float maxn, float b1,
                float ob1, float b2, float ob2, float eps, float wd, float lr, float bc1, float bc2, cudaStream_t st);
void ema_step(float* e, const float* p, long long n, float decay, float one_minus, cudaStream_t st);

}  // namespace pi05t
