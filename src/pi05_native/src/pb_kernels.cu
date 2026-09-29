// PiBehavior (2025 BEHAVIOR 1st place, IliaLarchenko/behavior-1k-solution) specific kernels.
// Source lines refer to src/b1k/models/pi_behavior.py of that repository (identical in the alstar8 2026 fork except
// for the stage table).
#include "pb_kernels.cuh"

namespace pi05 {
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
}  // namespace

// ---- task / stage tokens (pi_behavior.py:222-251, 452-517, 562-590) ----------------------------------------------------
// base = task_emb[task] (bf16) ; sincos = posemb_sincos(stage / max(n-1, 1), 1024, min_period=1e-3, max_period=1.0) (f32)
// all = concat(base, sincos, stage_emb[offset + stage]) -> f32 [4096]
// x rows: 0 base, 4 pure stage = concat(sincos, stage_emb); rows 1..3 come from the gated MLPs (pb_task_rows)
__global__ void pb_task_prep_kernel(PbTaskArgs a, float* all, bf16* x_rows) {
  pdl_entry();
  const int task = a.ints[0], stage = a.ints[1];
  const int n = a.num_stages[task];
  const float norm = __fdiv_rn((float)stage, fmaxf((float)n - 1.0f, 1.0f));
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
    const bf16 b = a.task_emb[(size_t)task * 2048 + i];
    all[i] = b2f(b);
    x_rows[i] = b;
  }
  for (int j = threadIdx.x; j < 512; j += blockDim.x) {
    const float frac = __fdiv_rn((float)j, 511.0f);  // jnp.linspace(0, 1, 512)
    const float period = __fmul_rn(1e-3f, powf(1000.0f, frac));
    const float f = __fmul_rn(__fmul_rn(__fdiv_rn(1.0f, period), 2.0f), 3.14159274101257324f);
    const float ang = __fmul_rn(norm, f);
    const float s = sinf(ang), c = cosf(ang);
    all[2048 + j] = s;
    all[2048 + 512 + j] = c;
    x_rows[4 * 2048 + j] = f2b(s);
    x_rows[4 * 2048 + 512 + j] = f2b(c);
  }
  const bf16* se = a.stage_emb + (size_t)(a.offsets[task] + stage) * 1024;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    all[3072 + i] = b2f(se[i]);
    x_rows[4 * 2048 + 1024 + i] = se[i];
  }
}

void launch_pb_task_prep(PbTaskArgs a, float* all, bf16* x_rows, cudaStream_t st) {
  launch_k(pb_task_prep_kernel, 1, 512, 0, st, a, all, x_rows);
}

// sf = concat(sincos * gate_sincos, stage_emb * gate_task_stage)   (f32)
__global__ void pb_stage_feat_kernel(const float* all, const float* g_sc, const float* g_ts, float* sf) {
  pdl_entry();
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    sf[i] = all[2048 + i] * g_sc[i];
    sf[1024 + i] = all[3072 + i] * g_ts[i];
  }
}

void launch_pb_stage_feat(const float* all, const float* g_sc, const float* g_ts, float* sf, cudaStream_t st) {
  launch_k(pb_stage_feat_kernel, 1, 512, 0, st, all, g_sc, g_ts, sf);
}

// rows 1 task_gated = base * gate_task, 2 balanced fusion, 3 stage-dominant; cast to bf16 at the llm input (gemma.py:400)
__global__ void pb_task_rows_kernel(const float* all, const float* g_t, const float* bal, const float* sd,
                                    bf16* x_rows) {
  pdl_entry();
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
    x_rows[1 * 2048 + i] = f2b(all[i] * g_t[i]);
    x_rows[2 * 2048 + i] = f2b(bal[i]);
    x_rows[3 * 2048 + i] = f2b(sd[i]);
  }
}

void launch_pb_task_rows(const float* all, const float* g_t, const float* bal, const float* sd, bf16* x_rows,
                         cudaStream_t st) {
  launch_k(pb_task_rows_kernel, 1, 512, 0, st, all, g_t, bal, sd, x_rows);
}

// ---- KV cache layer mixing (pi_behavior.py:35-90) ----------------------------------------------------------------------
// Parameters are restored as bf16 (restore_params(dtype=bf16)), so einsum(bf16 coeffs, bf16 cache) -> bf16 (f32
// accumulate), + bf16 bias -> bf16.   k2[d] = bf16(bf16(sum_s c[d, s] k[s]) + b[d])
__global__ void kv_transform_kernel(const bf16* __restrict__ kc, const bf16* __restrict__ vt, bf16* __restrict__ kc2,
                                    bf16* __restrict__ vt2, const bf16* kcoef, const bf16* vcoef, const bf16* kbias,
                                    const bf16* vbias, int depth, int sc, const int* d_T) {
  pdl_entry();
  __shared__ float ck[18 * 18], cv[18 * 18];
  for (int i = threadIdx.x; i < depth * depth; i += blockDim.x) {
    ck[i] = b2f(kcoef[i]);
    cv[i] = b2f(vcoef[i]);
  }
  __syncthreads();
  const int T = *d_T;
  const size_t per = (size_t)sc * 256;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < T * 256; idx += gridDim.x * blockDim.x) {
    const int t = idx / 256, j = idx % 256;
    float ks[18], vs[18];
    for (int s = 0; s < depth; ++s) {
      ks[s] = b2f(kc[s * per + (size_t)t * 256 + j]);
      vs[s] = b2f(vt[s * per + (size_t)j * sc + t]);
    }
    for (int d = 0; d < depth; ++d) {
      float a = 0.f, b = 0.f;
      for (int s = 0; s < depth; ++s) {
        a = fmaf(ck[d * depth + s], ks[s], a);
        b = fmaf(cv[d * depth + s], vs[s], b);
      }
      kc2[d * per + (size_t)t * 256 + j] = f2b(bfr(a) + b2f(kbias[d * 256 + j]));
      vt2[d * per + (size_t)j * sc + t] = f2b(bfr(b) + b2f(vbias[d * 256 + j]));
    }
  }
}

void launch_kv_transform(const bf16* kc, const bf16* vt, bf16* kc2, bf16* vt2, const bf16* kcoef, const bf16* vcoef,
                         const bf16* kbias, const bf16* vbias, int depth, int sc, const int* d_T, cudaStream_t st) {
  launch_k(kv_transform_kernel, 280, 256, 0, st, kc, vt, kc2, vt2, kcoef, vcoef, kbias, vbias, depth, sc, d_T);
}

// y = bf16(bf16(x . W^T) + b), x bf16 (nnx.Linear on a bf16 activation, e.g. stage_pred_from_vlm)
__global__ void linear_bf16in_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                     bf16* __restrict__ y, int in_dim, int out_dim) {
  pdl_entry();
  const int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (o >= out_dim) return;
  float acc = 0.f;
  for (int k = lane; k < in_dim; k += 32) acc = fmaf(b2f(x[k]), b2f(w[(size_t)o * in_dim + k]), acc);
  acc = wsum(acc);
  if (lane == 0) y[o] = f2b(bfr(acc) + b2f(b[o]));
}

void launch_linear_bf16in(const bf16* x, const bf16* w, const bf16* b, bf16* y, int in_dim, int out_dim,
                          cudaStream_t st) {
  launch_k(linear_bf16in_kernel, cdiv(out_dim, 8), 256, 0, st, x, w, b, y, in_dim, out_dim);
}

// ---- correlation-aware soft inpainting after one Euler step (pi_behavior.py:1061-1107) ---------------------------------
//   x[O] = (1 - t_new) x0_O + t_new z_O ; delta_O = x_desired - x[O] ; delta_U = C . delta_O ;
//   if max |delta_U| <= 1: x[U] += delta_U.   O = first nO flat entries (kept steps x 32 dims), U = the rest.
// Enabled per call by ints[2] (first chunk of an episode has nothing to inpaint).
__global__ void __launch_bounds__(1024) inpaint_kernel(float* x, const float* x0O, const float* zO, const float* C,
                                                       int nO, int nU, float t_new, const int* ints) {
  pdl_entry();
  if (!ints[2]) return;
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
  mx = wmax(mx);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
  __syncthreads();
  float gmax = 0.f;
  for (int i = 0; i < (int)(blockDim.x / 32); ++i) gmax = fmaxf(gmax, sh[i]);
  for (int i = threadIdx.x; i < nO; i += blockDim.x) x[i] = (1.0f - t_new) * x0O[i] + t_new * zO[i];
  if (gmax <= 1.0f && u < nU) x[nO + u] += du;
}

void launch_inpaint(float* x, const float* x0O, const float* zO, const float* C, int nO, int nU, float t_new,
                    const int* ints, cudaStream_t st) {
  launch_k(inpaint_kernel, 1, 1024, 0, st, x, x0O, zO, C, nO, nU, t_new, ints);
}

}  // namespace pi05
