// Hand-written bf16 tensor-core GEMM (mma.sync.m16n8k16, fp32 accumulate) with fused epilogues.
//
//   C[M, N] = A[M, K] . B[N, K]^T        (both operands K-contiguous, "NT")
//
// cp.async multi-stage pipeline, XOR-swizzled shared memory (conflict-free ldmatrix), optional
// split-K (partials to an fp32 workspace, reduced in fixed order by splitk_reduce), optional
// 2-level batching (attention heads) and optional device-side M/N/K so one CUDA graph serves every
// prompt length. The epilogue functor sees fp32 accumulator pairs (row, col..col+1) and decides how
// JAX would round them (see epilogues below).
#pragma once
#include "common.cuh"

namespace pi05 {

struct GemmParams {
  const bf16* A = nullptr;
  const bf16* B = nullptr;
  int lda = 0, ldb = 0;
  int M = 0, N = 0, K = 0;                         // upper bounds (grid is sized from these)
  const int* dM = nullptr;                          // optional device-side actual sizes
  const int* dN = nullptr;
  const int* dK = nullptr;
  int splits = 1;                                   // split-K factor (grid.z = batch * splits)
  int nb2 = 1;                                      // batch b -> (b / nb2, b % nb2)
  long long sA1 = 0, sA2 = 0, sB1 = 0, sB2 = 0;     // element strides per batch coordinate
  long long sC1 = 0, sC2 = 0;                       // passed to the epilogue as an offset
  float* ws = nullptr;                              // split-K workspace [splits][M][N]
  int* counters = nullptr;                          // split-K tile tickets (zeroed); null -> separate reduce kernel
  bool b_static = false;                            // B is a weight (not written by earlier kernels): prefetch before PDL wait
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int GEMM_BK = 32;  // K granularity every config supports (tails are zero-filled per 8 elements)

// XOR swizzle of 16-byte chunks inside a BK-wide bf16 row: conflict-free cp.async stores and ldmatrix loads.
// BK=32: 64-byte rows, 4 chunks, chunk ^= (row >> 1) & 3.  BK=64: 128-byte rows, 8 chunks, chunk ^= row & 7.
template <int BK>
__device__ __forceinline__ int swz(int r, int ch) {
  if constexpr (BK == 32) return r * BK + ((ch ^ ((r >> 1) & 3)) << 3);
  else return r * BK + ((ch ^ (r & 7)) << 3);
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int BK = 32>
struct GemmCfg {
  static constexpr int kThreads = WARPS_M * WARPS_N * 32;
  static constexpr int kSmem = STAGES * (BM + BN) * BK * 2;
};

// Grid: x = M tiles (fastest), y = N tiles, z = batch * splits. M-fastest order makes the CTAs that share one
// weight tile run back to back, so each weight byte comes from DRAM once (weights >> L2, activations fit in L2).
template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int BK, class Epi>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32)
    gemm_nt_kernel(GemmParams p, Epi epi) {
  pdl_trigger();  // pdl_wait() comes after the static weight prefetch below
  constexpr int CH = BK / 8;  // 16-byte chunks per row
  constexpr int NT = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MT = WM / 16, NT8 = WN / 8;
  static_assert(WN % 16 == 0 && WM % 16 == 0, "warp tile");
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* sA = reinterpret_cast<bf16*>(smem_raw);
  bf16* sB = sA + STAGES * BM * BK;

  const int M = p.dM ? *p.dM : p.M;
  const int N = p.dN ? *p.dN : p.N;
  const int K = p.dK ? *p.dK : p.K;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  if (m0 >= M || n0 >= N) return;
  const int split = blockIdx.z % p.splits;
  const int b = blockIdx.z / p.splits;
  const int b1 = b / p.nb2, b2 = b % p.nb2;
  const bf16* A = p.A + b1 * p.sA1 + b2 * p.sA2;
  const bf16* B = p.B + b1 * p.sB1 + b2 * p.sB2;
  const long long coff = b1 * p.sC1 + b2 * p.sC2;

  const int ktiles = (K + BK - 1) / BK;
  const int kper = (ktiles + p.splits - 1) / p.splits;
  const int kt0 = split * kper;
  const int nk = max(0, min(ktiles, kt0 + kper) - kt0);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;

  auto load_a = [&](int stage, int kt) {
    const int k0 = kt * BK;
    bf16* a = sA + stage * BM * BK;
#pragma unroll
    for (int c = tid; c < BM * CH; c += NT) {
      int r = c / CH, ch = c % CH;
      int gr = m0 + r, gk = k0 + ch * 8;
      bool ok = gr < M && gk < K;
      const bf16* src = ok ? A + (long long)gr * p.lda + gk : A;
      cp_async16(smem_u32(a + swz<BK>(r, ch)), src, ok ? 16 : 0);
    }
  };
  auto load_b = [&](int stage, int kt) {
    const int k0 = kt * BK;
    bf16* bb = sB + stage * BN * BK;
#pragma unroll
    for (int c = tid; c < BN * CH; c += NT) {
      int r = c / CH, ch = c % CH;
      int gr = n0 + r, gk = k0 + ch * 8;
      bool ok = gr < N && gk < K;
      const bf16* src = ok ? B + (long long)gr * p.ldb + gk : B;
      cp_async16(smem_u32(bb + swz<BK>(r, ch)), src, ok ? 16 : 0);
    }
  };
  auto load_stage = [&](int stage, int kt) {
    load_a(stage, kt);
    load_b(stage, kt);
  };

  float acc[MT][NT8][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  // Prologue: STAGES cp.async groups either way (group 0 = prefetched weights or empty), so the main loop's
  // wait_group<STAGES-2> finds stage i complete at iteration i.
  if (p.b_static) {
    // weights do not depend on the previous kernel: start streaming them before waiting for it (PDL overlap)
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < nk) load_b(s, kt0 + s);
    cp_async_commit();
    pdl_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < nk) load_a(s, kt0 + s);
      cp_async_commit();
    }
  } else {
    pdl_wait();
    cp_async_commit();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < nk) load_stage(s, kt0 + s);
      cp_async_commit();
    }
  }

  for (int i = 0; i < nk; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < nk) load_stage(nxt % STAGES, kt0 + nxt);
    cp_async_commit();
    const bf16* a = sA + (i % STAGES) * BM * BK;
    const bf16* bb = sB + (i % STAGES) * BN * BK;
#pragma unroll
    for (int kk = 0; kk < BK / 8; kk += 2) {  // two 8-wide chunks = k16
      uint32_t af[MT][4];
      uint32_t bfg[NT8][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        int r = wm * WM + mt * 16 + (lane & 15);
        ldmatrix_x4(af[mt], smem_u32(a + swz<BK>(r, kk + (lane >> 4))));
      }
#pragma unroll
      for (int np = 0; np < NT8 / 2; ++np) {
        int r = wn * WN + np * 16 + (lane & 7) + ((lane >> 4) << 3);
        uint32_t t[4];
        ldmatrix_x4(t, smem_u32(bb + swz<BK>(r, kk + ((lane >> 3) & 1))));
        bfg[2 * np][0] = t[0];
        bfg[2 * np][1] = t[1];
        bfg[2 * np + 1][0] = t[2];
        bfg[2 * np + 1][1] = t[3];
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma_bf16(acc[mt][nt], af[mt], bfg[nt][0], bfg[nt][1]);
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int row = m0 + wm * WM + mt * 16 + g;
      const int col = n0 + wn * WN + nt * 8 + 2 * t4;
      if (col >= N) continue;
      if (p.splits > 1) {
        float* w = p.ws + ((long long)b * p.splits + split) * p.M * p.N;  // b = 0 unless batched split-K
        if (row < M) *reinterpret_cast<float2*>(w + (long long)row * p.N + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (row + 8 < M)
          *reinterpret_cast<float2*>(w + (long long)(row + 8) * p.N + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      } else {
        if (row < M) epi(row, col, acc[mt][nt][0], acc[mt][nt][1], coff);
        if (row + 8 < M) epi(row + 8, col, acc[mt][nt][2], acc[mt][nt][3], coff);
      }
    }
  }
  if (p.splits > 1 && p.counters) {
    // In-kernel split-K reduction without a second launch: the last CTA of this output tile (ticket counter)
    // sums every split's partial in the fixed order 0..splits-1 (deterministic regardless of which CTA finishes
    // last), applies the epilogue and re-arms the counter for the next launch / graph replay.
    __shared__ int is_last;
    __threadfence();
    __syncthreads();
    int* ctr = p.counters + ((long long)b * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (tid == 0) is_last = atomicAdd(ctr, 1) == p.splits - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    const int rows = min(BM, M - m0), cols = min(BN, N - n0);
    for (int idx = tid; idx < rows * (cols / 2); idx += NT) {
      const int r = m0 + idx / (cols / 2), c = n0 + 2 * (idx % (cols / 2));
      const float* wb = p.ws + (long long)b * p.splits * p.M * p.N;
      float2 s = __ldcg(reinterpret_cast<const float2*>(wb + (long long)r * p.N + c));
      for (int k = 1; k < p.splits; ++k) {
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wb + (long long)k * p.M * p.N + (long long)r * p.N + c));
        s.x += v.x;
        s.y += v.y;
      }
      epi(r, c, s.x, s.y, coff);
    }
    if (tid == 0) *ctr = 0;
  }
}

// Sums split-K partials in a fixed order (deterministic) and applies the epilogue.
template <class Epi>
__global__ void splitk_reduce_kernel(const float* ws, int splits, int M, int N, const int* dM, Epi epi) {
  pdl_entry();
  const int Mr = dM ? *dM : M;
  const int half = N / 2;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < Mr * half; idx += gridDim.x * blockDim.x) {
    const int row = idx / half, col = 2 * (idx % half);
    float2 s = *reinterpret_cast<const float2*>(ws + (long long)row * N + col);
    for (int k = 1; k < splits; ++k) {
      float2 v = *reinterpret_cast<const float2*>(ws + (long long)k * M * N + (long long)row * N + col);
      s.x += v.x;
      s.y += v.y;
    }
    epi(row, col, s.x, s.y, 0);
  }
}

// ---- epilogues ------------------------------------------------------------------------------------
// JAX semantics being reproduced are noted per functor; `bfr` = round to bf16.

struct EpiBf16 {  // dot(bf16, bf16) -> bf16
  bf16* out;
  int ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    *reinterpret_cast<bf162*>(out + off + (long long)r * ldo + c) = __floats2bfloat162_rn(v0, v1);
  }
};

struct EpiF32 {  // einsum(..., preferred_element_type=f32)
  float* out;
  int ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    *reinterpret_cast<float2*>(out + off + (long long)r * ldo + c) = make_float2(v0, v1);
  }
};

struct EpiBias {  // flax Dense(dtype=bf16): y = bf16(dot) ; y += bias (bf16 add)
  bf16* out;
  int ldo;
  const bf16* bias;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float a = bfr(bfr(v0) + b2f(bias[c])), b = bfr(bfr(v1) + b2f(bias[c + 1]));
    *reinterpret_cast<bf162*>(out + off + (long long)r * ldo + c) = __floats2bfloat162_rn(a, b);
  }
};

struct EpiBiasGelu {  // SigLIP MLP fc1: nn.gelu(Dense(x))
  bf16* out;
  int ldo;
  const bf16* bias;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float a = gelu_bf16(bfr(bfr(v0) + b2f(bias[c]))), b = gelu_bf16(bfr(bfr(v1) + b2f(bias[c + 1])));
    *reinterpret_cast<bf162*>(out + off + (long long)r * ldo + c) = __floats2bfloat162_rn(a, b);
  }
};

struct EpiBiasResid {  // SigLIP: x = x + Dense(y)   (out may alias resid)
  bf16* out;
  int ldo;
  const bf16* bias;
  const bf16* resid;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    long long i = off + (long long)r * ldo + c;
    bf162 x = *reinterpret_cast<const bf162*>(resid + i);
    float a = bfr(__low2float(x) + bfr(bfr(v0) + b2f(bias[c])));
    float b = bfr(__high2float(x) + bfr(bfr(v1) + b2f(bias[c + 1])));
    *reinterpret_cast<bf162*>(out + i) = __floats2bfloat162_rn(a, b);
  }
};

struct EpiResid {  // Gemma: x = x + bf16(dot)   (gate None, gemma.py:453-459)
  bf16* out;
  int ldo;
  const bf16* resid;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    long long i = off + (long long)r * ldo + c;
    bf162 x = *reinterpret_cast<const bf162*>(resid + i);
    *reinterpret_cast<bf162*>(out + i) =
        __floats2bfloat162_rn(__low2float(x) + bfr(v0), __high2float(x) + bfr(v1));
  }
};

struct EpiGatedResid {  // action expert: x = x + bf16(dot) * gate   (bf16 ops)
  bf16* out;
  int ldo;
  const bf16* resid;
  const bf16* gate;  // [ldo] per column
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    long long i = off + (long long)r * ldo + c;
    bf162 x = *reinterpret_cast<const bf162*>(resid + i);
    float a = __low2float(x) + bfr(bfr(v0) * b2f(gate[c]));
    float b = __high2float(x) + bfr(bfr(v1) * b2f(gate[c + 1]));
    *reinterpret_cast<bf162*>(out + i) = __floats2bfloat162_rn(a, b);
  }
};

// Gemma attention input projection with RoPE, q scaling and KV-cache writes fused (gemma.py:185-206,424-440).
// The q and k weight rows are permuted at load time so that output columns (2j, 2j+1) of every 256-wide head hold
// dims (j, j+128) = the RoPE pair; v rows keep their order.  Per element exactly what JAX does:
//   x = bf16(dot); rot = bf16(f32 rotation of (x1, x2)); q = bf16(rot * 0.0625); k -> cache; v -> transposed cache.
struct EpiQKVRope {
  bf16* q;            // [rows][heads][256]
  bf16* kc;           // [pos][256]
  bf16* vt;           // [256][vt_ld]
  const float2* rope; // [pos][128] (cos, sin)
  const int* d_pos0;  // position of row 0 (prefix: null -> 0, suffix: number of prefix tokens)
  int heads, vt_ld;
  __device__ void operator()(int r, int c, float v0, float v1, long long) const {
    const int blk = c >> 8, cc = c & 255;
    const int pos = (d_pos0 ? *d_pos0 : 0) + r;
    if (blk <= heads) {
      const int j = cc >> 1;
      const float2 cs = rope[pos * 128 + j];
      const float x1 = bfr(v0), x2 = bfr(v1);
      const float o1 = bfr(x1 * cs.x - x2 * cs.y), o2 = bfr(x2 * cs.x + x1 * cs.y);
      if (blk < heads) {
        bf16* qo = q + ((long long)r * heads + blk) * 256;
        qo[j] = f2b(o1 * 0.0625f);
        qo[j + 128] = f2b(o2 * 0.0625f);
      } else {
        kc[(long long)pos * 256 + j] = f2b(o1);
        kc[(long long)pos * 256 + j + 128] = f2b(o2);
      }
    } else {
      vt[(long long)cc * vt_ld + pos] = f2b(v0);
      vt[(long long)(cc + 1) * vt_ld + pos] = f2b(v1);
    }
  }
};

// SigLIP qkv DenseGeneral (+bias, bf16) fused with the attention prep (flax attention.py:129-132):
// q -> bf16(q / bf16(sqrt(72)) = 8.5) into qs, k stays in place, v -> vt[img][head][dim][token].
struct EpiSiglipQKV {
  bf16* qkv;  // [rows][3456]  (k columns written here)
  bf16* qs;   // [rows][1152]
  bf16* vt;   // [img][16][72][256]
  const bf16* bias;
  __device__ void operator()(int r, int c, float v0, float v1, long long) const {
    const float a = bfr(bfr(v0) + b2f(bias[c])), b = bfr(bfr(v1) + b2f(bias[c + 1]));
    if (c < 1152) {
      *reinterpret_cast<bf162*>(qs + (long long)r * 1152 + c) =
          __floats2bfloat162_rn(__fdiv_rn(a, 8.5f), __fdiv_rn(b, 8.5f));
    } else if (c < 2304) {
      *reinterpret_cast<bf162*>(qkv + (long long)r * 3456 + c) = __floats2bfloat162_rn(a, b);
    } else {
      const int cc = c - 2304, im = r >> 8, tok = r & 255, h = cc / 72, d = cc % 72;  // d even: d, d+1 same head
      bf16* base = vt + (((long long)im * 16 + h) * 72 + d) * 256 + tok;
      base[0] = f2b(a);
      base[256] = f2b(b);
    }
  }
};

struct EpiGeluGate {  // Gemma MLP: gelu(x.Wg) * (x.Wu), columns interleaved (gate, up) -> out col c/2
  bf16* out;
  int ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float gte = gelu_bf16(bfr(v0));
    out[off + (long long)r * ldo + (c >> 1)] = f2b(gte * bfr(v1));
  }
};

}  // namespace pi05

namespace pi05 {
// Batched suffix (many episodes): EpiQKVRope with per-episode q / KV cache / position. Launched with sC1 = 1 so the
// epilogue offset is the episode index; every element is computed exactly as EpiQKVRope does for that episode.
struct EpiQKVRopeB {
  bf16* q;              // [episode][rows][heads][256]
  long long q_stride;   // elements per episode
  bf16* kc;             // layer base of episode 0's cache [pos][256]
  bf16* vt;             // layer base of episode 0's transposed V cache [256][vt_ld]
  long long kv_stride;  // elements per episode (depth * s_cap * 256)
  const float2* rope;
  const int* pos0;      // [episode] first position (= prefix tokens)
  int heads, vt_ld;
  __device__ void operator()(int r, int c, float v0, float v1, long long e) const {
    const int blk = c >> 8, cc = c & 255;
    const int pos = pos0[e] + r;
    if (blk <= heads) {
      const int j = cc >> 1;
      const float2 cs = rope[pos * 128 + j];
      const float x1 = bfr(v0), x2 = bfr(v1);
      const float o1 = bfr(x1 * cs.x - x2 * cs.y), o2 = bfr(x2 * cs.x + x1 * cs.y);
      if (blk < heads) {
        bf16* qo = q + e * q_stride + ((long long)r * heads + blk) * 256;
        qo[j] = f2b(o1 * 0.0625f);
        qo[j + 128] = f2b(o2 * 0.0625f);
      } else {
        bf16* k = kc + e * kv_stride;
        k[(long long)pos * 256 + j] = f2b(o1);
        k[(long long)pos * 256 + j + 128] = f2b(o2);
      }
    } else {
      bf16* v = vt + e * kv_stride;
      v[(long long)cc * vt_ld + pos] = f2b(v0);
      v[(long long)(cc + 1) * vt_ld + pos] = f2b(v1);
    }
  }
};
}  // namespace pi05
