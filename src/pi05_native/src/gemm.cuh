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

constexpr int GEMM_BK = 32;  // 64-byte rows: 4 x 16B chunks, chunk ^= (row >> 1) & 3

__device__ __forceinline__ int swz(int r, int ch) { return r * GEMM_BK + ((ch ^ ((r >> 1) & 3)) << 3); }

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
struct GemmCfg {
  static constexpr int kThreads = WARPS_M * WARPS_N * 32;
  static constexpr int kSmem = STAGES * (BM + BN) * GEMM_BK * 2;
};

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, class Epi>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32)
    gemm_nt_kernel(GemmParams p, Epi epi) {
  constexpr int BK = GEMM_BK;
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
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
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

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * BK;
    bf16* a = sA + stage * BM * BK;
    bf16* bb = sB + stage * BN * BK;
#pragma unroll
    for (int c = tid; c < BM * 4; c += NT) {
      int r = c >> 2, ch = c & 3;
      int gr = m0 + r, gk = k0 + ch * 8;
      bool ok = gr < M && gk < K;
      const bf16* src = ok ? A + (long long)gr * p.lda + gk : A;
      cp_async16(smem_u32(a + swz(r, ch)), src, ok ? 16 : 0);
    }
#pragma unroll
    for (int c = tid; c < BN * 4; c += NT) {
      int r = c >> 2, ch = c & 3;
      int gr = n0 + r, gk = k0 + ch * 8;
      bool ok = gr < N && gk < K;
      const bf16* src = ok ? B + (long long)gr * p.ldb + gk : B;
      cp_async16(smem_u32(bb + swz(r, ch)), src, ok ? 16 : 0);
    }
  };

  float acc[MT][NT8][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_stage(s, kt0 + s);
    cp_async_commit();
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
        ldmatrix_x4(af[mt], smem_u32(a + swz(r, kk + (lane >> 4))));
      }
#pragma unroll
      for (int np = 0; np < NT8 / 2; ++np) {
        int r = wn * WN + np * 16 + (lane & 7) + ((lane >> 4) << 3);
        uint32_t t[4];
        ldmatrix_x4(t, smem_u32(bb + swz(r, kk + ((lane >> 3) & 1))));
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
        float* w = p.ws + (long long)split * p.M * p.N;
        if (row < M) *reinterpret_cast<float2*>(w + (long long)row * p.N + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (row + 8 < M)
          *reinterpret_cast<float2*>(w + (long long)(row + 8) * p.N + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      } else {
        if (row < M) epi(row, col, acc[mt][nt][0], acc[mt][nt][1], coff);
        if (row + 8 < M) epi(row + 8, col, acc[mt][nt][2], acc[mt][nt][3], coff);
      }
    }
  }
}

// Sums split-K partials in a fixed order (deterministic) and applies the epilogue.
template <class Epi>
__global__ void splitk_reduce_kernel(const float* ws, int splits, int M, int N, const int* dM, Epi epi) {
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

struct EpiGeluGate {  // Gemma MLP: gelu(x.Wg) * (x.Wu), columns interleaved (gate, up) -> out col c/2
  bf16* out;
  int ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float gte = gelu_bf16(bfr(v0));
    out[off + (long long)r * ldo + (c >> 1)] = f2b(gte * bfr(v1));
  }
};

}  // namespace pi05
