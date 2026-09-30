// General bf16 tensor-core GEMM for training (forward and backward), fp32 accumulate:
//
//   C[M, N] = sum_k A(m, k) * B(k, n)
//     A_KM = true : A(m, k) = A[m * lda + k]   (K contiguous; activations x, dy)
//     A_KM = false: A(m, k) = A[k * lda + m]   (M contiguous; x^T in dW = x^T . dy)
//     B_KM = true : B(k, n) = B[n * ldb + k]   (K contiguous; W^T with W stored [out][in], or W [in][out] for dx)
//     B_KM = false: B(k, n) = B[k * ldb + n]   (N contiguous; openpi kernels stored [in][out], dy in dW)
//
// So openpi's native parameter layout ([in, out] Dense kernels, einsum weights) is used as is: forward y = x.W is
// (A_KM, !B_KM), input gradient dx = dy.W^T is (A_KM, B_KM), weight gradient dW = x^T.dy is (!A_KM, !B_KM).
// K-contiguous operands are staged in shared memory as [rows][BK], the others as [BK][rows]; both are XOR-swizzled
// in 16-byte chunks and read with ldmatrix (.trans for the M/N-contiguous tiles), so every case is conflict free.
// Requirements: the contiguous extent of every operand is a multiple of 8 elements and 16-byte aligned.
// Batched launches (grid.z) with two batch coordinates like the inference GEMM (b -> b / nb2, b % nb2).
#pragma once
#include "../../pi05_native/src/gemm.cuh"

namespace pi05t {
using namespace pi05;

struct TGemm {
  const bf16* A = nullptr;
  const bf16* B = nullptr;
  long long lda = 0, ldb = 0;
  int M = 0, N = 0, K = 0;
  int nb2 = 1;
  long long sA1 = 0, sA2 = 0, sB1 = 0, sB2 = 0, sC1 = 0, sC2 = 0;
};

__device__ __forceinline__ void ldmatrix_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// swizzled element offset of 16-byte chunk `ch` in row `r` of a tile whose rows hold CH chunks
template <int CH>
__device__ __forceinline__ int tswz(int r, int ch) {
  if constexpr (CH >= 8) return (r * CH + (ch ^ (r & 7))) * 8;
  else if constexpr (CH == 4) return (r * CH + (ch ^ ((r >> 1) & 3))) * 8;
  else return (r * CH + (ch ^ ((r >> 2) & 1))) * 8;
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, bool A_KM, bool B_KM, class Epi>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32) tgemm_kernel(TGemm p, Epi epi) {
  constexpr int BK = 32;
  constexpr int NT = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MT = WM / 16, NT8 = WN / 8;
  static_assert(WN % 16 == 0 && WM % 16 == 0, "warp tile");
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* sA = reinterpret_cast<bf16*>(smem_raw);
  bf16* sB = sA + STAGES * BM * BK;

  const int M = p.M, N = p.N, K = p.K;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int b = blockIdx.z, b1 = b / p.nb2, b2 = b % p.nb2;
  const bf16* A = p.A + b1 * p.sA1 + b2 * p.sA2;
  const bf16* B = p.B + b1 * p.sB1 + b2 * p.sB2;
  const long long coff = b1 * p.sC1 + b2 * p.sC2;
  const int ktiles = (K + BK - 1) / BK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;

  auto load_a = [&](int stage, int kt) {
    const int k0 = kt * BK;
    bf16* a = sA + stage * BM * BK;
    if constexpr (A_KM) {  // [BM][BK]
      constexpr int CH = BK / 8;
      for (int c = tid; c < BM * CH; c += NT) {
        const int r = c / CH, ch = c % CH, gr = m0 + r, gk = k0 + ch * 8;
        const bool ok = gr < M && gk < K;
        cp_async16(smem_u32(a + tswz<CH>(r, ch)), ok ? A + (long long)gr * p.lda + gk : A, ok ? 16 : 0);
      }
    } else {  // [BK][BM]
      constexpr int CH = BM / 8;
      for (int c = tid; c < BK * CH; c += NT) {
        const int r = c / CH, ch = c % CH, gk = k0 + r, gm = m0 + ch * 8;
        const bool ok = gk < K && gm < M;
        cp_async16(smem_u32(a + tswz<CH>(r, ch)), ok ? A + (long long)gk * p.lda + gm : A, ok ? 16 : 0);
      }
    }
  };
  auto load_b = [&](int stage, int kt) {
    const int k0 = kt * BK;
    bf16* bb = sB + stage * BN * BK;
    if constexpr (B_KM) {  // [BN][BK]
      constexpr int CH = BK / 8;
      for (int c = tid; c < BN * CH; c += NT) {
        const int r = c / CH, ch = c % CH, gn = n0 + r, gk = k0 + ch * 8;
        const bool ok = gn < N && gk < K;
        cp_async16(smem_u32(bb + tswz<CH>(r, ch)), ok ? B + (long long)gn * p.ldb + gk : B, ok ? 16 : 0);
      }
    } else {  // [BK][BN]
      constexpr int CH = BN / 8;
      for (int c = tid; c < BK * CH; c += NT) {
        const int r = c / CH, ch = c % CH, gk = k0 + r, gn = n0 + ch * 8;
        const bool ok = gk < K && gn < N;
        cp_async16(smem_u32(bb + tswz<CH>(r, ch)), ok ? B + (long long)gk * p.ldb + gn : B, ok ? 16 : 0);
      }
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
    if (s < ktiles) { load_a(s, s); load_b(s, s); }
    cp_async_commit();
  }
  for (int i = 0; i < ktiles; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < ktiles) { load_a(nxt % STAGES, nxt); load_b(nxt % STAGES, nxt); }
    cp_async_commit();
    const bf16* a = sA + (i % STAGES) * BM * BK;
    const bf16* bb = sB + (i % STAGES) * BN * BK;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t af[MT][4];
      uint32_t bfg[NT8][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int mb = wm * WM + mt * 16;
        if constexpr (A_KM) {
          const int r = mb + (lane & 15);
          ldmatrix_x4(af[mt], smem_u32(a + tswz<BK / 8>(r, (kk >> 3) + (lane >> 4))));
        } else {  // matrices (m0-7,k0-7) (m8-15,k0-7) (m0-7,k8-15) (m8-15,k8-15), rows of the tile are k
          const int j = lane >> 3;
          const int k = kk + ((j >> 1) << 3) + (lane & 7), m = mb + ((j & 1) << 3);
          ldmatrix_x4_t(af[mt], smem_u32(a + tswz<BM / 8>(k, m >> 3)));
        }
      }
#pragma unroll
      for (int np = 0; np < NT8 / 2; ++np) {
        const int nb = wn * WN + np * 16;
        uint32_t t[4];
        if constexpr (B_KM) {
          const int r = nb + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(t, smem_u32(bb + tswz<BK / 8>(r, (kk >> 3) + ((lane >> 3) & 1))));
        } else {  // matrices (k0-7,n0-7) (k8-15,n0-7) (k0-7,n8-15) (k8-15,n8-15)
          const int j = lane >> 3;
          const int k = kk + ((j & 1) << 3) + (lane & 7), n = nb + ((j >> 1) << 3);
          ldmatrix_x4_t(t, smem_u32(bb + tswz<BN / 8>(k, n >> 3)));
        }
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
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int row = m0 + wm * WM + mt * 16 + g;
      const int col = n0 + wn * WN + nt * 8 + 2 * t4;
      if (col >= N) continue;
      if (row < M) epi(row, col, acc[mt][nt][0], acc[mt][nt][1], coff);
      if (row + 8 < M) epi(row + 8, col, acc[mt][nt][2], acc[mt][nt][3], coff);
    }
}

template <bool A_KM, bool B_KM, class Epi>
void tgemm(const TGemm& p, const Epi& epi, int batch, cudaStream_t st) {
  if (p.M <= 0 || p.N <= 0 || batch <= 0) return;
  auto run = [&](auto kern, int bm, int bn, int threads, int smem) {
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
      PI05_CUDA(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      attr = true;
    }
    dim3 grid(cdiv(p.M, bm), cdiv(p.N, bn), batch);
    kern<<<grid, threads, smem, st>>>(p, epi);
    PI05_CUDA(cudaGetLastError());
  };
  if ((long long)p.M * p.N >= 256LL * 1024)
    run(tgemm_kernel<128, 128, 2, 4, 3, A_KM, B_KM, Epi>, 128, 128, 256, 3 * 256 * 32 * 2);
  else
    run(tgemm_kernel<64, 64, 2, 2, 4, A_KM, B_KM, Epi>, 64, 64, 128, 4 * 128 * 32 * 2);
}

// ---- epilogues ------------------------------------------------------------------------------------------------------
struct EStoreBf16 {  // bf16 = round(acc)
  bf16* out;
  long long ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    *reinterpret_cast<bf162*>(out + off + r * ldo + c) = __floats2bfloat162_rn(v0, v1);
  }
};
struct EStoreF32 {
  float* out;
  long long ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    *reinterpret_cast<float2*>(out + off + r * ldo + c) = make_float2(v0, v1);
  }
};
struct EAddF32 {  // f32 accumulate (gradient sums over samples / heads)
  float* out;
  long long ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float2* p = reinterpret_cast<float2*>(out + off + r * ldo + c);
    float2 x = *p;
    x.x += v0;
    x.y += v1;
    *p = x;
  }
};
struct EAddBf16F32 {  // f32 += bf16(acc): a bf16 einsum result summed in f32
  float* out;
  long long ldo;
  __device__ void operator()(int r, int c, float v0, float v1, long long off) const {
    float2* p = reinterpret_cast<float2*>(out + off + r * ldo + c);
    float2 x = *p;
    x.x += bfr(v0);
    x.y += bfr(v1);
    *p = x;
  }
};

}  // namespace pi05t
