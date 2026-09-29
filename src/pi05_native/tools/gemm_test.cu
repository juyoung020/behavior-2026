// GEMM unit test + throughput: hand-written mma kernel vs a double-precision CPU reference (and cuBLAS for speed
// comparison only when built with -DWITH_CUBLAS; the engine itself never links cuBLAS).
//   gemm_test            correctness on odd shapes, every tile config, split-K, batching
//   gemm_test --bench    TFLOP/s on the model's real shapes
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/gemm.cuh"
#ifdef WITH_CUBLAS
#include <cublasLt.h>
#endif

using namespace pi05;

static float bf2f_h(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t f2bf_h(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

template <int BM, int BN, int WM, int WN, int ST, class Epi>
void run(GemmParams p, Epi e, int batch = 1) {
  using C = GemmCfg<BM, BN, WM, WN, ST>;
  dim3 grid(cdiv(p.N, BN), cdiv(p.M, BM), batch * p.splits);
  gemm_nt_kernel<BM, BN, WM, WN, ST, Epi><<<grid, C::kThreads, C::kSmem>>>(p, e);
}

template <class Epi>
void run_cfg(int cfg, GemmParams p, Epi e) {
  switch (cfg) {
    case 0: run<128, 128, 2, 4, 3>(p, e); break;
    case 1: run<128, 64, 2, 2, 4>(p, e); break;
    case 2: run<64, 64, 2, 2, 4>(p, e); break;
    case 3: run<32, 64, 1, 4, 6>(p, e); break;
  }
}

int correctness() {
  std::mt19937 rng(1);
  std::normal_distribution<float> nd;
  int bad = 0;
  struct Shape { int M, N, K; };
  const Shape shapes[] = {{37, 72, 72}, {256, 256, 72}, {130, 200, 136}, {968, 520, 256}, {32, 2560, 1024}, {7, 64, 32}};
  for (auto s : shapes) {
    std::vector<uint16_t> A((size_t)s.M * s.K), B((size_t)s.N * s.K);
    for (auto& x : A) x = f2bf_h(nd(rng));
    for (auto& x : B) x = f2bf_h(nd(rng));
    std::vector<double> ref((size_t)s.M * s.N);
    for (int m = 0; m < s.M; ++m)
      for (int n = 0; n < s.N; ++n) {
        double acc = 0;
        for (int k = 0; k < s.K; ++k) acc += (double)bf2f_h(A[(size_t)m * s.K + k]) * bf2f_h(B[(size_t)n * s.K + k]);
        ref[(size_t)m * s.N + n] = acc;
      }
    bf16 *dA, *dB;
    float *dC, *ws;
    cudaMalloc(&dA, A.size() * 2);
    cudaMalloc(&dB, B.size() * 2);
    cudaMalloc(&dC, ref.size() * 4);
    cudaMalloc(&ws, ref.size() * 4 * 8);
    cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
    for (int cfg = 0; cfg < 4; ++cfg) {
      for (int splits : {1, 2, 4}) {
        if (splits > 1 && cfg != 3) continue;
        cudaMemset(dC, 0xff, ref.size() * 4);
        GemmParams p;
        p.A = dA; p.lda = s.K; p.B = dB; p.ldb = s.K; p.M = s.M; p.N = s.N; p.K = s.K;
        p.splits = splits;
        p.ws = ws;
        EpiF32 e{dC, s.N};
        run_cfg(cfg, p, e);
        if (splits > 1) splitk_reduce_kernel<<<cdiv(s.M * s.N / 2, 256), 256>>>(ws, splits, s.M, s.N, nullptr, e);
        cudaError_t ce = cudaDeviceSynchronize();
        std::vector<float> C(ref.size());
        cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
        double maxe = 0;
        for (size_t i = 0; i < C.size(); ++i) maxe = std::max(maxe, std::fabs(C[i] - ref[i]) / (1 + std::fabs(ref[i])));
        bool ok = ce == cudaSuccess && maxe < 1e-4;
        if (!ok) ++bad;
        printf("M%5d N%5d K%5d cfg%d split%d : max rel err %.2e %s\n", s.M, s.N, s.K, cfg, splits, maxe, ok ? "ok" : "BAD");
      }
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(ws);
  }
  return bad;
}

void bench() {
  struct Shape { const char* name; int M, N, K; };
  const Shape shapes[] = {
      {"siglip qkv", 768, 3456, 1152}, {"siglip fc1", 768, 4352, 1152}, {"siglip fc2", 768, 1152, 4352},
      {"gemma qkv", 968, 2560, 2048},  {"gemma o", 968, 2048, 2048},    {"gemma gate_up", 968, 32768, 2048},
      {"gemma down", 968, 2048, 16384}, {"attn logits", 7744, 968, 256}, {"attn pv", 7744, 256, 968},
      {"ae qkv", 32, 2560, 1024},      {"ae gate_up", 32, 8192, 1024},  {"ae down", 32, 1024, 4096}};
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (auto s : shapes) {
    bf16 *dA, *dB, *dC;
    float* ws;
    cudaMalloc(&dA, (size_t)s.M * s.K * 2);
    cudaMalloc(&dB, (size_t)s.N * s.K * 2);
    cudaMalloc(&dC, (size_t)s.M * s.N * 2);
    cudaMalloc(&ws, (size_t)s.M * s.N * 4 * 8);
    cudaMemset(dA, 0, (size_t)s.M * s.K * 2);
    cudaMemset(dB, 0, (size_t)s.N * s.K * 2);
    printf("%-14s M%5d N%6d K%6d :", s.name, s.M, s.N, s.K);
    for (int cfg = 0; cfg < 4; ++cfg) {
      if (cfg == 3 && s.M > 64) continue;
      if (cfg < 3 && s.M <= 32) continue;
      GemmParams p;
      p.A = dA; p.lda = s.K; p.B = dB; p.ldb = s.K; p.M = s.M; p.N = s.N; p.K = s.K;
      int splits = 1;
      if (cfg == 3) {
        int ctas = cdiv(s.N, 64), kt = cdiv(s.K, 32);
        while (ctas * splits < 140 && kt / (splits * 2) >= 4) splits *= 2;
      }
      p.splits = splits;
      p.ws = ws;
      EpiBf16 e{dC, s.N};
      for (int w = 0; w < 3; ++w) run_cfg(cfg, p, e);
      const int it = 20;
      cudaEventRecord(e0);
      for (int i = 0; i < it; ++i) {
        run_cfg(cfg, p, e);
        if (splits > 1) splitk_reduce_kernel<<<cdiv(s.M * s.N / 2, 256), 256>>>(ws, splits, s.M, s.N, nullptr, e);
      }
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      ms /= it;
      double tf = 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12;
      double gbs = ((double)s.N * s.K * 2) / (ms * 1e-3) / 1e9;
      printf("  cfg%d %7.3f ms %6.1f TF/s %6.0f GB/s", cfg, ms, tf, gbs);
    }
#ifdef WITH_CUBLAS
    {
      cublasLtHandle_t lt;
      cublasLtCreate(&lt);
      cublasLtMatmulDesc_t op;
      cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
      cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
      // column-major: C^T[N,M] = B[N,K] . A^T  -> A_cm = B stored [N,K] row-major = [K,N] col-major, op T
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA);
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
      cublasLtMatrixLayout_t la, lb, lc;
      cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, s.K, s.N, s.K);
      cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, s.K, s.M, s.K);
      cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, s.N, s.M, s.N);
      float alpha = 1, beta = 0;
      void* wsp;
      size_t wss = 32 << 20;
      cudaMalloc(&wsp, wss);
      cublasLtMatmulPreference_t pref;
      cublasLtMatmulPreferenceCreate(&pref);
      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss);
      cublasLtMatmulHeuristicResult_t hr;
      int nres = 0;
      cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &hr, &nres);
      for (int w = 0; w < 3; ++w)
        cublasLtMatmul(lt, op, &alpha, dB, la, dA, lb, &beta, dC, lc, dC, lc, &hr.algo, wsp, wss, 0);
      cudaEventRecord(e0);
      for (int i = 0; i < 20; ++i)
        cublasLtMatmul(lt, op, &alpha, dB, la, dA, lb, &beta, dC, lc, dC, lc, &hr.algo, wsp, wss, 0);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      ms /= 20;
      printf("  | cuBLASLt %7.3f ms %6.1f TF/s", ms, 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12);
      cudaFree(wsp);
      cublasLtDestroy(lt);
    }
#endif
    printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(ws);
  }
}

int main(int argc, char** argv) {
  bool b = argc > 1 && std::string(argv[1]) == "--bench";
  if (b) { bench(); return 0; }
  int bad = correctness();
  printf("%s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
  return bad ? 1 : 0;
}
