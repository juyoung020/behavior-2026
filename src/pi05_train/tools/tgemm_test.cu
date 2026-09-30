// Checks the training GEMM (all operand layouts) against a double-precision CPU reference on random bf16 data.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/tgemm.cuh"

using namespace pi05t;

static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <bool AK, bool BK>
int run(int M, int N, int K, int batch) {
  std::mt19937 rng(M * 7 + N * 13 + K);
  std::normal_distribution<float> nd;
  const long long asz = (long long)M * K, bsz = (long long)N * K;
  std::vector<float> ha(asz * batch), hb(bsz * batch);
  for (auto& v : ha) v = bf(nd(rng));
  for (auto& v : hb) v = bf(nd(rng));
  std::vector<bf16> a16(ha.size()), b16(hb.size());
  for (size_t i = 0; i < ha.size(); ++i) a16[i] = __float2bfloat16_rn(ha[i]);
  for (size_t i = 0; i < hb.size(); ++i) b16[i] = __float2bfloat16_rn(hb[i]);
  bf16 *da, *db;
  float* dc;
  cudaMalloc(&da, a16.size() * 2);
  cudaMalloc(&db, b16.size() * 2);
  cudaMalloc(&dc, (size_t)M * N * batch * 4);
  cudaMemcpy(da, a16.data(), a16.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b16.data(), b16.size() * 2, cudaMemcpyHostToDevice);
  TGemm p;
  p.A = da; p.B = db; p.M = M; p.N = N; p.K = K;
  p.lda = AK ? K : M;
  p.ldb = BK ? K : N;
  p.sA1 = asz; p.sB1 = bsz; p.sC1 = (long long)M * N;
  tgemm<AK, BK>(p, EStoreF32{dc, N}, batch, 0);
  cudaDeviceSynchronize();
  std::vector<float> hc((size_t)M * N * batch);
  cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost);
  double maxrel = 0;
  for (int z = 0; z < batch; ++z)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double s = 0, sa = 0;
        for (int k = 0; k < K; ++k) {
          const float av = AK ? ha[z * asz + (long long)m * K + k] : ha[z * asz + (long long)k * M + m];
          const float bv = BK ? hb[z * bsz + (long long)n * K + k] : hb[z * bsz + (long long)k * N + n];
          s += (double)av * bv;
          sa += std::fabs((double)av * bv);
        }
        const double d = std::fabs(hc[(size_t)z * M * N + (size_t)m * N + n] - s) / (sa + 1e-30);
        maxrel = std::max(maxrel, d);
      }
  cudaFree(da); cudaFree(db); cudaFree(dc);
  const bool ok = maxrel < 1e-5;
  printf("A_KM %d B_KM %d  M %4d N %4d K %4d batch %d : max |err|/sum|ab| %.2e %s\n", AK, BK, M, N, K, batch, maxrel,
         ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int bad = 0;
  const int shapes[][4] = {{32, 1024, 256, 1}, {256, 912, 256, 1}, {136, 200, 72, 2}, {1024, 1024, 40, 1},
                           {904, 256, 256, 1}, {64, 3072, 8, 2}};
  for (auto& s : shapes) {
    bad += run<true, true>(s[0], s[1], s[2], s[3]);
    bad += run<true, false>(s[0], s[1], s[2], s[3]);
    bad += run<false, false>(s[0], s[1], s[2], s[3]);
    bad += run<false, true>(s[0], s[1], s[2], s[3]);
  }
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
