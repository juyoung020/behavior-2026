// 층 2 시험: glibc 2.35 tanf 이식본의 CUDA 결과 = 이 PC libm (FTZ/DAZ 켬 — GPU -ftz=true 와 같은 조건), float 2^32 전부.
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/joints/glibc_tanf.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void kTan(uint64_t base, uint32_t n, float* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eng::glibc::tanf(__uint_as_float(uint32_t(base + i)));
}
static uint32_t fu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float uf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static float (*volatile ref_tanf)(float) = ::tanf;

int main() {
  const int nt = int(std::thread::hardware_concurrency());
  const uint32_t CH = 1u << 27;
  float* dOut;
  CK(cudaMalloc(&dOut, sizeof(float) * CH));
  std::vector<float> h(CH);
  uint64_t bad = 0;
  int64_t first = -1;
  for (uint64_t base = 0; base < (1ull << 32); base += CH) {
    kTan<<<(CH + 255) / 256, 256>>>(base, CH, dOut);
    CK(cudaGetLastError());
    CK(cudaMemcpy(h.data(), dOut, sizeof(float) * CH, cudaMemcpyDeviceToHost));
    std::atomic<uint64_t> b{0};
    std::atomic<int64_t> fi{-1};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));
        const uint32_t lo = uint32_t(uint64_t(CH) * t / nt), hi = uint32_t(uint64_t(CH) * (t + 1) / nt);
        uint64_t bb = 0;
        for (uint32_t i = lo; i < hi; ++i)
          if (fu(ref_tanf(uf(uint32_t(base + i)))) != fu(h[i]) && bb++ == 0) {
            int64_t e = -1;
            fi.compare_exchange_strong(e, int64_t(base + i));
          }
        b += bb;
      });
    for (auto& x : th) x.join();
    bad += b;
    if (first < 0 && fi >= 0) first = fi;
  }
  printf("  tanf GPU: 입력 2^32 개, libm(FTZ) 과 비트 다름 %" PRIu64, bad);
  if (first >= 0) printf("  첫 다름 x=%08x", uint32_t(first));
  printf("\n%s\n", bad == 0 ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  cudaFree(dOut);
  return bad == 0 ? 0 : 3;
}
