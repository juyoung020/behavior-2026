// 층 2 시험: glibc 2.35 atanf/asinf/acosf 이식본의 CUDA 결과 = 이 PC libm (FTZ/DAZ 켬 — GPU -ftz=true 와 같은 조건), float 2^32 전부.
// atan2f 는 무작위 쌍 (test_libm_joints 와 같은 생성 규칙).
//   test_libm_joints_gpu [--pairs N]
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "core/joints/glibc_trig.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace G = eng::glibcj;

__global__ void kUnary(int f, uint64_t base, uint32_t n, float* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = __uint_as_float(uint32_t(base + i));
  out[i] = f == 0 ? G::atanf(x) : (f == 1 ? G::asinf(x) : G::acosf(x));
}
__global__ void kAtan2(const float* y, const float* x, uint32_t n, float* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = G::atan2f(y[i], x[i]);
}

static uint32_t fu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float uf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static float (*volatile ref_atanf)(float) = ::atanf;
static float (*volatile ref_asinf)(float) = ::asinf;
static float (*volatile ref_acosf)(float) = ::acosf;
static float (*volatile ref_atan2f)(float, float) = ::atan2f;
static void ftzOn() { _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }

int main(int argc, char** argv) {
  uint64_t pairs = 1ull << 26;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = strtoull(argv[++i], nullptr, 10);
  const int nt = int(std::thread::hardware_concurrency());
  const uint32_t CH = 1u << 27;
  float* dOut;
  CK(cudaMalloc(&dOut, sizeof(float) * CH));
  std::vector<float> h(CH);
  const char* names[3] = {"atanf", "asinf", "acosf"};
  bool allOk = true;
  for (int f = 0; f < 3; ++f) {
    uint64_t bad = 0;
    int64_t first = -1;
    for (uint64_t base = 0; base < (1ull << 32); base += CH) {
      kUnary<<<(CH + 255) / 256, 256>>>(f, base, CH, dOut);
      CK(cudaGetLastError());
      CK(cudaMemcpy(h.data(), dOut, sizeof(float) * CH, cudaMemcpyDeviceToHost));
      std::atomic<uint64_t> b{0};
      std::atomic<int64_t> fi{-1};
      std::vector<std::thread> th;
      for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t]() {
          ftzOn();
          const uint32_t lo = uint32_t(uint64_t(CH) * t / nt), hi = uint32_t(uint64_t(CH) * (t + 1) / nt);
          uint64_t bb = 0;
          for (uint32_t i = lo; i < hi; ++i) {
            const float x = uf(uint32_t(base + i));
            const float r = f == 0 ? ref_atanf(x) : (f == 1 ? ref_asinf(x) : ref_acosf(x));
            // NaN: GPU 는 NaN 연산이 꼬리를 버릴 수 있으나 이식본은 비트로 만든다 -> 그대로 비교
            if (fu(r) != fu(h[i])) {
              if (bb++ == 0) { int64_t ex = -1; fi.compare_exchange_strong(ex, int64_t(base + i)); }
            }
          }
          b += bb;
        });
      for (auto& x : th) x.join();
      bad += b;
      if (first < 0 && fi >= 0) first = fi;
    }
    printf("  %-6s GPU: 입력 2^32 개, libm(FTZ) 과 비트 다름 %" PRIu64, names[f], bad);
    if (first >= 0) printf("  첫 다름 x=%08x", uint32_t(first));
    printf("\n");
    allOk &= bad == 0;
  }
  // atan2f
  {
    std::vector<float> ys(pairs), xs(pairs), rs(pairs);
    std::mt19937_64 rng(99);
    for (uint64_t i = 0; i < pairs; ++i) {
      const uint64_t r = rng();
      uint32_t ya = uint32_t(r), xa = uint32_t(r >> 32);
      if (i & 1) xa = (xa & 0x807fffffu) | (ya & 0x7f800000u);
      ys[i] = uf(ya);
      xs[i] = uf(xa);
    }
    float *dy, *dx, *dr;
    CK(cudaMalloc(&dy, sizeof(float) * pairs));
    CK(cudaMalloc(&dx, sizeof(float) * pairs));
    CK(cudaMalloc(&dr, sizeof(float) * pairs));
    CK(cudaMemcpy(dy, ys.data(), sizeof(float) * pairs, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xs.data(), sizeof(float) * pairs, cudaMemcpyHostToDevice));
    kAtan2<<<uint32_t((pairs + 255) / 256), 256>>>(dy, dx, uint32_t(pairs), dr);
    CK(cudaGetLastError());
    CK(cudaMemcpy(rs.data(), dr, sizeof(float) * pairs, cudaMemcpyDeviceToHost));
    ftzOn();
    uint64_t bad = 0;
    for (uint64_t i = 0; i < pairs; ++i)
      if (fu(ref_atan2f(ys[i], xs[i])) != fu(rs[i])) {
        if (bad == 0) printf("  atan2f 첫 다름 y=%08x x=%08x libm=%08x gpu=%08x\n", fu(ys[i]), fu(xs[i]), fu(ref_atan2f(ys[i], xs[i])), fu(rs[i]));
        bad++;
      }
    printf("  atan2f GPU: 무작위 %" PRIu64 " 쌍, libm(FTZ) 과 비트 다름 %" PRIu64 "\n", pairs, bad);
    allOk &= bad == 0;
    cudaFree(dy); cudaFree(dx); cudaFree(dr);
  }
  cudaFree(dOut);
  printf("%s\n", allOk ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  return allOk ? 0 : 3;
}
