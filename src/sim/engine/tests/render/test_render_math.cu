// 렌더 손 초월함수(shade.h: flog2, fexp2, fpow, srgb_to_lin, lin_to_srgb, sincos2pi) 층 1(CPU) = 층 2(GPU) 전수·표본 비트 비교.
//   test_render_math [표본 간격=1]
// flog2/srgb_to_lin: [2^-20, 4) 의 모든 float (간격으로 솎음), fexp2: [-30, 30] 의 모든 float, sincos2pi: [0,1) 의 모든 float.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <xmmintrin.h>

#include "core/render/shade.h"

using namespace eng::rnd;

__global__ void kMath(const float* in, float* out, int n, int fn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = in[i];
  float r = 0.0f;
  if (fn == 0) r = flog2(x);
  else if (fn == 1) r = fexp2(x);
  else if (fn == 2) r = srgb_to_lin(x);
  else if (fn == 3) r = lin_to_srgb(x);
  else if (fn == 4) { float s, c; sincos2pi(x, s, c); r = s + c * 3.0f; }
  out[i] = r;
}

static float host_fn(float x, int fn) {
  if (fn == 0) return flog2(x);
  if (fn == 1) return fexp2(x);
  if (fn == 2) return srgb_to_lin(x);
  if (fn == 3) return lin_to_srgb(x);
  float s, c;
  sincos2pi(x, s, c);
  return s + c * 3.0f;
}

int main(int argc, char** argv) {
  const int stride = argc > 1 ? atoi(argv[1]) : 1;
  _mm_setcsr(_mm_getcsr() | 0x8040u);  // GPU -ftz=true 와 같게
  const char* names[5] = {"flog2", "fexp2", "srgb_to_lin", "lin_to_srgb", "sincos2pi"};
  const float lo[5] = {9.5367432e-7f, -30.0f, 0.0f, 0.0f, 0.0f}, hi[5] = {4.0f, 30.0f, 1.0f, 1.0f, 0.99999994f};
  int bad_total = 0;
  for (int fn = 0; fn < 5; ++fn) {
    std::vector<float> xs;
    // lo..hi 의 모든 float (부호 넘김 처리: 음수 구간은 비트를 거꾸로)
    auto key = [](float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? int64_t(0x80000000u) - int64_t(u & 0x7fffffffu) : int64_t(0x80000000u) + u; };
    auto unkey = [](int64_t k) { uint32_t u = k >= 0x80000000LL ? uint32_t(k - 0x80000000LL) : (uint32_t(0x80000000LL - k) | 0x80000000u); float f; memcpy(&f, &u, 4); return f; };
    for (int64_t k = key(lo[fn]); k <= key(hi[fn]); k += stride) xs.push_back(unkey(k));
    const int n = int(xs.size());
    float *din, *dout;
    cudaMalloc(&din, size_t(n) * 4);
    cudaMalloc(&dout, size_t(n) * 4);
    cudaMemcpy(din, xs.data(), size_t(n) * 4, cudaMemcpyHostToDevice);
    kMath<<<(n + 255) / 256, 256>>>(din, dout, n, fn);
    std::vector<float> g(n);
    cudaMemcpy(g.data(), dout, size_t(n) * 4, cudaMemcpyDeviceToHost);
    cudaFree(din);
    cudaFree(dout);
    int bad = 0, shown = 0;
    for (int i = 0; i < n; ++i) {
      const float h = host_fn(xs[i], fn);
      if (memcmp(&h, &g[i], 4)) {
        ++bad;
        if (shown++ < 3) printf("  %s(%.9g = 0x%08x): 층1 %.9g 층2 %.9g\n", names[fn], xs[i], *(const uint32_t*)&xs[i], h, g[i]);
      }
    }
    printf("%-12s 입력 %d 개 중 비트 다름 %d\n", names[fn], n, bad);
    bad_total += bad;
  }
  return bad_total ? 1 : 0;
}
