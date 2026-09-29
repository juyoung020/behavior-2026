// Shared CUDA helpers for the pi0.5 engine.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define PI05_CUDA(call)                                                                        \
  do {                                                                                         \
    cudaError_t e_ = (call);                                                                   \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, \
              cudaGetErrorString(e_));                                                         \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

namespace pi05 {

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

// Round a float to bfloat16 (round to nearest even) and back: the value a JAX bf16 op would store.
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }

// jax.nn.gelu(x, approximate=True) evaluated on a bf16 tensor (flax nn.gelu default).
//   cdf = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)));  return x * cdf
// Python float constants are weakly typed, so they become bf16 constants; sqrt(2/pi) is cast to
// x.dtype explicitly (jax/_src/nn/functions.py). PI05_GELU_F32 keeps the chain in f32 between those
// bf16 constants (XLA fusion with excess precision); otherwise every op rounds to bf16.
__device__ __forceinline__ float gelu_bf16(float x) {
  const float c0 = 0.044677734375f;  // bf16(0.044715)
  const float c1 = 0.796875f;        // bf16(sqrt(2/pi) = 0.79788456)
#ifdef PI05_GELU_F32
  float x3 = x * x * x;
  float inner = c1 * (x + c0 * x3);
  float cdf = 0.5f * (1.0f + tanhf(inner));
  return bfr(x * cdf);
#else
  float x3 = bfr(bfr(x * x) * x);
  float inner = bfr(c1 * bfr(x + bfr(c0 * x3)));
  float cdf = bfr(0.5f * bfr(1.0f + bfr(tanhf(inner))));
  return bfr(x * cdf);
#endif
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace pi05
