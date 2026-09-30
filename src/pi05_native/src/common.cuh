// Shared CUDA helpers for the pi0.5 engine.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Programmatic dependent launch (sm_90+). Every kernel begins with pdl_entry(): first allow the next kernel in the
// stream to be scheduled (it only gets SM slots once all of our CTAs are running, i.e. it fills our tail), then wait
// until the previous kernel has completed and its writes are visible. Without the launch attribute (PI05_PDL=0) both
// instructions are no-ops, so correctness never depends on it.
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_entry() {
  pdl_trigger();
  pdl_wait();
}

inline bool pdl_enabled() {
  static const bool on = [] {
    const char* v = getenv("PI05_PDL");
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return !(v && v[0] == '0') && major >= 9;  // programmatic dependent launch needs sm_90+
  }();
  return on;
}

// Kernel launch with the programmatic-stream-serialization attribute (captured into the CUDA graph as a PDL edge).
template <typename... KArgs, typename... Args>
inline void launch_k(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t c = {};
  c.gridDim = grid;
  c.blockDim = block;
  c.dynamicSmemBytes = smem;
  c.stream = st;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[0].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = a;
  c.numAttrs = pdl_enabled() ? 1 : 0;
  PI05_CUDA(cudaLaunchKernelEx(&c, k, static_cast<KArgs>(args)...));
}

}  // namespace pi05
