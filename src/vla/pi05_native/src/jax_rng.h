// The openpi policy's start-noise stream, reproduced: jax.random.key(0), then per inference
//   rng, sample = jax.random.split(rng);  noise = jax.random.normal(sample, (1, ah, ad))
// (openpi policies/policy.py:75, models/pi0.py:231). JAX 0.5.3, threefry2x32 with jax_threefry_partitionable=True.
// Bits are identical to JAX; normal() = sqrt(2) * erfinv(uniform(-1+ulp, 1)) with XLA's ErfInv32 polynomial, so values
// can differ from JAX by the last ulp of log1p (host libm vs XLA).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace pi05 {

struct JaxKey {
  uint32_t k0 = 0, k1 = 0;
};

inline void threefry2x32(uint32_t k0, uint32_t k1, uint32_t& x0, uint32_t& x1) {
  auto rotl = [](uint32_t v, int r) { return (v << r) | (v >> (32 - r)); };
  const uint32_t ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  static const int R[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
  x0 += ks[0];
  x1 += ks[1];
  for (int i = 0; i < 5; ++i) {
    for (int r : R[i & 1]) {
      x0 += x1;
      x1 = rotl(x1, r);
      x1 ^= x0;
    }
    x0 += ks[(i + 1) % 3];
    x1 += ks[(i + 2) % 3] + (uint32_t)(i + 1);
  }
}

inline JaxKey jax_key(uint64_t seed) { return {(uint32_t)(seed >> 32), (uint32_t)seed}; }

// jax.random.split(key) (num=2), fold-like partitionable variant: counters (hi, lo) = (0, i)
inline void jax_split(const JaxKey& key, JaxKey* a, JaxKey* b) {
  const JaxKey k = key;  // a may alias key
  uint32_t x0 = 0, x1 = 0;
  threefry2x32(k.k0, k.k1, x0, x1);
  *a = {x0, x1};
  x0 = 0, x1 = 1;
  threefry2x32(k.k0, k.k1, x0, x1);
  *b = {x0, x1};
}

// XLA ErfInv32 (xla/hlo/builder/lib/math.cc)
inline float xla_erfinv(float x) {
  static const float lt[9] = {2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f, -4.39150654e-06f, 0.00021858087f,
                              -0.00125372503f, -0.00417768164f, 0.246640727f,    1.50140941f};
  static const float gt[9] = {-0.000200214257f, 0.000100950558f, 0.00134934322f, -0.00367342844f, 0.00573950773f,
                              -0.0076224613f,   0.00943887047f,  1.00167406f,    2.83297682f};
  float w = -log1pf(-x * x);
  const bool small = w < 5.0f;
  w = small ? w - 2.5f : sqrtf(w) - 3.0f;
  const float* c = small ? lt : gt;
  float p = c[0];
  for (int i = 1; i < 9; ++i) p = c[i] + p * w;
  if (fabsf(x) == 1.0f) return x * 3.40282347e38f;
  return p * x;
}

// jax.random.normal(key, (n,)) float32
inline void jax_normal(const JaxKey& k, float* out, int n) {
  float lo = nextafterf(-1.0f, 0.0f);
  const float hi = 1.0f;
  for (int i = 0; i < n; ++i) {
    uint32_t x0 = 0, x1 = (uint32_t)i;  // iota as (hi, lo)
    threefry2x32(k.k0, k.k1, x0, x1);
    const uint32_t bits = x0 ^ x1;
    const uint32_t fb = (bits >> 9) | 0x3F800000u;
    float f;
    memcpy(&f, &fb, 4);
    f -= 1.0f;
    float u = f * (hi - lo) + lo;
    if (u < lo) u = lo;
    out[i] = 1.41421354f * xla_erfinv(u);
  }
}

}  // namespace pi05
