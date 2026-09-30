// JAX random numbers of the openpi training step, reproduced on the host (threefry2x32, jax_threefry_partitionable):
//   scripts/train.py:137-150  train_rng = fold_in(train_rng0, step)
//   models/pi0.py:192-197      preprocess_rng, noise_rng, time_rng = split(train_rng, 3)
//                              noise = normal(noise_rng, [B, ah, ad]); time = beta(time_rng, 1.5, 1, [B]) * 0.999 + 0.001
//   models/model.py:183        per-camera augmentation keys = split(preprocess_rng, B)
// jax/_src/random.py: split, uniform, bernoulli, exponential, gamma (Marsaglia-Tsang, log space), beta.
// Bits are identical to JAX; float results can differ in the last ulp where the host libm (log, exp, sqrt, log1p)
// differs from XLA's, which in turn can (rarely) flip a gamma accept/reject decision.
#pragma once
#include <vector>

#include "../../pi05_native/src/jax_rng.h"

namespace pi05t {
using pi05::JaxKey;

inline JaxKey jr_child(const JaxKey& k, uint32_t i) {  // split(k, n)[i] == fold_in(k, i)
  uint32_t x0 = 0, x1 = i;
  pi05::threefry2x32(k.k0, k.k1, x0, x1);
  return {x0, x1};
}
inline std::vector<JaxKey> jr_split(const JaxKey& k, int n) {
  std::vector<JaxKey> v(n);
  for (int i = 0; i < n; ++i) v[i] = jr_child(k, (uint32_t)i);
  return v;
}
inline JaxKey jr_fold_in(const JaxKey& k, uint32_t data) { return jr_child(k, data); }
inline float jr_bits_to_unit(uint32_t bits) {
  const uint32_t fb = (bits >> 9) | 0x3F800000u;
  float f;
  memcpy(&f, &fb, 4);
  return f - 1.0f;
}
// jax.random.uniform(key, (n,), minval, maxval) float32
inline void jr_uniform(const JaxKey& k, float* out, int n, float minval = 0.f, float maxval = 1.f) {
  for (int i = 0; i < n; ++i) {
    uint32_t x0 = 0, x1 = (uint32_t)i;
    pi05::threefry2x32(k.k0, k.k1, x0, x1);
    float f = jr_bits_to_unit(x0 ^ x1) * (maxval - minval) + minval;
    out[i] = f < minval ? minval : f;
  }
}
inline float jr_uniform1(const JaxKey& k, float minval = 0.f, float maxval = 1.f) {
  float f;
  jr_uniform(k, &f, 1, minval, maxval);
  return f;
}
inline float jr_normal1(const JaxKey& k) {
  float f;
  pi05::jax_normal(k, &f, 1);
  return f;
}
// jax.random._gamma_one(key, alpha, log_space=True)
inline float jr_loggamma_one(JaxKey key, float alpha) {
  const float one = 1.f, zero = 0.f, third = 1.f / 3.f, half = 0.5f, squeeze = 0.0331f;
  const bool boost = alpha >= one;
  const float alpha_orig = alpha;
  if (!boost) alpha = alpha + one;
  const float d = alpha - third;
  const float c = third / sqrtf(d);
  const JaxKey kk = jr_child(key, 0), subkey = jr_child(key, 1);
  key = kk;
  float X = zero, V = one, U = 2.f;
  auto cond = [&] {
    return (U >= one - squeeze * (X * X)) && (logf(U) >= X * half + d * ((one - V) + logf(V)));
  };
  while (cond()) {
    const JaxKey k0 = jr_child(key, 0), xk = jr_child(key, 1), uk = jr_child(key, 2);
    key = k0;
    JaxKey ik = xk;
    float x = zero, v = -one;
    while (v <= zero) {
      const JaxKey a = jr_child(ik, 0), s = jr_child(ik, 1);
      ik = a;
      x = jr_normal1(s);
      v = one + x * c;
    }
    X = x * x;
    V = (v * v) * v;
    U = jr_uniform1(uk);
  }
  const float log_samples = -(-log1pf(-jr_uniform1(subkey)));  // -exponential(subkey)
  const float log_boost = (boost || log_samples == 0.f) ? zero : log_samples * (one / alpha_orig);
  return (logf(d) + logf(V)) + log_boost;
}
// jax.random.beta(key, a, b, (n,))
inline void jr_beta(const JaxKey& key, float a, float b, float* out, int n) {
  const JaxKey ka = jr_child(key, 0), kb = jr_child(key, 1);
  for (int i = 0; i < n; ++i) {
    const float la = jr_loggamma_one(jr_child(ka, (uint32_t)i), a);
    const float lb = jr_loggamma_one(jr_child(kb, (uint32_t)i), b);
    const float m = la > lb ? la : lb;
    const float ea = expf(la - m), eb = expf(lb - m);
    out[i] = ea / (ea + eb);
  }
}

// openpi random draws of one training step (batch B, action chunk ah x ad)
struct StepRandom {
  JaxKey preprocess;
  std::vector<float> noise;  // [B][ah*ad]
  std::vector<float> time;   // [B]
};
inline StepRandom jr_step(const JaxKey& train_rng0, uint32_t step, int B, int chunk) {
  const JaxKey r = jr_fold_in(train_rng0, step);
  StepRandom s;
  s.preprocess = jr_child(r, 0);
  const JaxKey nk = jr_child(r, 1), tk = jr_child(r, 2);
  s.noise.resize((size_t)B * chunk);
  pi05::jax_normal(nk, s.noise.data(), B * chunk);
  s.time.resize(B);
  jr_beta(tk, 1.5f, 1.0f, s.time.data(), B);
  for (float& t : s.time) t = t * 0.999f + 0.001f;
  return s;
}

}  // namespace pi05t
