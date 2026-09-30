// torch CPU 난수를 손으로 (층 1·2 공용): 기본 생성기 mt19937(torch 판) + inductor 난수(Philox4_32) + T.random_quaternion.
// OmniGibson 이 판 도중 쓰는 곳: 다지기·익히기 입자 방향(macro_particle_system.py:281 T.random_quaternion, torch.compile),
// 뿌리기 표본(particle_modifier.py:1445 th.rand, eager) 등. 평가기는 시작에 한 번 seed_everything → 판 시작 상태를 추출(get_rng_state)해 이어 간다.
// 원본 (torch 2.7.0, site-packages/torch/include): ATen/core/MT19937RNGEngine.h, ATen/core/PhiloxRNGEngine.h,
//   ATen/core/DistributionsHelper.h:53 (randint: 범위 ≥ 2^32 이면 random64 % 범위 + 시작), ATen/CPUGeneratorImpl random64 = (첫 << 32) | 둘째,
//   _inductor/codegen/cpp_prefix.h:561 normalized_rand_cpu. 컴파일된 random_quaternion = eager randint(-2^63, 2^63-1) 로 시드 하나 →
//   원소 x 마다 Philox4_32(시드 하위 32비트, 0, x) 첫 출력 → (v & 0x7fffffff) * 4.6566127342e-10f, 그다음 std::sqrt·std::sin/cos(float) = glibc.
#pragma once
#include <cstdint>
#include <cstring>

#include "core/common/glibc_sincosf.h"
#include "core/particles/visual.h"

namespace eng {
namespace particles {

struct TorchMT {  // at::mt19937_data_pod
  uint64_t seed;
  int32_t left;
  uint32_t next;
  uint32_t state[624];
};

// th.get_rng_state() 바이트(5056) → 엔진 상태 (CPUGeneratorImplStateLegacy: seed u64, left i32, seeded i32, next u64, state u64[624], ...)
PEHD bool torch_mt_from_bytes(const uint8_t* b, int n, TorchMT& m) {
  if (n < 24 + 624 * 8) return false;
  memcpy(&m.seed, b, 8);
  memcpy(&m.left, b + 8, 4);
  uint64_t nx;
  memcpy(&nx, b + 16, 8);
  m.next = (uint32_t)nx;
  for (int i = 0; i < 624; ++i) {
    uint64_t v;
    memcpy(&v, b + 24 + 8 * i, 8);
    m.state[i] = (uint32_t)v;
  }
  return true;
}
PEHD void torch_mt_seed(TorchMT& m, uint64_t seed) {
  m.seed = seed;
  m.state[0] = (uint32_t)(seed & 0xffffffffu);
  for (uint32_t j = 1; j < 624; ++j) m.state[j] = 1812433253u * (m.state[j - 1] ^ (m.state[j - 1] >> 30)) + j;
  m.left = 1;
  m.next = 0;
}
PEHD void torch_mt_next_state(TorchMT& m) {
  const uint32_t UM = 0x80000000u, LM = 0x7fffffffu, MA = 0x9908b0dfu;
  auto twist = [&](uint32_t u, uint32_t v) { return (((u & UM) | (v & LM)) >> 1) ^ ((v & 1u) ? MA : 0u); };
  uint32_t* p = m.state;
  m.left = 624;
  m.next = 0;
  for (int j = 624 - 397 + 1; --j; p++) *p = p[397] ^ twist(p[0], p[1]);
  for (int j = 397; --j; p++) *p = p[397 - 624] ^ twist(p[0], p[1]);
  *p = p[397 - 624] ^ twist(p[0], m.state[0]);
}
PEHD uint32_t torch_mt_u32(TorchMT& m) {
  if (--m.left == 0) torch_mt_next_state(m);
  uint32_t y = m.state[m.next++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= (y >> 18);
  return y;
}
PEHD uint64_t torch_mt_u64(TorchMT& m) {
  const uint64_t a = torch_mt_u32(m);
  const uint64_t b = torch_mt_u32(m);
  return (a << 32) | b;
}
// torch.randint(low, high) 한 원소 (int64, 범위 ≥ 2^32 경로)
PEHD int64_t torch_randint64(TorchMT& m, int64_t low, int64_t high) {
  const uint64_t range = (uint64_t)high - (uint64_t)low;
  return (int64_t)(torch_mt_u64(m) % range + (uint64_t)low);
}

// ---- Philox4_32 (첫 출력만) ----
PEHD uint32_t philox_first(uint64_t seed, uint64_t offset) {
  uint32_t c[4] = {(uint32_t)offset, (uint32_t)(offset >> 32), 0u, 0u};
  uint32_t k[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
  for (int r = 0; r < 10; ++r) {
    const uint64_t p0 = (uint64_t)0xD2511F53u * c[0], p1 = (uint64_t)0xCD9E8D57u * c[2];
    const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0, hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    const uint32_t n0 = hi1 ^ c[1] ^ k[0], n2 = hi0 ^ c[3] ^ k[1];
    c[0] = n0;
    c[1] = lo1;
    c[2] = n2;
    c[3] = lo0;
    if (r < 9) {
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
  }
  return c[0];
}
PEHD float inductor_rand(uint32_t seed, uint32_t offset) {
  return (float)(philox_first(seed, offset) & 0x7fffffffu) * 4.6566127342e-10f;
}

// T.random_quaternion(n) (torch.compile). out[n][4] (x, y, z, w). 기본 생성기에서 64비트 둘을 쓴다
PEHD void random_quaternion(TorchMT& m, int n, float* out) {
  const int64_t seed = torch_randint64(m, INT64_MIN, INT64_MAX);
  const uint32_t s32 = (uint32_t)seed;
  for (int i = 0; i < n; ++i) {
    const float r0 = inductor_rand(s32, (uint32_t)(4 * i)), r1 = inductor_rand(s32, (uint32_t)(4 * i + 1)),
                r2 = inductor_rand(s32, (uint32_t)(4 * i + 2));
    const float a = sqrt_rn(1.0f - r0), b = sqrt_rn(r0);
    const float t1 = r1 * 6.283185307179586f, t2 = r2 * 6.283185307179586f;
    out[4 * i] = a * eng::glibc::sinf(t1);
    out[4 * i + 1] = a * eng::glibc::cosf(t1);
    out[4 * i + 2] = b * eng::glibc::sinf(t2);
    out[4 * i + 3] = b * eng::glibc::cosf(t2);
  }
}

}  // namespace particles
}  // namespace eng
