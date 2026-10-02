// SSE/SSE2 명령 흉내 (호스트·CUDA 공용). PhysX aos(Vec3V/Vec4V/FloatV …) 코드를 그대로 옮기려고, PhysX 가 쓰는
// _mm_* 명령을 같은 의미로 구현한다 (core/common/aos.h 가 이것 위에서 PhysX 원본 식을 그대로 쓴다).
// 규칙
//  - 칸(lane)마다 IEEE float 연산 한 번 = SSE 한 번. FMA 없음(nvcc -fmad=false, 호스트 -ffp-contract=off).
//  - FTZ/DAZ 는 호출하는 쪽 환경을 따른다 (호스트: MXCSR, GPU: -ftz=true). PhysX 는 simulate 안에서 FTZ+DAZ.
//  - max/min 은 SSE 뜻 그대로: max(a,b) = a > b ? a : b (NaN·±0 이면 둘째 값), min 도 같은 식.
//  - rcpps / rsqrtps 는 근사 명령이라 CPU 마다 다르다 -> approx 표 흉내 (기본: AMD Zen 5 = 이 PC, 대회 측정 장비 Zen 4 와 같은지는 미확인).
//    표는 scripts/rcpgen.cpp 로 뽑았고, float 2^32 개 전부(DAZ 켬/끔) 하드웨어와 비트 동일 확인.
//  - comi*_ss 는 clang 판 의미(순서 있는 비교, NaN 이면 0). 공식 리눅스 omni.physx 는 clang 빌드.
#pragma once
#include <cstdint>
#include <cstring>

#include "pmath.h"

namespace eng {
namespace sse {

struct alignas(16) m128 { float f[4]; };
struct alignas(16) m128i { int32_t i[4]; };

EHD uint32_t fb(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
EHD float bf(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

// ---------------------------------------------------------------- 근사 역수·역제곱근 (표)
#if defined(__CUDACC__)
#define ENG_APPROX_TABLE(name) __device__ __constant__ static const uint32_t name##_d[4096]
#include "approx_tables_zen5.inc"
#undef ENG_APPROX_TABLE
#endif
#define ENG_APPROX_TABLE(name) static const uint32_t name##_h[4096]
#include "approx_tables_zen5.inc"
#undef ENG_APPROX_TABLE

EHD uint32_t rcp_tab(uint32_t i) {
#if defined(__CUDA_ARCH__)
  return kRcpTable_d[i];
#else
  return kRcpTable_h[i];
#endif
}
EHD uint32_t rsq_tab(int odd, uint32_t i) {
#if defined(__CUDA_ARCH__)
  return odd ? kRsqTable1_d[i] : kRsqTable0_d[i];
#else
  return odd ? kRsqTable1_h[i] : kRsqTable0_h[i];
#endif
}
EHD float rcp_approx(float x) {  // rcpss/rcpps 한 칸 (rcpgen.cpp 의 rcp_emu 와 같음)
  const uint32_t u = fb(x), s = u & 0x80000000u, e = (u >> 23) & 0xff, m = u & 0x7fffff;
  if (e == 0xff) return m ? bf(u | 0x400000u) : bf(s);
  if (e == 0) return bf(s | 0x7f800000u);
  const uint32_t r = rcp_tab(m >> 11);
  const int re = int((r >> 23) & 0xff) - (int(e) - 127);
  if (re >= 0xff) return bf(s | 0x7f800000u);
  if (re <= 0) return bf(s);
  return bf(s | (uint32_t(re) << 23) | (r & 0x7fffff));
}
EHD float rsqrt_approx(float x) {  // rsqrtss/rsqrtps 한 칸
  const uint32_t u = fb(x), s = u & 0x80000000u, e = (u >> 23) & 0xff, m = u & 0x7fffff;
  if (e == 0xff) { if (m) return bf(u | 0x400000u); return s ? bf(0xffc00000u) : bf(0); }
  if (e == 0) return bf(s | 0x7f800000u);
  if (s) return bf(0xffc00000u);
  const int E = int(e) - 127;
  const int odd = E & 1;
  const uint32_t r = rsq_tab(odd, m >> 11);
  const int k = odd ? (E - 1) / 2 : E / 2;
  const int re = int((r >> 23) & 0xff) - k;
  return bf((uint32_t(re) << 23) | (r & 0x7fffff));
}

// ---------------------------------------------------------------- 불러오기·쓰기·만들기
EHD m128 _mm_setzero_ps() { return m128{{0.0f, 0.0f, 0.0f, 0.0f}}; }
EHD m128 _mm_set_ps(float e3, float e2, float e1, float e0) { return m128{{e0, e1, e2, e3}}; }
EHD m128 _mm_set1_ps(float a) { return m128{{a, a, a, a}}; }
EHD m128 _mm_load1_ps(const float* p) { return _mm_set1_ps(*p); }
EHD m128 _mm_load_ps(const float* p) { return m128{{p[0], p[1], p[2], p[3]}}; }
EHD m128 _mm_loadu_ps(const float* p) { return _mm_load_ps(p); }
EHD m128 _mm_load_ss(const float* p) { return m128{{p[0], 0.0f, 0.0f, 0.0f}}; }
EHD void _mm_store_ps(float* p, m128 a) { p[0] = a.f[0]; p[1] = a.f[1]; p[2] = a.f[2]; p[3] = a.f[3]; }
EHD void _mm_storeu_ps(float* p, m128 a) { _mm_store_ps(p, a); }
EHD void _mm_store_ss(float* p, m128 a) { p[0] = a.f[0]; }
EHD float _mm_cvtss_f32(m128 a) { return a.f[0]; }
EHD m128i _mm_setzero_si128() { return m128i{{0, 0, 0, 0}}; }
EHD m128i _mm_set_epi32(int e3, int e2, int e1, int e0) { return m128i{{e0, e1, e2, e3}}; }
EHD m128i _mm_set1_epi32(int a) { return m128i{{a, a, a, a}}; }
EHD m128i _mm_load_si128(const m128i* p) { return *p; }
EHD void _mm_store_si128(m128i* p, m128i a) { *p = a; }
EHD m128 _mm_castsi128_ps(m128i a) { m128 r; memcpy(&r, &a, 16); return r; }
EHD m128i _mm_castps_si128(m128 a) { m128i r; memcpy(&r, &a, 16); return r; }

// ---------------------------------------------------------------- 칸 산술
#define ENG_LANE4(op) m128 r; for (int k = 0; k < 4; ++k) r.f[k] = (op); return r
EHD m128 _mm_add_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] + b.f[k]); }
EHD m128 _mm_sub_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] - b.f[k]); }
EHD m128 _mm_mul_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] * b.f[k]); }
EHD m128 _mm_div_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] / b.f[k]); }
EHD m128 _mm_sqrt_ps(m128 a) { ENG_LANE4(psqrt(a.f[k])); }
EHD m128 _mm_max_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] > b.f[k] ? a.f[k] : b.f[k]); }
EHD m128 _mm_min_ps(m128 a, m128 b) { ENG_LANE4(a.f[k] < b.f[k] ? a.f[k] : b.f[k]); }
EHD m128 _mm_rcp_ps(m128 a) { ENG_LANE4(rcp_approx(a.f[k])); }
EHD m128 _mm_rsqrt_ps(m128 a) { ENG_LANE4(rsqrt_approx(a.f[k])); }
EHD m128 _mm_add_ss(m128 a, m128 b) { a.f[0] = a.f[0] + b.f[0]; return a; }
EHD m128 _mm_sub_ss(m128 a, m128 b) { a.f[0] = a.f[0] - b.f[0]; return a; }
EHD m128 _mm_mul_ss(m128 a, m128 b) { a.f[0] = a.f[0] * b.f[0]; return a; }
EHD m128 _mm_div_ss(m128 a, m128 b) { a.f[0] = a.f[0] / b.f[0]; return a; }
EHD m128 _mm_sqrt_ss(m128 a) { a.f[0] = psqrt(a.f[0]); return a; }
EHD m128 _mm_rcp_ss(m128 a) { a.f[0] = rcp_approx(a.f[0]); return a; }
EHD m128 _mm_rsqrt_ss(m128 a) { a.f[0] = rsqrt_approx(a.f[0]); return a; }
EHD m128 _mm_max_ss(m128 a, m128 b) { a.f[0] = a.f[0] > b.f[0] ? a.f[0] : b.f[0]; return a; }
EHD m128 _mm_min_ss(m128 a, m128 b) { a.f[0] = a.f[0] < b.f[0] ? a.f[0] : b.f[0]; return a; }

// ---------------------------------------------------------------- 비트 논리 (float 칸을 비트로)
#define ENG_BITS4(op) m128 r; for (int k = 0; k < 4; ++k) { const uint32_t x = fb(a.f[k]), y = fb(b.f[k]); r.f[k] = bf(op); } return r
EHD m128 _mm_and_ps(m128 a, m128 b) { ENG_BITS4(x & y); }
EHD m128 _mm_andnot_ps(m128 a, m128 b) { ENG_BITS4(~x & y); }
EHD m128 _mm_or_ps(m128 a, m128 b) { ENG_BITS4(x | y); }
EHD m128 _mm_xor_ps(m128 a, m128 b) { ENG_BITS4(x ^ y); }

// ---------------------------------------------------------------- 비교 (참 = 모든 비트 1)
#define ENG_CMP4(c) m128 r; for (int k = 0; k < 4; ++k) r.f[k] = bf((c) ? 0xffffffffu : 0u); return r
EHD m128 _mm_cmpeq_ps(m128 a, m128 b) { ENG_CMP4(a.f[k] == b.f[k]); }
EHD m128 _mm_cmpgt_ps(m128 a, m128 b) { ENG_CMP4(a.f[k] > b.f[k]); }
EHD m128 _mm_cmpngt_ps(m128 a, m128 b) { ENG_CMP4(!(a.f[k] > b.f[k])); }  // cmpnltps 계열: NaN 이면 참 (SSE CMPNLEPS 인자 바꿈)
EHD m128 _mm_cmpge_ps(m128 a, m128 b) { ENG_CMP4(a.f[k] >= b.f[k]); }
EHD m128 _mm_cmplt_ps(m128 a, m128 b) { ENG_CMP4(a.f[k] < b.f[k]); }
EHD m128 _mm_cmple_ps(m128 a, m128 b) { ENG_CMP4(a.f[k] <= b.f[k]); }
EHD m128 _mm_cmpneq_ps(m128 a, m128 b) { ENG_CMP4(!(a.f[k] == b.f[k])); }
EHD int _mm_comieq_ss(m128 a, m128 b) { return a.f[0] == b.f[0]; }
EHD int _mm_comigt_ss(m128 a, m128 b) { return a.f[0] > b.f[0]; }
EHD int _mm_comige_ss(m128 a, m128 b) { return a.f[0] >= b.f[0]; }
EHD int _mm_comilt_ss(m128 a, m128 b) { return a.f[0] < b.f[0]; }
EHD int _mm_comile_ss(m128 a, m128 b) { return a.f[0] <= b.f[0]; }
EHD int _mm_comineq_ss(m128 a, m128 b) { return !(a.f[0] == b.f[0]); }
EHD int _mm_movemask_ps(m128 a) {
  int r = 0;
  for (int k = 0; k < 4; ++k) r |= int(fb(a.f[k]) >> 31) << k;
  return r;
}

// ---------------------------------------------------------------- 섞기
#ifndef _MM_SHUFFLE
#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif
#define ENG_MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
EHD m128 _mm_shuffle_ps(m128 a, m128 b, int imm) {
  return m128{{a.f[imm & 3], a.f[(imm >> 2) & 3], b.f[(imm >> 4) & 3], b.f[(imm >> 6) & 3]}};
}
EHD m128 _mm_unpacklo_ps(m128 a, m128 b) { return m128{{a.f[0], b.f[0], a.f[1], b.f[1]}}; }
EHD m128 _mm_unpackhi_ps(m128 a, m128 b) { return m128{{a.f[2], b.f[2], a.f[3], b.f[3]}}; }
EHD m128 _mm_movehl_ps(m128 a, m128 b) { return m128{{b.f[2], b.f[3], a.f[2], a.f[3]}}; }
EHD m128 _mm_movelh_ps(m128 a, m128 b) { return m128{{a.f[0], a.f[1], b.f[0], b.f[1]}}; }
EHD m128 _mm_move_ss(m128 a, m128 b) { a.f[0] = b.f[0]; return a; }

// ---------------------------------------------------------------- 정수 칸 (SSE2)
#define ENG_ILANE4(op) m128i r; for (int k = 0; k < 4; ++k) r.i[k] = (op); return r
EHD m128i _mm_add_epi32(m128i a, m128i b) { ENG_ILANE4(int32_t(uint32_t(a.i[k]) + uint32_t(b.i[k]))); }
EHD m128i _mm_sub_epi32(m128i a, m128i b) { ENG_ILANE4(int32_t(uint32_t(a.i[k]) - uint32_t(b.i[k]))); }
EHD m128i _mm_and_si128(m128i a, m128i b) { ENG_ILANE4(a.i[k] & b.i[k]); }
EHD m128i _mm_andnot_si128(m128i a, m128i b) { ENG_ILANE4(~a.i[k] & b.i[k]); }
EHD m128i _mm_or_si128(m128i a, m128i b) { ENG_ILANE4(a.i[k] | b.i[k]); }
EHD m128i _mm_xor_si128(m128i a, m128i b) { ENG_ILANE4(a.i[k] ^ b.i[k]); }
EHD m128i _mm_cmpeq_epi32(m128i a, m128i b) { ENG_ILANE4(a.i[k] == b.i[k] ? -1 : 0); }
EHD m128i _mm_cmpgt_epi32(m128i a, m128i b) { ENG_ILANE4(a.i[k] > b.i[k] ? -1 : 0); }
EHD m128i _mm_cmplt_epi32(m128i a, m128i b) { ENG_ILANE4(a.i[k] < b.i[k] ? -1 : 0); }
EHD m128i _mm_slli_epi32(m128i a, int n) { ENG_ILANE4(n > 31 ? 0 : int32_t(uint32_t(a.i[k]) << n)); }
EHD m128i _mm_srli_epi32(m128i a, int n) { ENG_ILANE4(n > 31 ? 0 : int32_t(uint32_t(a.i[k]) >> n)); }
EHD m128i _mm_srai_epi32(m128i a, int n) { ENG_ILANE4(a.i[k] >> (n > 31 ? 31 : n)); }
EHD m128i _mm_sll_epi32(m128i a, m128i c) {  // 이동 수 = c 의 아래 64 비트
  const uint64_t n = uint64_t(uint32_t(c.i[0])) | (uint64_t(uint32_t(c.i[1])) << 32);
  ENG_ILANE4(n > 31 ? 0 : int32_t(uint32_t(a.i[k]) << n));
}
EHD m128i _mm_srl_epi32(m128i a, m128i c) {
  const uint64_t n = uint64_t(uint32_t(c.i[0])) | (uint64_t(uint32_t(c.i[1])) << 32);
  ENG_ILANE4(n > 31 ? 0 : int32_t(uint32_t(a.i[k]) >> n));
}
EHD m128i _mm_cmpgt_epi16(m128i a, m128i b) {  // 16 비트 칸 8 개
  m128i r;
  int16_t x[8], y[8], z[8];
  memcpy(x, &a, 16);
  memcpy(y, &b, 16);
  for (int k = 0; k < 8; ++k) z[k] = x[k] > y[k] ? int16_t(-1) : int16_t(0);
  memcpy(&r, z, 16);
  return r;
}

// ---------------------------------------------------------------- 변환 (MXCSR 반올림 = 가장 가까운 짝수, PhysX 기본)
EHD int32_t cvt_rne(float x) {  // cvtps2dq: 범위 밖·NaN -> 0x80000000
  if (!(x >= -2147483648.0f && x < 2147483648.0f)) return int32_t(0x80000000u);
#if defined(__CUDA_ARCH__)
  return __float2int_rn(x);
#else
  const float r = std::nearbyint(x);  // 기본 반올림 모드(가장 가까운 짝수)
  return int32_t(r);
#endif
}
EHD int32_t cvt_trunc(float x) {  // cvttps2dq
  if (!(x >= -2147483648.0f && x < 2147483648.0f)) return int32_t(0x80000000u);
  return int32_t(x);
}
EHD m128i _mm_cvtps_epi32(m128 a) { ENG_ILANE4(cvt_rne(a.f[k])); }
EHD m128i _mm_cvttps_epi32(m128 a) { ENG_ILANE4(cvt_trunc(a.f[k])); }
EHD m128 _mm_cvtepi32_ps(m128i a) { ENG_LANE4(float(a.i[k])); }

#undef ENG_LANE4
#undef ENG_BITS4
#undef ENG_CMP4
#undef ENG_ILANE4

}  // namespace sse
}  // namespace eng
