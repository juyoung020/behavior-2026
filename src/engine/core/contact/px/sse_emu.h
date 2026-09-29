// SSE2 명령 흉내 (손으로 짬): PhysX aos(Vec3V/FloatV/BoolV) 코드가 쓰는 _mm_* 를 칸(lane)마다 같은 float 연산으로 옮긴 것.
// 목적: PhysX 5.6.1 리눅스 빌드(clang, SSE2 경로, __SSE4_2__ 없음)의 PxVecMathSSE.h / PxUnixSse2InlineAoS.h 를
//       이름만 em_* 로 바꿔 그대로 쓰면 호스트(층 1)와 GPU(층 2)에서 PhysX 와 비트까지 같은 결과가 나오게 하는 것.
// 규칙
//  - 칸 하나의 +,-,*,/,sqrt 는 IEEE float 한 번 (SSE 의 addps 등과 같음). FMA 축약 금지(-ffp-contract=off, CUDA -fmad=false).
//  - minps/maxps 는 "a<b ? a : b" / "a>b ? a : b" (NaN·±0 에서 두 번째 인자). comi* 는 NaN 이면 0 (clang 의 COMISS 해석).
//  - FTZ/DAZ: 호스트는 PhysX 처럼 simulate 구간을 MXCSR FTZ+DAZ 로 두고 돌린다(tests 의 FtzScope). GPU 는 -ftz=true.
//  - rcpps/rsqrtps(근사)는 CPU 제조사마다 값이 다르다. 호스트는 진짜 명령을 부르고, GPU 는 표(approx 표, 리드 common/approx.h 예정)로.
//    접촉 모듈 기준으로는 볼록-볼록 경로에서 안 쓰고, 척도 있는 볼록(M33Inverse)·삼각메시(FRsqrtFast)·상자-상자(V3RecipFast)에서 쓴다.
// 원본: physx/include/foundation/PxVecMathSSE.h, unix/sse2/PxUnixSse2InlineAoS.h (태그 107.3-omni-and-physx-5.6.1, BSD-3)
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

#if !defined(__CUDA_ARCH__) && (defined(__x86_64__) || defined(_M_X64))
#include <xmmintrin.h>
#define EM_HOST_X86 1
#endif

#if defined(__CUDACC__)
#define EHD __host__ __device__ __forceinline__
#define EHDI __host__ __device__ inline
#else
#define EHD inline __attribute__((always_inline))
#define EHDI inline
#endif

namespace eng {
namespace px {

struct alignas(16) em128 { float f[4]; };
struct alignas(16) em128i { int32_t i[4]; };

#define EM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

EHD uint32_t em_f2u(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u; std::memcpy(&u, &f, 4); return u;
#endif
}
EHD float em_u2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f; std::memcpy(&f, &u, 4); return f;
#endif
}
EHD float em_sqrt1(float x) {
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return std::sqrt(x);
#endif
}
EHD float em_div1(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

// ---- 근사 역수·역제곱근 (rcpps / rsqrtps). GPU 판은 표가 필요 -> 층 2 에서 이 경로를 쓰면 표를 붙인다.
#if defined(__CUDACC__)
__device__ const float* em_rcp_table_dev();   // (예정) 리드 approx.h
__device__ const float* em_rsqrt_table_dev();
#endif
EHD float em_rcp1(float x) {
#if defined(__CUDA_ARCH__)
  __trap();  // 아직 표 없음: 이 경로가 GPU 에서 불리면 멈춘다(조용히 다른 값을 내지 않게)
  return 0.0f;
#elif defined(EM_HOST_X86)
  return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
#else
  return 1.0f / x;
#endif
}
EHD float em_rsqrt1(float x) {
#if defined(__CUDA_ARCH__)
  __trap();
  return 0.0f;
#elif defined(EM_HOST_X86)
  return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
#else
  return 1.0f / std::sqrt(x);
#endif
}

// ---- 만들기·읽기·쓰기
EHD em128 em_setzero_ps() { return em128{{0.0f, 0.0f, 0.0f, 0.0f}}; }
EHD em128 em_set_ps(float e3, float e2, float e1, float e0) { return em128{{e0, e1, e2, e3}}; }
EHD em128 em_set_ss(float a) { return em128{{a, 0.0f, 0.0f, 0.0f}}; }
EHD em128 em_set1_ps(float a) { return em128{{a, a, a, a}}; }
EHD em128 em_load_ps(const float* p) { em128 r; std::memcpy(r.f, p, 16); return r; }
EHD em128 em_loadu_ps(const float* p) { em128 r; std::memcpy(r.f, p, 16); return r; }
EHD em128 em_load1_ps(const float* p) { float a; std::memcpy(&a, p, 4); return em128{{a, a, a, a}}; }
EHD void em_store_ps(float* p, const em128& a) { std::memcpy(p, a.f, 16); }
EHD void em_storeu_ps(float* p, const em128& a) { std::memcpy(p, a.f, 16); }
EHD void em_store_ss(float* p, const em128& a) { std::memcpy(p, &a.f[0], 4); }
EHD float em_cvtss_f32(const em128& a) { return a.f[0]; }

// ---- 칸별 산술
EHD em128 em_add_ps(const em128& a, const em128& b) { return em128{{a.f[0] + b.f[0], a.f[1] + b.f[1], a.f[2] + b.f[2], a.f[3] + b.f[3]}}; }
EHD em128 em_sub_ps(const em128& a, const em128& b) { return em128{{a.f[0] - b.f[0], a.f[1] - b.f[1], a.f[2] - b.f[2], a.f[3] - b.f[3]}}; }
EHD em128 em_mul_ps(const em128& a, const em128& b) { return em128{{a.f[0] * b.f[0], a.f[1] * b.f[1], a.f[2] * b.f[2], a.f[3] * b.f[3]}}; }
EHD em128 em_div_ps(const em128& a, const em128& b) {
  return em128{{em_div1(a.f[0], b.f[0]), em_div1(a.f[1], b.f[1]), em_div1(a.f[2], b.f[2]), em_div1(a.f[3], b.f[3])}};
}
EHD em128 em_sqrt_ps(const em128& a) { return em128{{em_sqrt1(a.f[0]), em_sqrt1(a.f[1]), em_sqrt1(a.f[2]), em_sqrt1(a.f[3])}}; }
EHD em128 em_rcp_ps(const em128& a) { return em128{{em_rcp1(a.f[0]), em_rcp1(a.f[1]), em_rcp1(a.f[2]), em_rcp1(a.f[3])}}; }
EHD em128 em_rsqrt_ps(const em128& a) { return em128{{em_rsqrt1(a.f[0]), em_rsqrt1(a.f[1]), em_rsqrt1(a.f[2]), em_rsqrt1(a.f[3])}}; }
EHD em128 em_rcp_ss(const em128& a) { return em128{{em_rcp1(a.f[0]), a.f[1], a.f[2], a.f[3]}}; }
EHD em128 em_add_ss(const em128& a, const em128& b) { return em128{{a.f[0] + b.f[0], a.f[1], a.f[2], a.f[3]}}; }
EHD em128 em_sub_ss(const em128& a, const em128& b) { return em128{{a.f[0] - b.f[0], a.f[1], a.f[2], a.f[3]}}; }
EHD em128 em_mul_ss(const em128& a, const em128& b) { return em128{{a.f[0] * b.f[0], a.f[1], a.f[2], a.f[3]}}; }
EHD float em_min1(float a, float b) { return a < b ? a : b; }  // MINPS: 둘 중 하나가 NaN 이거나 둘 다 0 이면 b
EHD float em_max1(float a, float b) { return a > b ? a : b; }
EHD em128 em_min_ps(const em128& a, const em128& b) { return em128{{em_min1(a.f[0], b.f[0]), em_min1(a.f[1], b.f[1]), em_min1(a.f[2], b.f[2]), em_min1(a.f[3], b.f[3])}}; }
EHD em128 em_max_ps(const em128& a, const em128& b) { return em128{{em_max1(a.f[0], b.f[0]), em_max1(a.f[1], b.f[1]), em_max1(a.f[2], b.f[2]), em_max1(a.f[3], b.f[3])}}; }

// ---- 비트 연산
#define EM_BITOP(name, expr)                                                          \
  EHD em128 name(const em128& a, const em128& b) {                                    \
    em128 r;                                                                          \
    for (int k = 0; k < 4; ++k) {                                                     \
      const uint32_t x = em_f2u(a.f[k]), y = em_f2u(b.f[k]);                          \
      r.f[k] = em_u2f(expr);                                                          \
    }                                                                                 \
    return r;                                                                         \
  }
EM_BITOP(em_and_ps, x & y)
EM_BITOP(em_or_ps, x | y)
EM_BITOP(em_xor_ps, x ^ y)
EM_BITOP(em_andnot_ps, (~x) & y)
#undef EM_BITOP

// ---- 비교 (참 = 0xFFFFFFFF). 순서 있는 비교: NaN 이면 거짓 (cmpneq 만 참)
EHD float em_mask(bool c) { return em_u2f(c ? 0xFFFFFFFFu : 0u); }
#define EM_CMP(name, op)                                                                                      \
  EHD em128 name(const em128& a, const em128& b) {                                                            \
    return em128{{em_mask(a.f[0] op b.f[0]), em_mask(a.f[1] op b.f[1]), em_mask(a.f[2] op b.f[2]), em_mask(a.f[3] op b.f[3])}}; \
  }
EM_CMP(em_cmpeq_ps, ==)
EM_CMP(em_cmpgt_ps, >)
EM_CMP(em_cmpge_ps, >=)
EM_CMP(em_cmplt_ps, <)
EM_CMP(em_cmple_ps, <=)
#undef EM_CMP
EHD em128 em_cmpneq_ps(const em128& a, const em128& b) {
  return em128{{em_mask(!(a.f[0] == b.f[0])), em_mask(!(a.f[1] == b.f[1])), em_mask(!(a.f[2] == b.f[2])), em_mask(!(a.f[3] == b.f[3]))}};
}
EHD int em_comieq_ss(const em128& a, const em128& b) { return a.f[0] == b.f[0] ? 1 : 0; }
EHD int em_comigt_ss(const em128& a, const em128& b) { return a.f[0] > b.f[0] ? 1 : 0; }
EHD int em_comige_ss(const em128& a, const em128& b) { return a.f[0] >= b.f[0] ? 1 : 0; }
EHD int em_comilt_ss(const em128& a, const em128& b) { return a.f[0] < b.f[0] ? 1 : 0; }
EHD int em_comile_ss(const em128& a, const em128& b) { return a.f[0] <= b.f[0] ? 1 : 0; }
EHD int em_movemask_ps(const em128& a) {
  return int((em_f2u(a.f[0]) >> 31) | ((em_f2u(a.f[1]) >> 31) << 1) | ((em_f2u(a.f[2]) >> 31) << 2) | ((em_f2u(a.f[3]) >> 31) << 3));
}

// ---- 섞기
EHD em128 em_shuffle_ps(const em128& a, const em128& b, int imm) {
  return em128{{a.f[imm & 3], a.f[(imm >> 2) & 3], b.f[(imm >> 4) & 3], b.f[(imm >> 6) & 3]}};
}
EHD em128 em_unpacklo_ps(const em128& a, const em128& b) { return em128{{a.f[0], b.f[0], a.f[1], b.f[1]}}; }
EHD em128 em_unpackhi_ps(const em128& a, const em128& b) { return em128{{a.f[2], b.f[2], a.f[3], b.f[3]}}; }
EHD em128 em_movelh_ps(const em128& a, const em128& b) { return em128{{a.f[0], a.f[1], b.f[0], b.f[1]}}; }
EHD em128 em_movehl_ps(const em128& a, const em128& b) { return em128{{b.f[2], b.f[3], a.f[2], a.f[3]}}; }
EHD em128 em_move_ss(const em128& a, const em128& b) { return em128{{b.f[0], a.f[1], a.f[2], a.f[3]}}; }

// ---- 정수
EHD em128i em_castps_si128(const em128& a) {
  return em128i{{int32_t(em_f2u(a.f[0])), int32_t(em_f2u(a.f[1])), int32_t(em_f2u(a.f[2])), int32_t(em_f2u(a.f[3]))}};
}
EHD em128 em_castsi128_ps(const em128i& a) {
  return em128{{em_u2f(uint32_t(a.i[0])), em_u2f(uint32_t(a.i[1])), em_u2f(uint32_t(a.i[2])), em_u2f(uint32_t(a.i[3]))}};
}
EHD em128i em_setzero_si128() { return em128i{{0, 0, 0, 0}}; }
EHD em128i em_set_epi32(int32_t e3, int32_t e2, int32_t e1, int32_t e0) { return em128i{{e0, e1, e2, e3}}; }
EHD em128i em_add_epi32(const em128i& a, const em128i& b) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = int32_t(uint32_t(a.i[k]) + uint32_t(b.i[k])); return r;
}
EHD em128i em_sub_epi32(const em128i& a, const em128i& b) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = int32_t(uint32_t(a.i[k]) - uint32_t(b.i[k])); return r;
}
EHD em128i em_cmpeq_epi32(const em128i& a, const em128i& b) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] == b.i[k] ? -1 : 0; return r;
}
EHD em128i em_cmpgt_epi32(const em128i& a, const em128i& b) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] > b.i[k] ? -1 : 0; return r;
}
EHD em128i em_cmpgt_epi16(const em128i& a, const em128i& b) {
  em128i r;
  for (int k = 0; k < 4; ++k) {
    const int16_t al = int16_t(uint32_t(a.i[k]) & 0xFFFF), ah = int16_t(uint32_t(a.i[k]) >> 16);
    const int16_t bl = int16_t(uint32_t(b.i[k]) & 0xFFFF), bh = int16_t(uint32_t(b.i[k]) >> 16);
    r.i[k] = int32_t((al > bl ? 0xFFFFu : 0u) | (ah > bh ? 0xFFFF0000u : 0u));
  }
  return r;
}
EHD em128i em_and_si128(const em128i& a, const em128i& b) { em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] & b.i[k]; return r; }
EHD em128i em_or_si128(const em128i& a, const em128i& b) { em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] | b.i[k]; return r; }
EHD em128i em_xor_si128(const em128i& a, const em128i& b) { em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] ^ b.i[k]; return r; }
EHD em128i em_andnot_si128(const em128i& a, const em128i& b) { em128i r; for (int k = 0; k < 4; ++k) r.i[k] = (~a.i[k]) & b.i[k]; return r; }
EHD em128i em_slli_epi32(const em128i& a, int c) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = c > 31 ? 0 : int32_t(uint32_t(a.i[k]) << c); return r;
}
EHD em128i em_srli_epi32(const em128i& a, int c) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = c > 31 ? 0 : int32_t(uint32_t(a.i[k]) >> c); return r;
}
EHD em128i em_srai_epi32(const em128i& a, int c) {
  em128i r; for (int k = 0; k < 4; ++k) r.i[k] = a.i[k] >> (c > 31 ? 31 : c); return r;
}
EHD em128i em_sll_epi32(const em128i& a, const em128i& c) {  // 개수 = 아래 64비트
  const uint64_t n = uint64_t(uint32_t(c.i[0])) | (uint64_t(uint32_t(c.i[1])) << 32);
  return em_slli_epi32(a, n > 31 ? 32 : int(n));
}
EHD em128i em_srl_epi32(const em128i& a, const em128i& c) {
  const uint64_t n = uint64_t(uint32_t(c.i[0])) | (uint64_t(uint32_t(c.i[1])) << 32);
  return em_srli_epi32(a, n > 31 ? 32 : int(n));
}
EHD em128 em_cvtepi32_ps(const em128i& a) {
#if defined(__CUDA_ARCH__)
  return em128{{__int2float_rn(a.i[0]), __int2float_rn(a.i[1]), __int2float_rn(a.i[2]), __int2float_rn(a.i[3])}};
#else
  return em128{{float(a.i[0]), float(a.i[1]), float(a.i[2]), float(a.i[3])}};
#endif
}
EHD int32_t em_cvt1(float x, bool trunc) {  // CVTPS2DQ / CVTTPS2DQ: 범위 밖·NaN 은 0x80000000
  if (!(x >= -2147483648.0f && x < 2147483648.0f)) return int32_t(0x80000000u);
#if defined(__CUDA_ARCH__)
  return trunc ? __float2int_rz(x) : __float2int_rn(x);
#else
  return trunc ? int32_t(x) : int32_t(std::nearbyint(x));  // 기본 MXCSR 반올림 = 짝수 쪽 가까운 값
#endif
}
EHD em128i em_cvtps_epi32(const em128& a) { return em128i{{em_cvt1(a.f[0], false), em_cvt1(a.f[1], false), em_cvt1(a.f[2], false), em_cvt1(a.f[3], false)}}; }
EHD em128i em_cvttps_epi32(const em128& a) { return em128i{{em_cvt1(a.f[0], true), em_cvt1(a.f[1], true), em_cvt1(a.f[2], true), em_cvt1(a.f[3], true)}}; }

}  // namespace px
}  // namespace eng
