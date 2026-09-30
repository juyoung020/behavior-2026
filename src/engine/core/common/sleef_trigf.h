// torch CPU 의 float32 sin/cos = SLEEF u10 벡터 함수 (Sleef_sinf16_u10 / cosf16_u10, at::vec, FMA 판) 를 스칼라로 옮김.
// 주의(09-30 확인): torch eager 의 th.cos/th.sin 은 이것이 아니라 MKL VML(core/omni/mkl_trig.h) 이다. 이 파일은 SLEEF 자체와 비트 같음(200 만 개 다름 0)만 확인.
// 원본: sleef 3.6 src/libm/sleefsimdsp.c xsinf_u1:969 / xcosf_u1:1067 (DETERMINISTIC 아님), src/common/df.h (ENABLE_FMA_SP 판),
//       misc.h PI_A2f·PI_B2f·PI_C2f·TRIGRANGEMAX2f. 쓰는 곳: 평가기 proprio 의 바닥 속도 회전(robot.py:1605 th.cos/th.sin, 0 차원 텐서도 같은 경로).
// |x| >= 125 (TRIGRANGEMAX2f) 는 rempif 경로 — 옮기지 않았다(바닥 yaw 는 ±π). 그 범위면 NaN 을 돌려 드러나게 한다.
// 확인: tests/common/test_sleef_trigf.py (torch 와 전수 비교).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SLHD __host__ __device__ inline
#else
#define SLHD inline
#endif

namespace eng {
namespace sleef {

struct f2 {
  float x, y;
};
SLHD float fma_(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return std::fmaf(a, b, c);
#endif
}
SLHD float rint_(float x) {  // 가장 가까운 짝수 (vrint, _mm512_roundscale 기본 모드)
#if defined(__CUDA_ARCH__)
  return rintf(x);
#else
  return std::nearbyintf(x);
#endif
}
SLHD f2 dfadd2_ff(float x, float y) {
  const float s = x + y, v = s - x;
  return {s, (x - (s - v)) + (y - v)};
}
SLHD f2 dfadd2_f2f(f2 x, float y) {
  const float s = x.x + y, v = s - x.x;
  const float t = (x.x - (s - v)) + (y - v);
  return {s, t + x.y};
}
SLHD f2 dfadd_f2f(f2 x, float y) {
  const float s = x.x + y;
  return {s, ((x.x - s) + y) + x.y};
}
SLHD f2 dfadd_ff(float x, float y) {
  const float s = x + y;
  return {s, (x - s) + y};
}
SLHD f2 dfadd_ff2(float x, f2 y) {
  const float s = x + y.x;
  return {s, ((x - s) + y.x) + y.y};
}
SLHD f2 dfsqu(f2 x) {
  const float s = x.x * x.x;
  return {s, fma_(x.x + x.x, x.y, fma_(x.x, x.x, -s))};
}
SLHD f2 dfmul_f2f2(f2 x, f2 y) {
  const float s = x.x * y.x;
  return {s, fma_(x.x, y.y, fma_(x.y, y.x, fma_(x.x, y.x, -s)))};
}
SLHD float dfmul_f(f2 x, f2 y) { return fma_(x.x, y.x, fma_(x.y, y.x, x.x * y.y)); }

constexpr float PI_A2f = 3.1414794921875f, PI_B2f = 0.00011315941810607910156f, PI_C2f = 1.9841872589410058936e-09f;
constexpr float TRIGRANGEMAX2f = 125.0f;

SLHD float poly_tail(f2 s, f2 t) {  // 공통 뒷부분: t × (1 + s·(-1/6 + s·u))
  const f2 t0 = t;
  const f2 sq = dfsqu(s);
  float u = 2.6083159809786593541503e-06f;
  u = fma_(u, sq.x, -0.0001981069071916863322258f);
  u = fma_(u, sq.x, 0.00833307858556509017944336f);
  const f2 x = dfadd_ff2(1.0f, dfmul_f2f2(dfadd_ff(-0.166666597127914428710938f, u * sq.x), sq));
  return dfmul_f(t0, x);
}

SLHD float sinf_u10(float d) {
  if (!(std::fabs(d) < TRIGRANGEMAX2f)) return NAN;
  const float u = rint_(d * float(M_1_PI));
  const int q = int(rint_(u));
  const float v = fma_(u, -PI_A2f, d);
  f2 s = dfadd2_ff(v, u * -PI_B2f);
  s = dfadd_f2f(s, u * -PI_C2f);
  float r = poly_tail(s, s);
  if ((q & 1) == 1) {
    uint32_t b;
    std::memcpy(&b, &r, 4);
    b ^= 0x80000000u;
    std::memcpy(&r, &b, 4);
  }
  if (d == 0.0f && std::signbit(d)) r = d;  // visnegzero
  return r;
}

SLHD float cosf_u10(float d) {
  if (!(std::fabs(d) < TRIGRANGEMAX2f)) return NAN;
  const float dq = fma_(rint_(fma_(d, float(M_1_PI), -0.5f)), 2.0f, 1.0f);
  const int q = int(rint_(dq));
  f2 s = dfadd2_ff(d, dq * (-PI_A2f * 0.5f));
  s = dfadd2_f2f(s, dq * (-PI_B2f * 0.5f));
  s = dfadd2_f2f(s, dq * (-PI_C2f * 0.5f));
  float r = poly_tail(s, s);
  if ((q & 2) == 0) {
    uint32_t b;
    std::memcpy(&b, &r, 4);
    b ^= 0x80000000u;
    std::memcpy(&r, &b, 4);
  }
  return r;
}

}  // namespace sleef
}  // namespace eng
