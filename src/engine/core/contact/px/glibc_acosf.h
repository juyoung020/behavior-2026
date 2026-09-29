// glibc 2.35 acosf 이식 (손으로 옮김, 호스트·GPU 공용). PhysX PxAcos(=::acosf) 와 비트 동일하게.
// 원본: glibc-2.35 sysdeps/ieee754/flt-32/e_acosf.c (Sun fdlibm 계열, float 연산만, x86_64 에 FMA 변종 없음).
// 검증: tests/contact/test_acosf_exhaustive.cpp 가 float 2^32 개 전부를 호스트 libm acosf 와 비트 비교 (FTZ/DAZ 켬·끔 둘 다).
// 쓰는 곳: PCM 다양체 무효화 판정 (GuPersistentContactManifold.h invalidate_BoxConvex, 회전량 -> 호 길이).
#pragma once
#include <cstdint>

#include "core/contact/px/sse_emu.h"

namespace eng {
namespace glibcx {

EHD float acosf(float x) {
  const float one = 1.0000000000e+00f, pi = 3.1415925026e+00f, pio2_hi = 1.5707962513e+00f, pio2_lo = 7.5497894159e-08f,
              pS0 = 1.6666667163e-01f, pS1 = -3.2556581497e-01f, pS2 = 2.0121252537e-01f, pS3 = -4.0055535734e-02f,
              pS4 = 7.9153501429e-04f, pS5 = 3.4793309169e-05f, qS1 = -2.4033949375e+00f, qS2 = 2.0209457874e+00f,
              qS3 = -6.8828397989e-01f, qS4 = 7.7038154006e-02f;
  float z, p, q, r, w, s, c, df;
  const int32_t hx = int32_t(px::em_f2u(x));
  const int32_t ix = hx & 0x7fffffff;
  if (ix == 0x3f800000) {  // |x|==1
    if (hx > 0) return 0.0f;
    else return pi + (float)2.0 * pio2_lo;
  } else if (ix > 0x3f800000) {  // |x| > 1 -> NaN
    // 실측(Ubuntu glibc 2.35 libm): NaN 입력은 조용한 NaN 으로 그대로(부호·꼬리 유지), 그 밖(|x|>1, ±inf)은 +qNaN 0x7fc00000
    // (원본 식 (x-x)/(x-x) 는 0xffc00000 를 내므로 쓰지 않는다 — 공개 acosf 는 오류 처리 감싸개를 거쳐 +NaN 을 돌려줌).
    // GPU 는 NaN 연산이 꼬리를 버리므로 식 대신 비트로 만든다.
    if (ix > 0x7f800000) return px::em_u2f(uint32_t(hx) | 0x00400000u);
    return px::em_u2f(0x7fc00000u);
  }
  if (ix < 0x3f000000) {  // |x| < 0.5
    if (ix <= 0x32800000) return pio2_hi + pio2_lo;
    z = x * x;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = px::em_div1(p, q);
    return pio2_hi - (x - (pio2_lo - x * r));
  } else if (hx < 0) {  // x < -0.5
    z = (one + x) * (float)0.5;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    s = px::em_sqrt1(z);
    r = px::em_div1(p, q);
    w = r * s - pio2_lo;
    return pi - (float)2.0 * (s + w);
  } else {  // x > 0.5
    z = (one - x) * (float)0.5;
    s = px::em_sqrt1(z);
    df = px::em_u2f(px::em_f2u(s) & 0xfffff000u);
    c = px::em_div1(z - df * df, s + df);
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = px::em_div1(p, q);
    w = r * s + c;
    return (float)2.0 * (df + w);
  }
}

}  // namespace glibcx
}  // namespace eng
