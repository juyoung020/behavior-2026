// joints 모듈(임시, 리드가 core/common 으로 올릴 것): glibc 2.35 tanf 를 그대로 옮긴 것 (CPU·CUDA 공용).
// PhysX 는 PxTan -> ::tanf (physx/include/foundation/PxMath.h:230). joints 에서는 원뿔 한계 ConeLimitHelperTanLess
// (source/common/src/CmConeLimitHelper.h:163) 에서만 쓴다. GPU 에서 CUDA tanf 는 결과가 달라 이식이 필요했다.
// 원본: glibc-2.35 sysdeps/ieee754/flt-32/s_tanf.c (rem_pio2f = s_sincosf.h 의 reduce_fast/reduce_large) + k_tanf.c.
// FMA 판 없음: x86_64 multiarch Makefile 의 s_tan-fma 는 double tan 뿐 -> 기본 빌드(SSE2) 식 그대로. 그래서 sinf/cosf(FMA 판)의
// reduce_fast 를 쓰지 않고 곱·뺄셈 두 번으로 따로 둔다(reduce_large 는 정수 연산이라 공용판을 그대로 쓴다).
// NaN/Inf: 원본 x - x. SSE 결과를 비트로 만든다(glibc::invalid_nan: Inf -> 0xffc00000, NaN -> 조용하게).
// 검증: tests/joints/test_tanf(.cpp CPU, _gpu.cu GPU) — float 2^32 전부 libm 과 비트 비교 (FTZ 켬·끔).
#pragma once
#include <cstdint>
#include <cstring>

#include "../common/glibc_sincosf.h"

namespace eng {
namespace glibc {

EHD float tf_uf(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
EHD float tf_fabsf(float x) { return tf_uf(asuint(x) & 0x7fffffffu); }

// k_tanf.c __kernel_tanf
EHD float kernel_tanf(float x, float y, int iy) {
  const float one = 1.0000000000e+00f, pio4 = 7.8539812565e-01f, pio4lo = 3.7748947079e-08f;
  const float T[13] = {3.3333334327e-01f, 1.3333334029e-01f, 5.3968254477e-02f, 2.1869488060e-02f, 8.8632395491e-03f,
                       3.5920790397e-03f, 1.4562094584e-03f, 5.8804126456e-04f, 2.4646313977e-04f, 7.8179444245e-05f,
                       7.1407252108e-05f, -1.8558637748e-05f, 2.5907305826e-05f};
  float z, r, v, w, s;
  const int32_t hx = int32_t(asuint(x));
  const int32_t ix = hx & 0x7fffffff;
  if (ix < 0x39000000) {  // x < 2**-13
    if ((int)x == 0) {
      if ((ix | (iy + 1)) == 0) return one / tf_fabsf(x);
      else if (iy == 1) return x;
      else return -one / x;
    }
  }
  if (ix >= 0x3f2ca140) {  // |x|>=0.6744
    if (hx < 0) { x = -x; y = -y; }
    z = pio4 - x;
    w = pio4lo - y;
    x = z + w;
    y = 0.0f;
    if (tf_fabsf(x) < 0x1p-13f) return float((1 - ((hx >> 30) & 2)) * iy) * (1.0f - float(2 * iy) * x);
  }
  z = x * x;
  w = z * z;
  r = T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
  v = z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
  s = z * x;
  r = y + z * (s * (r + v) + y);
  r += T[0] * s;
  w = x + r;
  if (ix >= 0x3f2ca140) {
    v = (float)iy;
    return (float)(1 - ((hx >> 30) & 2)) * (v - (float)2.0 * (x - (w * w / (w + v) - r)));
  }
  if (iy == 1) return w;
  float a, t;
  z = w;
  z = tf_uf(asuint(z) & 0xfffff000u);
  v = r - (z - x);
  t = a = -(float)1.0 / w;
  t = tf_uf(asuint(t) & 0xfffff000u);
  s = (float)1.0 + t * z;
  return t + a * (s + t * v);
}

// s_tanf.c rem_pio2f (FMA 없는 reduce_fast)
EHD int32_t rem_pio2f_tan(float x, float* y) {
  double dx = x;
  int n;
  const SinCosT* p = table();
  if (abstop12(x) < abstop12(120.0f)) {
    const double r = dx * p->hpi_inv;
    n = ((int32_t)r + 0x800000) >> 24;
    const double nh = double(n) * p->hpi;  // 곱 따로 (축약 금지)
    dx = dx - nh;
  } else {
    const uint32_t xi = asuint(x);
    const int sign = int(xi >> 31);
    dx = reduce_large(xi, &n);
    dx = sign ? -dx : dx;
  }
  y[0] = float(dx);
  y[1] = float(dx - double(y[0]));
  return n;
}

EHD float tanf(float x) {
  float y[2], z = 0.0f;
  const int32_t ix = int32_t(asuint(x) & 0x7fffffffu);
  if (ix <= 0x3f490fda) return kernel_tanf(x, z, 1);
  if (ix >= 0x7f800000) return invalid_nan(x);  // x - x
  const int32_t n = rem_pio2f_tan(x, y);
  return kernel_tanf(y[0], y[1], 1 - ((n & 1) << 1));
}

}  // namespace glibc
}  // namespace eng
