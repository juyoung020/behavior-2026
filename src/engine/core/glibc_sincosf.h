// glibc 2.35 의 sinf / cosf 를 그대로 옮긴 것 (CPU·CUDA 공용). PhysX 는 PxSinCos -> ::sinf / ::cosf 를 쓰므로
// (physx/include/foundation/PxMath.h:210) 리눅스 대회 환경(Ubuntu 22.04, glibc 2.35)과 비트까지 같으려면
// GPU 에서도 같은 알고리즘이어야 한다. CUDA 의 sinf/__sinf 는 결과가 다르다.
// 원본: glibc-2.35 sysdeps/ieee754/flt-32/{s_sinf.c, s_cosf.c, s_sincosf.h, s_sincosf_data.c, sincosf_poly.h}
// x86_64 glibc 는 CPU 에 FMA·AVX2 가 있으면 -mfma -mavx2 로 컴파일한 판(__sinf_fma)을 고른다
// (sysdeps/x86_64/fpu/multiarch/Makefile:34, ifunc-fma.h). 이때 gcc 가 a + b*c 를 fma(b, c, a) 로 묶는다 -> ENG_GLIBC_FMA=1.
// 검증: tests/test_sincosf.cu 가 float 전 범위(2^32 개)를 이 PC 의 libm 과 비트 비교한다.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#include "pmath.h"

#ifndef ENG_GLIBC_FMA
#define ENG_GLIBC_FMA 1
#endif

namespace eng {
namespace glibc {

// a + b*c  (FMA 판이면 한 번만 반올림)
EHD double madd(double b, double c, double a) {
#if ENG_GLIBC_FMA
  return fma(b, c, a);
#else
  return a + b * c;
#endif
}

struct SinCosT {
  double sign[4];
  double hpi_inv;  // 2/PI * 2^24 (TOINT_INTRINSICS = 0)
  double hpi;
  double c0, c1, c2, c3, c4;
  double s1, s2, s3;
};

#if defined(__CUDACC__)
__device__ __constant__ static SinCosT kTableD[2] = {
#define ENG_SINCOS_TABLE_BODY \
  {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0, 0x1p0, -0x1.ffffffd0c621cp-2, 0x1.55553e1068f19p-5, \
   -0x1.6c087e89a359dp-10, 0x1.99343027bf8c3p-16, -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13}, \
  {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0, -0x1p0, 0x1.ffffffd0c621cp-2, -0x1.55553e1068f19p-5, \
   0x1.6c087e89a359dp-10, -0x1.99343027bf8c3p-16, -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13}
    ENG_SINCOS_TABLE_BODY};
__device__ __constant__ static uint32_t kInvPio4D[24] = {
#define ENG_INV_PIO4_BODY \
  0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, \
  0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, \
  0x95993c43, 0x993c4390, 0x3c439041
    ENG_INV_PIO4_BODY};
#else
#define ENG_SINCOS_TABLE_BODY \
  {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0, 0x1p0, -0x1.ffffffd0c621cp-2, 0x1.55553e1068f19p-5, \
   -0x1.6c087e89a359dp-10, 0x1.99343027bf8c3p-16, -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13}, \
  {{1.0, -1.0, -1.0, 1.0}, 0x1.45F306DC9C883p+23, 0x1.921FB54442D18p0, -0x1p0, 0x1.ffffffd0c621cp-2, -0x1.55553e1068f19p-5, \
   0x1.6c087e89a359dp-10, -0x1.99343027bf8c3p-16, -0x1.555545995a603p-3, 0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13}
#define ENG_INV_PIO4_BODY \
  0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, \
  0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, \
  0x95993c43, 0x993c4390, 0x3c439041
#endif
static const SinCosT kTableH[2] = {ENG_SINCOS_TABLE_BODY};
static const uint32_t kInvPio4H[24] = {ENG_INV_PIO4_BODY};

EHD const SinCosT* table() {
#if defined(__CUDA_ARCH__)
  return kTableD;
#else
  return kTableH;
#endif
}
EHD const uint32_t* inv_pio4() {
#if defined(__CUDA_ARCH__)
  return kInvPio4D;
#else
  return kInvPio4H;
#endif
}

EHD uint32_t asuint(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
EHD uint32_t abstop12(float x) { return (asuint(x) >> 20) & 0x7ff; }

EHD double reduce_fast(double x, const SinCosT* p, int* np) {
  const double r = x * p->hpi_inv;
  const int n = ((int32_t)r + 0x800000) >> 24;
  *np = n;
  return madd(-double(n), p->hpi, x);  // x - n * p->hpi
}

EHD double reduce_large(uint32_t xi, int* np) {
  const uint32_t* arr = &inv_pio4()[(xi >> 26) & 15];
  const int shift = (xi >> 23) & 7;
  uint64_t n, res0, res1, res2;
  xi = (xi & 0xffffff) | 0x800000;
  xi <<= shift;
  res0 = xi * arr[0];
  res1 = (uint64_t)xi * arr[4];
  res2 = (uint64_t)xi * arr[8];
  res0 = (res2 >> 32) | (res0 << 32);
  res0 += res1;
  n = (res0 + (1ULL << 61)) >> 62;
  res0 -= n << 62;
  const double x = (double)(int64_t)res0;
  *np = int(n);
  return x * 0x1.921FB54442D18p-62;  // pi63
}

EHD float sinf_poly(double x, double x2, const SinCosT* p, int n) {
  if ((n & 1) == 0) {
    const double x3 = x * x2;
    const double s1 = madd(x2, p->s3, p->s2);
    const double x7 = x3 * x2;
    const double s = madd(x3, p->s1, x);
    return float(madd(x7, s1, s));
  } else {
    const double x4 = x2 * x2;
    const double c2 = madd(x2, p->c4, p->c3);
    const double c1 = madd(x2, p->c1, p->c0);
    const double x6 = x4 * x2;
    const double c = madd(x4, p->c2, c1);
    return float(madd(x6, c2, c));
  }
}

EHD float sinf(float y) {
  double x = y;
  int n;
  const SinCosT* p = &table()[0];
  const float pio4 = 0x1.921FB6p-1f;
  if (abstop12(y) < abstop12(pio4)) {
    const double s = x * x;
    if (abstop12(y) < abstop12(0x1p-12f)) return y;
    return sinf_poly(x, s, p, 0);
  } else if (abstop12(y) < abstop12(120.0f)) {
    x = reduce_fast(x, p, &n);
    const double s = p->sign[n & 3];
    if (n & 2) p = &table()[1];
    return sinf_poly(x * s, x * x, p, n);
  } else if (abstop12(y) < abstop12(INFINITY)) {
    const uint32_t xi = asuint(y);
    const int sign = xi >> 31;
    x = reduce_large(xi, &n);
    const double s = p->sign[(n + sign) & 3];
    if ((n + sign) & 2) p = &table()[1];
    return sinf_poly(x * s, x * x, p, n);
  }
  return (y - y) / (y - y);  // NaN (glibc __math_invalidf)
}

EHD float cosf(float y) {
  double x = y;
  int n;
  const SinCosT* p = &table()[0];
  const float pio4 = 0x1.921FB6p-1f;
  if (abstop12(y) < abstop12(pio4)) {
    const double x2 = x * x;
    if (abstop12(y) < abstop12(0x1p-12f)) return 1.0f;
    return sinf_poly(x, x2, p, 1);
  } else if (abstop12(y) < abstop12(120.0f)) {
    x = reduce_fast(x, p, &n);
    const double s = p->sign[n & 3];
    if (n & 2) p = &table()[1];
    return sinf_poly(x * s, x * x, p, n ^ 1);
  } else if (abstop12(y) < abstop12(INFINITY)) {
    const uint32_t xi = asuint(y);
    const int sign = xi >> 31;
    x = reduce_large(xi, &n);
    const double s = p->sign[(n + sign) & 3];
    if ((n + sign) & 2) p = &table()[1];
    return sinf_poly(x * s, x * x, p, n ^ 1);
  }
  return (y - y) / (y - y);
}

}  // namespace glibc
}  // namespace eng
