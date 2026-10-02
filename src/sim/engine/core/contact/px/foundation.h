// PhysX foundation (PxVec3·PxQuat·PxMat33·PxTransform·PxBounds3·PxPlane·PxMath) 의 우리 판 (eng::px). PhysX 링크 없음.
// 식은 원본 그대로(gen/translate.py 가 기계 번역한 gen_foundation_vec.inc), 이 파일은 그 바탕(형·상수·스칼라 수학)만 손으로 둔다.
// 원본: physx/include/foundation/PxSimpleTypes.h, PxMath.h, unix/PxUnixMathIntrinsics.h (태그 107.3-omni-and-physx-5.6.1, BSD-3)
// 비트 규칙: 제곱근은 IEEE(sqrtss 와 같음), sin/cos 은 glibc 2.35 이식본(common/glibc_sincosf.h). 다른 libm 함수는 없앴다
// (tan·asin·acos·atan·atan2·exp·pow·log 는 GPU 에서 값이 달라지므로 필요하면 리드가 glibc 이식 후 넣는다).
#pragma once
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>

#include "core/common/glibc_sincosf.h"
#include "core/contact/px/sse_emu.h"

#define E_ASSERT(...) ((void)0)
#define E_ASSERT2(...) ((void)0)
#define E_UNUSED(x) (void)(x)
#define E_STATIC_ASSERT(x) static_assert((x), #x)
#define E_OFFSET_OF(T, m) offsetof(T, m)
#define E_PLACEMENT_NEW(p, T) new (p) T
#define E_NOCOPY(Class) \
 protected:             \
  Class(const Class&);  \
  Class& operator=(const Class&);
#define EPX_FL __FILE__, __LINE__

#define EPX_MAX_F32 3.4028234663852885981170418348452e+38F
#define EPX_EPS_F32 FLT_EPSILON
#define EPX_MAX_REAL EPX_MAX_F32
#define EPX_EPS_REAL EPX_EPS_F32
#define EPX_NORMALIZATION_EPSILON float(1e-20f)
#define EPX_MAX_I8 INT8_MAX
#define EPX_MIN_I8 INT8_MIN
#define EPX_MAX_U8 UINT8_MAX
#define EPX_MAX_I16 INT16_MAX
#define EPX_MIN_I16 INT16_MIN
#define EPX_MAX_U16 UINT16_MAX
#define EPX_MAX_I32 INT32_MAX
#define EPX_MIN_I32 INT32_MIN
#define EPX_MAX_U32 UINT32_MAX
#define EPX_INVALID_U32 0xffffffff
#define EPX_INVALID_U16 0xffff
#define EPX_SIGN_BITMASK 0x80000000

namespace eng {
namespace px {

typedef int64_t PxI64;
typedef uint64_t PxU64;
typedef int32_t PxI32;
typedef uint32_t PxU32;
typedef int16_t PxI16;
typedef uint16_t PxU16;
typedef int8_t PxI8;
typedef uint8_t PxU8;
typedef float PxF32;
typedef double PxF64;
typedef float PxReal;
typedef PxI32 PxIntBool;
static const PxIntBool PxIntFalse = 0;
static const PxIntBool PxIntTrue = 1;

struct PxIdentity_t { };  // PxIDENTITY 등 (PxVec3.h 등의 생성자 표식)
enum PxZERO { PxZero };
enum PxIDENTITY { PxIdentity };
enum PxEMPTY { PxEmpty };

static constexpr float PxPi = float(3.141592653589793);
static constexpr float PxHalfPi = float(1.57079632679489661923);
static constexpr float PxTwoPi = float(6.28318530717958647692);
static constexpr float PxInvPi = float(0.31830988618379067154);
static constexpr float PxInvTwoPi = float(0.15915494309189533577);
static constexpr float PxPiDivTwo = float(1.57079632679489661923);
static constexpr float PxPiDivFour = float(0.78539816339744830962);
static constexpr float PxSqrt2 = float(1.4142135623730951);
static constexpr float PxInvSqrt2 = float(0.7071067811865476);

template <class A, class B>
EHD A PxUnionCast(B b) {
  static_assert(sizeof(A) == sizeof(B), "PxUnionCast size");
  A a;
  std::memcpy(&a, &b, sizeof(A));
  return a;
}

namespace intrinsics {
EHD float abs(float a) { return em_u2f(em_f2u(a) & 0x7fffffffu); }  // fabsf
EHD float fsel(float a, float b, float c) { return (a >= 0.0f) ? b : c; }
EHD float sign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
EHD float recip(float a) { return em_div1(1.0f, a); }
EHD float recipFast(float a) { return em_div1(1.0f, a); }
EHD float sqrt(float a) { return em_sqrt1(a); }
EHD float recipSqrt(float a) { return em_div1(1.0f, em_sqrt1(a)); }
EHD float recipSqrtFast(float a) { return em_div1(1.0f, em_sqrt1(a)); }
EHD float sin(float a) { return eng::glibc::sinf(a); }
EHD float cos(float a) { return eng::glibc::cosf(a); }
EHD float selectMin(float a, float b) { return a < b ? a : b; }
EHD float selectMax(float a, float b) { return a > b ? a : b; }
EHD bool isFinite(float a) { return !((em_f2u(a) & 0x7fffffff) >= 0x7f800000); }
}  // namespace intrinsics

template <class T> EHD T PxMax(T a, T b) { return a < b ? b : a; }
template <> EHD float PxMax(float a, float b) { return intrinsics::selectMax(a, b); }
template <class T> EHD T PxMin(T a, T b) { return a < b ? a : b; }
template <> EHD float PxMin(float a, float b) { return intrinsics::selectMin(a, b); }
EHD float PxAbs(float a) { return intrinsics::abs(a); }
EHD bool PxEquals(float a, float b, float eps) { return (PxAbs(a - b) < eps); }
EHD int32_t PxAbs(int32_t a) { return a < 0 ? -a : a; }
template <class T> EHD T PxClamp(T v, T lo, T hi) { return PxMin(hi, PxMax(lo, v)); }
EHD float PxSqrt(float a) { return intrinsics::sqrt(a); }
EHD float PxRecipSqrt(float a) { return intrinsics::recipSqrt(a); }
EHD PxF32 PxSqr(const PxF32 a) { return a * a; }
EHD float PxSin(float a) { return intrinsics::sin(a); }
EHD float PxCos(float a) { return intrinsics::cos(a); }
EHD void PxSinCos(const PxF32 a, PxF32& s, PxF32& c) { s = PxSin(a); c = PxCos(a); }  // PxMath.h:205 (CPU 경로)
EHD PxF32 PxDegToRad(const PxF32 a) { return 0.01745329251994329547f * a; }
EHD bool PxIsFinite(float f) { return intrinsics::isFinite(f); }
EHD float PxFloor(float a) { return ::floorf(a); }
EHD float PxCeil(float a) { return ::ceilf(a); }
EHD float PxSign(float a) { return intrinsics::sign(a); }
EHD float PxSign2(float a, float eps = FLT_EPSILON) { return (a < -eps) ? -1.0f : (a > eps) ? 1.0f : 0.0f; }

EHD uint32_t PxHighestSetBitUnsafe(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return uint32_t(31 - __clz(int(v)));
#else
  return uint32_t(31 - __builtin_clz(v));
#endif
}
EHD uint32_t PxLowestSetBitUnsafe(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return uint32_t(__ffs(int(v)) - 1);
#else
  return uint32_t(__builtin_ctz(v));
#endif
}
EHD uint32_t PxCountLeadingZeros(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return uint32_t(__clz(int(v)));
#else
  return v ? uint32_t(__builtin_clz(v)) : 32u;
#endif
}
EHD void PxPrefetchLine(const void*, uint32_t = 0) {}
EHD void PxPrefetch(const void*, uint32_t = 1) {}

}  // namespace px

#include "core/contact/px/gen_foundation_vec.inc"

}  // namespace eng
