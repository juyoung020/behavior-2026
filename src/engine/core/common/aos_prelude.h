// aos.h(생성물)가 기대하는 PhysX 이름들을 엔진 쪽 것으로 잇는다. 리드 관리.
// PhysX 원본 aos 코드가 참조하는 형·함수·매크로만 둔다 (physx/include/foundation 의 같은 이름과 같은 뜻).
#pragma once
#include <cstdint>
#include <cstring>

#include "pmath.h"
#include "sse_emu.h"

#define PX_ALIGN(alignment, decl) decl __attribute__((aligned(alignment)))  // PhysX unix PxPreprocessor.h 와 같음
#define PX_ALIGN_PREFIX(alignment)
#define PX_ALIGN_SUFFIX(alignment) __attribute__((aligned(alignment)))
#define PX_RESTRICT __restrict__
#define PX_UNUSED(x) (void)(x)
#define PX_ASSERT(x) ((void)0)
#define PX_EPS_REAL 1.192092896e-07F   // PxSimpleTypes.h
#define PX_MAX_REAL 3.402823466e+38F
#define PX_MAX_F32 3.4028234663852885981170418348452e+38F

namespace eng {

typedef float PxF32;
typedef double PxF64;
typedef int32_t PxI32;
typedef uint32_t PxU32;
typedef int16_t PxI16;
typedef uint16_t PxU16;
typedef int8_t PxI8;
typedef uint8_t PxU8;
typedef int PxIntBool;

// PhysX 값 형과 같은 배치 (x,y,z / x,y,z,w / 열 우선 3x3 / q 다음 p)
// PhysX 값 형: 같은 배치, 원본 aos 코드가 쓰는 생성자만. eng::V3/Q/Tf 와 서로 바뀐다.
struct PxVec3 {
  float x, y, z;
  EHD PxVec3() {}
  EHD PxVec3(float a, float b, float c) : x(a), y(b), z(c) {}
  EHD PxVec3(const V3& v) : x(v.x), y(v.y), z(v.z) {}
  EHD operator V3() const { return V3{x, y, z}; }
};
struct PxVec4 {
  float x, y, z, w;
  EHD PxVec4() {}
  EHD PxVec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
};
struct PxQuat {
  float x, y, z, w;
  EHD PxQuat() {}
  EHD PxQuat(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
  EHD PxQuat(const Q& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}
  EHD operator Q() const { return Q{x, y, z, w}; }
};
struct PxMat33 {
  PxVec3 column0, column1, column2;
  EHD PxMat33() {}
  EHD PxMat33(const PxVec3& a, const PxVec3& b, const PxVec3& c) : column0(a), column1(b), column2(c) {}
};
struct PxTransform {
  PxQuat q;
  PxVec3 p;
  EHD PxTransform() {}
  EHD PxTransform(const PxVec3& p_, const PxQuat& q_) : q(q_), p(p_) {}  // PhysX 생성자 순서 (p, q)
  EHD PxTransform(const Tf& t) : q(t.q), p(t.p) {}
  EHD operator Tf() const { return Tf{Q(q), V3(p)}; }
};

static constexpr float PxPi = 3.141592653589793f;
static constexpr float PxHalfPi = 1.57079632679489661923f;
static constexpr float PxTwoPi = 6.28318530717958647692f;
static constexpr float PxInvPi = 0.31830988618379067154f;
static constexpr float PxInvTwoPi = 0.15915494309189533577f;

template <class A, class B> EHD A PxUnionCast(B b) { A a; static_assert(sizeof(A) == sizeof(B), ""); memcpy(&a, &b, sizeof(A)); return a; }
EHD float PxAbs(float a) { return a < 0.0f ? -a : a; }  // PxMath.h: ::fabsf -> 부호 비트만 지움과 같음(NaN 제외)
EHD float PxClamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }  // PxMath.h:98 PxMin(hi, PxMax(lo, v))
EHD float PxFloor(float a) { return floorf(a); }
EHD float PxCeil(float a) { return ceilf(a); }
EHD bool PxIsFinite(float a) {
  const uint32_t u = sse::fb(a);
  return (u & 0x7f800000u) != 0x7f800000u;
}

// 전역 상수: 생성물이 호스트판(이름_h)과 장치판(이름_d) 두 벌을 만들고, 쓰는 곳은 ENG_G(이름) 으로 고른다
union ENG_U4F { uint32_t u[4]; float f[4]; };  // 비트 무늬 상수(마스크)용: 정수 칸으로 초기화

// FloatV 한 칸 표현 (선택, -DENG_AOS_FLOATV_ONE_LANE, docs/엔진_자체구현.md 12.7).
// PhysX 의 FloatV 는 네 칸이 늘 같다(시험으로 확인) -> 첫 칸만 들고 다니고, 벡터로 쓰일 때 네 칸으로 복제한다.
// 식은 생성물 그대로이고 형변환만 끼므로, 인라인 뒤 안 쓰이는 칸 계산은 컴파일러가 지운다.
namespace sse {
struct FV1 {
  float x;
  EHD FV1() {}
  EHD FV1(const m128& v) : x(v.f[0]) {}
  EHD operator m128() const { return m128{{x, x, x, x}}; }
};
}  // namespace sse
}  // namespace eng

#if defined(__CUDA_ARCH__)
#define ENG_G(name) name##_d
#else
#define ENG_G(name) name##_h
#endif
