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

#include "px_foundation.h"

namespace eng {
// Px 값 형·스칼라 수학은 eng::px 한 벌 (px_foundation.h, contact 번역본을 common 으로 올림 — 문서 12.7, 09-30 결정).
// 생성물 aos.h 는 eng 이름 공간에서 이름을 찾으므로 여기서 이어 준다 (별칭일 뿐, 형은 하나).
using px::PxF32; using px::PxF64; using px::PxI32; using px::PxU32; using px::PxI16; using px::PxU16; using px::PxI8; using px::PxU8;
using px::PxI64; using px::PxU64;
typedef int PxIntBool;
using px::PxVec3; using px::PxVec4; using px::PxQuat; using px::PxMat33; using px::PxTransform;
using px::PxPi; using px::PxHalfPi; using px::PxTwoPi; using px::PxInvPi; using px::PxInvTwoPi;
using px::PxUnionCast; using px::PxAbs; using px::PxClamp; using px::PxFloor; using px::PxCeil; using px::PxIsFinite;
using px::PxMax; using px::PxMin; using px::PxSqrt; using px::PxRecipSqrt;

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
