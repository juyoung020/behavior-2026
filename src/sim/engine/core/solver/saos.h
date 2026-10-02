// solver 의 aos(PhysX SIMD 수학) 입구. 09-29 결정(docs/엔진_자체구현.md 12.7)대로 수학은 core/common/aos.h 하나만 쓴다
// (PhysX 5.6.1 aos 헤더 7개에서 기계적으로 만든 생성물, 286 함수 CPU·GPU 비트 동일 확인). 이 파일은 solver 코드가 쓰던
// 짧은 이름(FV·V4·BV·BS·M33V)을 aos 형에 잇고, aos 에 없는 도우미 몇 개만 둔다.
//  - FV = FloatV, V4 = Vec3V/Vec4V/QuatV, BV = BoolV, BS = BoolV(FloatV 비교 결과, 네 칸이 같음), M33V = Mat33V.
//  - FloatV 한 칸 표현은 -DENG_AOS_FLOATV_ONE_LANE (빌드 전체에 같은 값이어야 함). solver 시험은 켬·끔 둘 다 PhysX 와 비트 동일.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#include "../common/aos.h"

#include "sv_hd.h"

namespace eng {
namespace sv {

using namespace eng::aos;

typedef aos::FloatV FV;
typedef aos::Vec4V V4;
typedef aos::BoolV BV;
typedef aos::BoolV BS;
typedef aos::Mat33V M33V;
using aos::FloatV;
using aos::Vec3V;
using aos::Vec4V;
using aos::QuatV;
using aos::BoolV;
using aos::VecCrossV;

constexpr float kMaxReal = 3.4028234663852885981170418348452e+38F;  // PX_MAX_F32
constexpr float kEpsReal = 1.192092896e-07F;                          // PX_EPS_F32

// 스칼라 도우미 (aos 밖, PhysX 스칼라 코드용)
SV_HD uint32_t f2u(float f) { return sse::fb(f); }
SV_HD float u2f(uint32_t u) { return sse::bf(u); }
SV_HD uint32_t pmaxu(uint32_t a, uint32_t b) { return a < b ? b : a; }  // PxMax<PxU32> (PxMath.h:75)
SV_HD uint32_t pminu(uint32_t a, uint32_t b) { return a < b ? a : b; }

// PX_TRANSPOSE_44_34 (PxVecMath.h:1267) — 입력을 덮어쓰는 매크로와 같은 결과(출력 A,B,C = 각 입력의 x,y,z)
SV_HD void Transpose44_34(V4 inA, V4 inB, V4 inC, V4 inD, V4& outA, V4& outB, V4& outC) {
  outA = V4UnpackXY(inA, inC);
  inA = V4UnpackZW(inA, inC);
  inC = V4UnpackXY(inB, inD);
  inB = V4UnpackZW(inB, inD);
  outB = V4UnpackZW(outA, inC);
  outA = V4UnpackXY(outA, inC);
  outC = V4UnpackXY(inA, inB);
}

}  // namespace sv
}  // namespace eng
