// solver 모듈 공통 자료형 (손으로 짬). PhysX 5.6.1 의 TGS 풀이 자료를 같은 뜻·같은 필드로 옮긴 것.
// 포인터 대신 번호(index)·바이트 위치(offset)로 잇는다 → 같은 코드가 호스트(층 1)와 GPU(층 2)에서 돈다.
// 원본: physx/include/solver/PxSolverDefs.h (PxTGSSolverBody*, PxSolverConstraintDesc, PxConstraintBatchHeader),
//       include/PxContact.h (PxContactPatch, PxContact), include/geomutils/PxContactPoint.h,
//       source/lowleveldynamics/src/DyFrictionPatch.h, DyCorrelationBuffer.h, DySolverConstraint1DStep.h,
//       DyTGSContactPrepBlock.cpp:89-152 (블록 헤더·점·마찰), DySolverConstraintTypes.h
#pragma once
#include <cstdint>

#include "solver_io.h"
#include "saos.h"

namespace eng {
namespace sv {

// PxVec3 <-> Vec3V 로드/저장 (V3LoadA/V3LoadU/V3LoadU_SafeReadW 는 W = +0, V4LoadA(&v.x) 는 W = 다음 필드)
SV_HD V4 ld4(const V3& v, float w) { return V4{v.x, v.y, v.z, w}; }
SV_HD V4 ldv(const V3& v) { return V4{v.x, v.y, v.z, 0.0f}; }
SV_HD V4 ldv3(const V3& v) { return V4{v.x, v.y, v.z, 0.0f}; }
SV_HD void stv(V4 a, V3& v) { v.x = a.f[0]; v.y = a.f[1]; v.z = a.f[2]; }

// ---------------- 단일 경로 제약 자료 (DySolverConstraint1DStep.h:47-117)
struct alignas(16) SolverContactHeaderStep {
  uint8_t type, flags, numNormalConstr, numFrictionConstr;
  float angDom0, angDom1, invMass0;
  V4 staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W;
  V3 normal;
  float maxPenBias, invMass1, minNormalForce;
  uint32_t broken;
  uint32_t frictionBrokenWriteback;  // 마찰 패치 번호(현 스텝 arena) 또는 NONE (원본은 바이트 포인터)
  uint32_t pad[2];
};
struct alignas(16) SolverContactPointStep {
  V3 raXnI;
  float separation;
  V3 rbXnI;
  float velMultiplier, targetVelocity, biasCoefficient, recipResponse, maxImpulse;
};
struct alignas(16) SolverContactFrictionStep {
  V4 normalXYZ_ErrorW;
  V4 raXnI_targetVelW;
  V4 rbXnI_velMultiplierW;
  float biasScale, appliedForce, frictionScale;
  uint32_t pad;
};

// ---------------- 4개 묶음 제약 자료 (DyTGSContactPrepBlock.cpp:89-152)
struct alignas(16) SolverContactHeaderStepBlock {
  enum { eHAS_MAX_IMPULSE = 1 << 0, eHAS_TARGET_VELOCITY = 1 << 1 };
  uint8_t type, numNormalConstr, numFrictionConstr, flag;
  uint8_t flags[4];
  uint8_t numNormalConstrs[4];
  uint8_t numFrictionConstrs[4];
  V4 staticFriction, dynamicFriction;
  V4 invMass0D0, invMass1D1, angDom0, angDom1;
  V4 normalX, normalY, normalZ;
  V4 maxPenBias;
  BV broken;
  uint32_t frictionBrokenWriteback[4];
};
struct alignas(16) SolverContactPointStepBlock {
  V4 raXnI[3];
  V4 rbXnI[3];
  V4 separation, velMultiplier, targetVelocity, biasCoefficient, recipResponse;
};
struct alignas(16) SolverContactFrictionStepBlock {
  V4 normal[3];
  V4 raXnI[3];
  V4 rbXnI[3];
  V4 error, velMultiplier, targetVel, biasCoefficient;
};

}  // namespace sv
}  // namespace eng
