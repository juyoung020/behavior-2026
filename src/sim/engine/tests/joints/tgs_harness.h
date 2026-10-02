// joints 시험 전용: 조인트 함수를 PhysX 와 같은 순서로 돌려 보기 위한 최소 TGS 틀 (CPU·CUDA 공용).
// 진짜 섬 묶기·분할·반복 루프는 solver 모듈 몫이다. 여기 것은 "섬 하나 = 몸체 몇 개 + 1D 조인트 몇 개(모두 stride 1)" 만 다룬다.
// 원본(physx/source/lowleveldynamics/src)
//   DyBodyCoreIntegrator.h:39 (bodyCoreComputeUnconstrainedVelocity), DyTGSDynamics.cpp:154 (copyToSolverBodyDataStep),
//   :1403 (integrateCoreStep), :1549 (copyBackBodies), :2515 (iterativeSolveIsland — 조인트가 있는 묶음 경로),
//   immediatemode/src/NpImmediateMode.cpp:1630 (PxSolveConstraintsTGS: 즉시 모드 루프, 함수 단위 시험의 대조군)
#pragma once
#include <cstdint>

#include "core/common/body.h"
#include "core/common/glibc_sincosf.h"
#include "core/joints/tgs_1d.h"

namespace eng {
namespace jtest {
using namespace eng::jnt;

EHD V3 safeRecip(const V3& v) { return V3{v.x == 0.f ? 0.f : 1.f / v.x, v.y == 0.f ? 0.f : 1.f / v.y, v.z == 0.f ? 0.f : 1.f / v.z}; }
EHD V3 safeSqrt(const V3& v) { return V3{v.x == 0.0f ? 0.0f : psqrt(v.x), v.y == 0.0f ? 0.0f : psqrt(v.y), v.z == 0.0f ? 0.0f : psqrt(v.z)}; }

// DyBodyCoreIntegrator.h:39
EHD void bodyCoreComputeUnconstrainedVelocity(const V3& gravity, float dt, float linearDamping, float angularDamping, float accelScale,
                                              float maxLinearVelocitySq, float maxAngularVelocitySq, V3& inOutLin, V3& inOutAng,
                                              bool disableGravity) {
  V3 lv = inOutLin, av = inOutAng;
  const float linearDampingTimesDT = linearDamping * dt;
  const float angularDampingTimesDT = angularDamping * dt;
  const float oneMinusLin = 1.0f - linearDampingTimesDT;
  const float oneMinusAng = 1.0f - angularDampingTimesDT;
  if (!disableGravity) {
    const V3 linearAccelTimesDT = gravity * dt * accelScale;
    lv += linearAccelTimesDT;
  }
  const float linVelMultiplier = fsel(oneMinusLin, oneMinusLin, 0.0f);
  const float angVelMultiplier = fsel(oneMinusAng, oneMinusAng, 0.0f);
  lv *= linVelMultiplier;
  av *= angVelMultiplier;
  const float linVelSq = magSq(lv);
  if (linVelSq > maxLinearVelocitySq) lv *= psqrt(maxLinearVelocitySq / linVelSq);
  const float angVelSq = magSq(av);
  if (angVelSq > maxAngularVelocitySq) av *= psqrt(maxAngularVelocitySq / angVelSq);
  inOutLin = lv;
  inOutAng = av;
}

// DyTGSDynamics.cpp:154
EHD void copyToSolverBodyDataStep(const V3& linearVelocity, const V3& angularVelocity, float invMass, const V3& invInertia, const Tf& globalPose,
                                  float maxDepenetrationVelocity, float maxContactImpulse, uint32_t nodeIndex, float reportThreshold,
                                  float maxAngVelSq, uint32_t lockFlags, bool isKinematic, TgsBodyVel& solverVel, TgsTxInertia& txI,
                                  TgsBodyData& data, float dt, bool gyroscopicForces) {
  const M33 rotation = mat_from_quat_simd(globalPose.q);
  const V3 sqrtInvInertia = safeSqrt(invInertia);
  const V3 sqrtBodySpaceInertia = safeRecip(sqrtInvInertia);
  txI.sqrtInvInertia = transformInertiaTensor(sqrtInvInertia, rotation);
  txI.body2WorldP = globalPose.p;
  txI.deltaBody2WorldQ = qid();
  const M33 sqrtInertia = transformInertiaTensor(sqrtBodySpaceInertia, rotation);
  V3 lv = linearVelocity;
  V3 av = angularVelocity;
  if (gyroscopicForces) {
    const V3 localInertia{invInertia.x == 0.f ? 0.f : 1.f / invInertia.x, invInertia.y == 0.f ? 0.f : 1.f / invInertia.y,
                          invInertia.z == 0.f ? 0.f : 1.f / invInertia.z};
    const V3 localAngVel = rotateInv(globalPose.q, av);
    const V3 origMom = mulc(localInertia, localAngVel);
    const V3 torque = -cross(localAngVel, origMom);
    V3 newMom = origMom + torque * dt;
    const float denom = mag(newMom);
    const float ratio = denom > 0.f ? mag(origMom) / denom : 0.f;
    newMom *= ratio;
    const V3 newDeltaAngVel = rotate(globalPose.q, mulc(invInertia, newMom) - localAngVel);
    av += newDeltaAngVel;
  }
  if (lockFlags) {
    if (lockFlags & LOCK_LX) lv.x = 0.f;
    if (lockFlags & LOCK_LY) lv.y = 0.f;
    if (lockFlags & LOCK_LZ) lv.z = 0.f;
    if (lockFlags & LOCK_AX) av.x = 0.f;
    if (lockFlags & LOCK_AY) av.y = 0.f;
    if (lockFlags & LOCK_AZ) av.z = 0.f;
  }
  solverVel.linearVelocity = lv;
  solverVel.angularVelocity = sqrtInertia * av;
  solverVel.deltaLinDt = V3{0.f, 0.f, 0.f};
  solverVel.deltaAngDt = V3{0.f, 0.f, 0.f};
  solverVel.lockFlags = uint16_t(lockFlags);
  solverVel.isKinematic = isKinematic;
  solverVel.maxAngVel = psqrt(maxAngVelSq);
  solverVel.partitionMask = 0;
  data.nodeIndex = nodeIndex;
  data.invMass = invMass;
  data.penBiasClamp = maxDepenetrationVelocity;
  data.maxContactImpulse = maxContactImpulse;
  data.reportThreshold = reportThreshold;
  data.originalLinearVelocity = lv;
  data.originalAngularVelocity = av;
}

// DyTGSDynamics.cpp:1403
EHD void integrateCoreStep(TgsBodyVel& vel, TgsTxInertia& txI, float dt) {
  const uint32_t lockFlags = vel.lockFlags;
  if (lockFlags) {
    if (lockFlags & LOCK_LX) vel.linearVelocity.x = 0.f;
    if (lockFlags & LOCK_LY) vel.linearVelocity.y = 0.f;
    if (lockFlags & LOCK_LZ) vel.linearVelocity.z = 0.f;
    if (lockFlags & LOCK_AX) vel.angularVelocity.x = 0.f;
    if (lockFlags & LOCK_AY) vel.angularVelocity.y = 0.f;
    if (lockFlags & LOCK_AZ) vel.angularVelocity.z = 0.f;
  }
  const V3 linearMotionVel = vel.linearVelocity;
  const V3 delta = linearMotionVel * dt;
  const V3 unmolestedAngVel = vel.angularVelocity;
  const V3 angularMotionVel = txI.sqrtInvInertia * vel.angularVelocity;
  const float w2 = magSq(angularMotionVel);
  txI.body2WorldP += delta;
  if (w2 != 0.0f) {
    const float w = psqrt(w2);
    const float v = dt * w * 0.5f;
    float s = glibc::sinf(v);
    const float q = glibc::cosf(v);
    s /= w;
    const V3 pqr = angularMotionVel * s;
    const Q quatVel{pqr.x, pqr.y, pqr.z, 0};
    Q result = quatVel * txI.deltaBody2WorldQ;
    result = result + txI.deltaBody2WorldQ * q;
    txI.deltaBody2WorldQ = normalized(result);
  }
  vel.deltaAngDt += unmolestedAngVel * dt;
  vel.deltaLinDt += delta;
}

// 세계(정적) 풀이 몸체 (DyTGSDynamics.cpp:299-309)
EHD void worldBody(TgsBodyVel& v, TgsTxInertia& t, TgsBodyData& d) {
  v = TgsBodyVel{};
  t = TgsTxInertia{};
  t.sqrtInvInertia = M33{V3{0, 0, 0}, V3{0, 0, 0}, V3{0, 0, 0}};
  t.body2WorldP = V3{0, 0, 0};
  t.deltaBody2WorldQ = qid();
  d = TgsBodyData{};
  d.penBiasClamp = -MAX_F32;
  d.maxContactImpulse = MAX_F32;
  d.nodeIndex = 0x7fffffffu;  // PX_INVALID_NODE (PxNodeIndex 가 아닌 u32 값 — 비교에는 안 씀)
  d.invMass = 0;
}

}  // namespace jtest
}  // namespace eng
