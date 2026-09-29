// TGS 풀이 몸체 함수 (손으로 짬, PhysX 5.6.1 과 비트 동일): 제약 없는 속도(중력·감쇠·제한), 풀이 몸체 만들기,
// 적분 한 번, 되쓰기, 잠 판정.
// 원본 (physx/source/lowleveldynamics/src/)
//   DyBodyCoreIntegrator.h:39      bodyCoreComputeUnconstrainedVelocity
//   DyTGSDynamics.cpp:154-274      copyToSolverBodyDataStep / copyToSolverBodyDataStepKinematic
//   DyTGSDynamics.cpp:1403-1476    integrateCoreStep
//   DyTGSDynamics.cpp:1549-1580    copyBackBodies
//   DySleep.cpp:35-236             updateWakeCounter / sleepCheck (DySleepingConfigulation.h 상수)
//   lowlevel/software/include/PxsRigidBody.h:47-53 내부 플래그
#pragma once
#include "../common/body.h"
#include "../common/glibc_sincosf.h"
#include "tgs_types.h"

namespace eng {
namespace sv {

enum : uint16_t {  // PxsRigidBody::mInternalFlags
  RB_FROZEN = 1 << 0,
  RB_FREEZE_THIS_FRAME = 1 << 1,
  RB_UNFREEZE_THIS_FRAME = 1 << 2,
  RB_ACTIVATE_THIS_FRAME = 1 << 3,
  RB_DEACTIVATE_THIS_FRAME = 1 << 4,
};
constexpr float PXD_FREEZE_INTERVAL = 1.5f;
constexpr float PXD_FREEZE_TOLERANCE = 0.25f;
constexpr float PXD_SLEEP_DAMPING = 0.5f;
constexpr float PXD_FREEZE_SCALE = 0.1f;

// DyBodyCoreIntegrator.h:39
SV_HD void bodyCoreComputeUnconstrainedVelocity(const V3& gravity, float dt, float linearDamping, float angularDamping, float accelScale,
                                                float maxLinearVelocitySq, float maxAngularVelocitySq, V3& inOutLinearVelocity,
                                                V3& inOutAngularVelocity, bool disableGravity) {
  V3 linearVelocity = inOutLinearVelocity;
  V3 angularVelocity = inOutAngularVelocity;
  const float linearDampingTimesDT = linearDamping * dt;
  const float angularDampingTimesDT = angularDamping * dt;
  const float oneMinusLinearDampingTimesDT = 1.0f - linearDampingTimesDT;
  const float oneMinusAngularDampingTimesDT = 1.0f - angularDampingTimesDT;
  if (!disableGravity) {
    const V3 linearAccelTimesDT = gravity * dt * accelScale;
    linearVelocity += linearAccelTimesDT;
  }
  const float linVelMultiplier = fsel(oneMinusLinearDampingTimesDT, oneMinusLinearDampingTimesDT, 0.0f);
  const float angVelMultiplier = fsel(oneMinusAngularDampingTimesDT, oneMinusAngularDampingTimesDT, 0.0f);
  linearVelocity *= linVelMultiplier;
  angularVelocity *= angVelMultiplier;
  const float linVelSq = magSq(linearVelocity);
  if (linVelSq > maxLinearVelocitySq) linearVelocity *= psqrt(maxLinearVelocitySq / linVelSq);
  const float angVelSq = magSq(angularVelocity);
  if (angVelSq > maxAngularVelocitySq) angularVelocity *= psqrt(maxAngularVelocitySq / angVelSq);
  inOutLinearVelocity = linearVelocity;
  inOutAngularVelocity = angularVelocity;
}

SV_HD void v3store(const V3& v, float* p) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }
SV_HD V3 v3load(const float* p) { return V3{p[0], p[1], p[2]}; }

// DyTGSDynamics.cpp:154
SV_HD void copyToSolverBodyDataStep(const V3& linearVelocity, const V3& angularVelocity, float invMass, const V3& invInertia, const Tf& globalPose,
                                    float maxDepenetrationVelocity, float maxContactImpulse, uint32_t nodeIndex, float reportThreshold,
                                    float maxAngVelSq, uint32_t lockFlags, bool isKinematic, SBodyVel& solverVel, SBodyTxI& txI,
                                    SBodyData& solverBodyData, float dt, bool gyroscopicForces) {
  const M33 rotation = mat_from_quat_simd(globalPose.q);  // PxMat33Padded(q)
  const V3 sqrtInvInertia{invInertia.x == 0.0f ? 0.0f : psqrt(invInertia.x), invInertia.y == 0.0f ? 0.0f : psqrt(invInertia.y),
                          invInertia.z == 0.0f ? 0.0f : psqrt(invInertia.z)};
  const V3 sqrtBodySpaceInertia{sqrtInvInertia.x == 0.f ? 0.f : 1.f / sqrtInvInertia.x, sqrtInvInertia.y == 0.f ? 0.f : 1.f / sqrtInvInertia.y,
                                sqrtInvInertia.z == 0.f ? 0.f : 1.f / sqrtInvInertia.z};
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
  v3store(lv, solverVel.lin);
  v3store(sqrtInertia * av, solverVel.ang);
  v3store(V3{0, 0, 0}, solverVel.deltaLinDt);
  v3store(V3{0, 0, 0}, solverVel.deltaAngDt);
  solverVel.lockFlags = uint16_t(lockFlags);
  solverVel.isKinematic = isKinematic;
  solverVel.maxAngVel = psqrt(maxAngVelSq);
  solverVel.partitionMask = 0;
  solverBodyData.nodeIndex = nodeIndex;
  solverBodyData.invMass = invMass;
  solverBodyData.penBiasClamp = maxDepenetrationVelocity;
  solverBodyData.maxContactImpulse = maxContactImpulse;
  solverBodyData.reportThreshold = reportThreshold;
  solverBodyData.originalLinearVelocity = lv;
  solverBodyData.originalAngularVelocity = av;
}

// DyTGSDynamics.cpp:245
SV_HD void copyToSolverBodyDataStepKinematic(const V3& linearVelocity, const V3& angularVelocity, const Tf& globalPose, float maxDepenetrationVelocity,
                                             float maxContactImpulse, uint32_t nodeIndex, float reportThreshold, float maxAngVelSq,
                                             SBodyVel& solverVel, SBodyTxI& txI, SBodyData& solverBodyData) {
  txI.body2WorldP = globalPose.p;
  txI.deltaBody2WorldQ = qid();
  txI.sqrtInvInertia = M33{V3{0, 0, 0}, V3{0, 0, 0}, V3{0, 0, 0}};
  v3store(V3{0, 0, 0}, solverVel.lin);
  v3store(V3{0, 0, 0}, solverVel.ang);
  v3store(V3{0, 0, 0}, solverVel.deltaLinDt);
  v3store(V3{0, 0, 0}, solverVel.deltaAngDt);
  solverVel.lockFlags = 0;
  solverVel.isKinematic = 1;
  solverVel.maxAngVel = psqrt(maxAngVelSq);
  solverVel.partitionMask = 0;
  solverVel.maxDynamicPartition = 0;
  solverVel.nbStaticInteractions = 0;
  solverBodyData.nodeIndex = nodeIndex;
  solverBodyData.invMass = 0.f;
  solverBodyData.penBiasClamp = maxDepenetrationVelocity;
  solverBodyData.maxContactImpulse = maxContactImpulse;
  solverBodyData.reportThreshold = reportThreshold;
  solverBodyData.originalLinearVelocity = linearVelocity;
  solverBodyData.originalAngularVelocity = angularVelocity;
}

// DyTGSDynamics.cpp:1403
SV_HD void integrateCoreStep(SBodyVel& vel, SBodyTxI& txInertia, float dt) {
  const uint32_t lockFlags = vel.lockFlags;
  if (lockFlags) {
    if (lockFlags & LOCK_LX) vel.lin[0] = 0.f;
    if (lockFlags & LOCK_LY) vel.lin[1] = 0.f;
    if (lockFlags & LOCK_LZ) vel.lin[2] = 0.f;
    if (lockFlags & LOCK_AX) vel.ang[0] = 0.f;
    if (lockFlags & LOCK_AY) vel.ang[1] = 0.f;
    if (lockFlags & LOCK_AZ) vel.ang[2] = 0.f;
  }
  const V3 linearMotionVel = v3load(vel.lin);
  const V3 delta = linearMotionVel * dt;
  const V3 unmolestedAngVel = v3load(vel.ang);
  const V3 angularMotionVel = txInertia.sqrtInvInertia * unmolestedAngVel;
  const float w2 = magSq(angularMotionVel);
  txInertia.body2WorldP += delta;
  if (w2 != 0.0f) {
    const float w = psqrt(w2);
    const float v = dt * w * 0.5f;
    float s = glibc::sinf(v);  // PxSinCos -> ::sinf/::cosf
    const float q = glibc::cosf(v);
    s /= w;
    const V3 pqr = angularMotionVel * s;
    const Q quatVel{pqr.x, pqr.y, pqr.z, 0.0f};
    Q result = quatVel * txInertia.deltaBody2WorldQ;
    result = result + txInertia.deltaBody2WorldQ * q;
    txInertia.deltaBody2WorldQ = normalized(result);
  }
  v3store(v3load(vel.deltaAngDt) + unmolestedAngVel * dt, vel.deltaAngDt);
  v3store(v3load(vel.deltaLinDt) + delta, vel.deltaLinDt);
}

// DySleep.cpp:35 updateWakeCounter + :226 sleepCheck
SV_HD float updateWakeCounter(Body& b, float dt, bool enableStabilization, const V3& motionLin, const V3& motionAng, bool hasStaticTouch) {
  const float wakeCounterResetTime = 20.0f * 0.02f;
  float wc = b.wakeCounter;
  if (enableStabilization) {
    const V3& t = b.invInertia;
    const V3 inertia{t.x > 0.0f ? 1.0f / t.x : 1.0f, t.y > 0.0f ? 1.0f / t.y : 1.0f, t.z > 0.0f ? 1.0f / t.z : 1.0f};
    const V3& sleepLinVelAcc = motionLin;
    const V3 sleepAngVelAcc = rotateInv(b.body2World.q, motionAng);
    float invMass = b.invMass;
    if (invMass == 0.0f) invMass = 1.0f;
    const float angular = dot(mulc(sleepAngVelAcc, sleepAngVelAcc), inertia) * invMass;
    const float linear = magSq(sleepLinVelAcc);
    const float frameNormalizedEnergy = 0.5f * (angular + linear);
    const float cf = hasStaticTouch ? float(pminu(10u, b.numCountedInteractions)) : 0.0f;
    const float freezeThresh = cf * b.freezeThreshold;
    b.freezeCount = pmax(b.freezeCount - dt, 0.0f);
    bool settled = true;
    float accelScale = pmin(1.0f, b.accelScale + dt);
    if (frameNormalizedEnergy >= freezeThresh) {
      settled = false;
      b.freezeCount = PXD_FREEZE_INTERVAL;
    }
    if (!hasStaticTouch) {
      accelScale = 1.0f;
      settled = false;
    }
    bool freeze = false;
    if (settled) {
      if (cf > 1.0f) {
        const float sleepDamping = PXD_SLEEP_DAMPING;
        const float sleepDampingTimesDT = sleepDamping * dt;
        const float d = 1.0f - sleepDampingTimesDT;
        b.linVel = b.linVel * d;
        b.angVel = b.angVel * d;
        accelScale = accelScale * 0.75f + 0.25f * PXD_FREEZE_SCALE;
      }
      freeze = b.freezeCount == 0.0f && frameNormalizedEnergy < (b.freezeThreshold * PXD_FREEZE_TOLERANCE);
    }
    b.accelScale = accelScale;
    const uint32_t wasFrozen = b.internalFlags & RB_FROZEN;
    uint16_t flags;
    if (freeze) {
      flags = uint16_t(RB_FROZEN);
      if (!wasFrozen) flags |= RB_FREEZE_THIS_FRAME;
      b.body2World = b.lastTransform;  // getLastCCDTransform: CCD 없음 -> 이번 스텝 시작 자세 (PxsRigidBody.h mLastTransform)
    } else {
      flags = 0;
      if (wasFrozen) flags |= RB_UNFREEZE_THIS_FRAME;
    }
    b.internalFlags = flags;
    if (wc < wakeCounterResetTime * 0.5f || wc < dt) {
      b.sleepLinVelAcc += sleepLinVelAcc;
      b.sleepAngVelAcc += sleepAngVelAcc;
      if (frameNormalizedEnergy >= b.sleepThreshold) {
        const float sleepAngular = dot(mulc(b.sleepAngVelAcc, b.sleepAngVelAcc), inertia) * invMass;
        const float sleepLinear = magSq(b.sleepLinVelAcc);
        const float normalizedEnergy = 0.5f * (sleepAngular + sleepLinear);
        const float sleepClusterFactor = float(1u + b.numCountedInteractions);
        const float threshold = sleepClusterFactor * b.sleepThreshold;
        if (normalizedEnergy >= threshold) {
          b.sleepLinVelAcc = V3{0, 0, 0};
          b.sleepAngVelAcc = V3{0, 0, 0};
          const float factor = b.sleepThreshold == 0.0f ? 2.0f : pmin(normalizedEnergy / threshold, 2.0f);
          const float oldWc = wc;
          wc = factor * 0.5f * wakeCounterResetTime + dt * (sleepClusterFactor - 1.0f);
          b.solverWakeCounter = wc;
          if (oldWc == 0.0f) b.internalFlags |= RB_ACTIVATE_THIS_FRAME;
          return wc;
        }
      }
    }
  } else {
    if (wc < wakeCounterResetTime * 0.5f || wc < dt) {
      const V3& t = b.invInertia;
      const V3 inertia{t.x > 0.0f ? 1.0f / t.x : 1.0f, t.y > 0.0f ? 1.0f / t.y : 1.0f, t.z > 0.0f ? 1.0f / t.z : 1.0f};
      const V3& sleepLinVelAcc = motionLin;
      const V3 sleepAngVelAcc = rotateInv(b.body2World.q, motionAng);
      b.sleepLinVelAcc += sleepLinVelAcc;
      b.sleepAngVelAcc += sleepAngVelAcc;
      float invMass = b.invMass;
      if (invMass == 0.0f) invMass = 1.0f;
      const float angular = dot(mulc(b.sleepAngVelAcc, b.sleepAngVelAcc), inertia) * invMass;
      const float linear = magSq(b.sleepLinVelAcc);
      const float normalizedEnergy = 0.5f * (angular + linear);
      const float clusterFactor = float(1 + b.numCountedInteractions);
      const float threshold = clusterFactor * b.sleepThreshold;
      if (normalizedEnergy >= threshold) {
        b.sleepLinVelAcc = V3{0, 0, 0};
        b.sleepAngVelAcc = V3{0, 0, 0};
        const float factor = threshold == 0.0f ? 2.0f : pmin(normalizedEnergy / threshold, 2.0f);
        const float oldWc = wc;
        wc = factor * 0.5f * wakeCounterResetTime + dt * (clusterFactor - 1.0f);
        b.solverWakeCounter = wc;
        uint16_t flags = 0;
        if (oldWc == 0.0f) flags |= RB_ACTIVATE_THIS_FRAME;
        b.internalFlags = flags;
        return wc;
      }
    }
  }
  wc = pmax(wc - dt, 0.0f);
  b.solverWakeCounter = wc;
  return wc;
}
SV_HD void sleepCheck(Body& b, float dt, bool enableStabilization, const V3& motionLin, const V3& motionAng, bool hasStaticTouch) {
  const float wc = updateWakeCounter(b, dt, enableStabilization, motionLin, motionAng, hasStaticTouch);
  if (wc == 0.0f) {
    b.internalFlags |= RB_DEACTIVATE_THIS_FRAME;
    b.sleepLinVelAcc = V3{0, 0, 0};
    b.sleepAngVelAcc = V3{0, 0, 0};
  }
}

// DyTGSDynamics.cpp:1549 copyBackBodies (몸체 하나)
SV_HD void copyBackBody(Body& b, const SBodyVel& vel, const SBodyTxI& txI, float invDt, float dt, bool enableStabilization, bool hasStaticTouch) {
  const V3 motionLin = v3load(vel.deltaLinDt) * invDt;
  const V3 motionAng = txI.sqrtInvInertia * (v3load(vel.deltaAngDt) * invDt);
  b.lastTransform = b.body2World;
  b.body2World.q = normalized(txI.deltaBody2WorldQ * b.body2World.q);
  b.body2World.p = txI.body2WorldP;
  b.linVel = v3load(vel.lin);
  b.angVel = txI.sqrtInvInertia * v3load(vel.ang);
  sleepCheck(b, dt, enableStabilization, motionLin, motionAng, hasStaticTouch);
}

}  // namespace sv
}  // namespace eng
