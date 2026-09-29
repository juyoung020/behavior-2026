// 엔진 강체 (손으로 짬, 1단계: 접촉·조인트 없는 자유 강체). PhysX 5.6.1 과 비트 동일이 목표.
// 옮긴 원본 (태그 107.3-omni-and-physx-5.6.1, physx/source/ 기준)
//   생성·API : physx/src/NpPhysics.cpp:522, NpRigidBodyTemplate.h:396-470, simulationcontroller/src/ScBodyCore.cpp:45-110
//   초기값   : lowlevel/api/include/PxvDynamics.h:139-175 (PxsBodyCore::init), software/include/PxsRigidBody.h:63
//   적분     : lowleveldynamics/src/DyBodyCoreIntegrator.h:39 (중력·감쇠·속도 제한)
//              DyTGSDynamics.cpp:154 (copyToSolverBodyDataStep), :1403 (integrateCoreStep), :1549 (copyBackBodies),
//              :2515-2572 (제약이 하나도 없는 묶음은 전체 dt 로 한 번만 적분)
//   잠       : DySleep.cpp:34-236 (updateWakeCounter, sleepCheck), simulationcontroller/src/ScScene.cpp:176 (wakeCounter 확정)
// 아직 안 옮긴 것: 잠든 몸체 비활성화(섬 관리자), 접촉·조인트, 운동학(kinematic), CCD, 외력(addForce).
#pragma once
#include <cstdint>

#include "pmath.h"

namespace eng {

enum : uint16_t { LOCK_LX = 1, LOCK_LY = 2, LOCK_LZ = 4, LOCK_AX = 8, LOCK_AY = 16, LOCK_AZ = 32 };

struct Body {
  // PxsBodyCore
  Tf body2World;
  Tf body2Actor;
  V3 linVel, angVel;
  float maxAngVelSq, maxLinVelSq, linDamping, angDamping;
  V3 invInertia;
  float invMass;
  float maxContactImpulse, maxPenBias, sleepThreshold, freezeThreshold;
  float wakeCounter, solverWakeCounter;
  uint32_t numCountedInteractions;
  uint16_t lockFlags;
  uint8_t disableGravity;
  uint8_t gyroscopic;          // PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES
  uint16_t solverIterationCounts;
  // PxsRigidBody
  float freezeCount, accelScale;
  V3 sleepLinVelAcc, sleepAngVelAcc;
  Tf lastTransform;
  uint16_t internalFlags;
};

struct SceneParams {
  V3 gravity;
  float speedScale = 10.0f;    // PxTolerancesScale::speed
  bool stabilization = false;  // PxSceneFlag::eENABLE_STABILIZATION
};

// ---- 생성·API (PhysX 와 같은 입력 -> 같은 내부 상태)
inline Body createRigidDynamic(const Tf& pose, const SceneParams& sp) {
  Body b{};
  b.body2World = normalized(pose);                  // NpPhysics.cpp:522 globalPose.getNormalized()
  b.body2Actor = Tf{qid(), V3{0, 0, 0}};
  b.linVel = V3{0, 0, 0};
  b.angVel = V3{0, 0, 0};
  b.maxAngVelSq = 100.0f * 100.0f;                  // ScBodyCore.cpp:52 (동적 강체)
  b.maxLinVelSq = 1e32f;
  b.linDamping = 0.0f;
  b.angDamping = 0.05f;
  b.invInertia = V3{1, 1, 1};
  b.invMass = 1.0f;
  b.maxContactImpulse = 1e32f;
  b.maxPenBias = -1e32f;
  b.sleepThreshold = 5e-5f * sp.speedScale * sp.speedScale;
  b.freezeThreshold = 2.5e-5f * sp.speedScale * sp.speedScale;
  b.wakeCounter = 20.0f * 0.02f;                    // Sc::Physics::sWakeCounterOnCreation
  b.solverWakeCounter = 0.0f;
  b.numCountedInteractions = 0;
  b.lockFlags = 0;
  b.disableGravity = 0;
  b.gyroscopic = 0;
  b.solverIterationCounts = (1 << 8) | 4;
  b.freezeCount = 1.5f;                             // ScBodySim.cpp:62 mLLBody(&core, PX_FREEZE_INTERVAL)
  b.accelScale = 1.0f;
  b.sleepLinVelAcc = V3{0, 0, 0};
  b.sleepAngVelAcc = V3{0, 0, 0};
  b.lastTransform = b.body2World;
  b.internalFlags = 0;
  return b;
}
inline void setCMassLocalPose(Body& b, const Tf& pose) {  // NpRigidDynamic.cpp:185 -> ScBodyCore.cpp:98
  const Tf p = normalized(pose);
  const Tf oldActor2World = b.body2World * inverse(b.body2Actor);
  b.body2World = oldActor2World * p;
  b.body2Actor = p;
}
inline void setMass(Body& b, float m) { b.invMass = m > 0.0f ? 1.0f / m : 0.0f; }
inline void setMassSpaceInertiaTensor(Body& b, const V3& m) {
  b.invInertia = V3{m.x == 0.0f ? 0.0f : 1.0f / m.x, m.y == 0.0f ? 0.0f : 1.0f / m.y, m.z == 0.0f ? 0.0f : 1.0f / m.z};
}
inline Tf getGlobalPose(const Body& b) { return b.body2World * inverse(b.body2Actor); }  // NpRigidDynamic.h:66

// ---- 한 simulate: 제약·접촉이 없는 묶음 경로 (DyTGSDynamics.cpp:2527-2570)
inline void stepFree(Body& b, const SceneParams& sp, float dt) {
  const float invDt = 1.0f / dt;
  // 1) preIntegrate: bodyCoreComputeUnconstrainedVelocity (DyBodyCoreIntegrator.h:39)
  {
    V3 lv = b.linVel, av = b.angVel;
    const float linearDampingTimesDT = b.linDamping * dt;
    const float angularDampingTimesDT = b.angDamping * dt;
    const float oneMinusLin = 1.0f - linearDampingTimesDT;
    const float oneMinusAng = 1.0f - angularDampingTimesDT;
    if (!b.disableGravity) {
      const V3 linearAccelTimesDT = sp.gravity * dt * b.accelScale;
      lv += linearAccelTimesDT;
    }
    const float linVelMultiplier = fsel(oneMinusLin, oneMinusLin, 0.0f);
    const float angVelMultiplier = fsel(oneMinusAng, oneMinusAng, 0.0f);
    lv *= linVelMultiplier;
    av *= angVelMultiplier;
    const float linVelSq = magSq(lv);
    if (linVelSq > b.maxLinVelSq) lv *= std::sqrt(b.maxLinVelSq / linVelSq);
    const float angVelSq = magSq(av);
    if (angVelSq > b.maxAngVelSq) av *= std::sqrt(b.maxAngVelSq / angVelSq);
    b.linVel = lv;
    b.angVel = av;
  }
  // 2) copyToSolverBodyDataStep (DyTGSDynamics.cpp:154)
  const M33 rotation = mat_from_quat_simd(b.body2World.q);
  const V3 sqrtInvInertia{b.invInertia.x == 0.0f ? 0.0f : std::sqrt(b.invInertia.x),
                          b.invInertia.y == 0.0f ? 0.0f : std::sqrt(b.invInertia.y),
                          b.invInertia.z == 0.0f ? 0.0f : std::sqrt(b.invInertia.z)};
  const V3 sqrtBodySpaceInertia{sqrtInvInertia.x == 0.0f ? 0.0f : 1.0f / sqrtInvInertia.x,
                                sqrtInvInertia.y == 0.0f ? 0.0f : 1.0f / sqrtInvInertia.y,
                                sqrtInvInertia.z == 0.0f ? 0.0f : 1.0f / sqrtInvInertia.z};
  const M33 txSqrtInvInertia = transformInertiaTensor(sqrtInvInertia, rotation);
  V3 body2WorldP = b.body2World.p;
  Q deltaQ = qid();
  const M33 sqrtInertia = transformInertiaTensor(sqrtBodySpaceInertia, rotation);
  V3 lv = b.linVel, av = b.angVel;
  if (b.gyroscopic) {
    const V3 localInertia{b.invInertia.x == 0.0f ? 0.0f : 1.0f / b.invInertia.x, b.invInertia.y == 0.0f ? 0.0f : 1.0f / b.invInertia.y,
                          b.invInertia.z == 0.0f ? 0.0f : 1.0f / b.invInertia.z};
    const V3 localAngVel = rotateInv(b.body2World.q, av);
    const V3 origMom = mulc(localInertia, localAngVel);
    const V3 torque = -cross(localAngVel, origMom);
    V3 newMom = origMom + torque * dt;
    const float denom = mag(newMom);
    const float ratio = denom > 0.0f ? mag(origMom) / denom : 0.0f;
    newMom *= ratio;
    const V3 newDeltaAngVel = rotate(b.body2World.q, mulc(b.invInertia, newMom) - localAngVel);
    av += newDeltaAngVel;
  }
  if (b.lockFlags) {
    if (b.lockFlags & LOCK_LX) lv.x = 0.0f;
    if (b.lockFlags & LOCK_LY) lv.y = 0.0f;
    if (b.lockFlags & LOCK_LZ) lv.z = 0.0f;
    if (b.lockFlags & LOCK_AX) av.x = 0.0f;
    if (b.lockFlags & LOCK_AY) av.y = 0.0f;
    if (b.lockFlags & LOCK_AZ) av.z = 0.0f;
  }
  V3 velLin = lv;
  V3 velAng = sqrtInertia * av;
  V3 deltaLinDt{0, 0, 0}, deltaAngDt{0, 0, 0};
  // 3) integrateCoreStep(vel, txInertia, mDt) 한 번 (DyTGSDynamics.cpp:1403)
  {
    if (b.lockFlags) {
      if (b.lockFlags & LOCK_LX) velLin.x = 0.0f;
      if (b.lockFlags & LOCK_LY) velLin.y = 0.0f;
      if (b.lockFlags & LOCK_LZ) velLin.z = 0.0f;
      if (b.lockFlags & LOCK_AX) velAng.x = 0.0f;
      if (b.lockFlags & LOCK_AY) velAng.y = 0.0f;
      if (b.lockFlags & LOCK_AZ) velAng.z = 0.0f;
    }
    const V3 linearMotionVel = velLin;
    const V3 delta = linearMotionVel * dt;
    const V3 unmolestedAngVel = velAng;
    const V3 angularMotionVel = txSqrtInvInertia * velAng;
    const float w2 = magSq(angularMotionVel);
    body2WorldP += delta;
    if (w2 != 0.0f) {
      const float w = std::sqrt(w2);
      const float v = dt * w * 0.5f;
      float s = std::sin(v);                         // PxSinCos -> ::sinf / ::cosf (PxMath.h:210)
      const float q = std::cos(v);
      s /= w;
      const V3 pqr = angularMotionVel * s;
      const Q quatVel{pqr.x, pqr.y, pqr.z, 0.0f};
      Q result = quatVel * deltaQ;
      result = result + deltaQ * q;
      deltaQ = normalized(result);
    }
    deltaAngDt += unmolestedAngVel * dt;
    deltaLinDt += delta;
  }
  // 4) copyBackBodies (DyTGSDynamics.cpp:1549)
  const V3 motionLin = deltaLinDt * invDt;
  const V3 motionAng = txSqrtInvInertia * (deltaAngDt * invDt);
  b.lastTransform = b.body2World;
  b.body2World.q = normalized(deltaQ * b.body2World.q);
  b.body2World.p = body2WorldP;
  b.linVel = velLin;
  b.angVel = txSqrtInvInertia * velAng;
  // 5) sleepCheck -> updateWakeCounter (DySleep.cpp), 정적 접촉 없음(hasStaticTouch = 0)
  {
    const float wakeCounterResetTime = 20.0f * 0.02f;
    float wc = b.wakeCounter;
    bool done = false;
    const V3& t = b.invInertia;
    const V3 inertia{t.x > 0.0f ? 1.0f / t.x : 1.0f, t.y > 0.0f ? 1.0f / t.y : 1.0f, t.z > 0.0f ? 1.0f / t.z : 1.0f};
    float invMass = b.invMass;
    if (invMass == 0.0f) invMass = 1.0f;
    if (sp.stabilization) {
      const V3 sleepLinVelAcc = motionLin;
      const V3 sleepAngVelAcc = rotateInv(b.body2World.q, motionAng);
      const float angular = dot(mulc(sleepAngVelAcc, sleepAngVelAcc), inertia) * invMass;
      const float linear = magSq(sleepLinVelAcc);
      const float frameNormalizedEnergy = 0.5f * (angular + linear);
      const float cf = 0.0f;                       // hasStaticTouch = 0
      const float freezeThresh = cf * b.freezeThreshold;
      b.freezeCount = pmax(b.freezeCount - dt, 0.0f);
      float accelScale = pmin(1.0f, b.accelScale + dt);
      if (frameNormalizedEnergy >= freezeThresh) b.freezeCount = 1.5f;  // PXD_FREEZE_INTERVAL
      accelScale = 1.0f;                           // !hasStaticTouch
      b.accelScale = accelScale;
      b.internalFlags = 0;                         // freeze = false, wasFrozen 은 이 단계에서 없음
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
            wc = factor * 0.5f * wakeCounterResetTime + dt * (sleepClusterFactor - 1.0f);
            b.solverWakeCounter = wc;
            done = true;
          }
        }
      }
    } else {
      if (wc < wakeCounterResetTime * 0.5f || wc < dt) {
        const V3 sleepLinVelAcc = motionLin;
        const V3 sleepAngVelAcc = rotateInv(b.body2World.q, motionAng);
        b.sleepLinVelAcc += sleepLinVelAcc;
        b.sleepAngVelAcc += sleepAngVelAcc;
        const float angular = dot(mulc(b.sleepAngVelAcc, b.sleepAngVelAcc), inertia) * invMass;
        const float linear = magSq(b.sleepLinVelAcc);
        const float normalizedEnergy = 0.5f * (angular + linear);
        const float clusterFactor = float(1 + b.numCountedInteractions);
        const float threshold = clusterFactor * b.sleepThreshold;
        if (normalizedEnergy >= threshold) {
          b.sleepLinVelAcc = V3{0, 0, 0};
          b.sleepAngVelAcc = V3{0, 0, 0};
          const float factor = threshold == 0.0f ? 2.0f : pmin(normalizedEnergy / threshold, 2.0f);
          wc = factor * 0.5f * wakeCounterResetTime + dt * (clusterFactor - 1.0f);
          b.solverWakeCounter = wc;
          b.internalFlags = 0;
          done = true;
        }
      }
    }
    if (!done) {
      wc = pmax(wc - dt, 0.0f);
      b.solverWakeCounter = wc;
    }
    if (wc == 0.0f) {  // 비활성화 표시 (섬 관리자 처리는 아직 안 옮김)
      b.sleepLinVelAcc = V3{0, 0, 0};
      b.sleepAngVelAcc = V3{0, 0, 0};
    }
  }
  // 6) ScAfterIntegrationTask: 깸 카운터 확정 (ScScene.cpp:176)
  b.wakeCounter = b.solverWakeCounter;
}

}  // namespace eng
