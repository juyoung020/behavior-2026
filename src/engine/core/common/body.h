// 공통: 동적 강체 상태와 생성·질량 API (모든 모듈이 쓰는 몸체 표현). 담당: 리드 (docs/엔진_자체구현.md 12절 모듈 경계).
// PhysX 5.6.1 과 같은 입력 -> 같은 내부 상태. 원본:
//   physx/src/NpPhysics.cpp:522, NpRigidBodyTemplate.h:396-470, simulationcontroller/src/ScBodyCore.cpp:45-110,
//   lowlevel/api/include/PxvDynamics.h:139-175 (PxsBodyCore::init), software/include/PxsRigidBody.h:63, ScBodySim.cpp:62
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
EHD Body createRigidDynamic(const Tf& pose, const SceneParams& sp) {
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
EHD void setCMassLocalPose(Body& b, const Tf& pose) {  // NpRigidDynamic.cpp:185 -> ScBodyCore.cpp:98
  const Tf p = normalized(pose);
  const Tf oldActor2World = b.body2World * inverse(b.body2Actor);
  b.body2World = oldActor2World * p;
  b.body2Actor = p;
}
EHD void setMass(Body& b, float m) { b.invMass = m > 0.0f ? 1.0f / m : 0.0f; }
EHD void setMassSpaceInertiaTensor(Body& b, const V3& m) {
  b.invInertia = V3{m.x == 0.0f ? 0.0f : 1.0f / m.x, m.y == 0.0f ? 0.0f : 1.0f / m.y, m.z == 0.0f ? 0.0f : 1.0f / m.z};
}
EHD Tf getGlobalPose(const Body& b) { return b.body2World * inverse(b.body2Actor); }  // NpRigidDynamic.h:66

}  // namespace eng
