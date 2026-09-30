// G1 비계: PhysX 내부 값 -> 우리 자료형 (g1_solver.cpp · g1_dump.cpp 공용). PhysX 내부 헤더(tests/solver/px_internal.h) 뒤에 넣는다.
#pragma once
#include "core/common/body.h"

namespace g1px {
using namespace physx;
inline eng::V3 toV(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
inline eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }

inline eng::Body bodyFrom(const PxsRigidBody& rb) {
  const PxsBodyCore& c = rb.getCore();
  eng::Body b{};
  b.body2World = toE(c.body2World);
  b.body2Actor = toE(c.getBody2Actor());
  b.linVel = toV(c.linearVelocity);
  b.angVel = toV(c.angularVelocity);
  b.maxAngVelSq = c.maxAngularVelocitySq;
  b.maxLinVelSq = c.maxLinearVelocitySq;
  b.linDamping = c.linearDamping;
  b.angDamping = c.angularDamping;
  b.invInertia = toV(c.inverseInertia);
  b.invMass = c.inverseMass;
  b.maxContactImpulse = c.maxContactImpulse;
  b.maxPenBias = c.maxPenBias;
  b.sleepThreshold = c.sleepThreshold;
  b.freezeThreshold = c.freezeThreshold;
  b.wakeCounter = c.wakeCounter;
  b.solverWakeCounter = c.solverWakeCounter;
  b.numCountedInteractions = c.numCountedInteractions;
  b.lockFlags = uint16_t(PxU8(c.lockFlags));
  b.disableGravity = c.disableGravity;
  b.gyroscopic = (c.mFlags & PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES) ? 1 : 0;
  b.solverIterationCounts = c.solverIterationCounts;
  b.freezeCount = rb.mFreezeCount;
  b.accelScale = rb.mAccelScale;
  b.sleepLinVelAcc = toV(rb.mSleepLinVelAcc);
  b.sleepAngVelAcc = toV(rb.mSleepAngVelAcc);
  b.lastTransform = toE(rb.mLastTransform);
  b.internalFlags = rb.mInternalFlags;
  return b;
}

}  // namespace g1px
