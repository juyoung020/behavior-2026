// PhysX 관절체 -> 우리 관절체(eng::art::Articulation) 상태 옮겨 담기 (시험·통합 비계 전용: PhysX 내부 헤더를 쓴다).
// 용도: 공식 OVD 재생(replay/ovd_replay) 안에서 PhysX 가 만든 관절체(로봇·서랍·문)를 어느 simulate 경계에서든 우리 엔진으로 바꿔 끼우기(문서 15절 S5).
//   최종 엔진은 PhysX 를 링크하지 않는다 — 이것은 비계와 시험에서만 쓴다.
// 방식: (1) 생성 순서로 링크 뼈대를 만들고 PhysX 코어 값을 비트 그대로 복사(PxsBodyCore·Dy::ArticulationJointCore·Dy::ArticulationCore,
//       흉내 관절은 공개 API 값) -> (2) 우리 addToScene -> (3) 장면 안 상태(Dy::ArticulationData 의 스텝 사이에 남는 칸, 링크 몸체, 잠 누적값)를
//       LL 순서로 덮어쓴다. 스텝마다 새로 계산되는 칸(응답 행렬·관성·내부 제약 등)은 옮기지 않는다.
// 조건: simulate 밖(fetchResults 뒤)에서 부른다. 힘줄 없음. 링크 수·dof 는 kMaxLinks/kMaxDofs 이하.
#pragma once
#include <vector>

#include "px_art_internal.h"
#include "random_art.h"
#include "PxsRigidBody.h"
#include "ScBodySim.h"
#include "NpArticulationLink.h"
#include "NpArticulationJointReducedCoordinate.h"
#include "DyFeatherstoneArticulationLink.h"

namespace artest {

inline eng::Q toEQ(const physx::PxQuat& q) { return eng::Q{q.x, q.y, q.z, q.w}; }
inline A::SV toESV(const physx::Cm::SpatialVectorF& v) { return A::SV{toE(v.top), toE(v.bottom)}; }

inline void copyBodyCore(const physx::PxsBodyCore& c, A::LinkBody& b) {
  b.body2World = toE(c.body2World);
  b.body2Actor = toE(c.body2Actor);
  b.linVel = toE(c.linearVelocity);
  b.angVel = toE(c.angularVelocity);
  b.maxAngVelSq = c.maxAngularVelocitySq;
  b.maxLinVelSq = c.maxLinearVelocitySq;
  b.linDamping = c.linearDamping;
  b.angDamping = c.angularDamping;
  b.invInertia = toE(c.inverseInertia);
  b.invMass = c.inverseMass;
  b.maxPenBias = c.maxPenBias;
  b.maxContactImpulse = c.maxContactImpulse;
  b.sleepThreshold = c.sleepThreshold;
  b.cfmScale = c.cfmScale;
  b.wakeCounter = c.wakeCounter;
  b.disableGravity = c.disableGravity;
  b.retainAccelerations = (c.mFlags & physx::PxRigidBodyFlag::eRETAIN_ACCELERATIONS) ? 1 : 0;
  b.numCountedInteractions = c.numCountedInteractions;
}
inline void copyJointCore(const physx::Dy::ArticulationJointCore& s, A::JointCore& j) {
  j.parentPose = toE(s.parentPose);
  j.childPose = toE(s.childPose);
  for (int i = 0; i < 6; ++i) {
    j.limLow[i] = s.limits[i].low;
    j.limHigh[i] = s.limits[i].high;
    const physx::PxArticulationDrive& d = s.drives[i];
    j.drives[i] = A::Drive{d.stiffness, d.damping, d.maxForce,
                           A::Envelope{d.envelope.maxEffort, d.envelope.maxActuatorVelocity, d.envelope.velocityDependentResistance,
                                       d.envelope.speedEffortGradient},
                           uint8_t(d.driveType)};
    j.targetP[i] = s.targetP[i];
    j.targetV[i] = s.targetV[i];
    j.armature[i] = s.armature[i];
    j.jointPos[i] = s.jointPos[i];
    j.jointVel[i] = s.jointVel[i];
    j.fStatic[i] = s.frictionParams[i].staticFrictionEffort;
    j.fDynamic[i] = s.frictionParams[i].dynamicFrictionEffort;
    j.fViscous[i] = s.frictionParams[i].viscousFrictionCoefficient;
    j.maxJointVelocity[i] = s.maxJointVelocity[i];
    j.dofIds[i] = s.dofIds[i];
    j.motion[i] = uint8_t(s.motion[i]);
    j.invDofIds[i] = s.invDofIds[i];
  }
  j.frictionCoefficient = s.frictionCoefficient;
  j.jointOffset = s.jointOffset;
  j.jCalcUpdateFrames = s.jCalcUpdateFrames ? 1 : 0;
  j.jointType = uint8_t(s.jointType);
}

// links: 생성 순서(우리 쪽 생성 번호가 된다). 비우면 px->getLinks() 순서.
// 반환: 성공 여부. 실패 이유는 out.err (ERR_*) 또는 false + 부모 없음.
inline bool snapshotFromPx(physx::PxArticulationReducedCoordinate* pxArt, A::Articulation& out, const A::SceneScale& sc,
                           std::vector<physx::PxArticulationLink*> links = {}) {
  using namespace physx;
  auto* np = static_cast<NpArticulationReducedCoordinate*>(pxArt);
  Dy::FeatherstoneArticulation* fa = llArticulation(pxArt);
  if (!fa) return false;
  Dy::ArticulationData& d = fa->mArticulationData;
  const PxU32 n = pxArt->getNbLinks();
  if (links.empty()) {
    links.resize(n);
    pxArt->getLinks(links.data(), n);
  }
  if (links.size() != n || n > A::kMaxLinks) return false;
  auto idxOf = [&](const PxArticulationLink* l) -> uint32_t {
    for (uint32_t i = 0; i < n; ++i)
      if (links[i] == l) return i;
    return A::kNone;
  };
  // (1) 뼈대 + 코어 값 (생성 번호 칸)
  A::createArticulation(out, sc);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t parent = A::kNone;
    if (PxArticulationJointReducedCoordinate* j = links[i]->getInboundJoint()) {
      parent = idxOf(&j->getParentArticulationLink());
      if (parent == A::kNone || parent >= i) return false;  // 부모가 먼저 만들어져 있어야 한다
    }
    if (A::createLink(out, parent, toE(links[i]->getGlobalPose()), sc) != i) return false;
  }
  for (uint32_t i = 0; i < n; ++i) {
    auto* nl = static_cast<NpArticulationLink*>(links[i]);
    copyBodyCore(nl->getCore().getCore(), out.bodies[i]);
    if (i) copyJointCore(static_cast<NpArticulationJointReducedCoordinate*>(links[i]->getInboundJoint())->getCore().getCore(), out.joints[i]);
  }
  const Dy::ArticulationCore& ac = np->getCore().getCore();
  out.solverIterationCounts = ac.solverIterationCounts;
  out.flags = uint8_t(PxU8(ac.flags) & (A::AF_FIX_BASE | A::AF_DRIVE_LIMITS_ARE_FORCES | A::AF_DISABLE_SELF_COLLISION));
  out.sleepThreshold = ac.sleepThreshold;
  out.freezeThreshold = ac.freezeThreshold;
  out.wakeCounter = ac.wakeCounter;
  // 흉내 관절 (공개 API 값 = 우리 createMimicJoint 인자, 만든 순서)
  const PxU32 nm = pxArt->getNbMimicJoints();
  std::vector<PxArticulationMimicJoint*> mj(nm);
  if (nm) pxArt->getMimicJoints(mj.data(), nm);
  for (PxArticulationMimicJoint* m : mj) {
    const uint32_t la = idxOf(&m->getJointA().getChildArticulationLink()), lb = idxOf(&m->getJointB().getChildArticulationLink());
    A::createMimicJoint(out, la, uint8_t(m->getAxisA()), lb, uint8_t(m->getAxisB()), m->getGearRatio(), m->getOffset(), m->getNaturalFrequency(),
                        m->getDampingRatio());
  }
  // (2) 장면에 넣기 (LL 순서·dof 배치·경로표)
  if (!A::addToScene(out)) return false;
  for (uint32_t i = 0; i < n; ++i)
    if (out.ll[i] != links[i]->getLinkIndex()) return false;  // LL 순서가 PhysX 와 같아야 한다
  if (out.dofs != d.getDofs()) return false;
  // (3) 장면 안 상태 (LL 순서)
  for (uint32_t l = 0; l < n; ++l) {
    const Dy::ArticulationLink& dl = d.mLinks[l];
    copyBodyCore(*dl.bodyCore, out.bodies[l]);
    if (l) copyJointCore(*dl.inboundJoint, out.joints[l]);
    const PxsRigidBody& rb = static_cast<NpArticulationLink*>(links[out.creation[l]])->getCore().getSim()->getLowLevelBody();
    out.bodies[l].sleepLinVelAcc = toE(rb.mSleepLinVelAcc);
    out.bodies[l].sleepAngVelAcc = toE(rb.mSleepAngVelAcc);
    out.links[l].cfm = dl.cfm;
    if (d.mLinksData) out.linkMaxPenBias[l] = d.mLinksData[l].maxPenBias;
    if (d.mMotionVelocities.size() > l) out.motionVelocities[l] = toESV(d.mMotionVelocities[l]);
    if (d.mPosIterMotionVelocities.size() > l) out.posIterMotionVelocities[l] = toESV(d.mPosIterMotionVelocities[l]);
    if (d.mMotionAccelerations.size() > l) out.motionAccelerations[l] = toESV(d.mMotionAccelerations[l]);
    if (d.mMotionAccelerationsInternal.size() > l) out.motionAccelerationsInternal[l] = toESV(d.mMotionAccelerationsInternal[l]);
    if (d.mSolverLinkSpatialDeltaVels.size() > l) out.solverLinkSpatialDeltaVels[l] = toESV(d.mSolverLinkSpatialDeltaVels[l]);
    if (d.mSolverLinkSpatialImpulses.size() > l) out.solverLinkSpatialImpulses[l] = toESV(d.mSolverLinkSpatialImpulses[l]);
    if (d.mDeltaMotionVector.size() > l) out.deltaMotion[l] = toESV(d.mDeltaMotionVector[l]);
    if (d.mPreTransform.size() > l) out.preTransform[l] = toE(d.mPreTransform[l]);
    if (d.mAccumulatedPoses.size() > l) out.accumulatedPoses[l] = toE(d.mAccumulatedPoses[l]);
    if (d.mDeltaQ.size() > l) out.deltaQ[l] = toEQ(d.mDeltaQ[l]);
    out.externalAcceleration[l] = d.mExternalAcceleration
                                      ? A::LinAng{toE(d.mExternalAcceleration[l].linear), toE(d.mExternalAcceleration[l].angular)}
                                      : A::LinAng{eng::V3{0, 0, 0}, eng::V3{0, 0, 0}};
  }
  for (uint32_t i = 0; i < out.dofs; ++i) {
    if (d.mJointAcceleration.size() > i) out.jointAcceleration[i] = d.mJointAcceleration[i];
    if (d.mJointInternalAcceleration.size() > i) out.jointInternalAcceleration[i] = d.mJointInternalAcceleration[i];
    if (d.mJointVelocity.size() > i) out.jointVelocity[i] = d.mJointVelocity[i];
    if (d.mJointNewVelocity.size() > i) out.jointNewVelocity[i] = d.mJointNewVelocity[i];
    if (d.mJointPosition.size() > i) out.jointPosition[i] = d.mJointPosition[i];
    if (d.mJointForce.size() > i) out.jointForce[i] = d.mJointForce[i];
    if (d.mJointTargetPositions.size() > i) out.jointTargetPositions[i] = d.mJointTargetPositions[i];
    if (d.mJointTargetVelocities.size() > i) out.jointTargetVelocities[i] = d.mJointTargetVelocities[i];
    if (d.mPosIterJointVelocities.size() > i) out.posIterJointVelocities[i] = d.mPosIterJointVelocities[i];
    if (d.mDeferredQstZ.size() > i) out.deferredQstZ[i] = d.mDeferredQstZ[i];
  }
  out.rootPreMotionVelocity = toESV(d.mRootPreMotionVelocity);
  out.rootDeferredZ = toESV(d.mRootDeferredZ);
  out.com = toE(d.mCOM);
  out.invSumMass = d.mInvSumMass;
  out.dt = d.mDt;
  out.dataDirty = d.mDataDirty ? 1 : 0;
  out.jointDirty = d.mJointDirty ? 1 : 0;
  out.jcalcDirty = fa->mJcalcDirty ? 1 : 0;
  out.wakeCounter = ac.wakeCounter;
  out.awake = pxArt->isSleeping() ? 0 : 1;
  // 잠 대기: 앞 simulate 의 sleepCheck 가 비활성 요청을 냈으면(관절체 깸 카운터 0 인데 아직 깨어 있음) 다음 simulate 를 풀고 재운다 (문서 16.2)
  out.readyForSleep = (out.awake && ac.wakeCounter == 0.0f) ? 1 : 0;
  return out.err == 0;
}

}  // namespace artest
