// articulation 모듈: 관절체 한 simulate (TGS). PhysX 5.6.1 과 연산 순서 그대로. 손으로 짬, PhysX 링크 없음.
// 담당: articulation 작업자 (docs/엔진_자체구현.md 14절). solver 가 부르는 함수 목록은 Dy::ArticulationPImpl(DyArticulationPImpl.h)과 같다:
//   computeUnconstrainedVelocitiesTGS / setupSolverConstraintsTGS / solveInternalConstraints / recordDeltaMotion(updateDeltaMotion) /
//   saveVelocityTGS / concludeInternalConstraints / writebackInternalConstraints / updateBodiesTGS / applyTgsSubstepForces,
//   결합점: getImpulseResponse / getImpulseSelfResponse / pxcFsApplyImpulse(s) / pxcFsGetVelocity / getLinkVelocity / getDeltaQ
// 원본 (physx/source/lowleveldynamics/src 기준)
//   DyFeatherstoneForwardDynamic.cpp:88 (전파 관성·ZA), :661 (computeArticulatedSpatialInertiaAndZ), :856 (응답 행렬),
//     :987-1358 (링크 가속도·내부 가속도·운동량 보존), :1479 (computeUnconstrainedVelocitiesTGS), :1582 (updateArticulation),
//     :1805 (computeUnconstrainedVelocitiesInternal), :1866-1983 (관절 위치 강제), :2260 (updateBodies)
//   DyFeatherstoneArticulation.cpp:975 (PxcFsFlushVelocity), :1037 (applyTgsSubstepForces), :1114 (recordDeltaMotion),
//     :1229 (pxcFsGetVelocity), :1400/1436 (pxcFsApplyImpulse(s)), :1581 (propagateTransform), :1888 (saveVelocityTGS),
//     :2015 (getImpulseSelfResponse), :2432 (setupInternalConstraintsRecursive), :3032 (setupInternalConstraints),
//     :3681 (computeRelativeTransformC2P), :3986 (computeLinkStates), :4560 (solveInternalJointConstraintRecursive), :4873, :5313
//   DyFeatherstoneInverseDynamic.cpp:116 (computeZAForceInv), DyArticulationMimicJoint.cpp (흉내 관절)
//   DyTGSDynamics.cpp:2531-2580 (접촉 묶음이 없는 섬의 관절체 풀이), :1898-1900 (stepDt·편향 계수), :1643 (invLengthScale)
//   simulationcontroller/src/ScArticulationSim.cpp:432 (sleepCheck), ScBodySim.cpp:581 (updateWakeCounter)
#pragma once
#include <cstdint>

#include "../common/glibc_trig.h"
#include "articulation.h"

namespace eng {
namespace art {

// ---------------------------------------------------------------- 정적 접촉/1D 행 자리 (solver/joints 가 채움)
// PhysX 는 링크와 정적 물체 사이 제약을 관절체 내부 풀이에서 푼다(DY_STATIC_CONTACTS_IN_INTERNAL_SOLVER = true,
// DyFeatherstoneArticulation.cpp:4323 solveStaticConstraint). 여기서는 링크마다 개수만 두고, 푸는 함수는 호출자가 준다.
struct NoStatic {
  EHD uint32_t count1D(const Articulation&, uint32_t) const { return 0; }
  EHD uint32_t countContact(const Articulation&, uint32_t) const { return 0; }
  // solve(desc 번호, 링크 속도 v, 충격 누적 imp, deltaV, 운동량, 회전, isTGS, elapsed, minPen, 위치반복?) — 없음
  template <class... T>
  EHD void solve1D(T&&...) const {}
  template <class... T>
  EHD void solveContact(T&&...) const {}
};

// ---------------------------------------------------------------- 결합점 (접촉·조인트가 관절체 링크에 붙을 때)
// getImpulseResponse(linkID, SpatialVector impulse) (DyFeatherstoneArticulation.cpp:859): 충격 (선형, 회전) -> 속도 변화 (선형, 회전)
EHD void getImpulseResponse(const Articulation& a, uint32_t linkID, const V3& impLinear, const V3& impAngular, V3& dvLinear,
                            V3& dvAngular) {
  const SV dv = responseOf(a.responseW[linkID], SV{impLinear, impAngular});
  dvLinear = dv.bottom;
  dvAngular = dv.top;
}
EHD SV getImpulseResponseW(const Articulation& a, uint32_t linkID, const SV& impulse) { return responseOf(a.responseW[linkID], impulse); }

// getImpulseSelfResponse (부모-자식 쌍, :2015). 충격·결과는 (선형, 회전) 쌍.
EHD void getImpulseSelfResponse(const Articulation& a, uint32_t linkID0, const V3& lin0, const V3& ang0, V3& dv0Lin, V3& dv0Ang,
                                uint32_t linkID1, const V3& lin1, const V3& ang1, V3& dv1Lin, V3& dv1Ang) {
  const Link& link = a.links[linkID1];
  if (link.parent == linkID0) {
    const SV imp1{lin1, ang1};
    const SV imp0{lin0, ang0};
    const SV Z1W{-imp1.top, -imp1.bottom};
    const uint32_t jointOffset1 = a.jointData[linkID1].jointOffset;
    const uint8_t dofCount1 = a.jointData[linkID1].nbDof;
    float qstZ[3] = {0.f, 0.f, 0.f};
    const SV Z0W = propagateImpulseW(a.rw[linkID1], Z1W, nullptr, &a.isInvStIS[jointOffset1], &a.worldMotionMatrix[jointOffset1],
                                     dofCount1, qstZ);
    const SV impulseDifW = imp0 - Z0W;
    const SV delV0W = getImpulseResponseW(a, linkID0, impulseDifW);
    const SV delV1W = propagateAccelerationW(a.rw[linkID1], delV0W, a.invStIs[linkID1], &a.worldMotionMatrix[jointOffset1],
                                             &a.isW[jointOffset1], qstZ, dofCount1, nullptr);
    dv0Lin = delV0W.bottom;
    dv0Ang = delV0W.top;
    dv1Lin = delV1W.bottom;
    dv1Ang = delV1W.top;
  } else {
    // getImpulseResponseSlow (:1916): 공통 조상까지 올렸다가 내린다 (같은 관절체의 두 링크 사이 접촉·조인트)
    uint32_t stack[kMaxLinks];
    uint32_t i0, i1;
    uint32_t id0 = linkID0, id1 = linkID1;
    for (i0 = id0, i1 = id1; i0 != i1;) {
      if (i0 < i1) i1 = a.links[i1].parent;
      else i0 = a.links[i0].parent;
    }
    const uint32_t common = i0;
    SV Z0{-lin0, -ang0};
    SV Z1{-lin1, -ang1};
    float qstZ[kMaxDofs];
    for (uint32_t d = 0; d < a.dofs; ++d) qstZ[d] = 0.0f;
    for (i0 = 0; id0 != common; id0 = a.links[id0].parent) {
      const uint32_t jointOffset = a.jointData[id0].jointOffset;
      const uint8_t dofCount = a.jointData[id0].nbDof;
      Z0 = propagateImpulseW(a.rw[id0], Z0, nullptr, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount, &qstZ[jointOffset]);
      stack[i0++] = id0;
    }
    for (i1 = i0; id1 != common; id1 = a.links[id1].parent) {
      const uint32_t jointOffset = a.jointData[id1].jointOffset;
      const uint8_t dofCount = a.jointData[id1].nbDof;
      Z1 = propagateImpulseW(a.rw[id1], Z1, nullptr, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount, &qstZ[jointOffset]);
      stack[i1++] = id1;
    }
    const SV ZZ = Z0 + Z1;
    const SV v = responseOf(a.responseW[common], -ZZ);
    SV dv1 = v;
    for (uint32_t index = i1; (index--) > i0;) {
      const uint32_t id = stack[index];
      const uint32_t jointOffset = a.jointData[id].jointOffset;
      const uint32_t dofCount = a.jointData[id].nbDof;
      dv1 = propagateAccelerationW(a.rw[id], dv1, a.invStIs[id], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset], &qstZ[jointOffset], dofCount,
                                   nullptr);
    }
    SV dv0 = v;
    for (uint32_t index = i0; (index--) > 0;) {
      const uint32_t id = stack[index];
      const uint32_t jointOffset = a.jointData[id].jointOffset;
      const uint32_t dofCount = a.jointData[id].nbDof;
      dv0 = propagateAccelerationW(a.rw[id], dv0, a.invStIs[id], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset], &qstZ[jointOffset], dofCount,
                                   nullptr);
    }
    dv0Lin = dv0.bottom;
    dv0Ang = dv0.top;
    dv1Lin = dv1.bottom;
    dv1Ang = dv1.top;
  }
}

// 링크 상태 조회 (:905-973): 풀이 전 속도(월드), TGS 누적 운동량, 회전 변화, 깊이 뚫림 한계, cfm, 가속도
EHD void getLinkVelocity(const Articulation& a, uint32_t linkID, V3& linear, V3& angular) {
  linear = a.motionVelocities[linkID].bottom;
  angular = a.motionVelocities[linkID].top;
}
EHD void getLinkMotionVector(const Articulation& a, uint32_t linkID, V3& linear, V3& angular) {
  linear = a.deltaMotion[linkID].bottom;
  angular = a.deltaMotion[linkID].top;
}
EHD const Q& getDeltaQ(const Articulation& a, uint32_t linkID) { return a.deltaQ[linkID]; }
EHD float getLinkMaxPenBias(const Articulation& a, uint32_t linkID) { return a.linkMaxPenBias[linkID]; }
EHD float getCfm(const Articulation& a, uint32_t linkID) { return a.links[linkID].cfm; }
EHD void getMotionAcceleration(const Articulation& a, uint32_t linkID, V3& linear, V3& angular) {  // :944 (CPU)
  linear = V3{0, 0, 0};
  angular = V3{0, 0, 0};
  if (a.dt > 0.0f) {
    const float invDt = 1.0f / a.dt;
    const SV linkAccel = a.motionAccelerations[linkID] + a.solverLinkSpatialDeltaVels[linkID] * invDt;
    linear = linkAccel.bottom;
    angular = linkAccel.top;
  }
}

// PxcFsFlushVelocity (:975)
EHD void flushVelocity(Articulation& a, SV* deltaV) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  if (fixBase) {
    deltaV[0] = svzero();
  } else {
    deltaV[0] = a.baseInvSpatialArticulatedInertiaW * -a.rootDeferredZ;
    a.motionVelocities[0] += deltaV[0];
    a.solverLinkSpatialDeltaVels[0] += deltaV[0];
  }
  for (uint32_t i = 1; i < a.nLinks; i++) {
    const Link& tLink = a.links[i];
    const JointData& jd = a.jointData[i];
    const SV dV = propagateAccelerationW(a.rw[i], deltaV[tLink.parent], a.invStIs[i], &a.worldMotionMatrix[jd.jointOffset],
                                         &a.isW[jd.jointOffset], &a.deferredQstZ[jd.jointOffset], jd.nbDof,
                                         &a.jointNewVelocity[jd.jointOffset]);
    deltaV[i] = dV;
    a.motionVelocities[i] += dV;
    a.solverLinkSpatialDeltaVels[i] += dV;
  }
  for (uint32_t i = 0; i < a.dofs; ++i) a.deferredQstZ[i] = 0.0f;
  a.rootDeferredZ = svzero();
}

// pxcFsGetVelocity (:1229) -> (선형, 회전)
EHD void pxcFsGetVelocity(const Articulation& a, uint32_t linkID, float* jointDofSpeeds, V3& linear, V3& angular) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  SV deltaV = svzero();
  if (!fixBase) deltaV = a.baseInvSpatialArticulatedInertiaW * -a.rootDeferredZ;
  const uint32_t startIndex = a.links[linkID].pathStart;
  const uint32_t elementCount = a.links[linkID].pathCount;
  const uint32_t elementCountMinusOne = (0 == elementCount) ? 0 : elementCount - 1;
  const uint32_t* pathToRootElements = &a.pathToRoot[startIndex];
  for (uint32_t i = 0; i < elementCountMinusOne; ++i) {
    const uint32_t index = pathToRootElements[i];
    const uint32_t jointOffset = a.jointData[index].jointOffset;
    const uint32_t dofCount = a.jointData[index].nbDof;
    deltaV = propagateAccelerationW(a.rw[index], deltaV, a.invStIs[index], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                                    &a.deferredQstZ[jointOffset], dofCount, nullptr);
  }
  float deltaJointDofSpeeds[3] = {0, 0, 0};
  float* optional = jointDofSpeeds ? deltaJointDofSpeeds : nullptr;
  for (uint32_t i = elementCountMinusOne; i < elementCount; ++i) {
    const uint32_t index = pathToRootElements[i];
    const uint32_t jointOffset = a.jointData[index].jointOffset;
    const uint32_t dofCount = a.jointData[index].nbDof;
    deltaV = propagateAccelerationW(a.rw[index], deltaV, a.invStIs[index], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                                    &a.deferredQstZ[jointOffset], dofCount, optional);
  }
  if (jointDofSpeeds) {
    const uint32_t jointOffset = a.jointData[linkID].jointOffset;
    const uint32_t dofCount = a.jointData[linkID].nbDof;
    for (uint32_t i = 0; i < dofCount; i++) jointDofSpeeds[i] = a.jointNewVelocity[jointOffset + i] + deltaJointDofSpeeds[i];
  }
  const SV vel = a.motionVelocities[linkID] + deltaV;
  linear = vel.bottom;
  angular = vel.top;
}

// pxcFsGetVelocities (:1299): 두 링크 속도를 공통 경로 한 번으로
EHD void pxcFsGetVelocities(const Articulation& a, uint32_t linkID, uint32_t linkID1, V3& lin0, V3& ang0, V3& lin1, V3& ang1) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  SV deltaV = svzero();
  if (!fixBase) deltaV = a.baseInvSpatialArticulatedInertiaW * (-a.rootDeferredZ);
  const Link& link0 = a.links[linkID];
  const Link& link1 = a.links[linkID1];
  const uint32_t* pathToRoot0 = &a.pathToRoot[link0.pathStart];
  const uint32_t* pathToRoot1 = &a.pathToRoot[link1.pathStart];
  const uint32_t numElems0 = link0.pathCount;
  const uint32_t numElems1 = link1.pathCount;
  uint32_t offset = 0;
  while (pathToRoot0[offset] == pathToRoot1[offset]) {
    const uint32_t index = pathToRoot0[offset++];
    if (offset >= numElems0 || offset >= numElems1) break;
    const uint32_t jointOffset = a.jointData[index].jointOffset;
    const uint32_t dofCount = a.jointData[index].nbDof;
    deltaV = propagateAccelerationW(a.rw[index], deltaV, a.invStIs[index], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                                    &a.deferredQstZ[jointOffset], dofCount, nullptr);
  }
  SV deltaV1 = deltaV;
  for (uint32_t idx = offset; idx < numElems0; ++idx) {
    const uint32_t index = pathToRoot0[idx];
    const uint32_t jointOffset = a.jointData[index].jointOffset;
    const uint32_t dofCount = a.jointData[index].nbDof;
    deltaV = propagateAccelerationW(a.rw[index], deltaV, a.invStIs[index], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                                    &a.deferredQstZ[jointOffset], dofCount, nullptr);
  }
  for (uint32_t idx = offset; idx < numElems1; ++idx) {
    const uint32_t index = pathToRoot1[idx];
    const uint32_t jointOffset = a.jointData[index].jointOffset;
    const uint32_t dofCount = a.jointData[index].nbDof;
    deltaV1 = propagateAccelerationW(a.rw[index], deltaV1, a.invStIs[index], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                                     &a.deferredQstZ[jointOffset], dofCount, nullptr);
  }
  const SV vel = a.motionVelocities[linkID] + deltaV;
  lin0 = vel.bottom;
  ang0 = vel.top;
  const SV vel1 = a.motionVelocities[linkID1] + deltaV1;
  lin1 = vel1.bottom;
  ang1 = vel1.top;
}

// pxcFsApplyImpulse (:1400): 월드 충격 (선형, 회전) + 선택적 관절 충격
EHD void pxcFsApplyImpulse(Articulation& a, uint32_t linkID, const V3& linear, const V3& angular, const float* jointImpulse) {
  a.jointDirty = 1;
  SV Z0{-linear, -angular};
  for (uint32_t i = linkID; i; i = a.links[i].parent) {
    const uint32_t jointOffset = a.jointData[i].jointOffset;
    const uint8_t dofCount = a.jointData[i].nbDof;
    a.solverLinkSpatialImpulses[i] += Z0;
    const float* jointImpulseToApply = (linkID == i) ? jointImpulse : nullptr;
    Z0 = propagateImpulseW(a.rw[i], Z0, jointImpulseToApply, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount,
                           &a.deferredQstZ[jointOffset]);
  }
  a.rootDeferredZ += Z0;
}

// pxcFsApplyImpulses (:1436)
EHD void pxcFsApplyImpulses(Articulation& a, uint32_t linkID1, const V3& linear1, const V3& angular1, const float* jointImpulse1,
                            uint32_t linkID2, const V3& linear2, const V3& angular2, const float* jointImpulse2) {
  a.jointDirty = 1;
  SV Z1{-linear1, -angular1};
  SV Z2{-linear2, -angular2};
  const Link& link1 = a.links[linkID1];
  const Link& link2 = a.links[linkID2];
  const uint32_t* pathToRoot1 = &a.pathToRoot[link1.pathStart];
  const uint32_t* pathToRoot2 = &a.pathToRoot[link2.pathStart];
  const uint32_t numElems1 = link1.pathCount;
  const uint32_t numElems2 = link2.pathCount;
  uint32_t offset = 0;
  uint32_t commonLink = 0;
  while (pathToRoot1[offset] == pathToRoot2[offset]) {
    commonLink = pathToRoot1[offset++];
    if (offset >= numElems1 || offset >= numElems2) break;
  }
  for (uint32_t i = linkID2; i != commonLink; i = a.links[i].parent) {
    const uint32_t jointOffset = a.jointData[i].jointOffset;
    const uint8_t dofCount = a.jointData[i].nbDof;
    const float* jointImpulseToApply = (linkID2 == i) ? jointImpulse2 : nullptr;
    a.solverLinkSpatialImpulses[i] += Z2;
    Z2 = propagateImpulseW(a.rw[i], Z2, jointImpulseToApply, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount,
                           &a.deferredQstZ[jointOffset]);
  }
  for (uint32_t i = linkID1; i != commonLink; i = a.links[i].parent) {
    const uint32_t jointOffset = a.jointData[i].jointOffset;
    const uint8_t dofCount = a.jointData[i].nbDof;
    const float* jointImpulseToApply = (linkID1 == i) ? jointImpulse1 : nullptr;
    a.solverLinkSpatialImpulses[i] += Z1;
    Z1 = propagateImpulseW(a.rw[i], Z1, jointImpulseToApply, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount,
                           &a.deferredQstZ[jointOffset]);
  }
  float jointImpulseToApplyAtCommonLink[3] = {0, 0, 0};
  if (((linkID1 == commonLink) && jointImpulse1) || ((linkID2 == commonLink) && jointImpulse2)) {
    const uint32_t linkIndices[2] = {linkID1, linkID2};
    const float* jointImpulses[2] = {jointImpulse1, jointImpulse2};
    const uint32_t dofCountAtCommonLink = a.jointData[commonLink].nbDof;
    for (uint32_t k = 0; k < 2; k++) {
      const uint32_t linkId = linkIndices[k];
      const float* jointImpulse = jointImpulses[k];
      if ((linkId == commonLink) && jointImpulse)
        for (uint32_t i = 0; i < dofCountAtCommonLink; i++) jointImpulseToApplyAtCommonLink[i] += jointImpulse[i];
    }
  }
  SV ZCommon = Z1 + Z2;
  for (uint32_t i = commonLink; i; i = a.links[i].parent) {
    const uint32_t jointOffset = a.jointData[i].jointOffset;
    const uint8_t dofCount = a.jointData[i].nbDof;
    const float* jointImpulseToApply = (commonLink == i) ? jointImpulseToApplyAtCommonLink : nullptr;
    a.solverLinkSpatialImpulses[i] += ZCommon;
    ZCommon = propagateImpulseW(a.rw[i], ZCommon, jointImpulseToApply, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset],
                                dofCount, &a.deferredQstZ[jointOffset]);
  }
  a.rootDeferredZ += ZCommon;
}

// ---------------------------------------------------------------- 준비: computeUnconstrainedVelocitiesTGS
// computeRelativeTransformC2P (:3681)
EHD void computeRelativeTransformC2P(Articulation& a) {
  a.accumulatedPoses[0] = a.bodies[0].body2World;
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    const uint32_t jointOffset = a.jointData[linkID].jointOffset;
    const uint32_t dofCount = a.jointData[linkID].nbDof;
    const Tf& body2World = a.bodies[linkID].body2World;
    const Tf& pBody2World = a.bodies[link.parent].body2World;
    a.rw[linkID] = body2World.p - pBody2World.p;
    for (uint32_t i = 0; i < dofCount; ++i) a.worldMotionMatrix[jointOffset + i] = svrotate(body2World, a.motionMatrix[jointOffset + i]);
    a.accumulatedPoses[linkID] = body2World;
  }
}

// computeLinkStates (:3986)
EHD void computeLinkStates(Articulation& a, float dt, float invLengthScale, const V3& gravity, bool externalForcesEveryTgsIterationEnabled) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  const uint32_t linkCount = a.nLinks;
  const float invDt = dt < 1e-6f ? kMaxF32 : 1.f / dt;
  SV rootLinkVel;
  {
    const LinkBody& core0 = a.bodies[0];
    rootLinkVel = fixBase ? svzero() : SV{core0.angVel, core0.linVel};
    a.motionVelocities[0] = rootLinkVel;
    a.motionAccelerations[0] = fixBase ? svzero() : a.motionAccelerations[0];
    a.coriolis[0] = svzero();
    a.rootPreMotionVelocity = rootLinkVel;
  }
  float ratio = 1.f;
  for (uint32_t linkID = 1; linkID < linkCount; ++linkID) {
    const JointData& jd = a.jointData[linkID];
    const float* jVelocity = &a.jointVelocity[jd.jointOffset];
    for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
      const float maxJVelocity = a.joints[linkID].maxJointVelocity[ind];  // 원본도 dof 번호로 축 배열을 읽는다
      const float jVel = jVelocity[ind];
      ratio = (jVel != 0.0f) ? pmin(ratio, maxJVelocity / fabsP(jVel)) : ratio;
    }
  }
  float sumMass = 0.f;
  V3 COM{0.f, 0.f, 0.f};
  for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
    Link& link = a.links[linkID];
    const LinkBody& bodyCore = a.bodies[linkID];
    a.linkMaxPenBias[linkID] = bodyCore.maxPenBias;
    link.cfm = (fixBase && linkID == 0) ? 0.f : bodyCore.cfmScale * invLengthScale;
    const V3& ii = bodyCore.invInertia;
    const V3 inertiaTensor{ii.x == 0.f ? 0.f : (1.f / ii.x), ii.y == 0.f ? 0.f : (1.f / ii.y), ii.z == 0.f ? 0.f : (1.f / ii.z)};
    const float invMass = bodyCore.invMass;
    const float m = invMass == 0.f ? 0.f : 1.0f / invMass;
    M33 Iw;
    SMat worldArticulatedInertia;
    {
      const M33 rot = m33FromQuat(a.accumulatedPoses[linkID].q);
      Iw = transformInertiaTensor(inertiaTensor, rot);
      worldArticulatedInertia.tl = m33zero();
      worldArticulatedInertia.tr = m33diag(V3{m, m, m});
      worldArticulatedInertia.bl = Iw;
    }
    a.worldSpatialArticulatedInertia[linkID] = worldArticulatedInertia;
    a.worldIsolatedInertia[linkID] = Iw;
    a.masses[linkID] = m;
    sumMass += m;
    COM += a.accumulatedPoses[linkID].p * m;
    SV vel;
    if (linkID != 0) {
      const SV pVel = a.motionVelocities[link.parent];
      vel = translateSV(-a.rw[linkID], pVel);
      SV coriolisVector{V3{0.f, 0.f, 0.f}, cross(pVel.top, cross(pVel.top, a.rw[linkID]))};
      const JointData& jd = a.jointData[linkID];
      if (jd.nbDof) {
        float* jVelocity = &a.jointVelocity[jd.jointOffset];
        SV deltaV = svzero();
        for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
          const float jVel = jVelocity[ind] * ratio;
          deltaV += a.worldMotionMatrix[jd.jointOffset + ind] * jVel;
          jVelocity[ind] = jVel;
        }
        vel.top += deltaV.top;
        vel.bottom += deltaV.bottom;
        coriolisVector += SV{cross(pVel.top, deltaV.top), cross(pVel.top, deltaV.bottom) * 2.0f + cross(deltaV.top, deltaV.bottom)};
      }
      a.coriolis[linkID] = coriolisVector;
      a.motionVelocities[linkID] = vel;
    } else {
      vel = rootLinkVel;
    }
    SV zExtForces, zDamping;
    {
      const V3 g = bodyCore.disableGravity ? V3{0.f, 0.f, 0.f} : gravity;
      const V3 extLinAccel = a.externalAcceleration[linkID].linear;
      const float lindamp = bodyCore.linDamping > 0.f ? pmin(bodyCore.linDamping, invDt) : 0.0f;
      const float linscale = (magSq(vel.bottom) > bodyCore.maxLinVelSq)
                                 ? (1.0f - (psqrt(bodyCore.maxLinVelSq) / psqrt(magSq(vel.bottom))))
                                 : 0.0f;
      zExtForces.top = (g + extLinAccel) * (m * (lindamp * dt - 1.0f));
      zDamping.top = vel.bottom * (m * (lindamp + linscale * invDt));
    }
    {
      const V3 extAngAccel = a.externalAcceleration[linkID].angular;
      const float angdamp = bodyCore.angDamping > 0.f ? pmin(bodyCore.angDamping, invDt) : 0.0f;
      const float angscale = (magSq(vel.top) > bodyCore.maxAngVelSq)
                                 ? (1.0f - (psqrt(bodyCore.maxAngVelSq) / psqrt(magSq(vel.top))))
                                 : 0.0f;
      zExtForces.bottom = Iw * (extAngAccel * (angdamp * dt - 1.0f));
      zDamping.bottom = Iw * (vel.top * (angdamp + angscale * invDt));
    }
    if (externalForcesEveryTgsIterationEnabled) {
      a.zaForces[linkID] = zDamping;
      a.externalAcceleration[linkID].linear = zExtForces.top;
      a.externalAcceleration[linkID].angular = zExtForces.bottom;
    } else {
      a.zaForces[linkID] = zExtForces + zDamping;
    }
    a.zaInternal[linkID] = SV{V3{0.f, 0.f, 0.f}, cross(vel.top, Iw * vel.top)};
  }
  const float invMass = 1.f / sumMass;
  a.com = COM * invMass;
  a.invSumMass = invMass;
}

// computePropagateSpatialInertia_ZA_ZIc (DyFeatherstoneForwardDynamic.cpp:88)
EHD SMat computePropagateSpatialInertia_ZA_ZIc(uint8_t jointType, uint8_t nbJointDofs, const SV* jointMotionMatricesW, const SV* jointISW,
                                               const float* jointTargetArmatures, const uint8_t* dofIds, const float* jointExternalForces,
                                               const SMat& linkArticulatedInertiaW, const SV& linkZExtW, const SV& linkZIntIcW,
                                               InvStIs& linkInvStISW, SV* jointDofISInvStISW, float* jointDofMinusStZExtW,
                                               float* jointDofQStZIntIcW, SV& deltaZAExtParent, SV& deltaZAIntIcParent) {
  deltaZAExtParent = linkZExtW;
  deltaZAIntIcParent = linkZIntIcW;
  SMat spatialInertia;
  switch (jointType) {
    case JT_PRISMATIC:
    case JT_REVOLUTE:
    case JT_REVOLUTE_UNWRAPPED: {
      const SV& sa = jointMotionMatricesW[0];
      const SV& Is = jointISW[0];
      float invStIS;
      {
        const uint32_t dofId = dofIds[0];
        const float stIs = (innerProduct(sa, Is) + jointTargetArmatures[dofId]);
        invStIS = ((stIs > 0.f) ? (1.f / stIs) : 0.f);
      }
      linkInvStISW.m[0][0] = invStIS;
      const SV isID = Is * invStIS;
      jointDofISInvStISW[0] = isID;
      const SV stI{Is.bottom, Is.top};
      spatialInertia = constructSM(isID, stI);
      {
        const float innerprod = innerProduct(sa, linkZExtW);
        const float diff = -innerprod;
        jointDofMinusStZExtW[0] = diff;
        deltaZAExtParent += isID * diff;
      }
      {
        const float innerprod = innerProduct(sa, linkZIntIcW);
        const float diff = (jointExternalForces ? jointExternalForces[0] : 0.0f) - innerprod;
        jointDofQStZIntIcW[0] = diff;
        deltaZAIntIcParent += isID * diff;
      }
      break;
    }
    case JT_SPHERICAL: {
      M33 D = m33identity();
      for (uint32_t ind = 0; ind < nbJointDofs; ++ind) {
        for (uint32_t ind2 = 0; ind2 < nbJointDofs; ++ind2) {
          const SV& sa = jointMotionMatricesW[ind2];
          m33set(D, int(ind), int(ind2), innerProduct(sa, jointISW[ind]));  // D[ind][ind2]: 열 ind, 행 ind2
        }
        const uint32_t dofId = dofIds[ind];
        m33set(D, int(ind), int(ind), m33get(D, int(ind), int(ind)) + jointTargetArmatures[dofId]);
      }
      const M33 invD = inverse(D);
      for (uint32_t ind = 0; ind < nbJointDofs; ++ind)
        for (uint32_t ind2 = 0; ind2 < nbJointDofs; ++ind2) linkInvStISW.m[ind][ind2] = m33get(invD, int(ind), int(ind2));
      SV columns[6];
      for (int c = 0; c < 6; ++c) columns[c] = svzero();
      for (uint32_t ind = 0; ind < nbJointDofs; ++ind) {
        SV isID = svzero();
        const SV& sa = jointMotionMatricesW[ind];
        const float stZ = innerProduct(sa, linkZExtW);
        const float stZInt = innerProduct(sa, linkZIntIcW);
        const float localQstZ = -stZ;
        const float localQstZInt = (jointExternalForces ? jointExternalForces[ind] : 0.0f) - stZInt;
        jointDofMinusStZExtW[ind] = localQstZ;
        jointDofQStZIntIcW[ind] = localQstZInt;
        for (uint32_t ind2 = 0; ind2 < nbJointDofs; ++ind2) {
          const SV& Is = jointISW[ind2];
          isID += Is * m33get(invD, int(ind), int(ind2));
        }
        columns[0] += isID * jointISW[ind].bottom.x;
        columns[1] += isID * jointISW[ind].bottom.y;
        columns[2] += isID * jointISW[ind].bottom.z;
        columns[3] += isID * jointISW[ind].top.x;
        columns[4] += isID * jointISW[ind].top.y;
        columns[5] += isID * jointISW[ind].top.z;
        jointDofISInvStISW[ind] = isID;
        deltaZAExtParent += isID * localQstZ;
        deltaZAIntIcParent += isID * localQstZInt;
      }
      spatialInertia = constructSM6(columns);
      break;
    }
    default:
      return linkArticulatedInertiaW;
  }
  spatialInertia = linkArticulatedInertiaW - spatialInertia;
  return spatialInertia;
}

// computeArticulatedSpatialInertiaAndZ (:661)
EHD void computeArticulatedSpatialInertiaAndZ(Articulation& a, const float* jointDofForces) {
  const uint32_t startIndex = uint32_t(a.nLinks - 1);
  for (uint32_t linkID = startIndex; linkID > 0; --linkID) {
    const Link& link = a.links[linkID];
    const JointCore& joint = a.joints[linkID];
    const JointData& jd = a.jointData[linkID];
    const uint32_t jointOffset = jd.jointOffset;
    const uint8_t nbDofs = jd.nbDof;
    for (uint8_t ind = 0; ind < nbDofs; ++ind) {
      const SV tmp = a.worldSpatialArticulatedInertia[linkID] * a.worldMotionMatrix[jointOffset + ind];
      a.isW[jointOffset + ind] = tmp;
    }
    SV deltaZAExtParent, deltaZAIntParent;
    SMat spatialInertiaW;
    {
      const SV linkZW = a.zaForces[linkID];
      const SV linkIcW = a.worldSpatialArticulatedInertia[linkID] * a.coriolis[linkID];
      const SV linkZIntIcW = a.zaInternal[linkID] + linkIcW;
      spatialInertiaW = computePropagateSpatialInertia_ZA_ZIc(
          joint.jointType, jd.nbDof, &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset], &joint.armature[0], &joint.dofIds[0],
          jointDofForces ? &jointDofForces[jointOffset] : nullptr, a.worldSpatialArticulatedInertia[linkID], linkZW, linkZIntIcW,
          a.invStIs[linkID], &a.isInvStIS[jointOffset], &a.qstZIc[jointOffset], &a.qstZIntIc[jointOffset], deltaZAExtParent,
          deltaZAIntParent);
    }
    {
      translateInertia(skew(a.rw[linkID]), spatialInertiaW);
      const float minPropagatedInertia = 0.f;
      spatialInertiaW.bl.c0.x = pmax(minPropagatedInertia, spatialInertiaW.bl.c0.x);
      spatialInertiaW.bl.c1.y = pmax(minPropagatedInertia, spatialInertiaW.bl.c1.y);
      spatialInertiaW.bl.c2.z = pmax(minPropagatedInertia, spatialInertiaW.bl.c2.z);
      a.worldSpatialArticulatedInertia[link.parent] += spatialInertiaW;
    }
    {
      const SV translatedZA = translateSV(a.rw[linkID], deltaZAExtParent);
      const SV translatedZAInt = translateSV(a.rw[linkID], deltaZAIntParent);
      a.zaForces[link.parent] += translatedZA;
      a.zaInternal[link.parent] += translatedZAInt;
    }
  }
  a.baseInvSpatialArticulatedInertiaW = invertInertiaV(a.worldSpatialArticulatedInertia[0]);
}

// computeArticulatedResponseMatrix (:856)
EHD void computeArticulatedResponseMatrix(Articulation& a) {
  const SMat& bi = a.baseInvSpatialArticulatedInertiaW;
  if (a.flags & AF_FIX_BASE) {
    for (int i = 0; i < 6; ++i) a.responseW[0].r[i] = svzero();
  } else {
    const M33 bottomRight = transpose(bi.tl);
    a.responseW[0].r[0] = SV{bi.tl.c0, bi.bl.c0};
    a.responseW[0].r[1] = SV{bi.tl.c1, bi.bl.c1};
    a.responseW[0].r[2] = SV{bi.tl.c2, bi.bl.c2};
    a.responseW[0].r[3] = SV{bi.tr.c0, bottomRight.c0};
    a.responseW[0].r[4] = SV{bi.tr.c1, bottomRight.c1};
    a.responseW[0].r[5] = SV{bi.tr.c2, bottomRight.c2};
    a.links[0].cfm *= pmax(a.responseW[0].r[0].bottom.x, pmax(a.responseW[0].r[1].bottom.y, a.responseW[0].r[2].bottom.z));
  }
  const SV testLinkImpulses[6] = {SV{V3{-1, 0, 0}, V3{0, 0, 0}}, SV{V3{0, -1, 0}, V3{0, 0, 0}}, SV{V3{0, 0, -1}, V3{0, 0, 0}},
                                  SV{V3{0, 0, 0}, V3{-1, 0, 0}}, SV{V3{0, 0, 0}, V3{0, -1, 0}}, SV{V3{0, 0, 0}, V3{0, 0, -1}}};
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const V3& parentLinkToChildLink = a.rw[linkID];
    const uint32_t jointOffset = a.jointData[linkID].jointOffset;
    const uint8_t dofCount = a.jointData[linkID].nbDof;
    const uint32_t parentLinkId = a.links[linkID].parent;
    for (uint32_t i = 0; i < 6; ++i) {
      const SV& testLinkImpulse = testLinkImpulses[i];
      float QMinusStZ[3] = {0.f, 0.f, 0.f};
      const SV Zp = propagateImpulseW(parentLinkToChildLink, testLinkImpulse, nullptr, &a.isInvStIS[jointOffset],
                                      &a.worldMotionMatrix[jointOffset], dofCount, QMinusStZ);
      const SV deltaVParent = -responseOf(a.responseW[parentLinkId], Zp);
      const SV deltaVChild = propagateAccelerationW(parentLinkToChildLink, deltaVParent, a.invStIs[linkID], &a.worldMotionMatrix[jointOffset],
                                                    &a.isW[jointOffset], QMinusStZ, dofCount, nullptr);
      a.responseW[linkID].r[i] = deltaVChild;
    }
    a.links[linkID].cfm *= pmax(a.responseW[linkID].r[0].bottom.x, pmax(a.responseW[linkID].r[1].bottom.y, a.responseW[linkID].r[2].bottom.z));
  }
}

// computeJointAccelerationW (:987)
EHD void computeJointAccelerationW(uint8_t nbJointDofs, const SV& pMotionAcceleration, const SV* jointDofISW, const InvStIs& linkInvStISW,
                                   const float* jointDofQStZIcW, float* jointAcceleration) {
  float tJAccel[6];
  for (uint32_t ind = 0; ind < nbJointDofs; ++ind) {
    const float temp = innerProduct(jointDofISW[ind], pMotionAcceleration);
    tJAccel[ind] = (jointDofQStZIcW[ind] - temp);
  }
  for (uint32_t ind = 0; ind < nbJointDofs; ++ind) {
    jointAcceleration[ind] = 0.f;
    for (uint32_t ind2 = 0; ind2 < nbJointDofs; ++ind2) jointAcceleration[ind] += linkInvStISW.m[ind2][ind] * tJAccel[ind2];
  }
}

// computeLinkAcceleration (:1014), doIC = false 로 불림
EHD void computeLinkAcceleration(Articulation& a, float dt) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  if (!fixBase) {
    const SV accel = -(a.baseInvSpatialArticulatedInertiaW * a.zaForces[0]);
    a.motionAccelerations[0] = accel;
    const SV deltaV = accel * dt;
    a.motionVelocities[0] += deltaV;
  }
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    const SV pMotionAcceleration = translateSV(-a.rw[linkID], a.motionAccelerations[link.parent]);
    const JointData& jd = a.jointData[linkID];
    float* jA = &a.jointAcceleration[jd.jointOffset];
    computeJointAccelerationW(jd.nbDof, pMotionAcceleration, &a.isW[jd.jointOffset], a.invStIs[linkID], &a.qstZIc[jd.jointOffset], jA);
    SV motionAcceleration = pMotionAcceleration;
    float* jointVelocity = &a.jointVelocity[jd.jointOffset];
    float* jointNewVelocity = &a.jointNewVelocity[jd.jointOffset];
    for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
      const float accel = jA[ind];
      const float jVel = jointVelocity[ind] + accel * dt;
      jointVelocity[ind] = jVel;
      jointNewVelocity[ind] = jVel;
      motionAcceleration.top += a.worldMotionMatrix[jd.jointOffset + ind].top * accel;
      motionAcceleration.bottom += a.worldMotionMatrix[jd.jointOffset + ind].bottom * accel;
    }
    a.motionAccelerations[linkID] = motionAcceleration;
    a.motionVelocities[linkID] += motionAcceleration * dt;
  }
}

// computeContributionToEnsembleMomentOfInertia (:1117)
EHD M33 ensembleContribution(const M33& IiW, float mi, const V3& r) {
  const float ax = r.x, ay = r.y, az = r.z;
  const float ax2 = ax * ax, ay2 = ay * ay, az2 = az * az;
  const float negAxAy = -ax * ay, negAxAz = -ax * az, negAyAz = -ay * az;
  const V3 col0{ay2 + az2, negAxAy, negAxAz};
  const V3 col1{negAxAy, ax2 + az2, negAyAz};
  const V3 col2{negAxAz, negAyAz, ax2 + ay2};
  return M33{IiW.c0 + col0 * mi, IiW.c1 + col1 * mi, IiW.c2 + col2 * mi};
}
// computeMomentum<computeCompoundInertia> (:1156)
EHD void computeMomentum(const Articulation& a, bool computeCompound, V3& linMomentumW, V3& angMomentumW, M33* compoundInertiaW) {
  const V3 rcomW = a.com;
  const float recipMass = a.invSumMass;
  linMomentumW = V3{0.0f, 0.0f, 0.0f};
  for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) {
    const V3& vi = a.motionVelocities[linkID].bottom;
    const float mi = a.masses[linkID];
    const V3 pi = vi * mi;
    linMomentumW += pi;
  }
  const V3 vcomW = linMomentumW * recipMass;
  angMomentumW = V3{0.0f, 0.0f, 0.0f};
  for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) {
    const float mi = a.masses[linkID];
    const M33& Ii = a.worldIsolatedInertia[linkID];
    const V3& vi = a.motionVelocities[linkID].bottom;
    const V3& wi = a.motionVelocities[linkID].top;
    const V3& ri = a.accumulatedPoses[linkID].p;
    const V3 riMinusrComW = ri - rcomW;
    const V3 viMinusvComW = vi - vcomW;
    const V3 angMomi = (Ii * wi) + cross(riMinusrComW * mi, viMinusvComW);
    angMomentumW += angMomi;
    if (computeCompound) *compoundInertiaW += ensembleContribution(Ii, mi, riMinusrComW);
  }
}

// computeLinkInternalAcceleration (:1219)
EHD void computeLinkInternalAcceleration(Articulation& a, float dt) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  V3 linMomentum0{0, 0, 0}, angMomentum0{0, 0, 0};
  if (!fixBase) computeMomentum(a, false, linMomentum0, angMomentum0, nullptr);
  if (!fixBase) {
    const SV accel = -(a.baseInvSpatialArticulatedInertiaW * a.zaInternal[0]);
    a.motionAccelerationsInternal[0] = accel;
    a.motionAccelerations[0] += accel;
    const SV deltaV = accel * dt;
    a.motionVelocities[0] += deltaV;
  } else {
    a.motionAccelerationsInternal[0] = svzero();
  }
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    const SV pMotionAcceleration = translateSV(-a.rw[linkID], a.motionAccelerationsInternal[link.parent]);
    const JointData& jd = a.jointData[linkID];
    float* jIntAccel = &a.jointInternalAcceleration[jd.jointOffset];
    computeJointAccelerationW(jd.nbDof, pMotionAcceleration, &a.isW[jd.jointOffset], a.invStIs[linkID], &a.qstZIntIc[jd.jointOffset],
                              jIntAccel);
    SV motionAcceleration = pMotionAcceleration + a.coriolis[linkID];
    float* jointVelocity = &a.jointVelocity[jd.jointOffset];
    float* jointNewVelocity = &a.jointNewVelocity[jd.jointOffset];
    float* jA = &a.jointAcceleration[jd.jointOffset];
    for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
      const float accel = jIntAccel[ind];
      const float jVel = jointVelocity[ind] + accel * dt;
      jointVelocity[ind] = jVel;
      jointNewVelocity[ind] = jVel;
      motionAcceleration.top += a.worldMotionMatrix[jd.jointOffset + ind].top * accel;
      motionAcceleration.bottom += a.worldMotionMatrix[jd.jointOffset + ind].bottom * accel;
      jA[ind] += accel;
    }
    a.motionAccelerationsInternal[linkID] = motionAcceleration;
    a.motionAccelerations[linkID] += motionAcceleration;
    const SV velDelta = (motionAcceleration)*dt;
    a.motionVelocities[linkID] += velDelta;
  }
  if (!fixBase) {
    V3 linMomentum1{0, 0, 0}, angMomentum1{0, 0, 0};
    M33 compoundInertia = m33zero();
    computeMomentum(a, true, linMomentum1, angMomentum1, &compoundInertia);
    const M33 invCompoundInertia = inverse(compoundInertia);
    float angRatio;
    {
      const float numerator = mag(angMomentum0);
      const float denominator = mag(angMomentum1);
      angRatio = denominator == 0.0f ? 1.0f : numerator / denominator;
    }
    const V3 deltaAngMom = angMomentum1 * (angRatio - 1.0f);
    const V3 deltaAng = invCompoundInertia * deltaAngMom;
    for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) {
      const V3 offset = (a.accumulatedPoses[linkID].p - a.com);
      const SV velChange{deltaAng, -cross(offset, deltaAng)};
      a.motionVelocities[linkID] += velChange;
      const float mass = a.masses[linkID];
      linMomentum1 += velChange.bottom * mass;
    }
    const V3 deltaLinMom = linMomentum0 - linMomentum1;
    const V3 deltaLin = deltaLinMom * a.invSumMass;
    for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) a.motionVelocities[linkID].bottom += deltaLin;
  }
}

// computeZAForceInv (DyFeatherstoneInverseDynamic.cpp:116) + computeJointTransmittedFrictionForce (ForwardDynamic.cpp:1374)
EHD void computeTransmittedForces(Articulation& a) {
  for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) {
    const LinkBody& core = a.bodies[linkID];
    const V3& ii = core.invInertia;
    const float m = core.invMass == 0.f ? 0.f : 1.0f / core.invMass;
    const V3 inertiaTensor{ii.x == 0.f ? 0.f : (1.f / ii.x), ii.y == 0.f ? 0.f : (1.f / ii.y), ii.z == 0.f ? 0.f : (1.f / ii.z)};
    SV Ia;
    Ia.bottom = rotate(core.body2World.q, mulc(rotateInv(core.body2World.q, a.motionAccelerations[linkID].top), inertiaTensor));
    Ia.top = a.motionAccelerations[linkID].bottom * m;
    a.transmittedForce[linkID] += Ia;
  }
  const uint32_t startIndex = a.nLinks - 1;
  for (uint32_t linkID = startIndex; linkID > 1; --linkID) {
    const Link& link = a.links[linkID];
    a.transmittedForce[link.parent] += translateSV(a.rw[linkID], a.transmittedForce[linkID]);
  }
  a.transmittedForce[0] = svzero();
}

// computeUnconstrainedVelocitiesTGS (:1479) -> computeUnconstrainedVelocitiesInternal (:1805) -> updateArticulation (:1582)
EHD void computeUnconstrainedVelocitiesTGS(Articulation& a, float dt, const V3& gravity, float invLengthScale,
                                           bool externalForcesEveryTgsIterationEnabled) {
  a.dt = dt;
  if (a.jcalcDirty) {
    a.jcalcDirty = 0;
    jcalc(a);
  }
  // mStatic*.clear / 개수 0 : 정적 제약은 호출자(NoStatic) 몫
  // ArticulationData::init (:469)
  for (uint32_t i = 0; i < a.nLinks; ++i) {
    a.deltaMotion[i] = svzero();
    a.posIterMotionVelocities[i] = svzero();
  }
  a.jointDirty = 0;
  // updateArticulation
  computeRelativeTransformC2P(a);
  computeLinkStates(a, a.dt, invLengthScale, gravity, externalForcesEveryTgsIterationEnabled);
  if (a.nLinks > 1)
    for (uint32_t linkID = 0; linkID < a.nLinks; ++linkID) a.transmittedForce[linkID] = a.zaForces[linkID] + a.zaInternal[linkID];
  computeArticulatedSpatialInertiaAndZ(a, externalForcesEveryTgsIterationEnabled ? nullptr : a.jointForce);
  computeArticulatedResponseMatrix(a);
  computeLinkAcceleration(a, a.dt);
  computeLinkInternalAcceleration(a, a.dt);
  for (uint32_t i = 0; i < a.nLinks; ++i) {
    a.solverLinkSpatialDeltaVels[i] = svzero();
    a.solverLinkSpatialImpulses[i] = svzero();
  }
  // computeUnconstrainedVelocitiesInternal 의 나머지
  if (a.nLinks > 1) computeTransmittedForces(a);
  a.dataDirty = 1;
  for (uint32_t i = 0; i < a.dofs; ++i) a.deferredQstZ[i] = 0.0f;
  a.rootDeferredZ = svzero();
  for (uint32_t i = 0; i < a.nLinks; ++i) {
    a.accumulatedPoses[i] = a.bodies[i].body2World;
    a.preTransform[i] = a.bodies[i].body2World;
    a.deltaQ[i] = qid();
  }
}

// ---------------------------------------------------------------- 내부 제약 준비
// setupInternalConstraintsRecursive (:2432)
EHDR void setupInternalConstraintsLink(Articulation& a, float stepDt, float dt, bool isTGS, uint32_t linkID, float maxForceScale) {
  const Link& link = a.links[linkID];
  const JointData& jd = a.jointData[linkID];
  const Link& pLink = a.links[link.parent];
  const JointCore& j = a.joints[linkID];
  const bool hasFriction = j.frictionCoefficient > 0.f;
  const float fCoefficient = j.frictionCoefficient * stepDt;
  const float transmissionForce = svmag(a.transmittedForce[linkID]) * fCoefficient;
  const Tf cA2w = a.bodies[link.parent].body2World * j.parentPose;
  const Tf cB2w = a.bodies[linkID].body2World * j.childPose;
  const uint32_t parent = link.parent;
  const float cfm = pmax(link.cfm, pLink.cfm);
  V3 driveError{0.f, 0.f, 0.f};
  V3 angles{0.f, 0.f, 0.f};
  V3 row[3] = {V3{0, 0, 0}, V3{0, 0, 0}, V3{0, 0, 0}};
  auto vset = [](V3& v, uint32_t i, float x) { (i == 0 ? v.x : (i == 1 ? v.y : v.z)) = x; };
  auto vget = [](const V3& v, uint32_t i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); };
  if (j.jointType == JT_SPHERICAL && jd.nbDof > 1) {
    V3 driveAxis{0.f, 0.f, 0.f};
    bool hasAngularDrives = false;
    uint32_t tmpDofId = 0;
    for (uint32_t i = 0; i < AX_X; ++i) {
      if (j.motion[i] != M_LOCKED) {
        const bool hasDrive = (j.motion[i] != M_LOCKED && j.drives[i].driveType != DT_NONE);
        if (hasDrive) {
          const V3 axis = a.motionMatrix[jd.jointOffset + tmpDofId].top;
          const float target = a.jointTargetPositions[jd.jointOffset + tmpDofId];
          driveAxis += axis * target;
          hasAngularDrives = true;
        }
        tmpDofId++;
      }
    }
    {
      const Q qB2qA = conj(cA2w.q) * cB2w.q;
      if (hasAngularDrives) {
        float angle = v3normalize(driveAxis);
        if (angle < 1e-12f) {
          driveAxis = V3{1.f, 0.f, 0.f};
          angle = 0.f;
        }
        Q targetQ = quatAA(angle, driveAxis);
        if (qdot(targetQ, qB2qA) < 0.f) targetQ = qneg(targetQ);
        const Q e = conj(targetQ) * qB2qA;
        driveError = V3{e.x, e.y, e.z} * -2.f;  // -2.f * getImaginaryPart()  (float * vec = 칸마다 곱, 교환 가능)
      }
      for (uint32_t i = 0, tmpDof = 0; i < AX_X; ++i) {
        if (j.motion[i] != M_LOCKED) {
          vset(angles, i, a.jointPosition[j.jointOffset + tmpDof]);
          row[i] = a.worldMotionMatrix[jd.jointOffset + tmpDof].top;
          tmpDof++;
        }
      }
    }
  } else {
    for (uint32_t i = 0; i < AX_X; ++i) {
      if (j.motion[i] != M_LOCKED) {
        vset(driveError, i, a.jointTargetPositions[j.jointOffset] - a.jointPosition[j.jointOffset]);
        vset(angles, i, a.jointPosition[j.jointOffset]);
        row[i] = a.worldMotionMatrix[jd.jointOffset].top;
      }
    }
  }
  uint32_t dofId = 0;
  for (uint32_t i = 0; i < AX_X; ++i) {
    if (j.motion[i] != M_LOCKED) {
      const V3 axis = row[i];
      V3 dv0L, dv0A, dv1L, dv1A;
      getImpulseSelfResponse(a, parent, V3{0, 0, 0}, axis, dv0L, dv0A, linkID, V3{0, 0, 0}, -axis, dv1L, dv1A);
      const float r0 = dot(dv0A, axis);
      const float r1 = dot(dv1A, axis);
      const float unitResponse = r0 - r1;
      const float recipResponse = unitResponse <= 0.f ? 0.f : 1.0f / (unitResponse + cfm);
      InternalConstraint& c = a.ic[a.nIc++];
      c.recipResponse = recipResponse;
      c.response = unitResponse;
      c.row0 = SV{V3{0, 0, 0}, axis};
      c.row1 = SV{V3{0, 0, 0}, axis};
      c.deltaVA = SV{dv0A, dv0L};
      c.deltaVB = SV{dv1A, dv1L};
      c.isLinearConstraint = 0;
      c.maxJointVelocity = j.maxJointVelocity[i];
      c.accumulatedFrictionImpulse = 0.0f;
      c.frictionMaxForce = hasFriction ? transmissionForce : 0.f;
      c.dynamicFrictionEffort = j.fDynamic[i];
      c.staticFrictionEffort = j.fStatic[i];
      c.viscousFrictionCoefficient = j.fViscous[i];
      const bool hasDrive = (j.motion[i] != M_LOCKED && j.drives[i].driveType != DT_NONE);
      c.driveImpulse = 0.0f;
      c.driveMaxImpulse = j.drives[i].maxForce * maxForceScale;
      c.envelope = j.drives[i].envelope;
      c.externalJointForce = a.jointForce[jd.jointOffset + dofId];
      if (hasDrive)
        c.drive = computeImplicitDriveParams(j.drives[i].driveType, j.drives[i].stiffness, j.drives[i].damping, isTGS ? stepDt : dt, dt,
                                             unitResponse, recipResponse, vget(driveError, i), a.jointTargetVelocities[j.jointOffset + dofId],
                                             isTGS);
      else
        c.drive = implicitDriveZero();
      if (j.motion[i] == M_LIMITED) {
        InternalLimit& lim = a.limits[a.nLimits++];
        const float jPos = vget(angles, i);
        lim.errorHigh = j.limHigh[i] - jPos;
        lim.errorLow = jPos - j.limLow[i];
        lim.lowImpulse = 0.f;
        lim.highImpulse = 0.f;
      }
      dofId++;
    }
  }
  for (uint32_t i = AX_X; i < 6; ++i) {
    if (j.motion[i] != M_LOCKED) {
      const V3 axis = a.worldMotionMatrix[jd.jointOffset + dofId].bottom;
      const V3 ang0 = cross(cA2w.p - a.bodies[link.parent].body2World.p, axis);
      const V3 ang1 = cross(cB2w.p - a.bodies[linkID].body2World.p, axis);
      V3 dv0L, dv0A, dv1L, dv1A;
      getImpulseSelfResponse(a, parent, axis, ang0, dv0L, dv0A, linkID, -axis, -ang1, dv1L, dv1A);
      const float r0 = dot(dv0L, axis) + dot(dv0A, ang0);
      const float r1 = dot(dv1L, axis) + dot(dv1A, ang1);
      const float unitResponse = r0 - r1;
      const float recipResponse = 1.0f / (unitResponse + cfm);
      InternalConstraint& c = a.ic[a.nIc++];
      c.response = unitResponse;
      c.recipResponse = recipResponse;
      c.row0 = SV{axis, ang0};
      c.row1 = SV{axis, ang1};
      c.deltaVA = SV{dv0A, dv0L};
      c.deltaVB = SV{dv1A, dv1L};
      c.isLinearConstraint = 1;
      c.maxJointVelocity = j.maxJointVelocity[i];
      c.accumulatedFrictionImpulse = 0.f;
      c.frictionMaxForce = hasFriction ? transmissionForce : 0.f;
      c.dynamicFrictionEffort = j.fDynamic[i];
      c.staticFrictionEffort = j.fStatic[i];
      c.viscousFrictionCoefficient = j.fViscous[i];
      const bool hasDrive = (j.motion[i] != M_LOCKED && (j.drives[i].envelope.maxEffort > 0.0f || j.drives[i].maxForce > 0.0f) &&
                             (j.drives[i].stiffness > 0.f || j.drives[i].damping > 0.f));
      c.driveImpulse = 0.0f;
      c.envelope = j.drives[i].envelope;
      c.driveMaxImpulse = j.drives[i].maxForce * maxForceScale;
      c.externalJointForce = a.jointForce[jd.jointOffset + dofId];
      if (hasDrive)
        c.drive = computeImplicitDriveParams(j.drives[i].driveType, j.drives[i].stiffness, j.drives[i].damping, isTGS ? stepDt : dt, dt,
                                             unitResponse, recipResponse,
                                             a.jointTargetPositions[j.jointOffset + dofId] - a.jointPosition[j.jointOffset + dofId],
                                             a.jointTargetVelocities[j.jointOffset + dofId], isTGS);
      else
        c.drive = implicitDriveZero();
      if (j.motion[i] == M_LIMITED) {
        InternalLimit& lim = a.limits[a.nLimits++];
        const float jPos = a.jointPosition[j.jointOffset + dofId];
        lim.errorHigh = j.limHigh[i] - jPos;
        lim.errorLow = jPos - j.limLow[i];
        lim.lowImpulse = 0.f;
        lim.highImpulse = 0.f;
      }
      dofId++;
    }
  }
  // dofLimitMask 의 잠긴 축 처리(:2721)는 dof 가 잠기지 않은 축에만 있으므로 실행되지 않는다.
  for (uint32_t i = 0; i < link.numChildren; ++i) setupInternalConstraintsLink(a, stepDt, dt, isTGS, link.childrenStart + i, maxForceScale);
}

// 흉내 관절 준비 (DyArticulationMimicJoint.cpp:46-257)
EHD float mimicSelfResponse(const Articulation& a, uint32_t linkIndex, uint32_t dof) {
  const uint32_t parentLinkIndex = a.links[linkIndex].parent;
  const V3& parentLinkToChildLink = a.rw[linkIndex];
  const uint32_t jointOffset = a.jointData[linkIndex].jointOffset;
  const uint8_t dofCount = a.jointData[linkIndex].nbDof;
  const float testJointImpulses[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const float* testJointImpulse = testJointImpulses[dof];
  float QMinusStZ[3] = {0.f, 0.f, 0.f};
  const SV Zp = propagateImpulseW(parentLinkToChildLink, svzero(), testJointImpulse, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset],
                                  dofCount, QMinusStZ);
  const SV deltaVParent = -responseOf(a.responseW[parentLinkIndex], Zp);
  float jointDeltaQDot[3] = {0, 0, 0};
  propagateAccelerationW(parentLinkToChildLink, deltaVParent, a.invStIs[linkIndex], &a.worldMotionMatrix[jointOffset], &a.isW[jointOffset],
                         QMinusStZ, dofCount, jointDeltaQDot);
  return jointDeltaQDot[dof];
}
EHD float mimicCrossResponse(Articulation& a, uint32_t linkA, uint32_t dofA, uint32_t linkB, uint32_t dofB) {
  float* QMinusSTZ = a.deferredQstZ;
  const uint32_t QMinusStZLength = kMaxDofs;  // mDeferredQstZ.size() (= dofs+1 칸) — 0 으로 지우기만 하므로 길이는 결과와 무관
  const float testJointImpulses[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const float* testJointImpulse = testJointImpulses[dofA];
  const SV testLinkImpulse = svzero();
  for (uint32_t i = 0; i < QMinusStZLength; ++i) QMinusSTZ[i] = 0.0f;
  const uint32_t* pathFromRootToLink = &a.pathToRoot[a.links[linkA].pathStart];
  const uint32_t numFromRootToLink = a.links[linkA].pathCount;
  const uint32_t* pathFromRootToOtherLink = &a.pathToRoot[a.links[linkB].pathStart];
  const uint32_t numFromRootToOtherLink = a.links[linkB].pathCount;
  SV Zp;
  {
    const uint32_t linkIndex = pathFromRootToLink[numFromRootToLink - 1];
    const uint32_t jointOffset = a.jointData[linkIndex].jointOffset;
    const uint8_t dofCount = a.jointData[linkIndex].nbDof;
    Zp = propagateImpulseW(a.rw[linkIndex], testLinkImpulse, testJointImpulse, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset],
                           dofCount, &QMinusSTZ[jointOffset]);
  }
  for (uint32_t k = 1; k < numFromRootToLink; k++) {
    const uint32_t linkIndex = pathFromRootToLink[numFromRootToLink - 1 - k];
    const uint32_t jointOffset = a.jointData[linkIndex].jointOffset;
    const uint8_t dofCount = a.jointData[linkIndex].nbDof;
    Zp = propagateImpulseW(a.rw[linkIndex], Zp, nullptr, &a.isInvStIS[jointOffset], &a.worldMotionMatrix[jointOffset], dofCount,
                           &QMinusSTZ[jointOffset]);
  }
  SV deltaVParent = -responseOf(a.responseW[0], Zp);
  float jointVelocity[3] = {0, 0, 0};
  for (uint32_t k = 0; k < numFromRootToOtherLink; k++) {
    const uint32_t linkIndex = pathFromRootToOtherLink[k];
    const uint32_t jointOffset = a.jointData[linkIndex].jointOffset;
    const uint8_t dofCount = a.jointData[linkIndex].nbDof;
    float* jointVelocityToUse = ((numFromRootToOtherLink - 1) == k) ? jointVelocity : nullptr;
    deltaVParent = propagateAccelerationW(a.rw[linkIndex], deltaVParent, a.invStIs[linkIndex], &a.worldMotionMatrix[jointOffset],
                                          &a.isW[jointOffset], &QMinusSTZ[jointOffset], dofCount, jointVelocityToUse);
  }
  for (uint32_t i = 0; i < QMinusStZLength; ++i) QMinusSTZ[i] = 0.0f;
  return jointVelocity[dofB];
}
EHD void setupInternalMimicJointConstraints(Articulation& a) {
  for (uint32_t i = 0; i < a.nMimic; i++) {
    const MimicCore& mc = a.mimic[i];
    MimicInternal& mi = a.mimicInternal[i];
    const uint32_t linkA = mc.linkA, linkB = mc.linkB;
    const uint32_t dofA = a.joints[linkA].invDofIds[mc.axisA];
    const uint32_t dofB = a.joints[linkB].invDofIds[mc.axisB];
    const float rAA = mimicSelfResponse(a, linkA, dofA);
    const float rBB = mimicSelfResponse(a, linkB, dofB);
    const float rBA = mimicCrossResponse(a, linkA, dofA, linkB, dofB);
    const float rAB = mimicCrossResponse(a, linkB, dofB, linkA, dofA);
    const float gearRatio = mc.gearRatio;
    const float recipEffectiveInertia = computeRecipMimicJointEffectiveInertia(rAA, rAB, rBB, rBA, gearRatio);
    mi.gearRatio = mc.gearRatio;
    mi.offset = mc.offset;
    mi.naturalFrequency = mc.naturalFrequency;
    mi.dampingRatio = mc.dampingRatio;
    mi.linkA = linkA;
    mi.linkB = linkB;
    mi.dofA = dofA;
    mi.dofB = dofB;
    mi.recipEffectiveInertia = recipEffectiveInertia;
  }
}

// setupSolverConstraintsTGS(desc, stepDt, invStepDt, totalDt) (:3193) -> setupInternalConstraints (:3032)
EHD void setupSolverConstraintsTGS(Articulation& a, float stepDt, float invStepDt, float totalDt) {
  (void)invStepDt;
  a.nIc = 0;
  a.nLimits = 0;
  const float dt = totalDt;
  const float maxForceScale = (a.flags & AF_DRIVE_LIMITS_ARE_FORCES) ? dt : 1.f;
  const Link& root = a.links[0];
  for (uint32_t i = 0; i < root.numChildren; ++i) setupInternalConstraintsLink(a, stepDt, dt, true, root.childrenStart + i, maxForceScale);
  setupInternalMimicJointConstraints(a);
}

// ---------------------------------------------------------------- 내부 제약 풀기
struct ProcessConfig {  // ArticulationConstraintProcessingConfig (DyCpuGpuArticulation.h:758)
  bool doMimic, doFrictionDrivePosLimit;
  uint8_t velLimit;  // 0 없음, 1 정적 제약 전, 2 정적 제약 후
  bool doStatic;
};
EHD ProcessConfig singlePassConfig(bool solveArticulationContactLast) {
  return solveArticulationContactLast ? ProcessConfig{true, true, 2, true} : ProcessConfig{true, true, 1, true};
}
EHD ProcessConfig firstPassConfig() { return ProcessConfig{true, true, 0, false}; }
EHD ProcessConfig secondPassConfig() { return ProcessConfig{false, false, 2, true}; }

struct SolveData {  // InternalConstraintSolverData (:4510)
  float dt, stepDt, invStepDt, elapsedTime, erp;
  bool isTGS, isVelIter, isExternalForceEveryStep;
};

// solveInternalMimicJointConstraints (DyArticulationMimicJoint.cpp:259)
EHD void solveInternalMimicJointConstraints(Articulation& a, float dt, float invDt, bool velocityIteration, float biasCoefficient) {
  for (uint32_t i = 0; i < a.nMimic; i++) {
    const MimicInternal& m = a.mimicInternal[i];
    const uint32_t jointOffsetA = a.joints[m.linkA].jointOffset;
    const uint32_t jointOffsetB = a.joints[m.linkB].jointOffset;
    const float qA = a.jointPosition[jointOffsetA + m.dofA];
    const float qB = a.jointPosition[jointOffsetB + m.dofB];
    float qADot = 0, qBDot = 0;
    {
      float jointDofSpeedsA[3] = {0, 0, 0};
      V3 l, an;
      pxcFsGetVelocity(a, m.linkA, jointDofSpeedsA, l, an);
      float jointDofSpeedsB[3] = {0, 0, 0};
      pxcFsGetVelocity(a, m.linkB, jointDofSpeedsB, l, an);
      qADot = jointDofSpeedsA[m.dofA];
      qBDot = jointDofSpeedsB[m.dofB];
    }
    float jointImpulseA[3] = {0, 0, 0};
    float jointImpulseB[3] = {0, 0, 0};
    {
      float jointDofImpA = 0, jointDofImpB = 0;
      computeMimicJointImpulses(biasCoefficient, dt, invDt, qA, qB, qADot, qBDot, m.gearRatio, m.offset, m.naturalFrequency, m.dampingRatio,
                                m.recipEffectiveInertia, velocityIteration, jointDofImpA, jointDofImpB);
      jointImpulseA[m.dofA] = jointDofImpA;
      jointImpulseB[m.dofB] = jointDofImpB;
    }
    const V3 zero{0, 0, 0};
    pxcFsApplyImpulses(a, m.linkA, zero, zero, jointImpulseA, m.linkB, zero, zero, jointImpulseB);
  }
}

// accumulateLinkImpulsesAndLinkVelocities (:4540)
EHD void accumulateLinkImpulses(float deltaF, const InternalConstraint& c, SV& i0, SV& i1, SV& parentV, SV& childV, SV& dv1) {
  i0 += c.row0 * deltaF;
  i1.top = i1.top - c.row1.top * deltaF;
  i1.bottom = i1.bottom - c.row1.bottom * deltaF;
  const SV deltaVP = c.deltaVA * (-deltaF);
  const SV deltaVC = c.deltaVB * (-deltaF);
  parentV += SV{deltaVP.top, deltaVP.bottom};
  childV += SV{deltaVC.top, deltaVC.bottom};
  dv1.top += deltaVC.top;
  dv1.bottom += deltaVC.bottom;
}

// solveInternalJointConstraintRecursive (:4560). 재귀 대신 명시적 스택 (GPU 에서도 되게). 연산 순서는 같다.
template <class Static>
EHDR SV solveJointLink(Articulation& a, const SolveData& data, uint32_t linkID, const SV& parentDeltaV, const ProcessConfig& cfg,
                      uint32_t& dofId, uint32_t& limitId, const Static& st);

template <class Static>
EHDR SV solveJointLinkBody(Articulation& a, const SolveData& data, uint32_t linkID, const SV& parentDeltaV, const ProcessConfig& cfg,
                          uint32_t& dofId, uint32_t& limitId, const Static& st) {
  const Link& link = a.links[linkID];
  const uint32_t startDofId = dofId;
  const JointData& jd = a.jointData[linkID];
  SV i1 = svzero();
  SV parentV = parentDeltaV + a.motionVelocities[link.parent];
  const SV parentVelContrib = propagateAccelerationW(a.rw[linkID], parentDeltaV, a.invStIs[linkID], &a.worldMotionMatrix[jd.jointOffset],
                                                     &a.isW[jd.jointOffset], &a.deferredQstZ[jd.jointOffset], jd.nbDof, nullptr);
  SV childV = a.motionVelocities[linkID] + parentVelContrib;
  SV i0 = svzero();
  SV dv1 = parentVelContrib;
  if (cfg.doFrictionDrivePosLimit || cfg.velLimit == 1) {
    for (uint32_t dof = 0; dof < jd.nbDof; ++dof) {
      if (a.nIc <= (startDofId + dof)) continue;
      InternalConstraint& c = a.ic[startDofId + dof];
      const float jointPDelta = innerProduct(c.row1, a.deltaMotion[linkID]) - innerProduct(c.row0, a.deltaMotion[link.parent]);
      float jointV = innerProduct(c.row1, childV) - innerProduct(c.row0, parentV);
      const bool newFrictionModel = c.staticFrictionEffort != 0.0f || c.viscousFrictionCoefficient != 0.0f;
      const bool isPerStep = (data.isTGS && data.isExternalForceEveryStep && !(data.isVelIter));
      const float effectiveTimestep = isPerStep ? data.stepDt : data.dt;
      float frictionDeltaF = 0.0f, driveDeltaF = 0.0f, posLimitDeltaF = 0.0f;
      if (cfg.doFrictionDrivePosLimit) {
        if (!newFrictionModel) {
          const float appliedFriction = data.isTGS ? 0.0f : c.accumulatedFrictionImpulse;
          const float frictionForce = pclamp(-jointV * c.recipResponse + appliedFriction, -c.frictionMaxForce, c.frictionMaxForce);
          c.accumulatedFrictionImpulse = frictionForce;
          frictionDeltaF = frictionForce - appliedFriction;
          jointV += frictionDeltaF * c.response;
        }
        if (c.envelope.maxEffort > 0.0f) {
          const float appliedDriveImpulse = isPerStep ? 0.0f : c.driveImpulse;
          const float maxImpulse = c.envelope.maxEffort * effectiveTimestep;
          const float speedImpulseGradient = c.envelope.speedEffortGradient / effectiveTimestep;
          const float velocityDependentResistance = c.envelope.velocityDependentResistance * effectiveTimestep;
          const float externalJointImpulse = c.externalJointForce * effectiveTimestep;
          const float unclampedImpulse = (data.isTGS && data.isVelIter)
                                             ? appliedDriveImpulse
                                             : computeDriveImpulse(appliedDriveImpulse, jointV, jointPDelta, data.elapsedTime, c.drive);
          const float clampedImpulse =
              clampDriveImpulse(jointV, appliedDriveImpulse + externalJointImpulse, unclampedImpulse + externalJointImpulse, c.response,
                                c.envelope.maxActuatorVelocity, maxImpulse, speedImpulseGradient, velocityDependentResistance) -
              externalJointImpulse;
          driveDeltaF = clampedImpulse - appliedDriveImpulse;
          c.driveImpulse += driveDeltaF;
        } else {
          const float unclampedImpulse = (data.isTGS && data.isVelIter)
                                             ? c.driveImpulse
                                             : computeDriveImpulse(c.driveImpulse, jointV, jointPDelta, data.elapsedTime, c.drive);
          const float clampedImpulse = pclamp(unclampedImpulse, -c.driveMaxImpulse, c.driveMaxImpulse);
          driveDeltaF = (clampedImpulse - c.driveImpulse);
          c.driveImpulse = clampedImpulse;
        }
        jointV += driveDeltaF * c.response;
        if (newFrictionModel) {
          const float staticFrictionImpulse = c.staticFrictionEffort * effectiveTimestep;
          const float dynamicFrictionImpulse = c.dynamicFrictionEffort * effectiveTimestep;
          const float viscousFrictionCoefficient = c.viscousFrictionCoefficient * effectiveTimestep;
          const float accumulatedFrictionImpulse = isPerStep ? 0.0f : c.accumulatedFrictionImpulse;
          const float totalFrictionImpulse = computeFrictionImpulse(accumulatedFrictionImpulse - jointV * c.recipResponse, staticFrictionImpulse,
                                                                    dynamicFrictionImpulse, viscousFrictionCoefficient, jointV);
          frictionDeltaF = totalFrictionImpulse - accumulatedFrictionImpulse;
          c.accumulatedFrictionImpulse += frictionDeltaF;
          jointV += frictionDeltaF * c.response;
        }
        if (jd.dofLimitMask & (1 << dof)) {
          InternalLimit& lim = a.limits[limitId++];
          posLimitDeltaF = computeLimitImpulse(data.stepDt, data.invStepDt, data.isVelIter, c.response, c.recipResponse, data.erp,
                                               lim.errorLow, lim.errorHigh, jointPDelta, lim.lowImpulse, lim.highImpulse, jointV);
        }
      }
      float velLimitDeltaF = 0.0f;
      const float maxJointVel = c.maxJointVelocity;
      if ((cfg.velLimit == 1) && (fabsP(jointV) > maxJointVel)) {
        const float newJointV = pclamp(jointV, -maxJointVel, maxJointVel);
        velLimitDeltaF = (newJointV - jointV) * c.recipResponse;
        jointV = newJointV;
      }
      const float deltaF = frictionDeltaF + driveDeltaF + posLimitDeltaF + velLimitDeltaF;
      if (deltaF != 0.f) accumulateLinkImpulses(deltaF, c, i0, i1, parentV, childV, dv1);
    }
  }
  SV i1Internal = i1;
  SV i1FromStatic = svzero();
  const uint32_t nbStatic1D = st.count1D(a, linkID);
  const uint32_t nbStaticContact = st.countContact(a, linkID);
  const bool processStatic = cfg.doStatic && ((nbStatic1D > 0) || (nbStaticContact > 0));
  if (processStatic) {
    const SV i1Before = i1;
    for (uint32_t i = 0; i < nbStatic1D; ++i) st.solve1D(a, linkID, i, childV, i1, dv1, data);
    for (uint32_t i = 0; i < nbStaticContact; ++i) st.solveContact(a, linkID, i, childV, i1, dv1, data);
    i1FromStatic = i1 - i1Before;
  }
  if (cfg.velLimit == 2) {
    const SV i1BeforeVelLimit = i1;
    SV deltaVParent = svzero();
    if (processStatic) {
      const SV propagated = propagateImpulseW(a.rw[linkID], i1FromStatic, nullptr, &a.isInvStIS[jd.jointOffset],
                                              &a.worldMotionMatrix[jd.jointOffset], jd.nbDof, nullptr);
      deltaVParent = -getImpulseResponseW(a, link.parent, propagated);
    }
    for (uint32_t dof = 0; dof < jd.nbDof; ++dof) {
      if (a.nIc <= (startDofId + dof)) continue;
      const InternalConstraint& c = a.ic[startDofId + dof];
      float jointV = innerProduct(c.row1, childV) - innerProduct(c.row0, parentV + deltaVParent);
      float velLimitDeltaF = 0.0f;
      const float maxJointVel = c.maxJointVelocity;
      if (fabsP(jointV) > maxJointVel) {
        const float newJointV = pclamp(jointV, -maxJointVel, maxJointVel);
        velLimitDeltaF = (newJointV - jointV) * c.recipResponse;
        jointV = newJointV;
      }
      if (velLimitDeltaF != 0.f) accumulateLinkImpulses(velLimitDeltaF, c, i0, i1, parentV, childV, dv1);
    }
    i1Internal += (i1 - i1BeforeVelLimit);
  }
  dofId = startDofId + jd.nbDof;
  const uint32_t numChildren = link.numChildren;
  const uint32_t offset = link.childrenStart;
  for (uint32_t i = 0; i < numChildren; ++i) {
    const uint32_t child = offset + i;
    const SV childImp = solveJointLink(a, data, child, dv1, cfg, dofId, limitId, st);
    i1 += childImp;
    if ((numChildren - i) > 1) {
      const SV deltaV = responseOf(a.responseW[linkID], -childImp);
      dv1 += deltaV;
      childV += deltaV;
    }
  }
  SV propagatedImpulseAtParentW = propagateImpulseW(a.rw[linkID], i1, nullptr, &a.isInvStIS[jd.jointOffset], &a.worldMotionMatrix[jd.jointOffset],
                                                    jd.nbDof, &a.deferredQstZ[jd.jointOffset]);
  a.solverLinkSpatialImpulses[linkID] += (i1 - i1Internal);
  return SV{i0.top, i0.bottom} + propagatedImpulseAtParentW;
}
template <class Static>
EHDR SV solveJointLink(Articulation& a, const SolveData& data, uint32_t linkID, const SV& parentDeltaV, const ProcessConfig& cfg,
                      uint32_t& dofId, uint32_t& limitId, const Static& st) {
  return solveJointLinkBody(a, data, linkID, parentDeltaV, cfg, dofId, limitId, st);
}

// solveInternalJointConstraints (:4873)
template <class Static>
EHD void solveInternalJointConstraints(Articulation& a, float dt, float stepDt, float invStepDt, bool isVelIter, bool isTGS, const ProcessConfig& cfg,
                                       float elapsedTime, float biasCoefficient, bool isExternalForcesEveryTgsIterationEnabled, const Static& st) {
  const float erp = biasCoefficient;
  const bool fixBase = a.flags & AF_FIX_BASE;
  SV rootLinkDeltaV = svzero();
  if (!fixBase) rootLinkDeltaV = a.baseInvSpatialArticulatedInertiaW * -a.rootDeferredZ;
  SV rootLinkV = rootLinkDeltaV + a.motionVelocities[0];
  SV im0 = svzero();
  {
    const uint32_t nbStatic1D = st.count1D(a, 0);
    if (cfg.doStatic && nbStatic1D) {
      const SolveData d0{dt, stepDt, invStepDt, elapsedTime, erp, isTGS, isVelIter, isExternalForcesEveryTgsIterationEnabled};
      for (uint32_t i = 0; i < nbStatic1D; ++i) st.solve1D(a, 0u, i, rootLinkV, im0, rootLinkDeltaV, d0);
    }
    const uint32_t nbStaticContact = st.countContact(a, 0);
    if (cfg.doStatic && nbStaticContact) {
      const SolveData d0{dt, stepDt, invStepDt, elapsedTime, erp, isTGS, isVelIter, isExternalForcesEveryTgsIterationEnabled};
      for (uint32_t i = 0; i < nbStaticContact; ++i) st.solveContact(a, 0u, i, rootLinkV, im0, rootLinkDeltaV, d0);
    }
  }
  const SolveData data{dt, stepDt, invStepDt, elapsedTime, erp, isTGS, isVelIter, isExternalForcesEveryTgsIterationEnabled};
  uint32_t dofId = 0, limitId = 0;
  const uint32_t numChildren = a.links[0].numChildren;
  const uint32_t offset = a.links[0].childrenStart;
  for (uint32_t i = 0; i < numChildren; ++i) {
    const uint32_t child = offset + i;
    const SV imp = solveJointLink(a, data, child, rootLinkDeltaV, cfg, dofId, limitId, st);
    im0 += imp;
    if (!fixBase && (numChildren - 1) != 0) rootLinkDeltaV += a.baseInvSpatialArticulatedInertiaW * (-imp);
  }
  a.rootDeferredZ += im0;
  a.jointDirty = 1;
}

// solveInternalConstraints (:5313)
template <class Static>
EHD void solveInternalConstraints(Articulation& a, float dt, float stepDt, float invStepDt, bool velocityIteration, bool isTGS,
                                  const ProcessConfig& cfg, float elapsedTime, float biasCoefficient, bool extEvery, const Static& st) {
  if (cfg.doMimic) solveInternalMimicJointConstraints(a, stepDt, invStepDt, velocityIteration, biasCoefficient);  // 힘줄(tendon) 없음
  solveInternalJointConstraints(a, dt, stepDt, invStepDt, velocityIteration, isTGS, cfg, elapsedTime, biasCoefficient, extEvery, st);
}

// ---------------------------------------------------------------- 서브스텝 적분
EHD void computeSphericalJointPositionsQ(const Q& newRot, const Q& pBody2WorldRot, Q& newParentToChild) {
  newParentToChild = normalized(conj(newRot) * pBody2WorldRot);
  if (newParentToChild.w < 0.f) newParentToChild = qneg(newParentToChild);
}
// PxQuat::toRadiansAndUnitAxis (PxQuat.h:158): PxAtan2 -> ::atan2f (PxMath.h). 호스트·GPU 모두 glibc 2.35 e_atan2f.c 이식본
// (core/common/glibc_trig.h, libm 과 비트 동일 — tests/common/test_glibc_trig). 구면 관절에서만 쓴다.
EHD void toRadiansAndUnitAxis(const Q& q, float& angle, V3& axis) {
  const float quatEpsilon = 1.0e-8f;
  const float s2 = q.x * q.x + q.y * q.y + q.z * q.z;
  if (s2 < quatEpsilon * quatEpsilon) {
    angle = 0.0f;
    axis = V3{1.0f, 0.0f, 0.0f};
  } else {
    const float s = 1.0f / psqrt(s2);
    axis = V3{q.x, q.y, q.z} * s;
    angle = fabsP(q.w) < quatEpsilon ? 3.14159265358979323846f : glibc::atan2f(s2 * s, q.w) * 2.0f;
  }
}

// propagateTransform (:1581)
EHD Tf propagateTransform(Articulation& a, uint32_t linkID, float dt, const Tf& pBody2World, const Tf& currentTransform) {
  const JointData& jd = a.jointData[linkID];
  const Q relativeQuat = a.relativeQuat[linkID];
  const JointCore& joint = a.joints[linkID];
  float* jVelocity = &a.jointNewVelocity[jd.jointOffset];
  float* jPosition = &a.jointPosition[jd.jointOffset];
  const SV* motionMatrix = &a.motionMatrix[jd.jointOffset];
  Q newParentToChild = qid();
  V3 r{0, 0, 0};
  const V3 childOffset = -joint.childPose.p;
  const V3 parentOffset = joint.parentPose.p;
  switch (joint.jointType) {
    case JT_PRISMATIC: {
      float tJointPosition = jPosition[0] + (jVelocity[0]) * dt;
      const uint32_t dofId = joint.dofIds[0];
      if (joint.motion[dofId] == M_LIMITED) {
        if (tJointPosition < (joint.limLow[dofId])) tJointPosition = joint.limLow[dofId];
        if (tJointPosition > (joint.limHigh[dofId])) tJointPosition = joint.limHigh[dofId];
      }
      jPosition[0] = tJointPosition;
      newParentToChild = relativeQuat;
      const V3 e = rotate(newParentToChild, parentOffset);
      const V3 d = childOffset;
      r = e + d + motionMatrix[0].bottom * tJointPosition;
      break;
    }
    case JT_REVOLUTE:
    case JT_REVOLUTE_UNWRAPPED: {
      const float tJointPosition = jPosition[0] + (jVelocity[0]) * dt;
      jPosition[0] = tJointPosition;
      const V3& u = motionMatrix[0].top;
      Q jointRotation = quatAA(-tJointPosition, u);
      if (jointRotation.w < 0) jointRotation = qneg(jointRotation);
      newParentToChild = normalized(jointRotation * relativeQuat);
      const V3 e = rotate(newParentToChild, parentOffset);
      const V3 d = childOffset;
      r = e + d;
      break;
    }
    case JT_SPHERICAL: {
      const SV worldVel = a.motionVelocities[linkID];
      const Tf oldTransform = currentTransform;
      V3 worldAngVel = worldVel.top;
      const float dist = v3normalize(worldAngVel) * dt;
      Q newWorldQ;
      if (dist > 1e-6f) newWorldQ = quatAA(dist, worldAngVel) * oldTransform.q;
      else newWorldQ = oldTransform.q;
      computeSphericalJointPositionsQ(newWorldQ, pBody2World.q, newParentToChild);
      Q jointRotation = newParentToChild * conj(relativeQuat);
      if (jointRotation.w < 0.0f) jointRotation = qneg(jointRotation);
      V3 axis;
      float angle;
      toRadiansAndUnitAxis(jointRotation, angle, axis);
      axis *= angle;
      for (uint32_t i = 0; i < jd.nbDof; ++i) {
        const V3 sa = a.motionMatrix[jd.jointOffset + i].top;
        const float ang = -dot(sa, axis);
        jPosition[i] = ang;
      }
      const V3 e = rotate(newParentToChild, parentOffset);
      const V3 d = childOffset;
      r = e + d;
      break;
    }
    case JT_FIX: {
      newParentToChild = relativeQuat;
      const V3 e = rotate(newParentToChild, parentOffset);
      const V3 d = childOffset;
      r = e + d;
      break;
    }
    default:
      break;
  }
  Tf cBody2World;
  cBody2World.q = normalized(pBody2World.q * conj(newParentToChild));
  cBody2World.p = pBody2World.p + rotate(cBody2World.q, r);
  return cBody2World;
}

// recordDeltaMotion (= ArticulationPImpl::updateDeltaMotion) (:1114)
EHD void recordDeltaMotion(Articulation& a, float dt, SV* deltaVScratch) {
  if (a.jointDirty) flushVelocity(a, deltaVScratch);
  a.dt = dt;
  const bool fixBase = a.flags & AF_FIX_BASE;
  if (!fixBase) {
    SV& motionVelocity = a.motionVelocities[0];
    const Tf preTrans = a.accumulatedPoses[0];
    const V3 lin = motionVelocity.bottom;
    const V3 ang = motionVelocity.top;
    const V3 newP = preTrans.p + lin * dt;
    const Tf newPose{pxExp(ang * dt) * preTrans.q, newP};
    a.accumulatedPoses[0] = newPose;
    Q dq = newPose.q * conj(a.preTransform[0].q);
    if (dq.w < 0.f) dq = qneg(dq);
    a.deltaQ[0] = dq;
    const SV delta = motionVelocity * dt;
    a.deltaMotion[0] += delta;
    a.posIterMotionVelocities[0] += delta;
  }
  for (uint32_t linkID = 1; linkID < a.nLinks; linkID++) {
    const Tf newPose = propagateTransform(a, linkID, dt, a.accumulatedPoses[a.links[linkID].parent], a.accumulatedPoses[linkID]);
    Q dq = newPose.q * conj(a.preTransform[linkID].q);
    if (dq.w < 0.f) dq = qneg(dq);
    a.deltaQ[linkID] = dq;
    const V3 lin = (newPose.p - a.preTransform[linkID].p);
    const SV delta = a.motionVelocities[linkID] * dt;
    a.deltaMotion[linkID].top += delta.top;
    a.deltaMotion[linkID].bottom = lin;
    a.posIterMotionVelocities[linkID] += delta;
    a.accumulatedPoses[linkID] = newPose;
  }
}

// saveVelocityTGS (:1888)
EHD void saveVelocityTGS(Articulation& a, float invDtF32) {
  for (uint32_t i = 0; i < a.nLinks; ++i) a.posIterMotionVelocities[i] = a.posIterMotionVelocities[i] * invDtF32;
}

// applyTgsSubstepForces (:1037) — eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS 가 켜졌을 때만
EHD void applyTgsSubstepForces(Articulation& a, float stepDt, SV* extForcesArticulatedYW) {
  const bool fixBase = a.flags & AF_FIX_BASE;
  a.jointDirty = 1;
  for (uint32_t linkID = 0; linkID < a.nLinks; linkID++) extForcesArticulatedYW[linkID] = svzero();
  const int32_t startIndex = int32_t(a.nLinks) - 1;
  for (int32_t linkID = startIndex; linkID > 0; --linkID) {
    const Link& link = a.links[linkID];
    const LinAng& ext = a.externalAcceleration[linkID];
    const SV isolatedYW{ext.linear, ext.angular};
    const SV articulatedYW = isolatedYW * stepDt + extForcesArticulatedYW[linkID];
    const JointData& jd = a.jointData[linkID];
    SV parentYW = articulatedYW;
    for (uint8_t jointDof = 0; jointDof < jd.nbDof; jointDof++) {
      const uint32_t jointIdx = jd.jointOffset + jointDof;
      const float jointForce = a.jointForce[jointIdx];
      const float stZY = jointForce * stepDt - innerProduct(a.worldMotionMatrix[jointIdx], articulatedYW);
      parentYW += a.isInvStIS[jointIdx] * stZY;
      a.deferredQstZ[jointIdx] += stZY;
    }
    const uint32_t parentLinkID = link.parent;
    if (parentLinkID == 0 && fixBase) continue;
    parentYW = translateSV(a.rw[linkID], parentYW);
    extForcesArticulatedYW[parentLinkID] += parentYW;
  }
  if (!fixBase) {
    const LinAng& ext = a.externalAcceleration[0];
    const SV isolatedYW{ext.linear, ext.angular};
    const SV articulatedYW = isolatedYW * stepDt + extForcesArticulatedYW[0];
    a.rootDeferredZ += articulatedYW;
  }
}

// ---------------------------------------------------------------- 되쓰기: updateBodiesTGS (ForwardDynamic.cpp:2260, integrateJointPositions = false)
EHD void computeAndEnforceJointPositions(Articulation& a) {  // :1924
  const float twoPi = 6.28318530717958647692f;
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    JointCore& joint = a.joints[linkID];
    const JointData& jd = a.jointData[linkID];
    float* jPositions = &a.jointPosition[jd.jointOffset];
    if (joint.jointType == JT_SPHERICAL) {
      Q newParentToChild;
      computeSphericalJointPositionsQ(a.bodies[linkID].body2World.q, a.bodies[link.parent].body2World.q, newParentToChild);
      Q jointRotation = newParentToChild * conj(a.relativeQuat[linkID]);
      if (jointRotation.w < 0.0f) jointRotation = qneg(jointRotation);
      float radians;
      V3 axis;
      toRadiansAndUnitAxis(jointRotation, radians, axis);
      axis *= radians;
      for (uint32_t d = 0; d < jd.nbDof; ++d) jPositions[d] = -dot(a.motionMatrix[jd.jointOffset + d].top, axis);
    } else if (joint.jointType == JT_REVOLUTE) {
      float jPos = jPositions[0];
      if (jPos > twoPi) jPos -= twoPi * 2.f;
      else if (jPos < -twoPi) jPos += twoPi * 2.f;
      jPos = pclamp(jPos, -twoPi * 2.f, twoPi * 2.f);
      jPositions[0] = jPos;
    } else if (joint.jointType == JT_PRISMATIC) {
      const uint32_t dofId = joint.dofIds[0];
      if (joint.motion[dofId] == M_LIMITED) {
        if (jPositions[0] < (joint.limLow[dofId])) jPositions[0] = joint.limLow[dofId];
        if (jPositions[0] > (joint.limHigh[dofId])) jPositions[0] = joint.limHigh[dofId];
      }
    }
  }
}

EHD void updateBodiesTGS(Articulation& a, float dt, SV* deltaVScratch) {
  const uint32_t linkCount = a.nLinks;
  a.dt = dt;
  if (a.jointDirty) flushVelocity(a, deltaVScratch);
  SV momentum0 = svzero();
  V3 posMomentum{0.f, 0.f, 0.f};
  const bool fixBase = a.flags & AF_FIX_BASE;
  if (!fixBase) {
    const V3 COM = a.com;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
      const float mass = a.masses[linkID];
      momentum0.top += a.motionVelocities[linkID].bottom * mass;
      posMomentum += a.posIterMotionVelocities[linkID].bottom * mass;
    }
    const V3 rootVel = momentum0.top * a.invSumMass;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
      const float mass = a.masses[linkID];
      const V3 offsetMass = (a.preTransform[linkID].p - COM) * mass;
      const V3 angMom = (a.worldIsolatedInertia[linkID] * a.motionVelocities[linkID].top) + cross(offsetMass, a.motionVelocities[linkID].bottom - rootVel);
      momentum0.bottom += angMom;
    }
  }
  for (uint32_t linkID = 0; linkID < linkCount; ++linkID) a.bodies[linkID].body2World = normalized(a.accumulatedPoses[linkID]);
  computeAndEnforceJointPositions(a);
  if (!fixBase) {
    V3 COM = a.bodies[0].body2World.p * a.masses[0];
    a.accumulatedPoses[0] = a.bodies[0].body2World;
    V3 sumLinMom = a.motionVelocities[0].bottom * a.masses[0];
    for (uint32_t linkID = 1; linkID < linkCount; ++linkID) {
      const uint32_t parent = a.links[linkID].parent;
      const Tf childPose = a.bodies[linkID].body2World;
      a.accumulatedPoses[linkID] = childPose;
      const V3 rwv = childPose.p - a.accumulatedPoses[parent].p;
      a.rw[linkID] = rwv;
      const JointData& jd = a.jointData[linkID];
      const float* jVelocity = &a.jointNewVelocity[jd.jointOffset];
      SV vel = translateSV(-rwv, a.motionVelocities[parent]);
      SV deltaV = svzero();
      for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
        const float jVel = jVelocity[ind];
        deltaV += a.motionMatrix[jd.jointOffset + ind] * jVel;
      }
      vel.top += rotate(childPose.q, deltaV.top);
      vel.bottom += rotate(childPose.q, deltaV.bottom);
      a.motionVelocities[linkID] = vel;
      const float mass = a.masses[linkID];
      COM += childPose.p * mass;
      sumLinMom += vel.bottom * mass;
    }
    COM *= a.invSumMass;
    M33 sumInertia = m33zero();
    V3 sumAngMom{0.f, 0.f, 0.f};
    const V3 rootLinVel = sumLinMom * a.invSumMass;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
      const float mass = a.masses[linkID];
      const V3 offset = a.accumulatedPoses[linkID].p - COM;
      const M33 R = m33FromQuat(a.accumulatedPoses[linkID].q);
      const V3 invInertiaDiag = a.bodies[linkID].invInertia;
      const V3 inertiaDiag{1.f / invInertiaDiag.x, 1.f / invInertiaDiag.y, 1.f / invInertiaDiag.z};
      const V3 offsetMass = offset * mass;
      const M33 inertia = transformInertiaTensor(inertiaDiag, R);
      sumInertia += ensembleContribution(inertia, mass, offset);
      sumAngMom += inertia * a.motionVelocities[linkID].top;
      sumAngMom += cross(offsetMass, a.motionVelocities[linkID].bottom - rootLinVel);
    }
    const M33 invSumInertia = inverse(sumInertia);
    const float aDenom = mag(sumAngMom);
    const float angRatio = aDenom == 0.f ? 0.f : mag(momentum0.bottom) / aDenom;
    const V3 angMomDelta = sumAngMom * (angRatio - 1.f);
    const V3 angDelta = invSumInertia * angMomDelta;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
      const V3 offset = (a.accumulatedPoses[linkID].p - COM);
      const SV velChange{angDelta, -cross(offset, angDelta)};
      a.motionVelocities[linkID] += velChange;
      const float mass = a.masses[linkID];
      sumLinMom += velChange.bottom * mass;
    }
    const V3 linDelta = (momentum0.top - sumLinMom) * a.invSumMass;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID) a.motionVelocities[linkID].bottom += linDelta;
    {
      const V3 predictedCOM = a.com + posMomentum * (a.invSumMass * dt);
      const V3 posCorrection = predictedCOM - COM;
      for (uint32_t linkID = 0; linkID < linkCount; ++linkID) a.bodies[linkID].body2World.p += posCorrection;
      COM += posCorrection;
    }
  }
  {  // updateJointProperties (:1971)
    const float invDt = 1.f / a.dt;
    for (uint32_t i = 0; i < a.dofs; ++i) {
      const float jNewVel = a.jointNewVelocity[i];
      const float delta = jNewVel - a.jointVelocity[i];
      a.jointVelocity[i] = jNewVel;
      a.jointAcceleration[i] += delta * invDt;
    }
  }
  for (uint32_t linkID = 0; linkID < linkCount; ++linkID) {
    a.bodies[linkID].linVel = a.motionVelocities[linkID].bottom;
    a.bodies[linkID].angVel = a.motionVelocities[linkID].top;
    if (!a.bodies[linkID].retainAccelerations) a.externalAcceleration[linkID] = LinAng{V3{0, 0, 0}, V3{0, 0, 0}};
  }
}

// ---------------------------------------------------------------- 잠: ArticulationSim::sleepCheck (ScArticulationSim.cpp:432)
// 반환: true = 모든 링크 깸 카운터가 0 -> 섬 관리자에 비활성 요청(deactivateNode). 실제 재우기(putToSleep)는 섬 관리자 차례에.
EHD float updateLinkWakeCounter(LinkBody& b, float dt, float energyThreshold, const V3& motionLinear, const V3& motionAngular) {  // ScBodySim.cpp:581
  const float wakeCounterResetTime = 20.0f * 0.02f;
  float wc = b.wakeCounter;
  {
    V3 bcSleepLinVelAcc = b.sleepLinVelAcc;
    V3 bcSleepAngVelAcc = b.sleepAngVelAcc;
    if (wc < wakeCounterResetTime * 0.5f || wc < dt) {
      const Tf& body2World = b.body2World;
      const V3 t = b.invInertia;
      const V3 inertia{t.x > 0.0f ? 1.0f / t.x : 1.0f, t.y > 0.0f ? 1.0f / t.y : 1.0f, t.z > 0.0f ? 1.0f / t.z : 1.0f};
      const V3 sleepLinVelAcc = motionLinear;
      const V3 sleepAngVelAcc = rotateInv(body2World.q, motionAngular);
      bcSleepLinVelAcc += sleepLinVelAcc;
      bcSleepAngVelAcc += sleepAngVelAcc;
      float invMass = b.invMass;
      if (invMass == 0.0f) invMass = 1.0f;
      const float angular = dot(mulc(bcSleepAngVelAcc, bcSleepAngVelAcc), inertia) * invMass;
      const float linear = magSq(bcSleepLinVelAcc);
      const float normalizedEnergy = 0.5f * (angular + linear);
      const float clusterFactor = float(1 + b.numCountedInteractions);
      const float threshold = clusterFactor * energyThreshold;
      if (normalizedEnergy >= threshold) {
        b.sleepLinVelAcc = V3{0, 0, 0};  // resetSleepFilter
        b.sleepAngVelAcc = V3{0, 0, 0};
        const float factor = threshold == 0.0f ? 2.0f : pmin(normalizedEnergy / threshold, 2.0f);
        wc = factor * 0.5f * wakeCounterResetTime + dt * (clusterFactor - 1.0f);
        b.wakeCounter = wc;
        return wc;
      }
    }
    b.sleepLinVelAcc = bcSleepLinVelAcc;
    b.sleepAngVelAcc = bcSleepAngVelAcc;
  }
  wc = pmax(wc - dt, 0.0f);
  b.wakeCounter = wc;
  return wc;
}
EHD bool sleepCheck(Articulation& a, float dt) {
  if (!a.nLinks) return false;
  if (!a.awake) return false;
  const float sleepThreshold = a.sleepThreshold;
  float maxTimer = 0.0f, minTimer = kMaxF32;
  for (uint32_t i = 0; i < a.nLinks; i++) {
    const SV& mv = a.posIterMotionVelocities[i];  // getMotionVelocity (:937)
    const float timer = updateLinkWakeCounter(a.bodies[i], dt, sleepThreshold, mv.bottom, mv.top);
    maxTimer = pmax(maxTimer, timer);
    minTimer = pmin(minTimer, timer);
  }
  a.wakeCounter = maxTimer;
  if (maxTimer != 0.0f) {
    if (minTimer == 0.0f)
      for (uint32_t i = 0; i < a.nLinks; i++) a.bodies[i].wakeCounter = pmax(1e-6f, a.bodies[i].wakeCounter);
    return false;
  }
  for (uint32_t i = 0; i < a.nLinks; i++) {
    a.bodies[i].sleepLinVelAcc = V3{0, 0, 0};
    a.bodies[i].sleepAngVelAcc = V3{0, 0, 0};
  }
  a.readyForSleep = 1;
  return true;
}
// ArticulationSim::putToSleep (:406)
EHD void putToSleep(Articulation& a) {
  for (uint32_t i = 0; i < a.nLinks; i++) {
    a.bodies[i].wakeCounter = 0.0f;
    a.bodies[i].linVel = V3{0.0f, 0.0f, 0.0f};
    a.bodies[i].angVel = V3{0.0f, 0.0f, 0.0f};
  }
  a.wakeCounter = 0.0f;
  a.awake = 0;
  a.readyForSleep = 0;
}

// ---------------------------------------------------------------- 관절체만 있는 섬 한 simulate (DyTGSDynamics.cpp:2531-2580 + 앞뒤)
struct StepParams {
  V3 gravity;
  float dt;
  float lengthScale = 1.0f;
  uint32_t posIters = 0, velIters = 0;  // 0 이면 관절체 자기 값 (섬 묶음의 최댓값이 들어와야 한다 — solver 몫)
  bool externalForcesEveryTgsIteration = false;
  bool solveArticulationContactLast = false;
};
template <class Static = NoStatic>
EHD void stepAlone(Articulation& a, const StepParams& p, const Static& st = Static{}) {
  if (!a.inScene || !a.awake) return;
  // 앞 스텝 sleepCheck 가 비활성 요청(deactivateNode)을 냈으면: 이번 스텝 풀이는 그대로 하고(섬 관리가 풀이와 병렬이라 풀이 목록에 남음),
  // 적분 뒤 sleepCheck 없이 재운다 (ScPipeline.cpp:2689-2707 afterIntegration, 관절체는 강체와 달리 자세를 되돌리지 않음)
  const bool deactivating = a.readyForSleep != 0;
  const float dt = p.dt;
  const uint32_t posIters = p.posIters ? p.posIters : uint32_t(a.solverIterationCounts & 0xff);
  const uint32_t velIters = p.velIters ? p.velIters : uint32_t(a.solverIterationCounts >> 8);
  const float stepDt = dt / float(posIters);             // :1898
  const float invStepDt = 1.f / stepDt;                  // :1899
  const float biasCoefficient = 2.f * psqrt(1.f / float(posIters));  // :1900
  const float invDt = 1.f / dt;                          // mInvDt
  const float invLengthScale = 1.f / p.lengthScale;      // :1660
  SV scratch[kMaxLinks];
  computeUnconstrainedVelocitiesTGS(a, dt, p.gravity, invLengthScale, p.externalForcesEveryTgsIteration);
  setupSolverConstraintsTGS(a, stepDt, invStepDt, dt);
  const ProcessConfig single = singlePassConfig(p.solveArticulationContactLast);
  const float recipStepDt = 1.0f / stepDt;
  float elapsedTime = 0.0f;
  for (uint32_t it = 0; it < posIters; it++) {
    if (p.externalForcesEveryTgsIteration) applyTgsSubstepForces(a, stepDt, scratch);
    solveInternalConstraints(a, dt, stepDt, recipStepDt, false, true, single, elapsedTime, biasCoefficient, p.externalForcesEveryTgsIteration, st);
    recordDeltaMotion(a, stepDt, scratch);
    elapsedTime += stepDt;
  }
  saveVelocityTGS(a, invDt);
  // concludeInternalConstraints: 정적 제약만 (호출자 몫)
  for (uint32_t it = 0; it < velIters; ++it)
    solveInternalConstraints(a, dt, stepDt, recipStepDt, true, true, single, elapsedTime, biasCoefficient, p.externalForcesEveryTgsIteration, st);
  // writebackInternalConstraints: 정적 제약만
  updateBodiesTGS(a, dt, scratch);
  if (deactivating) putToSleep(a);
  else sleepCheck(a, dt);
}

}  // namespace art
}  // namespace eng
