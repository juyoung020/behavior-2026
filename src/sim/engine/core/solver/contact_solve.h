// TGS 접촉 제약 풀이·되쓰기 (손으로 짬, PhysX 5.6.1 과 비트 동일 목표). 단일 경로와 4개 묶음 경로.
// 원본 (physx/source/lowleveldynamics/src/)
//   DyTGSContactPrep.cpp:1492-1861  solveDynamicContactsStep / solveContact
//   DyTGSContactPrep.cpp:1863-1937  writeBackContact (동역학에 남는 것: 마찰 패치 broken 표시)
//   DyTGSContactPrep.cpp:2784       concludeContactStep (빈 함수)
//   DyTGSContactPrepBlock.cpp:2280-2765 solveContact4_Block, :2775-2937 writeBackContact4_Block, :3532 concludeContact4_Block (빈 함수)
// 접촉 힘 보고(contactForces)·마찰 충격 보고(writeBackFriction)·잔차 누적(residual)은 동역학에 영향이 없어 옮기지 않았다.
#pragma once
#include "saos.h"
#include "tgs_types.h"

namespace eng {
namespace sv {

SV_HD V4 ldf3(const float* p) { return V4{p[0], p[1], p[2], 0.0f}; }  // V3LoadA
SV_HD void stf3(V4 a, float* p) { p[0] = a.f[0]; p[1] = a.f[1]; p[2] = a.f[2]; }

// DyTGSContactPrep.cpp:1492
SV_HD FV solveDynamicContactsStep(SolverContactPointStep* contacts, uint32_t nbContactPoints, V4 contactNormal, FV invMassA, FV invMassB,
                                  V4& linVel0_, V4& angState0_, V4& linVel1_, V4& angState1_, float* forceBuffer, V4 angMotion0,
                                  V4 angMotion1, V4 linRelMotion, FV maxPenBias, FV angD0, FV angD1, FV minPen, FV elapsedTime) {
  V4 linVel0 = linVel0_;
  V4 angState0 = angState0_;
  V4 linVel1 = linVel1_;
  V4 angState1 = angState1_;
  const FV zero = FZero();
  FV accumulatedNormalImpulse = zero;
  const V4 normalInvMass0 = V3Scale(contactNormal, invMassA);
  const V4 normalInvMass1 = V3Scale(contactNormal, invMassB);
  const FV deltaV = V3Dot(linRelMotion, contactNormal);
  for (uint32_t i = 0; i < nbContactPoints; i++) {
    SolverContactPointStep& c = contacts[i];
    const V4 raXnI = ldv3(c.raXnI);
    const V4 rbXnI = ldv3(c.rbXnI);
    const FV angDelta0 = V3Dot(angMotion0, raXnI);
    const FV angDelta1 = V3Dot(angMotion1, rbXnI);
    const FV deltaAng = FSub(angDelta0, angDelta1);
    const FV targetVel = FLoad(c.targetVelocity);
    const FV deltaBias = FSub(FAdd(deltaV, deltaAng), FMul(targetVel, elapsedTime));
    const FV biasCoefficient = FLoad(c.biasCoefficient);
    const FV sep = FMax(minPen, FAdd(FLoad(c.separation), deltaBias));
    const FV bias = FMin(FNeg(maxPenBias), FMul(biasCoefficient, sep));
    const V4 v0 = V3MulAdd(linVel0, contactNormal, V3Mul(angState0, raXnI));
    const V4 v1 = V3MulAdd(linVel1, contactNormal, V3Mul(angState1, rbXnI));
    const FV normalVel = V3SumElems(V3Sub(v0, v1));
    const FV velMultiplier = FLoad(c.velMultiplier);
    const FV biasNV = FMul(bias, FLoad(c.recipResponse));
    const FV lambda = FNegScaleSub(FSub(normalVel, targetVel), velMultiplier, biasNV);
    const FV appliedForce = FLoad(forceBuffer[i]);
    const FV maxImpulse = FLoad(c.maxImpulse);
    const FV _deltaF = FMax(lambda, FNeg(appliedForce));
    const FV _newForce = FAdd(appliedForce, _deltaF);
    const FV newForce = FMin(_newForce, maxImpulse);
    const FV deltaF = FSub(newForce, appliedForce);
    linVel0 = V3ScaleAdd(normalInvMass0, deltaF, linVel0);
    linVel1 = V3NegScaleSub(normalInvMass1, deltaF, linVel1);
    angState0 = V3ScaleAdd(raXnI, FMul(deltaF, angD0), angState0);
    angState1 = V3NegScaleSub(rbXnI, FMul(deltaF, angD1), angState1);
    FStore(newForce, &forceBuffer[i]);
    accumulatedNormalImpulse = FAdd(accumulatedNormalImpulse, newForce);
  }
  linVel0_ = linVel0;
  angState0_ = angState0;
  linVel1_ = linVel1;
  angState1_ = angState1;
  return accumulatedNormalImpulse;
}

// DyTGSContactPrep.cpp:1581 solveContact
SV_HDN void solveContact(const SDesc& desc, SBodyVel* vels, ByteArena& arena, bool doFriction, float minPenetration, float elapsedTimeF32) {
  SBodyVel& b0 = vels[desc.bodyA];
  SBodyVel& b1 = vels[desc.bodyB];
  const FV minPen = FLoad(minPenetration);
  V4 linVel0 = ldf3(b0.lin);
  V4 linVel1 = ldf3(b1.lin);
  V4 angState0 = ldf3(b0.ang);
  V4 angState1 = ldf3(b1.ang);
  const V4 angMotion0 = ldf3(b0.deltaAngDt);
  const V4 angMotion1 = ldf3(b1.deltaAngDt);
  const V4 linMotion0 = ldf3(b0.deltaLinDt);
  const V4 linMotion1 = ldf3(b1.deltaLinDt);
  const V4 relMotion = V3Sub(linMotion0, linMotion1);
  const FV zero = FZero();
  const FV elapsedTime = FLoad(elapsedTimeF32);
  uint8_t* currPtr = arenaPtr<uint8_t>(arena, desc.constraint);
  const uint8_t* last = currPtr + (uint32_t(desc.constraintLengthOver16) << 4);
  while (currPtr < last) {
    SolverContactHeaderStep* hdr = reinterpret_cast<SolverContactHeaderStep*>(currPtr);
    currPtr += sizeof(SolverContactHeaderStep);
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    SolverContactPointStep* contacts = reinterpret_cast<SolverContactPointStep*>(currPtr);
    currPtr += numNormalConstr * sizeof(SolverContactPointStep);
    float* forceBuffer = reinterpret_cast<float*>(currPtr);
    currPtr += sizeof(float) * ((numNormalConstr + 3) & (~3u));
    SolverContactFrictionStep* frictions = reinterpret_cast<SolverContactFrictionStep*>(currPtr);
    currPtr += numFrictionConstr * sizeof(SolverContactFrictionStep);
    const FV invMassA = FLoad(hdr->invMass0);
    const FV invMassB = FLoad(hdr->invMass1);
    const FV angDom0 = FLoad(hdr->angDom0);
    const FV angDom1 = FLoad(hdr->angDom1);
    const V4 contactNormal = ldv3(hdr->normal);
    const FV maxPenBias = FLoad(hdr->maxPenBias);
    const FV accumulatedNormalImpulse =
        solveDynamicContactsStep(contacts, numNormalConstr, contactNormal, invMassA, invMassB, linVel0, angState0, linVel1, angState1, forceBuffer,
                                 angMotion0, angMotion1, relMotion, maxPenBias, angDom0, angDom1, minPen, elapsedTime);
    FStore(accumulatedNormalImpulse, &hdr->minNormalForce);
    if (numFrictionConstr && doFriction) {
      const FV staticFrictionCof = V4GetX(hdr->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W);
      const FV dynamicFrictionCof = V4GetY(hdr->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W);
      const FV maxFrictionImpulse = FMul(staticFrictionCof, accumulatedNormalImpulse);
      const FV maxDynFrictionImpulse = FMul(dynamicFrictionCof, accumulatedNormalImpulse);
      const FV negMaxDynFrictionImpulse = FNeg(maxDynFrictionImpulse);
      BoolV broken = BFFFF();
      const uint32_t numFrictionPairs = (numFrictionConstr & 6);
      for (uint32_t i = 0; i < numFrictionPairs; i += 2) {
        SolverContactFrictionStep& f0 = frictions[i];
        SolverContactFrictionStep& f1 = frictions[i + 1];
        const FV frictionScale = FLoad(f0.frictionScale);
        const V4 normalXYZ_ErrorW0 = f0.normalXYZ_ErrorW;
        const V4 raXnI_targetVelW0 = f0.raXnI_targetVelW;
        const V4 rbXnI_velMultiplierW0 = f0.rbXnI_velMultiplierW;
        const V4 normalXYZ_ErrorW1 = f1.normalXYZ_ErrorW;
        const V4 raXnI_targetVelW1 = f1.raXnI_targetVelW;
        const V4 rbXnI_velMultiplierW1 = f1.rbXnI_velMultiplierW;
        const V4 normal0 = Vec3V_From_Vec4V(normalXYZ_ErrorW0);
        const V4 normal1 = Vec3V_From_Vec4V(normalXYZ_ErrorW1);
        const FV initialError0 = V4GetW(normalXYZ_ErrorW0);
        const FV initialError1 = V4GetW(normalXYZ_ErrorW1);
        const FV biasScale = FLoad(f0.biasScale);
        const V4 raXnI0 = Vec3V_From_Vec4V(raXnI_targetVelW0);
        const V4 rbXnI0 = Vec3V_From_Vec4V(rbXnI_velMultiplierW0);
        const V4 raXnI1 = Vec3V_From_Vec4V(raXnI_targetVelW1);
        const V4 rbXnI1 = Vec3V_From_Vec4V(rbXnI_velMultiplierW1);
        const FV appliedForce0 = FLoad(f0.appliedForce);
        const FV appliedForce1 = FLoad(f1.appliedForce);
        const FV targetVel0 = V4GetW(raXnI_targetVelW0);
        const FV targetVel1 = V4GetW(raXnI_targetVelW1);
        FV deltaV0 = FAdd(FSub(V3Dot(raXnI0, angMotion0), V3Dot(rbXnI0, angMotion1)), V3Dot(normal0, relMotion));
        FV deltaV1 = FAdd(FSub(V3Dot(raXnI1, angMotion0), V3Dot(rbXnI1, angMotion1)), V3Dot(normal1, relMotion));
        deltaV0 = FSub(deltaV0, FMul(targetVel0, elapsedTime));
        deltaV1 = FSub(deltaV1, FMul(targetVel1, elapsedTime));
        const FV error0 = FAdd(initialError0, deltaV0);
        const FV error1 = FAdd(initialError1, deltaV1);
        const FV bias0 = FMul(error0, biasScale);
        const FV bias1 = FMul(error1, biasScale);
        const FV velMultiplier0 = V4GetW(rbXnI_velMultiplierW0);
        const FV velMultiplier1 = V4GetW(rbXnI_velMultiplierW1);
        const V4 delLinVel00 = V3Scale(normal0, invMassA);
        const V4 delLinVel10 = V3Scale(normal0, invMassB);
        const V4 delLinVel01 = V3Scale(normal1, invMassA);
        const V4 delLinVel11 = V3Scale(normal1, invMassB);
        const V4 v00 = V3MulAdd(linVel0, normal0, V3Mul(angState0, raXnI0));
        const V4 v10 = V3MulAdd(linVel1, normal0, V3Mul(angState1, rbXnI0));
        const FV normalVel0 = V3SumElems(V3Sub(v00, v10));
        const V4 v01 = V3MulAdd(linVel0, normal1, V3Mul(angState0, raXnI1));
        const V4 v11 = V3MulAdd(linVel1, normal1, V3Mul(angState1, rbXnI1));
        const FV normalVel1 = V3SumElems(V3Sub(v01, v11));
        const FV tmp10 = FNegScaleSub(FSub(bias0, targetVel0), velMultiplier0, appliedForce0);
        const FV tmp11 = FNegScaleSub(FSub(bias1, targetVel1), velMultiplier1, appliedForce1);
        const FV totalImpulse0 = FNegScaleSub(normalVel0, velMultiplier0, tmp10);
        const FV totalImpulse1 = FNegScaleSub(normalVel1, velMultiplier1, tmp11);
        const FV totalImpulse = FSqrt(FAdd(FMul(totalImpulse0, totalImpulse0), FMul(totalImpulse1, totalImpulse1)));
        const BS clamp = FIsGrtr(totalImpulse, FMul(frictionScale, maxFrictionImpulse));
        const FV totalClamped = FSel(clamp, FMin(FMul(frictionScale, maxDynFrictionImpulse), totalImpulse), totalImpulse);
        const FV ratio = FSel(FIsGrtr(totalImpulse, zero), FDiv(totalClamped, totalImpulse), zero);
        const FV newAppliedForce0 = FMul(totalImpulse0, ratio);
        const FV newAppliedForce1 = FMul(totalImpulse1, ratio);
        broken = BOr(broken, clamp);
        const FV deltaF0 = FSub(newAppliedForce0, appliedForce0);
        const FV deltaF1 = FSub(newAppliedForce1, appliedForce1);
        linVel0 = V3ScaleAdd(delLinVel00, deltaF0, V3ScaleAdd(delLinVel01, deltaF1, linVel0));
        linVel1 = V3NegScaleSub(delLinVel10, deltaF0, V3NegScaleSub(delLinVel11, deltaF1, linVel1));
        angState0 = V3ScaleAdd(raXnI0, FMul(deltaF0, angDom0), V3ScaleAdd(raXnI1, FMul(deltaF1, angDom0), angState0));
        angState1 = V3NegScaleSub(rbXnI0, FMul(deltaF0, angDom1), V3NegScaleSub(rbXnI1, FMul(deltaF1, angDom1), angState1));
        FStore(newAppliedForce0, &f0.appliedForce);
        FStore(newAppliedForce1, &f1.appliedForce);
      }
      for (uint32_t i = numFrictionPairs; i < numFrictionConstr; i++) {
        SolverContactFrictionStep& f = frictions[i];
        const FV frictionScale = FLoad(f.frictionScale);
        const V4 raXnI_targetVelW = f.raXnI_targetVelW;
        const V4 rbXnI_velMultiplierW = f.rbXnI_velMultiplierW;
        const V4 raXnI = Vec3V_From_Vec4V(raXnI_targetVelW);
        const V4 rbXnI = Vec3V_From_Vec4V(rbXnI_velMultiplierW);
        const FV appliedForce = FLoad(f.appliedForce);
        const FV targetVel = V4GetW(raXnI_targetVelW);
        const FV velMultiplier = V4GetW(rbXnI_velMultiplierW);
        const V4 v0 = V3Mul(angState0, raXnI);
        const V4 v1 = V3Mul(angState1, rbXnI);
        const FV normalVel = V3SumElems(V3Sub(v0, v1));
        const FV tmp1 = FNegScaleSub(FNeg(targetVel), velMultiplier, appliedForce);
        const FV totalImpulse = FNegScaleSub(normalVel, velMultiplier, tmp1);
        const BS clamp = FIsGrtr(FAbs(totalImpulse), FMul(frictionScale, maxFrictionImpulse));
        const FV totalClamped =
            FMin(FMul(frictionScale, maxDynFrictionImpulse), FMax(FMul(frictionScale, negMaxDynFrictionImpulse), totalImpulse));
        const FV newAppliedForce = FSel(clamp, totalClamped, totalImpulse);
        broken = BOr(broken, clamp);
        const FV deltaF = FSub(newAppliedForce, appliedForce);
        angState0 = V3ScaleAdd(raXnI, FMul(deltaF, angDom0), angState0);
        angState1 = V3NegScaleSub(rbXnI, FMul(deltaF, angDom1), angState1);
        FStore(newAppliedForce, &f.appliedForce);
      }
      Store_From_BoolV(broken, &hdr->broken);
    }
  }
  stf3(linVel0, b0.lin);
  stf3(linVel1, b1.lin);
  stf3(angState0, b0.ang);
  stf3(angState1, b1.ang);
}

// DyTGSContactPrep.cpp:1863 writeBackContact — 동역학에 남는 부분: broken 이면 이번 스텝 마찰 패치에 표시
SV_HDN void writeBackContact(const SDesc& desc, ByteArena& arena, FrictionArena& frictionCur) {
  uint8_t* cPtr = arenaPtr<uint8_t>(arena, desc.constraint);
  const uint8_t* last = cPtr + (uint32_t(desc.constraintLengthOver16) << 4);
  while (cPtr < last) {
    const SolverContactHeaderStep* hdr = reinterpret_cast<const SolverContactHeaderStep*>(cPtr);
    cPtr += sizeof(SolverContactHeaderStep);
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    cPtr += sizeof(SolverContactPointStep) * numNormalConstr;
    cPtr += sizeof(float) * ((numNormalConstr + 3) & (~3u));
    if (hdr->broken && hdr->frictionBrokenWriteback != NONE) frictionCur.data[hdr->frictionBrokenWriteback].broken = 1;
    cPtr += numFrictionConstr * sizeof(SolverContactFrictionStep);
  }
}

// DyTGSContactPrepBlock.cpp:2280 solveContact4_Block
SV_HDN void solveContact4_Block(const SDesc* desc, SBodyVel* vels, ByteArena& arena, float minPenetration, float elapsedTimeF32) {
  SBodyVel* b0[4] = {&vels[desc[0].bodyA], &vels[desc[1].bodyA], &vels[desc[2].bodyA], &vels[desc[3].bodyA]};
  SBodyVel* b1[4] = {&vels[desc[0].bodyB], &vels[desc[1].bodyB], &vels[desc[2].bodyB], &vels[desc[3].bodyB]};
  const V4 minPen = V4Load(minPenetration);
  const V4 elapsedTime = V4Load(elapsedTimeF32);
  const V4 vZero = V4Zero();
  // PX_TRANSPOSE_44 로 네 칸 속도를 X/Y/Z 줄로 (W 줄은 건드리지 않고 그대로 되쓰므로 생략)
  V4 linVel0T0 = V4{b0[0]->lin[0], b0[1]->lin[0], b0[2]->lin[0], b0[3]->lin[0]};
  V4 linVel0T1 = V4{b0[0]->lin[1], b0[1]->lin[1], b0[2]->lin[1], b0[3]->lin[1]};
  V4 linVel0T2 = V4{b0[0]->lin[2], b0[1]->lin[2], b0[2]->lin[2], b0[3]->lin[2]};
  V4 linVel1T0 = V4{b1[0]->lin[0], b1[1]->lin[0], b1[2]->lin[0], b1[3]->lin[0]};
  V4 linVel1T1 = V4{b1[0]->lin[1], b1[1]->lin[1], b1[2]->lin[1], b1[3]->lin[1]};
  V4 linVel1T2 = V4{b1[0]->lin[2], b1[1]->lin[2], b1[2]->lin[2], b1[3]->lin[2]};
  V4 angState0T0 = V4{b0[0]->ang[0], b0[1]->ang[0], b0[2]->ang[0], b0[3]->ang[0]};
  V4 angState0T1 = V4{b0[0]->ang[1], b0[1]->ang[1], b0[2]->ang[1], b0[3]->ang[1]};
  V4 angState0T2 = V4{b0[0]->ang[2], b0[1]->ang[2], b0[2]->ang[2], b0[3]->ang[2]};
  V4 angState1T0 = V4{b1[0]->ang[0], b1[1]->ang[0], b1[2]->ang[0], b1[3]->ang[0]};
  V4 angState1T1 = V4{b1[0]->ang[1], b1[1]->ang[1], b1[2]->ang[1], b1[3]->ang[1]};
  V4 angState1T2 = V4{b1[0]->ang[2], b1[1]->ang[2], b1[2]->ang[2], b1[3]->ang[2]};
  const V4 linDelta0T0 = V4{b0[0]->deltaLinDt[0], b0[1]->deltaLinDt[0], b0[2]->deltaLinDt[0], b0[3]->deltaLinDt[0]};
  const V4 linDelta0T1 = V4{b0[0]->deltaLinDt[1], b0[1]->deltaLinDt[1], b0[2]->deltaLinDt[1], b0[3]->deltaLinDt[1]};
  const V4 linDelta0T2 = V4{b0[0]->deltaLinDt[2], b0[1]->deltaLinDt[2], b0[2]->deltaLinDt[2], b0[3]->deltaLinDt[2]};
  const V4 linDelta1T0 = V4{b1[0]->deltaLinDt[0], b1[1]->deltaLinDt[0], b1[2]->deltaLinDt[0], b1[3]->deltaLinDt[0]};
  const V4 linDelta1T1 = V4{b1[0]->deltaLinDt[1], b1[1]->deltaLinDt[1], b1[2]->deltaLinDt[1], b1[3]->deltaLinDt[1]};
  const V4 linDelta1T2 = V4{b1[0]->deltaLinDt[2], b1[1]->deltaLinDt[2], b1[2]->deltaLinDt[2], b1[3]->deltaLinDt[2]};
  const V4 angDelta0T0 = V4{b0[0]->deltaAngDt[0], b0[1]->deltaAngDt[0], b0[2]->deltaAngDt[0], b0[3]->deltaAngDt[0]};
  const V4 angDelta0T1 = V4{b0[0]->deltaAngDt[1], b0[1]->deltaAngDt[1], b0[2]->deltaAngDt[1], b0[3]->deltaAngDt[1]};
  const V4 angDelta0T2 = V4{b0[0]->deltaAngDt[2], b0[1]->deltaAngDt[2], b0[2]->deltaAngDt[2], b0[3]->deltaAngDt[2]};
  const V4 angDelta1T0 = V4{b1[0]->deltaAngDt[0], b1[1]->deltaAngDt[0], b1[2]->deltaAngDt[0], b1[3]->deltaAngDt[0]};
  const V4 angDelta1T1 = V4{b1[0]->deltaAngDt[1], b1[1]->deltaAngDt[1], b1[2]->deltaAngDt[1], b1[3]->deltaAngDt[1]};
  const V4 angDelta1T2 = V4{b1[0]->deltaAngDt[2], b1[1]->deltaAngDt[2], b1[2]->deltaAngDt[2], b1[3]->deltaAngDt[2]};
  uint8_t* currPtr = arenaPtr<uint8_t>(arena, desc[0].constraint);
  const uint8_t* last = currPtr + (uint32_t(desc[0].constraintLengthOver16) << 4);
  const V4 vMax = V4Splat(FMax());
  SolverContactHeaderStepBlock* hdr = reinterpret_cast<SolverContactHeaderStepBlock*>(currPtr);
  const V4 invMassA = hdr->invMass0D0;
  const V4 invMassB = hdr->invMass1D1;
  const V4 sumInvMass = V4Add(invMassA, invMassB);
  const V4 linDeltaX = V4Sub(linDelta0T0, linDelta1T0);
  const V4 linDeltaY = V4Sub(linDelta0T1, linDelta1T1);
  const V4 linDeltaZ = V4Sub(linDelta0T2, linDelta1T2);
  while (currPtr < last) {
    hdr = reinterpret_cast<SolverContactHeaderStepBlock*>(currPtr);
    currPtr = reinterpret_cast<uint8_t*>(hdr + 1);
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    const bool hasMaxImpulse = (hdr->flag & SolverContactHeaderStepBlock::eHAS_MAX_IMPULSE) != 0;
    V4* appliedForces = reinterpret_cast<V4*>(currPtr);
    currPtr += sizeof(V4) * numNormalConstr;
    SolverContactPointStepBlock* contacts = reinterpret_cast<SolverContactPointStepBlock*>(currPtr);
    const V4* maxImpulses;
    currPtr = reinterpret_cast<uint8_t*>(contacts + numNormalConstr);
    uint32_t maxImpulseMask = 0;
    if (hasMaxImpulse) {
      maxImpulseMask = 0xFFFFFFFF;
      maxImpulses = reinterpret_cast<V4*>(currPtr);
      currPtr += sizeof(V4) * numNormalConstr;
    } else {
      maxImpulses = &vMax;
    }
    V4* frictionAppliedForce = reinterpret_cast<V4*>(currPtr);
    currPtr += sizeof(V4) * numFrictionConstr;
    const SolverContactFrictionStepBlock* frictions = reinterpret_cast<SolverContactFrictionStepBlock*>(currPtr);
    currPtr += numFrictionConstr * sizeof(SolverContactFrictionStepBlock);
    V4 accumulatedNormalImpulse = vZero;
    const V4 angD0 = hdr->angDom0;
    const V4 angD1 = hdr->angDom1;
    const V4 _normalT0 = hdr->normalX;
    const V4 _normalT1 = hdr->normalY;
    const V4 _normalT2 = hdr->normalZ;
    V4 contactNormalVel1 = V4Mul(linVel0T0, _normalT0);
    V4 contactNormalVel3 = V4Mul(linVel1T0, _normalT0);
    contactNormalVel1 = V4MulAdd(linVel0T1, _normalT1, contactNormalVel1);
    contactNormalVel3 = V4MulAdd(linVel1T1, _normalT1, contactNormalVel3);
    contactNormalVel1 = V4MulAdd(linVel0T2, _normalT2, contactNormalVel1);
    contactNormalVel3 = V4MulAdd(linVel1T2, _normalT2, contactNormalVel3);
    const V4 maxPenBias = hdr->maxPenBias;
    V4 relVel1 = V4Sub(contactNormalVel1, contactNormalVel3);
    V4 deltaNormalV = V4Mul(linDeltaX, _normalT0);
    deltaNormalV = V4MulAdd(linDeltaY, _normalT1, deltaNormalV);
    deltaNormalV = V4MulAdd(linDeltaZ, _normalT2, deltaNormalV);
    V4 accumDeltaF = vZero;
    for (uint32_t i = 0; i < numNormalConstr; i++) {
      const SolverContactPointStepBlock& c = contacts[i];
      const V4 appliedForce = appliedForces[i];
      const V4 maxImpulse = maxImpulses[i & maxImpulseMask];
      V4 contactNormalVel2 = V4Mul(c.raXnI[0], angState0T0);
      V4 contactNormalVel4 = V4Mul(c.rbXnI[0], angState1T0);
      contactNormalVel2 = V4MulAdd(c.raXnI[1], angState0T1, contactNormalVel2);
      contactNormalVel4 = V4MulAdd(c.rbXnI[1], angState1T1, contactNormalVel4);
      contactNormalVel2 = V4MulAdd(c.raXnI[2], angState0T2, contactNormalVel2);
      contactNormalVel4 = V4MulAdd(c.rbXnI[2], angState1T2, contactNormalVel4);
      const V4 normalVel = V4Add(relVel1, V4Sub(contactNormalVel2, contactNormalVel4));
      V4 angDelta0 = V4Mul(angDelta0T0, c.raXnI[0]);
      V4 angDelta1 = V4Mul(angDelta1T0, c.rbXnI[0]);
      angDelta0 = V4MulAdd(angDelta0T1, c.raXnI[1], angDelta0);
      angDelta1 = V4MulAdd(angDelta1T1, c.rbXnI[1], angDelta1);
      angDelta0 = V4MulAdd(angDelta0T2, c.raXnI[2], angDelta0);
      angDelta1 = V4MulAdd(angDelta1T2, c.rbXnI[2], angDelta1);
      const V4 deltaAng = V4Sub(angDelta0, angDelta1);
      const V4 targetVel = c.targetVelocity;
      const V4 deltaBias = V4Sub(V4Add(deltaNormalV, deltaAng), V4Mul(targetVel, elapsedTime));
      const V4 biasCoefficient = c.biasCoefficient;
      const V4 sep = V4Max(minPen, V4Add(c.separation, deltaBias));
      const V4 bias = V4Min(V4Neg(maxPenBias), V4Mul(biasCoefficient, sep));
      const V4 velMultiplier = c.velMultiplier;
      const V4 tVelBias = V4Mul(bias, c.recipResponse);
      const V4 _deltaF = V4Max(V4Sub(tVelBias, V4Mul(V4Sub(normalVel, targetVel), velMultiplier)), V4Neg(appliedForce));
      const V4 newAppliedForce = V4Min(V4Add(appliedForce, _deltaF), maxImpulse);
      const V4 deltaF = V4Sub(newAppliedForce, appliedForce);
      accumDeltaF = V4Add(accumDeltaF, deltaF);
      const V4 angDetaF0 = V4Mul(deltaF, angD0);
      const V4 angDetaF1 = V4Mul(deltaF, angD1);
      relVel1 = V4MulAdd(sumInvMass, deltaF, relVel1);
      angState0T0 = V4MulAdd(c.raXnI[0], angDetaF0, angState0T0);
      angState1T0 = V4NegMulSub(c.rbXnI[0], angDetaF1, angState1T0);
      angState0T1 = V4MulAdd(c.raXnI[1], angDetaF0, angState0T1);
      angState1T1 = V4NegMulSub(c.rbXnI[1], angDetaF1, angState1T1);
      angState0T2 = V4MulAdd(c.raXnI[2], angDetaF0, angState0T2);
      angState1T2 = V4NegMulSub(c.rbXnI[2], angDetaF1, angState1T2);
      appliedForces[i] = newAppliedForce;
      accumulatedNormalImpulse = V4Add(accumulatedNormalImpulse, newAppliedForce);
    }
    const V4 accumDeltaF_IM0 = V4Mul(accumDeltaF, invMassA);
    const V4 accumDeltaF_IM1 = V4Mul(accumDeltaF, invMassB);
    linVel0T0 = V4MulAdd(_normalT0, accumDeltaF_IM0, linVel0T0);
    linVel1T0 = V4NegMulSub(_normalT0, accumDeltaF_IM1, linVel1T0);
    linVel0T1 = V4MulAdd(_normalT1, accumDeltaF_IM0, linVel0T1);
    linVel1T1 = V4NegMulSub(_normalT1, accumDeltaF_IM1, linVel1T1);
    linVel0T2 = V4MulAdd(_normalT2, accumDeltaF_IM0, linVel0T2);
    linVel1T2 = V4NegMulSub(_normalT2, accumDeltaF_IM1, linVel1T2);
    if (numFrictionConstr) {  // doFriction 은 항상 true
      const V4 staticFric = hdr->staticFriction;
      const V4 dynamicFric = hdr->dynamicFriction;
      const V4 maxFrictionImpulse = V4Add(V4Mul(staticFric, accumulatedNormalImpulse), V4Load(1e-5f));
      const V4 maxDynFrictionImpulse = V4Mul(dynamicFric, accumulatedNormalImpulse);
      BV broken = BFFFF();
      for (uint32_t i = 0; i < numFrictionConstr; i += 2) {
        const SolverContactFrictionStepBlock& f0 = frictions[i];
        const SolverContactFrictionStepBlock& f1 = frictions[i + 1];
        const V4 appliedForce0 = frictionAppliedForce[i];
        const V4 appliedForce1 = frictionAppliedForce[i + 1];
        const V4 normalT00 = f0.normal[0];
        const V4 normalT10 = f0.normal[1];
        const V4 normalT20 = f0.normal[2];
        const V4 normalT01 = f1.normal[0];
        const V4 normalT11 = f1.normal[1];
        const V4 normalT21 = f1.normal[2];
        V4 normalVel10 = V4Mul(linVel0T0, normalT00);
        V4 normalVel20 = V4Mul(f0.raXnI[0], angState0T0);
        V4 normalVel30 = V4Mul(linVel1T0, normalT00);
        V4 normalVel40 = V4Mul(f0.rbXnI[0], angState1T0);
        V4 normalVel11 = V4Mul(linVel0T0, normalT01);
        V4 normalVel21 = V4Mul(f1.raXnI[0], angState0T0);
        V4 normalVel31 = V4Mul(linVel1T0, normalT01);
        V4 normalVel41 = V4Mul(f1.rbXnI[0], angState1T0);
        normalVel10 = V4MulAdd(linVel0T1, normalT10, normalVel10);
        normalVel20 = V4MulAdd(f0.raXnI[1], angState0T1, normalVel20);
        normalVel30 = V4MulAdd(linVel1T1, normalT10, normalVel30);
        normalVel40 = V4MulAdd(f0.rbXnI[1], angState1T1, normalVel40);
        normalVel11 = V4MulAdd(linVel0T1, normalT11, normalVel11);
        normalVel21 = V4MulAdd(f1.raXnI[1], angState0T1, normalVel21);
        normalVel31 = V4MulAdd(linVel1T1, normalT11, normalVel31);
        normalVel41 = V4MulAdd(f1.rbXnI[1], angState1T1, normalVel41);
        normalVel10 = V4MulAdd(linVel0T2, normalT20, normalVel10);
        normalVel20 = V4MulAdd(f0.raXnI[2], angState0T2, normalVel20);
        normalVel30 = V4MulAdd(linVel1T2, normalT20, normalVel30);
        normalVel40 = V4MulAdd(f0.rbXnI[2], angState1T2, normalVel40);
        normalVel11 = V4MulAdd(linVel0T2, normalT21, normalVel11);
        normalVel21 = V4MulAdd(f1.raXnI[2], angState0T2, normalVel21);
        normalVel31 = V4MulAdd(linVel1T2, normalT21, normalVel31);
        normalVel41 = V4MulAdd(f1.rbXnI[2], angState1T2, normalVel41);
        const V4 normalVel0_tmp1 = V4Add(normalVel10, normalVel20);
        const V4 normalVel0_tmp2 = V4Add(normalVel30, normalVel40);
        const V4 normalVel0 = V4Sub(normalVel0_tmp1, normalVel0_tmp2);
        const V4 normalVel1_tmp1 = V4Add(normalVel11, normalVel21);
        const V4 normalVel1_tmp2 = V4Add(normalVel31, normalVel41);
        const V4 normalVel1 = V4Sub(normalVel1_tmp1, normalVel1_tmp2);
        V4 deltaV0 = V4Mul(linDeltaX, normalT00);
        deltaV0 = V4MulAdd(linDeltaY, normalT10, deltaV0);
        deltaV0 = V4MulAdd(linDeltaZ, normalT20, deltaV0);
        V4 deltaV1 = V4Mul(linDeltaX, normalT01);
        deltaV1 = V4MulAdd(linDeltaY, normalT11, deltaV1);
        deltaV1 = V4MulAdd(linDeltaZ, normalT21, deltaV1);
        V4 angDelta00 = V4Mul(angDelta0T0, f0.raXnI[0]);
        V4 angDelta10 = V4Mul(angDelta1T0, f0.rbXnI[0]);
        angDelta00 = V4MulAdd(angDelta0T1, f0.raXnI[1], angDelta00);
        angDelta10 = V4MulAdd(angDelta1T1, f0.rbXnI[1], angDelta10);
        angDelta00 = V4MulAdd(angDelta0T2, f0.raXnI[2], angDelta00);
        angDelta10 = V4MulAdd(angDelta1T2, f0.rbXnI[2], angDelta10);
        V4 angDelta01 = V4Mul(angDelta0T0, f1.raXnI[0]);
        V4 angDelta11 = V4Mul(angDelta1T0, f1.rbXnI[0]);
        angDelta01 = V4MulAdd(angDelta0T1, f1.raXnI[1], angDelta01);
        angDelta11 = V4MulAdd(angDelta1T1, f1.rbXnI[1], angDelta11);
        angDelta01 = V4MulAdd(angDelta0T2, f1.raXnI[2], angDelta01);
        angDelta11 = V4MulAdd(angDelta1T2, f1.rbXnI[2], angDelta11);
        const V4 deltaAng0 = V4Sub(angDelta00, angDelta10);
        const V4 deltaAng1 = V4Sub(angDelta01, angDelta11);
        const V4 deltaBias0 = V4Sub(V4Add(deltaV0, deltaAng0), V4Mul(f0.targetVel, elapsedTime));
        const V4 deltaBias1 = V4Sub(V4Add(deltaV1, deltaAng1), V4Mul(f1.targetVel, elapsedTime));
        const V4 error0 = V4Add(f0.error, deltaBias0);
        const V4 error1 = V4Add(f1.error, deltaBias1);
        const V4 bias0 = V4Mul(error0, f0.biasCoefficient);
        const V4 bias1 = V4Mul(error1, f1.biasCoefficient);
        const V4 tmp10 = V4NegMulSub(V4Sub(bias0, f0.targetVel), f0.velMultiplier, appliedForce0);
        const V4 tmp11 = V4NegMulSub(V4Sub(bias1, f1.targetVel), f1.velMultiplier, appliedForce1);
        const V4 totalImpulse0 = V4NegMulSub(normalVel0, f0.velMultiplier, tmp10);
        const V4 totalImpulse1 = V4NegMulSub(normalVel1, f1.velMultiplier, tmp11);
        const V4 totalImpulse = V4Sqrt(V4MulAdd(totalImpulse0, totalImpulse0, V4Mul(totalImpulse1, totalImpulse1)));
        const BV clamped = V4IsGrtr(totalImpulse, maxFrictionImpulse);
        broken = BOr(broken, clamped);
        const V4 totalClamped = V4Sel(broken, V4Min(totalImpulse, maxDynFrictionImpulse), totalImpulse);
        const V4 ratio = V4Sel(V4IsGrtr(totalImpulse, vZero), V4Div(totalClamped, totalImpulse), vZero);
        const V4 newAppliedForce0 = V4Mul(totalImpulse0, ratio);
        const V4 newAppliedForce1 = V4Mul(totalImpulse1, ratio);
        const V4 deltaF0 = V4Sub(newAppliedForce0, appliedForce0);
        const V4 deltaF1 = V4Sub(newAppliedForce1, appliedForce1);
        frictionAppliedForce[i] = newAppliedForce0;
        frictionAppliedForce[i + 1] = newAppliedForce1;
        const V4 deltaFIM00 = V4Mul(deltaF0, invMassA);
        const V4 deltaFIM10 = V4Mul(deltaF0, invMassB);
        const V4 angDetaF00 = V4Mul(deltaF0, angD0);
        const V4 angDetaF10 = V4Mul(deltaF0, angD1);
        const V4 deltaFIM01 = V4Mul(deltaF1, invMassA);
        const V4 deltaFIM11 = V4Mul(deltaF1, invMassB);
        const V4 angDetaF01 = V4Mul(deltaF1, angD0);
        const V4 angDetaF11 = V4Mul(deltaF1, angD1);
        linVel0T0 = V4MulAdd(normalT00, deltaFIM00, V4MulAdd(normalT01, deltaFIM01, linVel0T0));
        linVel1T0 = V4NegMulSub(normalT00, deltaFIM10, V4NegMulSub(normalT01, deltaFIM11, linVel1T0));
        angState0T0 = V4MulAdd(f0.raXnI[0], angDetaF00, V4MulAdd(f1.raXnI[0], angDetaF01, angState0T0));
        angState1T0 = V4NegMulSub(f0.rbXnI[0], angDetaF10, V4NegMulSub(f1.rbXnI[0], angDetaF11, angState1T0));
        linVel0T1 = V4MulAdd(normalT10, deltaFIM00, V4MulAdd(normalT11, deltaFIM01, linVel0T1));
        linVel1T1 = V4NegMulSub(normalT10, deltaFIM10, V4NegMulSub(normalT11, deltaFIM11, linVel1T1));
        angState0T1 = V4MulAdd(f0.raXnI[1], angDetaF00, V4MulAdd(f1.raXnI[1], angDetaF01, angState0T1));
        angState1T1 = V4NegMulSub(f0.rbXnI[1], angDetaF10, V4NegMulSub(f1.rbXnI[1], angDetaF11, angState1T1));
        linVel0T2 = V4MulAdd(normalT20, deltaFIM00, V4MulAdd(normalT21, deltaFIM01, linVel0T2));
        linVel1T2 = V4NegMulSub(normalT20, deltaFIM10, V4NegMulSub(normalT21, deltaFIM11, linVel1T2));
        angState0T2 = V4MulAdd(f0.raXnI[2], angDetaF00, V4MulAdd(f1.raXnI[2], angDetaF01, angState0T2));
        angState1T2 = V4NegMulSub(f0.rbXnI[2], angDetaF10, V4NegMulSub(f1.rbXnI[2], angDetaF11, angState1T2));
      }
      hdr->broken = broken;
    }
  }
  // 되쓰기 (PX_TRANSPOSE_44 역전치 + V4StoreA; W 칸은 원래 값 그대로라 xyz 만 쓴다)
  const V4* L0[3] = {&linVel0T0, &linVel0T1, &linVel0T2};
  const V4* L1[3] = {&linVel1T0, &linVel1T1, &linVel1T2};
  const V4* A0[3] = {&angState0T0, &angState0T1, &angState0T2};
  const V4* A1[3] = {&angState1T0, &angState1T1, &angState1T2};
  for (int k = 0; k < 3; ++k) {
    b0[0]->lin[k] = L0[k]->f[0]; b0[1]->lin[k] = L0[k]->f[1]; b0[2]->lin[k] = L0[k]->f[2]; b0[3]->lin[k] = L0[k]->f[3];
    b0[0]->ang[k] = A0[k]->f[0]; b0[1]->ang[k] = A0[k]->f[1]; b0[2]->ang[k] = A0[k]->f[2]; b0[3]->ang[k] = A0[k]->f[3];
  }
  const float* lanes1L[4];
  (void)lanes1L;
  for (int a = 0; a < 4; ++a) {
    if (desc[a].bodyBDataIndex != 0) {
      for (int k = 0; k < 3; ++k) {
        const V4& l = *L1[k];
        const V4& g = *A1[k];
        b1[a]->lin[k] = l.f[a];
        b1[a]->ang[k] = g.f[a];
      }
    }
  }
}

// DyTGSContactPrepBlock.cpp:2775 writeBackContact4_Block — broken 표시만
SV_HDN void writeBackContact4_Block(const SDesc* desc, ByteArena& arena, FrictionArena& frictionCur) {
  uint8_t* currPtr = arenaPtr<uint8_t>(arena, desc[0].constraint);
  const uint8_t* last = currPtr + (uint32_t(desc[0].constraintLengthOver16) << 4);
  while (currPtr < last) {
    SolverContactHeaderStepBlock* hdr = reinterpret_cast<SolverContactHeaderStepBlock*>(currPtr);
    currPtr = reinterpret_cast<uint8_t*>(hdr + 1);
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    currPtr += sizeof(V4) * numNormalConstr;
    currPtr += numNormalConstr * sizeof(SolverContactPointStepBlock);
    const bool hasMaxImpulse = (hdr->flag & SolverContactHeaderStepBlock::eHAS_MAX_IMPULSE) != 0;
    if (hasMaxImpulse) currPtr += sizeof(V4) * numNormalConstr;
    currPtr += sizeof(V4) * numFrictionConstr;
    currPtr += numFrictionConstr * sizeof(SolverContactFrictionStepBlock);
    if (numFrictionConstr) {
      uint32_t broken[4];
      BStoreA(hdr->broken, broken);  // DyTGSContactPrepBlock.cpp 원본과 같음
      const uint8_t* frictionCounts = hdr->numNormalConstrs;  // 원본도 numNormalConstrs 를 본다 (DyTGSContactPrepBlock.cpp:2897)
      for (uint32_t a = 0; a < 4; ++a)
        if (frictionCounts[a] && broken[a] && hdr->frictionBrokenWriteback[a] != NONE) frictionCur.data[hdr->frictionBrokenWriteback[a]].broken = 1;
    }
  }
}

}  // namespace sv
}  // namespace eng
