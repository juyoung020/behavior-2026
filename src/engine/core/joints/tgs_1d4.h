// joints 모듈: TGS 1D 제약 4개 묶음(SIMD 4칸) 경로 — PhysX 5.6.1 과 바이트까지 같게 (CPU·CUDA 공용).
// 언제 쓰나: solver 의 분할(partition)에서 같은 분할에 강체-강체 1D 제약이 4개 이어서 모이면 머리 stride=4 로 묶는다
// (DyTGSDynamics.cpp:2204-2224). 4개 모두 행이 1개 이상이면 이 경로, 아니면 하나씩(tgs_1d.h). 공식 radio 장면은 정적-동적
// 고정 조인트 51개(meta 링크)라 깨어 있는 동안 이 경로를 탄다(tests/joints/ovd_joint_census).
// 원본(physx/source/lowleveldynamics/src 기준)
//   DyTGSContactPrepBlock.cpp:154 (SolverConstraint1DHeaderStep4), :186 (SolverConstraint1DStep4), :1635 (setOrthoData),
//   :1659 (셰이더 4번 + setupSolverConstraintStep4), :1722 (setupSolverConstraintStep4), :57 (QuatRotate4),
//   :2952 (V4Dot3 3성분), :2957 (solve1DStep4), :3418 (writeBack1D4), :3583 (conclude1DStep4)
//   shared/DyCpuGpu1dConstraint.h:718 (computeResolvedGeometricErrorTGSBlock), :797 (computeMinBiasTGSBlock)
// 블록은 0 으로 채우지 않는다(원본도 PxMemSet 없음) — 머리의 pad0[3] 만 안 쓰이는 바이트다.
#pragma once
#include <cstdint>

#include "../common/aos.h"
#include "d6_prep.h"
#include "joint_types.h"
#include "tgs_1d.h"

namespace eng {
namespace jnt {

struct alignas(16) Sc1DHeader4 {
  uint8_t type;
  uint8_t pad0[3];
  uint32_t count;       // 네 조인트 행 수의 최댓값
  uint8_t counts[4];
  uint8_t breakable[4];
  aos::Vec4V linBreakImpulse, angBreakImpulse, invMass0D0, invMass1D1, angD0, angD1;
  aos::Vec4V body0WorkOffset[3], rAWorld[3], rBWorld[3];
  aos::Vec4V angOrthoAxis0X[3], angOrthoAxis0Y[3], angOrthoAxis0Z[3];
  aos::Vec4V angOrthoAxis1X[3], angOrthoAxis1Y[3], angOrthoAxis1Z[3];
  aos::Vec4V angOrthoRecipResponse[3], angOrthoError[3];
};
struct alignas(16) Sc1DRow4 {
  aos::Vec4V lin0[3]; aos::Vec4V error;
  aos::Vec4V lin1[3]; aos::Vec4V biasScale;
  aos::Vec4V ang0[3]; aos::Vec4V velMultiplier;
  aos::Vec4V ang1[3];
  aos::Vec4V velTarget;
  aos::Vec4V minImpulse, maxImpulse, appliedForce;
  aos::Vec4V maxBias, angularErrorScale;
  uint32_t flags[4];
};
struct alignas(16) Sc1DRow4R : Sc1DRow4 { aos::Vec4V residualVelIter, residualPosIter; };  // 잔차 보고 켰을 때

EHD uint32_t blockLength4(uint32_t maxRows, bool residual) {
  return uint32_t(sizeof(Sc1DHeader4)) + (residual ? uint32_t(sizeof(Sc1DRow4R)) : uint32_t(sizeof(Sc1DRow4))) * maxRows;
}

// setOrthoData (DyTGSContactPrepBlock.cpp:1635)
EHD void setOrthoData(float ang0X, float ang0Y, float ang0Z, float ang1X, float ang1Y, float ang1Z, float recipResponse, float error,
                      float& oa0x, float& oa0y, float& oa0z, float& oa1x, float& oa1y, float& oa1z, float& orr, float& oerr,
                      bool disableProcessing, uint32_t solveHint, uint32_t& flags, uint32_t& orthoCount, bool finished) {
  if (!finished && !disableProcessing) {
    if (solveHint == SH_ROTATIONAL_EQUALITY) {
      flags |= SC_ROT_EQ;
      oa0x = ang0X; oa0y = ang0Y; oa0z = ang0Z;
      oa1x = ang1X; oa1y = ang1Y; oa1z = ang1Z;
      orr = recipResponse;
      oerr = error;
      orthoCount++;
    } else if (solveHint & SH_EQUALITY) {
      flags |= SC_ORTHO_TARGET;
    }
  }
}

EHD float& lane(aos::Vec4V& v, int i) { return reinterpret_cast<float*>(&v)[i]; }
EHD float laneC(const aos::Vec4V& v, int i) { return reinterpret_cast<const float*>(&v)[i]; }

// ---- setupSolverConstraintStep4 (DyTGSContactPrepBlock.cpp:1722). p[a].rows 는 셰이더가 채운 행(각 numRows>0).
// blk 는 blockLength4(maxRows)+16 바이트 이상. 반환 = 머리 길이(16 배수, 네 desc 공통).
EHD uint32_t setupSolverConstraintStep4(PrepIn* p, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt, uint32_t maxRows,
                                        float lengthScale, float biasCoefficient, bool isResidualReportingEnabled) {
  using namespace aos;
  const Vec4V zero = V4Zero();
  Row* allSorted[MAX_CONSTRAINT_ROWS * 4];
  uint32_t startIndex[4];
  V4a angSqrtInvInertia0[MAX_CONSTRAINT_ROWS * 4];
  V4a angSqrtInvInertia1[MAX_CONSTRAINT_ROWS * 4];
  uint32_t numRows = 0;
  for (uint32_t a = 0; a < 4; ++a) {
    startIndex[a] = numRows;
    PrepIn& d = p[a];
    Row** sorted = allSorted + numRows;
    for (uint32_t i = 0; i < d.numRows; ++i) {
      if (d.rows[i].flags & RF_ANGULAR_CONSTRAINT) {
        if (d.rows[i].solveHint == SH_EQUALITY) d.rows[i].solveHint = SH_ROTATIONAL_EQUALITY;
        else if (d.rows[i].solveHint == SH_INEQUALITY) d.rows[i].solveHint = SH_ROTATIONAL_INEQUALITY;
      }
    }
    preprocessRows(sorted, d.rows, angSqrtInvInertia0 + numRows, angSqrtInvInertia1 + numRows, d.numRows, d.txI0->sqrtInvInertia,
                   d.txI1->sqrtInvInertia, d.data0->invMass, d.data1->invMass, d.invMassScales, d.disablePreprocessing, d.improvedSlerp);
    numRows += d.numRows;
  }
  const uint32_t stride = isResidualReportingEnabled ? uint32_t(sizeof(Sc1DRow4R)) : uint32_t(sizeof(Sc1DRow4));
  const uint32_t constraintLength = uint32_t(sizeof(Sc1DHeader4)) + stride * maxRows;
  const float erp = 0.5f * biasCoefficient;
  const bool kin[4][2] = {{p[0].body0->isKinematic != 0, p[0].body1->isKinematic != 0}, {p[1].body0->isKinematic != 0, p[1].body1->isKinematic != 0},
                          {p[2].body0->isKinematic != 0, p[2].body1->isKinematic != 0}, {p[3].body0->isKinematic != 0, p[3].body1->isKinematic != 0}};

  uint8_t* currPtr = blk;
  Sc1DHeader4* header = reinterpret_cast<Sc1DHeader4*>(currPtr);
  currPtr += sizeof(Sc1DHeader4);
  header->pad0[0] = header->pad0[1] = header->pad0[2] = 0;  // 원본은 안 씀(쓰레기) — 비교에서 뺀다

  const Vec4V invMassScale0 = V4LoadXYZW(p[0].invMassScales.linear0, p[1].invMassScales.linear0, p[2].invMassScales.linear0, p[3].invMassScales.linear0);
  const Vec4V invMassScale1 = V4LoadXYZW(p[0].invMassScales.linear1, p[1].invMassScales.linear1, p[2].invMassScales.linear1, p[3].invMassScales.linear1);
  const Vec4V iMass0 = V4LoadXYZW(p[0].data0->invMass, p[1].data0->invMass, p[2].data0->invMass, p[3].data0->invMass);
  const Vec4V iMass1 = V4LoadXYZW(p[0].data1->invMass, p[1].data1->invMass, p[2].data1->invMass, p[3].data1->invMass);
  const Vec4V invMass0 = V4Mul(iMass0, invMassScale0);
  const Vec4V invMass1 = V4Mul(iMass1, invMassScale1);
  const Vec4V invInertiaScale0 = V4LoadXYZW(p[0].invMassScales.angular0, p[1].invMassScales.angular0, p[2].invMassScales.angular0, p[3].invMassScales.angular0);
  const Vec4V invInertiaScale1 = V4LoadXYZW(p[0].invMassScales.angular1, p[1].invMassScales.angular1, p[2].invMassScales.angular1, p[3].invMassScales.angular1);

  auto ld3 = [](const V3& v) { const float f[4] = {v.x, v.y, v.z, 0.0f}; return V4LoadU(f); };  // w 는 전치(44_34)에서 안 쓰인다
  Vec4V workOffset0 = ld3(p[0].body0WorldOffset), workOffset1 = ld3(p[1].body0WorldOffset);
  Vec4V workOffset2 = ld3(p[2].body0WorldOffset), workOffset3 = ld3(p[3].body0WorldOffset);
  Vec4V workOffsetX, workOffsetY, workOffsetZ;
  PX_TRANSPOSE_44_34(workOffset0, workOffset1, workOffset2, workOffset3, workOffsetX, workOffsetY, workOffsetZ);

  const FloatV dtV = FLoad(simDt);
  const Vec4V linBreakForce = V4LoadXYZW(p[0].linBreakForce, p[1].linBreakForce, p[2].linBreakForce, p[3].linBreakForce);
  const Vec4V angBreakForce = V4LoadXYZW(p[0].angBreakForce, p[1].angBreakForce, p[2].angBreakForce, p[3].angBreakForce);
  for (int a = 0; a < 4; ++a) header->breakable[a] = uint8_t((p[a].linBreakForce != MAX_F32) || (p[a].angBreakForce != MAX_F32));
  header->invMass0D0 = invMass0;
  header->invMass1D1 = invMass1;
  header->angD0 = invInertiaScale0;
  header->angD1 = invInertiaScale1;
  header->body0WorkOffset[0] = workOffsetX;
  header->body0WorkOffset[1] = workOffsetY;
  header->body0WorkOffset[2] = workOffsetZ;
  header->count = maxRows;
  header->type = SC_TYPE_BLOCK_1D;
  header->linBreakImpulse = V4Scale(linBreakForce, dtV);
  header->angBreakImpulse = V4Scale(angBreakForce, dtV);
  for (int a = 0; a < 4; ++a) header->counts[a] = uint8_t(p[a].numRows);

  Vec4V ca2WX, ca2WY, ca2WZ, cb2WX, cb2WY, cb2WZ;
  Vec4V ca2W0 = ld3(p[0].cA2w), ca2W1 = ld3(p[1].cA2w), ca2W2 = ld3(p[2].cA2w), ca2W3 = ld3(p[3].cA2w);
  Vec4V cb2W0 = ld3(p[0].cB2w), cb2W1 = ld3(p[1].cB2w), cb2W2 = ld3(p[2].cB2w), cb2W3 = ld3(p[3].cB2w);
  PX_TRANSPOSE_44_34(ca2W0, ca2W1, ca2W2, ca2W3, ca2WX, ca2WY, ca2WZ);
  PX_TRANSPOSE_44_34(cb2W0, cb2W1, cb2W2, cb2W3, cb2WX, cb2WY, cb2WZ);
  Vec4V pos00 = ld3(p[0].txI0->body2WorldP), pos01 = ld3(p[0].txI1->body2WorldP);
  Vec4V pos10 = ld3(p[1].txI0->body2WorldP), pos11 = ld3(p[1].txI1->body2WorldP);
  Vec4V pos20 = ld3(p[2].txI0->body2WorldP), pos21 = ld3(p[2].txI1->body2WorldP);
  Vec4V pos30 = ld3(p[3].txI0->body2WorldP), pos31 = ld3(p[3].txI1->body2WorldP);
  Vec4V pos0X, pos0Y, pos0Z, pos1X, pos1Y, pos1Z;
  PX_TRANSPOSE_44_34(pos00, pos10, pos20, pos30, pos0X, pos0Y, pos0Z);
  PX_TRANSPOSE_44_34(pos01, pos11, pos21, pos31, pos1X, pos1Y, pos1Z);
  Vec4V linVel00 = ld3(p[0].data0->originalLinearVelocity), linVel01 = ld3(p[0].data1->originalLinearVelocity);
  Vec4V angState00 = ld3(p[0].data0->originalAngularVelocity), angState01 = ld3(p[0].data1->originalAngularVelocity);
  Vec4V linVel10 = ld3(p[1].data0->originalLinearVelocity), linVel11 = ld3(p[1].data1->originalLinearVelocity);
  Vec4V angState10 = ld3(p[1].data0->originalAngularVelocity), angState11 = ld3(p[1].data1->originalAngularVelocity);
  Vec4V linVel20 = ld3(p[2].data0->originalLinearVelocity), linVel21 = ld3(p[2].data1->originalLinearVelocity);
  Vec4V angState20 = ld3(p[2].data0->originalAngularVelocity), angState21 = ld3(p[2].data1->originalAngularVelocity);
  Vec4V linVel30 = ld3(p[3].data0->originalLinearVelocity), linVel31 = ld3(p[3].data1->originalLinearVelocity);
  Vec4V angState30 = ld3(p[3].data0->originalAngularVelocity), angState31 = ld3(p[3].data1->originalAngularVelocity);
  Vec4V linVel0T0, linVel0T1, linVel0T2, linVel1T0, linVel1T1, linVel1T2;
  Vec4V angState0T0, angState0T1, angState0T2, angState1T0, angState1T1, angState1T2;
  PX_TRANSPOSE_44_34(linVel00, linVel10, linVel20, linVel30, linVel0T0, linVel0T1, linVel0T2);
  PX_TRANSPOSE_44_34(linVel01, linVel11, linVel21, linVel31, linVel1T0, linVel1T1, linVel1T2);
  PX_TRANSPOSE_44_34(angState00, angState10, angState20, angState30, angState0T0, angState0T1, angState0T2);
  PX_TRANSPOSE_44_34(angState01, angState11, angState21, angState31, angState1T0, angState1T1, angState1T2);
  header->rAWorld[0] = V4Sub(ca2WX, pos0X);
  header->rAWorld[1] = V4Sub(ca2WY, pos0Y);
  header->rAWorld[2] = V4Sub(ca2WZ, pos0Z);
  header->rBWorld[0] = V4Sub(cb2WX, pos1X);
  header->rBWorld[1] = V4Sub(cb2WY, pos1Y);
  header->rBWorld[2] = V4Sub(cb2WZ, pos1Z);

  uint32_t index[4], endIndex[4];
  for (int a = 0; a < 4; ++a) {
    index[a] = startIndex[a];
    endIndex[a] = startIndex[a] + p[a].numRows - 1;
  }
  const Vec4V one = V4One();
  uint32_t orthoCount[4] = {0, 0, 0, 0};
  for (uint32_t a = 0; a < 3; ++a) {
    header->angOrthoAxis0X[a] = V4Zero(); header->angOrthoAxis0Y[a] = V4Zero(); header->angOrthoAxis0Z[a] = V4Zero();
    header->angOrthoAxis1X[a] = V4Zero(); header->angOrthoAxis1Y[a] = V4Zero(); header->angOrthoAxis1Z[a] = V4Zero();
    header->angOrthoRecipResponse[a] = V4Zero();
    header->angOrthoError[a] = V4Zero();
  }

  for (uint32_t a = 0; a < maxRows; ++a) {
    const bool finished[4] = {a >= p[0].numRows, a >= p[1].numRows, a >= p[2].numRows, a >= p[3].numRows};
    const BoolV bFinished = BLoad(finished);
    Sc1DRow4* c = reinterpret_cast<Sc1DRow4*>(currPtr);
    currPtr += stride;
    Row* con[4] = {allSorted[index[0]], allSorted[index[1]], allSorted[index[2]], allSorted[index[3]]};
    const bool angularConstraint[4] = {(con[0]->flags & RF_ANGULAR_CONSTRAINT) != 0, (con[1]->flags & RF_ANGULAR_CONSTRAINT) != 0,
                                       (con[2]->flags & RF_ANGULAR_CONSTRAINT) != 0, (con[3]->flags & RF_ANGULAR_CONSTRAINT) != 0};
    const BoolV bAngularConstraint = BLoad(angularConstraint);
    Vec4V cangDelta00 = V4LoadA(&angSqrtInvInertia0[index[0]].x), cangDelta01 = V4LoadA(&angSqrtInvInertia0[index[1]].x);
    Vec4V cangDelta02 = V4LoadA(&angSqrtInvInertia0[index[2]].x), cangDelta03 = V4LoadA(&angSqrtInvInertia0[index[3]].x);
    Vec4V cangDelta10 = V4LoadA(&angSqrtInvInertia1[index[0]].x), cangDelta11 = V4LoadA(&angSqrtInvInertia1[index[1]].x);
    Vec4V cangDelta12 = V4LoadA(&angSqrtInvInertia1[index[2]].x), cangDelta13 = V4LoadA(&angSqrtInvInertia1[index[3]].x);
    for (int k = 0; k < 4; ++k) index[k] = index[k] == endIndex[k] ? index[k] : index[k] + 1;

    float minI[4], maxI[4];
    for (int k = 0; k < 4; ++k) {  // computeMinMaxImpulseOrForceAsImpulse
      const float driveScale = ((con[k]->flags & RF_HAS_DRIVE_LIMIT) && p[k].driveLimitsAreForces) ? simDt : 1.0f;
      minI[k] = con[k]->minImpulse * driveScale;
      maxI[k] = con[k]->maxImpulse * driveScale;
    }
    const Vec4V minImpulse = V4LoadXYZW(minI[0], minI[1], minI[2], minI[3]);
    const Vec4V maxImpulse = V4LoadXYZW(maxI[0], maxI[1], maxI[2], maxI[3]);

    Vec4V clin00 = V4LoadA(&con[0]->linear0.x), clin01 = V4LoadA(&con[1]->linear0.x), clin02 = V4LoadA(&con[2]->linear0.x), clin03 = V4LoadA(&con[3]->linear0.x);
    Vec4V clin0X, clin0Y, clin0Z;
    PX_TRANSPOSE_44_34(clin00, clin01, clin02, clin03, clin0X, clin0Y, clin0Z);
    Vec4V cang00 = V4LoadA(&con[0]->angular0.x), cang01 = V4LoadA(&con[1]->angular0.x), cang02 = V4LoadA(&con[2]->angular0.x), cang03 = V4LoadA(&con[3]->angular0.x);
    Vec4V cang0X, cang0Y, cang0Z;
    PX_TRANSPOSE_44_34(cang00, cang01, cang02, cang03, cang0X, cang0Y, cang0Z);
    Vec4V cang10 = V4LoadA(&con[0]->angular1.x), cang11 = V4LoadA(&con[1]->angular1.x), cang12 = V4LoadA(&con[2]->angular1.x), cang13 = V4LoadA(&con[3]->angular1.x);
    Vec4V cang1X, cang1Y, cang1Z;
    PX_TRANSPOSE_44_34(cang10, cang11, cang12, cang13, cang1X, cang1Y, cang1Z);
    Vec4V angDelta0X, angDelta0Y, angDelta0Z;
    PX_TRANSPOSE_44_34(cangDelta00, cangDelta01, cangDelta02, cangDelta03, angDelta0X, angDelta0Y, angDelta0Z);

    c->flags[0] = c->flags[1] = c->flags[2] = c->flags[3] = 0;
    c->lin0[0] = V4Sel(bFinished, zero, clin0X);
    c->lin0[1] = V4Sel(bFinished, zero, clin0Y);
    c->lin0[2] = V4Sel(bFinished, zero, clin0Z);
    c->ang0[0] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang0X, zero);
    c->ang0[1] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang0Y, zero);
    c->ang0[2] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang0Z, zero);
    c->angularErrorScale = V4Sel(bAngularConstraint, one, zero);
    c->minImpulse = minImpulse;
    c->maxImpulse = maxImpulse;
    c->appliedForce = zero;
    if (isResidualReportingEnabled) {
      Sc1DRow4R* cc = static_cast<Sc1DRow4R*>(c);
      cc->residualPosIter = zero;
      cc->residualVelIter = zero;
    }
    const Vec4V lin0MagSq = V4MulAdd(clin0Z, clin0Z, V4MulAdd(clin0Y, clin0Y, V4Mul(clin0X, clin0X)));
    const Vec4V cang0DotAngDelta = V4MulAdd(angDelta0Z, angDelta0Z, V4MulAdd(angDelta0Y, angDelta0Y, V4Mul(angDelta0X, angDelta0X)));
    Vec4V unitResponse = V4MulAdd(lin0MagSq, invMass0, V4Mul(cang0DotAngDelta, invInertiaScale0));
    Vec4V clin10 = V4LoadA(&con[0]->linear1.x), clin11 = V4LoadA(&con[1]->linear1.x), clin12 = V4LoadA(&con[2]->linear1.x), clin13 = V4LoadA(&con[3]->linear1.x);
    Vec4V clin1X, clin1Y, clin1Z;
    PX_TRANSPOSE_44_34(clin10, clin11, clin12, clin13, clin1X, clin1Y, clin1Z);
    Vec4V angDelta1X, angDelta1Y, angDelta1Z;
    PX_TRANSPOSE_44_34(cangDelta10, cangDelta11, cangDelta12, cangDelta13, angDelta1X, angDelta1Y, angDelta1Z);
    const Vec4V lin1MagSq = V4MulAdd(clin1Z, clin1Z, V4MulAdd(clin1Y, clin1Y, V4Mul(clin1X, clin1X)));
    const Vec4V cang1DotAngDelta = V4MulAdd(angDelta1Z, angDelta1Z, V4MulAdd(angDelta1Y, angDelta1Y, V4Mul(angDelta1X, angDelta1X)));
    c->lin1[0] = V4Sel(bFinished, zero, clin1X);
    c->lin1[1] = V4Sel(bFinished, zero, clin1Y);
    c->lin1[2] = V4Sel(bFinished, zero, clin1Z);
    c->ang1[0] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang1X, zero);
    c->ang1[1] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang1Y, zero);
    c->ang1[2] = V4Sel(BAndNot(bAngularConstraint, bFinished), cang1Z, zero);
    unitResponse = V4Add(unitResponse, V4MulAdd(lin1MagSq, invMass1, V4Mul(cang1DotAngDelta, invInertiaScale1)));
    const Vec4V lnormalVel0 = V4MulAdd(clin0X, linVel0T0, V4MulAdd(clin0Y, linVel0T1, V4Mul(clin0Z, linVel0T2)));
    const Vec4V lnormalVel1 = V4MulAdd(clin1X, linVel1T0, V4MulAdd(clin1Y, linVel1T1, V4Mul(clin1Z, linVel1T2)));
    const Vec4V angVel0 = V4MulAdd(cang0X, angState0T0, V4MulAdd(cang0Y, angState0T1, V4Mul(cang0Z, angState0T2)));
    const Vec4V angVel1 = V4MulAdd(angDelta1X, angState1T0, V4MulAdd(angDelta1Y, angState1T1, V4Mul(angDelta1Z, angState1T2)));
    const Vec4V normalVel0 = V4Add(lnormalVel0, angVel0);
    const Vec4V normalVel1 = V4Add(lnormalVel1, angVel1);
    const Vec4V normalVel = V4Sub(normalVel0, normalVel1);
    angDelta0X = V4Mul(angDelta0X, invInertiaScale0);
    angDelta0Y = V4Mul(angDelta0Y, invInertiaScale0);
    angDelta0Z = V4Mul(angDelta0Z, invInertiaScale0);
    angDelta1X = V4Mul(angDelta1X, invInertiaScale1);
    angDelta1Y = V4Mul(angDelta1Y, invInertiaScale1);
    angDelta1Z = V4Mul(angDelta1Z, invInertiaScale1);
    {
      float recipResponses[4], originalError[4];
      for (int i = 0; i < 4; i++) {
        if (a < p[i].numRows) {
          const Row& ci = *con[i];
          const float geometricErrorI = ci.geometricError;
          originalError[i] = geometricErrorI;
          const float jointSpeedI = laneC(normalVel, i);
          const float unitResponseI = laneC(unitResponse, i);
          const float recipUnitResponseI = computeRecipUnitResponse(unitResponseI, p[i].minResponseThreshold);
          recipResponses[i] = recipUnitResponseI;
          const float maxBiasVelocityI =
              computeMaxBiasVelocityTGS(ci.flags, jointSpeedI, ci.mod1, ci.mod0, geometricErrorI, false, lengthScale, recipSimDt);
          float initJointSpeedI = 0.f;
          if (kin[i][0]) initJointSpeedI -= laneC(normalVel0, i);
          if (kin[i][1]) initJointSpeedI += laneC(normalVel1, i);
          const Consts k = compute1dConstantsTGS(ci.flags, ci.mod0, ci.mod1, ci.mod0, ci.mod1, geometricErrorI, ci.velocityTarget, jointSpeedI,
                                                 initJointSpeedI, unitResponseI, recipUnitResponseI, erp, stepDt, recipStepDt);
          lane(c->biasScale, i) = k.biasScale;
          lane(c->error, i) = k.error;
          lane(c->velMultiplier, i) = k.velMultiplier;
          lane(c->velTarget, i) = k.targetVel;
          lane(c->maxBias, i) = maxBiasVelocityI;
        } else {
          lane(c->biasScale, i) = 0.0f;
          lane(c->error, i) = 0.0f;
          lane(c->velMultiplier, i) = 0.0f;
          lane(c->velTarget, i) = 0.0f;
          lane(c->maxBias, i) = 0.0f;
        }
        raiseInternalFlagsTGS(con[i]->flags, con[i]->solveHint, c->flags[i]);
      }
      for (int i = 0; i < 4; ++i) {
        const uint32_t oc = orthoCount[i];
        setOrthoData(laneC(angDelta0X, i), laneC(angDelta0Y, i), laneC(angDelta0Z, i), laneC(angDelta1X, i), laneC(angDelta1Y, i),
                     laneC(angDelta1Z, i), recipResponses[i], originalError[i], lane(header->angOrthoAxis0X[oc], i),
                     lane(header->angOrthoAxis0Y[oc], i), lane(header->angOrthoAxis0Z[oc], i), lane(header->angOrthoAxis1X[oc], i),
                     lane(header->angOrthoAxis1Y[oc], i), lane(header->angOrthoAxis1Z[oc], i), lane(header->angOrthoRecipResponse[oc], i),
                     lane(header->angOrthoError[oc], i), p[i].disablePreprocessing, con[i]->solveHint, c->flags[i], orthoCount[i],
                     a >= p[i].numRows);
      }
    }
  }
  reinterpret_cast<uint32_t*>(currPtr)[0] = 0;  // 원본도 끝에 8 바이트를 0 으로 쓴다 (+16 여유 안)
  reinterpret_cast<uint32_t*>(currPtr)[1] = 0;
  return constraintLength;
}

// 셰이더 4번 + 준비 (DyTGSContactPrepBlock.cpp:1659). 네 조인트 중 행이 0 인 것이 있으면 0 을 돌려준다(= 하나씩 경로로).
EHD uint32_t prepareD6Step4(const D6Data* const data[4], const uint16_t flags[4], const float linBreak[4], const float angBreak[4],
                            const float minResp[4], const Tf* frame0[4], const Tf* frame1[4], const TgsBodyVel* b0[4], const TgsBodyVel* b1[4],
                            const TgsTxInertia* t0[4], const TgsTxInertia* t1[4], const TgsBodyData* d0[4], const TgsBodyData* d1[4],
                            Row* rows /* MAX_CONSTRAINT_ROWS*4 */, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt,
                            float lengthScale, float biasCoefficient, bool residual) {
  PrepIn p[4];
  uint32_t maxRows = 0;
  Row* r = rows;
  for (int a = 0; a < 4; ++a) {
    PrepIn& d = p[a];
    d.disablePreprocessing = (flags[a] & CF_DISABLE_PREPROCESSING) != 0;
    d.improvedSlerp = (flags[a] & CF_IMPROVED_SLERP) != 0;
    d.driveLimitsAreForces = (flags[a] & CF_DRIVE_LIMITS_ARE_FORCES) != 0;
    d.extendedLimits = (flags[a] & CF_ENABLE_EXTENDED_LIMITS) != 0;
    d.disableConstraint = (flags[a] & CF_DISABLE_CONSTRAINT) != 0;
    setupConstraintRows(r, MAX_CONSTRAINT_ROWS);
    uint32_t n = 0;
    if (!d.disableConstraint) {
      const PrepOut o = d6SolverPrep(r, *data[a], *frame0[a], *frame1[a], d.extendedLimits);
      n = o.numRows;
      d.invMassScales = o.invMassScale;
      d.body0WorldOffset = o.body0WorldOffset;
      d.cA2w = o.cA2w;
      d.cB2w = o.cB2w;
    }
    if (n == 0) return 0;
    maxRows = n > maxRows ? n : maxRows;
    d.rows = r;
    d.numRows = n;
    r += n;
    d.bodyFrame0 = *frame0[a];
    d.bodyFrame1 = *frame1[a];
    d.body0 = b0[a]; d.body1 = b1[a]; d.txI0 = t0[a]; d.txI1 = t1[a]; d.data0 = d0[a]; d.data1 = d1[a];
    d.linkIndexA = RIGID_BODY; d.linkIndexB = RIGID_BODY;
    d.linBreakForce = linBreak[a]; d.angBreakForce = angBreak[a]; d.minResponseThreshold = minResp[a];
    if (b0[a]->isKinematic) d.invMassScales.angular0 = 0.0f;
    if (b1[a]->isKinematic) d.invMassScales.angular1 = 0.0f;
  }
  return setupSolverConstraintStep4(p, blk, stepDt, simDt, recipStepDt, recipSimDt, maxRows, lengthScale, biasCoefficient, residual);
}

// QuatRotate4 (DyTGSContactPrepBlock.cpp:57)
EHD void QuatRotate4(const aos::Vec4V& qx, const aos::Vec4V& qy, const aos::Vec4V& qz, const aos::Vec4V& qw, const aos::Vec4V& vx,
                     const aos::Vec4V& vy, const aos::Vec4V& vz, aos::Vec4V& rX, aos::Vec4V& rY, aos::Vec4V& rZ) {
  using namespace aos;
  const Vec4V two = V4Splat(FLoad(2.f));
  const Vec4V nhalf = V4Splat(FLoad(-0.5f));
  const Vec4V w2 = V4MulAdd(qw, qw, nhalf);
  const Vec4V ax = V4Mul(vx, w2), ay = V4Mul(vy, w2), az = V4Mul(vz, w2);
  const Vec4V crX = V4NegMulSub(qz, vy, V4Mul(qy, vz));
  const Vec4V crY = V4NegMulSub(qx, vz, V4Mul(qz, vx));
  const Vec4V crZ = V4NegMulSub(qy, vx, V4Mul(qx, vy));
  const Vec4V tempX = V4MulAdd(crX, qw, ax), tempY = V4MulAdd(crY, qw, ay), tempZ = V4MulAdd(crZ, qw, az);
  Vec4V dotuv = V4Mul(qx, vx);
  dotuv = V4MulAdd(qy, vy, dotuv);
  dotuv = V4MulAdd(qz, vz, dotuv);
  rX = V4Mul(V4MulAdd(qx, dotuv, tempX), two);
  rY = V4Mul(V4MulAdd(qy, dotuv, tempY), two);
  rZ = V4Mul(V4MulAdd(qz, dotuv, tempZ), two);
}
EHD aos::Vec4V V4Dot3x(const aos::Vec4V& x0, const aos::Vec4V& y0, const aos::Vec4V& z0, const aos::Vec4V& x1, const aos::Vec4V& y1,
                       const aos::Vec4V& z1) {
  using namespace aos;
  return V4MulAdd(x0, x1, V4MulAdd(y0, y1, V4Mul(z0, z1)));
}

// ---- solve1DStep4 (DyTGSContactPrepBlock.cpp:2957). b[k][0/1] = k 번째 조인트의 두 몸체 (같은 몸체를 여러 칸이 가리킬 수 있음: 세계)
EHD void solve1DStep4(uint8_t* blk, TgsBodyVel* const b[4][2], const TgsTxInertia* const t[4][2], float elapsedTimeF32, bool residual,
                      bool isPositionIteration) {
  using namespace aos;
  if (blk == nullptr) return;
  const FloatV elapsedTime = FLoad(elapsedTimeF32);
  auto ldv = [](const TgsBodyVel* bv, int which) { return V4LoadA(which == 0 ? &bv->linearVelocity.x : &bv->angularVelocity.x); };
  Vec4V linVel00 = ldv(b[0][0], 0), linVel01 = ldv(b[0][1], 0), angState00 = ldv(b[0][0], 1), angState01 = ldv(b[0][1], 1);
  Vec4V linVel10 = ldv(b[1][0], 0), linVel11 = ldv(b[1][1], 0), angState10 = ldv(b[1][0], 1), angState11 = ldv(b[1][1], 1);
  Vec4V linVel20 = ldv(b[2][0], 0), linVel21 = ldv(b[2][1], 0), angState20 = ldv(b[2][0], 1), angState21 = ldv(b[2][1], 1);
  Vec4V linVel30 = ldv(b[3][0], 0), linVel31 = ldv(b[3][1], 0), angState30 = ldv(b[3][0], 1), angState31 = ldv(b[3][1], 1);
  Vec4V linVel0T0, linVel0T1, linVel0T2, linVel0T3, linVel1T0, linVel1T1, linVel1T2, linVel1T3;
  Vec4V angState0T0, angState0T1, angState0T2, angState0T3, angState1T0, angState1T1, angState1T2, angState1T3;
  PX_TRANSPOSE_44(linVel00, linVel10, linVel20, linVel30, linVel0T0, linVel0T1, linVel0T2, linVel0T3);
  PX_TRANSPOSE_44(linVel01, linVel11, linVel21, linVel31, linVel1T0, linVel1T1, linVel1T2, linVel1T3);
  PX_TRANSPOSE_44(angState00, angState10, angState20, angState30, angState0T0, angState0T1, angState0T2, angState0T3);
  PX_TRANSPOSE_44(angState01, angState11, angState21, angState31, angState1T0, angState1T1, angState1T2, angState1T3);
  auto ldd = [](const TgsBodyVel* bv, int which) { return V4LoadA(which == 0 ? &bv->deltaLinDt.x : &bv->deltaAngDt.x); };
  Vec4V linDelta00 = ldd(b[0][0], 0), linDelta01 = ldd(b[0][1], 0), angDelta00 = ldd(b[0][0], 1), angDelta01 = ldd(b[0][1], 1);
  Vec4V linDelta10 = ldd(b[1][0], 0), linDelta11 = ldd(b[1][1], 0), angDelta10 = ldd(b[1][0], 1), angDelta11 = ldd(b[1][1], 1);
  Vec4V linDelta20 = ldd(b[2][0], 0), linDelta21 = ldd(b[2][1], 0), angDelta20 = ldd(b[2][0], 1), angDelta21 = ldd(b[2][1], 1);
  Vec4V linDelta30 = ldd(b[3][0], 0), linDelta31 = ldd(b[3][1], 0), angDelta30 = ldd(b[3][0], 1), angDelta31 = ldd(b[3][1], 1);
  Vec4V linDelta0T0, linDelta0T1, linDelta0T2, linDelta1T0, linDelta1T1, linDelta1T2;
  Vec4V angDelta0T0, angDelta0T1, angDelta0T2, angDelta1T0, angDelta1T1, angDelta1T2;
  PX_TRANSPOSE_44_34(linDelta00, linDelta10, linDelta20, linDelta30, linDelta0T0, linDelta0T1, linDelta0T2);
  PX_TRANSPOSE_44_34(linDelta01, linDelta11, linDelta21, linDelta31, linDelta1T0, linDelta1T1, linDelta1T2);
  PX_TRANSPOSE_44_34(angDelta00, angDelta10, angDelta20, angDelta30, angDelta0T0, angDelta0T1, angDelta0T2);
  PX_TRANSPOSE_44_34(angDelta01, angDelta11, angDelta21, angDelta31, angDelta1T0, angDelta1T1, angDelta1T2);

  const Sc1DHeader4* header = reinterpret_cast<const Sc1DHeader4*>(blk);
  uint8_t* base = blk + sizeof(Sc1DHeader4);
  auto ldI = [](const TgsTxInertia* tx, int col) -> Vec4V {  // V4LoadU(&column0.x) — 뒤 열의 x 가 w 칸 (전치에서 안 쓰임)
    const float* f = &tx->sqrtInvInertia.c0.x + 3 * col;
    if (col < 2) return V4LoadU(f);
    return Vec4V_From_Vec3V(V3LoadU(f));
  };
  Vec4V invInertia00X = ldI(t[0][0], 0), invInertia00Y = ldI(t[0][0], 1), invInertia00Z = ldI(t[0][0], 2);
  Vec4V invInertia10X = ldI(t[1][0], 0), invInertia10Y = ldI(t[1][0], 1), invInertia10Z = ldI(t[1][0], 2);
  Vec4V invInertia20X = ldI(t[2][0], 0), invInertia20Y = ldI(t[2][0], 1), invInertia20Z = ldI(t[2][0], 2);
  Vec4V invInertia30X = ldI(t[3][0], 0), invInertia30Y = ldI(t[3][0], 1), invInertia30Z = ldI(t[3][0], 2);
  Vec4V invInertia01X = ldI(t[0][1], 0), invInertia01Y = ldI(t[0][1], 1), invInertia01Z = ldI(t[0][1], 2);
  Vec4V invInertia11X = ldI(t[1][1], 0), invInertia11Y = ldI(t[1][1], 1), invInertia11Z = ldI(t[1][1], 2);
  Vec4V invInertia21X = ldI(t[2][1], 0), invInertia21Y = ldI(t[2][1], 1), invInertia21Z = ldI(t[2][1], 2);
  Vec4V invInertia31X = ldI(t[3][1], 0), invInertia31Y = ldI(t[3][1], 1), invInertia31Z = ldI(t[3][1], 2);
  Vec4V invInertia0X0, invInertia0X1, invInertia0X2, invInertia0Y0, invInertia0Y1, invInertia0Y2, invInertia0Z0, invInertia0Z1, invInertia0Z2;
  Vec4V invInertia1X0, invInertia1X1, invInertia1X2, invInertia1Y0, invInertia1Y1, invInertia1Y2, invInertia1Z0, invInertia1Z1, invInertia1Z2;
  PX_TRANSPOSE_44_34(invInertia00X, invInertia10X, invInertia20X, invInertia30X, invInertia0X0, invInertia0Y0, invInertia0Z0);
  PX_TRANSPOSE_44_34(invInertia00Y, invInertia10Y, invInertia20Y, invInertia30Y, invInertia0X1, invInertia0Y1, invInertia0Z1);
  PX_TRANSPOSE_44_34(invInertia00Z, invInertia10Z, invInertia20Z, invInertia30Z, invInertia0X2, invInertia0Y2, invInertia0Z2);
  PX_TRANSPOSE_44_34(invInertia01X, invInertia11X, invInertia21X, invInertia31X, invInertia1X0, invInertia1Y0, invInertia1Z0);
  PX_TRANSPOSE_44_34(invInertia01Y, invInertia11Y, invInertia21Y, invInertia31Y, invInertia1X1, invInertia1Y1, invInertia1Z1);
  PX_TRANSPOSE_44_34(invInertia01Z, invInertia11Z, invInertia21Z, invInertia31Z, invInertia1X2, invInertia1Y2, invInertia1Z2);
  const Vec4V invInertiaScale0 = header->angD0;
  const Vec4V invInertiaScale1 = header->angD1;
  Vec4V rot00 = V4LoadA(&t[0][0]->deltaBody2WorldQ.x), rot01 = V4LoadA(&t[0][1]->deltaBody2WorldQ.x);
  Vec4V rot10 = V4LoadA(&t[1][0]->deltaBody2WorldQ.x), rot11 = V4LoadA(&t[1][1]->deltaBody2WorldQ.x);
  Vec4V rot20 = V4LoadA(&t[2][0]->deltaBody2WorldQ.x), rot21 = V4LoadA(&t[2][1]->deltaBody2WorldQ.x);
  Vec4V rot30 = V4LoadA(&t[3][0]->deltaBody2WorldQ.x), rot31 = V4LoadA(&t[3][1]->deltaBody2WorldQ.x);
  Vec4V rot0X, rot0Y, rot0Z, rot0W, rot1X, rot1Y, rot1Z, rot1W;
  PX_TRANSPOSE_44(rot00, rot10, rot20, rot30, rot0X, rot0Y, rot0Z, rot0W);
  PX_TRANSPOSE_44(rot01, rot11, rot21, rot31, rot1X, rot1Y, rot1Z, rot1W);
  Vec4V raX, raY, raZ, rbX, rbY, rbZ;
  QuatRotate4(rot0X, rot0Y, rot0Z, rot0W, header->rAWorld[0], header->rAWorld[1], header->rAWorld[2], raX, raY, raZ);
  QuatRotate4(rot1X, rot1Y, rot1Z, rot1W, header->rBWorld[0], header->rBWorld[1], header->rBWorld[2], rbX, rbY, rbZ);
  const Vec4V raMotionX = V4Sub(V4Add(raX, linDelta0T0), header->rAWorld[0]);
  const Vec4V raMotionY = V4Sub(V4Add(raY, linDelta0T1), header->rAWorld[1]);
  const Vec4V raMotionZ = V4Sub(V4Add(raZ, linDelta0T2), header->rAWorld[2]);
  const Vec4V rbMotionX = V4Sub(V4Add(rbX, linDelta1T0), header->rBWorld[0]);
  const Vec4V rbMotionY = V4Sub(V4Add(rbY, linDelta1T1), header->rBWorld[1]);
  const Vec4V rbMotionZ = V4Sub(V4Add(rbZ, linDelta1T2), header->rBWorld[2]);
  const Vec4V mass0 = header->invMass0D0;
  const Vec4V mass1 = header->invMass1D1;
  const VecU32V orthoMask = U4Load(SC_ORTHO_TARGET);
  const VecU32V limitMask = U4Load(SC_INEQUALITY);
  const VecU32V springFlagMask = U4Load(SC_SPRING);
  const Vec4V zero = V4Zero();
  const Vec4V one = V4One();
  const Vec4V error0 = V4Add(header->angOrthoError[0],
                             V4Sub(V4Dot3x(header->angOrthoAxis0X[0], header->angOrthoAxis0Y[0], header->angOrthoAxis0Z[0], angDelta0T0, angDelta0T1, angDelta0T2),
                                   V4Dot3x(header->angOrthoAxis1X[0], header->angOrthoAxis1Y[0], header->angOrthoAxis1Z[0], angDelta1T0, angDelta1T1, angDelta1T2)));
  const Vec4V error1 = V4Add(header->angOrthoError[1],
                             V4Sub(V4Dot3x(header->angOrthoAxis0X[1], header->angOrthoAxis0Y[1], header->angOrthoAxis0Z[1], angDelta0T0, angDelta0T1, angDelta0T2),
                                   V4Dot3x(header->angOrthoAxis1X[1], header->angOrthoAxis1Y[1], header->angOrthoAxis1Z[1], angDelta1T0, angDelta1T1, angDelta1T2)));
  const Vec4V error2 = V4Add(header->angOrthoError[2],
                             V4Sub(V4Dot3x(header->angOrthoAxis0X[2], header->angOrthoAxis0Y[2], header->angOrthoAxis0Z[2], angDelta0T0, angDelta0T1, angDelta0T2),
                                   V4Dot3x(header->angOrthoAxis1X[2], header->angOrthoAxis1Y[2], header->angOrthoAxis1Z[2], angDelta1T0, angDelta1T1, angDelta1T2)));
  const uint32_t stride = residual ? uint32_t(sizeof(Sc1DRow4R)) : uint32_t(sizeof(Sc1DRow4));
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; ++i) {
    Sc1DRow4& c = *reinterpret_cast<Sc1DRow4*>(base);
    const Vec4V cangVel0X = V4Add(c.ang0[0], V4NegMulSub(raZ, c.lin0[1], V4Mul(raY, c.lin0[2])));
    const Vec4V cangVel0Y = V4Add(c.ang0[1], V4NegMulSub(raX, c.lin0[2], V4Mul(raZ, c.lin0[0])));
    const Vec4V cangVel0Z = V4Add(c.ang0[2], V4NegMulSub(raY, c.lin0[0], V4Mul(raX, c.lin0[1])));
    const Vec4V cangVel1X = V4Add(c.ang1[0], V4NegMulSub(rbZ, c.lin1[1], V4Mul(rbY, c.lin1[2])));
    const Vec4V cangVel1Y = V4Add(c.ang1[1], V4NegMulSub(rbX, c.lin1[2], V4Mul(rbZ, c.lin1[0])));
    const Vec4V cangVel1Z = V4Add(c.ang1[2], V4NegMulSub(rbY, c.lin1[0], V4Mul(rbX, c.lin1[1])));
    const VecU32V flags = U4LoadA(c.flags);
    const BoolV useOrtho = V4IsEqU32(V4U32and(flags, orthoMask), orthoMask);
    const Vec4V angOrthoCoefficient = V4Sel(useOrtho, one, zero);
    Vec4V delAngVel0X = V4Mul(invInertia0X0, cangVel0X);
    Vec4V delAngVel0Y = V4Mul(invInertia0X1, cangVel0X);
    Vec4V delAngVel0Z = V4Mul(invInertia0X2, cangVel0X);
    delAngVel0X = V4MulAdd(invInertia0Y0, cangVel0Y, delAngVel0X);
    delAngVel0Y = V4MulAdd(invInertia0Y1, cangVel0Y, delAngVel0Y);
    delAngVel0Z = V4MulAdd(invInertia0Y2, cangVel0Y, delAngVel0Z);
    delAngVel0X = V4MulAdd(invInertia0Z0, cangVel0Z, delAngVel0X);
    delAngVel0Y = V4MulAdd(invInertia0Z1, cangVel0Z, delAngVel0Y);
    delAngVel0Z = V4MulAdd(invInertia0Z2, cangVel0Z, delAngVel0Z);
    Vec4V delAngVel1X = V4Mul(invInertia1X0, cangVel1X);
    Vec4V delAngVel1Y = V4Mul(invInertia1X1, cangVel1X);
    Vec4V delAngVel1Z = V4Mul(invInertia1X2, cangVel1X);
    delAngVel1X = V4MulAdd(invInertia1Y0, cangVel1Y, delAngVel1X);
    delAngVel1Y = V4MulAdd(invInertia1Y1, cangVel1Y, delAngVel1Y);
    delAngVel1Z = V4MulAdd(invInertia1Y2, cangVel1Y, delAngVel1Z);
    delAngVel1X = V4MulAdd(invInertia1Z0, cangVel1Z, delAngVel1X);
    delAngVel1Y = V4MulAdd(invInertia1Z1, cangVel1Z, delAngVel1Y);
    delAngVel1Z = V4MulAdd(invInertia1Z2, cangVel1Z, delAngVel1Z);
    Vec4V err = c.error;
    {
      const Sc1DHeader4& h = *header;
      const Vec4V proj0 = V4Mul(V4MulAdd(h.angOrthoAxis0X[0], delAngVel0X, V4MulAdd(h.angOrthoAxis0Y[0], delAngVel0Y,
                                V4MulAdd(h.angOrthoAxis0Z[0], delAngVel0Z, V4MulAdd(h.angOrthoAxis1X[0], delAngVel1X,
                                V4MulAdd(h.angOrthoAxis1Y[0], delAngVel1Y, V4Mul(h.angOrthoAxis1Z[0], delAngVel1Z)))))), h.angOrthoRecipResponse[0]);
      const Vec4V proj1 = V4Mul(V4MulAdd(h.angOrthoAxis0X[1], delAngVel0X, V4MulAdd(h.angOrthoAxis0Y[1], delAngVel0Y,
                                V4MulAdd(h.angOrthoAxis0Z[1], delAngVel0Z, V4MulAdd(h.angOrthoAxis1X[1], delAngVel1X,
                                V4MulAdd(h.angOrthoAxis1Y[1], delAngVel1Y, V4Mul(h.angOrthoAxis1Z[1], delAngVel1Z)))))), h.angOrthoRecipResponse[1]);
      const Vec4V proj2 = V4Mul(V4MulAdd(h.angOrthoAxis0X[2], delAngVel0X, V4MulAdd(h.angOrthoAxis0Y[2], delAngVel0Y,
                                V4MulAdd(h.angOrthoAxis0Z[2], delAngVel0Z, V4MulAdd(h.angOrthoAxis1X[2], delAngVel1X,
                                V4MulAdd(h.angOrthoAxis1Y[2], delAngVel1Y, V4Mul(h.angOrthoAxis1Z[2], delAngVel1Z)))))), h.angOrthoRecipResponse[2]);
      const Vec4V delta0X = V4MulAdd(h.angOrthoAxis0X[0], proj0, V4MulAdd(h.angOrthoAxis0X[1], proj1, V4Mul(h.angOrthoAxis0X[2], proj2)));
      const Vec4V delta0Y = V4MulAdd(h.angOrthoAxis0Y[0], proj0, V4MulAdd(h.angOrthoAxis0Y[1], proj1, V4Mul(h.angOrthoAxis0Y[2], proj2)));
      const Vec4V delta0Z = V4MulAdd(h.angOrthoAxis0Z[0], proj0, V4MulAdd(h.angOrthoAxis0Z[1], proj1, V4Mul(h.angOrthoAxis0Z[2], proj2)));
      const Vec4V delta1X = V4MulAdd(h.angOrthoAxis1X[0], proj0, V4MulAdd(h.angOrthoAxis1X[1], proj1, V4Mul(h.angOrthoAxis1X[2], proj2)));
      const Vec4V delta1Y = V4MulAdd(h.angOrthoAxis1Y[0], proj0, V4MulAdd(h.angOrthoAxis1Y[1], proj1, V4Mul(h.angOrthoAxis1Y[2], proj2)));
      const Vec4V delta1Z = V4MulAdd(h.angOrthoAxis1Z[0], proj0, V4MulAdd(h.angOrthoAxis1Z[1], proj1, V4Mul(h.angOrthoAxis1Z[2], proj2)));
      delAngVel0X = V4NegMulSub(delta0X, angOrthoCoefficient, delAngVel0X);
      delAngVel0Y = V4NegMulSub(delta0Y, angOrthoCoefficient, delAngVel0Y);
      delAngVel0Z = V4NegMulSub(delta0Z, angOrthoCoefficient, delAngVel0Z);
      delAngVel1X = V4NegMulSub(delta1X, angOrthoCoefficient, delAngVel1X);
      delAngVel1Y = V4NegMulSub(delta1Y, angOrthoCoefficient, delAngVel1Y);
      delAngVel1Z = V4NegMulSub(delta1Z, angOrthoCoefficient, delAngVel1Z);
      const Vec4V orthoBasisError = V4Mul(c.biasScale, V4MulAdd(error0, proj0, V4MulAdd(error1, proj1, V4Mul(error2, proj2))));
      err = V4Sub(err, V4Mul(orthoBasisError, angOrthoCoefficient));
    }
    // (원본은 여기서 ang0I*/ang1I* 를 계산하지만 쓰지 않는다 — 결과에 영향 없음)
    const Vec4V clinVel0X = c.lin0[0], clinVel0Y = c.lin0[1], clinVel0Z = c.lin0[2];
    const Vec4V clinVel1X = c.lin1[0], clinVel1Y = c.lin1[1], clinVel1Z = c.lin1[2];
    const BoolV isSpringConstraint = V4IsEqU32(V4U32and(flags, springFlagMask), springFlagMask);
    // computeResolvedGeometricErrorTGSBlock (shared/DyCpuGpu1dConstraint.h:718)
    Vec4V errorChange;
    {
      const Vec4V deltaAng0 = V4MulAdd(delAngVel0X, angDelta0T0, V4MulAdd(delAngVel0Y, angDelta0T1, V4Mul(delAngVel0Z, angDelta0T2)));
      const Vec4V deltaAng1 = V4MulAdd(delAngVel1X, angDelta1T0, V4MulAdd(delAngVel1Y, angDelta1T1, V4Mul(delAngVel1Z, angDelta1T2)));
      const Vec4V deltaAng = V4Mul(c.angularErrorScale, V4Sub(deltaAng0, deltaAng1));
      const Vec4V deltaLin0 = V4MulAdd(clinVel0X, raMotionX, V4MulAdd(clinVel0Y, raMotionY, V4Mul(clinVel0Z, raMotionZ)));
      const Vec4V deltaLin1 = V4MulAdd(clinVel1X, rbMotionX, V4MulAdd(clinVel1Y, rbMotionY, V4Mul(clinVel1Z, rbMotionZ)));
      const Vec4V deltaLin = V4Sub(deltaLin0, deltaLin1);
      const Vec4V motion = V4Add(deltaLin, deltaAng);
      errorChange = V4Sel(isSpringConstraint, motion, V4NegScaleSub(c.velTarget, elapsedTime, motion));
    }
    const Vec4V dotDelAngVel0 = V4MulAdd(delAngVel0X, delAngVel0X, V4MulAdd(delAngVel0Y, delAngVel0Y, V4Mul(delAngVel0Z, delAngVel0Z)));
    const Vec4V dotDelAngVel1 = V4MulAdd(delAngVel1X, delAngVel1X, V4MulAdd(delAngVel1Y, delAngVel1Y, V4Mul(delAngVel1Z, delAngVel1Z)));
    const Vec4V dotClinVel0 = V4MulAdd(clinVel0X, clinVel0X, V4MulAdd(clinVel0Y, clinVel0Y, V4Mul(clinVel0Z, clinVel0Z)));
    const Vec4V dotClinVel1 = V4MulAdd(clinVel1X, clinVel1X, V4MulAdd(clinVel1Y, clinVel1Y, V4Mul(clinVel1Z, clinVel1Z)));
    const Vec4V resp0 = V4MulAdd(mass0, dotClinVel0, V4Mul(invInertiaScale0, dotDelAngVel0));
    const Vec4V resp1 = V4MulAdd(mass1, dotClinVel1, V4Mul(invInertiaScale1, dotDelAngVel1));
    const Vec4V response = V4Add(resp0, resp1);
    const Vec4V recipResponse = V4Sel(V4IsGrtr(response, V4Zero()), V4Recip(response), V4Zero());
    const Vec4V vMul = V4Sel(isSpringConstraint, c.velMultiplier, V4Mul(recipResponse, c.velMultiplier));
    // computeMinBiasTGSBlock
    const BoolV isInequality = V4IsEqU32(V4U32and(flags, limitMask), limitMask);
    const Vec4V minBias = V4Neg(V4Sel(isInequality, Vec4V_From_FloatV(FMax()), c.maxBias));
    const Vec4V unclampedBias = V4MulAdd(errorChange, c.biasScale, err);
    const Vec4V bias = V4Clamp(unclampedBias, minBias, c.maxBias);
    const Vec4V constant = V4Sel(isSpringConstraint, V4Add(bias, c.velTarget), V4Mul(recipResponse, V4Add(bias, c.velTarget)));
    const Vec4V normalVel0 = V4MulAdd(clinVel0X, linVel0T0, V4MulAdd(clinVel0Y, linVel0T1, V4Mul(clinVel0Z, linVel0T2)));
    const Vec4V normalVel1 = V4MulAdd(clinVel1X, linVel1T0, V4MulAdd(clinVel1Y, linVel1T1, V4Mul(clinVel1Z, linVel1T2)));
    const Vec4V angVel0 = V4MulAdd(delAngVel0X, angState0T0, V4MulAdd(delAngVel0Y, angState0T1, V4Mul(delAngVel0Z, angState0T2)));
    const Vec4V angVel1 = V4MulAdd(delAngVel1X, angState1T0, V4MulAdd(delAngVel1Y, angState1T1, V4Mul(delAngVel1Z, angState1T2)));
    const Vec4V normalVel = V4Add(V4Sub(normalVel0, normalVel1), V4Sub(angVel0, angVel1));
    const Vec4V unclampedForce = V4Add(c.appliedForce, V4MulAdd(vMul, normalVel, constant));
    const Vec4V clampedForce = V4Clamp(unclampedForce, c.minImpulse, c.maxImpulse);
    const Vec4V deltaF = V4Sub(clampedForce, c.appliedForce);
    c.appliedForce = clampedForce;
    if (residual) {  // Dy::calculateResidualV4 (DyResidualAccumulator.h:51)
      Sc1DRow4R& cc = static_cast<Sc1DRow4R&>(c);
      const Vec4V r = V4Sel(V4IsEq(vMul, V4Zero()), V4Zero(), V4DivFast(deltaF, vMul));
      if (isPositionIteration) cc.residualPosIter = r;
      else cc.residualVelIter = r;
    }
    const Vec4V deltaFIM0 = V4Mul(deltaF, mass0);
    const Vec4V deltaFIM1 = V4Mul(deltaF, mass1);
    const Vec4V angDetaF0 = V4Mul(deltaF, invInertiaScale0);
    const Vec4V angDetaF1 = V4Mul(deltaF, invInertiaScale1);
    linVel0T0 = V4MulAdd(clinVel0X, deltaFIM0, linVel0T0);
    linVel1T0 = V4NegMulSub(clinVel1X, deltaFIM1, linVel1T0);
    angState0T0 = V4MulAdd(delAngVel0X, angDetaF0, angState0T0);
    angState1T0 = V4NegMulSub(delAngVel1X, angDetaF1, angState1T0);
    linVel0T1 = V4MulAdd(clinVel0Y, deltaFIM0, linVel0T1);
    linVel1T1 = V4NegMulSub(clinVel1Y, deltaFIM1, linVel1T1);
    angState0T1 = V4MulAdd(delAngVel0Y, angDetaF0, angState0T1);
    angState1T1 = V4NegMulSub(delAngVel1Y, angDetaF1, angState1T1);
    linVel0T2 = V4MulAdd(clinVel0Z, deltaFIM0, linVel0T2);
    linVel1T2 = V4NegMulSub(clinVel1Z, deltaFIM1, linVel1T2);
    angState0T2 = V4MulAdd(delAngVel0Z, angDetaF0, angState0T2);
    angState1T2 = V4NegMulSub(delAngVel1Z, angDetaF1, angState1T2);
    base += stride;
  }
  PX_TRANSPOSE_44(linVel0T0, linVel0T1, linVel0T2, linVel0T3, linVel00, linVel10, linVel20, linVel30);
  PX_TRANSPOSE_44(linVel1T0, linVel1T1, linVel1T2, linVel1T3, linVel01, linVel11, linVel21, linVel31);
  PX_TRANSPOSE_44(angState0T0, angState0T1, angState0T2, angState0T3, angState00, angState10, angState20, angState30);
  PX_TRANSPOSE_44(angState1T0, angState1T1, angState1T2, angState1T3, angState01, angState11, angState21, angState31);
  // 원본과 같은 순서로 저장 (같은 몸체를 두 칸이 가리키면 나중 것이 남는다)
  V4StoreA(linVel00, &b[0][0]->linearVelocity.x); V4StoreA(angState00, &b[0][0]->angularVelocity.x);
  V4StoreA(linVel10, &b[1][0]->linearVelocity.x); V4StoreA(angState10, &b[1][0]->angularVelocity.x);
  V4StoreA(linVel20, &b[2][0]->linearVelocity.x); V4StoreA(angState20, &b[2][0]->angularVelocity.x);
  V4StoreA(linVel30, &b[3][0]->linearVelocity.x); V4StoreA(angState30, &b[3][0]->angularVelocity.x);
  V4StoreA(linVel01, &b[0][1]->linearVelocity.x); V4StoreA(angState01, &b[0][1]->angularVelocity.x);
  V4StoreA(linVel11, &b[1][1]->linearVelocity.x); V4StoreA(angState11, &b[1][1]->angularVelocity.x);
  V4StoreA(linVel21, &b[2][1]->linearVelocity.x); V4StoreA(angState21, &b[2][1]->angularVelocity.x);
  V4StoreA(linVel31, &b[3][1]->linearVelocity.x); V4StoreA(angState31, &b[3][1]->angularVelocity.x);
}

// ---- conclude1DStep4 (DyTGSContactPrepBlock.cpp:3583)
EHD void conclude1DStep4(uint8_t* blk, bool residual) {
  using namespace aos;
  if (blk == nullptr) return;
  const Sc1DHeader4* header = reinterpret_cast<const Sc1DHeader4*>(blk);
  uint8_t* base = blk + sizeof(Sc1DHeader4);
  const uint32_t stride = residual ? uint32_t(sizeof(Sc1DRow4R)) : uint32_t(sizeof(Sc1DRow4));
  const VecI32V keepBiasMask = I4Load(int32_t(SC_KEEP_BIAS));
  const VecI32V isSpringMask = I4Load(int32_t(SC_SPRING));
  const Vec4V zero = V4Zero();
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; ++i) {
    Sc1DRow4& c = *reinterpret_cast<Sc1DRow4*>(base);
    const VecI32V flags = I4LoadA(reinterpret_cast<const int32_t*>(c.flags));
    const BoolV keepBias = VecI32V_IsEq(VecI32V_And(flags, keepBiasMask), keepBiasMask);
    c.biasScale = V4Sel(keepBias, c.biasScale, zero);
    c.error = V4Sel(keepBias, c.error, zero);
    const BoolV isSpring = VecI32V_IsEq(VecI32V_And(flags, isSpringMask), isSpringMask);
    c.biasScale = V4Sel(isSpring, zero, c.biasScale);
    c.error = V4Sel(isSpring, zero, c.error);
    c.velMultiplier = V4Sel(isSpring, zero, c.velMultiplier);
    c.velTarget = V4Sel(isSpring, zero, c.velTarget);
    base += stride;
  }
}

// ---- writeBack1D4 (DyTGSContactPrepBlock.cpp:3418). wb[k] 가 nullptr 이면 그 칸은 안 씀
EHD void writeBack1D4(const uint8_t* blk, Writeback* const wb[4], bool residualEnabled) {
  using namespace aos;
  if (!(wb[0] || wb[1] || wb[2] || wb[3])) return;
  const Sc1DHeader4* header = reinterpret_cast<const Sc1DHeader4*>(blk);
  const uint8_t* base = blk + sizeof(Sc1DHeader4);
  const uint32_t stride = residualEnabled ? uint32_t(sizeof(Sc1DRow4R)) : uint32_t(sizeof(Sc1DRow4));
  const Vec4V zero = V4Zero();
  Vec4V linX(zero), linY(zero), linZ(zero), angX(zero), angY(zero), angZ(zero);
  Vec4V residual(zero), residualPosIter(zero);
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; i++) {
    const Sc1DRow4* c = reinterpret_cast<const Sc1DRow4*>(base);
    const VecI32V flags = I4LoadU(reinterpret_cast<const int32_t*>(&c->flags[0]));
    const VecI32V mask = I4Load(int32_t(SC_OUTPUT_FORCE));
    const VecI32V masked = VecI32V_And(flags, mask);
    const BoolV isEq = VecI32V_IsEq(masked, mask);
    const Vec4V appliedForce = V4Sel(isEq, c->appliedForce, zero);
    if (residualEnabled) {
      const Sc1DRow4R* cc = static_cast<const Sc1DRow4R*>(c);
      residual = V4MulAdd(cc->residualVelIter, cc->residualVelIter, residual);
      residualPosIter = V4MulAdd(cc->residualPosIter, cc->residualPosIter, residualPosIter);
    }
    linX = V4MulAdd(c->lin0[0], appliedForce, linX);
    linY = V4MulAdd(c->lin0[1], appliedForce, linY);
    linZ = V4MulAdd(c->lin0[2], appliedForce, linZ);
    angX = V4MulAdd(c->ang0[0], appliedForce, angX);
    angY = V4MulAdd(c->ang0[1], appliedForce, angY);
    angZ = V4MulAdd(c->ang0[2], appliedForce, angZ);
    base += stride;
  }
  angX = V4Sub(angX, V4NegMulSub(header->body0WorkOffset[0], linY, V4Mul(header->body0WorkOffset[1], linZ)));
  angY = V4Sub(angY, V4NegMulSub(header->body0WorkOffset[1], linZ, V4Mul(header->body0WorkOffset[2], linX)));
  angZ = V4Sub(angZ, V4NegMulSub(header->body0WorkOffset[2], linX, V4Mul(header->body0WorkOffset[0], linY)));
  const Vec4V linLenSq = V4MulAdd(linZ, linZ, V4MulAdd(linY, linY, V4Mul(linX, linX)));
  const Vec4V angLenSq = V4MulAdd(angZ, angZ, V4MulAdd(angY, angY, V4Mul(angX, angX)));
  const Vec4V linLen = V4Sqrt(linLenSq);
  const Vec4V angLen = V4Sqrt(angLenSq);
  const BoolV broken = BOr(V4IsGrtr(linLen, header->linBreakImpulse), V4IsGrtr(angLen, header->angBreakImpulse));
  alignas(16) uint32_t iBroken[4];
  BStoreA(broken, iBroken);
  alignas(16) float residual4[4], residual4PosIter[4];
  V4StoreA(residual, residual4);
  V4StoreA(residualPosIter, residual4PosIter);
  Vec4V lin0, lin1, lin2, lin3, ang0, ang1, ang2, ang3;
  PX_TRANSPOSE_34_44(linX, linY, linZ, lin0, lin1, lin2, lin3);
  PX_TRANSPOSE_34_44(angX, angY, angZ, ang0, ang1, ang2, ang3);
  const Vec4V ls[4] = {lin0, lin1, lin2, lin3}, as[4] = {ang0, ang1, ang2, ang3};
  for (int k = 0; k < 4; ++k) {
    if (!wb[k]) continue;
    PxVec3 t;
    V3StoreU(Vec3V_From_Vec4V_WUndefined(ls[k]), t); wb[k]->linearImpulse = t;
    V3StoreU(Vec3V_From_Vec4V_WUndefined(as[k]), t); wb[k]->angularImpulse = t;
    const bool isBroken = header->breakable[k] ? (iBroken[k] != 0) : false;
    wb[k]->broken_residualPosIter = f2u(residual4PosIter[k]);  // setCombined
    wb[k]->broken_residualPosIter = setBit(wb[k]->broken_residualPosIter, 31, isBroken);
    wb[k]->residual = residual4[k];
  }
}

}  // namespace jnt
}  // namespace eng
