// joints 모듈: TGS 1D 제약 (조인트 행 -> 풀이 제약 블록 준비, 반복 한 번 풀기, 마무리, 되쓰기). PhysX 5.6.1 과 바이트까지 같은 블록.
// solver 모듈의 TGS 반복 루프가 부르는 함수들이다 (docs 12절 경계: 섬 묶기·분할·배치·반복 루프 = solver).
// 원본(physx/source/lowleveldynamics 기준)
//   준비      : src/DyTGSContactPrep.cpp:1939 (setupSolverConstraintStep), :2166 (SetupSolverConstraintStep: 셰이더 호출)
//   행 전처리 : src/DyConstraintSetup.cpp:492 (preprocessRows), :307 (orthogonalize), :146 (diagonalize — eIMPROVED_SLERP 때만)
//   상수 계산 : shared/DyCpuGpu1dConstraint.h (computeMinMaxImpulseOrForceAsImpulse, computeJointSpeedTGS, computeMaxBiasVelocityTGS,
//               compute1dConstraintSolverConstantsTGS, raiseInternalFlagsTGS, computeResolvedGeometricErrorTGS, computeMinBiasTGS)
//   풀기      : src/DyTGSContactPrep.cpp:2418 (solve1DStep), :2750 (conclude1DStep), :2837 (writeBack1DStep)
//   관절체 쪽 : src/DyTGSContactPrep.cpp:212-320 (SolverExtBodyStep, getImpulseResponse), :2211 (solveExt1D), :2309 (solveExt1DStep)
// 블록 배치(바이트): 머리 Sc1DHeader(176 B) + 행 Sc1DRow(96 B) 또는 Sc1DRowExt(160 B) × 개수. 앞에서 0 으로 채운다(PxMemSet).
// 아직 안 옮긴 것: 4개 묶음 SIMD 경로(DyTGSContactPrepBlock.cpp setupSolverConstraintStep4/solve1D4 — 강체-강체 조인트 4개가
//   한 분할에 모일 때), eIMPROVED_SLERP 대각화, 잔차(residual) 보고의 rcpps 근사 나눗셈.
#pragma once
#include <cstdint>

#include "../common/aos.h"
#include "d6_prep.h"
#include "joint_types.h"

namespace eng {
namespace jnt {


// Sc1DRow union 비트 (DySolverConstraint1DStep.h:207-240)
EHD uint32_t setBit(uint32_t value, uint32_t bit, bool state) { return state ? (value | (1u << bit)) : (value & ~(1u << bit)); }
EHD void setUseAngularError(Sc1DRow& s, bool b) { s.useAngularError_residualPosIter = setBit(s.useAngularError_residualPosIter, 31, b); }
EHD float getUseAngularError(const Sc1DRow& s) { return (s.useAngularError_residualPosIter & 0x80000000u) ? 1.0f : 0.0f; }
EHD void setPositionIterationResidual(Sc1DRow& s, float r) {
  const bool b = getUseAngularError(s) != 0.0f;
  s.useAngularError_residualPosIter = f2u(r);
  setUseAngularError(s, b);
}
EHD float getPositionIterationResidual(const Sc1DRow& s) { return fabsf(u2f(s.useAngularError_residualPosIter)); }
EHD void initRow(Sc1DRow& c, const V3& l0, const V3& l1, const V3& a0, const V3& a1, float minI, float maxI) {
  c.lin0 = l0; c.lin1 = l1; c.ang0 = a0; c.ang1 = a1;
  c.minImpulse = minI; c.maxImpulse = maxI;
  c.flags = 0; c.appliedForce = 0;
  setUseAngularError(c, true);
  c.residualVelIter = 0.0f;
  setPositionIterationResidual(c, 0.0f);
}

// ---- 행 전처리 (DyConstraintSetup.cpp) — aos 코드 그대로
struct alignas(16) V4a { float x, y, z, w; };  // PX_ALIGN(16, PxVec4)
struct MassProps {
  aos::FloatV invMass0, invMass1, invInertiaScale0, invInertiaScale1;
  EHD MassProps(float imass0, float imass1, const InvMassScale& ims)
      : invMass0(aos::FLoad(imass0 * ims.linear0)), invMass1(aos::FLoad(imass1 * ims.linear1)),
        invInertiaScale0(aos::FLoad(ims.angular0)), invInertiaScale1(aos::FLoad(ims.angular1)) {}
};

// orthogonalize (DyConstraintSetup.cpp:307)
EHD void orthogonalize(Row** row, V4a* angSqrtInvInertia0, V4a* angSqrtInvInertia1, uint32_t rowCount, uint32_t eqRowCount, const MassProps& m) {
  using namespace aos;
  const FloatV zero = FZero();
  Vec3V lin1m[6], ang1m[6], lin1[6], ang1[6];
  Vec4V lin0m[6], ang0m[6];
  Vec4V lin0AndG[6], ang0AndT[6];
  for (uint32_t i = 0; i < rowCount; i++) {
    Vec4V l0AndG = V4LoadA(&row[i]->linear0.x);
    Vec4V a0AndT = V4LoadA(&row[i]->angular0.x);
    Vec3V l1 = Vec3V_From_Vec4V(V4LoadA(&row[i]->linear1.x));
    Vec3V a1 = Vec3V_From_Vec4V(V4LoadA(&row[i]->angular1.x));
    Vec4V angSqrtL0 = V4LoadA(&angSqrtInvInertia0[i].x);
    Vec4V angSqrtL1 = V4LoadA(&angSqrtInvInertia1[i].x);
    const uint32_t eliminationRows = i < eqRowCount ? i : eqRowCount;
    for (uint32_t j = 0; j < eliminationRows; j++) {
      const Vec3V s0 = V3MulAdd(l1, lin1m[j], Vec3V_From_Vec4V_WUndefined(V4Mul(l0AndG, lin0m[j])));
      const Vec3V s1 = V3MulAdd(Vec3V_From_Vec4V_WUndefined(angSqrtL1), ang1m[j], Vec3V_From_Vec4V_WUndefined(V4Mul(angSqrtL0, ang0m[j])));
      const FloatV t = V3SumElems(V3Add(s0, s1));
      l0AndG = V4NegScaleSub(lin0AndG[j], t, l0AndG);
      a0AndT = V4NegScaleSub(ang0AndT[j], t, a0AndT);
      l1 = V3NegScaleSub(lin1[j], t, l1);
      a1 = V3NegScaleSub(ang1[j], t, a1);
      angSqrtL0 = V4NegScaleSub(V4LoadA(&angSqrtInvInertia0[j].x), t, angSqrtL0);
      angSqrtL1 = V4NegScaleSub(V4LoadA(&angSqrtInvInertia1[j].x), t, angSqrtL1);
    }
    V4StoreA(l0AndG, &row[i]->linear0.x);
    V4StoreA(a0AndT, &row[i]->angular0.x);
    PxVec3 t3;
    V3StoreA(l1, t3); row[i]->linear1 = t3;
    V3StoreA(a1, t3); row[i]->angular1 = t3;
    V4StoreA(angSqrtL0, &angSqrtInvInertia0[i].x);
    V4StoreA(angSqrtL1, &angSqrtInvInertia1[i].x);
    if (i < eqRowCount) {
      lin0AndG[i] = l0AndG;
      ang0AndT[i] = a0AndT;
      lin1[i] = l1;
      ang1[i] = a1;
      const Vec3V l0 = Vec3V_From_Vec4V(l0AndG);
      const Vec3V l0m = V3Scale(l0, m.invMass0);
      const Vec3V l1m = V3Scale(l1, m.invMass1);
      const Vec4V a0m = V4Scale(angSqrtL0, m.invInertiaScale0);
      const Vec4V a1m = V4Scale(angSqrtL1, m.invInertiaScale1);
      const Vec3V s0 = V3MulAdd(l0, l0m, V3Mul(l1, l1m));
      const Vec4V s1 = V4MulAdd(a0m, angSqrtL0, V4Mul(a1m, angSqrtL1));
      const FloatV s = V3SumElems(V3Add(s0, Vec3V_From_Vec4V_WUndefined(s1)));
      const FloatV a = FSel(FIsGrtr(s, zero), FRecip(s), zero);
      lin0m[i] = V4Scale(V4ClearW(Vec4V_From_Vec3V(l0m)), a);
      ang0m[i] = V4Scale(V4ClearW(a0m), a);
      lin1m[i] = V3Scale(l1m, a);
      ang1m[i] = V3Scale(Vec3V_From_Vec4V_WUndefined(a1m), a);
    }
  }
}

// M33MulV4 (DyConstraintSetup.cpp:476)
EHD aos::Vec3V M33MulV4(const aos::Mat33V& a, const aos::Vec4V b) {
  using namespace aos;
  const FloatV x = V4GetX(b), y = V4GetY(b), z = V4GetZ(b);
  const Vec3V v0 = V3Scale(a.col0, x), v1 = V3Scale(a.col1, y), v2 = V3Scale(a.col2, z);
  const Vec3V v0PlusV1 = V3Add(v0, v1);
  return V3Add(v0PlusV1, v2);
}

// preprocessRows (DyConstraintSetup.cpp:492). 반환: 0 = 성공, 1 = 대각화(eIMPROVED_SLERP) 필요한데 아직 안 옮김
EHD int preprocessRows(Row** sorted, Row* rows, V4a* angSqrtInvInertia0, V4a* angSqrtInvInertia1, uint32_t rowCount,
                       const M33& sqrtInvInertia0F32, const M33& sqrtInvInertia1F32, float invMass0, float invMass1,
                       const InvMassScale& ims, bool disablePreprocessing, bool diagonalizeDrive) {
  using namespace aos;
  for (uint32_t i = 0; i < rowCount; i++) {  // 삽입 정렬 (안정)
    Row* r = rows + i;
    uint32_t j = i;
    for (; j > 0 && r->solveHint < sorted[j - 1]->solveHint; j--) sorted[j] = sorted[j - 1];
    sorted[j] = r;
  }
  const Mat33V sqrtInvInertia0 = Mat33V(V3LoadU_SafeReadW(PxVec3(sqrtInvInertia0F32.c0)), V3LoadU_SafeReadW(PxVec3(sqrtInvInertia0F32.c1)),
                                        V3LoadU(PxVec3(sqrtInvInertia0F32.c2)));
  const Mat33V sqrtInvInertia1 = Mat33V(V3LoadU_SafeReadW(PxVec3(sqrtInvInertia1F32.c0)), V3LoadU_SafeReadW(PxVec3(sqrtInvInertia1F32.c1)),
                                        V3LoadU(PxVec3(sqrtInvInertia1F32.c2)));
  for (uint32_t i = 0; i < rowCount; ++i) {
    const Vec3V angDelta0 = M33MulV4(sqrtInvInertia0, V4LoadA(&sorted[i]->angular0.x));
    const Vec3V angDelta1 = M33MulV4(sqrtInvInertia1, V4LoadA(&sorted[i]->angular1.x));
    V4StoreA(Vec4V_From_Vec3V(angDelta0), &angSqrtInvInertia0[i].x);
    V4StoreA(Vec4V_From_Vec3V(angDelta1), &angSqrtInvInertia1[i].x);
  }
  if (disablePreprocessing) return 0;
  const MassProps m(invMass0, invMass1, ims);
  int status = 0;
  for (uint32_t i = 0; i < rowCount;) {
    const uint32_t groupMajorId = uint32_t(sorted[i]->solveHint >> 8), start = i++;
    while (i < rowCount && uint32_t(sorted[i]->solveHint >> 8) == groupMajorId) i++;
    if (groupMajorId == 4 || groupMajorId == 8) {
      uint32_t bCount = start;
      for (; bCount < i && (sorted[bCount]->solveHint & 255) == 0; bCount++) {}
      orthogonalize(sorted + start, angSqrtInvInertia0 + start, angSqrtInvInertia1 + start, i - start, bCount - start, m);
    }
    if (groupMajorId == 1 && diagonalizeDrive) status = 1;  // TODO: diagonalize (DyConstraintSetup.cpp:146)
  }
  return status;
}

// ---- 상수 계산 (shared/DyCpuGpu1dConstraint.h)
EHD float computeRecipUnitResponse(float unitResponse, float minRowResponse) { return unitResponse <= minRowResponse ? 0 : 1.0f / unitResponse; }
EHD float computeBounceVelocity(uint16_t flags, float jointSpeed, float bounceThreshold, float restitution, float geometricError) {
  const float bounceVel = jointSpeed * (-restitution);
  if ((flags & RF_RESTITUTION) && (-jointSpeed > bounceThreshold) && ((bounceVel * geometricError) <= 0.0f)) return bounceVel;
  return 0.0f;
}
EHD float computeMaxBiasVelocityTGS(uint16_t flags, float jointSpeed, float bounceThreshold, float restitution, float geometricError,
                                    bool isExtended, float lengthScale, float recipSimDt) {
  float maxBiasSpeed = MAX_F32;
  if (flags & RF_SPRING) {
    maxBiasSpeed = MAX_F32;
  } else {
    const float bounceVel = computeBounceVelocity(flags, jointSpeed, bounceThreshold, restitution, geometricError);
    if (bounceVel != 0.0f) maxBiasSpeed = 0;
    else if (flags & RF_ANGULAR_CONSTRAINT) maxBiasSpeed = recipSimDt * 0.75f;
    else maxBiasSpeed = isExtended ? recipSimDt * 1.5f * lengthScale : recipSimDt * 15.f * lengthScale;
  }
  return maxBiasSpeed;
}
struct Consts { float biasScale, error, velMultiplier, targetVel; };
EHD Consts compute1dConstantsTGS(uint16_t flags, float springStiffness, float springDamping, float restitution, float bounceThreshold,
                                 float geometricError, float velocityTarget, float jointSpeed, float initJointSpeed,
                                 float unitResponse, float recipUnitResponse, float erp, float stepDt, float recipStepDt) {
  Consts d{0.0f, 0.0f, 0.0f, 0.0f};
  if (flags & RF_SPRING) {
    const float a = stepDt * (stepDt * springStiffness + springDamping);
    const float b = stepDt * (springDamping * velocityTarget);
    if (flags & RF_ACCELERATION_SPRING) {
      const float x = 1.0f / (1.0f + a);
      const float biasScale = -x * springStiffness * stepDt * recipUnitResponse;
      const float velMultiplier = -x * a * recipUnitResponse;
      d.biasScale = biasScale;
      d.error = biasScale * geometricError;
      d.velMultiplier = velMultiplier;
      d.targetVel = x * b * recipUnitResponse - velMultiplier * initJointSpeed;
    } else {
      const float x = 1.0f / (1.0f + a * unitResponse);
      const float biasScale = -x * springStiffness * stepDt;
      const float velMultiplier = -x * a;
      d.biasScale = biasScale;
      d.error = biasScale * geometricError;
      d.velMultiplier = velMultiplier;
      d.targetVel = x * b - velMultiplier * initJointSpeed;
    }
  } else {
    const float bounceVel = computeBounceVelocity(flags, jointSpeed, bounceThreshold, restitution, geometricError);
    if (bounceVel != 0.0f) {
      const float velMultiplier = -1.0f;
      d.biasScale = 0.f;
      d.error = 0.f;
      d.velMultiplier = velMultiplier;
      d.targetVel = bounceVel - velMultiplier * initJointSpeed;
    } else {
      const float velMultiplier = -1.0f;
      const float biasScale = -recipStepDt * erp;
      d.biasScale = biasScale;
      d.error = geometricError * biasScale;
      d.velMultiplier = velMultiplier;
      d.targetVel = velocityTarget - velMultiplier * initJointSpeed;
    }
  }
  return d;
}
EHD void raiseInternalFlagsTGS(uint16_t externalFlags, uint16_t hint, uint32_t& internalFlags) {
  if (externalFlags & RF_SPRING) internalFlags |= SC_SPRING;
  if (externalFlags & RF_ACCELERATION_SPRING) internalFlags |= SC_ACCELERATION_SPRING;
  if (externalFlags & RF_OUTPUT_FORCE) internalFlags |= SC_OUTPUT_FORCE;
  if (externalFlags & RF_KEEPBIAS) internalFlags |= SC_KEEP_BIAS;
  if (hint & 1) internalFlags |= SC_INEQUALITY;
}

// ---- 관절체 쪽 접근 (articulation 모듈이 채운다). 강체만 있는 조인트는 NoArt.
// Art 가 줘야 하는 것 (PhysX FeatherstoneArticulation 과 같은 의미·같은 연산):
//   float getCfm(uint32_t link)
//   void getImpulseResponse(uint32_t link, const V3& impLin, const V3& impAng, V3& dvLin, V3& dvAng)   // 입력은 이미 dom 배율 적용됨
//   void getLinkVelocity(uint32_t link, V3& lin, V3& ang)                                             // Cm::SpatialVectorV (w=0 적재)
//   (풀기용 함수들은 아래 solveExt1DStep 머리말)
struct NoArt {
  EHD float getCfm(uint32_t) const { return 0.0f; }
  EHD void getImpulseResponse(uint32_t, const V3&, const V3&, V3& a, V3& b) const { a = V3{0, 0, 0}; b = V3{0, 0, 0}; }
  EHD void getLinkVelocity(uint32_t, V3& a, V3& b) const { a = V3{0, 0, 0}; b = V3{0, 0, 0}; }
  EHD void getVelocity(uint32_t, V3& a, V3& b) const { a = V3{0, 0, 0}; b = V3{0, 0, 0}; }
  EHD void getVelocities(uint32_t, uint32_t, V3& a, V3& b, V3& c, V3& d) const { a = b = c = d = V3{0, 0, 0}; }
  EHD void getMotionVector(uint32_t, V3& a, V3& b) const { a = V3{0, 0, 0}; b = V3{0, 0, 0}; }
  EHD Q getDeltaQ(uint32_t) const { return qid(); }
  EHD void applyImpulse(uint32_t, const V3&, const V3&) {}
  EHD void applyImpulses(uint32_t, const V3&, const V3&, uint32_t, const V3&, const V3&) {}
};

// ---- setupSolverConstraintStep (DyTGSContactPrep.cpp:1939). blk 는 blockLength(numRows, isExtended) 바이트 이상.
// 반환: numRows (0 이면 블록 없음). outCount 는 실제로 쓴 행 수(관절체 퇴화 행 건너뜀).
template <class Art>
EHD uint32_t setupSolverConstraintStep(PrepIn& p, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt,
                                       float lengthScale, float biasCoefficient, const Art& artA, const Art& artB,
                                       int* status = nullptr) {
  if (p.numRows == 0) return 0;
  const bool isExtended = p.linkIndexA != RIGID_BODY || p.linkIndexB != RIGID_BODY;
  const bool isKinematic0 = p.linkIndexA == RIGID_BODY && p.body0->isKinematic;
  const bool isKinematic1 = p.linkIndexB == RIGID_BODY && p.body1->isKinematic;
  const uint32_t stride = isExtended ? uint32_t(sizeof(Sc1DRowExt)) : uint32_t(sizeof(Sc1DRow));
  const uint32_t constraintLength = uint32_t(sizeof(Sc1DHeader)) + stride * p.numRows;
  for (uint32_t i = 0; i < constraintLength; ++i) blk[i] = 0;  // PxMemSet(desc.constraint, 0, constraintLength)

  Sc1DHeader* header = reinterpret_cast<Sc1DHeader*>(blk);
  uint8_t* constraints = blk + sizeof(Sc1DHeader);
  // init(header, ...) (DySolverConstraint1DStep.h:161)
  header->type = uint8_t(isExtended ? SC_TYPE_EXT_1D : SC_TYPE_RB_1D);
  header->count = uint8_t(p.numRows);
  header->dominance = 0;
  header->linearInvMassScale0 = p.invMassScales.linear0;
  header->angularInvMassScale0 = p.invMassScales.angular0;
  header->linearInvMassScale1 = -p.invMassScales.linear1;
  header->angularInvMassScale1 = -p.invMassScales.angular1;
  header->body0WorldOffset = p.body0WorldOffset;
  header->linBreakImpulse = p.linBreakForce * simDt;
  header->angBreakImpulse = p.angBreakForce * simDt;
  header->breakable = uint8_t((p.linBreakForce != MAX_F32) || (p.angBreakForce != MAX_F32));
  header->invMass0D0 = p.data0->invMass * p.invMassScales.linear0;
  header->invMass1D1 = p.data1->invMass * p.invMassScales.linear1;
  header->rAWorld = p.cA2w - p.bodyFrame0.p;
  header->rBWorld = p.cB2w - p.bodyFrame1.p;

  Row* sorted[MAX_CONSTRAINT_ROWS];
  V4a angSqrtInvInertia0[MAX_CONSTRAINT_ROWS];
  V4a angSqrtInvInertia1[MAX_CONSTRAINT_ROWS];
  for (uint32_t i = 0; i < p.numRows; ++i) {
    if (p.rows[i].flags & RF_ANGULAR_CONSTRAINT) {
      if (p.rows[i].solveHint == SH_EQUALITY) p.rows[i].solveHint = SH_ROTATIONAL_EQUALITY;
      else if (p.rows[i].solveHint == SH_INEQUALITY) p.rows[i].solveHint = SH_ROTATIONAL_INEQUALITY;
    }
  }
  const int st = preprocessRows(sorted, p.rows, angSqrtInvInertia0, angSqrtInvInertia1, p.numRows, p.txI0->sqrtInvInertia,
                                p.txI1->sqrtInvInertia, p.data0->invMass, p.data1->invMass, p.invMassScales,
                                isExtended || p.disablePreprocessing, p.improvedSlerp);
  if (status) *status = st;

  const float erp = 0.5f * biasCoefficient;
  uint32_t orthoCount = 0;
  uint32_t outCount = 0;
  float cfm = 0.f;
  if (isExtended) {
    const float c0 = p.linkIndexA == RIGID_BODY ? 0.f : artA.getCfm(p.linkIndexA);
    const float c1 = p.linkIndexB == RIGID_BODY ? 0.f : artB.getCfm(p.linkIndexB);
    cfm = pmax(c0, c1);
  }

  for (uint32_t i = 0; i < p.numRows; i++) {
    Sc1DRow& s = *reinterpret_cast<Sc1DRow*>(constraints);
    const Row& c = *sorted[i];
    // computeMinMaxImpulseOrForceAsImpulse
    const float driveScale = ((c.flags & RF_HAS_DRIVE_LIMIT) && p.driveLimitsAreForces) ? simDt : 1.0f;
    const float minImpulse = c.minImpulse * driveScale;
    const float maxImpulse = c.maxImpulse * driveScale;
    float unitResponse;
    float jointSpeedForRestitutionBounce = 0.0f;
    float initJointSpeed = 0.0f;
    if (!isExtended) {
      const V3 angSqrtInvInertia0V3{angSqrtInvInertia0[i].x, angSqrtInvInertia0[i].y, angSqrtInvInertia0[i].z};
      const V3 angSqrtInvInertia1V3{angSqrtInvInertia1[i].x, angSqrtInvInertia1[i].y, angSqrtInvInertia1[i].z};
      initRow(s, c.linear0, c.linear1, c.angular0, c.angular1, minImpulse, maxImpulse);
      const float linSumMass = magSq(s.lin0) * p.data0->invMass * p.invMassScales.linear0 +
                               magSq(s.lin1) * p.data1->invMass * p.invMassScales.linear1;
      const float resp0 = magSq(angSqrtInvInertia0V3) * p.invMassScales.angular0;
      const float resp1 = magSq(angSqrtInvInertia1V3) * p.invMassScales.angular1;
      unitResponse = resp0 + resp1 + linSumMass;
      // PxTGSSolverBodyData::projectVelocity (PxSolverDefs.h:609)
      const float vel0 = dot(p.data0->originalLinearVelocity, s.lin0) + dot(p.data0->originalAngularVelocity, s.ang0);
      const float vel1 = dot(p.data1->originalLinearVelocity, s.lin1) + dot(p.data1->originalAngularVelocity, s.ang1);
      // computeJointSpeedTGS
      initJointSpeed = 0.f;
      if (isKinematic0) initJointSpeed -= vel0;
      if (isKinematic1) initJointSpeed += vel1;
      jointSpeedForRestitutionBounce = vel0 - vel1;
      if (!(c.flags & RF_ANGULAR_CONSTRAINT)) {
        s.ang0 = V3{0.f, 0.f, 0.f};
        s.ang1 = V3{0.f, 0.f, 0.f};
        setUseAngularError(s, false);
      }
    } else {
      // createImpulseResponseVector (DyTGSContactPrep.cpp:212)
      auto respVec = [&](const V3& lin, const V3& ang, uint32_t link, const TgsTxInertia* txI, V3& oLin, V3& oAng) {
        oLin = lin;
        oAng = link == RIGID_BODY ? txI->sqrtInvInertia * ang : ang;
      };
      V3 r0l, r0a, r1l, r1a;
      respVec(c.linear0, c.angular0, p.linkIndexA, p.txI0, r0l, r0a);
      respVec(vneg(c.linear1), vneg(c.angular1), p.linkIndexB, p.txI1, r1l, r1a);
      initRow(s, r0l, vneg(r1l), r0a, vneg(r1a), minImpulse, maxImpulse);
      Sc1DRowExt& e = static_cast<Sc1DRowExt&>(s);
      // getImpulseResponse (DyTGSContactPrep.cpp:229, allowSelfCollision = false)
      V3 d0l, d0a, d1l, d1a;
      if (p.linkIndexA == RIGID_BODY) {
        d0l = r0l * p.data0->invMass * p.invMassScales.linear0;
        d0a = r0a * p.invMassScales.angular0;
      } else {
        artA.getImpulseResponse(p.linkIndexA, r0l * p.invMassScales.linear0, r0a * p.invMassScales.angular0, d0l, d0a);
      }
      float response = dot(r0l, d0l) + dot(r0a, d0a);
      if (p.linkIndexB == RIGID_BODY) {
        d1l = r1l * p.data1->invMass * p.invMassScales.linear1;
        d1a = r1a * p.invMassScales.angular1;
      } else {
        artB.getImpulseResponse(p.linkIndexB, r1l * p.invMassScales.linear1, r1a * p.invMassScales.angular1, d1l, d1a);
      }
      response += dot(r1l, d1l) + dot(r1a, d1a);
      e.deltaVA.lin[0] = d0l.x; e.deltaVA.lin[1] = d0l.y; e.deltaVA.lin[2] = d0l.z;
      e.deltaVA.ang[0] = d0a.x; e.deltaVA.ang[1] = d0a.y; e.deltaVA.ang[2] = d0a.z;
      e.deltaVB.lin[0] = d1l.x; e.deltaVB.lin[1] = d1l.y; e.deltaVB.lin[2] = d1l.z;
      e.deltaVB.ang[0] = d1a.x; e.deltaVB.ang[1] = d1a.y; e.deltaVB.ang[2] = d1a.z;
      unitResponse = response;
      if (unitResponse < 1e-12f) continue;  // DY_ARTICULATION_MIN_RESPONSE: 이 칸은 다음 행이 덮어쓴다
      unitResponse += cfm;
      // SolverExtBodyStep::projectVelocity (DyTGSContactPrep.cpp:305)
      auto proj = [&](uint32_t link, const TgsBodyData* d, const Art& art, const V3& lin, const V3& ang) -> float {
        if (link == RIGID_BODY) return dot(d->originalLinearVelocity, lin) + dot(d->originalAngularVelocity, ang);
        V3 vl3, va3;
        art.getLinkVelocity(link, vl3, va3);
        const aos::Vec3V vl = aos::V3LoadA(PxVec3(vl3)), va = aos::V3LoadA(PxVec3(va3));
        float f;  // SpatialVectorV::dot(SpatialVectorV(SpatialVector)) = V3SumElems(lin*lin' + ang*ang')
        aos::FStore(aos::V3SumElems(aos::V3Add(aos::V3Mul(vl, aos::V3LoadA(PxVec3(lin))), aos::V3Mul(va, aos::V3LoadA(PxVec3(ang))))), &f);
        return f;
      };
      const float vel0 = proj(p.linkIndexA, p.data0, artA, s.lin0, s.ang0);
      const float vel1 = proj(p.linkIndexB, p.data1, artB, s.lin1, s.ang1);
      initJointSpeed = 0.f;
      if (isKinematic0) initJointSpeed -= vel0;
      if (isKinematic1) initJointSpeed += vel1;
      jointSpeedForRestitutionBounce = vel0 - vel1;
      if (!(c.flags & RF_ANGULAR_CONSTRAINT)) setUseAngularError(s, false);
    }
    const float recipUnitResponse = computeRecipUnitResponse(unitResponse, p.minResponseThreshold);
    s.recipResponse = recipUnitResponse;
    s.maxBias = computeMaxBiasVelocityTGS(c.flags, jointSpeedForRestitutionBounce, c.mod1, c.mod0, c.geometricError, isExtended,
                                          lengthScale, recipSimDt);
    const Consts k = compute1dConstantsTGS(c.flags, c.mod0, c.mod1, c.mod0, c.mod1, c.geometricError, c.velocityTarget,
                                           jointSpeedForRestitutionBounce, initJointSpeed, unitResponse, recipUnitResponse, erp,
                                           stepDt, recipStepDt);
    s.biasScale = k.biasScale;
    s.error = k.error;
    s.velTarget = k.targetVel;
    s.velMultiplier = k.velMultiplier;
    raiseInternalFlagsTGS(c.flags, c.solveHint, s.flags);
    if (!(isExtended || p.disablePreprocessing)) {
      if (c.solveHint == SH_ROTATIONAL_EQUALITY) {
        s.flags |= SC_ROT_EQ;
        const V4a& a0 = angSqrtInvInertia0[i];
        const V4a& a1 = angSqrtInvInertia1[i];
        header->angOrthoAxis0_recipResponseW[orthoCount] =
            V4f{a0.x * p.invMassScales.angular0, a0.y * p.invMassScales.angular0, a0.z * p.invMassScales.angular0, recipUnitResponse};
        header->angOrthoAxis1_Error[orthoCount].x = a1.x * p.invMassScales.angular1;
        header->angOrthoAxis1_Error[orthoCount].y = a1.y * p.invMassScales.angular1;
        header->angOrthoAxis1_Error[orthoCount].z = a1.z * p.invMassScales.angular1;
        header->angOrthoAxis1_Error[orthoCount].w = c.geometricError;
        orthoCount++;
      } else if (c.solveHint & SH_EQUALITY) {
        s.flags |= SC_ORTHO_TARGET;
      }
    }
    constraints += stride;
    outCount++;
  }
  header->count = uint8_t(outCount);
  return p.numRows;
}

// ---- SetupSolverConstraintStep (DyTGSContactPrep.cpp:2166): 셰이더 실행 + 운동학 몸체 각 질량 척도 0 + 준비
// rows 는 MAX_CONSTRAINT_ROWS 칸 이상. constraintFlags = D6Joint::constraintFlags.
template <class Art>
EHD uint32_t prepareD6Step(const D6Data& data, uint16_t constraintFlags, float linBreakForce, float angBreakForce, float minResponseThreshold,
                           const Tf& bodyFrame0, const Tf& bodyFrame1, const TgsBodyVel& b0, const TgsBodyVel& b1,
                           const TgsTxInertia& t0, const TgsTxInertia& t1, const TgsBodyData& d0, const TgsBodyData& d1,
                           uint32_t linkIndexA, uint32_t linkIndexB, Row* rows, uint8_t* blk, float stepDt, float simDt,
                           float recipStepDt, float recipSimDt, float lengthScale, float biasCoefficient, const Art& artA,
                           const Art& artB, uint32_t* outLength, int* status = nullptr) {
  PrepIn p;
  p.disablePreprocessing = (constraintFlags & CF_DISABLE_PREPROCESSING) != 0;  // setupConstraintFlags (DyConstraintPrep.h:93)
  p.improvedSlerp = (constraintFlags & CF_IMPROVED_SLERP) != 0;
  p.driveLimitsAreForces = (constraintFlags & CF_DRIVE_LIMITS_ARE_FORCES) != 0;
  p.extendedLimits = (constraintFlags & CF_ENABLE_EXTENDED_LIMITS) != 0;
  p.disableConstraint = (constraintFlags & CF_DISABLE_CONSTRAINT) != 0;
  setupConstraintRows(rows, MAX_CONSTRAINT_ROWS);
  p.invMassScales = InvMassScale{1.f, 1.f, 1.f, 1.f};
  p.body0WorldOffset = V3{0.0f, 0.0f, 0.0f};
  p.cA2w = V3{0, 0, 0};
  p.cB2w = V3{0, 0, 0};
  if (p.disableConstraint) {
    p.numRows = 0;
  } else {
    const PrepOut o = d6SolverPrep(rows, data, bodyFrame0, bodyFrame1, p.extendedLimits);
    p.numRows = o.numRows;
    p.invMassScales = o.invMassScale;
    p.body0WorldOffset = o.body0WorldOffset;
    p.cA2w = o.cA2w;
    p.cB2w = o.cB2w;
  }
  p.rows = rows;
  p.bodyFrame0 = bodyFrame0;
  p.bodyFrame1 = bodyFrame1;
  p.body0 = &b0; p.body1 = &b1; p.txI0 = &t0; p.txI1 = &t1; p.data0 = &d0; p.data1 = &d1;
  p.linkIndexA = linkIndexA; p.linkIndexB = linkIndexB;
  p.linBreakForce = linBreakForce; p.angBreakForce = angBreakForce; p.minResponseThreshold = minResponseThreshold;
  if (linkIndexA == RIGID_BODY && b0.isKinematic) p.invMassScales.angular0 = 0.f;
  if (linkIndexB == RIGID_BODY && b1.isKinematic) p.invMassScales.angular1 = 0.f;
  const uint32_t n = setupSolverConstraintStep(p, blk, stepDt, simDt, recipStepDt, recipSimDt, lengthScale, biasCoefficient, artA, artB, status);
  if (outLength) *outLength = n ? blockLength(p.numRows, linkIndexA != RIGID_BODY || linkIndexB != RIGID_BODY) : 0;
  return n;
}

// ---- solve1DStep (DyTGSContactPrep.cpp:2418) — 강체-강체. aos 코드 그대로.
// residual: 잔차 보고(PxSceneFlag::eENABLE_SOLVER_RESIDUAL_REPORTING) 켰을 때 (cache.contactErrorAccumulator != NULL)
EHD void solve1DStep(uint8_t* blk, TgsBodyVel& b0, TgsBodyVel& b1, const TgsTxInertia& txI0, const TgsTxInertia& txI1, float elapsedTime,
                     bool residual = false, bool isPositionIteration = true) {
  using namespace aos;
  if (blk == nullptr) return;
  const FloatV elapsed = FLoad(elapsedTime);
  const Sc1DHeader* header = reinterpret_cast<const Sc1DHeader*>(blk);
  Sc1DRow* base = reinterpret_cast<Sc1DRow*>(blk + sizeof(Sc1DHeader));

  Vec3V linVel0 = V3LoadA(PxVec3(b0.linearVelocity));
  Vec3V linVel1 = V3LoadA(PxVec3(b1.linearVelocity));
  Vec3V angState0 = V3LoadA(PxVec3(b0.angularVelocity));
  Vec3V angState1 = V3LoadA(PxVec3(b1.angularVelocity));
  const Mat33V sqrtInvInertia0 = Mat33V(V3LoadU_SafeReadW(PxVec3(txI0.sqrtInvInertia.c0)), V3LoadU_SafeReadW(PxVec3(txI0.sqrtInvInertia.c1)),
                                        V3LoadU(PxVec3(txI0.sqrtInvInertia.c2)));
  const Mat33V sqrtInvInertia1 = Mat33V(V3LoadU_SafeReadW(PxVec3(txI1.sqrtInvInertia.c0)), V3LoadU_SafeReadW(PxVec3(txI1.sqrtInvInertia.c1)),
                                        V3LoadU(PxVec3(txI1.sqrtInvInertia.c2)));
  const FloatV invMass0 = FLoad(header->invMass0D0);
  const FloatV invMass1 = FLoad(header->invMass1D1);
  const FloatV invInertiaScale0 = FLoad(header->angularInvMassScale0);
  const FloatV invInertiaScale1 = FLoad(header->angularInvMassScale1);

  Vec3V raMotion, rbMotion;
  VecCrossV raCross, rbCross;
  Vec3V lin0, ang0, lin1, ang1;
  {
    const QuatV deltaRotA = QuatVLoadA(&txI0.deltaBody2WorldQ.x);
    const QuatV deltaRotB = QuatVLoadA(&txI1.deltaBody2WorldQ.x);
    const Vec3V raPrev = V3LoadA(PxVec3(header->rAWorld));
    const Vec3V rbPrev = V3LoadA(PxVec3(header->rBWorld));
    const Vec3V ra = QuatRotate(deltaRotA, raPrev);
    const Vec3V rb = QuatRotate(deltaRotB, rbPrev);
    raCross = V3PrepareCross(ra);
    rbCross = V3PrepareCross(rb);
    lin0 = V3LoadA(PxVec3(b0.deltaLinDt));
    ang0 = V3LoadA(PxVec3(b0.deltaAngDt));
    lin1 = V3LoadA(PxVec3(b1.deltaLinDt));
    ang1 = V3LoadA(PxVec3(b1.deltaAngDt));
    raMotion = V3Sub(V3Add(ra, lin0), raPrev);
    rbMotion = V3Sub(V3Add(rb, lin1), rbPrev);
  }
  const Vec4V ang0Ortho0_recipResponseW = V4LoadA(&header->angOrthoAxis0_recipResponseW[0].x);
  const Vec4V ang0Ortho1_recipResponseW = V4LoadA(&header->angOrthoAxis0_recipResponseW[1].x);
  const Vec4V ang0Ortho2_recipResponseW = V4LoadA(&header->angOrthoAxis0_recipResponseW[2].x);
  const Vec4V ang1Ortho0_Error0 = V4LoadA(&header->angOrthoAxis1_Error[0].x);
  const Vec4V ang1Ortho1_Error1 = V4LoadA(&header->angOrthoAxis1_Error[1].x);
  const Vec4V ang1Ortho2_Error2 = V4LoadA(&header->angOrthoAxis1_Error[2].x);
  const FloatV recipResponse0 = V4GetW(ang0Ortho0_recipResponseW);
  const FloatV recipResponse1 = V4GetW(ang0Ortho1_recipResponseW);
  const FloatV recipResponse2 = V4GetW(ang0Ortho2_recipResponseW);
  const Vec3V ang0Ortho0 = Vec3V_From_Vec4V(ang0Ortho0_recipResponseW);
  const Vec3V ang0Ortho1 = Vec3V_From_Vec4V(ang0Ortho1_recipResponseW);
  const Vec3V ang0Ortho2 = Vec3V_From_Vec4V(ang0Ortho2_recipResponseW);
  const Vec3V ang1Ortho0 = Vec3V_From_Vec4V(ang1Ortho0_Error0);
  const Vec3V ang1Ortho1 = Vec3V_From_Vec4V(ang1Ortho1_Error1);
  const Vec3V ang1Ortho2 = Vec3V_From_Vec4V(ang1Ortho2_Error2);
  const FloatV error0 = FAdd(V4GetW(ang1Ortho0_Error0), FSub(V3Dot(ang0Ortho0, ang0), V3Dot(ang1Ortho0, ang1)));
  const FloatV error1 = FAdd(V4GetW(ang1Ortho1_Error1), FSub(V3Dot(ang0Ortho1, ang0), V3Dot(ang1Ortho1, ang1)));
  const FloatV error2 = FAdd(V4GetW(ang1Ortho2_Error2), FSub(V3Dot(ang0Ortho2, ang0), V3Dot(ang1Ortho2, ang1)));
  const VecU32V springFlagMask = U4Load(SC_SPRING);

  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; ++i, base++) {
    Sc1DRow& c = *base;
    const Vec3V clinVel0 = V3LoadA(PxVec3(c.lin0));
    const Vec3V clinVel1 = V3LoadA(PxVec3(c.lin1));
    const Vec3V cangVel0_ = V3LoadA(PxVec3(c.ang0));
    const Vec3V cangVel1_ = V3LoadA(PxVec3(c.ang1));
    const FloatV angularErrorScale = FLoad(getUseAngularError(c));
    const FloatV biasScale = FLoad(c.biasScale);
    const FloatV maxBias = FLoad(c.maxBias);
    const FloatV targetVel = FLoad(c.velTarget);
    const FloatV appliedForce = FLoad(c.appliedForce);
    const FloatV velMultiplier = FLoad(c.velMultiplier);
    const FloatV maxImpulse = FLoad(c.maxImpulse);
    const FloatV minImpulse = FLoad(c.minImpulse);
    const Vec3V cangVel0 = V3Add(cangVel0_, V3Cross(raCross, clinVel0));
    const Vec3V cangVel1 = V3Add(cangVel1_, V3Cross(rbCross, clinVel1));
    FloatV error = FLoad(c.error);
    const FloatV minBias = FNeg((c.flags & SC_INEQUALITY) ? FMax() : maxBias);  // computeMinBiasTGS
    Vec3V raXnI = M33MulV3(sqrtInvInertia0, cangVel0);
    Vec3V rbXnI = M33MulV3(sqrtInvInertia1, cangVel1);
    if (c.flags & SC_ORTHO_TARGET) {
      const FloatV proj0 = FMul(V3SumElems(V3MulAdd(raXnI, ang0Ortho0, V3Mul(rbXnI, ang1Ortho0))), recipResponse0);
      const FloatV proj1 = FMul(V3SumElems(V3MulAdd(raXnI, ang0Ortho1, V3Mul(rbXnI, ang1Ortho1))), recipResponse1);
      const FloatV proj2 = FMul(V3SumElems(V3MulAdd(raXnI, ang0Ortho2, V3Mul(rbXnI, ang1Ortho2))), recipResponse2);
      const Vec3V delta0 = V3ScaleAdd(ang0Ortho0, proj0, V3ScaleAdd(ang0Ortho1, proj1, V3Scale(ang0Ortho2, proj2)));
      const Vec3V delta1 = V3ScaleAdd(ang1Ortho0, proj0, V3ScaleAdd(ang1Ortho1, proj1, V3Scale(ang1Ortho2, proj2)));
      raXnI = V3Sub(raXnI, delta0);
      rbXnI = V3Sub(rbXnI, delta1);
      const FloatV orthoBasisError = FMul(biasScale, FScaleAdd(error0, proj0, FScaleAdd(error1, proj1, FMul(error2, proj2))));
      error = FSub(error, orthoBasisError);
    }
    const BoolV isSpringConstraint = V4IsEqU32(V4U32and(U4Load(c.flags), springFlagMask), springFlagMask);
    // computeResolvedGeometricErrorTGS (shared/DyCpuGpu1dConstraint.h:689)
    const FloatV deltaAng = FMul(angularErrorScale, FSub(V3Dot(raXnI, ang0), V3Dot(rbXnI, ang1)));
    const FloatV deltaLin = FSub(V3Dot(clinVel0, raMotion), V3Dot(clinVel1, rbMotion));
    const FloatV motion = FAdd(deltaLin, deltaAng);
    const FloatV errorChange = FSel(isSpringConstraint, motion, FNegScaleSub(targetVel, elapsed, motion));

    const FloatV resp0 = FScaleAdd(invMass0, V3Dot(clinVel0, clinVel0), V3SumElems(V3Mul(V3Scale(raXnI, invInertiaScale0), raXnI)));
    const FloatV resp1 = FSub(FMul(invMass1, V3Dot(clinVel1, clinVel1)), V3SumElems(V3Mul(V3Scale(rbXnI, invInertiaScale1), rbXnI)));
    const FloatV response = FAdd(resp0, resp1);
    const FloatV recipResponse = FSel(FIsGrtr(response, FZero()), FRecip(response), FZero());
    const FloatV vMul = FSel(isSpringConstraint, velMultiplier, FMul(recipResponse, velMultiplier));
    const FloatV unclampedBias = FScaleAdd(errorChange, biasScale, error);
    const FloatV bias = FClamp(unclampedBias, minBias, maxBias);
    const FloatV constant = FSel(isSpringConstraint, FAdd(bias, targetVel), FMul(recipResponse, FAdd(bias, targetVel)));
    const Vec3V v0 = V3MulAdd(linVel0, clinVel0, V3Mul(angState0, raXnI));
    const Vec3V v1 = V3MulAdd(linVel1, clinVel1, V3Mul(angState1, rbXnI));
    const FloatV normalVel = V3SumElems(V3Sub(v0, v1));
    const FloatV unclampedForce = FAdd(appliedForce, FScaleAdd(vMul, normalVel, constant));
    const FloatV clampedForce = FClamp(unclampedForce, minImpulse, maxImpulse);
    const FloatV deltaF = FSub(clampedForce, appliedForce);
    FStore(clampedForce, &c.appliedForce);
    if (residual) {  // Dy::calculateResidual (DyResidualAccumulator.h:45): FSel(vMul==0, 0, FDivFast(deltaF, vMul)) — rcpps 는 common/aos 표 흉내
      float r;
      FStore(FSel(FIsEq(vMul, FZero()), FZero(), FDivFast(deltaF, vMul)), &r);
      if (isPositionIteration) setPositionIterationResidual(c, r);
      else c.residualVelIter = r;
    }
    linVel0 = V3ScaleAdd(clinVel0, FMul(deltaF, invMass0), linVel0);
    linVel1 = V3NegScaleSub(clinVel1, FMul(deltaF, invMass1), linVel1);
    angState0 = V3ScaleAdd(raXnI, FMul(deltaF, invInertiaScale0), angState0);
    angState1 = V3ScaleAdd(rbXnI, FMul(deltaF, invInertiaScale1), angState1);
  }
  PxVec3 t;
  V3StoreA(linVel0, t); b0.linearVelocity = t;
  V3StoreA(angState0, t); b0.angularVelocity = t;
  V3StoreA(linVel1, t); b1.linearVelocity = t;
  V3StoreA(angState1, t); b1.angularVelocity = t;
}

// ---- 관절체 쪽 1D 풀기 (DyTGSContactPrep.cpp:2211 solveExt1D, :2309 solveExt1DStep)
// Art(관절체 접근) 이 줘야 하는 것 — PhysX FeatherstoneArticulation 의 같은 이름 함수와 같은 뜻 (articulation 모듈 art_step.h 에 있음):
//   void getVelocity(uint32_t link, V3& lin, V3& ang)                     // pxcFsGetVelocity(link) (Cm::SpatialVector -> w=0 으로 적재)
//   void getVelocities(uint32_t l0, uint32_t l1, V3& lin0, V3& ang0, V3& lin1, V3& ang1)  // pxcFsGetVelocities
//   void getMotionVector(uint32_t link, V3& lin, V3& ang)                 // getLinkMotionVector
//   Q getDeltaQ(uint32_t link)
//   void applyImpulse(uint32_t link, const V3& lin, const V3& ang)        // pxcFsApplyImpulse(link, lin, ang, NULL)
//   void applyImpulses(uint32_t l0, const V3& lin0, const V3& ang0, uint32_t l1, const V3& lin1, const V3& ang1)  // pxcFsApplyImpulses
// artA/artB: 링크가 속한 관절체 (강체 쪽은 nullptr). 같은 관절체면 같은 포인터.
EHD void solveExt1D(uint8_t* blk, aos::Vec3V& linVel0, aos::Vec3V& linVel1, aos::Vec3V& angVel0, aos::Vec3V& angVel1, const aos::Vec3V& linMotion0,
                    const aos::Vec3V& linMotion1, const aos::Vec3V& angMotion0, const aos::Vec3V& angMotion1, const aos::QuatV& rotA,
                    const aos::QuatV& rotB, float elapsedTimeF32, aos::Vec3V& linImpulse0, aos::Vec3V& linImpulse1, aos::Vec3V& angImpulse0,
                    aos::Vec3V& angImpulse1, bool isPositionIteration) {
  using namespace aos;
  const Sc1DHeader* header = reinterpret_cast<const Sc1DHeader*>(blk);
  Sc1DRowExt* base = reinterpret_cast<Sc1DRowExt*>(blk + sizeof(Sc1DHeader));
  const FloatV elapsedTime = FLoad(elapsedTimeF32);
  const Vec3V raPrev = V3LoadA(PxVec3(header->rAWorld));
  const Vec3V rbPrev = V3LoadA(PxVec3(header->rBWorld));
  const Vec3V ra = QuatRotate(rotA, V3LoadA(PxVec3(header->rAWorld)));
  const Vec3V rb = QuatRotate(rotB, V3LoadA(PxVec3(header->rBWorld)));
  const Vec3V raMotion = V3Sub(V3Add(ra, linMotion0), raPrev);
  const Vec3V rbMotion = V3Sub(V3Add(rb, linMotion1), rbPrev);
  Vec3V li0 = V3Zero(), li1 = V3Zero(), ai0 = V3Zero(), ai1 = V3Zero();
  const VecU32V springFlagMask = U4Load(SC_SPRING);
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; ++i, base++) {
    Sc1DRowExt& c = *base;
    const Vec3V clinVel0 = V3LoadA(PxVec3(c.lin0));
    const Vec3V clinVel1 = V3LoadA(PxVec3(c.lin1));
    const Vec3V cangVel0 = V3LoadA(PxVec3(c.ang0));
    const Vec3V cangVel1 = V3LoadA(PxVec3(c.ang1));
    const FloatV recipResponse = FLoad(c.recipResponse);
    const FloatV targetVel = FLoad(c.velTarget);
    const BoolV isSpringConstraint = V4IsEqU32(V4U32and(U4Load(c.flags), springFlagMask), springFlagMask);
    // computeResolvedGeometricErrorTGS (deltaLin0=raMotion, deltaLin1=rbMotion, cLinVel0/1, deltaAngInertia0/1 = angMotion0/1,
    //   cAngVelInertia0/1 = cangVel0/1)
    const FloatV angErrScale = FLoad(getUseAngularError(c));
    const FloatV deltaAng = FMul(angErrScale, FSub(V3Dot(cangVel0, angMotion0), V3Dot(cangVel1, angMotion1)));
    const FloatV deltaLin = FSub(V3Dot(clinVel0, raMotion), V3Dot(clinVel1, rbMotion));
    const FloatV motion = FAdd(deltaLin, deltaAng);
    const FloatV errorChange = FSel(isSpringConstraint, motion, FNegScaleSub(targetVel, elapsedTime, motion));
    const FloatV biasScale = FLoad(c.biasScale);
    const FloatV maxBias = FLoad(c.maxBias);
    const FloatV vMul = FSel(isSpringConstraint, FLoad(c.velMultiplier), FMul(recipResponse, FLoad(c.velMultiplier)));
    const FloatV appliedForce = FLoad(c.appliedForce);
    const FloatV unclampedBias = FScaleAdd(errorChange, biasScale, FLoad(c.error));
    const FloatV minBias = FNeg((c.flags & SC_INEQUALITY) ? FMax() : maxBias);
    const FloatV bias = FClamp(unclampedBias, minBias, maxBias);
    const FloatV constant = FSel(isSpringConstraint, FAdd(bias, targetVel), FMul(recipResponse, FAdd(bias, targetVel)));
    const FloatV maxImpulse = FLoad(c.maxImpulse);
    const FloatV minImpulse = FLoad(c.minImpulse);
    const Vec3V v0 = V3MulAdd(linVel0, clinVel0, V3Mul(angVel0, cangVel0));
    const Vec3V v1 = V3MulAdd(linVel1, clinVel1, V3Mul(angVel1, cangVel1));
    const FloatV normalVel = V3SumElems(V3Sub(v0, v1));
    const FloatV unclampedForce = FAdd(appliedForce, FScaleAdd(vMul, normalVel, constant));
    const FloatV clampedForce = FMin(maxImpulse, (FMax(minImpulse, unclampedForce)));
    const FloatV deltaF = FSub(clampedForce, appliedForce);
    FStore(clampedForce, &c.appliedForce);
    float residual;  // Dy::calculateResidual (DyResidualAccumulator.h:45) — 관절체 쪽은 늘 저장
    FStore(FSel(FIsEq(vMul, FZero()), FZero(), FDivFast(deltaF, vMul)), &residual);
    if (isPositionIteration) setPositionIterationResidual(c, residual);
    else c.residualVelIter = residual;
    FStore(clampedForce, &base->appliedForce);
    li0 = V3ScaleAdd(clinVel0, deltaF, li0);
    ai0 = V3ScaleAdd(cangVel0, deltaF, ai0);
    li1 = V3ScaleAdd(clinVel1, deltaF, li1);
    ai1 = V3ScaleAdd(cangVel1, deltaF, ai1);
    linVel0 = V3ScaleAdd(V3LoadA(c.deltaVA.lin), deltaF, linVel0);
    angVel0 = V3ScaleAdd(V3LoadA(c.deltaVA.ang), deltaF, angVel0);
    linVel1 = V3ScaleAdd(V3LoadA(c.deltaVB.lin), deltaF, linVel1);
    angVel1 = V3ScaleAdd(V3LoadA(c.deltaVB.ang), deltaF, angVel1);
  }
  linImpulse0 = V3Scale(li0, FLoad(header->linearInvMassScale0));
  linImpulse1 = V3Scale(li1, FLoad(header->linearInvMassScale1));
  angImpulse0 = V3Scale(ai0, FLoad(header->angularInvMassScale0));
  angImpulse1 = V3Scale(ai1, FLoad(header->angularInvMassScale1));
}

template <class Art>
EHD void solveExt1DStep(uint8_t* blk, uint32_t linkIndexA, uint32_t linkIndexB, Art* artA, Art* artB, TgsBodyVel* bodyA, TgsBodyVel* bodyB,
                        const TgsTxInertia* txIA, const TgsTxInertia* txIB, float elapsedTimeF32, bool isPositionIteration) {
  using namespace aos;
  if (blk == nullptr) return;
  Vec3V linVel0, angVel0, linVel1, angVel1, linMotion0, angMotion0, linMotion1, angMotion1;
  QuatV rotA, rotB;
  auto ld = [](const V3& v) { return V3LoadA(PxVec3(v)); };
  auto ldq = [](const Q& q) { const float f[4] = {q.x, q.y, q.z, q.w}; return QuatVLoadU(f); };
  if (artA == artB) {
    V3 l0, a0, l1, a1;
    artA->getVelocities(linkIndexA, linkIndexB, l0, a0, l1, a1);
    linVel0 = ld(l0); angVel0 = ld(a0); linVel1 = ld(l1); angVel1 = ld(a1);
    V3 m0l, m0a, m1l, m1a;
    artA->getMotionVector(linkIndexA, m0l, m0a);
    artB->getMotionVector(linkIndexB, m1l, m1a);
    linMotion0 = ld(m0l); angMotion0 = ld(m0a); linMotion1 = ld(m1l); angMotion1 = ld(m1a);
    rotA = ldq(artA->getDeltaQ(linkIndexA));
    rotB = ldq(artB->getDeltaQ(linkIndexB));
  } else {
    if (linkIndexA == RIGID_BODY) {
      linVel0 = ld(bodyA->linearVelocity); angVel0 = ld(bodyA->angularVelocity);
      linMotion0 = ld(bodyA->deltaLinDt); angMotion0 = ld(bodyA->deltaAngDt);
      rotA = QuatVLoadA(&txIA->deltaBody2WorldQ.x);
    } else {
      V3 l, a, ml, ma;
      artA->getVelocity(linkIndexA, l, a);
      rotA = ldq(artA->getDeltaQ(linkIndexA));
      artA->getMotionVector(linkIndexA, ml, ma);
      linVel0 = ld(l); angVel0 = ld(a); linMotion0 = ld(ml); angMotion0 = ld(ma);
    }
    if (linkIndexB == RIGID_BODY) {
      linVel1 = ld(bodyB->linearVelocity); angVel1 = ld(bodyB->angularVelocity);
      linMotion1 = ld(bodyB->deltaLinDt); angMotion1 = ld(bodyB->deltaAngDt);
      rotB = QuatVLoadA(&txIB->deltaBody2WorldQ.x);
    } else {
      V3 l, a, ml, ma;
      artB->getVelocity(linkIndexB, l, a);
      rotB = ldq(artB->getDeltaQ(linkIndexB));
      artB->getMotionVector(linkIndexB, ml, ma);
      linVel1 = ld(l); angVel1 = ld(a); linMotion1 = ld(ml); angMotion1 = ld(ma);
    }
  }
  Vec3V li0, li1, ai0, ai1;
  solveExt1D(blk, linVel0, linVel1, angVel0, angVel1, linMotion0, linMotion1, angMotion0, angMotion1, rotA, rotB, elapsedTimeF32, li0, li1, ai0,
             ai1, isPositionIteration);
  PxVec3 t, t2, t3, t4;
  if (artA == artB) {
    V3StoreA(li0, t); V3StoreA(ai0, t2); V3StoreA(li1, t3); V3StoreA(ai1, t4);
    artA->applyImpulses(linkIndexA, t, t2, linkIndexB, t3, t4);
  } else {
    if (linkIndexA == RIGID_BODY) {
      V3StoreA(linVel0, t); bodyA->linearVelocity = t;
      V3StoreA(angVel0, t); bodyA->angularVelocity = t;
    } else {
      V3StoreA(li0, t); V3StoreA(ai0, t2);
      artA->applyImpulse(linkIndexA, t, t2);
    }
    if (linkIndexB == RIGID_BODY) {
      V3StoreA(linVel1, t); bodyB->linearVelocity = t;
      V3StoreA(angVel1, t); bodyB->angularVelocity = t;
    } else {
      V3StoreA(li1, t); V3StoreA(ai1, t2);
      artB->applyImpulse(linkIndexB, t, t2);
    }
  }
}

// ---- conclude1DStep (DyTGSContactPrep.cpp:2750): 마지막 위치 반복 뒤, 속도 반복 전에
EHD void conclude1DStep(uint8_t* blk) {
  if (blk == nullptr) return;
  const Sc1DHeader* header = reinterpret_cast<const Sc1DHeader*>(blk);
  uint8_t* base = blk + sizeof(Sc1DHeader);
  const uint32_t stride = header->type == SC_TYPE_RB_1D ? uint32_t(sizeof(Sc1DRow)) : uint32_t(sizeof(Sc1DRowExt));
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; ++i, base += stride) {
    Sc1DRow& c = *reinterpret_cast<Sc1DRow*>(base);
    if (!(c.flags & SC_KEEP_BIAS)) {
      c.biasScale = 0.f;
      c.error = 0.f;
    }
    if (c.flags & SC_SPRING) {
      c.velMultiplier = 0.0f;
      c.biasScale = 0.0f;
      c.error = 0.0f;
      c.velTarget = 0.0f;
    }
  }
}

// ---- writeBack1DStep (DyTGSContactPrep.cpp:2837)
EHD void writeBack1DStep(const uint8_t* blk, Writeback* wb) {
  if (!wb || !blk) return;
  const Sc1DHeader* header = reinterpret_cast<const Sc1DHeader*>(blk);
  const uint8_t* base = blk + sizeof(Sc1DHeader);
  const uint32_t stride = header->type == SC_TYPE_EXT_1D ? uint32_t(sizeof(Sc1DRowExt)) : uint32_t(sizeof(Sc1DRow));
  V3 lin{0, 0, 0}, ang{0, 0, 0};
  float constraintErrorSq = 0.0f;
  float constraintErrorPosIterSq = 0.0f;
  const uint32_t count = header->count;
  for (uint32_t i = 0; i < count; i++) {
    const Sc1DRow* c = reinterpret_cast<const Sc1DRow*>(base);
    if (c->flags & SC_OUTPUT_FORCE) {
      lin += c->lin0 * c->appliedForce;
      ang += (c->ang0 + cross(c->lin0, header->rAWorld)) * c->appliedForce;
    }
    float err = c->residualVelIter;
    constraintErrorSq += err * err;
    err = getPositionIterationResidual(*c);
    constraintErrorPosIterSq += err * err;
    base += stride;
  }
  ang = ang - cross(header->body0WorldOffset, lin);
  wb->linearImpulse = lin;
  wb->angularImpulse = ang;
  const uint32_t isBroken = header->breakable ? uint32_t(mag(lin) > header->linBreakImpulse || mag(ang) > header->angBreakImpulse) : 0u;
  wb->broken_residualPosIter = f2u(constraintErrorPosIterSq);  // setCombined
  wb->broken_residualPosIter = setBit(wb->broken_residualPosIter, 31, isBroken != 0);
  wb->residual = constraintErrorSq;
}
EHD bool writebackIsBroken(const Writeback& wb) { return (wb.broken_residualPosIter & 0x80000000u) != 0; }

}  // namespace jnt
}  // namespace eng
