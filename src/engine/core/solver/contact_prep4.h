// TGS 접촉 제약 준비 — 4개 묶음(SIMD 4칸) 경로 (손으로 짬, PhysX 5.6.1 과 비트 동일 목표).
// 분할에서 같은 종류 접촉 4개가 한 머리에 묶이면 PhysX 는 이 경로로 4개를 한꺼번에 준비·풀이한다. 칸마다 식 순서가
// 단일 경로와 다르므로(예: 내적을 z,y,x 순 MulAdd 로) 따로 옮긴다. 끝난 칸(hasFinished)의 쓰레기 값까지 원본과 같게 계산한다.
// 원본 (physx/source/lowleveldynamics/src/)
//   DyContactPrepShared.h:184-233      CorrelationListIterator
//   DyTGSContactPrepBlock.cpp:219-1227 setupFinalizeSolverConstraints4Step
//   DyTGSContactPrepBlock.cpp:1278-1633 computeBlockStreamByteSizes4 / reserveBlockStreams4 / createFinalizeSolverContacts4Step (두 판)
#pragma once
#include "contact_prep.h"

namespace eng {
namespace sv {

enum PrepState : uint32_t { PREP_OUT_OF_MEMORY = 0, PREP_UNBATCHABLE = 1, PREP_SUCCESS = 2 };

struct CorrelationListIterator {
  const CorrelationBuffer& buffer;
  uint32_t currPatch;
  uint32_t currContact;
  SV_HD CorrelationListIterator(const CorrelationBuffer& correlationBuffer, uint32_t startPatch) : buffer(correlationBuffer) {
    uint32_t newPatch = startPatch, newContact = 0;
    while (newPatch != CorrelationBuffer::LIST_END && newContact == buffer.contactPatches[newPatch].count) {
      newPatch = buffer.contactPatches[newPatch].next;
      newContact = 0;
    }
    currPatch = newPatch;
    currContact = newContact;
  }
  SV_HD bool hasNextContact() const { return (currPatch != CorrelationBuffer::LIST_END && currContact < buffer.contactPatches[currPatch].count); }
  SV_HD void nextContact(uint32_t& patch, uint32_t& contact) {
    patch = currPatch;
    contact = currContact;
    uint32_t newPatch = currPatch, newContact = currContact + 1;
    while (newPatch != CorrelationBuffer::LIST_END && newContact == buffer.contactPatches[newPatch].count) {
      newPatch = buffer.contactPatches[newPatch].next;
      newContact = 0;
    }
    currPatch = newPatch;
    currContact = newContact;
  }
};

// 몸체 3x3 행렬 칸 로드: V4LoadU(&col0.x) 는 W 로 다음 열의 x 를 읽지만 전치(Transpose44_34)에서 버려진다.
SV_HD V4 ldcol(const M33& m, int c) {
  const V3& v = c == 0 ? m.c0 : (c == 1 ? m.c1 : m.c2);
  const float w = c == 0 ? m.c1.x : (c == 1 ? m.c2.x : 0.0f);
  return V4{v.x, v.y, v.z, w};
}

SV_HDN void setupFinalizeSolverConstraints4Step(const PrepCtx& P, TGSContactDesc* descs, const CorrelationBuffer& c, uint8_t* workspace,
                                               float invDtF32, float totalDtF32, float invTotalDtF32, float dtF32, float bounceThresholdF32,
                                               float biasCoefficient, V4 invMassScale0, V4 invInertiaScale0, V4 invMassScale1,
                                               V4 invInertiaScale1) {
  const V4 solverOffsetSlop = V4LoadXYZW(descs[0].offsetSlop, descs[1].offsetSlop, descs[2].offsetSlop, descs[3].offsetSlop);
  const V4 zero = V4Zero();
  const V4 one = V4One();
  const BV bFalse = BFFFF();
  const BV bTrue = BTTTT();
  const FV fZero = FZero();
  uint8_t flags[4] = {uint8_t(descs[0].hasForceThresholds ? 1 : 0), uint8_t(descs[1].hasForceThresholds ? 1 : 0),
                      uint8_t(descs[2].hasForceThresholds ? 1 : 0), uint8_t(descs[3].hasForceThresholds ? 1 : 0)};
  const bool hasMaxImpulse = descs[0].hasMaxImpulse || descs[1].hasMaxImpulse || descs[2].hasMaxImpulse || descs[3].hasMaxImpulse;
  bool isDynamic = false;
  bool hasKinematic = false;
  float kinematicScale0F32[4];
  float kinematicScale1F32[4];
  for (uint32_t a = 0; a < 4; ++a) {
    isDynamic = isDynamic || (descs[a].bodyState1 == BS_DYNAMIC);
    hasKinematic = hasKinematic || descs[a].bodyState1 == BS_KINEMATIC;
    kinematicScale0F32[a] = P.vels[descs[a].b0].isKinematic ? 1.f : 0.f;
    kinematicScale1F32[a] = P.vels[descs[a].b1].isKinematic ? 1.f : 0.f;
  }
  const V4 kinematicScale0 = V4LoadU(kinematicScale0F32);
  const V4 kinematicScale1 = V4LoadU(kinematicScale1F32);
  const uint32_t constraintSize = sizeof(SolverContactPointStepBlock);
  const uint32_t frictionSize = sizeof(SolverContactFrictionStepBlock);
  uint8_t* ptr = workspace;
  const V4 dom0 = invMassScale0;
  const V4 dom1 = invMassScale1;
  const V4 angDom0 = invInertiaScale0;
  const V4 angDom1 = invInertiaScale1;
  const SBodyData* d0[4] = {&P.datas[descs[0].b0], &P.datas[descs[1].b0], &P.datas[descs[2].b0], &P.datas[descs[3].b0]};
  const SBodyData* d1[4] = {&P.datas[descs[0].b1], &P.datas[descs[1].b1], &P.datas[descs[2].b1], &P.datas[descs[3].b1]};
  const SBodyTxI* t0i[4] = {&P.txI[descs[0].b0], &P.txI[descs[1].b0], &P.txI[descs[2].b0], &P.txI[descs[3].b0]};
  const SBodyTxI* t1i[4] = {&P.txI[descs[0].b1], &P.txI[descs[1].b1], &P.txI[descs[2].b1], &P.txI[descs[3].b1]};
  const V4 maxPenBias = V4Max(V4LoadXYZW(d0[0]->penBiasClamp, d0[1]->penBiasClamp, d0[2]->penBiasClamp, d0[3]->penBiasClamp),
                              V4LoadXYZW(d1[0]->penBiasClamp, d1[1]->penBiasClamp, d1[2]->penBiasClamp, d1[3]->penBiasClamp));
  const V4 restDistance = V4LoadXYZW(descs[0].restDistance, descs[1].restDistance, descs[2].restDistance, descs[3].restDistance);
  // 속도 (V4LoadA: W 는 다음 필드, 전치에서 버려짐)
  V4 linVelT00, linVelT10, linVelT20, linVelT01, linVelT11, linVelT21;
  V4 angVelT00, angVelT10, angVelT20, angVelT01, angVelT11, angVelT21;
  Transpose44_34(ld4(d0[0]->originalLinearVelocity, d0[0]->maxContactImpulse), ld4(d0[1]->originalLinearVelocity, d0[1]->maxContactImpulse),
                 ld4(d0[2]->originalLinearVelocity, d0[2]->maxContactImpulse), ld4(d0[3]->originalLinearVelocity, d0[3]->maxContactImpulse),
                 linVelT00, linVelT10, linVelT20);
  Transpose44_34(ld4(d1[0]->originalLinearVelocity, d1[0]->maxContactImpulse), ld4(d1[1]->originalLinearVelocity, d1[1]->maxContactImpulse),
                 ld4(d1[2]->originalLinearVelocity, d1[2]->maxContactImpulse), ld4(d1[3]->originalLinearVelocity, d1[3]->maxContactImpulse),
                 linVelT01, linVelT11, linVelT21);
  Transpose44_34(ld4(d0[0]->originalAngularVelocity, d0[0]->penBiasClamp), ld4(d0[1]->originalAngularVelocity, d0[1]->penBiasClamp),
                 ld4(d0[2]->originalAngularVelocity, d0[2]->penBiasClamp), ld4(d0[3]->originalAngularVelocity, d0[3]->penBiasClamp),
                 angVelT00, angVelT10, angVelT20);
  Transpose44_34(ld4(d1[0]->originalAngularVelocity, d1[0]->penBiasClamp), ld4(d1[1]->originalAngularVelocity, d1[1]->penBiasClamp),
                 ld4(d1[2]->originalAngularVelocity, d1[2]->penBiasClamp), ld4(d1[3]->originalAngularVelocity, d1[3]->penBiasClamp),
                 angVelT01, angVelT11, angVelT21);
  const V4 vrelX = V4Sub(linVelT00, linVelT01);
  const V4 vrelY = V4Sub(linVelT10, linVelT11);
  const V4 vrelZ = V4Sub(linVelT20, linVelT21);
  const V4 invMass0 = V4LoadXYZW(d0[0]->invMass, d0[1]->invMass, d0[2]->invMass, d0[3]->invMass);
  const V4 invMass1 = V4LoadXYZW(d1[0]->invMass, d1[1]->invMass, d1[2]->invMass, d1[3]->invMass);
  const V4 invMass0D0 = V4Mul(dom0, invMass0);
  const V4 invMass1D1 = V4Mul(dom1, invMass1);
  V4 invInertia0X0, invInertia0X1, invInertia0X2, invInertia0Y0, invInertia0Y1, invInertia0Y2, invInertia0Z0, invInertia0Z1, invInertia0Z2;
  V4 invInertia1X0, invInertia1X1, invInertia1X2, invInertia1Y0, invInertia1Y1, invInertia1Y2, invInertia1Z0, invInertia1Z1, invInertia1Z2;
  // invInertia{a}{b}{X|Y|Z} = desc a 몸체 b 의 열 0/1/2. 전치: X열 4개 -> (invInertia0X0, 0Y0, 0Z0) 등 (DyTGSContactPrepBlock.cpp:366)
  Transpose44_34(ldcol(t0i[0]->sqrtInvInertia, 0), ldcol(t0i[1]->sqrtInvInertia, 0), ldcol(t0i[2]->sqrtInvInertia, 0),
                 ldcol(t0i[3]->sqrtInvInertia, 0), invInertia0X0, invInertia0Y0, invInertia0Z0);
  Transpose44_34(ldcol(t0i[0]->sqrtInvInertia, 1), ldcol(t0i[1]->sqrtInvInertia, 1), ldcol(t0i[2]->sqrtInvInertia, 1),
                 ldcol(t0i[3]->sqrtInvInertia, 1), invInertia0X1, invInertia0Y1, invInertia0Z1);
  Transpose44_34(ldcol(t0i[0]->sqrtInvInertia, 2), ldcol(t0i[1]->sqrtInvInertia, 2), ldcol(t0i[2]->sqrtInvInertia, 2),
                 ldcol(t0i[3]->sqrtInvInertia, 2), invInertia0X2, invInertia0Y2, invInertia0Z2);
  Transpose44_34(ldcol(t1i[0]->sqrtInvInertia, 0), ldcol(t1i[1]->sqrtInvInertia, 0), ldcol(t1i[2]->sqrtInvInertia, 0),
                 ldcol(t1i[3]->sqrtInvInertia, 0), invInertia1X0, invInertia1Y0, invInertia1Z0);
  Transpose44_34(ldcol(t1i[0]->sqrtInvInertia, 1), ldcol(t1i[1]->sqrtInvInertia, 1), ldcol(t1i[2]->sqrtInvInertia, 1),
                 ldcol(t1i[3]->sqrtInvInertia, 1), invInertia1X1, invInertia1Y1, invInertia1Z1);
  Transpose44_34(ldcol(t1i[0]->sqrtInvInertia, 2), ldcol(t1i[1]->sqrtInvInertia, 2), ldcol(t1i[2]->sqrtInvInertia, 2),
                 ldcol(t1i[3]->sqrtInvInertia, 2), invInertia1X2, invInertia1Y2, invInertia1Z2);
  const FV invDt = FLoad(invDtF32);
  const float scale = pmin(0.8f, biasCoefficient);
  const FV p8 = FLoad(scale);
  const FV frictionBiasScale = FMul(invDt, p8);
  const V4 totalDt = V4Load(totalDtF32);
  const FV invTotalDt = FLoad(invTotalDtF32);
  const V4 p84 = V4Splat(p8);
  const V4 bounceThreshold = V4Splat(FLoad(bounceThresholdF32));
  const V4 invDtp8 = V4Splat(FMul(invDt, p8));
  const FV dt = FLoad(dtF32);
  V4 bodyFrame0pX, bodyFrame0pY, bodyFrame0pZ, bodyFrame1pX, bodyFrame1pY, bodyFrame1pZ;
  Transpose44_34(ldv(descs[0].bodyFrame0.p), ldv(descs[1].bodyFrame0.p), ldv(descs[2].bodyFrame0.p), ldv(descs[3].bodyFrame0.p), bodyFrame0pX,
                 bodyFrame0pY, bodyFrame0pZ);
  Transpose44_34(ldv(descs[0].bodyFrame1.p), ldv(descs[1].bodyFrame1.p), ldv(descs[2].bodyFrame1.p), ldv(descs[3].bodyFrame1.p), bodyFrame1pX,
                 bodyFrame1pY, bodyFrame1pZ);
  V4 bodyFrame0q[4], bodyFrame1q[4];
  for (int a = 0; a < 4; ++a) {
    bodyFrame0q[a] = V4{descs[a].bodyFrame0.q.x, descs[a].bodyFrame0.q.y, descs[a].bodyFrame0.q.z, descs[a].bodyFrame0.q.w};
    bodyFrame1q[a] = V4{descs[a].bodyFrame1.q.x, descs[a].bodyFrame1.q.y, descs[a].bodyFrame1.q.z, descs[a].bodyFrame1.q.w};
  }
  uint32_t frictionPatchWritebackAddrIndex[4] = {0, 0, 0, 0};
  uint32_t frictionIndex[4] = {0, 0, 0, 0};
  const uint32_t maxPatches = pmaxu(descs[0].numFrictionPatches, pmaxu(descs[1].numFrictionPatches, pmaxu(descs[2].numFrictionPatches, descs[3].numFrictionPatches)));
  const V4 p1 = V4Splat(FLoad(0.0001f));
  const V4 orthoThreshold = V4Splat(FLoad(0.70710678f));
  uint32_t contact[4] = {0, 0, 0, 0};
  uint32_t patch[4] = {0, 0, 0, 0};
  uint8_t flag = 0;
  if (hasMaxImpulse) flag |= SolverContactHeaderStepBlock::eHAS_MAX_IMPULSE;
  bool hasFinished[4];
  for (uint32_t i = 0; i < maxPatches; i++) {
    uint32_t clampedContacts[4], firstPatch[4];
    const ContactPoint* contactBase[4];
    for (int a = 0; a < 4; ++a) {
      hasFinished[a] = i >= descs[a].numFrictionPatches;
      frictionIndex[a] = hasFinished[a] ? frictionIndex[a] : descs[a].startFrictionPatchIndex + i;
      clampedContacts[a] = hasFinished[a] ? 0 : c.frictionPatchContactCounts[frictionIndex[a]];
    }
    for (int a = 0; a < 4; ++a) firstPatch[a] = c.correlationListHeads[frictionIndex[a]];
    for (int a = 0; a < 4; ++a) contactBase[a] = descs[a].contacts + c.contactPatches[firstPatch[a]].start;
    const V4 restitution = V4Neg(V4LoadXYZW(contactBase[0]->restitution, contactBase[1]->restitution, contactBase[2]->restitution,
                                            contactBase[3]->restitution));
    const V4 damping = V4LoadXYZW(contactBase[0]->damping, contactBase[1]->damping, contactBase[2]->damping, contactBase[3]->damping);
    const bool accelSpring_[] = {!!(contactBase[0]->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING),
                                 !!(contactBase[1]->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING),
                                 !!(contactBase[2]->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING),
                                 !!(contactBase[3]->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING)};
    const BV accelSpring = BLoad(accelSpring_);
    SolverContactHeaderStepBlock* header = reinterpret_cast<SolverContactHeaderStepBlock*>(ptr);
    ptr += sizeof(SolverContactHeaderStepBlock);
    header->flags[0] = flags[0];
    header->flags[1] = flags[1];
    header->flags[2] = flags[2];
    header->flags[3] = flags[3];
    header->flag = flag;
    const uint32_t totalContacts = pmaxu(clampedContacts[0], pmaxu(clampedContacts[1], pmaxu(clampedContacts[2], clampedContacts[3])));
    V4* appliedNormalForces = reinterpret_cast<V4*>(ptr);
    ptr += sizeof(V4) * totalContacts;
    for (uint32_t k = 0; k < totalContacts; ++k) appliedNormalForces[k] = V4Zero();
    header->numNormalConstr = uint8_t(totalContacts);
    for (int a = 0; a < 4; ++a) header->numNormalConstrs[a] = uint8_t(clampedContacts[a]);
    header->invMass0D0 = invMass0D0;
    header->invMass1D1 = invMass1D1;
    header->angDom0 = angDom0;
    header->angDom1 = angDom1;
    V4* maxImpulse = reinterpret_cast<V4*>(ptr + constraintSize * totalContacts);
    V4 normalX, normalY, normalZ;
    Transpose44_34(ld4(contactBase[0]->normal, contactBase[0]->separation), ld4(contactBase[1]->normal, contactBase[1]->separation),
                   ld4(contactBase[2]->normal, contactBase[2]->separation), ld4(contactBase[3]->normal, contactBase[3]->separation), normalX,
                   normalY, normalZ);
    header->normalX = normalX;
    header->normalY = normalY;
    header->normalZ = normalZ;
    header->maxPenBias = maxPenBias;
    const V4 norVel0 = V4MulAdd(normalZ, linVelT20, V4MulAdd(normalY, linVelT10, V4Mul(normalX, linVelT00)));
    const V4 norVel1 = V4MulAdd(normalZ, linVelT21, V4MulAdd(normalY, linVelT11, V4Mul(normalX, linVelT01)));
    const V4 relNorVel = V4Sub(norVel0, norVel1);
    uint32_t finished = (uint32_t(hasFinished[0])) | ((uint32_t(hasFinished[1])) << 1) | ((uint32_t(hasFinished[2])) << 2) |
                        ((uint32_t(hasFinished[3])) << 3);
    CorrelationListIterator iter0(c, firstPatch[0]);
    CorrelationListIterator iter1(c, firstPatch[1]);
    CorrelationListIterator iter2(c, firstPatch[2]);
    CorrelationListIterator iter3(c, firstPatch[3]);
    if (!hasFinished[0]) iter0.nextContact(patch[0], contact[0]);
    if (!hasFinished[1]) iter1.nextContact(patch[1], contact[1]);
    if (!hasFinished[2]) iter2.nextContact(patch[2], contact[2]);
    if (!hasFinished[3]) iter3.nextContact(patch[3], contact[3]);
    uint8_t* p = ptr;
    uint32_t contactCount = 0;
    uint32_t newFinished = (uint32_t(hasFinished[0] || !iter0.hasNextContact())) | ((uint32_t(hasFinished[1] || !iter1.hasNextContact())) << 1) |
                           ((uint32_t(hasFinished[2] || !iter2.hasNextContact())) << 2) |
                           ((uint32_t(hasFinished[3] || !iter3.hasNextContact())) << 3);
    BV bFinished = BLoad(hasFinished);
    while (finished != 0xf) {
      finished = newFinished;
      ++contactCount;
      SolverContactPointStepBlock* solverContact = reinterpret_cast<SolverContactPointStepBlock*>(p);
      p += constraintSize;
      const ContactPoint& con0 = descs[0].contacts[c.contactPatches[patch[0]].start + contact[0]];
      const ContactPoint& con1 = descs[1].contacts[c.contactPatches[patch[1]].start + contact[1]];
      const ContactPoint& con2 = descs[2].contacts[c.contactPatches[patch[2]].start + contact[2]];
      const ContactPoint& con3 = descs[3].contacts[c.contactPatches[patch[3]].start + contact[3]];
      {
        V4 pointX, pointY, pointZ;
        Transpose44_34(ld4(con0.point, con0.maxImpulse), ld4(con1.point, con1.maxImpulse), ld4(con2.point, con2.maxImpulse),
                       ld4(con3.point, con3.maxImpulse), pointX, pointY, pointZ);
        V4 cTargetVelX, cTargetVelY, cTargetVelZ;
        Transpose44_34(ld4(con0.targetVel, con0.staticFriction), ld4(con1.targetVel, con1.staticFriction), ld4(con2.targetVel, con2.staticFriction),
                       ld4(con3.targetVel, con3.staticFriction), cTargetVelX, cTargetVelY, cTargetVelZ);
        const V4 separation = V4LoadXYZW(con0.separation, con1.separation, con2.separation, con3.separation);
        const V4 cTargetNorVel = V4MulAdd(cTargetVelX, normalX, V4MulAdd(cTargetVelY, normalY, V4Mul(cTargetVelZ, normalZ)));
        const V4 raX = V4Sub(pointX, bodyFrame0pX);
        const V4 raY = V4Sub(pointY, bodyFrame0pY);
        const V4 raZ = V4Sub(pointZ, bodyFrame0pZ);
        const V4 rbX = V4Sub(pointX, bodyFrame1pX);
        const V4 rbY = V4Sub(pointY, bodyFrame1pY);
        const V4 rbZ = V4Sub(pointZ, bodyFrame1pZ);
        V4 raXnX = V4NegMulSub(raZ, normalY, V4Mul(raY, normalZ));
        V4 raXnY = V4NegMulSub(raX, normalZ, V4Mul(raZ, normalX));
        V4 raXnZ = V4NegMulSub(raY, normalX, V4Mul(raX, normalY));
        V4 rbXnX = V4NegMulSub(rbZ, normalY, V4Mul(rbY, normalZ));
        V4 rbXnY = V4NegMulSub(rbX, normalZ, V4Mul(rbZ, normalX));
        V4 rbXnZ = V4NegMulSub(rbY, normalX, V4Mul(rbX, normalY));
        const V4 relAngVel0 = V4MulAdd(raXnZ, angVelT20, V4MulAdd(raXnY, angVelT10, V4Mul(raXnX, angVelT00)));
        const V4 relAngVel1 = V4MulAdd(rbXnZ, angVelT21, V4MulAdd(rbXnY, angVelT11, V4Mul(rbXnX, angVelT01)));
        const V4 relAng = V4Sub(relAngVel0, relAngVel1);
        const V4 slop = V4Mul(solverOffsetSlop, V4Max(V4Sel(V4IsEq(relNorVel, zero), V4Splat(FMax()), V4Div(relAng, relNorVel)), V4One()));
        raXnX = V4Sel(V4IsGrtr(slop, V4Abs(raXnX)), zero, raXnX);
        raXnY = V4Sel(V4IsGrtr(slop, V4Abs(raXnY)), zero, raXnY);
        raXnZ = V4Sel(V4IsGrtr(slop, V4Abs(raXnZ)), zero, raXnZ);
        V4 delAngVel0X = V4Mul(invInertia0X0, raXnX);
        V4 delAngVel0Y = V4Mul(invInertia0X1, raXnX);
        V4 delAngVel0Z = V4Mul(invInertia0X2, raXnX);
        delAngVel0X = V4MulAdd(invInertia0Y0, raXnY, delAngVel0X);
        delAngVel0Y = V4MulAdd(invInertia0Y1, raXnY, delAngVel0Y);
        delAngVel0Z = V4MulAdd(invInertia0Y2, raXnY, delAngVel0Z);
        delAngVel0X = V4MulAdd(invInertia0Z0, raXnZ, delAngVel0X);
        delAngVel0Y = V4MulAdd(invInertia0Z1, raXnZ, delAngVel0Y);
        delAngVel0Z = V4MulAdd(invInertia0Z2, raXnZ, delAngVel0Z);
        const V4 dotDelAngVel0 = V4MulAdd(delAngVel0X, delAngVel0X, V4MulAdd(delAngVel0Y, delAngVel0Y, V4Mul(delAngVel0Z, delAngVel0Z)));
        V4 unitResponse = V4MulAdd(dotDelAngVel0, angDom0, invMass0D0);
        V4 vrel0 = V4Add(norVel0, relAngVel0);
        V4 vrel1 = V4Add(norVel1, relAngVel1);
        V4 delAngVel1X = zero, delAngVel1Y = zero, delAngVel1Z = zero;
        if (isDynamic) {
          rbXnX = V4Sel(V4IsGrtr(slop, V4Abs(rbXnX)), zero, rbXnX);
          rbXnY = V4Sel(V4IsGrtr(slop, V4Abs(rbXnY)), zero, rbXnY);
          rbXnZ = V4Sel(V4IsGrtr(slop, V4Abs(rbXnZ)), zero, rbXnZ);
          delAngVel1X = V4Mul(invInertia1X0, rbXnX);
          delAngVel1Y = V4Mul(invInertia1X1, rbXnX);
          delAngVel1Z = V4Mul(invInertia1X2, rbXnX);
          delAngVel1X = V4MulAdd(invInertia1Y0, rbXnY, delAngVel1X);
          delAngVel1Y = V4MulAdd(invInertia1Y1, rbXnY, delAngVel1Y);
          delAngVel1Z = V4MulAdd(invInertia1Y2, rbXnY, delAngVel1Z);
          delAngVel1X = V4MulAdd(invInertia1Z0, rbXnZ, delAngVel1X);
          delAngVel1Y = V4MulAdd(invInertia1Z1, rbXnZ, delAngVel1Y);
          delAngVel1Z = V4MulAdd(invInertia1Z2, rbXnZ, delAngVel1Z);
          const V4 dotDelAngVel1 = V4MulAdd(delAngVel1X, delAngVel1X, V4MulAdd(delAngVel1Y, delAngVel1Y, V4Mul(delAngVel1Z, delAngVel1Z)));
          const V4 resp1 = V4MulAdd(dotDelAngVel1, angDom1, invMass1D1);
          unitResponse = V4Add(unitResponse, resp1);
        }
        V4 vrel = V4Sub(vrel0, vrel1);
        solverContact->rbXnI[0] = delAngVel1X;
        solverContact->rbXnI[1] = delAngVel1Y;
        solverContact->rbXnI[2] = delAngVel1Z;
        V4 penetration = V4Sub(separation, restDistance);
        const V4 penetrationInvDt = V4Scale(penetration, invTotalDt);
        const BV isSep = V4IsGrtr(penetration, zero);
        const BV isGreater2 = BAnd(BAnd(V4IsGrtr(zero, restitution), V4IsGrtr(bounceThreshold, vrel)), V4IsGrtr(V4Neg(vrel), penetrationInvDt));
        const V4 ratio = V4Sel(isGreater2, V4Add(totalDt, V4Div(penetration, vrel)), zero);
        const V4 recipResponse = V4Sel(V4IsGrtr(unitResponse, zero), V4Recip(unitResponse), zero);
        const BV isCompliant = V4IsGrtr(restitution, zero);
        const V4 rdt = V4Scale(restitution, dt);
        const BV collidingWithVrel = V4IsGrtr(V4Neg(vrel), penetrationInvDt);
        const V4 dampingIfEnabled = V4Sel(BAndNot(isSep, collidingWithVrel), zero, damping);
        const V4 a = V4Scale(V4Add(dampingIfEnabled, rdt), dt);
        const V4 massIfAccelElseOne = V4Sel(accelSpring, recipResponse, one);
        const V4 oneIfAccelElseR = V4Sel(accelSpring, one, unitResponse);
        const V4 x = V4Recip(V4MulAdd(a, oneIfAccelElseR, one));
        const V4 velMultiplier = V4Sel(isCompliant, V4Mul(V4Mul(x, a), massIfAccelElseOne), recipResponse);
        const V4 scaledBias = V4Neg(V4Sel(isCompliant, V4Mul(rdt, V4Mul(x, oneIfAccelElseR)), V4Sel(isSep, V4Splat(invDt), invDtp8)));
        V4 targetVelocity = V4NegMulSub(vrel0, kinematicScale0, V4MulAdd(vrel1, kinematicScale1, V4Sel(isGreater2, V4Mul(vrel, restitution), zero)));
        penetration = V4MulAdd(targetVelocity, ratio, penetration);
        solverContact->raXnI[0] = delAngVel0X;
        solverContact->raXnI[1] = delAngVel0Y;
        solverContact->raXnI[2] = delAngVel0Z;
        solverContact->velMultiplier = V4Sel(bFinished, zero, velMultiplier);
        solverContact->targetVelocity = V4Add(cTargetNorVel, targetVelocity);
        solverContact->separation = penetration;
        solverContact->biasCoefficient = V4Sel(bFinished, zero, scaledBias);
        solverContact->recipResponse = V4Sel(bFinished, zero, recipResponse);
        if (hasMaxImpulse)
          maxImpulse[contactCount - 1] = V4Merge(FLoad(con0.maxImpulse), FLoad(con1.maxImpulse), FLoad(con2.maxImpulse), FLoad(con3.maxImpulse));
      }
      if (!(finished & 0x1)) {
        iter0.nextContact(patch[0], contact[0]);
        newFinished |= uint32_t(!iter0.hasNextContact());
      } else
        bFinished = BSetX(bFinished, bTrue);
      if (!(finished & 0x2)) {
        iter1.nextContact(patch[1], contact[1]);
        newFinished |= (uint32_t(!iter1.hasNextContact()) << 1);
      } else
        bFinished = BSetY(bFinished, bTrue);
      if (!(finished & 0x4)) {
        iter2.nextContact(patch[2], contact[2]);
        newFinished |= (uint32_t(!iter2.hasNextContact()) << 2);
      } else
        bFinished = BSetZ(bFinished, bTrue);
      if (!(finished & 0x8)) {
        iter3.nextContact(patch[3], contact[3]);
        newFinished |= (uint32_t(!iter3.hasNextContact()) << 3);
      } else
        bFinished = BSetW(bFinished, bTrue);
    }
    ptr = p;
    if (hasMaxImpulse) ptr += sizeof(V4) * totalContacts;

    // ---- 마찰
    V4 maxImpulseScale = V4One();
    {
      const FrictionPatch* fpp[4] = {&c.frictionPatches[frictionIndex[0]], &c.frictionPatches[frictionIndex[1]], &c.frictionPatches[frictionIndex[2]],
                                     &c.frictionPatches[frictionIndex[3]]};
      uint32_t clampedAnchorCount[4];
      for (int a = 0; a < 4; ++a)
        clampedAnchorCount[a] = hasFinished[a] || (contactBase[a]->materialFlags & MAT_DISABLE_FRICTION) ? 0 : fpp[a]->anchorCount;
      const uint32_t maxAnchorCount =
          pmaxu(clampedAnchorCount[0], pmaxu(clampedAnchorCount[1], pmaxu(clampedAnchorCount[2], clampedAnchorCount[3])));
      float staticFriction[4];
      float dynamicFriction[4];
      for (int a = 0; a < 4; ++a) {
        const float coeff = (clampedAnchorCount[a] == 2) ? 0.5f : 1.f;
        staticFriction[a] = contactBase[a]->staticFriction * coeff;
        dynamicFriction[a] = contactBase[a]->dynamicFriction * coeff;
      }
      header->numFrictionConstr = uint8_t(maxAnchorCount * 2);
      for (int a = 0; a < 4; ++a) header->numFrictionConstrs[a] = uint8_t(clampedAnchorCount[a] * 2);
      header->type = SC_TYPE_BLOCK_RB_CONTACT;
      if (maxAnchorCount) {
        const BV cond = V4IsGrtr(orthoThreshold, V4Abs(normalX));
        const V4 t0FallbackX = V4Sel(cond, zero, V4Neg(normalY));
        const V4 t0FallbackY = V4Sel(cond, V4Neg(normalZ), normalX);
        const V4 t0FallbackZ = V4Sel(cond, normalY, zero);
        const V4 vrelSubNorVelX = V4NegMulSub(normalX, relNorVel, vrelX);
        const V4 vrelSubNorVelY = V4NegMulSub(normalY, relNorVel, vrelY);
        const V4 vrelSubNorVelZ = V4NegMulSub(normalZ, relNorVel, vrelZ);
        const V4 lenSqvrelSubNorVelZ =
            V4MulAdd(vrelSubNorVelX, vrelSubNorVelX, V4MulAdd(vrelSubNorVelY, vrelSubNorVelY, V4Mul(vrelSubNorVelZ, vrelSubNorVelZ)));
        const BV bcon2 = V4IsGrtr(lenSqvrelSubNorVelZ, p1);
        V4 t0X = V4Sel(bcon2, vrelSubNorVelX, t0FallbackX);
        V4 t0Y = V4Sel(bcon2, vrelSubNorVelY, t0FallbackY);
        V4 t0Z = V4Sel(bcon2, vrelSubNorVelZ, t0FallbackZ);
        const V4 recipLen = V4Rsqrt(V4MulAdd(t0Z, t0Z, V4MulAdd(t0Y, t0Y, V4Mul(t0X, t0X))));
        t0X = V4Mul(t0X, recipLen);
        t0Y = V4Mul(t0Y, recipLen);
        t0Z = V4Mul(t0Z, recipLen);
        V4 t1X = V4NegMulSub(normalZ, t0Y, V4Mul(normalY, t0Z));
        V4 t1Y = V4NegMulSub(normalX, t0Z, V4Mul(normalZ, t0X));
        V4 t1Z = V4NegMulSub(normalY, t0X, V4Mul(normalX, t0Y));
        uint32_t index[4] = {0, 0, 0, 0};
        header->broken = bFalse;
        for (int a = 0; a < 4; ++a)
          header->frictionBrokenWriteback[a] = descs[a].frictionPtr == NONE ? NONE : descs[a].frictionPtr + frictionPatchWritebackAddrIndex[a];
        V4* appliedForces = reinterpret_cast<V4*>(ptr);
        ptr += sizeof(V4) * header->numFrictionConstr;
        for (uint32_t k = 0; k < header->numFrictionConstr; ++k) appliedForces[k] = V4Zero();
        for (uint32_t j = 0; j < maxAnchorCount; j++) {
          SolverContactFrictionStepBlock* f0 = reinterpret_cast<SolverContactFrictionStepBlock*>(ptr);
          ptr += frictionSize;
          SolverContactFrictionStepBlock* f1 = reinterpret_cast<SolverContactFrictionStepBlock*>(ptr);
          ptr += frictionSize;
          for (int a = 0; a < 4; ++a) index[a] = j < clampedAnchorCount[a] ? j : index[a];
          if (j >= clampedAnchorCount[0]) maxImpulseScale = V4SetX(maxImpulseScale, fZero);
          if (j >= clampedAnchorCount[1]) maxImpulseScale = V4SetY(maxImpulseScale, fZero);
          if (j >= clampedAnchorCount[2]) maxImpulseScale = V4SetZ(maxImpulseScale, fZero);
          if (j >= clampedAnchorCount[3]) maxImpulseScale = V4SetW(maxImpulseScale, fZero);
          t0X = V4Mul(maxImpulseScale, t0X);
          t0Y = V4Mul(maxImpulseScale, t0Y);
          t0Z = V4Mul(maxImpulseScale, t0Z);
          t1X = V4Mul(maxImpulseScale, t1X);
          t1Y = V4Mul(maxImpulseScale, t1Y);
          t1Z = V4Mul(maxImpulseScale, t1Z);
          // V4LoadU(&body0Anchors[index].x): W 는 다음 float (전치에서 버려지지만 QuatRotate4V 의 W 칸에만 영향 -> 버려짐)
          V4 ra[4], rb[4];
          for (int a = 0; a < 4; ++a) {
            const FrictionPatch& fp = *fpp[a];
            const float wa = index[a] == 0 ? fp.body0Anchors[1].x : fp.body1Anchors[0].x;
            const float wb = index[a] == 0 ? fp.body1Anchors[1].x : fp.relativeQuat.x;
            ra[a] = QuatRotate4V(bodyFrame0q[a], ld4(fp.body0Anchors[index[a]], wa));
            rb[a] = QuatRotate4V(bodyFrame1q[a], ld4(fp.body1Anchors[index[a]], wb));
          }
          V4 raX, raY, raZ, rbX, rbY, rbZ;
          Transpose44_34(ra[0], ra[1], ra[2], ra[3], raX, raY, raZ);
          const V4 raWorldX = V4Add(raX, bodyFrame0pX);
          const V4 raWorldY = V4Add(raY, bodyFrame0pY);
          const V4 raWorldZ = V4Add(raZ, bodyFrame0pZ);
          Transpose44_34(rb[0], rb[1], rb[2], rb[3], rbX, rbY, rbZ);
          const V4 rbWorldX = V4Add(rbX, bodyFrame1pX);
          const V4 rbWorldY = V4Add(rbY, bodyFrame1pY);
          const V4 rbWorldZ = V4Add(rbZ, bodyFrame1pZ);
          const V4 errorX = V4Sub(raWorldX, rbWorldX);
          const V4 errorY = V4Sub(raWorldY, rbWorldY);
          const V4 errorZ = V4Sub(raWorldZ, rbWorldZ);
          V4 tv[4];
          for (int a = 0; a < 4; ++a) {
            const uint32_t contactIndex = c.contactID[frictionIndex[a]][index[a]];
            // 원본은 0xFFFF 일 때 네 칸 모두 contactBase0 (첫 desc) 의 targetVel 을 읽는다 (DyTGSContactPrepBlock.cpp:1016-1019)
            const ContactPoint& tc = contactIndex == 0xFFFF ? *contactBase[0] : descs[a].contacts[contactIndex];
            tv[a] = ld4(tc.targetVel, tc.staticFriction);
          }
          V4 targetVelX, targetVelY, targetVelZ;
          Transpose44_34(tv[0], tv[1], tv[2], tv[3], targetVelX, targetVelY, targetVelZ);
          for (int dir = 0; dir < 2; ++dir) {
            const V4 tX = dir == 0 ? t0X : t1X;
            const V4 tY = dir == 0 ? t0Y : t1Y;
            const V4 tZ = dir == 0 ? t0Z : t1Z;
            SolverContactFrictionStepBlock* f = dir == 0 ? f0 : f1;
            V4 raXnX = V4NegMulSub(raZ, tY, V4Mul(raY, tZ));
            V4 raXnY = V4NegMulSub(raX, tZ, V4Mul(raZ, tX));
            V4 raXnZ = V4NegMulSub(raY, tX, V4Mul(raX, tY));
            raXnX = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(raXnX)), zero, raXnX);
            raXnY = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(raXnY)), zero, raXnY);
            raXnZ = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(raXnZ)), zero, raXnZ);
            V4 delAngVel0X = V4Mul(invInertia0X0, raXnX);
            V4 delAngVel0Y = V4Mul(invInertia0X1, raXnX);
            V4 delAngVel0Z = V4Mul(invInertia0X2, raXnX);
            delAngVel0X = V4MulAdd(invInertia0Y0, raXnY, delAngVel0X);
            delAngVel0Y = V4MulAdd(invInertia0Y1, raXnY, delAngVel0Y);
            delAngVel0Z = V4MulAdd(invInertia0Y2, raXnY, delAngVel0Z);
            delAngVel0X = V4MulAdd(invInertia0Z0, raXnZ, delAngVel0X);
            delAngVel0Y = V4MulAdd(invInertia0Z1, raXnZ, delAngVel0Y);
            delAngVel0Z = V4MulAdd(invInertia0Z2, raXnZ, delAngVel0Z);
            const V4 dotDelAngVel0 = V4MulAdd(delAngVel0Z, delAngVel0Z, V4MulAdd(delAngVel0Y, delAngVel0Y, V4Mul(delAngVel0X, delAngVel0X)));
            V4 resp = V4MulAdd(dotDelAngVel0, angDom0, invMass0D0);
            const V4 tVel0 = V4MulAdd(tZ, linVelT20, V4MulAdd(tY, linVelT10, V4Mul(tX, linVelT00)));
            V4 vr0 = V4MulAdd(raXnZ, angVelT20, V4MulAdd(raXnY, angVelT10, V4MulAdd(raXnX, angVelT00, tVel0)));
            V4 delAngVel1X = zero, delAngVel1Y = zero, delAngVel1Z = zero;
            V4 vr1 = zero;
            if (isDynamic) {
              V4 rbXnX = V4NegMulSub(rbZ, tY, V4Mul(rbY, tZ));
              V4 rbXnY = V4NegMulSub(rbX, tZ, V4Mul(rbZ, tX));
              V4 rbXnZ = V4NegMulSub(rbY, tX, V4Mul(rbX, tY));
              rbXnX = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(rbXnX)), zero, rbXnX);
              rbXnY = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(rbXnY)), zero, rbXnY);
              rbXnZ = V4Sel(V4IsGrtr(solverOffsetSlop, V4Abs(rbXnZ)), zero, rbXnZ);
              delAngVel1X = V4Mul(invInertia1X0, rbXnX);
              delAngVel1Y = V4Mul(invInertia1X1, rbXnX);
              delAngVel1Z = V4Mul(invInertia1X2, rbXnX);
              delAngVel1X = V4MulAdd(invInertia1Y0, rbXnY, delAngVel1X);
              delAngVel1Y = V4MulAdd(invInertia1Y1, rbXnY, delAngVel1Y);
              delAngVel1Z = V4MulAdd(invInertia1Y2, rbXnY, delAngVel1Z);
              delAngVel1X = V4MulAdd(invInertia1Z0, rbXnZ, delAngVel1X);
              delAngVel1Y = V4MulAdd(invInertia1Z1, rbXnZ, delAngVel1Y);
              delAngVel1Z = V4MulAdd(invInertia1Z2, rbXnZ, delAngVel1Z);
              const V4 dotDelAngVel1 = V4MulAdd(delAngVel1Z, delAngVel1Z, V4MulAdd(delAngVel1Y, delAngVel1Y, V4Mul(delAngVel1X, delAngVel1X)));
              const V4 resp1 = V4MulAdd(dotDelAngVel1, angDom1, invMass1D1);
              resp = V4Add(resp, resp1);
              const V4 tVel1 = V4MulAdd(tZ, linVelT21, V4MulAdd(tY, linVelT11, V4Mul(tX, linVelT01)));
              vr1 = V4MulAdd(rbXnZ, angVelT21, V4MulAdd(rbXnY, angVelT11, V4MulAdd(rbXnX, angVelT01, tVel1)));
            } else if (hasKinematic) {
              const V4 rbXnX = V4NegMulSub(rbZ, tY, V4Mul(rbY, tZ));
              const V4 rbXnY = V4NegMulSub(rbX, tZ, V4Mul(rbZ, tX));
              const V4 rbXnZ = V4NegMulSub(rbY, tX, V4Mul(rbX, tY));
              const V4 tVel1 = V4MulAdd(tZ, linVelT21, V4MulAdd(tY, linVelT11, V4Mul(tX, linVelT01)));
              vr1 = V4MulAdd(rbXnZ, angVelT21, V4MulAdd(rbXnY, angVelT11, V4MulAdd(rbXnX, angVelT01, tVel1)));
            }
            f->rbXnI[0] = delAngVel1X;
            f->rbXnI[1] = delAngVel1Y;
            f->rbXnI[2] = delAngVel1Z;
            const V4 velMultiplier = V4Mul(maxImpulseScale, V4Sel(V4IsGrtr(resp, zero), V4Div(p84, resp), zero));
            const V4 error = V4MulAdd(tZ, errorZ, V4MulAdd(tY, errorY, V4Mul(tX, errorX)));
            const V4 targetVel =
                V4NegMulSub(vr0, kinematicScale0, V4MulAdd(vr1, kinematicScale1, V4MulAdd(tZ, targetVelZ, V4MulAdd(tY, targetVelY, V4Mul(tX, targetVelX)))));
            f->normal[0] = tX;
            f->normal[1] = tY;
            f->normal[2] = tZ;
            f->raXnI[0] = delAngVel0X;
            f->raXnI[1] = delAngVel0Y;
            f->raXnI[2] = delAngVel0Z;
            f->error = error;
            f->velMultiplier = velMultiplier;
            f->biasCoefficient = V4Splat(frictionBiasScale);
            f->targetVel = targetVel;
          }
        }
        header->dynamicFriction = V4LoadA(dynamicFriction);
        header->staticFriction = V4LoadA(staticFriction);
        for (int a = 0; a < 4; ++a) frictionPatchWritebackAddrIndex[a]++;
      }
    }
  }
}

// DyTGSContactPrepBlock.cpp:1278
SV_HDN void computeBlockStreamByteSizes4(TGSContactDesc* descs, uint32_t& _solverConstraintByteSize, uint32_t* _axisConstraintCount,
                                        const CorrelationBuffer& c) {
  uint32_t maxPatches = 0;
  uint32_t maxContactCount[CorrelationBuffer::MAX_FRICTION_PATCHES];
  uint32_t maxFrictionCount[CorrelationBuffer::MAX_FRICTION_PATCHES];
  for (uint32_t k = 0; k < CorrelationBuffer::MAX_FRICTION_PATCHES; ++k) maxContactCount[k] = maxFrictionCount[k] = 0;
  bool hasMaxImpulse = false;
  for (uint32_t a = 0; a < 4; ++a) {
    uint32_t axisConstraintCount = 0;
    hasMaxImpulse = hasMaxImpulse || descs[a].hasMaxImpulse;
    for (uint32_t i = 0; i < descs[a].numFrictionPatches; i++) {
      const uint32_t ind = i + descs[a].startFrictionPatchIndex;
      const FrictionPatch& frictionPatch = c.frictionPatches[ind];
      const bool haveFriction = (frictionPatch.materialFlags & MAT_DISABLE_FRICTION) == 0 && frictionPatch.anchorCount != 0;
      if (c.frictionPatchContactCounts[ind] != 0) {
        maxContactCount[i] = pmaxu(c.frictionPatchContactCounts[ind], maxContactCount[i]);
        axisConstraintCount += c.frictionPatchContactCounts[ind];
        if (haveFriction) {
          const uint32_t fricCount = uint32_t(c.frictionPatches[ind].anchorCount) * 2;
          maxFrictionCount[i] = pmaxu(fricCount, maxFrictionCount[i]);
          axisConstraintCount += fricCount;
        }
      }
    }
    maxPatches = pmaxu(descs[a].numFrictionPatches, maxPatches);
    _axisConstraintCount[a] = axisConstraintCount;
  }
  uint32_t totalContacts = 0, totalFriction = 0;
  for (uint32_t a = 0; a < maxPatches; ++a) {
    totalContacts += maxContactCount[a];
    totalFriction += maxFrictionCount[a];
  }
  const uint32_t headerSize = sizeof(SolverContactHeaderStepBlock) * maxPatches;
  uint32_t constraintSize = (sizeof(SolverContactPointStepBlock) * totalContacts) + (sizeof(SolverContactFrictionStepBlock) * totalFriction);
  constraintSize += sizeof(V4) * (totalContacts + totalFriction);
  if (hasMaxImpulse) constraintSize += sizeof(V4) * totalContacts;
  _solverConstraintByteSize = ((constraintSize + headerSize + 0x0f) & ~0x0fu);
}

// DyTGSContactPrepBlock.cpp:1398 (상관 버퍼판)
SV_HDN PrepState createFinalizeSolverContacts4StepCorr(PrepCtx& P, CorrelationBuffer& c, TGSContactDesc* blockDescs, float invDtF32,
                                                      float totalDtF32, float invTotalDtF32, float dt, float bounceThresholdF32,
                                                      float frictionOffsetThreshold, float correlationDistance, float biasCoefficient) {
  float invMassScale0[4], invMassScale1[4], invInertiaScale0[4], invInertiaScale1[4];
  c.frictionPatchCount = 0;
  c.contactPatchCount = 0;
  for (uint32_t a = 0; a < 4; ++a) {
    TGSContactDesc& blockDesc = blockDescs[a];
    invMassScale0[a] = blockDesc.invMassScales[0];
    invMassScale1[a] = blockDesc.invMassScales[2];
    invInertiaScale0[a] = blockDesc.invMassScales[1];
    invInertiaScale1[a] = blockDesc.invMassScales[3];
    blockDesc.startFrictionPatchIndex = c.frictionPatchCount;
    if (!(blockDesc.disableStrongFriction)) {
      const bool valid = getFrictionPatches(c, blockDesc.frictionPrev, blockDesc.frictionPrevCount, blockDesc.bodyFrame0, blockDesc.bodyFrame1,
                                            correlationDistance);
      if (!valid) return PREP_UNBATCHABLE;
    }
    blockDesc.startContactPatchIndex = c.contactPatchCount;
    if (!createContactPatches(c, blockDesc.contacts, blockDesc.numContacts, PXC_SAME_NORMAL)) return PREP_UNBATCHABLE;
    blockDesc.numContactPatches = uint16_t(c.contactPatchCount - blockDesc.startContactPatchIndex);
    const bool overflow = correlatePatches(c, blockDesc.contacts, blockDesc.bodyFrame0, blockDesc.bodyFrame1, PXC_SAME_NORMAL,
                                           blockDesc.startContactPatchIndex, blockDesc.startFrictionPatchIndex);
    if (overflow) return PREP_UNBATCHABLE;
    growPatches(c, blockDesc.contacts, blockDesc.bodyFrame0, blockDesc.bodyFrame1, blockDesc.startFrictionPatchIndex,
                frictionOffsetThreshold + blockDescs[a].restDistance);
    for (uint32_t p = c.frictionPatchCount; p > blockDesc.startFrictionPatchIndex; --p) {
      if (c.correlationListHeads[p - 1] == 0xffff) {
        for (uint32_t p2 = p; p2 < c.frictionPatchCount; ++p2) {
          c.correlationListHeads[p2 - 1] = c.correlationListHeads[p2];
          c.frictionPatchContactCounts[p2 - 1] = c.frictionPatchContactCounts[p2];
        }
        c.frictionPatchCount--;
      }
    }
    blockDesc.numFrictionPatches = c.frictionPatchCount - blockDesc.startFrictionPatchIndex;
  }
  uint32_t frictionPatchArray[4];
  uint32_t frictionPatchCounts[4];
  for (uint32_t a = 0; a < 4; ++a) {
    TGSContactDesc& blockDesc = blockDescs[a];
    uint32_t numFrictionPatches = 0;
    for (uint32_t i = blockDesc.startFrictionPatchIndex; i < blockDesc.numFrictionPatches + blockDesc.startFrictionPatchIndex; i++)
      if (c.correlationListHeads[i] != CorrelationBuffer::LIST_END) numFrictionPatches++;
    uint32_t off = NONE;
    if (numFrictionPatches > 0) {
      off = frictionAlloc(*P.frictionCur, numFrictionPatches);
      if (off == NONE) return PREP_OUT_OF_MEMORY;
    }
    frictionPatchArray[a] = off;
    frictionPatchCounts[a] = numFrictionPatches;
  }
  uint32_t solverConstraintByteSize = 0;
  uint32_t axisConstraintCount[4];
  computeBlockStreamByteSizes4(blockDescs, solverConstraintByteSize, axisConstraintCount, c);
  uint32_t constraintOff = NONE;
  if (solverConstraintByteSize > 0) {
    if ((solverConstraintByteSize + 16u) > 16384) return PREP_UNBATCHABLE;
    constraintOff = arenaAlloc(*P.constraints, solverConstraintByteSize + 16u);
    if (constraintOff == NONE) return PREP_OUT_OF_MEMORY;
  }
  for (uint32_t a = 0; a < 4; ++a) {
    TGSContactDesc& blockDesc = blockDescs[a];
    SDesc& desc = *blockDesc.desc;
    blockDesc.frictionPtr = frictionPatchArray[a];
    blockDesc.frictionCount = frictionPatchCounts[a];
    if (frictionPatchArray[a] != NONE) {
      FrictionPatch* fp = P.frictionCur->data + frictionPatchArray[a];
      for (uint32_t i = 0; i < blockDesc.numFrictionPatches; i++)
        if (c.correlationListHeads[blockDesc.startFrictionPatchIndex + i] != CorrelationBuffer::LIST_END)
          *fp++ = c.frictionPatches[blockDesc.startFrictionPatchIndex + i];
    }
    blockDesc.axisConstraintCount += axisConstraintCount[a];
    desc.constraint = solverConstraintByteSize ? constraintOff : NONE;
    desc.constraintLengthOver16 = uint16_t(solverConstraintByteSize / 16);
  }
  uint8_t* solverConstraint = arenaPtr<uint8_t>(*P.constraints, constraintOff);
  setupFinalizeSolverConstraints4Step(P, blockDescs, c, solverConstraint, invDtF32, totalDtF32, invTotalDtF32, dt, bounceThresholdF32, biasCoefficient,
                                      V4LoadA(invMassScale0), V4LoadA(invInertiaScale0), V4LoadA(invMassScale1), V4LoadA(invInertiaScale1));
  *reinterpret_cast<uint32_t*>(solverConstraint + solverConstraintByteSize) = 0;
  return PREP_SUCCESS;
}

// DyTGSContactPrepBlock.cpp:1550 (접촉 관리자 출력판). 네 칸 접촉은 한 버퍼에 이어 붙인다(최대 64).
SV_HDN PrepState createFinalizeSolverContacts4Step(PrepCtx& P, const CMOutput* const* cmOutputs, TGSContactDesc* blockDescs, float invDtF32,
                                                  float totalDtF32, float invTotalDtF32, float dtF32, float bounceThresholdF32,
                                                  float frictionOffsetThreshold, float correlationDistance, float biasCoefficient) {
  for (uint32_t a = 0; a < 4; ++a) blockDescs[a].desc->constraintLengthOver16 = 0;
  P.contactBufferCount = 0;
  for (uint32_t a = 0; a < 4; ++a) {
    TGSContactDesc& blockDesc = blockDescs[a];
    blockDesc.contacts = P.contactBuffer + P.contactBufferCount;
    if ((P.contactBufferCount + cmOutputs[a]->nbContacts) > 64 || (blockDesc.torsionalPatchRadius != 0.f || blockDesc.minTorsionalPatchRadius != 0.f))
      return PREP_UNBATCHABLE;
    bool hasMaxImpulse = false;
    bool hasTargetVelocity = false;
    float invMassScale0, invMassScale1, invInertiaScale0, invInertiaScale1;
    const float defaultMaxImpulse = pmin(P.datas[blockDesc.b0].maxContactImpulse, P.datas[blockDesc.b1].maxContactImpulse);
    const uint32_t contactCount = extractContacts(P.contactBuffer, P.contactBufferCount, *cmOutputs[a], hasMaxImpulse, hasTargetVelocity,
                                                  invMassScale0, invMassScale1, invInertiaScale0, invInertiaScale1, defaultMaxImpulse);
    if (contactCount == 0 || hasTargetVelocity) return PREP_UNBATCHABLE;
    blockDesc.numContacts = contactCount;
    blockDesc.hasMaxImpulse = hasMaxImpulse;
    blockDesc.disableStrongFriction = blockDesc.disableStrongFriction || hasTargetVelocity;
    blockDesc.invMassScales[0] *= invMassScale0;
    blockDesc.invMassScales[2] *= invMassScale1;
    blockDesc.invMassScales[1] *= P.vels[blockDesc.b0].isKinematic ? 0.f : invInertiaScale0;
    blockDesc.invMassScales[3] *= P.vels[blockDesc.b1].isKinematic ? 0.f : invInertiaScale1;
  }
  return createFinalizeSolverContacts4StepCorr(P, *P.corr, blockDescs, invDtF32, totalDtF32, invTotalDtF32, dtF32, bounceThresholdF32,
                                               frictionOffsetThreshold, correlationDistance, biasCoefficient);
}

}  // namespace sv
}  // namespace eng
