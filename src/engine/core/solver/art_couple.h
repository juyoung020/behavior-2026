// solver <-> articulation 결합 (손으로 짬, PhysX 5.6.1 과 비트 동일 목표): 관절체 링크가 낀 접촉(ext)의 준비·풀이·되쓰기,
// 링크-정적 제약을 관절체 내부 풀이로 넘기는 자리(관절체별 정적 목록), 관절체 내부 풀이에 넘기는 Static 구현.
// 원본 (physx/source/lowleveldynamics/src/)
//   DyTGSContactPrep.cpp:57-171    computeBlockStreamByteSizesStep (useExtContacts = true: 점 112 B, 마찰 128 B)
//   DyTGSContactPrep.cpp:173-320   SolverExtBodyStep / createImpulseResponseVector / getImpulseResponse (aos·스칼라 두 판)
//   DyTGSContactPrep.cpp:818-1295  setupExtSolverContactStep / setupFinalizeExtSolverContactsStep
//   DyTGSContactPrep.cpp:1297-1490 createFinalizeSolverContactsStep (ext 갈래)
//   DyTGSContactPrep.cpp:2879-3309 solveExtContactsStep / solveExtContactStep (링크 속도 읽기·충격 적용)
//   DyTGSContactPrep.cpp:1863      writeBackContact (ext 보폭)
//   DyFeatherstoneArticulation.cpp:2076-2255 prepareStaticConstraintsTGS, :4323 solveStaticConstraint, :4434 writeback/conclude internal
// 관절체 쪽 계산(응답·속도·충격 적용)은 articulation 모듈 함수(art_step.h, docs 16.4)를 부른다.
#pragma once
#include "../articulation/art_adapters.h"
#include "../articulation/art_static.h"
#include "../joints/tgs_1d.h"
#include "contact_prep.h"
#include "contact_solve.h"

namespace eng {
namespace sv {

// DySolverConstraint1DStep.h:94,119
struct alignas(16) SolverContactPointStepExt : SolverContactPointStep {
  V4 linDeltaVA, linDeltaVB, angDeltaVA, angDeltaVB;
};
struct alignas(16) SolverContactFrictionStepExt : SolverContactFrictionStep {
  V4 linDeltaVA, linDeltaVB, angDeltaVA, angDeltaVB;
};
static_assert(sizeof(SolverContactPointStep) == 48, "PhysX SolverContactPointStep");
static_assert(sizeof(SolverContactPointStepExt) == 112, "PhysX SolverContactPointStepExt");
static_assert(sizeof(SolverContactFrictionStepExt) == 128, "PhysX SolverContactFrictionStepExt");

template <class D, class S>
SV_HD D bitCopy(const S& s) {
  static_assert(sizeof(D) == sizeof(S), "");
  D d;
  memcpy(&d, &s, sizeof(D));
  return d;
}

SV_HD bool isArtDesc(const SDesc& d) { return d.linkIndexA != RIGID_BODY || d.linkIndexB != RIGID_BODY; }  // isArticulationConstraint

// ---------------- SolverExtBodyStep (DyTGSContactPrep.cpp:173)
struct ExtBody {
  art::Articulation* art;  // 링크면 관절체, 아니면 nullptr
  const SBodyVel* body;
  const SBodyTxI* txI;
  const SBodyData* data;
  uint32_t link;  // RIGID_BODY 또는 LL 링크 번호
  SV_HD bool isKinematic() const { return link == RIGID_BODY && body->isKinematic; }
  SV_HD float getCFM() const { return link == RIGID_BODY ? 0.f : art::getCfm(*art, link); }
  // getVelocity: 강체는 원래 속도(data), 링크는 getLinkVelocity (V3LoadA -> W 0)
  SV_HD void getVelocity(V4& lin, V4& ang) const {
    if (link == RIGID_BODY) {
      lin = ldv(data->originalLinearVelocity);
      ang = ldv(data->originalAngularVelocity);
    } else {
      V3 l, a;
      art::getLinkVelocity(*art, link, l, a);
      lin = ldv(l);
      ang = ldv(a);
    }
  }
};

// createImpulseResponseVector (aos 판, :220): 강체면 각 부분에 sqrtInvInertia (M33Load = 세 열 V3LoadU)
SV_HD void extRespVec(V4 linear, V4 angular, const ExtBody& b, V4& oLin, V4& oAng) {
  oLin = linear;
  oAng = b.link == RIGID_BODY ? M33MulV3(m33v(b.txI->sqrtInvInertia), angular) : angular;
}

// getImpulseResponse (aos 판, :273). 링크 쪽은 art::getImpulseResponse(SpatialVectorV 판과 같은 식, docs 16.3 결합점 시험)
SV_HD FV extImpulseResponse(const ExtBody& b0, V4 i0l, V4 i0a, V4& d0l, V4& d0a, FV dom0, FV angDom0, const ExtBody& b1, V4 i1l, V4 i1a, V4& d1l,
                           V4& d1a, FV dom1, FV angDom1) {
  if (b0.link == RIGID_BODY) {
    d0l = V3Scale(i0l, FMul(FLoad(b0.data->invMass), dom0));
    d0a = V3Scale(i0a, angDom0);
  } else {
    const V4 sl = V3Scale(i0l, dom0), sa = V3Scale(i0a, angDom0);  // SpatialVectorV::scale
    V3 l, a;
    art::getImpulseResponse(*b0.art, b0.link, V3{sl.f[0], sl.f[1], sl.f[2]}, V3{sa.f[0], sa.f[1], sa.f[2]}, l, a);
    d0l = ldv(l);
    d0a = ldv(a);
  }
  V4 response = V3Add(V3Mul(i0l, d0l), V3Mul(i0a, d0a));
  if (b1.link == RIGID_BODY) {
    d1l = V3Scale(i1l, FMul(FLoad(b1.data->invMass), dom1));
    d1a = V3Scale(i1a, angDom1);
  } else {
    const V4 sl = V3Scale(i1l, dom1), sa = V3Scale(i1a, angDom1);
    V3 l, a;
    art::getImpulseResponse(*b1.art, b1.link, V3{sl.f[0], sl.f[1], sl.f[2]}, V3{sa.f[0], sa.f[1], sa.f[2]}, l, a);
    d1l = ldv(l);
    d1a = ldv(a);
  }
  response = V3Add(response, V3Add(V3Mul(i1l, d1l), V3Mul(i1a, d1a)));
  return V3SumElems(response);
}

// getImpulseResponse (스칼라 판, :229, allowSelfCollision = false) — 비틀림 마찰 한 곳에서만
SV_HD float extImpulseResponseScalar(const ExtBody& b0, const V3& i0l, const V3& i0a, V3& d0l, V3& d0a, float dom0, float angDom0,
                                     const ExtBody& b1, const V3& i1l, const V3& i1a, V3& d1l, V3& d1a, float dom1, float angDom1) {
  if (b0.link == RIGID_BODY) {
    d0l = i0l * b0.data->invMass * dom0;
    d0a = i0a * angDom0;
  } else {
    art::getImpulseResponse(*b0.art, b0.link, i0l * dom0, i0a * angDom0, d0l, d0a);
  }
  float response = dot(i0l, d0l) + dot(i0a, d0a);  // SpatialVector::dot
  if (b1.link == RIGID_BODY) {
    d1l = i1l * b1.data->invMass * dom1;
    d1a = i1a * angDom1;
  } else {
    art::getImpulseResponse(*b1.art, b1.link, i1l * dom1, i1a * angDom1, d1l, d1a);
  }
  response += dot(i1l, d1l) + dot(i1a, d1a);
  return response;
}

// DyTGSContactPrep.cpp:818 setupExtSolverContactStep. 인자 이름은 원본 그대로(호출자는 invDt 자리에 invTotalDt, invStepDt 자리에 invDt 를 넣는다).
SV_HD FV setupExtSolverContactStep(const ExtBody& b0, const ExtBody& b1, FV d0, FV d1, FV angD0, FV angD1, V4 bodyFrame0p, V4 bodyFrame1p, V4 normal,
                                   FV invDt, FV invDtp8, FV invStepDt, FV totalDt, FV dt, FV restDistance, FV restitution, FV damping,
                                   BS accelerationSpring, FV bounceThreshold, const ContactPoint& contact, SolverContactPointStepExt& solverContact,
                                   bool isKinematic0, bool isKinematic1, FV cfm, V4 v0lin, V4 v0ang, V4 v1lin, V4 v1ang, V4 solverOffsetSlop,
                                   FV norVel0, FV norVel1) {
  const FV zero = FZero();
  const FV separation = FLoad(contact.separation);
  FV penetration = FSub(separation, restDistance);
  const V4 ra = V3Sub(ldv(contact.point), bodyFrame0p);
  const V4 rb = V3Sub(ldv(contact.point), bodyFrame1p);
  V4 raXn = V3Cross(ra, normal);
  V4 rbXn = V3Cross(rb, normal);
  V4 dv0l, dv0a, dv1l, dv1a;
  const FV vRelAng = V3SumElems(V3Sub(V3Mul(v0ang, raXn), V3Mul(v1ang, rbXn)));
  const FV vRelLin = FSub(norVel0, norVel1);
  const V4 slop = V3Scale(solverOffsetSlop, FMax(FSel(FIsEq(vRelLin, zero), FOne(), FDiv(vRelAng, vRelLin)), FOne()));
  raXn = V3Sel(V3IsGrtr(slop, V3Abs(raXn)), V3Zero(), raXn);
  rbXn = V3Sel(V3IsGrtr(slop, V3Abs(rbXn)), V3Zero(), rbXn);
  const FV angV0 = V3Dot(v0ang, raXn);
  const FV angV1 = V3Dot(v1ang, rbXn);
  V4 r0l, r0a, r1l, r1a;
  extRespVec(normal, raXn, b0, r0l, r0a);
  extRespVec(V3Neg(normal), V3Neg(rbXn), b1, r1l, r1a);
  const FV unitResponse = extImpulseResponse(b0, r0l, r0a, dv0l, dv0a, d0, angD0, b1, r1l, r1a, dv1l, dv1a, d1, angD1);
  const FV vrel = FAdd(vRelAng, vRelLin);
  const BS isSeparated = FIsGrtr(penetration, zero);
  FV scaledBias, velMultiplier;
  const FV recipResponse = FSel(FIsGrtr(unitResponse, FZero()), FRecip(FAdd(unitResponse, cfm)), zero);
  const FV penetrationInvDt = FMul(penetration, invDt);
  if (FAllGrtr(zero, restitution)) {
    const BS collidingWithVrel = FIsGrtr(FNeg(vrel), penetrationInvDt);
    computeCompliantContactCoefficientsTGS(dt, restitution, damping, recipResponse, unitResponse, accelerationSpring, isSeparated, collidingWithVrel,
                                           velMultiplier, scaledBias);
  } else {
    velMultiplier = recipResponse;
    scaledBias = FNeg(FSel(isSeparated, invStepDt, invDtp8));
  }
  const BS isGreater2 = BAnd(BAnd(FIsGrtr(restitution, zero), FIsGrtr(bounceThreshold, vrel)), FIsGrtr(FNeg(vrel), penetrationInvDt));
  FV targetVelocity = FSel(isGreater2, FMul(FNeg(vrel), restitution), zero);
  const FV cTargetVel = V3Dot(ldv(contact.targetVel), normal);
  targetVelocity = FAdd(targetVelocity, cTargetVel);
  const FV deltaF = FMax(FMul(FAdd(targetVelocity, FSub(FNeg(penetrationInvDt), vrel)), velMultiplier), zero);
  const FV ratio = FAdd(totalDt, FSel(isGreater2, FDiv(penetration, vrel), FNeg(totalDt)));
  penetration = FScaleAdd(targetVelocity, ratio, penetration);
  if (isKinematic0) targetVelocity = FSub(targetVelocity, FAdd(norVel0, angV0));
  if (isKinematic1) targetVelocity = FAdd(targetVelocity, FAdd(norVel1, angV1));
  FStore(scaledBias, &solverContact.biasCoefficient);
  FStore(targetVelocity, &solverContact.targetVelocity);
  FStore(recipResponse, &solverContact.recipResponse);
  const V4 raXnI_Sepw = V4SetW(Vec4V_From_Vec3V(r0a), penetration);
  const V4 rbXnI_velMulW = V4SetW(Vec4V_From_Vec3V(V3Neg(r1a)), velMultiplier);
  V4StoreA(raXnI_Sepw, &solverContact.raXnI.x);   // raXnI + separation
  V4StoreA(rbXnI_velMulW, &solverContact.rbXnI.x);  // rbXnI + velMultiplier
  solverContact.linDeltaVA = dv0l;
  solverContact.angDeltaVA = dv0a;
  solverContact.linDeltaVB = dv1l;
  solverContact.angDeltaVB = dv1a;
  solverContact.maxImpulse = contact.maxImpulse;
  return deltaF;
}

// DyTGSContactPrep.cpp:960 setupFinalizeExtSolverContactsStep
SV_HDN void setupFinalizeExtSolverContactsStep(const ContactPoint* buffer, const CorrelationBuffer& c, const Tf& bodyFrame0, const Tf& bodyFrame1,
                                              uint8_t* workspace, const ExtBody& b0, const ExtBody& b1, float invDtF32, float invTotalDtF32,
                                              float totalDtF32, float dtF32, float bounceThresholdF32, float invMassScale0, float invInertiaScale0,
                                              float invMassScale1, float invInertiaScale1, float restDist, uint32_t frictionDataPtr,
                                              float torsionalPatchRadiusF32, float minTorsionalPatchRadiusF32, float biasCoefficient,
                                              float solverOffsetSlop) {
  const bool isKinematic0 = b0.isKinematic();
  const bool isKinematic1 = b1.isKinematic();
  const bool hasTorsionalFriction = torsionalPatchRadiusF32 > 0.f || minTorsionalPatchRadiusF32 > 0.f;
  const FV quarter = FLoad(0.25f);
  uint8_t* ptr = workspace;
  const FV zero = FZero();
  const float maxPenBias0 = b0.link == RIGID_BODY ? b0.data->penBiasClamp : art::getLinkMaxPenBias(*b0.art, b0.link);
  const float maxPenBias1 = b1.link == RIGID_BODY ? b1.data->penBiasClamp : art::getLinkMaxPenBias(*b1.art, b1.link);
  const float maxPenBias = pmax(maxPenBias0, maxPenBias1);
  V4 v0lin, v0ang, v1lin, v1ang;
  b0.getVelocity(v0lin, v0ang);
  b1.getVelocity(v1lin, v1ang);
  const FV d0 = FLoad(invMassScale0);
  const FV d1 = FLoad(invMassScale1);
  const FV angD0 = FLoad(invInertiaScale0);
  const FV angD1 = FLoad(invInertiaScale1);
  const V4 bodyFrame0p = ldv(bodyFrame0.p);
  const V4 bodyFrame1p = ldv(bodyFrame1.p);
  V4 sfdf = V4Zero();
  sfdf = V4SetZ(sfdf, d0);
  sfdf = V4SetW(sfdf, d1);
  const FV restDistance = FLoad(restDist);
  uint32_t frictionPatchWritebackAddrIndex = 0;
  const FV invDt = FLoad(invDtF32);
  const FV invTotalDt = FLoad(invTotalDtF32);
  const float scale = pmin(0.8f, biasCoefficient);
  const FV p8 = FLoad(scale);
  const FV bounceThreshold = FLoad(bounceThresholdF32);
  const FV totalDt = FLoad(totalDtF32);
  const FV dt = FLoad(dtF32);
  const FV invDtp8 = FMul(invDt, p8);
  const FV cfm = FLoad(pmax(b0.getCFM(), b1.getCFM()));
  const uint8_t flags = 0;
  const V4 offsetSlop = V3Load(solverOffsetSlop);
  for (uint32_t i = 0; i < c.frictionPatchCount; i++) {
    const uint32_t contactCount = c.frictionPatchContactCounts[i];
    if (contactCount == 0) continue;
    const FrictionPatch& frictionPatch = c.frictionPatches[i];
    const ContactPoint* contactBase0 = buffer + c.contactPatches[c.correlationListHeads[i]].start;
    const bool disableStrongFriction = !!(contactBase0->materialFlags & MAT_DISABLE_FRICTION);
    sfdf = V4SetX(sfdf, FLoad(contactBase0->staticFriction));
    sfdf = V4SetY(sfdf, FLoad(contactBase0->dynamicFriction));
    const float frictionBiasScale = disableStrongFriction ? 0.f : invDtF32 * 0.8f;
    SolverContactHeaderStep* header = reinterpret_cast<SolverContactHeaderStep*>(ptr);
    ptr += sizeof(SolverContactHeaderStep);
    const bool haveFriction = (disableStrongFriction == 0);
    header->numNormalConstr = uint8_t(contactCount);
    header->numFrictionConstr = uint8_t(haveFriction ? frictionPatch.anchorCount * 2 : 0);
    header->type = SC_TYPE_EXT_CONTACT;
    header->flags = flags;
    const FV restitution = FLoad(contactBase0->restitution);
    const FV damping = FLoad(contactBase0->damping);
    const BS accelSpring = BLoad(!!(contactBase0->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING));
    header->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W = sfdf;
    header->angDom0 = invInertiaScale0;
    header->angDom1 = invInertiaScale1;
    header->invMass0 = 0.f;  // 원본은 안 씀(쓰레기) — ext 풀이·되쓰기가 읽지 않는다
    header->invMass1 = 0.f;
    header->frictionBrokenWriteback = NONE;
    const V4 normal = ldv(buffer[c.contactPatches[c.correlationListHeads[i]].start].normal);
    stv(normal, header->normal);
    header->maxPenBias = maxPenBias;
    FV maxPenetration = FZero();
    FV accumulatedImpulse = FZero();
    const FV norVel0 = V3Dot(v0lin, normal);
    const FV norVel1 = V3Dot(v1lin, normal);
    for (uint32_t patch = c.correlationListHeads[i]; patch != CorrelationBuffer::LIST_END; patch = c.contactPatches[patch].next) {
      const uint32_t count = c.contactPatches[patch].count;
      const ContactPoint* contactBase = buffer + c.contactPatches[patch].start;
      uint8_t* p = ptr;
      for (uint32_t j = 0; j < count; j++) {
        const ContactPoint& contact = contactBase[j];
        SolverContactPointStepExt* solverContact = reinterpret_cast<SolverContactPointStepExt*>(p);
        p += sizeof(SolverContactPointStepExt);
        const FV deltaF = setupExtSolverContactStep(b0, b1, d0, d1, angD0, angD1, bodyFrame0p, bodyFrame1p, normal, invTotalDt, invDtp8, invDt,
                                                    totalDt, dt, restDistance, restitution, damping, accelSpring, bounceThreshold, contact,
                                                    *solverContact, isKinematic0, isKinematic1, cfm, v0lin, v0ang, v1lin, v1ang, offsetSlop, norVel0,
                                                    norVel1);
        accumulatedImpulse = FAdd(accumulatedImpulse, deltaF);
        maxPenetration = FMin(FLoad(contact.separation), maxPenetration);
      }
      ptr = p;
    }
    accumulatedImpulse = FMul(FDiv(accumulatedImpulse, FLoad(float(contactCount))), quarter);
    FStore(accumulatedImpulse, &header->minNormalForce);
    float* forceBuffer = reinterpret_cast<float*>(ptr);
    for (uint32_t k = 0; k < contactCount; ++k) forceBuffer[k] = 0.0f;
    ptr += sizeof(float) * ((contactCount + 3) & (~3u));
    header->broken = 0;
    if (haveFriction) {
      const V4 linVrel = V3Sub(v0lin, v1lin);
      const FV orthoThreshold = FLoad(0.70710678f);
      const FV p1 = FLoad(0.0001f);
      const FV normalX = V3GetX(normal);
      const FV normalY = V3GetY(normal);
      const FV normalZ = V3GetZ(normal);
      const V4 t0Fallback1 = V3Merge(zero, FNeg(normalZ), normalY);
      const V4 t0Fallback2 = V3Merge(FNeg(normalY), normalX, zero);
      const V4 t0Fallback = V3Sel(FIsGrtr(orthoThreshold, FAbs(normalX)), t0Fallback1, t0Fallback2);
      V4 t0 = V3Sub(linVrel, V3Scale(normal, V3Dot(normal, linVrel)));
      t0 = V3Sel(FIsGrtr(V3LengthSq(t0), p1), t0, t0Fallback);
      t0 = V3Normalize(t0);
      const VecCrossV t0Cross = V3PrepareCross(t0);
      const V4 t1 = V3Cross(normal, t0Cross);
      const VecCrossV t1Cross = V3PrepareCross(t1);
      header->frictionBrokenWriteback = frictionDataPtr == NONE ? NONE : frictionDataPtr + frictionPatchWritebackAddrIndex;
      const float frictionScale = (frictionPatch.anchorCount == 2) ? 0.5f : 1.f;
      for (uint32_t j = 0; j < frictionPatch.anchorCount; j++) {
        SolverContactFrictionStepExt* f0 = reinterpret_cast<SolverContactFrictionStepExt*>(ptr);
        ptr += sizeof(SolverContactFrictionStepExt);
        SolverContactFrictionStepExt* f1 = reinterpret_cast<SolverContactFrictionStepExt*>(ptr);
        ptr += sizeof(SolverContactFrictionStepExt);
        const V4 ra = ldv(rotate(bodyFrame0.q, frictionPatch.body0Anchors[j]));  // V3LoadU(q.rotate(anchor)) 스칼라 회전
        const V4 rb = ldv(rotate(bodyFrame1.q, frictionPatch.body1Anchors[j]));
        const V4 error = V3Sub(V3Add(ra, bodyFrame0p), V3Add(rb, bodyFrame1p));
        const uint32_t index = c.contactPatches[c.correlationListHeads[i]].start;
        for (int k = 0; k < 2; ++k) {
          SolverContactFrictionStepExt* f = k == 0 ? f0 : f1;
          const V4 tk = k == 0 ? t0 : t1;
          const VecCrossV& tc = k == 0 ? t0Cross : t1Cross;
          V4 raXn = V3Cross(ra, tc);
          V4 rbXn = V3Cross(rb, tc);
          raXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(raXn)), V3Zero(), raXn);
          rbXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(rbXn)), V3Zero(), rbXn);
          V4 dv0l, dv0a, dv1l, dv1a, r0l, r0a, r1l, r1a;
          extRespVec(tk, raXn, b0, r0l, r0a);
          extRespVec(V3Neg(tk), V3Neg(rbXn), b1, r1l, r1a);
          const FV resp = extImpulseResponse(b0, r0l, r0a, dv0l, dv0a, d0, angD0, b1, r1l, r1a, dv1l, dv1a, d1, angD1);
          const FV velMultiplier = FSel(FIsGrtr(resp, FEps()), FDiv(p8, FAdd(cfm, resp)), zero);
          FV targetVel = V3Dot(ldv(buffer[index].targetVel), tk);
          // SpatialVectorV::dot = V3SumElems(lin*lin' + ang*ang')
          if (isKinematic0) targetVel = FSub(targetVel, V3SumElems(V3Add(V3Mul(v0lin, r0l), V3Mul(v0ang, r0a))));
          if (isKinematic1) targetVel = FSub(targetVel, V3SumElems(V3Add(V3Mul(v1lin, r1l), V3Mul(v1ang, r1a))));
          f->normalXYZ_ErrorW = V4SetW(Vec4V_From_Vec3V(tk), V3Dot(error, tk));
          f->raXnI_targetVelW = V4SetW(Vec4V_From_Vec3V(r0a), targetVel);
          f->rbXnI_velMultiplierW = V4SetW(V4Neg(Vec4V_From_Vec3V(r1a)), velMultiplier);
          f->appliedForce = 0.f;
          f->biasScale = frictionBiasScale;
          f->linDeltaVA = dv0l;
          f->linDeltaVB = dv1l;
          f->angDeltaVA = dv0a;
          f->angDeltaVB = dv1a;
          f->frictionScale = frictionScale;
          f->pad = 0;
        }
      }
      if (hasTorsionalFriction && frictionPatch.anchorCount == 1) {
        const FV torsionalPatchRadius = FLoad(torsionalPatchRadiusF32);
        const FV minTorsionalPatchRadius = FLoad(minTorsionalPatchRadiusF32);
        const FV torsionalFriction = FMax(minTorsionalPatchRadius, FSqrt(FMul(FMax(zero, FNeg(maxPenetration)), torsionalPatchRadius)));
        header->numFrictionConstr++;
        SolverContactFrictionStepExt* f = reinterpret_cast<SolverContactFrictionStepExt*>(ptr);
        ptr += sizeof(SolverContactFrictionStepExt);
        // createImpulseResponseVector (스칼라 판, :212)
        const V3 z{0.f, 0.f, 0.f};
        const V3 hn = header->normal;
        const V3 r0a = b0.link == RIGID_BODY ? b0.txI->sqrtInvInertia * hn : hn;
        const V3 r1a = b1.link == RIGID_BODY ? b1.txI->sqrtInvInertia * (-hn) : -hn;
        V3 dv0l, dv0a, dv1l, dv1a;
        const float ur = extImpulseResponseScalar(b0, z, r0a, dv0l, dv0a, invMassScale0, invInertiaScale0, b1, z, r1a, dv1l, dv1a, invMassScale1,
                                                  invInertiaScale1);
        const FV resp = FLoad(ur);
        const FV velMultiplier = FSel(FIsGrtr(resp, FEps()), FDiv(p8, FAdd(cfm, resp)), zero);
        f->normalXYZ_ErrorW = V4Zero();
        f->raXnI_targetVelW = V4ClearW(V4{r0a.x, r0a.y, r0a.z, 0.0f});
        f->rbXnI_velMultiplierW = V4SetW(V4Neg(V4{r1a.x, r1a.y, r1a.z, 0.0f}), velMultiplier);
        f->biasScale = 0.f;
        f->appliedForce = 0.f;
        FStore(torsionalFriction, &f->frictionScale);
        f->linDeltaVA = ldv(dv0l);
        f->linDeltaVB = ldv(dv1l);
        f->angDeltaVA = ldv(dv0a);
        f->angDeltaVB = ldv(dv1a);
        f->pad = 0;
      }
    }
    frictionPatchWritebackAddrIndex++;
  }
}

// computeBlockStreamByteSizesStep (useExtContacts = true)
SV_HD void computeBlockStreamByteSizesStepExt(const CorrelationBuffer& c, uint32_t& _solverConstraintByteSize, uint32_t& _numFrictionPatches,
                                              float torsionalPatchRadius) {
  uint32_t solverConstraintByteSize = 0, numFrictionPatches = 0;
  for (uint32_t i = 0; i < c.frictionPatchCount; i++) {
    if (c.correlationListHeads[i] != CorrelationBuffer::LIST_END) numFrictionPatches++;
    const FrictionPatch& frictionPatch = c.frictionPatches[i];
    const bool haveFriction = (frictionPatch.materialFlags & MAT_DISABLE_FRICTION) == 0;
    if (c.frictionPatchContactCounts[i] != 0) {
      solverConstraintByteSize += sizeof(SolverContactHeaderStep);
      solverConstraintByteSize += c.frictionPatchContactCounts[i] * sizeof(SolverContactPointStepExt);
      solverConstraintByteSize += sizeof(float) * ((c.frictionPatchContactCounts[i] + 3) & (~3u));
      if (haveFriction) {
        uint32_t nbAnchors = uint32_t(c.frictionPatches[i].anchorCount * 2);
        if (torsionalPatchRadius > 0.f && c.frictionPatches[i].anchorCount == 1) nbAnchors++;
        solverConstraintByteSize += nbAnchors * sizeof(SolverContactFrictionStepExt);
      }
    }
  }
  _numFrictionPatches = numFrictionPatches;
  _solverConstraintByteSize = ((solverConstraintByteSize + 0x0f) & ~0x0fu);
}

// createFinalizeSolverContactsStep (접촉 관리자 출력판 :1441 + 상관 버퍼판 :1297 의 ext 갈래)
SV_HDN bool createFinalizeSolverContactsStepExt(PrepCtx& P, TGSContactDesc& contactDesc, const CMOutput& output, const ExtBody& e0, const ExtBody& e1,
                                               float invDtF32, float invTotalDt, float totalDtF32, float dt, float bounceThresholdF32,
                                               float frictionOffsetThreshold, float correlationDistance, float biasCoefficient) {
  P.contactBufferCount = 0;
  {
    float invMassScale0 = 1.f, invMassScale1 = 1.f, invInertiaScale0 = 1.f, invInertiaScale1 = 1.f;
    contactDesc.invMassScales[1] = (contactDesc.bodyState0 != BS_ARTICULATION && P.vels[contactDesc.b0].isKinematic) ? 0.f : contactDesc.invMassScales[1];
    contactDesc.invMassScales[3] = (contactDesc.bodyState1 != BS_ARTICULATION && P.vels[contactDesc.b1].isKinematic) ? 0.f : contactDesc.invMassScales[3];
    bool hasMaxImpulse = false, hasTargetVelocity = false;
    const uint32_t numContacts = extractContacts(P.contactBuffer, P.contactBufferCount, output, hasMaxImpulse, hasTargetVelocity, invMassScale0,
                                                 invMassScale1, invInertiaScale0, invInertiaScale1, contactDesc.maxImpulse);
    contactDesc.contacts = P.contactBuffer;
    contactDesc.numContacts = numContacts;
    contactDesc.disableStrongFriction = contactDesc.disableStrongFriction || hasTargetVelocity;
    contactDesc.hasMaxImpulse = hasMaxImpulse;
    contactDesc.invMassScales[0] *= invMassScale0;
    contactDesc.invMassScales[2] *= invMassScale1;
    contactDesc.invMassScales[1] *= invInertiaScale0;
    contactDesc.invMassScales[3] *= invInertiaScale1;
  }
  CorrelationBuffer& c = *P.corr;
  c.frictionPatchCount = 0;
  c.contactPatchCount = 0;
  const bool disableStrongFriction = contactDesc.disableStrongFriction;
  SDesc& desc = *contactDesc.desc;
  desc.constraintLengthOver16 = 0;
  if (contactDesc.numContacts == 0) {
    contactDesc.frictionPtr = NONE;
    contactDesc.frictionCount = 0;
    desc.constraint = NONE;
    return true;
  }
  if (!disableStrongFriction)
    getFrictionPatches(c, contactDesc.frictionPrev, contactDesc.frictionPrevCount, contactDesc.bodyFrame0, contactDesc.bodyFrame1, correlationDistance);
  bool overflow = !createContactPatches(c, contactDesc.contacts, contactDesc.numContacts, PXC_SAME_NORMAL);
  overflow = correlatePatches(c, contactDesc.contacts, contactDesc.bodyFrame0, contactDesc.bodyFrame1, PXC_SAME_NORMAL, 0, 0) || overflow;
  (void)overflow;
  growPatches(c, contactDesc.contacts, contactDesc.bodyFrame0, contactDesc.bodyFrame1, 0, frictionOffsetThreshold + contactDesc.restDistance);
  uint32_t solverConstraintByteSize = 0, numFrictionPatches = 0;
  computeBlockStreamByteSizesStepExt(c, solverConstraintByteSize, numFrictionPatches,
                                     pmax(contactDesc.torsionalPatchRadius, contactDesc.minTorsionalPatchRadius));
  uint32_t constraintOff = NONE, frictionOff = NONE;
  if (solverConstraintByteSize > 0) constraintOff = arenaAlloc(*P.constraints, solverConstraintByteSize + 16u);
  if (numFrictionPatches > 0 && (solverConstraintByteSize == 0 || constraintOff != NONE)) frictionOff = frictionAlloc(*P.frictionCur, numFrictionPatches);
  const bool successfulReserve = (solverConstraintByteSize == 0 || constraintOff != NONE) && (numFrictionPatches == 0 || frictionOff != NONE);
  contactDesc.frictionPtr = NONE;
  contactDesc.frictionCount = 0;
  desc.constraint = NONE;
  desc.constraintLengthOver16 = 0;
  if (successfulReserve) {
    contactDesc.frictionPtr = frictionOff;
    desc.constraint = solverConstraintByteSize ? constraintOff : NONE;
    contactDesc.frictionCount = numFrictionPatches;
    desc.constraintLengthOver16 = uint16_t(solverConstraintByteSize / 16);
    if (frictionOff != NONE) {
      FrictionPatch* fp = P.frictionCur->data + frictionOff;
      for (uint32_t i = 0; i < c.frictionPatchCount; i++)
        if (c.frictionPatchContactCounts[i]) *fp++ = c.frictionPatches[i];
    }
    if (desc.constraint != NONE) {
      uint8_t* solverConstraint = arenaPtr<uint8_t>(*P.constraints, constraintOff);
      setupFinalizeExtSolverContactsStep(contactDesc.contacts, c, contactDesc.bodyFrame0, contactDesc.bodyFrame1, solverConstraint, e0, e1, invDtF32,
                                         invTotalDt, totalDtF32, dt, bounceThresholdF32, contactDesc.invMassScales[0], contactDesc.invMassScales[1],
                                         contactDesc.invMassScales[2], contactDesc.invMassScales[3], contactDesc.restDistance, frictionOff,
                                         contactDesc.torsionalPatchRadius, contactDesc.minTorsionalPatchRadius, biasCoefficient,
                                         contactDesc.offsetSlop);
      *reinterpret_cast<uint32_t*>(solverConstraint + solverConstraintByteSize) = 0;
    }
  }
  return successfulReserve;
}

// ---------------- 풀이: solveExtContactsStep (:2879) + solveExtContactStep 본체 (:2953)
SV_HD FV solveExtContactsStep(SolverContactPointStepExt* contacts, uint32_t nbContactPoints, V4 contactNormal, V4& linVel0, V4& angVel0, V4& linVel1,
                              V4& angVel1, V4& li0, V4& ai0, V4& li1, V4& ai1, V4 linDeltaA, V4 linDeltaB, V4 angDeltaA, V4 angDeltaB, FV maxPenBias,
                              float* appliedForceBuffer, FV minPen, FV elapsedTime) {
  const FV deltaV = V3Dot(contactNormal, V3Sub(linDeltaA, linDeltaB));
  FV accumulatedNormalImpulse = FZero();
  for (uint32_t i = 0; i < nbContactPoints; i++) {
    SolverContactPointStepExt& c = contacts[i];
    const V4 raXn = ldv3(c.raXnI);
    const V4 rbXn = ldv3(c.rbXnI);
    const FV appliedForce = FLoad(appliedForceBuffer[i]);
    const FV velMultiplier = FLoad(c.velMultiplier);
    const FV recipResponse = FLoad(c.recipResponse);
    V4 v = V3MulAdd(linVel0, contactNormal, V3Mul(angVel0, raXn));
    v = V3Sub(v, V3MulAdd(linVel1, contactNormal, V3Mul(angVel1, rbXn)));
    const FV normalVel = V3SumElems(v);
    const FV angDelta0 = V3Dot(angDeltaA, raXn);
    const FV angDelta1 = V3Dot(angDeltaB, rbXn);
    const FV deltaAng = FSub(angDelta0, angDelta1);
    const FV targetVel = FLoad(c.targetVelocity);
    const FV deltaBias = FSub(FAdd(deltaV, deltaAng), FMul(targetVel, elapsedTime));
    const FV biasCoefficient = FLoad(c.biasCoefficient);
    const FV sep = FMax(minPen, FAdd(FLoad(c.separation), deltaBias));
    const FV bias = FMin(FNeg(maxPenBias), FMul(biasCoefficient, sep));
    const FV tVelBias = FMul(bias, recipResponse);
    const FV _deltaF = FMax(FSub(tVelBias, FMul(FSub(normalVel, targetVel), velMultiplier)), FNeg(appliedForce));
    const FV _newForce = FAdd(appliedForce, _deltaF);
    const FV newForce = FMin(_newForce, FLoad(c.maxImpulse));
    const FV deltaF = FSub(newForce, appliedForce);
    const V4 raXnI = c.angDeltaVA;
    const V4 rbXnI = c.angDeltaVB;
    linVel0 = V3ScaleAdd(c.linDeltaVA, deltaF, linVel0);
    angVel0 = V3ScaleAdd(raXnI, deltaF, angVel0);
    linVel1 = V3ScaleAdd(c.linDeltaVB, deltaF, linVel1);
    angVel1 = V3ScaleAdd(rbXnI, deltaF, angVel1);
    li0 = V3ScaleAdd(contactNormal, deltaF, li0);
    ai0 = V3ScaleAdd(raXn, deltaF, ai0);
    li1 = V3ScaleAdd(contactNormal, deltaF, li1);
    ai1 = V3ScaleAdd(rbXn, deltaF, ai1);
    FStore(newForce, &appliedForceBuffer[i]);
    accumulatedNormalImpulse = FAdd(accumulatedNormalImpulse, newForce);
  }
  return accumulatedNormalImpulse;
}

SV_HDN void solveExtContactStepCore(uint8_t* constraint, uint32_t lengthOver16, V4& linVel0, V4& linVel1, V4& angVel0, V4& angVel1, V4 linDelta0,
                                    V4 linDelta1, V4 angDelta0, V4 angDelta1, V4& linImpulse0, V4& linImpulse1, V4& angImpulse0, V4& angImpulse1,
                                    float minPenetration, float elapsedTimeF32) {
  const FV elapsedTime = FLoad(elapsedTimeF32);
  const FV minPen = FLoad(minPenetration);
  const FV zero = FZero();
  const uint8_t* last = constraint + lengthOver16 * 16u;
  uint8_t* currPtr = constraint;
  const V4 relMotion = V3Sub(linDelta0, linDelta1);
  while (currPtr < last) {
    SolverContactHeaderStep* hdr = reinterpret_cast<SolverContactHeaderStep*>(currPtr);
    currPtr += sizeof(SolverContactHeaderStep);
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    SolverContactPointStepExt* contacts = reinterpret_cast<SolverContactPointStepExt*>(currPtr);
    currPtr += numNormalConstr * sizeof(SolverContactPointStepExt);
    float* appliedForceBuffer = reinterpret_cast<float*>(currPtr);
    currPtr += sizeof(float) * ((numNormalConstr + 3) & (~3u));
    SolverContactFrictionStepExt* frictions = reinterpret_cast<SolverContactFrictionStepExt*>(currPtr);
    currPtr += numFrictionConstr * sizeof(SolverContactFrictionStepExt);
    V4 li0 = V3Zero(), li1 = V3Zero(), ai0 = V3Zero(), ai1 = V3Zero();
    const V4 contactNormal = ldv3(hdr->normal);
    const FV accumulatedNormalImpulse =
        FMax(solveExtContactsStep(contacts, numNormalConstr, contactNormal, linVel0, angVel0, linVel1, angVel1, li0, ai0, li1, ai1, linDelta0, linDelta1,
                                  angDelta0, angDelta1, FLoad(hdr->maxPenBias), appliedForceBuffer, minPen, elapsedTime),
             FLoad(hdr->minNormalForce));
    if (numFrictionConstr) {
      const FV maxFrictionImpulse = FMul(V4GetX(hdr->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W), accumulatedNormalImpulse);
      const FV maxDynFrictionImpulse = FMul(V4GetY(hdr->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W), accumulatedNormalImpulse);
      BoolV broken = BFFFF();
      const uint32_t numFrictionPairs = numFrictionConstr & 6;
      for (uint32_t i = 0; i < numFrictionPairs; i += 2) {
        SolverContactFrictionStepExt& f0 = frictions[i];
        SolverContactFrictionStepExt& f1 = frictions[i + 1];
        const V4 normalXYZ_ErrorW0 = f0.normalXYZ_ErrorW;
        const V4 raXn_targetVelW0 = f0.raXnI_targetVelW;
        const V4 rbXn_velMultiplierW0 = f0.rbXnI_velMultiplierW;
        const V4 normalXYZ_ErrorW1 = f1.normalXYZ_ErrorW;
        const V4 raXn_targetVelW1 = f1.raXnI_targetVelW;
        const V4 rbXn_velMultiplierW1 = f1.rbXnI_velMultiplierW;
        const V4 normal0 = Vec3V_From_Vec4V(normalXYZ_ErrorW0);
        const V4 raXn0 = Vec3V_From_Vec4V(raXn_targetVelW0);
        const V4 rbXn0 = Vec3V_From_Vec4V(rbXn_velMultiplierW0);
        const V4 raXnI0 = f0.angDeltaVA;
        const V4 rbXnI0 = f0.angDeltaVB;
        const V4 normal1 = Vec3V_From_Vec4V(normalXYZ_ErrorW1);
        const V4 raXn1 = Vec3V_From_Vec4V(raXn_targetVelW1);
        const V4 rbXn1 = Vec3V_From_Vec4V(rbXn_velMultiplierW1);
        const V4 raXnI1 = f1.angDeltaVA;
        const V4 rbXnI1 = f1.angDeltaVB;
        const FV frictionScale = FLoad(f0.frictionScale);
        const FV biasScale = FLoad(f0.biasScale);
        const FV appliedForce0 = FLoad(f0.appliedForce);
        const FV velMultiplier0 = V4GetW(rbXn_velMultiplierW0);
        const FV targetVel0 = V4GetW(raXn_targetVelW0);
        const FV initialError0 = V4GetW(normalXYZ_ErrorW0);
        const FV appliedForce1 = FLoad(f1.appliedForce);
        const FV velMultiplier1 = V4GetW(rbXn_velMultiplierW1);
        const FV targetVel1 = V4GetW(raXn_targetVelW1);
        const FV initialError1 = V4GetW(normalXYZ_ErrorW1);
        const FV error0 = FAdd(initialError0, FNegScaleSub(targetVel0, elapsedTime,
                                                           FAdd(FSub(V3Dot(raXn0, angDelta0), V3Dot(rbXn0, angDelta1)), V3Dot(normal0, relMotion))));
        const FV error1 = FAdd(initialError1, FNegScaleSub(targetVel1, elapsedTime,
                                                           FAdd(FSub(V3Dot(raXn1, angDelta0), V3Dot(rbXn1, angDelta1)), V3Dot(normal1, relMotion))));
        const FV bias0 = FMul(error0, biasScale);
        const FV bias1 = FMul(error1, biasScale);
        const V4 v00 = V3MulAdd(linVel0, normal0, V3Mul(angVel0, raXn0));
        const V4 v10 = V3MulAdd(linVel1, normal0, V3Mul(angVel1, rbXn0));
        const FV normalVel0 = V3SumElems(V3Sub(v00, v10));
        const V4 v01 = V3MulAdd(linVel0, normal1, V3Mul(angVel0, raXn1));
        const V4 v11 = V3MulAdd(linVel1, normal1, V3Mul(angVel1, rbXn1));
        const FV normalVel1 = V3SumElems(V3Sub(v01, v11));
        const FV tmp10 = FNegScaleSub(FSub(bias0, targetVel0), velMultiplier0, appliedForce0);
        const FV tmp11 = FNegScaleSub(FSub(bias1, targetVel1), velMultiplier1, appliedForce1);
        const FV totalImpulse0 = FNegScaleSub(normalVel0, velMultiplier0, tmp10);
        const FV totalImpulse1 = FNegScaleSub(normalVel1, velMultiplier1, tmp11);
        const FV totalImpulse = FSqrt(FAdd(FMul(totalImpulse0, totalImpulse0), FMul(totalImpulse1, totalImpulse1)));
        const BS clamp = FIsGrtr(totalImpulse, FMul(maxFrictionImpulse, frictionScale));
        const FV totalClamped = FSel(clamp, FMin(FMul(maxDynFrictionImpulse, frictionScale), totalImpulse), totalImpulse);
        const FV ratio = FSel(FIsGrtr(totalImpulse, zero), FDiv(totalClamped, totalImpulse), zero);
        const FV newAppliedForce0 = FMul(ratio, totalImpulse0);
        const FV newAppliedForce1 = FMul(ratio, totalImpulse1);
        broken = BOr(broken, clamp);
        const FV deltaF0 = FSub(newAppliedForce0, appliedForce0);
        const FV deltaF1 = FSub(newAppliedForce1, appliedForce1);
        linVel0 = V3ScaleAdd(f0.linDeltaVA, deltaF0, V3ScaleAdd(f1.linDeltaVA, deltaF1, linVel0));
        angVel0 = V3ScaleAdd(raXnI0, deltaF0, V3ScaleAdd(raXnI1, deltaF1, angVel0));
        linVel1 = V3ScaleAdd(f0.linDeltaVB, deltaF0, V3ScaleAdd(f1.linDeltaVB, deltaF1, linVel1));
        angVel1 = V3ScaleAdd(rbXnI0, deltaF0, V3ScaleAdd(rbXnI1, deltaF1, angVel1));
        li0 = V3ScaleAdd(normal0, deltaF0, V3ScaleAdd(normal1, deltaF1, li0));
        ai0 = V3ScaleAdd(raXn0, deltaF0, V3ScaleAdd(raXn1, deltaF1, ai0));
        li1 = V3ScaleAdd(normal0, deltaF0, V3ScaleAdd(normal1, deltaF1, li1));
        ai1 = V3ScaleAdd(rbXn0, deltaF0, V3ScaleAdd(rbXn1, deltaF1, ai1));
        FStore(newAppliedForce0, &f0.appliedForce);
        FStore(newAppliedForce1, &f1.appliedForce);
      }
      for (uint32_t i = numFrictionPairs; i < numFrictionConstr; i++) {
        SolverContactFrictionStepExt& f = frictions[i];
        const V4 raXn_targetVelW = f.raXnI_targetVelW;
        const V4 rbXn_velMultiplierW = f.rbXnI_velMultiplierW;
        const V4 raXn = Vec3V_From_Vec4V(raXn_targetVelW);
        const V4 rbXn = Vec3V_From_Vec4V(rbXn_velMultiplierW);
        const V4 raXnI = f.angDeltaVA;
        const V4 rbXnI = f.angDeltaVB;
        const FV frictionScale = FLoad(f.frictionScale);
        const FV appliedForce = FLoad(f.appliedForce);
        const FV velMultiplier = V4GetW(rbXn_velMultiplierW);
        const FV targetVel = V4GetW(raXn_targetVelW);
        const FV negMaxDynFrictionImpulse = FNeg(maxDynFrictionImpulse);
        const V4 v0 = V3Mul(angVel0, raXn);
        const V4 v1 = V3Mul(angVel1, rbXn);
        const FV normalVel = V3SumElems(V3Sub(v0, v1));
        const FV tmp1 = FNegScaleSub(FNeg(targetVel), velMultiplier, appliedForce);
        const FV totalImpulse = FNegScaleSub(normalVel, velMultiplier, tmp1);
        const BS clamp = FIsGrtr(FAbs(totalImpulse), FMul(maxFrictionImpulse, frictionScale));
        const FV totalClamped = FMin(FMul(maxDynFrictionImpulse, frictionScale), FMax(FMul(negMaxDynFrictionImpulse, frictionScale), totalImpulse));
        const FV newAppliedForce = FSel(clamp, totalClamped, totalImpulse);
        broken = BOr(broken, clamp);
        const FV deltaF = FSub(newAppliedForce, appliedForce);
        linVel0 = V3ScaleAdd(f.linDeltaVA, deltaF, linVel0);
        angVel0 = V3ScaleAdd(raXnI, deltaF, angVel0);
        linVel1 = V3ScaleAdd(f.linDeltaVB, deltaF, linVel1);
        angVel1 = V3ScaleAdd(rbXnI, deltaF, angVel1);
        ai0 = V3ScaleAdd(raXn, deltaF, ai0);
        ai1 = V3ScaleAdd(rbXn, deltaF, ai1);
        FStore(newAppliedForce, &f.appliedForce);
      }
      Store_From_BoolV(broken, &hdr->broken);
    }
    const V4 sfdf = hdr->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W;
    linImpulse0 = V3ScaleAdd(li0, V4GetZ(sfdf), linImpulse0);  // getDominance0
    angImpulse0 = V3ScaleAdd(ai0, FLoad(hdr->angDom0), angImpulse0);
    linImpulse1 = V3NegScaleSub(li1, V4GetW(sfdf), linImpulse1);  // getDominance1
    angImpulse1 = V3NegScaleSub(ai1, FLoad(hdr->angDom1), angImpulse1);
  }
}

// ---------------- 관절체 쪽 몸 읽기·쓰기 (solveExtContactStep(desc, ...) :3209, solveExt1DStep 과 같은 방식)
SV_HD V4 v3to4(const V3& v) { return V4{v.x, v.y, v.z, 0.0f}; }  // V3LoadA(SpatialVectorF 칸)
SV_HD V3 v4to3(V4 v) { return V3{v.f[0], v.f[1], v.f[2]}; }

// 분할 안의 EXT_CONTACT 하나 (:3209). vels = 풀이 몸체, arts = 판의 관절체.
SV_HDN void solveExtContactDesc(const SDesc& desc, SBodyVel* vels, art::Articulation* arts, ByteArena& arena, float minPenetration,
                                float elapsedTimeF32) {
  if (desc.constraint == NONE) return;
  V4 linVel0, angVel0, linVel1, angVel1, linDelta0, angDelta0, linDelta1, angDelta1;
  art::Articulation* artA = desc.linkIndexA == RIGID_BODY ? nullptr : &arts[desc.bodyA];
  art::Articulation* artB = desc.linkIndexB == RIGID_BODY ? nullptr : &arts[desc.bodyB];
  const bool same = artA && artA == artB;
  if (same) {
    V3 l0, a0, l1, a1;
    art::pxcFsGetVelocities(*artA, desc.linkIndexA, desc.linkIndexB, l0, a0, l1, a1);
    linVel0 = v3to4(l0); angVel0 = v3to4(a0); linVel1 = v3to4(l1); angVel1 = v3to4(a1);
    V3 m0l, m0a, m1l, m1a;
    art::getLinkMotionVector(*artA, desc.linkIndexA, m0l, m0a);
    art::getLinkMotionVector(*artB, desc.linkIndexB, m1l, m1a);
    linDelta0 = v3to4(m0l); angDelta0 = v3to4(m0a); linDelta1 = v3to4(m1l); angDelta1 = v3to4(m1a);
  } else {
    if (!artA) {
      const SBodyVel& b = vels[desc.bodyA];
      linVel0 = ldf3(b.lin); angVel0 = ldf3(b.ang); linDelta0 = ldf3(b.deltaLinDt); angDelta0 = ldf3(b.deltaAngDt);
    } else {
      V3 l, a, ml, ma;
      art::pxcFsGetVelocity(*artA, desc.linkIndexA, nullptr, l, a);
      art::getLinkMotionVector(*artA, desc.linkIndexA, ml, ma);
      linVel0 = v3to4(l); angVel0 = v3to4(a); linDelta0 = v3to4(ml); angDelta0 = v3to4(ma);
    }
    if (!artB) {
      const SBodyVel& b = vels[desc.bodyB];
      linVel1 = ldf3(b.lin); angVel1 = ldf3(b.ang); linDelta1 = ldf3(b.deltaLinDt); angDelta1 = ldf3(b.deltaAngDt);
    } else {
      V3 l, a, ml, ma;
      art::pxcFsGetVelocity(*artB, desc.linkIndexB, nullptr, l, a);
      art::getLinkMotionVector(*artB, desc.linkIndexB, ml, ma);
      linVel1 = v3to4(l); angVel1 = v3to4(a); linDelta1 = v3to4(ml); angDelta1 = v3to4(ma);
    }
  }
  V4 li0 = V3Zero(), li1 = V3Zero(), ai0 = V3Zero(), ai1 = V3Zero();
  solveExtContactStepCore(arenaPtr<uint8_t>(arena, desc.constraint), desc.constraintLengthOver16, linVel0, linVel1, angVel0, angVel1, linDelta0, linDelta1,
                          angDelta0, angDelta1, li0, li1, ai0, ai1, minPenetration, elapsedTimeF32);
  if (same) {
    art::pxcFsApplyImpulses(*artA, desc.linkIndexA, v4to3(li0), v4to3(ai0), nullptr, desc.linkIndexB, v4to3(li1), v4to3(ai1), nullptr);
  } else {
    if (!artA) {
      stf3(linVel0, vels[desc.bodyA].lin);
      stf3(angVel0, vels[desc.bodyA].ang);
    } else {
      art::pxcFsApplyImpulse(*artA, desc.linkIndexA, v4to3(li0), v4to3(ai0), nullptr);
    }
    if (!artB) {
      stf3(linVel1, vels[desc.bodyB].lin);
      stf3(angVel1, vels[desc.bodyB].ang);
    } else {
      art::pxcFsApplyImpulse(*artB, desc.linkIndexB, v4to3(li1), v4to3(ai1), nullptr);
    }
  }
}

// 분할 안의 EXT_1D 하나 (joints solveExt1DStep 에 ArtRef 를 넘긴다)
SV_HDN void solveExt1DDesc(const SDesc& desc, SBodyVel* vels, const SBodyTxI* txI, art::Articulation* arts, ByteArena& arena, float elapsed, bool posIter,
                           bool conclude) {
  if (desc.constraint == NONE) return;
  uint8_t* blk = arenaPtr<uint8_t>(arena, desc.constraint);
  const art::ArtRef ra{desc.linkIndexA == RIGID_BODY ? nullptr : &arts[desc.bodyA]};
  const art::ArtRef rb{desc.linkIndexB == RIGID_BODY ? nullptr : &arts[desc.bodyB]};
  const bool same = ra.a && ra.a == rb.a;
  jnt::TgsBodyVel b0 = bitCopy<jnt::TgsBodyVel>(vels[desc.linkIndexA == RIGID_BODY ? desc.bodyA : 0u]);
  jnt::TgsBodyVel b1 = bitCopy<jnt::TgsBodyVel>(vels[desc.linkIndexB == RIGID_BODY ? desc.bodyB : 0u]);
  const jnt::TgsTxInertia t0 = bitCopy<jnt::TgsTxInertia>(txI[desc.bodyADataIndex]);
  const jnt::TgsTxInertia t1 = bitCopy<jnt::TgsTxInertia>(txI[desc.bodyBDataIndex]);
  jnt::solveExt1DStep(blk, desc.linkIndexA, desc.linkIndexB, ra, rb, same, &b0, &b1, &t0, &t1, elapsed, posIter);
  if (!same) {
    if (desc.linkIndexA == RIGID_BODY) vels[desc.bodyA] = bitCopy<SBodyVel>(b0);
    if (desc.linkIndexB == RIGID_BODY) vels[desc.bodyB] = bitCopy<SBodyVel>(b1);
  }
  if (conclude) jnt::conclude1DStep(blk);
}

// writeBackContact (:1863) — ext 보폭도. 동역학에 남는 것: broken 이면 이번 스텝 마찰 패치에 표시
SV_HDN void writeBackContactAny(const SDesc& desc, ByteArena& arena, FrictionArena& frictionCur) {
  if (desc.constraint == NONE) return;
  uint8_t* cPtr = arenaPtr<uint8_t>(arena, desc.constraint);
  const uint8_t* last = cPtr + (uint32_t(desc.constraintLengthOver16) << 4);
  while (cPtr < last) {
    const SolverContactHeaderStep* hdr = reinterpret_cast<const SolverContactHeaderStep*>(cPtr);
    cPtr += sizeof(SolverContactHeaderStep);
    const bool ext = hdr->type == SC_TYPE_EXT_CONTACT;
    const uint32_t numNormalConstr = hdr->numNormalConstr;
    const uint32_t numFrictionConstr = hdr->numFrictionConstr;
    cPtr += (ext ? sizeof(SolverContactPointStepExt) : sizeof(SolverContactPointStep)) * numNormalConstr;
    cPtr += sizeof(float) * ((numNormalConstr + 3) & (~3u));
    if (hdr->broken && hdr->frictionBrokenWriteback != NONE) frictionCur.data[hdr->frictionBrokenWriteback].broken = 1;
    cPtr += numFrictionConstr * (ext ? sizeof(SolverContactFrictionStepExt) : sizeof(SolverContactFrictionStep));
  }
}

}  // namespace sv
}  // namespace eng
