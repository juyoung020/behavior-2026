// TGS 접촉 제약 준비 (손으로 짬, PhysX 5.6.1 과 비트 동일 목표): 마찰 패치 상관(correlation), 단일 경로 준비.
// 원본 (physx/source/lowleveldynamics/src/)
//   DyFrictionCorrelation.cpp:44-290  createContactPatches / correlatePatches / growPatches
//   DyContactPrepShared.h:53-182      pointsAreClose / isSeparated / getFrictionPatches / extractContacts
//   DyContactPrepShared.h:272-293     computeCompliantContactCoefficientsTGS
//   DyTGSContactPrep.cpp:57-171       computeBlockStreamByteSizesStep / reserveBlockStreams
//   DyTGSContactPrep.cpp:322-816      constructContactConstraintStep / setupFinalizeSolverConstraints
//   DyTGSContactPrep.cpp:1297-1490    createFinalizeSolverContactsStep (두 판)
// 관절체(ext) 접촉 경로는 articulation 과 붙일 때 옮긴다(아직 안 옮김).
#pragma once
#include "saos.h"
#include "tgs_types.h"

namespace eng {
namespace sv {

// ---------------- PxTransform / PxVec3 / PxBounds3 (스칼라) — pmath 위에 PhysX 식 그대로
SV_HD V3 tfTransform(const Tf& t, const V3& v) { return rotate(t.q, v) + t.p; }      // PxTransform::transform(vec)
SV_HD V3 tfTransformInv(const Tf& t, const V3& v) { return rotateInv(t.q, v - t.p); }  // transformInv(vec)
SV_HD Tf tfTransformInv(const Tf& t, const Tf& src) {                                  // PxTransform.h:162
  const Q qinv = conj(t.q);
  return Tf{qinv * src.q, rotate(qinv, src.p - t.p)};
}
SV_HD V3 vmin3(const V3& a, const V3& b) { return V3{pmin(a.x, b.x), pmin(a.y, b.y), pmin(a.z, b.z)}; }
SV_HD V3 vmax3(const V3& a, const V3& b) { return V3{pmax(a.x, b.x), pmax(a.y, b.y), pmax(a.z, b.z)}; }
SV_HD void boundsSetEmpty(Bounds3& b) {
  const float e = kMaxReal * 0.25f;  // PX_MAX_BOUNDS_EXTENTS
  b.minimum = V3{e, e, e};
  b.maximum = V3{-e, -e, -e};
}
SV_HD void boundsInclude(Bounds3& b, const Bounds3& o) {
  b.minimum = vmin3(b.minimum, o.minimum);
  b.maximum = vmax3(b.maximum, o.maximum);
}

// ---------------- 상관 (DyFrictionCorrelation.cpp)
SV_HD void initContactPatch(CorrelationBuffer::ContactPatchData& patch, uint16_t index, float restitution, float staticFriction,
                            float dynamicFriction, uint8_t flags) {
  patch.start = index;
  patch.count = 1;
  patch.next = 0;
  patch.flags = flags;
  patch.restitution = restitution;
  patch.staticFriction = staticFriction;
  patch.dynamicFriction = dynamicFriction;
}

SV_HDN bool createContactPatches(CorrelationBuffer& fb, const ContactPoint* cb, uint32_t contactCount, float normalTolerance) {
  uint32_t contactPatchCount = fb.contactPatchCount;
  if (contactPatchCount == MAX_CONTACTS) return false;
  if (contactCount > 0) {
    CorrelationBuffer::ContactPatchData* currentPatchData = fb.contactPatches + contactPatchCount;
    const ContactPoint* contacts = cb;
    initContactPatch(fb.contactPatches[contactPatchCount++], 0, contacts[0].restitution, contacts[0].staticFriction,
                     contacts[0].dynamicFriction, contacts[0].materialFlags);
    V4 minV = ld4(contacts[0].point, contacts[0].maxImpulse);  // V4LoadA(&point.x): W = maxImpulse
    V4 maxV = minV;
    uint32_t patchIndex = 0;
    uint8_t count = 1;
    for (uint32_t i = 1; i < contactCount; i++) {
      const ContactPoint& curContact = contacts[i];
      const ContactPoint& preContact = contacts[patchIndex];
      if (curContact.staticFriction == preContact.staticFriction && curContact.dynamicFriction == preContact.dynamicFriction &&
          curContact.restitution == preContact.restitution && dot(curContact.normal, preContact.normal) >= normalTolerance) {
        const V4 ptV = ld4(curContact.point, curContact.maxImpulse);
        minV = V4Min(minV, ptV);
        maxV = V4Max(maxV, ptV);
        count++;
      } else {
        if (contactPatchCount == MAX_CONTACTS) return false;
        patchIndex = i;
        currentPatchData->count = count;
        count = 1;
        currentPatchData->patchBounds.minimum = V3{minV.x, minV.y, minV.z};  // StoreBounds
        currentPatchData->patchBounds.maximum = V3{maxV.x, maxV.y, maxV.z};
        currentPatchData = fb.contactPatches + contactPatchCount;
        initContactPatch(fb.contactPatches[contactPatchCount++], uint16_t(i), curContact.restitution, curContact.staticFriction,
                         curContact.dynamicFriction, curContact.materialFlags);
        minV = ld4(curContact.point, curContact.maxImpulse);
        maxV = minV;
      }
    }
    if (count != 1) currentPatchData->count = count;
    currentPatchData->patchBounds.minimum = V3{minV.x, minV.y, minV.z};
    currentPatchData->patchBounds.maximum = V3{maxV.x, maxV.y, maxV.z};
  }
  fb.contactPatchCount = contactPatchCount;
  return true;
}

SV_HD void initFrictionPatch(FrictionPatch& p, const V3& worldNormal, const Tf& body0Pose, const Tf& body1Pose, float restitution,
                             float staticFriction, float dynamicFriction, uint8_t materialFlags) {
  p.body0Normal = rotateInv(body0Pose.q, worldNormal);
  p.body1Normal = rotateInv(body1Pose.q, worldNormal);
  p.relativeQuat = conj(body0Pose.q) * body1Pose.q;
  p.anchorCount = 0;
  p.broken = 0;
  p.staticFriction = staticFriction;
  p.dynamicFriction = dynamicFriction;
  p.restitution = restitution;
  p.materialFlags = materialFlags;
}

SV_HDN bool correlatePatches(CorrelationBuffer& fb, const ContactPoint* cb, const Tf& bodyFrame0, const Tf& bodyFrame1,
                            float normalTolerance, uint32_t startContactPatchIndex, uint32_t startFrictionPatchIndex) {
  bool overflow = false;
  uint32_t frictionPatchCount = fb.frictionPatchCount;
  for (uint32_t i = startContactPatchIndex; i < fb.contactPatchCount; i++) {
    CorrelationBuffer::ContactPatchData& c = fb.contactPatches[i];
    const V3 patchNormal = cb[c.start].normal;
    uint32_t j = startFrictionPatchIndex;
    for (; j < frictionPatchCount &&
           ((dot(patchNormal, fb.frictionPatchWorldNormal[j]) < normalTolerance) || fb.frictionPatches[j].restitution != c.restitution ||
            fb.frictionPatches[j].staticFriction != c.staticFriction || fb.frictionPatches[j].dynamicFriction != c.dynamicFriction);
         j++)
      ;
    if (j == frictionPatchCount) {
      overflow |= j == CorrelationBuffer::MAX_FRICTION_PATCHES;
      if (overflow) continue;
      initFrictionPatch(fb.frictionPatches[frictionPatchCount], patchNormal, bodyFrame0, bodyFrame1, c.restitution, c.staticFriction,
                        c.dynamicFriction, c.flags);
      fb.frictionPatchWorldNormal[j] = patchNormal;
      fb.frictionPatchContactCounts[frictionPatchCount] = c.count;
      fb.patchBounds[frictionPatchCount] = c.patchBounds;
      fb.contactID[frictionPatchCount][0] = 0xffff;
      fb.contactID[frictionPatchCount++][1] = 0xffff;
      c.next = CorrelationBuffer::LIST_END;
    } else {
      boundsInclude(fb.patchBounds[j], c.patchBounds);
      fb.frictionPatchContactCounts[j] += c.count;
      c.next = uint16_t(fb.correlationListHeads[j]);
    }
    fb.correlationListHeads[j] = i;
  }
  fb.frictionPatchCount = frictionPatchCount;
  return overflow;
}

SV_HDN void growPatches(CorrelationBuffer& fb, const ContactPoint* cb, const Tf& bodyFrame0, const Tf& bodyFrame1,
                       uint32_t frictionPatchStartIndex, float frictionOffsetThreshold) {
  for (uint32_t i = frictionPatchStartIndex; i < fb.frictionPatchCount; i++) {
    FrictionPatch& fp = fb.frictionPatches[i];
    if (fp.anchorCount == 2 || fb.correlationListHeads[i] == CorrelationBuffer::LIST_END) {
      const V3 dim = fb.patchBounds[i].maximum - fb.patchBounds[i].minimum;  // getDimensions
      const float frictionPatchDiagonalSq = magSq(dim);
      const float anchorSqDistance = magSq(fp.body0Anchors[0] - fp.body0Anchors[1]);
      if (fb.frictionPatchContactCounts[i] == 0 || (anchorSqDistance * 4.f) >= frictionPatchDiagonalSq) continue;
      fp.anchorCount = 0;
    }
    V3 worldAnchors[2];
    uint16_t anchorCount = 0;
    float pointDistSq = 0.0f, dist0, dist1;
    if (fp.anchorCount == 1) worldAnchors[anchorCount++] = tfTransform(bodyFrame0, fp.body0Anchors[0]);
    const float eps = 1e-8f;
    for (uint32_t patch = fb.correlationListHeads[i]; patch != CorrelationBuffer::LIST_END; patch = fb.contactPatches[patch].next) {
      CorrelationBuffer::ContactPatchData& cp = fb.contactPatches[patch];
      for (uint16_t j = 0; j < cp.count; j++) {
        const V3& worldPoint = cb[cp.start + j].point;
        if (cb[cp.start + j].separation < frictionOffsetThreshold) {
          switch (anchorCount) {
            case 0:
              fb.contactID[i][0] = uint16_t(cp.start + j);
              worldAnchors[0] = worldPoint;
              anchorCount++;
              break;
            case 1:
              pointDistSq = magSq(worldPoint - worldAnchors[0]);
              if (pointDistSq > eps) {
                fb.contactID[i][1] = uint16_t(cp.start + j);
                worldAnchors[1] = worldPoint;
                anchorCount++;
              }
              break;
            default:
              dist0 = magSq(worldPoint - worldAnchors[0]);
              dist1 = magSq(worldPoint - worldAnchors[1]);
              if (dist0 > dist1) {
                if (dist0 > pointDistSq) {
                  fb.contactID[i][1] = uint16_t(cp.start + j);
                  worldAnchors[1] = worldPoint;
                  pointDistSq = dist0;
                }
              } else if (dist1 > pointDistSq) {
                fb.contactID[i][0] = uint16_t(cp.start + j);
                worldAnchors[0] = worldPoint;
                pointDistSq = dist1;
              }
          }
        }
      }
    }
    for (uint32_t j = fp.anchorCount; j < anchorCount; j++) {
      fp.body0Anchors[j] = tfTransformInv(bodyFrame0, worldAnchors[j]);
      fp.body1Anchors[j] = tfTransformInv(bodyFrame1, worldAnchors[j]);
    }
    if (anchorCount == 0) fp.body0Anchors[0] = fp.body1Anchors[0] = V3{0.0f, 0.0f, 0.0f};
    fp.anchorCount = anchorCount;
  }
}

// ---------------- DyContactPrepShared.h
SV_HD bool pointsAreClose(const Tf& body1ToBody0, const V3& localAnchor0, const V3& localAnchor1, const V3& axis, float correlDist) {
  const V3 body0PatchPoint1 = tfTransform(body1ToBody0, localAnchor1);
  return u2f(f2u(dot(localAnchor0 - body0PatchPoint1, axis)) & 0x7fffffffu) < correlDist;  // PxAbs
}
SV_HD bool isSeparated(const FrictionPatch& patch, const Tf& body1ToBody0, float correlationDistance) {
  for (uint32_t a = 0; a < patch.anchorCount; ++a)
    if (!pointsAreClose(body1ToBody0, patch.body0Anchors[a], patch.body1Anchors[a], patch.body0Normal, correlationDistance)) return true;
  return false;
}
// frictionCookie = 지난 스텝 arena 의 패치 배열(없으면 nullptr)
SV_HDN bool getFrictionPatches(CorrelationBuffer& c, const FrictionPatch* frictionCookie, uint32_t frictionPatchCount, const Tf& bodyFrame0,
                              const Tf& bodyFrame1, float correlationDistance) {
  if (frictionCookie == nullptr || frictionPatchCount == 0) return true;
  const FrictionPatch* patches = frictionCookie;
  bool evaluated = false;
  Tf body1ToBody0;
  while (frictionPatchCount--) {
    const FrictionPatch& patch = *patches++;
    if (!patch.broken) {
      if (patch.anchorCount != 0 && !(patch.materialFlags & MAT_DISABLE_STRONG_FRICTION)) {
        if (!evaluated) {
          body1ToBody0 = tfTransformInv(bodyFrame0, bodyFrame1);
          evaluated = true;
        }
        if (dot(patch.body0Normal, rotate(body1ToBody0.q, patch.body1Normal)) > PXC_SAME_NORMAL) {
          if (!isSeparated(patch, body1ToBody0, correlationDistance)) {
            if (c.frictionPatchCount == CorrelationBuffer::MAX_FRICTION_PATCHES) return false;
            c.contactID[c.frictionPatchCount][0] = 0xffff;
            c.contactID[c.frictionPatchCount][1] = 0xffff;
            c.frictionPatchWorldNormal[c.frictionPatchCount] = rotate(bodyFrame0.q, patch.body0Normal);
            c.frictionPatchContactCounts[c.frictionPatchCount] = 0;
            boundsSetEmpty(c.patchBounds[c.frictionPatchCount]);
            c.correlationListHeads[c.frictionPatchCount] = CorrelationBuffer::LIST_END;
            c.frictionPatches[c.frictionPatchCount++] = patch;
          }
        }
      }
    }
  }
  return true;
}

// 접촉 관리자 출력 (PxsContactManagerOutput 단순 스트림)
struct CMOutput {
  const ContactPatchIn* patches;
  const ContactIn* contacts;
  uint32_t nbPatches, nbContacts;
};

// extractContacts (DyContactPrepShared.h:133). 단순 스트림만(수정 가능 스트림은 overflow 로 알린다).
SV_HDN uint32_t extractContacts(ContactPoint* buffer, uint32_t& bufferCount, const CMOutput& np, bool& hasMaxImpulse, bool& hasTargetVelocity,
                               float& invMassScale0, float& invMassScale1, float& invInertiaScale0, float& invInertiaScale1,
                               float defaultMaxImpulse) {
  uint32_t numContacts = bufferCount, origContactCount = bufferCount;
  const ContactPatchIn* patch = np.patches;
  const bool forceNoResponse = patch ? (patch->internalFlags & PATCH_FORCE_NO_RESPONSE) != 0 : true;
  if (!forceNoResponse) {
    invMassScale0 = patch->invMassScale[0];
    invMassScale1 = patch->invMassScale[2];
    invInertiaScale0 = patch->invMassScale[1];
    invInertiaScale1 = patch->invMassScale[3];
    hasMaxImpulse = (patch->internalFlags & PATCH_HAS_MAX_IMPULSE) != 0;
    hasTargetVelocity = (patch->internalFlags & PATCH_HAS_TARGET_VELOCITY) != 0;
    const ContactIn* ct = np.contacts;
    for (uint32_t p = 0; p < np.nbPatches; ++p) {
      const ContactPatchIn& pp = np.patches[p];
      for (uint32_t k = 0; k < pp.nbContacts; ++k, ++ct) {
        const float maxImpulse = hasMaxImpulse ? kMaxReal : defaultMaxImpulse;  // 단순 스트림 getMaxImpulse = PX_MAX_REAL
        if (maxImpulse != 0.f) {
          ContactPoint& d = buffer[numContacts];
          d.normal = pp.normal;
          d.point = ct->point;
          d.separation = ct->separation;
          d.materialFlags = pp.materialFlags;
          d.maxImpulse = maxImpulse;
          d.staticFriction = pp.staticFriction;
          d.dynamicFriction = pp.dynamicFriction;
          d.restitution = pp.restitution;
          d.damping = pp.damping;
          d.targetVel = V3{0.0f, 0.0f, 0.0f};
          d.internalFaceIndex1 = 0xffffffffu;
          ++numContacts;
        }
      }
    }
  }
  bufferCount = numContacts;
  return numContacts - origContactCount;
}

// ---------------- 접촉 기술자 (PxTGSSolverContactDesc 에서 쓰는 것만)
struct TGSContactDesc {
  uint32_t b0, b1;                 // 풀이 몸체 풀 번호 (vel·txI·data 공통)
  Tf bodyFrame0, bodyFrame1;
  float invMassScales[4];          // linear0, angular0, linear1, angular1
  uint8_t bodyState0, bodyState1;
  bool hasForceThresholds, disableStrongFriction, hasMaxImpulse;
  float restDistance, maxCCDSeparation, maxImpulse, torsionalPatchRadius, minTorsionalPatchRadius, offsetSlop;
  // 마찰: 지난 스텝 패치(읽기) -> 이번 스텝 패치(쓰기, frictionPtr = 현 arena 번호)
  const FrictionPatch* frictionPrev;
  uint32_t frictionPrevCount;
  uint32_t frictionPtr;
  uint32_t frictionCount;
  const ContactPoint* contacts;
  uint32_t numContacts;
  // 4개 묶음용
  uint32_t startFrictionPatchIndex, numFrictionPatches, startContactPatchIndex, numContactPatches, axisConstraintCount;
  SDesc* desc;
};

struct PrepCtx {
  SBodyVel* vels;
  SBodyTxI* txI;
  SBodyData* datas;
  ByteArena* constraints;
  FrictionArena* frictionCur;
  CorrelationBuffer* corr;
  ContactPoint* contactBuffer;  // MAX_CONTACTS
  uint32_t contactBufferCount;
};

// DyContactPrepShared.h:272
SV_HD void computeCompliantContactCoefficientsTGS(FV dt, FV restitution, FV damping, FV recipResponse, FV unitResponse, BS accelerationSpring,
                                                  BS isSeparated_, BS collidingWithVrel, FV& velMultiplier, FV& biasCoeff) {
  const FV nrdt = FMul(dt, restitution);
  const FV dampingIfEnabled = FSel(BAndNot(isSeparated_, collidingWithVrel), FZero(), damping);  // computeCompliantDamping
  const FV a = FMul(dt, FSub(dampingIfEnabled, nrdt));
  const FV one = FOne();
  const FV massIfAccelElseOne = FSel(accelerationSpring, recipResponse, one);
  const FV oneIfAccelElseR = FSel(accelerationSpring, one, unitResponse);
  const FV x = FRecip(FScaleAdd(a, oneIfAccelElseR, one));
  velMultiplier = FMul(FMul(x, a), massIfAccelElseOne);
  biasCoeff = FMul(nrdt, FMul(x, oneIfAccelElseR));
}

// DyTGSContactPrep.cpp:322
SV_HD FV constructContactConstraintStep(const M33V& sqrtInvInertia0, const M33V& sqrtInvInertia1, FV invMassNorLenSq0, FV invMassNorLenSq1,
                                        FV angD0, FV angD1, V4 bodyFrame0p, V4 bodyFrame1p, V4 normal, FV norVel0, FV norVel1,
                                        const VecCrossV& norCross, V4 angVel0, V4 angVel1, FV invDtp8, FV invStepDt, FV totalDt,
                                        FV invTotalDt, FV restDistance, FV restitution, FV bounceThreshold, const ContactPoint& contact,
                                        SolverContactPointStep& solverContact, bool isKinematic0, bool isKinematic1, V4 solverOffsetSlop,
                                        FV dt, FV damping, BS accelerationSpring) {
  const FV zero = FZero();
  const V4 point = ldv(contact.point);
  const FV separation = FLoad(contact.separation);
  const FV cTargetVel = V3Dot(normal, ldv(contact.targetVel));
  const V4 ra = V3Sub(point, bodyFrame0p);
  const V4 rb = V3Sub(point, bodyFrame1p);
  V4 raXn = V3Cross(ra, norCross);
  V4 rbXn = V3Cross(rb, norCross);
  const FV angV0 = V3Dot(raXn, angVel0);
  const FV angV1 = V3Dot(rbXn, angVel1);
  const FV vRelAng = FSub(angV0, angV1);
  const FV vRelLin = FSub(norVel0, norVel1);
  const V4 slop = V3Scale(solverOffsetSlop, FMax(FSel(FIsEq(vRelLin, zero), FMax(), FDiv(vRelAng, vRelLin)), FOne()));
  const FV vrel1 = FAdd(norVel0, angV0);
  const FV vrel2 = FAdd(norVel1, angV1);
  const FV vrel = FSub(vrel1, vrel2);
  raXn = V3Sel(V3IsGrtr(slop, V3Abs(raXn)), V3Zero(), raXn);
  rbXn = V3Sel(V3IsGrtr(slop, V3Abs(rbXn)), V3Zero(), rbXn);
  const V4 raXnInertia = M33MulV3(sqrtInvInertia0, raXn);
  const V4 rbXnInertia = M33MulV3(sqrtInvInertia1, rbXn);
  const FV i0 = FMul(V3Dot(raXnInertia, raXnInertia), angD0);
  const FV i1 = FMul(V3Dot(rbXnInertia, rbXnInertia), angD1);
  const FV resp0 = FAdd(invMassNorLenSq0, i0);
  const FV resp1 = FSub(i1, invMassNorLenSq1);
  const FV unitResponse = FAdd(resp0, resp1);
  const FV penetration = FSub(separation, restDistance);
  const BS isSep = FIsGrtr(penetration, zero);
  const FV penetrationInvDt = FMul(penetration, invTotalDt);
  const BS isGreater2 = BAnd(BAnd(FIsGrtr(restitution, zero), FIsGrtr(bounceThreshold, vrel)), FIsGrtr(FNeg(vrel), penetrationInvDt));
  const FV ratio = FAdd(totalDt, FSel(isGreater2, FDiv(penetration, vrel), FNeg(totalDt)));
  const FV recipResponse = FSel(FIsGrtr(unitResponse, zero), FRecip(unitResponse), zero);
  FV biasCoeff, velMultiplier;
  if (FAllGrtr(zero, restitution)) {
    const BS collidingWithVrel = FIsGrtr(FNeg(vrel), penetrationInvDt);
    computeCompliantContactCoefficientsTGS(dt, restitution, damping, recipResponse, unitResponse, accelerationSpring, isSep,
                                           collidingWithVrel, velMultiplier, biasCoeff);
  } else {
    biasCoeff = FNeg(FSel(isSep, invStepDt, invDtp8));
    velMultiplier = recipResponse;
  }
  FV totalError = penetration;
  const FV sumVRel = vrel;
  FV targetVelocity = FAdd(cTargetVel, FSel(isGreater2, FMul(FNeg(sumVRel), restitution), zero));
  totalError = FScaleAdd(targetVelocity, ratio, totalError);
  if (isKinematic0) targetVelocity = FSub(targetVelocity, vrel1);
  if (isKinematic1) targetVelocity = FAdd(targetVelocity, vrel2);
  stv(raXnInertia, solverContact.raXnI);
  stv(rbXnInertia, solverContact.rbXnI);
  FStore(velMultiplier, &solverContact.velMultiplier);
  FStore(totalError, &solverContact.separation);
  FStore(biasCoeff, &solverContact.biasCoefficient);
  FStore(targetVelocity, &solverContact.targetVelocity);
  FStore(recipResponse, &solverContact.recipResponse);
  solverContact.maxImpulse = contact.maxImpulse;
  return penetration;
}

SV_HD M33V m33v(const M33& m) {  // Mat33V(V3LoadU_SafeReadW(col0), V3LoadU_SafeReadW(col1), V3LoadU(col2)) — W 모두 +0
  return M33V{ldv(m.c0), ldv(m.c1), ldv(m.c2)};
}

// DyTGSContactPrep.cpp:426 setupFinalizeSolverConstraints (강체-강체/정적)
SV_HDN void setupFinalizeSolverConstraints(const PrepCtx& P, const TGSContactDesc& contactDesc, const CorrelationBuffer& c, uint8_t* workspace,
                                          const SBodyVel& b0, const SBodyVel& b1, float invDtF32, float totalDtF32, float invTotalDtF32,
                                          float dtF32, float bounceThresholdF32, bool hasForceThreshold, bool staticOrKinematicBody,
                                          uint32_t frictionDataPtr, bool disableStrongFriction, float biasCoefficient) {
  const bool hasTorsionalFriction = contactDesc.torsionalPatchRadius > 0.0f || contactDesc.minTorsionalPatchRadius > 0.0f;
  const bool isKinematic0 = b0.isKinematic;
  const bool isKinematic1 = b1.isKinematic;
  const SBodyData& data0 = P.datas[contactDesc.b0];
  const SBodyData& data1 = P.datas[contactDesc.b1];
  const uint8_t flags = uint8_t(hasForceThreshold ? 1 : 0);  // eHAS_FORCE_THRESHOLDS
  const uint8_t type = staticOrKinematicBody ? SC_TYPE_STATIC_CONTACT : SC_TYPE_RB_CONTACT;
  const FV zero = FZero();
  const FV d0 = FLoad(contactDesc.invMassScales[0]);
  const FV d1 = FLoad(contactDesc.invMassScales[2]);
  const FV angD0 = FLoad(contactDesc.invMassScales[1]);
  const FV angD1 = FLoad(contactDesc.invMassScales[3]);
  const V4 offsetSlop = V3Load(contactDesc.offsetSlop);
  const FV nDom1fV = FNeg(d1);
  const FV invMass0 = FLoad(data0.invMass);
  const FV invMass1 = FLoad(data1.invMass);
  const FV invMass0_dom0fV = FMul(d0, invMass0);
  const FV invMass1_dom1fV = FMul(nDom1fV, invMass1);
  V4 sfdf = V4Zero();  // staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W
  sfdf = V4SetZ(sfdf, invMass0_dom0fV);
  sfdf = V4SetW(sfdf, invMass1_dom1fV);
  const FV restDistance = FLoad(contactDesc.restDistance);
  const float maxPenBias = pmax(data0.penBiasClamp, data1.penBiasClamp);
  const V4 bodyFrame0p = ldv(contactDesc.bodyFrame0.p);
  const V4 bodyFrame1p = ldv(contactDesc.bodyFrame1.p);
  const V4 bodyFrame0q = V4{contactDesc.bodyFrame0.q.x, contactDesc.bodyFrame0.q.y, contactDesc.bodyFrame0.q.z, contactDesc.bodyFrame0.q.w};
  const V4 bodyFrame1q = V4{contactDesc.bodyFrame1.q.x, contactDesc.bodyFrame1.q.y, contactDesc.bodyFrame1.q.z, contactDesc.bodyFrame1.q.w};
  uint32_t frictionPatchWritebackAddrIndex = 0;
  const V4 linVel0 = ldv(data0.originalLinearVelocity);
  const V4 linVel1 = ldv(data1.originalLinearVelocity);
  const V4 angVel0 = ldv(data0.originalAngularVelocity);
  const V4 angVel1 = ldv(data1.originalAngularVelocity);
  const M33V sqrtInvInertia0 = m33v(P.txI[contactDesc.b0].sqrtInvInertia);
  const M33V sqrtInvInertia1 = m33v(P.txI[contactDesc.b1].sqrtInvInertia);
  const FV invDt = FLoad(invDtF32);
  const FV dt = FLoad(dtF32);
  const FV totalDt = FLoad(totalDtF32);
  const FV invTotalDt = FLoad(invTotalDtF32);
  const float scale = pmin(0.8f, biasCoefficient);
  const FV p8 = FLoad(scale);
  const FV bounceThreshold = FLoad(bounceThresholdF32);
  const FV invDtp8 = FMul(invDt, p8);
  const float frictionBiasScale = disableStrongFriction ? 0.f : invDtF32 * scale;
  const ContactPoint* buffer = contactDesc.contacts;
  uint8_t* ptr = workspace;
  for (uint32_t i = 0; i < c.frictionPatchCount; i++) {
    const uint32_t contactCount = c.frictionPatchContactCounts[i];
    if (contactCount == 0) continue;
    const FrictionPatch& frictionPatch = c.frictionPatches[i];
    const uint32_t firstPatch = c.correlationListHeads[i];
    const ContactPoint* contactBase0 = buffer + c.contactPatches[firstPatch].start;
    const float combinedRestitution = contactBase0->restitution;
    const float combinedDamping = contactBase0->damping;
    SolverContactHeaderStep* header = reinterpret_cast<SolverContactHeaderStep*>(ptr);
    ptr += sizeof(SolverContactHeaderStep);
    header->flags = flags;
    header->minNormalForce = 0.f;
    FStore(invMass0_dom0fV, &header->invMass0);
    FStore(FNeg(invMass1_dom1fV), &header->invMass1);
    const FV restitution = FLoad(combinedRestitution);
    const FV damping = FLoad(combinedDamping);
    const V4 normal = ldv(buffer[c.contactPatches[c.correlationListHeads[i]].start].normal);
    const FV normalLenSq = V3LengthSq(normal);
    const VecCrossV norCross = V3PrepareCross(normal);
    const FV norVel0 = V3Dot(linVel0, normal);
    const FV norVel1 = V3Dot(linVel1, normal);
    const FV invMassNorLenSq0 = FMul(invMass0_dom0fV, normalLenSq);
    const FV invMassNorLenSq1 = FMul(invMass1_dom1fV, normalLenSq);
    stv(normal, header->normal);
    header->maxPenBias = contactBase0->restitution < 0.f ? -kMaxReal : maxPenBias;
    FV maxPenetration = FMax();
    const BS accelSpring = BLoad(!!(contactBase0->materialFlags & MAT_COMPLIANT_ACCELERATION_SPRING));
    for (uint32_t patch = c.correlationListHeads[i]; patch != CorrelationBuffer::LIST_END; patch = c.contactPatches[patch].next) {
      const uint32_t count = c.contactPatches[patch].count;
      const ContactPoint* contactBase = buffer + c.contactPatches[patch].start;
      uint8_t* p = ptr;
      for (uint32_t j = 0; j < count; j++) {
        const ContactPoint& contact = contactBase[j];
        SolverContactPointStep* solverContact = reinterpret_cast<SolverContactPointStep*>(p);
        p += sizeof(SolverContactPointStep);
        maxPenetration = FMin(maxPenetration, constructContactConstraintStep(
                                                  sqrtInvInertia0, sqrtInvInertia1, invMassNorLenSq0, invMassNorLenSq1, angD0, angD1, bodyFrame0p,
                                                  bodyFrame1p, normal, norVel0, norVel1, norCross, angVel0, angVel1, invDtp8, invDt, totalDt,
                                                  invTotalDt, restDistance, restitution, bounceThreshold, contact, *solverContact, isKinematic0,
                                                  isKinematic1, offsetSlop, dt, damping, accelSpring));
      }
      ptr = p;
    }
    float* forceBuffers = reinterpret_cast<float*>(ptr);
    for (uint32_t k = 0; k < contactCount; ++k) forceBuffers[k] = 0.0f;
    ptr += ((contactCount + 3) & (~3u)) * sizeof(float);
    const float staticFriction = contactBase0->staticFriction;
    const float dynamicFriction = contactBase0->dynamicFriction;
    const bool disableFriction = !!(contactBase0->materialFlags & MAT_DISABLE_FRICTION);
    sfdf = V4SetX(sfdf, FLoad(staticFriction));
    sfdf = V4SetY(sfdf, FLoad(dynamicFriction));
    const bool haveFriction = (disableFriction == 0 && frictionPatch.anchorCount != 0);
    header->numNormalConstr = uint8_t(contactCount);
    header->numFrictionConstr = uint8_t(haveFriction ? frictionPatch.anchorCount * 2 : 0);
    header->type = type;
    header->staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W = sfdf;
    FStore(angD0, &header->angDom0);
    FStore(angD1, &header->angDom1);
    header->broken = 0;
    header->frictionBrokenWriteback = NONE;
    if (haveFriction) {
      const V4 linVrel = V3Sub(linVel0, linVel1);
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
      const V4 t1 = V3Normalize(V3Cross(norCross, t0Cross));
      const uint32_t writeback = frictionDataPtr == NONE ? NONE : frictionDataPtr + frictionPatchWritebackAddrIndex;
      const FV norVel00 = V3Dot(linVel0, t0);
      const FV norVel01 = V3Dot(linVel1, t0);
      const FV norVel10 = V3Dot(linVel0, t1);
      const FV norVel11 = V3Dot(linVel1, t1);
      const V4 relTr = V3Sub(bodyFrame0p, bodyFrame1p);
      header->frictionBrokenWriteback = writeback;
      const float frictionScale = (frictionPatch.anchorCount == 2) ? 0.5f : 1.f;
      for (uint32_t j = 0; j < frictionPatch.anchorCount; j++) {
        SolverContactFrictionStep* f0 = reinterpret_cast<SolverContactFrictionStep*>(ptr);
        ptr += sizeof(SolverContactFrictionStep);
        SolverContactFrictionStep* f1 = reinterpret_cast<SolverContactFrictionStep*>(ptr);
        ptr += sizeof(SolverContactFrictionStep);
        const V4 body0Anchor = ldv(frictionPatch.body0Anchors[j]);
        const V4 body1Anchor = ldv(frictionPatch.body1Anchors[j]);
        const V4 ra = QuatRotate(bodyFrame0q, body0Anchor);
        const V4 rb = QuatRotate(bodyFrame1q, body1Anchor);
        uint32_t index = c.contactID[i][j];
        index = index == 0xFFFF ? c.contactPatches[c.correlationListHeads[i]].start : index;
        const V4 tvel = ldv(buffer[index].targetVel);
        const V4 error = V3Add(V3Sub(ra, rb), relTr);
        {
          V4 raXn = V3Cross(ra, t0);
          V4 rbXn = V3Cross(rb, t0);
          raXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(raXn)), V3Zero(), raXn);
          rbXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(rbXn)), V3Zero(), rbXn);
          const V4 raXnInertia = M33MulV3(sqrtInvInertia0, raXn);
          const V4 rbXnInertia = M33MulV3(sqrtInvInertia1, rbXn);
          const FV resp0 = FAdd(invMassNorLenSq0, FMul(V3Dot(raXnInertia, raXnInertia), angD0));
          const FV resp1 = FSub(FMul(V3Dot(rbXnInertia, rbXnInertia), angD1), invMassNorLenSq1);
          const FV unitResponse = FAdd(resp0, resp1);
          const FV velMultiplier = FSel(FIsGrtr(unitResponse, zero), FDiv(p8, unitResponse), zero);
          FV targetVel = V3Dot(tvel, t0);
          if (isKinematic0) targetVel = FSub(targetVel, FAdd(norVel00, V3Dot(raXn, angVel0)));
          if (isKinematic1) targetVel = FAdd(targetVel, FAdd(norVel01, V3Dot(rbXn, angVel1)));
          f0->normalXYZ_ErrorW = V4SetW(t0, V3Dot(error, t0));
          f0->raXnI_targetVelW = V4SetW(raXnInertia, targetVel);
          f0->rbXnI_velMultiplierW = V4SetW(rbXnInertia, velMultiplier);
          f0->appliedForce = 0.f;
          f0->frictionScale = frictionScale;
          f0->biasScale = frictionBiasScale;
        }
        {
          FV targetVel = V3Dot(tvel, t1);
          V4 raXn = V3Cross(ra, t1);
          V4 rbXn = V3Cross(rb, t1);
          raXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(raXn)), V3Zero(), raXn);
          rbXn = V3Sel(V3IsGrtr(offsetSlop, V3Abs(rbXn)), V3Zero(), rbXn);
          const V4 raXnInertia = M33MulV3(sqrtInvInertia0, raXn);
          const V4 rbXnInertia = M33MulV3(sqrtInvInertia1, rbXn);
          const FV resp0 = FAdd(invMassNorLenSq0, FMul(V3Dot(raXnInertia, raXnInertia), angD0));
          const FV resp1 = FSub(FMul(V3Dot(rbXnInertia, rbXnInertia), angD1), invMassNorLenSq1);
          const FV unitResponse = FAdd(resp0, resp1);
          const FV velMultiplier = FSel(FIsGrtr(unitResponse, zero), FDiv(p8, unitResponse), zero);
          if (isKinematic0) targetVel = FSub(targetVel, FAdd(norVel10, V3Dot(raXn, angVel0)));
          if (isKinematic1) targetVel = FAdd(targetVel, FAdd(norVel11, V3Dot(rbXn, angVel1)));
          f1->normalXYZ_ErrorW = V4SetW(t1, V3Dot(error, t1));
          f1->raXnI_targetVelW = V4SetW(raXnInertia, targetVel);
          f1->rbXnI_velMultiplierW = V4SetW(rbXnInertia, velMultiplier);
          f1->appliedForce = 0.f;
          f1->frictionScale = frictionScale;
          f1->biasScale = frictionBiasScale;
        }
      }
      if (hasTorsionalFriction && frictionPatch.anchorCount == 1) {
        const FV torsionalPatchRadius = FLoad(contactDesc.torsionalPatchRadius);
        const FV minTorsionalPatchRadius = FLoad(contactDesc.minTorsionalPatchRadius);
        const FV torsionalFriction = FMax(minTorsionalPatchRadius, FSqrt(FMul(FMax(zero, FNeg(maxPenetration)), torsionalPatchRadius)));
        header->numFrictionConstr++;
        SolverContactFrictionStep* f = reinterpret_cast<SolverContactFrictionStep*>(ptr);
        ptr += sizeof(SolverContactFrictionStep);
        const V4 raXnInertia = M33MulV3(sqrtInvInertia0, normal);
        const V4 rbXnInertia = M33MulV3(sqrtInvInertia1, normal);
        const FV resp0 = FMul(V3Dot(raXnInertia, raXnInertia), angD0);
        const FV resp1 = FMul(V3Dot(rbXnInertia, rbXnInertia), angD1);
        const FV unitResponse = FAdd(resp0, resp1);
        const FV velMultiplier = FSel(FIsGrtr(unitResponse, zero), FDiv(p8, unitResponse), zero);
        FV targetVel = zero;
        if (isKinematic0) targetVel = V3Dot(normal, angVel0);
        if (isKinematic1) targetVel = V3Dot(normal, angVel1);
        f->normalXYZ_ErrorW = V4Zero();
        f->raXnI_targetVelW = V4SetW(raXnInertia, targetVel);
        f->rbXnI_velMultiplierW = V4SetW(rbXnInertia, velMultiplier);
        f->biasScale = 0.f;
        f->appliedForce = 0.f;
        FStore(torsionalFriction, &f->frictionScale);
      }
    }
    frictionPatchWritebackAddrIndex++;
  }
}

// DyTGSContactPrep.cpp:57 computeBlockStreamByteSizesStep (강체만: useExtContacts = false)
SV_HD void computeBlockStreamByteSizesStep(const CorrelationBuffer& c, uint32_t& _solverConstraintByteSize, uint32_t& _numFrictionPatches,
                                           uint32_t& _axisConstraintCount, float torsionalPatchRadius) {
  uint32_t solverConstraintByteSize = 0, numFrictionPatches = 0, axisConstraintCount = 0;
  for (uint32_t i = 0; i < c.frictionPatchCount; i++) {
    if (c.correlationListHeads[i] != CorrelationBuffer::LIST_END) numFrictionPatches++;
    const FrictionPatch& frictionPatch = c.frictionPatches[i];
    const bool haveFriction = (frictionPatch.materialFlags & MAT_DISABLE_FRICTION) == 0;
    if (c.frictionPatchContactCounts[i] != 0) {
      solverConstraintByteSize += sizeof(SolverContactHeaderStep);
      solverConstraintByteSize += c.frictionPatchContactCounts[i] * sizeof(SolverContactPointStep);
      solverConstraintByteSize += sizeof(float) * ((c.frictionPatchContactCounts[i] + 3) & (~3u));
      axisConstraintCount += c.frictionPatchContactCounts[i];
      if (haveFriction) {
        uint32_t nbAnchors = uint32_t(c.frictionPatches[i].anchorCount * 2);
        if (torsionalPatchRadius > 0.f && c.frictionPatches[i].anchorCount == 1) nbAnchors++;
        solverConstraintByteSize += nbAnchors * sizeof(SolverContactFrictionStep);
        axisConstraintCount += nbAnchors;
      }
    }
  }
  _numFrictionPatches = numFrictionPatches;
  _axisConstraintCount = axisConstraintCount;
  _solverConstraintByteSize = ((solverConstraintByteSize + 0x0f) & ~0x0fu);
}

// DyTGSContactPrep.cpp:1297 createFinalizeSolverContactsStep (상관 버퍼판)
SV_HDN bool createFinalizeSolverContactsStepCorr(PrepCtx& P, TGSContactDesc& contactDesc, CorrelationBuffer& c, float invDtF32, float invTotalDtF32,
                                                float totalDtF32, float dtF32, float bounceThresholdF32, float frictionOffsetThreshold,
                                                float correlationDistance, float biasCoefficient) {
  c.frictionPatchCount = 0;
  c.contactPatchCount = 0;
  const bool hasForceThreshold = contactDesc.hasForceThresholds;
  const bool staticOrKinematicBody = contactDesc.bodyState1 == BS_KINEMATIC || contactDesc.bodyState1 == BS_STATIC;
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
    getFrictionPatches(c, contactDesc.frictionPrev, contactDesc.frictionPrevCount, contactDesc.bodyFrame0, contactDesc.bodyFrame1,
                       correlationDistance);
  bool overflow = !createContactPatches(c, contactDesc.contacts, contactDesc.numContacts, PXC_SAME_NORMAL);
  overflow = correlatePatches(c, contactDesc.contacts, contactDesc.bodyFrame0, contactDesc.bodyFrame1, PXC_SAME_NORMAL, 0, 0) || overflow;
  (void)overflow;
  growPatches(c, contactDesc.contacts, contactDesc.bodyFrame0, contactDesc.bodyFrame1, 0, frictionOffsetThreshold + contactDesc.restDistance);
  uint32_t solverConstraintByteSize = 0, numFrictionPatches = 0, axisConstraintCount = 0;
  computeBlockStreamByteSizesStep(c, solverConstraintByteSize, numFrictionPatches, axisConstraintCount,
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
      setupFinalizeSolverConstraints(P, contactDesc, c, solverConstraint, P.vels[contactDesc.b0], P.vels[contactDesc.b1], invDtF32, totalDtF32,
                                     invTotalDtF32, dtF32, bounceThresholdF32, hasForceThreshold, staticOrKinematicBody, frictionOff,
                                     disableStrongFriction, biasCoefficient);
      *reinterpret_cast<uint32_t*>(solverConstraint + solverConstraintByteSize) = 0;
    }
  }
  return successfulReserve;
}

// DyTGSContactPrep.cpp:1441 createFinalizeSolverContactsStep (접촉 관리자 출력판)
SV_HDN bool createFinalizeSolverContactsStep(PrepCtx& P, TGSContactDesc& contactDesc, const CMOutput& output, float invDtF32, float invTotalDt,
                                            float totalDtF32, float dt, float bounceThresholdF32, float frictionOffsetThreshold,
                                            float correlationDistance, float biasCoefficient) {
  P.contactBufferCount = 0;
  uint32_t numContacts = 0;
  {
    float invMassScale0 = 1.f, invMassScale1 = 1.f, invInertiaScale0 = 1.f, invInertiaScale1 = 1.f;
    contactDesc.invMassScales[1] = (contactDesc.bodyState0 != BS_ARTICULATION && P.vels[contactDesc.b0].isKinematic) ? 0.f : contactDesc.invMassScales[1];
    contactDesc.invMassScales[3] = (contactDesc.bodyState1 != BS_ARTICULATION && P.vels[contactDesc.b1].isKinematic) ? 0.f : contactDesc.invMassScales[3];
    bool hasMaxImpulse = false, hasTargetVelocity = false;
    numContacts = extractContacts(P.contactBuffer, P.contactBufferCount, output, hasMaxImpulse, hasTargetVelocity, invMassScale0, invMassScale1,
                                  invInertiaScale0, invInertiaScale1, contactDesc.maxImpulse);
    contactDesc.contacts = P.contactBuffer;
    contactDesc.numContacts = numContacts;
    contactDesc.disableStrongFriction = contactDesc.disableStrongFriction || hasTargetVelocity;
    contactDesc.hasMaxImpulse = hasMaxImpulse;
    contactDesc.invMassScales[0] *= invMassScale0;
    contactDesc.invMassScales[2] *= invMassScale1;
    contactDesc.invMassScales[1] *= invInertiaScale0;
    contactDesc.invMassScales[3] *= invInertiaScale1;
  }
  return createFinalizeSolverContactsStepCorr(P, contactDesc, *P.corr, invDtF32, invTotalDt, totalDtF32, dt, bounceThresholdF32,
                                              frictionOffsetThreshold, correlationDistance, biasCoefficient);
}

}  // namespace sv
}  // namespace eng
