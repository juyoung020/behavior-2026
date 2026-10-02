// Sc 층 입력 조각 (문서 15.3, 리드): 모양 세계 자세(변환 캐시 칸)와 경계 상자를 몸체 자세에서 만든다 — PhysX 와 같은 식·순서.
// 원본: simulationcontroller/src/ScShapeSimBase.cpp:215 getAbsPoseAligned, :307·:321 updateCached,
//       common/src/CmTransformUtils.h:39 transformFast·:53 transformInvFast·:77 getStaticGlobalPoseAligned·:103 getDynamicGlobalPoseAligned (aos SSE),
//       lowlevelaabb BoundsArray::updateBounds -> Gu::computeBounds(pose, 0, 1) (contact 모듈 gu_bounds.inc).
// 갱신 규칙 (칸은 아래 식으로 "그때의" 자세에서 계산해 두고, 다음 갱신까지 그대로 든다 — G1 Sc 그림자 replay/g1_sc.cpp 가 radio500_h·시연 0 에서 모든 칸 비트 같음 확인):
//  - 모양 넣기: initSubsystemsDependingOnElementID (ScShapeSimBase.cpp:176) 가 넣는 순간 자세로.
//  - 적분 뒤: 깨어 있고 안 얼린 강체 (ScScene.cpp:186), 이번에 잠든 강체는 스텝 전 자세로 되돌린 뒤 (ScPipeline.cpp:2640 근처),
//    깨어 있는 관절체 링크 (updateArticulationAfterIntegration). 이번 스텝에 잠든 관절체는 고치지 않는다(putToSleep, ScPipeline.cpp:2706)
//    -> 풀이기가 옮긴 링크 자세와 달리 풀이 전 값을 든다.
//  - 사용자 자세 쓰기 (setBody2World -> 모양 더럽힘): 다음 simulate 의 preRigidBodyNarrowPhase (ScPipeline.cpp:199) 에서.
//  - 질량 중심 자세 바꾸기 (setCMassLocalPose, ScBodyCore.cpp:98): body2World·body2Actor 가 바뀌어도 더럽힘 표시가 없다 -> 옛 값을 든다.
// 층 1·2 공용(EHD).
#pragma once
#include "core/contact/px/aos.h"  // contact 모듈의 PhysX aos 번역판 (core/common/aos.h 와는 한 번역 단위에 같이 못 씀)
#include "core/contact/px/gu.h"

namespace eng {
namespace scene {

namespace scaos {
using namespace eng::px::aos;
// CmTransformUtils.h:39
EHD void transformFast(const FloatV wa, const Vec3V va, const Vec3V pa, const FloatV wb, const Vec3V vb, const Vec3V pb, FloatV& wo, Vec3V& vo, Vec3V& po) {
  wo = FSub(FMul(wa, wb), V3Dot(va, vb));
  vo = V3ScaleAdd(va, wb, V3ScaleAdd(vb, wa, V3Cross(va, vb)));
  const Vec3V t1 = V3Scale(pb, FScaleAdd(wa, wa, FLoad(-0.5f)));
  const Vec3V t2 = V3ScaleAdd(V3Cross(va, pb), wa, t1);
  const Vec3V t3 = V3ScaleAdd(va, V3Dot(va, pb), t2);
  po = V3ScaleAdd(t3, FLoad(2.f), pa);
}
// CmTransformUtils.h:53
EHD void transformInvFast(const FloatV wa, const Vec3V va, const Vec3V pa, const FloatV wb, const Vec3V vb, const Vec3V pb, FloatV& wo, Vec3V& vo, Vec3V& po) {
  wo = FScaleAdd(wa, wb, V3Dot(va, vb));
  vo = V3NegScaleSub(va, wb, V3ScaleAdd(vb, wa, V3Cross(vb, va)));
  const Vec3V pt = V3Sub(pb, pa);
  const Vec3V t1 = V3Scale(pt, FScaleAdd(wa, wa, FLoad(-0.5f)));
  const Vec3V t2 = V3ScaleAdd(V3Cross(pt, va), wa, t1);
  const Vec3V t3 = V3ScaleAdd(va, V3Dot(va, pt), t2);
  po = V3Add(t3, t3);
}
}  // namespace scaos

// actor2World * shape2Actor (CmTransformUtils.h:77)
EHD void staticGlobalPose(const px::PxTransform& a2w, const px::PxTransform& s2a, px::PxTransform& out) {
  using namespace scaos;
  alignas(16) float A[8] = {a2w.q.x, a2w.q.y, a2w.q.z, a2w.q.w, a2w.p.x, a2w.p.y, a2w.p.z, 0.f};
  alignas(16) float S[8] = {s2a.q.x, s2a.q.y, s2a.q.z, s2a.q.w, s2a.p.x, s2a.p.y, s2a.p.z, 0.f};
  const Vec3V aPos = V3LoadA(A + 4);
  const QuatV aRot = QuatVLoadA(A);
  const Vec3V sPos = V3LoadA(S + 4);
  const QuatV sRot = QuatVLoadA(S);
  Vec3V v, p;
  FloatV w;
  transformFast(V4GetW(aRot), Vec3V_From_Vec4V(aRot), aPos, V4GetW(sRot), Vec3V_From_Vec4V(sRot), sPos, w, v, p);
  alignas(16) float O[4];
  alignas(16) px::PxVec3 P;
  V4StoreA(V4SetW(v, w), O);
  V3StoreA(p, P);
  out.q = px::PxQuat(O[0], O[1], O[2], O[3]);
  out.p = P;
}
// body2World * body2Actor.getInverse() * shape2Actor (CmTransformUtils.h:103)
EHD void dynamicGlobalPose(const px::PxTransform& b2w, const px::PxTransform& s2a, const px::PxTransform& b2a, px::PxTransform& out) {
  using namespace scaos;
  alignas(16) float W[8] = {b2w.q.x, b2w.q.y, b2w.q.z, b2w.q.w, b2w.p.x, b2w.p.y, b2w.p.z, 0.f};
  alignas(16) float S[8] = {s2a.q.x, s2a.q.y, s2a.q.z, s2a.q.w, s2a.p.x, s2a.p.y, s2a.p.z, 0.f};
  alignas(16) float B[8] = {b2a.q.x, b2a.q.y, b2a.q.z, b2a.q.w, b2a.p.x, b2a.p.y, b2a.p.z, 0.f};
  const Vec3V sPos = V3LoadA(S + 4);
  const QuatV sRot = QuatVLoadA(S);
  const Vec3V bPos = V3LoadA(B + 4);
  const QuatV bRot = QuatVLoadA(B);
  const Vec3V wPos = V3LoadA(W + 4);
  const QuatV wRot = QuatVLoadA(W);
  Vec3V v1, p1, v2, p2;
  FloatV w1, w2;
  transformInvFast(V4GetW(bRot), Vec3V_From_Vec4V(bRot), bPos, V4GetW(sRot), Vec3V_From_Vec4V(sRot), sPos, w1, v1, p1);
  transformFast(V4GetW(wRot), Vec3V_From_Vec4V(wRot), wPos, w1, v1, p1, w2, v2, p2);
  alignas(16) float O[4];
  alignas(16) px::PxVec3 P;
  V4StoreA(V4SetW(v2, w2), O);
  V3StoreA(p2, P);
  out.q = px::PxQuat(O[0], O[1], O[2], O[3]);
  out.p = P;
}

// ShapeSimBase::getAbsPoseAligned (ScShapeSimBase.cpp:215)
//  정적: 모양 자세가 항등(PxShapeCoreFlag::eIDT_TRANSFORM)이면 행위자 자세 그대로, 아니면 staticGlobalPose
//  동적: 몸체-행위자가 항등이 아니면(PxRigidBodyFlag::eRESERVED 가 꺼짐) dynamicGlobalPose, 항등이면 staticGlobalPose(body2World, shape2Actor)
struct ShapePoseIn {
  px::PxTransform shape2Actor;
  uint8_t isStatic, idtShape, idtBody2Actor, pad;
};
EHD void absPose(const ShapePoseIn& s, const px::PxTransform& actorOrBody2World, const px::PxTransform& body2Actor, px::PxTransform& out) {
  if (s.isStatic) {
    if (s.idtShape) {
      out = actorOrBody2World;
      return;
    }
    staticGlobalPose(actorOrBody2World, s.shape2Actor, out);
    return;
  }
  if (!s.idtBody2Actor) {
    dynamicGlobalPose(actorOrBody2World, s.shape2Actor, body2Actor, out);
    return;
  }
  staticGlobalPose(actorOrBody2World, s.shape2Actor, out);
}
// 한 모양 칸 갱신 (ShapeSimBase::updateCached): 변환 캐시 칸과 경계 상자 (Gu::computeBounds(pose, 0, 1))
EHD void updateShapeCached(const ShapePoseIn& s, const px::PxGeometry& geom, const px::PxTransform& actorOrBody2World, const px::PxTransform& body2Actor,
                           px::PxTransform& cacheOut, px::PxBounds3& boundsOut) {
  absPose(s, actorOrBody2World, body2Actor, cacheOut);
  px::Gu::computeBounds(boundsOut, geom, cacheOut, 0.0f, 1.0f);
}

}  // namespace scene
}  // namespace eng
