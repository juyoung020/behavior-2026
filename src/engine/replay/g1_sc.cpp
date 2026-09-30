// G1 Sc 층 입력 그림자 (문서 15.3, 리드, 비계 전용): 넓은 단계가 읽는 순간(작업 ScScene.broadPhaseFirstPass 직전) PhysX 의
// 변환 캐시(모양 세계 자세)와 경계 상자 배열을, 우리 식(core/scene/sc_inputs.h: getAbsPoseAligned + Gu::computeBounds)으로
// 그 순간의 행위자·몸체 자세에서 모든 모양에 대해 다시 만들어 비트 비교한다. (어느 칸을 언제 고치는지 = 갱신 규칙은 이 비교가 모든 칸 같음을 보이면
// "그 순간 자세로 다시 계산한 값 = PhysX 값" 이 성립한다는 뜻.) 바뀜 비트맵·접촉 거리도 같이 본다.
// 켜기: G1_SC=1 (디스패처 필요).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../tests/solver/px_internal.h"
#include "../tests/articulation/px_art_internal.h"
#include "GuConvexMesh.h"
#include "NpShape.h"
#include "NpArticulationLink.h"
#include "ScShapeSim.h"
#include "ScStaticSim.h"
#include "core/contact/narrowphase.h"
#include "g1_hooks.h"

using namespace physx;
namespace ep = eng::px;

void g1sc_update(const float* s2a, const unsigned char* flags, const float* a2w, const float* b2a, const eng::contact::ShapeGeom& g, float* pose, float* bounds);

namespace {

struct ScShadow {
  bool inited = false, on = false;
  PxScene* scene = nullptr;
  uint64_t sim = 0, steps = 0, shapes = 0, badPose = 0, badBounds = 0, badDist = 0, unsup = 0, holds = 0, heldShapes = 0, artHolds = 0;
  long long firstPose = -1, firstBounds = -1;
  int show = 0;
  long long trace = -1, traceFrom = 0, traceTo = -1;
} SS;

template <class T, class... Args>
void zput(T& dst, Args... args) {
  alignas(T) unsigned char buf[sizeof(T)] = {};
  new (buf) T(args...);
  memcpy(static_cast<void*>(&dst), buf, sizeof(T));
}

// PhysX 기하 -> 우리 기하 (볼록은 PhysX 가 구운 덩어리를 그대로 가리킴: 배치 같음)
bool geomOf(const PxGeometry& pg, eng::contact::ShapeGeom& o) {
  switch (pg.getType()) {
    case PxGeometryType::eSPHERE: o.type = eng::contact::eSPHERE; zput(o.sphere, static_cast<const PxSphereGeometry&>(pg).radius); return true;
    case PxGeometryType::ePLANE: o.type = eng::contact::ePLANE; zput(o.plane); return true;
    case PxGeometryType::eCAPSULE: {
      const auto& g = static_cast<const PxCapsuleGeometry&>(pg);
      o.type = eng::contact::eCAPSULE; zput(o.capsule, g.radius, g.halfHeight); return true;
    }
    case PxGeometryType::eBOX: {
      const auto& g = static_cast<const PxBoxGeometry&>(pg);
      o.type = eng::contact::eBOX; zput(o.box, g.halfExtents.x, g.halfExtents.y, g.halfExtents.z); return true;
    }
    case PxGeometryType::eCONVEXMESH: {
      const auto& g = static_cast<const PxConvexMeshGeometry&>(pg);
      ep::PxMeshScale s;
      s.scale = ep::PxVec3(g.scale.scale.x, g.scale.scale.y, g.scale.scale.z);
      s.rotation = ep::PxQuat(g.scale.rotation.x, g.scale.rotation.y, g.scale.rotation.z, g.scale.rotation.w);
      const auto* hull = reinterpret_cast<const ep::Gu::ConvexHullData*>(&static_cast<const Gu::ConvexMesh*>(g.convexMesh)->getHull());
      o.type = eng::contact::eCONVEXMESH; zput(o.convex, hull, s, g.meshFlags); return true;
    }
    default: return false;
  }
}

std::vector<PxRigidActor*> actorsOf(PxScene* scene) {
  std::vector<PxRigidActor*> out;
  const PxU32 na = scene->getNbActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC);
  std::vector<PxActor*> acts(na);
  scene->getActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC, acts.data(), na);
  for (PxActor* a : acts) out.push_back(static_cast<PxRigidActor*>(a));
  const PxU32 nArt = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(nArt);
  scene->getArticulations(arts.data(), nArt);
  for (PxArticulationReducedCoordinate* art : arts) {
    std::vector<PxArticulationLink*> links(art->getNbLinks());
    art->getLinks(links.data(), PxU32(links.size()));
    for (PxArticulationLink* l : links) out.push_back(l);
  }
  return out;
}

// 한 모양을 지금 자세로 다시 계산 (우리 식). false = 못 옮긴 기하
bool calc(Sc::ShapeSim* ssim, float* P7, float* BB, unsigned char* fl) {
  const PxsShapeCore& pc = ssim->getCore().getCore();
  fl[0] = fl[2] = 0;
  const PxTransform s2a = pc.getTransform();
  fl[1] = pc.mShapeCoreFlags.isSet(PxShapeCoreFlag::eIDT_TRANSFORM) ? 1 : 0;
  PxTransform a2w, b2a(PxIdentity);
  Sc::ActorSim& as = ssim->getActor();
  if (as.getActorType() == PxActorType::eRIGID_STATIC) {
    fl[0] = 1;
    a2w = static_cast<Sc::StaticSim&>(as).getStaticCore().getCore().body2World;
  } else {
    const PxsBodyCore& bc = static_cast<Sc::BodySim&>(as).getBodyCore().getCore();
    fl[2] = bc.hasIdtBody2Actor() ? 1 : 0;
    a2w = bc.body2World;
    b2a = bc.getBody2Actor();
  }
  eng::contact::ShapeGeom g;
  memset(static_cast<void*>(&g), 0, sizeof(g));
  if (!geomOf(pc.mGeometry.getGeometry(), g)) return false;
  auto f7 = [](const PxTransform& t, float* o) { o[0] = t.q.x; o[1] = t.q.y; o[2] = t.q.z; o[3] = t.q.w; o[4] = t.p.x; o[5] = t.p.y; o[6] = t.p.z; };
  float S7[7], A7[7], B7[7];
  f7(s2a, S7);
  f7(a2w, A7);
  f7(b2a, B7);
  g1sc_update(S7, fl, A7, B7, g, P7, BB);
  return true;
}

// 갱신 규칙 모형: 질량 중심 자세 바꾸기(setCMassLocalPose, ScBodyCore.cpp:98)는 body2World·body2Actor 를 고치지만 모양을 더럽힘 표시하지 않는다
// -> 그 몸체 모양의 캐시 칸은 옛 값(바꾸기 직전 자세로 계산한 값)을 그대로 든다. 다음 갱신(적분 뒤 깨어 있고 안 얼린 몸체, setBody2World 의 더럽힘)에서 풀린다.
struct Held {
  float pose[7], bounds[6];
  Sc::BodySim* body;
};
std::map<PxU32, Held> gHeld;
void holdBody(Sc::BodySim* b) {
  Sc::ElementSim** el = b->getElements();
  for (PxU32 k = 0, n = b->getNbElements(); k < n; ++k) {
    Sc::ShapeSim* ss = static_cast<Sc::ShapeSim*>(el[k]);
    Held h;
    h.body = b;
    unsigned char fl[3];
    if (!calc(ss, h.pose, h.bounds, fl)) continue;
    if (!gHeld.count(ss->getElementID())) gHeld[ss->getElementID()] = h;  // 이미 들고 있으면 더 옛 값 유지
    ++SS.holds;
  }
}
void releaseBody(Sc::BodySim* b) {
  Sc::ElementSim** el = b->getElements();
  for (PxU32 k = 0, n = b->getNbElements(); k < n; ++k) gHeld.erase(el[k]->getElementID());
}

void check() {
  Sc::Scene& sc = static_cast<NpScene*>(SS.scene)->getScScene();
  PxsTransformCache& tc = sc.getLowLevelContext()->getTransformCache();
  Bp::BoundsArray& ba = sc.getBoundsArray();
  const float* cd = sc.getLowLevelContext()->getContactDistances();
  ++SS.steps;
  for (PxRigidActor* a : actorsOf(SS.scene)) {
    const PxU32 n = a->getNbShapes();
    std::vector<PxShape*> sh(n);
    a->getShapes(sh.data(), n);
    for (PxShape* s : sh) {
      Sc::ShapeCore& core = static_cast<NpShape*>(s)->getCore();
      Sc::ShapeSim* ssim = core.getExclusiveSim();
      if (!ssim) continue;
      const PxU32 e = ssim->getElementID();
      const PxsShapeCore& pc = core.getCore();
      float P7[7], BB[6];
      unsigned char fl[3];
      if (!calc(ssim, P7, BB, fl)) { ++SS.unsup; continue; }
      auto hi = gHeld.find(e);
      if (hi != gHeld.end()) {
        ++SS.heldShapes;
        memcpy(P7, hi->second.pose, 28);
        memcpy(BB, hi->second.bounds, 24);
      }
      struct { float q[4], p[3]; } pose{{P7[0], P7[1], P7[2], P7[3]}, {P7[4], P7[5], P7[6]}};
      ++SS.shapes;
      const PxsCachedTransform& pt = tc.getTransformCache(e);
      if (memcmp(&pt.transform, &pose, 28)) {
        if (SS.firstPose < 0) SS.firstPose = (long long)SS.sim;
        if (SS.show > 0) {
          --SS.show;
          Sc::ActorSim& as = ssim->getActor();
          const bool st = as.getActorType() == PxActorType::eRIGID_STATIC;
          fprintf(stderr, "[g1 sc 다름] sim %llu 요소 %u %s 깨어있음 %d 얼림 %d 들고있음 %d 자세 PhysX (%.9g %.9g %.9g | %.9g %.9g %.9g %.9g) 우리 (%.9g %.9g %.9g | %.9g %.9g %.9g %.9g) 정적 %d idt %d/%d\n",
                  (unsigned long long)SS.sim, e, a->getName() ? a->getName() : "?", st ? -1 : int(static_cast<Sc::BodySim&>(as).isActive()),
                  st ? -1 : int(static_cast<Sc::BodySim&>(as).isFrozen()), int(hi != gHeld.end()), pt.transform.p.x, pt.transform.p.y, pt.transform.p.z,
                  pt.transform.q.x, pt.transform.q.y, pt.transform.q.z, pt.transform.q.w, pose.p[0], pose.p[1], pose.p[2], pose.q[0], pose.q[1], pose.q[2],
                  pose.q[3], fl[0], fl[1], fl[2]);
        }
        ++SS.badPose;
      }
      if (memcmp(&ba.getBounds(e), BB, 24)) {
        if (SS.firstBounds < 0) SS.firstBounds = (long long)SS.sim;
        ++SS.badBounds;
      }
      if (cd && memcmp(&cd[e], &pc.mContactOffset, 4)) ++SS.badDist;
    }
  }
}

// 진단: 한 요소의 캐시·다시 계산·몸체 자세를 작업마다 찍는다 (G1_SC_TRACE=요소, G1_SC_TRACE_SIM=from:to)
void traceElem(const char* where) {
  if (SS.trace < 0 || (long long)SS.sim < SS.traceFrom || (long long)SS.sim > SS.traceTo) return;
  Sc::Scene& sc = static_cast<NpScene*>(SS.scene)->getScScene();
  Sc::ShapeSim* ss = reinterpret_cast<Sc::ShapeSim*>(sc.getAABBManager()->getUserData(PxU32(SS.trace)));
  if (!ss) {
    fprintf(stderr, "[g1 sc 추적] sim %llu %-40s 요소 없음\n", (unsigned long long)SS.sim, where);
    return;
  }
  float P7[7], BB[6];
  unsigned char fl[3];
  calc(ss, P7, BB, fl);
  const PxsCachedTransform& pt = sc.getLowLevelContext()->getTransformCache().getTransformCache(PxU32(SS.trace));
  const PxsBodyCore& bc = static_cast<Sc::BodySim&>(ss->getActor()).getBodyCore().getCore();
  fprintf(stderr, "[g1 sc 추적] sim %llu %-40s 캐시 z %.9g qw %.9g 다시 z %.9g qw %.9g body2World z %.9g 깨어 %d\n", (unsigned long long)SS.sim, where, pt.transform.p.z, pt.transform.q.w, P7[6], P7[3],
          bc.body2World.p.z, int(static_cast<Sc::BodySim&>(ss->getActor()).isActive()));
}

// 관절체 잠들기: 이번 스텝에 잠든 관절체는 풀이기가 링크 자세를 이미 옮겼지만(UpdateArticTask) 적분 뒤 갱신은 깨어 있는 관절체만 하므로
// (ScPipeline.cpp:2690 updateArticulationAfterIntegration -> 2702 putToSleep 는 캐시를 안 고침) 캐시는 풀이 전 값을 든다.
// -> 풀이 직전(작업 ScScene.advanceStep)에 깨어 있는 관절체 링크 모양의 값을 계산해 두고, 끝 단계에서 잠든 것만 들고 있음으로 옮긴다.
std::map<PxU32, Held> gPreSolve;
void preSolveSnapshot() {
  gPreSolve.clear();
  const PxU32 nArt = SS.scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(nArt);
  SS.scene->getArticulations(arts.data(), nArt);
  for (PxArticulationReducedCoordinate* art : arts) {
    std::vector<PxArticulationLink*> links(art->getNbLinks());
    art->getLinks(links.data(), PxU32(links.size()));
    for (PxArticulationLink* l : links) {
      Sc::BodySim* b = static_cast<NpArticulationLink*>(l)->getCore().getSim();
      if (!b || !b->isActive()) continue;
      Sc::ElementSim** el = b->getElements();
      for (PxU32 k = 0, n = b->getNbElements(); k < n; ++k) {
        Sc::ShapeSim* ss = static_cast<Sc::ShapeSim*>(el[k]);
        Held h;
        h.body = b;
        unsigned char fl[3];
        if (!calc(ss, h.pose, h.bounds, fl)) continue;
        auto hi = gHeld.find(ss->getElementID());
        gPreSolve[ss->getElementID()] = hi != gHeld.end() ? hi->second : h;  // 이미 옛 값을 들고 있었으면 그것
      }
    }
  }
}
void artSleepHold() {
  for (auto& kv : gPreSolve)
    if (!kv.second.body->isActive()) {
      if (!gHeld.count(kv.first)) ++SS.artHolds;
      gHeld[kv.first] = kv.second;
    }
  gPreSolve.clear();
}

// 적분 뒤 갱신 (ScScene.cpp:186 깨어 있고 안 얼린 몸체의 모양은 updateCached): 끝 단계 시작 때 그런 몸체의 들고 있음을 푼다
void afterIntegrationRelease() {
  for (auto it = gHeld.begin(); it != gHeld.end();) {
    Sc::BodySim* b = it->second.body;
    it = (b->isActive() && !b->isFrozen()) ? gHeld.erase(it) : std::next(it);
  }
}

}  // namespace

void g1_sc_before(PxScene* scene, uint64_t sim) {
  if (!SS.inited) {
    SS.inited = true;
    SS.on = getenv("G1_SC") != nullptr;
    SS.show = getenv("G1_SC_SHOW") ? atoi(getenv("G1_SC_SHOW")) : 0;
    if (getenv("G1_SC_TRACE")) SS.trace = atoll(getenv("G1_SC_TRACE"));
    if (getenv("G1_SC_TRACE_SIM")) sscanf(getenv("G1_SC_TRACE_SIM"), "%lld:%lld", &SS.traceFrom, &SS.traceTo);
  }
  if (!SS.on) return;
  SS.scene = scene;
  SS.sim = sim;
  traceElem("(simulate 시작)");
}
void g1_sc_task(const char* name) {
  if (!SS.on || !SS.scene) return;
  traceElem(name);
  if (!strcmp(name, "ScScene.broadPhaseFirstPass")) check();
  else if (!strcmp(name, "ScScene.advanceStep")) preSolveSnapshot();
  else if (!strcmp(name, "ScScene.finalizationPhase")) {
    afterIntegrationRelease();
    artSleepHold();
  }
}
void g1_sc_report() {
  if (!SS.on) return;
  printf("G1 Sc 입력 그림자 (넓은 단계 직전, 모든 모양을 그 순간 자세로 다시 계산): 스텝 %" PRIu64 ", 모양·스텝 %" PRIu64 " (못 옮긴 기하 %" PRIu64 ")\n", SS.steps, SS.shapes,
         SS.unsup);
  printf("  변환 캐시 자세 다름 %" PRIu64 "%s, 경계 상자 다름 %" PRIu64 "%s, 접촉 거리(=contactOffset) 다름 %" PRIu64 "\n", SS.badPose,
         SS.firstPose >= 0 ? (" (첫 simulate " + std::to_string(SS.firstPose) + ")").c_str() : "", SS.badBounds,
         SS.firstBounds >= 0 ? (" (첫 simulate " + std::to_string(SS.firstBounds) + ")").c_str() : "", SS.badDist);
  printf("  갱신 규칙: 질량 중심 자세 바꾸기로 옛 값 든 모양 %" PRIu64 " 번, 관절체 잠들기로 %" PRIu64 " 번, 비교 때 옛 값 쓴 모양·스텝 %" PRIu64 "\n", SS.holds, SS.artHolds,
         SS.heldShapes);
}

// ---- 갱신 규칙 입력 --wrap (Np -> Sc 호출: NpRigidDynamic.cpp·NpArticulationLink.cpp -> ScBodyCore.cpp)
#define W(name) __wrap_##name
#define R(name) __real_##name
extern "C" {
void R(_ZN5physx2Sc8BodyCore17setCMassLocalPoseERKNS_12PxTransformTIfEE)(Sc::BodyCore*, const PxTransform&);
void W(_ZN5physx2Sc8BodyCore17setCMassLocalPoseERKNS_12PxTransformTIfEE)(Sc::BodyCore* c, const PxTransform& p) {
  if (getenv("G1_SC") && c->getSim()) holdBody(c->getSim());
  R(_ZN5physx2Sc8BodyCore17setCMassLocalPoseERKNS_12PxTransformTIfEE)(c, p);
}
void R(_ZN5physx2Sc8BodyCore13setBody2WorldERKNS_12PxTransformTIfEE)(Sc::BodyCore*, const PxTransform&);
void W(_ZN5physx2Sc8BodyCore13setBody2WorldERKNS_12PxTransformTIfEE)(Sc::BodyCore* c, const PxTransform& p) {
  R(_ZN5physx2Sc8BodyCore13setBody2WorldERKNS_12PxTransformTIfEE)(c, p);
  if (getenv("G1_SC") && c->getSim()) releaseBody(c->getSim());  // 모양 더럽힘 -> 넓은 단계 전 preRigidBodyNarrowPhase 에서 다시 계산 (ScPipeline.cpp:199)
}
}
