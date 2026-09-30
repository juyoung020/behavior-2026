// G1 Sc 층 입력 그림자 (문서 15.3, 리드, 비계 전용): 넓은 단계가 읽는 순간(작업 ScScene.broadPhaseFirstPass 직전) PhysX 의
// 변환 캐시(모양 세계 자세)와 경계 상자 배열을, 우리 식(core/scene/sc_inputs.h: getAbsPoseAligned + Gu::computeBounds)으로
// 그 순간의 행위자·몸체 자세에서 모든 모양에 대해 다시 만들어 비트 비교한다. (어느 칸을 언제 고치는지 = 갱신 규칙은 이 비교가 모든 칸 같음을 보이면
// "그 순간 자세로 다시 계산한 값 = PhysX 값" 이 성립한다는 뜻.) 바뀜 비트맵·접촉 거리도 같이 본다.
// 켜기: G1_SC=1 (디스패처 필요).
#include <cinttypes>
#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
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
#include "core/scene/id_pool.h"
#include "core/scene/sc_scene.h"
#include <unordered_map>
#include "foundation/PxInlineArray.h"
#include "core/scene/bp_log.h"
#include "core/scene/pairs_log.h"
#include "g1_hooks.h"

using namespace physx;
void flushTemplates(physx::PxScene* scene, uint64_t sim);  // 새 행위자 틀 창 닫기 (아래)
namespace ep = eng::px;

void g1sc_update(const float* s2a, const unsigned char* flags, const float* a2w, const float* b2a, const eng::contact::ShapeGeom& g, float* pose, float* bounds);

namespace {

struct ScShadow {
  bool inited = false, on = false, inSim = false;
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

// 번호 매김 그림자 (G1_SC_IDS=1): simulate 시작마다 PhysX 요소·행위자·조인트 번호 추적기를 읽어, 우리 IdTracker(core/scene/id_pool.h)가
// "지난 스텝 끝에 대기 목록을 풀고(postReportsCleanup) 이번 창에서 c 번 새로 받음" 으로 같은 상태가 되는 c 가 있는지 본다.
// 풀어 줄 순서(대기 목록)는 PhysX 에서 읽은 것을 쓴다(삭제 API 호출 순서 = 우리 엔진 입력).
struct IdShadow {
  const char* name;
  eng::scene::IdTracker ours;
  bool init = false;
  uint64_t steps = 0, creates = 0, releases = 0, bad = 0;
  long long firstBad = -1;
};
IdShadow gIds[3] = {{"요소"}, {"행위자"}, {"조인트"}};
bool samePool(const eng::scene::IdPool& a, const Sc::ObjectIDTracker& t) {
  if (a.cur != t.mIDPool.mCurrentID || a.freeIds.size() != t.mIDPool.mFreeIDs.size()) return false;
  for (size_t i = 0; i < a.freeIds.size(); ++i)
    if (a.freeIds[i] != t.mIDPool.mFreeIDs[PxU32(i)]) return false;
  return true;
}
void idsStep(IdShadow& S, Sc::ObjectIDTracker& t) {
  ++S.steps;
  if (!S.init) {
    S.init = true;
    S.ours.pool.cur = t.mIDPool.mCurrentID;
    S.ours.pool.freeIds.assign(t.mIDPool.mFreeIDs.begin(), t.mIDPool.mFreeIDs.end());
  } else {
    S.ours.endStep();  // 지난 simulate 끝 (fetchResults 의 postReportsCleanup)
    // 이번 창의 새로 받기 수 c 찾기 (최대 빈 번호 + 4096)
    eng::scene::IdPool p = S.ours.pool;
    bool ok = samePool(p, t);
    uint32_t c = 0;
    while (!ok && c < p.freeIds.size() + 4096u) {
      p.getNew();
      ++c;
      ok = samePool(p, t);
    }
    if (ok) {
      S.ours.pool = p;
      S.creates += c;
    } else {
      ++S.bad;
      if (S.firstBad < 0) {
        S.firstBad = (long long)SS.sim;
        fprintf(stderr, "[g1 sc 번호 다름] %s sim %llu 우리 cur %u 빈 %zu / PhysX cur %u 빈 %u\n", S.name, (unsigned long long)SS.sim, S.ours.pool.cur,
                S.ours.pool.freeIds.size(), t.mIDPool.mCurrentID, t.mIDPool.mFreeIDs.size());
      }
      S.ours.pool.cur = t.mIDPool.mCurrentID;  // 다시 맞춤
      S.ours.pool.freeIds.assign(t.mIDPool.mFreeIDs.begin(), t.mIDPool.mFreeIDs.end());
    }
  }
  // 이번 창에서 지운 번호 (대기 목록, 넣은 순서) -> 이번 스텝 끝에 풀림
  S.ours.pending.assign(t.mPendingReleasedIDs.begin(), t.mPendingReleasedIDs.end());
  S.releases += S.ours.pending.size();
  if (!S.ours.pending.empty() && getenv("G1_SC_IDS_SHOW")) {
    fprintf(stderr, "[g1 sc 번호] %s sim %llu 지움 %zu:", S.name, (unsigned long long)SS.sim, S.ours.pending.size());
    for (uint32_t id : S.ours.pending) fprintf(stderr, " %u", id);
    fprintf(stderr, "\n");
  }
}
void idsCheck() {
  Sc::Scene& sc = static_cast<NpScene*>(SS.scene)->getScScene();
  idsStep(gIds[0], sc.getElementIDPool());
  idsStep(gIds[1], sc.getActorIDTracker());
  idsStep(gIds[2], sc.getConstraintIDTracker());
}

}  // namespace

#define W(name) __wrap_##name
#define R(name) __real_##name

// ==== Sc 편집 그림자 (G1_SC_EDIT=1): 판 도중 추가·삭제·다시 넣기를 우리 API(core/scene/sc_scene.h)로도 돌려 PhysX 와 비교 ====
// PhysX Sc::Scene::addBody/addStatic/removeBody/removeStatic, ShapeSimBase::reinsertBroadPhase 를 가로채, 진짜 호출 동안의 넓은 단계·섬 호출을
// 받아 두고(g1_sc_note_*), 같은 입력으로 우리 API 를 돌린다. 우리 API 가 모듈에서 돌려받는 값(넣기 대기였나·섬 노드 번호)은 PhysX 가 받은 값을
// 차례대로 준다. 비교: 모듈 호출 열(종류·인자), 행위자·요소 번호, 새 칸의 변환 캐시·경계 상자·접촉 거리, 스텝마다 번호 추적기 상태.
namespace {
namespace es = eng::scene;
struct EdCall {
  int kind;  // 0 bpAdd 1 bpRemove 2 islandAdd 3 islandDeactivate 4 islandRemove
  uint32_t idx = 0, group = 0, agg = 0, vt = 0, env = 0, result = 0;
  float cd = 0.f;
  uint64_t node = 0;
  int a = 0, b = 0;
};
struct EditShadow {
  bool on = false, inited = false, capturing = false;
  es::ScScene E;
  std::unordered_map<const void*, int32_t> handle;  // Sc::ActorSim* -> 우리 손잡이
  std::vector<const PxActor*> hPx;                  // 손잡이 -> PhysX 행위자 (닫힌 고리: 건드림 확인)
  std::vector<EdCall> px;
  uint64_t adds = 0, removes = 0, reinserts = 0, flagChanges = 0, untracked = 0, attaches = 0, detaches = 0, shapeChanges = 0, calls = 0, bad = 0, unsup = 0, idBad = 0, cellBad = 0, trackerBad = 0;
  long long firstBad = -1;
  int show = 5;
  std::map<uint64_t, std::map<std::string, int>> bySim;  // 진단: simulate 별 사건 수
  // 새 행위자 틀 (G1_SC_TEMPLATES=<폴더>): 편집 창에서 넣은 강체들을 창 끝(다음 simulate 앞)에 장면 파일 형식으로 쓴다
  std::string tmplDir;
  std::vector<PxActor*> addedWin;
  std::vector<std::string> removedWin;
  uint64_t tmplFiles = 0;
  bool tmplStopped = false;  // 장면 해체(정적 행위자 삭제)가 시작되면 더 쓰지 않음
} ES;
void edNote(const char* what) { ES.bySim[SS.sim][what]++; }

struct ReplayModules : es::ScModules {
  const std::vector<EdCall>& px;
  std::vector<EdCall> ours;
  explicit ReplayModules(const std::vector<EdCall>& p) : px(p) {}
  const EdCall* at() const { return ours.size() < px.size() ? &px[ours.size()] : nullptr; }
  bool bpAdd(const es::BpOp& o) override {
    const EdCall* p = at();
    EdCall c{0};
    c.idx = o.index; c.group = o.group; c.agg = o.agg; c.vt = o.volumeType; c.env = o.env; c.cd = o.contactDistance;
    c.result = p && p->kind == 0 ? p->result : 1u;
    ours.push_back(c);
    return c.result != 0;
  }
  bool bpRemove(uint32_t index) override {
    const EdCall* p = at();
    EdCall c{1};
    c.idx = index;
    c.result = p && p->kind == 1 ? p->result : 0u;
    ours.push_back(c);
    return c.result != 0;
  }
  uint64_t islandAddNode(bool awake, bool kine) override {
    const EdCall* p = at();
    EdCall c{2};
    c.a = awake; c.b = kine;
    c.node = p && p->kind == 2 ? p->node : ~0ull;
    ours.push_back(c);
    return c.node;
  }
  void islandDeactivateNode(uint64_t n) override { EdCall c{3}; c.node = n; ours.push_back(c); }
  void islandRemoveNode(uint64_t n) override { EdCall c{4}; c.node = n; ours.push_back(c); }
};

bool sameCall(const EdCall& x, const EdCall& y) {
  if (x.kind != y.kind) return false;
  switch (x.kind) {
    case 0: return x.idx == y.idx && x.group == y.group && x.agg == y.agg && x.vt == y.vt && x.env == y.env && !memcmp(&x.cd, &y.cd, 4) && x.result == y.result;
    case 1: return x.idx == y.idx && x.result == y.result;
    case 2: return x.a == y.a && x.b == y.b && x.node == y.node;
    default: return x.node == y.node;
  }
}
void printCall(const char* who, const EdCall& c) {
  fprintf(stderr, "    %s 종류 %d 칸 %u 무리 %u 집합 %u 부피 %u 판 %u 거리 %.9g 결과 %u 노드 %llx 인자 %d/%d\n", who, c.kind, c.idx, c.group, c.agg, c.vt, c.env, c.cd,
          c.result, (unsigned long long)c.node, c.a, c.b);
}
void edBad(const char* what) {
  ++ES.bad;
  if (ES.firstBad < 0) ES.firstBad = (long long)SS.sim;
  if (ES.show > 0) {
    --ES.show;
    fprintf(stderr, "[g1 sc 편집 다름] sim %llu %s\n", (unsigned long long)SS.sim, what);
  }
}
void compareCalls(const ReplayModules& m, const char* what) {
  ES.calls += m.ours.size();
  static const bool all = getenv("G1_SC_EDIT_ALL") != nullptr;
  if (all && SS.sim < 690) {
    fprintf(stderr, "[g1 sc 편집] sim %llu %s\n",(unsigned long long)SS.sim, what);
    for (const EdCall& c : ES.px) printCall("PhysX", c);
    for (const EdCall& c : m.ours) printCall("우리 ", c);
  }
  bool ok = m.ours.size() == ES.px.size();
  for (size_t i = 0; ok && i < m.ours.size(); ++i) ok = sameCall(m.ours[i], ES.px[i]);
  if (!ok) {
    const bool showIt = ES.show > 0;
    edBad(what);
    if (showIt) {
      for (const EdCall& c : ES.px) printCall("PhysX", c);
      for (const EdCall& c : m.ours) printCall("우리 ", c);
    }
  }
}

es::ScShapeIn shapeIn(const Sc::ShapeCore& core, bool* okOut) {
  es::ScShapeIn si;
  const PxsShapeCore& pc = core.getCore();
  memset(static_cast<void*>(&si.geom), 0, sizeof(si.geom));
  *okOut = geomOf(pc.mGeometry.getGeometry(), si.geom);
  const PxTransform t = pc.getTransform();
  si.localPose = ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w));
  si.contactOffset = pc.mContactOffset;
  si.shapeFlags = uint32_t(pc.mShapeFlags);
  si.idtShape = pc.mShapeCoreFlags.isSet(PxShapeCoreFlag::eIDT_TRANSFORM) ? 1 : 0;
  return si;
}
ep::PxTransform toEp(const PxTransform& t) { return ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)); }

void copyTracker(es::IdTracker& o, Sc::ObjectIDTracker& t) {
  o.pool.cur = t.mIDPool.mCurrentID;
  o.pool.freeIds.assign(t.mIDPool.mFreeIDs.begin(), t.mIDPool.mFreeIDs.end());
  o.pending.assign(t.mPendingReleasedIDs.begin(), t.mPendingReleasedIDs.end());
  o.deleted.clear();
  for (uint32_t id : o.pending) {
    if (o.deleted.size() <= id) o.deleted.resize(id + 1, 0);
    o.deleted[id] = 1;
  }
}
bool sameTracker(const es::IdTracker& o, Sc::ObjectIDTracker& t) {
  if (!samePool(o.pool, t) || o.pending.size() != t.mPendingReleasedIDs.size()) return false;
  for (size_t i = 0; i < o.pending.size(); ++i)
    if (o.pending[i] != t.mPendingReleasedIDs[PxU32(i)]) return false;
  return true;
}

// PhysX 장면 전체 -> 우리 Sc 장면 (record = 편집 그림자의 행위자 표도 채움)
void scCapture(PxScene* scene, es::ScScene& E, bool record) {
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  copyTracker(E.elementIds, sc.getElementIDPool());
  copyTracker(E.actorIds, sc.getActorIDTracker());
  const uint32_t maxE = E.elementIds.maxId();
  E.grow(maxE ? maxE - 1 : 0);
  PxsTransformCache& tc = sc.getLowLevelContext()->getTransformCache();
  Bp::BoundsArray& ba = sc.getBoundsArray();
  const float* cd = sc.getLowLevelContext()->getContactDistances();
  for (PxRigidActor* a : actorsOf(scene)) {
    const PxU32 n = a->getNbShapes();
    if (!n) continue;
    std::vector<PxShape*> sh(n);
    a->getShapes(sh.data(), n);
    Sc::ShapeSim* first = static_cast<NpShape*>(sh[0])->getCore().getExclusiveSim();
    if (!first) continue;
    Sc::ActorSim& as = first->getActor();
    const int32_t h = int32_t(E.actors.size());
    E.actors.emplace_back();
    es::ScActorRec& r = E.actors.back();
    r.alive = 1;
    r.actorID = as.getActorID();
    if (as.getActorType() == PxActorType::eRIGID_STATIC) {
      r.kind = 0;
      r.pose = toEp(static_cast<Sc::StaticSim&>(as).getStaticCore().getActor2World());
      r.body2Actor = ep::PxTransform(ep::PxIdentity);
    } else {
      Sc::BodySim& bs = static_cast<Sc::BodySim&>(as);
      const PxsBodyCore& bc = bs.getBodyCore().getCore();
      r.kind = as.getActorType() == PxActorType::eARTICULATION_LINK ? 2 : 1;
      r.pose = toEp(bc.body2World);
      r.body2Actor = toEp(bc.getBody2Actor());
      r.idtBody2Actor = bc.hasIdtBody2Actor() ? 1 : 0;
      r.kinematic = bs.isKinematic() ? 1 : 0;
      r.forcedKineNotif = bs.hasForcedKinematicNotif() ? 1 : 0;
      r.node = bs.getNodeIndex().getInd();
    }
    Sc::ElementSim** el = as.getElements();
    for (PxU32 k = 0, ne = as.getNbElements(); k < ne; ++k) {
      Sc::ShapeSim* ss = static_cast<Sc::ShapeSim*>(el[k]);
      const uint32_t e = ss->getElementID();
      E.grow(e);
      es::ScShapeRec& s = E.shapes[e];
      bool ok = true;
      s.in = shapeIn(ss->getCore(), &ok);
      if (!ok && record) ++ES.unsup;
      s.actor = h;
      s.alive = 1;
      s.inBp = ss->isInBroadPhase() ? 1 : 0;
      r.elements.push_back(e);
      const PxTransform& t = tc.getTransformCache(e).transform;
      E.cache[e] = toEp(t);
      memcpy(&E.bounds[e], &ba.getBounds(e), 24);
      if (cd) E.contactDist[e] = cd[e];
      E.cacheFlags[e] = tc.getTransformCache(e).flags;
    }
    if (!record) continue;
    ES.handle[&as] = h;
    if (ES.hPx.size() <= size_t(h)) ES.hPx.resize(size_t(h) + 1, nullptr);
    ES.hPx[size_t(h)] = as.getPxActor();
  }
}
// 첫 simulate 앞: PhysX 장면 전체를 우리 Sc 장면으로 (넘겨받기)
void edInit() {
  ES.inited = true;
  scCapture(SS.scene, ES.E, true);
}

void edBegin() {
  ES.px.clear();
  ES.capturing = true;
}
// 새로 넣은 우리 행위자를 PhysX 결과와 비교 (번호·칸)
void edCheckAdded(int32_t h, Sc::ActorSim& as) {
  Sc::Scene& sc = static_cast<NpScene*>(SS.scene)->getScScene();
  const es::ScActorRec& r = ES.E.actors[size_t(h)];
  bool ok = r.actorID == as.getActorID() && r.elements.size() == as.getNbElements();
  Sc::ElementSim** el = as.getElements();
  for (PxU32 k = 0; ok && k < as.getNbElements(); ++k) ok = r.elements[k] == el[k]->getElementID();
  if (!ok) {
    ++ES.idBad;
    edBad("행위자·요소 번호");
    return;
  }
  PxsTransformCache& tc = sc.getLowLevelContext()->getTransformCache();
  Bp::BoundsArray& ba = sc.getBoundsArray();
  const float* cd = sc.getLowLevelContext()->getContactDistances();
  for (uint32_t e : r.elements) {
    const PxTransform& t = tc.getTransformCache(e).transform;
    const ep::PxTransform& o = ES.E.cache[e];
    const float a7[7] = {t.q.x, t.q.y, t.q.z, t.q.w, t.p.x, t.p.y, t.p.z}, b7[7] = {o.q.x, o.q.y, o.q.z, o.q.w, o.p.x, o.p.y, o.p.z};
    if (memcmp(a7, b7, 28) || memcmp(&ba.getBounds(e), &ES.E.bounds[e], 24) || (cd && memcmp(&cd[e], &ES.E.contactDist[e], 4))) {
      ++ES.cellBad;
      edBad("새 칸(변환 캐시·경계 상자·접촉 거리)");
    }
  }
}
}  // namespace

void g1_sc_note_bp(uint32_t type, uint32_t index, uint32_t group, uint32_t agg, uint32_t vt, uint32_t env, float cd, uint32_t result) {
  if (!ES.capturing) {
    // 편집 API 로 못 덮은 넓은 단계 구조 변경 (simulate 밖에서 일어난 것만 — 안쪽 집합체 등은 BpRuntime 이 스스로)
    if (ES.on && ES.inited && !SS.inSim) {
      ++ES.untracked;
      edNote("덮지 못한 넓은 단계 호출");
      if (getenv("G1_SC_EDIT_ALL") && SS.sim < 690) {
        fprintf(stderr, "[g1 sc 편집 밖 bp] sim %llu 종류 %u 칸 %u 결과 %u\n", (unsigned long long)SS.sim, type, index, result);
        void* bt[16];
        const int nb = backtrace(bt, 16);
        backtrace_symbols_fd(bt, nb, 2);
      }
    }
    return;
  }
  EdCall c{int(type == 0 ? 0 : 1)};
  c.idx = index;
  c.result = result;
  if (type == 0) { c.group = group; c.agg = agg; c.vt = vt; c.env = env; c.cd = cd; }
  ES.px.push_back(c);
}
void g1_sc_note_island(int op, uint64_t node, int a, int b) {
  if (!ES.capturing) return;
  EdCall c{2 + op};
  c.node = node;
  c.a = a;
  c.b = b;
  ES.px.push_back(c);
}

// ---- 가로채기 (NpScene -> ScScene, ScActorCore -> ShapeSimBase)
extern "C" {
void R(_ZN5physx2Sc5Scene7addBodyERNS0_8BodyCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3Eb)(Sc::Scene*, Sc::BodyCore&, NpShape* const*, PxU32, size_t, PxBounds3*, bool);
void W(_ZN5physx2Sc5Scene7addBodyERNS0_8BodyCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3Eb)(Sc::Scene* self, Sc::BodyCore& body, NpShape* const* shapes, PxU32 n,
                                                                                       size_t off, PxBounds3* ob, bool compound) {
  const bool track = ES.on && ES.inited;
  es::ScActorIn in{};
  bool ok = true;
  if (track) {
    const PxsBodyCore& bc = body.getCore();
    in.kind = 1;
    in.pose = toEp(bc.body2World);
    in.body2Actor = toEp(bc.getBody2Actor());
    in.idtBody2Actor = bc.hasIdtBody2Actor() ? 1 : 0;
    in.kinematic = (body.getFlags() & PxRigidBodyFlag::eKINEMATIC) ? 1 : 0;
    in.forcedKineNotif = (body.getFlags() & (PxRigidBodyFlag::eFORCE_KINE_KINE_NOTIFICATIONS | PxRigidBodyFlag::eFORCE_STATIC_KINE_NOTIFICATIONS)) ? 1 : 0;
    in.awake = (body.getWakeCounter() > 0.f || !body.getLinearVelocity().isZero() || !body.getAngularVelocity().isZero()) ? 1 : 0;
    for (PxU32 i = 0; i < n; ++i) {
      bool gok = true;
      in.shapes.push_back(shapeIn(*reinterpret_cast<Sc::ShapeCore*>(size_t(shapes[i]) + off), &gok));
      ok = ok && gok;
    }
    edBegin();
  }
  R(_ZN5physx2Sc5Scene7addBodyERNS0_8BodyCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3Eb)(self, body, shapes, n, off, ob, compound);
  if (!ES.tmplDir.empty() && SS.inited && body.getSim()) ES.addedWin.push_back(body.getSim()->getPxActor());
  if (!track) return;
  ES.capturing = false;
  if (!ok || compound) { ++ES.unsup; return; }
  ReplayModules m(ES.px);
  const int32_t h = ES.E.addActor(in, m);
  ++ES.adds;
  edNote("추가");
  compareCalls(m, "추가(동적) 모듈 호출");
  if (Sc::BodySim* sim = body.getSim()) {
    ES.handle[sim] = h;
    if (ES.hPx.size() <= size_t(h)) ES.hPx.resize(size_t(h) + 1, nullptr);
    ES.hPx[size_t(h)] = sim->getPxActor();
    edCheckAdded(h, *sim);
  }
}
void R(_ZN5physx2Sc5Scene9addStaticERNS0_10StaticCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3E)(Sc::Scene*, Sc::StaticCore&, NpShape* const*, PxU32, size_t, PxBounds3*);
void W(_ZN5physx2Sc5Scene9addStaticERNS0_10StaticCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3E)(Sc::Scene* self, Sc::StaticCore& st, NpShape* const* shapes, PxU32 n,
                                                                                         size_t off, PxBounds3* ob) {
  const bool track = ES.on && ES.inited;
  es::ScActorIn in{};
  bool ok = true;
  if (track) {
    in.kind = 0;
    in.pose = toEp(st.getActor2World());
    in.body2Actor = ep::PxTransform(ep::PxIdentity);
    in.idtBody2Actor = 1;
    for (PxU32 i = 0; i < n; ++i) {
      bool gok = true;
      in.shapes.push_back(shapeIn(*reinterpret_cast<Sc::ShapeCore*>(size_t(shapes[i]) + off), &gok));
      ok = ok && gok;
    }
    edBegin();
  }
  R(_ZN5physx2Sc5Scene9addStaticERNS0_10StaticCoreEPKPNS_7NpShapeEjmPNS_9PxBounds3E)(self, st, shapes, n, off, ob);
  if (!ES.tmplDir.empty() && SS.inited && st.getSim()) ES.addedWin.push_back(st.getSim()->getPxActor());
  if (!track) return;
  ES.capturing = false;
  if (!ok) { ++ES.unsup; return; }
  ReplayModules m(ES.px);
  const int32_t h = ES.E.addActor(in, m);
  ++ES.adds;
  edNote("추가 정적");
  compareCalls(m, "추가(정적) 모듈 호출");
  if (Sc::StaticSim* sim = st.getSim()) {
    ES.handle[sim] = h;
    edCheckAdded(h, *sim);
  }
}
void R(_ZN5physx2Sc5Scene10removeBodyERNS0_8BodyCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(
    Sc::Scene*, Sc::BodyCore&, PxInlineArray<const Sc::ShapeCore*, 64>&, bool);
void W(_ZN5physx2Sc5Scene10removeBodyERNS0_8BodyCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(
    Sc::Scene* self, Sc::BodyCore& body, PxInlineArray<const Sc::ShapeCore*, 64>& rs, bool wake) {
  int32_t h = -1;
  const void* key = body.getSim();
  if (!ES.tmplDir.empty() && !ES.tmplStopped && SS.inited && body.getSim() && !ES.addedWin.empty()) {
    // 전이 규칙은 한 창에서 빼기를 모두 한 뒤 넣는다 -> 넣은 뒤의 빼기 = 기록 끝 장면 해체: 열린 창을 닫고 멈춤
    flushTemplates(SS.scene, SS.sim + 1);
    ES.tmplStopped = true;
  }
  if (!ES.tmplDir.empty() && !ES.tmplStopped && ES.removedWin.size() >= 32) {  // 넣기 없는 창의 대량 빼기 = 해체
    ES.removedWin.clear();
    ES.tmplStopped = true;
  }
  if (!ES.tmplDir.empty() && !ES.tmplStopped && SS.inited && body.getSim()) {
    const char* nm = body.getSim()->getPxActor()->getName();
    ES.removedWin.push_back(nm ? nm : "?");
  }
  if (ES.on && ES.inited && key) {
    auto it = ES.handle.find(key);
    if (it != ES.handle.end()) h = it->second;
    if (h >= 0 && ES.E.actors[size_t(h)].kind == 2) { ++ES.unsup; h = -1; }  // 링크는 아직
    if (h >= 0) edBegin();
  }
  R(_ZN5physx2Sc5Scene10removeBodyERNS0_8BodyCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(self, body, rs, wake);
  if (h < 0) return;
  ES.capturing = false;
  ReplayModules m(ES.px);
  ES.E.removeActor(h, m);
  ++ES.removes;
  edNote("삭제");
  ES.handle.erase(key);
  compareCalls(m, "삭제(동적) 모듈 호출");
}
void R(_ZN5physx2Sc5Scene12removeStaticERNS0_10StaticCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(
    Sc::Scene*, Sc::StaticCore&, PxInlineArray<const Sc::ShapeCore*, 64>&, bool);
void W(_ZN5physx2Sc5Scene12removeStaticERNS0_10StaticCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(
    Sc::Scene* self, Sc::StaticCore& st, PxInlineArray<const Sc::ShapeCore*, 64>& rs, bool wake) {
  int32_t h = -1;
  const void* key = st.getSim();
  if (!ES.tmplDir.empty() && !ES.tmplStopped && SS.scene) {  // 판 도중에는 정적을 지우지 않는다 -> 해체 시작: 열린 창을 닫고 멈춤
    flushTemplates(SS.scene, SS.sim + 1);
    ES.tmplStopped = true;
  }
  if (ES.on && ES.inited && key) {
    auto it = ES.handle.find(key);
    if (it != ES.handle.end()) h = it->second;
    if (h >= 0) edBegin();
  }
  R(_ZN5physx2Sc5Scene12removeStaticERNS0_10StaticCoreERNS_13PxInlineArrayIPKNS0_9ShapeCoreELj64ENS_21PxReflectionAllocatorIS7_EEEEb)(self, st, rs, wake);
  if (h < 0) return;
  ES.capturing = false;
  ReplayModules m(ES.px);
  ES.E.removeActor(h, m);
  ++ES.removes;
  edNote("삭제 정적");
  ES.handle.erase(key);
  compareCalls(m, "삭제(정적) 모듈 호출");
}
void R(_ZN5physx2Sc12ShapeSimBase12onFlagChangeENS_7PxFlagsINS_11PxShapeFlag4EnumEhEE)(Sc::ShapeSimBase*, PxShapeFlags);
void W(_ZN5physx2Sc12ShapeSimBase12onFlagChangeENS_7PxFlagsINS_11PxShapeFlag4EnumEhEE)(Sc::ShapeSimBase* self, PxShapeFlags oldFlags) {
  const bool track = ES.on && ES.inited;
  const uint32_t e = self->getElementID();
  const uint32_t nf = uint32_t(PxU8(self->getCore().getFlags()));
  if (track) edBegin();
  R(_ZN5physx2Sc12ShapeSimBase12onFlagChangeENS_7PxFlagsINS_11PxShapeFlag4EnumEhEE)(self, oldFlags);
  if (!track) return;
  ES.capturing = false;
  if (e >= ES.E.shapes.size() || !ES.E.shapes[e].alive) { ++ES.unsup; return; }
  if (ES.E.shapes[e].in.shapeFlags != uint32_t(PxU8(oldFlags))) edBad("플래그 바꾸기 전 값");
  ReplayModules m(ES.px);
  ES.E.setShapeFlags(e, nf, m);
  ++ES.flagChanges;
  edNote("플래그");
  compareCalls(m, "플래그 바꾸기 모듈 호출");
}
// 모양 붙이기·떼기·바꾸기 (NpShapeManager·NpShape·NpScene -> ScRigidCore)
void R(_ZN5physx2Sc9RigidCore15addShapeToSceneERNS0_9ShapeCoreE)(Sc::RigidCore*, Sc::ShapeCore&);
void W(_ZN5physx2Sc9RigidCore15addShapeToSceneERNS0_9ShapeCoreE)(Sc::RigidCore* self, Sc::ShapeCore& sc) {
  const bool track = ES.on && ES.inited && self->getSim();
  int32_t h = -1;
  if (track) {
    auto it = ES.handle.find(static_cast<Sc::ActorSim*>(self->getSim()));
    if (it != ES.handle.end()) h = it->second;
    if (h >= 0) edBegin();
  }
  R(_ZN5physx2Sc9RigidCore15addShapeToSceneERNS0_9ShapeCoreE)(self, sc);
  if (h < 0) {
    if (track) ++ES.unsup;
    return;
  }
  ES.capturing = false;
  bool ok = true;
  const es::ScShapeIn si = shapeIn(sc, &ok);
  if (!ok) { ++ES.unsup; return; }
  ReplayModules m(ES.px);
  const uint32_t e = ES.E.attachShape(h, si, m);
  ++ES.attaches;
  edNote("모양 붙이기");
  compareCalls(m, "모양 붙이기 모듈 호출");
  Sc::ShapeSim* ss = sc.getExclusiveSim();
  if (!ss || ss->getElementID() != e) {
    ++ES.idBad;
    edBad("모양 붙이기 요소 번호");
  }
}
void R(_ZN5physx2Sc9RigidCore20removeShapeFromSceneERNS0_9ShapeCoreEb)(Sc::RigidCore*, Sc::ShapeCore&, bool);
void W(_ZN5physx2Sc9RigidCore20removeShapeFromSceneERNS0_9ShapeCoreEb)(Sc::RigidCore* self, Sc::ShapeCore& sc, bool wake) {
  const bool track = ES.on && ES.inited && self->getSim();
  uint32_t e = 0xffffffffu;
  if (track) {
    if (Sc::ShapeSim* ss = sc.getExclusiveSim()) e = ss->getElementID();
    if (e < ES.E.shapes.size() && ES.E.shapes[e].alive) edBegin();
    else e = 0xffffffffu;
  }
  R(_ZN5physx2Sc9RigidCore20removeShapeFromSceneERNS0_9ShapeCoreEb)(self, sc, wake);
  if (e == 0xffffffffu) {
    if (track) ++ES.unsup;
    return;
  }
  ES.capturing = false;
  ReplayModules m(ES.px);
  ES.E.detachShape(e, m);
  ++ES.detaches;
  edNote("모양 떼기");
  compareCalls(m, "모양 떼기 모듈 호출");
}
void R(_ZN5physx2Sc9RigidCore13onShapeChangeERNS0_9ShapeCoreENS_7PxFlagsINS0_21ShapeChangeNotifyFlag4EnumEjEE)(Sc::RigidCore*, Sc::ShapeCore&,
                                                                                                                   Sc::ShapeChangeNotifyFlags);
void W(_ZN5physx2Sc9RigidCore13onShapeChangeERNS0_9ShapeCoreENS_7PxFlagsINS0_21ShapeChangeNotifyFlag4EnumEjEE)(Sc::RigidCore* self, Sc::ShapeCore& sc,
                                                                                                                   Sc::ShapeChangeNotifyFlags f) {
  const bool track = ES.on && ES.inited && self->getSim();
  uint32_t e = 0xffffffffu;
  if (track) {
    if (Sc::ShapeSim* ss = sc.getExclusiveSim()) e = ss->getElementID();
    if (e < ES.E.shapes.size() && ES.E.shapes[e].alive) edBegin();
    else e = 0xffffffffu;
  }
  R(_ZN5physx2Sc9RigidCore13onShapeChangeERNS0_9ShapeCoreENS_7PxFlagsINS0_21ShapeChangeNotifyFlag4EnumEjEE)(self, sc, f);
  if (e == 0xffffffffu) {
    if (track) ++ES.unsup;
    return;
  }
  ES.capturing = false;
  ReplayModules m(ES.px);
  bool ok = true;
  const es::ScShapeIn si = shapeIn(sc, &ok);
  es::ScShapeRec& r = ES.E.shapes[e];
  // PhysX RigidCore::onShapeChange 순서: 기하 -> 거르기 다시 -> 모양 자세 -> 거르기 자료 -> 접촉 거리 -> 쉼 거리
  if (f & Sc::ShapeChangeNotifyFlag::eGEOMETRY) { r.in.geom = si.geom; edNote("모양 바꾸기(기하)"); }
  if (f & Sc::ShapeChangeNotifyFlag::eRESET_FILTERING) { ES.E.resetFiltering(e, m); edNote("거르기 다시"); }
  if (f & Sc::ShapeChangeNotifyFlag::eSHAPE2BODY) { r.in.localPose = si.localPose; r.in.idtShape = si.idtShape; edNote("모양 바꾸기(자세)"); }
  if (f & Sc::ShapeChangeNotifyFlag::eCONTACTOFFSET) { ES.E.setContactOffset(e, si.contactOffset); edNote("모양 바꾸기(접촉 거리)"); }
  ++ES.shapeChanges;
  compareCalls(m, "모양 바꾸기 모듈 호출");
}
void R(_ZN5physx2Sc12ShapeSimBase18reinsertBroadPhaseEv)(Sc::ShapeSimBase*);
void W(_ZN5physx2Sc12ShapeSimBase18reinsertBroadPhaseEv)(Sc::ShapeSimBase* self) {
  const bool track = ES.on && ES.inited;
  const uint32_t e = self->getElementID();
  if (track) edBegin();
  R(_ZN5physx2Sc12ShapeSimBase18reinsertBroadPhaseEv)(self);
  if (!track) return;
  ES.capturing = false;
  if (e >= ES.E.shapes.size() || !ES.E.shapes[e].alive) { ++ES.unsup; return; }
  ReplayModules m(ES.px);
  ES.E.reinsertShape(e, m);
  ++ES.reinserts;
  edNote("다시 넣기");
  compareCalls(m, "다시 넣기 모듈 호출");
  const int32_t h = ES.E.shapes.size() > self->getElementID() ? ES.E.shapes[self->getElementID()].actor : -1;
  if (h < 0 || !ES.E.shapes[self->getElementID()].alive) {
    ++ES.idBad;
    edBad("다시 넣기 요소 번호");
  }
}
}

// 창 끝(다음 simulate 앞, 또는 기록 끝의 장면 해체 직전)에 틀 파일·목록 한 줄을 쓴다
void flushTemplates(PxScene* scene, uint64_t sim) {
  if (ES.tmplDir.empty() || ES.tmplStopped || (ES.addedWin.empty() && ES.removedWin.empty())) return;
  {
    // 창 끝: 이름이 붙은 뒤라 여기서 쓴다. 목록 파일 한 줄 = simulate, 뺀 행위자들, 넣은 틀 파일과 행위자들
    std::string path;
    if (!ES.addedWin.empty()) {
      path = ES.tmplDir + "/add_sim" + std::to_string(sim) + ".scene";
      if (!g1_dump_actors(scene, ES.addedWin, sim, path.c_str())) fprintf(stderr, "[g1 sc] 틀 파일 쓰기 실패: %s\n", path.c_str());
      else ++ES.tmplFiles;
    }
    if (FILE* f = fopen((ES.tmplDir + "/templates.txt").c_str(), "a")) {
      fprintf(f, "simulate %llu 뺌 %zu", (unsigned long long)sim, ES.removedWin.size());
      for (const std::string& n : ES.removedWin) fprintf(f, " %s", n.c_str());
      fprintf(f, " | 넣음 %zu %s", ES.addedWin.size(), path.empty() ? "-" : path.c_str());
      for (PxActor* a : ES.addedWin) fprintf(f, " %s", a->getName() ? a->getName() : "?");
      fprintf(f, "\n");
      fclose(f);
    }
    ES.addedWin.clear();
    ES.removedWin.clear();
  }
}

void g1_sc_before(PxScene* scene, uint64_t sim) {
  if (!SS.inited) {
    SS.inited = true;
    SS.on = getenv("G1_SC") != nullptr;
    SS.show = getenv("G1_SC_SHOW") ? atoi(getenv("G1_SC_SHOW")) : 0;
    ES.on = SS.on && getenv("G1_SC_EDIT") != nullptr;

    if (const char* d = getenv("G1_SC_TEMPLATES")) ES.tmplDir = d;
    if (getenv("G1_SC_TRACE")) SS.trace = atoll(getenv("G1_SC_TRACE"));
    if (getenv("G1_SC_TRACE_SIM")) sscanf(getenv("G1_SC_TRACE_SIM"), "%lld:%lld", &SS.traceFrom, &SS.traceTo);
  }
  if (!SS.on) return;
  SS.scene = scene;
  SS.sim = sim;
  SS.inSim = true;
  traceElem("(simulate 시작)");
  flushTemplates(scene, sim);
  if (getenv("G1_SC_IDS")) idsCheck();
  if (ES.on) {
    Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
    if (!ES.inited) edInit();
    else if (!sameTracker(ES.E.elementIds, sc.getElementIDPool()) || !sameTracker(ES.E.actorIds, sc.getActorIDTracker())) {
      ++ES.trackerBad;
      edBad("번호 추적기 상태 (simulate 앞)");
      copyTracker(ES.E.elementIds, sc.getElementIDPool());
      copyTracker(ES.E.actorIds, sc.getActorIDTracker());
    }
  }
}
// fetchResults 뒤: 스텝 끝 번호 풀기 (postReportsCleanup)
void g1_sc_after(PxScene*, uint64_t) {
  SS.inSim = false;
  if (ES.on && ES.inited) ES.E.endStep();
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
// 진단 (G1_EDIT_DUMP=from:to, G1_DUMP_AT 로 기록을 켜야 함): 판 도중 추가·삭제 때 넓은 단계 연산·쌍 관리층 앞 연산을 찍는다
static void editDump() {
  const char* e = getenv("G1_EDIT_DUMP");
  if (!e) return;
  long long a = 0, b = -1;
  sscanf(e, "%lld:%lld", &a, &b);
  if (const eng::scene::BpLog* bl = g1_bp_log()) {
    for (const eng::scene::BpOp& o : bl->ops)
      if ((long long)o.frame >= a && (long long)o.frame <= b)
        printf("[편집 bp] 프레임 %u 종류 %u 칸 %u 무리 %u 집합 %u 부피 %u 판 %u 최대 %u 거리 %.9g 결과 %u\n", o.frame, o.type, o.index, o.group, o.agg, o.volumeType, o.env,
               o.maxNum, o.contactDistance, o.result);
  }
  if (const eng::scene::PairsLog* pl = g1_pairs_log()) {
    for (size_t k = 0; k < pl->steps.size(); ++k) {
      const eng::scene::PairsStep& st = pl->steps[k];
      const size_t prevA = k ? pl->steps[k - 1].actors.size() : pl->actors0.size(), prevS = k ? pl->steps[k - 1].shapes.size() : pl->shapes0.size();
      bool any = st.actors.size() != prevA || st.shapes.size() != prevS;
      for (const auto& o : st.ops) any |= o.type != eng::scene::PPO_API_RESET && o.type != eng::scene::PPO_REFILTER;
      if (!any || (long long)k < a || (long long)k > b) continue;
      printf("[편집 쌍] 기록 스텝 %zu 행위자 %zu->%zu 모양 %zu->%zu 연산", k, prevA, st.actors.size(), prevS, st.shapes.size());
      for (const auto& o : st.ops)
        if (o.type != eng::scene::PPO_API_RESET) printf(" (%d %d %d %d)", o.type, o.a, o.b, o.c);
      printf("\n");
    }
  }
}

void g1_sc_report() {
  if (!SS.on) return;
  if (ES.on && getenv("G1_SC_EDIT_SIMS")) {
    int k = 0;
    for (auto& kv : ES.bySim) {
      if (k++ >= 30) break;
      printf("  [편집 simulate %llu]", (unsigned long long)kv.first);
      for (auto& w : kv.second) printf(" %s %d", w.first.c_str(), w.second);
      printf("\n");
    }
  }
  if (ES.on)
    printf("  편집 API(sc_scene.h): 추가 %" PRIu64 ", 삭제 %" PRIu64 ", 다시 넣기 %" PRIu64 ", 플래그 %" PRIu64 ", 모양 붙이기 %" PRIu64 " 떼기 %" PRIu64 " 바꾸기 %" PRIu64 ", 모듈 호출 %" PRIu64 " — 다름 %" PRIu64 " (번호 %" PRIu64 ", 칸 %" PRIu64
           ", 추적기 %" PRIu64 ")%s, 못 받은 것 %" PRIu64 ", 덮지 못한 넓은 단계 호출 %" PRIu64 "\n",
           ES.adds, ES.removes, ES.reinserts, ES.flagChanges, ES.attaches, ES.detaches, ES.shapeChanges, ES.calls, ES.bad, ES.idBad, ES.cellBad, ES.trackerBad,
           ES.firstBad >= 0 ? (" 첫 simulate " + std::to_string(ES.firstBad)).c_str() : "", ES.unsup, ES.untracked);
  editDump();
  for (const IdShadow& S : gIds)
    if (S.steps)
      printf("  번호 매김 %s: 스텝 %" PRIu64 ", 새로 받기 %" PRIu64 ", 지움 %" PRIu64 ", 다름 %" PRIu64 "%s\n", S.name, S.steps, S.creates, S.releases, S.bad,
             S.firstBad >= 0 ? (" (첫 simulate " + std::to_string(S.firstBad) + ")").c_str() : "");
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

// ---- 닫힌 고리 2단: 우리 Sc 층(ES.E)의 입력 조각을 우리 몸체 결과로 갱신하고 contact 층이 쓴다 (G1_LOOP_SC=1, G1_SC_EDIT 필요)
bool g1_sc_loop_on() { return ES.on && ES.inited && getenv("G1_LOOP_SC") != nullptr; }
void g1_sc_update_actor(const void* actorSim, const eng::Tf& b2w, const eng::Tf& b2a, bool frozen) {
  if (!g1_sc_loop_on()) return;
  auto it = ES.handle.find(actorSim);
  if (it == ES.handle.end()) return;
  auto tf = [](const eng::Tf& t) { return ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)); };
  ES.E.updateActorCached(it->second, tf(b2w), tf(b2a), frozen);
}
eng::scene::ScScene* g1_sc_scene() { return g1_sc_loop_on() ? &ES.E : nullptr; }
// 장면 파일 뜨기(g1_dump.cpp)·env 적재 대조(g1_env.cpp): 지금 PhysX 장면을 우리 Sc 장면으로 (편집 그림자와 무관)
void g1_sc_capture(physx::PxScene* scene, eng::scene::ScScene& out) {
  out = eng::scene::ScScene{};
  scCapture(scene, out, false);
}
const PxActor* g1_sc_actor_px(int32_t h) { return h >= 0 && size_t(h) < ES.hPx.size() ? ES.hPx[size_t(h)] : nullptr; }

// 진단 (G1_SEGV=1): 덤프 없이 호출 스택만 찍고 끝냄. 프로그램 시작 때 걸고, 스택 넘침도 잡게 대체 스택을 쓴다
__attribute__((constructor)) static void g1SegvInstall() {
  if (!getenv("G1_SEGV")) return;
  {  // backtrace 는 처음 부를 때 libgcc 를 불러오며 malloc 을 쓴다 -> 미리 한 번 (더미 힙이 깨진 뒤에도 핸들러가 돌게)
    void* w[2];
    backtrace(w, 2);
  }
  static char altStack[1 << 16];
  stack_t ss{};
  ss.ss_sp = altStack;
  ss.ss_size = sizeof(altStack);
  sigaltstack(&ss, nullptr);
  struct sigaction sa{};
  sa.sa_flags = SA_ONSTACK | SA_RESETHAND;
  sa.sa_flags |= SA_SIGINFO;
  sa.sa_sigaction = [](int sig, siginfo_t* si, void* uc) {
    // 스택 되짚기(backtrace)가 깨진 스택에서 다시 죽으므로 레지스터만: 명령 주소·스택 위 반환 주소 후보
    const ucontext_t* u = static_cast<const ucontext_t*>(uc);
    const uintptr_t rip = uintptr_t(u->uc_mcontext.gregs[REG_RIP]), rsp = uintptr_t(u->uc_mcontext.gregs[REG_RSP]),
                    rbp = uintptr_t(u->uc_mcontext.gregs[REG_RBP]);
    char buf[512];
    int n = snprintf(buf, sizeof(buf), "[g1 segv] 신호 %d 주소 %p rip %p rsp %p rbp %p 기준 g1_sc_report %p\n", sig, si->si_addr, (void*)rip, (void*)rsp, (void*)rbp,
                     (void*)&g1_sc_report);
    (void)!write(2, buf, size_t(n));
    // 스택 위 64 칸 (반환 주소 후보; 실행 파일 안 주소만 addr2line 으로 보면 됨)
    const uintptr_t* sp = reinterpret_cast<const uintptr_t*>(rsp);
    for (int k = 0; k < 64; ++k) {
      n = snprintf(buf, sizeof(buf), "[g1 segv 스택] %p\n", (void*)sp[k]);
      (void)!write(2, buf, size_t(n));
    }
    _exit(3);
  };
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
}
