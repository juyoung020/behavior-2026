// G1 쌍 관리층 그림자 (문서 15.3 v1-b, 리드, 비계 전용): 공식 radio 재생 중 PhysX Sc 층의 접촉 쌍 처리(거르기·상호작용·접촉 관리자 풀·
// 좁은 단계 목록·닿음 사건·섬 호출)를 우리 쌍 관리층(core/contact/sc_pairs.h)으로 같은 입력을 넣어 나란히 돌리고 매 스텝 비교한다.
// 입력·비교 방식은 contact 작업자의 tests/contact/test_sc_pairs.cpp 그대로(새 겹침·사라진 겹침·좁은 단계 결과·상호작용 활성/비활성을 --wrap 으로 뜸),
// 다른 점: 장면이 공식 radio 이고, 거르개는 omni(replay/omni_filter.h — 셰이더는 같은 함수, pair-found 는 재생기가 PhysX 에 준 콜백 객체를 그대로 부름),
// 장면 변경(조인트 생성·제거, 모양 거르기 자료 바꿈, 운동학 전환, 행위자 추가·제거)은 스텝마다 PhysX Sc 상태를 비교해 알아낸다.
// 섬 호출 가로채기는 g1_islands.cpp 의 --wrap 이 g1p_* 로 넘겨 준다(같은 기호를 두 번 감쌀 수 없음).
// 켜기: G1_PAIRS=1. G1_PAIRS_SHOW=1 이면 첫 다름을 자세히.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <cstddef>
#include <set>
#include <vector>

#include "PxPhysicsAPI.h"
#define private public
#define protected public
#include "NpScene.h"
#include "NpShape.h"
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "NpArticulationLink.h"
#include "NpArticulationReducedCoordinate.h"
#include "ScScene.h"
#include "ScBodySim.h"
#include "ScStaticSim.h"
#include "ScShapeSim.h"
#include "ScShapeInteraction.h"
#include "ScNPhaseCore.h"
#include "ScArticulationSim.h"
#include "ScArticulationCore.h"
#include "ScConstraintSim.h"
#include "ScConstraintInteraction.h"
#include "NpConstraint.h"
#include "PxsContext.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsNphaseImplementationContext.h"
#include "PxsSimpleIslandManager.h"
#include "PxsIslandSim.h"
#include "BpAABBManagerBase.h"
#undef private
#undef protected

#include "core/contact/sc_pairs.h"
#include "core/scene/pairs_log.h"
#include "core/scene/scene_file.h"
#include "omni_filter.h"
#include "g1_hooks.h"

using namespace physx;
namespace ss = eng::contact::sc;

namespace {

struct IslandCall {
  int op;
  uint64_t a, b, c, d, e;
  bool operator==(const IslandCall& o) const { return op == o.op && a == o.a && b == o.b && c == o.c && d == o.d && e == o.e; }
};
enum Op { OP_ADD_CM, OP_PREALLOC, OP_ADD_PREALLOC, OP_DELAYED, OP_CONNECT, OP_DISCONNECT, OP_REMOVE, OP_SET_RIGID_CM, OP_CLEAR_RIGID_CM, OP_DEACT_EDGE, OP_COUNT };
const char* kOpName[OP_COUNT] = {"addContactManager", "preallocateContactManagers", "addPreallocatedContactManager", "addDelayedDirtyEdges",
                                 "setEdgeConnected", "setEdgeDisconnected", "removeConnection", "setEdgeRigidCM", "clearEdgeRigidCM", "deactivateEdge"};
struct Capture {
  bool on = false;
  std::vector<IslandCall> calls[OP_COUNT];
  std::vector<uint32_t> preallocHandles, addCmEdges;
  struct Chunk { const void* base; std::vector<int32_t> pairs; };
  std::vector<Chunk> createdShapeChunks;
  std::vector<int32_t> createdTrigger, removedPairs;
  std::vector<int32_t> npCms;
  std::vector<uint8_t> npStatus, npPatches;
  std::vector<uint32_t> npWuFlags, npRest, npNpIndex, npShape0, npShape1, npDom;
  std::vector<std::pair<int32_t, int32_t>> touchFound, touchLost;
  struct Act { bool activate; int32_t e0, e1; bool result; bool afterFill; };
  std::vector<Act> acts;
  bool fillSeen = false;
  void clearStep() {
    acts.clear();
    fillSeen = false;
    for (auto& v : calls) v.clear();
    preallocHandles.clear();
    addCmEdges.clear();
    createdShapeChunks.clear();
    createdTrigger.clear();
    removedPairs.clear();
    npCms.clear(); npStatus.clear(); npPatches.clear(); npWuFlags.clear(); npRest.clear(); npNpIndex.clear();
    npShape0.clear(); npShape1.clear(); npDom.clear();
    touchFound.clear(); touchLost.clear();
  }
} G;
std::mutex gCapMu;
std::map<const PxsShapeCore*, int32_t> gCoreToElem;
std::map<int32_t, Sc::ElementSim*> gElemSim;
bool gInSimulate = false;  // simulate 안의 호출은 우리 층이 스스로 한다
std::vector<std::pair<int32_t, int32_t>> gApiResets;  // simulate 밖(사용자 자세 set 등)의 resetManagerCachedState, 부른 순서
uint64_t gSleepShapeChange = 0;

uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
std::pair<int32_t, int32_t> siElems(const Sc::Interaction* it) {
  const Sc::ElementSimInteraction* e = static_cast<const Sc::ElementSimInteraction*>(it);
  return {int32_t(e->getElement0().getElementID()), int32_t(e->getElement1().getElementID())};
}
uint64_t packPair(std::pair<int32_t, int32_t> p) { return (uint64_t(uint32_t(p.first)) << 32) | uint32_t(p.second); }

}  // namespace

// ---- 섬 호출 (g1_islands.cpp 의 --wrap 에서)
void g1p_addCM(void* cmv, uint64_t n0, uint64_t n1, void* it, uint32_t t, uint32_t e) {
  if (!G.on) return;
  PxsContactManager* cm = static_cast<PxsContactManager*>(cmv);
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_ADD_CM].push_back(IslandCall{OP_ADD_CM, cm ? cm->getIndex() : 0xffffffffull, n0, n1, it ? packPair(siElems(static_cast<Sc::Interaction*>(it))) : 0, t});
  G.addCmEdges.push_back(e);
}
void g1p_prealloc(uint32_t n, const uint32_t* h) {
  if (!G.on) return;
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_PREALLOC].push_back(IslandCall{OP_PREALLOC, n, 0, 0, 0, 0});
  for (uint32_t i = 0; i < n; ++i) G.preallocHandles.push_back(h[i]);
}
void g1p_addPrealloc(uint32_t e, void* cmv, uint64_t n0, uint64_t n1, void* it) {
  if (!G.on) return;
  PxsContactManager* cm = static_cast<PxsContactManager*>(cmv);
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_ADD_PREALLOC].push_back(IslandCall{OP_ADD_PREALLOC, e, cm ? cm->getIndex() : 0xffffffffull, n0, n1, it ? packPair(siElems(static_cast<Sc::Interaction*>(it))) : 0});
}
void g1p_delayed(uint32_t n, const uint32_t* e) {
  if (!G.on) return;
  uint64_t h = n;
  for (uint32_t i = 0; i < n; ++i) h = h * 1000003ull + e[i];
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_DELAYED].push_back(IslandCall{OP_DELAYED, n, h, 0, 0, 0});
}
void g1p_edge(int op, uint32_t e) {
  if (!G.on) return;
  const int o = op == 5 ? OP_DISCONNECT : op == 6 ? OP_REMOVE : op == 8 ? OP_CLEAR_RIGID_CM : OP_DEACT_EDGE;
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[o].push_back(IslandCall{o, e, 0, 0, 0, 0});
}
void g1p_connect(uint32_t e, uint32_t t) {
  if (!G.on) return;
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_CONNECT].push_back(IslandCall{OP_CONNECT, e, t, 0, 0, 0});
}
void g1p_setRigidCM(uint32_t e, void* cmv) {
  if (!G.on) return;
  PxsContactManager* cm = static_cast<PxsContactManager*>(cmv);
  std::lock_guard<std::mutex> lk(gCapMu);
  G.calls[OP_SET_RIGID_CM].push_back(IslandCall{OP_SET_RIGID_CM, e, cm ? cm->getIndex() : 0xffffffffull, 0, 0, 0});
}
bool g1p_isConstraintEdge(void* sim, uint32_t e) {  // removeConnection: 조인트 간선은 joints 몫이라 기록하지 않는다
  if (!G.on) return false;
  const Sc::Interaction* it = static_cast<IG::SimpleIslandManager*>(sim)->getInteractionFromEdgeIndex(e);
  return it && it->getType() == Sc::InteractionType::eCONSTRAINTSHADER;
}

// ---- 입력 --wrap (test_sc_pairs.cpp 와 같음)
#define W(name) __wrap_##name
#define R(name) __real_##name
extern "C" {
bool R(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(Sc::ShapeInteraction*);
void R(_ZNK5physx2Sc10NPhaseCore17runOverlapFiltersEjPNS_2Bp11AABBOverlapEPNS0_10FilterInfoERjS7_)(const Sc::NPhaseCore*, PxU32, Bp::AABBOverlap*,
                                                                                                  Sc::FilterInfo*, PxU32&, PxU32&);
void W(_ZNK5physx2Sc10NPhaseCore17runOverlapFiltersEjPNS_2Bp11AABBOverlapEPNS0_10FilterInfoERjS7_)(const Sc::NPhaseCore* c, PxU32 n, Bp::AABBOverlap* p,
                                                                                                  Sc::FilterInfo* f, PxU32& k, PxU32& s) {
  if (G.on) {
    Capture::Chunk ch;
    ch.base = p;
    for (PxU32 i = 0; i < n; ++i) {
      ch.pairs.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(p[i].mUserData0)->getElementID()));
      ch.pairs.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(p[i].mUserData1)->getElementID()));
    }
    std::lock_guard<std::mutex> lk(gCapMu);
    G.createdShapeChunks.push_back(ch);
  }
  R(_ZNK5physx2Sc10NPhaseCore17runOverlapFiltersEjPNS_2Bp11AABBOverlapEPNS0_10FilterInfoERjS7_)(c, n, p, f, k, s);
}
void R(_ZN5physx2Sc10NPhaseCore23onTriggerOverlapCreatedEPKNS_2Bp11AABBOverlapEj)(Sc::NPhaseCore*, const Bp::AABBOverlap*, PxU32);
void W(_ZN5physx2Sc10NPhaseCore23onTriggerOverlapCreatedEPKNS_2Bp11AABBOverlapEj)(Sc::NPhaseCore* c, const Bp::AABBOverlap* p, PxU32 n) {
  if (G.on)
    for (PxU32 i = 0; i < n; ++i) {
      G.createdTrigger.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(p[i].mUserData0)->getElementID()));
      G.createdTrigger.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(p[i].mUserData1)->getElementID()));
    }
  R(_ZN5physx2Sc10NPhaseCore23onTriggerOverlapCreatedEPKNS_2Bp11AABBOverlapEj)(c, p, n);
}
void R(_ZN5physx2Sc10NPhaseCore16onOverlapRemovedEPNS0_10ElementSimES3_jPvRNS_31PxsContactManagerOutputIteratorE)(Sc::NPhaseCore*, Sc::ElementSim*,
                                                                                                             Sc::ElementSim*, PxU32, void*,
                                                                                                             PxsContactManagerOutputIterator&);
void W(_ZN5physx2Sc10NPhaseCore16onOverlapRemovedEPNS0_10ElementSimES3_jPvRNS_31PxsContactManagerOutputIteratorE)(Sc::NPhaseCore* c, Sc::ElementSim* a,
                                                                                                             Sc::ElementSim* b, PxU32 ccd, void* es,
                                                                                                             PxsContactManagerOutputIterator& o) {
  if (G.on) {
    G.removedPairs.push_back(int32_t(a->getElementID()));
    G.removedPairs.push_back(int32_t(b->getElementID()));
  }
  R(_ZN5physx2Sc10NPhaseCore16onOverlapRemovedEPNS0_10ElementSimES3_jPvRNS_31PxsContactManagerOutputIteratorE)(c, a, b, ccd, es, o);
}
bool R(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(Sc::ShapeInteraction*, PxsContactManager*);
bool W(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(Sc::ShapeInteraction* si, PxsContactManager* cm) {
  const bool r = R(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(si, cm);
  if (G.on) { const auto e = siElems(si); G.acts.push_back(Capture::Act{true, e.first, e.second, r, G.fillSeen}); }
  return r;
}
void R(_ZNK5physx2Sc16ShapeInteraction23resetManagerCachedStateEv)(const Sc::ShapeInteraction*);
void W(_ZNK5physx2Sc16ShapeInteraction23resetManagerCachedStateEv)(const Sc::ShapeInteraction* si) {
  if (G.on && !gInSimulate) gApiResets.push_back(siElems(si));
  R(_ZNK5physx2Sc16ShapeInteraction23resetManagerCachedStateEv)(si);
}
void R(_ZN5physx2Sc16ShapeInteraction26onShapeChangeWhileSleepingEb)(Sc::ShapeInteraction*, bool);
void W(_ZN5physx2Sc16ShapeInteraction26onShapeChangeWhileSleepingEb)(Sc::ShapeInteraction* si, bool d) {
  if (G.on && !gInSimulate) ++gSleepShapeChange;
  R(_ZN5physx2Sc16ShapeInteraction26onShapeChangeWhileSleepingEb)(si, d);
}
bool W(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(Sc::ShapeInteraction* si) {
  const auto e = siElems(si);
  const bool r = R(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(si);
  if (G.on) G.acts.push_back(Capture::Act{false, e.first, e.second, r, G.fillSeen});
  return r;
}
void R(_ZN5physx10PxsContext22fillManagerTouchEventsEPNS_27PxvContactManagerTouchEventERjS2_S3_S2_S3_)(PxsContext*, PxvContactManagerTouchEvent*, PxU32&,
                                                                                                    PxvContactManagerTouchEvent*, PxU32&,
                                                                                                    PxvContactManagerTouchEvent*, PxU32&);
void W(_ZN5physx10PxsContext22fillManagerTouchEventsEPNS_27PxvContactManagerTouchEventERjS2_S3_S2_S3_)(PxsContext* ctx, PxvContactManagerTouchEvent* nt,
                                                                                                    PxU32& nc, PxvContactManagerTouchEvent* lt,
                                                                                                    PxU32& lc, PxvContactManagerTouchEvent* ct,
                                                                                                    PxU32& cc) {
  if (G.on) {
    PxsNphaseImplementationContext* np = static_cast<PxsNphaseImplementationContext*>(ctx->getNphaseImplementationContext());
    const PxsContactManagers& L = np->mNarrowPhasePairs;
    for (PxU32 i = 0; i < L.mContactManagerMapping.size(); ++i) {
      PxsContactManager* cm = L.mContactManagerMapping[i];
      const PxcNpWorkUnit& u = cm->getWorkUnit();
      G.npCms.push_back(int32_t(cm->getIndex()));
      G.npStatus.push_back(L.mOutputContactManagers[i].statusFlag);
      G.npPatches.push_back(L.mOutputContactManagers[i].nbPatches);
      G.npWuFlags.push_back(u.mFlags);
      G.npRest.push_back(fbits(u.mRestDistance));
      G.npNpIndex.push_back(u.mNpIndex);
      G.npShape0.push_back(uint32_t(gCoreToElem[u.getShapeCore0()]));
      G.npShape1.push_back(uint32_t(gCoreToElem[u.getShapeCore1()]));
      G.npDom.push_back(uint32_t(u.getDominance0()) | (uint32_t(u.getDominance1()) << 8));
    }
  }
  R(_ZN5physx10PxsContext22fillManagerTouchEventsEPNS_27PxvContactManagerTouchEventERjS2_S3_S2_S3_)(ctx, nt, nc, lt, lc, ct, cc);
  if (G.on) {
    G.fillSeen = true;
    for (PxU32 i = 0; i < nc; ++i) G.touchFound.push_back(siElems(reinterpret_cast<Sc::ShapeInteraction*>(nt[i].getCMTouchEventUserData())));
    for (PxU32 i = 0; i < lc; ++i) G.touchLost.push_back(siElems(reinterpret_cast<Sc::ShapeInteraction*>(lt[i].getCMTouchEventUserData())));
  }
}
}  // extern "C"

namespace {

namespace sc2 = eng::scene;

// ---- 우리 섬 갈고리: 돌려줄 값은 기록(PhysX 가 받은 값), 호출은 모아 PhysX 호출과 비교
struct Hooks : public sc2::PairsHooks {
  std::vector<IslandCall> mine[OP_COUNT];
  void onCall(int op, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) override { mine[op].push_back(IslandCall{op, a, b, c, d, e}); }
  void clearMine() {
    for (auto& v : mine) v.clear();
  }
};

// ---- 상태
struct PairsShadow {
  bool inited = false, on = false, started = false;
  PxScene* scene = nullptr;
  Sc::Scene* sc = nullptr;
  std::unique_ptr<ss::ScPairs> Mp{new ss::ScPairs};  // 파일 넘겨받기 때 새로 만들려고 포인터로
  Hooks hooks;
  std::map<const Sc::ActorSim*, int32_t> actorIndex;
  std::vector<const PxActor*> actorPx;          // 행위자 번호 -> PhysX 행위자 (pair-found 콜백에 넘김)
  std::map<int32_t, const PxShape*> shapePx;    // 요소 번호 -> PhysX 모양
  std::map<const Sc::ArticulationSim*, int32_t> artIndex;
  std::map<const Sc::Interaction*, int32_t> extIndex;  // 조인트·관절체 관절 상호작용 -> 우리 자리표
  std::map<int32_t, ss::FilterData> lastFD;             // 요소 -> 지난 거르기 자료
  std::map<int32_t, uint32_t> lastAttr;                 // 행위자 -> 지난 거르기 속성
  std::set<int32_t> liveElems;
  // 이번 스텝 입력 (넘겨받기 기록과 같은 꼴)
  sc2::PairsStep step;
  std::vector<const Sc::Interaction*> stepAdded;  // 이번 스텝 새 조인트 상호작용 (EXT_ADD 순서)
  // 넘겨받기 기록 (G1_DUMP_AT 앞까지) / 파일에서 다시 세우기 (G1_PAIRS_FROM)
  long long logUntil = -1;
  sc2::PairsLog log;
  std::string from;
  bool fromDone = false;
  // 통계
  uint64_t steps = 0, bad = 0, cmpList = 0, cmpEvents = 0, cmpCalls = 0, cmpActor = 0, cmpPool = 0;
  uint64_t nCreated = 0, nRemoved = 0, nTouch = 0, nAct = 0, nDeact = 0, nRefilter = 0, nKinToggle = 0, nJointAdd = 0, nJointRemove = 0, nShapeAdd = 0,
           nShapeRemove = 0, nApiReset = 0, nSleepChange = 0;
  long long firstBad = -1;
  std::string firstWhat;
  bool show = false;
  uint64_t curSim = 0;
} PS;

bool logging() { return PS.logUntil >= 0 && (long long)PS.curSim < PS.logUntil; }

void fail(const char* what) {
  if (!PS.bad) {
    PS.firstBad = (long long)PS.curSim;
    PS.firstWhat = what;
    if (PS.show) printf("  [g1 pairs] simulate %llu 첫 다름: %s\n", (unsigned long long)PS.curSim, what);
  }
  ++PS.bad;
}

// omni 거르개를 우리 모양으로 (셰이더 = 같은 함수)
uint32_t ourShader(uint32_t a0, const ss::FilterData& f0, uint32_t a1, const ss::FilterData& f1, uint32_t& pf, const void* data) {
  PxFilterData d0(f0.word0, f0.word1, f0.word2, f0.word3), d1(f1.word0, f1.word1, f1.word2, f1.word3);
  PxPairFlags p;
  const PxFilterFlags r = engine::OmniFilterShader(a0, d0, a1, d1, p, data, 0);
  pf = PxU16(p);
  return PxU16(r);
}
// pair-found = 재생기가 PhysX 에 준 콜백 객체를 그대로 (행위자·모양 번호 -> PhysX 객체)
uint32_t ourPairFound(uint64_t id, uint32_t a0, const ss::FilterData& f0, int32_t actor0, int32_t shape0, uint32_t a1, const ss::FilterData& f1, int32_t actor1,
                      int32_t shape1, uint32_t& pf, const void*) {
  PxSimulationFilterCallback* cb = PS.sc->getFilterCallbackFast();
  if (!cb) return 0;
  PxFilterData d0(f0.word0, f0.word1, f0.word2, f0.word3), d1(f1.word0, f1.word1, f1.word2, f1.word3);
  PxPairFlags p = PxPairFlags(PxU16(pf));
  const PxActor* x0 = actor0 >= 0 && size_t(actor0) < PS.actorPx.size() ? PS.actorPx[size_t(actor0)] : nullptr;
  const PxActor* x1 = actor1 >= 0 && size_t(actor1) < PS.actorPx.size() ? PS.actorPx[size_t(actor1)] : nullptr;
  const PxShape* s0 = PS.shapePx.count(shape0) ? PS.shapePx[shape0] : nullptr;
  const PxShape* s1 = PS.shapePx.count(shape1) ? PS.shapePx[shape1] : nullptr;
  if (!x0 || !x1 || !s0 || !s1) { fail("pair-found: 행위자·모양 없음"); return 0; }
  const PxFilterFlags r = cb->pairFound(id, a0, d0, x0, s0, a1, d1, x1, s1, p);
  pf = PxU16(p);
  return PxU16(r);
}

int32_t actorOf(Sc::ActorSim* as) {
  auto it = PS.actorIndex.find(as);
  if (it != PS.actorIndex.end()) return it->second;
  const int32_t idx = int32_t(PS.step.actors.size());
  PS.actorIndex[as] = idx;
  PS.step.actors.push_back(sc2::PairsActorIn{});
  PS.actorPx.push_back(nullptr);
  return idx;
}
// 행위자 입력 값 (ss::Actor 칸과 같은 뜻)
void refreshActor(Sc::ActorSim* as, const PxActor* px) {
  const int32_t ai = actorOf(as);
  sc2::PairsActorIn& A = PS.step.actors[size_t(ai)];
  PS.actorPx[size_t(ai)] = px;
  A = sc2::PairsActorIn{};
  A.articulation = -1;
  A.parentLinkId = ss::INVALID;
  A.type = int32_t(as->getActorType());
  A.filterAttr = as->getFilterAttributes();
  A.actorID = as->getActorID();
  A.nodeIndex = as->getNodeIndex().getInd();
  A.hasConstraints = as->readInternalFlag(Sc::ActorSim::BF_HAS_CONSTRAINTS) ? 1 : 0;
  A.dominanceGroup = as->getActorCore().getDominanceGroup();
  if (as->isDynamicRigid()) {
    Sc::BodySim* bs = static_cast<Sc::BodySim*>(as);
    A.fixedBaseLink = bs->getLowLevelBody().mCore->fixedBaseLink != 0;
    A.offsetSlop = bs->getBodyCore().getCore().offsetSlop;
    A.forceStaticKineNotif = bs->getBodyCore().getCore().mFlags.isSet(PxRigidBodyFlag::eFORCE_STATIC_KINE_NOTIFICATIONS);
    A.forceKineKineNotif = bs->getBodyCore().getCore().mFlags.isSet(PxRigidBodyFlag::eFORCE_KINE_KINE_NOTIFICATIONS);
    if (bs->isArticulationLink()) {
      Sc::ArticulationSim* asim = bs->getArticulation();
      auto aj = PS.artIndex.find(asim);
      if (aj == PS.artIndex.end()) aj = PS.artIndex.emplace(asim, int32_t(PS.artIndex.size())).first;
      A.articulation = aj->second;
      A.linkId = bs->getNodeIndex().articulationLinkId();
      A.parentLinkId = asim->getLink(A.linkId).parent;
      A.artDisableSelfCollision = asim->getCore().getArticulationFlags().isSet(PxArticulationFlag::eDISABLE_SELF_COLLISION);
    }
  }
}

// 장면의 모든 강체 행위자 (정적·동적·관절체 링크)
std::vector<PxRigidActor*> sceneActors(PxScene* scene) {
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

// 이번 스텝 앞 입력을 PhysX 에서 뜬다 (PS.step 의 앞 칸). 장면 변경은 PhysX 가 한 순서대로 연산 목록에.
void captureScene(bool first) {
  PS.step.shapes.resize(PS.Mp->shapes.size());
  for (size_t i = 0; i < PS.Mp->shapes.size(); ++i) PS.step.shapes[i] = PS.Mp->shapes[i];  // 없어진 모양의 칸도 그대로 이어 간다
  PS.step.ops.clear();
  PS.step.joints.clear();
  std::set<int32_t> seen;
  std::vector<int32_t> refiltered;
  std::vector<std::pair<int32_t, bool>> kinToggles;
  for (PxRigidActor* a : sceneActors(PS.scene)) {
    const PxU32 n = a->getNbShapes();
    std::vector<PxShape*> sh(n);
    a->getShapes(sh.data(), n);
    for (PxShape* s : sh) {
      Sc::ShapeCore& core = static_cast<NpShape*>(s)->getCore();
      Sc::ShapeSim* sim = core.getExclusiveSim();
      if (!sim) continue;
      const int32_t e = int32_t(sim->getElementID());
      seen.insert(e);
      if (size_t(e) >= PS.step.shapes.size()) PS.step.shapes.resize(size_t(e) + 1);
      ss::Shape& S = PS.step.shapes[size_t(e)];
      Sc::ActorSim* as = &sim->getActor();
      const int32_t ai = actorOf(as);
      const uint32_t oldAttr = PS.lastAttr.count(ai) ? PS.lastAttr[ai] : 0xffffffffu;
      refreshActor(as, a);
      const uint32_t newAttr = PS.step.actors[size_t(ai)].filterAttr;
      if (!first && oldAttr != 0xffffffffu && ((oldAttr ^ newAttr) & ss::FilterObj::eKINEMATIC))
        if (std::find_if(kinToggles.begin(), kinToggles.end(), [&](auto& k) { return k.first == ai; }) == kinToggles.end())
          kinToggles.push_back({ai, (newAttr & ss::FilterObj::eKINEMATIC) != 0});
      PS.lastAttr[ai] = newAttr;
      if (!first && !PS.liveElems.count(e)) ++PS.nShapeAdd;
      S.valid = true;
      S.actor = ai;
      S.geomType = int32_t(sim->getGeometryType());
      S.trigger = (sim->getFlags() & PxShapeFlag::eTRIGGER_SHAPE) != 0;
      const PxFilterData fd = core.getSimulationFilterData();
      const ss::FilterData nfd{fd.word0, fd.word1, fd.word2, fd.word3};
      auto lf = PS.lastFD.find(e);
      if (!first && lf != PS.lastFD.end() && PS.liveElems.count(e) &&
          (lf->second.word0 != nfd.word0 || lf->second.word1 != nfd.word1 || lf->second.word2 != nfd.word2 || lf->second.word3 != nfd.word3))
        refiltered.push_back(e);
      PS.lastFD[e] = nfd;
      S.fd = nfd;
      S.restOffset = sim->getRestOffset();
      S.torsionalPatchRadius = sim->getTorsionalPatchRadius();
      S.minTorsionalPatchRadius = sim->getMinTorsionalPatchRadius();
      S.transformCacheId = sim->getTransformCacheID();
      gCoreToElem[&core.getCore()] = e;
      gElemSim[e] = sim;
      PS.shapePx[e] = s;
    }
  }
  // 조인트 충돌 표 (Scene::findConstraintCore: 행위자 쌍의 첫 조인트)
  {
    std::set<uint64_t> keys;
    const PxU32 nc = PS.scene->getNbConstraints();
    std::vector<PxConstraint*> cs(nc);
    PS.scene->getConstraints(cs.data(), nc);
    for (PxConstraint* c : cs) {
      Sc::ConstraintSim* csim = static_cast<NpConstraint*>(c)->getCore().getSim();
      if (!csim) continue;
      const Sc::Interaction* ci = reinterpret_cast<const Sc::Interaction*>(csim->getInteraction());
      if (!ci) continue;
      Sc::ActorSim* s0 = &const_cast<Sc::Interaction*>(ci)->getActorSim0();
      Sc::ActorSim* s1 = &const_cast<Sc::Interaction*>(ci)->getActorSim1();
      if (!PS.actorIndex.count(s0) || !PS.actorIndex.count(s1)) continue;
      const int32_t a0 = PS.actorIndex[s0], a1 = PS.actorIndex[s1];
      const uint64_t k = (uint64_t(uint32_t(std::min(a0, a1))) << 32) | uint32_t(std::max(a0, a1));
      if (!keys.insert(k).second) continue;
      Sc::ConstraintCore* cc = PS.sc->findConstraintCore(s0, s1);
      PS.step.joints.push_back(sc2::PairsJoint{k, cc ? (cc->getFlags().isSet(PxConstraintFlag::eCOLLISION_ENABLED) ? 1u : 0u) : 1u, 0});
    }
  }
  if (first) {
    PS.liveElems = seen;
    return;
  }
  // PhysX 순서: 거르기 자료·운동학 전환 -> 조인트 -> 모양·행위자 빼기 (test_sc_pairs 와 같음) -> 사용자 자세 set 의 관리자 다시 등록
  for (int32_t e : refiltered) {
    PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_REFILTER, e, 0, 0});
    ++PS.nRefilter;
  }
  for (auto& k : kinToggles) {
    PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_KIN, k.first, k.second ? 1 : 0, 0});
    ++PS.nKinToggle;
    fail("운동학 전환 (모양 다시 넣기 번호 추적은 아직)");
  }
  std::set<const Sc::Interaction*> nowExt;
  std::vector<const Sc::Interaction*> added;
  for (auto& kv : PS.actorIndex) {
    const Sc::ActorSim* as = kv.first;
    for (PxU32 i = 0; i < as->getActorInteractionCount(); ++i) {
      const Sc::Interaction* it = as->getActorInteractions()[i];
      if (it->getType() <= Sc::InteractionType::eMARKER) continue;
      nowExt.insert(it);
      if (!PS.extIndex.count(it) && std::find(added.begin(), added.end(), it) == added.end()) added.push_back(it);
    }
  }
  for (auto it = PS.extIndex.begin(); it != PS.extIndex.end();) {
    if (!nowExt.count(it->first)) {
      PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_EXT_REMOVE, it->second, 0, 0});
      ++PS.nJointRemove;
      it = PS.extIndex.erase(it);
    } else
      ++it;
  }
  PS.stepAdded.assign(added.begin(), added.end());
  for (const Sc::Interaction* it : added) {
    Sc::ActorSim* s0 = &const_cast<Sc::Interaction*>(it)->getActorSim0();
    Sc::ActorSim* s1 = &const_cast<Sc::Interaction*>(it)->getActorSim1();
    const int32_t a0 = PS.actorIndex.count(s0) ? PS.actorIndex[s0] : -1;
    const int32_t a1 = PS.actorIndex.count(s1) ? PS.actorIndex[s1] : -1;
    PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_EXT_ADD, a0, a1, int32_t(it->getType())});
    ++PS.nJointAdd;
  }
  for (int32_t e : PS.liveElems)
    if (!seen.count(e)) {
      PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_SHAPE_REMOVE, e, 0, 0});
      ++PS.nShapeRemove;
    }
  PS.liveElems = seen;
  for (auto& e : gApiResets) {
    PS.step.ops.push_back(sc2::PairsPreOp{sc2::PPO_API_RESET, e.first, e.second, 0});
    ++PS.nApiReset;
  }
}

void setFilters(ss::ScPairs& M) {
  M.filterShader = ourShader;
  M.filterShaderData = PS.scene->getFilterShaderData();
  M.filterPairFound = PS.sc->getFilterCallbackFast() ? ourPairFound : nullptr;
}

void start(PxScene* scene) {
  PS.started = true;
  PS.scene = scene;
  PS.sc = &static_cast<NpScene*>(scene)->getScScene();
  PS.hooks.m = PS.Mp.get();
  PS.Mp->islands = &PS.hooks;
  setFilters(*PS.Mp);
  captureScene(true);
  sc2::PairsLog& L = PS.log;
  L = sc2::PairsLog{};
  L.eltsPerSlab = PS.sc->getLowLevelContext()->mContactManagerPool.mEltsPerSlab;
  L.kineKine = int32_t(PS.sc->getKineKineFilteringMode());
  L.staticKine = int32_t(PS.sc->getStaticKineFilteringMode());
  L.actors0 = PS.step.actors;
  L.shapes0 = PS.step.shapes;
  L.joints0 = PS.step.joints;
  // 조인트·관절체 관절 상호작용: 행위자 목록에 섞인 순서 그대로 자리표로
  std::map<const Sc::Interaction*, int32_t> rec;
  for (auto& kv : PS.actorIndex) {
    const Sc::ActorSim* as = kv.first;
    for (PxU32 i = 0; i < as->getActorInteractionCount(); ++i) {
      const Sc::Interaction* it = as->getActorInteractions()[i];
      if (it->getType() <= Sc::InteractionType::eMARKER) {
        fail("시작 때 이미 겹침 상호작용이 있음");
        continue;
      }
      auto e = rec.find(it);
      if (e == rec.end()) {
        Sc::ActorSim* s0 = &const_cast<Sc::Interaction*>(it)->getActorSim0();
        Sc::ActorSim* s1 = &const_cast<Sc::Interaction*>(it)->getActorSim1();
        const int32_t a0 = PS.actorIndex.count(s0) ? PS.actorIndex[s0] : -1;
        const int32_t a1 = PS.actorIndex.count(s1) ? PS.actorIndex[s1] : -1;
        e = rec.emplace(it, int32_t(L.extRecords.size())).first;
        L.extRecords.push_back(sc2::PairsPreOp{0, a0, a1, int32_t(it->getType())});
      }
      L.appends.push_back({kv.second, e->second});
    }
  }
  const std::vector<int32_t> ids = sc2::pairsStart(*PS.Mp, L);
  for (auto& kv : rec) PS.extIndex[kv.first] = ids[size_t(kv.second)];
  L.valid = true;
  G.clearStep();
  G.on = true;
}

}  // namespace

// simulate 바로 앞
void g1_pairs_before(PxScene* scene, uint64_t sim) {
  if (!PS.inited) {
    PS.inited = true;
    PS.on = getenv("G1_PAIRS") != nullptr;
    PS.show = getenv("G1_PAIRS_SHOW") != nullptr;
    if (const char* a = getenv("G1_DUMP_AT")) PS.logUntil = atoll(a);
    if (const char* f = getenv("G1_PAIRS_FROM")) PS.from = f;
  }
  if (!PS.on) return;
  PS.curSim = sim;
  if (!PS.started) {
    start(scene);
  } else if (scene != PS.scene) {
    return;
  } else {
    // 파일 넘겨받기: 경계 simulate 에서 우리 층을 기록으로 처음부터 다시 세운다 (그 뒤로만 비교)
    if (!PS.from.empty() && !PS.fromDone) {
      static sc2::SceneFile ff;
      static int state = 0;
      if (state == 0) {
        std::string err;
        state = sc2::readScene(PS.from.c_str(), ff, &err) && ff.pairs.valid ? 1 : 2;
        if (state == 2) fprintf(stderr, "[g1 pairs] 파일 쌍 관리층 기록 읽기 실패: %s\n", err.c_str());
      }
      if (state == 1 && sim == ff.h.sim) {
        PS.fromDone = true;
        PS.Mp.reset(new ss::ScPairs);
        setFilters(*PS.Mp);
        uint32_t actBad = 0;
        sc2::PairsLog lg = ff.pairs;
        if (getenv("G1_PAIRS_FROM_NEG") && lg.steps.size() > 1) {  // 음성 대조: 새 겹침이 가장 많은 스텝 뒤쪽 입력(겹침)을 비우면 달라져야 한다
          size_t best = 0;
          for (size_t k = 1; k < lg.steps.size(); ++k)
            if (lg.steps[k].created.size() > lg.steps[best].created.size()) best = k;
          lg.steps[best].created.clear();
        }
        sc2::pairsReplay(*PS.Mp, PS.hooks, lg, &actBad);
        PS.hooks.m = PS.Mp.get();
        PS.Mp->islands = &PS.hooks;
        printf("G1 쌍 관리층 넘겨받기: simulate %llu 에서 파일 기록(스텝 %zu)으로 우리 층을 다시 세움 (활성화 재생 어긋남 %u) — 이 뒤로만 비교\n",
               (unsigned long long)sim, ff.pairs.steps.size(), actBad);
        PS.steps = PS.bad = PS.cmpList = PS.cmpEvents = PS.cmpCalls = PS.cmpActor = PS.cmpPool = 0;
        PS.firstBad = -1;
      }
    }
    captureScene(false);
    std::vector<int32_t> extIds;
    sc2::pairsPre(*PS.Mp, PS.hooks, PS.step, &extIds);
    for (size_t k = 0; k < extIds.size() && k < PS.stepAdded.size(); ++k) PS.extIndex[PS.stepAdded[k]] = extIds[k];
    PS.nSleepChange += gSleepShapeChange;  // 잠든 쌍의 onShapeChangeWhileSleeping: 깨우기 목록만 (쌍 관리층 영향 없음)
  }
  gApiResets.clear();
  gSleepShapeChange = 0;
  gInSimulate = true;
  // 새 겹침 때 쓸 노드 활성 상태 = simulate 직전 추측 섬
  const IG::IslandSim& is = PS.sc->getSimpleIslandManager()->getSpeculativeIslandSim();
  PS.step.nodes.clear();
  PS.hooks.nodeActive.clear();
  for (auto& kv : PS.actorIndex) {
    const PxNodeIndex n = kv.first->getNodeIndex();
    if (n.isValid()) {
      const bool act = is.getNode(n).isActive();
      PS.step.nodes.push_back(sc2::PairsNode{n.getInd(), act ? 1u : 0u, 0});
      PS.hooks.nodeActive[n.getInd()] = act;
    }
  }
}

// fetchResults 뒤: 이 스텝 입력을 우리 층에 PhysX 순서로 넣고 비교
void g1_pairs_after(PxScene* scene, uint64_t sim) {
  if (!PS.on || !PS.started || scene != PS.scene) return;
  (void)sim;
  gInSimulate = false;
  ++PS.steps;
  ss::ScPairs& M = *PS.Mp;
  Hooks& hooks = PS.hooks;
  std::lock_guard<std::mutex> lk(gCapMu);
  hooks.clearMine();
  // 뒤 입력 (PhysX 가 받은 것)
  sc2::PairsStep& S = PS.step;
  std::sort(G.createdShapeChunks.begin(), G.createdShapeChunks.end(), [](const Capture::Chunk& a, const Capture::Chunk& b) { return a.base < b.base; });
  S.created.clear();
  for (auto& c : G.createdShapeChunks) S.created.insert(S.created.end(), c.pairs.begin(), c.pairs.end());
  S.createdTrigger = G.createdTrigger;
  S.removedPairs = G.removedPairs;
  S.acts.clear();
  for (const auto& a : G.acts) S.acts.push_back(sc2::PairsAct{uint8_t(a.activate), uint8_t(a.result), uint8_t(a.afterFill), 0, a.e0, a.e1});
  S.npStatus = G.npStatus;
  S.npPatches = G.npPatches;
  S.preallocHandles = G.preallocHandles;
  S.addCmEdges = G.addCmEdges;
  PS.nCreated += S.created.size() / 2;
  PS.nRemoved += S.removedPairs.size() / 2;
  for (const auto& a : G.acts)
    if (a.result) (a.activate ? PS.nAct : PS.nDeact)++;
  auto check = [&](int stage) {
    if (stage == 0) {
      bool same = M.npMain.size() == G.npCms.size();
      for (uint32_t i = 0; same && i < M.npMain.size(); ++i) {
        const ss::ContactManager& c = M.cmsData[size_t(M.npMain.cms[i])];
        ++PS.cmpList;
        same = M.npMain.cms[i] == G.npCms[i] && uint32_t(c.shape0) == G.npShape0[i] && uint32_t(c.shape1) == G.npShape1[i] && c.wuFlags == G.npWuFlags[i] &&
               fbits(c.restDistance) == G.npRest[i] && c.npIndex == G.npNpIndex[i] && (uint32_t(c.dominance0) | (uint32_t(c.dominance1) << 8)) == G.npDom[i];
        if (!same && PS.show && !PS.bad)
          printf("    칸 %u: PhysX cm %d (%u,%u) 플래그 %x np %x / 우리 cm %d (%d,%d) 플래그 %x np %x\n", i, G.npCms[i], G.npShape0[i], G.npShape1[i], G.npWuFlags[i],
                 G.npNpIndex[i], M.npMain.cms[i], c.shape0, c.shape1, c.wuFlags, c.npIndex);
      }
      if (!same) {
        if (PS.show && !PS.bad) printf("    목록 길이 PhysX %zu / 우리 %u\n", G.npCms.size(), M.npMain.size());
        fail("좁은 단계 목록");
      }
    } else {
      auto cmpEv = [&](const ss::Vec<ss::TouchEvent>& ours, const std::vector<std::pair<int32_t, int32_t>>& px, const char* name) {
        bool same = ours.size() == px.size();
        for (size_t i = 0; same && i < ours.size(); ++i) {
          ++PS.cmpEvents;
          const ss::Interaction& I = M.inters[size_t(ours[i].inter)];
          same = I.elem0 == px[i].first && I.elem1 == px[i].second;
        }
        if (!same) fail(name);
      };
      cmpEv(M.touchFound, G.touchFound, "닿음 시작 사건");
      cmpEv(M.touchLost, G.touchLost, "닿음 끝 사건");
      PS.nTouch += G.touchFound.size();
    }
  };
  if (sc2::pairsPost(M, hooks, S, check)) fail("활성화 재생");
  if (logging()) PS.log.steps.push_back(S);
  // 섬 호출 (종류별 순서)
  for (int op = 0; op < OP_COUNT; ++op) {
    if (op == OP_DELAYED) continue;  // isDirty 는 섬 상태 (solver)
    std::vector<IslandCall> a = G.calls[op], b = hooks.mine[op];
    if (op == OP_ADD_PREALLOC) {
      auto byEdge = [](const IslandCall& x, const IslandCall& y) { return x.a < y.a; };
      std::sort(a.begin(), a.end(), byEdge);
      std::sort(b.begin(), b.end(), byEdge);
    }
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) {
      ++PS.cmpCalls;
      same = a[i] == b[i];
    }
    if (!same) {
      if (PS.show && !PS.bad) printf("    %s: PhysX %zu 번 / 우리 %zu 번\n", kOpName[op], a.size(), b.size());
      fail(kOpName[op]);
    }
  }
  // 스텝 끝: 행위자 상호작용 목록, 풀 빈 칸, 좁은 단계 목록
  for (auto& kv : PS.actorIndex) {
    const Sc::ActorSim* as = kv.first;
    const ss::Actor& A = M.actors[size_t(kv.second)];
    const PxU32 n = as->getActorInteractionCount();
    bool same = n == A.interactions.size();
    for (PxU32 i = 0; same && i < n; ++i) {
      ++PS.cmpActor;
      const Sc::Interaction* it = as->getActorInteractions()[i];
      const ss::Interaction& I = M.inters[size_t(A.interactions[i])];
      if (it->getType() > Sc::InteractionType::eMARKER) {
        const auto e = PS.extIndex.find(it);
        same = e != PS.extIndex.end() && e->second == A.interactions[i];
        continue;
      }
      const auto e = siElems(it);
      same = int(it->getType()) == int(I.type) && e.first == I.elem0 && e.second == I.elem1;
    }
    if (!same) fail("행위자 상호작용 목록");
  }
  {
    PxsContext* ctx = PS.sc->getLowLevelContext();
    const auto& pool = ctx->mContactManagerPool;
    bool same = pool.mFreeCount == M.cmPool.freeList.size();
    for (PxU32 i = 0; same && i < pool.mFreeCount; ++i) same = int32_t(pool.mFreeList[i]->getIndex()) == M.cmPool.freeList[i];
    ++PS.cmpPool;
    if (!same) fail("접촉 관리자 풀 빈 칸 목록");
    const PxsContactManagers& L = static_cast<PxsNphaseImplementationContext*>(ctx->getNphaseImplementationContext())->mNarrowPhasePairs;
    same = L.mContactManagerMapping.size() == M.npMain.size();
    for (PxU32 i = 0; same && i < M.npMain.size(); ++i) same = int32_t(L.mContactManagerMapping[i]->getIndex()) == M.npMain.cms[i];
    if (!same) fail("스텝 끝 좁은 단계 목록");
  }
  G.clearStep();
}

// 장면 파일 뜨기(g1_dump.cpp)가 부른다: 지금까지(경계 앞) 모은 쌍 관리층 입력 기록
const eng::scene::PairsLog* g1_pairs_log() {
  if (!PS.on || !PS.log.valid) return nullptr;
  PxSimulationFilterCallback* cb = PS.sc ? PS.sc->getFilterCallbackFast() : nullptr;
  PS.log.reportAll = cb ? (static_cast<engine::OmniFilterCallback*>(cb)->report_all ? 1u : 0u) : 0u;
  return &PS.log;
}

void g1_pairs_report() {
  if (!PS.on) return;
  printf("G1 쌍 관리층 그림자 (omni 거르개, PhysX Sc 층과 같은 입력): 스텝 %" PRIu64 " — 새 겹침 %" PRIu64 ", 사라진 겹침 %" PRIu64 ", 닿음 시작 %" PRIu64
         ", 활성화 %" PRIu64 "/%" PRIu64 ", 거르기 자료 바꿈 %" PRIu64 ", 운동학 전환 %" PRIu64 ", 조인트 생김 %" PRIu64 " 없어짐 %" PRIu64 ", 모양 생김 %" PRIu64 " 없어짐 %" PRIu64 "\n",
         PS.steps, PS.nCreated, PS.nRemoved, PS.nTouch, PS.nAct, PS.nDeact, PS.nRefilter, PS.nKinToggle, PS.nJointAdd, PS.nJointRemove, PS.nShapeAdd, PS.nShapeRemove);
  printf("  API 관리자 다시 등록 %" PRIu64 ", 잠든 몸체 모양 변경 %" PRIu64 "\n", PS.nApiReset, PS.nSleepChange);
  printf("  비교: 좁은 단계 칸 %" PRIu64 ", 닿음 사건 %" PRIu64 ", 섬 호출 %" PRIu64 ", 행위자 상호작용 %" PRIu64 ", 풀·목록 %" PRIu64 " — 다름 %" PRIu64 "%s\n", PS.cmpList,
         PS.cmpEvents, PS.cmpCalls, PS.cmpActor, PS.cmpPool, PS.bad,
         PS.bad ? ("  첫 다름 simulate " + std::to_string(PS.firstBad) + " " + PS.firstWhat).c_str() : "");
}
