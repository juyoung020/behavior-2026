// 층 1 시험 (contact, 쌍 관리층): core/contact/sc_pairs.h = PhysX 5.6.1 Sc 층의 접촉 쌍 처리?
// PhysX 장면을 돌리면서 링커 --wrap 으로 다음을 뜬다:
//  - 입력: 새 겹침(NPhaseCore::runOverlapFilters·onTriggerOverlapCreated 입구), 사라진 겹침(NPhaseCore::onOverlapRemoved 입구),
//          좁은 단계 결과(PxsContext::fillManagerTouchEvents 입구 때 합쳐진 목록의 칸별 상태·패치 수)
//  - 비교 대상: 섬 관리자 호출(SimpleIslandManager·IslandSim, 종류별 순서·인자), 좁은 단계 목록(접촉 관리자 번호·모양·작업 단위 값),
//          닿음 시작/끝 사건 순서, 스텝 끝 행위자 상호작용 목록·접촉 관리자 풀 빈 칸 목록
// 같은 입력을 우리 층에 PhysX 순서로 넣고 매 스텝 비교한다. 잠은 끈다(모든 몸체 깨어 있음, 운동학 몸체는 매 스텝 목표를 줌).
//   test_sc_pairs [--bodies N] [--steps S] [--seed X] [--threads T]
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <cstddef>
#include <random>
#include <set>
#include <vector>

#include "px_bridge.h"  // 볼록 굽기·우리 모양 변환 (cxt::)
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
#include "core/contact/np_step.h"

using namespace physx;
namespace ss = eng::contact::sc;
namespace ec = eng::contact;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

// ---------------------------------------------------------------------------------------------------------------------
// 뜬 자료 (--wrap)
struct IslandCall {
  int op;
  uint64_t a, b, c, d, e;  // 연산마다 뜻이 다름
  bool operator==(const IslandCall& o) const { return op == o.op && a == o.a && b == o.b && c == o.c && d == o.d && e == o.e; }
};
enum Op { OP_ADD_CM, OP_PREALLOC, OP_ADD_PREALLOC, OP_DELAYED, OP_CONNECT, OP_DISCONNECT, OP_REMOVE, OP_SET_RIGID_CM, OP_CLEAR_RIGID_CM, OP_DEACT_EDGE, OP_COUNT };
static const char* kOpName[OP_COUNT] = {"addContactManager", "preallocateContactManagers", "addPreallocatedContactManager", "addDelayedDirtyEdges",
                                        "setEdgeConnected", "setEdgeDisconnected", "removeConnection", "setEdgeRigidCM", "clearEdgeRigidCM",
                                        "deactivateEdge"};
struct Capture {
  bool on = false;
  std::vector<IslandCall> calls[OP_COUNT];
  std::vector<uint32_t> preallocHandles;  // preallocateContactManagers 가 돌려준 간선 번호 (재생 때 우리 층에 돌려줌)
  std::vector<uint32_t> addCmEdges;
  // 입력
  struct Chunk { const void* base; std::vector<int32_t> pairs; };
  std::vector<Chunk> createdShapeChunks;
  std::vector<int32_t> createdTrigger;
  std::vector<int32_t> removedPairs;  // onOverlapRemoved 순서 (모양·트리거 섞임 — 모양 목록 다음 트리거 목록)
  // 좁은 단계 결과 (합친 목록 칸별)
  std::vector<int32_t> npCms;
  std::vector<uint8_t> npStatus, npPatches;
  std::vector<uint32_t> npWuFlags, npRest, npNpIndex, npShape0, npShape1, npDom;
  std::vector<std::pair<int32_t, int32_t>> touchFound, touchLost;
  // --np: 좁은 단계 입력(변환 캐시·접촉 거리)과 PhysX 출력 스트림
  std::vector<PxTransform> tcPose;
  std::vector<uint32_t> tcFlags;
  std::vector<float> contactDist;
  std::vector<uint16_t> npNbContacts;
  std::vector<std::vector<uint8_t>> npPatchBytes, npContactBytes;
  // 상호작용 활성·비활성 (섬 관리가 부르는 것: Sc::activateInteraction / deactivateInteraction 경유)
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
    tcPose.clear(); tcFlags.clear(); contactDist.clear(); npNbContacts.clear(); npPatchBytes.clear(); npContactBytes.clear();
    touchFound.clear(); touchLost.clear();
  }
} G;
static std::mutex gCapMu;  // PhysX 작업 스레드들이 동시에 부르는 감싸개가 있다 (섬 넣기 작업 등)
static std::map<const PxsShapeCore*, int32_t> gCoreToElem;
static std::map<int32_t, Sc::ElementSim*> gElemSim;
static Sc::Scene* gSc = nullptr;

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static std::pair<int32_t, int32_t> siElems(const Sc::Interaction* it) {
  const Sc::ElementSimInteraction* e = static_cast<const Sc::ElementSimInteraction*>(it);
  return {int32_t(e->getElement0().getElementID()), int32_t(e->getElement1().getElementID())};
}
static uint64_t packPair(std::pair<int32_t, int32_t> p) { return (uint64_t(uint32_t(p.first)) << 32) | uint32_t(p.second); }

#define W(name) __wrap_##name
#define R(name) __real_##name
typedef IG::SimpleIslandManager SIM;
extern "C" {
PxU32 R(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM*, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
PxU32 W(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM* s, PxsContactManager* cm, PxNodeIndex n0, PxNodeIndex n1, Sc::Interaction* it, IG::Edge::EdgeType t) {
  const PxU32 e = R(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(s, cm, n0, n1, it, t);
  if (G.on) {
    G.calls[OP_ADD_CM].push_back(IslandCall{OP_ADD_CM, cm ? cm->getIndex() : 0xffffffffull, n0.getInd(), n1.getInd(), it ? packPair(siElems(it)) : 0, uint64_t(t)});
    G.addCmEdges.push_back(e);
  }
  return e;
}
void R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM*, PxU32, PxU32*);
void W(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM* s, PxU32 n, PxU32* h) {
  R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(s, n, h);
  if (G.on) {
    G.calls[OP_PREALLOC].push_back(IslandCall{OP_PREALLOC, n, 0, 0, 0, 0});
    for (PxU32 i = 0; i < n; ++i) G.preallocHandles.push_back(h[i]);
  }
}
bool R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM*, PxU32, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
bool W(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM* s, PxU32 e, PxsContactManager* cm, PxNodeIndex n0, PxNodeIndex n1, Sc::Interaction* it, IG::Edge::EdgeType t) {
  const bool r = R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(s, e, cm, n0, n1, it, t);
  if (G.on) {
    std::lock_guard<std::mutex> lk(gCapMu);
    G.calls[OP_ADD_PREALLOC].push_back(IslandCall{OP_ADD_PREALLOC, e, cm ? cm->getIndex() : 0xffffffffull, n0.getInd(), n1.getInd(), it ? packPair(siElems(it)) : 0});
  }
  return r;
}
void R(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim*, PxU32, const PxU32*);
void W(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim* s, PxU32 n, const PxU32* e) {
  if (G.on) {
    uint64_t h = n;
    for (PxU32 i = 0; i < n; ++i) h = h * 1000003ull + e[i];
    G.calls[OP_DELAYED].push_back(IslandCall{OP_DELAYED, n, h, 0, 0, 0});
  }
  R(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(s, n, e);
}
#define WRAP_EDGE(OP, NAME)                                    \
  void R(NAME)(SIM*, PxU32);                                   \
  void W(NAME)(SIM * s, PxU32 e) {                             \
    if (G.on) { std::lock_guard<std::mutex> lk(gCapMu); G.calls[OP].push_back(IslandCall{OP, e, 0, 0, 0, 0}); } \
    R(NAME)(s, e);                                             \
  }
WRAP_EDGE(OP_DISCONNECT, _ZN5physx2IG19SimpleIslandManager19setEdgeDisconnectedEj)
// removeConnection: 조인트 간선(ConstraintInteraction)은 joints 몫이라 기록하지 않는다
void R(_ZN5physx2IG19SimpleIslandManager16removeConnectionEj)(SIM*, PxU32);
void W(_ZN5physx2IG19SimpleIslandManager16removeConnectionEj)(SIM* s, PxU32 e) {
  if (G.on) {
    const Sc::Interaction* it = s->getInteractionFromEdgeIndex(e);
    if (!it || it->getType() != Sc::InteractionType::eCONSTRAINTSHADER) G.calls[OP_REMOVE].push_back(IslandCall{OP_REMOVE, e, 0, 0, 0, 0});
  }
  R(_ZN5physx2IG19SimpleIslandManager16removeConnectionEj)(s, e);
}
WRAP_EDGE(OP_CLEAR_RIGID_CM, _ZN5physx2IG19SimpleIslandManager16clearEdgeRigidCMEj)
WRAP_EDGE(OP_DEACT_EDGE, _ZN5physx2IG19SimpleIslandManager14deactivateEdgeEj)
void R(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(SIM*, PxU32, IG::Edge::EdgeType);
void W(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(SIM* s, PxU32 e, IG::Edge::EdgeType t) {
  if (G.on) {
    std::lock_guard<std::mutex> lk(gCapMu);
    G.calls[OP_CONNECT].push_back(IslandCall{OP_CONNECT, e, uint64_t(t), 0, 0, 0});
  }
  R(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(s, e, t);
}
void R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM*, PxU32, PxsContactManager*);
void W(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM* s, PxU32 e, PxsContactManager* cm) {
  if (G.on) G.calls[OP_SET_RIGID_CM].push_back(IslandCall{OP_SET_RIGID_CM, e, cm ? cm->getIndex() : 0xffffffffull, 0, 0, 0});
  R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(s, e, cm);
}
// 입력: 새 겹침 (거르기 작업 입구 — 작업 순서는 스레드마다 달라도 덩어리 주소로 되맞춘다)
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
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
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
// 상호작용 활성·비활성 (ScScene.o 의 activateInteraction/deactivateInteraction 이 부른 것만 잡힌다)
bool R(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(Sc::ShapeInteraction*, PxsContactManager*);
bool W(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(Sc::ShapeInteraction* si, PxsContactManager* cm) {
  const bool r = R(_ZN5physx2Sc16ShapeInteraction10onActivateEPNS_17PxsContactManagerE)(si, cm);
  if (G.on) { const auto e = siElems(si); G.acts.push_back(Capture::Act{true, e.first, e.second, r, G.fillSeen}); }
  return r;
}
bool R(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(Sc::ShapeInteraction*);
bool W(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(Sc::ShapeInteraction* si) {
  const auto e = siElems(si);
  const bool r = R(_ZN5physx2Sc16ShapeInteraction12onDeactivateEv)(si);
  if (G.on) G.acts.push_back(Capture::Act{false, e.first, e.second, r, G.fillSeen});
  return r;
}
// 좁은 단계 결과 + 닿음 사건
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
      const PxsContactManagerOutput& o = L.mOutputContactManagers[i];
      G.npNbContacts.push_back(o.nbContacts);
      G.npPatchBytes.emplace_back(o.contactPatches, o.contactPatches + (o.contactPatches ? sizeof(PxContactPatch) * o.nbPatches : 0));
      G.npContactBytes.emplace_back(o.contactPoints, o.contactPoints + (o.contactPoints ? sizeof(PxContact) * o.nbContacts : 0));
    }
    PxsTransformCache& tcache = ctx->getTransformCache();
    for (PxU32 k = 0; k < tcache.getTotalSize(); ++k) {
      G.tcPose.push_back(tcache.getTransformCache(k).transform);
      G.tcFlags.push_back(tcache.getTransformCache(k).flags);
    }
    const PxReal* cd = ctx->getContactDistances();
    for (PxU32 k = 0; k < tcache.getTotalSize(); ++k) G.contactDist.push_back(cd[k]);
  }
  R(_ZN5physx10PxsContext22fillManagerTouchEventsEPNS_27PxvContactManagerTouchEventERjS2_S3_S2_S3_)(ctx, nt, nc, lt, lc, ct, cc);
  if (G.on) {
    G.fillSeen = true;
    for (PxU32 i = 0; i < nc; ++i) G.touchFound.push_back(siElems(reinterpret_cast<Sc::ShapeInteraction*>(nt[i].getCMTouchEventUserData())));
    for (PxU32 i = 0; i < lc; ++i) G.touchLost.push_back(siElems(reinterpret_cast<Sc::ShapeInteraction*>(lt[i].getCMTouchEventUserData())));
  }
}
}  // extern "C"

// ---------------------------------------------------------------------------------------------------------------------
// 거르개: omni.physx 와 같은 꼴 (PhysXScene.cpp:48) — 충돌 그룹 쌍 표(word2), 거른 쌍(word1), 보고 있으면 eCALLBACK
struct FilterTables {
  std::set<uint64_t> groupPairs, filteredPairs;
  bool anyReport = true;
};
static FilterTables gFT;
static uint64_t pk(uint32_t a, uint32_t b) { if (a > b) std::swap(a, b); return (uint64_t(a) << 32) | b; }
static uint32_t ourShader(uint32_t a0, const ss::FilterData& f0, uint32_t a1, const ss::FilterData& f1, uint32_t& pf, const void*) {
  pf = ss::PairFlag::eCONTACT_DEFAULT | ss::PairFlag::eDETECT_CCD_CONTACT;
  if ((a0 & ss::FilterObj::eTRIGGER) || (a1 & ss::FilterObj::eTRIGGER)) { pf = ss::PairFlag::eTRIGGER_DEFAULT; return 0; }
  if (f0.word2 && f1.word2 && gFT.groupPairs.count(pk(f0.word2, f1.word2))) return ss::FilterFlag::eKILL;
  if ((f0.word3 & 2u) || (f1.word3 & 2u)) pf |= ss::PairFlag::eMODIFY_CONTACTS;
  if ((f0.word3 & 4u) || (f1.word3 & 4u)) return ss::FilterFlag::eCALLBACK;
  if (f0.word1 && f1.word1 && gFT.filteredPairs.count(pk(f0.word1, f1.word1))) return ss::FilterFlag::eKILL;
  if (gFT.anyReport) return ss::FilterFlag::eCALLBACK;
  return 0;
}
static PxFilterFlags pxShader(PxFilterObjectAttributes a0, PxFilterData d0, PxFilterObjectAttributes a1, PxFilterData d1, PxPairFlags& pf, const void*, PxU32) {
  uint32_t f = 0;
  const ss::FilterData e0{d0.word0, d0.word1, d0.word2, d0.word3}, e1{d1.word0, d1.word1, d1.word2, d1.word3};
  const uint32_t r = ourShader(a0, e0, a1, e1, f, nullptr);
  pf = PxPairFlags(PxU16(f));
  return PxFilterFlags(PxU16(r));
}
// pairFound: 행위자 번호가 짝수인 쌍은 알림 플래그를 붙이고(보고 쌍), 운동학·정적끼리는 풀이 끔 (omni OmniFilterCallback 과 같은 꼴)
static uint32_t ourPairFound(uint64_t, uint32_t a0, const ss::FilterData& f0, int32_t, int32_t, uint32_t a1, const ss::FilterData& f1, int32_t, int32_t,
                             uint32_t& pf, const void*) {
  if ((f0.word3 & 4u) || (f1.word3 & 4u)) pf &= ~ss::PairFlag::eSOLVE_CONTACT;
  if ((f0.word0 & 1u) || (f1.word0 & 1u)) {
    pf |= ss::PairFlag::eNOTIFY_TOUCH_LOST | ss::PairFlag::eNOTIFY_TOUCH_FOUND | ss::PairFlag::eNOTIFY_TOUCH_PERSISTS | ss::PairFlag::eNOTIFY_CONTACT_POINTS;
    const bool k0 = (a0 & ss::FilterObj::eKINEMATIC) || (a0 & ss::FilterObj::eTYPE_MASK) == 0;
    const bool k1 = (a1 & ss::FilterObj::eKINEMATIC) || (a1 & ss::FilterObj::eTYPE_MASK) == 0;
    if (k0 && k1) { pf &= ~ss::PairFlag::eSOLVE_CONTACT; pf |= ss::PairFlag::eDETECT_DISCRETE_CONTACT; }
  }
  return 0;
}
struct PxCallback : public PxSimulationFilterCallback {
  PxFilterFlags pairFound(PxU64 id, PxFilterObjectAttributes a0, PxFilterData d0, const PxActor*, const PxShape*, PxFilterObjectAttributes a1, PxFilterData d1,
                          const PxActor*, const PxShape*, PxPairFlags& pf) override {
    uint32_t f = PxU16(pf);
    const ss::FilterData e0{d0.word0, d0.word1, d0.word2, d0.word3}, e1{d1.word0, d1.word1, d1.word2, d1.word3};
    const uint32_t r = ourPairFound(id, a0, e0, 0, 0, a1, e1, 0, 0, f, nullptr);
    pf = PxPairFlags(PxU16(f));
    return PxFilterFlags(PxU16(r));
  }
  void pairLost(PxU64, PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData, bool) override {}
  bool statusChange(PxU64&, PxPairFlags&, PxFilterFlags&) override { return false; }
};

// ---------------------------------------------------------------------------------------------------------------------
// 우리 층 섬 갈고리: PhysX 가 한 호출(종류별 순서)과 맞춰 보고, 돌려줄 값은 PhysX 가 받은 값을 준다
struct Hooks : public ss::IslandHooks {
  const Capture* cap = nullptr;
  size_t pos[OP_COUNT] = {};
  size_t preallocPos = 0, addCmPos = 0;
  uint64_t bad = 0, cmp = 0;
  int step = 0;
  std::vector<IslandCall> mine[OP_COUNT];
  ss::ScPairs* m = nullptr;
  void reset(const Capture* c, int s) {
    cap = c; step = s;
    for (auto& p : pos) p = 0;
    for (auto& v : mine) v.clear();
    preallocPos = addCmPos = 0;
  }
  uint64_t pairOf(int32_t it) const { return it < 0 ? 0 : packPair({m->inters[size_t(it)].elem0, m->inters[size_t(it)].elem1}); }
  void rec(const IslandCall& c) { mine[c.op].push_back(c); }
  uint32_t addContactManager(int32_t cm, uint64_t n0, uint64_t n1, int32_t it, int32_t t) override {
    rec(IslandCall{OP_ADD_CM, cm < 0 ? 0xffffffffull : uint64_t(cm), n0, n1, pairOf(it), uint64_t(t)});
    return addCmPos < cap->addCmEdges.size() ? cap->addCmEdges[addCmPos++] : 0xffffffffu;
  }
  void preallocateContactManagers(uint32_t n, uint32_t* h) override {
    rec(IslandCall{OP_PREALLOC, n, 0, 0, 0, 0});
    for (uint32_t i = 0; i < n; ++i) h[i] = preallocPos < cap->preallocHandles.size() ? cap->preallocHandles[preallocPos++] : 0xffffffffu;
  }
  bool addPreallocatedContactManager(uint32_t e, int32_t cm, uint64_t n0, uint64_t n1, int32_t it, int32_t) override {
    rec(IslandCall{OP_ADD_PREALLOC, e, cm < 0 ? 0xffffffffull : uint64_t(cm), n0, n1, pairOf(it)});
    return false;  // isDirty 는 섬 상태에 달림 -> 지연 간선 비교는 따로 (아래)
  }
  void addDelayedDirtyEdges(uint32_t, const uint32_t*) override {}
  void setEdgeConnected(uint32_t e, int32_t t) override { rec(IslandCall{OP_CONNECT, e, uint64_t(t), 0, 0, 0}); }
  void setEdgeDisconnected(uint32_t e) override { rec(IslandCall{OP_DISCONNECT, e, 0, 0, 0, 0}); }
  void removeConnection(uint32_t e) override { rec(IslandCall{OP_REMOVE, e, 0, 0, 0, 0}); }
  void setEdgeRigidCM(uint32_t e, int32_t cm) override { rec(IslandCall{OP_SET_RIGID_CM, e, uint64_t(uint32_t(cm)), 0, 0, 0}); }
  void clearEdgeRigidCM(uint32_t e) override { rec(IslandCall{OP_CLEAR_RIGID_CM, e, 0, 0, 0, 0}); }
  void deactivateEdge(uint32_t e) override { rec(IslandCall{OP_DEACT_EDGE, e, 0, 0, 0, 0}); }
  // 추측 섬 노드 활성: 새 겹침 만들 때 = 지난 스텝 끝 상태, 활성화 재생 때 = PhysX 가 낸 결과를 그대로 (forced)
  std::map<uint64_t, bool> nodeActive;
  int forced = -1;
  bool isSpeculativeNodeActive(uint64_t n) override {
    if (forced >= 0) return forced != 0;
    const auto it = nodeActive.find(n);
    return it != nodeActive.end() && it->second;
  }
  bool isSpeculativeNodeActiveOrActivating(uint64_t n) override { return isSpeculativeNodeActive(n); }
  int forcedDeact = -1;  // 비활성화 재생: PhysX 결과가 참이면 두 행위자 모두 비활성으로 답한다
  bool isActorActive(int32_t a) override { return forcedDeact >= 0 ? forcedDeact == 0 : !m->actors[size_t(a)].isStatic(); }
  void internalWakeUp(int32_t) override {}
  void addToLostTouchList(int32_t, int32_t) override {}
};

// --np: 좁은 단계 칸 자료 (다양체 + 출력 스트림). 쌍 관리층이 목록을 바꿀 때 같이 옮긴다.
struct SlotData {
  ec::ManifoldSlot man;
  ec::NpSlotOutput<32, 256> out;
};
struct Caches : public ss::CacheHooks {
  std::vector<SlotData> L[2];
  void create(bool nl, uint32_t slot, int32_t g0, int32_t g1) override {
    SlotData& d = L[nl][slot];
    d.out = ec::NpSlotOutput<32, 256>();
    ec::initManifold(d.man, g0 < g1 ? g0 : g1, g0 < g1 ? g1 : g0);
  }
  void move(bool dn, uint32_t dst, bool sn, uint32_t src) override {
    if (dn == sn && dst == src) return;
    ec::copyManifoldSlot(L[dn][dst].man, L[sn][src].man);
    L[dn][dst].out = L[sn][src].out;
  }
  void destroy(bool, uint32_t) override {}
  void resize(bool nl, uint32_t n) override { L[nl].resize(n); }
};

// ---------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  int nb = 150, steps = 300, seed = 1, threads = 2;
  bool sleep = false;     // --sleep: 잠 켬 (몸체가 잠들고, 가끔 깨우고, 운동학 일부는 멈춤)
  int removeEvery = 0;    // --remove N: N 스텝마다 동적 행위자 하나 빼고 다음 스텝에 하나 넣기
  int refilterEvery = 0;  // --refilter N: N 스텝마다 모양 거르기 자료 바꾸기 + 몸체 운동학 전환 (재거르기·convert 경로)
  int jointEvery = 0;     // --joints N: N 스텝마다 조인트 하나 만들거나(충돌 끔/켬) 없애기 (행위자 목록 자리표 재생)
  bool ourNp = false;     // --np: 좁은 단계도 우리 것(np_step.h)으로 돌려 PhysX 출력(상태·패치·접촉 스트림)과 비교
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--sleep")) sleep = true;
    else if (!strcmp(argv[i], "--remove") && i + 1 < argc) removeEvery = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--refilter") && i + 1 < argc) refilterEvery = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--joints") && i + 1 < argc) jointEvery = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--np")) ourNp = true;
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxInitExtensions(*phys, nullptr);
  static PxCallback pxCb;
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(PxU32(threads));
  sd.filterShader = pxShader;
  sd.filterCallback = &pxCb;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM;
  PxScene* scene = phys->createScene(sd);
  gSc = &static_cast<NpScene*>(scene)->getScScene();

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.4f, 0.1f);
  std::vector<PxConvexMesh*> hulls;  // 볼록 (BEHAVIOR 물체는 거의 볼록 조각)
  for (int h = 0; h < 24; ++h) {
    PxConvexMesh* m = cxt::cookHull(phys, cxt::randomCloud(rng, h % 4, (h % 3 == 0) ? 40 : 12), false, 64);
    if (m) hulls.push_back(m);
  }
  // 충돌 그룹 (word2 1~4) 과 죽이는 그룹 쌍, 거른 쌍 번호(word1)
  gFT.groupPairs.insert(pk(1, 2));
  gFT.groupPairs.insert(pk(3, 3));
  for (int i = 0; i < 20; ++i) gFT.filteredPairs.insert(pk(1 + uint32_t(P(rng) * 30), 1 + uint32_t(P(rng) * 30)));
  auto randFD = [&]() {
    PxFilterData d;
    d.word0 = P(rng) < 0.5f ? 1u : 0u;  // 보고 쌍 표시
    d.word1 = P(rng) < 0.5f ? 1 + uint32_t(P(rng) * 30) : 0;
    d.word2 = P(rng) < 0.6f ? 1 + uint32_t(P(rng) * 4) : 0;
    d.word3 = P(rng) < 0.05f ? 4u : (P(rng) < 0.05f && !ourNp ? 2u : 0u);  // 2 = 접촉 수정(omni 표면 속도): --np 에서는 뺌(수정 가능 스트림은 아직)
    return d;
  };
  auto addShape = [&](PxRigidActor* a, const PxGeometry& g, const PxTransform& local, bool trigger) {
    PxShape* s = PxRigidActorExt::createExclusiveShape(*a, g, *mat);
    s->setLocalPose(local);
    s->setContactOffset(0.01f + 0.03f * P(rng));
    s->setRestOffset(P(rng) < 0.3f ? 0.002f * P(rng) : 0.0f);
    if (trigger) { s->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false); s->setFlag(PxShapeFlag::eTRIGGER_SHAPE, true); }
    s->setSimulationFilterData(randFD());
    return s;
  };
  auto randGeom = [&](PxGeometryHolder& h) {
    const float r = P(rng);
    if (r < 0.35f) {  // 볼록 (크기 늘임 있음/없음)
      PxConvexMesh* m = hulls[size_t(P(rng) * float(hulls.size())) % hulls.size()];
      h.storeAny(PxConvexMeshGeometry(m, P(rng) < 0.5f ? PxMeshScale(1.0f) : PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)))));
      return;
    }
    if (r < 0.55f) h.storeAny(PxSphereGeometry(0.05f + 0.2f * P(rng)));
    else if (r < 0.8f) h.storeAny(PxBoxGeometry(0.05f + 0.2f * P(rng), 0.05f + 0.2f * P(rng), 0.05f + 0.2f * P(rng)));
    else h.storeAny(PxCapsuleGeometry(0.04f + 0.1f * P(rng), 0.05f + 0.2f * P(rng)));
  };
  std::vector<PxRigidActor*> actorsPx;
  std::vector<PxRigidDynamic*> kinematics;
  // 바닥·정적
  {
    PxRigidStatic* g = phys->createRigidStatic(PxTransform(PxVec3(0, 0, -0.5f)));
    addShape(g, PxBoxGeometry(5.0f, 5.0f, 0.5f), PxTransform(PxIdentity), false);
    scene->addActor(*g);
    actorsPx.push_back(g);
    for (int i = 0; i < 6; ++i) {
      PxRigidStatic* s = phys->createRigidStatic(PxTransform(PxVec3(3.5f * U(rng), 3.5f * U(rng), 0.2f)));
      PxGeometryHolder h; randGeom(h);
      addShape(s, h.any(), PxTransform(PxIdentity), i == 0);  // 하나는 트리거
      scene->addActor(*s);
      actorsPx.push_back(s);
    }
  }
  // 동적 몸체 (일부는 운동학, 일부는 조인트로 이음)
  std::vector<PxRigidDynamic*> dyns;
  for (int i = 0; i < nb; ++i) {
    PxQuat q(U(rng), U(rng), U(rng), U(rng)); q.normalize();
    PxRigidDynamic* d = phys->createRigidDynamic(PxTransform(PxVec3(3.0f * U(rng), 3.0f * U(rng), 0.3f + 0.4f * float(i % 10) + P(rng)), q));
    const int ns = (i % 6 == 0) ? 2 : 1;
    for (int k = 0; k < ns; ++k) {
      PxGeometryHolder h; randGeom(h);
      addShape(d, h.any(), PxTransform(PxVec3(0.1f * U(rng), 0.1f * U(rng), 0.1f * U(rng))), false);
    }
    if (i % 17 == 3) d->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true), kinematics.push_back(d);
    else PxRigidBodyExt::updateMassAndInertia(*d, 300.0f);
    if (!sleep) d->setSleepThreshold(0.0f);
    scene->addActor(*d);
    actorsPx.push_back(d);
    dyns.push_back(d);
  }
  // 조인트 (충돌 끔 기본)
  std::vector<std::pair<PxRigidActor*, PxRigidActor*>> jointed;
  std::vector<PxJoint*> liveJoints;
  std::map<PxJoint*, int32_t> jointExt;  // 조인트 -> 우리 층 자리표 번호
  for (int j = 0; j + 1 < int(dyns.size()); j += 11) {
    PxRigidDynamic* a = dyns[size_t(j)];
    PxRigidDynamic* b = dyns[size_t(j + 1)];
    if (a->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC || b->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC) continue;
    PxD6Joint* jt = PxD6JointCreate(*phys, a, PxTransform(PxIdentity), b, PxTransform(PxVec3(0.3f, 0, 0)));
    jt->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    if (j % 3 == 0) jt->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);
    jointed.push_back({a, b});
    liveJoints.push_back(jt);
  }
  // 관절체 2 개 (하나는 뿌리 고정), 묶음(자기 충돌 켬)
  std::vector<PxArticulationReducedCoordinate*> arts;
  for (int a = 0; a < 2; ++a) {
    PxArticulationReducedCoordinate* art = phys->createArticulationReducedCoordinate();
    if (a == 0) art->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
    if (!sleep) art->setSleepThreshold(0.0f);
    PxArticulationLink* parent = nullptr;
    const PxVec3 base(2.0f * U(rng), 2.0f * U(rng), 0.8f);
    for (int l = 0; l < 5; ++l) {
      PxArticulationLink* link = art->createLink(parent, PxTransform(base + PxVec3(0.0f, 0.0f, 0.25f * float(l))));
      PxGeometryHolder h; randGeom(h);
      addShape(link, h.any(), PxTransform(PxIdentity), false);
      PxRigidBodyExt::updateMassAndInertia(*link, 200.0f);
      if (parent) {
        PxArticulationJointReducedCoordinate* jt = link->getInboundJoint();
        jt->setJointType(PxArticulationJointType::eREVOLUTE);
        jt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
        jt->setParentPose(PxTransform(PxVec3(0, 0, 0.125f)));
        jt->setChildPose(PxTransform(PxVec3(0, 0, -0.125f)));
      }
      parent = link;
    }
    PxAggregate* agg = phys->createAggregate(8, 8, true);
    agg->addArticulation(*art);
    scene->addAggregate(*agg);
    arts.push_back(art);
  }
  printf("몸체 %d(운동학 %zu), 조인트 %zu, 관절체 %zu\n", nb, kinematics.size(), jointed.size(), arts.size());

  // ---- 우리 층 준비 (PhysX 쪽 Sc 자료에서 입력을 읽는다)
  ss::ScPairs M;
  Caches caches;
  std::vector<ec::NpShape> npShapes;
  std::vector<ec::MaterialData> ourMats(64);
  {
    ec::MaterialData& md = ourMats[static_cast<NpMaterial*>(mat)->mMaterial.mMaterialIndex];
    md.dynamicFriction = mat->getDynamicFriction();
    md.staticFriction = mat->getStaticFriction();
    md.restitution = mat->getRestitution();
    md.damping = mat->getDamping();
    md.flags = uint16_t(PxU16(mat->getFlags()));
    md.fricCombineMode = uint8_t(mat->getFrictionCombineMode());
    md.restCombineMode = uint8_t(mat->getRestitutionCombineMode());
    md.dampingCombineMode = uint8_t(mat->getDampingCombineMode());
  }
  ec::NpParams npParams;
  npParams.meshContactMargin = 0.01f * tol.length;
  npParams.toleranceLength = tol.length;
  Hooks hooks;
  hooks.m = &M;
  M.islands = &hooks;
  if (ourNp) M.caches = &caches;
  M.filterShader = ourShader;
  M.filterPairFound = ourPairFound;
  std::map<const Sc::ActorSim*, int32_t> actorIndex;
  std::map<const Sc::ArticulationSim*, int32_t> artIndex;
  auto actorOf = [&](Sc::ActorSim* as) -> int32_t {
    auto it = actorIndex.find(as);
    if (it != actorIndex.end()) return it->second;
    const int32_t idx = int32_t(M.actors.size());
    actorIndex[as] = idx;
    M.actors.push_back(ss::Actor());
    return idx;
  };
  auto refreshActor = [&](Sc::ActorSim* as) {
    ss::Actor& A = M.actors[size_t(actorOf(as))];
    A.type = int32_t(as->getActorType());
    A.filterAttr = as->getFilterAttributes();
    A.actorID = as->getActorID();
    A.nodeIndex = as->getNodeIndex().getInd();
    A.hasConstraints = as->readInternalFlag(Sc::ActorSim::BF_HAS_CONSTRAINTS);
    A.dominanceGroup = as->getActorCore().getDominanceGroup();
    if (as->isDynamicRigid()) {
      Sc::BodySim* bs = static_cast<Sc::BodySim*>(as);
      A.fixedBaseLink = bs->getLowLevelBody().mCore->fixedBaseLink != 0;
      A.offsetSlop = bs->getBodyCore().getCore().offsetSlop;
      A.forceStaticKineNotif = bs->getBodyCore().getCore().mFlags.isSet(PxRigidBodyFlag::eFORCE_STATIC_KINE_NOTIFICATIONS);
      A.forceKineKineNotif = bs->getBodyCore().getCore().mFlags.isSet(PxRigidBodyFlag::eFORCE_KINE_KINE_NOTIFICATIONS);
      if (bs->isArticulationLink()) {
        Sc::ArticulationSim* asim = bs->getArticulation();
        auto ai = artIndex.find(asim);
        if (ai == artIndex.end()) ai = artIndex.emplace(asim, int32_t(artIndex.size())).first;
        A.articulation = ai->second;
        A.linkId = bs->getNodeIndex().articulationLinkId();
        A.parentLinkId = asim->getLink(A.linkId).parent;
        A.artDisableSelfCollision = asim->getCore().getArticulationFlags().isSet(PxArticulationFlag::eDISABLE_SELF_COLLISION);
      }
    }
  };
  auto syncScene = [&]() {  // 모양·행위자 입력을 PhysX 에서 (행위자는 처음 볼 때 등록)
    for (PxRigidActor* a : actorsPx) {
      const PxU32 n = a->getNbShapes();
      std::vector<PxShape*> sh(n);
      a->getShapes(sh.data(), n);
      for (PxShape* s : sh) {
        Sc::ShapeCore& core = static_cast<NpShape*>(s)->getCore();
        Sc::ShapeSim* sim = core.getExclusiveSim();
        if (!sim) continue;
        const int32_t e = int32_t(sim->getElementID());
        if (size_t(e) >= M.shapes.size()) M.shapes.resize(size_t(e) + 1);
        ss::Shape& S = M.shapes[size_t(e)];
        Sc::ActorSim* as = &sim->getActor();
        refreshActor(as);
        S.valid = true;
        S.actor = actorOf(as);
        S.geomType = int32_t(sim->getGeometryType());
        S.trigger = (sim->getFlags() & PxShapeFlag::eTRIGGER_SHAPE) != 0;
        const PxFilterData fd = core.getSimulationFilterData();
        S.fd = ss::FilterData{fd.word0, fd.word1, fd.word2, fd.word3};
        S.restOffset = sim->getRestOffset();
        S.torsionalPatchRadius = sim->getTorsionalPatchRadius();
        S.minTorsionalPatchRadius = sim->getMinTorsionalPatchRadius();
        S.transformCacheId = sim->getTransformCacheID();
        if (size_t(e) >= npShapes.size()) npShapes.resize(size_t(e) + 1);
        {
          ec::NpShape& ns = npShapes[size_t(e)];
          ns.geom = ec::ShapeGeom();
          const PxGeometry& g = s->getGeometry();
          ns.geom.type = int32_t(g.getType());
          if (g.getType() == PxGeometryType::eSPHERE) ns.geom.sphere = eng::px::PxSphereGeometry(static_cast<const PxSphereGeometry&>(g).radius);
          else if (g.getType() == PxGeometryType::eBOX) {
            const PxVec3 h = static_cast<const PxBoxGeometry&>(g).halfExtents;
            ns.geom.box = eng::px::PxBoxGeometry(h.x, h.y, h.z);
          } else if (g.getType() == PxGeometryType::eCONVEXMESH) {
            ns.geom.convex = cxt::toE(static_cast<const PxConvexMeshGeometry&>(g));
          } else if (g.getType() == PxGeometryType::eCAPSULE) {
            const PxCapsuleGeometry& c = static_cast<const PxCapsuleGeometry&>(g);
            ns.geom.capsule = eng::px::PxCapsuleGeometry(c.radius, c.halfHeight);
          }
          ns.material = core.getCore().mMaterialIndex;
        }
        gCoreToElem[&core.getCore()] = e;
        gElemSim[e] = sim;
      }
    }
    M.jointPairs.clear();  // 조인트가 없어지면 거르기에서도 빠진다 (Scene::findConstraintCore)
    for (auto& jp : jointed) {
      Sc::ActorSim* s0 = static_cast<NpRigidDynamic*>(jp.first)->getCore().getSim();
      Sc::ActorSim* s1 = static_cast<NpRigidDynamic*>(jp.second)->getCore().getSim();
      const int32_t a0 = actorOf(s0), a1 = actorOf(s1);
      Sc::ConstraintCore* cc = gSc->findConstraintCore(s0, s1);
      M.jointPairs[(uint64_t(uint32_t(std::min(a0, a1))) << 32) | uint32_t(std::max(a0, a1))] =
          ss::JointPairInfo{cc ? cc->getFlags().isSet(PxConstraintFlag::eCOLLISION_ENABLED) : true};
    }
  };
  for (PxArticulationReducedCoordinate* art : arts) {
    std::vector<PxArticulationLink*> links(art->getNbLinks());
    art->getLinks(links.data(), PxU32(links.size()));
    for (PxArticulationLink* l : links) actorsPx.push_back(l);
  }

  PxsContext* ctx = gSc->getLowLevelContext();
  PxsNphaseImplementationContext* npc = static_cast<PxsNphaseImplementationContext*>(ctx->getNphaseImplementationContext());
  M.cmPool.eltsPerSlab = ctx->mContactManagerPool.mEltsPerSlab;
  {
    const auto& pool = ctx->mContactManagerPool;
    printf("처음 풀: 판 %u 개, 빈 칸 %u 개", pool.mSlabCount, pool.mFreeCount);
    if (pool.mFreeCount) printf(" (첫 %u, 끝 %u)", pool.mFreeList[0]->getIndex(), pool.mFreeList[pool.mFreeCount - 1]->getIndex());
    printf("\n");
  }
  M.kineKineFilteringMode = int32_t(gSc->getKineKineFilteringMode());
  M.staticKineFilteringMode = int32_t(gSc->getStaticKineFilteringMode());

  // 조인트·관절체 관절 상호작용 (행위자 목록에 섞여 있다) -> 우리 층에 자리표로 같은 순서로
  syncScene();
  std::map<const Sc::Interaction*, int32_t> extIndex;
  for (auto& kv : actorIndex) {
    const Sc::ActorSim* as = kv.first;
    for (PxU32 i = 0; i < as->getActorInteractionCount(); ++i) {
      const Sc::Interaction* it = as->getActorInteractions()[i];
      if (it->getType() <= Sc::InteractionType::eMARKER) continue;
      auto e = extIndex.find(it);
      if (e == extIndex.end()) {
        const int32_t a0 = actorIndex.count(&it->getActorSim0()) ? actorIndex[&it->getActorSim0()] : -1;
        const int32_t a1 = actorIndex.count(&it->getActorSim1()) ? actorIndex[&it->getActorSim1()] : -1;
        e = extIndex.emplace(it, M.newInteractionRecord(a0, a1, uint8_t(it->getType()))).first;
      }
      M.appendToActorList(kv.second, e->second);
    }
  }
  for (PxJoint* j : liveJoints) {  // 처음 조인트 -> 자리표 번호 (나중에 없앨 때)
    const Sc::ConstraintSim* cs = static_cast<NpConstraint*>(j->getConstraint())->getCore().getSim();
    const Sc::Interaction* ci = cs ? reinterpret_cast<const Sc::Interaction*>(cs->getInteraction()) : nullptr;
    if (ci && extIndex.count(ci)) jointExt[j] = extIndex[ci];
  }
  printf("조인트·관절 상호작용 자리표 %zu 개\n", extIndex.size());
  uint64_t bad = 0, cmpList = 0, cmpEvents = 0, cmpCalls = 0, cmpActor = 0, cmpNp = 0, npPoints = 0;
  int firstBad = -1;
  auto fail = [&](int s, const char* what) {
    if (!bad) { firstBad = s; printf("  [스텝 %d] 첫 다름: %s\n", s, what); }
    ++bad;
  };
  const float dt = 1.0f / 60.0f;
  uint64_t nCreated = 0, nRemoved = 0, nTouch = 0, nTrig = 0, nAct = 0, nDeact = 0, nRemovedActors = 0, nRefilterApi = 0, nKinToggle = 0, nJointOps = 0;
  for (int step = 0; step < steps; ++step) {
    for (size_t k = 0; k < kinematics.size(); ++k) {  // 운동학 몸체 옮기기 (잠 켬이면 절반은 150 스텝 뒤 멈춤)
      if (sleep && (k & 1) && step > 150) continue;
      PxTransform t = kinematics[k]->getGlobalPose();
      t.p += PxVec3(0.02f * std::sin(0.05f * float(step) + float(k)), 0.02f * std::cos(0.07f * float(step) + float(k)), 0.0f);
      kinematics[k]->setKinematicTarget(t);
    }
    if (sleep && step % 37 == 20 && !dyns.empty()) {  // 가끔 깨우기·밀기
      PxRigidDynamic* d = dyns[size_t(P(rng) * float(dyns.size())) % dyns.size()];
      if (!(d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)) {
        if (P(rng) < 0.5f) d->wakeUp();
        else d->addForce(PxVec3(0, 0, 200.0f));
      }
    }
    syncScene();
    G.clearStep();
    // API: 거르기 자료 바꾸기·운동학 전환 (우리 층에는 같은 순서로 더러움 표시를 넣는다)
    struct ApiOp { int kind; int32_t elem; Sc::ActorSim* actor; bool toKinematic; std::vector<int32_t> oldElems; };
    std::vector<ApiOp> apiOps;
    if (refilterEvery > 0 && step % refilterEvery == 1) {
      G.on = true;  // API 때 PhysX 가 하는 섬 호출(모양 다시 넣기 -> removeConnection 등)도 이 스텝 기록에
      PxRigidActor* a = actorsPx[size_t(P(rng) * float(actorsPx.size())) % actorsPx.size()];
      std::vector<PxShape*> sh(a->getNbShapes());
      a->getShapes(sh.data(), PxU32(sh.size()));
      PxShape* s0 = sh[0];
      if (!(s0->getFlags() & PxShapeFlag::eTRIGGER_SHAPE)) {
        s0->setSimulationFilterData(randFD());
        apiOps.push_back(ApiOp{0, int32_t(static_cast<NpShape*>(s0)->getCore().getExclusiveSim()->getElementID()), nullptr, false, {}});
        nRefilterApi++;
      }
      for (int tries = 0; tries < 10; ++tries) {
        PxRigidDynamic* d = dyns[size_t(P(rng) * float(dyns.size())) % dyns.size()];
        bool joined = false;
        for (auto& jp : jointed) joined |= jp.first == d || jp.second == d;
        if (joined) continue;
        const bool toKin = !(d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC);
        if (!toKin && std::find(kinematics.begin(), kinematics.end(), d) != kinematics.end()) continue;  // 처음부터 운동학인 것은 그대로
        std::vector<int32_t> oldElems;  // 운동학 전환은 모양을 넓은 단계에 다시 넣는다(새 요소 번호, ScShapeSimBase.cpp:336 updateBPGroup)
        {
          std::vector<PxShape*> dsh(d->getNbShapes());
          d->getShapes(dsh.data(), PxU32(dsh.size()));
          for (PxShape* x : dsh) oldElems.push_back(int32_t(static_cast<NpShape*>(x)->getCore().getExclusiveSim()->getElementID()));
        }
        d->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, toKin);
        if (!toKin) PxRigidBodyExt::updateMassAndInertia(*d, 300.0f);
        apiOps.push_back(ApiOp{1, -1, static_cast<NpRigidDynamic*>(d)->getCore().getSim(), toKin, oldElems});
        nKinToggle++;
        break;
      }
      G.on = false;
    }
    // API: 조인트 만들기·없애기 (Sc::ConstraintInteraction: registerInActors(행위자 0 다음 1) / destroy)
    struct JointOp { bool add; Sc::ActorSim* a0; Sc::ActorSim* a1; PxJoint* j; const Sc::Interaction* ci; };
    std::vector<JointOp> jointOps;
    if (jointEvery > 0 && step % jointEvery == 2) {
      G.on = true;
      if (!liveJoints.empty() && P(rng) < 0.5f) {
        const size_t ji = size_t(P(rng) * float(liveJoints.size())) % liveJoints.size();
        PxJoint* j = liveJoints[ji];
        PxRigidActor *x0, *x1;
        j->getActors(x0, x1);
        jointOps.push_back(JointOp{false, nullptr, nullptr, j, nullptr});
        j->release();
        liveJoints.erase(liveJoints.begin() + long(ji));
        for (size_t q = 0; q < jointed.size(); ++q)
          if (jointed[q].first == x0 && jointed[q].second == x1) { jointed.erase(jointed.begin() + long(q)); break; }
        nJointOps++;
      } else {
        for (int tries = 0; tries < 20; ++tries) {
          PxRigidDynamic* a = dyns[size_t(P(rng) * float(dyns.size())) % dyns.size()];
          PxRigidDynamic* b = dyns[size_t(P(rng) * float(dyns.size())) % dyns.size()];
          if (a == b || (a->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC) || (b->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)) continue;
          bool dup = false;
          for (auto& jp : jointed) dup |= (jp.first == a && jp.second == b) || (jp.first == b && jp.second == a);
          if (dup) continue;
          PxD6Joint* jt = PxD6JointCreate(*phys, a, PxTransform(PxIdentity), b, PxTransform(PxVec3(0.3f, 0, 0)));
          jt->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
          if (P(rng) < 0.3f) jt->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);
          jointed.push_back({a, b});
          liveJoints.push_back(jt);
          {
            const Sc::ConstraintSim* cs = static_cast<NpConstraint*>(jt->getConstraint())->getCore().getSim();
            jointOps.push_back(JointOp{true, static_cast<NpRigidDynamic*>(a)->getCore().getSim(), static_cast<NpRigidDynamic*>(b)->getCore().getSim(), jt,
                                       cs ? reinterpret_cast<const Sc::Interaction*>(cs->getInteraction()) : nullptr});
          }
          nJointOps++;
          break;
        }
      }
      G.on = false;
      syncScene();
    }
    // 가끔 행위자 빼기(API) — 그때 PhysX 가 하는 섬 호출도 이 스텝 기록에 넣고, 우리 층도 스텝 재생 앞에서 같은 모양 순서로 뺀다
    std::vector<int32_t> removedElems;
    if (removeEvery > 0 && step % removeEvery == removeEvery / 2 && apiOps.empty() && jointOps.empty()) {  // 같은 스텝의 API 조작과 겹치지 않게
      for (int tries = 0; tries < 20; ++tries) {
        const size_t di = size_t(P(rng) * float(dyns.size())) % dyns.size();
        PxRigidDynamic* d = dyns[di];
        if (d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC) continue;
        bool joined = false;
        for (auto& jp : jointed) joined |= jp.first == d || jp.second == d;
        if (joined) continue;
        std::vector<PxShape*> sh(d->getNbShapes());
        d->getShapes(sh.data(), PxU32(sh.size()));
        for (PxShape* s : sh) removedElems.push_back(int32_t(static_cast<NpShape*>(s)->getCore().getExclusiveSim()->getElementID()));
        G.on = true;
        scene->removeActor(*d);
        G.on = false;
        actorsPx.erase(std::find(actorsPx.begin(), actorsPx.end(), static_cast<PxRigidActor*>(d)));
        dyns.erase(dyns.begin() + long(di));
        nRemovedActors++;
        break;
      }
    }
    if (removeEvery > 0 && step % removeEvery == removeEvery / 2 + 1) {  // 다음 스텝에 새 몸체 하나 넣기 (번호 재사용이 뺀 스텝과 겹치지 않게)
      PxRigidDynamic* n = phys->createRigidDynamic(PxTransform(PxVec3(3.0f * U(rng), 3.0f * U(rng), 2.0f + P(rng))));
      PxGeometryHolder h; randGeom(h);
      addShape(n, h.any(), PxTransform(PxIdentity), false);
      PxRigidBodyExt::updateMassAndInertia(*n, 300.0f);
      if (!sleep) n->setSleepThreshold(0.0f);
      scene->addActor(*n);
      actorsPx.push_back(n);
      dyns.push_back(n);
      syncScene();
    }
    if (!apiOps.empty()) syncScene();  // API 로 바뀐 거르기 자료·운동학 플래그를 우리 입력에 (재거르기가 새 값으로 돈다)
    // 새 겹침 때 쓸 노드 활성 상태 = simulate 직전 추측 섬 (API 깨움·새 행위자 반영 뒤)
    {
      const IG::IslandSim& is = gSc->getSimpleIslandManager()->getSpeculativeIslandSim();
      hooks.nodeActive.clear();
      for (auto& kv : actorIndex) {
        const PxNodeIndex n = kv.first->getNodeIndex();
        if (n.isValid()) hooks.nodeActive[n.getInd()] = is.getNode(n).isActive();
      }
    }
    G.on = true;
    scene->simulate(dt);
    scene->fetchResults(true);
    G.on = false;

    if (getenv("SC_DEBUG") && step < 2) {
      printf("  [디버그 %d] PhysX 섬 호출:", step);
      for (int op = 0; op < OP_COUNT; ++op) printf(" %s=%zu", kOpName[op], G.calls[op].size());
      printf("\n  풀: 판 %u 빈 칸 %u, 새 겹침 덩어리 %zu\n", ctx->mContactManagerPool.mSlabCount, ctx->mContactManagerPool.mFreeCount, G.createdShapeChunks.size());
    }
    // ---- 우리 층에 같은 입력을 PhysX 순서로
    hooks.reset(&G, step);
    // 새 겹침: 거르기 작업 덩어리를 주소 순서로 = AABB 관리자 겹침 목록 순서
    std::sort(G.createdShapeChunks.begin(), G.createdShapeChunks.end(), [](const Capture::Chunk& a, const Capture::Chunk& b) { return a.base < b.base; });
    std::vector<int32_t> created;
    for (auto& c : G.createdShapeChunks) created.insert(created.end(), c.pairs.begin(), c.pairs.end());
    nCreated += created.size() / 2;
    nTrig += G.createdTrigger.size() / 2;
    for (const ApiOp& op : apiOps) {
      if (op.kind == 0) M.setElementInteractionsDirty(op.elem, ss::DirtyFlag::eFILTER_STATE, ss::IFlag::eFILTERABLE);
      else {
        M.setActorsInteractionsDirty(actorOf(op.actor), ss::DirtyFlag::eBODY_KINEMATIC, -1,
                                     op.toKinematic ? ss::IFlag::eFILTERABLE : uint8_t(ss::IFlag::eFILTERABLE | ss::IFlag::eCONSTRAINT));
        for (int32_t e : op.oldElems) M.onVolumeRemoved(e, true);  // updateBPGroup -> reinsertBroadPhase
      }
    }
    // (PhysX 순서: 거르기 자료·운동학 전환 -> 조인트 -> 행위자 빼기)
    for (const JointOp& op : jointOps) {
      if (op.add) {
        const int32_t id = M.addExternalInteraction(actorOf(op.a0), actorOf(op.a1), ss::eCONSTRAINTSHADER);
        jointExt[op.j] = id;
        if (op.ci) extIndex[op.ci] = id;
      } else {
        auto je = jointExt.find(op.j);
        if (je != jointExt.end()) {
          for (auto it = extIndex.begin(); it != extIndex.end(); ++it)
            if (it->second == je->second) { extIndex.erase(it); break; }
          M.removeExternalInteraction(je->second);
          jointExt.erase(je);
        }
      }
    }
    for (int32_t e : removedElems) M.onVolumeRemoved(e, true);
    M.updateDirtyInteractions();
    M.finishBroadPhase(G.createdTrigger.data(), uint32_t(G.createdTrigger.size() / 2), created.data(), uint32_t(created.size() / 2));
    // 좁은 단계 결과: 합친 목록 칸별로 (PhysX 의 칸 순서와 우리 순서가 같아야 한다)
    auto replayActs = [&](bool afterFill) {
      for (const auto& a : G.acts) {
        if (a.afterFill != afterFill) continue;
        const int32_t it = M.findInteraction(a.e0, a.e1);
        if (it < 0) { fail(step, "활성화 재생: 상호작용 없음"); continue; }
        bool r;
        if (a.activate) {
          hooks.forced = a.result ? 1 : 0;
          r = M.activateInteraction(it);
        } else {
          hooks.forcedDeact = a.result ? 1 : 0;
          r = M.deactivateInteraction(it);
        }
        hooks.forced = -1;
        hooks.forcedDeact = -1;
        if (r) (a.activate ? nAct : nDeact)++;
        if (r != a.result) fail(step, "활성화 결과");
      }
    };
    replayActs(false);
    M.beginNarrowPhase();
    if (ourNp) {  // 좁은 단계: 기존 목록(1차) 다음 새 목록(2차), PhysX 와 같은 입력(변환 캐시·접촉 거리)
      std::vector<ec::CachedTransform> tc(G.tcPose.size());
      for (size_t k = 0; k < tc.size(); ++k) {
        const PxTransform& t = G.tcPose[k];
        tc[k].transform = eng::px::PxTransform32(eng::px::PxTransform(eng::px::PxVec3(t.p.x, t.p.y, t.p.z), eng::px::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)));
        tc[k].flags = G.tcFlags[k];
      }
      static eng::px::PxContactBuffer nbuf;
      const unsigned oldCsr = _mm_getcsr();
      _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));  // PX_SIMD_GUARD (FTZ+DAZ)
      for (int nl = 0; nl < 2; ++nl) {
        ss::NpList& L = nl ? M.npNew : M.npMain;
        for (uint32_t slot = 0; slot < L.size(); ++slot) {
          SlotData& sd = caches.L[nl][slot];
          sd.out.statusFlag = L.outputs[slot].statusFlag;
          const ss::ContactManager& cm = M.cmsData[size_t(L.cms[slot])];
          const ec::NpWorkUnit wu{cm.wuFlags, cm.shape0, cm.shape1};
          ec::discreteNarrowPhasePCM(wu, npShapes.data(), tc.data(), G.contactDist.data(), ourMats.data(), npParams, sd.man, nbuf, sd.out);
          M.narrowPhaseResult(nl != 0, slot, sd.out.statusFlag, sd.out.nbPatches);
        }
      }
      _mm_setcsr(oldCsr & ~unsigned(_MM_EXCEPT_MASK));
    }
    M.mergeNarrowPhase();
    {
      bool same = M.npMain.size() == G.npCms.size();
      for (uint32_t i = 0; same && i < M.npMain.size(); ++i) {
        const ss::ContactManager& c = M.cmsData[size_t(M.npMain.cms[i])];
        ++cmpList;
        same = M.npMain.cms[i] == G.npCms[i] && uint32_t(c.shape0) == G.npShape0[i] && uint32_t(c.shape1) == G.npShape1[i] && c.wuFlags == G.npWuFlags[i] &&
               fbits(c.restDistance) == G.npRest[i] && c.npIndex == G.npNpIndex[i] && (uint32_t(c.dominance0) | (uint32_t(c.dominance1) << 8)) == G.npDom[i];
        if (!same) {
          if (!bad)
            printf("    칸 %u: PhysX cm %d (%u,%u) 플래그 %x rest %08x np %x / 우리 cm %d (%d,%d) 플래그 %x rest %08x np %x\n", i, G.npCms[i], G.npShape0[i],
                   G.npShape1[i], G.npWuFlags[i], G.npRest[i], G.npNpIndex[i], M.npMain.cms[i], c.shape0, c.shape1, c.wuFlags, fbits(c.restDistance),
                   c.npIndex);
        }
      }
      if (!same) {
        if (!bad) {
          printf("    목록 길이 PhysX %zu / 우리 %u\n      PhysX:", G.npCms.size(), M.npMain.size());
          for (size_t i = 0; i < G.npCms.size() && i < 40; ++i) printf(" %d(%u,%u)", G.npCms[i], G.npShape0[i], G.npShape1[i]);
          printf("\n      우리 :");
          for (uint32_t i = 0; i < M.npMain.size() && i < 40; ++i)
            printf(" %d(%d,%d)", M.npMain.cms[i], M.cmsData[size_t(M.npMain.cms[i])].shape0, M.cmsData[size_t(M.npMain.cms[i])].shape1);
          printf("\n");
          std::set<std::pair<int, int>> px, ours;
          for (size_t i = 0; i < G.npCms.size(); ++i) px.insert({int(G.npShape0[i]), int(G.npShape1[i])});
          for (uint32_t i = 0; i < M.npMain.size(); ++i) ours.insert({M.cmsData[size_t(M.npMain.cms[i])].shape0, M.cmsData[size_t(M.npMain.cms[i])].shape1});
          for (auto& p : ours)
            if (!px.count(p)) {
              const ss::Actor& a0 = M.actors[size_t(M.shapes[size_t(p.first)].actor)];
              const ss::Actor& a1 = M.actors[size_t(M.shapes[size_t(p.second)].actor)];
              printf("      우리만: (%d,%d) 행위자 종류 %d/%d 운동학 %d/%d 노드활성 %d/%d\n", p.first, p.second, a0.type, a1.type, a0.isKinematic(), a1.isKinematic(),
                     int(hooks.isSpeculativeNodeActive(a0.nodeIndex)), int(hooks.isSpeculativeNodeActive(a1.nodeIndex)));
              const Sc::ElementSimInteraction* pi = gSc->getNPhaseCore()->findInteraction(gElemSim[p.first], gElemSim[p.second]);
              const int32_t oi = M.findInteraction(p.first, p.second);
              if (pi) {
                const auto pe = siElems(pi);
                printf("        PhysX 상호작용: 종류 %d 순서 (%d,%d) 활성 %d 관리자 %d | 우리: 종류 %d 순서 (%d,%d) 활성 %d\n", int(pi->getType()), pe.first, pe.second,
                       int(pi->readInteractionFlag(Sc::InteractionFlag::eIS_ACTIVE) != 0),
                       pi->getType() == Sc::InteractionType::eOVERLAP ? int(static_cast<const Sc::ShapeInteraction*>(pi)->getContactManager() != nullptr) : -1,
                       oi >= 0 ? int(M.inters[size_t(oi)].type) : -1, oi >= 0 ? M.inters[size_t(oi)].elem0 : -1, oi >= 0 ? M.inters[size_t(oi)].elem1 : -1,
                       oi >= 0 ? int((M.inters[size_t(oi)].iflags & ss::IFlag::eIS_ACTIVE) != 0) : -1);
              } else {
                printf("        PhysX 상호작용 없음\n");
              }
            }
          for (auto& p : px)
            if (!ours.count(p)) printf("      PhysX만: (%d,%d)\n", p.first, p.second);
          for (const auto& a : G.acts) printf("      활성화 기록: %s (%d,%d) -> %d %s\n", a.activate ? "켬" : "끔", a.e0, a.e1, int(a.result), a.afterFill ? "(좁은 단계 뒤)" : "");
        }
        fail(step, "좁은 단계 목록");
      }
      if (same && !ourNp)
        for (uint32_t i = 0; i < M.npMain.size(); ++i) M.narrowPhaseResult(false, i, G.npStatus[i], G.npPatches[i]);
      if (same && ourNp) {  // 우리 좁은 단계 결과 = PhysX 출력?
        for (uint32_t i = 0; i < M.npMain.size(); ++i) {
          const SlotData& sd = caches.L[0][i];
          ++cmpNp;
          bool ok = sd.out.statusFlag == G.npStatus[i] && sd.out.nbPatches == G.npPatches[i] &&  // 상태 바이트 전체 (09-30: 닿음 비트만 보다가 startContacts 상태 0 누락을 놓침)
                    sd.out.nbContacts == G.npNbContacts[i];
          if (ok && sd.out.nbContacts) {
            ok = G.npPatchBytes[i].size() == sizeof(ec::ContactPatch) * sd.out.nbPatches && G.npContactBytes[i].size() == sizeof(ec::Contact) * sd.out.nbContacts &&
                 !memcmp(G.npContactBytes[i].data(), sd.out.stream.contacts, G.npContactBytes[i].size());
            for (uint32_t k = 0; ok && k < sd.out.nbPatches; ++k)  // 패치: 채움(pad) 10 바이트 빼고 (PhysX 는 초기화 안 함)
              ok = !memcmp(G.npPatchBytes[i].data() + k * sizeof(ec::ContactPatch), &sd.out.stream.patches[k], offsetof(ec::ContactPatch, pad));
            npPoints += sd.out.nbContacts;
          }
          if (!ok) {
            if (!bad) printf("    좁은 단계 칸 %u (%u,%u): 상태 PhysX %x 우리 %x, 패치 %u/%u, 점 %u/%u\n", i, G.npShape0[i], G.npShape1[i], G.npStatus[i], sd.out.statusFlag,
                             G.npPatches[i], sd.out.nbPatches, G.npNbContacts[i], sd.out.nbContacts);
            if (!bad && sd.out.nbContacts) {
              const ec::ContactPatch* pp = reinterpret_cast<const ec::ContactPatch*>(G.npPatchBytes[i].data());
              const ec::Contact* pc = reinterpret_cast<const ec::Contact*>(G.npContactBytes[i].data());
              for (uint32_t k = 0; k < sd.out.nbPatches && k < 2; ++k) {
                const ec::ContactPatch& a = pp[k];
                const ec::ContactPatch& b = sd.out.stream.patches[k];
                printf("      패치 %u: n (%.9g %.9g %.9g)/(%.9g %.9g %.9g) 마찰 %.9g/%.9g %.9g/%.9g 반발 %.9g/%.9g 감쇠 %.9g/%.9g 질량 %g %g %g %g/%g %g %g %g 시작 %u/%u 수 %u/%u 재질플래그 %u/%u 내부 %u/%u 재질 %u,%u/%u,%u\n", k,
                       a.normal.x, a.normal.y, a.normal.z, b.normal.x, b.normal.y, b.normal.z, a.staticFriction, b.staticFriction, a.dynamicFriction, b.dynamicFriction,
                       a.restitution, b.restitution, a.damping, b.damping, a.linear0, a.angular0, a.linear1, a.angular1, b.linear0, b.angular0, b.linear1, b.angular1,
                       a.startContactIndex, b.startContactIndex, a.nbContacts, b.nbContacts, a.materialFlags, b.materialFlags, a.internalFlags, b.internalFlags,
                       a.materialIndex0, a.materialIndex1, b.materialIndex0, b.materialIndex1);
              }
              for (uint32_t k = 0; k < sd.out.nbContacts && k < 4; ++k)
                printf("      점 %u: (%.9g %.9g %.9g) %.9g / (%.9g %.9g %.9g) %.9g\n", k, pc[k].contact.x, pc[k].contact.y, pc[k].contact.z, pc[k].separation,
                       sd.out.stream.contacts[k].contact.x, sd.out.stream.contacts[k].contact.y, sd.out.stream.contacts[k].contact.z, sd.out.stream.contacts[k].separation);
            }
            fail(step, "좁은 단계 출력");
          }
        }
      }
    }
    M.fillTouchEvents();
    {
      auto cmpEv = [&](const ss::Vec<ss::TouchEvent>& ours, const std::vector<std::pair<int32_t, int32_t>>& px, const char* name) {
        bool same = ours.size() == px.size();
        for (size_t i = 0; same && i < ours.size(); ++i) {
          ++cmpEvents;
          const ss::Interaction& I = M.inters[size_t(ours[i].inter)];
          same = I.elem0 == px[i].first && I.elem1 == px[i].second;
        }
        if (!same) {
          if (!bad) printf("    %s: PhysX %zu / 우리 %zu\n", name, px.size(), ours.size());
          fail(step, name);
        }
      };
      cmpEv(M.touchFound, G.touchFound, "닿음 시작 사건");
      cmpEv(M.touchLost, G.touchLost, "닿음 끝 사건");
      nTouch += G.touchFound.size();
    }
    M.processNewTouches();
    M.setEdgesConnected();
    // 사라진 겹침: onOverlapRemoved 순서 = 모양 목록 다음 트리거 목록
    std::vector<int32_t> lostS, lostT;
    for (size_t i = 0; i < G.removedPairs.size(); i += 2) {
      const bool trig = M.shapes[size_t(G.removedPairs[i])].trigger || M.shapes[size_t(G.removedPairs[i + 1])].trigger;
      (trig ? lostT : lostS).push_back(G.removedPairs[i]);
      (trig ? lostT : lostS).push_back(G.removedPairs[i + 1]);
    }
    nRemoved += G.removedPairs.size() / 2;
    M.processLostContacts(lostS.data(), uint32_t(lostS.size() / 2), lostT.data(), uint32_t(lostT.size() / 2));
    M.processNarrowPhaseLostTouchEventsIslands();
    M.processNarrowPhaseLostTouchEvents();
    M.processLostContacts2();
    M.lostTouchReports();
    M.unregisterInteractions();
    M.destroyManagers();
    M.processLostContacts3();
    replayActs(true);

    // ---- 섬 호출 비교 (종류별 순서)
    for (int op = 0; op < OP_COUNT; ++op) {
      if (op == OP_DELAYED) continue;  // isDirty 는 섬 상태 (solver) — 우리 갈고리는 늘 거짓
      std::vector<IslandCall> a = G.calls[op], b = hooks.mine[op];
      if (op == OP_ADD_PREALLOC) {  // 섬 넣기 작업(IslandInsertionTask)이 여럿이면 스레드마다 부르는 순서가 다르다 -> 간선 번호로 맞춘다
        auto byEdge = [](const IslandCall& x, const IslandCall& y) { return x.a < y.a; };
        std::sort(a.begin(), a.end(), byEdge);
        std::sort(b.begin(), b.end(), byEdge);
      }
      bool same = a.size() == b.size();
      for (size_t i = 0; same && i < a.size(); ++i) { ++cmpCalls; same = a[i] == b[i]; }
      if (!same) {
        if (!bad) {
          printf("    %s: PhysX %zu 번 / 우리 %zu 번\n", kOpName[op], a.size(), b.size());
          for (size_t i = 0; i < std::max(a.size(), b.size()) && i < 30; ++i)
            printf("      %zu: PhysX (%" PRIx64 ",%" PRIx64 ",%" PRIx64 ",%" PRIx64 ",%" PRIx64 ") 우리 (%" PRIx64 ",%" PRIx64 ",%" PRIx64 ",%" PRIx64 ",%" PRIx64 ")\n",
                   i, i < a.size() ? a[i].a : 0, i < a.size() ? a[i].b : 0, i < a.size() ? a[i].c : 0, i < a.size() ? a[i].d : 0, i < a.size() ? a[i].e : 0,
                   i < b.size() ? b[i].a : 0, i < b.size() ? b[i].b : 0, i < b.size() ? b[i].c : 0, i < b.size() ? b[i].d : 0, i < b.size() ? b[i].e : 0);
        }
        fail(step, kOpName[op]);
      }
    }
    // ---- 스텝 끝: 행위자 상호작용 목록 (종류·모양 쌍, 순서), 풀 빈 칸 목록, 좁은 단계 목록
    for (auto& kv : actorIndex) {
      const Sc::ActorSim* as = kv.first;
      const ss::Actor& A = M.actors[size_t(kv.second)];
      const PxU32 n = as->getActorInteractionCount();
      bool same = n == A.interactions.size();
      for (PxU32 i = 0; same && i < n; ++i) {
        ++cmpActor;
        const Sc::Interaction* it = as->getActorInteractions()[i];
        const ss::Interaction& I = M.inters[size_t(A.interactions[i])];
        if (it->getType() > Sc::InteractionType::eMARKER) {
          const auto e = extIndex.find(it);
          same = e != extIndex.end() && e->second == A.interactions[i];
          continue;
        }
        const auto e = siElems(it);
        same = int(it->getType()) == int(I.type) && e.first == I.elem0 && e.second == I.elem1;
      }
      if (!same) {
        if (!bad) printf("    행위자 %d 상호작용 PhysX %u / 우리 %zu\n", kv.second, n, A.interactions.size());
        fail(step, "행위자 상호작용 목록");
      }
    }
    {
      const auto& pool = ctx->mContactManagerPool;
      bool same = pool.mFreeCount == M.cmPool.freeList.size();
      for (PxU32 i = 0; same && i < pool.mFreeCount; ++i) same = int32_t(pool.mFreeList[i]->getIndex()) == M.cmPool.freeList[i];
      if (!same) {
        if (!bad) printf("    풀 빈 칸 PhysX %u / 우리 %zu\n", pool.mFreeCount, M.cmPool.freeList.size());
        fail(step, "접촉 관리자 풀 빈 칸 목록");
      }
      const PxsContactManagers& L = npc->mNarrowPhasePairs;
      same = L.mContactManagerMapping.size() == M.npMain.size();
      for (PxU32 i = 0; same && i < M.npMain.size(); ++i) same = int32_t(L.mContactManagerMapping[i]->getIndex()) == M.npMain.cms[i];
      if (!same) fail(step, "스텝 끝 좁은 단계 목록");
    }
    if (bad && step > firstBad + 2) break;
  }
  printf("스텝 %d: 새 겹침 %" PRIu64 "(트리거 %" PRIu64 "), 사라진 겹침 %" PRIu64 ", 닿음 시작 %" PRIu64 ", 상호작용 활성화 %" PRIu64 " 비활성화 %" PRIu64 ", 뺀 행위자 %" PRIu64 ", 거르기 자료 바꿈 %" PRIu64 ", 운동학 전환 %" PRIu64 ", 조인트 만듦·없앰 %" PRIu64 "\n", steps, nCreated,
         nTrig, nRemoved, nTouch, nAct, nDeact, nRemovedActors, nRefilterApi, nKinToggle, nJointOps);
  printf("비교: 좁은 단계 칸 %" PRIu64 ", 닿음 사건 %" PRIu64 ", 섬 호출 %" PRIu64 ", 행위자 상호작용 %" PRIu64 " / 다름 %" PRIu64, cmpList, cmpEvents, cmpCalls, cmpActor,
         bad);
  if (bad) printf(" (첫 다름 스텝 %d)", firstBad);
  if (ourNp) printf("\n우리 좁은 단계(np_step.h): 칸 %" PRIu64 " (상태·패치 수·점 수·패치/점 바이트), 접촉점 %" PRIu64, cmpNp, npPoints);
  printf("\n%s\n", bad ? "결과: 다름 있음" : "결과: 쌍 관리 전부 같음");
  return bad ? 1 : 0;
}
