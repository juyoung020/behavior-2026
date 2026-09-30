// G1 섬 관리자 그림자 + 넘겨받기 시험 (문서 15.3 v1-b, 리드, 비계 전용).
// 공식 radio 재생 중 PhysX 섬 관리자(IG::SimpleIslandManager)에 들어가는 호출을 링커 --wrap 으로 기록해 우리 섬 관리(core/solver/islands.h)에 같은 순서로 넣고,
// 풀이 직전(UpdateContinuationTask)·fetchResults 뒤에 두 섬 시뮬(정확·추측)의 내부 상태 전부를 PhysX 와 비교한다.
// 기록·적용·비교 코드는 solver 작업자의 tests/solver/test_islands.cpp 를 그대로 옮겨 왔다(생성: 리드 스크립트, 원본은 고치지 않음).
// 새로 더한 것: 경계 simulate 에서 PhysX 상태 -> 우리 상태(eng::scene::IslandMgrState, 장면 파일에 들어감)로 **넘겨받아** 그 뒤로 이어 가기.
//   G1_ISLANDS=1 G1_ISLANDS_FROM=<엔진 장면 파일>   (파일의 경계 simulate 에서 파일의 섬 상태로 시작)
//   G1_ISLANDS=1 G1_ISLANDS_AT=<simulate>            (그 simulate 에서 살아 있는 PhysX 에서 바로 넘겨받음)
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../tests/solver/px_internal.h"
#include "core/solver/islands.h"
#include "core/scene/island_state.h"
#include "core/scene/scene_file.h"
#include "g1_hooks.h"

using namespace physx;
namespace ig = eng::ig;

// ---------------- 호출 기록
enum Op : uint32_t {
  OP_ADD_NODE, OP_REMOVE_NODE, OP_ADD_CM, OP_PREALLOC_CMS, OP_ADD_PREALLOC_CM, OP_ADD_CONSTRAINT, OP_ACTIVATE, OP_DEACTIVATE, OP_SLEEP,
  OP_REMOVE_CONN, OP_FIRST_PASS, OP_ADD_SPEC_ACT, OP_SECOND1, OP_SECOND2, OP_THIRD, OP_SET_CONNECTED, OP_SET_DISCONNECTED, OP_DEACT_EDGE,
  OP_SET_RIGID_CM, OP_CLEAR_RIGID_CM, OP_SET_KINEMATIC, OP_SET_DYNAMIC, OP_DELAYED_DIRTY, OP_SIM_REMOVE_DESTROYED, OP_SIM_PROCESS_LOST,
  OP_POST_THIRD, OP_SECOND,
};
struct Rec {
  uint32_t op;
  uint32_t sim;  // 0 = 관리자, 1 = 정확, 2 = 추측
  uint32_t a, b, c, d, e;
  uint64_t p, q;
  std::vector<uint32_t> list;
  uint32_t result;
};
static std::mutex gRecM;
static std::vector<Rec> gRecs;
static thread_local int tDepth = 0;
static IG::SimpleIslandManager* gIM = nullptr;
struct Depth {
  bool top;
  Depth() : top(tDepth == 0) { tDepth++; }
  ~Depth() { tDepth--; }
};
static bool gOn = false;
static void rec(Rec r) {
  if (!gOn) return;
  std::lock_guard<std::mutex> l(gRecM);
  gRecs.push_back(std::move(r));
}
static uint32_t simId(const IG::IslandSim* s) {
  if (!gIM) return 9;
  if (s == &gIM->getAccurateIslandSim()) return 1;
  if (s == &gIM->getSpeculativeIslandSim()) return 2;
  return 9;
}
static uint64_t ni(const PxNodeIndex& n) { return n.getInd(); }

using SIM = IG::SimpleIslandManager;
// 쌍 관리층 그림자(g1_pairs.cpp)가 PhysX 의 섬 호출도 받는다 (같은 기호를 두 번 --wrap 할 수 없어서 여기서 넘김)
void g1p_addCM(void* cm, uint64_t n0, uint64_t n1, void* it, uint32_t t, uint32_t edge);
void g1p_prealloc(uint32_t n, const uint32_t* h);
void g1p_addPrealloc(uint32_t e, void* cm, uint64_t n0, uint64_t n1, void* it);
void g1p_delayed(uint32_t n, const uint32_t* e);
void g1p_edge(int op, uint32_t e);  // 5 끊음 6 떼어냄(removeConnection, 조인트 간선 뺌) 8 강체 관리자 지움 9 비활성
void g1p_connect(uint32_t e, uint32_t t);
void g1p_setRigidCM(uint32_t e, void* cm);
bool g1p_isConstraintEdge(void* sim, uint32_t e);
#define W(name) __wrap_##name
#define R(name) __real_##name
extern "C" {
// SimpleIslandManager
PxNodeIndex R(_ZN5physx2IG19SimpleIslandManager7addNodeEbbNS0_4Node8NodeTypeEPv)(SIM*, bool, bool, IG::Node::NodeType, void*);
PxNodeIndex W(_ZN5physx2IG19SimpleIslandManager7addNodeEbbNS0_4Node8NodeTypeEPv)(SIM* s, bool act, bool kin, IG::Node::NodeType t, void* o) {
  gIM = s;
  Depth d;
  const PxNodeIndex r = R(_ZN5physx2IG19SimpleIslandManager7addNodeEbbNS0_4Node8NodeTypeEPv)(s, act, kin, t, o);
  if (d.top) rec(Rec{OP_ADD_NODE, 0, act, kin, uint32_t(t), 0, 0, uint64_t(o), 0, {}, r.index()});
  if (d.top) g1_sc_note_island(0, r.getInd(), act, kin);
  return r;
}
void R(_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE)(SIM*, PxNodeIndex);
void W(_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE)(SIM* s, PxNodeIndex n) {
  Depth d;
  if (d.top) rec(Rec{OP_REMOVE_NODE, 0, n.index(), 0, 0, 0, 0, ni(n), 0, {}, 0});
  if (d.top) g1_sc_note_island(2, ni(n), 0, 0);
  R(_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE)(s, n);
}
PxU32 R(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM*, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
PxU32 W(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM* s, PxsContactManager* cm, PxNodeIndex n1, PxNodeIndex n2, Sc::Interaction* it, IG::Edge::EdgeType t) {
  Depth d;
  const PxU32 r = R(_ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
      s, cm, n1, n2, it, t);
  if (d.top) rec(Rec{OP_ADD_CM, 0, uint32_t(t), 0, 0, 0, 0, ni(n1), ni(n2), {uint32_t(uintptr_t(cm) & 0xffffffffu), uint32_t(uintptr_t(cm) >> 32)}, r});
  g1p_addCM(cm, n1.getInd(), n2.getInd(), it, uint32_t(t), r);
  return r;
}
void R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM*, PxU32, PxU32*);
void W(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM* s, PxU32 nb, PxU32* handles) {
  Depth d;
  R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(s, nb, handles);
  if (d.top) rec(Rec{OP_PREALLOC_CMS, 0, nb, 0, 0, 0, 0, 0, 0, std::vector<uint32_t>(handles, handles + nb), 0});
  g1p_prealloc(nb, handles);
}
bool R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM*, PxU32, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
bool W(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM* s, PxU32 h, PxsContactManager* cm, PxNodeIndex n1, PxNodeIndex n2, Sc::Interaction* it, IG::Edge::EdgeType t) {
  Depth d;
  const bool r = R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
      s, h, cm, n1, n2, it, t);
  if (d.top) rec(Rec{OP_ADD_PREALLOC_CM, 0, h, uint32_t(t), 0, 0, 0, ni(n1), ni(n2), {uint32_t(uintptr_t(cm) & 0xffffffffu), uint32_t(uintptr_t(cm) >> 32)}, r});
  g1p_addPrealloc(h, cm, n1.getInd(), n2.getInd(), it);
  return r;
}
PxU32 R(_ZN5physx2IG19SimpleIslandManager13addConstraintEPNS_2Dy10ConstraintENS_11PxNodeIndexES5_PNS_2Sc11InteractionE)(SIM*, Dy::Constraint*, PxNodeIndex,
                                                                                                                     PxNodeIndex, Sc::Interaction*);
PxU32 W(_ZN5physx2IG19SimpleIslandManager13addConstraintEPNS_2Dy10ConstraintENS_11PxNodeIndexES5_PNS_2Sc11InteractionE)(SIM* s, Dy::Constraint* c,
                                                                                                                     PxNodeIndex n1, PxNodeIndex n2,
                                                                                                                     Sc::Interaction* it) {
  Depth d;
  const PxU32 r = R(_ZN5physx2IG19SimpleIslandManager13addConstraintEPNS_2Dy10ConstraintENS_11PxNodeIndexES5_PNS_2Sc11InteractionE)(s, c, n1, n2, it);
  if (d.top) rec(Rec{OP_ADD_CONSTRAINT, 0, 0, 0, 0, 0, 0, ni(n1), ni(n2), {}, r});
  return r;
}
#define WRAP_NODE(OPC, MANGLED)                                        \
  void R(MANGLED)(SIM*, PxNodeIndex);                                  \
  void W(MANGLED)(SIM * s, PxNodeIndex n) {                            \
    Depth d;                                                           \
    if (d.top) rec(Rec{OPC, 0, n.index(), 0, 0, 0, 0, ni(n), 0, {}, 0}); \
    R(MANGLED)(s, n);                                                  \
  }
WRAP_NODE(OP_ACTIVATE, _ZN5physx2IG19SimpleIslandManager12activateNodeENS_11PxNodeIndexE)
void R(_ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE)(SIM*, PxNodeIndex);
void W(_ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE)(SIM* s, PxNodeIndex n) {
  Depth d;
  if (d.top) rec(Rec{OP_DEACTIVATE, 0, n.index(), 0, 0, 0, 0, ni(n), 0, {}, 0});
  if (d.top) g1_sc_note_island(1, ni(n), 0, 0);
  R(_ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE)(s, n);
}
WRAP_NODE(OP_SLEEP, _ZN5physx2IG19SimpleIslandManager14putNodeToSleepENS_11PxNodeIndexE)
WRAP_NODE(OP_SET_KINEMATIC, _ZN5physx2IG19SimpleIslandManager12setKinematicENS_11PxNodeIndexE)
WRAP_NODE(OP_SET_DYNAMIC, _ZN5physx2IG19SimpleIslandManager10setDynamicENS_11PxNodeIndexE)
#define WRAP_EDGE(OPC, PAIROP, MANGLED)                            \
  void R(MANGLED)(SIM*, PxU32);                                    \
  void W(MANGLED)(SIM * s, PxU32 e) {                              \
    Depth d;                                                       \
    if (d.top) rec(Rec{OPC, 0, e, 0, 0, 0, 0, 0, 0, {}, 0});       \
    if (PAIROP != 6 || !g1p_isConstraintEdge(s, e)) g1p_edge(PAIROP, e); \
    R(MANGLED)(s, e);                                              \
  }
WRAP_EDGE(OP_REMOVE_CONN, 6, _ZN5physx2IG19SimpleIslandManager16removeConnectionEj)
WRAP_EDGE(OP_SET_DISCONNECTED, 5, _ZN5physx2IG19SimpleIslandManager19setEdgeDisconnectedEj)
WRAP_EDGE(OP_DEACT_EDGE, 9, _ZN5physx2IG19SimpleIslandManager14deactivateEdgeEj)
WRAP_EDGE(OP_CLEAR_RIGID_CM, 8, _ZN5physx2IG19SimpleIslandManager16clearEdgeRigidCMEj)
#define WRAP_VOID(OPC, MANGLED)                                   \
  void R(MANGLED)(SIM*);                                          \
  void W(MANGLED)(SIM * s) {                                      \
    Depth d;                                                      \
    if (d.top) rec(Rec{OPC, 0, 0, 0, 0, 0, 0, 0, 0, {}, 0});      \
    R(MANGLED)(s);                                                \
  }
WRAP_VOID(OP_FIRST_PASS, _ZN5physx2IG19SimpleIslandManager18firstPassIslandGenEv)
WRAP_VOID(OP_ADD_SPEC_ACT, _ZN5physx2IG19SimpleIslandManager31additionalSpeculativeActivationEv)
WRAP_VOID(OP_SECOND1, _ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart1Ev)
WRAP_VOID(OP_SECOND2, _ZN5physx2IG19SimpleIslandManager24secondPassIslandGenPart2Ev)
WRAP_VOID(OP_SECOND, _ZN5physx2IG19SimpleIslandManager19secondPassIslandGenEv)
void R(_ZN5physx2IG19SimpleIslandManager18thirdPassIslandGenEPNS_10PxBaseTaskE)(SIM*, PxBaseTask*);
void W(_ZN5physx2IG19SimpleIslandManager18thirdPassIslandGenEPNS_10PxBaseTaskE)(SIM* s, PxBaseTask* t) {
  Depth d;
  if (d.top) rec(Rec{OP_THIRD, 0, 0, 0, 0, 0, 0, 0, 0, {}, 0});
  R(_ZN5physx2IG19SimpleIslandManager18thirdPassIslandGenEPNS_10PxBaseTaskE)(s, t);
}
void R(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(SIM*, PxU32, IG::Edge::EdgeType);
void W(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(SIM* s, PxU32 e, IG::Edge::EdgeType t) {
  Depth d;
  if (d.top) rec(Rec{OP_SET_CONNECTED, 0, e, uint32_t(t), 0, 0, 0, 0, 0, {}, 0});
  g1p_connect(e, uint32_t(t));
  R(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(s, e, t);
}
void R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM*, PxU32, PxsContactManager*);
void W(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM* s, PxU32 e, PxsContactManager* cm) {
  Depth d;
  if (d.top) rec(Rec{OP_SET_RIGID_CM, 0, e, 0, 0, 0, 0, uint64_t(cm), 0, {}, 0});
  g1p_setRigidCM(e, cm);
  R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(s, e, cm);
}
// IslandSim (바깥에서 직접 부르는 것)
void R(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim*, PxU32, const PxU32*);
void W(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim* s, PxU32 nb, const PxU32* h) {
  Depth d;
  if (d.top) rec(Rec{OP_DELAYED_DIRTY, simId(s), nb, 0, 0, 0, 0, 0, 0, std::vector<uint32_t>(h, h + nb), 0});
  g1p_delayed(nb, h);
  R(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(s, nb, h);
}
void R(_ZN5physx2IG9IslandSim20removeDestroyedEdgesEv)(IG::IslandSim*);
void W(_ZN5physx2IG9IslandSim20removeDestroyedEdgesEv)(IG::IslandSim* s) {
  Depth d;
  if (d.top) rec(Rec{OP_SIM_REMOVE_DESTROYED, simId(s), 0, 0, 0, 0, 0, 0, 0, {}, 0});
  R(_ZN5physx2IG9IslandSim20removeDestroyedEdgesEv)(s);
}
typedef PxArray<PxNodeIndex, PxReflectionAllocator<PxNodeIndex> > NodeArr;
void R(_ZN5physx2IG9IslandSim16processLostEdgesERKNS_7PxArrayINS_11PxNodeIndexENS_21PxReflectionAllocatorIS3_EEEEbbj)(IG::IslandSim*, const NodeArr&, bool, bool,
                                                                                                                   PxU32);
void W(_ZN5physx2IG9IslandSim16processLostEdgesERKNS_7PxArrayINS_11PxNodeIndexENS_21PxReflectionAllocatorIS3_EEEEbbj)(IG::IslandSim* s, const NodeArr& nodes,
                                                                                                                   bool allow, bool permit, PxU32 lim) {
  Depth d;
  if (d.top) {
    std::vector<uint32_t> l;
    for (PxU32 k = 0; k < nodes.size(); ++k) l.push_back(nodes[k].index());
    rec(Rec{OP_SIM_PROCESS_LOST, simId(s), allow, permit, lim, 0, 0, 0, 0, l, 0});
  }
  R(_ZN5physx2IG9IslandSim16processLostEdgesERKNS_7PxArrayINS_11PxNodeIndexENS_21PxReflectionAllocatorIS3_EEEEbbj)(s, nodes, allow, permit, lim);
}
}  // extern "C"


// ---------------- 적용 상태
static eng::scene::IslandStore* gOur = nullptr;
static size_t gApplied = 0;
static uint64_t gResultBad = 0;
static ig::NodeIndex toN(uint64_t ind) {
  const PxNodeIndex n(ind);
  return ig::NodeIndex{n.index(), uint32_t(ind >> 32)};
}

// 기록 한 줄을 우리 섬 관리에 적용. 결과(번호)가 PhysX 와 다르면 errs++
static void apply(eng::scene::IslandStore& O, const Rec& r, uint64_t& resultBad) {
  ig::IslandManager& M = O.M;
  switch (r.op) {
    case OP_ADD_NODE: {
      const uint32_t h = ig::addNode(M, r.a != 0, r.b != 0, uint8_t(r.c), uint32_t(r.p & 0xffffffffu));
      if (h != r.result) resultBad++;
      break;
    }
    case OP_REMOVE_NODE: ig::removeNode(M, r.a); break;
    case OP_ADD_CM: {
      const uint32_t h = ig::addContactManager(M, r.list[0], toN(r.p), toN(r.q), r.a);
      if (h != r.result) resultBad++;
      break;
    }
    case OP_PREALLOC_CMS: {
      std::vector<uint32_t> hs(r.a);
      ig::preallocateContactManagers(M, r.a, hs.data());
      if (hs != r.list) resultBad++;
      break;
    }
    case OP_ADD_PREALLOC_CM: {
      const bool st = ig::addPreallocatedContactManager(M, r.a, r.list[0], toN(r.p), toN(r.q), r.b);
      if (uint32_t(st) != r.result) resultBad++;
      break;
    }
    case OP_ADD_CONSTRAINT: {
      const uint32_t h = ig::addConstraint(M, 0, toN(r.p), toN(r.q));
      if (h != r.result) resultBad++;
      break;
    }
    case OP_ACTIVATE: ig::activateNode(M, r.a); break;
    case OP_DEACTIVATE: ig::deactivateNode(M, r.a); break;
    case OP_SLEEP: ig::putNodeToSleep(M, r.a); break;
    case OP_REMOVE_CONN: ig::removeConnection(M, r.a); break;
    case OP_FIRST_PASS: ig::firstPassIslandGen(M); break;
    case OP_ADD_SPEC_ACT: ig::additionalSpeculativeActivation(M); break;
    case OP_SECOND1: ig::secondPassIslandGenPart1(M); break;
    case OP_SECOND2: ig::secondPassIslandGenPart2(M); break;
    case OP_SECOND:
      ig::secondPassIslandGenPart1(M);
      ig::secondPassIslandGenPart2(M);
      break;
    case OP_THIRD: ig::clearDeactivations(M.accurate); break;  // 나머지는 작업이 돌 때 기록된 호출로
    case OP_SET_CONNECTED: ig::setEdgeConnected(M, r.a, r.b); break;
    case OP_SET_DISCONNECTED: ig::setEdgeDisconnected(M, r.a); break;
    case OP_DEACT_EDGE: break;  // GPU 자료만
    case OP_SET_RIGID_CM: M.constraintOrCm[r.a] = uint32_t(r.p & 0xffffffffu); break;
    case OP_CLEAR_RIGID_CM: M.constraintOrCm[r.a] = ig::INVALID_EDGE; break;
    case OP_SET_KINEMATIC:
    case OP_SET_DYNAMIC: M.err |= 0x10000u; break;  // 아직 안 옮김
    case OP_DELAYED_DIRTY: ig::addDelayedDirtyEdges(r.sim == 1 ? M.accurate : M.speculative, r.a, r.list.data()); break;
    case OP_SIM_REMOVE_DESTROYED: ig::removeDestroyedEdges(r.sim == 1 ? M.accurate : M.speculative); break;
    case OP_SIM_PROCESS_LOST:
      ig::processLostEdges(r.sim == 1 ? M.accurate : M.speculative, r.list.data(), uint32_t(r.list.size()), r.a != 0, r.b != 0);
      break;
    case OP_POST_THIRD:
      for (uint32_t a = 0; a < M.destroyedNodes.size; ++a) M.nodeHandles.freeHandle(M.destroyedNodes[a], M.err);
      M.destroyedNodes.size = 0;
      for (uint32_t a = 0; a < M.destroyedEdges.size; ++a) M.edgeHandles.freeHandle(M.destroyedEdges[a], M.err);
      M.destroyedEdges.size = 0;
      break;
  }
}

// class 기본 비공개 멤버(private: 표시 없이 선언된 것)는 #define private public 으로 안 열린다 -> 명시적 인스턴스화로 멤버 포인터를 얻는다
template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) { return M; }
};
typedef IG::HandleManager<PxU32> HM;
struct RobNodeHandles { typedef HM IG::SimpleIslandManager::*type; friend type get(RobNodeHandles); };
struct RobEdgeHandles { typedef HM IG::SimpleIslandManager::*type; friend type get(RobEdgeHandles); };
struct RobFree { typedef PxArray<PxU32> HM::*type; friend type get(RobFree); };
struct RobCur { typedef PxU32 HM::*type; friend type get(RobCur); };
template struct Rob<RobNodeHandles, &IG::SimpleIslandManager::mNodeHandles>;
template struct Rob<RobEdgeHandles, &IG::SimpleIslandManager::mEdgeHandles>;
template struct Rob<RobFree, &HM::mFreeHandles>;
template struct Rob<RobCur, &HM::mCurrentHandle>;
static bool sameHandles(const HM& p, const ig::HandleManager& s) {
  const PxArray<PxU32>& fh = p.*get(RobFree());
  bool ok = p.*get(RobCur()) == s.currentHandle && fh.size() == s.freeHandles.size;
  for (PxU32 k = 0; ok && k < fh.size(); ++k) ok = fh[k] == s.freeHandles[k];
  return ok;
}

// ---------------- 상태 비교 (PhysX IslandSim 내부 vs 우리)
struct Cmp {
  uint64_t n = 0, bad = 0;
  std::string first;
  void chk(bool ok, const char* what, uint32_t idx) {
    n++;
    if (!ok) {
      bad++;
      if (first.empty()) first = std::string(what) + " #" + std::to_string(idx);
    }
  }
};
static void compareSim(const IG::IslandSim& P, const ig::IslandSim& S, Cmp& c) {
  c.chk(P.mNodes.size() == S.nodes.size, "노드 수", 0);
  for (PxU32 i = 0; i < P.mNodes.size() && i < S.nodes.size; ++i) {
    const IG::Node& a = P.mNodes[i];
    const ig::Node& b = S.nodes[i];
    c.chk(a.mFirstEdgeIndex == b.firstEdgeIndex && a.mFlags == b.flags && a.mType == b.type && a.mStaticTouchCount == b.staticTouchCount &&
              a.mNextNode.index() == b.nextNode && a.mPrevNode.index() == b.prevNode && a.mActiveRefCount == b.activeRefCount,
          "노드", i);
    c.chk(P.mActiveNodeIndex[i] == S.activeNodeIndex[i], "activeNodeIndex", i);
    c.chk(P.mHopCounts[i] == S.hopCounts[i], "hopCounts", i);
    c.chk(P.mFastRoute[i].index() == S.fastRoute[i], "fastRoute", i);
    c.chk(P.mIslandIds[i] == S.islandIds[i], "islandIds", i);
  }
  c.chk(P.mEdges.size() == S.edges.size, "간선 수", 0);
  for (PxU32 i = 0; i < P.mEdges.size() && i < S.edges.size; ++i) {
    const IG::Edge& a = P.mEdges[i];
    const ig::Edge& b = S.edges[i];
    c.chk(a.mEdgeType == b.type && a.mEdgeState == b.state && a.mNextIslandEdge == b.nextIslandEdge && a.mPrevIslandEdge == b.prevIslandEdge, "간선", i);
  }
  c.chk(P.mEdgeInstances.size() == S.edgeInstances.size, "간선 사례 수", 0);
  for (PxU32 i = 0; i < P.mEdgeInstances.size() && i < S.edgeInstances.size; ++i)
    c.chk(P.mEdgeInstances[i].mNextEdge == S.edgeInstances[i].next && P.mEdgeInstances[i].mPrevEdge == S.edgeInstances[i].prev, "간선 사례", i);
  c.chk(P.mIslands.size() == S.islands.size, "섬 수", 0);
  for (PxU32 i = 0; i < P.mIslands.size() && i < S.islands.size; ++i) {
    const IG::Island& a = P.mIslands[i];
    const ig::Island& b = S.islands[i];
    bool ok = a.mRootNode.index() == b.rootNode && a.mLastNode.index() == b.lastNode && a.mActiveIndex == b.activeIndex;
    for (int t = 0; t < 5; ++t)
      ok = ok && a.mNodeCount[t] == b.nodeCount[t] && a.mFirstEdge[t] == b.firstEdge[t] && a.mLastEdge[t] == b.lastEdge[t] &&
           a.mEdgeCount[t] == b.edgeCount[t];
    c.chk(ok, "섬", i);
    c.chk(P.mIslandStaticTouchCount[i] == S.islandStaticTouchCount[i], "섬 정적 닿음 수", i);
    const bool pa = i < P.mIslandAwake.getWordCount() * 32 && P.mIslandAwake.test(i);
    c.chk(pa == S.islandAwake.test(i), "섬 깸 비트", i);
  }
  auto cmpArr = [&](const auto& pa, const ig::Arr<uint32_t>& sa, const char* what, auto get) {
    bool ok = pa.size() == sa.size;
    for (PxU32 k = 0; ok && k < pa.size(); ++k) ok = get(pa[k]) == sa[k];
    c.chk(ok, what, 0);
  };
  auto idN = [](const PxNodeIndex& x) { return x.index(); };
  auto idU = [](PxU32 x) { return x; };
  for (int t = 0; t < 5; ++t) {
    cmpArr(P.mActiveNodes[t], S.activeNodes[t], "활성 노드 목록", idN);
    cmpArr(P.mNodesToPutToSleep[t], S.nodesToPutToSleep[t], "재울 노드 목록", idN);
    cmpArr(P.mActivatedEdges[t], S.activatedEdges[t], "활성화 간선 목록", idU);
    cmpArr(P.mDirtyEdges[t], S.dirtyEdges[t], "dirty 간선 목록", idU);
    cmpArr(P.mDeactivatingEdges[t], S.deactivatingEdges[t], "비활성화 간선 목록", idU);
    c.chk(P.mActiveEdgeCount[t] == S.activeEdgeCount[t], "활성 간선 수", t);
    c.chk(P.mInitialActiveNodeCount[t] == S.initialActiveNodeCount[t], "초기 활성 노드 수", t);
  }
  cmpArr(P.mActiveKinematicNodes, S.activeKinematicNodes, "활성 운동학 목록", idN);
  cmpArr(P.mActiveIslands, S.activeIslands, "활성 섬 목록", idU);
  cmpArr(P.mActivatingNodes, S.activatingNodes, "깨울 노드 목록", idN);
  cmpArr(P.mDestroyedEdges, S.destroyedEdges, "파괴 간선 목록", idU);
  {
    bool ok = true;
    for (uint32_t i = 0; i < P.mNodes.size(); ++i) {
      const bool pd = i < P.mDirtyMap.getWordCount() * 32 && P.mDirtyMap.test(i);
      ok = ok && (pd == S.dirtyMap.test(i));
    }
    c.chk(ok, "dirty 노드 표", 0);
  }
  c.chk(sameHandles(P.mIslandHandles, S.islandHandles), "섬 번호 관리", 0);
}
static void compareAll(const eng::scene::IslandStore& O, Cmp& acc, Cmp& spec, Cmp& mgr) {
  compareSim(gIM->getAccurateIslandSim(), O.M.accurate, acc);
  compareSim(gIM->getSpeculativeIslandSim(), O.M.speculative, spec);
  const bool ok = sameHandles(gIM->*get(RobNodeHandles()), O.M.nodeHandles) && sameHandles(gIM->*get(RobEdgeHandles()), O.M.edgeHandles);
  mgr.chk(ok, "노드·간선 번호 관리", 0);
}

static Cmp gAccPre, gAccPost, gSpecPost, gMgrPost;

static void applyPending() {
  std::lock_guard<std::mutex> l(gRecM);
  for (; gApplied < gRecs.size(); ++gApplied) apply(*gOur, gRecs[gApplied], gResultBad);
}


// SimpleIslandManager 의 클래스 기본 비공개 멤버 (#define private public 으로 안 열림) -> 명시적 인스턴스화로
struct RobDestroyedNodes { typedef PxArray<PxNodeIndex> IG::SimpleIslandManager::*type; friend type get(RobDestroyedNodes); };
struct RobDestroyedEdges { typedef PxArray<IG::EdgeIndex> IG::SimpleIslandManager::*type; friend type get(RobDestroyedEdges); };
struct RobCpuData { typedef IG::CPUExternalData IG::SimpleIslandManager::*type; friend type get(RobCpuData); };
struct RobAuxCpuData { typedef IG::AuxCpuData IG::SimpleIslandManager::*type; friend type get(RobAuxCpuData); };
struct RobConnectedMap { typedef PxBitMap IG::SimpleIslandManager::*type; friend type get(RobConnectedMap); };
template struct Rob<RobDestroyedNodes, &IG::SimpleIslandManager::mDestroyedNodes>;
template struct Rob<RobDestroyedEdges, &IG::SimpleIslandManager::mDestroyedEdges>;
template struct Rob<RobCpuData, &IG::SimpleIslandManager::mCpuData>;
template struct Rob<RobAuxCpuData, &IG::SimpleIslandManager::mAuxCpuData>;
template struct Rob<RobConnectedMap, &IG::SimpleIslandManager::mConnectedMap>;

// ---------------- PhysX 섬 관리자 -> 우리 상태 (compareSim 의 거꾸로)
namespace {
template <class A>
std::vector<uint32_t> nodeIdx(const A& a) {
  std::vector<uint32_t> v(a.size());
  for (PxU32 k = 0; k < a.size(); ++k) v[k] = a[k].index();
  return v;
}
template <class A>
std::vector<uint32_t> u32s(const A& a) {
  std::vector<uint32_t> v(a.size());
  for (PxU32 k = 0; k < a.size(); ++k) v[k] = a[k];
  return v;
}
std::vector<uint32_t> bitWords(const PxBitMap& b) {
  return std::vector<uint32_t>(b.getWords(), b.getWords() + b.getWordCount());
}
eng::scene::HandleState handles(const HM& h) {
  eng::scene::HandleState s;
  s.freeHandles = u32s(h.*get(RobFree()));
  s.currentHandle = h.*get(RobCur());
  return s;
}
template <class F>
void captureSim(const IG::IslandSim& P, eng::scene::IslandSimState& s, F objectId) {
  s.islandHandles = handles(P.mIslandHandles);
  s.nodes.resize(P.mNodes.size());
  for (PxU32 i = 0; i < P.mNodes.size(); ++i) {
    const IG::Node& a = P.mNodes[i];
    ig::Node& b = s.nodes[i];
    b.firstEdgeIndex = a.mFirstEdgeIndex;
    b.flags = a.mFlags;
    b.type = a.mType;
    b.staticTouchCount = a.mStaticTouchCount;
    b.nextNode = a.mNextNode.index();
    b.prevNode = a.mPrevNode.index();
    b.activeRefCount = a.mActiveRefCount;
    b.object = objectId(a.mObject, a.mType);
  }
  s.activeNodeIndex = u32s(P.mActiveNodeIndex);
  s.hopCounts = u32s(P.mHopCounts);
  s.fastRoute = nodeIdx(P.mFastRoute);
  s.islandIds = u32s(P.mIslandIds);
  s.edges.resize(P.mEdges.size());
  for (PxU32 i = 0; i < P.mEdges.size(); ++i) {
    const IG::Edge& a = P.mEdges[i];
    s.edges[i] = ig::Edge{uint16_t(a.mEdgeType), uint16_t(a.mEdgeState), a.mNextIslandEdge, a.mPrevIslandEdge};
  }
  s.edgeInstances.resize(P.mEdgeInstances.size());
  for (PxU32 i = 0; i < P.mEdgeInstances.size(); ++i) s.edgeInstances[i] = ig::EdgeInstance{P.mEdgeInstances[i].mNextEdge, P.mEdgeInstances[i].mPrevEdge};
  s.islands.resize(P.mIslands.size());
  for (PxU32 i = 0; i < P.mIslands.size(); ++i) {
    const IG::Island& a = P.mIslands[i];
    ig::Island& b = s.islands[i];
    b.rootNode = a.mRootNode.index();
    b.lastNode = a.mLastNode.index();
    b.activeIndex = a.mActiveIndex;
    for (int k = 0; k < 5; ++k) {
      b.nodeCount[k] = a.mNodeCount[k];
      b.firstEdge[k] = a.mFirstEdge[k];
      b.lastEdge[k] = a.mLastEdge[k];
      b.edgeCount[k] = a.mEdgeCount[k];
    }
  }
  s.islandStaticTouchCount = u32s(P.mIslandStaticTouchCount);
  for (int k = 0; k < 5; ++k) {
    s.activeNodes[k] = nodeIdx(P.mActiveNodes[k]);
    s.nodesToPutToSleep[k] = nodeIdx(P.mNodesToPutToSleep[k]);
    s.activatedEdges[k] = u32s(P.mActivatedEdges[k]);
    s.dirtyEdges[k] = u32s(P.mDirtyEdges[k]);
    s.islandSplitEdges[k] = u32s(P.mIslandSplitEdges[k]);
    s.deactivatingEdges[k] = u32s(P.mDeactivatingEdges[k]);
    s.activeEdgeCount[k] = P.mActiveEdgeCount[k];
    s.initialActiveNodeCount[k] = P.mInitialActiveNodeCount[k];
  }
  s.activeKinematicNodes = nodeIdx(P.mActiveKinematicNodes);
  s.islandAwake = bitWords(P.mIslandAwake);
  s.dirtyMap = bitWords(P.mDirtyMap);
  s.activeIslands = u32s(P.mActiveIslands);
  s.activatingNodes = nodeIdx(P.mActivatingNodes);
  s.destroyedEdges = u32s(P.mDestroyedEdges);
}
}  // namespace

// 장면 파일 뜨기(g1_dump.cpp)가 부른다. objectId: 노드 객체(PxsRigidBody* / FeatherstoneArticulation*) -> 장면 파일 번호,
// edgeObject: 간선 객체(접촉 관리자 / Dy::Constraint) -> 장면 파일 번호
bool g1_islands_capture(PxScene* scene, eng::scene::IslandMgrState& s, uint32_t (*objectId)(const void*, uint32_t, void*),
                        uint32_t (*edgeObject)(const void*, void*), void* user) {
  IG::SimpleIslandManager& im = *static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager();
  s = eng::scene::IslandMgrState{};
  s.nodeHandles = handles(im.*get(RobNodeHandles()));
  s.edgeHandles = handles(im.*get(RobEdgeHandles()));
  s.destroyedNodes = nodeIdx((im.*get(RobDestroyedNodes())));
  s.destroyedEdges = u32s((im.*get(RobDestroyedEdges())));
  const auto& eni = (im.*get(RobCpuData())).mEdgeNodeIndices;
  s.edgeNodeIndices.resize(eni.size());
  for (PxU32 k = 0; k < eni.size(); ++k) s.edgeNodeIndices[k] = ig::NodeIndex{eni[k].index(), uint32_t(eni[k].getInd() >> 32)};
  const auto& co = (im.*get(RobAuxCpuData())).mConstraintOrCm;
  s.constraintOrCm.resize(co.size());
  for (PxU32 k = 0; k < co.size(); ++k) s.constraintOrCm[k] = co[k] ? edgeObject(co[k], user) : ig::INVALID_EDGE;
  s.connectedMap = bitWords((im.*get(RobConnectedMap())));
  auto obj = [&](const void* o, uint32_t type) { return objectId(o, type, user); };
  captureSim(im.getAccurateIslandSim(), s.accurate, obj);
  captureSim(im.getSpeculativeIslandSim(), s.speculative, obj);
  s.valid = true;
  return true;
}

// ---------------- 그림자 + 넘겨받기
namespace {
struct IslShadow {
  bool inited = false;
  long long at = -1;
  std::string from;
  eng::scene::IslandStore store;
  bool running = false;
  uint64_t steps = 0;
} IS;
uint32_t ptrLow(const void* o, uint32_t, void*) { return uint32_t(uintptr_t(o) & 0xffffffffu); }
uint32_t ptrLowE(const void* o, void*) { return uint32_t(uintptr_t(o) & 0xffffffffu); }
}  // namespace

void g1_islands_before(PxScene* scene, uint64_t sim) {
  if (!IS.inited) {
    IS.inited = true;
    gOn = getenv("G1_ISLANDS") != nullptr;
    if (const char* a = getenv("G1_ISLANDS_AT")) IS.at = atoll(a);
    if (const char* f = getenv("G1_ISLANDS_FROM")) IS.from = f;
  }
  if (!gOn) return;
  eng::scene::IslandMgrState st;
  bool take = false;
  if (!IS.from.empty()) {
    static eng::scene::SceneFile ff;
    static int state = 0;
    if (state == 0) {
      std::string err;
      state = eng::scene::readScene(IS.from.c_str(), ff, &err) && ff.islands.valid ? 1 : 2;
      if (state == 2) fprintf(stderr, "[g1 섬] 파일 섬 상태 읽기 실패: %s\n", err.c_str());
    }
    if (state == 1 && sim == ff.h.sim) {
      st = ff.islands;
      take = true;
    }
  } else if (IS.at >= 0 && (long long)sim == IS.at) {
    g1_islands_capture(scene, st, ptrLow, ptrLowE, nullptr);
    take = true;
  }
  if (!take) return;
  gIM = static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager();
  const bool ok = IS.store.load(st);
  {
    std::lock_guard<std::mutex> l(gRecM);
    gRecs.clear();
    gApplied = 0;
  }
  gOur = &IS.store;
  IS.running = true;
  // 넘겨받은 직후 상태가 PhysX 와 같은지부터
  Cmp a, s, m;
  compareAll(IS.store, a, s, m);
  printf("G1 섬 넘겨받기: simulate %llu 에서 %s (노드 %zu, 간선 %zu, 활성 섬 %zu) 적재 %s — 직후 비교: 정확 %" PRIu64 "/%" PRIu64 " 추측 %" PRIu64 "/%" PRIu64
         " 번호 %" PRIu64 "/%" PRIu64 " 다름 %s\n",
         (unsigned long long)sim, IS.from.empty() ? "살아 있는 PhysX" : "파일", st.accurate.nodes.size(), st.edgeNodeIndices.size() / 2,
         st.accurate.activeIslands.size(), ok ? "됨" : "실패(용량)", a.bad, a.n, s.bad, s.n, m.bad, m.n, (a.first + s.first + m.first).c_str());
}

void g1_islands_task(const char* name) {
  if (!gOn) return;
  if (!strcmp(name, "PostThirdPassTask")) rec(Rec{OP_POST_THIRD, 0, 0, 0, 0, 0, 0, 0, 0, {}, 0});
  if (gOur && !strcmp(name, "UpdateContinuationTask")) {
    applyPending();
    compareSim(gIM->getAccurateIslandSim(), gOur->M.accurate, gAccPre);
  }
}

void g1_islands_after(PxScene*, uint64_t) {
  if (!gOn || !gOur) return;
  applyPending();
  compareAll(*gOur, gAccPost, gSpecPost, gMgrPost);
  ++IS.steps;
}

void g1_islands_report() {
  if (!gOn) return;
  printf("G1 섬 관리자 그림자 (넘겨받은 뒤 %" PRIu64 " simulate, 호출 적용 %zu)\n", IS.steps, gApplied);
  printf("  정확 섬 시뮬 (풀이 직전): 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gAccPre.n, gAccPre.bad, gAccPre.first.c_str());
  printf("  정확 섬 시뮬 (스텝 뒤)  : 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gAccPost.n, gAccPost.bad, gAccPost.first.c_str());
  printf("  추측 섬 시뮬 (스텝 뒤)  : 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gSpecPost.n, gSpecPost.bad, gSpecPost.first.c_str());
  printf("  번호 관리               : 비교 %" PRIu64 ", 다름 %" PRIu64 ", 호출 결과(번호) 다름 %" PRIu64 ", 우리 오류 0x%x/0x%x/0x%x\n", gMgrPost.n, gMgrPost.bad,
         gResultBad, gOur ? gOur->M.err : 0u, gOur ? gOur->M.accurate.err : 0u, gOur ? gOur->M.speculative.err : 0u);
}
