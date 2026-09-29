// 섬 관리 시험: 손으로 짠 섬 관리(core/solver/islands.h) = PhysX 5.6.1 IslandSim/SimpleIslandManager ?
// 방법: 링커 --wrap 으로 Sc 층이 섬 관리자에 하는 호출(노드 추가/삭제, 간선 추가·연결·끊기, 깨우기·재우기, 1·2·3차 섬 생성)을
// 순서대로 기록한다(스레드마다 깊이를 세어 바깥 호출만). 3차 섬 생성은 작업(task)으로 나중에 돌므로 그 안의
// IslandSim::removeDestroyedEdges / processLostEdges 호출과 PostThirdPassTask 시작을 기록한다.
// 우리 섬 관리에 같은 호출열을 넣고, (1) 풀이 직전(UpdateContinuationTask) (2) fetchResults 뒤에 두 섬 시뮬(정확·추측)의
// 내부 상태 전부(노드·간선·간선 사례·섬·활성 목록·활성화 간선·재울 노드·dirty 표·번호 관리)를 PhysX 와 비교한다.
//   test_islands [--scene boxes|pile] [--n N] [--steps S] [--seed K]
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "px_internal.h"
#include "core/solver/islands.h"

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
static void rec(Rec r) {
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
  return r;
}
void R(_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE)(SIM*, PxNodeIndex);
void W(_ZN5physx2IG19SimpleIslandManager10removeNodeENS_11PxNodeIndexE)(SIM* s, PxNodeIndex n) {
  Depth d;
  if (d.top) rec(Rec{OP_REMOVE_NODE, 0, n.index(), 0, 0, 0, 0, ni(n), 0, {}, 0});
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
  return r;
}
void R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM*, PxU32, PxU32*);
void W(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(SIM* s, PxU32 nb, PxU32* handles) {
  Depth d;
  R(_ZN5physx2IG19SimpleIslandManager26preallocateContactManagersEjPj)(s, nb, handles);
  if (d.top) rec(Rec{OP_PREALLOC_CMS, 0, nb, 0, 0, 0, 0, 0, 0, std::vector<uint32_t>(handles, handles + nb), 0});
}
bool R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM*, PxU32, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
bool W(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
    SIM* s, PxU32 h, PxsContactManager* cm, PxNodeIndex n1, PxNodeIndex n2, Sc::Interaction* it, IG::Edge::EdgeType t) {
  Depth d;
  const bool r = R(_ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE)(
      s, h, cm, n1, n2, it, t);
  if (d.top) rec(Rec{OP_ADD_PREALLOC_CM, 0, h, uint32_t(t), 0, 0, 0, ni(n1), ni(n2), {uint32_t(uintptr_t(cm) & 0xffffffffu), uint32_t(uintptr_t(cm) >> 32)}, r});
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
WRAP_NODE(OP_DEACTIVATE, _ZN5physx2IG19SimpleIslandManager14deactivateNodeENS_11PxNodeIndexE)
WRAP_NODE(OP_SLEEP, _ZN5physx2IG19SimpleIslandManager14putNodeToSleepENS_11PxNodeIndexE)
WRAP_NODE(OP_SET_KINEMATIC, _ZN5physx2IG19SimpleIslandManager12setKinematicENS_11PxNodeIndexE)
WRAP_NODE(OP_SET_DYNAMIC, _ZN5physx2IG19SimpleIslandManager10setDynamicENS_11PxNodeIndexE)
#define WRAP_EDGE(OPC, MANGLED)                                    \
  void R(MANGLED)(SIM*, PxU32);                                    \
  void W(MANGLED)(SIM * s, PxU32 e) {                              \
    Depth d;                                                       \
    if (d.top) rec(Rec{OPC, 0, e, 0, 0, 0, 0, 0, 0, {}, 0});       \
    R(MANGLED)(s, e);                                              \
  }
WRAP_EDGE(OP_REMOVE_CONN, _ZN5physx2IG19SimpleIslandManager16removeConnectionEj)
WRAP_EDGE(OP_SET_DISCONNECTED, _ZN5physx2IG19SimpleIslandManager19setEdgeDisconnectedEj)
WRAP_EDGE(OP_DEACT_EDGE, _ZN5physx2IG19SimpleIslandManager14deactivateEdgeEj)
WRAP_EDGE(OP_CLEAR_RIGID_CM, _ZN5physx2IG19SimpleIslandManager16clearEdgeRigidCMEj)
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
  R(_ZN5physx2IG19SimpleIslandManager16setEdgeConnectedEjNS0_4Edge8EdgeTypeE)(s, e, t);
}
void R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM*, PxU32, PxsContactManager*);
void W(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(SIM* s, PxU32 e, PxsContactManager* cm) {
  Depth d;
  if (d.top) rec(Rec{OP_SET_RIGID_CM, 0, e, 0, 0, 0, 0, uint64_t(cm), 0, {}, 0});
  R(_ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE)(s, e, cm);
}
// IslandSim (바깥에서 직접 부르는 것)
void R(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim*, PxU32, const PxU32*);
void W(_ZN5physx2IG9IslandSim20addDelayedDirtyEdgesEjPKj)(IG::IslandSim* s, PxU32 nb, const PxU32* h) {
  Depth d;
  if (d.top) rec(Rec{OP_DELAYED_DIRTY, simId(s), nb, 0, 0, 0, 0, 0, 0, std::vector<uint32_t>(h, h + nb), 0});
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

// ---------------- 우리 섬 관리 (용량 고정)
struct OurIG {
  std::vector<std::vector<uint32_t>> u32;
  std::vector<std::vector<ig::Node>> nodes;
  std::vector<std::vector<ig::Edge>> edges;
  std::vector<std::vector<ig::EdgeInstance>> einst;
  std::vector<std::vector<ig::Island>> islands;
  std::vector<std::vector<ig::QueueElement>> pqs;
  std::vector<std::vector<ig::TraversalState>> vis;
  std::vector<ig::NodeIndex> eni;
  ig::IslandManager M;
  uint32_t* u(uint32_t n) {
    u32.emplace_back(n, 0u);
    return u32.back().data();
  }
  ig::Arr<uint32_t> au(uint32_t n) { return ig::Arr<uint32_t>{u(n), 0, n}; }
  ig::Bits bits(uint32_t nbits) { return ig::Bits{u((nbits + 31) / 32), 0, (nbits + 31) / 32}; }
  void initSim(ig::IslandSim& S, uint32_t CN, uint32_t CE) {
    S = ig::IslandSim{};
    S.islandHandles = ig::HandleManager{au(CN + 8), 0};
    nodes.emplace_back(CN);
    S.nodes = ig::Arr<ig::Node>{nodes.back().data(), 0, CN};
    S.activeNodeIndex = u(CN);
    S.hopCounts = u(CN);
    S.fastRoute = u(CN);
    S.islandIds = u(CN);
    edges.emplace_back(CE);
    S.edges = ig::Arr<ig::Edge>{edges.back().data(), 0, CE};
    einst.emplace_back(2 * CE);
    S.edgeInstances = ig::Arr<ig::EdgeInstance>{einst.back().data(), 0, 2 * CE};
    islands.emplace_back(CN + 8);
    S.islands = ig::Arr<ig::Island>{islands.back().data(), 0, CN + 8};
    S.islandStaticTouchCount = u(CN + 8);
    for (uint32_t t = 0; t < ig::NODE_TYPES; ++t) {
      S.activeNodes[t] = au(CN);
      S.nodesToPutToSleep[t] = au(CN);
    }
    S.activeKinematicNodes = au(CN);
    for (uint32_t t = 0; t < ig::EDGE_TYPES; ++t) {
      S.activatedEdges[t] = au(CE);
      S.dirtyEdges[t] = au(CE);
      S.islandSplitEdges[t] = au(CE + 1);
      S.deactivatingEdges[t] = au(CE);
    }
    S.islandAwake = bits(CN + 64);
    S.activeIslands = au(CN + 8);
    S.dirtyMap = bits(CN + 64);
    S.activatingNodes = au(CN);
    S.destroyedEdges = au(CE);
    pqs.emplace_back(CN + 8);
    S.pq = ig::Arr<ig::QueueElement>{pqs.back().data(), 0, CN + 8};
    vis.emplace_back(CN + 8);
    S.visitedNodes = ig::Arr<ig::TraversalState>{vis.back().data(), 0, CN + 8};
    S.visitedState = bits(CN + 64);
    S.cpu = &M.cpu;
  }
  void init(uint32_t CN, uint32_t CE) {
    M = ig::IslandManager{};
    M.nodeHandles = ig::HandleManager{au(CN), 0};
    M.edgeHandles = ig::HandleManager{au(CE), 0};
    M.destroyedNodes = au(CN);
    M.destroyedEdges = au(CE);
    eni.assign(2 * CE, ig::NodeIndex{ig::INVALID_NODE, 0});
    M.cpu = ig::CpuData{eni.data(), 2 * CE};
    M.constraintOrCm = u(CE);
    M.connectedMap = bits(CE + 64);
    initSim(M.accurate, CN, CE);
    initSim(M.speculative, CN, CE);
  }
};

static ig::NodeIndex toN(uint64_t ind) {
  const PxNodeIndex n(ind);
  return ig::NodeIndex{n.index(), uint32_t(ind >> 32)};
}

// 기록 한 줄을 우리 섬 관리에 적용. 결과(번호)가 PhysX 와 다르면 errs++
static void apply(OurIG& O, const Rec& r, uint64_t& resultBad) {
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
static void compareAll(const OurIG& O, Cmp& acc, Cmp& spec, Cmp& mgr) {
  compareSim(gIM->getAccurateIslandSim(), O.M.accurate, acc);
  compareSim(gIM->getSpeculativeIslandSim(), O.M.speculative, spec);
  const bool ok = sameHandles(gIM->*get(RobNodeHandles()), O.M.nodeHandles) && sameHandles(gIM->*get(RobEdgeHandles()), O.M.edgeHandles);
  mgr.chk(ok, "노드·간선 번호 관리", 0);
}

// ---------------- 디스패처: UpdateContinuationTask 직전에 비교, PostThirdPassTask 시작을 기록
static OurIG* gOur = nullptr;
static size_t gApplied = 0;
static uint64_t gResultBad = 0;
static Cmp gAccPre, gAccPost, gSpecPost, gMgrPost, gSpecPre, gMgrPre;
static void applyPending() {
  std::lock_guard<std::mutex> l(gRecM);
  for (; gApplied < gRecs.size(); ++gApplied) apply(*gOur, gRecs[gApplied], gResultBad);
}
class HookDispatcher : public PxCpuDispatcher {
 public:
  HookDispatcher() : mThread([this] { loop(); }) {}
  ~HookDispatcher() override {
    {
      std::lock_guard<std::mutex> l(mM);
      mQuit = true;
    }
    mCv.notify_all();
    mThread.join();
  }
  void submitTask(PxBaseTask& task) override {
    {
      std::lock_guard<std::mutex> l(mM);
      mQ.push_back(&task);
    }
    mCv.notify_one();
  }
  PxU32 getWorkerCount() const override { return 1; }

 private:
  void loop() {
    for (;;) {
      PxBaseTask* t = nullptr;
      {
        std::unique_lock<std::mutex> l(mM);
        mCv.wait(l, [this] { return mQuit || !mQ.empty(); });
        if (mQ.empty()) return;
        t = mQ.front();
        mQ.pop_front();
      }
      if (!strcmp(t->getName(), "PostThirdPassTask")) rec(Rec{OP_POST_THIRD, 0, 0, 0, 0, 0, 0, 0, 0, {}, 0});
      if (gOur && !strcmp(t->getName(), "UpdateContinuationTask")) {
        applyPending();
        compareSim(gIM->getAccurateIslandSim(), gOur->M.accurate, gAccPre);
      }
      t->run();
      t->release();
    }
  }
  std::mutex mM;
  std::condition_variable mCv;
  std::deque<PxBaseTask*> mQ;
  bool mQuit = false;
  std::thread mThread;
};

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

int main(int argc, char** argv) {
  std::string scene = "boxes";
  int n = 64, steps = 600, seed = 1, churn = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--scene") && i + 1 < argc) scene = argv[++i];
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--churn")) churn = 1;
  }
  OurIG our;
  our.init(1u << 14, 1u << 17);
  gOur = &our;
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  HookDispatcher disp;
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = &disp;
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  PxScene* pscene = phys->createScene(sd);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.5f, 0.0f);
  pscene->addActor(*PxCreatePlane(*phys, PxPlane(0, 0, 1, 0), *mat));
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double nn = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / nn), float(b / nn), float(c / nn), float(d / nn));
  };
  std::vector<PxRigidDynamic*> px;
  auto addBody = [&](const PxTransform& pose, const PxGeometry& g, const PxVec3& lv) {
    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    PxRigidActorExt::createExclusiveShape(*a, g, *mat);
    PxRigidBodyExt::updateMassAndInertia(*a, 600.0f);
    pscene->addActor(*a);
    a->setLinearVelocity(lv);
    px.push_back(a);
  };
  if (scene == "boxes") {
    const int side = int(std::ceil(std::sqrt(double(n))));
    for (int i = 0; i < n; ++i) {
      const float x = float(i % side) * 0.9f, y = float(i / side) * 0.9f;  // 가까이 두어 섬이 합치고 쪼개지게
      const int stack = 1 + (i % 3);
      float z = 0.0f;
      for (int s = 0; s < stack; ++s) {
        const PxVec3 he(0.1f + 0.2f * P(rng), 0.1f + 0.2f * P(rng), 0.08f + 0.1f * P(rng));
        z += he.z + 0.002f;
        addBody(PxTransform(PxVec3(x, y, z + 0.1f * P(rng)), (i % 4 == 0) ? rq() : PxQuat(3.14f * U(rng), PxVec3(0, 0, 1))), PxBoxGeometry(he),
                PxVec3(1.5f * U(rng), 1.5f * U(rng), 0.0f));
        z += he.z + 0.002f;
      }
    }
  } else {
    for (int i = 0; i < n; ++i)
      addBody(PxTransform(PxVec3(0.5f * U(rng), 0.5f * U(rng), 0.3f + 0.2f * float(i)), rq()), PxSphereGeometry(0.1f + 0.05f * P(rng)),
              PxVec3(0.5f * U(rng), 0.5f * U(rng), 0));
  }
  const float dt = 1.0f / 120.0f;
  std::mt19937 crng(seed + 77);
  uint64_t nRemoved = 0, nAdded = 0, nSleepApi = 0, nWakeApi = 0, nKick = 0;
  for (int s = 1; s <= steps; ++s) {
    if (churn) {  // 몸체 지우기·새로 넣기(노드·간선 번호 재사용), API 재우기·깨우기, 밀기(자는 섬 깨우기)
      if (s % 40 == 0 && px.size() > 4) {
        for (int k = 0; k < 2; ++k) {
          const size_t i = crng() % px.size();
          px[i]->release();
          px[i] = px.back();
          px.pop_back();
          nRemoved++;
        }
      }
      if (s % 40 == 20) {
        for (int k = 0; k < 2; ++k) {
          const PxTransform pose(PxVec3(2.0f * U(crng), 2.0f * U(crng), 0.6f + 0.3f * float(k)), rq());
          if (k & 1)
            addBody(pose, PxSphereGeometry(0.15f), PxVec3(0, 0, 0));
          else
            addBody(pose, PxBoxGeometry(0.15f, 0.12f, 0.1f), PxVec3(0, 0, 0));
          nAdded++;
        }
      }
      if (s % 30 == 10 && !px.empty()) {
        px[crng() % px.size()]->putToSleep();
        nSleepApi++;
      }
      if (s % 30 == 25 && !px.empty()) {
        px[crng() % px.size()]->wakeUp();
        nWakeApi++;
      }
      if (s % 70 == 35 && !px.empty()) {
        px[crng() % px.size()]->addForce(PxVec3(0, 0, 400.0f), PxForceMode::eIMPULSE);
        nKick++;
      }
    }
    pscene->simulate(dt);
    pscene->fetchResults(true);
    applyPending();
    compareAll(our, gAccPost, gSpecPost, gMgrPost);
    if ((gAccPost.bad || gSpecPost.bad || gMgrPost.bad || gAccPre.bad || gResultBad) && s < 100000) {
      printf("[다름] step %d: 정확(풀이 직전) %s / 정확(스텝 뒤) %s / 추측 %s / 관리 %s / 결과 번호 다름 %" PRIu64 "\n", s, gAccPre.first.c_str(),
             gAccPost.first.c_str(), gSpecPost.first.c_str(), gMgrPost.first.c_str(), gResultBad);
      break;
    }
  }
  uint64_t ops[32] = {0};
  for (auto& r : gRecs) ops[r.op]++;
  printf("\n장면 %s: 몸체 %zu 개 x %d 스텝, 섬 관리 호출 %zu 개 (노드 추가 %" PRIu64 ", 쌍 미리받기 %" PRIu64 ", 닿음 연결 %" PRIu64 ", 끊기 %" PRIu64
         ", 간선 제거 %" PRIu64 ", 깨우기 %" PRIu64 ", 재우기 %" PRIu64 ", 3차 섬 작업 %" PRIu64 ", 3차 뒤 %" PRIu64 ")\n",
         scene.c_str(), px.size(), steps, gRecs.size(), ops[OP_ADD_NODE], ops[OP_ADD_PREALLOC_CM], ops[OP_SET_CONNECTED], ops[OP_SET_DISCONNECTED],
         ops[OP_REMOVE_CONN], ops[OP_ACTIVATE], ops[OP_DEACTIVATE], ops[OP_SIM_PROCESS_LOST], ops[OP_POST_THIRD]);
  if (churn)
    printf("  흔들기: 지움 %" PRIu64 ", 새로 넣음 %" PRIu64 ", API 재우기 %" PRIu64 ", API 깨우기 %" PRIu64 ", 밀기 %" PRIu64 ", 노드 삭제 호출 %" PRIu64 "\n", nRemoved, nAdded,
           nSleepApi, nWakeApi, nKick, ops[OP_REMOVE_NODE]);
  printf("  정확 섬 시뮬 (풀이 직전): 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gAccPre.n, gAccPre.bad, gAccPre.first.c_str());
  printf("  정확 섬 시뮬 (스텝 뒤)  : 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gAccPost.n, gAccPost.bad, gAccPost.first.c_str());
  printf("  추측 섬 시뮬 (스텝 뒤)  : 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", gSpecPost.n, gSpecPost.bad, gSpecPost.first.c_str());
  printf("  번호 관리               : 비교 %" PRIu64 ", 다름 %" PRIu64 ", 호출 결과(번호) 다름 %" PRIu64 ", 우리 오류 0x%x/0x%x/0x%x\n", gMgrPost.n, gMgrPost.bad,
         gResultBad, our.M.err, our.M.accurate.err, our.M.speculative.err);
  const bool ok = !gAccPre.bad && !gAccPost.bad && !gSpecPost.bad && !gMgrPost.bad && !gResultBad && !our.M.err && !our.M.accurate.err &&
                  !our.M.speculative.err;
  printf("%s\n", ok ? "결과: 섬 관리 상태 전부 같음" : "결과: 불일치 있음");
  gOur = nullptr;
  pscene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
