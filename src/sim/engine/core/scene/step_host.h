// env 한 스텝 순서기 (문서 15.3 닫힌 고리 4단, 리드): PhysX Sc::Scene::simulate 가 한 스텝에 모듈을 부르는 순서를 우리 모듈로 그대로.
// 이 파일 ① = 섬 관리자와 쌍 관리층을 **직접** 잇는 부분 (기록 대신):
//   - LiveIslands: 쌍 관리층(core/contact/sc_pairs.h)의 섬 갈고리를 우리 섬 관리(core/solver/islands.h)에 바로 건다.
//     돌려받는 값(간선 번호·isDirty)·추측 섬 노드 활성·행위자 활성 모두 우리 상태에서 (pairs_log.h 의 PairsHooks 는 PhysX 기록에서).
//   - Sc 활성 몰이 (ScPipeline.cpp·ScSleep.cpp 를 옮김):
//       activateEdgesInternal  (ScPipeline.cpp:1448, secondPassNarrowPhase 의 additionalSpeculativeActivation 바로 뒤)
//       wakeObjectsUp          (ScSleep.cpp, secondPassIslandGenPart2 바로 뒤)
//       putObjectsToSleep + 상호작용 재우기 (ScPipeline.cpp:2449 postThirdPassIslandGen)
//   - 한 스텝 차례 (추적 G1_ISL_TRACE 로 확인한 PhysX 순서, radio500_h):
//       창 API -> updateDirtyInteractions -> 넓은 단계 -> finishBroadPhase(prealloc/addPrealloc/delayedDirty) -> firstPassIslandGen
//       -> additionalSpeculativeActivation + activateEdgesInternal -> 좁은 단계 -> processNewTouches/setEdgesConnected -> secondPass1
//       -> secondPass2 + wakeObjectsUp -> 풀이 -> processLostContacts..unregisterInteractions -> thirdPassIslandGen(작업 시작)
//       -> destroyManagers -> processLostContacts3 -> PostThirdPass(번호 돌려줌) -> postThirdPassIslandGen -> afterIntegration
// 아직 바깥(기록) 몫: 노드 추가·삭제·조인트 간선·API 깨우기/재우기, 풀이 뒤 몸체 잠 판정의 activateNode/deactivateNode,
//   internalWakeUp 의 activateNode (깨우기 수치가 풀이 쪽에 있어서). 그림자 시험: replay/g1_host.cpp (G1_HOST).
// PhysX 없이 쓴다.
#pragma once
#include <cstdint>
#include <vector>

#include "core/scene/pairs_log.h"
#include "core/solver/islands.h"

namespace eng {
namespace scene {

// 섬 관리자 호출 (replay/g1_islands.cpp 의 기록 Op 와 같은 번호)
enum IslOpCode : uint32_t {
  ISL_ADD_NODE, ISL_REMOVE_NODE, ISL_ADD_CM, ISL_PREALLOC_CMS, ISL_ADD_PREALLOC_CM, ISL_ADD_CONSTRAINT, ISL_ACTIVATE, ISL_DEACTIVATE, ISL_SLEEP,
  ISL_REMOVE_CONN, ISL_FIRST_PASS, ISL_ADD_SPEC_ACT, ISL_SECOND1, ISL_SECOND2, ISL_THIRD, ISL_SET_CONNECTED, ISL_SET_DISCONNECTED, ISL_DEACT_EDGE,
  ISL_SET_RIGID_CM, ISL_CLEAR_RIGID_CM, ISL_SET_KINEMATIC, ISL_SET_DYNAMIC, ISL_DELAYED_DIRTY, ISL_SIM_REMOVE_DESTROYED, ISL_SIM_PROCESS_LOST,
  ISL_POST_THIRD, ISL_SECOND, ISL_FLUSH, ISL_OP_COUNT  // FLUSH = Sc::Scene::flush (PxScene::flushSimulation, 창)
};
inline const char* islOpName(uint32_t op) {
  static const char* nm[] = {"addNode", "removeNode", "addCM", "preallocCMs", "addPreallocCM", "addConstraint", "activateNode", "deactivateNode", "putNodeToSleep",
                             "removeConnection", "firstPass", "addSpecAct", "second1", "second2", "third", "setConnected", "setDisconnected", "deactEdge",
                             "setRigidCM", "clearRigidCM", "setKinematic", "setDynamic", "delayedDirty", "simRemoveDestroyed", "simProcessLost", "postThird", "second", "flush"};
  return op < ISL_OP_COUNT ? nm[op] : "?";
}
// 한 호출. 칸 뜻은 기록과 같다: sim(0 관리자 1 정확 2 추측), a/b/c = 번호·종류, p/q = 노드(PxNodeIndex 원값), list, result = 돌려준 값.
// removeConnection 의 c = 1 이면 조인트 간선 (바깥 몫).
struct IslOp {
  uint32_t op = 0, sim = 0, a = 0, b = 0, c = 0;
  uint64_t p = 0, q = 0;
  std::vector<uint32_t> list;
  uint32_t result = 0;
};
inline ig::NodeIndex islNode(uint64_t ind) { return ig::NodeIndex{uint32_t(ind & 0xffffffffu), uint32_t(ind >> 32)}; }

// 쌍 관리층이 내는 호출인가 (나머지는 바깥 또는 순서기의 마디)
inline bool islPairsOp(const IslOp& r) {
  switch (r.op) {
    case ISL_ADD_CM: case ISL_PREALLOC_CMS: case ISL_ADD_PREALLOC_CM: case ISL_DELAYED_DIRTY: case ISL_SET_CONNECTED: case ISL_SET_DISCONNECTED:
    case ISL_DEACT_EDGE: case ISL_SET_RIGID_CM: case ISL_CLEAR_RIGID_CM:
      return true;
    case ISL_REMOVE_CONN: return r.c == 0;
    default: return false;
  }
}
inline bool islExternalOp(const IslOp& r) {
  switch (r.op) {
    case ISL_ADD_NODE: case ISL_REMOVE_NODE: case ISL_ADD_CONSTRAINT: case ISL_ACTIVATE: case ISL_DEACTIVATE: case ISL_SLEEP: case ISL_SET_KINEMATIC:
    case ISL_SET_DYNAMIC:
      return true;
    case ISL_REMOVE_CONN: return r.c != 0;
    default: return false;
  }
}
// 같은 호출인가 (접촉 관리자 주소 같은 PhysX 전용 칸은 빼고)
inline bool islSame(const IslOp& x, const IslOp& y) {
  if (x.op != y.op) return false;
  switch (x.op) {
    case ISL_ADD_CM: return x.a == y.a && x.p == y.p && x.q == y.q && x.result == y.result;
    case ISL_PREALLOC_CMS: return x.a == y.a && x.list == y.list;
    case ISL_ADD_PREALLOC_CM: return x.a == y.a && x.b == y.b && x.p == y.p && x.q == y.q && x.result == y.result;
    case ISL_DELAYED_DIRTY: return x.sim == y.sim && x.list == y.list;
    case ISL_SET_CONNECTED: return x.a == y.a && x.b == y.b;
    default: return x.a == y.a;
  }
}
// 바깥 호출을 우리 섬 관리에 (돌려준 값이 기록과 다르면 false)
inline bool islApplyExternal(ig::IslandManager& M, const IslOp& r) {
  switch (r.op) {
    case ISL_ADD_NODE: return ig::addNode(M, r.a != 0, r.b != 0, uint8_t(r.c), uint32_t(r.p & 0xffffffffu)) == r.result;
    case ISL_REMOVE_NODE: ig::removeNode(M, r.a); return true;
    case ISL_ADD_CONSTRAINT: return ig::addConstraint(M, 0, islNode(r.p), islNode(r.q)) == r.result;
    case ISL_ACTIVATE: ig::activateNode(M, r.a); return true;
    case ISL_DEACTIVATE: ig::deactivateNode(M, r.a); return true;
    case ISL_SLEEP: ig::putNodeToSleep(M, r.a); return true;
    case ISL_REMOVE_CONN: ig::removeConnection(M, r.a); return true;
    default: M.err |= 0x10000u; return false;  // setKinematic/setDynamic 아직 안 옮김
  }
}

// ---- 깸 카운터 (Sc 층이 보는 BodyCore/ArticulationCore 값). 노드 원값(PxNodeIndex::getInd)으로 찾는다.
constexpr float kWakeReset = 20.0f * 0.02f;  // ScActorSim.h:67 ScInternalWakeCounterResetValue
struct HostBodyWake {
  uint64_t node = 0;
  float wc = 0.0f;        // BodyCore::getWakeCounter
  float solverWc = 0.0f;  // PxsBodyCore::solverWakeCounter (풀이가 낸 값, afterIntegration 에서 옮겨 적음)
  float solveWc = 0.0f;   // 풀이가 읽은 값 (순서기가 풀이 앞에 떠 둠: 사라진 닿음의 깨우기가 풀이 뒤에 wc 를 올려도 풀이는 앞 값을 봤다)
  uint8_t kinematic = 0, link = 0, pad[2] = {0, 0};
};
struct HostArtWake {
  uint32_t node = 0;              // 관절체 섬 노드 번호
  float wc = 0.0f;                // ArticulationCore 깸 카운터 (sleepCheck 뒤 = 링크 최댓값)
  std::vector<uint64_t> links;    // 링크 노드 원값 (링크 번호 순)
};
struct HostWake {
  std::vector<HostBodyWake> bodies;  // 원값 순으로 정렬
  std::vector<HostArtWake> arts;
  HostBodyWake* body(uint64_t node) {
    size_t lo = 0, hi = bodies.size();
    while (lo < hi) {
      const size_t m = (lo + hi) / 2;
      if (bodies[m].node < node) lo = m + 1;
      else hi = m;
    }
    return lo < bodies.size() && bodies[lo].node == node ? &bodies[lo] : nullptr;
  }
  HostArtWake* art(uint32_t node) {
    for (HostArtWake& a : arts)
      if (a.node == node) return &a;
    return nullptr;
  }
};

// ---- 섬 갈고리: 우리 섬 관리에 바로
struct LiveIslands : public ss::IslandHooks {
  ig::IslandManager* M = nullptr;
  ss::ScPairs* P = nullptr;
  std::vector<uint8_t>* active = nullptr;  // 행위자별 ActorSim::isActive
  HostWake* wake = nullptr;                // 깸 카운터 (없으면 internalWakeUp 은 활성 표시만)
  bool defer = false;                      // 창(simulate 밖): 호출을 모아 두고 부르는 쪽이 차례에 맞춰 넣는다
  std::vector<IslOp> out;                  // 이 스텝에 낸 호출 (차례)
  std::vector<uint8_t> applied;            // out 과 같은 길이: 섬 관리에 넣었나 (defer 때는 부르는 쪽이 차례에 맞춰 넣는다)
  uint32_t deferBad = 0;                   // defer 중 돌려받을 값이 있는 호출 (창에서는 없어야)
  std::vector<int32_t> wakeReq;            // internalWakeUp 받은 행위자 (섬 activateNode 는 아직 바깥 기록)
  struct LostTouch { int32_t a0, a1; uint32_t id0, id1; };  // Sc::SimpleBodyPair (행위자 + 그때의 행위자 번호)
  std::vector<LostTouch> lostTouch;       // Scene::mLostTouchPairs (다음 스텝 넓은 단계 뒤에 처리)
  std::vector<uint32_t> releasedIds;      // Scene::markReleasedBodyIDForLostTouch (창에서 지운 몸체의 행위자 번호, 엔진 편집 API 가 채움)

  void applyOp(IslOp& r) {
    switch (r.op) {
      case ISL_ADD_CM: r.result = ig::addContactManager(*M, r.b, islNode(r.p), islNode(r.q), r.a); break;
      case ISL_PREALLOC_CMS: r.list.resize(r.a); ig::preallocateContactManagers(*M, r.a, r.list.data()); break;
      case ISL_ADD_PREALLOC_CM: r.result = ig::addPreallocatedContactManager(*M, r.a, r.c, islNode(r.p), islNode(r.q), r.b) ? 1u : 0u; break;
      case ISL_DELAYED_DIRTY: ig::addDelayedDirtyEdges(M->speculative, uint32_t(r.list.size()), r.list.data()); break;
      case ISL_SET_CONNECTED: ig::setEdgeConnected(*M, r.a, r.b); break;
      case ISL_SET_DISCONNECTED: ig::setEdgeDisconnected(*M, r.a); break;
      case ISL_REMOVE_CONN: ig::removeConnection(*M, r.a); break;
      case ISL_SET_RIGID_CM: M->constraintOrCm[r.a] = r.b; break;
      case ISL_CLEAR_RIGID_CM: M->constraintOrCm[r.a] = ig::INVALID_EDGE; break;
      case ISL_ACTIVATE: ig::activateNode(*M, r.a); break;
      case ISL_DEACTIVATE: ig::deactivateNode(*M, r.a); break;
      default: break;  // deactivateEdge: GPU 자료만
    }
  }
  IslOp& emit(IslOp r, bool needsResult) {
    out.push_back(r);
    applied.push_back(0);
    if (!defer || needsResult) {
      if (defer) ++deferBad;
      apply(out.size() - 1);
    }
    return out.back();
  }
  void apply(size_t k) {
    if (applied[k]) return;
    applied[k] = 1;
    applyOp(out[k]);
  }
  void clearStep() {
    out.clear();
    applied.clear();
    wakeReq.clear();  // lostTouch 는 다음 simulate 의 processLostTouchPairs 가 비운다
  }

  static uint32_t cmId(int32_t cm) { return cm < 0 ? ig::INVALID_EDGE : uint32_t(cm); }
  uint32_t addContactManager(int32_t cm, uint64_t n0, uint64_t n1, int32_t, int32_t t) override {
    IslOp r;
    r.op = ISL_ADD_CM; r.a = uint32_t(t); r.b = cmId(cm); r.p = n0; r.q = n1;
    return emit(r, true).result;
  }
  void preallocateContactManagers(uint32_t n, uint32_t* h) override {
    IslOp r;
    r.op = ISL_PREALLOC_CMS; r.a = n;
    const IslOp& o = emit(r, true);
    for (uint32_t i = 0; i < n; ++i) h[i] = o.list[i];
  }
  bool addPreallocatedContactManager(uint32_t e, int32_t cm, uint64_t n0, uint64_t n1, int32_t, int32_t t) override {
    IslOp r;
    r.op = ISL_ADD_PREALLOC_CM; r.a = e; r.b = uint32_t(t); r.c = cmId(cm); r.p = n0; r.q = n1;
    return emit(r, true).result != 0;
  }
  void addDelayedDirtyEdges(uint32_t n, const uint32_t* e) override {
    IslOp r;
    r.op = ISL_DELAYED_DIRTY; r.sim = 2; r.a = n; r.list.assign(e, e + n);
    emit(r, false);
  }
  void setEdgeConnected(uint32_t e, int32_t t) override { IslOp r; r.op = ISL_SET_CONNECTED; r.a = e; r.b = uint32_t(t); emit(r, false); }
  void setEdgeDisconnected(uint32_t e) override { IslOp r; r.op = ISL_SET_DISCONNECTED; r.a = e; emit(r, false); }
  void removeConnection(uint32_t e) override { IslOp r; r.op = ISL_REMOVE_CONN; r.a = e; emit(r, false); }  // 무효 간선은 ig 쪽이 그냥 돌아감
  void setEdgeRigidCM(uint32_t e, int32_t cm) override { IslOp r; r.op = ISL_SET_RIGID_CM; r.a = e; r.b = cmId(cm); emit(r, false); }
  void clearEdgeRigidCM(uint32_t e) override { IslOp r; r.op = ISL_CLEAR_RIGID_CM; r.a = e; emit(r, false); }
  void deactivateEdge(uint32_t e) override { IslOp r; r.op = ISL_DEACT_EDGE; r.a = e; emit(r, false); }
  bool specFlag(uint64_t n, uint8_t mask) const {
    const uint32_t id = uint32_t(n & 0xffffffffu);
    return id != ig::INVALID_NODE && id < M->speculative.nodes.size && (M->speculative.nodes.d[id].flags & mask) != 0;
  }
  bool isSpeculativeNodeActive(uint64_t n) override { return specFlag(n, ig::N_ACTIVE); }
  bool isSpeculativeNodeActiveOrActivating(uint64_t n) override { return specFlag(n, ig::N_ACTIVE | ig::N_ACTIVATING); }
  bool isActorActive(int32_t a) override { return a >= 0 && size_t(a) < active->size() && (*active)[size_t(a)] != 0; }
  void markNodeActive(uint64_t node) {
    for (uint32_t k = 0; k < P->actors.size() && k < active->size(); ++k)
      if (P->actors[k].nodeIndex == node) (*active)[k] = 1;
  }
  void emitActivate(uint64_t node) {
    IslOp r;
    r.op = ISL_ACTIVATE; r.a = uint32_t(node & 0xffffffffu); r.p = node;
    emit(r, false);
  }
  // BodySim::internalWakeUpBase (ScBodySim.cpp:541): 운동학이 아니고 깸 카운터가 모자라면 올리고 setActive(true) + activateNode
  void wakeBody(uint64_t node, bool kinematic) {
    HostBodyWake* b = wake->body(node);
    if (!b || kinematic || !(b->wc < kWakeReset)) return;
    b->wc = kWakeReset;
    markNodeActive(node);
    emitActivate(node);
  }
  // BodySim::internalWakeUp (ScBodySim.cpp:527) / ArticulationSim::internalWakeUp (ScArticulationSim.cpp:501)
  void internalWakeUp(int32_t a) override {
    wakeReq.push_back(a);
    const ss::Actor& A = P->actors[size_t(a)];
    if (!wake) {  // 깸 표 없이: 활성 표시만
      if (A.isKinematic()) return;
      if (A.articulation >= 0) {
        for (uint32_t k = 0; k < P->actors.size(); ++k)
          if (P->actors[k].articulation == A.articulation && k < active->size()) (*active)[k] = 1;
      } else if (size_t(a) < active->size()) {
        (*active)[size_t(a)] = 1;
      }
      return;
    }
    if (A.articulation >= 0) {
      HostArtWake* w = wake->art(uint32_t(A.nodeIndex & 0xffffffffu));
      if (!w || !(w->wc < kWakeReset)) return;
      w->wc = kWakeReset;
      for (uint64_t l : w->links) wakeBody(l, false);  // internalWakeUpArticulationLink -> Base (운동학 검사는 링크 몸체 것: 링크는 운동학 아님)
    } else {
      wakeBody(A.nodeIndex, A.isKinematic());
    }
  }
  void addToLostTouchList(int32_t a0, int32_t a1) override { lostTouch.push_back({a0, a1, P->actors[size_t(a0)].actorID, P->actors[size_t(a1)].actorID}); }
};

// ---- Sc 활성 몰이
// SimpleIslandManager::getInteractionFromEdgeIndex (접촉 관리자 간선만: 우리 층의 겹침 상호작용이 쥔 간선)
inline std::vector<int32_t> hostEdgeInteractions(const ss::ScPairs& P, uint32_t nEdges) {
  std::vector<int32_t> m(nEdges, -1);
  for (uint32_t i = 0; i < P.inters.size(); ++i) {
    const ss::Interaction& I = P.inters[i];
    if (I.alive && I.type == ss::eOVERLAP && I.edge != ss::INVALID && I.edge < nEdges) m[I.edge] = int32_t(i);
  }
  return m;
}
// ScPipeline.cpp:1448 activateEdgesInternal(eCONTACT_MANAGER)
inline void hostActivateEdges(ig::IslandManager& M, ss::ScPairs& P, std::vector<PairsAct>* acts) {
  const ig::IslandSim& S = M.speculative;
  const std::vector<int32_t> e2i = hostEdgeInteractions(P, S.edges.size);
  const ig::Arr<uint32_t>& A = S.activatedEdges[ig::eCONTACT_MANAGER];
  for (uint32_t k = 0; k < A.size; ++k) {
    const uint32_t e = A.d[k];
    const int32_t it = e < e2i.size() ? e2i[e] : -1;
    if (it < 0 || (P.inters[size_t(it)].iflags & ss::IFlag::eIS_ACTIVE)) continue;
    if (!(S.edges.d[e].state & ig::E_ACTIVE)) continue;
    const bool r = P.activateInteraction(it);
    if (acts) acts->push_back(PairsAct{1, uint8_t(r), 0, 0, P.inters[size_t(it)].elem0, P.inters[size_t(it)].elem1});
  }
}
// 노드 -> 행위자 (관절체는 링크 전부)
inline void hostSetNodeActive(ss::ScPairs& P, std::vector<uint8_t>& active, uint32_t node, bool on, std::vector<uint32_t>* changed) {
  bool any = false;
  for (uint32_t k = 0; k < P.actors.size() && k < active.size(); ++k) {
    const uint64_t n = P.actors[k].nodeIndex;
    if (uint32_t(n & 0xffffffffu) != node || P.actors[k].isStatic()) continue;
    if ((active[k] != 0) != on) any = true;
    active[k] = on ? 1 : 0;
  }
  if (any && changed) changed->push_back(node);
}
// ScSleep.cpp wakeObjectsUp / putObjectsToSleep (정확 섬): 강체 다음 관절체. node.isActive()==on 인 것만.
inline void hostSetActiveFromIslands(ig::IslandManager& M, ss::ScPairs& P, std::vector<uint8_t>& active, bool on, std::vector<uint32_t>* changed) {
  const ig::IslandSim& S = M.accurate;
  for (uint32_t t = ig::eRIGID_BODY_TYPE; t <= ig::eARTICULATION_TYPE; ++t) {
    const ig::Arr<uint32_t>& L = on ? S.activeNodes[t] : S.nodesToPutToSleep[t];
    const uint32_t from = on ? S.initialActiveNodeCount[t] : 0u;
    for (uint32_t k = from; k < L.size; ++k) {
      const uint32_t id = L.d[k];
      if (((S.nodes.d[id].flags & ig::N_ACTIVE) != 0) == on) hostSetNodeActive(P, active, id, on, changed);
    }
  }
}
// ScPipeline.cpp:2449 postThirdPassIslandGen 의 상호작용 재우기 (추측 섬, 조인트 간선 빼고)
inline void hostDeactivateEdges(ig::IslandManager& M, ss::ScPairs& P, std::vector<PairsAct>* acts) {
  const ig::IslandSim& S = M.speculative;
  const std::vector<int32_t> e2i = hostEdgeInteractions(P, S.edges.size);
  for (uint32_t t = 0; t < ig::EDGE_TYPES; ++t) {
    if (t == ig::eCONSTRAINT) continue;
    const ig::Arr<uint32_t>& L = S.deactivatingEdges[t];
    for (uint32_t k = 0; k < L.size; ++k) {
      const uint32_t e = L.d[k];
      const int32_t it = e < e2i.size() ? e2i[e] : -1;
      if (it < 0 || !(P.inters[size_t(it)].iflags & ss::IFlag::eIS_ACTIVE)) continue;
      if (S.edges.d[e].state & ig::E_ACTIVE) continue;
      const int32_t e0 = P.inters[size_t(it)].elem0, e1 = P.inters[size_t(it)].elem1;
      const bool r = P.deactivateInteraction(it);
      if (acts) acts->push_back(PairsAct{0, uint8_t(r), 1, 0, e0, e1});
    }
  }
}

// ---- ScPipeline.cpp:853 processLostTouchPairs (postBroadPhaseStage2, 다음 simulate 의 넓은 단계 뒤): 지난 스텝 잃은 닿음 쌍 중
//      한쪽만 자고 있으면 둘 다 깨운다(둘 다 자면 그대로). 지워진 몸체가 끼면 남은 쪽만 깨운다.
//      지워짐 = 그 쌍을 적을 때의 행위자 번호가 풀린 번호 표(ScScene.cpp:1687 removeBody)에 있음. 표를 모르는 호출자(그림자)를 위해
//      행위자 칸의 번호가 바뀌었거나 모양이 하나도 안 남은 경우도 지워짐으로 본다.
inline uint32_t hostProcessLostTouchPairs(ss::ScPairs& P, LiveIslands& L) {
  uint32_t deleted = 0;
  const std::vector<LiveIslands::LostTouch> pairs = L.lostTouch;
  L.lostTouch.clear();
  auto released = [&](int32_t a, uint32_t id) {
    for (uint32_t r : L.releasedIds)
      if (r == id) return true;
    if (P.actors[size_t(a)].actorID != id) return true;
    for (uint32_t e = 0; e < P.shapes.size(); ++e)
      if (P.shapes[e].valid && P.shapes[e].actor == a) return false;
    return true;
  };
  for (const LiveIslands::LostTouch& pr : pairs) {
    const bool d1 = released(pr.a0, pr.id0), d2 = released(pr.a1, pr.id1);
    if (d1 || d2) {
      ++deleted;
      if (!d1) L.internalWakeUp(pr.a0);
      if (!d2) L.internalWakeUp(pr.a1);
      continue;
    }
    const bool a1 = L.isActorActive(pr.a0), a2 = L.isActorActive(pr.a1);
    if (!a1 && !a2) continue;
    if (!a1 || !a2) {
      L.internalWakeUp(pr.a0);
      L.internalWakeUp(pr.a1);
    }
  }
  L.releasedIds.clear();
  return deleted;
}

// ---- afterIntegration 의 깸/잠 요청 (풀이가 낸 깸 카운터 -> 섬 관리자)
// 강체: SimulationController::updateScBodyAndShapeSim (ScScene.cpp:300, 옛 작업 나누기 gUseNewTaskAllocationScheme=false):
//   정확 섬 활성 강체 노드를 모양 수 누적 256 이 넘기 전까지 한 작업으로 묶고, 작업마다 몸체를 돌며 깸 카운터를 옮겨 적은 뒤
//   활성화(이번 프레임 깸 = 풀이 앞 0 -> 풀이 뒤 > 0, DySleep.cpp:161/210) 전부 activateNode, 이어서 비활성화(풀이 뒤 0, DySleep.cpp:232) 전부 deactivateNode.
// 관절체: updateArticulationAfterIntegration (ScSimulationController.cpp:100) -> ArticulationSim::sleepCheck (ScArticulationSim.cpp:432):
//   첫 링크가 깨어 있을 때만. 링크마다 BodySim::updateWakeCounter — 앞 값 0 에서 에너지로 깨면 activateNode(링크 노드) (ScBodySim.cpp:625),
//   최댓값 0 이면 deactivateNode(관절체 노드). 깸 카운터 수치 자체는 풀이·관절체 모듈 몫이라 여기서는 앞·뒤 값을 받는다:
//   링크 뒤 값 1e-6 은 "하나도 안 자게" 고친 값(에너지로 깬 값은 0.2 이상)이라 활성화가 아니다.
inline uint32_t hostShapeCount(const ss::ScPairs& P, uint64_t node) {
  uint32_t n = 0;
  for (uint32_t e = 0; e < P.shapes.size(); ++e) {
    const ss::Shape& S = P.shapes[e];
    if (S.valid && S.actor >= 0 && P.actors[size_t(S.actor)].nodeIndex == node) ++n;
  }
  return n;
}
inline void hostSnapshotSolveWake(HostWake& W) {
  for (HostBodyWake& b : W.bodies) b.solveWc = b.wc;
}
inline void hostAfterIntegration(ig::IslandManager& M, ss::ScPairs& P, LiveIslands& L, HostWake& W, const HostWake& post) {
  const ig::IslandSim& S = M.accurate;
  // 강체
  {
    const ig::Arr<uint32_t>& A = S.activeNodes[ig::eRIGID_BODY_TYPE];
    std::vector<uint64_t> nodes;
    for (uint32_t i = 0; i < A.size; ++i) nodes.push_back(uint64_t(A.d[i]));  // 강체 노드 원값 = 번호 (링크 번호 0)
    auto chunk = [&](size_t from, size_t to) {
      std::vector<uint64_t> act, deact;
      for (size_t i = from; i < to; ++i) {
        HostBodyWake* b = W.body(nodes[i]);
        const HostBodyWake* q = const_cast<HostWake&>(post).body(nodes[i]);
        if (!b || !q || b->kinematic) continue;
        const float before = b->solveWc, after = q->solverWc;
        b->wc = after;
        b->solverWc = after;  // PxsBodyCore::solverWakeCounter = 풀이가 쓴 값 (Sc 가 그 값을 깸 카운터로 옮김 — 두 칸이 같아짐)
        if (before == 0.0f && after > 0.0f) act.push_back(nodes[i]);
        else if (after == 0.0f) deact.push_back(nodes[i]);
      }
      for (uint64_t n : act) L.emitActivate(n);
      for (uint64_t n : deact) {
        IslOp r;
        r.op = ISL_DEACTIVATE; r.a = uint32_t(n); r.p = n;
        L.emit(r, false);
      }
    };
    size_t start = 0;
    uint32_t nbShapes = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nbShapes >= 256) {
        chunk(start, i);
        start = i;
        nbShapes = 0;
      }
      const uint32_t c = hostShapeCount(P, nodes[i]);
      nbShapes += c > 1 ? c : 1;
    }
    if (nbShapes) chunk(start, nodes.size());
    // 이번에 재운 강체: 깸 카운터 0 (ScPipeline.cpp:2676)
    const ig::Arr<uint32_t>& D = S.nodesToPutToSleep[ig::eRIGID_BODY_TYPE];
    for (uint32_t i = 0; i < D.size; ++i)
      if (HostBodyWake* b = W.body(uint64_t(D.d[i]))) b->wc = 0.0f;
  }
  // 관절체
  {
    const ig::Arr<uint32_t>& A = S.activeNodes[ig::eARTICULATION_TYPE];
    for (uint32_t i = 0; i < A.size; ++i) {
      HostArtWake* w = W.art(A.d[i]);
      const HostArtWake* q = const_cast<HostWake&>(post).art(A.d[i]);
      if (!w || !q || w->links.empty()) continue;
      bool link0Active = (S.nodes.d[A.d[i]].flags & ig::N_ACTIVE) != 0;  // 첫 링크가 모양 없는 링크면 섬 노드 활성으로
      for (uint32_t k = 0; k < P.actors.size() && k < L.active->size(); ++k)
        if (P.actors[k].nodeIndex == w->links[0]) link0Active = (*L.active)[k] != 0;
      if (!link0Active) continue;
      for (uint64_t l : w->links) {
        HostBodyWake* b = W.body(l);
        const HostBodyWake* qb = const_cast<HostWake&>(post).body(l);
        if (!b || !qb) continue;
        if (b->wc == 0.0f && qb->wc > 1e-6f) L.emitActivate(l);
        b->wc = qb->wc;
      }
      w->wc = q->wc;
      if (q->wc == 0.0f) {
        IslOp r;
        r.op = ISL_DEACTIVATE; r.a = w->node; r.p = w->node;
        L.emit(r, false);
      }
    }
    // 이번에 재운 관절체: ArticulationSim::putToSleep (ScArticulationSim.cpp:406) 링크 깸 카운터 0
    const ig::Arr<uint32_t>& D = S.nodesToPutToSleep[ig::eARTICULATION_TYPE];
    for (uint32_t i = 0; i < D.size; ++i)
      if (HostArtWake* w = W.art(D.d[i]))
        for (uint64_t l : w->links)
          if (HostBodyWake* b = W.body(l)) b->wc = 0.0f;
  }
}

// ---- 쌍 관리층 마디 (pairs_log.h pairsPost 를 섬 마디 사이로 나눈 것). 입력(새·사라진 겹침, 좁은 단계 결과)은 PairsStep 꼴.
inline void hostPairsBP(ss::ScPairs& P, const PairsStep& S, LiveIslands* L = nullptr) {
  if (L) hostProcessLostTouchPairs(P, *L);  // 추정 자리: PhysX 는 새 상호작용 만들기와 섬 넣기 사이 (섬 호출 차례 비교로 확인)
  P.updateDirtyInteractions();
  P.finishBroadPhase(S.createdTrigger.data(), uint32_t(S.createdTrigger.size() / 2), S.created.data(), uint32_t(S.created.size() / 2));
}
inline void hostPairsNP(ss::ScPairs& P, const PairsStep& S) {
  P.beginNarrowPhase();
  P.mergeNarrowPhase();
  if (P.npMain.size() == S.npStatus.size())
    for (uint32_t i = 0; i < P.npMain.size(); ++i) P.narrowPhaseResult(false, i, S.npStatus[i], S.npPatches[i]);
  P.fillTouchEvents();
  P.processNewTouches();
  P.setEdgesConnected();
}
inline void hostPairsLost(ss::ScPairs& P, const PairsStep& S) {
  std::vector<int32_t> lostS, lostT;
  for (size_t i = 0; i + 1 < S.removedPairs.size(); i += 2) {
    const bool trig = P.shapes[size_t(S.removedPairs[i])].trigger || P.shapes[size_t(S.removedPairs[i + 1])].trigger;
    (trig ? lostT : lostS).push_back(S.removedPairs[i]);
    (trig ? lostT : lostS).push_back(S.removedPairs[i + 1]);
  }
  P.processLostContacts(lostS.data(), uint32_t(lostS.size() / 2), lostT.data(), uint32_t(lostT.size() / 2));
  P.processNarrowPhaseLostTouchEventsIslands();
  P.processNarrowPhaseLostTouchEvents();
  P.processLostContacts2();
  P.lostTouchReports();
  P.unregisterInteractions();
}
inline void hostPairsLost3(ss::ScPairs& P) {  // destroyManagers (thirdPassIslandGen 을 띄운 뒤) + processLostContacts3
  P.destroyManagers();
  P.processLostContacts3();
}

// ---- 섬 3차 (PxsSimpleIslandManager.cpp thirdPassIslandGen 을 작업 단위로: 번호 돌려주기는 PostThirdPass 에서)
inline void hostThirdBegin(ig::IslandManager& M) { ig::clearDeactivations(M.accurate); }
inline void hostThirdSim(ig::IslandManager& M, ig::IslandSim& S) {
  ig::removeDestroyedEdges(S);
  ig::processLostEdges(S, M.destroyedNodes.d, M.destroyedNodes.size, true, true);
}
inline void hostPostThird(ig::IslandManager& M) {
  for (uint32_t a = 0; a < M.destroyedNodes.size; ++a) M.nodeHandles.freeHandle(M.destroyedNodes[a], M.err);
  M.destroyedNodes.size = 0;
  for (uint32_t a = 0; a < M.destroyedEdges.size; ++a) M.edgeHandles.freeHandle(M.destroyedEdges[a], M.err);
  M.destroyedEdges.size = 0;
}

// ---- 한 simulate 의 고정 차례 (env 스텝 함수와 순서기 그림자의 "고정 차례" 모드가 같이 쓴다)
// Ph 가 채우는 마디: bp() = 넓은 단계 + 새 겹침을 쌍 관리층에(dirty·finishBroadPhase), np() = 좁은 단계·새 닿음·연결,
//                   solve(post) = 풀이(post 에 풀이 뒤 깸 카운터), lost() = 사라진 겹침..unregisterInteractions, lost3() = destroyManagers·processLostContacts3,
//                   afterIntegration(post) = 적분 뒤 풀이 쪽 (관절체 잠 판정 등, post 의 링크·관절체 깸 카운터)
// 풀이 앞 반쪽 (N 판을 풀이 자리에서 모아 한 번에 풀 때 판마다 부름)
template <class Ph>
inline void hostSimulatePre(ig::IslandManager& M, ss::ScPairs& P, LiveIslands& L, std::vector<uint8_t>& active, HostWake& W, Ph& ph) {
  hostProcessLostTouchPairs(P, L);  // postBroadPhaseStage2 (새 상호작용 만들기와 섬 넣기 사이 — 섬 호출이 서로 안 걸려 앞에 둠)
  ph.bp();
  ig::firstPassIslandGen(M);
  ig::additionalSpeculativeActivation(M);
  hostActivateEdges(M, P, ph.acts());
  ph.np();
  ig::secondPassIslandGenPart1(M);
  ig::secondPassIslandGenPart2(M);
  hostSnapshotSolveWake(W);
  hostSetActiveFromIslands(M, P, active, true, nullptr);
}
// 풀이 뒤 반쪽 (post = 풀이가 낸 깸 카운터 표)
template <class Ph>
inline void hostSimulatePost(ig::IslandManager& M, ss::ScPairs& P, LiveIslands& L, std::vector<uint8_t>& active, HostWake& W, HostWake& post, Ph& ph) {
  ph.lost();
  hostThirdBegin(M);
  hostThirdSim(M, M.speculative);
  hostThirdSim(M, M.accurate);
  ph.lost3();
  hostPostThird(M);
  hostSetActiveFromIslands(M, P, active, false, nullptr);
  hostDeactivateEdges(M, P, ph.acts());
  ph.afterIntegration(post);  // 풀이 쪽 적분 뒤 (관절체 잠 판정·재운 몸체 되돌리기·Sc 칸) — post 에 관절체 깸 카운터
  hostAfterIntegration(M, P, L, W, post);
}
template <class Ph>
inline void hostSimulateOrder(ig::IslandManager& M, ss::ScPairs& P, LiveIslands& L, std::vector<uint8_t>& active, HostWake& W, Ph& ph) {
  hostSimulatePre(M, P, L, active, W, ph);
  HostWake post = W;
  ph.solve(post);
  hostSimulatePost(M, P, L, active, W, post, ph);
}

}  // namespace scene
}  // namespace eng
