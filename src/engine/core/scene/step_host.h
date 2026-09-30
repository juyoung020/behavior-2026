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
  ISL_POST_THIRD, ISL_SECOND, ISL_OP_COUNT
};
inline const char* islOpName(uint32_t op) {
  static const char* nm[] = {"addNode", "removeNode", "addCM", "preallocCMs", "addPreallocCM", "addConstraint", "activateNode", "deactivateNode", "putNodeToSleep",
                             "removeConnection", "firstPass", "addSpecAct", "second1", "second2", "third", "setConnected", "setDisconnected", "deactEdge",
                             "setRigidCM", "clearRigidCM", "setKinematic", "setDynamic", "delayedDirty", "simRemoveDestroyed", "simProcessLost", "postThird", "second"};
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

// ---- 섬 갈고리: 우리 섬 관리에 바로
struct LiveIslands : public ss::IslandHooks {
  ig::IslandManager* M = nullptr;
  ss::ScPairs* P = nullptr;
  std::vector<uint8_t>* active = nullptr;  // 행위자별 ActorSim::isActive
  bool defer = false;                      // 창(simulate 밖): 호출을 모아 두고 부르는 쪽이 차례에 맞춰 넣는다
  std::vector<IslOp> out;                  // 이 스텝에 낸 호출 (차례)
  std::vector<uint8_t> applied;            // out 과 같은 길이: 섬 관리에 넣었나 (defer 때는 부르는 쪽이 차례에 맞춰 넣는다)
  uint32_t deferBad = 0;                   // defer 중 돌려받을 값이 있는 호출 (창에서는 없어야)
  std::vector<int32_t> wakeReq;            // internalWakeUp 받은 행위자 (섬 activateNode 는 아직 바깥 기록)
  std::vector<std::pair<int32_t, int32_t>> lostTouch;

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
    wakeReq.clear();
    lostTouch.clear();
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
  // BodySim::internalWakeUp: 운동학이 아니고 깨우기 수치가 모자라면 setActive(true) (+ activateNode). 관절체는 링크 전부.
  // 잠든 몸체의 수치는 0 이라 "모자람" = 잠들어 있음. 이미 깨어 있으면 활성 표시는 그대로라 여기서는 표시만 켠다.
  void internalWakeUp(int32_t a) override {
    wakeReq.push_back(a);
    const ss::Actor& A = P->actors[size_t(a)];
    if (A.isKinematic()) return;
    if (A.articulation >= 0) {
      for (uint32_t k = 0; k < P->actors.size(); ++k)
        if (P->actors[k].articulation == A.articulation && k < active->size()) (*active)[k] = 1;
    } else if (size_t(a) < active->size()) {
      (*active)[size_t(a)] = 1;
    }
  }
  void addToLostTouchList(int32_t a0, int32_t a1) override { lostTouch.push_back({a0, a1}); }
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

// ---- 쌍 관리층 마디 (pairs_log.h pairsPost 를 섬 마디 사이로 나눈 것). 입력(새·사라진 겹침, 좁은 단계 결과)은 PairsStep 꼴.
inline void hostPairsBP(ss::ScPairs& P, const PairsStep& S) {
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

}  // namespace scene
}  // namespace eng
