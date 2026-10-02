// 쌍 관리층 상태 넘겨받기 (문서 15.3 v1-b, 리드): 우리 쌍 관리층(core/contact/sc_pairs.h)은 PhysX Sc 층을 옮긴 것이라 내부 상태
// (상호작용 표·행위자 목록 순서·접촉 관리자 풀 빈 칸·좁은 단계 목록)를 PhysX 객체에서 통째로 옮기지 않고, 넓은 단계(bp_log.h)처럼
// **불러오기 구간의 입력 기록**을 장면 파일에 담아 적재 때 다시 넣는다.
// 한 스텝 입력 = 앞(simulate 전): 행위자·모양 입력 값, 조인트 충돌 표, 장면 변경 연산(거르기 자료 바꿈·운동학 전환·조인트 상호작용 생김/없어짐·
//                모양 빼기·사용자 자세 set 이 부른 관리자 다시 등록), 추측 섬 노드 활성 상태
//              뒤(Sc 한 스텝): 새 겹침(모양·트리거), 사라진 겹침, 상호작용 활성/비활성과 그 결과, 좁은 단계 결과(칸별 상태·패치 수), 섬 관리자가 돌려준 간선 번호
// 적용 순서는 PhysX Sc::Scene::simulate 순서(tests/contact/test_sc_pairs.cpp 와 같음). 그림자 시험(replay/g1_pairs.cpp)도 같은 함수로 돈다.
// PhysX 없이 쓴다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "core/contact/sc_pairs.h"

namespace eng {
namespace scene {

namespace ss = contact::sc;

// 섬 호출 종류 (비교·기록용)
enum PairsIslandOp { PIO_ADD_CM, PIO_PREALLOC, PIO_ADD_PREALLOC, PIO_DELAYED, PIO_CONNECT, PIO_DISCONNECT, PIO_REMOVE, PIO_SET_RIGID_CM, PIO_CLEAR_RIGID_CM, PIO_DEACT_EDGE, PIO_COUNT };

// 행위자 입력 값 (ss::Actor 중 이 층이 관리하는 칸 빼고)
struct PairsActorIn {
  int32_t type;
  uint32_t filterAttr, actorID;
  uint64_t nodeIndex;
  int32_t articulation;
  uint32_t linkId, parentLinkId;
  float offsetSlop;
  uint8_t fixedBaseLink, artDisableSelfCollision, hasConstraints, dominanceGroup, forceStaticKineNotif, forceKineKineNotif, pad[2];
};
inline PairsActorIn actorIn(const ss::Actor& a) {
  PairsActorIn o{};
  o.type = a.type; o.filterAttr = a.filterAttr; o.actorID = a.actorID; o.nodeIndex = a.nodeIndex; o.articulation = a.articulation;
  o.linkId = a.linkId; o.parentLinkId = a.parentLinkId; o.offsetSlop = a.offsetSlop; o.fixedBaseLink = a.fixedBaseLink;
  o.artDisableSelfCollision = a.artDisableSelfCollision; o.hasConstraints = a.hasConstraints; o.dominanceGroup = a.dominanceGroup;
  o.forceStaticKineNotif = a.forceStaticKineNotif; o.forceKineKineNotif = a.forceKineKineNotif;
  return o;
}
inline void setActorIn(ss::Actor& a, const PairsActorIn& o) {
  a.type = o.type; a.filterAttr = o.filterAttr; a.actorID = o.actorID; a.nodeIndex = o.nodeIndex; a.articulation = o.articulation;
  a.linkId = o.linkId; a.parentLinkId = o.parentLinkId; a.offsetSlop = o.offsetSlop; a.fixedBaseLink = o.fixedBaseLink != 0;
  a.artDisableSelfCollision = o.artDisableSelfCollision != 0; a.hasConstraints = o.hasConstraints != 0; a.dominanceGroup = o.dominanceGroup;
  a.forceStaticKineNotif = o.forceStaticKineNotif != 0; a.forceKineKineNotif = o.forceKineKineNotif != 0;
}

enum PairsPreOpType : int32_t { PPO_REFILTER = 0, PPO_KIN = 1, PPO_EXT_REMOVE = 2, PPO_EXT_ADD = 3, PPO_SHAPE_REMOVE = 4, PPO_API_RESET = 5 };
struct PairsPreOp {
  int32_t type, a, b, c;  // REFILTER(elem) KIN(actor, toKin) EXT_REMOVE(ext) EXT_ADD(a0, a1, type) SHAPE_REMOVE(elem) API_RESET(e0, e1)
};
struct PairsAct {
  uint8_t activate, result, afterFill, pad;
  int32_t e0, e1;
};
struct PairsJoint {
  uint64_t key;
  uint32_t coll, pad;
};
struct PairsNode {
  uint64_t node;
  uint32_t active, pad;
};

struct PairsStep {
  // 앞
  std::vector<PairsActorIn> actors;
  std::vector<ss::Shape> shapes;
  std::vector<PairsJoint> joints;
  std::vector<PairsPreOp> ops;
  std::vector<PairsNode> nodes;
  // 뒤
  std::vector<int32_t> created, createdTrigger, removedPairs;
  std::vector<PairsAct> acts;
  std::vector<uint8_t> npStatus, npPatches;
  std::vector<uint32_t> preallocHandles, addCmEdges;
};
struct PairsLog {
  bool valid = false;
  uint32_t eltsPerSlab = 0;
  int32_t kineKine = 0, staticKine = 0;
  uint32_t reportAll = 0;  // omni pair-found: 접촉 보고 대상 = 모든 쌍 (재생기 --contact-report-all)
  // 시작: 행위자·모양 입력 + 조인트·관절체 관절 상호작용 자리표(행위자 목록 순서)
  std::vector<PairsActorIn> actors0;
  std::vector<ss::Shape> shapes0;
  std::vector<PairsJoint> joints0;
  std::vector<PairsPreOp> extRecords;         // (a0, a1, type)
  std::vector<std::pair<int32_t, int32_t>> appends;  // (행위자, 자리표 번호)
  std::vector<PairsStep> steps;
};

// ---- 섬 갈고리: 돌려줄 값은 기록(PhysX 가 받은 값)에서. 호출은 onCall 로 알린다(그림자 시험이 PhysX 호출과 비교)
struct PairsHooks : public ss::IslandHooks {
  ss::ScPairs* m = nullptr;
  const std::vector<uint32_t>* prealloc = nullptr;
  const std::vector<uint32_t>* addCm = nullptr;
  size_t preallocPos = 0, addCmPos = 0;
  std::map<uint64_t, bool> nodeActive;
  int forced = -1, forcedDeact = -1;
  virtual void onCall(int, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {}
  uint64_t pairOf(int32_t it) const {
    return it < 0 ? 0 : (uint64_t(uint32_t(m->inters[size_t(it)].elem0)) << 32) | uint32_t(m->inters[size_t(it)].elem1);
  }
  uint32_t addContactManager(int32_t cm, uint64_t n0, uint64_t n1, int32_t it, int32_t t) override {
    onCall(PIO_ADD_CM, cm < 0 ? 0xffffffffull : uint64_t(cm), n0, n1, pairOf(it), uint64_t(t));
    return addCm && addCmPos < addCm->size() ? (*addCm)[addCmPos++] : 0xffffffffu;
  }
  void preallocateContactManagers(uint32_t n, uint32_t* h) override {
    onCall(PIO_PREALLOC, n, 0, 0, 0, 0);
    for (uint32_t i = 0; i < n; ++i) h[i] = prealloc && preallocPos < prealloc->size() ? (*prealloc)[preallocPos++] : 0xffffffffu;
  }
  bool addPreallocatedContactManager(uint32_t e, int32_t cm, uint64_t n0, uint64_t n1, int32_t it, int32_t) override {
    onCall(PIO_ADD_PREALLOC, e, cm < 0 ? 0xffffffffull : uint64_t(cm), n0, n1, pairOf(it));
    return false;
  }
  void addDelayedDirtyEdges(uint32_t, const uint32_t*) override {}
  void setEdgeConnected(uint32_t e, int32_t t) override { onCall(PIO_CONNECT, e, uint64_t(t), 0, 0, 0); }
  void setEdgeDisconnected(uint32_t e) override { onCall(PIO_DISCONNECT, e, 0, 0, 0, 0); }
  void removeConnection(uint32_t e) override { onCall(PIO_REMOVE, e, 0, 0, 0, 0); }
  void setEdgeRigidCM(uint32_t e, int32_t cm) override { onCall(PIO_SET_RIGID_CM, e, uint64_t(uint32_t(cm)), 0, 0, 0); }
  void clearEdgeRigidCM(uint32_t e) override { onCall(PIO_CLEAR_RIGID_CM, e, 0, 0, 0, 0); }
  void deactivateEdge(uint32_t e) override { onCall(PIO_DEACT_EDGE, e, 0, 0, 0, 0); }
  bool isSpeculativeNodeActive(uint64_t n) override {
    if (forced >= 0) return forced != 0;
    const auto it = nodeActive.find(n);
    return it != nodeActive.end() && it->second;
  }
  bool isSpeculativeNodeActiveOrActivating(uint64_t n) override { return isSpeculativeNodeActive(n); }
  bool isActorActive(int32_t a) override { return forcedDeact >= 0 ? forcedDeact == 0 : !m->actors[size_t(a)].isStatic(); }
  void internalWakeUp(int32_t) override {}
  void addToLostTouchList(int32_t, int32_t) override {}
};

// ---- 적용
inline void pairsSetInputs(ss::ScPairs& M, const std::vector<PairsActorIn>& actors, const std::vector<ss::Shape>& shapes, const std::vector<PairsJoint>& joints) {
  if (M.actors.size() < actors.size()) M.actors.resize(actors.size());
  for (size_t i = 0; i < actors.size(); ++i) setActorIn(M.actors[i], actors[i]);
  if (M.shapes.size() < shapes.size()) M.shapes.resize(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) M.shapes[i] = shapes[i];
  M.jointPairs.clear();
  for (const PairsJoint& j : joints) M.jointPairs[j.key] = ss::JointPairInfo{j.coll != 0};
}
// 시작 (장면 첫 simulate 앞). 돌려준 벡터 = 자리표 번호 (extRecords 순서)
inline std::vector<int32_t> pairsStart(ss::ScPairs& M, const PairsLog& L) {
  M.cmPool.eltsPerSlab = L.eltsPerSlab;
  M.kineKineFilteringMode = L.kineKine;
  M.staticKineFilteringMode = L.staticKine;
  pairsSetInputs(M, L.actors0, L.shapes0, L.joints0);
  std::vector<int32_t> ids;
  for (const PairsPreOp& r : L.extRecords) ids.push_back(M.newInteractionRecord(r.a, r.b, uint8_t(r.c)));
  for (const auto& ap : L.appends) M.appendToActorList(ap.first, ids[size_t(ap.second)]);
  return ids;
}
// 앞 입력 값 + 장면 변경 연산만 (섬 갈고리 상태는 안 건드림: 순서기 step_host.h 도 이것을 쓴다)
inline void pairsPreOps(ss::ScPairs& M, const PairsStep& S, std::vector<int32_t>* extIds = nullptr) {
  pairsSetInputs(M, S.actors, S.shapes, S.joints);
  for (const PairsPreOp& o : S.ops) {
    switch (o.type) {
      case PPO_REFILTER: M.setElementInteractionsDirty(o.a, ss::DirtyFlag::eFILTER_STATE, ss::IFlag::eFILTERABLE); break;
      case PPO_KIN:
        M.setActorsInteractionsDirty(o.a, ss::DirtyFlag::eBODY_KINEMATIC, -1, o.b ? ss::IFlag::eFILTERABLE : uint8_t(ss::IFlag::eFILTERABLE | ss::IFlag::eCONSTRAINT));
        break;
      case PPO_EXT_REMOVE: M.removeExternalInteraction(o.a); break;
      case PPO_EXT_ADD: {
        const int32_t id = M.addExternalInteraction(o.a, o.b, uint8_t(o.c));
        if (extIds) extIds->push_back(id);
        break;
      }
      case PPO_SHAPE_REMOVE: M.onVolumeRemoved(o.a, true); break;
      case PPO_API_RESET: {
        const int32_t it = M.findInteraction(o.a, o.b);
        if (it >= 0) M.userResetManagerCachedState(it);
        break;
      }
    }
  }
}
// 한 스텝 앞 (simulate 직전). extIds: 조인트 상호작용 자리표 번호가 생기면 돌려줌 (EXT_ADD 순서)
inline void pairsPre(ss::ScPairs& M, PairsHooks& H, const PairsStep& S, std::vector<int32_t>* extIds = nullptr) {
  pairsPreOps(M, S, extIds);
  H.nodeActive.clear();
  for (const PairsNode& n : S.nodes) H.nodeActive[n.node] = n.active != 0;
}
// 한 스텝 뒤 (Sc 한 스텝). check(단계 이름) 는 그림자 시험이 중간에 비교하려고 넘긴다 (없으면 null).
// 좁은 단계 결과는 목록 길이가 같을 때만 넣는다 (그림자 시험과 같은 규칙).
template <class Check>
inline uint32_t pairsPost(ss::ScPairs& M, PairsHooks& H, const PairsStep& S, Check check) {
  uint32_t actBad = 0;
  H.prealloc = &S.preallocHandles;
  H.addCm = &S.addCmEdges;
  H.preallocPos = H.addCmPos = 0;
  M.updateDirtyInteractions();
  M.finishBroadPhase(S.createdTrigger.data(), uint32_t(S.createdTrigger.size() / 2), S.created.data(), uint32_t(S.created.size() / 2));
  auto replayActs = [&](bool afterFill) {
    for (const PairsAct& a : S.acts) {
      if ((a.afterFill != 0) != afterFill) continue;
      const int32_t it = M.findInteraction(a.e0, a.e1);
      if (it < 0) { ++actBad; continue; }
      bool r;
      if (a.activate) {
        H.forced = a.result ? 1 : 0;
        r = M.activateInteraction(it);
      } else {
        H.forcedDeact = a.result ? 1 : 0;
        r = M.deactivateInteraction(it);
      }
      H.forced = -1;
      H.forcedDeact = -1;
      if (r != (a.result != 0)) ++actBad;
    }
  };
  replayActs(false);
  M.beginNarrowPhase();
  M.mergeNarrowPhase();
  check(0);  // 좁은 단계 목록 (결과 넣기 전)
  if (M.npMain.size() == S.npStatus.size())
    for (uint32_t i = 0; i < M.npMain.size(); ++i) M.narrowPhaseResult(false, i, S.npStatus[i], S.npPatches[i]);
  M.fillTouchEvents();
  check(1);  // 닿음 사건
  M.processNewTouches();
  M.setEdgesConnected();
  std::vector<int32_t> lostS, lostT;
  for (size_t i = 0; i + 1 < S.removedPairs.size(); i += 2) {
    const bool trig = M.shapes[size_t(S.removedPairs[i])].trigger || M.shapes[size_t(S.removedPairs[i + 1])].trigger;
    (trig ? lostT : lostS).push_back(S.removedPairs[i]);
    (trig ? lostT : lostS).push_back(S.removedPairs[i + 1]);
  }
  M.processLostContacts(lostS.data(), uint32_t(lostS.size() / 2), lostT.data(), uint32_t(lostT.size() / 2));
  M.processNarrowPhaseLostTouchEventsIslands();
  M.processNarrowPhaseLostTouchEvents();
  M.processLostContacts2();
  M.lostTouchReports();
  M.unregisterInteractions();
  M.destroyManagers();
  M.processLostContacts3();
  replayActs(true);
  return actBad;
}
// 기록 전체를 다시 넣어 경계 상태로. 거르개 함수는 부르는 쪽이 M 에 이미 걸어 둔다.
inline bool pairsReplay(ss::ScPairs& M, PairsHooks& H, const PairsLog& L, uint32_t* actBad = nullptr) {
  H.m = &M;
  M.islands = &H;
  pairsStart(M, L);
  uint32_t bad = 0;
  for (const PairsStep& S : L.steps) {
    pairsPre(M, H, S);
    bad += pairsPost(M, H, S, [](int) {});
  }
  if (actBad) *actBad = bad;
  return true;
}

// ---- 파일
namespace pio {
template <class T>
inline bool w(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  return fwrite(&n, 8, 1, f) == 1 && (n == 0 || fwrite(static_cast<const void*>(v.data()), sizeof(T), n, f) == n);
}
template <class T>
inline bool r(FILE* f, std::vector<T>& v) {
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1 || n > (1ull << 32)) return false;
  v.resize(n);
  return n == 0 || fread(static_cast<void*>(v.data()), sizeof(T), n, f) == n;
}
}  // namespace pio
inline bool writePairsLog(FILE* f, const PairsLog& L) {
  const uint32_t hdr[7] = {0x50524c31u, L.eltsPerSlab, uint32_t(L.kineKine), uint32_t(L.staticKine), L.reportAll, uint32_t(sizeof(ss::Shape)), uint32_t(L.steps.size())};
  bool ok = fwrite(hdr, 4, 7, f) == 7 && pio::w(f, L.actors0) && pio::w(f, L.shapes0) && pio::w(f, L.joints0) && pio::w(f, L.extRecords) && pio::w(f, L.appends);
  for (const PairsStep& S : L.steps)
    ok = ok && pio::w(f, S.actors) && pio::w(f, S.shapes) && pio::w(f, S.joints) && pio::w(f, S.ops) && pio::w(f, S.nodes) && pio::w(f, S.created) &&
         pio::w(f, S.createdTrigger) && pio::w(f, S.removedPairs) && pio::w(f, S.acts) && pio::w(f, S.npStatus) && pio::w(f, S.npPatches) &&
         pio::w(f, S.preallocHandles) && pio::w(f, S.addCmEdges);
  return ok;
}
inline bool readPairsLog(FILE* f, PairsLog& L) {
  uint32_t hdr[7];
  if (fread(hdr, 4, 7, f) != 7 || hdr[0] != 0x50524c31u || hdr[5] != sizeof(ss::Shape)) return false;
  L.eltsPerSlab = hdr[1]; L.kineKine = int32_t(hdr[2]); L.staticKine = int32_t(hdr[3]); L.reportAll = hdr[4];
  bool ok = pio::r(f, L.actors0) && pio::r(f, L.shapes0) && pio::r(f, L.joints0) && pio::r(f, L.extRecords) && pio::r(f, L.appends);
  L.steps.resize(hdr[6]);
  for (PairsStep& S : L.steps)
    ok = ok && pio::r(f, S.actors) && pio::r(f, S.shapes) && pio::r(f, S.joints) && pio::r(f, S.ops) && pio::r(f, S.nodes) && pio::r(f, S.created) &&
         pio::r(f, S.createdTrigger) && pio::r(f, S.removedPairs) && pio::r(f, S.acts) && pio::r(f, S.npStatus) && pio::r(f, S.npPatches) &&
         pio::r(f, S.preallocHandles) && pio::r(f, S.addCmEdges);
  L.valid = ok;
  return ok;
}

}  // namespace scene
}  // namespace eng
