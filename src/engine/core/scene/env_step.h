// env 한 스텝 함수 + 편집 창 약속 (문서 12.3, 리드): step_host.h 의 조각(섬↔쌍 관리층 직접 연결, Sc 활성 몰이, 깸/잠 요청)과
// contact 장면 단위(scene_step.h: 우리 넓은 단계·쌍 관리·좁은 단계), Sc 입력 조각(sc_scene.h)을 PhysX Sc::Scene::simulate 차례로 한 함수에 묶는다.
// 기록 없이 돈다. 모듈이 아직 안 붙은 자리는 호출자 갈고리(EnvSolve)다:
//   풀이·관절체: EnvSolve::solve — 활성 섬(isl.M.accurate)·접촉 입력(C.S 좁은 단계 칸 + contactSolverInput)을 받아 풀고,
//                풀이 뒤 깸 카운터(post: 강체 solverWc, 링크 wc, 관절체 wc)를 채우고 ScScene::updateActorCached 로 자세를 적는다.
// 차례 (G1_ISL_TRACE 로 확인한 PhysX 차례, 순서기 그림자 replay/g1_host.cpp 에서 다섯 기록 전 구간 다름 0):
//   넓은 단계(입력 = ScScene 경계 상자·접촉 거리·바뀜 비트) -> 잃은 닿음 쌍 -> dirty·새 겹침(섬 넣기) -> 섬 1차 -> 추측 추가 활성 + 간선 활성
//   -> 좁은 단계·새 닿음·연결 -> 섬 2차 ①② + 풀이 앞 깸 카운터 + 깨우기 -> 풀이(EnvSolve) -> 사라진 겹침·잃은 닿음 -> 섬 3차
//   -> destroyManagers·processLostContacts3 -> 번호 돌려주기 -> 재우기·상호작용 재우기 -> afterIntegration 깸/잠 요청 -> 스텝 끝
// 호스트 전용. 판 N 개면 판마다 하나씩 (G2 는 같은 차례를 GPU 커널로).
#pragma once
#include <cstring>
#include <vector>

#include "core/scene/env_runtime.h"
#include "core/scene/island_state.h"
#include "core/scene/sc_scene.h"
#include "core/scene/step_host.h"

namespace eng {
namespace scene {

struct EnvStep;

// 풀이·관절체 자리
struct EnvSolve {
  virtual ~EnvSolve() {}
  virtual void solve(EnvStep& E, HostWake& post) = 0;             // 풀이 (post: 강체 solverWc)
  virtual void afterIntegration(EnvStep& E, HostWake& post) { (void)E; (void)post; }  // 적분 뒤 (post: 링크·관절체 wc)
};

// Sc 편집(sc_scene.h)이 넓은 단계·섬·쌍 관리층으로 퍼지는 길 — PhysX 가 편집 API 안에서 부르는 차례 그대로 (sc_scene.h 가 차례를 정함)
struct EnvModules : public ScModules {
  EnvStep* E = nullptr;
  bool bpAdd(const BpOp& o) override;
  bool bpRemove(uint32_t index) override;
  uint64_t islandAddNode(bool awake, bool kine) override;
  void islandDeactivateNode(uint64_t node) override;
  void islandRemoveNode(uint64_t node) override;
  void pairsVolumeRemoved(uint32_t elem, bool wakeOnLostTouch) override;
  void bodyReleased(uint32_t actorID) override;
  void pairsActorAdded(int32_t h, const ScActorIn& in) override;
  void pairsShapeAdded(int32_t h, uint32_t elem) override;
};

struct EnvStep {
  EnvContact* C = nullptr;     // 넓은 단계(C->bp) + 쌍 관리층·좁은 단계(C->S)
  IslandStore* isl = nullptr;  // 섬 관리 (정확·추측)
  ScScene* sc = nullptr;       // Sc 입력 조각 (경계 상자·변환 캐시·접촉 거리·바뀜 비트, 편집 API)
  EnvSolve* solver = nullptr;
  LiveIslands live;
  HostWake wake;               // 깸 카운터 표 (풀이 모듈과 주고받음)
  std::vector<uint8_t> active; // 쌍 관리층 행위자별 ActorSim::isActive
  std::vector<int32_t> pairsOfSc;  // ScScene 손잡이 -> 쌍 관리층 행위자 번호 (적재 때 행위자 번호로 맞춤, 새 행위자는 끝에 붙임)
  EnvModules mods;
  bool contactDistChanged = false;  // Sc mHasContactDistanceChanged (편집 API 가 켬)
  uint64_t steps = 0;
  // 묶기 (적재 뒤 한 번)
  void bind() {
    live.M = &isl->M;
    live.P = &C->S->pairs;
    live.active = &active;
    live.wake = &wake;
    C->S->pairs.islands = &live;
    mods.E = this;
    if (active.size() < C->S->pairs.actors.size()) active.resize(C->S->pairs.actors.size(), 0);
  }
  ScModules& modules() { return mods; }
};

inline bool EnvModules::bpAdd(const BpOp& o) {
  BpRuntime& bp = *E->C->bp;
  bp.bounds->initEntry(o.index);
  px::Bp::AggregateHandle ea = EPX_INVALID_U32;  // 관절체 링크 모양은 집합체 안 (BpRuntime::apply 와 같은 표)
  if (o.agg != 0xffffffffu) {
    auto it = bp.aggMap.find(o.agg);
    ea = it == bp.aggMap.end() ? EPX_INVALID_U32 : it->second;
  }
  return bp.m->addBounds(o.index, o.contactDistance, px::Bp::FilterGroup::Enum(o.group), contact::userOfElem(o.index), ea,
                         px::Bp::ElementType::Enum(o.volumeType), o.env);
}
inline bool EnvModules::bpRemove(uint32_t index) { return E->C->bp->m->removeBounds(index); }
inline uint64_t EnvModules::islandAddNode(bool awake, bool kine) { return ig::addNode(E->isl->M, awake, kine, ig::eRIGID_BODY_TYPE, 0); }
inline void EnvModules::islandDeactivateNode(uint64_t node) { ig::deactivateNode(E->isl->M, uint32_t(node & 0xffffffffu)); }
inline void EnvModules::islandRemoveNode(uint64_t node) { ig::removeNode(E->isl->M, uint32_t(node & 0xffffffffu)); }
inline void EnvModules::pairsVolumeRemoved(uint32_t elem, bool wakeOnLostTouch) {
  ss::ScPairs& P = E->C->S->pairs;
  P.onVolumeRemoved(int32_t(elem), wakeOnLostTouch);
  if (elem < P.shapes.size()) P.shapes[elem].valid = false;
}
inline void EnvModules::bodyReleased(uint32_t actorID) { E->live.releasedIds.push_back(actorID); }
// 쌍 관리층 모양 칸 (Sc::ShapeCore -> ss::Shape, g1_pairs.cpp captureScene 과 같은 칸)
inline void envPairsShapeRow(EnvStep& E, int32_t h, uint32_t e) {
  ss::ScPairs& P = E.C->S->pairs;
  const ScShapeRec& r = E.sc->shapes[e];
  if (P.shapes.size() <= e) P.shapes.resize(e + 1);
  ss::Shape& S = P.shapes[e];
  S.valid = true;
  S.actor = size_t(h) < E.pairsOfSc.size() ? E.pairsOfSc[size_t(h)] : -1;
  S.geomType = int32_t(r.in.geom.type);
  S.trigger = (r.in.shapeFlags & kShapeTrigger) != 0;
  S.fd = ss::FilterData{r.in.filter[0], r.in.filter[1], r.in.filter[2], r.in.filter[3]};
  S.restOffset = r.in.restOffset;
  S.torsionalPatchRadius = r.in.torsionalPatchRadius;
  S.minTorsionalPatchRadius = r.in.minTorsionalPatchRadius;
  S.transformCacheId = e;
}
// 쌍 관리층 행위자 칸 (Sc::ActorSim: 종류·거르기 속성(ScActorSim.h:52 PxFilterObjectType + eKINEMATIC + Ex)·행위자 번호·섬 노드·지배·알림 플래그)
inline void EnvModules::pairsActorAdded(int32_t h, const ScActorIn& in) {
  ss::ScPairs& P = E->C->S->pairs;
  const ScActorRec& r = E->sc->actors[size_t(h)];
  ss::Actor A;
  const bool dyn = r.kind != 0;
  A.type = dyn ? ss::eRIGID_DYNAMIC : ss::eRIGID_STATIC;
  A.filterAttr = dyn ? (uint32_t(ss::FilterObj::eTYPE_RIGID_DYNAMIC) | ss::FilterObj::eEX_RIGID_DYNAMIC | (r.kinematic ? uint32_t(ss::FilterObj::eKINEMATIC) : 0u))
                     : (uint32_t(ss::FilterObj::eTYPE_RIGID_STATIC) | ss::FilterObj::eEX_RIGID_STATIC);
  A.actorID = r.actorID;
  A.nodeIndex = dyn ? r.node : ss::INVALID_NODE;
  A.dominanceGroup = in.dominance;
  A.offsetSlop = in.offsetSlop;
  A.forceStaticKineNotif = in.forceStaticKineNotif != 0;
  A.forceKineKineNotif = in.forceKineKineNotif != 0;
  const int32_t pa = int32_t(P.actors.size());
  P.actors.push_back(A);
  if (E->pairsOfSc.size() <= size_t(h)) E->pairsOfSc.resize(size_t(h) + 1, -1);
  E->pairsOfSc[size_t(h)] = pa;
  E->active.resize(P.actors.size(), 0);
  E->active[size_t(pa)] = (dyn && in.awake) ? 1 : 0;  // BodySim 생성: setActive(isAwake) (ScBodySim.cpp:108)
  for (uint32_t e : r.elements) envPairsShapeRow(*E, h, e);
}
inline void EnvModules::pairsShapeAdded(int32_t h, uint32_t elem) { envPairsShapeRow(*E, h, elem); }

// 넓은 단계 결과를 요소 번호 쌍으로 (scene_step.h contactBroadPhase 의 grab 과 같음)
inline void envGrabOverlaps(contact::ContactScene& S) {
  auto grab = [&](bool created, px::Bp::ElementType::Enum type, std::vector<int32_t>& out) {
    out.clear();
    px::PxU32 n = 0;
    const px::Bp::AABBOverlap* o = created ? S.aabb->getCreatedOverlaps(type, n) : S.aabb->getDestroyedOverlaps(type, n);
    for (px::PxU32 i = 0; i < n; ++i) {
      out.push_back(contact::elemOfUser(o[i].mUserData0));
      out.push_back(contact::elemOfUser(o[i].mUserData1));
    }
  };
  grab(true, px::Bp::ElementType::eSHAPE, S.createdShape);
  grab(true, px::Bp::ElementType::eTRIGGER, S.createdTrigger);
  grab(false, px::Bp::ElementType::eSHAPE, S.destroyedShape);
  grab(false, px::Bp::ElementType::eTRIGGER, S.destroyedTrigger);
}

// 한 스텝. 넓은 단계 입력 = Sc 칸(모양) + 집합체 칸은 AABB 관리자가 든 값 그대로 (집합체 칸은 입력이 아님 — 문서 3단 ①), 좁은 단계 변환 캐시·접촉 거리 = Sc 칸
struct EnvPhases {
  EnvStep& E;
  contact::ContactScene& S;
  void bp() {
    ScScene& sc = *E.sc;
    BpRuntime& bpr = *E.C->bp;
    // 칸 수 = 요소 번호 최댓값 (모양·집합체가 같은 번호 표를 씀 — BoundsArray::size 는 용량이라 쓰면 안 됨)
    const uint32_t nb = uint32_t(sc.elementIds.maxId() > sc.bounds.size() ? sc.elementIds.maxId() : sc.bounds.size());
    std::vector<px::PxBounds3> b(nb);
    for (uint32_t e = 0; e < nb; ++e) {
      const bool isShape = e < sc.shapes.size() && sc.shapes[e].alive;
      if (isShape) b[e] = sc.bounds[e];
      else if (e < bpr.bounds->size()) b[e] = bpr.bounds->begin()[e];
    }
    // 접촉 거리·바뀜 비트도 넓은 단계 칸 전체 길이로 (집합체 칸 = AABB 관리자가 든 값, 비트 0)
    std::vector<float> d(nb, 0.0f);
    for (uint32_t e = 0; e < nb; ++e) {
      const bool isShape = e < sc.shapes.size() && sc.shapes[e].alive;
      if (isShape && e < sc.contactDist.size()) d[e] = sc.contactDist[e];
      else if (e < bpr.dist->size()) d[e] = bpr.dist->begin()[e];
    }
    std::vector<uint32_t> w((nb + 31) / 32, 0u);
    for (size_t k = 0; k < w.size() && k < sc.changed.size(); ++k) w[k] = sc.changed[k];
    bool changed = false;
    for (uint32_t x : w) changed = changed || x != 0;
    if (getenv("G1_ENV_TRACE"))
      fprintf(stderr, "[env bp] 칸 %u (넓은 단계 %u, Sc %zu) 거리 %u 비트 낱말 %zu 바뀜 %d\n", nb, uint32_t(bpr.bounds->size()), sc.bounds.size(),
              uint32_t(bpr.dist->size()), w.size(), int(changed));
    bpr.step(b.data(), nb, changed, d.data(), nb, w.data(), uint32_t(w.size()), E.contactDistChanged);
    E.contactDistChanged = false;
    std::fill(sc.changed.begin(), sc.changed.end(), 0u);  // AABB 관리자가 바뀜 표를 쓰고 비움 (다음 표시는 이번 적분 뒤)
    envGrabOverlaps(S);
    contact::contactPairs(S);
  }
  void np() {
    ScScene& sc = *E.sc;
    std::vector<contact::CachedTransform> tc(sc.cache.size());
    for (size_t k = 0; k < tc.size(); ++k) {
      tc[k].transform = px::PxTransform32(sc.cache[k]);
      tc[k].flags = sc.cacheFlags[k];
    }
    contact::contactNarrowPhase(S, tc.data(), sc.contactDist.data());
  }
  void solve(HostWake& post) {
    if (E.solver) E.solver->solve(E, post);
  }
  void afterIntegration(HostWake& post) {
    if (E.solver) E.solver->afterIntegration(E, post);
  }
  void lost() {
    ss::ScPairs& M = S.pairs;
    M.processLostContacts(S.destroyedShape.data(), uint32_t(S.destroyedShape.size() / 2), S.destroyedTrigger.data(), uint32_t(S.destroyedTrigger.size() / 2));
    M.processNarrowPhaseLostTouchEventsIslands();
    M.processNarrowPhaseLostTouchEvents();
    M.processLostContacts2();
    M.lostTouchReports();
    M.unregisterInteractions();
  }
  void lost3() {
    S.pairs.destroyManagers();
    S.pairs.processLostContacts3();
  }
  std::vector<PairsAct>* acts() { return nullptr; }
};
inline void envStep(EnvStep& E) {
  contact::ContactScene& S = *E.C->S;
  E.live.clearStep();
  if (E.active.size() < S.pairs.actors.size()) E.active.resize(S.pairs.actors.size(), 0);
  EnvPhases ph{E, S};
  hostSimulateOrder(E.isl->M, S.pairs, E.live, E.active, E.wake, ph);
  contact::contactBroadPhaseEnd(S);
  E.C->bp->endStep();
  E.sc->endStep();
  ++E.steps;
}

// ---- 편집 창 (스텝 사이) — 공식 OmniGibson 물체 지우기(removing_objects) 차례. particles·omni 가 이 자리에 얹는다.
//   (1) 상태 뜨기(dump_state) -> (2) 지울 물체를 무덤 자리(100,100,100)로 순간이동 -> (3) 물리 한 스텝 더(og.sim.step_physics, 제어·렌더 없음)
//   -> (4) 떼기·지우기·새 물체 넣기·거르기 다시 -> (5) 상태 되돌리기(load_state)
// 모양·몸체 편집은 ScScene API(removeActor / detachShape / addActor / attachShape / resetFiltering)를 E.modules() 와 함께 부른다:
// ScScene 이 PhysX 차례(모양마다 넓은 단계 빼기 -> 쌍 관리층 onVolumeRemoved, 몸체 번호 풀기 -> 섬 노드 빼기)로 모듈을 부른다.
struct EditWindow {
  virtual ~EditWindow() {}
  virtual void dumpState(EnvStep& E) { (void)E; }        // (1)
  virtual void teleportToGrave(EnvStep& E) { (void)E; }  // (2) 강체 API 자세 쓰기 (core/joints/rigid_api.h setGlobalPose, autowake)
  virtual bool extraPhysicsStep() const { return false; } // (3) true 면 envEditWindow 가 envStep 한 번
  virtual void edit(EnvStep& E) { (void)E; }             // (4)
  virtual void loadState(EnvStep& E) { (void)E; }        // (5) 자세·속도·관절 쓰기
};
inline void envEditWindow(EnvStep& E, EditWindow& w) {
  w.dumpState(E);
  w.teleportToGrave(E);
  if (w.extraPhysicsStep()) envStep(E);
  w.edit(E);
  w.loadState(E);
}

}  // namespace scene
}  // namespace eng
