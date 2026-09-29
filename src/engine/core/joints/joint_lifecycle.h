// joints 모듈: 조인트 붙이기·떼기(생성·해제)의 장면 쪽 부수효과 — PhysX 5.6.1 과 같은 순서·같은 번호 (CPU·CUDA 공용).
// 보조 잡기(OmniGibson robots/robot.py:3597 create_joint, :2059 delete_or_deactivate_prim)는 USD 조인트를 만들고 지우고,
// omni.physx 는 그때 PxD6JointCreate / PxJoint::release 를 부른다. 그 둘이 장면에 남기는 것:
// 원본(physx/source 기준)
//   생성 : physx/src/NpConstraint.cpp (생성자 -> NpScene::addToConstraintList -> scAddConstraint)
//          simulationcontroller/src/ScScene.cpp:1690 (Sc::Scene::addConstraint: ConstraintSim 생성 -> addConstraintToMap)
//          ScConstraintSim.cpp:57 (생성자: 제약 번호 createID, writeback[id].initialize(), 상수 블록 0, break 판정, 상호작용 생성)
//          ScConstraintInteraction.cpp:39 (onActivate, 몸체마다 onConstraintAttach = BF_HAS_CONSTRAINTS + numCountedInteractions++,
//                                          섬 관리자 addConstraint(간선))
//   해제 : ScScene.cpp:1701 (removeConstraint: 지도에서 지움 -> ConstraintSim 파괴)
//          ScConstraintSim.cpp:88 (파괴자: 상호작용 destroy -> releaseID(미룸) -> 상수 블록 반납)
//          ScConstraintInteraction.cpp:73 (destroy: 섬 관리자 removeConnection(간선) -> 몸체마다 onConstraintDetach)
//          ScBodySim.cpp:722 (onConstraintDetach: numCountedInteractions--, 다른 제약 상호작용이 없으면 BF_HAS_CONSTRAINTS 내림)
//   번호 : ScObjectIDTracker.h:41 (releaseID 는 미뤘다가 postReportsCleanup(ScScene.cpp:1886, fetchResults 끝)에서 풂)
//          common/src/CmIDPool.h:40 (IDPoolBase: 빈 번호 목록 뒤에서 꺼냄, 맨 끝 번호를 풀면 mCurrentID 를 줄임)
//   거르개: ScFiltering.cpp:299 (filterJointedBodies: 두 몸체 모두 BF_HAS_CONSTRAINTS 가 없으면 false, 아니면 지도에서 찾아
//          eCOLLISION_ENABLED 가 없으면 쌍을 버림). ScScene.cpp:1715,1739,3027 (지도: 노드 번호 작은 쪽이 앞인 (ActorSim, ActorSim) 쌍,
//          PxHashMap::insert 는 이미 있으면 덮지 않고, erase 는 어느 조인트를 지우든 그 쌍 칸을 지운다).
// 주의(원본 동작 그대로): 조인트를 만들어도 이미 있는 접촉 쌍은 다시 거르지 않는다 — 거르개는 쌍이 새로 생길 때만 본다.
// 깨우기: 조인트 생성·해제 자체는 몸체를 깨우지 않는다(wakeCounter 불변). 깸의 퍼짐은 섬 관리자가 간선 추가/제거를 처리할 때
//   생긴다(PxsIslandSim processNewEdges/processLostEdges — solver 몫). 여기서는 그 간선 사건을 PhysX 순서대로 내놓는다.
//   관절체 링크도 같다(ArticulationLink 의 BodySim 에 counted/flag, 섬 노드 = 관절체 노드).
#pragma once
#include <cstdint>

#include "../common/pmath.h"

namespace eng {
namespace jnt {

enum : uint32_t { NO_ACTOR = 0xffffffffu };
enum : uint16_t { BF_HAS_CONSTRAINTS = 1 << 8 };  // Sc::ActorSim::InternalFlags (ScActorSim.h:120)

// Cm::IDPoolBase + Sc::ObjectIDTracker (CAP = 판 하나에 동시에 있을 수 있는 제약 수 상한)
template <uint32_t CAP>
struct ConstraintIdTracker {
  uint32_t current = 0;
  uint32_t nFree = 0, nPending = 0;
  uint32_t freeIds[CAP];
  uint32_t pending[CAP];
  bool overflow = false;
  EHD uint32_t createID() {  // IDPoolBase::getNewID
    if (nFree) return freeIds[--nFree];
    if (current >= CAP) overflow = true;
    return current++;
  }
  EHD void releaseID(uint32_t id) {  // ObjectIDTracker::releaseID (markIDAsDeleted + 미룸)
    if (nPending < CAP) pending[nPending++] = id; else overflow = true;
  }
  EHD void freeID(uint32_t id) {  // IDPoolBase::freeID
    if (id == current - 1) --current;
    else if (nFree < CAP) freeIds[nFree++] = id;
    else overflow = true;
  }
  EHD void processPendingReleases() {  // fetchResults 끝 (Sc::Scene::postReportsCleanup)
    for (uint32_t i = 0; i < nPending; i++) freeID(pending[i]);
    nPending = 0;
  }
};

// 몸체 쪽 (Sc::BodySim) — 강체·관절체 링크 모두. 정적 행위자는 없음(ActorSim 이지만 flag 를 올리지 않는다).
struct JointBodyState {
  uint32_t numCountedInteractions;  // PxsBodyCore::numCountedInteractions (접촉 상호작용도 올린다 — 합계는 몸체 공통 칸)
  uint32_t numConstraints;          // 이 몸체에 붙은 살아 있는 제약 상호작용 수 (BF_HAS_CONSTRAINTS = numConstraints > 0)
};

// 장면 쪽 제약 한 개
struct ConstraintSlot {
  uint32_t id;          // Dy::Constraint::index = writeback 칸 = getGPUIndex (eGPU_COMPATIBLE 일 때)
  uint32_t actor[2];    // 행위자 번호 (NO_ACTOR = 세계)
  uint8_t dynamic[2];   // 동적(강체·링크) 이면 1 — BodySim 이 있음
  uint8_t alive;
  uint16_t flags;       // PxConstraintFlags
  uint32_t edge;        // 섬 간선 번호 (solver 가 채움)
};

enum : uint8_t { EV_EDGE_ADD = 1, EV_EDGE_REMOVE = 2 };
struct EdgeEvent {  // 섬 관리자에 줄 사건 (PhysX 호출 순서 그대로)
  uint8_t type;
  uint32_t slot;       // ConstraintSlot 번호
  uint32_t actor[2];   // 동적이 아니면 NO_ACTOR (PxNodeIndex() 에 해당)
};

// 쌍 지도 (Sc::Scene::mConstraintMap): 행위자 쌍 -> 먼저 들어간 제약 칸
struct PairEntry {
  uint32_t a, b;   // 정렬된 행위자 번호 (정적 포함, 세계 = NO_ACTOR)
  uint32_t slot;
};

EHD void pairKey(uint32_t a0, uint32_t a1, uint32_t& a, uint32_t& b) {
  a = a0 < a1 ? a0 : a1;
  b = a0 < a1 ? a1 : a0;
}

template <uint32_t MAXC, uint32_t MAXB, uint32_t MAXEV>
struct JointScene {
  ConstraintIdTracker<MAXC> ids;
  ConstraintSlot slot[MAXC];
  uint32_t nSlots = 0;
  JointBodyState body[MAXB];
  PairEntry map[MAXC];
  uint32_t nMap = 0;
  EdgeEvent ev[MAXEV];
  uint32_t nEv = 0;
  bool overflow = false;

  EHD void pushEv(uint8_t t, uint32_t s) {
    if (nEv >= MAXEV) { overflow = true; return; }
    EdgeEvent& e = ev[nEv++];
    e.type = t;
    e.slot = s;
    for (int k = 0; k < 2; ++k) e.actor[k] = slot[s].dynamic[k] ? slot[s].actor[k] : NO_ACTOR;
  }
  EHD int findPair(uint32_t a0, uint32_t a1) const {
    uint32_t a, b;
    pairKey(a0, a1, a, b);
    for (uint32_t i = 0; i < nMap; ++i)
      if (map[i].a == a && map[i].b == b) return int(i);
    return -1;
  }

  // PxD6JointCreate(두 행위자가 장면에 있을 때). 반환 = 칸 번호. 칸의 id 가 writeback·getGPUIndex 번호.
  // actor 가 NO_ACTOR 면 세계(ScScene 의 정적 닻), dynamic=0 이면 정적 행위자.
  EHD uint32_t addConstraint(uint32_t actor0, bool dyn0, uint32_t actor1, bool dyn1, uint16_t flags) {
    uint32_t s = 0;  // 칸 번호는 우리 것(뜻 없음): 빈 칸 재사용
    while (s < nSlots && slot[s].alive) ++s;
    if (s == nSlots) {
      if (nSlots >= MAXC) { overflow = true; return 0; }
      nSlots++;
    }
    ConstraintSlot& c = slot[s];
    c.id = ids.createID();  // ScConstraintSim.cpp:63
    c.actor[0] = actor0; c.actor[1] = actor1;
    c.dynamic[0] = uint8_t(dyn0 && actor0 != NO_ACTOR);
    c.dynamic[1] = uint8_t(dyn1 && actor1 != NO_ACTOR);
    c.alive = 1;
    c.flags = flags;
    c.edge = 0xffffffffu;
    // ConstraintInteraction 생성자: onConstraintAttach (몸체 0, 1 순서) -> 섬 간선
    for (int k = 0; k < 2; ++k)
      if (c.dynamic[k]) { body[c.actor[k]].numConstraints++; body[c.actor[k]].numCountedInteractions++; }
    pushEv(EV_EDGE_ADD, s);
    // Sc::Scene::addConstraintToMap: 이미 있으면 그대로 (PxHashMap::insert)
    if (findPair(actor0, actor1) < 0) {
      if (nMap >= MAXC) overflow = true;
      else { pairKey(actor0, actor1, map[nMap].a, map[nMap].b); map[nMap].slot = s; nMap++; }
    }
    return s;
  }

  // PxJoint::release (장면 안). 순서: 지도에서 지움 -> 간선 제거 -> 몸체 detach -> 번호 반납(미룸)
  EHD void removeConstraint(uint32_t s) {
    ConstraintSlot& c = slot[s];
    if (!c.alive) return;
    const int m = findPair(c.actor[0], c.actor[1]);  // removeConstraintFromMap: 그 쌍 칸을 지움 (다른 조인트가 넣은 칸이어도)
    if (m >= 0) {  // 배열 순서는 뜻이 없다 (해시 지도) — 마지막 칸으로 메움
      map[m] = map[nMap - 1];
      nMap--;
    }
    pushEv(EV_EDGE_REMOVE, s);
    for (int k = 0; k < 2; ++k)
      if (c.dynamic[k]) { body[c.actor[k]].numCountedInteractions--; body[c.actor[k]].numConstraints--; }
    ids.releaseID(c.id);
    c.alive = 0;
  }

  // fetchResults 끝
  EHD void postReportsCleanup() { ids.processPendingReleases(); }

  // ScFiltering.cpp:299 filterJointedBodies — true 면 이 쌍은 충돌하지 않는다(쌍 생성 때만 본다)
  EHD bool jointedPairFiltered(uint32_t actor0, bool dyn0, uint32_t actor1, bool dyn1) const {
    const bool h0 = dyn0 && body[actor0].numConstraints > 0, h1 = dyn1 && body[actor1].numConstraints > 0;
    if (!h0 && !h1) return false;
    const int m = findPair(actor0, actor1);
    return m >= 0 ? !(slot[map[m].slot].flags & (1u << 3)) : false;  // PxConstraintFlag::eCOLLISION_ENABLED
  }
  EHD bool hasConstraints(uint32_t b) const { return body[b].numConstraints > 0; }
};

}  // namespace jnt
}  // namespace eng
