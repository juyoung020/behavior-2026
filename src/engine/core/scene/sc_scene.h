// Sc 장면 층: 판 도중 강체 추가·삭제·다시 넣기 API (문서 15.3·20.4, 리드).
// PhysX 가 같은 일을 할 때의 번호 매김과 모듈 호출 순서를 그대로 따른다. 원본:
//   추가  Sc::Scene::addBody/addStatic (ScScene.cpp:2335·2301, 한꺼번에 넣기 2483·2498 도 같은 순서)
//         = ActorSim 생성(행위자 번호 ScActorSim.cpp:73) -> [동적] 섬 addNode(깨어 있음, 운동학, 강체) 후 안 깨어 있으면 deactivateNode (ScBodySim.cpp:86-118)
//         -> 모양마다(행위자 모양 순서) ShapeSim 생성 = 요소 번호(ScElementSim.h:111) -> 변환 캐시·경계 상자(sc_inputs.h)
//         -> 넓은 단계면 addBounds(요소, 접촉 거리 = contactOffset, 무리 = ((행위자 번호+1)<<3)|종류, 트리거면 eTRIGGER) + 접촉 거리 칸
//   삭제  Sc::Scene::removeBody/removeStatic (2367·2314) = 모양마다(요소 순서) removeBounds -> 요소 번호 풀기(대기)
//         -> [동적] 섬 removeNode -> 행위자 번호 풀기(대기). 쌍 관리층에는 사라진 모양이 PPO_SHAPE_REMOVE 로 간다(요소 오름차순, pairs_log.h).
//   다시 넣기 ShapeSimBase::reinsertBroadPhase (ScShapeSimBase.cpp:106, resetFiltering 등)
//         = removeBounds -> 넣기 대기 중이 아니었으면 요소 번호 풀고 새로 받음 -> 칸 다시 채움 -> addBounds
//   플래그  ShapeSimBase::onFlagChange (ScShapeSimBase.cpp:250) = 시뮬레이션·트리거 플래그로 넓은 단계에 들고 남 (setShapeFlags)
//   번호 풀기는 스텝 끝(postReportsCleanup)에만 실제로 된다 -> endStep().
// 모듈(넓은 단계·섬 관리자)은 ScModules 로 부른다: 돌려주는 값(넣기 대기였나, 섬 노드 번호)이 다음 순서를 정하므로 호출 순서가 곧 PhysX 순서다.
// 검증: replay/g1_sc.cpp (G1_SC_EDIT=1) 가 PhysX Sc::Scene::addBody 등을 가로채 같은 입력으로 이 API 를 돌리고
//       번호·모듈 호출 열·칸 값을 PhysX 와 비교한다 (chop_slice0: 통나무 삭제 1·반쪽 추가 2·다시 넣기 1).
// 아직: 관절체 링크·집합체(aggregate) 안 행위자·운동학 전환은 받지 않는다(unsupported 로 알림).
#pragma once
#include <cstdint>
#include <vector>

#include "core/contact/narrowphase.h"
#include "core/scene/bp_log.h"
#include "core/scene/id_pool.h"
#include "core/scene/sc_inputs.h"

namespace eng {
namespace scene {

// PxShapeFlag
constexpr uint32_t kShapeSimulation = 1u << 0, kShapeSceneQuery = 1u << 1, kShapeTrigger = 1u << 2;
// Bp::FilterType (BpFiltering.h:60)
enum BpFilterType : uint32_t { kBpStatic = 0, kBpKinematic = 1, kBpDynamic = 2 };
constexpr uint32_t kBpTypeShift = 3;  // BP_FILTERING_TYPE_SHIFT_BIT

struct ScShapeIn {
  contact::ShapeGeom geom;
  px::PxTransform localPose;
  float contactOffset;
  uint32_t shapeFlags;  // PxShapeFlags
  uint8_t idtShape, pad[3];
};
struct ScActorIn {
  uint32_t kind;          // kStatic / kDynamic (scene_file.h ActorKind)
  px::PxTransform pose;   // 정적: actor2World, 동적: body2World
  px::PxTransform body2Actor;
  uint8_t idtBody2Actor, kinematic, forcedKineNotif, awake;  // awake = wakeCounter>0 또는 속도≠0 (ScBodySim.cpp:80)
  std::vector<ScShapeIn> shapes;
};

// 모듈 호출. 돌려주는 값은 모듈이 정한다 (층 1·2 = 우리 BpRuntime·섬 관리자, 검증 = PhysX 기록)
struct ScModules {
  virtual ~ScModules() {}
  virtual bool bpAdd(const BpOp& o) = 0;       // AABBManager::addBounds
  virtual bool bpRemove(uint32_t index) = 0;   // AABBManager::removeBounds -> 넣기 대기(추가 맵)에 있었나
  virtual void bpReserve(uint32_t index) { (void)index; }  // reserveSpaceForBounds (넓은 단계 밖 모양)
  virtual bool bpMarkedForRemove(uint32_t index) { (void)index; return false; }  // AABBManager::isMarkedForRemove (이번 창에 빼기 대기)
  virtual uint64_t islandAddNode(bool awake, bool kine) = 0;  // SimpleIslandManager::addNode(.., eRIGID_BODY_TYPE, ..) -> PxNodeIndex
  virtual void islandDeactivateNode(uint64_t node) = 0;
  virtual void islandRemoveNode(uint64_t node) = 0;
};

struct ScShapeRec {
  int32_t actor = -1;
  ScShapeIn in;
  uint8_t alive = 0, inBp = 0, pad[2];
};
struct ScActorRec {
  uint32_t kind = 0, actorID = 0;
  px::PxTransform pose, body2Actor;
  uint8_t idtBody2Actor = 1, kinematic = 0, forcedKineNotif = 0, alive = 0;
  uint64_t node = ~0ull;
  std::vector<uint32_t> elements;  // 모양 순서 (PxRigidActor::getShapes)
};

struct ScScene {
  IdTracker elementIds, actorIds;
  std::vector<ScShapeRec> shapes;  // 요소 번호별
  std::vector<ScActorRec> actors;  // 손잡이별 (손잡이는 우리 쪽 번호, PhysX 행위자 번호 = actorID)
  std::vector<px::PxTransform> cache;
  std::vector<px::PxBounds3> bounds;
  std::vector<float> contactDist;
  uint32_t unsupported = 0;

  static uint32_t bpGroup(const ScActorRec& a) {  // Bp::getFilterGroup (BpFiltering.h:80)
    if (a.kind == 0) return 0;  // eSTATICS
    const uint32_t type = (a.kinematic && !a.forcedKineNotif) ? kBpKinematic : kBpDynamic;
    return ((a.actorID + 1u) << kBpTypeShift) | type;
  }
  static bool inBroadPhase(uint32_t flags) { return (flags & (kShapeSimulation | kShapeTrigger)) != 0; }  // isBroadPhase (ScShapeSimBase)

  void grow(uint32_t e) {
    if (shapes.size() <= e) {
      shapes.resize(e + 1);
      cache.resize(e + 1);
      bounds.resize(e + 1);
      contactDist.resize(e + 1, 0.f);
    }
  }
  // initSubsystemsDependingOnElementID (ScShapeSimBase.cpp:176)
  void initShape(uint32_t e, ScModules& m) {
    ScShapeRec& s = shapes[e];
    const ScActorRec& a = actors[size_t(s.actor)];
    ShapePoseIn pin{};
    pin.shape2Actor = s.in.localPose;
    pin.isStatic = a.kind == 0;
    pin.idtShape = s.in.idtShape;
    pin.idtBody2Actor = a.idtBody2Actor;
    updateShapeCached(pin, s.in.geom.get(), a.pose, a.body2Actor, cache[e], bounds[e]);
    if (inBroadPhase(s.in.shapeFlags)) {
      addToBp(e, m);
    } else {
      m.bpReserve(e);
      s.inBp = 0;
    }
    contactDist[e] = s.in.contactOffset;  // Scene::updateContactDistance
  }
  // internalAddToBroadPhase (ScShapeSimBase.cpp:160) = addToAABBMgr(contactOffset, 무리, 종류)
  void addToBp(uint32_t e, ScModules& m) {
    ScShapeRec& s = shapes[e];
    BpOp o{};
    o.type = BP_ADD;
    o.index = e;
    o.contactDistance = s.in.contactOffset;
    o.group = bpGroup(actors[size_t(s.actor)]);
    o.agg = 0xffffffffu;  // 집합체 밖
    o.volumeType = (s.in.shapeFlags & kShapeTrigger) ? 1u : 0u;
    o.env = 0xffffffffu;
    o.result = m.bpAdd(o) ? 1u : 0u;
    s.inBp = 1;
  }
  // 모양 플래그 바꾸기 (ShapeSimBase::onFlagChange, ScShapeSimBase.cpp:250): 넓은 단계 들고 남, 트리거 바뀜은 다시 넣기
  void setShapeFlags(uint32_t e, uint32_t newFlags, ScModules& m) {
    ScShapeRec& s = shapes[e];
    const uint32_t oldFlags = s.in.shapeFlags;
    s.in.shapeFlags = newFlags;
    const bool oldBp = inBroadPhase(oldFlags), newBp = inBroadPhase(newFlags);
    if (oldBp != newBp) {
      if (!oldBp && newBp) {
        if ((newFlags & kShapeTrigger) && m.bpMarkedForRemove(e))
          reinsertShape(e, m);
        else
          addToBp(e, m);
      } else {
        if (s.inBp) m.bpRemove(e);  // internalRemoveFromBroadPhase
        s.inBp = 0;
      }
    } else if (((oldFlags ^ newFlags) & kShapeTrigger) != 0) {
      reinsertShape(e, m);
    }
  }

  // 추가: 돌려주는 값 = 손잡이
  int32_t addActor(const ScActorIn& in, ScModules& m) {
    const int32_t h = int32_t(actors.size());
    actors.emplace_back();
    ScActorRec& a = actors.back();
    a.kind = in.kind;
    a.pose = in.pose;
    a.body2Actor = in.body2Actor;
    a.idtBody2Actor = in.kind == 0 ? 1 : in.idtBody2Actor;
    a.kinematic = in.kinematic;
    a.forcedKineNotif = in.forcedKineNotif;
    a.alive = 1;
    a.actorID = actorIds.create();
    if (in.kind != 0) {
      a.node = m.islandAddNode(in.awake != 0, in.kinematic != 0);
      if (!in.awake) m.islandDeactivateNode(a.node);
    }
    for (const ScShapeIn& si : in.shapes) {
      const uint32_t e = elementIds.create();
      grow(e);
      ScShapeRec& s = shapes[e];
      s.actor = h;
      s.in = si;
      s.alive = 1;
      actors[size_t(h)].elements.push_back(e);
      initShape(e, m);
    }
    return h;
  }

  // 장면 안 행위자에 모양 붙이기 (RigidCore::addShapeToScene -> Scene::addShape_, ScScene.cpp:2410): ShapeSim 생성 = 요소 번호 + 칸 + 넓은 단계
  uint32_t attachShape(int32_t h, const ScShapeIn& si, ScModules& m) {
    const uint32_t e = elementIds.create();
    grow(e);
    ScShapeRec& s = shapes[e];
    s.actor = h;
    s.in = si;
    s.alive = 1;
    actors[size_t(h)].elements.push_back(e);  // ElementSim 생성이 행위자 모양 표 끝에 넣음 (ScElementSim.cpp:106)
    initShape(e, m);
    return e;
  }
  // 모양 떼기 (RigidCore::removeShapeFromScene -> Scene::removeShape_, ScScene.cpp:2422): 넓은 단계 빼기 -> ShapeSim 소멸(번호 풀기, 표에서 끝 것과 바꿔 빼기)
  void detachShape(uint32_t e, ScModules& m) {
    ScShapeRec& s = shapes[e];
    if (s.inBp) m.bpRemove(e);
    s.inBp = 0;
    std::vector<uint32_t>& el = actors[size_t(s.actor)].elements;
    for (size_t i = 0; i < el.size(); ++i)
      if (el[i] == e) {
        el[i] = el.back();  // replaceWithLast (ScElementSim.cpp:125)
        el.pop_back();
        break;
      }
    s.alive = 0;
    s.actor = -1;
    elementIds.release(e);
  }
  // 거르기 다시 (RigidCore::onShapeChange eRESET_FILTERING -> ShapeSimBase::onResetFiltering, ScShapeSimBase.cpp:83)
  void resetFiltering(uint32_t e, ScModules& m) {
    if (shapes[e].inBp) reinsertShape(e, m);
  }
  // 접촉 거리 바꾸기 (eCONTACTOFFSET -> onContactOffsetChange): 넓은 단계에 있으면 접촉 거리 칸을 고침 (AABBManager::setContactDistance)
  void setContactOffset(uint32_t e, float off) {
    shapes[e].in.contactOffset = off;
    if (shapes[e].inBp) contactDist[e] = off;
  }

  // 삭제 (ShapeSim 소멸 -> BodySim 소멸 순)
  void removeActor(int32_t h, ScModules& m) {
    ScActorRec& a = actors[size_t(h)];
    for (uint32_t e : a.elements) {
      ScShapeRec& s = shapes[e];
      if (s.inBp) m.bpRemove(e);
      s.inBp = 0;
      s.alive = 0;
      s.actor = -1;
      elementIds.release(e);
    }
    if (a.kind != 0) m.islandRemoveNode(a.node);
    actorIds.release(a.actorID);
    a.alive = 0;
    a.elements.clear();
  }

  // 다시 넣기 (resetFiltering 등). 넣기 대기 중이던 모양은 번호를 그대로 둔다
  void reinsertShape(uint32_t e, ScModules& m) {
    ScShapeRec s = shapes[e];
    bool pending = false;
    if (s.inBp) pending = m.bpRemove(e);
    shapes[e].inBp = 0;
    uint32_t ne = e;
    if (!pending) {
      elementIds.release(e);
      ne = elementIds.create();
      grow(ne);
      if (ne != e) {
        shapes[e].alive = 0;
        shapes[e].inBp = 0;
        shapes[e].actor = -1;
        std::vector<uint32_t>& el = actors[size_t(s.actor)].elements;
        for (uint32_t& x : el)
          if (x == e) x = ne;
      }
    }
    shapes[ne] = s;
    shapes[ne].inBp = 0;
    initShape(ne, m);
  }
  void reinsertActor(int32_t h, ScModules& m) {
    const std::vector<uint32_t> el = actors[size_t(h)].elements;
    for (uint32_t e : el) reinsertShape(e, m);
  }

  // 스텝 끝 (postReportsCleanup)
  void endStep() {
    elementIds.endStep();
    actorIds.endStep();
  }
};

}  // namespace scene
}  // namespace eng
