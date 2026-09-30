// Sc 입력 조각 상태 저장·적재 (문서 15.3 닫힌 고리 4단 E1, 리드): 장면 파일에서 env 를 세울 때 ScScene(sc_scene.h)을 경계 그대로 되살린다.
// 담는 것: 요소·행위자 번호 표(Cm::IDPool + ObjectIDTracker 대기 목록), 행위자 기록(종류·번호·섬 노드·자세·요소 차례), 요소별 칸
// (변환 캐시·경계 상자·접촉 거리·캐시 플래그·넓은 단계 안인가·모양 입력), 넓은 단계 바뀜 비트.
// 모양 기하는 담지 않고 장면 파일 모양 번호(sceneShape)로 틀(SceneShared)에서 가져온다(볼록 덩어리 주소가 틀에 걸려 있어서).
// PhysX 없이 쓴다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/scene/sc_scene.h"

namespace eng {
namespace scene {

static_assert(sizeof(px::PxTransform) == 28, "PxTransform = 쿼터니언 4 + 위치 3");
static_assert(sizeof(px::PxBounds3) == 24, "PxBounds3 = 최소 3 + 최대 3");

struct ScStateTracker {
  uint32_t cur = 0;
  std::vector<uint32_t> freeIds, pending;
};
struct ScStateActor {
  uint32_t kind, actorID, alive, elemStart, elemCount;
  uint8_t idtBody2Actor, kinematic, forcedKineNotif, pad;
  uint64_t node;
  float pose[7], body2Actor[7];
};
struct ScStateShape {
  int32_t actor;
  uint32_t alive, sceneShape, inBp, shapeFlags;
  float contactOffset;
  uint8_t idtShape, pad[3];
  float localPose[7], cache[7], bounds[6], contactDist;
  uint32_t cacheFlags;
};
struct ScState {
  bool valid = false;
  ScStateTracker elementIds, actorIds;
  std::vector<ScStateActor> actors;
  std::vector<uint32_t> elems;  // 행위자별 요소 차례를 이어 붙임
  std::vector<ScStateShape> shapes;  // 요소 번호별
  std::vector<uint32_t> changed;
};

inline ScStateTracker scSaveTracker(const IdTracker& t) {
  ScStateTracker o;
  o.cur = t.pool.cur;
  o.freeIds = t.pool.freeIds;
  o.pending = t.pending;
  return o;
}
inline void scLoadTracker(IdTracker& t, const ScStateTracker& o) {
  t.pool.cur = o.cur;
  t.pool.freeIds = o.freeIds;
  t.pending = o.pending;
  t.deleted.clear();
  for (uint32_t id : o.pending) {
    if (t.deleted.size() <= id) t.deleted.resize(id + 1, 0);
    t.deleted[id] = 1;
  }
}

// elemToSceneShape: 요소 번호 -> 장면 파일 모양 번호 (없으면 kNone 과 같은 0xffffffff)
inline ScState scSave(const ScScene& S, const std::vector<uint32_t>& elemToSceneShape) {
  ScState o;
  o.valid = true;
  o.elementIds = scSaveTracker(S.elementIds);
  o.actorIds = scSaveTracker(S.actorIds);
  for (const ScActorRec& a : S.actors) {
    ScStateActor r{};
    r.kind = a.kind;
    r.actorID = a.actorID;
    r.alive = a.alive;
    r.elemStart = uint32_t(o.elems.size());
    r.elemCount = uint32_t(a.elements.size());
    r.idtBody2Actor = a.idtBody2Actor;
    r.kinematic = a.kinematic;
    r.forcedKineNotif = a.forcedKineNotif;
    r.node = a.node;
    memcpy(r.pose, &a.pose, 28);
    memcpy(r.body2Actor, &a.body2Actor, 28);
    o.elems.insert(o.elems.end(), a.elements.begin(), a.elements.end());
    o.actors.push_back(r);
  }
  for (uint32_t e = 0; e < S.shapes.size(); ++e) {
    const ScShapeRec& s = S.shapes[e];
    ScStateShape r{};
    r.actor = s.actor;
    r.alive = s.alive;
    r.sceneShape = e < elemToSceneShape.size() ? elemToSceneShape[e] : 0xffffffffu;
    r.inBp = s.inBp;
    r.shapeFlags = s.in.shapeFlags;
    r.contactOffset = s.in.contactOffset;
    r.idtShape = s.in.idtShape;
    memcpy(r.localPose, &s.in.localPose, 28);
    memcpy(r.cache, &S.cache[e], 28);
    memcpy(r.bounds, &S.bounds[e], 24);
    r.contactDist = S.contactDist[e];
    r.cacheFlags = S.cacheFlags[e];
    o.shapes.push_back(r);
  }
  o.changed = S.changed;
  return o;
}

// geomOfSceneShape(k) -> 그 장면 모양의 기하 (주소가 걸린 것). 반환 = 기하를 못 찾은 살아 있는 요소 수
template <class GeomOf>
inline uint32_t scLoad(ScScene& S, const ScState& o, GeomOf geomOfSceneShape) {
  uint32_t missing = 0;
  S = ScScene{};
  scLoadTracker(S.elementIds, o.elementIds);
  scLoadTracker(S.actorIds, o.actorIds);
  for (const ScStateActor& r : o.actors) {
    ScActorRec a;
    a.kind = r.kind;
    a.actorID = r.actorID;
    a.alive = uint8_t(r.alive);
    a.idtBody2Actor = r.idtBody2Actor;
    a.kinematic = r.kinematic;
    a.forcedKineNotif = r.forcedKineNotif;
    a.node = r.node;
    memcpy(&a.pose, r.pose, 28);
    memcpy(&a.body2Actor, r.body2Actor, 28);
    a.elements.assign(o.elems.begin() + r.elemStart, o.elems.begin() + r.elemStart + r.elemCount);
    S.actors.push_back(a);
  }
  if (!o.shapes.empty()) S.grow(uint32_t(o.shapes.size() - 1));
  for (uint32_t e = 0; e < o.shapes.size(); ++e) {
    const ScStateShape& r = o.shapes[e];
    ScShapeRec& s = S.shapes[e];
    s.actor = r.actor;
    s.alive = uint8_t(r.alive);
    s.inBp = uint8_t(r.inBp);
    s.in.shapeFlags = r.shapeFlags;
    s.in.contactOffset = r.contactOffset;
    s.in.idtShape = r.idtShape;
    memcpy(&s.in.localPose, r.localPose, 28);
    const contact::ShapeGeom* g = r.sceneShape != 0xffffffffu ? geomOfSceneShape(r.sceneShape) : nullptr;
    if (g) s.in.geom = *g;
    else if (r.alive) ++missing;
    memcpy(&S.cache[e], r.cache, 28);
    memcpy(&S.bounds[e], r.bounds, 24);
    S.contactDist[e] = r.contactDist;
    S.cacheFlags[e] = r.cacheFlags;
  }
  S.changed = o.changed;
  return missing;
}

// 파일 절 (장면 파일 v8 뒤에 붙는 선택 절, 머리 "SCSTATE1")
namespace detail_sc {
template <class T>
inline bool wv(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  return fwrite(&n, 8, 1, f) == 1 && (n == 0 || fwrite(v.data(), sizeof(T), size_t(n), f) == size_t(n));
}
template <class T>
inline bool rv(FILE* f, std::vector<T>& v) {
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1 || n >= (1ull << 32)) return false;
  v.resize(size_t(n));
  return n == 0 || fread(v.data(), sizeof(T), size_t(n), f) == size_t(n);
}
inline bool wt(FILE* f, const ScStateTracker& t) { return fwrite(&t.cur, 4, 1, f) == 1 && wv(f, t.freeIds) && wv(f, t.pending); }
inline bool rt(FILE* f, ScStateTracker& t) { return fread(&t.cur, 4, 1, f) == 1 && rv(f, t.freeIds) && rv(f, t.pending); }
}  // namespace detail_sc
constexpr char kScStateMagic[8] = {'S', 'C', 'S', 'T', 'A', 'T', 'E', '1'};
inline bool writeScState(FILE* f, const ScState& s) {
  using namespace detail_sc;
  const uint32_t sizes[2] = {uint32_t(sizeof(ScStateActor)), uint32_t(sizeof(ScStateShape))};
  return fwrite(kScStateMagic, 1, 8, f) == 8 && fwrite(sizes, 4, 2, f) == 2 && wt(f, s.elementIds) && wt(f, s.actorIds) && wv(f, s.actors) &&
         wv(f, s.elems) && wv(f, s.shapes) && wv(f, s.changed);
}
// 머리가 없으면(옛 파일) false 이고 s.valid = false
inline bool readScState(FILE* f, ScState& s) {
  using namespace detail_sc;
  s = ScState{};
  char m[8];
  if (fread(m, 1, 8, f) != 8 || memcmp(m, kScStateMagic, 8)) return false;
  uint32_t sizes[2];
  if (fread(sizes, 4, 2, f) != 2 || sizes[0] != sizeof(ScStateActor) || sizes[1] != sizeof(ScStateShape)) return false;
  s.valid = rt(f, s.elementIds) && rt(f, s.actorIds) && rv(f, s.actors) && rv(f, s.elems) && rv(f, s.shapes) && rv(f, s.changed);
  return s.valid;
}

}  // namespace scene
}  // namespace eng
