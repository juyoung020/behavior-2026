// particles 편집 창의 몸체 API (core/particles/edit_window.h BodyApi) 를 env 상태 위에 (문서 12.3, 리드).
// 몸체 = EnvSolveImpl 의 Body 칸 (장면 파일 번호 + 판 도중 넣은 것), 손잡이 = ScScene 손잡이(행위자 기록 번호, 0 이상).
// 자세 쓰기는 PhysX 처럼 행위자 상호작용에 변환 바뀜을 알린다 (RigidSim::notifyShapesOfTransformChange -> notifyActorInteractionsOfTransformChange,
//   ScShapeSimBase.cpp:389): 겹침 상호작용마다 resetManagerCachedState, 잠든 동적이면 onShapeChangeWhileSleeping(잃은 닿음 목록).
// 정적 자세(setStaticPose, NpRigidStatic::setGlobalPose :70): 정규화 -> Sc StaticCore::setActor2World(모양 칸 dirty + 상호작용 알림)
//   -> updateShaderComs: 붙은 조인트의 c2b[정적 쪽] = getCom(정적).transformInv(국소 틀), getCom = 새 자세.getInverse() (ExtJoint.h:549).
// 모양 없는 동적 행위자(meta__*_link 등)는 Sc 행위자가 없다 -> 몸체 번호 손잡이 -2 - 몸체 번호 (bodyHandle 이 골라 줌). 이 손잡이는 Sc 칸이 없어
// setGlobalPose 에서 Sc 칸 갱신만 빠지고, 몸체·깸 카운터 표·섬 요청은 같다.
// 부수효과는 joints 모듈 core/joints/rigid_api.h (NpRigidDynamic 과 같은 입력 -> 같은 상태) 로 계산하고, 돌려받은 섬 요청(깨움·잠 준비·바로 재움)과
// Sc 층(깸 카운터 표·행위자 활성·Sc 칸)을 PhysX 가 같은 API 안에서 하는 차례로 반영한다:
//   setGlobalPose : BodyCore::setBody2World (Sc 칸은 다음 simulate 의 preRigidBodyNarrowPhase 에서 다시 계산 — 창 안에 읽는 이가 없어 바로 계산해도 같다)
//                   -> 깨우기 (autowake) -> activateNode
//   setLinearVelocity -> setAngularVelocity (각각 깨우기)
// 호스트 전용.
#pragma once
#include <algorithm>
#include <vector>

#include "core/joints/d6_joint.h"
#include "core/joints/rigid_api.h"
#include "core/particles/edit_window.h"
#include "core/scene/env_solve.h"

namespace eng {
namespace scene {

struct EnvBodyApi : public particles::BodyApi {
  EnvStep& E;
  EnvSolveImpl& S;
  std::vector<jnt::BodySimState> sim;  // 몸체 번호별 BodySim 쪽 API 상태 (속도 수정 누적 등)
  uint64_t reqActivate = 0, reqDeactivate = 0, reqSleep = 0, unknown = 0, staticWrites = 0, jointC2b = 0, interReset = 0, lostTouchAdd = 0, triggerSkip = 0;

  EnvBodyApi(EnvStep& e, EnvSolveImpl& s) : E(e), S(s) { sim.assign(S.bodies.size(), jnt::makeBodySimState()); }

  static int32_t directHandle(uint32_t body) { return -2 - int32_t(body); }
  // 몸체 번호 -> 손잡이: 그 몸체의 섬 노드를 가진 Sc 행위자가 있으면 그 손잡이, 없으면(모양 없는 행위자) 몸체 번호 손잡이
  int32_t bodyHandle(uint32_t body) const {
    for (size_t h = 0; h < E.sc->actors.size(); ++h) {
      const ScActorRec& a = E.sc->actors[h];
      if (a.alive && a.kind != 0 && S.bodyOf(uint32_t(a.node & 0xffffffffu)) == int32_t(body)) return int32_t(h);
    }
    return directHandle(body);
  }
  uint64_t nodeOf(int32_t h) const {
    if (h <= -2) {  // 몸체 번호 손잡이: 섬 노드 = 그 몸체를 가리키는 노드 (강체 노드 원값 = 노드 번호)
      const int32_t b = -2 - h;
      for (uint32_t n = 0; n < S.bodyOfNode.size(); ++n)
        if (S.bodyOfNode[n] == b) return uint64_t(n);
      return ~0ull;
    }
    return h >= 0 && size_t(h) < E.sc->actors.size() ? E.sc->actors[size_t(h)].node : ~0ull;
  }
  int32_t bodyOf(int32_t h) const {
    const uint64_t n = nodeOf(h);
    return n == ~0ull ? -1 : S.bodyOf(uint32_t(n & 0xffffffffu));
  }
  // BodySim::isActive = 정확 섬 노드 활성 (스텝 경계)
  bool nodeActive(uint64_t n) const {
    const uint32_t id = uint32_t(n & 0xffffffffu);
    const ig::IslandSim& A = E.isl->M.accurate;
    return id < A.nodes.size && (A.nodes.d[id].flags & ig::N_ACTIVE) != 0;
  }
  // 섬 요청 + Sc 층 깸 카운터 표
  void apply(int32_t h, uint32_t req, const Body& b) {
    const uint64_t n = nodeOf(h);
    if (HostBodyWake* w = E.wake.body(n)) w->wc = b.wakeCounter;
    const uint32_t id = uint32_t(n & 0xffffffffu);
    switch (req) {
      case jnt::REQ_ACTIVATE:
        ig::activateNode(E.isl->M, id);
        E.live.markNodeActive(n);
        ++reqActivate;
        break;
      case jnt::REQ_DEACTIVATE:
        ig::deactivateNode(E.isl->M, id);
        ++reqDeactivate;
        break;
      case jnt::REQ_SLEEP_NOW:
        ig::putNodeToSleep(E.isl->M, id);
        ++reqSleep;
        break;
      default: break;
    }
  }
  jnt::BodySimState& st(int32_t b, int32_t h) {
    if (sim.size() <= size_t(b)) sim.resize(size_t(b) + 1, jnt::makeBodySimState());
    jnt::BodySimState& s = sim[size_t(b)];
    s.active = nodeActive(nodeOf(h)) ? 1 : 0;
    if (h >= 0) {
      s.kinematic = E.sc->actors[size_t(h)].kinematic;
    } else {
      const HostBodyWake* w = E.wake.body(nodeOf(h));
      s.kinematic = w ? w->kinematic : 0;
    }
    return s;
  }

  // notifyActorInteractionsOfTransformChange (Sc 손잡이 h 의 쌍 관리층 행위자)
  void transformChanged(int32_t h, bool isDynamic, bool isAsleep) {
    if (h < 0 || size_t(h) >= E.pairsOfSc.size()) return;
    const int32_t pa = E.pairsOfSc[size_t(h)];
    ss::ScPairs& P = E.C->S->pairs;
    if (pa < 0 || size_t(pa) >= P.actors.size()) return;
    const ss::Actor& A = P.actors[size_t(pa)];
    std::vector<int32_t> its;
    for (uint32_t k = 0; k < A.interactions.size(); ++k) its.push_back(A.interactions[k]);
    for (int32_t it : its) {
      if (it < 0 || size_t(it) >= P.inters.size()) continue;
      const ss::Interaction& I = P.inters[size_t(it)];
      if (!I.alive) continue;
      if (I.type == ss::eOVERLAP) {
        P.userResetManagerCachedState(it);
        ++interReset;
        // ShapeInteraction::onShapeChangeWhileSleeping (ScShapeInteraction.cpp:1143)
        if (isAsleep && isDynamic && I.cm < 0 && !(I.siFlags & ss::SiFlag::TOUCH_KNOWN) && I.actor1 >= 0 &&
            P.actors[size_t(I.actor1)].isDynamicRigid() && !(I.siFlags & ss::SiFlag::CONTACTS_RESPONSE_DISABLED)) {
          E.live.addToLostTouchList(I.actor0, I.actor1);
          ++lostTouchAdd;
        }
      } else if (I.type == ss::eTRIGGER) {
        ++triggerSkip;  // TriggerInteraction::forceProcessingThisFrame (트리거는 아직 안 옮김)
      }
    }
  }
  // ---- 정적 (particles BodyApi 정적 입구)
  bool hasStaticApi() const override { return true; }
  Tf staticPose(int32_t h) override {
    if (h < 0 || size_t(h) >= E.sc->actors.size() || E.sc->actors[size_t(h)].kind != 0) {
      ++unknown;
      return Tf{qid(), V3{0, 0, 0}};
    }
    const px::PxTransform& t = E.sc->actors[size_t(h)].pose;
    return Tf{Q{t.q.x, t.q.y, t.q.z, t.q.w}, V3{t.p.x, t.p.y, t.p.z}};
  }
  void setStaticPose(int32_t h, const Tf& pose) override {
    if (h < 0 || size_t(h) >= E.sc->actors.size() || !E.sc->actors[size_t(h)].alive || E.sc->actors[size_t(h)].kind != 0) {
      ++unknown;
      return;
    }
    ++staticWrites;
    const Tf np = normalized(pose);
    ScActorRec& a = E.sc->actors[size_t(h)];
    a.pose = px::PxTransform(px::PxVec3(np.p.x, np.p.y, np.p.z), px::PxQuat(np.q.x, np.q.y, np.q.z, np.q.w));  // StaticCore::setActor2World
    for (uint32_t e : a.elements) E.sc->markDirty(e);                                                     // markBoundsForUpdate
    transformChanged(h, false, true);
    // updateShaderComs -> 조인트 c2b
    const Tf com = jnt::comOf(jnt::ACTOR_STATIC, np);  // getCom(정적) = getGlobalPose().getInverse()
    for (size_t k = 0; k < S.joints.size() && k < S.jointExt.size(); ++k) {
      if (S.jointDead.size() > k && S.jointDead[k]) continue;
      for (int s = 0; s < 2; ++s) {
        if (S.jointExt[k].staticID[s] != a.actorID) continue;
        const Tf& l = S.jointExt[k].local[s];
        S.joints[k].data.c2b[s] = jnt::toTf32(jnt::transformInvTf(com, l));  // Ext::Joint::onComShift (joints d6_joint.h 와 같은 식)
        ++jointC2b;
      }
    }
  }

  Tf actorPose(int32_t h) override {  // NpRigidDynamic::getGlobalPose = body2World * body2Actor^-1
    const int32_t b = bodyOf(h);
    if (b < 0) { ++unknown; return Tf{qid(), V3{0, 0, 0}}; }
    const Body& x = S.bodies[size_t(b)];
    return x.body2World * inverse(x.body2Actor);
  }
  void setActorPose(int32_t h, const Tf& pose) override {
    const int32_t b = bodyOf(h);
    if (b < 0) { ++unknown; return; }
    Body& x = S.bodies[size_t(b)];
    const uint32_t req = jnt::setGlobalPose(x, st(b, h), pose, true, true, kWakeReset);
    if (h >= 0) {  // Sc 칸 (모양 없는 행위자는 없음)
      const ScActorRec& a = E.sc->actors[size_t(h)];
      E.sc->updateActorCached(h, px::PxTransform(px::PxVec3(x.body2World.p.x, x.body2World.p.y, x.body2World.p.z),
                                                 px::PxQuat(x.body2World.q.x, x.body2World.q.y, x.body2World.q.z, x.body2World.q.w)),
                              a.body2Actor, false);
      transformChanged(h, true, !nodeActive(nodeOf(h)));  // BodySim::postBody2WorldChange -> notifyShapesOfTransformChange
    }
    apply(h, req, x);
  }
  void velocity(int32_t h, V3& lin, V3& ang) override {
    const int32_t b = bodyOf(h);
    if (b < 0) { ++unknown; lin = ang = V3{0, 0, 0}; return; }
    lin = S.bodies[size_t(b)].linVel;
    ang = S.bodies[size_t(b)].angVel;
  }
  void setVelocity(int32_t h, const V3& lin, const V3& ang) override {
    const int32_t b = bodyOf(h);
    if (b < 0) { ++unknown; return; }
    Body& x = S.bodies[size_t(b)];
    apply(h, jnt::setLinearVelocity(x, st(b, h), lin, true, true, kWakeReset), x);
    apply(h, jnt::setAngularVelocity(x, st(b, h), ang, true, true, kWakeReset), x);
  }
  // spawn_add 로 만든 몸체: 몸체 칸·섬 노드 -> 몸체 표·깸 카운터 표
  void bodyAdded(int32_t h, const Body& b) override {
    const uint64_t n = nodeOf(h);
    if (n == ~0ull) { ++unknown; return; }
    const uint32_t id = uint32_t(n & 0xffffffffu);
    const uint32_t bi = uint32_t(S.bodies.size());
    S.bodies.push_back(b);
    S.jointsOnBody.push_back(0);
    if (S.bodyOfNode.size() <= id) S.bodyOfNode.resize(size_t(id) + 1, -1);
    if (S.artOfNode.size() <= id) S.artOfNode.resize(size_t(id) + 1, -1);
    S.bodyOfNode[id] = int32_t(bi);
    sim.resize(S.bodies.size(), jnt::makeBodySimState());
    HostBodyWake w;
    w.node = n;
    w.wc = b.wakeCounter;
    w.solverWc = b.solverWakeCounter;
    w.kinematic = E.sc->actors[size_t(h)].kinematic;
    auto it = std::lower_bound(E.wake.bodies.begin(), E.wake.bodies.end(), w, [](const HostBodyWake& x, const HostBodyWake& y) { return x.node < y.node; });
    E.wake.bodies.insert(it, w);
  }
  void bodyRemoved(int32_t h) override {
    const uint64_t n = nodeOf(h);
    const uint32_t id = uint32_t(n & 0xffffffffu);
    if (n == ~0ull || id >= S.bodyOfNode.size()) { ++unknown; return; }
    S.bodyOfNode[id] = -1;  // 몸체 칸은 남긴다 (번호 안정)
    auto it = std::lower_bound(E.wake.bodies.begin(), E.wake.bodies.end(), n, [](const HostBodyWake& x, uint64_t v) { return x.node < v; });
    if (it != E.wake.bodies.end() && it->node == n) E.wake.bodies.erase(it);
  }
};

}  // namespace scene
}  // namespace eng
