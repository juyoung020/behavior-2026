// base 창 관절체 쪽 엔진 입력 (particles BaseArtApi, 문서 12.3 base 창 약속) 을 env 상태 위에 — engine-solver-art.
// 관절체 = EnvSolveImpl::arts[a] (a = 장면 파일 관절체 번호 = BaseObject::art), 섬 노드 = artOfNode 가 a 인 노드, 링크 노드 원값 = 노드 | ((LL<<1)|1)<<32.
// PhysX 가 같은 API 안에서 하는 차례 그대로:
//   applyCache (NpArticulationReducedCoordinate.cpp:153): Dy applyCache(값 쓰기·jcalc·링크 순간이동·링크 속도, art::applyCache 의 앞부분)
//     -> [위치·뿌리 자세면] 링크마다 scSetBody2World (BodyCore::setBody2World: Sc 칸 갱신)
//     -> wakeUpInternal(force, autowake=true) (:1137): 자고 있거나 깸 카운터 < 0.4 면 링크마다 scWakeUpInternal(wc) 다음 관절체 코어 wakeUp(wc).
//   링크 scWakeUpInternal(wc) = BodyCore::wakeUp = setWakeCounter(wc, force) -> BodySim::wakeUp(setActive(true), activateNode) -> postSetWakeCounter(activateNode)
//   putToSleep (:1177): 링크마다 BodyCore::putToSleep (속도 0, 깸 0, BodySim::putToSleep: setActive(false), putNodeToSleep) -> 코어 깸 0
//   wakeUp (:1161): 링크마다 scWakeUpInternal(0.4) -> 코어 wakeUp(0.4)
//   isSleeping = !뿌리 링크 BodySim::isActive (ScArticulationSim.cpp:496)
// Sc 층 상태 = 리드 env 표: 행위자 활성(E.active), 깸 카운터 표(E.wake: 관절체·링크), 섬(E.isl). 코어 값(Articulation 의 깸 카운터)도 같이 맞춘다.
// 아직 안 하는 것 (강체 EnvBodyApi 와 같음): BodySim::setActive(false) 의 상호작용 재우기(deactivateInteractions), lastCCDTransform(링크 칸 없음).
// 호스트 전용.
#pragma once
#include <vector>

#include "core/articulation/articulation.h"
#include "core/particles/env_base_api.h"
#include "core/scene/env_solve.h"

namespace eng {
namespace scene {

struct EnvArtApi : public particles::BaseArtApi {
  EnvStep& E;
  EnvSolveImpl& S;
  uint64_t unknown = 0, reqActivate = 0, reqSleep = 0;

  EnvArtApi(EnvStep& e, EnvSolveImpl& s) : E(e), S(s) {}

  art::Articulation* artOf(int32_t a) { return a >= 0 && size_t(a) < S.arts.size() ? &S.arts[size_t(a)] : nullptr; }
  uint32_t nodeOf(int32_t a) const {
    for (uint32_t n = 0; n < S.artOfNode.size(); ++n)
      if (S.artOfNode[n] == a) return n;
    return ig::INVALID_NODE;
  }
  static uint64_t linkNode(uint32_t node, uint32_t ll) { return uint64_t(node) | (uint64_t((ll << 1) | 1u) << 32); }
  // 행위자 활성 (BodySim::isActive) 표시: 이 링크 노드의 행위자
  void setLinkActive(uint64_t ln, bool on) {
    if (on) {
      E.live.markNodeActive(ln);
      return;
    }
    const ss::ScPairs& P = E.C->S->pairs;
    for (uint32_t k = 0; k < P.actors.size() && k < E.active.size(); ++k)
      if (P.actors[k].nodeIndex == ln) E.active[k] = 0;
  }
  bool linkActive(uint32_t node, uint32_t ll) const {
    const uint64_t ln = linkNode(node, ll);
    const ss::ScPairs& P = E.C->S->pairs;
    for (uint32_t k = 0; k < P.actors.size() && k < E.active.size(); ++k)
      if (P.actors[k].nodeIndex == ln) return E.active[k] != 0;
    const ig::IslandSim& A = E.isl->M.accurate;  // 모양 없는 링크: 섬 노드 활성으로
    return node < A.nodes.size && (A.nodes.d[node].flags & ig::N_ACTIVE) != 0;
  }
  // Sc 깸 카운터 표
  void setLinkWc(uint32_t node, uint32_t ll, float wc) {
    if (HostBodyWake* w = E.wake.body(linkNode(node, ll))) w->wc = wc;
  }
  void setArtWc(uint32_t node, float wc) {
    if (HostArtWake* w = E.wake.art(node)) w->wc = wc;
  }
  // 링크 BodyCore::wakeUp(wc) (setWakeCounter(wc, true) -> BodySim::wakeUp + postSetWakeCounter)
  void linkWakeUp(art::Articulation& A, uint32_t node, uint32_t ll, float wc) {
    A.bodies[ll].wakeCounter = wc;
    setLinkWc(node, ll, wc);
    setLinkActive(linkNode(node, ll), true);
    ig::activateNode(E.isl->M, node);  // notifyWakeUp
    ig::activateNode(E.isl->M, node);  // postSetWakeCounter -> notifyNotReadyForSleeping (이미 활성/활성 중이면 아무것도 안 함)
    ++reqActivate;
  }
  // NpArticulationReducedCoordinate::wakeUpInternal(force, autowake = true) (:1137). 순서: 링크 (생성 순서 = mArticulationLinks) -> 코어
  void wakeUpInternal(art::Articulation& A, uint32_t node, bool sleepingBefore) {
    float wc = A.wakeCounter;
    bool needs = sleepingBefore;
    if (wc < kWakeReset) {
      wc = kWakeReset;
      needs = true;
    }
    if (!needs) return;
    for (uint32_t c = 0; c < A.nLinks; ++c) linkWakeUp(A, node, A.ll[c], wc);
    A.wakeCounter = wc;  // ArticulationCore::wakeUp
    setArtWc(node, wc);
  }
  // Sc 칸 (BodyCore::setBody2World -> 모양 자세 캐시): 링크 행위자마다
  void linkPosesToSc(const art::Articulation& A, uint32_t node) {
    ScScene& sc = *E.sc;
    for (uint32_t c = 0; c < A.nLinks; ++c) {
      const uint32_t ll = A.ll[c];
      const uint64_t ln = linkNode(node, ll);
      for (size_t h = 0; h < sc.actors.size(); ++h) {
        const ScActorRec& r = sc.actors[h];
        if (!r.alive || r.node != ln) continue;
        const Tf& b = A.bodies[ll].body2World;
        sc.updateActorCached(int32_t(h), px::PxTransform(px::PxVec3(b.p.x, b.p.y, b.p.z), px::PxQuat(b.q.x, b.q.y, b.q.z, b.q.w)), r.body2Actor, false);
      }
    }
  }
  // applyCache 한 번 (flag = art::CF_*)
  void apply(int32_t a, const art::CacheIn& c, uint32_t flag) {
    art::Articulation* A = artOf(a);
    const uint32_t node = nodeOf(a);
    if (!A || node == ig::INVALID_NODE) {
      ++unknown;
      return;
    }
    const bool sleepingBefore = sleeping(a);
    // Dy applyCache 몫만: 깨우기는 아래 Sc 차례로 (art::applyCache 의 wakeUpInternal 은 autowake=false + awake=1 이면 아무것도 안 함)
    const uint8_t awake = A->awake;
    A->awake = 1;
    EnvFtz f;
    art::applyCache(*A, c, flag, false);
    A->awake = awake;
    if (flag & (art::CF_POSITION | art::CF_ROOT_TRANSFORM)) linkPosesToSc(*A, node);
    wakeUpInternal(*A, node, sleepingBefore);
  }

  // ---- BaseArtApi
  bool sleeping(int32_t a) override {
    const art::Articulation* A = artOf(a);
    const uint32_t node = nodeOf(a);
    if (!A || node == ig::INVALID_NODE) {
      ++unknown;
      return true;
    }
    return A->nLinks ? !linkActive(node, A->ll[0]) : true;
  }
  Tf rootPose(int32_t a) override {  // 뿌리 링크 getGlobalPose = body2World * body2Actor^-1
    const art::Articulation* A = artOf(a);
    if (!A || !A->nLinks) {
      ++unknown;
      return Tf{qid(), V3{0, 0, 0}};
    }
    const art::LinkBody& b = A->bodies[A->ll[0]];
    return b.body2World * inverse(b.body2Actor);
  }
  void rootVelocity(int32_t a, V3& lin, V3& ang) override {
    const art::Articulation* A = artOf(a);
    if (!A || !A->nLinks) {
      ++unknown;
      lin = ang = V3{0, 0, 0};
      return;
    }
    const art::LinkBody& b = A->bodies[A->ll[0]];
    lin = b.linVel;
    ang = b.angVel;
  }
  void jointState(int32_t a, std::vector<float>& pos, std::vector<float>& vel) override {
    const art::Articulation* A = artOf(a);
    if (!A) {
      ++unknown;
      pos.clear();
      vel.clear();
      return;
    }
    const float* jp = A->jointPosition;
    const float* jv = A->jointVelocity;
    pos.assign(jp, jp + A->dofs);
    vel.assign(jv, jv + A->dofs);
  }
  void applyRootTransform(int32_t a, const Tf& pose) override {
    art::CacheIn c{};
    c.rootTransform = pose;
    apply(a, c, art::CF_ROOT_TRANSFORM);
  }
  void applyRootVelocities(int32_t a, const V3& lin, const V3& ang) override {
    art::CacheIn c{};
    c.rootLinVel = lin;
    c.rootAngVel = ang;
    apply(a, c, art::CF_ROOT_VELOCITIES);
  }
  void applyJointPositions(int32_t a, const std::vector<float>& pos) override {
    const art::Articulation* A = artOf(a);
    if (!A || pos.size() < A->dofs) {
      ++unknown;
      return;
    }
    art::CacheIn c{};
    c.jointPosition = pos.data();
    apply(a, c, art::CF_POSITION);
  }
  void applyJointVelocities(int32_t a, const std::vector<float>& vel) override {
    const art::Articulation* A = artOf(a);
    if (!A || vel.size() < A->dofs) {
      ++unknown;
      return;
    }
    art::CacheIn c{};
    c.jointVelocity = vel.data();
    apply(a, c, art::CF_VELOCITY);
  }
  void putToSleep(int32_t a) override {
    art::Articulation* A = artOf(a);
    const uint32_t node = nodeOf(a);
    if (!A || node == ig::INVALID_NODE) {
      ++unknown;
      return;
    }
    for (uint32_t c = 0; c < A->nLinks; ++c) {  // 링크 BodyCore::putToSleep (생성 순서)
      const uint32_t ll = A->ll[c];
      art::LinkBody& b = A->bodies[ll];
      b.linVel = V3{0.0f, 0.0f, 0.0f};
      b.angVel = V3{0.0f, 0.0f, 0.0f};
      b.wakeCounter = 0.0f;  // setWakeCounter(0): 링크는 notifyReadyForSleeping 이 아무것도 안 함
      setLinkWc(node, ll, 0.0f);
      setLinkActive(linkNode(node, ll), false);  // BodySim::putToSleep -> setActive(false)
      ig::putNodeToSleep(E.isl->M, node);        // notifyPutToSleep
      ++reqSleep;
    }
    A->wakeCounter = 0.0f;  // ArticulationCore::putToSleep
    setArtWc(node, 0.0f);
  }
  void wakeUp(int32_t a) override {
    art::Articulation* A = artOf(a);
    const uint32_t node = nodeOf(a);
    if (!A || node == ig::INVALID_NODE) {
      ++unknown;
      return;
    }
    for (uint32_t c = 0; c < A->nLinks; ++c) linkWakeUp(*A, node, A->ll[c], kWakeReset);
    A->wakeCounter = kWakeReset;
    setArtWc(node, kWakeReset);
  }
};

}  // namespace scene
}  // namespace eng
