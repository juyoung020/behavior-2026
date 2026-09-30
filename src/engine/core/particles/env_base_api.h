// base 창(base_state.h) 의 엔진 입력 BaseStateApi 를 env 상태 위에 (문서 12.3 base 창 약속).
//   강체 (particles): 리드 EnvBodyApi (core/scene/env_body_api.h) + joints rigid_api.h 의 putToSleep·wakeUp (NpRigidDynamic.cpp:504,517).
//     뿌리 자세 쓰기 = setGlobalPose(autowake), 선/각속도 쓰기 = 지금 선·각을 읽어 한쪽만 바꿔 setLinearVelocity → setAngularVelocity (텐서 뷰 set_velocities 6 칸),
//     psi 잠·깨움 = putToSleep / wakeUp(리셋값), is_sleeping = BodySim 비활성 (BodyCore::isSleeping).
//     운동학 전용 뿌리 자세 쓰기 = XForm(USD) 쓰기 → PhysX 정적 setGlobalPose (공식 OVD 확인) — 값은 dump 의 XForm 자세(BaseObject::xform), EnvBodyApi::setStaticPose.
//   관절체 (engine-solver-art 구현): BaseArtApi 로 넘긴다.
// 호스트 전용.
#pragma once
#include <cstdint>
#include <vector>

#include "core/particles/base_state.h"
#include "core/scene/env_body_api.h"

namespace eng {
namespace particles {

// 관절체 쪽 (engine-solver-art): PhysX 원본대로 applyCache·putToSleep/wakeUp·섬/링크 활성까지. a = env 관절체 번호 (BaseObject::art)
struct BaseArtApi {
  virtual ~BaseArtApi() {}
  virtual bool sleeping(int32_t a) = 0;                                        // PxArticulationReducedCoordinate::isSleeping
  virtual Tf rootPose(int32_t a) = 0;                                          // 뿌리 링크 getGlobalPose
  virtual void rootVelocity(int32_t a, V3& lin, V3& ang) = 0;                  // 뿌리 링크 선·각속도
  virtual void jointState(int32_t a, std::vector<float>& pos, std::vector<float>& vel) = 0;  // 캐시 jointPosition/jointVelocity (LL dof 순서)
  virtual void applyRootTransform(int32_t a, const Tf& pose) = 0;              // applyCache(eROOT_TRANSFORM)
  virtual void applyRootVelocities(int32_t a, const V3& lin, const V3& ang) = 0;  // applyCache(eROOT_VELOCITIES)
  virtual void applyJointPositions(int32_t a, const std::vector<float>& pos) = 0;   // applyCache(ePOSITION)
  virtual void applyJointVelocities(int32_t a, const std::vector<float>& vel) = 0;  // applyCache(eVELOCITY)
  virtual void putToSleep(int32_t a) = 0;                                      // NpArticulationReducedCoordinate::putToSleep (:1177)
  virtual void wakeUp(int32_t a) = 0;                                          // NpArticulationReducedCoordinate::wakeUp (:1161)
};

struct EnvBaseApi : BaseStateApi {
  scene::EnvBodyApi& B;
  BaseArtApi* art = nullptr;
  uint64_t artMissing = 0;  // 관절체 쪽이 아직 없어 건너뛴 호출

  explicit EnvBaseApi(scene::EnvBodyApi& b) : B(b) {}

  // ---- 강체
  bool rigidSleeping(int32_t h) override { return !B.nodeActive(B.nodeOf(h)); }
  void rigidSleep(int32_t h) override {
    const int32_t b = B.bodyOf(h);
    if (b < 0) { ++B.unknown; return; }
    Body& x = B.S.bodies[size_t(b)];
    B.apply(h, jnt::putToSleep(x, B.st(b, h)), x);
  }
  void rigidWake(int32_t h) override {
    const int32_t b = B.bodyOf(h);
    if (b < 0) { ++B.unknown; return; }
    Body& x = B.S.bodies[size_t(b)];
    B.apply(h, jnt::wakeUp(x, B.st(b, h), scene::kWakeReset), x);
  }

  // ---- 뿌리 (종류별)
  Tf rootPose(const BaseObject& o) override {
    if (o.kind == BASE_ART) {
      if (art) return art->rootPose(o.art);
      ++artMissing;
      return Tf{qid(), V3{0, 0, 0}};
    }
    if (o.kind == BASE_KINEMATIC) {  // XFormPrim.get_position_orientation (운동학 물체는 안 움직이므로 추출 값)
      if (o.hasXform) return o.xform;
      return B.hasStaticApi() ? B.staticPose(o.root) : B.actorPose(o.root);
    }
    return B.actorPose(o.root);
  }
  void rootVelocity(const BaseObject& o, V3& lin, V3& ang) override {
    if (o.kind == BASE_ART) {
      if (art) art->rootVelocity(o.art, lin, ang);
      else ++artMissing, lin = ang = V3{0, 0, 0};
      return;
    }
    B.velocity(o.root, lin, ang);
  }
  void jointState(const BaseObject& o, std::vector<float>& pos, std::vector<float>& vel) override {
    if (o.kind == BASE_ART && art) art->jointState(o.art, pos, vel);
    else ++artMissing;
  }
  void setRootPose(const BaseObject& o, const Tf& pose) override {
    if (o.kind == BASE_ART) {
      if (art) art->applyRootTransform(o.art, pose);
      else ++artMissing;
    } else if (o.kind == BASE_RIGID) {
      B.setActorPose(o.root, pose);
    } else if (o.kind == BASE_KINEMATIC && B.hasStaticApi()) {
      B.setStaticPose(o.root, pose);  // XForm 쓰기 → USD → PhysX 정적 setGlobalPose (공식 OVD: 운동학 물체 34 개가 되돌리기 때 정적 자세를 받음)
    }
  }
  void setRootLinVel(const BaseObject& o, const V3& v) override {
    V3 l, a;
    rootVelocity(o, l, a);
    if (o.kind == BASE_ART) {
      if (art) art->applyRootVelocities(o.art, v, a);
    } else if (o.kind == BASE_RIGID) {
      B.setVelocity(o.root, v, a);
    }
  }
  void setRootAngVel(const BaseObject& o, const V3& v) override {
    V3 l, a;
    rootVelocity(o, l, a);
    if (o.kind == BASE_ART) {
      if (art) art->applyRootVelocities(o.art, l, v);
    } else if (o.kind == BASE_RIGID) {
      B.setVelocity(o.root, l, v);
    }
  }
  void setJointPositions(const BaseObject& o, const std::vector<float>& v) override {
    if (o.kind == BASE_ART && art) art->applyJointPositions(o.art, v);
    else ++artMissing;
  }
  void setJointVelocities(const BaseObject& o, const std::vector<float>& v) override {
    if (o.kind == BASE_ART && art) art->applyJointVelocities(o.art, v);
    else ++artMissing;
  }
  bool artSleeping(int32_t a) override {
    if (art) return art->sleeping(a);
    ++artMissing;
    return false;
  }
  void artSleep(int32_t a) override {
    if (art) art->putToSleep(a);
    else ++artMissing;
  }
  void artWake(int32_t a) override {
    if (art) art->wakeUp(a);
    else ++artMissing;
  }
};

}  // namespace particles
}  // namespace eng
