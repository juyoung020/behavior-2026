// base 창 = OmniGibson 전체 상태 뜨기·되돌리기 (og.sim.dump_state / load_state) 의 물리 쪽을 엔진으로 (조정자 결정 09-30, B9 자리).
// envEditWindow(core/scene/env_step.h 12.3) 의 dumpState/loadState 자리에 붙는다 (particles edit_window.h TransitionEditWindow::base).
// 원본 (BEHAVIOR-1K v3.9.3-post1) 차례:
//   Simulator._load_state → scene.load_state → Scene._load_state (scene_base.py:1279, 장면 자세 같으면 그대로) → 등록부 (registry_utils.py:434)
//   **계 등록부가 물체 등록부보다 먼저** (scene_base.py:597 system_registry 를 먼저 add) — 입자 계는 edit_window.h 가 앞에서 한다.
//   물체마다 등록부 차례로 (scene.objects), EntityPrim._load_state (entity_prim.py:1565):
//     ① 기록이 잠이면 지금 뿌리 자세를 읽어 비교 (allclose atol 1e-7 rtol 0, 1-|dot(q)| < 1e-7)
//     ② root_link._load_state: 자세 set (RigidDynamicPrim.set_position_orientation → 뷰 set_world_poses) → set_linear_velocity → set_angular_velocity
//        (뷰의 속도 쓰기는 지금 6 칸을 읽어 한쪽만 바꿔 6 칸을 다시 씀 → 두 번 모두 선·각 쓰기)
//     ③ 관절이 있으면 set_joint_positions → set_joint_velocities (관절체 뷰, 전체 dof)
//     ④ ①이 잠이고 자세 같으면 sleep() 아니면 wake()  — 관절체는 관절체 하나(psi), 아니면 RigidDynamicPrim 링크마다(psi)
//     ⑤ 로봇이면 제어기·보조 잡기 상태, 모든 물체의 비운동 상태(object_states) — 물리 쓰기 없음 (엔진 omni 모듈 몫, 여기서는 갈고리)
//   운동학 전용 물체(kinematic_only)는 is_asleep = False → wake() 는 동적 링크가 없어 아무것도 안 함, 자세는 XForm(USD/Fabric) 쓰기.
// PhysX 에 닿는 때 (공식 탐침 tests/particles/probe_load_order.py, 09-30): 텐서 쓰기(자세·속도·관절)는 **부른 즉시** PhysX 에 닿고 자동으로 깨운다.
//   쓰기 → sleep() = 잠, sleep() → 쓰기 = 깨어 있음, load_state(dump) = 잠 유지. 즉 파이썬 호출 차례 = PhysX 차례.
// 텐서 쓰기 → PhysX 호출 (리드 재생기 ovd_replay.cpp 대응표): 강체 동체 = setGlobalPose/setLinearVelocity/setAngularVelocity (autowake),
//   관절체 뿌리(강체 뷰로 뿌리 링크) = applyCache(eROOT_TRANSFORM / eROOT_VELOCITIES), dof = applyCache(ePOSITION / eVELOCITY),
//   psi = PxRigidDynamic / PxArticulationReducedCoordinate 의 putToSleep·wakeUp. 계산은 joints(rigid_api.h)·articulation(articulation.h) 몫 → BaseStateApi.
// 물체 표는 장면 추출(extract_particles.py particles_spec.json "base_objects")에서.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/particles/edit_window.h"
#include "core/particles/spray.h"

namespace eng {
namespace particles {

enum BaseKind : uint8_t { BASE_RIGID = 0, BASE_ART = 1, BASE_KINEMATIC = 2, BASE_NONE = 3 };

struct BaseObject {
  uint8_t kind = BASE_NONE;
  uint8_t robot = 0;
  int32_t root = -1;               // 뿌리 행위자 손잡이 (BASE_RIGID / BASE_KINEMATIC)
  std::vector<int32_t> links;      // BASE_RIGID: RigidDynamicPrim 링크 손잡이 (링크 차례) — sleep/wake 대상
  int32_t art = -1;                // BASE_ART: 관절체 번호
  uint32_t nJoints = 0;            // 관절 쓰기 여부 (n_joints > 0)
  uint8_t hasXform = 0;            // BASE_KINEMATIC: dump 가 읽는 XForm 자세(USD/Fabric 합성)를 장면 추출에서 떠 둠 — PhysX 정적 자세와 몇 ulp 다를 수 있음
  Tf xform{};
};
struct BaseObjectState {
  uint8_t asleep = 0;
  Tf pose;                         // 뿌리 링크 자세 (강체 뷰 get_world_poses = 행위자 전역 자세)
  V3 lin{0, 0, 0}, ang{0, 0, 0};   // 뿌리 링크 속도
  std::vector<float> jpos, jvel;   // 관절체 캐시 칸 그대로 (뷰의 부호·순서는 되돌리기에서 상쇄)
};

// 엔진 쪽 "텐서 뷰 층" (리드 구현: EnvBodyApi 옆). OmniGibson 이 부르는 단위 그대로 받는다 — PhysX 호출로 풀기(아래)는 구현 몫:
//   강체 뿌리: 자세 = setGlobalPose(autowake), 선속도 쓰기 = 지금 6 칸 읽고 선만 바꿔 setLinearVelocity→setAngularVelocity (각속도 쓰기도 같은 꼴)
//   관절체 뿌리: 자세 = applyCache(eROOT_TRANSFORM), 속도 = applyCache(eROOT_VELOCITIES) (6 칸), 관절 = applyCache(ePOSITION / eVELOCITY) 전체 dof
//   psi: 강체 PxRigidDynamic::putToSleep / wakeUp, 관절체 PxArticulationReducedCoordinate::putToSleep / wakeUp (OVD 에 API 사건 없음 — 탐침 09-30)
struct BaseStateApi {
  virtual ~BaseStateApi() {}
  virtual bool rigidSleeping(int32_t h) = 0;  // psi.is_sleeping (강체 뿌리 링크)
  virtual bool artSleeping(int32_t a) = 0;    // psi.is_sleeping (관절체)
  virtual Tf rootPose(const BaseObject& o) = 0;                         // root_link.get_position_orientation (뷰 get_world_poses)
  virtual void rootVelocity(const BaseObject& o, V3& lin, V3& ang) = 0; // root_link.get_linear/angular_velocity
  virtual void jointState(const BaseObject& o, std::vector<float>& pos, std::vector<float>& vel) = 0;  // get_joint_positions/velocities
  virtual void setRootPose(const BaseObject& o, const Tf& pose) = 0;    // root_link.set_position_orientation
  virtual void setRootLinVel(const BaseObject& o, const V3& v) = 0;     // root_link.set_linear_velocity
  virtual void setRootAngVel(const BaseObject& o, const V3& v) = 0;     // root_link.set_angular_velocity
  virtual void setJointPositions(const BaseObject& o, const std::vector<float>& v) = 0;
  virtual void setJointVelocities(const BaseObject& o, const std::vector<float>& v) = 0;
  virtual void rigidSleep(int32_t h) = 0;  // RigidDynamicPrim.sleep (psi.put_to_sleep)
  virtual void rigidWake(int32_t h) = 0;
  virtual void artSleep(int32_t a) = 0;    // EntityPrim.sleep (관절체, psi)
  virtual void artWake(int32_t a) = 0;
  // ⑤ 로봇 제어기·보조 잡기, 비운동 상태 (물리 쓰기 없음)
  virtual void dumpExtra(size_t obj) { (void)obj; }
  virtual void loadExtra(size_t obj) { (void)obj; }
};

// ① 잠 유지 판정: th.allclose(cur, pos, rtol=0, atol=1e-7) 과 (1 - |th.dot(cur, orn)|) < 1e-7 (float32, th.dot = MKL sdot)
inline bool base_same_pose(const Tf& cur, const Tf& rec) {
  const float cp[3] = {cur.p.x, cur.p.y, cur.p.z}, rp[3] = {rec.p.x, rec.p.y, rec.p.z};
  for (int k = 0; k < 3; ++k)
    if (!(std::fabs(cp[k] - rp[k]) <= 1e-7f)) return false;
  const float cq[4] = {cur.q.x, cur.q.y, cur.q.z, cur.q.w}, rq[4] = {rec.q.x, rec.q.y, rec.q.z, rec.q.w};
  float d;
  if (mklsp::lib().dot) {
    int n = 4, one = 1;
    d = mklsp::lib().dot(&n, cq, &one, rq, &one);
  } else {
    d = ((cq[0] * rq[0] + cq[1] * rq[1]) + cq[2] * rq[2]) + cq[3] * rq[3];
  }
  return (1.0f - std::fabs(d)) < 1e-7f;
}

struct BaseStateWindow : scene::EditWindow {
  BaseStateApi* api = nullptr;
  const std::vector<BaseObject>* objects = nullptr;  // 등록부 차례
  std::vector<uint8_t> present;                      // 뜬 뒤 지워진 물체는 0 (removing_objects 가 상태에서 뺌)
  std::vector<BaseObjectState> saved;

  bool isAsleep(const BaseObject& o) {  // EntityPrim.is_asleep (운동학 전용·물리 없음 = False)
    if (o.kind == BASE_ART) return api->artSleeping(o.art);
    if (o.kind == BASE_RIGID) return api->rigidSleeping(o.root);
    return false;
  }
  void dumpState(scene::EnvStep& E) override {
    (void)E;
    const size_t n = objects->size();
    saved.assign(n, BaseObjectState());
    present.assign(n, 1);
    for (size_t i = 0; i < n; ++i) {  // EntityPrim._dump_state: is_asleep, 뿌리(자세, 속도), 관절
      const BaseObject& o = (*objects)[i];
      if (o.kind == BASE_NONE) continue;
      BaseObjectState& s = saved[i];
      s.asleep = isAsleep(o) ? 1 : 0;
      s.pose = api->rootPose(o);
      if (o.kind != BASE_KINEMATIC) api->rootVelocity(o, s.lin, s.ang);
      if (o.nJoints > 0) api->jointState(o, s.jpos, s.jvel);
      api->dumpExtra(i);
    }
  }
  // removing_objects: 지운 물체는 상태에서 빠진다 (edit 전에 알려 줌)
  void markRemoved(size_t obj) {
    if (obj < present.size()) present[obj] = 0;
  }
  void loadOne(size_t i) {
    const BaseObject& o = (*objects)[i];
    const BaseObjectState& s = saved[i];
    bool keep = false;
    if (s.asleep) keep = base_same_pose(api->rootPose(o), s.pose);  // ①
    api->setRootPose(o, s.pose);                                      // ② root_link._load_state
    if (o.kind != BASE_KINEMATIC) {
      api->setRootLinVel(o, s.lin);
      api->setRootAngVel(o, s.ang);
    }
    if (o.nJoints > 0) {  // ③
      api->setJointPositions(o, s.jpos);
      api->setJointVelocities(o, s.jvel);
    }
    const bool sleep = s.asleep && keep;  // ④
    if (o.kind == BASE_ART) {
      if (sleep) api->artSleep(o.art);
      else api->artWake(o.art);
    } else {
      for (int32_t h : o.links) {
        if (sleep) api->rigidSleep(h);
        else api->rigidWake(h);
      }
    }
    api->loadExtra(i);  // ⑤
  }
  void loadState(scene::EnvStep& E) override {
    (void)E;
    for (size_t i = 0; i < objects->size() && i < saved.size(); ++i)
      if ((*objects)[i].kind != BASE_NONE && present[i]) loadOne(i);
  }
};

}  // namespace particles
}  // namespace eng
