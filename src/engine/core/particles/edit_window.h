// 전이 편집 실행 (P2·P3 물리 결합, 리드 편집 창 약속 core/scene/env_step.h 12.3 에 얹음): 한 스텝의 전이 편집 목록(spawn.h Edit)을
// 공식 OmniGibson 실행 차례대로 ScScene·몸체 API 에 적용한다. 원본 (BEHAVIOR-1K v3.9.3-post1):
//   ① 규칙 step() 안 (execute_transition 앞): DicingRule → generate_particles (macro_particle_system.py:269,1420)
//      = 새 입자 prim 마다 강체 생성(add_particle → update_handles) → 모든 입자 set_transforms(기존 = 원점→중심 그대로, 새 = 중심·방향)
//        → 모든 입자 속도(기존 값, 새 것 0)                                                        [runRuleEdits: EDIT_PARTICLES_ADD]
//   ② execute_transition (transition_rules.py:182) 에 지울 물체가 있으면 removing_objects (simulator.py:1011) = envEditWindow 다섯 자리:
//      (1) dump_state: 물체 등록부(→ base) 다음 계 등록부(→ 입자 원점 자세·속도)
//      (2) 지울 물체마다 무덤 자세로 set_position_orientation (기준 링크 = 행위자 0)                 [EDIT_REMOVE_BEGIN.poses]
//      (3) step_physics 한 번
//      (4) 물체마다 삭제 (행위자마다 ScScene::removeActor)                                            [EDIT_REMOVE_OBJECT]
//      (5) load_state: 물체(→ base) 다음 계 (입자: particle_reload 로 자세, 뜬 속도)                    [EDIT_PARTICLES_RESET]
//   ③ 그다음 넣을 물체마다 add_object → set_bbox_center_position_orientation                         [EDIT_ADD_OBJECT, loadState 뒤]
//   ④ 다음 스텝 처음에 비물리 상태 물려받기                                                            [EDIT_INHERIT_STATES — 물리 아님, 호출자]
// 물체(강체 여럿)의 전체 상태 뜨기·되돌리기(자세·속도·관절)는 omni/리드 쪽 창(base)이 하고, 여기서는 그 앞뒤에 입자 계를 붙인다(공식 등록부 차례).
// 몸체 상태(자세·속도·깸)는 풀이 모듈이 들고 있으므로 BodyApi 로 부른다 (core/joints/rigid_api.h setGlobalPose·setLinearVelocity 부수효과 포함).
// 새 강체의 두 자세 (B10, 리드 곁기록 쓰기 열 09-30): 생성 = 입력 자세(방향은 PhysX getNormalized) 그대로, 그 뒤 omni.physx 의 USD→PhysX 동기화가
//   USD 왕복 자세(spawn.h usd_roundtrip_quat) + 속도 0 을 setGlobalPose·속도로 다시 쓴다. 동기화는 "다음 물체 넣기 직전"과 "simulate 직전"에 일어나고,
//   그때마다 이 창에서 넣은 물체 전부를 넣은 차례대로 다시 쓴다(양파 반쪽: 생성 0 → 동기 {0} → 생성 1 → 동기 {0,1} → simulate).
//   다진 입자: 생성(자세 없음) → 입자 자세 set(입력 A) → 속도 → 동기 (왕복 C) → simulate. [pending/flushSync]
#pragma once
#include <cstdint>
#include <vector>

#include "core/particles/spawn.h"
#include "core/scene/env_step.h"

namespace eng {
namespace particles {

// 몸체 상태 쪽 (풀이 모듈 구현). h = ScScene 손잡이
struct BodyApi {
  virtual ~BodyApi() {}
  virtual Tf actorPose(int32_t h) = 0;                                  // PxRigidActor::getGlobalPose (뷰 get_transforms)
  virtual void setActorPose(int32_t h, const Tf& pose) = 0;             // NpRigidDynamic::setGlobalPose (autowake, 방향 getNormalized)
  virtual void velocity(int32_t h, V3& lin, V3& ang) = 0;
  virtual void setVelocity(int32_t h, const V3& lin, const V3& ang) = 0;  // setLinearVelocity → setAngularVelocity
  virtual void bodyAdded(int32_t h, const Body& b) = 0;                 // spawn_add 로 만든 몸체 등록
  virtual void bodyRemoved(int32_t h) = 0;
};

// 새 행위자 출처 (리드 g1_sc 틀 + 우리 키 대응 build_spawn_table)
struct SpawnSource {
  const SpawnTemplate* T = nullptr;
  uint32_t actor = 0;
  float usd_scale[3] = {1, 1, 1};  // USD 왕복 척도 (반쪽 = 물체 척도, 입자 = 입자 prim 척도)
  bool raw_q = false;              // 입자: 정규화 전 방향으로 USD 행렬
};
struct SpawnResolver {
  virtual ~SpawnResolver() {}
  virtual bool half(int32_t src, int32_t part, SpawnSource& out) = 0;  // 반쪽 (원본 물체, 부분 번호)
  virtual bool particle(int32_t system, SpawnSource& out) = 0;         // 입자 계 틀
};

struct ObjectRt {
  std::vector<int32_t> actors;  // 행위자 0 = 기준 링크
};
struct ParticleSystemRt {
  std::vector<int32_t> actors;  // 입자 순서 (뷰 순서)
  float off[3] = {0, 0, 0};     // _particle_offset
  // (1) 에서 뜬 값
  std::vector<float> dumpTfs;   // [n][7] 원점 자세 (p, q)
  std::vector<V3> dumpLin, dumpAng;
};

struct TransitionEditWindow : scene::EditWindow {
  scene::EditWindow* base = nullptr;  // 물체 전체 상태 (omni/리드), 없으면 건너뜀
  BodyApi* body = nullptr;
  SpawnResolver* spawn = nullptr;
  std::vector<ObjectRt>* objects = nullptr;
  std::vector<ParticleSystemRt>* systems = nullptr;
  float scenePose[16], scenePoseInv[16];  // scene._pose_info (float 4x4 행 우선)
  const std::vector<Edit>* edits = nullptr;
  int32_t failed = 0;                     // 틀을 못 찾은 편집 수
  struct Pending {
    int32_t h;
    Tf pose;  // USD 왕복 행위자 자세
  };
  std::vector<Pending> pending;           // 이 창에서 넣어 USD 동기화를 기다리는 몸체 (넣은 차례)

  // USD → PhysX 동기화 한 번: 기다리는 몸체 전부 자세·속도 0 을 다시 쓴다
  void flushSync() {
    const V3 z{0, 0, 0};
    for (const Pending& q : pending) {
      body->setActorPose(q.h, q.pose);
      body->setVelocity(q.h, z, z);
    }
  }
  // 새 몸체 하나: 생성은 입력 자세, 동기화 대기에 USD 왕복 자세
  int32_t spawnOne(scene::EnvStep& E, const SpawnSource& src, const Pose7& p) {
    Body b, rt;
    const int32_t h = spawn_add(*E.sc, E.modules(), *src.T, src.actor, p, b);  // 생성: normalized(입력)
    if (h < 0) return h;
    body->bodyAdded(h, b);
    scene::ScActorIn tmp;
    actor_from_template(*src.T, src.actor, p, tmp, rt, src.usd_scale, src.raw_q);  // 왕복 몸체 자세 → 행위자 자세
    pending.push_back(Pending{h, getGlobalPose(rt)});
    return h;
  }

  TransitionEditWindow() {
    for (int i = 0; i < 16; ++i) scenePose[i] = scenePoseInv[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  }
  bool hasRemoval() const {
    for (const Edit& e : *edits)
      if (e.kind == EDIT_REMOVE_BEGIN) return true;
    return false;
  }
  static Tf tf7(const float* t) { return Tf{Q{t[3], t[4], t[5], t[6]}, V3{t[0], t[1], t[2]}}; }

  // ① 규칙 step() 안의 입자 생성 (창 밖, execute_transition 앞)
  void runRuleEdits(scene::EnvStep& E) {
    for (const Edit& e : *edits) {
      if (e.kind != EDIT_PARTICLES_ADD) continue;
      ParticleSystemRt& S = (*systems)[size_t(e.system)];
      SpawnSource src;
      if (!spawn->particle(e.system, src)) {
        ++failed;
        continue;
      }
      // 기존 입자 get (원점 → 중심) — 새 prim 을 만들기 전에 뜬다 (generate_particles 첫 줄)
      const size_t n0 = S.actors.size();
      std::vector<Pose7> set(n0);
      std::vector<V3> lin(n0), ang(n0);
      for (size_t i = 0; i < n0; ++i) {
        const Tf a = body->actorPose(S.actors[i]);
        const float t[7] = {a.p.x, a.p.y, a.p.z, a.q.x, a.q.y, a.q.z, a.q.w};
        float c[3];
        physical_center(t, S.off, c);
        set[i] = particle_frame_from_center(c, t + 3, S.off);  // 공식: 중심이 그대로 원점
      }
      for (const Pose7& p : e.poses) {  // 새 입자 강체 (원점 자세, 방향 정규화 전)
        const int32_t h = spawnOne(E, src, p);
        if (h < 0) {
          ++failed;
          continue;
        }
        S.actors.push_back(h);
        set.push_back(p);
      }
      // 자세 set (뷰 set_transforms, 모든 입자) → 속도 get (기존 값) → 속도 set (기존, 새 것 0)
      for (size_t i = 0; i < S.actors.size() && i < set.size(); ++i) body->setActorPose(S.actors[i], pose7_tf(set[i]));
      for (size_t i = 0; i < n0; ++i) body->velocity(S.actors[i], lin[i], ang[i]);
      const V3 z{0, 0, 0};
      for (size_t i = 0; i < S.actors.size(); ++i) body->setVelocity(S.actors[i], i < n0 ? lin[i] : z, i < n0 ? ang[i] : z);
    }
  }

  // ② removing_objects 다섯 자리
  void dumpState(scene::EnvStep& E) override {
    if (base) base->dumpState(E);
    for (ParticleSystemRt& S : *systems) {
      const size_t n = S.actors.size();
      S.dumpTfs.resize(7 * n);
      S.dumpLin.resize(n);
      S.dumpAng.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const Tf a = body->actorPose(S.actors[i]);
        float* t = &S.dumpTfs[7 * i];
        t[0] = a.p.x, t[1] = a.p.y, t[2] = a.p.z, t[3] = a.q.x, t[4] = a.q.y, t[5] = a.q.z, t[6] = a.q.w;
        body->velocity(S.actors[i], S.dumpLin[i], S.dumpAng[i]);
      }
    }
  }
  void teleportToGrave(scene::EnvStep& E) override {
    (void)E;
    for (const Edit& e : *edits) {
      if (e.kind != EDIT_REMOVE_BEGIN) continue;
      // e.poses[i] ↔ 지울 물체 차례: EDIT_REMOVE_OBJECT 편집의 object 순서
      size_t i = 0;
      for (const Edit& r : *edits) {
        if (r.kind != EDIT_REMOVE_OBJECT) continue;
        if (i < e.poses.size() && !(*objects)[size_t(r.object)].actors.empty()) body->setActorPose((*objects)[size_t(r.object)].actors[0], pose7_tf(e.poses[i]));
        ++i;
      }
    }
    flushSync();  // step_physics 의 simulate 직전 동기화 (① 에서 만든 입자)
  }
  bool extraPhysicsStep() const override { return hasRemoval(); }
  void edit(scene::EnvStep& E) override {
    for (const Edit& e : *edits) {
      if (e.kind != EDIT_REMOVE_OBJECT) continue;
      ObjectRt& o = (*objects)[size_t(e.object)];
      for (int32_t h : o.actors) {
        E.sc->removeActor(h, E.modules());
        body->bodyRemoved(h);
      }
      o.actors.clear();
    }
  }
  void loadState(scene::EnvStep& E) override {
    if (base) base->loadState(E);
    for (ParticleSystemRt& S : *systems) {  // 계 등록부: _sync_particles(수 같음) → 자세 set → 크기(같음) → 속도 set
      const size_t n = S.actors.size();
      if (S.dumpTfs.size() != 7 * n) continue;
      for (size_t i = 0; i < n; ++i) body->setActorPose(S.actors[i], pose7_tf(particle_reload(&S.dumpTfs[7 * i], S.off, scenePose, scenePoseInv)));
      for (size_t i = 0; i < n; ++i) body->setVelocity(S.actors[i], S.dumpLin[i], S.dumpAng[i]);
    }
    addObjects(E);  // ③ 은 load_state 뒤 (execute_transition 이 removing_objects 를 나온 뒤 넣음)
  }
  // ③ 새 물체 넣기 (지울 것이 없는 스텝이면 창 없이 이것만)
  void addObjects(scene::EnvStep& E) {
    for (const Edit& e : *edits) {
      if (e.kind != EDIT_ADD_OBJECT) continue;
      SpawnSource src;
      if (!spawn->half(e.src, e.part, src) || e.poses.empty()) {
        ++failed;
        continue;
      }
      flushSync();  // 다음 물체 넣기 직전 동기화 (앞서 넣은 것들)
      const int32_t h = spawnOne(E, src, e.poses[0]);
      if (h < 0) {
        ++failed;
        continue;
      }
      if (size_t(e.object) >= objects->size()) objects->resize(size_t(e.object) + 1);
      (*objects)[size_t(e.object)].actors.push_back(h);
    }
  }
};

// 한 스텝의 전이 편집 전부 (규칙 step 뒤, 환경 스텝 사이): ① → (지울 것 있으면 ② 창 + ③, 없으면 ③ 만) → 동기화
// 주의: 다진 입자가 있는 창에서는 ① 뒤의 동기화가 removing_objects 의 step_physics(물리 한 스텝) 직전에도 한 번 든다(그 simulate 앞) — extraPhysicsStep 앞 flushSync.
// 끝에 simulate 직전 동기화 (기다리던 몸체 전부) — 다음 envStep 앞
inline void runTransitionEdits(scene::EnvStep& E, TransitionEditWindow& w) {
  w.pending.clear();
  w.runRuleEdits(E);
  if (w.hasRemoval()) scene::envEditWindow(E, w);
  else w.addObjects(E);
  w.flushSync();
  w.pending.clear();
}

}  // namespace particles
}  // namespace eng
