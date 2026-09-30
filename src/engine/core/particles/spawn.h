// 전이 편집 목록 (P2·P3 물리 결합 (c), 조정자 결정 09-30): 한 스텝의 전이 결과를 공식 실행 순서대로 편집으로 낸다.
// 실행(ScScene 추가·삭제, 몸체 만들기, 상태 저장·재적재)은 리드의 장면 층이 한다. 새 행위자의 물리 입력은 "새 행위자 틀"(리드 g1_sc 가 수확 기록에서 씀)을
// 키(원본 물체·부분 번호 / 입자 계)로 찾는다. 원본 (BEHAVIOR-1K v3.9.3-post1):
//   TransitionRuleAPI.step (transition_rules.py:156): ① 지난 스텝에 넣은 물체의 비물리 상태 물려받기(obj_init_info) → ② 활성 규칙마다 step()
//     (다지기는 여기서 바로 입자 강체를 만든다: DicingRule.transition → generate_particles_from_link → generate_particles) → ③ execute_transition
//   execute_transition (:182): 지울 물체가 있으면 og.sim.batch_remove_objects → removing_objects (simulator.py:1011):
//       전체 상태 저장 → 지울 물체를 무덤 (100,100,100) 부터 x 로 max(aabb 범위)씩 옮겨 가며 순간이동(방향 0,0,0,1) → og.sim.step_physics() (simulate 1 번)
//       → 물체마다 삭제 → 뷰 갱신 → 규칙 가지치기 → 전체 상태 재적재 (지운 물체는 빼고)
//     그다음 넣을 물체마다 scene.add_object → set_bbox_center_position_orientation (slicing.h bbox_center_to_base 자세)
//   MacroPhysicalParticleSystem.generate_particles (macro_particle_system.py:269,1420): 새 입자 prim 을 만들고, **기존 입자까지 전부**
//     set_particles_position_orientation(중심 → 원점: p - quat2mat(q) @ offset) 과 속도(기존 값 + 새 것 0)를 다시 넣는다.
//     기존 입자의 중심은 get(원점 → 중심: contain.h physical_center) 에서 오므로 원점이 중심을 거쳐 한 번 왕복한다(끝비트가 바뀔 수 있음 — 공식 그대로).
// 리드가 확인할 것(요청 중): 무덤 순간이동·step_physics 창과 상태 재적재가 기록(chop_slice0)의 어느 simulate 인지.
#pragma once
#include <cstdint>
#include <vector>

#include "core/particles/contain.h"
#include "core/particles/slicing.h"
#include "core/particles/trng.h"

namespace eng {
namespace particles {

enum EditKind : int32_t {
  EDIT_INHERIT_STATES = 0,   // 지난 스텝에 넣은 물체(object)가 원본(src)의 비물리 상태를 물려받음 (온도·익음 등; 늦은 묶임: 한 전이의 반쪽 모두 마지막 원본)
  EDIT_PARTICLES_ADD = 1,    // 입자 계(tmpl = 계 틀)에 강체 n 개: 원점 자세 poses[n], 속도 0. 기존 입자 전부 다시 놓기(EDIT_PARTICLES_RESET)가 뒤따름
  EDIT_PARTICLES_RESET = 2,  // 입자 계의 기존 입자 전부를 왕복한 원점 자세로 다시 놓고 속도 유지
  EDIT_REMOVE_BEGIN = 3,     // 전체 상태 저장 + 무덤 순간이동(objects, 자세 poses) + 물리 1 스텝
  EDIT_REMOVE_OBJECT = 4,    // 물체 삭제 (행위자·모양)
  EDIT_REMOVE_END = 5,       // 뷰 갱신 + 규칙 가지치기 + 전체 상태 재적재
  EDIT_ADD_OBJECT = 6,       // 물체 추가: 틀(src 물체의 part 번호) 을 기준 링크 자세 pose 로
};
struct Pose7 {
  float p[3], q[4];
};
struct Edit {
  int32_t kind = 0;
  int32_t object = -1;  // 대상 물체 번호 (장면 물체 표)
  int32_t src = -1;     // 원본 물체 번호 (틀 키, 상태 물려받기)
  int32_t part = -1;    // 부분 번호 (반쪽 틀 키)
  int32_t system = -1;  // 입자 계 번호
  std::vector<Pose7> poses;
};

// 반쪽 하나의 추가 자세: slicing.h 의 사슬 (공식 실제 기록과 비트 동일, tests/particles/test_slice_capture)
struct HalfSpec {
  float bb_pos[3], bb_orn[4], bb_size[3];  // object_parts 메타데이터
  float native_bb[3], base_link_offset[3]; // 반쪽 에셋 (수확)
};
inline Pose7 half_add_pose(const float src_pos[3], const float src_orn[4], const float src_scale[3], const HalfSpec& h) {
  float bp[3], bo[4], bb[3], sc[3];
  Pose7 r;
  slice_part_bbox(src_pos, src_orn, src_scale, h.bb_pos, h.bb_orn, h.bb_size, bp, bo, bb);
  half_scale(bb, h.native_bb, sc);
  bbox_center_to_base(bp, bo, sc, h.base_link_offset, r.p);
  for (int k = 0; k < 4; ++k) r.q[k] = bo[k];
  return r;
}

// 무덤 순간이동 자세 (removing_objects): x 를 물체마다 max(aabb 범위)만큼. pos[0] 은 파이썬 float 100.0 으로 시작하지만
// `pos[0] += max(ob.aabb_extent)` 에서 0 차원 float32 텐서가 되므로 누적은 float32 (aabb 범위 = omni AABB 상태)
inline void graveyard_poses(const float* aabb_extents /*[n][3]*/, int n, std::vector<Pose7>& out) {
  float x = 100.0f;
  out.clear();
  for (int i = 0; i < n; ++i) {
    const float* e = aabb_extents + 3 * i;
    out.push_back(Pose7{{x, 100.0f, 100.0f}, {0.0f, 0.0f, 0.0f, 1.0f}});
    float m = e[0];  // 파이썬 max: 앞에서부터 더 큰 것 (같으면 앞)
    if (e[1] > m) m = e[1];
    if (e[2] > m) m = e[2];
    x = x + m;
  }
}

// 다진 입자 원점 자세: 중심 c 와 방향 q (random_quaternion) → 원점 = c - quat2mat(q) @ offset (배치 quat2mat, 순차 곱)
inline Pose7 particle_frame_from_center(const float c[3], const float q[4], const float off[3]) {
  float R[9];
  quat2mat_batched(q, R);
  Pose7 r;
  for (int i = 0; i < 3; ++i) r.p[i] = c[i] - ((R[3 * i] * off[0] + R[3 * i + 1] * off[1]) + R[3 * i + 2] * off[2]);
  for (int k = 0; k < 4; ++k) r.q[k] = q[k];
  return r;
}

}  // namespace particles
}  // namespace eng
