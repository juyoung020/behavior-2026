// 자르기 전이(SlicingRule) + 자르개 켜짐(SlicerActive) — OmniGibson 파이썬 층을 손으로 (층 1·2 공용).
// 원본 (BEHAVIOR-1K v3.9.3-post1, omnigibson/):
//   transition_rules.py:923 SlicingRule — 후보: 능력 sliceable / slicer (scene.objects 순서)
//     조건 1 TouchingAnyCondition(:357): sliceable 중 "slicer 후보 아무것과" 최근 접촉(current_only=False) 한 것만 남김
//     조건 2 StateCondition(:411): slicer 후보를 SlicerActive == True 인 것으로 거름. 하나도 없으면 전이 없음
//       → 켜진 자르개가 하나라도 있으면, 꺼진 자르개에 닿은 것도 잘린다 (조건이 따로 거르므로 — 공식 그대로)
//     전이(:942): 반쪽마다 bb 중심 = pos + T.quat2mat(orn) @ (bb_pos * scale), 방향 = T.quat_multiply(orn, bb_orn),
//       bounding_box = bb_size * scale (척도가 고르지 않으면 scale = |T.quat2mat(bb_orn) @ scale|), 이름 half_<원본>_<i>.
//       반쪽이 물려받는 상태는 **이번 전이에서 마지막으로 자른 물체**의 dump_state (lambda 늦은 묶임, :985).
//   objects/dataset_object.py:383 반쪽 척도 = bounding_box / ig:nativeBB (nativeBB > 1e-4 인 축만, 나머지 1)
//   objects/dataset_object.py:472 set_bbox_center_position_orientation: 기준 위치 = bb 중심 + T.pose_transform(0, orn, -scale*offsetBaseLink, 단위)[0]
//   object_states/slicer_active.py warp 커널 3 개 (스텝마다): 앞 지움 → 현재 닿음(자르개 링크 행 × 자를 것 링크 열, 최근 행렬) → 뒤 갱신
// 연산 순서: T.* (torch.compile) 는 omni agframe.h 에서 찾은 것(단일 quat2mat 나비 합, 4x4 곱 순차, mat2pose)을 그대로 쓴다.
//   eager `R @ v`(3x3 @ 3) 는 tests/particles/test_slice 후보 대조로 정함.
#pragma once
#include <cmath>
#include <cstdint>

#include "core/omni/agframe.h"
#include "core/particles/visual.h"

namespace eng {
namespace particles {

namespace agf = eng::omni::agf;

// eager aten::mv (3x3 @ 3): 순차 곱·합, FMA 없음 (대조로 확인)
PEHD void mv3(const float R[9], const float v[3], float o[3]) {
  for (int i = 0; i < 3; ++i) o[i] = (R[3 * i] * v[0] + R[3 * i + 1] * v[1]) + R[3 * i + 2] * v[2];
}
// T.quat_multiply(q1, q0) (transform_utils.py:187, inductor: 왼쪽부터, FMA 없음)
PEHD void quat_multiply(const float q1[4], const float q0[4], float o[4]) {
  const float x0 = q0[0], y0 = q0[1], z0 = q0[2], w0 = q0[3], x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
  o[0] = ((x1 * w0 + y1 * z0) - z1 * y0) + w1 * x0;
  o[1] = ((-x1 * z0 + y1 * w0) + z1 * x0) + w1 * y0;
  o[2] = ((x1 * y0 - y1 * x0) + z1 * w0) + w1 * z0;
  o[3] = ((-x1 * x0 - y1 * y0) - z1 * z0) + w1 * w0;
}

// 반쪽 하나의 bb 자세·크기 (SlicingRule.transition 한 부분). part_bb_* 는 메타데이터(파이썬 float → float32)
PEHD void slice_part_bbox(const float pos[3], const float orn[4], const float scale[3], const float part_bb_pos[3],
                          const float part_bb_orn[4], const float part_bb_size[3], float out_pos[3], float out_orn[4], float out_bb[3]) {
  float s[3] = {scale[0], scale[1], scale[2]};
  if (!(scale[0] == scale[0] && scale[1] == scale[0] && scale[2] == scale[0])) {  // th.all(scale == scale[0])
    float Rb[9];
    agf::quat2mat(part_bb_orn, Rb);
    mv3(Rb, scale, s);
    for (int k = 0; k < 3; ++k) s[k] = std::fabs(s[k]);
  }
  float R[9], v[3], rv[3];
  agf::quat2mat(orn, R);
  for (int k = 0; k < 3; ++k) v[k] = part_bb_pos[k] * s[k];
  mv3(R, v, rv);
  for (int k = 0; k < 3; ++k) out_pos[k] = pos[k] + rv[k];
  quat_multiply(orn, part_bb_orn, out_orn);
  for (int k = 0; k < 3; ++k) out_bb[k] = part_bb_size[k] * s[k];
}
// 반쪽 물체 척도 (dataset_object.py:383)
PEHD void half_scale(const float bb[3], const float native_bb[3], float scale[3]) {
  for (int k = 0; k < 3; ++k) scale[k] = native_bb[k] > 1e-4f ? div_rn(bb[k], native_bb[k]) : 1.0f;
}
// set_bbox_center_position_orientation → 기준 링크 위치 (방향은 그대로)
PEHD void bbox_center_to_base(const float bb_pos[3], const float orn[4], const float scale[3], const float base_link_offset[3],
                              float base_pos[3]) {
  float c[3], m1[16], m0[16], h[16], p[3], q[4];
  for (int k = 0; k < 3; ++k) c[k] = -scale[k] * base_link_offset[k];
  const float z3[3] = {0.0f, 0.0f, 0.0f}, id[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  agf::pose2mat(z3, orn, m1);
  agf::pose2mat(c, id, m0);
  agf::mm4(m1, m0, h, 1);
  agf::mat2pose(h, p, q);
  for (int k = 0; k < 3; ++k) base_pos[k] = bb_pos[k] + p[k];
}

// ---- SlicerActive (자르개 물체 하나) ----
struct SlicerState {
  uint8_t value;       // 켜짐 (처음 1)
  int32_t prev_touch;  // PREVIOUSLY_TOUCHING
  float delay;         // DELAY_COUNTER (초)
};
// 스텝마다 한 번 (TensorizedState 갱신 = _refresh_state_caches, 전이 규칙보다 먼저). touching = 이번 스텝 최근 접촉 행렬에서
// 자르개 링크 행 × 모든 자를 것 링크 열 중 하나라도 1. dt = 환경 스텝 시간(float32). 2.0 초 = m.REACTIVATION_DELAY
PEHD void slicer_active_update(SlicerState& s, bool touching, float dt) {
  if (s.prev_touch != 0) {  // _slicer_pre_clear_kernel
    s.delay = 0.0f;
    s.value = 0;
  }
  const int32_t cur = touching ? 1 : 0;  // _slicer_zero_currently_touching_kernel + is_in_contact_batch_warp(atomic_max)
  if (s.value == 0) {                    // _slicer_post_update_kernel
    if (!cur)
      s.delay = s.delay + dt;
    else
      s.delay = 0.0f;
  }
  if (s.delay >= 2.0f) s.value = 1;
  s.prev_touch = cur;
}

// ---- SlicingRule 판단 (한 스텝) ----
// n_sl 개 자를 것(scene.objects 순서), n_kn 개 자르개. touch(i, j) = 자를 것 i 와 자르개 j 가 최근 접촉(행 = i 링크, 열 = j 링크).
// 반환 = 자를 것 수, out[] = 자를 것 번호(후보 순서). 조건 2 에서 켜진 자르개가 없으면 0.
template <class TouchFn>
PEHD int slicing_rule_select(int n_sl, int n_kn, const uint8_t* knife_active, TouchFn touch, int* out) {
  if (n_sl == 0 || n_kn == 0) return 0;  // 규칙이 활성이 아님 (get_rule_candidates: 거르개 둘 다 비어 있지 않아야)
  int m = 0;
  for (int i = 0; i < n_sl; ++i) {
    bool t = false;
    for (int j = 0; j < n_kn && !t; ++j) t = touch(i, j);
    if (t) out[m++] = i;
  }
  if (m == 0) return 0;
  bool any_active = false;
  for (int j = 0; j < n_kn; ++j) any_active = any_active || knife_active[j] != 0;
  return any_active ? m : 0;
}

// ---- BDDL 범위 칸 (future/real, tasks/behavior_task.py:737~809) ----
// 칸 순서 = 범위 dict 순서 (agent 먼저, 그다음 compiled_task.object_scope 순서). val[j] = 묶인 물체/계 번호, -1 = None.
// matches(j) = 칸 j 의 범주 집합에 새 물체 범주가 드는가 (og_categories_from_bddl_inst), is_sys[j] = 칸이 물질(계) 인가.
// 물체 추가: 첫 빈 물체 칸 하나. 물체 삭제: 그 물체가 묶인 첫 칸을 비움. 계 초기화: 첫 빈 계 칸 중 첫 범주가 계 이름과 같은 칸 하나.
template <class MatchFn>
PEHD int scope_on_add(int n, const uint8_t* is_sys, int32_t* val, MatchFn matches, int32_t oid) {
  for (int j = 0; j < n; ++j)
    if (val[j] == -1 && !is_sys[j] && matches(j)) {
      val[j] = oid;
      return j;
    }
  return -1;
}
PEHD int scope_on_remove(int n, const uint8_t* is_sys, int32_t* val, int32_t oid) {
  for (int j = 0; j < n; ++j)
    if (val[j] == oid && !is_sys[j]) {
      val[j] = -1;
      return j;
    }
  return -1;
}
template <class MatchFn>
PEHD int scope_on_system_init(int n, const uint8_t* is_sys, int32_t* val, MatchFn first_cat_is, int32_t sid) {
  for (int j = 0; j < n; ++j)
    if (val[j] == -1 && is_sys[j] && first_cat_is(j)) {
      val[j] = sid;
      return j;
    }
  return -1;
}

}  // namespace particles
}  // namespace eng
