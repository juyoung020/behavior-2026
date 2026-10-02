// R1Pro 제어기 (손으로 짬). 평가기 설정(OmniGibson/omnigibson/eval/r1pro.yaml)의 제어기 묶음과 비트까지 같게:
//   base    HolonomicBaseJointController, velocity, 입력 [-1,1] → 출력 [-0.75,-0.75,-1]..[0.75,0.75,1]
//   trunk / arm_left / arm_right   JointController, position, 절대값, 입력·출력 한계 없음, 임피던스 없음
//   gripper_left / gripper_right    MultiFingerGripperController, smooth, position, 입력 [-1,1] → 손가락 한계 평균
// 원본: controllers/controller_base.py:304 _preprocess_command, :419 clip_control, :436 step
//       controllers/joint_controller.py:274 _update_goal, :316 compute_control, :380 compute_no_op_goal
//       controllers/holonomic_base_joint_controller.py:98 _update_goal
//       controllers/multi_finger_gripper_controller.py:201 _preprocess_command, :218 compute_control, :366 compute_no_op_goal
//       robots/robot.py:758 apply_action (rz 감기), :795 행동 자르기 순서
// 제어기는 numpy float32 백엔드(macros.py:152). 수 연산 흉내는 core/omni/npf32.h.
// 행동(23) = base 3 | trunk 4 | arm_left 7 | gripper_left 1 | arm_right 7 | gripper_right 1 (r1pro.yaml raw_controller_order)
// 한 스텝: apply_action(행동 → 목표) 한 번, 그다음 물리 서브스텝마다 step(목표 → 드라이브 목표). 목표가 안 바뀌면 출력도 같다.
#pragma once
#include <cmath>
#include <cstdint>

#include "core/omni/npf32.h"

namespace eng {
namespace omni {
namespace ctrl {

constexpr int kMaxDof = 32;
enum Group { G_BASE = 0, G_TRUNK = 1, G_ARM_L = 2, G_GRIP_L = 3, G_ARM_R = 4, G_GRIP_R = 5, G_N = 6 };

// 로봇 형태별 고정값 (판끼리 공유). 한계는 robot.control_limits (float32) 그대로.
struct R1ProConfig {
  int n_dof;
  int base_dof[3];  // x, y, rz 가상 관절 (base_control_idx)
  int trunk_dof[4];
  int arm_dof[2][7];
  int grip_dof[2][2];
  float pos_lo[kMaxDof], pos_hi[kMaxDof];
  float vel_lo[kMaxDof], vel_hi[kMaxDof];
  uint8_t has_limit[kMaxDof];
  // base 명령 한계 (r1pro.yaml)
  float base_in_lo[3], base_in_hi[3], base_out_lo[3], base_out_hi[3];
  // ---- init() 가 파이썬과 같은 식으로 미리 계산 (controller_base.py:323 _command_scale_factor 등)
  float base_scale[3], base_out_tf[3], base_in_tf[3];
  float grip_in_lo, grip_in_hi;  // "default" = (-1, 1)
  float grip_out_lo[2], grip_out_hi[2], grip_scale[2], grip_out_tf[2], grip_in_tf[2];
};

// 판마다 바뀌는 제어기 상태 (goal + goal_set, controller_base.py:152~)
struct R1ProState {
  float base_goal[3];
  float trunk_goal[4];
  float arm_goal[2][7];
  float grip_goal[2];
  uint8_t goal_set[G_N];
};

// 이번 서브스텝 드라이브 목표. set_pos / set_vel 이 1 인 자유도만 PhysX 에 쓴다.
struct DriveTargets {
  float pos[kMaxDof];
  float vel[kMaxDof];
  uint8_t set_pos[kMaxDof];
  uint8_t set_vel[kMaxDof];
};

OEHD void init(R1ProConfig& c) {
  for (int k = 0; k < 3; ++k) {
    c.base_scale[k] = std::fabs(c.base_out_hi[k] - c.base_out_lo[k]) / std::fabs(c.base_in_hi[k] - c.base_in_lo[k]);
    c.base_out_tf[k] = (c.base_out_hi[k] + c.base_out_lo[k]) / 2.0f;
    c.base_in_tf[k] = (c.base_in_hi[k] + c.base_in_lo[k]) / 2.0f;
  }
  c.grip_in_lo = -1.0f;
  c.grip_in_hi = 1.0f;
  for (int a = 0; a < 2; ++a) {
    // smooth: 출력 한계 = 손가락 위치 한계의 np.mean (float32 합 후 2 로 나눔)
    const int f0 = c.grip_dof[a][0], f1 = c.grip_dof[a][1];
    c.grip_out_lo[a] = (c.pos_lo[f0] + c.pos_lo[f1]) / 2.0f;
    c.grip_out_hi[a] = (c.pos_hi[f0] + c.pos_hi[f1]) / 2.0f;
    c.grip_scale[a] = std::fabs(c.grip_out_hi[a] - c.grip_out_lo[a]) / std::fabs(c.grip_in_hi - c.grip_in_lo);
    c.grip_out_tf[a] = (c.grip_out_hi[a] + c.grip_out_lo[a]) / 2.0f;
    c.grip_in_tf[a] = (c.grip_in_hi + c.grip_in_lo) / 2.0f;
  }
}

OEHD void reset(R1ProState& s) {  // controller_base.py:490 reset
  for (int k = 0; k < 3; ++k) s.base_goal[k] = 0.0f;
  for (int k = 0; k < 4; ++k) s.trunk_goal[k] = 0.0f;
  for (int a = 0; a < 2; ++a) {
    for (int k = 0; k < 7; ++k) s.arm_goal[a][k] = 0.0f;
    s.grip_goal[a] = 0.0f;
  }
  for (int g = 0; g < G_N; ++g) s.goal_set[g] = 0;
}

// ---- robot.py:767 apply_action 앞부분: rz 가상 관절이 [-pi, pi] 밖이면 감아서 순간이동.
// 비교·감기 모두 torch float32 (파이썬 math.pi 는 float32 로 바뀌어 계산됨, 확인: gen_ctrl_ref.py wrap 시험).
// wrap_angle (utils/geometry_utils.py:12) = (x + pi) % (2 pi) - pi, torch.remainder = fmod 후 부호 맞춤.
OEHD bool rz_needs_wrap(float rz) {
  const float pi = 3.14159265358979323846f;
  return rz < -pi || rz > pi;
}
OEHD float wrap_angle_f32(float x) {
  const float pi = 3.14159265358979323846f;
  const float two_pi = 6.28318530717958647692f;
  const float t = x + pi;
  float m = std::fmod(t, two_pi);
  if (m != 0.0f && ((two_pi < 0.0f) != (m < 0.0f))) m = m + two_pi;
  return m - pi;
}

// ---- 목표 갱신 (ControllerView.update_goal → BaseController.update_goal)
// joint (절대 위치): _preprocess_command 는 입력 한계 None 이라 그대로, _update_goal 에서 위치 한계로 clip
OEHD void joint_pos_update_goal(const R1ProConfig& c, const int* dof, int n, const float* cmd, float* goal) {
  for (int k = 0; k < n; ++k) goal[k] = np_clip(cmd[k], c.pos_lo[dof[k]], c.pos_hi[dof[k]]);
}

// gripper smooth: [command[0]] → 입력 clip → 비례 (command - in_tf) * scale + out_tf
OEHD float gripper_update_goal(const R1ProConfig& c, int arm, float cmd0) {
  float x = np_clip(cmd0, c.grip_in_lo, c.grip_in_hi);
  return (x - c.grip_in_tf[arm]) * c.grip_scale[arm] + c.grip_out_tf[arm];
}

// base (속도): 입력 clip·비례 → 로봇(base_footprint) 좌표 명령을 관절체 뿌리(canonical) 좌표로 회전 → 속도 한계 clip
//   base_p/q: 바닥 링크 자세(get_position_orientation), root_p/q: 관절체 뿌리 자세(get_root_position_orientation)
OEHD void base_update_goal(const R1ProConfig& c, const float cmd[3], const float base_p[3], const float base_q[4],
                           const float root_p[3], const float root_q[4], float goal[3]) {
  float x[3];
  for (int k = 0; k < 3; ++k) {
    x[k] = np_clip(cmd[k], c.base_in_lo[k], c.base_in_hi[k]);
    x[k] = (x[k] - c.base_in_tf[k]) * c.base_scale[k] + c.base_out_tf[k];
  }
  const M4 base_pose = np_pose2mat(base_p, base_q);
  const M4 canonical_pose = np_pose2mat(root_p, root_q);
  const M4 c2b = np_matmul4(np_pose_inv(canonical_pose), base_pose);
  M4 cmd_in_base = m4_eye();
  cmd_in_base.m[0][3] = x[0];
  cmd_in_base.m[1][3] = x[1];
  M4 rot = m4_eye();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) rot.m[i][j] = c2b.m[i][j];
  const M4 cmd_in_canon = np_matmul4(rot, cmd_in_base);
  const float v[3] = {cmd_in_canon.m[0][3], cmd_in_canon.m[1][3], x[2]};
  for (int k = 0; k < 3; ++k) goal[k] = np_clip(v[k], c.vel_lo[c.base_dof[k]], c.vel_hi[c.base_dof[k]]);
}

// robot.apply_action 의 제어기 부분: 행동 23 개를 순서대로 잘라 각 제어기 목표로
OEHD void apply_action(const R1ProConfig& c, R1ProState& s, const float action[23], const float base_p[3],
                       const float base_q[4], const float root_p[3], const float root_q[4]) {
  base_update_goal(c, action + 0, base_p, base_q, root_p, root_q, s.base_goal);
  s.goal_set[G_BASE] = 1;
  joint_pos_update_goal(c, c.trunk_dof, 4, action + 3, s.trunk_goal);
  s.goal_set[G_TRUNK] = 1;
  joint_pos_update_goal(c, c.arm_dof[0], 7, action + 7, s.arm_goal[0]);
  s.goal_set[G_ARM_L] = 1;
  s.grip_goal[0] = gripper_update_goal(c, 0, action[14]);
  s.goal_set[G_GRIP_L] = 1;
  joint_pos_update_goal(c, c.arm_dof[1], 7, action + 15, s.arm_goal[1]);
  s.goal_set[G_ARM_R] = 1;
  s.grip_goal[1] = gripper_update_goal(c, 1, action[22]);
  s.goal_set[G_GRIP_R] = 1;
}

// ---- 서브스텝마다 (ControllerView.step_all, 그룹 등록 순서 = base, trunk, arm_left, gripper_left, arm_right, gripper_right)
// 목표가 아직 없으면 no-op 목표를 먼저 채운다 (controller_base.py:453). joint_pos 는 지금 관절 위치 (float32).
OEHD void step(const R1ProConfig& c, R1ProState& s, const float* joint_pos, DriveTargets& out) {
  for (int d = 0; d < c.n_dof; ++d) {
    out.set_pos[d] = 0;
    out.set_vel[d] = 0;
  }
  // base: 속도 제어. no-op 목표 = 0 (joint_controller.py:386)
  if (!s.goal_set[G_BASE]) {
    for (int k = 0; k < 3; ++k) s.base_goal[k] = 0.0f;
    s.goal_set[G_BASE] = 1;
  }
  for (int k = 0; k < 3; ++k) {
    const int d = c.base_dof[k];
    out.vel[d] = np_clip(s.base_goal[k], c.vel_lo[d], c.vel_hi[d]);  // clip_control (속도는 한계 없는 관절 예외 없음)
    out.set_vel[d] = 1;
  }
  // 위치 제어 관절: u = target → clip_control (한계 없는 관절은 되돌림) → 위치 목표 u, 속도 목표 u * 0
  auto pos_group = [&](int g, const int* dof, int n, float* goal) {
    if (!s.goal_set[g]) {  // no-op: 지금 관절 위치 (clip 없이)
      for (int k = 0; k < n; ++k) goal[k] = joint_pos[dof[k]];
      s.goal_set[g] = 1;
    }
    for (int k = 0; k < n; ++k) {
      const int d = dof[k];
      const float u = goal[k];
      const float cl = c.has_limit[d] ? np_clip(u, c.pos_lo[d], c.pos_hi[d]) : u;
      out.pos[d] = cl;
      out.vel[d] = cl * 0.0f;
      out.set_pos[d] = 1;
      out.set_vel[d] = 1;
    }
  };
  pos_group(G_TRUNK, c.trunk_dof, 4, s.trunk_goal);
  for (int a = 0; a < 2; ++a) {
    pos_group(a == 0 ? G_ARM_L : G_ARM_R, c.arm_dof[a], 7, s.arm_goal[a]);
    const int g = a == 0 ? G_GRIP_L : G_GRIP_R;
    if (!s.goal_set[g]) {  // smooth no-op: 두 손가락 위치의 np.mean
      s.grip_goal[a] = (joint_pos[c.grip_dof[a][0]] + joint_pos[c.grip_dof[a][1]]) / 2.0f;
      s.goal_set[g] = 1;
    }
    for (int k = 0; k < 2; ++k) {
      const int d = c.grip_dof[a][k];
      const float u = s.grip_goal[a] * 1.0f;  // target_batch * ones(control_dim)
      const float cl = c.has_limit[d] ? np_clip(u, c.pos_lo[d], c.pos_hi[d]) : u;
      out.pos[d] = cl;
      out.vel[d] = cl * 0.0f;
      out.set_pos[d] = 1;
      out.set_vel[d] = 1;
    }
  }
}

// 보조 잡기가 보는 "지금 잡으려 하는가" (robot.py:845~862, grasping_direction = lower, 위치 제어):
//   applying_grasp = any(control < 관절 상한)   control = 마지막으로 보낸 손가락 위치 목표
OEHD bool applying_grasp(const R1ProConfig& c, int arm, const DriveTargets& last) {
  bool any = false;
  for (int k = 0; k < 2; ++k) {
    const int d = c.grip_dof[arm][k];
    any = any || (last.pos[d] < c.pos_hi[d]);
  }
  return any;
}

}  // namespace ctrl
}  // namespace omni
}  // namespace eng
