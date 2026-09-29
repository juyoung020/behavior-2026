// 보조 잡기 (assisted grasping, 손으로 짬). 평가기 r1pro.yaml grasping_mode: assisted.
// 원본: robots/robot.py:835 _handle_assisted_grasping, :3130 _calculate_in_hand_object, :3206 _find_gripper_raycast_collisions,
//       :3272 _handle_release_window, :2047 _release_grasp, :3497 _maybe_establish_grasp, :2564 _get_assisted_grasp_joint_type,
//       :3525 _establish_grasp (관절 틀), :2003 _find_gripper_contacts, :1332 _find_finger_contact_position
// 불리는 때: 물리 서브스텝마다 제어기 step 다음, PhysX simulate 전 (simulator.py:1579~1587).
// 이 파일이 옮기는 것 = 판단(언제 잡고 놓는가, 무엇을 잡는가, 어떤 관절인가). 입력으로 받는 것 = 손가락 접촉 목록
//   (RigidContactAPI 접촉 행렬, contact 모듈), 광선 적중 목록(장면 질의, joints 모듈), 링크 자세. 출력 = 관절 생성/삭제 요청.
#pragma once
#include <cmath>
#include <cstdint>

#include "core/omni/npf32.h"

namespace eng {
namespace omni {
namespace ag {

constexpr int kMaxCand = 64;

// robot.py:76~82 매크로
struct Params {
  double physics_dt = 1.0 / 120.0;  // og.sim.get_physics_dt() = 1 / timeStepsPerSecond (파이썬 double)
  double grasp_window = 0.3;        // m.GRASP_WINDOW
  double release_window = 1.0 / 30.0;  // m.RELEASE_WINDOW
  float mass_threshold = 10.0f;     // m.ASSIST_GRASP_MASS_THRESHOLD (파이썬 float 와 torch float32 질량 비교)
};

// 팔 하나의 상태. None = -1
struct ArmState {
  int32_t obj_in_hand = -1;      // _ag_obj_in_hand (물체 번호)
  int32_t link_in_hand = -1;
  int32_t release_counter = -1;  // _ag_release_counter
  int32_t grasp_counter = -1;    // _ag_grasp_counter
  uint8_t has_constraint = 0;    // _ag_obj_constraints 가 있는가
};

// _calculate_in_hand_object 입력 (팔 하나, 이 서브스텝)
struct CandidateIn {
  int n;                       // 손가락과 닿은 (로봇 아닌) 링크 수 = _find_gripper_contacts 의 contact_data
  int32_t link[kMaxCand];      // 링크 번호 (장면 전체)
  uint8_t fingers[kMaxCand];   // 이 팔 손가락 중 몇 개가 닿았나 (robot_contact_links[path] ∩ finger paths 크기)
  uint8_t ray_hit[kMaxCand];   // _find_gripper_raycast_collisions 결과에 들었나 (assisted 모드는 교집합만 후보)
  int32_t obj[kMaxCand];       // 링크가 속한 등록 물체 (없으면 -1 → object_registry 가 None)
  uint8_t dynamic[kMaxCand];   // RigidDynamicPrim 인가 (고정/운동학 링크는 제외)
  float pos[kMaxCand][3];      // 링크 위치 (get_position_orientation()[0])
};

// th.norm(3-벡터) = sqrt(fma(v2,v2, fma(v1,v1, v0*v0)))  (torch CPU, 공식 파이썬으로 확인: gen_ag_ref.py)
OEHD float torch_norm3(float x, float y, float z) {
  const float s = f32fma(z, z, f32fma(y, y, x * x));
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(s);
#else
  return std::sqrt(s);
#endif
}

// 잡을 물체 찾기. 없으면 -1, 있으면 후보 번호를 돌려준다.
// 파이썬: 후보 = 접촉 ∩ 광선 (집합) → 등록 물체의 동적 링크만 → 집게 중심(eef 위치)까지 거리로 정렬 → 첫째 →
//         손가락 둘 이상이 닿았나 → 물체.  (거리가 완전히 같은 후보끼리는 파이썬 set 순서라 비결정 — 여기선 앞 번호)
OEHD int calculate_in_hand(const CandidateIn& c, const float eef_pos[3]) {
  int best = -1;
  float best_d = 0.0f;
  for (int i = 0; i < c.n; ++i) {
    if (!c.ray_hit[i]) continue;
    if (c.obj[i] < 0 || !c.dynamic[i]) continue;
    const float d = torch_norm3(c.pos[i][0] - eef_pos[0], c.pos[i][1] - eef_pos[1], c.pos[i][2] - eef_pos[2]);
    if (best < 0 || d < best_d) {  // sorted(): 작은 것이 먼저, 같으면 먼저 들어온 것
      best = i;
      best_d = d;
    }
  }
  if (best < 0) return -1;
  if (c.fingers[best] < 2) return -1;  // touching_at_least_two_fingers (assisted)
  return best;
}

enum Event : uint8_t { EV_NONE = 0, EV_RELEASE = 1, EV_TRY_GRASP = 2 };

// _handle_assisted_grasping 의 팔 하나. EV_TRY_GRASP 이면 호출자가 _maybe_establish_grasp 를 한다
// (관절 종류가 None 이거나 접촉점이 없으면 잡지 않음 → 그때는 establish_failed() 를 부르지 않아도 상태는 같다).
// in_hand: calculate_in_hand 결과 (필요할 때만 쓰임, -1 = None). target 에 잡을 후보 번호.
OEHD Event step_arm(const Params& p, ArmState& s, bool applying_grasp, int in_hand, int* target) {
  *target = -1;
  if (s.obj_in_hand >= 0) {
    if (s.release_counter >= 0) {
      // _handle_release_window
      s.release_counter += 1;
      const double t = (double)s.release_counter * p.physics_dt;
      if (t >= p.release_window) {
        s.obj_in_hand = -1;
        s.link_in_hand = -1;
        s.release_counter = -1;
      }
      return EV_NONE;
    }
    if (!applying_grasp) {
      // _release_grasp: 관절 삭제, 놓기 창 시작
      s.has_constraint = 0;
      s.release_counter = 0;
      return EV_RELEASE;
    }
    return EV_NONE;
  }
  if (applying_grasp) {
    if (s.grasp_counter >= 0) {
      if (in_hand < 0) {
        s.grasp_counter = -1;
      } else {
        s.grasp_counter += 1;
        const double t = (double)s.grasp_counter * p.physics_dt;
        if (t >= p.grasp_window) {
          *target = in_hand;
          s.grasp_counter = -1;
          return EV_TRY_GRASP;
        }
      }
    } else if (in_hand >= 0) {
      s.grasp_counter = 0;
    }
  } else {
    s.grasp_counter = -1;
  }
  return EV_NONE;
}

// _maybe_establish_grasp 가 성공해 관절을 만들었을 때 (_create_assisted_grasp_joint)
OEHD void grasp_established(ArmState& s, int obj, int link) {
  s.obj_in_hand = obj;
  s.link_in_hand = link;
  s.has_constraint = 1;
}

// _get_assisted_grasp_joint_type: 0 = 못 잡음(None), 1 = FixedJoint, 2 = SphericalJoint
//   mass: 링크 질량 (torch float32 → 파이썬 비교는 float32 값 그대로 > 10.0)
//   has_nonfixed_ancestor: 링크에서 뿌리까지 올라가는 관절 중 고정이 아닌 것이 있나 (nx.edge_dfs reverse)
OEHD int joint_type(const Params& p, float mass, bool obj_fixed_base, bool is_root_link, bool has_nonfixed_ancestor) {
  if ((double)mass > (double)p.mass_threshold && !(obj_fixed_base && !is_root_link)) return 0;
  return has_nonfixed_ancestor ? 2 : 1;
}

}  // namespace ag
}  // namespace omni
}  // namespace eng
