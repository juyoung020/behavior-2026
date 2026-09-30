// 평가기 관측 가공 (손으로, 공식과 비트 동일): proprio 61 (robots/robot.py:1541 _get_proprioception_dict, eval_utils.py:121 PROPRIOCEPTION_INDICES R1Pro),
// cam_rel_poses 21 (evaluator.py:576 _preprocess_obs), 카메라 세계 자세 (usd_utils.py:2138 get_world_pose = usdrt Gf, gfmat.h).
// 입력은 엔진 물리(float32): 관절 위치·속도(텐서 API dof 순서), 바닥(base_footprint) 링크·손끝 링크·카메라가 붙은 링크 자세.
#pragma once
#include <cstdint>

#include "core/omni/mkl_trig.h"
#include "core/omni/agframe.h"
#include "core/omni/gfmat.h"

namespace eng {
namespace omni {
namespace obs {

struct R1ProIdx {           // s1_setup.txt group 줄 (텐서 API dof 번호)
  int base[3];              // 바닥 제어 dof (x, y, rz)
  int yaw_dof;              // base_idx[5] (바닥 가상 관절 6 개 중 rz)
  int trunk[4];
  int arm[2][7];            // 왼쪽, 오른쪽
  int grip[2][2];
};

struct Pose {
  float p[3];
  float q[4];  // x, y, z, w
};

// proprio 61: base_qvel 3 | 왼팔 qpos 7, qvel 7 | 왼 손끝 pos 3, quat 4 | 왼 집게 qpos 2, qvel 2 | 오른쪽 같은 순서 | 몸통 qpos 4, qvel 4
inline void proprio_r1pro(const R1ProIdx& ix, const float* jp, const float* jv, const Pose& base, const Pose eef[2], float out[61]) {
  int k = 0;
  // _get_base_qvel_for_proprioception (robot.py:1605): yaw = base_qpos_6dof[5], th.cos/sin = MKL VML HA (mkl_trig.h)
  const float yaw = jp[ix.yaw_dof];
  const float c = mkl::cosf(yaw), s = mkl::sinf(yaw);
  const float v0 = jv[ix.base[0]], v1 = jv[ix.base[1]], v2 = jv[ix.base[2]];
  out[k++] = c * v0 + s * v1;
  out[k++] = -s * v0 + c * v1;
  out[k++] = v2;
  for (int arm = 0; arm < 2; ++arm) {
    for (int d = 0; d < 7; ++d) out[k++] = jp[ix.arm[arm][d]];
    for (int d = 0; d < 7; ++d) out[k++] = jv[ix.arm[arm][d]];
    float rp[3], rq[4];
    agf::relative_pose_transform(eef[arm].p, eef[arm].q, base.p, base.q, rp, rq);
    for (int i = 0; i < 3; ++i) out[k++] = rp[i];
    for (int i = 0; i < 4; ++i) out[k++] = rq[i];
    for (int d = 0; d < 2; ++d) out[k++] = jp[ix.grip[arm][d]];
    for (int d = 0; d < 2; ++d) out[k++] = jv[ix.grip[arm][d]];
  }
  for (int d = 0; d < 4; ++d) out[k++] = jp[ix.trunk[d]];
  for (int d = 0; d < 4; ++d) out[k++] = jv[ix.trunk[d]];
}

// 카메라 prim 세계 자세: fabric 세계 행렬 = local(사슬, double, 카메라 쪽부터) × 링크 세계 행렬(PhysX 자세 → SetRotate 식) → usdrt 분해
inline Pose camera_world(const double* local16, int n_local, const Pose& link) {
  gf::M4 M = gf::from_physx_pose(link.p, link.q);
  for (int i = 0; i < n_local; ++i) {  // obs_engine: for c in chain: M = local @ M (사슬은 카메라 prim 부터 위로)
    gf::M4 L;
    for (int k = 0; k < 16; ++k) L.m[k / 4][k % 4] = local16[i * 16 + k];
    M = gf::mul(L, M);
  }
  Pose o;
  gf::world_pose_f32(M, o.p, o.q);
  return o;
}

// cam_rel_poses: 카메라마다 relative_pose_transform(cam, base) 의 pos 3 + quat 4
inline void cam_rel_pose(const Pose& cam, const Pose& base, float out7[7]) {
  agf::relative_pose_transform(cam.p, cam.q, base.p, base.q, out7, out7 + 3);
}

}  // namespace obs
}  // namespace omni
}  // namespace eng
