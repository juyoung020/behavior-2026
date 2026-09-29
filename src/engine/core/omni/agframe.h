// 보조 잡기 관절 틀 (robots/robot.py:3525 _establish_grasp) 을 손으로 짬 (EHD).
//   parent_frame = T.relative_pose_transform(contact, [0,0,0,1], eef_pos, eef_orn), pos / robot.scale
//   child_frame  = T.relative_pose_transform(contact, [0,0,0,1], link_pos, link_orn), pos / obj.scale
// T.relative_pose_transform (utils/transform_utils.py:950) 은 Linux 에서 torch.compile(inductor CPU) 이다. inductor 가 만든 C++ 는
// -ffp-contract=off -fno-unsafe-math-optimizations 로 짓는다(torch/_inductor/cpp_builder.py:539, config.py:937) → FMA·재배열 없음.
// 그래서 식을 inductor 생성 코드의 연산 순서대로 옮기면 비트가 같다 (TORCH_LOGS=output_code 로 읽음, tests/omni/gen_agframe_ref.py):
//   - 4 원소 제곱합(quat 정규화) = 벡터 줄이기: (q0²+q2²) + (q1²+q3²)          (at::vec::vec_reduce_all, 폭 8/16 같음)
//   - 3 원소 합(norm·sum) = 스칼라 순서: ((0 + a) + b) + c
//   - 정규화는 곱이 아니라 나눗셈 (q / sqrt(n2))
//   - pose_inv @ mat1 은 extern mm (aten.mm) — 후보 셋을 시험해 순서대로 곱·합(FMA 없음, mm_order=1)이 20,000 행 140,000 값 전부 같음
//     (fma 사슬은 11 % 다름). 주의: CPU BLAS 경로라 CPU 종류가 다르면 바뀔 수 있다 (이 PC Zen5 AVX512 에서 확인)
// 원본 (utils/transform_utils.py): pose2mat, quat2mat, pose_inv, relative_pose_transform:950, mat2pose → decompose_mat → mat2quat
#pragma once
#include <cmath>
#include <cstdint>

#include "core/omni/warp_f32.h"

namespace eng {
namespace omni {
namespace agf {

OEHD float sq_reduce4(const float q[4]) {  // vec_reduce_all 의 나비 순서
  const float a = q[0] * q[0], b = q[1] * q[1], c = q[2] * q[2], d = q[3] * q[3];
  return 0.0f + ((a + c) + (b + d));
}
OEHD float sum3(float a, float b, float c) { return ((0.0f + a) + b) + c; }

// quat2mat (정규화 포함) → 3x3 행 우선
OEHD void quat2mat(const float q_in[4], float r[9]) {
  const float n = wf::sqrt_rn(sq_reduce4(q_in));
  const float x = wf::div_rn(q_in[0], n), y = wf::div_rn(q_in[1], n), z = wf::div_rn(q_in[2], n), w = wf::div_rn(q_in[3], n);
  const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, xw = x * w, yw = y * w, zw = z * w;
  r[0] = 1.0f - (yy + zz) * 2.0f;
  r[1] = (xy - zw) * 2.0f;
  r[2] = (xz + yw) * 2.0f;
  r[3] = (xy + zw) * 2.0f;
  r[4] = 1.0f - (xx + zz) * 2.0f;
  r[5] = (yz - xw) * 2.0f;
  r[6] = (xz - yw) * 2.0f;
  r[7] = (yz + xw) * 2.0f;
  r[8] = 1.0f - (xx + yy) * 2.0f;
}
OEHD void pose2mat(const float p[3], const float q[4], float m[16]) {
  float r[9];
  quat2mat(q, r);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m[i * 4 + j] = r[i * 3 + j];
    m[i * 4 + 3] = p[i];
  }
  m[12] = m[13] = m[14] = 0.0f;
  m[15] = 1.0f;
}
// pose_inv: R^T, t' = ((-R_0i t0) + (-R_1i t1)) + (-R_2i t2)
OEHD void pose_inv(const float m[16], float o[16]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) o[i * 4 + j] = m[j * 4 + i];
    o[i * 4 + 3] = ((-m[0 * 4 + i] * m[3]) + (-m[1 * 4 + i] * m[7])) + (-m[2 * 4 + i] * m[11]);
  }
  o[12] = o[13] = o[14] = 0.0f;
  o[15] = 1.0f;
}
// 4x4 @ 4x4 (extern mm). order 1 = 공식과 같음(순서대로 곱·합). 0·2 는 비교용 후보 (fma 사슬)
OEHD void mm4(const float a[16], const float b[16], float c[16], int order) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float s;
      if (order == 0) {
        s = a[i * 4] * b[j];
        for (int k = 1; k < 4; ++k) s = wf::fma_(a[i * 4 + k], b[k * 4 + j], s);
      } else if (order == 1) {
        s = a[i * 4] * b[j];
        for (int k = 1; k < 4; ++k) s = s + a[i * 4 + k] * b[k * 4 + j];
      } else {
        s = 0.0f;
        for (int k = 0; k < 4; ++k) s = wf::fma_(a[i * 4 + k], b[k * 4 + j], s);
      }
      c[i * 4 + j] = s;
    }
}
// mat2quat (행렬 m 행 우선 3x3)
OEHD void mat2quat(const float m[9], float q[4]) {
  const float m00 = m[0], m01 = m[1], m02 = m[2], m10 = m[3], m11 = m[4], m12 = m[5], m20 = m[6], m21 = m[7], m22 = m[8];
  const float trace = (m00 + m11) + m22;
  const bool tp = trace > 0.0f;
  const bool c1 = (m00 > m11) && (m00 > m22) && !tp;
  const bool c2 = (m11 > m22) && !(tp || c1);
  const bool c3 = !(tp || c1 || c2);
  float sq = tp ? wf::sqrt_rn(trace + 1.0f) * 2.0f : 0.0f;
  float qw = tp ? 0.25f * sq : 0.0f;
  float qx = tp ? wf::div_rn(m21 - m12, sq) : 0.0f;
  float qy = tp ? wf::div_rn(m02 - m20, sq) : 0.0f;
  float qz = tp ? wf::div_rn(m10 - m01, sq) : 0.0f;
  if (c1) {
    sq = wf::sqrt_rn(((1.0f + m00) - m11) - m22) * 2.0f;
    qw = wf::div_rn(m21 - m12, sq);
    qx = 0.25f * sq;
    qy = wf::div_rn(m01 + m10, sq);
    qz = wf::div_rn(m02 + m20, sq);
  }
  if (c2) {
    sq = wf::sqrt_rn(((1.0f + m11) - m00) - m22) * 2.0f;
    qw = wf::div_rn(m02 - m20, sq);
    qx = wf::div_rn(m01 + m10, sq);
    qy = 0.25f * sq;
    qz = wf::div_rn(m12 + m21, sq);
  }
  if (c3) {
    sq = wf::sqrt_rn(((1.0f + m22) - m00) - m11) * 2.0f;
    qw = wf::div_rn(m10 - m01, sq);
    qx = wf::div_rn(m02 + m20, sq);
    qy = wf::div_rn(m12 + m21, sq);
    qz = 0.25f * sq;
  }
  const float v[4] = {qx, qy, qz, qw};
  const float n = wf::sqrt_rn(sq_reduce4(v));
  for (int i = 0; i < 4; ++i) q[i] = wf::div_rn(v[i], n);
}
// mat2pose → decompose_mat (transform_utils.py decompose_mat): 위치 = 마지막 열 / H33, 회전 = 그람-슈미트 뒤 mat2quat
OEHD void mat2pose(const float H[16], float pos[3], float quat[4]) {
  const float d = H[15];
  float row[3][3];
  for (int i = 0; i < 3; ++i) {
    pos[i] = wf::div_rn(H[i * 4 + 3], d);
    for (int j = 0; j < 3; ++j) row[i][j] = wf::div_rn(H[j * 4 + i], d);  // M = H^T / d
  }
  const float s0 = wf::sqrt_rn(sum3(row[0][0] * row[0][0], row[0][1] * row[0][1], row[0][2] * row[0][2]));
  for (int j = 0; j < 3; ++j) row[0][j] = wf::div_rn(row[0][j], s0);
  const float sh0 = sum3(row[0][0] * row[1][0], row[0][1] * row[1][1], row[0][2] * row[1][2]);
  for (int j = 0; j < 3; ++j) row[1][j] = row[1][j] - row[0][j] * sh0;
  const float s1 = wf::sqrt_rn(sum3(row[1][0] * row[1][0], row[1][1] * row[1][1], row[1][2] * row[1][2]));
  for (int j = 0; j < 3; ++j) row[1][j] = wf::div_rn(row[1][j], s1);
  const float sh1 = sum3(row[0][0] * row[2][0], row[0][1] * row[2][1], row[0][2] * row[2][2]);
  for (int j = 0; j < 3; ++j) row[2][j] = row[2][j] - row[0][j] * sh1;
  const float sh2 = sum3(row[1][0] * row[2][0], row[1][1] * row[2][1], row[1][2] * row[2][2]);
  for (int j = 0; j < 3; ++j) row[2][j] = row[2][j] - row[1][j] * sh2;
  const float s2 = wf::sqrt_rn(sum3(row[2][0] * row[2][0], row[2][1] * row[2][1], row[2][2] * row[2][2]));
  for (int j = 0; j < 3; ++j) row[2][j] = wf::div_rn(row[2][j], s2);
  const float cr[3] = {row[1][1] * row[2][2] - row[1][2] * row[2][1], row[1][2] * row[2][0] - row[1][0] * row[2][2],
                       row[1][0] * row[2][1] - row[1][1] * row[2][0]};
  const float dot = sum3(row[0][0] * cr[0], row[0][1] * cr[1], row[0][2] * cr[2]);
  if (dot < 0.0f)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) row[i][j] = -row[i][j];
  float rm[9];  // mat2quat(row^T)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) rm[i * 3 + j] = row[j][i];
  mat2quat(rm, quat);
}
// relative_pose_transform(pos1, quat1, pos0, quat0) = mat2pose(pose_inv(pose2mat(pos0,quat0)) @ pose2mat(pos1,quat1))
OEHD void relative_pose_transform(const float p1[3], const float q1[4], const float p0[3], const float q0[4], float pos[3],
                                  float quat[4], int mm_order = 1) {
  float m0[16], m1[16], i0[16], h[16];
  pose2mat(p0, q0, m0);
  pose2mat(p1, q1, m1);
  pose_inv(m0, i0);
  mm4(i0, m1, h, mm_order);
  mat2pose(h, pos, quat);
}
// _establish_grasp 한쪽 틀: 접촉점(세계) 을 링크 틀로, 관절 틀 방향 [0,0,0,1]. pos 는 척도로 나눔 (eager 나눗셈)
OEHD void grasp_frame(const float contact[3], const float link_pos[3], const float link_q[4], const float scale[3], float pos[3],
                      float quat[4], int mm_order = 1) {
  const float jq[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  relative_pose_transform(contact, jq, link_pos, link_q, pos, quat, mm_order);
  for (int i = 0; i < 3; ++i) pos[i] = wf::div_rn(pos[i], scale[i]);
}

}  // namespace agf
}  // namespace omni
}  // namespace eng
