// OmniGibson 물체 상태 (손으로 짠 C++, EHD). 원본 warp 커널을 한 스레드(칸) 단위 함수로 옮겼다.
// 층 1 은 이 함수들을 칸마다 순서대로 부르고, 층 2 는 CUDA 스레드마다 부른다(원자 연산 자리는 순서와 무관한 OR/min/max).
//   AABB        object_states/aabb.py:12 _aabb_init_kernel, utils/usd_utils.py:478 _aabb_reduce_kernel, :515 fallback
//   Open        object_states/open_state.py:16 _open_update_kernel
//   접촉 행렬    utils/usd_utils.py:127 _body_awake_at_step, :189 _update_contact_matrices_kernel, :260 _update_body_transforms_kernel
//   Touching    object_states/touching.py (obj_to_col, pair, finalize)
//   Inside      object_states/inside.py (inv_world, aabb_prefilter, halfspace_test, mesh_reduce, finalize)
//   Adjacency   object_states/adjacency.py:24 (+ warp/native/mesh.h:1842 mesh_query_ray_anyhit, intersect.h:317 woop)
//   OnTop/Under/NextTo  on_top.py, under.py, next_to.py
// 수 연산 축약(FMA) 규칙은 core/omni/warp_f32.h 머리말.
#pragma once
#include <cstdint>

#include "core/omni/warp_f32.h"

namespace eng {
namespace omni {
namespace st {

using wf::M44;

// ---------------- AABB ----------------
OEHD void aabb_init(float* row6) {
  const float inf = wf::bitsf(0x7f800000u);
  row6[0] = inf, row6[1] = inf, row6[2] = inf;
  row6[3] = -inf, row6[4] = -inf, row6[5] = -inf;
}
// 꼭짓점 하나를 세계 좌표로 (PTX: world_i = fma(p2,M_i2, fma(p0,M_i0, p1*M_i1)) + M_i3)
OEHD void aabb_vertex_world(const M44& M, const float pt[3], float w[3]) { wf::transform_point(M, pt, w); }
// warp atomic_min/max (builtin.h:1596/1659): val < old 일 때만 바꿈 (부호 있는 0 끼리는 먼저 온 값이 남는다 — GPU 순서 비결정)
OEHD void atomic_min_f(float* a, float v) {
  if (v < *a) *a = (*a < v ? *a : v);
}
OEHD void atomic_max_f(float* a, float v) {
  if (v > *a) *a = (*a > v ? *a : v);
}
OEHD void aabb_accumulate(float* row6, const float w[3]) {
  atomic_min_f(&row6[0], w[0]);
  atomic_min_f(&row6[1], w[1]);
  atomic_min_f(&row6[2], w[2]);
  atomic_max_f(&row6[3], w[0]);
  atomic_max_f(&row6[4], w[1]);
  atomic_max_f(&row6[5], w[2]);
}
// 충돌 메시가 하나도 없는 물체: 바닥 링크 위치를 점 AABB 로
OEHD void aabb_fallback(float* row6, const float* base_pose7) {
  if (wf::isinf_(row6[0])) {
    for (int k = 0; k < 3; ++k) {
      row6[k] = base_pose7[k];
      row6[3 + k] = base_pose7[k];
    }
  }
}

// ---------------- Open ----------------
OEHD uint8_t open_value(const float* jp_row, int max_dof, const uint8_t* mask, const float* th1, const float* d1,
                        const float* th2, const float* d2, uint8_t both_sides) {
  uint8_t any1 = 0, any2 = 0;
  for (int d = 0; d < max_dof; ++d) {
    if (mask[d] != 0) {
      const float p = jp_row[d];
      if ((p - th1[d]) * d1[d] > 0.0f) any1 = 1;
      if ((p - th2[d]) * d2[d] > 0.0f) any2 = 1;
    }
  }
  return both_sides ? (uint8_t)(any1 & any2) : any1;
}

// ---------------- 접촉 행렬 (RigidContactAPI.update) ----------------
// all_tf (N,B,7), prev_tf (B,7), net (N,R,3), b2r (B,)
struct ContactIn {
  const float* all_tf;
  const float* prev_tf;
  const float* net;
  const int32_t* b2r;
  int B, R;
  float pos_eps, ori_eps;
};
OEHD bool body_awake_at_step(const ContactIn& c, int i, int b) {
  const float* t0 = (i == 0) ? (c.prev_tf + b * 7) : (c.all_tf + ((size_t)(i - 1) * c.B + b) * 7);
  const float* t1 = c.all_tf + ((size_t)i * c.B + b) * 7;
  // PTX: max(|dx|,|dy|,|dz|) > eps  (= 셋 중 하나라도 > eps, NaN 은 거짓)
  const float dx = wf::wabs(t1[0] - t0[0]), dy = wf::wabs(t1[1] - t0[1]), dz = wf::wabs(t1[2] - t0[2]);
  const bool pos_changed = dx > c.pos_eps || dy > c.pos_eps || dz > c.pos_eps;
  // qdot = fma(qw0,qw1, fma(qz0,qz1, fma(qx0,qx1, qy0*qy1)))
  const float qdot = wf::fma_(t0[6], t1[6], wf::fma_(t0[5], t1[5], wf::fma_(t0[3], t1[3], t0[4] * t1[4])));
  const bool ori_changed = wf::wabs(qdot) < (1.0f - c.ori_eps);
  bool awake = pos_changed || ori_changed;
  const int r = c.b2r[b];
  if (r >= 0) {
    const float* f = c.net + ((size_t)i * c.R + r) * 3;
    if (f[0] != 0.0f || f[1] != 0.0f || f[2] != 0.0f) awake = true;
  }
  return awake;
}
// (r, c) 한 칸. imp (N,R,C,3). cm/ccm 은 제자리 갱신.
OEHD void contact_update_rc(const ContactIn& in, const float* imp, int C, const int32_t* row_to_rigid,
                            const int32_t* col_to_rigid, int n_steps, int r, int col, uint8_t* cm, uint8_t* ccm) {
  const int row_b = row_to_rigid[r];
  const int col_b = col_to_rigid[col];
  int last_awake = -1;
  uint8_t any_contact = 0;
  for (int i = 0; i < n_steps; ++i) {
    const bool ra = body_awake_at_step(in, i, row_b);
    bool ca = false;
    if (col_b >= 0) ca = body_awake_at_step(in, i, col_b);
    if (ra || ca) {
      last_awake = i;
      const float* v = imp + (((size_t)i * in.R + r) * C + col) * 3;
      if (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f) any_contact = 1;
    }
  }
  const size_t k = (size_t)r * C + col;
  if (last_awake >= 0) {
    const float* v = imp + (((size_t)last_awake * in.R + r) * C + col) * 3;
    ccm[k] = (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f) ? 1 : 0;
    cm[k] = any_contact;
  } else {
    cm[k] = ccm[k];
  }
}
// 몸체 b 의 마지막 깬 스텝 자세를 저장 (contact_update 를 모든 칸에 한 뒤에 부를 것 — prev_tf 를 제자리 갱신)
OEHD void body_transform_update(const ContactIn& in, int n_steps, int b, float* body_tf) {
  int last_awake = -1;
  for (int i = 0; i < n_steps; ++i)
    if (body_awake_at_step(in, i, b)) last_awake = i;
  if (last_awake >= 0)
    for (int k = 0; k < 7; ++k) body_tf[b * 7 + k] = in.all_tf[((size_t)last_awake * in.B + b) * 7 + k];
}

// ---------------- Touching ----------------
// 판 하나: obj_to_col[i,c] = OR_r (row_mask[i,r] & cm[r,c]); pair[i,j] |= OR_c (obj_to_col[i,c] & col_mask[j,c])
OEHD uint8_t touching_obj_to_col(const uint8_t* row_mask, const uint8_t* cm, int R, int C, int i, int c) {
  for (int r = 0; r < R; ++r)
    if (row_mask[(size_t)i * R + r] && cm[(size_t)r * C + c]) return 1;
  return 0;
}
OEHD uint8_t touching_finalize(const int32_t* pair, int N, int i, int j) {  // pair: 판 하나 (N,N)
  if (i == j) return 0;
  return (pair[(size_t)i * N + j] > 0 || pair[(size_t)j * N + i] > 0) ? 1 : 0;
}

// ---------------- Inside ----------------
OEHD void aabb_center(const float* a6, float c[3]) {
  c[0] = (a6[0] + a6[3]) * 0.5f;
  c[1] = (a6[1] + a6[4]) * 0.5f;
  c[2] = (a6[2] + a6[5]) * 0.5f;
}
OEHD int32_t inside_prefilter(const float* aabb_s, const int32_t* aidx, int i, int j) {  // aabb_s: 판 하나 (N_aabb,6)
  if (i == j) return 0;
  const int ai = aidx[i], aj = aidx[j];
  if (ai < 0 || aj < 0) return 0;
  float c[3];
  aabb_center(aabb_s + ai * 6, c);
  const float* b = aabb_s + aj * 6;
  const bool in = c[0] >= b[0] && c[0] <= b[3] && c[1] >= b[1] && c[1] <= b[4] && c[2] >= b[2] && c[2] <= b[5];
  return in ? 1 : 0;
}
OEHD M44 inside_inv_world(const M44& pose_parent, const M44& inv_local_w_scale) {
  return wf::mat44_mul(inv_local_w_scale, wf::rigid_inverse(pose_parent));
}
// 면 하나 반공간 시험: 참이면 "밖" (outside_flag 를 1 로)
OEHD bool inside_face_outside(const M44& inv_world, const float c_world[3], const float fc[3], const float fn[3]) {
  float pl[3];
  wf::transform_point(inv_world, c_world, pl);
  const float d[3] = {pl[0] - fc[0], pl[1] - fc[1], pl[2] - fc[2]};
  return wf::dot3(d, fn) >= 0.0f;
}

OEHD uint8_t inside_finalize(const int32_t* pair, int N, int i, int j) {  // pair: 판 하나 (N,N)
  if (i == j) return 0;
  return pair[(size_t)i * N + j] > 0 ? 1 : 0;
}

// ---------------- Adjacency (광선) ----------------
// warp intersect.h:307 diff_product (소스에 fmaf 명시)
OEHD float diff_product(float a, float b, float c, float d) {
  const float cd = c * d;
  const float diff = wf::fma_(a, b, -cd);
  const float err = wf::fma_(-c, d, cd);
  return diff + err;
}
OEHD int longest_axis_abs(const float v[3]) {  // warp vec.h longest_axis(abs): 엄격한 > 로 첫 최대
  float lmax = wf::wabs(v[0]);
  int ret = 0;
  for (int i = 1; i < 3; ++i) {
    const float l = wf::wabs(v[i]);
    if (l > lmax) {
      ret = i;
      lmax = l;
    }
  }
  return ret;
}
// warp intersect.h:317 intersect_ray_tri_woop. 축약(PTX): Ax = fma(-Sx, A[kz], A[kx]) 등 (ptxas 가 mul+sub 를 묶음),
// det = (U+V)+W, T = fma(W, Cz, fma(U, Az, V*Bz)), t = rcp(det) * T
OEHD bool ray_tri_woop(const float p[3], const float dir[3], const float a[3], const float b[3], const float c[3],
                       float* t_out) {
  const int kz = longest_axis_abs(dir);
  int kx = kz + 1;
  if (kx == 3) kx = 0;
  int ky = kx + 1;
  if (ky == 3) ky = 0;
  if (dir[kz] < 0.0f) {
    const int tmp = kx;
    kx = ky;
    ky = tmp;
  }
  const float Sx = wf::div_rn(dir[kx], dir[kz]);
  const float Sy = wf::div_rn(dir[ky], dir[kz]);
  const float Sz = wf::rcp_rn(dir[kz]);
  const float A[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
  const float B[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
  const float C[3] = {c[0] - p[0], c[1] - p[1], c[2] - p[2]};
  const float Ax = wf::fma_(-Sx, A[kz], A[kx]);
  const float Ay = wf::fma_(-Sy, A[kz], A[ky]);
  const float Bx = wf::fma_(-Sx, B[kz], B[kx]);
  const float By = wf::fma_(-Sy, B[kz], B[ky]);
  const float Cx = wf::fma_(-Sx, C[kz], C[kx]);
  const float Cy = wf::fma_(-Sy, C[kz], C[ky]);
  float U = diff_product(Cx, By, Cy, Bx);
  float V = diff_product(Ax, Cy, Ay, Cx);
  float W = diff_product(Bx, Ay, By, Ax);
  if (U == 0.0f || V == 0.0f || W == 0.0f) {
    U = (float)((double)Cx * (double)By - (double)Cy * (double)Bx);
    V = (float)((double)Ax * (double)Cy - (double)Ay * (double)Cx);
    W = (float)((double)Bx * (double)Ay - (double)By * (double)Ax);
  }
  if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) return false;
  const float det = (U + V) + W;
  if (det == 0.0f) return false;
  const float Az = Sz * A[kz];
  const float Bz = Sz * B[kz];
  const float Cz = Sz * C[kz];
  const float T = wf::fma_(W, Cz, wf::fma_(U, Az, V * Bz));
  const uint32_t det_sign = wf::fbits(det) & 0x80000000u;
  if (wf::bitsf(wf::fbits(T) ^ det_sign) < 0.0f) return false;
  const float rcp_det = wf::rcp_rn(det);
  *t_out = rcp_det * T;
  return true;
}
// 광선을 링크 좌표로: origin_local_i = dot3((P_0i,P_1i,P_2i), o) - s_i, dir_local_i = fma(s_i, -0, dot3(.., d))
OEHD void adjacency_ray_local(const M44& P, const float o_w[3], const float d_w[3], float o_l[3], float d_l[3]) {
  float s[3];
  wf::rigid_inverse_s(P, s);
  for (int i = 0; i < 3; ++i) {
    const float col[3] = {P(0, i), P(1, i), P(2, i)};
    o_l[i] = wf::dot3(col, o_w) - s[i];
    d_l[i] = wf::fma_(s[i], -0.0f, wf::dot3(col, d_w));
  }
}
// mesh_query_ray_anyhit 을 삼각형 전수로 (BVH 는 경계를 1e-3 넓혀 보수적이라 유효 적중을 버리지 않는다 — 결과 같음, 문서 참고)
OEHD bool mesh_anyhit_bruteforce(const float* pts, const int32_t* tri, int n_tri, const float o[3], const float d[3],
                                 float max_t) {
  for (int f = 0; f < n_tri; ++f) {
    const float* a = pts + tri[f * 3 + 0] * 3;
    const float* b = pts + tri[f * 3 + 1] * 3;
    const float* c = pts + tri[f * 3 + 2] * 3;
    float t;
    if (ray_tri_woop(o, d, a, b, c, &t))
      if (t < max_t && t >= 0.0f) return true;
  }
  return false;
}

// ---------------- ToggledOn (object_states/toggle.py) ----------------
// 1) 손가락이 물체에 닿았나: usd_utils.py:74 _is_in_contact_batch_kernel (with-mask 방식, 질의 행 마스크 하나를 모든 물체가 공유)
OEHD int32_t toggle_contact(const uint8_t* query_row_mask, const uint8_t* cm, int R, int C, const uint8_t* with_mask_o) {
  for (int r = 0; r < R; ++r) {
    if (!query_row_mask[r]) continue;
    for (int c = 0; c < C; ++c)
      if (with_mask_o[c] && cm[(size_t)r * C + c]) return 1;
  }
  return 0;
}
// 2) warp intersect.h:58 closest_point_to_triangle → (u, v). ptxas 축약(SASS): 두 곱의 차는 앞 곱을 묶음
//    vc = fma(d1,d4,-(d3*d2)), vb = fma(d5,d2,-(d1*d6)), va = fma(d3,d6,-(d5*d4)), 안쪽 u = fma(-vc, denom, 1-v)
OEHD void closest_point_uv(const float a[3], const float b[3], const float c[3], const float p[3], float* uo, float* vo) {
  const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const float ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const float d1 = wf::dot3(ab, ap), d2 = wf::dot3(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) { *uo = 1.0f; *vo = 0.0f; return; }
  const float bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const float d3 = wf::dot3(ab, bp), d4 = wf::dot3(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) { *uo = 0.0f; *vo = 1.0f; return; }
  const float vc = wf::fma_(d1, d4, -(d3 * d2));
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = wf::div_rn(d1, d1 - d3);
    *uo = 1.0f - v;
    *vo = v;
    return;
  }
  const float cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const float d5 = wf::dot3(ab, cp), d6 = wf::dot3(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) { *uo = 0.0f; *vo = 0.0f; return; }
  const float vb = wf::fma_(d5, d2, -(d1 * d6));
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = wf::div_rn(d2, d2 - d6);
    *uo = 1.0f - w;
    *vo = 0.0f;
    return;
  }
  const float va = wf::fma_(d3, d6, -(d5 * d4));
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float w = wf::div_rn(d4 - d3, (d4 - d3) + (d5 - d6));
    const float v = 1.0f - w;
    *uo = (1.0f - v) - w;
    *vo = v;
    return;
  }
  const float denom = wf::rcp_rn((va + vb) + vc);
  const float v = vb * denom;
  *uo = wf::fma_(-vc, denom, 1.0f - v);
  *vo = v;
}
OEHD void cross3(const float a[3], const float b[3], float o[3]) {  // 앞 곱을 묶음 (SASS)
  o[0] = wf::fma_(a[1], b[2], -(a[2] * b[1]));
  o[1] = wf::fma_(a[2], b[0], -(a[0] * b[2]));
  o[2] = wf::fma_(a[0], b[1], -(a[1] * b[0]));
}
// 3) warp mesh.h:512 mesh_query_point_no_sign 의 참/거짓 = 가늘지 않은 삼각형 중 거리² < max_dist² 가 하나라도 있나 (삼각형 전수)
OEHD bool mesh_point_within(const float* pts, const int32_t* tri, int n_tri, const float x[3], float max_dist) {
  float min_d2 = max_dist * max_dist;
  const float lim = min_d2;
  for (int f = 0; f < n_tri; ++f) {
    const float* p = pts + tri[f * 3 + 0] * 3;
    const float* q = pts + tri[f * 3 + 1] * 3;
    const float* r = pts + tri[f * 3 + 2] * 3;
    const float e0[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
    const float e1[3] = {r[0] - p[0], r[1] - p[1], r[2] - p[2]};
    const float e2[3] = {r[0] - q[0], r[1] - q[1], r[2] - q[2]};
    float n[3];
    cross3(e0, e1, n);
    const float len = wf::sqrt_rn(wf::dot3(n, n));
    const float den = (wf::dot3(e0, e0) + wf::dot3(e1, e1)) + wf::dot3(e2, e2);
    if (wf::div_rn(len, den) < 1.e-6f) continue;  // 가는 삼각형 건너뜀
    float u, v;
    closest_point_uv(p, q, r, x, &u, &v);
    const float w = (1.0f - u) - v;
    float d[3];
    for (int k = 0; k < 3; ++k) {
      const float ck = wf::fma_(w, r[k], wf::fma_(u, p[k], v * q[k]));
      d[k] = ck - x[k];
    }
    const float d2 = wf::dot3(d, d);
    if (d2 < min_d2) min_d2 = d2;
  }
  return min_d2 < lim;
}
// 4) _check_overlap_kernel 한 쌍 (표식, 손가락 링크): 표식 중심 = P_부모 * (offset,1), 손가락 좌표 = R_f^T x - s
OEHD bool toggle_marker_overlap(const M44& P_parent, const float offset[3], float radius, const M44& P_finger,
                                const float* pts, const int32_t* tri, int n_tri) {
  float w[3];
  wf::transform_point(P_parent, offset, w);
  float s[3], xl[3];
  wf::rigid_inverse_s(P_finger, s);
  for (int i = 0; i < 3; ++i) {
    const float col[3] = {P_finger(0, i), P_finger(1, i), P_finger(2, i)};
    xl[i] = wf::fma_(-s[i], 1.0f, wf::dot3(col, w));  // SASS: FFMA(-s, w4=1, dot)
  }
  return mesh_point_within(pts, tri, n_tri, xl, radius);
}
// 5) _set_toggle_value_kernel: mask==2 인 동안 dt 를 더하고, 처음 문턱을 넘는 스텝에 값을 뒤집는다 (float32)
OEHD void toggle_set_value(uint8_t* value, int32_t* mask, float* time, float threshold, float dt) {
  const int eligible = (*mask == 2) ? 1 : 0;
  const float prev = *time;
  *time = eligible ? prev + dt : 0.0f;
  const float now = *time;
  int flip = 0;
  if (prev < threshold && now >= threshold) flip = eligible;
  *mask = 0;
  if (flip) *value = (uint8_t)(1 - *value);
}

// ---------------- OnTop / Under / NextTo (칸 하나) ----------------
// adj: 판 하나 (Na,Na,K), touch: 판 하나 (Nt,Nt)
OEHD uint8_t on_top_value(const uint8_t* touch, int Nt, const uint8_t* adj, int Na, int K, const int32_t* tidx,
                          const int32_t* jidx, int i, int j) {
  if (i == j) return 0;
  const int ti = tidx[i], tj = tidx[j], ai = jidx[i], aj = jidx[j];
  if (ti < 0 || tj < 0 || ai < 0 || aj < 0) return 0;
  const bool touching = touch[(size_t)ti * Nt + tj] != 0;
  const uint8_t* aij = adj + ((size_t)ai * Na + aj) * K;
  const bool above = aij[0] != 0, below = aij[1] != 0;
  return (touching && below && !above) ? 1 : 0;
}
OEHD uint8_t under_value(const uint8_t* adj, int Na, int K, const int32_t* jidx, int i, int j) {
  if (i == j) return 0;
  const int ai = jidx[i], aj = jidx[j];
  if (ai < 0 || aj < 0) return 0;
  const uint8_t* aij = adj + ((size_t)ai * Na + aj) * K;
  const uint8_t* aji = adj + ((size_t)aj * Na + ai) * K;
  const bool other_above = aij[0] != 0, other_below = aij[1] != 0, self_above = aji[0] != 0;
  return (other_above && !other_below && !self_above) ? 1 : 0;
}
// next_to.py: gap = max(0, max(lo_i,lo_j) - min(hi_i,hi_j)), 거리 = sqrt(fma(gz,gz, fma(gx,gx, gy*gy))),
// 문턱 = div.rn(순서대로 더한 치수 6개, 3) * float32(1/6)
OEHD uint8_t next_to_value(const float* aabb_s, const uint8_t* adj, int Na, int K, const int32_t* aidx,
                           const int32_t* jidx, int i, int j, int k_start, int k_end) {
  if (i == j) return 0;
  const int a_i = aidx[i], a_j = aidx[j], ad_i = jidx[i], ad_j = jidx[j];
  if (a_i < 0 || a_j < 0 || ad_i < 0 || ad_j < 0) return 0;
  const float* bi = aabb_s + a_i * 6;
  const float* bj = aabb_s + a_j * 6;
  float g[3];
  for (int k = 0; k < 3; ++k) g[k] = wf::wmax(0.0f, wf::wmax(bi[k], bj[k]) - wf::wmin(bi[3 + k], bj[3 + k]));
  const float dist = wf::sqrt_rn(wf::fma_(g[2], g[2], wf::fma_(g[0], g[0], g[1] * g[1])));
  float sum = (bi[3] - bi[0]) + (bi[4] - bi[1]);
  sum = sum + (bi[5] - bi[2]);
  sum = sum + (bj[3] - bj[0]);
  sum = sum + (bj[4] - bj[1]);
  sum = sum + (bj[5] - bj[2]);
  const float thr = wf::div_rn(sum, 3.0f) * (float)(1.0 / 6.0);
  if (dist > thr) return 0;
  uint8_t hit = 0;
  const uint8_t* aij = adj + ((size_t)ad_i * Na + ad_j) * K;
  const uint8_t* aji = adj + ((size_t)ad_j * Na + ad_i) * K;
  for (int k = k_start; k < k_end; ++k) {
    if (aij[k]) hit = 1;
    if (aji[k]) hit = 1;
  }
  return hit;
}

}  // namespace st
}  // namespace omni
}  // namespace eng
