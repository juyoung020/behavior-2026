// 시각 입자 도포기(분무기, ParticleApplier) 의 앞뒤. MKL VML cos/sin(dlopen, core/omni/mkl_trig.h) 만 호스트 — 나머지는 PEHD (층 2: test_p4_gpu).
//   A. 투영 원뿔 표본 (particle_modifier.py:1430): th.rand(2,2) → r·θ → 끝점 (r cos θ, r sin θ, -h), 시작점 원점 → 분무기 링크 틀로
//      (get_particle_positions_from_frame: 점*척도 → T.pose2mat(링크) @ 단위(점) 의 3 열, 작은 bmm 순차)
//   B. 무리 척도 (system_base.py:672): 상대 척도면 th.rand(n,3)*(max-min)+min (min·max 는 :634), 아니면 th.rand(n,3)*0+1 → / pow(prod(물체 척도), 1/3)
//      표본 크기 = 척도 * 입자 틀 aabb * 분무기 평균 척도
//   C. 적중에 입자 붙이기 (:1312 → macro_particle_system.py:507, :830): 무리(맞은 물체) 첫 등장 순서로 묶어 만들고,
//      국소 행렬 = inv(링크 scaled_transform) @ [quat2mat(q) | p]
// 연산 순서는 tests/particles/test_applier 대조로 확인.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/omni/agframe.h"
#include "core/omni/mkl_trig.h"
#include "core/particles/trng.h"
#include "core/particles/visual.h"

namespace eng {
namespace particles {

namespace agf = eng::omni::agf;

// 3 원소 곱 (th.prod, 순차), 1/3 거듭제곱 (th.pow 0 차원 + 파이썬 float 지수 → double pow 뒤 float, 대조 2 만 개)
inline float avg_scale3(const float s[3]) { return (float)std::pow((double)((s[0] * s[1]) * s[2]), 1.0 / 3.0); }

// A. 광선 시작·끝 (n = 2). link_p/q = 분무기 링크 PhysX 자세, obj_scale = 분무기 척도, ext = 투영 메시 척도(extents)
//   층 2 (GPU): 난수 뽑기(applier_cone_draw)와 마무리(applier_cone_finish)는 장치에서, 그 사이 cos·sin(MKL VML)만 호스트가 채운다.
PEHD void applier_cone_draw(TorchMT& rng, const float ext[3], float rr[2], float th[2]) {
  const float r = ext[0] / 2.0f;
  float rt[4];
  for (int k = 0; k < 4; ++k) rt[k] = torch_rand_float(rng);
  const float two_pi = (float)(3.141592653589793 * 2.0);
  for (int i = 0; i < 2; ++i) rr[i] = rt[2 * i] * r, th[i] = rt[2 * i + 1] * two_pi;
}
PEHD void applier_cone_finish(const float ext[3], const float link_p[3], const float link_q[4], const float obj_scale[3], const float rr[2], const float cs[2],
                              const float sn[2], float start[2][3], float end[2][3]) {
  const float h = ext[2];
  float pts[4][3];
  for (int i = 0; i < 2; ++i) {
    pts[i][0] = pts[i][1] = pts[i][2] = 0.0f;
    pts[2 + i][0] = rr[i] * cs[i];
    pts[2 + i][1] = rr[i] * sn[i];
    pts[2 + i][2] = -h * 1.0f;
  }
  float M[16];
  agf::pose2mat(link_p, link_q, M);
  for (int i = 0; i < 4; ++i) {
    float p[3], o[3];
    for (int k = 0; k < 3; ++k) p[k] = pts[i][k] * obj_scale[k];
    for (int r = 0; r < 3; ++r) o[r] = ((M[4 * r] * p[0] + M[4 * r + 1] * p[1]) + M[4 * r + 2] * p[2]) + M[4 * r + 3] * 1.0f;
    float* dst = i < 2 ? start[i] : end[i - 2];
    for (int k = 0; k < 3; ++k) dst[k] = o[k];
  }
}
inline void applier_cone_rays(TorchMT& rng, const float ext[3], const float link_p[3], const float link_q[4], const float obj_scale[3], float start[2][3],
                              float end[2][3]) {
  float rr[2], th[2], cs[2], sn[2];
  applier_cone_draw(rng, ext, rr, th);
  for (int i = 0; i < 2; ++i) cs[i] = eng::omni::mkl::cosf(th[i]), sn[i] = eng::omni::mkl::sinf(th[i]);
  applier_cone_finish(ext, link_p, link_q, obj_scale, rr, cs, sn, start, end);
}

// B'. 범위(min·max)를 알 때 (무리 만들 때 정해진 값을 추출한 경우). avg = avg_scale3(무리 물체 척도) — 무리마다 고정이라 층 2 는 호스트 값을 받는다
PEHD void group_scales_avg(TorchMT& rng, const float mn[3], const float mx[3], float avg, int n, float* out) {
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) {
      const float u = torch_rand_float(rng);
      out[3 * i + k] = div_rn(u * (mx[k] - mn[k]) + mn[k], avg);
    }
}
inline void group_scales_mm(TorchMT& rng, const float mn[3], const float mx[3], const float grp_scale[3], int n, float* out) {
  group_scales_avg(rng, mn, mx, avg_scale3(grp_scale), n, out);
}
// 상대 척도 범위 (system_base.py:634). rel 이 아니면 1
PEHD void group_scale_range(bool rel, const float grp_aabb[3], const float tmpl[3], float mn[3], float mx[3]) {
  for (int k = 0; k < 3; ++k) mn[k] = mx[k] = 1.0f;
  if (rel) {
    float a = grp_aabb[0], b = grp_aabb[1], c = grp_aabb[2], t;  // th.median(3) = 가운데 값
    if (a > b) t = a, a = b, b = t;
    if (b > c) t = b, b = c, c = t;
    if (a > b) t = a, a = b, b = t;
    const float med = b;
    float lo = 0.06f * med, hi = 0.1f * med;
    lo = lo < 0.002f ? 0.002f : (lo > 0.02f ? 0.02f : lo);
    hi = hi < 0.01f ? 0.01f : (hi > 0.1f ? 0.1f : hi);
    mn[0] = div_rn(lo, tmpl[0]), mn[1] = div_rn(lo, tmpl[1]), mn[2] = 1.0f;
    mx[0] = div_rn(hi, tmpl[0]), mx[1] = div_rn(hi, tmpl[1]), mx[2] = 1.0f;
  }
}
// B. 무리 척도 n 개. rel = 상대 척도 계. grp_aabb = 무리 물체 aabb 범위(무리 만들 때), tmpl = 입자 틀 aabb 범위, grp_scale = 무리 물체 척도
inline void group_scales(TorchMT& rng, bool rel, const float grp_aabb[3], const float tmpl[3], const float grp_scale[3], int n, float* out) {
  float mn[3], mx[3];
  group_scale_range(rel, grp_aabb, tmpl, mn, mx);
  group_scales_mm(rng, mn, mx, grp_scale, n, out);
}

// 도포기 한 스텝 (ParticleModifier._update, particle_modifier.py:697; 5 스텝마다, 조건 all() 은 앞에서부터 단락):
//   [켜짐 == True] → [한도: 누적 != 한도(1e6)] → [겹침: _check_overlap] → Saturated 재확인 → _modify_particles
// 겹침 질의는 joints 몫이라 overlap(half, center) 로 받는다(켜져 있고 한도 안일 때만 부름 — 공식과 같은 호출 횟수).
// 반환: 이번 스텝에 _modify_particles 를 부르는가. step_counter 는 판마다 0 에서 시작(_current_step), 부를 때마다 갱신.
template <class OverlapFn>
PEHD bool applier_step(int32_t& step_counter, bool toggled, int64_t modified, int64_t limit, const float link_tf[16], const float* hull, int nh,
                         OverlapFn overlap) {
  bool go = false;
  if (step_counter == 0 && toggled && modified != limit) {
    float lo[3], hi[3];
    visual_aabb(link_tf, hull, nh, lo, hi);  // 투영 방식: 여유 없이 visual_aabb 그대로 (particle_modifier.py:450)
    float half[3], ctr[3];
    for (int k = 0; k < 3; ++k) half[k] = (hi[k] - lo[k]) / 2.0f, ctr[k] = (hi[k] + lo[k]) / 2.0f;
    go = overlap(half, ctr) && modified != limit;
  }
  step_counter = (step_counter + 1) % 5;  // m.N_STEPS_PER_APPLICATION
  return go;
}

// C. 붙은 입자 하나의 국소 행렬 (set_particle_position_orientation): inv(link_tf) @ global
PEHD void attach_local_mat(const float link_tf[16], const float p[3], const float q[4], float out[16]) {
  float g[16] = {0};
  float R[9];
  agf::quat2mat(q, R);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) g[4 * i + j] = R[3 * i + j];
    g[4 * i + 3] = p[i];
  }
  g[15] = 1.0f;
  float inv[16];
  inv4_mkl(link_tf, inv);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out[4 * i + j] = ((inv[4 * i] * g[j] + inv[4 * i + 1] * g[4 + j]) + inv[4 * i + 2] * g[8 + j]) + inv[4 * i + 3] * g[12 + j];
}

}  // namespace particles
}  // namespace eng
