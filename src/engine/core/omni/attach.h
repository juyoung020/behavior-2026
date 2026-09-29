// AttachedTo 붙이기 판정 (손으로 짬, EHD). object_states/attached_to.py:219 _update → :315 _find_attachment_links 의 정렬 시험:
//   pos_diff = th.norm(child_pos - parent_pos)                       (eager ATen: sqrt(fma(z,z, fma(y,y, x*x))), assisted_grasp.h 와 같음)
//   orn_diff = T.get_orientation_diff_in_radian(child_orn, parent_orn) (utils/transform_utils.py:1212, torch.compile inductor CPU)
//   붙음 = pos_diff < 0.05 and orn_diff < deg2rad(15) (float32 텐서와 파이썬 수 비교)
// inductor 생성 코드 순서 (TORCH_LOGS=output_code, tests/omni/gen_attach_ref.py, -ffp-contract=off):
//   quat_distance(q1=child, q0=parent): d = 벡터 줄이기 (p0c0+p2c2)+(p1c1+p3c3), n = 같은 순서 제곱합, inv = conj / n,
//   q1 = d<0 ? -q1 : q1, quat_multiply 는 식 그대로 왼쪽부터; quat2axisangle: w 자르기 [-1,1], den = sqrt(1 - w*w),
//   den != 0 이면 (q_i * 2 * acos(w)) / den (acos = glibc acosf, std::acos(float)), norm = sqrt(0 + ((a²+c²)+b²))
// 붙인 뒤(관절 만들기·자세 맞춤 _attach:393)는 아직 안 옮김 — 문서 17 절.
#pragma once
#include <cmath>
#include <cstdint>

#include "core/common/glibc_trig.h"
#include "core/omni/assisted_grasp.h"
#include "core/omni/warp_f32.h"

namespace eng {
namespace omni {
namespace att {

OEHD float red4(const float a[4], const float b[4]) {
  return 0.0f + ((a[0] * b[0] + a[2] * b[2]) + (a[1] * b[1] + a[3] * b[3]));
}
OEHD float orientation_diff(const float child[4], const float parent[4]) {
  const float d = red4(parent, child);
  const float n = red4(parent, parent);
  const float x0 = wf::div_rn(-parent[0], n), y0 = wf::div_rn(-parent[1], n), z0 = wf::div_rn(-parent[2], n),
              w0 = wf::div_rn(parent[3], n);
  const bool neg = d < 0.0f;
  const float x1 = neg ? -child[0] : child[0], y1 = neg ? -child[1] : child[1], z1 = neg ? -child[2] : child[2],
              w1 = neg ? -child[3] : child[3];
  const float q[4] = {((x1 * w0 + y1 * z0) - z1 * y0) + w1 * x0, ((-x1 * z0 + y1 * w0) + z1 * x0) + w1 * y0,
                      ((x1 * y0 - y1 * x0) + z1 * w0) + w1 * z0, ((-x1 * x0 - y1 * y0) - z1 * z0) + w1 * w0};
  float w = q[3];
  w = (w != w) ? w : (w > -1.0f ? w : -1.0f);  // max_propagate_nan
  w = (w != w) ? w : (w < 1.0f ? w : 1.0f);    // min_propagate_nan
  const float den = wf::sqrt_rn(1.0f - w * w);
  float r[3] = {0.0f, 0.0f, 0.0f};
  if (den != 0.0f) {
    const float ac = eng::glibc::acosf(w);
    for (int i = 0; i < 3; ++i) r[i] = wf::div_rn((q[i] * 2.0f) * ac, den);
  }
  return wf::sqrt_rn(0.0f + ((r[0] * r[0] + r[2] * r[2]) + r[1] * r[1]));
}
OEHD float pos_diff(const float c[3], const float p[3]) { return ag::torch_norm3(c[0] - p[0], c[1] - p[1], c[2] - p[2]); }
// 붙음 판정 (문턱은 float32 로 바꿔 비교: 텐서 dtype 을 따름 — 시험으로 확인)
OEHD bool aligned(const float cpos[3], const float corn[4], const float ppos[3], const float porn[4], float pos_thresh = 0.05f,
                  float orn_thresh = 0.2617993950843811f) {
  return pos_diff(cpos, ppos) < pos_thresh && orientation_diff(corn, porn) < orn_thresh;
}

}  // namespace att
}  // namespace omni
}  // namespace eng
