// core/omni/obs.h 를 파이썬 시험(test_obs.py)에서 부르는 C 입구 (시험 전용). 자세 = p3 + q4 (x,y,z,w) float32
#include "core/omni/obs.h"

using namespace eng::omni::obs;

static Pose P(const float* s) {
  Pose o;
  for (int i = 0; i < 3; ++i) o.p[i] = s[i];
  for (int i = 0; i < 4; ++i) o.q[i] = s[3 + i];
  return o;
}

extern "C" {
// idx: base3, yaw, trunk4, arm 2x7, grip 2x2 (int32 29 개)
void obs_proprio(const int32_t* idx, const float* jp, const float* jv, const float* base7, const float* eefL7, const float* eefR7, float* out61) {
  R1ProIdx ix;
  int k = 0;
  for (int i = 0; i < 3; ++i) ix.base[i] = idx[k++];
  ix.yaw_dof = idx[k++];
  for (int i = 0; i < 4; ++i) ix.trunk[i] = idx[k++];
  for (int a = 0; a < 2; ++a)
    for (int i = 0; i < 7; ++i) ix.arm[a][i] = idx[k++];
  for (int a = 0; a < 2; ++a)
    for (int i = 0; i < 2; ++i) ix.grip[a][i] = idx[k++];
  const Pose eef[2] = {P(eefL7), P(eefR7)};
  proprio_r1pro(ix, jp, jv, P(base7), eef, out61);
}
// 카메라 하나: local 사슬 n 개(double 16 씩), 링크 자세 -> 카메라 세계 자세 7, base 에 대한 상대 7
void obs_cam(const double* locals, int n, const float* link7, const float* base7, float* world7, float* rel7) {
  const Pose w = camera_world(locals, n, P(link7));
  for (int i = 0; i < 3; ++i) world7[i] = w.p[i];
  for (int i = 0; i < 4; ++i) world7[3 + i] = w.q[i];
  cam_rel_pose(w, P(base7), rel7);
}
}
