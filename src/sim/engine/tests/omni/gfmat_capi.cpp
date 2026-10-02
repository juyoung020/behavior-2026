// core/omni/gfmat.h 를 파이썬(test_gfmat.py)에서 usdrt.Gf 와 비교하려고 내놓는 얇은 C 입구 (시험 전용)
#include "core/omni/gfmat.h"

using namespace eng::omni::gf;

extern "C" {
// m16: 행 우선 4x4 double -> 쿼터니언 (x,y,z,w) double, 이동 3 double (RemoveScaleShear().ExtractRotationQuat(), ExtractTranslation())
void gf_rss_quat(const double* m16, double* q4, double* t3) {
  M4 m;
  for (int i = 0; i < 16; ++i) m.m[i / 4][i % 4] = m16[i];
  const M4 r = rt_remove_scale_shear(m);
  extract_rotation_quat(r, q4);
  t3[0] = m.m[3][0];
  t3[1] = m.m[3][1];
  t3[2] = m.m[3][2];
}
// a*b (Gf operator*)
void gf_mul(const double* a16, const double* b16, double* o16) {
  M4 a, b;
  for (int i = 0; i < 16; ++i) { a.m[i / 4][i % 4] = a16[i]; b.m[i / 4][i % 4] = b16[i]; }
  const M4 r = mul(a, b);
  for (int i = 0; i < 16; ++i) o16[i] = r.m[i / 4][i % 4];
}
}
extern "C" {
void gf_rss(const double* m16, double* o16) {
  M4 m;
  for (int i = 0; i < 16; ++i) m.m[i / 4][i % 4] = m16[i];
  const M4 r = rt_remove_scale_shear(m);
  for (int i = 0; i < 16; ++i) o16[i] = r.m[i / 4][i % 4];
}
void gf_quat_only(const double* m16, double* q4) {
  M4 m;
  for (int i = 0; i < 16; ++i) m.m[i / 4][i % 4] = m16[i];
  extract_rotation_quat(m, q4);
}
}
