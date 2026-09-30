// Contains / Filled (object_states/contains.py, filled.py) + 물리 입자 중심 (macro_particle_system.py:1302) — 층 1·2 공용.
//   중심 = 강체 위치 + T.quat2mat(방향) @ offset  (배치 quat2mat = 평가기 동적 커널(나비 합), 행렬·벡터 곱 순차)
//   안 판정 = 용기 메타링크 메시마다 GeomPrim.check_points_in_volume (Mesh: find_simplex, start 는 입자 순서로 이어짐 / 원기둥 등: visual.h)
//   Contains = 안 개수 > 0,  Filled = pow(2r, 3) * 개수 / link.volume > 0.2  (float32 텐서 연산, volume 은 파이썬 float)
#pragma once
#include <cmath>
#include <cstdint>

#include "core/particles/delaunay.h"
#include "core/particles/visual.h"

namespace eng {
namespace particles {

PEHD void physical_center(const float tf7[7], const float off[3], float c[3]) {
  float R[9];
  quat2mat_batched(tf7 + 3, R);
  for (int i = 0; i < 3; ++i) {
    const float v = (R[3 * i] * off[0] + R[3 * i + 1] * off[1]) + R[3 * i + 2] * off[2];  // eager matmul: 순차, FMA 없음 (FMA 후보는 1,729/79,800 다름)
    c[i] = tf7[i] + v;
  }
}

struct ContainerMesh {
  int32_t kind;   // MeshKind (visual.h), 5 = Mesh (Delaunay)
  float tf[16];   // 메시 scaled_transform
  double attr[3]; // radius height size
  Delaunay3 dl;   // kind == 5
};
// 입자 중심들 중 용기 안 개수. in[] 에 표시 (공식 in_volume). 메시 여러 개면 OR (메시마다 모든 점, start 이어짐)
PEHD int contained_count(const ContainerMesh* ms, int nm, const float* centers, int n, uint8_t* in) {
  for (int i = 0; i < n; ++i) in[i] = 0;
  for (int m = 0; m < nm; ++m) {
    float inv[16];
    inv4_mkl(ms[m].tf, inv);
    int start = 0;
    for (int i = 0; i < n; ++i) {
      float L[3];
      proj_local(inv, centers + 3 * i, L);
      bool v;
      if (ms[m].kind == 5) {
        const double x[3] = {L[0], L[1], L[2]};
        v = dl_find_simplex(ms[m].dl, x, &start) >= 0;
      } else {
        v = in_mesh_volume(ms[m].kind, ms[m].attr, L);
      }
      in[i] = in[i] | (uint8_t)v;
    }
  }
  int k = 0;
  for (int i = 0; i < n; ++i) k += in[i];
  return k;
}
PEHD bool filled_value(float radius, int n_in, double link_volume) {
  const float d = radius * 2.0f;
  const float pv = d * d * d;  // torch pow(x, 3)
  const float prop = div_rn(pv * (float)n_in, (float)link_volume);
  return prop > 0.2f;
}

}  // namespace particles
}  // namespace eng
