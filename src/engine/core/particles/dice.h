// 다지기 입자 자리 (DicingRule → PhysicalParticleSystem.generate_particles_from_link, system_base.py:896) — 층 1·2 공용.
//   격자: 축마다 th.arange(lo + r, hi - r, 2r) (float32 0 차원 텐서 인자) → meshgrid(ij) → 평탄화 순서 x 바깥 · z 안쪽
//   안 판정: 링크 충돌 메시마다 GeomPrim.check_points_in_volume (linalg.inv → 국소 → find_simplex ≥ 0), 메시끼리 OR.
//     공식은 메시마다 모든 격자점을 한 번에 넣으므로 find_simplex 의 start 는 메시마다 격자 순서로 이어진다 (단락 없음).
//   (그다음 generate_particles: T.random_quaternion(n) 이 방향 — trng.h, sample_scales 가 th.rand(n,3) 로 생성기 3n 번 소비)
#pragma once
#include <cmath>
#include <cstdint>

#include "core/particles/delaunay.h"
#include "core/particles/visual.h"

namespace eng {
namespace particles {

// th.arange (float32, CPU) 길이·값. ATen RangeFactoriesKernel: acc = double. cpu_serial_kernel_vec 가 Vectorized<float>(여기서는 8 칸, AVX2 판)
// 두 개씩(16 원소) 돌고: 벡터 기준값 b = float(start + step*idx), 칸 k = float((double)b + step*k); 남은 n % 16 개는 스칼라 float(start + step*idx).
// (대조 1,500 개 arange 로 확인. 다른 CPU 기능 판(AVX512 arange 커널)이면 칸 수가 바뀔 수 있다 — Zen 5·Zen 4 평가 장비는 같은 판으로 본다)
PEHD int64_t arange_len(float start, float end, float step) { return (int64_t)ceil(((double)end - (double)start) / (double)step); }
PEHD float arange_at(float start, float step, int64_t i, int64_t n) {
  const double s = start, d = step;
  if (i < n / 16 * 16) {
    const int64_t b = i / 8 * 8;
    const float base = (float)(s + d * (double)b);
    return (float)((double)base + d * (double)(i - b));
  }
  return (float)(s + d * (double)i);
}

struct DiceMesh {
  float tf[16];  // 충돌 메시 scaled_transform (float32, 행 우선)
  Delaunay3 dl;  // 메시 점(국소)의 삼각분할 (scipy 로 미리)
};

// 격자를 만들고 안쪽 점만 out 에 (공식 순서). 반환 = 개수 (cap 넘으면 -1). lo·hi = 링크 visual_aabb, r = 입자 반지름(float32)
PEHD int64_t dice_grid(const float lo[3], const float hi[3], float r, const DiceMesh* ms, int nm, float* out, int64_t cap) {
  const float step = r * 2.0f;  // particle_particle_rest_distance
  float st[3];
  int64_t n[3];
  for (int k = 0; k < 3; ++k) {
    st[k] = lo[k] + r;
    n[k] = arange_len(st[k], hi[k] - r, step);
    if (n[k] < 0) n[k] = 0;
  }
  float inv[4][16];
  int start[4] = {0, 0, 0, 0};
  for (int m = 0; m < nm; ++m) inv4_mkl(ms[m].tf, inv[m]);
  int64_t cnt = 0;
  for (int64_t i = 0; i < n[0]; ++i)
    for (int64_t j = 0; j < n[1]; ++j)
      for (int64_t k = 0; k < n[2]; ++k) {
        const float p[3] = {arange_at(st[0], step, i, n[0]), arange_at(st[1], step, j, n[1]), arange_at(st[2], step, k, n[2])};
        bool in = false;
        for (int m = 0; m < nm; ++m) {
          float L[3];
          proj_local(inv[m], p, L);
          const double x[3] = {L[0], L[1], L[2]};
          in = (dl_find_simplex(ms[m].dl, x, &start[m]) >= 0) || in;
        }
        if (in) {
          if (cnt >= cap) return -1;
          out[3 * cnt] = p[0];
          out[3 * cnt + 1] = p[1];
          out[3 * cnt + 2] = p[2];
          ++cnt;
        }
      }
  return cnt;
}

}  // namespace particles
}  // namespace eng
