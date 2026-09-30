// scipy.spatial.Delaunay.find_simplex (scipy 1.15.3, scipy/spatial/_qhull.pyx:1212~1580, 2071) 을 손으로 (층 1·2 공용, double).
// OmniGibson 이 메시 부피 안 판정에 쓴다: prims/geom_prim.py:205 check_local_points_in_volume = find_simplex(점) >= 0
//   (용기 fillable 메타링크 → Contains/Filled, 다지기 입자 격자, 투영 메시가 Mesh 인 제거기·도포기).
// 삼각분할(Qhull)은 메시마다 한 번 scipy 로 떠 온다 (tests/particles/gen_delaunay_ref.py 가 내보내는 모양). 찾기만 옮긴다.
// 주의: 한 번 부른 점 묶음 안에서 시작 단체(start)가 다음 점으로 이어진다 (find_simplex :2144). 경계 eps 근처에서만 결과(-1 여부)가
// 시작 단체에 따라 달라질 수 있으므로 공식과 같은 순서로 점을 넣는다.
#pragma once
#include <cmath>
#include <cstdint>

#include "core/particles/visual.h"

namespace eng {
namespace particles {

struct Delaunay3 {
  int32_t nsimplex;
  const double* transform;  // [nsimplex][3*4]  (ndim*(ndim+1)) — 역행렬 3x3 행 우선 + 기준점 3
  const int32_t* neighbors; // [nsimplex][4]
  const double* equations;  // [nsimplex][5]
  double paraboloid_scale, paraboloid_shift;
  double min_bound[3], max_bound[3];
};

PEHD bool dl_outside_bbox(const Delaunay3& d, const double x[3], double eps) {
  for (int i = 0; i < 3; ++i)
    if (x[i] < d.min_bound[i] - eps || x[i] > d.max_bound[i] + eps) return true;
  return false;
}
PEHD double dl_bary_single(const double* T, const double x[3], double c[4], int i) {
  if (i == 3) {
    c[3] = 1.0;
    for (int j = 0; j < 3; ++j) c[3] -= c[j];
  } else {
    c[i] = 0;
    for (int j = 0; j < 3; ++j) c[i] += T[3 * i + j] * (x[j] - T[9 + j]);
  }
  return c[i];
}
PEHD void dl_bary(const double* T, const double x[3], double c[4]) {
  c[3] = 1.0;
  for (int i = 0; i < 3; ++i) {
    c[i] = 0;
    for (int j = 0; j < 3; ++j) c[i] += T[3 * i + j] * (x[j] - T[9 + j]);
    c[3] -= c[i];
  }
}
PEHD bool dl_bary_inside(const double* T, const double x[3], double c[4], double eps) {
  c[3] = 1.0;
  for (int i = 0; i < 3; ++i) {
    c[i] = 0;
    for (int j = 0; j < 3; ++j) c[i] += T[3 * i + j] * (x[j] - T[9 + j]);
    c[3] -= c[i];
    if (!(-eps <= c[i] && c[i] <= 1 + eps)) return false;
  }
  return -eps <= c[3] && c[3] <= 1 + eps;
}
PEHD int dl_bruteforce(const Delaunay3& d, double c[4], const double x[3], double eps, double eps_broad) {
  if (dl_outside_bbox(d, x, eps)) return -1;
  for (int s = 0; s < d.nsimplex; ++s) {
    const double* T = d.transform + 12 * s;
    if (T[0] == T[0]) {
      if (dl_bary_inside(T, x, c, eps)) return s;
    } else {
      for (int k = 0; k < 4; ++k) {
        const int nb = d.neighbors[4 * s + k];
        if (nb == -1) continue;
        const double* Tn = d.transform + 12 * nb;
        if (Tn[0] != Tn[0]) continue;
        dl_bary(Tn, x, c);
        bool inside = true;
        for (int m = 0; m < 4; ++m) {
          const double lo = d.neighbors[4 * nb + m] == s ? -eps_broad : -eps;
          if (!(lo <= c[m] && c[m] <= 1 + eps)) {
            inside = false;
            break;
          }
        }
        if (inside) return nb;
      }
    }
  }
  return -1;
}
PEHD int dl_directed(const Delaunay3& d, double c[4], const double x[3], int* start, double eps, double eps_broad) {
  int s = *start;
  if (s < 0 || s >= d.nsimplex) s = 0;
  bool done = false;
  for (int cyc = 0; cyc < 1 + d.nsimplex / 4; ++cyc) {
    if (s == -1) {
      done = true;
      break;
    }
    const double* T = d.transform + 12 * s;
    int inside = 1;
    for (int k = 0; k < 4; ++k) {
      dl_bary_single(T, x, c, k);
      if (c[k] < -eps) {
        const int m = d.neighbors[4 * s + k];
        if (m == -1) {
          *start = s;
          return -1;
        }
        s = m;
        inside = -1;
        break;
      } else if (c[k] <= 1 + eps) {
      } else {
        inside = 0;
      }
    }
    if (inside == -1) continue;
    if (inside == 0) s = dl_bruteforce(d, c, x, eps, eps_broad);
    done = true;
    break;
  }
  if (!done) s = dl_bruteforce(d, c, x, eps, eps_broad);
  *start = s;
  return s;
}
PEHD double dl_distplane(const Delaunay3& d, int s, const double z[4]) {
  const double* e = d.equations + 5 * s;
  double dist = e[4];
  for (int k = 0; k < 4; ++k) dist += e[k] * z[k];
  return dist;
}
// _find_simplex (:1470). start 는 점 사이로 이어진다 (처음 0)
PEHD int dl_find_simplex(const Delaunay3& d, const double x[3], int* start) {
  const double eps = 100 * 2.220446049250313e-16;
  const double eps_broad = sqrt(eps);
  double c[4];
  if (dl_outside_bbox(d, x, eps)) return -1;
  if (d.nsimplex <= 0) return -1;
  int s = *start;
  if (s < 0 || s >= d.nsimplex) s = 0;
  double z[4];
  z[3] = 0;
  for (int i = 0; i < 3; ++i) {
    z[i] = x[i];
    z[3] += x[i] * x[i];
  }
  z[3] *= d.paraboloid_scale;
  z[3] += d.paraboloid_shift;
  double best = dl_distplane(d, s, z);
  bool changed = true;
  while (changed) {
    if (best > 0) break;
    changed = false;
    for (int k = 0; k < 4; ++k) {
      const int nb = d.neighbors[4 * s + k];
      if (nb == -1) continue;
      const double dist = dl_distplane(d, nb, z);
      if (dist > best + eps * (1 + fabs(best))) {
        s = nb;
        best = dist;
        changed = true;
      }
    }
  }
  *start = s;
  return dl_directed(d, c, x, start, eps, eps_broad);
}

}  // namespace particles
}  // namespace eng
