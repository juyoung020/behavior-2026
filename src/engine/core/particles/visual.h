// 시각 입자(MacroVisualParticleSystem) + 입자 제거기(ParticleRemover) + Covered — OmniGibson 파이썬 층을 손으로 (층 1·2 공용).
// 평가기는 GPU 동역학을 끄므로(eval/evaluator.py:57) 먼지·흙·녹 같은 "시각 입자" 는 물리가 없다: 입자 = (무리 물체, 붙은 링크, 국소 4x4).
// 정답 = 공식 함수를 가짜 물체에 붙여 부른 결과 (tests/particles/gen_visual_ref.py). 연산 순서는 후보 대조로 찾았다 (docs 20.5).
//
// 원본 (BEHAVIOR-1K v3.9.3-post1, omnigibson/):
//   국소 행렬   systems/macro_particle_system.py:749 _modify_batch_particles_position_orientation(local=True)
//               = zeros(n,4,4); [:, 3,3]=1; [:, :3,3]=위치; [:, :3,:3]=T.quat2mat(방향)   (T.* 는 Linux 에서 torch.compile)
//   세계 위치   :664 _compute_batch_particles_position_orientation = (link.scaled_transform @ 국소)[:, :3, 3]  (eager bmm)
//   제거기      object_states/particle_modifier.py:697 _update → :927 ParticleRemover._modify_particles
//               ADJACENCY: :524 _check_in_mesh = 링크 visual_aabb(prims/rigid_prim.py:572, T.transform_points) ± 0.02, 열린 구간
//               PROJECTION: prims/geom_prim.py:229 (hom @ th.linalg.inv(W).T) → utils/geometry_utils.py 원기둥·원뿔·상자·구
//               한도: 제거한 수 누적(ModifiedParticles) == 한도(기본 200) 이면 Saturated → 더 안 지움. 낮은 입자 번호부터 지움.
//   Covered     object_states/covered.py:59 (무리 입자 수 ≥ 1)
//
// 찾은 연산 순서 (Zen5, torch 2.7.0 MKL, Linux 평가기와 같은 venv):
//   T.quat2mat 배치: 노름 = sqrt(((x²+y²)+z²)+w²) 순차. 단 배치 크기 1 이면 inductor 가 따로 특수화해 나비 (x²+z²)+(y²+w²)
//   bmm·torch.mm(4 원소 내적): 왼쪽부터 순차 곱·합, FMA 없음
//   th.linalg.inv(4x4, MKL getrf+getrs): 부분 피벗 LU, 열 스케일은 역수 곱, 갱신 c - l*u (FMA 없음),
//     뒤 대입은 곱들을 k 내림차순으로 먼저 합한 뒤 빼고 대각 역수를 곱함: x_i = (b_i - ((p3 + p2) + p1)) * (1/u_ii)
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define PEHD __host__ __device__ inline
#else
#define PEHD inline
#endif

namespace eng {
namespace particles {

PEHD float sqrt_rn(float x) {
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return std::sqrt(x);
#endif
}
PEHD float fmaf_(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return std::fmaf(a, b, c);
#endif
}
PEHD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

// ---- 국소 행렬 ----------------------------------------------------------------------------------------------
// T.quat2mat (transform_utils.py:367): q / norm(q) → 바깥곱 → 1 - 2*(yy+zz) ...  single = 배치 크기 1
PEHD void quat2mat_batched(const float q[4], bool single, float r[9]) {
  const float a = q[0] * q[0], b = q[1] * q[1], c = q[2] * q[2], d = q[3] * q[3];
  const float s = single ? 0.0f + ((a + c) + (b + d)) : ((a + b) + c) + d;
  const float n = sqrt_rn(s);
  const float x = div_rn(q[0], n), y = div_rn(q[1], n), z = div_rn(q[2], n), w = div_rn(q[3], n);
  const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, xw = x * w, yw = y * w, zw = z * w;
  r[0] = 1.0f - 2.0f * (yy + zz);
  r[1] = 2.0f * (xy - zw);
  r[2] = 2.0f * (xz + yw);
  r[3] = 2.0f * (xy + zw);
  r[4] = 1.0f - 2.0f * (xx + zz);
  r[5] = 2.0f * (yz - xw);
  r[6] = 2.0f * (xz - yw);
  r[7] = 2.0f * (yz + xw);
  r[8] = 1.0f - 2.0f * (xx + yy);
}
// 국소 4x4 (행 우선). single = 이 입자가 든 set_*_local_pose 호출의 배치 크기가 1
PEHD void local_mat(const float p[3], const float q[4], bool single, float m[16]) {
  float r[9];
  quat2mat_batched(q, single, r);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m[i * 4 + j] = r[i * 3 + j];
    m[i * 4 + 3] = p[i];
  }
  m[12] = m[13] = m[14] = 0.0f;
  m[15] = 1.0f;
}

// ---- 세계 위치 (bmm 의 3 열) -------------------------------------------------------------------------------------
PEHD float dot4(const float* a, const float* b0, const float* b1, const float* b2, const float* b3) {
  return ((a[0] * *b0 + a[1] * *b1) + a[2] * *b2) + a[3] * *b3;
}
PEHD void particle_world_pos(const float L[16], const float M[16], float o[3]) {
  for (int r = 0; r < 3; ++r) o[r] = dot4(L + 4 * r, M + 3, M + 7, M + 11, M + 15);
}

// ---- 제거기 ADJACENCY: 링크 visual_aabb ----------------------------------------------------------------------
// T.transform_points(points, M): |M - I|.max() < 1e-8 이면 그대로, 아니면 torch.mm(M, [p,1]^T)^T
PEHD bool near_identity(const float M[16]) {
  float mx = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const float d = std::fabs(M[i] - ((i % 5 == 0) ? 1.0f : 0.0f));
    mx = d > mx ? d : mx;
  }
  return mx < 1e-8f;
}
PEHD void visual_aabb(const float M[16], const float* hull, int nh, float lo[3], float hi[3]) {
  const bool id = near_identity(M);
  for (int h = 0; h < nh; ++h) {
    const float p4[4] = {hull[3 * h], hull[3 * h + 1], hull[3 * h + 2], 1.0f};
    for (int k = 0; k < 3; ++k) {
      const float v = id ? p4[k] : ((M[4 * k] * p4[0] + M[4 * k + 1] * p4[1]) + M[4 * k + 2] * p4[2]) + M[4 * k + 3] * p4[3];
      if (h == 0 || v < lo[k]) lo[k] = v;
      if (h == 0 || v > hi[k]) hi[k] = v;
    }
  }
}
// _check_in_mesh (ADJACENCY): lower -= 0.02; upper += 0.02; (lower < p) & (p < upper) 세 축 모두
PEHD bool in_relaxed_aabb(const float lo[3], const float hi[3], const float p[3]) {
  bool in = true;
  for (int k = 0; k < 3; ++k) in = in && (lo[k] - 0.02f) < p[k] && p[k] < (hi[k] + 0.02f);
  return in;
}

// ---- 제거기 PROJECTION: 투영 메시 부피 ---------------------------------------------------------------------------
// th.linalg.inv (MKL). W, X 행 우선. LAPACK 에는 열 우선 A 로 넘어간다 (a[i + 4j] = W[i][j])
PEHD void inv4_mkl(const float W[16], float X[16]) {
  float a[16];
  int piv[4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a[i + 4 * j] = W[i * 4 + j];
  for (int j = 0; j < 4; ++j) {
    int p = j;
    float mx = std::fabs(a[j + 4 * j]);
    for (int i = j + 1; i < 4; ++i)
      if (std::fabs(a[i + 4 * j]) > mx) {
        mx = std::fabs(a[i + 4 * j]);
        p = i;
      }
    piv[j] = p;
    if (p != j)
      for (int k = 0; k < 4; ++k) {
        const float t = a[j + 4 * k];
        a[j + 4 * k] = a[p + 4 * k];
        a[p + 4 * k] = t;
      }
    const float rc = div_rn(1.0f, a[j + 4 * j]);
    for (int i = j + 1; i < 4; ++i) a[i + 4 * j] = a[i + 4 * j] * rc;
    for (int k = j + 1; k < 4; ++k)
      for (int i = j + 1; i < 4; ++i) a[i + 4 * k] = a[i + 4 * k] - a[i + 4 * j] * a[j + 4 * k];
  }
  for (int c = 0; c < 4; ++c) {
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    x[c] = 1.0f;
    for (int j = 0; j < 4; ++j)
      if (piv[j] != j) {
        const float t = x[j];
        x[j] = x[piv[j]];
        x[piv[j]] = t;
      }
    // 앞 대입 (단위 하삼각): 순차 빼기 x_i = (x_i - l_i0 x_0) - l_i1 x_1 (대조 930/930). 우리 행렬은 행 3 = 0 0 0 1 이라
    // 항이 셋인 줄은 늘 0 곱이 섞여 묶음 차이를 드러내지 않았다 — 일반 행렬에 쓰려면 다시 대조할 것
    for (int k = 0; k < 4; ++k)
      for (int i = k + 1; i < 4; ++i) x[i] = x[i] - x[k] * a[i + 4 * k];
    for (int i = 3; i >= 0; --i) {
      const float rc = div_rn(1.0f, a[i + 4 * i]);
      if (i == 3) {
        x[i] = x[i] * rc;
        continue;
      }
      float s = a[i + 4 * 3] * x[3];
      for (int k = 2; k > i; --k) s = s + a[i + 4 * k] * x[k];
      x[i] = (x[i] - s) * rc;
    }
    for (int i = 0; i < 4; ++i) X[i * 4 + c] = x[i];
  }
}
// hom @ inv.T 의 앞 세 칸: o_j = 순차 내적(p, inv 행 j)
PEHD void proj_local(const float inv[16], const float p[3], float o[3]) {
  for (int j = 0; j < 3; ++j) o[j] = ((p[0] * inv[4 * j] + p[1] * inv[4 * j + 1]) + p[2] * inv[4 * j + 2]) + 1.0f * inv[4 * j + 3];
}
enum MeshKind : int32_t { ADJACENCY = 0, CYLINDER = 1, CONE = 2, CUBE = 3, SPHERE = 4 };
// geometry_utils.py. attr = USD radius, height, size (double). 파이썬 스칼라는 float32 텐서와 섞일 때 float32 로 바뀐다
PEHD bool in_mesh_volume(int kind, const double attr[3], const float L[3]) {
  const float radius = (float)attr[0], height = (float)attr[1];
  if (kind == CYLINDER || kind == CONE) {
    const bool in_h = (float)(-attr[1] / 2.0) < L[2] && L[2] < (float)(attr[1] / 2.0);
    const float nr = sqrt_rn(fmaf_(L[1], L[1], L[0] * L[0]));
    if (kind == CYLINDER) return in_h && nr < radius;
    const float t = div_rn(L[2] + (float)(attr[1] / 2.0), height);
    return in_h && nr < radius * (1.0f - t);
  }
  if (kind == CUBE) {
    const float lo = (float)(-attr[2] / 2.0), hi = (float)(attr[2] / 2.0);
    return lo < L[0] && L[0] < hi && lo < L[1] && L[1] < hi && lo < L[2] && L[2] < hi;
  }
  return sqrt_rn(fmaf_(L[2], L[2], fmaf_(L[1], L[1], L[0] * L[0]))) < radius;  // SPHERE: size = radius 속성 (geom_prim.py:207)
}

// ---- 입자 모음 · 제거 한 번 ----------------------------------------------------------------------------------------
// 한 시각 입자계(판 하나). 입자 번호 = 만든 순서. 공식 dict 순서 = 살아 있는 것의 번호 순.
struct VisualSet {
  int32_t n;            // 지금까지 만든 입자 수 (죽은 것 포함)
  const int32_t* link;  // [n] 붙은 링크 번호 (link_tf 배열 번호)
  const int32_t* group; // [n] 무리(물체) 번호
  const float* lm;      // [n][16] 국소 행렬
  uint8_t* alive;       // [n]
};
struct RemoverGeom {
  int32_t kind;        // MeshKind
  int32_t nh;          // ADJACENCY: 링크 볼록 껍질 점 수
  const float* hull;   // [nh][3] 링크 국소 (visual_boundary_points_local)
  double attr[3];      // PROJECTION: radius height size
};
// ParticleRemover._modify_particles (시각 입자). tf = ADJACENCY 면 링크 scaled_transform, PROJECTION 이면 투영 메시 scaled_transform.
// modified = 이 제거기·이 계의 누적 제거 수 (ModifiedParticles). 반환 = 이번에 지운 수.
PEHD int remove_visual(VisualSet& s, const float* link_tf, const RemoverGeom& g, const float tf[16], int32_t limit, int32_t& modified) {
  int n_alive = 0;
  for (int i = 0; i < s.n; ++i) n_alive += s.alive[i];
  if (n_alive == 0) return 0;
  float lo[3], hi[3], inv[16];
  if (g.kind == ADJACENCY)
    visual_aabb(tf, g.hull, g.nh, lo, hi);
  else
    inv4_mkl(tf, inv);
  const int budget = limit - modified;
  int removed = 0;
  for (int i = 0; i < s.n && removed < budget; ++i) {
    if (!s.alive[i]) continue;
    float p[3];
    particle_world_pos(link_tf + 16 * s.link[i], s.lm + 16 * i, p);
    bool in;
    if (g.kind == ADJACENCY) {
      in = in_relaxed_aabb(lo, hi, p);
    } else {
      float L[3];
      proj_local(inv, p, L);
      in = in_mesh_volume(g.kind, g.attr, L);
    }
    if (in) {
      s.alive[i] = 0;
      ++removed;
    }
  }
  modified += removed;
  return removed;
}

// Covered(obj, 시각 계) = 무리 입자 수 ≥ 1 (covered.py:59, m.VISUAL_PARTICLE_THRESHOLD = 1)
PEHD bool covered_visual(const VisualSet& s, int group) {
  for (int i = 0; i < s.n; ++i)
    if (s.alive[i] && s.group[i] == group) return true;
  return false;
}

// ---- 제거기 한 스텝 조건 (ParticleModifier._update, :697) --------------------------------------------------------
// 조건 종류 (particle_modifier.py:535 _generate_condition + 기본 조건 :894)
enum CondKind : int32_t { COND_TOGGLED = 0, COND_SATURATED = 1, COND_NONEMPTY = 2, COND_LIMIT = 3 };
struct Cond {
  int32_t kind;
  int32_t value;  // TOGGLED: 원하는 켜짐 값 / SATURATED: 계 번호
};
// 한 제거기가 한 계를 이번 스텝에 지울 수 있나 (check_conditions_for_system + Saturated 재확인).
// toggled = 제거기 물체의 ToggledOn 값, n_alive_sys = 그 계 입자 수, sat_of_sys(k) = 이 물체가 계 k 에 포화인가(활성 계만),
// limit·modified = 이 계 한도·누적.
template <class SatFn>
PEHD bool remover_may_modify(const Cond* conds, int ncond, bool toggled, int n_alive_sys, int32_t limit, int32_t modified,
                             SatFn sat_of_sys) {
  for (int c = 0; c < ncond; ++c) {
    const Cond& k = conds[c];
    bool ok = true;
    if (k.kind == COND_TOGGLED) ok = toggled == (k.value != 0);
    else if (k.kind == COND_SATURATED) ok = sat_of_sys(k.value);
    else if (k.kind == COND_NONEMPTY) ok = n_alive_sys > 0;
    else ok = modified != limit;  // COND_LIMIT: not Saturated
    if (!ok) return false;
  }
  return modified != limit;  // _update 의 Saturated 재확인 (:707)
}

}  // namespace particles
}  // namespace eng
