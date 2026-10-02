// 뿌리기 표본 sampling_utils.sample_cuboid_on_object (도포기 ParticleApplier._modify_particles, particle_modifier.py:1267) — 호스트 전용.
// 장면 질의(광선)는 입력: RayFn(origin, dir, distance) 가 PhysX raycast_all 이 알린 순서대로 적중 목록을 준다(joints 몫).
// 연산 순서 (대조로 확인, tests/particles/test_spray):
//   get_parallel_rays: 방향 = end - start, th.randn(3) (trng.h, double Box-Muller + 캐시), 노름 = eager FMA 꼴,
//     th.linalg.cross = fma(a1,b2, -(a2*b1)) 꼴, 격자 = linspace(-o, o, steps) ij, 원점 = start + (gx*o1 + gy*o2) 순차
//   raytest: diff = dst - src, dist = eager 노름, dir = diff / dist. raycast_all 적중 중 무시 몸체 빼고 거리 안정 정렬 첫 번째
//   법선 = 적중 법선 / 행 노름, 유사도 = arccos(clip(N @ c / (|N| |c|), -1, 1)) < 1.0 (mv = MKL sgemv_, arccos = MKL vmsAcos)
//   fit_plane (fitplane.h, MKL), 평면 법선 부호 = sign(dot(src[중심] - 중심점, 법선)) (sdot_), 평면 거리 = |(P - c) @ n| (sgemv_)
//   회전 = T.align_vector_sets: B = bmm(v1ᵀ, v2) (작은 크기 ATen 순차 루프), MKL sgesdd('A'), u@vh (MKL sgemm_), det 부호로 뒤집기, mat2quat (agframe)
// MKL 은 fitplane.h 처럼 dlopen (ENGINE_MKL_LIB). GPU 판은 이 결과만 호스트에서 넘겨받는다(5 스텝마다 표본 2 개).
#pragma once
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/omni/agframe.h"
#include "core/particles/fitplane.h"
#include "core/particles/trng.h"

namespace eng {
namespace particles {

namespace agf = eng::omni::agf;

namespace mklsp {
typedef void (*sgemv_t)(const char*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*,
                        float*, const int*);
typedef float (*sdot_t)(const int*, const float*, const int*, const float*, const int*);
typedef void (*vms_t)(int64_t, const float*, float*, int64_t);
struct Lib {
  sgemv_t gemv = nullptr;
  sdot_t dot = nullptr;
  vms_t acos = nullptr;
  bool load() {
    if (gemv) return true;
    const char* cands[] = {getenv("ENGINE_MKL_LIB"), "libmkl_rt.so.2", "libmkl_rt.so"};
    for (const char* p : cands) {
      if (!p) continue;
      void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
      if (!h) continue;
      gemv = (sgemv_t)dlsym(h, "sgemv_");
      dot = (sdot_t)dlsym(h, "sdot_");
      acos = (vms_t)dlsym(h, "vmsAcos");
      if (gemv && dot && acos) return true;
    }
    fprintf(stderr, "spray: MKL sgemv_/sdot_/vmsAcos 를 못 찾음 (ENGINE_MKL_LIB)\n");
    return false;
  }
};
inline Lib& lib() {
  static Lib l;
  l.load();
  return l;
}
constexpr int64_t kVmlMode = 0x2 | 0x140000 | 0x100;  // VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE (mkl_trig.h 와 같음)
// 행 우선 (k,3) @ (3,) → y[k] : torch 는 열 우선 3xk 행렬에 'T' 로 sgemv_
inline void mv_k3(const float* A, int k, const float* x, float* y) {
  const int three = 3, kk = k, one_i = 1;
  const float one = 1.0f, zero = 0.0f;
  lib().gemv("T", &three, &kk, &one, A, &three, x, &one_i, &zero, y, &one_i);
}
}  // namespace mklsp

inline float enorm3(const float* v) { return std::sqrt(std::fmaf(v[2], v[2], std::fmaf(v[1], v[1], v[0] * v[0]))); }
inline void tcross(const float* a, const float* b, float* r) {
  r[0] = std::fmaf(a[1], b[2], -(a[2] * b[1]));
  r[1] = std::fmaf(a[2], b[0], -(a[0] * b[2]));
  r[2] = std::fmaf(a[0], b[1], -(a[1] * b[0]));
}
// th.linspace(s, e, n) (float32, CPU, 16 개 미만 스칼라 경로): step = (e - s)/(n - 1), 앞 절반 fma(step, i, s), 뒤 절반 fma(-step, n-1-i, e)
// (ATen RangeFactoriesKernel 이 AVX2 판에서 FMA 로 묶임 — 대조 3.6 만 개). 16 개 이상은 벡터 arange 경로(아직 안 옮김, 여기선 격자 ≤ 15)
inline float tlinspace(float s, float e, int n, int i) {
  if (n == 1) return s;
  const float step = (e - s) / (float)(n - 1);
  return i < n / 2 ? std::fmaf(step, (float)i, s) : std::fmaf(-step, (float)(n - 1 - i), e);
}

struct RayHitIn {
  float pos[3], nrm[3];
  double dist;  // 파이썬 float (PhysX float → double)
  int32_t body; // 강체 번호
};
struct CuboidResult {
  bool ok = false;
  float centroid[3], normal[3], rot[4];
  int32_t body = -1;
  int n_hit = 0;  // 적중 광선 수 (회전 계산에 쓴 점 수)
};

// 표본 하나 (start/end 한 쌍 — 도포기는 표본마다 시도 1 번). ignore(body) = 무시할 몸체인가.
// rayfn(src, dir, dist, std::vector<RayHitIn>& out) : raycast_all 이 알린 순서로 적중을 채움
template <class RayFn, class IgnoreFn>
inline CuboidResult sample_cuboid_one(TorchMT& rng, const float start[3], const float end[3], const float dims[3], RayFn rayfn, IgnoreFn ignore,
                                      float padding_len = 0.005f, float new_ray_dist = 0.1f, float tol = 1.0f, float plane_thr = 0.05f) {
  CuboidResult R;
  // ---- get_parallel_rays ----
  const float rd[3] = {end[0] - start[0], end[1] - start[1], end[2] - start[2]};
  float rv[3];
  for (int k = 0; k < 3; ++k) rv[k] = torch_randn_float(rng);
  float nn = enorm3(rv);
  for (int k = 0; k < 3; ++k) rv[k] /= nn;
  float o1[3], o2[3];
  tcross(rd, rv, o1);
  nn = enorm3(o1);
  for (int k = 0; k < 3; ++k) o1[k] /= nn;
  tcross(rd, o1, o2);
  for (int k = 0; k < 3; ++k) o2[k] = -o2[k];
  nn = enorm3(o2);
  for (int k = 0; k < 3; ++k) o2[k] /= nn;
  const float off[2] = {dims[0] / 2.0f, dims[1] / 2.0f};
  int steps[2];
  for (int k = 0; k < 2; ++k) {
    steps[k] = (int)(off[k] / new_ray_dist) * 2 + 1;
    if (steps[k] < 3) steps[k] = 3;
  }
  const int nr = steps[0] * steps[1];
  std::vector<float> src(3 * nr), grid(2 * nr);
  std::vector<uint8_t> hit(nr, 0);
  std::vector<RayHitIn> res(nr);
  std::vector<RayHitIn> buf;
  for (int i = 0, r = 0; i < steps[0]; ++i)
    for (int j = 0; j < steps[1]; ++j, ++r) {
      const float gx = tlinspace(-off[0], off[0], steps[0], i), gy = tlinspace(-off[1], off[1], steps[1], j);
      grid[2 * r] = gx;
      grid[2 * r + 1] = gy;
      float so[3], de[3];
      for (int k = 0; k < 3; ++k) {
        const float t = gx * o1[k] + gy * o2[k];
        so[k] = start[k] + t;
        de[k] = end[k] + t;
        src[3 * r + k] = so[k];
      }
      const float df[3] = {de[0] - so[0], de[1] - so[1], de[2] - so[2]};
      const float dist = enorm3(df);
      const float dir[3] = {df[0] / dist, df[1] / dist, df[2] / dist};
      buf.clear();
      rayfn(so, dir, dist, buf);
      int best = -1;
      for (int q = 0; q < (int)buf.size(); ++q) {  // 무시 몸체 빼고, 거리 안정 정렬 첫 번째
        if (ignore(buf[q].body)) continue;
        if (best < 0 || buf[q].dist < buf[best].dist) best = q;
      }
      if (best >= 0) {
        hit[r] = 1;
        res[r] = buf[best];
      }
    }
  // ---- check_rays_hit_object (문턱 0 → 늘 통과) ----
  const int center = nr / 2;
  if (!hit[center]) return R;
  std::vector<float> hp, hn;
  int fc = -1;
  for (int r = 0; r < nr; ++r)
    if (hit[r]) {
      if (r == center) fc = (int)hp.size() / 3;
      for (int k = 0; k < 3; ++k) hp.push_back(res[r].pos[k]), hn.push_back(res[r].nrm[k]);
    }
  const int k = (int)hp.size() / 3;
  for (int q = 0; q < k; ++q) {
    const float n2 = enorm3(&hn[3 * q]);
    for (int c = 0; c < 3; ++c) hn[3 * q + c] /= n2;
  }
  const float* cn = &hn[3 * fc];
  const float* cp = &hp[3 * fc];
  auto similar = [&](const float* c, const float* N, int m) {
    std::vector<float> dots(m), ang(m);
    mklsp::mv_k3(N, m, c, dots.data());
    const float cnn = enorm3(c);
    for (int q = 0; q < m; ++q) {
      float v = dots[q] / (enorm3(N + 3 * q) * cnn);
      v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
      dots[q] = v;
    }
    mklsp::lib().acos(m, dots.data(), ang.data(), mklsp::kVmlMode);
    for (int q = 0; q < m; ++q)
      if (!(ang[q] < tol)) return false;
    return true;
  };
  if (!similar(cn, hn.data(), k)) return R;
  float pc[3], pn[3];
  if (!fit_plane(hp.data(), k, pc, pn)) return R;
  float p2s[3] = {src[3 * center] - pc[0], src[3 * center + 1] - pc[1], src[3 * center + 2] - pc[2]};
  {
    const int three = 3, one_i = 1;
    const float dt = mklsp::lib().dot(&three, p2s, &one_i, pn, &one_i);
    const float sg = dt > 0 ? 1.0f : (dt < 0 ? -1.0f : 0.0f);
    for (int c = 0; c < 3; ++c) pn[c] *= sg;
  }
  if (!similar(cn, pn, 1)) return R;
  {  // check_distance_to_plane (적중만)
    std::vector<float> d(k), X(3 * k);
    for (int q = 0; q < k; ++q)
      for (int c = 0; c < 3; ++c) X[3 * q + c] = hp[3 * q + c] - pc[c];
    mklsp::mv_k3(X.data(), k, pn, d.data());
    for (int q = 0; q < k; ++q)
      if (std::fabs(d[q]) > plane_thr) return R;
  }
  // 모든 광선 적중점 (빗나간 것은 0) → 평면 투영 (|거리| 로 빼는 공식 식 그대로) + 바닥 여유
  std::vector<float> P(3 * nr), X(3 * nr), dd(nr);
  for (int r = 0; r < nr; ++r)
    for (int c = 0; c < 3; ++c) {
      P[3 * r + c] = hit[r] ? res[r].pos[c] : 0.0f;
      X[3 * r + c] = P[3 * r + c] - pc[c];
    }
  mklsp::mv_k3(X.data(), nr, pn, dd.data());
  float pad[3];
  for (int c = 0; c < 3; ++c) pad[c] = padding_len * pn[c];
  for (int r = 0; r < nr; ++r) {
    const float a = std::fabs(dd[r]);
    for (int c = 0; c < 3; ++c) P[3 * r + c] = (P[3 * r + c] - a * pn[c]) + pad[c];
  }
  float cc[3];
  for (int c = 0; c < 3; ++c) cc[c] = P[3 * center + c] + pn[c] * dims[2] / 2.0f;
  // ---- compute_rotation_from_grid_sample → T.align_vector_sets ----
  int nh = 0;
  for (int r = 0; r < nr; ++r) nh += hit[r];
  R.n_hit = nh;
  if (nh < 3) return R;
  std::vector<float> v1(3 * nh), v2(3 * nh);
  for (int r = 0, q = 0; r < nr; ++r)
    if (hit[r]) {
      for (int c = 0; c < 3; ++c) v1[3 * q + c] = P[3 * r + c] - cc[c];
      v2[3 * q] = grid[2 * r];
      v2[3 * q + 1] = grid[2 * r + 1];
      v2[3 * q + 2] = -dims[2] / 2.0f;
      ++q;
    }
  float B[9];  // B[i][k] = sum_j v1[j][i] v2[j][k]  (ATen 작은 bmm 순차 루프)
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c) {
      float s = 0.0f;
      for (int q = 0; q < nh; ++q) s += v1[3 * q + i] * v2[3 * q + c];
      B[i * 3 + c] = s;
    }
  float A[9], S[3], U[9], VT[9], wq;
  int iw[24], info, lw = -1;
  const int three = 3;
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c) A[i + 3 * c] = B[i * 3 + c];
  auto& FL = mklfp::lib();
  FL.gesdd("A", &three, &three, A, &three, S, U, &three, VT, &three, &wq, &lw, iw, &info);
  lw = std::max(1, (int)wq);
  float work[1024];
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c) A[i + 3 * c] = B[i * 3 + c];
  FL.gesdd("A", &three, &three, A, &three, S, U, &three, VT, &three, work, &lw, iw, &info);
  // u, vh 행 우선 값: u[i][j] = U[i + 3j], vh[i][j] = VT[i + 3j]
  auto mm_uvh = [&](float* C) {  // C = u @ vh (행 우선 결과) — 열 우선 계산 C^T = vhᵀ uᵀ
    const float one = 1.0f, zero = 0.0f;
    float Ct[9];
    FL.gemm("T", "T", &three, &three, &three, &one, VT, &three, U, &three, &zero, Ct, &three);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) C[i * 3 + j] = Ct[j + 3 * i];
  };
  float C[9];
  mm_uvh(C);
  const double det = (double)C[0] * ((double)C[4] * C[8] - (double)C[5] * C[7]) - (double)C[1] * ((double)C[3] * C[8] - (double)C[5] * C[6]) +
                     (double)C[2] * ((double)C[3] * C[7] - (double)C[4] * C[6]);
  if (det < 0) {
    for (int i = 0; i < 3; ++i) U[i + 3 * 2] = -U[i + 3 * 2];  // u[:, -1] = -u[:, -1]
    mm_uvh(C);
  }
  float q[4];
  agf::mat2quat(C, q);
  for (int c = 0; c < 3; ++c) cc[c] -= pad[c];  // undo_cuboid_bottom_padding
  R.ok = true;
  memcpy(R.centroid, cc, 12);
  memcpy(R.normal, pn, 12);
  memcpy(R.rot, q, 16);
  R.body = res[center].body;
  return R;
}

}  // namespace particles
}  // namespace eng
