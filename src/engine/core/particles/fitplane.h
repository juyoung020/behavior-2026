// sampling_utils.fit_plane (평행 광선 적중점 → 평면 중심·법선) — 호스트 전용. 뿌리기(P4) 표본 검사에 쓴다.
//   ctr = points.mean(0)   : ATen row_sum (4 칸 ILP + 16 단 계단 누적, SumKernel.cpp multi_row_sum) / k
//   x = points - ctr ; M = x.T @ x : MKL sgemm_("N","T", 3,3,k, 1, x,3, x,3, 0, C,3) (열 우선 C = 행 우선 M)
//   U = th.linalg.svd(M).U : MKL sgesdd_("A", 3,3, M(열 우선), …, lwork 질의 뒤 max(1, work[0]), iwork 8*3)
//   normal = U[:, -1] / th.norm(U[:, -1])  (eager norm: sqrt(fma(z,z, fma(y,y, x*x))))
// MKL 은 닫힌 소스라 옮기지 않고 같은 라이브러리를 dlopen 으로 부른다 (core/omni/mkl_trig.h 와 같은 방식, ENGINE_MKL_LIB =
// torch/lib/libtorch_cpu.so — MKL 이 정적으로 들어 있고 sgemm_·sgesdd_ 를 내보냄). 확인: 2,000 행렬 U·법선 비트 동일 (tests/particles/test_fitplane).
// 위험: MKL 은 CPU 명령어 집합마다 다른 코드로 갈 수 있다 — 공식도 같은 영향을 받으므로 같은 CPU 에서 비교한다(15 절과 같음). GPU 판은 이 부분만 호스트에서 계산해 넘긴다.
#pragma once
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace eng {
namespace particles {
namespace mklfp {

typedef void (*sgemm_t)(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*,
                        const int*, const float*, float*, const int*);
typedef void (*sgesdd_t)(const char*, const int*, const int*, float*, const int*, float*, float*, const int*, float*, const int*, float*,
                         const int*, int*, int*);
struct Lib {
  sgemm_t gemm = nullptr;
  sgesdd_t gesdd = nullptr;
  bool load() {
    if (gemm) return true;
    const char* cands[] = {getenv("ENGINE_MKL_LIB"), "libmkl_rt.so.2", "libmkl_rt.so"};
    for (const char* p : cands) {
      if (!p) continue;
      void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
      if (!h) continue;
      gemm = (sgemm_t)dlsym(h, "sgemm_");
      gesdd = (sgesdd_t)dlsym(h, "sgesdd_");
      if (gemm && gesdd) return true;
    }
    fprintf(stderr, "fitplane: MKL sgemm_/sgesdd_ 를 못 찾음 (ENGINE_MKL_LIB=<torch/lib/libtorch_cpu.so>)\n");
    return false;
  }
};
inline Lib& lib() {
  static Lib l;
  l.load();
  return l;
}

}  // namespace mklfp

// ATen row_sum (SumKernel.cpp): 원소 k 개 (보폭 stride) 합. ILP 4 칸 × 계단 누적(num_levels 4)
inline float aten_row_sum(const float* x, int64_t stride, int64_t size) {
  const int64_t ilp = 4, n = size / ilp;  // multi_row_sum(행 = ilp 묶음 n 개, 열 = 4)
  float acc[4][4] = {};
  int64_t lp = 4;
  {
    int64_t c = 0, v = n > 0 ? n - 1 : 0;  // CeilLog2(n)
    while ((int64_t(1) << c) < n) ++c;
    (void)v;
    lp = std::max<int64_t>(4, c / 4);
  }
  const int64_t step = int64_t(1) << lp, mask = step - 1;
  int64_t i = 0;
  for (; i + step <= n;) {
    for (int64_t j = 0; j < step; ++j, ++i)
      for (int k = 0; k < 4; ++k) acc[0][k] += x[(i * ilp + k) * stride];
    for (int j = 1; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        acc[j][k] += acc[j - 1][k];
        acc[j - 1][k] = 0;
      }
      if ((i & (mask << (j * lp))) != 0) break;
    }
  }
  for (; i < n; ++i)
    for (int k = 0; k < 4; ++k) acc[0][k] += x[(i * ilp + k) * stride];
  for (int j = 1; j < 4; ++j)
    for (int k = 0; k < 4; ++k) acc[0][k] += acc[j][k];
  float s = acc[0][0];
  float part[4] = {acc[0][0], acc[0][1], acc[0][2], acc[0][3]};
  for (int64_t r = n * ilp; r < size; ++r) part[0] += x[r * stride];
  s = part[0];
  for (int k = 1; k < 4; ++k) s += part[k];
  return s;
}

// 반환 false = 점이 3 개 미만 (공식은 None)
inline bool fit_plane(const float* pts, int k, float ctr[3], float normal[3]) {
  if (k < 3) return false;
  for (int j = 0; j < 3; ++j) ctr[j] = aten_row_sum(pts + j, 3, k) / (float)k;
  float* x = new float[3 * k];
  for (int r = 0; r < k; ++r)
    for (int j = 0; j < 3; ++j) x[3 * r + j] = pts[3 * r + j] - ctr[j];
  auto& L = mklfp::lib();
  const int three = 3, kk = k;
  const float one = 1.0f, zero = 0.0f;
  float C[9];
  L.gemm("N", "T", &three, &three, &kk, &one, x, &three, x, &three, &zero, C, &three);
  delete[] x;
  // C 는 열 우선 = 행 우선 M 의 전치 자리; sgesdd 에는 열 우선 M 을 넘긴다: A[i + 3j] = M[i][j] = C[i + 3j]
  float A[9], S[3], U[9], VT[9], wq;
  int iw[24], info, lw = -1;
  for (int q = 0; q < 9; ++q) A[q] = C[q];
  L.gesdd("A", &three, &three, A, &three, S, U, &three, VT, &three, &wq, &lw, iw, &info);
  lw = std::max(1, (int)wq);
  float work[1024];
  for (int q = 0; q < 9; ++q) A[q] = C[q];
  L.gesdd("A", &three, &three, A, &three, S, U, &three, VT, &three, work, &lw, iw, &info);
  const float nv[3] = {U[0 + 3 * 2], U[1 + 3 * 2], U[2 + 3 * 2]};  // U[:, -1]
  const float nn = std::sqrt(std::fmaf(nv[2], nv[2], std::fmaf(nv[1], nv[1], nv[0] * nv[0])));
  for (int j = 0; j < 3; ++j) normal[j] = nv[j] / nn;
  return true;
}

}  // namespace particles
}  // namespace eng
