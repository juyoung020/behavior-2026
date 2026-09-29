// OmniGibson 파이썬 층이 쓰는 float32 수학을 비트까지 흉내 (손으로 짬, PhysX 링크 없음, EHD = 층 1·2 공용).
// 제어기는 numpy 백엔드(gm.USE_NUMPY_CONTROLLER_BACKEND = True, macros.py:152) + numba 변환 함수(transform_utils_np.py)를 쓴다.
// 연산 순서는 공식 파이썬을 같은 입력으로 돌려 찾았다 (tests/omni/gen_ctrl_ref.py → test_controllers):
//   - numba _quat2mat (transform_utils_np.py:378): float32 곱·합 그대로 (FMA 없음)
//   - numba _pose_inv (:771) 의 -R^T.dot(t): 0·1 행은 (p0+p1)+p2, 2 행은 fma(a2,b2, fma(a0,b0, a1*b1))
//     (numba 가 비연속 배열 dot 을 스스로 짠 루프로 푸는데, 두 행 벡터 몸통 + 한 행 꼬리가 FMA 로 축약된 결과로 보임)
//   - numpy 4x4 @ 4x4 (OpenBLAS SkylakeX sgemm): c = fma(a3,b3, fma(a2,b2, fma(a1,b1, a0*b0)))
// 주의: 위 둘(numba 꼬리, OpenBLAS 커널)은 CPU 가 고른 경로라 CPU 종류(AVX512/FMA 유무)가 다르면 바뀔 수 있다.
//   확인한 CPU: Ryzen 9 9950X (Zen 5, numpy 1.26.0 OpenBLAS 코어 "SkylakeX"). 대회 측정 장비 7950X(Zen 4)도 AVX512 라 같은 경로일 것(추정).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define OEHD __host__ __device__ __forceinline__
#else
#define OEHD inline
#endif

namespace eng {
namespace omni {

OEHD float f32fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return std::fmaf(a, b, c);
#endif
}

// np.clip (numpy 1.26 umath/clip.cpp): _NPY_MIN(_NPY_MAX(x, lo), hi),
//   _NPY_MAX(a,b) = isnan(a) ? a : (a > b ? a : b), _NPY_MIN(a,b) = isnan(a) ? a : (a < b ? a : b)
//   → 부호 있는 0 도 이 규칙대로 (clip(-0, lo=+0) = +0).
OEHD float np_clip(float x, float lo, float hi) {
  float y = (x != x) ? x : (x > lo ? x : lo);
  y = (y != y) ? y : (y < hi ? y : hi);
  return y;
}

// 4x4 행렬 (행 우선, numpy 와 같음)
struct M4 {
  float m[4][4];
};

OEHD M4 m4_zero() {
  M4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r.m[i][j] = 0.0f;
  return r;
}
OEHD M4 m4_eye() {
  M4 r = m4_zero();
  for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
  return r;
}

// numba _quat2mat (transform_utils_np.py:378). q = (x, y, z, w)
OEHD void np_quat2mat(const float q[4], float R[3][3]) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, xw = x * w, yw = y * w, zw = z * w;
  R[0][0] = 1.0f - 2.0f * (yy + zz);
  R[0][1] = 2.0f * (xy - zw);
  R[0][2] = 2.0f * (xz + yw);
  R[1][0] = 2.0f * (xy + zw);
  R[1][1] = 1.0f - 2.0f * (xx + zz);
  R[1][2] = 2.0f * (yz - xw);
  R[2][0] = 2.0f * (xz - yw);
  R[2][1] = 2.0f * (yz + xw);
  R[2][2] = 1.0f - 2.0f * (xx + yy);
}

// numba pose2mat (transform_utils_np.py:698)
OEHD M4 np_pose2mat(const float p[3], const float q[4]) {
  M4 r = m4_zero();
  float R[3][3];
  np_quat2mat(q, R);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = R[i][j];
  r.m[0][3] = p[0];
  r.m[1][3] = p[1];
  r.m[2][3] = p[2];
  r.m[3][3] = 1.0f;
  return r;
}

// numba _pose_inv (transform_utils_np.py:771)
OEHD M4 np_pose_inv(const M4& a) {
  M4 r = m4_zero();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[j][i];
  const float t0 = a.m[0][3], t1 = a.m[1][3], t2 = a.m[2][3];
  for (int i = 0; i < 2; ++i) {
    const float s = (r.m[i][0] * t0 + r.m[i][1] * t1) + r.m[i][2] * t2;
    r.m[i][3] = -s;
  }
  {
    const float s = f32fma(r.m[2][2], t2, f32fma(r.m[2][0], t0, r.m[2][1] * t1));
    r.m[2][3] = -s;
  }
  r.m[3][3] = 1.0f;
  return r;
}

// numpy a @ b (float32 4x4, OpenBLAS sgemm)
OEHD M4 np_matmul4(const M4& a, const M4& b) {
  M4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float s = a.m[i][0] * b.m[0][j];
      s = f32fma(a.m[i][1], b.m[1][j], s);
      s = f32fma(a.m[i][2], b.m[2][j], s);
      s = f32fma(a.m[i][3], b.m[3][j], s);
      r.m[i][j] = s;
    }
  return r;
}

}  // namespace omni
}  // namespace eng
