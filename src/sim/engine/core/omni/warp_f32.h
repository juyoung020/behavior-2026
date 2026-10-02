// OmniGibson 물체 상태의 warp 커널(CUDA, NVRTC --fmad=true)을 비트까지 흉내 내는 float32 수학 (손으로 짬, EHD).
// warp 는 곱-덧셈 축약(FMA)을 켠 채로 컴파일한다(warp/native/warp.cu:3889 "--fmad=true"). 그래서 식을 그대로 옮기면
// 끝비트가 다르다. 아래 순서는 공식 Linux 실행이 남긴 warp 캐시 PTX(~/behavior-linux/appdata/global/warp_cache/*.sm120.ptx)와
// 그것을 ptxas 로 굽은 SASS 에서 읽었다 (NVVM 이 fma.rn 으로 묶은 곳 + ptxas 가 mul/add 짝을 FFMA 로 묶은 곳).
//   dot3(a, b)          = fma(a2,b2, fma(a0,b0, a1*b1))        (warp vec dot, mat*vec 의 행마다)
//   transform_point(M,p) = dot3(M_i, p) + M_i3                   (usd_utils _aabb_reduce_kernel, inside 반공간 시험)
//   mat44 * mat44       = fma(a_i3,b_3j, fma(a_i2,b_2j, fma(a_i1,b_1j, fma(a_i0,b_0j, 0))))  (inside _inside_inv_world_kernel)
//   rigid_inverse 의 -R^T t: s_i = fma(P_2i,t2, fma(P_0i,t0, P_1i*t1)), nt_i = -s_i   (usd_utils.py:30)
// 이 층의 GPU 판(층 2)은 -fmad=false 로 짓고 여기 적힌 fma 만 명시적으로 쓴다(FTZ 도 끔: warp 는 FTZ 를 쓰지 않는다).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#include "core/omni/npf32.h"

namespace eng {
namespace omni {
namespace wf {

OEHD float fma_(float a, float b, float c) { return f32fma(a, b, c); }

OEHD float sqrt_rn(float x) {
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return std::sqrt(x);
#endif
}
OEHD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
OEHD float rcp_rn(float a) { return div_rn(1.0f, a); }
OEHD float wmin(float a, float b) { return a < b ? a : b; }  // warp builtin.h:457
OEHD float wmax(float a, float b) { return a > b ? a : b; }  // warp builtin.h:458
OEHD float wabs(float a) { return std::fabs(a); }
OEHD uint32_t fbits(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}
OEHD float bitsf(uint32_t u) {
  float x;
  memcpy(&x, &u, 4);
  return x;
}
OEHD bool isinf_(float x) { return (fbits(x) & 0x7fffffffu) == 0x7f800000u; }

// 4x4 행 우선 (wp.mat44 메모리 배치와 같음)
struct M44 {
  float m[16];
  OEHD float operator()(int i, int j) const { return m[i * 4 + j]; }
};

OEHD float dot3(const float a[3], const float b[3]) { return fma_(a[2], b[2], fma_(a[0], b[0], a[1] * b[1])); }

// usd_utils.py:541 _poses_to_matrices_kernel. SASS 에서 읽은 축약:
//   |q|^2 = fma(qw,qw, fma(qz,qz, fma(qx,qx, qy*qy))), 정규화는 sqrt.rn / rcp.rn
//   xy-zw = fma(qx,qy,-zw), xy+zw = fma(qx,qy,zw), yz-xw = fma(qy,qz,-xw), yz+xw = fma(qy,qz,xw),
//   xz+yw = fma(qx,qz,yw), xz-yw = fma(qx,qz,-yw), xx+zz = fma(qx,qx,zz), xx+yy = fma(qx,qx,yy), yy+zz = yy+zz (축약 없음)
//   2*(s) = s+s, 1-2*(s) = 1-(s+s)
OEHD M44 pose_to_mat(const float* pose7) {
  const float px = pose7[0], py = pose7[1], pz = pose7[2];
  float qx = pose7[3], qy = pose7[4], qz = pose7[5], qw = pose7[6];
  const float n2 = fma_(qw, qw, fma_(qz, qz, fma_(qx, qx, qy * qy)));
  const float n = sqrt_rn(n2);
  const float inv = rcp_rn(n);
  qx = qx * inv;
  qy = qy * inv;
  qz = qz * inv;
  qw = qw * inv;
  const float yy = qy * qy, zz = qz * qz;
  const float zw = qz * qw, xw = qx * qw, yw = qy * qw;
  const float xy_m_zw = fma_(qx, qy, -zw), xy_p_zw = fma_(qx, qy, zw);
  const float yz_m_xw = fma_(qy, qz, -xw), yz_p_xw = fma_(qy, qz, xw);
  const float xz_p_yw = fma_(qx, qz, yw), xz_m_yw = fma_(qx, qz, -yw);
  const float yy_zz = yy + zz;
  const float xx_zz = fma_(qx, qx, zz);
  const float xx_yy = fma_(qx, qx, yy);
  M44 r;
  r.m[0] = 1.0f - (yy_zz + yy_zz);
  r.m[1] = xy_m_zw + xy_m_zw;
  r.m[2] = xz_p_yw + xz_p_yw;
  r.m[3] = px;
  r.m[4] = xy_p_zw + xy_p_zw;
  r.m[5] = 1.0f - (xx_zz + xx_zz);
  r.m[6] = yz_m_xw + yz_m_xw;
  r.m[7] = py;
  r.m[8] = xz_m_yw + xz_m_yw;
  r.m[9] = yz_p_xw + yz_p_xw;
  r.m[10] = 1.0f - (xx_yy + xx_yy);
  r.m[11] = pz;
  r.m[12] = 0.0f;
  r.m[13] = 0.0f;
  r.m[14] = 0.0f;
  r.m[15] = 1.0f;
  return r;
}

// M * (p, 1): 행마다 dot3(M_i, p) + M_i3   (PTX: mul, fma, fma, add)
OEHD void transform_point(const M44& M, const float p[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    const float row[3] = {M(i, 0), M(i, 1), M(i, 2)};
    out[i] = dot3(row, p) + M(i, 3);
  }
}

// rigid_inverse_mat44 (usd_utils.py:30) 의 s_i (nt_i = -s_i)
OEHD void rigid_inverse_s(const M44& P, float s[3]) {
  const float t[3] = {P(0, 3), P(1, 3), P(2, 3)};
  for (int i = 0; i < 3; ++i) {
    const float col[3] = {P(0, i), P(1, i), P(2, i)};  // R^T 의 i 행 = R 의 i 열
    s[i] = dot3(col, t);
  }
}
OEHD M44 rigid_inverse(const M44& P) {
  float s[3];
  rigid_inverse_s(P, s);
  M44 r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r.m[i * 4 + j] = P(j, i);
    r.m[i * 4 + 3] = -s[i];
  }
  r.m[12] = 0.0f;
  r.m[13] = 0.0f;
  r.m[14] = 0.0f;
  r.m[15] = 1.0f;
  return r;
}

// wp.mul(mat44, mat44): k 순서 fma 사슬, 누산기 0 에서 시작 (SASS: FFMA ..., RZ)
OEHD M44 mat44_mul(const M44& A, const M44& B) {
  M44 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float acc = fma_(A(i, 0), B(0, j), 0.0f);
      acc = fma_(A(i, 1), B(1, j), acc);
      acc = fma_(A(i, 2), B(2, j), acc);
      acc = fma_(A(i, 3), B(3, j), acc);
      r.m[i * 4 + j] = acc;
    }
  return r;
}

}  // namespace wf
}  // namespace omni
}  // namespace eng
