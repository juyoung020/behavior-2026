// OmniGibson 카메라·링크 세계 자세 읽기 (utils/usd_utils.py:2138 get_world_pose) 를 손으로:
//   matrix.RemoveScaleShear().ExtractRotationQuat() / ExtractTranslation() -> float32
// 원본 = OpenUSD Gf (v24.05, pxr/base/gf/matrix4d.cpp: Factor :904, _Jacobi3 :961, RemoveScaleShear :1063, ExtractRotationQuat :1080,
//        Orthonormalize :467, operator*= :585, _GetDeterminant3 :436 / vec3d.cpp GfOrthogonalizeBasis :111 / vec3d.h Normalize :269)
// 식·연산 순서를 원본 그대로 옮겼다 (double, FMA 없이 -ffp-contract=off). usdrt.Gf 결과와 비트 대조: tests/omni/test_gfmat.py
// 행 벡터 규약 (p_world = p * M, 행 3 = 이동), m[r][c] = m[r*4+c].
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define GFHD __host__ __device__ inline
#else
#define GFHD inline
#endif

namespace eng {
namespace omni {
namespace gf {

struct M4 {
  double m[4][4];
};
struct V3 {
  double v[3];
};

GFHD M4 ident(double s = 1.0) {
  M4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r.m[i][j] = i == j ? s : 0.0;
  return r;
}

// operator*= (matrix4d.cpp:585): 원소마다 a0*b0 + a1*b1 + a2*b2 + a3*b3 (왼쪽부터)
GFHD M4 mul(const M4& t, const M4& b) {
  M4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      r.m[i][j] = t.m[i][0] * b.m[0][j] + t.m[i][1] * b.m[1][j] + t.m[i][2] * b.m[2][j] + t.m[i][3] * b.m[3][j];
  return r;
}

GFHD M4 transpose(const M4& a) {
  M4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r.m[j][i] = a.m[i][j];
  return r;
}

GFHD double det3(const M4& a) {  // _GetDeterminant3(0,1,2, 0,1,2)
  const int r1 = 0, r2 = 1, r3 = 2, c1 = 0, c2 = 1, c3 = 2;
  return (a.m[r1][c1] * a.m[r2][c2] * a.m[r3][c3] + a.m[r1][c2] * a.m[r2][c3] * a.m[r3][c1] +
          a.m[r1][c3] * a.m[r2][c1] * a.m[r3][c2] - a.m[r1][c1] * a.m[r2][c3] * a.m[r3][c2] -
          a.m[r1][c2] * a.m[r2][c1] * a.m[r3][c3] - a.m[r1][c3] * a.m[r2][c2] * a.m[r3][c1]);
}

// _Jacobi3 (matrix4d.cpp:961, Open Inventor SbMatrix::Jacobi3)
GFHD void jacobi3(const M4& self, double ev[3], double evec[3][3]) {
  ev[0] = self.m[0][0];
  ev[1] = self.m[1][1];
  ev[2] = self.m[2][2];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) evec[i][j] = i == j ? 1.0 : 0.0;
  M4 a = self;
  double b[3] = {ev[0], ev[1], ev[2]};
  double z[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 50; i++) {
    double sm = 0.0;
    for (int p = 0; p < 2; p++)
      for (int q = p + 1; q < 3; q++) sm += std::fabs(a.m[p][q]);
    if (sm == 0.0) return;
    double thresh = (i < 3 ? (.2 * sm / (3 * 3)) : 0.0);
    for (int p = 0; p < 3; p++) {
      for (int q = p + 1; q < 3; q++) {
        double g = 100.0 * std::fabs(a.m[p][q]);
        if (i > 3 && (std::fabs(ev[p]) + g == std::fabs(ev[p])) && (std::fabs(ev[q]) + g == std::fabs(ev[q])))
          a.m[p][q] = 0.0;
        else if (std::fabs(a.m[p][q]) > thresh) {
          double h = ev[q] - ev[p];
          double t;
          if (std::fabs(h) + g == std::fabs(h)) {
            t = a.m[p][q] / h;
          } else {
            double theta = 0.5 * h / a.m[p][q];
            t = 1.0 / (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
            if (theta < 0.0) t = -t;
          }
          double c = 1.0 / std::sqrt(1.0 + t * t);
          double s = t * c;
          double tau = s / (1.0 + c);
          h = t * a.m[p][q];
          z[p] -= h;
          z[q] += h;
          ev[p] -= h;
          ev[q] += h;
          a.m[p][q] = 0.0;
          for (int j = 0; j < p; j++) {
            g = a.m[j][p];
            h = a.m[j][q];
            a.m[j][p] = g - s * (h + g * tau);
            a.m[j][q] = h + s * (g - h * tau);
          }
          for (int j = p + 1; j < q; j++) {
            g = a.m[p][j];
            h = a.m[j][q];
            a.m[p][j] = g - s * (h + g * tau);
            a.m[j][q] = h + s * (g - h * tau);
          }
          for (int j = q + 1; j < 3; j++) {
            g = a.m[p][j];
            h = a.m[q][j];
            a.m[p][j] = g - s * (h + g * tau);
            a.m[q][j] = h + s * (g - h * tau);
          }
          for (int j = 0; j < 3; j++) {
            g = evec[j][p];
            h = evec[j][q];
            evec[j][p] = g - s * (h + g * tau);
            evec[j][q] = h + s * (g - h * tau);
          }
        }
      }
    }
    for (int p = 0; p < 3; p++) {
      ev[p] = b[p] += z[p];
      z[p] = 0;
    }
  }
}

// Factor (matrix4d.cpp:904) 중 u (= R S^-1 R^T A) 와 이동만 쓴다
GFHD bool factor(const M4& self, M4* u, double t[3], double eps = 1e-10) {
  M4 a;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) a.m[i][j] = self.m[i][j];
    a.m[3][i] = a.m[i][3] = 0.0;
    t[i] = self.m[3][i];
  }
  a.m[3][3] = 1.0;
  double det = det3(a);
  double detSign = (det < 0.0 ? -1.0 : 1.0);
  bool isSingular = det * detSign < eps;
  M4 b = mul(a, transpose(a));
  double ev[3], evec[3][3];
  jacobi3(b, ev, evec);
  M4 r = ident();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = evec[i][j];
  r.m[0][3] = r.m[1][3] = r.m[2][3] = 0.0;
  r.m[3][0] = r.m[3][1] = r.m[3][2] = 0.0;
  r.m[3][3] = 1.0;
  M4 sInv = ident();
  for (int i = 0; i < 3; i++) {
    double s = ev[i] < eps ? detSign * eps : detSign * std::sqrt(ev[i]);
    sInv.m[i][i] = 1.0 / s;
  }
  // *u = *r * sInv * r->GetTranspose() * a  (왼쪽부터 곱함)
  *u = mul(mul(mul(r, sInv), transpose(r)), a);
  return !isSingular;
}

GFHD double dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
GFHD void normalize(double v[3], double eps = 1e-10) {  // vec3d.h:269 — 나누기는 역수 곱 (operator/= :219)
  double length = std::sqrt(dot(v, v));
  double inv = 1.0 / ((length > eps) ? length : eps);
  v[0] *= inv;
  v[1] *= inv;
  v[2] *= inv;
}
GFHD bool is_close(const double a[3], const double b[3], double tol) {
  double d[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  return dot(d, d) <= tol * tol;
}

// GfOrthogonalizeBasis (vec3d.cpp:111), normalize = true, eps = GF_MIN_ORTHO_TOLERANCE 1e-6
GFHD bool orthogonalize_basis(double tx[3], double ty[3], double tz[3], double eps = 1e-6) {
  double ax[3], ay[3], az[3], bx[3], by[3], bz[3], cx[3], cy[3], cz[3];
  normalize(tx);
  normalize(ty);
  normalize(tz);
  for (int k = 0; k < 3; ++k) { ax[k] = tx[k]; ay[k] = ty[k]; az[k] = tz[k]; }
  if (is_close(ax, ay, eps) || is_close(ax, az, eps) || is_close(ay, az, eps)) return false;
  const int MAX_ITERS = 20;
  int iter;
  for (iter = 0; iter < MAX_ITERS; ++iter) {
    for (int k = 0; k < 3; ++k) { bx[k] = tx[k]; by[k] = ty[k]; bz[k] = tz[k]; }
    // bx -= GfDot(ay,bx) * ay  (스칼라 × 벡터 = 원소마다 곱)
    double d;
    d = dot(ay, bx); for (int k = 0; k < 3; ++k) bx[k] -= ay[k] * d;
    d = dot(az, bx); for (int k = 0; k < 3; ++k) bx[k] -= az[k] * d;
    d = dot(ax, by); for (int k = 0; k < 3; ++k) by[k] -= ax[k] * d;
    d = dot(az, by); for (int k = 0; k < 3; ++k) by[k] -= az[k] * d;
    d = dot(ax, bz); for (int k = 0; k < 3; ++k) bz[k] -= ax[k] * d;
    d = dot(ay, bz); for (int k = 0; k < 3; ++k) bz[k] -= ay[k] * d;
    // cx = 0.5*(*tx + bx)
    for (int k = 0; k < 3; ++k) {
      cx[k] = (tx[k] + bx[k]) * 0.5;
      cy[k] = (ty[k] + by[k]) * 0.5;
      cz[k] = (tz[k] + bz[k]) * 0.5;
    }
    normalize(cx);
    normalize(cy);
    normalize(cz);
    double xd[3], yd[3], zd[3];
    for (int k = 0; k < 3; ++k) { xd[k] = tx[k] - cx[k]; yd[k] = ty[k] - cy[k]; zd[k] = tz[k] - cz[k]; }
    double error = dot(xd, xd) + dot(yd, yd) + dot(zd, zd);
    if (error < eps * eps) break;
    for (int k = 0; k < 3; ++k) { tx[k] = cx[k]; ty[k] = cy[k]; tz[k] = cz[k]; }
    for (int k = 0; k < 3; ++k) { ax[k] = tx[k]; ay[k] = ty[k]; az[k] = tz[k]; }
  }
  return iter < MAX_ITERS;
}

// Orthonormalize (matrix4d.cpp:467)
GFHD void orthonormalize(M4& a) {
  double r0[3] = {a.m[0][0], a.m[0][1], a.m[0][2]};
  double r1[3] = {a.m[1][0], a.m[1][1], a.m[1][2]};
  double r2[3] = {a.m[2][0], a.m[2][1], a.m[2][2]};
  orthogonalize_basis(r0, r1, r2);
  for (int k = 0; k < 3; ++k) { a.m[0][k] = r0[k]; a.m[1][k] = r1[k]; a.m[2][k] = r2[k]; }
  if (a.m[3][3] != 1.0 && !(std::fabs(a.m[3][3] - 0.0) < 1e-10)) {
    a.m[3][0] /= a.m[3][3];
    a.m[3][1] /= a.m[3][3];
    a.m[3][2] /= a.m[3][3];
    a.m[3][3] = 1.0;
  }
}

// RemoveScaleShear (matrix4d.cpp:1063)
GFHD M4 remove_scale_shear(const M4& self) {
  M4 u;
  double t[3];
  if (!factor(self, &u, t)) return self;
  orthonormalize(u);
  M4 tr = ident();
  tr.m[3][0] = t[0];
  tr.m[3][1] = t[1];
  tr.m[3][2] = t[2];
  return mul(u, tr);
}

// ExtractRotationQuat (matrix4d.cpp:1080) -> (x, y, z, w) double
GFHD void extract_rotation_quat(const M4& m, double q[4]) {
  int i;
  if (m.m[0][0] > m.m[1][1])
    i = (m.m[0][0] > m.m[2][2] ? 0 : 2);
  else
    i = (m.m[1][1] > m.m[2][2] ? 1 : 2);
  double im[3], r;
  if (m.m[0][0] + m.m[1][1] + m.m[2][2] > m.m[i][i]) {
    r = 0.5 * std::sqrt(m.m[0][0] + m.m[1][1] + m.m[2][2] + m.m[3][3]);
    im[0] = (m.m[1][2] - m.m[2][1]) / (4.0 * r);
    im[1] = (m.m[2][0] - m.m[0][2]) / (4.0 * r);
    im[2] = (m.m[0][1] - m.m[1][0]) / (4.0 * r);
  } else {
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    double qq = 0.5 * std::sqrt(m.m[i][i] - m.m[j][j] - m.m[k][k] + m.m[3][3]);
    im[i] = qq;
    im[j] = (m.m[i][j] + m.m[j][i]) / (4 * qq);
    im[k] = (m.m[k][i] + m.m[i][k]) / (4 * qq);
    r = (m.m[j][k] - m.m[k][j]) / (4 * qq);
  }
  // GfClamp(r, -1, 1)
  r = r < -1.0 ? -1.0 : (r > 1.0 ? 1.0 : r);
  q[0] = im[0];
  q[1] = im[1];
  q[2] = im[2];
  q[3] = r;
}

// ---------------- usdrt 판 (OmniGibson 이 실제로 부르는 것: usdrt.Gf.Matrix4d, kit/dev/fabric/include/usdrt/gf) ----------------
// usdrt 는 pxr 와 다르게 짰다: RemoveScaleShear(matrix.h:1253) = 행 세 개만 GfOrthogonalizeBasis(vec.h:851) + 행렬식 음수면 z 행 뒤집기
// (Factor·Jacobi 안 씀). 정규화는 역수 곱이 아니라 나눗셈(vec.h:171 Normalize, operator/= :233), 길이² 0 이면 0 벡터.
GFHD void rt_normalize(double v[3]) {
  const double l2 = dot(v, v);
  if (l2 != 0.0) {
    const double l = std::sqrt(l2);
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
  } else {
    v[0] *= 0;
    v[1] *= 0;
    v[2] *= 0;
  }
}
GFHD void rt_complement(const double v[3], const double u[3], double o[3]) {  // v - u * (v·u)
  const double d = dot(v, u);
  for (int k = 0; k < 3; ++k) o[k] = v[k] - u[k] * d;
}
GFHD bool rt_orthogonalize_basis(double pa[3], double pb[3], double pc[3]) {
  double a[3] = {pa[0], pa[1], pa[2]}, b[3] = {pb[0], pb[1], pb[2]}, c[3] = {pc[0], pc[1], pc[2]};
  rt_normalize(a);
  rt_normalize(b);
  rt_normalize(c);
  rt_normalize(pa);
  rt_normalize(pb);
  rt_normalize(pc);
  const double tolerance = 4.8e-7;
  const double toleranceSq = tolerance * tolerance;
  const int maxIterations = 32;
  int iteration;
  for (iteration = 0; iteration < maxIterations; ++iteration) {
    double nA[3], nB[3], nC[3], t[3];
    rt_complement(a, b, t);
    rt_complement(t, c, nA);
    rt_complement(b, c, t);
    rt_complement(t, a, nB);
    rt_complement(c, a, t);
    rt_complement(t, b, nC);
    if (iteration == 0) {
      const double la = dot(nA, nA), lb = dot(nB, nB), lc = dot(nC, nC);
      if (!(la >= toleranceSq) || !(lb >= toleranceSq) || !(lc >= toleranceSq)) return false;
    }
    for (int k = 0; k < 3; ++k) {  // S(0.5) * (a + newA): 먼저 더하고 곱함
      nA[k] = 0.5 * (a[k] + nA[k]);
      nB[k] = 0.5 * (b[k] + nB[k]);
      nC[k] = 0.5 * (c[k] + nC[k]);
    }
    rt_normalize(nA);
    rt_normalize(nB);
    rt_normalize(nC);
    double da[3], db[3], dc[3];
    for (int k = 0; k < 3; ++k) { da[k] = nA[k] - a[k]; db[k] = nB[k] - b[k]; dc[k] = nC[k] - c[k]; }
    const double changeSq = dot(da, da) + dot(db, db) + dot(dc, dc);
    if (changeSq < toleranceSq) break;
    for (int k = 0; k < 3; ++k) { a[k] = nA[k]; b[k] = nB[k]; c[k] = nC[k]; }
    for (int k = 0; k < 3; ++k) { pa[k] = a[k]; pb[k] = b[k]; pc[k] = c[k]; }
  }
  return iteration < maxIterations;
}
GFHD M4 rt_remove_scale_shear(const M4& self) {
  double r0[3] = {self.m[0][0], self.m[0][1], self.m[0][2]};
  double r1[3] = {self.m[1][0], self.m[1][1], self.m[1][2]};
  double r2[3] = {self.m[2][0], self.m[2][1], self.m[2][2]};
  if (!rt_orthogonalize_basis(r0, r1, r2)) return self;
  const double cr[3] = {r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]};
  if (dot(r0, cr) < 0)
    for (int k = 0; k < 3; ++k) r2[k] = -r2[k];
  M4 o = ident();
  for (int k = 0; k < 3; ++k) { o.m[0][k] = r0[k]; o.m[1][k] = r1[k]; o.m[2][k] = r2[k]; o.m[3][k] = self.m[3][k]; }
  return o;
}

// get_world_pose: 위치·쿼터니언 float32 (th.tensor(..., float32) 는 double -> float 반올림). usdrt 판을 쓴다
GFHD void world_pose_f32(const M4& world, float pos[3], float quat[4]) {
  const M4 rs = rt_remove_scale_shear(world);
  double q[4];
  extract_rotation_quat(rs, q);
  pos[0] = float(world.m[3][0]);
  pos[1] = float(world.m[3][1]);
  pos[2] = float(world.m[3][2]);
  for (int k = 0; k < 4; ++k) quat[k] = float(q[k]);
}

// PhysX 자세(float32 쿼터니언 x,y,z,w + 위치) -> pxr GfMatrix4d::SetRotate(GfQuatd) 식의 double 행렬 (link 세계 행렬, obs_engine._rot_pxr 와 같음)
GFHD M4 from_physx_pose(const float p[3], const float q[4]) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  M4 m = ident();
  m.m[0][0] = 1.0 - 2.0 * (y * y + z * z);
  m.m[0][1] = 2.0 * (x * y + z * w);
  m.m[0][2] = 2.0 * (z * x - y * w);
  m.m[1][0] = 2.0 * (x * y - z * w);
  m.m[1][1] = 1.0 - 2.0 * (z * z + x * x);
  m.m[1][2] = 2.0 * (y * z + x * w);
  m.m[2][0] = 2.0 * (z * x + y * w);
  m.m[2][1] = 2.0 * (y * z - x * w);
  m.m[2][2] = 1.0 - 2.0 * (y * y + x * x);
  m.m[3][0] = p[0];
  m.m[3][1] = p[1];
  m.m[3][2] = p[2];
  return m;
}

}  // namespace gf
}  // namespace omni
}  // namespace eng
