// joints 모듈: D6 조인트 셰이더 (상수 블록 + 두 몸체 자세 -> Px1DConstraint 행들). PhysX 5.6.1 과 비트 동일 목표. CPU·CUDA 공용.
// 원본(physx/ 기준)
//   셰이더   : source/physxextensions/src/ExtD6Joint.cpp:1002 (D6JointSolverPrep) 와 그 위 보조 함수(:790, :971-999)
//   행 도우미: source/physxextensions/src/ExtConstraintHelper.h (ConstraintHelper, computeJacobianAxes, _linear/_angular)
//   각도     : source/physxextensions/src/ExtJoint.h:66 (computeSwingAngle), ExtD6Joint.cpp:722 (computePhi)
//   원뿔 한계: source/common/src/CmConeLimitHelper.h:52,163 (ConeLimitHelperTanLess), include/foundation/PxMathUtils.h:204 (PxEllipseClamp)
//   SIMD    : include/foundation/PxSIMDHelpers.h:77 (transformMultiply), PxMat33Padded = QuatGetMat33V (common/pmath.h mat_from_quat_simd)
// 행 배열은 호출 전에 setupConstraintRows 로 초기화한다 (DyConstraintPrep.h:115 와 같음).
// libm: 한계(limit) 경로만 atan2f/asinf/acosf/tanf 를 쓴다(PhysX 는 ::atan2f 등, PxMath.h:245-315). atan2f/asinf/acosf 는
//       glibc 2.35 이식본(glibc_trig.h)을 CPU·GPU 공용으로 쓴다. tanf(원뿔 한계)만 아직 libm/CUDA 그대로 — GPU 비트 다를 수 있음.
#pragma once
#include <cmath>
#include <cstdint>

#include "../common/aos.h"
#include "d6_joint.h"
#include "glibc_trig.h"

namespace eng {
namespace jnt {

namespace lm {
// glibc 2.35 이식본 (glibc_trig.h) — CPU·GPU 같은 코드. CPU 에서 libm 과 float 2^32 전수 비트 동일 확인(tests/joints/test_libm_joints)
EHD float atan2f_(float y, float x) { return glibcj::atan2f(y, x); }
EHD float asinf_(float x) { return glibcj::asinf(x); }
EHD float acosf_(float x) { return glibcj::acosf(x); }
#if defined(__CUDA_ARCH__)
EHD float tanf_(float x) { return ::tanf(x); }  // TODO: glibc s_tanf 이식 (원뿔 한계에서만 씀) — GPU 비트 다를 수 있음
#else
inline float tanf_(float x) { return ::tanf(x); }
#endif
EHD float pclamp(float v, float lo, float hi) { return pmin(hi, pmax(lo, v)); }  // PxClamp = PxMin(hi, PxMax(lo, v))
EHD float PxAsin(float f) { return asinf_(pclamp(f, -1.0f, 1.0f)); }
EHD float PxAcos(float f) { return acosf_(pclamp(f, -1.0f, 1.0f)); }
EHD float PxAtan2(float y, float x) { return atan2f_(y, x); }
EHD float PxTan(float a) { return tanf_(a); }
EHD float PxRecipSqrt(float a) { return 1.0f / psqrt(a); }
}  // namespace lm

EHD void setupConstraintRows(Row* rows, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    Row& c = rows[i];
    c = Row{};
    c.minImpulse = -MAX_F32;
    c.maxImpulse = MAX_F32;
  }
}

// aos::transformMultiply<.,.> (physx/include/foundation/PxSIMDHelpers.h:77 transformKernelVec4) — out = a * b
EHD Tf transformMultiplyV(const Tf& a, const Tf& b) {
  using namespace aos;
  const float ap[4] = {a.p.x, a.p.y, a.p.z, 0.0f}, aq[4] = {a.q.x, a.q.y, a.q.z, a.q.w};
  const float bp[4] = {b.p.x, b.p.y, b.p.z, 0.0f}, bq[4] = {b.q.x, b.q.y, b.q.z, b.q.w};
  const Vec4V aPos = V4LoadU(ap), aRot = V4LoadU(aq), bPos = V4LoadU(bp), bRot = V4LoadU(bq);
  const FloatV wa = V4GetW(aRot), wb = V4GetW(bRot);
  const Vec4V va = aRot, pa = aPos, vb = bRot, pb = bPos;
  const FloatV wo = FSub(FMul(wa, wb), V4Dot3(va, vb));
  const Vec4V vo = V4ScaleAdd(va, wb, V4ScaleAdd(vb, wa, V4Cross(va, vb)));
  const Vec4V t1 = V4Scale(pb, FScaleAdd(wa, wa, FLoad(-0.5f)));
  const Vec4V t2 = V4ScaleAdd(V4Cross(va, pb), wa, t1);
  const Vec4V t3 = V4ScaleAdd(va, V4Dot3(va, pb), t2);
  const Vec4V po = V4ScaleAdd(t3, FLoad(2.0f), pa);
  float o[4], q[4];
  V4StoreU(po, o);
  V4StoreU(V4SetW(vo, wo), q);
  return Tf{Q{q[0], q[1], q[2], q[3]}, V3{o[0], o[1], o[2]}};
}

// ---- 스칼라 PxVec3/PxQuat 보조 (PxVec3.h, PxQuat.h 의 연산 순서)
EHD V3 vsub(const V3& a, const V3& b) { return a - b; }
EHD V3 vscale(const V3& a, float s) { return a * s; }
EHD V3 vneg(const V3& a) { return V3{-a.x, -a.y, -a.z}; }  // PxVec3::operator- (부호 반전)
EHD Q qneg(const Q& a) { return Q{-a.x, -a.y, -a.z, -a.w}; }
EHD float qdot(const Q& a, const Q& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
EHD V3 vgetNormalized(const V3& v) {  // PxVec3::getNormalized
  const float m = magSq(v);
  return m > 0.0f ? v * lm::PxRecipSqrt(m) : V3{0, 0, 0};
}
EHD V3 vdiv(const V3& v, float f) { f = 1.0f / f; return V3{v.x * f, v.y * f, v.z * f}; }  // PxVec3::operator/
EHD V3 basis0(const Q& q) {
  const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
  return V3{(q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2};
}
EHD V3 basis1(const Q& q) {
  const float y2 = q.y * 2.0f, w2 = q.w * 2.0f;
  return V3{(-q.z * w2) + q.x * y2, (q.w * w2) - 1.0f + q.y * y2, (q.x * w2) + q.z * y2};
}
EHD V3 basis2(const Q& q) {
  const float z2 = q.z * 2.0f, w2 = q.w * 2.0f;
  return V3{(q.y * w2) + q.x * z2, (-q.x * w2) + q.y * z2, (q.w * w2) - 1.0f + q.z * z2};
}
EHD Q qnormalize(Q q) {  // PxQuat::normalize (mag != 0 일 때만)
  const float mag = psqrt(qmagSq(q));
  if (mag != 0.0f) {
    const float imag = 1.0f / mag;
    q.x *= imag; q.y *= imag; q.z *= imag; q.w *= imag;
  }
  return q;
}
EHD void separateSwingTwist(const Q& q, Q& swing, Q& twist) {  // PxMathUtils.h:272
  twist = q.x != 0.0f ? normalized(Q{q.x, 0, 0, q.w}) : qid();
  swing = q * conj(twist);
}
EHD float computeSwingAngle(float swingYZ, float swingW) { return 4.0f * lm::PxAtan2(swingYZ, 1.0f + swingW); }
EHD float computePhi(const Q& q) {  // ExtD6Joint.cpp:722
  const Q twist = qnormalize(q);
  float angle = lm::PxAcos(twist.w) * 2.0f;  // PxQuat::getAngle
  if (twist.x < 0.0f) angle = -angle;
  return angle;
}
EHD V3 col(const M33& m, int i) { return i == 0 ? m.c0 : (i == 1 ? m.c1 : m.c2); }
EHD float comp(const V3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

// computeJacobianAxes (ExtConstraintHelper.h:84)
EHD void computeJacobianAxes(V3 row[3], const Q& qa, const Q& qb) {
  const float wa = qa.w, wb = qb.w;
  const V3 va{qa.x, qa.y, qa.z}, vb{qb.x, qb.y, qb.z};
  const V3 c = vb * wa + va * wb;
  const float d0 = wa * wb;
  const float d1 = dot(va, vb);
  const float d = d0 - d1;
  row[0] = (va * vb.x + vb * va.x + V3{d, c.z, -c.y}) * 0.5f;
  row[1] = (va * vb.y + vb * va.y + V3{-c.z, d, c.x}) * 0.5f;
  row[2] = (va * vb.z + vb * va.z + V3{c.y, -c.x, d}) * 0.5f;
  if ((d0 + d1) != 0.0f) return;
  const float eps = 1.192092896e-07f;  // PX_EPS_F32 = FLT_EPSILON
  row[0].x += eps;
  row[1].y += eps;
  row[2].z += eps;
}

// ---- ConstraintHelper (ExtConstraintHelper.h:130)
struct Helper {
  Row* rows;
  Row* cur;
  V3 ra, rb;

  EHD Row* _linear(const V3& axis, const V3& ra_, const V3& rb_, float posErr, uint16_t hint, Row* c) {
    c->solveHint = hint;
    c->linear0 = axis;
    c->angular0 = cross(ra_, axis);
    c->linear1 = axis;
    c->angular1 = cross(rb_, axis);
    c->geometricError = posErr;
    return c;
  }
  EHD Row* _angular(const V3& axis, float posErr, uint16_t hint, Row* c) {
    c->solveHint = hint;
    c->linear0 = V3{0, 0, 0};
    c->angular0 = axis;
    c->linear1 = V3{0, 0, 0};
    c->angular1 = axis;
    c->geometricError = posErr;
    c->flags |= RF_ANGULAR_CONSTRAINT;
    return c;
  }
  EHD Row* linear(const V3& axis, float posErr, uint16_t hint) { return _linear(axis, ra, rb, posErr, hint, cur++); }
  EHD Row* angular(const V3& axis, float posErr, uint16_t hint) { return _angular(axis, posErr, hint, cur++); }

  template <class L>
  EHD void addLimit(Row* c, const L& limit) {
    uint16_t flags = uint16_t(c->flags | RF_OUTPUT_FORCE);
    if (isSoft(limit)) {
      flags |= RF_SPRING;
      c->mod0 = limit.stiffness;
      c->mod1 = limit.damping;
    } else {
      c->solveHint = SH_INEQUALITY;
      c->mod0 = limit.restitution;
      c->mod1 = limit.bounceThreshold;
      if (c->geometricError > 0.0f) flags |= RF_KEEPBIAS;
      if (limit.restitution > 0.0f) flags |= RF_RESTITUTION;
    }
    c->flags = flags;
    c->minImpulse = 0.0f;
  }
  EHD void addDrive(Row* c, float velTarget, const Drive& drive) {
    c->velocityTarget = velTarget;
    uint16_t flags = uint16_t(c->flags | RF_SPRING | RF_HAS_DRIVE_LIMIT);
    if (drive.flags & DRIVE_FLAG_ACCELERATION) flags |= RF_ACCELERATION_SPRING;
    if (drive.flags & DRIVE_FLAG_OUTPUT_FORCE) flags |= RF_OUTPUT_FORCE;
    c->flags = flags;
    c->mod0 = drive.stiffness;
    c->mod1 = drive.damping;
    c->minImpulse = -drive.forceLimit;
    c->maxImpulse = drive.forceLimit;
  }
  EHD void angularHard(const V3& axis, float posErr) {
    Row* c = angular(axis, posErr, SH_EQUALITY);
    c->flags |= RF_OUTPUT_FORCE;
  }
  template <class L>
  EHD void linearLimit(const V3& axis, float ordinate, float limitValue, const L& limit) {
    if (!isSoft(limit) || ordinate > limitValue) addLimit(linear(axis, limitValue - ordinate, SH_NONE), limit);
  }
  template <class L>
  EHD void angularLimit(const V3& axis, float error, const L& limit) { addLimit(angular(axis, error, SH_NONE), limit); }
  template <class L>
  EHD void anglePair(float angle, float lower, float upper, const V3& axis, const L& limit) {
    const bool softLimit = isSoft(limit);
    if (!softLimit || angle < lower) angularLimit(vneg(axis), -(lower - angle), limit);
    if (!softLimit || angle > upper) angularLimit(axis, (upper - angle), limit);
  }
  EHD void linearDrive(const V3& axis, float velTarget, float error, const Drive& drive) {
    addDrive(linear(axis, error, SH_NONE), velTarget, drive);
  }
  EHD void angularDrive(const V3& axis, float velTarget, float error, const Drive& drive, uint16_t hint = SH_NONE) {
    addDrive(angular(axis, error, hint), velTarget, drive);
  }
  EHD uint32_t count() const { return uint32_t(cur - rows); }

  // prepareLockedAxes (ExtConstraintHelper.h:230)
  EHD void prepareLockedAxes(const Q& qA, const Q& qB, const V3& cB2cAp, uint32_t lin, uint32_t ang, V3& raOut, V3& rbOut) {
    Row* current = cur;
    V3 errorVector{0.0f, 0.0f, 0.0f};
    V3 ra_ = ra;
    V3 rb_ = rb;
    if (lin) {
      const M33 axes = mat_from_quat_simd(qA);
      if (lin & 1) errorVector = errorVector - axes.c0 * cB2cAp.x;
      if (lin & 2) errorVector = errorVector - axes.c1 * cB2cAp.y;
      if (lin & 4) errorVector = errorVector - axes.c2 * cB2cAp.z;
      ra_ = ra_ + errorVector;
      if (lin & 1) _linear(axes.c0, ra_, rb_, -cB2cAp.x, SH_EQUALITY, current++);
      if (lin & 2) _linear(axes.c1, ra_, rb_, -cB2cAp.y, SH_EQUALITY, current++);
      if (lin & 4) _linear(axes.c2, ra_, rb_, -cB2cAp.z, SH_EQUALITY, current++);
    }
    if (ang) {
      const Q qB2qA = conj(qA) * qB;
      V3 row[3];
      computeJacobianAxes(row, qA, qB);
      if (ang & 1) _angular(row[0], -qB2qA.x, SH_EQUALITY, current++);
      if (ang & 2) _angular(row[1], -qB2qA.y, SH_EQUALITY, current++);
      if (ang & 4) _angular(row[2], -qB2qA.z, SH_EQUALITY, current++);
    }
    raOut = ra_;
    rbOut = rb_;
    for (Row* front = cur; front < current; front++) front->flags |= RF_OUTPUT_FORCE;
    cur = current;
  }
};

// ---- 원뿔 한계 (CmConeLimitHelper.h, PxMathUtils.h:204)
EHD V3 ellipseClamp(const V3& point, const V3& radii) {
  const uint32_t MAX_ITERATIONS = 20;
  const float convergenceThreshold = 1e-4f;
  const V3 q{0, fabsf(point.y), fabsf(point.z)};
  const float tinyEps = 1e-6f;
  if (radii.y >= radii.z) {
    if (q.z < tinyEps) return V3{0, point.y > 0 ? radii.y : -radii.y, 0};
  } else {
    if (q.y < tinyEps) return V3{0, 0, point.z > 0 ? radii.z : -radii.z};
  }
  V3 denom;
  const V3 e2 = mulc(radii, radii), eq = mulc(radii, q);
  float t = pmax(eq.y - e2.y, eq.z - e2.z);
  for (uint32_t i = 0; i < MAX_ITERATIONS; i++) {
    denom = V3{0, 1 / (t + e2.y), 1 / (t + e2.z)};
    const V3 denom2 = mulc(eq, denom);
    const V3 fv = mulc(denom2, denom2);
    const float f = fv.y + fv.z - 1;
    if (f < convergenceThreshold) return mulc(mulc(e2, point), denom);
    const float df = dot(fv, denom) * -2.0f;
    t = t - f / df;
  }
  const V3 r = mulc(mulc(e2, point), denom);
  const float ry = r.y / radii.y, rz = r.z / radii.z;
  return r * lm::PxRecipSqrt(ry * ry + rz * rz);
}
EHD float computeAxisAndError(const V3& r, const V3& d, const V3& twistAxis, V3& axis) {
  const V3 p{1.0f, 0.0f, 0.0f};
  const float r2 = dot(r, r), a = 1.0f - r2, b = 1.0f / (1.0f + r2), b2 = b * b;
  const float v1 = 2.0f * a * b2;
  const V3 v2{a, 2.0f * r.z, -2.0f * r.y};
  const V3 coneLine = V3{v1 * v2.x, v1 * v2.y, v1 * v2.z} - p;
  const float rd = dot(r, d);
  const float dv1 = -4.0f * rd * (3.0f - r2) * b2 * b;
  const V3 dv2{-2.0f * rd, 2.0f * d.z, -2.0f * d.y};
  const V3 coneNormal = V3{v1 * dv2.x, v1 * dv2.y, v1 * dv2.z} + V3{dv1 * v2.x, dv1 * v2.y, dv1 * v2.z};
  axis = vdiv(cross(coneLine, coneNormal), mag(coneNormal));
  return dot(cross(coneLine, axis), twistAxis);
}
EHD void coneGetLimit(float yMax, float zMax, const Q& swing, V3& axis, float& error) {
  const V3 twistAxis = basis0(swing);
  const V3 swingAngle{0.0f, 4.0f * lm::PxAtan2(swing.y, 1.0f + swing.w), 4.0f * lm::PxAtan2(swing.z, 1.0f + swing.w)};
  const V3 p = ellipseClamp(swingAngle, V3{0.0f, yMax, zMax});
  const V3 normal{0.0f, p.y / (yMax * yMax), p.z / (zMax * zMax)};  // PxSqr(x) = x*x
  const V3 r{0.0f, lm::PxTan(p.y / 4.0f), lm::PxTan(p.z / 4.0f)}, d{0.0f, normal.y, normal.z};
  error = computeAxisAndError(r, d, twistAxis, axis);
}

// ---- D6JointSolverPrep (ExtD6Joint.cpp:1002). bA2w/bB2w = 두 몸체 질량중심 자세 (정적/세계 = 항등)
EHD PrepOut d6SolverPrep(Row* constraints, const D6Data& data, const Tf& bA2w, const Tf& bB2w, bool useExtendedLimits) {
  PrepOut out;
  // ConstraintHelper 생성자 (ExtConstraintHelper.h:144)
  out.invMassScale = data.invMassScale;
  Tf cA2w = transformMultiplyV(bA2w, fromTf32(data.c2b[0]));
  Tf cB2w = transformMultiplyV(bB2w, fromTf32(data.c2b[1]));
  Helper ch;
  ch.rows = constraints;
  ch.cur = constraints;
  ch.ra = cB2w.p - bA2w.p;   // V4Sub(cB2wV, bA2w.p) — 칸별 뺄셈이라 스칼라와 같음
  ch.rb = cB2w.p - bB2w.p;
  out.body0WorldOffset = ch.ra;

  const uint32_t SWING1_FLAG = 1u << AX_SWING1, SWING2_FLAG = 1u << AX_SWING2, TWIST_FLAG = 1u << AX_TWIST;
  const uint32_t ANGULAR_MASK = SWING1_FLAG | SWING2_FLAG | TWIST_FLAG;
  const uint32_t LINEAR_MASK = 1u << AX_X | 1u << AX_Y | 1u << AX_Z;
  const Drive* drives = data.drive;
  uint32_t locked = data.locked;
  const uint32_t limited = data.limited;
  const uint32_t driving = data.driving;

  if (!useExtendedLimits) {  // applyNeighborhoodOperator
    if (qdot(cA2w.q, cB2w.q) < 0.0f) cB2w.q = qneg(cB2w.q);
  }
  const Tf cB2cA = transformInvTf(cA2w, cB2w);
  const M33 cA2w_m = mat_from_quat_simd(cA2w.q);
  const M33 cB2w_m = mat_from_quat_simd(cB2w.q);
  const V3 bX = cB2w_m.c0;
  const V3 aY = cA2w_m.c1;
  const V3 aZ = cA2w_m.c2;

  if (driving & ((1u << DR_X) | (1u << DR_Y) | (1u << DR_Z))) {
    const V3 posErr = data.drivePosition.p - cB2cA.p;
    for (int i = 0; i < 3; i++) {
      if (driving & (1u << (DR_X + i)))
        ch.linearDrive(col(cA2w_m, i), -comp(data.driveLinearVelocity, i), comp(posErr, i), drives[i]);
    }
  }

  if (driving & ((1u << DR_SLERP) | (1u << DR_SWING) | (1u << DR_TWIST) | (1u << DR_SWING1) | (1u << DR_SWING2))) {
    const Q d2cA_q = qdot(cB2cA.q, data.drivePosition.q) > 0.0f ? data.drivePosition.q : qneg(data.drivePosition.q);
    const V3& v = data.driveAngularVelocity;
    const Q delta = conj(d2cA_q) * cB2cA.q;
    if (driving & (1u << DR_SLERP)) {
      const V3 velTarget = vneg(rotate(cA2w.q, data.driveAngularVelocity));
      V3 axis[3] = {V3{1.0f, 0.0f, 0.0f}, V3{0.0f, 1.0f, 0.0f}, V3{0.0f, 0.0f, 1.0f}};
      if (drives[5].stiffness != 0.0f) computeJacobianAxes(axis, cA2w.q * d2cA_q, cB2w.q);
      const V3 im{delta.x, delta.y, delta.z};
      for (int i = 0; i < 3; i++)
        ch.angularDrive(axis[i], dot(axis[i], velTarget), -comp(im, i), drives[5], SH_SLERP_SPRING);
    } else {
      if (driving & (1u << DR_TWIST)) ch.angularDrive(cA2w_m.c0, v.x, -2.0f * delta.x, drives[4]);
      if (driving & (1u << DR_SWING)) {
        const V3 err = basis0(delta);
        if (!(locked & SWING1_FLAG)) ch.angularDrive(aY, v.y, err.z, drives[3]);
        if (!(locked & SWING2_FLAG)) ch.angularDrive(aZ, v.z, -err.y, drives[3]);
      } else if (driving & ((1u << DR_SWING1) | (1u << DR_SWING2))) {
        const V3 err = basis0(delta);
        if (driving & (1u << DR_SWING1)) ch.angularDrive(aY, v.y, err.z, drives[3]);
        if (driving & (1u << DR_SWING2)) ch.angularDrive(aZ, v.z, -err.y, drives[5]);
      }
    }
  }

  if (limited & ANGULAR_MASK) {
    Q swing, twist;
    separateSwingTwist(cB2cA.q, swing, twist);
    if ((limited & SWING1_FLAG) && (limited & SWING2_FLAG)) {
      if (data.mUseConeLimit) {  // setupConeSwingLimits
        V3 axis;
        float error;
        coneGetLimit(data.swingLimit.yAngle, data.swingLimit.zAngle, swing, axis, error);
        ch.angularLimit(rotate(cA2w.q, axis), error, data.swingLimit);
      }
      if (data.mUsePyramidLimits) {  // setupPyramidSwingLimits(.., true, true)
        const Q q = cA2w.q * swing;
        const LimitPyramid& l = data.pyramidSwingLimit;
        ch.anglePair(computeSwingAngle(swing.y, swing.w), l.yAngleMin, l.yAngleMax, basis1(q), l);
        ch.anglePair(computeSwingAngle(swing.z, swing.w), l.zAngleMin, l.zAngleMax, basis2(q), l);
      }
    } else {
      if (limited & SWING1_FLAG) {
        if (locked & SWING2_FLAG) {
          if (data.mUsePyramidLimits) {
            const Q q = cA2w.q * swing;
            const LimitPyramid& l = data.pyramidSwingLimit;
            ch.anglePair(computeSwingAngle(swing.y, swing.w), l.yAngleMin, l.yAngleMax, basis1(q), l);
          } else {
            ch.anglePair(computeSwingAngle(swing.y, swing.w), -data.swingLimit.yAngle, data.swingLimit.yAngle, aY, data.swingLimit);
          }
        } else {
          if (!data.mUsePyramidLimits)
            ch.anglePair(lm::PxAsin(-dot(aZ, bX)), -data.swingLimit.yAngle, data.swingLimit.yAngle, vgetNormalized(cross(aZ, bX)),
                         data.swingLimit);
          // else: PhysX 는 오류 메시지만 낸다 (이중 피라미드 미지원)
        }
      }
      if (limited & SWING2_FLAG) {
        if (locked & SWING1_FLAG) {
          if (data.mUsePyramidLimits) {
            const Q q = cA2w.q * swing;
            const LimitPyramid& l = data.pyramidSwingLimit;
            ch.anglePair(computeSwingAngle(swing.z, swing.w), l.zAngleMin, l.zAngleMax, basis2(q), l);
          } else {
            ch.anglePair(computeSwingAngle(swing.z, swing.w), -data.swingLimit.zAngle, data.swingLimit.zAngle, aZ, data.swingLimit);
          }
        } else {
          if (!data.mUsePyramidLimits)
            ch.anglePair(lm::PxAsin(dot(aY, bX)), -data.swingLimit.zAngle, data.swingLimit.zAngle, vgetNormalized(vneg(cross(aY, bX))),
                         data.swingLimit);
        }
      }
    }
    if (limited & TWIST_FLAG)
      ch.anglePair(computePhi(twist), data.twistLimit.lower, data.twistLimit.upper, cB2w_m.c0, data.twistLimit);
  }

  if (limited & LINEAR_MASK) {
    if (data.mUseDistanceLimit) {  // computeLimitedDistance (ExtD6Joint.cpp:790)
      V3 limitDir{0.0f, 0.0f, 0.0f};
      for (int i = 0; i < 3; i++)
        if (data.limited & (1u << (AX_X + i))) limitDir = limitDir + col(cA2w_m, i) * comp(cB2cA.p, i);
      const float distance = mag(limitDir);
      if (distance > data.distanceMinDist)
        ch.linearLimit(limitDir * (1.0f / distance), distance, data.distanceLimit.value, data.distanceLimit);
    }
    if (data.mUseNewLinearLimits) {
      const V3& bOriginInA = cB2cA.p;
      const LinearLimitPair* L[3] = {&data.linearLimitX, &data.linearLimitY, &data.linearLimitZ};
      for (int i = 0; i < 3; i++) {
        if ((limited & (1u << (AX_X + i))) && L[i]->lower <= L[i]->upper) {  // setupLinearLimit
          const V3 axis = col(cA2w_m, i);
          const float origin = comp(bOriginInA, i);
          ch.linearLimit(axis, origin, L[i]->upper, *L[i]);
          ch.linearLimit(vneg(axis), -origin, -L[i]->lower, *L[i]);
        }
      }
    }
  }

  const uint32_t angularLocked = locked & ANGULAR_MASK;
  if (angularLocked == SWING1_FLAG) {
    ch.angularHard(cross(bX, aZ), -dot(bX, aZ));
    locked &= ~SWING1_FLAG;
  } else if (angularLocked == SWING2_FLAG) {
    locked &= ~SWING2_FLAG;
    ch.angularHard(cross(bX, aY), -dot(bX, aY));
  }

  V3 ra, rb;
  ch.prepareLockedAxes(cA2w.q, cB2w.q, cB2cA.p, locked & 7, locked >> 3, ra, rb);
  out.cA2w = ra + bA2w.p;
  out.cB2w = rb + bB2w.p;
  out.numRows = ch.count();
  return out;
}

}  // namespace jnt
}  // namespace eng
