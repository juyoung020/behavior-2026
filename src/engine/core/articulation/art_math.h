// articulation 모듈 수학: PhysX 5.6.1 의 PxMat33(스칼라)·Dy::SpatialMatrix·Cm::SpatialVectorF·관절체 전파 함수를 연산 순서 그대로.
// 담당: articulation 작업자 (docs/엔진_자체구현.md 12절, 14절). PhysX 링크 없음, 호스트·GPU 공용(EHD).
// 원본 (태그 107.3-omni-and-physx-5.6.1, physx/ 기준)
//   include/foundation/PxMat33.h (스칼라 행렬), source/common/src/CmSpatialVector.h,
//   source/lowleveldynamics/include/DyFeatherstoneArticulationUtils.h (SpatialMatrix, TestImpulseResponse),
//   source/lowleveldynamics/shared/DyCpuGpuArticulation.h (드라이브·마찰·한계·전파·흉내 관절 식),
//   include/foundation/PxVecMathSSE.h, unix/sse2/PxUnixSse2InlineAoS.h (aos 판: V3Dot 합산 순서, V3Neg = 0-x)
// 규칙: 식을 정리하지 말 것. 원본 줄을 주석으로 단다. 0.f 로 시작해 += 하는 곳도 그대로 (−0 → +0 이 달라진다).
#pragma once
#include <cstdint>

#include "../common/pmath.h"

namespace eng {
namespace art {

// ---------------------------------------------------------------- PxMat33 (스칼라, PxMat33.h)
EHD M33 m33(const V3& c0, const V3& c1, const V3& c2) { return M33{c0, c1, c2}; }
EHD M33 m33zero() { return M33{V3{0, 0, 0}, V3{0, 0, 0}, V3{0, 0, 0}}; }
EHD M33 m33identity() { return M33{V3{1, 0, 0}, V3{0, 1, 0}, V3{0, 0, 1}}; }
EHD M33 m33diag(const V3& d) { return M33{V3{d.x, 0, 0}, V3{0, d.y, 0}, V3{0, 0, d.z}}; }  // createDiagonal
// explicit PxMat33(const PxQuat&) (PxMat33.h:136) — SIMD 판(mat_from_quat_simd)과 다르다
EHD M33 m33FromQuat(const Q& q) {
  const float x = q.x, y = q.y, z = q.z, w = q.w;
  const float x2 = x + x, y2 = y + y, z2 = z + z;
  const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
  const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
  const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
  return M33{V3{1.0f - yy - zz, xy + zw, xz - yw}, V3{xy - zw, 1.0f - xx - zz, yz + xw}, V3{xz + yw, yz - xw, 1.0f - xx - yy}};
}
EHD V3 mulT(const M33& m, const V3& v) { return V3{dot(m.c0, v), dot(m.c1, v), dot(m.c2, v)}; }  // transformTranspose
EHD M33 operator*(const M33& a, const M33& b) { return M33{a * b.c0, a * b.c1, a * b.c2}; }
EHD M33 operator+(const M33& a, const M33& b) { return M33{a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2}; }
EHD M33 operator-(const M33& a, const M33& b) { return M33{a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2}; }
EHD M33 operator*(const M33& a, float s) { return M33{a.c0 * s, a.c1 * s, a.c2 * s}; }
EHD M33& operator+=(M33& a, const M33& b) { a.c0 += b.c0; a.c1 += b.c1; a.c2 += b.c2; return a; }
EHD M33 transpose(const M33& a) { return M33{V3{a.c0.x, a.c1.x, a.c2.x}, V3{a.c0.y, a.c1.y, a.c2.y}, V3{a.c0.z, a.c1.z, a.c2.z}}; }
EHD M33 neg0(const M33& a) {  // aos M33Neg: V3Neg = _mm_sub_ps(0, f) (PxVecMathSSE.h:829) — 부호 반전이 아니라 0 에서 뺌
  return M33{V3{0.0f - a.c0.x, 0.0f - a.c0.y, 0.0f - a.c0.z}, V3{0.0f - a.c1.x, 0.0f - a.c1.y, 0.0f - a.c1.z},
             V3{0.0f - a.c2.x, 0.0f - a.c2.y, 0.0f - a.c2.z}};
}
EHD float det(const M33& m) { return dot(m.c0, cross(m.c1, m.c2)); }  // getDeterminant (PxMat33.h:245)
EHD M33 inverse(const M33& m) {                                        // getInverse (PxMat33.h:215)
  const float d = det(m);
  if (d != 0.0f) {
    const float invDet = 1.0f / d;
    M33 r;
    r.c0.x = invDet * (m.c1.y * m.c2.z - m.c2.y * m.c1.z);
    r.c0.y = invDet * -(m.c0.y * m.c2.z - m.c2.y * m.c0.z);
    r.c0.z = invDet * (m.c0.y * m.c1.z - m.c0.z * m.c1.y);
    r.c1.x = invDet * -(m.c1.x * m.c2.z - m.c1.z * m.c2.x);
    r.c1.y = invDet * (m.c0.x * m.c2.z - m.c0.z * m.c2.x);
    r.c1.z = invDet * -(m.c0.x * m.c1.z - m.c0.z * m.c1.x);
    r.c2.x = invDet * (m.c1.x * m.c2.y - m.c1.y * m.c2.x);
    r.c2.y = invDet * -(m.c0.x * m.c2.y - m.c0.y * m.c2.x);
    r.c2.z = invDet * (m.c0.x * m.c1.y - m.c1.x * m.c0.y);
    return r;
  }
  return m33identity();
}
EHD float m33get(const M33& m, int col, int row) {
  const V3& c = col == 0 ? m.c0 : (col == 1 ? m.c1 : m.c2);
  return row == 0 ? c.x : (row == 1 ? c.y : c.z);
}
EHD void m33set(M33& m, int col, int row, float v) {
  V3& c = col == 0 ? m.c0 : (col == 1 ? m.c1 : m.c2);
  if (row == 0) c.x = v; else if (row == 1) c.y = v; else c.z = v;
}
// FeatherstoneArticulation::constructSkewSymmetricMatrix (DyFeatherstoneArticulation.h:833)
EHD M33 skew(const V3& r) { return M33{V3{0.0f, r.z, -r.y}, V3{-r.z, 0.0f, r.x}, V3{r.y, -r.x, 0.0f}}; }

// SSE V3Dot (PxVecMathSSE.h:965, SSE4.2 없음 — linux-carbonite 빌드는 -msse4 안 씀): (x + z) + (y + w*w'), w 칸 = ±0
EHD float dotSSE(const V3& a, const V3& b) {
  const float px = a.x * b.x, py = a.y * b.y, pz = a.z * b.z;
  return (px + pz) + (py + 0.0f * 0.0f);
}
// SpatialMatrix::invertSym33 aos 판 (DyFeatherstoneArticulationUtils.h:510)
EHD M33 invertSym33V(const M33& in) {
  const V3 v0 = cross(in.c1, in.c2);
  const V3 v1 = cross(in.c2, in.c0);
  const V3 v2 = cross(in.c0, in.c1);
  const float d = dotSSE(v0, in.c0);
  const float recipDet = 1.0f / d;  // FRecip = _mm_div_ps(1, a)
  if (!(d == 0.0f))
    return M33{v0 * recipDet, V3{v0.y, v1.y, v1.z} * recipDet, V3{v0.z, v1.z, v2.z} * recipDet};
  return m33identity();
}

// ---------------------------------------------------------------- 공간 벡터 (CmSpatialVector.h)
// SpatialVectorF / UnAlignedSpatialVector: 운동 벡터는 top = 각속도, bottom = 선속도. 충격/힘은 top = 선형, bottom = 회전.
struct SV {
  V3 top, bottom;
};
EHD SV sv(const V3& t, const V3& b) { return SV{t, b}; }
EHD SV svzero() { return SV{V3{0, 0, 0}, V3{0, 0, 0}}; }
EHD SV operator+(const SV& a, const SV& b) { return SV{a.top + b.top, a.bottom + b.bottom}; }
EHD SV operator-(const SV& a, const SV& b) { return SV{a.top - b.top, a.bottom - b.bottom}; }
EHD SV operator-(const SV& a) { return SV{-a.top, -a.bottom}; }
EHD SV operator*(const SV& a, float s) { return SV{a.top * s, a.bottom * s}; }
EHD SV& operator+=(SV& a, const SV& b) { a.top += b.top; a.bottom += b.bottom; return a; }
EHD SV& operator-=(SV& a, const SV& b) {
  a.top = a.top - b.top;
  a.bottom = a.bottom - b.bottom;
  return a;
}
EHD float innerProduct(const SV& a, const SV& b) { return dot(a.bottom, b.top) + dot(a.top, b.bottom); }
EHD float svdot(const SV& a, const SV& b) { return dot(a.top, b.top) + dot(a.bottom, b.bottom); }
EHD float svmag(const SV& a) { return mag(a.top) + mag(a.bottom); }
EHD SV svrotate(const Tf& t, const SV& v) { return SV{rotate(t.q, v.top), rotate(t.q, v.bottom)}; }
// translateSpatialVector(offset, s) = (s.top, s.bottom + offset x s.top) (DyCpuGpuArticulation.h:466)
EHD SV translateSV(const V3& offset, const SV& s) { return SV{s.top, s.bottom + cross(offset, s.top)}; }

// ---------------------------------------------------------------- SpatialMatrix (DyFeatherstoneArticulationUtils.h:220)
struct SMat {
  M33 tl, tr, bl;  // topLeft, topRight(질량), bottomLeft(관성)
};
EHD SV operator*(const SMat& m, const SV& s) {  // :268
  const V3 top = m.tl * s.top + m.tr * s.bottom;
  const V3 bottom = m.bl * s.top + mulT(m.tl, s.bottom);
  return SV{top, bottom};
}
EHD SMat operator-(const SMat& a, const SMat& b) { return SMat{a.tl - b.tl, a.tr - b.tr, a.bl - b.bl}; }
EHD SMat& operator+=(SMat& a, const SMat& b) { a.tl += b.tl; a.tr += b.tr; a.bl += b.bl; return a; }
EHD SMat smatzero() { return SMat{m33zero(), m33zero(), m33zero()}; }
// constructSpatialMatrix(Is, stI) (:364)
EHD SMat constructSM(const SV& Is, const SV& stI) {
  const M33 tl{Is.top * stI.top.x, Is.top * stI.top.y, Is.top * stI.top.z};
  const M33 tr{Is.top * stI.bottom.x, Is.top * stI.bottom.y, Is.top * stI.bottom.z};
  const M33 bl{Is.bottom * stI.top.x, Is.bottom * stI.top.y, Is.bottom * stI.top.z};
  return SMat{tl, tr, bl};
}
// constructSpatialMatrix(columns[6]) (:389)
EHD SMat constructSM6(const SV* c) {
  return SMat{M33{c[0].top, c[1].top, c[2].top}, M33{c[3].top, c[4].top, c[5].top}, M33{c[0].bottom, c[1].bottom, c[2].bottom}};
}
// invertInertiaV (aos, :564). aos 행렬 연산은 칸마다 스칼라와 같은 순서(M33MulV3 = (c0*x + c1*y) + c2*z), M33Neg 만 0-x.
EHD SMat invertInertiaV(const SMat& in) {
  M33 aa = in.bl, ll = in.tr;
  const M33 la = in.tl;
  aa = (aa + transpose(aa)) * 0.5f;
  ll = (ll + transpose(ll)) * 0.5f;
  const M33 AAInv = invertSym33V(aa);
  const M33 z = neg0(la) * AAInv;
  const M33 S = ll + z * transpose(la);
  const M33 LL = invertSym33V(S);
  const M33 LA = LL * z;
  const M33 AA = AAInv + transpose(z) * LA;
  return SMat{transpose(LA), AA, LL};
}
// FeatherstoneArticulation::translateInertia aos 판 (DyFeatherstoneArticulation.cpp:829)
EHD void translateInertia(const M33& sTod, SMat& inertia) {
  const M33 dTos = transpose(sTod);
  const M33 tL = inertia.tl, tR = inertia.tr, bL = inertia.bl;
  const M33 bl = sTod * tL + bL;
  const M33 br = sTod * tR + transpose(tL);
  const M33 bottomLeft = bl + br * dTos;
  inertia.tl = tL + tR * dTos;
  inertia.bl = (bottomLeft + transpose(bottomLeft)) * 0.5f;
}

struct InvStIs {
  float m[3][3];
};

// TestImpulseResponse (:617): 6 가지 단위 충격에 대한 링크 속도 변화
struct TIR {
  SV r[6];
};
// getLinkDeltaVImpulseResponse(SpatialVectorF) (:622). aos 덧셈은 칸마다 왼쪽부터.
EHD SV responseOf(const TIR& t, const SV& imp) {
  const float ix = imp.top.x, iy = imp.top.y, iz = imp.top.z;
  const float ia = imp.bottom.x, ib = imp.bottom.y, ic = imp.bottom.z;
  const V3 top = t.r[0].top * ix + t.r[1].top * iy + t.r[2].top * iz + t.r[3].top * ia + t.r[4].top * ib + t.r[5].top * ic;
  const V3 bottom = t.r[0].bottom * ix + t.r[1].bottom * iy + t.r[2].bottom * iz + t.r[3].bottom * ia + t.r[4].bottom * ib +
                    t.r[5].bottom * ic;
  return SV{top, bottom};
}

// ---------------------------------------------------------------- 전파 (DyCpuGpuArticulation.h)
// propagateImpulseW (:494)
EHD SV propagateImpulseW(const V3& parentToChild, const SV& YChildW, const float* jointDofImpulse, const SV* jointDofISInvStISW,
                         const SV* jointDofMotionMatrixW, uint32_t dofCount, float* jointDofQMinusStY) {
  SV YParentW = svzero();
  for (uint32_t ind = 0; ind < dofCount; ++ind) {
    const SV& sa = jointDofMotionMatrixW[ind];
    const float Qv = jointDofImpulse ? jointDofImpulse[ind] : 0.0f;
    const float QMinusStY = Qv - innerProduct(sa, YChildW);
    YParentW += jointDofISInvStISW[ind] * QMinusStY;
    if (jointDofQMinusStY) jointDofQMinusStY[ind] += QMinusStY;
  }
  YParentW += YChildW;
  return translateSV(parentToChild, YParentW);
}
// propagateAccelerationW (:549)
EHD SV propagateAccelerationW(const V3& parentToChild, const SV& parentLinkAccelerationW, const InvStIs& invStISW,
                              const SV* motionMatrixW, const SV* IsW, const float* QMinusSTZ, uint32_t dofCount,
                              float* jointAcceleration) {
  SV motionAccelerationW = translateSV(-parentToChild, parentLinkAccelerationW);
  float tJAccel[3];
  for (uint32_t ind = 0; ind < dofCount; ++ind) {
    const float temp = innerProduct(IsW[ind], motionAccelerationW);
    tJAccel[ind] = (QMinusSTZ[ind] - temp);
  }
  for (uint32_t ind = 0; ind < dofCount; ++ind) {
    float jVel = 0.f;
    for (uint32_t ind2 = 0; ind2 < dofCount; ++ind2) jVel += invStISW.m[ind2][ind] * tJAccel[ind2];
    motionAccelerationW.top += motionMatrixW[ind].top * jVel;
    motionAccelerationW.bottom += motionMatrixW[ind].bottom * jVel;
    if (jointAcceleration) jointAcceleration[ind] += jVel;
  }
  return motionAccelerationW;
}

// ---------------------------------------------------------------- 드라이브·마찰·한계 (DyCpuGpuArticulation.h)
struct ImplicitDrive {  // ArticulationImplicitDriveDesc (:47)
  float targetVelPlusInitialBias, biasCoefficient, velMultiplier, impulseMultiplier, targetPosBias;
};
EHD ImplicitDrive implicitDriveZero() { return ImplicitDrive{0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; }
EHD ImplicitDrive computeImplicitDriveParamsForceDrive(float stiffness, float damping, float dt, float simDt, float unitResponse,
                                                       float geomError, float targetVelocity, bool isTGS) {  // :74
  ImplicitDrive d = implicitDriveZero();
  const float a = dt * (dt * stiffness + damping);
  const float b = dt * (damping * targetVelocity);
  const float x = unitResponse > 0.f ? 1.0f / (1.0f + a * unitResponse) : 0.f;
  const float initialBias = geomError - (simDt - dt) * targetVelocity;
  const float driveBiasCoefficient = stiffness * x * dt;
  d.targetVelPlusInitialBias = (x * b) + driveBiasCoefficient * initialBias;
  d.velMultiplier = -x * a;
  d.biasCoefficient = driveBiasCoefficient;
  d.impulseMultiplier = isTGS ? 1.f : 1.0f - x;
  d.targetPosBias = isTGS ? driveBiasCoefficient * targetVelocity : 0.f;
  return d;
}
EHD ImplicitDrive computeImplicitDriveParamsAccelerationDrive(float stiffness, float damping, float dt, float simDt,
                                                              float recipUnitResponse, float geomError, float targetVelocity,
                                                              bool isTGS) {  // :92
  ImplicitDrive d = implicitDriveZero();
  const float a = dt * (dt * stiffness + damping);
  const float b = dt * (damping * targetVelocity);
  const float x = 1.0f / (1.0f + a);
  const float initialBias = geomError - (simDt - dt) * targetVelocity;
  const float driveBiasCoefficient = stiffness * x * recipUnitResponse * dt;
  d.targetVelPlusInitialBias = (x * b * recipUnitResponse) + driveBiasCoefficient * initialBias;
  d.velMultiplier = -x * a * recipUnitResponse;
  d.biasCoefficient = driveBiasCoefficient;
  d.impulseMultiplier = isTGS ? 1.f : 1.0f - x;
  d.targetPosBias = isTGS ? driveBiasCoefficient * targetVelocity : 0.f;
  return d;
}
EHD ImplicitDrive computeImplicitDriveParams(uint8_t driveType, float stiffness, float damping, float dt, float simDt,
                                             float unitResponse, float recipUnitResponse, float geomError, float targetVelocity,
                                             bool isTGS) {  // :125
  if (driveType == 0) return computeImplicitDriveParamsForceDrive(stiffness, damping, dt, simDt, unitResponse, geomError, targetVelocity, isTGS);
  if (driveType == 1)
    return computeImplicitDriveParamsAccelerationDrive(stiffness, damping, dt, simDt, recipUnitResponse, geomError, targetVelocity, isTGS);
  return implicitDriveZero();
}
EHD float fabsP(float a) { return ::fabsf(a); }                                 // PxAbs = ::fabsf (PxUnixMathIntrinsics.h:57)
EHD float pclamp(float v, float lo, float hi) { return pmin(hi, pmax(lo, v)); }  // PxClamp = PxMin(hi, PxMax(lo, v)) (PxMath.h:142)
EHD float computeFrictionImpulse(float frictionImpulse, float staticFrictionImpulse, float dynamicFrictionImpulse,
                                 float viscousFrictionCoefficient, float jointVel) {  // :160
  if (fabsP(frictionImpulse) > staticFrictionImpulse) {
    frictionImpulse = pclamp(frictionImpulse, -dynamicFrictionImpulse - viscousFrictionCoefficient * fabsP(jointVel),
                             dynamicFrictionImpulse + viscousFrictionCoefficient * fabsP(jointVel));
  }
  return frictionImpulse;
}
EHD float computeDriveImpulse(float accumulatedDriveImpulse, float jointVel, float jointDeltaPos, float elapsedTime,
                              const ImplicitDrive& d) {  // :179
  const float unclampedForce = accumulatedDriveImpulse * d.impulseMultiplier + jointVel * d.velMultiplier +
                               d.targetVelPlusInitialBias - jointDeltaPos * d.biasCoefficient + elapsedTime * d.targetPosBias;
  return unclampedForce;
}
constexpr float kEpsF32 = 1.192092896e-07f;  // PX_EPS_F32 = FLT_EPSILON
constexpr float kMaxF32 = 3.402823466e+38f;  // PX_MAX_F32
struct Pair {
  float first, second;
};
EHD Pair boundsNeg(float velDeviation, float maxV, float denom1, float denom2, float impulse) {  // :193
  float bound1, bound2;
  if (fabsP(denom1) < kEpsF32) return Pair{-kMaxF32, kMaxF32};
  if (fabsP(denom2) < kEpsF32) {
    bound1 = (velDeviation - maxV) / denom1;
    if (fabsP(velDeviation + maxV) < kEpsF32) bound2 = kMaxF32;
    else bound2 = (velDeviation + maxV) > 0.0f ? kMaxF32 : -kMaxF32;
    return Pair{bound1, pmin(0.0f, bound2)};
  }
  bound1 = (velDeviation - maxV) / denom1;
  bound2 = (velDeviation + maxV) / denom2;
  if (denom2 >= 0.0f) return Pair{pmax(bound1, impulse), pmin(0.0f, bound2)};
  return Pair{pmax(pmax(bound1, bound2), impulse), 0.0f};
}
EHD Pair boundsPos(float velDeviation, float maxV, float denom1, float denom2, float impulse) {  // :232
  float bound1, bound2;
  if (fabsP(denom2) < kEpsF32) return Pair{-kMaxF32, kMaxF32};
  if (fabsP(denom1) < kEpsF32) {
    bound2 = (velDeviation + maxV) / denom2;
    if (fabsP(velDeviation - maxV) < kEpsF32) bound1 = -kMaxF32;
    else bound1 = (velDeviation - maxV) > 0.0f ? kMaxF32 : -kMaxF32;
    return Pair{pmax(0.0f, bound1), bound2};
  }
  bound1 = (velDeviation - maxV) / denom1;
  bound2 = (velDeviation + maxV) / denom2;
  if (denom1 >= 0.0f) return Pair{pmax(0.0f, bound1), pmin(bound2, impulse)};
  return Pair{0.0f, pmin(pmin(bound1, bound2), impulse)};
}
EHD float clampDriveImpulse(float jointVel0, float driveImpulse0, float driveImpulse, float response, float maxJointVel,
                            float maxImpulse, float speedImpulseGradient, float velocityDependentResistance) {  // :279
  float velDeviation = driveImpulse0 * response - jointVel0;
  Pair bounds1, bounds2;
  if (driveImpulse > 0.0f) {
    float denom1 = response - speedImpulseGradient;
    float denom2 = response + speedImpulseGradient;
    bounds1 = boundsPos(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
    if (velocityDependentResistance == 0.0f) bounds2 = Pair{-maxImpulse, maxImpulse};
    else {
      denom1 = response - 1.0f / velocityDependentResistance;
      denom2 = response + 1.0f / velocityDependentResistance;
      bounds2 = boundsPos(velDeviation, maxImpulse / velocityDependentResistance, denom1, denom2, driveImpulse);
    }
  } else {
    float denom1 = response + speedImpulseGradient;
    float denom2 = response - speedImpulseGradient;
    bounds1 = boundsNeg(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
    if (velocityDependentResistance == 0.0f) bounds2 = Pair{-maxImpulse, maxImpulse};
    else {
      denom1 = response + 1.0f / velocityDependentResistance;
      denom2 = response - 1.0f / velocityDependentResistance;
      bounds2 = boundsNeg(velDeviation, maxImpulse / velocityDependentResistance, denom1, denom2, driveImpulse);
    }
  }
  float lowerBound = pmax(bounds1.first, bounds2.first);
  float upperBound = pmin(bounds1.second, bounds2.second);
  if (lowerBound > upperBound) return 0.0f;
  return pclamp(driveImpulse, lowerBound, upperBound);
}
// computeLimitImpulse (:352)
EHD float computeLimitImpulse(float dt, float recipDt, bool isVelIter, float response, float recipResponse, float erp, float errorLow,
                              float errorHigh, float jointPDelta, float& lowImpulse_, float& highImpulse_, float& jointV_) {
  float jointV = jointV_;
  float lowImpulse = lowImpulse_;
  float highImpulse = highImpulse_;
  const float futureDeltaJointP = jointPDelta + jointV * dt;
  const float currErrLow = errorLow + jointPDelta;
  const float nextErrLow = errorLow + futureDeltaJointP;
  const float currErrHigh = errorHigh - jointPDelta;
  const float nextErrHigh = errorHigh - futureDeltaJointP;
  bool limited = false, newLow = false, newHigh = false;
  const float tolerance = 0.f;
  float deltaF = 0.f;
  if (currErrLow < tolerance || nextErrLow < tolerance) {
    float newJointV = jointV;
    limited = true;
    if (currErrLow < tolerance) {
      if (!isVelIter) newJointV = -currErrLow * recipDt * erp;
    } else {
      newJointV = -currErrLow * recipDt;
    }
    const float deltaV = newJointV - jointV;
    deltaF = pmax(lowImpulse + deltaV * recipResponse, 0.f) - lowImpulse;
    lowImpulse += deltaF;
    newLow = true;
  } else if (currErrHigh < tolerance || nextErrHigh < tolerance) {
    float newJointV = jointV;
    limited = true;
    if (currErrHigh < tolerance) {
      if (!isVelIter) newJointV = currErrHigh * recipDt * erp;
    } else
      newJointV = currErrHigh * recipDt;
    const float deltaV = newJointV - jointV;
    deltaF = pmin(highImpulse + deltaV * recipResponse, 0.f) - highImpulse;
    highImpulse += deltaF;
    newHigh = true;
  }
  if (!limited) {
    const float impulseForZeroVel = -jointV * recipResponse;
    if (jointV > 0.f) {
      deltaF = pmax(impulseForZeroVel, -lowImpulse);
      lowImpulse += deltaF;
      newLow = true;
    } else {
      deltaF = pmin(impulseForZeroVel, -highImpulse);
      highImpulse += deltaF;
      newHigh = true;
    }
  }
  jointV += deltaF * response;
  if (newLow) lowImpulse_ = lowImpulse;
  if (newHigh) highImpulse_ = highImpulse;
  jointV_ = jointV;
  return deltaF;
}
// 흉내 관절 (:603, :637)
EHD float computeRecipMimicJointEffectiveInertia(float rAA, float rAB, float rBB, float rBA, float gearRatio) {
  return (rAA + gearRatio * (rAB + rBA) + gearRatio * gearRatio * rBB);
}
EHD void computeMimicJointImpulses(float biasCoefficient, float dt, float invDt, float qA, float qB, float qADot, float qBDot,
                                   float gearRatio, float offset, float naturalFrequency, float dampingRatio, float r,
                                   bool isVelocityIteration, float& jointImpA, float& jointImpB) {
  float erp = 0.0f;
  float cfm = 0.0f;
  if (naturalFrequency <= 0 || dampingRatio <= 0 || isVelocityIteration) {
    erp = isVelocityIteration ? 0.0f : biasCoefficient;
    cfm = 0.0f;
  } else {
    const float mu = naturalFrequency;
    const float zeta = dampingRatio;
    const float kp = mu * mu / r;
    const float kd = 2.0f * mu * zeta / r;
    cfm = 1.0f / (dt * (dt * kp + kd));
    erp = (dt * kp) / (dt * kp + kd);
  }
  const float C = qA + gearRatio * qB + offset;
  const float b = erp * C * invDt;
  const float Jv = qADot + gearRatio * qBDot;
  const float effectiveInertia = 1.0f / (r + cfm);
  const float lambda = -(b + Jv) * effectiveInertia;
  jointImpA = lambda;
  jointImpB = gearRatio * lambda;
}

// ---------------------------------------------------------------- 쿼터니언 보조 (PxQuat.h, PxMathUtils.h)
EHD float qdot(const Q& a, const Q& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
EHD Q qneg(const Q& a) { return Q{-a.x, -a.y, -a.z, -a.w}; }
// PxQuat(angleRadians, unitAxis) (PxQuat.h:86): PxSinCos -> ::sinf/::cosf (glibc 이식본을 쓰는 쪽에서 넣어 준다)
template <class SinCos>
EHD Q quatAngleAxis(float angle, const V3& unitAxis, SinCos sc) {
  const float a = angle * 0.5f;
  float s, c;
  sc(a, s, c);
  return Q{unitAxis.x * s, unitAxis.y * s, unitAxis.z * s, c};
}
EHD V3 v3normalizedFast(const V3& v) {  // PxVec3::getNormalized: m>0 ? v * PxRecipSqrt(m) : 0 (PxRecipSqrt = 1/sqrtf)
  const float m = magSq(v);
  return m > 0.0f ? v * (1.0f / psqrt(m)) : V3{0, 0, 0};
}
EHD float v3normalize(V3& v) {  // PxVec3::normalize: m = magnitude; if(m>0) *this /= m (= * (1/m))
  const float m = mag(v);
  if (m > 0.0f) v *= (1.0f / m);
  return m;
}

}  // namespace art
}  // namespace eng
