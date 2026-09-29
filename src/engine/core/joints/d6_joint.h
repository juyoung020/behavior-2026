// joints 모듈: D6 조인트 객체와 API (생성·setter·prepareData) — PhysX 5.6.1 Ext::D6Joint 와 같은 입력 -> 같은 상수 블록.
// omni.physx 는 USD 의 모든 조인트(Fixed/Revolute/Prismatic/Spherical/D6)를 PxD6JointCreate 로 만든다(docs 12절 경계).
// 원본(physx/source/physxextensions/src 기준)
//   생성      : ExtD6Joint.cpp:39 (D6Joint 생성자), ExtJoint.h:630 (JointT 생성자: 틀 정규화·c2b·질량 척도)
//   setter    : ExtD6Joint.cpp:117(setMotion) 309(setDrive) 334(setDistanceLimit) 356(setLinearLimit) 419(setTwistLimit)
//               444(setPyramidSwingLimit) 482(setSwingLimit) 508(setDrivePosition) 527(setDriveVelocity) 579(setAngularDriveConfig)
//               ExtJoint.h:239(setLocalPose) 383-432(질량 척도) 331(setBreakForce) 485(onComShift)
//   준비      : ExtD6Joint.cpp:655 (prepareData: locked/limited/driving 비트 계산)
//   깨우기    : ExtJoint.h:674 (wakeUpActors: 깸 카운터 < 재설정값이면 wakeUp)
//   제약 기본 : physx/src/NpConstraint.cpp:159 (scSetFlags(shaders.flag) -> 기본 플래그 = eGPU_COMPATIBLE 뿐),
//               simulationcontroller/src/ScConstraintCore.cpp:35 (break force = MAX, minResponseThreshold = 0)
// 여기서 "몸체" 는 번호로만 안다. 조인트 틀 계산에 필요한 것은 각 행위자의 질량중심 자세(getCom)뿐이다:
//   동적 강체·관절체 링크 = body2Actor(getCMassLocalPose), 정적 = 전역 자세의 역, 없음(세계) = 항등 (ExtJoint.h:600).
#pragma once
#include <cstdint>

#include "joint_types.h"

namespace eng {
namespace jnt {

enum ActorKind : uint8_t { ACTOR_NONE = 0, ACTOR_DYNAMIC = 1, ACTOR_STATIC = 2, ACTOR_LINK = 3 };

struct D6Joint {
  D6Data data;
  Tf localPose[2];
  uint32_t actor[2];        // 몸체 번호 (ACTOR_NONE 이면 무시)
  uint8_t kind[2];          // ActorKind
  bool recomputeMotion;
  uint16_t constraintFlags; // PxConstraintFlags
  float linBreakForce, angBreakForce, minResponseThreshold;
};

// PxTransform::transformInv(const PxTransform&) (PxTransform.h:162): qinv.rotate(src.p - p), qinv * src.q
EHD Tf transformInvTf(const Tf& t, const Tf& src) {
  const Q qinv = conj(t.q);
  return Tf{qinv * src.q, rotate(qinv, src.p - t.p)};
}
// PxTransform::getInverse (PxTransform.h:114): (q.rotateInv(-p), q.getConjugate())
EHD Tf tfGetInverse(const Tf& t) { return Tf{conj(t.q), rotateInv(t.q, -t.p)}; }
EHD Tf32 toTf32(const Tf& t) { return Tf32{t.q, t.p, 0.0f}; }
EHD Tf fromTf32(const Tf32& t) { return Tf{t.q, t.p}; }

// getCom(actor) 입력: 동적/링크 = body2Actor, 정적 = 전역 자세, 없음 = 무시
EHD Tf comOf(uint8_t kind, const Tf& body2ActorOrStaticPose) {
  if (kind == ACTOR_NONE) return Tf{qid(), V3{0, 0, 0}};
  if (kind == ACTOR_STATIC) return tfGetInverse(body2ActorOrStaticPose);
  return body2ActorOrStaticPose;
}

// ---- 한계 생성자 (PxJointLimit.h)
EHD AngularLimitPair angularLimitPair(float lower, float upper) { return AngularLimitPair{0.0f, 0.5f, 0.0f, 0.0f, upper, lower}; }
EHD AngularLimitPair angularLimitPairSoft(float lower, float upper, float stiffness, float damping) {
  return AngularLimitPair{0.0f, 0.0f, stiffness, damping, upper, lower};
}
EHD LimitCone limitCone(float y, float z) { return LimitCone{0.0f, 0.5f, 0.0f, 0.0f, y, z}; }
EHD LimitPyramid limitPyramid(float y0, float y1, float z0, float z1) { return LimitPyramid{0.0f, 0.5f, 0.0f, 0.0f, y0, y1, z0, z1}; }
EHD LinearLimit linearLimit(float extent) { return LinearLimit{0.0f, 0.0f, 0.0f, 0.0f, extent}; }
EHD LinearLimitPair linearLimitPair(float lengthScale, float lower, float upper) {
  return LinearLimitPair{0.0f, 2.0f * lengthScale, 0.0f, 0.0f, upper, lower};
}
EHD LinearLimitPair linearLimitPairSoft(float lower, float upper, float stiffness, float damping) {
  return LinearLimitPair{0.0f, 0.0f, stiffness, damping, upper, lower};
}
EHD Drive defaultDrive() { return Drive{0.0f, 0.0f, MAX_F32, 0u}; }

// ---- 생성 (PxD6JointCreate). com0/com1 = comOf(...) 결과. lengthScale = PxTolerancesScale::length
EHD D6Joint createD6(uint8_t kind0, uint32_t actor0, const Tf& com0, const Tf& localFrame0, uint8_t kind1, uint32_t actor1,
                     const Tf& com1, const Tf& localFrame1, float lengthScale) {
  D6Joint j{};
  j.kind[0] = kind0; j.kind[1] = kind1;
  j.actor[0] = actor0; j.actor[1] = actor1;
  // JointT 생성자 (ExtJoint.h:640)
  j.localPose[0] = normalized(localFrame0);
  j.localPose[1] = normalized(localFrame1);
  D6Data& d = j.data;
  d.c2b[0] = toTf32(transformInvTf(com0, j.localPose[0]));
  d.c2b[1] = toTf32(transformInvTf(com1, j.localPose[1]));
  d.invMassScale = InvMassScale{1.0f, 1.0f, 1.0f, 1.0f};
  // D6Joint 생성자 (ExtD6Joint.cpp:39)
  for (int i = 0; i < 6; ++i) d.motion[i] = M_LOCKED;
  const float halfPi = PI / 2;
  d.twistLimit = angularLimitPair(-PI / 2, halfPi);
  d.swingLimit = limitCone(halfPi, halfPi);
  d.pyramidSwingLimit = limitPyramid(-PI / 2, halfPi, -PI / 2, halfPi);
  d.distanceLimit = linearLimit(MAX_F32);
  d.distanceMinDist = 1e-6f * lengthScale;
  d.linearLimitX = linearLimitPair(lengthScale, -MAX_F32 / 3.0f, MAX_F32 / 3.0f);
  d.linearLimitY = d.linearLimitX;
  d.linearLimitZ = d.linearLimitX;
  for (int i = 0; i < 6; ++i) d.drive[i] = defaultDrive();
  d.drivePosition = Tf{qid(), V3{0, 0, 0}};
  d.driveLinearVelocity = V3{0, 0, 0};
  d.driveAngularVelocity = V3{0, 0, 0};
  d.mUseDistanceLimit = d.mUseNewLinearLimits = d.mUseConeLimit = d.mUsePyramidLimits = false;
  d.angularDriveConfig = ADC_LEGACY;
  d.locked = d.limited = d.driving = 0;  // PhysX 는 prepareData 전까지 미정 (PX_ALLOC) — 첫 prepareData 에서 채워진다
  j.recomputeMotion = true;
  j.constraintFlags = CF_GPU_COMPATIBLE;  // NpConstraint.cpp:159 scSetFlags(D6 셰이더 표 플래그)
  j.linBreakForce = MAX_F32;
  j.angBreakForce = MAX_F32;
  j.minResponseThreshold = 0.0f;
  return j;
}

// ---- setter (모두 markDirty: 다음 simulate 에서 상수 블록을 다시 복사 — 여기서는 data 를 바로 쓰므로 할 일 없음)
EHD void setMotion(D6Joint& j, uint32_t axis, uint32_t m) { j.data.motion[axis] = m; j.recomputeMotion = true; }

// gDriveTypeToIndexMap (ExtD6Joint.cpp:227): SWING1 -> SWING 칸(3), SWING2 -> SLERP 칸(5)
EHD uint32_t driveDataIndex(uint32_t type) {
  const uint32_t map[8] = {0, 1, 2, 3, 4, 5, 3, 5};
  return map[type];
}
// isDriveTypeAllowed (ExtD6Joint.cpp:150, checked 빌드에서만 거름 — 공식 DLL 은 checked)
EHD bool driveTypeAllowed(uint32_t type, uint8_t cfg) {
  if (cfg == ADC_SWING_TWIST) return type <= DR_Z || type == DR_SWING1 || type == DR_SWING2 || type == DR_TWIST;
  if (cfg == ADC_SLERP) return type <= DR_Z || type == DR_SLERP;
  return type <= DR_Z || type == DR_SWING || type == DR_TWIST || type == DR_SLERP;
}
EHD bool setDrive(D6Joint& j, uint32_t type, const Drive& dr) {
  if (!driveTypeAllowed(type, j.data.angularDriveConfig)) return false;
  j.data.drive[driveDataIndex(type)] = dr;
  j.recomputeMotion = true;
  return true;
}
EHD void setDistanceLimit(D6Joint& j, const LinearLimit& l) { j.data.distanceLimit = l; j.data.mUseDistanceLimit = true; }
EHD void setLinearLimit(D6Joint& j, uint32_t axis, const LinearLimitPair& l) {
  if (axis == AX_X) j.data.linearLimitX = l;
  else if (axis == AX_Y) j.data.linearLimitY = l;
  else if (axis == AX_Z) j.data.linearLimitZ = l;
  else return;
  j.data.mUseNewLinearLimits = true;
}
EHD void setTwistLimit(D6Joint& j, const AngularLimitPair& l) { j.data.twistLimit = l; }
EHD void setPyramidSwingLimit(D6Joint& j, const LimitPyramid& l) { j.data.pyramidSwingLimit = l; j.data.mUsePyramidLimits = true; }
EHD void setSwingLimit(D6Joint& j, const LimitCone& l) { j.data.swingLimit = l; j.data.mUseConeLimit = true; }
// setDrivePosition / setDriveVelocity: autowake 면 호출자가 wakeUpActors 규칙(아래)을 적용해야 한다.
EHD void setDrivePosition(D6Joint& j, const Tf& pose) { j.data.drivePosition = normalized(pose); }
EHD void setDriveVelocity(D6Joint& j, const V3& lin, const V3& ang) { j.data.driveLinearVelocity = lin; j.data.driveAngularVelocity = ang; }
EHD void setAngularDriveConfig(D6Joint& j, uint8_t cfg) {
  D6Data& d = j.data;
  if (cfg == d.angularDriveConfig) return;
  if (cfg == ADC_SWING_TWIST) {
    d.drive[3] = defaultDrive(); d.drive[5] = defaultDrive(); d.drive[4] = defaultDrive();
  } else if (cfg == ADC_SLERP) {
    d.drive[5] = defaultDrive();
  } else {
    d.drive[3] = defaultDrive(); d.drive[4] = defaultDrive(); d.drive[5] = defaultDrive();
  }
  d.angularDriveConfig = cfg;
  j.recomputeMotion = true;
}
// PxJoint::setLocalPose: 정규화 후 c2b 다시 계산 (com = comOf(해당 행위자))
EHD void setLocalPose(D6Joint& j, int a, const Tf& pose, const Tf& com) {
  const Tf p = normalized(pose);
  j.localPose[a] = p;
  j.data.c2b[a] = toTf32(transformInvTf(com, p));
}
// PxConstraintConnector::onComShift: 행위자의 질량중심 자세가 바뀌면 (setCMassLocalPose) 부른다
EHD void onComShift(D6Joint& j, int a, const Tf& com) { j.data.c2b[a] = toTf32(transformInvTf(com, j.localPose[a])); }
EHD void setInvMassScale0(D6Joint& j, float s) { j.data.invMassScale.linear0 = s; }
EHD void setInvInertiaScale0(D6Joint& j, float s) { j.data.invMassScale.angular0 = s; }
EHD void setInvMassScale1(D6Joint& j, float s) { j.data.invMassScale.linear1 = s; }
EHD void setInvInertiaScale1(D6Joint& j, float s) { j.data.invMassScale.angular1 = s; }
EHD void setBreakForce(D6Joint& j, float f, float t) { j.linBreakForce = f; j.angBreakForce = t; }
// Sc::ConstraintCore::setFlags: eGPU_COMPATIBLE 은 처음 값 유지 (ScConstraintCore.cpp:52)
EHD void setConstraintFlags(D6Joint& j, uint16_t f) { j.constraintFlags = uint16_t(f | (j.constraintFlags & CF_GPU_COMPATIBLE)); }

// ---- prepareData (ExtD6Joint.cpp:655). simulate 때 상수 블록을 넘기기 직전에 부른다.
EHD bool isDriveActive(const D6Data& d, uint32_t idx) { return d.drive[idx].stiffness != 0 || d.drive[idx].damping != 0; }
EHD const D6Data& prepareData(D6Joint& j) {
  D6Data& d = j.data;
  if (j.recomputeMotion) {
    j.recomputeMotion = false;
    d.driving = 0; d.limited = 0; d.locked = 0;
    for (uint32_t i = 0; i < 6; i++) {
      if (d.motion[i] == M_LIMITED) d.limited |= 1u << i;
      else if (d.motion[i] == M_LOCKED) d.locked |= 1u << i;
    }
    if (isDriveActive(d, 0) && d.motion[AX_X] != M_LOCKED) d.driving |= 1u << DR_X;
    if (isDriveActive(d, 1) && d.motion[AX_Y] != M_LOCKED) d.driving |= 1u << DR_Y;
    if (isDriveActive(d, 2) && d.motion[AX_Z] != M_LOCKED) d.driving |= 1u << DR_Z;
    const bool swing1Locked = d.motion[AX_SWING1] == M_LOCKED;
    const bool swing2Locked = d.motion[AX_SWING2] == M_LOCKED;
    const bool twistLocked = d.motion[AX_TWIST] == M_LOCKED;
    if (d.angularDriveConfig == ADC_SWING_TWIST) {
      if (isDriveActive(d, 4) && !twistLocked) d.driving |= 1u << DR_TWIST;
      if (isDriveActive(d, 3) && !swing1Locked) d.driving |= 1u << DR_SWING1;
      if (isDriveActive(d, 5) && !swing2Locked) d.driving |= 1u << DR_SWING2;
    } else if (d.angularDriveConfig == ADC_SLERP) {
      if (isDriveActive(d, 5) && !swing1Locked && !swing2Locked && !twistLocked) d.driving |= 1u << DR_SLERP;
    } else {
      if (isDriveActive(d, 5) && !swing1Locked && !swing2Locked && !twistLocked) {
        d.driving |= 1u << DR_SLERP;
      } else {
        if (isDriveActive(d, 4) && !twistLocked) d.driving |= 1u << DR_TWIST;
        if (isDriveActive(d, 3) && (!swing1Locked || !swing2Locked)) d.driving |= 1u << DR_SWING;
      }
    }
  }
  return d;
}

// ---- wakeUpActors 규칙 (ExtJoint.h:674): 장면에 있는 동적(비운동학) 강체는 wakeCounter < 재설정값이면 wakeUp(),
// 관절체 링크는 그 관절체의 wakeCounter 를 같은 규칙으로. 실제 wakeUp 은 강체 API(rigid_api.h)가 한다.
EHD bool needsWake(float wakeCounter, float wakeCounterResetValue) { return wakeCounter < wakeCounterResetValue; }

}  // namespace jnt
}  // namespace eng
