// joints 모듈: 동적 강체 API 의 부수효과 (순간이동·속도 설정·힘/토크·깨우기/재우기) — PhysX 5.6.1 과 같은 입력 -> 같은 상태.
// 원본(physx/source 기준)
//   API      : physx/src/NpRigidDynamic.cpp:87 (setGlobalPose), :227/:261 (setLinear/AngularVelocity), :295 (addForce), :322 (setForceAndTorque),
//              :350 (addTorque), :377/:401 (clearForce/Torque), :479 (setWakeCounter), :504 (wakeUp), :517 (putToSleep),
//              :599 (wakeUpInternalNoKinematicTest), NpRigidDynamic.h:148 (wakeUpInternal)
//   힘 변환  : physx/src/NpRigidBodyTemplate.h:507 (addSpatialForce), :558 (setSpatialForce), :609 (clearSpatialForce), :315 (전역 역관성)
//   Sc 층    : simulationcontroller/src/ScBodyCore.cpp:84 (setBody2World), :495 (setWakeCounter), :527 (putToSleep),
//              ScBodySim.cpp:302-375 (속도 수정 누적), :408 (postBody2WorldChange = lastTransform 저장), :414 (postSetWakeCounter),
//              :469 (deactivate: 속도 0), :513 (putToSleep), :656 (updateForces: simulate 의 beforeSolver 에서 속도에 더함),
//              ScBodySim.h:224 (setForcesToDefaults), ScSimStateData.h:68 (VelocityMod)
// 섬(활성/비활성) 관리는 solver 몫이다. 여기 함수는 "활성 여부"를 입력으로 받고, 섬 관리자에게 줄 요청(깨움/잠 준비)을 돌려준다.
#pragma once
#include <cstdint>

#include "../common/body.h"

namespace eng {
namespace jnt {

enum ForceMode : uint32_t { FM_FORCE = 0, FM_IMPULSE = 1, FM_VELOCITY_CHANGE = 2, FM_ACCELERATION = 3 };  // PxForceMode
enum : uint8_t { VMF_GRAVITY_DIRTY = 1 << 0, VMF_ACC_DIRTY = 1 << 1, VMF_VEL_DIRTY = 1 << 2 };          // Sc::VelocityModFlags

// Sc::BodySim 쪽 상태 중 API 가 건드리는 것 (몸체마다)
struct BodySimState {
  V3 linearPerSec, angularPerSec, linearPerStep, angularPerStep;  // VelocityMod
  uint8_t velModFlags;   // VMF_*
  uint8_t hasVelMod;     // mSimStateData 존재 (한 번 만들어지면 남는다)
  uint8_t active;        // BodySim::isActive (섬 관리자가 정함)
  uint8_t kinematic;     // PxRigidBodyFlag::eKINEMATIC
  uint8_t retainAccelerations;  // PxRigidBodyFlag::eRETAIN_ACCELERATIONS
  uint8_t disableSimulation;    // PxActorFlag::eDISABLE_SIMULATION
};
// 섬 관리자에게 줄 요청
// REQ_ACTIVATE = activateNode, REQ_DEACTIVATE = deactivateNode(잠 준비), REQ_SLEEP_NOW = putNodeToSleep(바로 재움)
enum : uint32_t { REQ_NONE = 0, REQ_ACTIVATE = 1, REQ_DEACTIVATE = 2, REQ_SLEEP_NOW = 3 };

EHD BodySimState makeBodySimState() {
  BodySimState s{};
  s.active = 1;
  return s;
}

// ---- 깨우기 (ScBodyCore.cpp:495 setWakeCounter, ScBodySim.cpp:414 postSetWakeCounter)
EHD uint32_t scSetWakeCounter(Body& b, BodySimState& s, float wakeCounter, bool forceWakeUp) {
  b.wakeCounter = wakeCounter;
  uint32_t req = REQ_NONE;
  if ((wakeCounter > 0.0f) || forceWakeUp) s.active = 1;  // sim->wakeUp() -> setActive(true)
  if ((wakeCounter > 0.0f) || forceWakeUp) {
    req = REQ_ACTIVATE;  // notifyNotReadyForSleeping -> activateNode
  } else {
    // checkSleepReadinessBesidesWakeCounter (ScBodySim.h:253): 속도 0 이고 걸린 힘/속도 수정도 0
    bool readyForSleep = isZero(b.linVel) && isZero(b.angVel);
    if (s.velModFlags & VMF_ACC_DIRTY) readyForSleep = readyForSleep && (!s.hasVelMod || (isZero(s.linearPerSec) && isZero(s.angularPerSec)));
    if (s.velModFlags & VMF_VEL_DIRTY) readyForSleep = readyForSleep && (!s.hasVelMod || (isZero(s.linearPerStep) && isZero(s.angularPerStep)));
    if (readyForSleep) req = REQ_DEACTIVATE;  // notifyReadyForSleeping -> deactivateNode
  }
  return req;
}
// NpRigidDynamic::wakeUpInternalNoKinematicTest (NpRigidDynamic.cpp:599)
EHD uint32_t wakeUpInternalNoKinematicTest(Body& b, BodySimState& s, bool forceWakeUp, bool autowake, float wakeCounterResetValue) {
  float wakeCounter = b.wakeCounter;
  bool needsWakingUp = (!s.active) && (autowake || forceWakeUp);
  if (autowake && (wakeCounter < wakeCounterResetValue)) {
    wakeCounter = wakeCounterResetValue;
    needsWakingUp = true;
  }
  if (needsWakingUp) return scSetWakeCounter(b, s, wakeCounter, true);  // BodyCore::wakeUp(wc) = setWakeCounter(wc, true)
  return REQ_NONE;
}

// ---- 순간이동 (NpRigidDynamic.cpp:87). inScene = 장면에 있음
EHD uint32_t setGlobalPose(Body& b, BodySimState& s, const Tf& pose, bool autowake, bool inScene, float wakeCounterResetValue) {
  const Tf newPose = normalized(pose);
  b.body2World = newPose * b.body2Actor;  // scSetBody2World
  b.lastTransform = b.body2World;         // postBody2WorldChange -> saveLastCCDTransform
  if (inScene && autowake && !s.disableSimulation && !s.kinematic)  // wakeUpInternal (운동학은 목표가 있을 때만 깨어 있음)
    return wakeUpInternalNoKinematicTest(b, s, false, true, wakeCounterResetValue);
  return REQ_NONE;
}

// ---- 속도 설정 (NpRigidDynamic.cpp:227, 261)
EHD uint32_t setLinearVelocity(Body& b, BodySimState& s, const V3& v, bool autowake, bool inScene, float wakeCounterResetValue) {
  b.linVel = v;
  return inScene ? wakeUpInternalNoKinematicTest(b, s, !isZero(v), autowake, wakeCounterResetValue) : REQ_NONE;
}
EHD uint32_t setAngularVelocity(Body& b, BodySimState& s, const V3& v, bool autowake, bool inScene, float wakeCounterResetValue) {
  b.angVel = v;
  return inScene ? wakeUpInternalNoKinematicTest(b, s, !isZero(v), autowake, wakeCounterResetValue) : REQ_NONE;
}

// ---- 힘·토크 (NpRigidBodyTemplate.h:507)
EHD M33 globalInvInertia(const Body& b) { return transformInertiaTensor(b.invInertia, mat_from_quat_simd(b.body2World.q)); }  // :315
EHD void accumulate(BodySimState& s, const V3* lin, const V3* ang, bool perSec, bool set) {
  if (perSec) s.velModFlags |= VMF_ACC_DIRTY;  // notifyDirtySpatialAcceleration
  else s.velModFlags |= VMF_VEL_DIRTY;         // notifyDirtySpatialVelocity
  if (!s.hasVelMod) {                          // setupSimStateData: VelocityMod 0 으로
    s.hasVelMod = 1;
    s.linearPerSec = s.angularPerSec = s.linearPerStep = s.angularPerStep = V3{0, 0, 0};
  }
  if (perSec) {
    if (lin) s.linearPerSec = set ? *lin : s.linearPerSec + *lin;
    if (ang) s.angularPerSec = set ? *ang : s.angularPerSec + *ang;
  } else {
    if (lin) s.linearPerStep = s.linearPerStep + *lin;
    if (ang) s.angularPerStep = s.angularPerStep + *ang;
  }
}
EHD void spatialForce(const Body& b, BodySimState& s, const V3* force, const V3* torque, uint32_t mode, bool set) {
  V3 la, aa;
  switch (mode) {
    case FM_FORCE:
      if (force) { la = *force * b.invMass; force = &la; }
      if (torque) { aa = globalInvInertia(b) * *torque; torque = &aa; }
      accumulate(s, force, torque, true, set);
      break;
    case FM_ACCELERATION: accumulate(s, force, torque, true, set); break;
    case FM_IMPULSE:
      if (force) { la = *force * b.invMass; force = &la; }
      if (torque) { aa = globalInvInertia(b) * *torque; torque = &aa; }
      accumulate(s, force, torque, false, false);  // set 판도 누적 (NpRigidBodyTemplate.h:581)
      break;
    case FM_VELOCITY_CHANGE: accumulate(s, force, torque, false, false); break;
  }
}
EHD uint32_t addForce(Body& b, BodySimState& s, const V3& force, uint32_t mode, bool autowake, float wakeCounterResetValue) {
  if (s.kinematic || s.disableSimulation) return REQ_NONE;  // PhysX 는 오류 보고 후 무시
  spatialForce(b, s, &force, nullptr, mode, false);
  return wakeUpInternalNoKinematicTest(b, s, !isZero(force), autowake, wakeCounterResetValue);
}
EHD uint32_t addTorque(Body& b, BodySimState& s, const V3& torque, uint32_t mode, bool autowake, float wakeCounterResetValue) {
  if (s.kinematic || s.disableSimulation) return REQ_NONE;
  spatialForce(b, s, nullptr, &torque, mode, false);
  return wakeUpInternalNoKinematicTest(b, s, !isZero(torque), autowake, wakeCounterResetValue);
}
EHD uint32_t setForceAndTorque(Body& b, BodySimState& s, const V3& force, const V3& torque, uint32_t mode, float wakeCounterResetValue) {
  if (s.kinematic || s.disableSimulation) return REQ_NONE;
  spatialForce(b, s, &force, &torque, mode, true);
  return wakeUpInternalNoKinematicTest(b, s, !isZero(force), true, wakeCounterResetValue);  // 원본도 force 만 본다
}
// clearForce/clearTorque (NpRigidBodyTemplate.h:609, ScBodySim.cpp:330,361)
EHD void clearSpatialForce(BodySimState& s, uint32_t mode, bool force, bool torque) {
  if (mode == FM_FORCE || mode == FM_ACCELERATION) {
    s.velModFlags |= VMF_ACC_DIRTY;
    if (s.hasVelMod) {
      if (force) s.linearPerSec = V3{0, 0, 0};
      if (torque) s.angularPerSec = V3{0, 0, 0};
    }
  } else {
    s.velModFlags |= VMF_VEL_DIRTY;
    if (s.hasVelMod) {
      if (force) s.linearPerStep = V3{0, 0, 0};
      if (torque) s.angularPerStep = V3{0, 0, 0};
    }
  }
}

// ---- 깨우기·재우기 API (NpRigidDynamic.cpp:479, 504, 517)
EHD uint32_t setWakeCounter(Body& b, BodySimState& s, float w) { return scSetWakeCounter(b, s, w, false); }
EHD uint32_t wakeUp(Body& b, BodySimState& s, float wakeCounterResetValue) { return scSetWakeCounter(b, s, wakeCounterResetValue, true); }
EHD void setForcesToDefaults(BodySimState& s) {  // ScBodySim.h:224 (eRETAIN_ACCELERATIONS 없을 때)
  if (!s.retainAccelerations) {
    if (s.hasVelMod) s.linearPerSec = s.angularPerSec = s.linearPerStep = s.angularPerStep = V3{0, 0, 0};
    s.velModFlags = 0;  // 중력 플래그는 우리 표현에서 안 씀 (중력은 적분에서 직접)
  } else {
    if (s.hasVelMod) s.linearPerStep = s.angularPerStep = V3{0, 0, 0};
    s.velModFlags &= uint8_t(~VMF_VEL_DIRTY);
  }
}
EHD uint32_t putToSleep(Body& b, BodySimState& s) {
  b.linVel = V3{0.0f, 0.0f, 0.0f};
  b.angVel = V3{0.0f, 0.0f, 0.0f};
  scSetWakeCounter(b, s, 0.0f, false);
  // BodySim::putToSleep (ScBodySim.cpp:513): 속도 수정 지움 + setActive(false)(-> deactivate: setForcesToDefaults) + putNodeToSleep
  s.velModFlags |= VMF_ACC_DIRTY | VMF_VEL_DIRTY;
  if (s.hasVelMod) s.linearPerSec = s.angularPerSec = s.linearPerStep = s.angularPerStep = V3{0, 0, 0};
  s.active = 0;
  setForcesToDefaults(s);
  return REQ_SLEEP_NOW;
}
// 섬 관리자가 몸체를 비활성으로 돌릴 때 (BodySim::deactivate, ScBodySim.cpp:469)
EHD void onDeactivate(Body& b, BodySimState& s) {
  b.linVel = V3{0.0f, 0.0f, 0.0f};
  b.angVel = V3{0.0f, 0.0f, 0.0f};
  setForcesToDefaults(s);
  s.active = 0;
}

// ---- simulate 의 beforeSolver: 누적된 힘을 속도에 더함 (ScBodySim.cpp:656 updateForces, ScBodyCore.h:72 updateVelocities)
EHD bool applyForces(Body& b, BodySimState& s, float dt) {
  const bool accDirty = (s.velModFlags & VMF_ACC_DIRTY) != 0;
  const bool velDirty = (s.velModFlags & VMF_VEL_DIRTY) != 0;
  bool applied = false;
  if ((accDirty || velDirty) && s.hasVelMod) {
    V3 linVelDt{0.0f, 0.0f, 0.0f}, angVelDt{0.0f, 0.0f, 0.0f};
    if (velDirty) {
      linVelDt = s.linearPerStep;
      angVelDt = s.angularPerStep;
    }
    if (accDirty) {
      linVelDt += s.linearPerSec * dt;
      angVelDt += s.angularPerSec * dt;
    }
    b.linVel += linVelDt;
    b.angVel += angVelDt;
    applied = true;
  }
  setForcesToDefaults(s);
  return applied;
}

}  // namespace jnt
}  // namespace eng
