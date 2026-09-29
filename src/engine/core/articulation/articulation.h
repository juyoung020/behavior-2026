// articulation 모듈: PhysX 5.6.1 축약좌표 관절체(Featherstone)의 자료 구조 + 생성·설정 API + 장면 넣기. 손으로 짬, PhysX 링크 없음.
// 담당: articulation 작업자 (docs/엔진_자체구현.md 12·14절). 풀이 한 스텝은 art_step.h.
// 원본 (태그 107.3-omni-and-physx-5.6.1, physx/source 기준)
//   자료 구조 : lowleveldynamics/include/DyFeatherstoneArticulation.h:198 (ArticulationData), DyArticulationJointCore.h:67,
//               DyVArticulation.h:69 (ArticulationLink), lowlevel/api/include/PxvDynamics.h:139 (PxsBodyCore::init)
//   생성·API  : physx/src/NpArticulationReducedCoordinate.cpp:962 (createLink), NpFactory.cpp:250 (링크+관절 생성),
//               NpArticulationJointReducedCoordinate.cpp:487,518 (setParentPose/ChildPose), NpArticulationLink.cpp:197 (setCMassLocalPose),
//               simulationcontroller/src/ScBodyCore.cpp:45 (링크 기본값), ScArticulationCore.cpp:38, ScArticulationJointCore.cpp
//   장면 넣기 : physx/src/NpScene.cpp:1198 (addArticulationInternal, 링크 BFS 순서), ScArticulationSim.cpp:178,283,299,346
//               DyFeatherstoneArticulation.cpp:270-419 (dof 배치·경로), :3209 (teleportLinks), :3320 (computeLinkVelocities), :3378 (jcalc)
// 용량은 고정(동적 할당 없음). 넘치면 err 에 표시.
#pragma once
#include <cstdint>

#include "../common/glibc_sincosf.h"
#include "../common/pmath.h"
#include "art_math.h"

#ifndef ENG_ART_MAX_LINKS
#define ENG_ART_MAX_LINKS 64
#endif
#ifndef ENG_ART_MAX_DOFS
#define ENG_ART_MAX_DOFS 64
#endif
#ifndef ENG_ART_MAX_PATH
#define ENG_ART_MAX_PATH 1024
#endif
#ifndef ENG_ART_MAX_MIMIC
#define ENG_ART_MAX_MIMIC 8
#endif

namespace eng {
namespace art {

constexpr uint32_t kMaxLinks = ENG_ART_MAX_LINKS;
constexpr uint32_t kMaxDofs = ENG_ART_MAX_DOFS;
constexpr uint32_t kMaxPath = ENG_ART_MAX_PATH;
constexpr uint32_t kMaxMimic = ENG_ART_MAX_MIMIC;
constexpr uint32_t kNone = 0xffffffffu;

// PxSolverDefs.h:268-330
enum Axis : uint8_t { AX_TWIST = 0, AX_SWING1 = 1, AX_SWING2 = 2, AX_X = 3, AX_Y = 4, AX_Z = 5 };
enum Motion : uint8_t { M_LOCKED = 0, M_LIMITED = 1, M_FREE = 2 };
enum JointType : uint8_t { JT_FIX = 0, JT_PRISMATIC = 1, JT_REVOLUTE = 2, JT_REVOLUTE_UNWRAPPED = 3, JT_SPHERICAL = 4, JT_UNDEFINED = 5 };
enum DriveType : uint8_t { DT_FORCE = 0, DT_ACCELERATION = 1, DT_NONE = 2 };
enum ArtFlag : uint8_t { AF_FIX_BASE = 1, AF_DRIVE_LIMITS_ARE_FORCES = 2, AF_DISABLE_SELF_COLLISION = 4 };
enum ErrBits : uint32_t { ERR_LINKS = 1, ERR_DOFS = 2, ERR_PATH = 4, ERR_MIMIC = 8, ERR_UNDEFINED_JOINT = 16, ERR_STATE = 32 };

struct Envelope {  // PxPerformanceEnvelope
  float maxEffort, maxActuatorVelocity, velocityDependentResistance, speedEffortGradient;
};
struct Drive {  // PxArticulationDrive
  float stiffness, damping, maxForce;
  Envelope envelope;
  uint8_t driveType;
};
EHD Drive makeDrive(float stiffness, float damping, float maxForce, uint8_t type = DT_FORCE) {
  return Drive{stiffness, damping, maxForce, Envelope{0.0f, 0.0f, 0.0f, 0.0f}, type};
}
EHD Drive makeDriveEnvelope(float stiffness, float damping, const Envelope& env, uint8_t type = DT_FORCE) {
  return Drive{stiffness, damping, 0.0f, env, type};
}

// Dy::ArticulationJointCore (DyArticulationJointCore.h:67)
struct JointCore {
  Tf parentPose, childPose;
  float limLow[6], limHigh[6];
  Drive drives[6];
  float targetP[6], targetV[6], armature[6];
  float jointPos[6], jointVel[6];
  float frictionCoefficient;
  float fStatic[6], fDynamic[6], fViscous[6];  // PxJointFrictionParams
  float maxJointVelocity[6];
  uint32_t jointOffset;
  uint8_t dofIds[6], motion[6], invDofIds[6];
  uint8_t jCalcUpdateFrames, jointType;
};

// 링크 몸체 = PxsBodyCore 중 관절체가 쓰는 칸 + PxsRigidBody 잠 누적값 (PxvDynamics.h:139, PxsRigidBody.h)
struct LinkBody {
  Tf body2World, body2Actor;
  V3 linVel, angVel;
  float maxAngVelSq, maxLinVelSq, linDamping, angDamping;
  V3 invInertia;
  float invMass;
  float maxPenBias, maxContactImpulse, sleepThreshold, cfmScale;
  float wakeCounter;
  uint8_t disableGravity, retainAccelerations;
  V3 sleepLinVelAcc, sleepAngVelAcc;
  uint32_t numCountedInteractions;
};

// Dy::ArticulationLink (DyVArticulation.h:69) — bodyCore/inboundJoint 는 같은 번호의 bodies[]/joints[]
struct Link {
  uint32_t pathStart, childrenStart;
  uint16_t pathCount, numChildren;
  uint32_t parent;
  float cfm;
};
struct JointData {  // ArticulationJointCoreData (DyFeatherstoneArticulationJointData.h:44)
  uint32_t jointOffset;
  uint8_t nbDof, dofLimitMask;
};

// ArticulationInternalConstraint (DyFeatherstoneArticulation.h:113-162)
struct InternalConstraint {
  SV row0, row1, deltaVA, deltaVB;
  float recipResponse, response;
  ImplicitDrive drive;
  Envelope envelope;
  float externalJointForce, driveImpulse, driveMaxImpulse;
  float dynamicFrictionEffort, staticFrictionEffort, viscousFrictionCoefficient, frictionMaxForce, accumulatedFrictionImpulse;
  float maxJointVelocity;
  uint8_t isLinearConstraint;
};
struct InternalLimit {
  float errorLow, errorHigh, lowImpulse, highImpulse;
};
struct MimicCore {  // Dy::ArticulationMimicJointCore (링크 번호는 LL 번호)
  uint32_t linkA, linkB, axisA, axisB;
  float gearRatio, offset, naturalFrequency, dampingRatio;
};
struct MimicInternal {  // ArticulationInternalMimicJoint (DyFeatherstoneArticulation.h:165)
  uint32_t linkA, dofA, linkB, dofB;
  float gearRatio, offset, naturalFrequency, dampingRatio, recipEffectiveInertia;
};

// Cm::SpatialVector (linear, angular) — 외부 가속도 칸
struct LinAng {
  V3 linear, angular;
};

struct Articulation {
  // ---- 구성 (Sc::ArticulationCore + Dy)
  uint32_t nLinks, dofs, nMimic;
  uint32_t err;
  uint16_t solverIterationCounts;  // (vel << 8) | pos
  uint8_t flags;
  uint8_t inScene, awake, readyForSleep, jcalcDirty, jointDirty, dataDirty;
  float sleepThreshold, freezeThreshold, wakeCounter;
  // 생성 순서 <-> LL 순서 (장면에 넣은 뒤부터 LL 순서로 저장)
  uint32_t ll[kMaxLinks], creation[kMaxLinks], cparent[kMaxLinks];

  Link links[kMaxLinks];
  LinkBody bodies[kMaxLinks];
  JointCore joints[kMaxLinks];  // joints[i] = 링크 i 의 들어오는 관절 (0 은 안 씀)
  JointData jointData[kMaxLinks];
  MimicCore mimic[kMaxMimic];
  uint32_t pathToRoot[kMaxPath];
  uint32_t nPath;

  // ---- ArticulationData (DyFeatherstoneArticulation.h:381-464)
  SV rootPreMotionVelocity, rootDeferredZ;
  float jointAcceleration[kMaxDofs], jointInternalAcceleration[kMaxDofs], jointVelocity[kMaxDofs], jointNewVelocity[kMaxDofs + 3];
  float jointPosition[kMaxDofs], jointForce[kMaxDofs], jointTargetPositions[kMaxDofs], jointTargetVelocities[kMaxDofs];
  float posIterJointVelocities[kMaxDofs];
  SV posIterMotionVelocities[kMaxLinks], motionVelocities[kMaxLinks], solverLinkSpatialDeltaVels[kMaxLinks],
      solverLinkSpatialImpulses[kMaxLinks], motionAccelerations[kMaxLinks], motionAccelerationsInternal[kMaxLinks],
      coriolis[kMaxLinks], zaInternal[kMaxLinks], zaForces[kMaxLinks], transmittedForce[kMaxLinks];
  InternalConstraint ic[kMaxDofs];
  InternalLimit limits[kMaxDofs];
  uint32_t nIc, nLimits;
  MimicInternal mimicInternal[kMaxMimic];
  float deferredQstZ[kMaxDofs];
  SV deltaMotion[kMaxLinks];
  Tf preTransform[kMaxLinks];
  TIR responseW[kMaxLinks];
  SMat worldSpatialArticulatedInertia[kMaxLinks];
  M33 worldIsolatedInertia[kMaxLinks];
  float masses[kMaxLinks];
  InvStIs invStIs[kMaxLinks];
  SV isW[kMaxDofs];
  float qstZIc[kMaxDofs], qstZIntIc[kMaxDofs];
  SV jointAxis[kMaxDofs], motionMatrix[kMaxDofs], worldMotionMatrix[kMaxDofs];
  SV isInvStIS[kMaxDofs];
  V3 rw[kMaxLinks];
  Q relativeQuat[kMaxLinks];
  Tf accumulatedPoses[kMaxLinks];
  Q deltaQ[kMaxLinks];
  SMat baseInvSpatialArticulatedInertiaW;
  float invSumMass;
  V3 com;
  float dt;
  float linkMaxPenBias[kMaxLinks];  // ArticulationLinkData::maxPenBias
  LinAng externalAcceleration[kMaxLinks];
};

// ---------------------------------------------------------------- 쿼터니언 보조 (glibc sinf/cosf)
struct GlibcSinCos {
  EHD void operator()(float a, float& s, float& c) const {
    s = glibc::sinf(a);
    c = glibc::cosf(a);
  }
};
EHD Q quatAA(float angle, const V3& axis) { return quatAngleAxis(angle, axis, GlibcSinCos{}); }
EHD Q pxExp(const V3& v) {  // PxExp (PxMathUtils.h:143)
  const float m = magSq(v);
  return m < 1e-24f ? qid() : quatAA(psqrt(m), v * (1.0f / psqrt(m)));
}
EHD Tf transformInv(const Tf& t, const Tf& src) {  // PxTransform::transformInv (PxTransform.h:162)
  const Q qinv = conj(t.q);
  return Tf{qinv * src.q, rotate(qinv, src.p - t.p)};
}
// DyArticulationJointCore.h:40 rotateAndNormalize
EHD V3 rotateAndNormalize(const Q& q, const V3& v) {
  const float vx = v.x, vy = v.y, vz = v.z;
  const float x = q.x, y = q.y, z = q.z, w = q.w;
  const float w2 = w * w - 0.5f;
  const float dot2 = (x * vx + y * vy + z * vz);
  const V3 rotated{(vx * w2 + (y * vz - z * vy) * w + x * dot2), (vy * w2 + (z * vx - x * vz) * w + y * dot2),
                   (vz * w2 + (x * vy - y * vx) * w + z * dot2)};
  return v3normalizedFast(rotated);
}

// ---------------------------------------------------------------- 생성 (장면에 넣기 전, 생성 순서 번호)
struct SceneScale {
  float length = 1.0f, speed = 10.0f;  // PxTolerancesScale (omni: 1/metersPerUnit, 10/metersPerUnit)
};

EHD void initJointCore(JointCore& j, const Tf& parentFrame, const Tf& childFrame) {  // ArticulationJointCore::init (:98)
  j.parentPose = parentFrame;
  j.childPose = childFrame;
  j.jointOffset = 0;
  j.jCalcUpdateFrames = 1;
  j.frictionCoefficient = 0.05f;
  for (int i = 0; i < 6; ++i) j.maxJointVelocity[i] = 100.0f;
  j.jointType = JT_UNDEFINED;
  for (int i = 0; i < 6; ++i) {
    j.limLow[i] = 0.0f;
    j.limHigh[i] = 0.0f;
    j.drives[i] = makeDrive(0.0f, 0.0f, 0.0f, DT_NONE);
    j.fStatic[i] = j.fDynamic[i] = j.fViscous[i] = 0.0f;
    j.targetP[i] = j.targetV[i] = j.armature[i] = j.jointPos[i] = j.jointVel[i] = 0.0f;
    j.dofIds[i] = 0xff;
    j.invDofIds[i] = 0xff;
    j.motion[i] = M_LOCKED;
  }
}

// PxPhysics::createArticulationReducedCoordinate + Sc::ArticulationCore() (ScArticulationCore.cpp:38)
EHD void createArticulation(Articulation& a, const SceneScale& sc) {
  a.nLinks = 0;
  a.dofs = 0;
  a.nMimic = 0;
  a.err = 0;
  a.solverIterationCounts = (1 << 8) | 4;
  a.flags = 0;
  a.inScene = 0;
  a.awake = 0;
  a.readyForSleep = 0;
  a.jcalcDirty = 1;
  a.jointDirty = 0;
  a.dataDirty = 1;
  a.sleepThreshold = 5e-5f * sc.speed * sc.speed;
  a.freezeThreshold = 5e-6f * sc.speed * sc.speed;
  a.wakeCounter = 20.0f * 0.02f;  // Sc::Physics::sWakeCounterOnCreation
  a.nPath = 0;
  a.dt = 0.0f;
}

// createLink (NpArticulationReducedCoordinate.cpp:962, NpFactory.cpp:250, ScBodyCore.cpp:45) -> 생성 번호
EHD uint32_t createLink(Articulation& a, uint32_t parent, const Tf& poseIn, const SceneScale& sc) {
  if (a.nLinks >= kMaxLinks || a.inScene) {
    a.err |= ERR_LINKS;
    return kNone;
  }
  const uint32_t i = a.nLinks++;
  const Tf pose = normalized(poseIn);  // pose.getNormalized()
  LinkBody& b = a.bodies[i];
  b.body2World = pose;
  b.body2Actor = Tf{qid(), V3{0, 0, 0}};
  b.linVel = V3{0, 0, 0};
  b.angVel = V3{0, 0, 0};
  b.maxPenBias = -1e32f;
  b.maxAngVelSq = 50.0f * 50.0f;
  b.maxLinVelSq = 100.f * 100.f * sc.length * sc.length;
  b.linDamping = 0.05f;
  b.angDamping = 0.05f;
  b.invInertia = V3{1.0f, 1.0f, 1.0f};
  b.invMass = 1.0f;
  b.maxContactImpulse = 1e32f;
  b.sleepThreshold = 5e-5f * sc.speed * sc.speed;
  b.cfmScale = 0.025f;
  b.wakeCounter = 20.0f * 0.02f;
  b.disableGravity = 0;
  b.retainAccelerations = 0;
  b.sleepLinVelAcc = V3{0, 0, 0};
  b.sleepAngVelAcc = V3{0, 0, 0};
  b.numCountedInteractions = 0;
  a.cparent[i] = parent;
  if (parent != kNone) {
    // parentPose = parent->getCMassLocalPose().transformInv(pose), childPose = identity
    initJointCore(a.joints[i], transformInv(a.bodies[parent].body2Actor, pose), Tf{qid(), V3{0, 0, 0}});
  } else {
    initJointCore(a.joints[i], Tf{qid(), V3{0, 0, 0}}, Tf{qid(), V3{0, 0, 0}});
  }
  a.ll[i] = i;
  a.creation[i] = i;
  return i;
}

// 생성 번호 -> 저장 칸 번호 (장면에 넣기 전에는 같다)
EHD uint32_t slot(const Articulation& a, uint32_t creationIdx) { return a.ll[creationIdx]; }

// ---- 링크 API (NpRigidBodyTemplate.h, NpArticulationLink.cpp)
// checked 빌드의 PX_CHECK_AND_RETURN 을 그대로: 거부되는 호출은 무시 (NpRigidBodyTemplate.h:419-460)
EHD bool finiteF(float x) { return x - x == 0.0f; }  // PxIsFinite
EHD void linkSetMass(Articulation& a, uint32_t l, float m) {
  if (!finiteF(m) || !(m > 0.0f)) return;  // 링크는 질량 > 0
  a.bodies[slot(a, l)].invMass = m > 0.0f ? 1.0f / m : 0.0f;
}
EHD void linkSetMassSpaceInertiaTensor(Articulation& a, uint32_t l, const V3& m) {
  if (!finiteF(m.x) || !finiteF(m.y) || !finiteF(m.z) || !(m.x > 0.0f && m.y > 0.0f && m.z > 0.0f)) return;  // 링크는 성분 > 0
  a.bodies[slot(a, l)].invInertia =
      V3{m.x == 0.0f ? 0.0f : 1.0f / m.x, m.y == 0.0f ? 0.0f : 1.0f / m.y, m.z == 0.0f ? 0.0f : 1.0f / m.z};
}
// NpArticulationLink::setCMassLocalPose (NpArticulationLink.cpp:197) + Sc::BodyCore::setCMassLocalPose (ScBodyCore.cpp:98)
EHD void linkSetCMassLocalPose(Articulation& a, uint32_t l, const Tf& poseIn) {
  const uint32_t s = slot(a, l);
  LinkBody& b = a.bodies[s];
  const Tf p = normalized(poseIn);
  const Tf oldpose = b.body2Actor;
  const Tf comShift = transformInv(p, oldpose);
  const Tf oldActor2World = b.body2World * inverse(b.body2Actor);
  b.body2World = oldActor2World * p;
  b.body2Actor = p;
  if (a.cparent[l] != kNone) {
    JointCore& j = a.joints[s];
    j.childPose = comShift * j.childPose;
    j.jCalcUpdateFrames = 1;
  }
  for (uint32_t c = 0; c < a.nLinks; ++c) {  // mChildLinks 순서 = 생성 순서
    if (a.cparent[c] == l) {
      JointCore& j = a.joints[slot(a, c)];
      j.parentPose = comShift * j.parentPose;
      j.jCalcUpdateFrames = 1;
    }
  }
  if (a.inScene) a.jcalcDirty = 1;
}
EHD void linkSetLinearDamping(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].linDamping = v; }
EHD void linkSetAngularDamping(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].angDamping = v; }
EHD void linkSetMaxLinearVelocity(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].maxLinVelSq = v * v; }
EHD void linkSetMaxAngularVelocity(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].maxAngVelSq = v * v; }
EHD void linkSetCfmScale(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].cfmScale = v; }
EHD void linkSetMaxDepenetrationVelocity(Articulation& a, uint32_t l, float v) { a.bodies[slot(a, l)].maxPenBias = -v; }
EHD void linkSetDisableGravity(Articulation& a, uint32_t l, bool v) { a.bodies[slot(a, l)].disableGravity = v ? 1 : 0; }

// ---- 관절 API (NpArticulationJointReducedCoordinate.cpp, ScArticulationJointCore.cpp). l = 자식 링크의 생성 번호
EHD JointCore& J(Articulation& a, uint32_t l) { return a.joints[slot(a, l)]; }
EHD void markSimDirty(Articulation& a) {
  if (a.inScene) a.jcalcDirty = 1;  // Sc::ArticulationJointCore::setSimDirty
}
EHD void jointSetType(Articulation& a, uint32_t l, uint8_t t) {
  if (t == JT_UNDEFINED || a.inScene) return;  // NpArticulationJointReducedCoordinate.cpp:78-85
  J(a, l).jointType = t;
}
// isValidMotion (:98)
EHD bool isValidMotion(const JointCore& j, uint8_t axis, uint8_t motion) {
  bool valid = true;
  switch (j.jointType) {
    case JT_PRISMATIC:
      if (axis < AX_X && motion != M_LOCKED) valid = false;
      else if (motion != M_LOCKED)
        for (uint32_t i = AX_X; i <= AX_Z; i++)
          if (i != axis && j.motion[i] != M_LOCKED) valid = false;
      break;
    case JT_REVOLUTE:
    case JT_REVOLUTE_UNWRAPPED:
      if (axis >= AX_X && motion != M_LOCKED) valid = false;
      else if (motion != M_LOCKED)
        for (uint32_t i = AX_TWIST; i < AX_X; i++)
          if (i != axis && j.motion[i] != M_LOCKED) valid = false;
      break;
    case JT_SPHERICAL:
      if (axis >= AX_X && motion != M_LOCKED) valid = false;
      break;
    case JT_FIX:
      if (motion != M_LOCKED) valid = false;
      break;
    default:
      break;
  }
  return valid;
}
EHD void jointSetMotion(Articulation& a, uint32_t l, uint8_t axis, uint8_t m) {
  if (J(a, l).jointType == JT_UNDEFINED || !isValidMotion(J(a, l), axis, m) || a.inScene) return;
  J(a, l).motion[axis] = m;
}
EHD void jointSetLimit(Articulation& a, uint32_t l, uint8_t axis, float lo, float hi) {  // :292-294
  const uint8_t t = J(a, l).jointType;
  if (!(finiteF(lo) && finiteF(hi) && lo <= hi)) return;
  if (t == JT_SPHERICAL && !(fabsP(lo) <= 3.14159265358979323846f && fabsP(hi) <= 3.14159265358979323846f)) return;
  if (t == JT_REVOLUTE && !(fabsP(lo) <= 2.0f * 3.14159265358979323846f && fabsP(hi) <= 2.0f * 3.14159265358979323846f)) return;
  J(a, l).limLow[axis] = lo;
  J(a, l).limHigh[axis] = hi;
  markSimDirty(a);
}
EHD void jointSetDrive(Articulation& a, uint32_t l, uint8_t axis, const Drive& d) {  // :328 성능곡선 값 >= 0
  if (!(d.envelope.maxActuatorVelocity >= 0.0f && d.envelope.maxEffort >= 0.0f && d.envelope.velocityDependentResistance >= 0.0f &&
        d.envelope.speedEffortGradient >= 0.0f))
    return;
  J(a, l).drives[axis] = d;
  markSimDirty(a);
}
EHD void jointSetArmature(Articulation& a, uint32_t l, uint8_t axis, float v) {
  if (J(a, l).armature[axis] != v) {
    J(a, l).armature[axis] = v;
    markSimDirty(a);
  }
}
EHD void jointSetFrictionCoefficient(Articulation& a, uint32_t l, float v) {
  J(a, l).frictionCoefficient = v;
  markSimDirty(a);
}
EHD void jointSetFrictionParams(Articulation& a, uint32_t l, uint8_t axis, float st, float dyn, float visc) {
  if (!(st >= dyn)) return;  // :209 정적 >= 동적
  J(a, l).fStatic[axis] = st;
  J(a, l).fDynamic[axis] = dyn;
  J(a, l).fViscous[axis] = visc;
  markSimDirty(a);
}
EHD void jointSetMaxJointVelocity(Articulation& a, uint32_t l, uint8_t axis, float v) {
  J(a, l).maxJointVelocity[axis] = v;
  markSimDirty(a);
}
EHD void jointSetMaxJointVelocityAll(Articulation& a, uint32_t l, float v) {
  for (int i = 0; i < 6; ++i) J(a, l).maxJointVelocity[i] = v;
  markSimDirty(a);
}
// setParentPose: scSetParentPose(mParent->getCMassLocalPose().transformInv(t.getNormalized()))
EHD void jointSetParentPose(Articulation& a, uint32_t l, const Tf& t) {
  const uint32_t p = a.cparent[l];
  if (p == kNone) return;
  JointCore& j = J(a, l);
  const Tf np = transformInv(a.bodies[slot(a, p)].body2Actor, normalized(t));
  if (!(np.p.x == j.parentPose.p.x && np.p.y == j.parentPose.p.y && np.p.z == j.parentPose.p.z && np.q.x == j.parentPose.q.x &&
        np.q.y == j.parentPose.q.y && np.q.z == j.parentPose.q.z && np.q.w == j.parentPose.q.w)) {
    j.parentPose = np;
    j.jCalcUpdateFrames = 1;
    markSimDirty(a);
  }
}
EHD void jointSetChildPose(Articulation& a, uint32_t l, const Tf& t) {
  JointCore& j = J(a, l);
  const Tf np = transformInv(a.bodies[slot(a, l)].body2Actor, normalized(t));
  if (!(np.p.x == j.childPose.p.x && np.p.y == j.childPose.p.y && np.p.z == j.childPose.p.z && np.q.x == j.childPose.q.x &&
        np.q.y == j.childPose.q.y && np.q.z == j.childPose.q.z && np.q.w == j.childPose.q.w)) {
    j.childPose = np;
    j.jCalcUpdateFrames = 1;
    markSimDirty(a);
  }
}

// NpArticulationReducedCoordinate::autoWakeInternal (:1119): 깸 카운터가 0.4 보다 작으면 0.4 로
EHD void autoWake(Articulation& a) {
  const float reset = 20.0f * 0.02f;
  if (a.wakeCounter < reset) {
    for (uint32_t i = 0; i < a.nLinks; ++i) a.bodies[i].wakeCounter = reset;
    a.wakeCounter = reset;
    a.awake = 1;           // 링크 wakeUp -> 섬 관리자 activateNode (비활성 요청도 취소)
    a.readyForSleep = 0;
  }
}
// setDriveTarget / setDriveVelocity (NpArticulationJointReducedCoordinate.cpp:386, ScArticulationJointCore.cpp setTargetP/V)
EHD bool angleOk(uint8_t t, float v) {  // :390-391, :543-544
  if (t == JT_SPHERICAL && !(fabsP(v) <= 3.14159265358979323846f)) return false;
  if (t == JT_REVOLUTE && !(fabsP(v) <= 2.0f * 3.14159265358979323846f)) return false;
  return true;
}
EHD void jointSetDriveTarget(Articulation& a, uint32_t l, uint8_t axis, float v, bool autowake = true) {
  if (!angleOk(J(a, l).jointType, v)) return;
  if (autowake && a.inScene) autoWake(a);
  JointCore& j = J(a, l);
  j.targetP[axis] = v;
  if (a.inScene) {
    const uint32_t dofId = j.invDofIds[axis];
    if (dofId != 0xff) a.jointTargetPositions[a.jointData[slot(a, l)].jointOffset + dofId] = v;
  }
}
EHD void jointSetDriveVelocity(Articulation& a, uint32_t l, uint8_t axis, float v, bool autowake = true) {
  if (autowake && a.inScene) autoWake(a);
  JointCore& j = J(a, l);
  j.targetV[axis] = v;
  if (a.inScene) {
    const uint32_t dofId = j.invDofIds[axis];
    if (dofId != 0xff) a.jointTargetVelocities[a.jointData[slot(a, l)].jointOffset + dofId] = v;
  }
}
EHD void jointSetJointPosition(Articulation& a, uint32_t l, uint8_t axis, float v) {
  if (!finiteF(v) || !angleOk(J(a, l).jointType, v)) return;
  JointCore& j = J(a, l);
  j.jointPos[axis] = v;
  if (a.inScene) {
    const uint32_t dofId = j.invDofIds[axis];
    if (dofId != 0xff) a.jointPosition[a.jointData[slot(a, l)].jointOffset + dofId] = v;
  }
}
EHD void jointSetJointVelocity(Articulation& a, uint32_t l, uint8_t axis, float v) {
  if (!finiteF(v)) return;
  JointCore& j = J(a, l);
  j.jointVel[axis] = v;
  if (a.inScene) {
    const uint32_t dofId = j.invDofIds[axis];
    if (dofId != 0xff) a.jointVelocity[a.jointData[slot(a, l)].jointOffset + dofId] = v;
  }
}

// ---- 관절체 API
EHD void artSetSolverIterationCounts(Articulation& a, uint32_t pos, uint32_t vel) {
  a.solverIterationCounts = uint16_t(((vel & 0xff) << 8) | (pos & 0xff));
}
EHD void artSetFlag(Articulation& a, uint8_t f, bool v) { a.flags = v ? uint8_t(a.flags | f) : uint8_t(a.flags & ~f); }
EHD void artSetSleepThreshold(Articulation& a, float v) { a.sleepThreshold = v; }
EHD void artSetStabilizationThreshold(Articulation& a, float v) { a.freezeThreshold = v; }
// createMimicJoint(jointA, axisA, jointB, axisB, gearRatio, offset, naturalFrequency, dampingRatio) — 장면에 넣기 전만
EHD void createMimicJoint(Articulation& a, uint32_t linkA, uint8_t axisA, uint32_t linkB, uint8_t axisB, float gearRatio, float offset,
                          float naturalFrequency = 0.0f, float dampingRatio = 0.0f) {
  if (a.nMimic >= kMaxMimic || a.inScene) {
    a.err |= ERR_MIMIC;
    return;
  }
  a.mimic[a.nMimic++] = MimicCore{linkA, linkB, axisA, axisB, gearRatio, offset, naturalFrequency, dampingRatio};
}

// ---------------------------------------------------------------- 장면에 넣기 (NpScene.cpp:1198)
// jcalc: 관절 틀 (DyFeatherstoneArticulation.cpp:3378, DyArticulationJointCore.h:130 setJointFrame, :146 computeMotionMatrix)
EHD void setJointFrame(Articulation& a, uint32_t linkID) {
  JointCore& j = a.joints[linkID];
  const JointData& jd = a.jointData[linkID];
  if (!j.jCalcUpdateFrames) return;
  a.relativeQuat[linkID] = normalized(j.childPose.q * conj(j.parentPose.q));
  SV* mm = &a.motionMatrix[jd.jointOffset];
  const SV* ax = &a.jointAxis[jd.jointOffset];
  const V3 childOffset = -j.childPose.p;
  switch (j.jointType) {
    case JT_PRISMATIC: {
      const V3 u = rotateAndNormalize(j.childPose.q, ax[0].bottom);
      mm[0] = SV{V3{0.0f, 0.0f, 0.0f}, u};
      break;
    }
    case JT_REVOLUTE:
    case JT_REVOLUTE_UNWRAPPED: {
      const V3 u = rotateAndNormalize(j.childPose.q, ax[0].top);
      const V3 uXd = cross(u, childOffset);
      mm[0] = SV{u, uXd};
      break;
    }
    case JT_SPHERICAL: {
      for (uint32_t ind = 0; ind < jd.nbDof; ++ind) {
        const V3 u = rotateAndNormalize(j.childPose.q, ax[ind].top);
        const V3 uXd = cross(u, childOffset);
        mm[ind] = SV{u, uXd};
      }
      break;
    }
    default:
      break;
  }
  j.jCalcUpdateFrames = 0;
}
EHD void jcalc(Articulation& a) {
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    if (a.joints[linkID].jointType == JT_UNDEFINED) {  // PX_CHECK_AND_RETURN
      a.err |= ERR_UNDEFINED_JOINT;
      return;
    }
    setJointFrame(a, linkID);
  }
}

// teleportLinks (DyFeatherstoneArticulation.cpp:3209): 뿌리 자세와 관절 위치로 링크 자세를 다시 계산
EHD void teleportLinks(Articulation& a) {
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    const JointData& jd = a.jointData[linkID];
    const Tf pBody2World = a.bodies[link.parent].body2World;
    const JointCore& joint = a.joints[linkID];
    const float* jPosition = &a.jointPosition[jd.jointOffset];
    Q newParentToChild = qid();
    V3 r{0, 0, 0};
    const V3 childOffset = -joint.childPose.p;
    const V3 parentOffset = joint.parentPose.p;
    const Q relativeQuat = a.relativeQuat[linkID];
    switch (joint.jointType) {
      case JT_PRISMATIC: {
        newParentToChild = relativeQuat;
        const V3 e = rotate(newParentToChild, parentOffset);
        const V3 d = childOffset;
        const V3& u = a.motionMatrix[jd.jointOffset].bottom;
        r = e + d + u * jPosition[0];
        break;
      }
      case JT_REVOLUTE:
      case JT_REVOLUTE_UNWRAPPED: {
        const V3& u = a.motionMatrix[jd.jointOffset].top;
        Q jointRotation = quatAA(-jPosition[0], u);
        if (jointRotation.w < 0) jointRotation = qneg(jointRotation);
        newParentToChild = normalized(jointRotation * relativeQuat);
        const V3 e = rotate(newParentToChild, parentOffset);
        const V3 d = childOffset;
        r = e + d;
        break;
      }
      case JT_SPHERICAL: {
        Q jointRotation = qid();
        {
          V3 axis{0.f, 0.f, 0.f};
          for (uint32_t d = 0; d < jd.nbDof; ++d) axis += a.motionMatrix[jd.jointOffset + d].top * -jPosition[d];
          const float angle = v3normalize(axis);
          jointRotation = angle < 1e-10f ? qid() : quatAA(angle, axis);
          if (jointRotation.w < 0.f) jointRotation = qneg(jointRotation);
        }
        newParentToChild = normalized(jointRotation * relativeQuat);
        const V3 e = rotate(newParentToChild, parentOffset);
        const V3 d = childOffset;
        r = e + d;
        break;
      }
      case JT_FIX: {
        newParentToChild = relativeQuat;
        const V3 e = rotate(newParentToChild, parentOffset);
        const V3 d = childOffset;
        r = e + d;
        break;
      }
      default:
        break;
    }
    Tf& body2World = a.bodies[linkID].body2World;
    body2World.q = normalized(pBody2World.q * conj(newParentToChild));
    body2World.p = pBody2World.p + rotate(body2World.q, r);
  }
}

// computeLinkVelocities(ArticulationData&) (DyFeatherstoneArticulation.cpp:3320)
EHD void computeLinkVelocitiesAPI(Articulation& a) {
  const LinkBody& root = a.bodies[0];
  a.motionVelocities[0].top = root.angVel;
  a.motionVelocities[0].bottom = root.linVel;
  for (uint32_t linkID = 1; linkID < a.nLinks; ++linkID) {
    const Link& link = a.links[linkID];
    LinkBody& bodyCore = a.bodies[linkID];
    const Tf body2World = bodyCore.body2World;
    const LinkBody& pbodyCore = a.bodies[link.parent];
    const SV parentVel{pbodyCore.angVel, pbodyCore.linVel};
    const Tf pBody2World = pbodyCore.body2World;
    const V3 rwv = body2World.p - pBody2World.p;
    SV vel = translateSV(-rwv, parentVel);
    const JointData& jd = a.jointData[linkID];
    const float* jVelocity = &a.jointVelocity[jd.jointOffset];
    SV deltaV = svzero();
    for (uint32_t ind = 0; ind < jd.nbDof; ++ind) deltaV += a.motionMatrix[jd.jointOffset + ind] * jVelocity[ind];
    vel.top += rotate(body2World.q, deltaV.top);
    vel.bottom += rotate(body2World.q, deltaV.bottom);
    bodyCore.linVel = vel.bottom;
    bodyCore.angVel = vel.top;
    a.motionVelocities[linkID] = vel;
  }
}

// updateKinematic (ScArticulationSim.cpp:346)
EHD void updateKinematic(Articulation& a, bool position, bool velocity) {
  if (a.jcalcDirty) {
    jcalc(a);
    a.jcalcDirty = 0;
  }
  if (position) teleportLinks(a);
  if (position || velocity) computeLinkVelocitiesAPI(a);
}

// addArticulation: 링크를 LL(BFS) 순서로 다시 놓고 dof·경로를 만든 뒤 teleport. 성공하면 true.
EHD bool addToScene(Articulation& a) {
  if (a.inScene || a.nLinks == 0 || a.err) return false;
  const uint32_t n = a.nLinks;
  // BFS (NpScene.cpp:1233-1263): 루트, 그 다음 링크마다 자식(생성 순서)
  uint32_t order[kMaxLinks];
  uint32_t cnt = 1, cur = 0;
  order[0] = 0;
  while (cur < cnt) {
    const uint32_t p = order[cur++];
    for (uint32_t c = 0; c < n; ++c)
      if (a.cparent[c] == p) order[cnt++] = c;
  }
  if (cnt != n) {
    a.err |= ERR_STATE;
    return false;
  }
  // 재배치 (생성 번호 칸 -> LL 칸). 장면에 넣기 전에는 slot = 생성 번호.
  for (uint32_t i = 0; i < n; ++i) {
    a.creation[i] = order[i];
    a.ll[order[i]] = i;
  }
  {
    // 제자리 순열: 임시 한 칸씩 (GPU 에서도 되게 — 정렬 전 사본 없이 순환 따라가기)
    uint8_t done[kMaxLinks];
    for (uint32_t i = 0; i < n; ++i) done[i] = 0;
    for (uint32_t s = 0; s < n; ++s) {
      if (done[s]) continue;
      // 순환: LL 칸 s 에는 생성 번호 order[s] 의 값이 와야 한다
      uint32_t d = s;
      LinkBody tb = a.bodies[s];
      JointCore tj = a.joints[s];
      while (true) {
        done[d] = 1;
        const uint32_t src = order[d];
        if (src == s) {
          a.bodies[d] = tb;
          a.joints[d] = tj;
          break;
        }
        a.bodies[d] = a.bodies[src];
        a.joints[d] = a.joints[src];
        d = src;
      }
    }
  }
  // checkArticulationLink (NpScene.cpp:996): 질량 0 -> 1, 관성 0 -> (1,1,1)
  for (uint32_t i = 0; i < n; ++i) {
    LinkBody& b = a.bodies[i];
    if (b.invMass == 0.0f) b.invMass = 1.0f;  // getMass()==0 <=> invMass==0 (setMass(1): 1/1)
    const V3 inertia0{b.invInertia.x == 0.0f ? 0.0f : 1.0f / b.invInertia.x, b.invInertia.y == 0.0f ? 0.0f : 1.0f / b.invInertia.y,
                      b.invInertia.z == 0.0f ? 0.0f : 1.0f / b.invInertia.z};
    if (inertia0.x == 0.0f || inertia0.y == 0.0f || inertia0.z == 0.0f) b.invInertia = V3{1.0f, 1.0f, 1.0f};
  }
  // ArticulationSim::addBody (ScArticulationSim.cpp:178)
  for (uint32_t i = 0; i < n; ++i) {
    Link& l = a.links[i];
    l.pathStart = 0;
    l.pathCount = 0;
    l.childrenStart = kNone;
    l.numChildren = 0;
    l.cfm = 0.0f;
    if (i == 0) {
      l.parent = kNone;
    } else {
      const uint32_t p = a.ll[a.cparent[a.creation[i]]];
      l.parent = p;
      if (a.links[p].childrenStart == kNone) a.links[p].childrenStart = i;
      a.links[p].numChildren++;
      a.joints[i].jCalcUpdateFrames = 1;
    }
    a.bodies[i].wakeCounter = a.wakeCounter;  // BodySim::setArticulation -> setWakeCounterFromSim
    a.externalAcceleration[i] = LinAng{V3{0, 0, 0}, V3{0, 0, 0}};
  }
  // createLLStructure -> setupLinks -> setupDofs (DyFeatherstoneArticulation.cpp:330-360)
  uint32_t totalDofs = 0;
  for (uint32_t linkID = 1; linkID < n; ++linkID) {
    const JointCore& j = a.joints[linkID];
    for (int i = 0; i < 6; ++i)
      if (j.motion[i] != M_LOCKED) totalDofs++;
  }
  if (totalDofs > kMaxDofs) {
    a.err |= ERR_DOFS;
    return false;
  }
  a.dofs = totalDofs;
  // resizeJointData (:182) 는 가속도·속도·위치·힘·목표만 0 으로. 나머지도 0 으로 둔다(값을 쓰기 전에 읽지 않음).
  for (uint32_t i = 0; i < kMaxDofs; ++i) {
    a.jointAcceleration[i] = a.jointInternalAcceleration[i] = a.jointVelocity[i] = 0.0f;
    a.jointPosition[i] = a.jointForce[i] = a.jointTargetPositions[i] = a.jointTargetVelocities[i] = 0.0f;
    a.posIterJointVelocities[i] = a.deferredQstZ[i] = a.qstZIc[i] = a.qstZIntIc[i] = 0.0f;
  }
  for (uint32_t i = 0; i < kMaxDofs + 3; ++i) a.jointNewVelocity[i] = 0.0f;
  // configureDofs (:286)
  uint32_t totalDof = 0;
  for (uint32_t linkID = 1; linkID < n; ++linkID) {
    JointCore& j = a.joints[linkID];
    JointData& jd = a.jointData[linkID];
    jd.nbDof = 0;
    jd.dofLimitMask = 0;
    for (uint8_t i = 0; i < 6; ++i) {
      if (j.motion[i] != M_LOCKED) {
        SV axis = svzero();
        if (i < 3) (i == 0 ? axis.top.x : (i == 1 ? axis.top.y : axis.top.z)) = 1.f;
        else (i == 3 ? axis.bottom.x : (i == 4 ? axis.bottom.y : axis.bottom.z)) = 1.f;
        a.jointAxis[totalDof + jd.nbDof] = axis;
        j.invDofIds[i] = jd.nbDof;
        j.dofIds[jd.nbDof] = i;
        if (j.motion[i] == M_LIMITED) jd.dofLimitMask |= uint8_t(1 << jd.nbDof);
        jd.nbDof++;
      }
    }
    jd.jointOffset = totalDof;
    j.jointOffset = totalDof;
    totalDof += jd.nbDof;
  }
  a.jointData[0].jointOffset = 0xffffffffu;
  a.jointData[0].nbDof = 0;
  a.jointData[0].dofLimitMask = 0;
  // 흉내 관절: 링크 번호를 LL 로 (NpScene addArticulationMimicJointInternal)
  for (uint32_t m = 0; m < a.nMimic; ++m) {
    a.mimic[m].linkA = a.ll[a.mimic[m].linkA];
    a.mimic[m].linkB = a.ll[a.mimic[m].linkB];
  }
  a.inScene = 1;
  // 깸 상태: 깸 카운터 0.4 로 생성 -> 깨어 있음
  a.awake = a.wakeCounter != 0.0f ? 1 : 0;
  a.readyForSleep = 0;
  // initializeConfiguration (ScArticulationSim.cpp:299)
  jcalc(a);
  a.jcalcDirty = 0;
  for (uint32_t linkID = 1; linkID < n; ++linkID) {
    const JointCore& j = a.joints[linkID];
    const JointData& jd = a.jointData[linkID];
    for (uint8_t i = 0; i < jd.nbDof; ++i) {
      const uint32_t dofId = j.dofIds[i];
      a.jointPosition[jd.jointOffset + i] = j.jointPos[dofId];
      a.jointVelocity[jd.jointOffset + i] = j.jointVel[dofId];
      a.jointTargetPositions[jd.jointOffset + i] = j.targetP[dofId];
      a.jointTargetVelocities[jd.jointOffset + i] = j.targetV[dofId];
    }
  }
  // initPathToRoot (:371)
  {
    a.links[0].pathCount = 0;
    a.links[0].pathStart = 0;
    uint32_t total = 1;
    for (uint32_t linkID = 1; linkID < n; ++linkID) {
      uint32_t parent = a.links[linkID].parent;
      uint32_t c = 1;
      while (parent != 0) {
        parent = a.links[parent].parent;
        c++;
      }
      a.links[linkID].pathStart = total;
      a.links[linkID].pathCount = uint16_t(c);
      total += c;
    }
    if (total > kMaxPath) {
      a.err |= ERR_PATH;
      return false;
    }
    a.nPath = total;
    a.pathToRoot[0] = 0;
    for (uint32_t linkID = 1; linkID < n; ++linkID) {
      uint32_t* pr = &a.pathToRoot[a.links[linkID].pathStart];
      uint32_t k = a.links[linkID].pathCount;
      pr[--k] = linkID;
      uint32_t parent = a.links[linkID].parent;
      while (parent != 0) {
        pr[--k] = parent;
        parent = a.links[parent].parent;
      }
    }
  }
  // updateKinematicInternal(ePOSITION | eVELOCITY) (NpScene.cpp:1302)
  updateKinematic(a, true, true);
  return true;
}

// ---- 장면 안 API
// applyCache (DyFeatherstoneInverseDynamic.cpp:285) + wakeUpInternal (NpArticulationReducedCoordinate.cpp:1137)
enum CacheFlag : uint32_t {
  CF_VELOCITY = 1 << 0,
  CF_ACCELERATION = 1 << 1,
  CF_POSITION = 1 << 2,
  CF_FORCE = 1 << 3,
  CF_LINK_VELOCITY = 1 << 4,
  CF_LINK_ACCELERATION = 1 << 5,
  CF_ROOT_TRANSFORM = 1 << 6,
  CF_ROOT_VELOCITIES = 1 << 7,
  CF_LINK_FORCE = 1 << 13,
  CF_LINK_TORQUE = 1 << 14,
  CF_JOINT_TARGET_POSITIONS = 1 << 11,
  CF_JOINT_TARGET_VELOCITIES = 1 << 12,
};
struct CacheIn {  // PxArticulationCache 에서 쓰는 칸 (LL dof 순서)
  const float* jointVelocity;
  const float* jointPosition;
  const float* jointForce;
  const float* jointTargetPositions;
  const float* jointTargetVelocities;
  Tf rootTransform;  // 행위자 틀
  V3 rootLinVel, rootAngVel;
  const V3* linkForce;   // LL 링크 순서
  const V3* linkTorque;
};
EHD void wakeUpInternal(Articulation& a, bool forceWakeUp, bool autowake) {
  const float reset = 20.0f * 0.02f;
  float wc = a.wakeCounter;
  bool needsWakingUp = (!a.awake) && (autowake || forceWakeUp);
  if (autowake && (wc < reset)) {
    wc = reset;
    needsWakingUp = true;
  }
  if (needsWakingUp) {
    for (uint32_t i = 0; i < a.nLinks; ++i) a.bodies[i].wakeCounter = wc;
    a.wakeCounter = wc;
    a.awake = 1;
    a.readyForSleep = 0;
  }
}
EHD void applyCache(Articulation& a, const CacheIn& c, uint32_t flag, bool autowake = true) {
  bool shouldWake = false;
  const uint32_t dofCount = a.dofs;
  if (flag & CF_VELOCITY) {
    for (uint32_t i = 0; i < dofCount; ++i) {
      const float jv = c.jointVelocity[i];
      shouldWake = shouldWake || jv != 0.f;
      a.jointVelocity[i] = jv;
    }
  }
  if (flag & CF_ROOT_TRANSFORM) a.bodies[0].body2World = c.rootTransform * a.bodies[0].body2Actor;
  if (flag & CF_ROOT_VELOCITIES) {
    a.bodies[0].linVel = c.rootLinVel;
    a.bodies[0].angVel = c.rootAngVel;
    shouldWake = shouldWake || !isZero(c.rootLinVel) || !isZero(c.rootAngVel);
  }
  if (flag & CF_POSITION)
    for (uint32_t i = 0; i < dofCount; ++i) a.jointPosition[i] = c.jointPosition[i];
  if (flag & CF_FORCE) {
    for (uint32_t i = 0; i < dofCount; ++i) {
      const float jf = c.jointForce[i];
      shouldWake = shouldWake || jf != 0.f;
      a.jointForce[i] = jf;
    }
  }
  if (flag & CF_JOINT_TARGET_POSITIONS) {
    for (uint32_t i = 0; i < dofCount; ++i) {
      const float jt = c.jointTargetPositions[i];
      shouldWake = shouldWake || jt != a.jointPosition[i];
      a.jointTargetPositions[i] = jt;
    }
  }
  if (flag & CF_JOINT_TARGET_VELOCITIES) {
    for (uint32_t i = 0; i < dofCount; ++i) {
      const float jv = c.jointTargetVelocities[i];
      shouldWake = shouldWake || jv != a.jointVelocity[i];
      a.jointTargetVelocities[i] = jv;
    }
  }
  if (flag & CF_LINK_FORCE) {  // :366 선형 외부 가속도 = 힘 * invMass
    for (uint32_t i = 0; i < a.nLinks; ++i) {
      const V3 linkForce = c.linkForce[i];
      shouldWake = shouldWake || !(linkForce.x == 0.0f && linkForce.y == 0.0f && linkForce.z == 0.0f);
      a.externalAcceleration[i].linear = linkForce * a.bodies[i].invMass;
    }
  }
  if (flag & CF_LINK_TORQUE) {  // :380
    for (uint32_t i = 0; i < a.nLinks; ++i) {
      const Q& q = a.bodies[i].body2World.q;
      const V3 localLinkTorque = rotateInv(q, c.linkTorque[i]);
      shouldWake = shouldWake || !(localLinkTorque.x == 0.0f && localLinkTorque.y == 0.0f && localLinkTorque.z == 0.0f);
      const V3 localAccel = mulc(a.bodies[i].invInertia, localLinkTorque);
      a.externalAcceleration[i].angular = rotate(q, localAccel);
    }
  }
  // :403 관절 틀이 바뀌었으면 jcalc, 위치/뿌리 자세면 링크 순간이동, 속도류면 링크 속도 다시 계산
  if (a.jcalcDirty) jcalc(a);
  a.jcalcDirty = 0;
  if (flag & (CF_POSITION | CF_ROOT_TRANSFORM)) teleportLinks(a);
  if (flag & (CF_VELOCITY | CF_POSITION | CF_ROOT_VELOCITIES | CF_ROOT_TRANSFORM)) computeLinkVelocitiesAPI(a);
  wakeUpInternal(a, shouldWake, autowake);
}

// 링크 행위자 자세 (PxRigidActor::getGlobalPose = body2World * body2Actor^-1)
EHD Tf linkGlobalPose(const Articulation& a, uint32_t creationIdx) {
  const LinkBody& b = a.bodies[a.ll[creationIdx]];
  return b.body2World * inverse(b.body2Actor);
}

}  // namespace art
}  // namespace eng
