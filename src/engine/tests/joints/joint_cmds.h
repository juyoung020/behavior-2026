// joints 시험 공용 (PhysX 쪽): joint_gen.h 의 명령을 PhysX 조인트에 적용 + 형 변환. PhysX 헤더를 쓰는 번역 단위에서만.
#pragma once
#include "PxPhysicsAPI.h"
#include "joint_gen.h"

namespace jcmd {
using namespace physx;
using namespace jgen;

inline eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
inline eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
inline PxTransform toPx(const eng::Tf& t) { return PxTransform(PxVec3(t.p.x, t.p.y, t.p.z), PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)); }
inline PxQuat toPx(const eng::Q& q) { return PxQuat(q.x, q.y, q.z, q.w); }
inline PxVec3 toPx(const eng::V3& v) { return PxVec3(v.x, v.y, v.z); }

inline void applyPx(PxD6Joint* j, const Cmd& c) {
  auto lp = [&](PxJointLimitParameters& l) { l.restitution = c.f[4]; l.bounceThreshold = c.f[5]; l.stiffness = c.f[6]; l.damping = c.f[7]; };
  switch (c.op) {
    case MOTION: j->setMotion(PxD6Axis::Enum(c.a), PxD6Motion::Enum(c.b)); break;
    case ADC: j->setAngularDriveConfig(PxD6AngularDriveConfig::Enum(c.a)); break;
    case DRIVE: {
      PxD6JointDrive d(c.f[0], c.f[1], c.f[2], c.b & 1);
      if (c.b & 2) d.flags |= PxD6JointDriveFlag::eOUTPUT_FORCE;
      j->setDrive(PxD6Drive::Enum(c.a), d);
      break;
    }
    case DPOS: j->setDrivePosition(PxTransform(PxVec3(c.f[4], c.f[5], c.f[6]), PxQuat(c.f[0], c.f[1], c.f[2], c.f[3])), c.b != 0); break;
    case DVEL: j->setDriveVelocity(PxVec3(c.f[0], c.f[1], c.f[2]), PxVec3(c.f[3], c.f[4], c.f[5]), c.b != 0); break;
    case TWIST: { PxJointAngularLimitPair l(c.f[0], c.f[1]); lp(l); j->setTwistLimit(l); break; }
    case SWING: { PxJointLimitCone l(c.f[0], c.f[1]); lp(l); j->setSwingLimit(l); break; }
    case PYR: { PxJointLimitPyramid l(c.f[0], c.f[1], c.f[2], c.f[3]); lp(l); j->setPyramidSwingLimit(l); break; }
    case LIN: { PxJointLinearLimitPair l(PxTolerancesScale(), c.f[0], c.f[1]); lp(l); j->setLinearLimit(PxD6Axis::Enum(c.a), l); break; }
    case DIST: { PxJointLinearLimit l(c.f[0]); lp(l); j->setDistanceLimit(l); break; }
    case FLAG: j->setConstraintFlag(PxConstraintFlag::Enum(c.a), c.b != 0); break;
    case IMS:
      if (c.a == 0) j->setInvMassScale0(c.f[0]);
      else if (c.a == 1) j->setInvInertiaScale0(c.f[0]);
      else if (c.a == 2) j->setInvMassScale1(c.f[0]);
      else j->setInvInertiaScale1(c.f[0]);
      break;
    case BREAK: j->setBreakForce(c.f[0], c.f[1]); break;
  }
}

}  // namespace jcmd
