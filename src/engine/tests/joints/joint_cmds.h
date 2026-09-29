// joints 시험 공용: 같은 조인트 설정 명령을 PhysX 와 우리 엔진에 똑같이 적용한다 (명령 목록 = omni.physx 가 USD 조인트를
// 만드는 순서를 흉내: omni/extensions/runtime/source/omni.physx/plugins/usdInterface/UsdInterface.cpp:3498-4190).
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "core/joints/d6_joint.h"

namespace jcmd {
using namespace physx;

enum Op { MOTION, ADC, DRIVE, DPOS, DVEL, TWIST, SWING, PYR, LIN, DIST, FLAG, IMS, BREAK };
struct Cmd {
  Op op;
  int a = 0, b = 0;
  float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

inline eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
inline eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
inline PxTransform toPx(const eng::Tf& t) { return PxTransform(PxVec3(t.p.x, t.p.y, t.p.z), PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)); }

// 한계 매개변수 (restitution, bounce, stiffness, damping) 는 f[4..7]
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

inline void applyEng(eng::jnt::D6Joint& j, const Cmd& c) {
  using namespace eng::jnt;
  auto fill = [&](auto& l) { l.restitution = c.f[4]; l.bounceThreshold = c.f[5]; l.stiffness = c.f[6]; l.damping = c.f[7]; };
  switch (c.op) {
    case MOTION: setMotion(j, uint32_t(c.a), uint32_t(c.b)); break;
    case ADC: setAngularDriveConfig(j, uint8_t(c.a)); break;
    case DRIVE: {
      Drive d{c.f[0], c.f[1], c.f[2], (c.b & 1) ? DRIVE_FLAG_ACCELERATION : 0u};
      if (c.b & 2) d.flags |= DRIVE_FLAG_OUTPUT_FORCE;
      setDrive(j, uint32_t(c.a), d);
      break;
    }
    case DPOS: setDrivePosition(j, eng::Tf{{c.f[0], c.f[1], c.f[2], c.f[3]}, {c.f[4], c.f[5], c.f[6]}}); break;
    case DVEL: setDriveVelocity(j, eng::V3{c.f[0], c.f[1], c.f[2]}, eng::V3{c.f[3], c.f[4], c.f[5]}); break;
    case TWIST: { AngularLimitPair l{}; l.lower = c.f[0]; l.upper = c.f[1]; fill(l); setTwistLimit(j, l); break; }
    case SWING: { LimitCone l{}; l.yAngle = c.f[0]; l.zAngle = c.f[1]; fill(l); setSwingLimit(j, l); break; }
    case PYR: { LimitPyramid l{}; l.yAngleMin = c.f[0]; l.yAngleMax = c.f[1]; l.zAngleMin = c.f[2]; l.zAngleMax = c.f[3]; fill(l); setPyramidSwingLimit(j, l); break; }
    case LIN: { LinearLimitPair l{}; l.lower = c.f[0]; l.upper = c.f[1]; fill(l); setLinearLimit(j, uint32_t(c.a), l); break; }
    case DIST: { LinearLimit l{}; l.value = c.f[0]; fill(l); setDistanceLimit(j, l); break; }
    case FLAG: {
      uint16_t f = j.constraintFlags;
      f = c.b ? uint16_t(f | uint16_t(c.a)) : uint16_t(f & ~uint16_t(c.a));
      setConstraintFlags(j, f);
      break;
    }
    case IMS:
      if (c.a == 0) setInvMassScale0(j, c.f[0]);
      else if (c.a == 1) setInvInertiaScale0(j, c.f[0]);
      else if (c.a == 2) setInvMassScale1(j, c.f[0]);
      else setInvInertiaScale1(j, c.f[0]);
      break;
    case BREAK: setBreakForce(j, c.f[0], c.f[1]); break;
  }
}

// 조인트 종류 (omni 가 만드는 모양)
enum Kind { K_FIXED = 0, K_SPHERE, K_SPHERE_CONE, K_REVOLUTE, K_PRISMATIC, K_D6, K_SLERP, K_LEGACY_SWING, K_DISTANCE, K_ONE_SWING, K_COUNT };
static const char* kKindNames[K_COUNT] = {"고정", "구", "구+원뿔", "회전", "직동", "D6", "슬러프", "옛스윙", "거리", "스윙하나잠금"};

struct Gen {
  std::mt19937 rng;
  std::uniform_real_distribution<float> U{-1.0f, 1.0f}, P{0.0f, 1.0f};
  explicit Gen(uint32_t s) : rng(s) {}
  float u() { return U(rng); }
  float p() { return P(rng); }
  int i(int n) { return int(rng() % uint32_t(n)); }
  PxQuat q() {
    double a = u(), b = u(), c = u(), d = u();
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  }
  PxVec3 v(float s) { return PxVec3(s * u(), s * u(), s * u()); }
  void limitParams(Cmd& c, bool soft) {
    c.f[4] = (i(3) == 0) ? 0.3f * p() : 0.0f;  // restitution
    c.f[5] = 0.5f * p();                       // bounceThreshold
    c.f[6] = soft ? 50.0f * p() + 1.0f : 0.0f;
    c.f[7] = soft ? 5.0f * p() : 0.0f;
  }
  Cmd drive(int type) {
    Cmd c{DRIVE};
    c.a = type;
    c.f[0] = (i(4) == 0) ? 0.0f : 1000.0f * p();
    c.f[1] = 50.0f * p();
    c.f[2] = (i(3) == 0) ? PX_MAX_F32 : 200.0f * p() + 1.0f;
    c.b = i(4);  // bit0 가속, bit1 출력힘
    return c;
  }
  // omni UsdInterface.cpp 순서를 따른 명령 목록
  std::vector<Cmd> joint(int kind) {
    std::vector<Cmd> cs;
    auto M = [&](int ax, int m) { Cmd c{MOTION}; c.a = ax; c.b = m; cs.push_back(c); };
    auto lockAll = [&]() { M(4, 0); M(5, 0); M(3, 0); M(0, 0); M(1, 0); M(2, 0); };
    switch (kind) {
      case K_FIXED: lockAll(); break;
      case K_SPHERE: M(4, 2); M(5, 2); M(3, 2); M(0, 0); M(1, 0); M(2, 0); break;
      case K_SPHERE_CONE: {
        M(4, 2); M(5, 2); M(3, 2); M(0, 0); M(1, 0); M(2, 0);
        M(4, 1); M(5, 1);
        Cmd c{SWING}; c.f[0] = 0.2f + 1.3f * p(); c.f[1] = 0.2f + 1.3f * p(); limitParams(c, i(3) == 0); cs.push_back(c);
        break;
      }
      case K_REVOLUTE: {
        lockAll();
        if (i(3) != 0) {
          Cmd f{FLAG}; f.a = PxConstraintFlag::eENABLE_EXTENDED_LIMITS; f.b = i(2);
          M(3, 1);
          Cmd t{TWIST}; t.f[0] = -2.5f * p() - 0.05f; t.f[1] = 2.5f * p() + 0.05f; limitParams(t, i(4) == 0); cs.push_back(t);
          cs.push_back(f);
        } else {
          M(3, 2);
        }
        if (i(2)) {
          cs.push_back(drive(PxD6Drive::eTWIST));
          Cmd v{DVEL}; v.f[3] = 2.0f * u(); v.b = 1; cs.push_back(v);
          Cmd d{DPOS}; const float a = 1.5f * u(); d.f[0] = std::sin(a * 0.5f); d.f[3] = std::cos(a * 0.5f); d.b = 1; cs.push_back(d);
        }
        break;
      }
      case K_PRISMATIC: {
        lockAll();
        if (i(3) != 0) {
          M(0, 1);
          Cmd l{LIN}; l.a = 0; l.f[0] = -0.3f * p() - 0.01f; l.f[1] = 0.3f * p() + 0.01f; limitParams(l, i(4) == 0); cs.push_back(l);
        } else {
          M(0, 2);
        }
        if (i(2)) {
          cs.push_back(drive(PxD6Drive::eX));
          Cmd v{DVEL}; v.f[0] = 0.5f * u(); v.b = 1; cs.push_back(v);
          Cmd d{DPOS}; d.f[3] = 1.0f; d.f[4] = 0.2f * u(); d.b = 1; cs.push_back(d);
        }
        break;
      }
      case K_D6: {
        Cmd a{ADC}; a.a = PxD6AngularDriveConfig::eSWING_TWIST; cs.push_back(a);
        M(4, 2); M(5, 2); M(3, 2); M(0, 2); M(1, 2); M(2, 2);
        bool rotLimit = false;
        bool pyr = false;
        Cmd py{PYR}; py.f[0] = 0; py.f[1] = 0; py.f[2] = 0; py.f[3] = 0;
        for (int ax = 0; ax < 6; ++ax) {
          const int r = i(3);  // 0 잠금, 1 한계, 2 자유
          if (r == 0) { M(ax, 0); continue; }
          if (r == 2) continue;
          if (ax < 3) {
            M(ax, 1);
            Cmd l{LIN}; l.a = ax; l.f[0] = -0.3f * p() - 0.01f; l.f[1] = 0.3f * p() + 0.01f; limitParams(l, i(4) == 0); cs.push_back(l);
          } else if (ax == 3) {
            M(3, 1);
            Cmd t{TWIST}; t.f[0] = -2.5f * p() - 0.05f; t.f[1] = 2.5f * p() + 0.05f; limitParams(t, i(4) == 0); cs.push_back(t);
            rotLimit = true;
          } else if (ax == 4) {
            pyr = true; py.f[0] = -1.4f * p() - 0.05f; py.f[1] = 1.4f * p() + 0.05f; M(4, 1); rotLimit = true;
          } else {
            pyr = true; py.f[2] = -1.4f * p() - 0.05f; py.f[3] = 1.4f * p() + 0.05f; M(5, 1); rotLimit = true;
          }
        }
        if (pyr) { limitParams(py, i(4) == 0); cs.push_back(py); }
        if (rotLimit) { Cmd f{FLAG}; f.a = PxConstraintFlag::eENABLE_EXTENDED_LIMITS; f.b = i(2); cs.push_back(f); }
        const int drives[6] = {PxD6Drive::eX, PxD6Drive::eY, PxD6Drive::eZ, PxD6Drive::eTWIST, PxD6Drive::eSWING1, PxD6Drive::eSWING2};
        for (int k = 0; k < 6; ++k) if (i(2)) cs.push_back(drive(drives[k]));
        const PxQuat qx(1.2f * u(), PxVec3(1, 0, 0)), qy(1.2f * u(), PxVec3(0, 1, 0)), qz(1.2f * u(), PxVec3(0, 0, 1));
        const PxQuat rot = qz * qy * qx;
        Cmd d{DPOS}; d.f[0] = rot.x; d.f[1] = rot.y; d.f[2] = rot.z; d.f[3] = rot.w; d.f[4] = 0.1f * u(); d.f[5] = 0.1f * u(); d.f[6] = 0.1f * u(); d.b = 1;
        cs.push_back(d);
        Cmd v{DVEL}; for (int k = 0; k < 6; ++k) v.f[k] = 0.5f * u(); v.b = 1; cs.push_back(v);
        break;
      }
      case K_SLERP: {
        Cmd a{ADC}; a.a = (i(2) == 0) ? PxD6AngularDriveConfig::eSLERP : PxD6AngularDriveConfig::eLEGACY; cs.push_back(a);
        M(4, 2); M(5, 2); M(3, 2); M(0, 0); M(1, 0); M(2, 0);
        cs.push_back(drive(PxD6Drive::eSLERP));
        const PxQuat r = q();
        Cmd d{DPOS}; d.f[0] = r.x; d.f[1] = r.y; d.f[2] = r.z; d.f[3] = r.w; d.b = 1; cs.push_back(d);
        Cmd v{DVEL}; for (int k = 3; k < 6; ++k) v.f[k] = 0.5f * u(); v.b = 1; cs.push_back(v);
        break;
      }
      case K_LEGACY_SWING: {
        M(3, 2); M(4, i(2) ? 2 : 0); M(5, 2); M(0, 0); M(1, 0); M(2, 0);
        cs.push_back(drive(PxD6Drive::eSWING));
        cs.push_back(drive(PxD6Drive::eTWIST));
        const PxQuat r = q();
        Cmd d{DPOS}; d.f[0] = r.x; d.f[1] = r.y; d.f[2] = r.z; d.f[3] = r.w; d.b = 1; cs.push_back(d);
        Cmd v{DVEL}; for (int k = 3; k < 6; ++k) v.f[k] = 0.5f * u(); v.b = 1; cs.push_back(v);
        break;
      }
      case K_DISTANCE: {
        M(4, 2); M(5, 2); M(3, 2); M(0, 1); M(1, 1); M(2, i(2) ? 1 : 0);
        Cmd c{DIST}; c.f[0] = 0.05f + 0.3f * p(); limitParams(c, i(3) == 0); cs.push_back(c);
        break;
      }
      case K_ONE_SWING: {
        M(3, 2); M(0, 0); M(1, 0); M(2, 0);
        if (i(2)) { M(4, 0); M(5, i(2) ? 2 : 1); } else { M(5, 0); M(4, i(2) ? 2 : 1); }
        Cmd s{SWING}; s.f[0] = 0.2f + 1.3f * p(); s.f[1] = 0.2f + 1.3f * p(); limitParams(s, i(3) == 0); cs.push_back(s);
        break;
      }
    }
    // omni 공통 (UsdInterface.cpp:4181-4185): break force, 충돌, 드라이브 한계 = 힘
    Cmd b{BREAK}; b.f[0] = (i(4) == 0) ? 500.0f * p() + 1.0f : PX_MAX_F32; b.f[1] = (i(4) == 0) ? 500.0f * p() + 1.0f : PX_MAX_F32; cs.push_back(b);
    Cmd c1{FLAG}; c1.a = PxConstraintFlag::eCOLLISION_ENABLED; c1.b = i(2); cs.push_back(c1);
    Cmd c2{FLAG}; c2.a = PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES; c2.b = 1; cs.push_back(c2);
    if (i(8) == 0) { Cmd c3{FLAG}; c3.a = PxConstraintFlag::eDISABLE_PREPROCESSING; c3.b = 1; cs.push_back(c3); }
    if (i(6) == 0) { Cmd m{IMS}; m.a = i(4); m.f[0] = 0.5f + p(); cs.push_back(m); }
    return cs;
  }
};

}  // namespace jcmd
