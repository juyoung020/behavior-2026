// joints 시험 공용 (PhysX 없음): 조인트 설정 명령 목록 만들기와 우리 엔진에 적용.
// 명령 목록 = omni.physx 가 USD 조인트를 만드는 순서를 흉내 (omni/extensions/runtime/source/omni.physx/plugins/usdInterface/
// UsdInterface.cpp:3498-4190: 고정/회전/직동/구 -> PxD6JointCreate + setMotion..., D6 -> eSWING_TWIST 드라이브 + 피라미드 한계).
// PhysX 쪽 적용은 joint_cmds.h (PhysX 헤더가 필요한 번역 단위에서만).
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/joints/d6_joint.h"

namespace jgen {

// 값은 PhysX 열거와 같다 (PxD6Axis / PxD6Motion / PxD6Drive / PxD6AngularDriveConfig / PxConstraintFlag)
enum Op { MOTION, ADC, DRIVE, DPOS, DVEL, TWIST, SWING, PYR, LIN, DIST, FLAG, IMS, BREAK };
struct Cmd {
  Op op;
  int a = 0, b = 0;
  float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};
static const float kMaxF = 3.40282346638528859812e+38F;

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

enum Kind { K_FIXED = 0, K_SPHERE, K_SPHERE_CONE, K_REVOLUTE, K_PRISMATIC, K_D6, K_SLERP, K_LEGACY_SWING, K_DISTANCE, K_ONE_SWING, K_COUNT };
static const char* kKindNames[K_COUNT] = {"고정", "구", "구+원뿔", "회전", "직동", "D6", "슬러프", "옛스윙", "거리", "스윙하나잠금"};
// PxConstraintFlag 값
enum : int { F_COLLISION = 1 << 3, F_DRIVE_FORCES = 1 << 5, F_DISABLE_PRE = 1 << 8, F_EXTENDED = 1 << 9 };
// PxD6Drive 값
enum : int { D_X = 0, D_Y = 1, D_Z = 2, D_SWING = 3, D_TWIST = 4, D_SLERP = 5, D_SWING1 = 6, D_SWING2 = 7 };

inline eng::Q qmul(const eng::Q& a, const eng::Q& b) { return a * b; }
// PxQuat(angle, axis) = (axis * sin(a/2), cos(a/2)) — PxSinCos 는 glibc sinf/cosf (이 값은 입력일 뿐이라 같은 값을 양쪽에 넣으면 된다)
inline eng::Q qaxis(float angle, int ax) {
  const float a = angle * 0.5f;
  const float s = std::sin(a), c = std::cos(a);
  return eng::Q{ax == 0 ? s : 0.0f, ax == 1 ? s : 0.0f, ax == 2 ? s : 0.0f, c};
}

struct Gen {
  std::mt19937 rng;
  std::uniform_real_distribution<float> U{-1.0f, 1.0f}, P{0.0f, 1.0f};
  explicit Gen(uint32_t s) : rng(s) {}
  float u() { return U(rng); }
  float p() { return P(rng); }
  int i(int n) { return int(rng() % uint32_t(n)); }
  eng::Q q() {
    double a = u(), b = u(), c = u(), d = u();
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return eng::Q{float(a / n), float(b / n), float(c / n), float(d / n)};
  }
  eng::V3 v(float s) { const float x = s * u(), y = s * u(), z = s * u(); return eng::V3{x, y, z}; }
  void limitParams(Cmd& c, bool soft) {
    c.f[4] = (i(3) == 0) ? 0.3f * p() : 0.0f;
    c.f[5] = 0.5f * p();
    c.f[6] = soft ? 50.0f * p() + 1.0f : 0.0f;
    c.f[7] = soft ? 5.0f * p() : 0.0f;
  }
  Cmd drive(int type) {
    Cmd c{DRIVE};
    c.a = type;
    c.f[0] = (i(4) == 0) ? 0.0f : 1000.0f * p();
    c.f[1] = 50.0f * p();
    c.f[2] = (i(3) == 0) ? kMaxF : 200.0f * p() + 1.0f;
    c.b = i(4);
    return c;
  }
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
          Cmd f{FLAG}; f.a = F_EXTENDED; f.b = i(2);
          M(3, 1);
          Cmd t{TWIST}; t.f[0] = -2.5f * p() - 0.05f; t.f[1] = 2.5f * p() + 0.05f; limitParams(t, i(4) == 0); cs.push_back(t);
          cs.push_back(f);
        } else {
          M(3, 2);
        }
        if (i(2)) {
          cs.push_back(drive(D_TWIST));
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
          cs.push_back(drive(D_X));
          Cmd v{DVEL}; v.f[0] = 0.5f * u(); v.b = 1; cs.push_back(v);
          Cmd d{DPOS}; d.f[3] = 1.0f; d.f[4] = 0.2f * u(); d.b = 1; cs.push_back(d);
        }
        break;
      }
      case K_D6: {
        Cmd a{ADC}; a.a = 0; cs.push_back(a);  // eSWING_TWIST
        M(4, 2); M(5, 2); M(3, 2); M(0, 2); M(1, 2); M(2, 2);
        bool rotLimit = false, pyr = false;
        Cmd py{PYR};
        for (int ax = 0; ax < 6; ++ax) {
          const int r = i(3);
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
        if (rotLimit) { Cmd f{FLAG}; f.a = F_EXTENDED; f.b = i(2); cs.push_back(f); }
        const int drives[6] = {D_X, D_Y, D_Z, D_TWIST, D_SWING1, D_SWING2};
        for (int k = 0; k < 6; ++k) if (i(2)) cs.push_back(drive(drives[k]));
        const float ax0 = 1.2f * u(), ay0 = 1.2f * u(), az0 = 1.2f * u();
        const eng::Q rot = qmul(qmul(qaxis(az0, 2), qaxis(ay0, 1)), qaxis(ax0, 0));
        Cmd d{DPOS}; d.f[0] = rot.x; d.f[1] = rot.y; d.f[2] = rot.z; d.f[3] = rot.w; d.f[4] = 0.1f * u(); d.f[5] = 0.1f * u(); d.f[6] = 0.1f * u(); d.b = 1;
        cs.push_back(d);
        Cmd v{DVEL}; for (int k = 0; k < 6; ++k) v.f[k] = 0.5f * u(); v.b = 1; cs.push_back(v);
        break;
      }
      case K_SLERP: {
        Cmd a{ADC}; a.a = (i(2) == 0) ? 1 : 2; cs.push_back(a);  // eSLERP / eLEGACY
        M(4, 2); M(5, 2); M(3, 2); M(0, 0); M(1, 0); M(2, 0);
        cs.push_back(drive(D_SLERP));
        const eng::Q r = q();
        Cmd d{DPOS}; d.f[0] = r.x; d.f[1] = r.y; d.f[2] = r.z; d.f[3] = r.w; d.b = 1; cs.push_back(d);
        Cmd v{DVEL}; for (int k = 3; k < 6; ++k) v.f[k] = 0.5f * u(); v.b = 1; cs.push_back(v);
        break;
      }
      case K_LEGACY_SWING: {
        M(3, 2); M(4, i(2) ? 2 : 0); M(5, 2); M(0, 0); M(1, 0); M(2, 0);
        cs.push_back(drive(D_SWING));
        cs.push_back(drive(D_TWIST));
        const eng::Q r = q();
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
    Cmd b{BREAK}; b.f[0] = (i(4) == 0) ? 500.0f * p() + 1.0f : kMaxF; b.f[1] = (i(4) == 0) ? 500.0f * p() + 1.0f : kMaxF; cs.push_back(b);
    Cmd c1{FLAG}; c1.a = F_COLLISION; c1.b = i(2); cs.push_back(c1);
    Cmd c2{FLAG}; c2.a = F_DRIVE_FORCES; c2.b = 1; cs.push_back(c2);
    if (i(8) == 0) { Cmd c3{FLAG}; c3.a = F_DISABLE_PRE; c3.b = 1; cs.push_back(c3); }
    if (i(6) == 0) { Cmd m{IMS}; m.a = i(4); m.f[0] = 0.5f + p(); cs.push_back(m); }
    return cs;
  }
};

}  // namespace jgen
