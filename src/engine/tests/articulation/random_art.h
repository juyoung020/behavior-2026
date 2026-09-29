// articulation 시험 공용: 무작위 관절체를 PhysX 와 우리 엔진에 같은 API 로 만들고(Mirror), 매 스텝 같은 입력을 넣는다.
// 입력 계획(ArtInputs + stepInputsEng)은 호스트·GPU 공용(EHD) — GPU 판도 같은 스텝에 같은 드라이브 목표·applyCache 를 넣는다.
// 판(env) 번호 e 는 드라이브 목표 위상만 바꾼다 (t_e = s*dt + e*0.01). 판 0 = PhysX 와 같은 입력.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "core/articulation/art_step.h"

namespace artest {
namespace A = eng::art;

constexpr uint32_t kMaxDriven = 48;
struct ArtInputs {  // 한 관절체에 넣는 입력 (POD, GPU 로 복사)
  uint32_t n;
  uint32_t link[kMaxDriven];  // 생성 번호
  uint8_t axis[kMaxDriven];
  uint8_t spherical[kMaxDriven];
  uint8_t touched;
};

EHD float envTime(int s, int env, float dt) { return float(s) * dt + float(env) * 0.01f; }
EHD float sinE(float x) {
#if defined(__CUDA_ARCH__)
  return eng::glibc::sinf(x);
#else
  return ::sinf(x);  // glibc libm (이식본과 비트 동일, test_sincosf)
#endif
}
EHD float cosE(float x) {
#if defined(__CUDA_ARCH__)
  return eng::glibc::cosf(x);
#else
  return ::cosf(x);
#endif
}
EHD float targetValue(const ArtInputs& in, uint32_t d, float t) {
  const bool rot = in.axis[d] < 3;
  const float amp = rot ? (in.spherical[d] ? 0.6f : 1.0f) : 0.15f;
  return amp * sinE(0.7f * t * float(1 + d));
}
EHD float targetVelValue(float t) { return 0.3f * cosE(1.1f * t); }

// 우리 엔진 쪽: 스텝 s 의 입력 (PhysX 쪽은 applyInputsPx 와 같은 순서·값)
EHD void stepInputsEng(A::Articulation& a, const ArtInputs& in, int s, int env, float dt) {
  const float t = envTime(s, env, dt);
  if (in.touched) {
    for (uint32_t d = 0; d < in.n; ++d) {
      A::jointSetDriveTarget(a, in.link[d], in.axis[d], targetValue(in, d, t));
      if ((d % 2) == 1) A::jointSetDriveVelocity(a, in.link[d], in.axis[d], targetVelValue(t));
    }
  }
  const uint32_t nd = a.dofs;
  float buf[A::kMaxDofs];
  if (s == 150 && nd) {
    for (uint32_t i = 0; i < nd; ++i) buf[i] = a.jointPosition[i] + 0.05f;
    A::CacheIn c{};
    c.jointPosition = buf;
    A::applyCache(a, c, A::CF_POSITION);
  }
  if (s == 250 && nd) {
    for (uint32_t i = 0; i < nd; ++i) buf[i] = 0.2f * sinE(float(i));
    A::CacheIn c{};
    c.jointVelocity = buf;
    A::applyCache(a, c, A::CF_VELOCITY);
  }
  if (s == 300 && !(a.flags & A::AF_FIX_BASE)) {
    A::CacheIn c{};
    c.rootLinVel = eng::V3{0.3f, -0.1f, 0.5f};
    c.rootAngVel = eng::V3{0.2f, 0.4f, -0.3f};
    A::applyCache(a, c, A::CF_ROOT_VELOCITIES);
  }
  if (s == 350 && !(a.flags & A::AF_FIX_BASE)) {
    eng::Tf rt = A::linkGlobalPose(a, 0);
    rt.p = rt.p + eng::V3{0.0f, 0.0f, 1.0f};  // PxVec3 += (0,0,1)
    A::CacheIn c{};
    c.rootTransform = rt;
    A::applyCache(a, c, A::CF_ROOT_TRANSFORM);
  }
  if (s >= 400 && s < 450 && nd) {
    for (uint32_t i = 0; i < nd; ++i) buf[i] = 0.5f * sinE(float(i + uint32_t(s)));
    A::CacheIn c{};
    c.jointForce = buf;
    A::applyCache(a, c, A::CF_FORCE);
  }
  if (s == 450 && nd) {
    for (uint32_t i = 0; i < nd; ++i) buf[i] = 0.0f;
    A::CacheIn c{};
    c.jointForce = buf;
    A::applyCache(a, c, A::CF_FORCE);
  }
}

}  // namespace artest

#ifndef ARTEST_NO_PHYSX
#include "PxPhysicsAPI.h"

namespace artest {
using namespace physx;

inline eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
inline eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

// 한 관절체를 양쪽에 똑같이 만드는 거울
struct Mirror {
  PxArticulationReducedCoordinate* px = nullptr;
  std::vector<PxArticulationLink*> pl;  // 생성 순서
  std::unique_ptr<A::Articulation> e;
  A::SceneScale sc;
  ArtInputs in{};
  std::vector<uint8_t> jtype;
  PxArticulationCache* cache = nullptr;

  void create(PxPhysics* phys) {
    px = phys->createArticulationReducedCoordinate();
    e.reset(new A::Articulation);
    A::createArticulation(*e, sc);
    in.n = 0;
    in.touched = 1;
  }
  uint32_t link(uint32_t parent, const PxTransform& pose) {
    PxArticulationLink* l = px->createLink(parent == A::kNone ? nullptr : pl[parent], pose);
    pl.push_back(l);
    jtype.push_back(0);
    return A::createLink(*e, parent, toE(pose), sc);
  }
  void mass(uint32_t l, float m, const PxVec3& I) {
    pl[l]->setMass(m);
    pl[l]->setMassSpaceInertiaTensor(I);
    A::linkSetMass(*e, l, m);
    A::linkSetMassSpaceInertiaTensor(*e, l, toE(I));
  }
  void cmass(uint32_t l, const PxTransform& t) {
    pl[l]->setCMassLocalPose(t);
    A::linkSetCMassLocalPose(*e, l, toE(t));
  }
  PxArticulationJointReducedCoordinate* J(uint32_t l) { return pl[l]->getInboundJoint(); }
  void type(uint32_t l, PxArticulationJointType::Enum t) {
    J(l)->setJointType(t);
    A::jointSetType(*e, l, uint8_t(t));
    jtype[l] = uint8_t(t);
  }
  void motion(uint32_t l, PxArticulationAxis::Enum ax, PxArticulationMotion::Enum m) {
    J(l)->setMotion(ax, m);
    A::jointSetMotion(*e, l, uint8_t(ax), uint8_t(m));
  }
  void limit(uint32_t l, PxArticulationAxis::Enum ax, float lo, float hi) {
    J(l)->setLimitParams(ax, PxArticulationLimit(lo, hi));
    A::jointSetLimit(*e, l, uint8_t(ax), lo, hi);
  }
  void drive(uint32_t l, PxArticulationAxis::Enum ax, float k, float d, float fmax, PxArticulationDriveType::Enum t) {
    J(l)->setDriveParams(ax, PxArticulationDrive(k, d, fmax, t));
    A::jointSetDrive(*e, l, uint8_t(ax), A::makeDrive(k, d, fmax, uint8_t(t)));
  }
  void driveEnv(uint32_t l, PxArticulationAxis::Enum ax, float k, float d, const PxPerformanceEnvelope& env,
                PxArticulationDriveType::Enum t) {
    J(l)->setDriveParams(ax, PxArticulationDrive(k, d, env, t));
    A::jointSetDrive(*e, l, uint8_t(ax),
                     A::makeDriveEnvelope(k, d, A::Envelope{env.maxEffort, env.maxActuatorVelocity, env.velocityDependentResistance,
                                                            env.speedEffortGradient},
                                          uint8_t(t)));
  }
  void parentPose(uint32_t l, const PxTransform& t) {
    J(l)->setParentPose(t);
    A::jointSetParentPose(*e, l, toE(t));
  }
  void childPose(uint32_t l, const PxTransform& t) {
    J(l)->setChildPose(t);
    A::jointSetChildPose(*e, l, toE(t));
  }
  void armature(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setArmature(ax, v);
    A::jointSetArmature(*e, l, uint8_t(ax), v);
  }
  void frictionCoef(uint32_t l, float v) {
    J(l)->setFrictionCoefficient(v);
    A::jointSetFrictionCoefficient(*e, l, v);
  }
  void frictionParams(uint32_t l, PxArticulationAxis::Enum ax, float st, float dy, float vi) {
    J(l)->setFrictionParams(ax, PxJointFrictionParams(st, dy, vi));
    A::jointSetFrictionParams(*e, l, uint8_t(ax), st, dy, vi);
  }
  void maxJointVel(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setMaxJointVelocity(ax, v);
    A::jointSetMaxJointVelocity(*e, l, uint8_t(ax), v);
  }
  void jointPos(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setJointPosition(ax, v);
    A::jointSetJointPosition(*e, l, uint8_t(ax), v);
  }
  void jointVel(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setJointVelocity(ax, v);
    A::jointSetJointVelocity(*e, l, uint8_t(ax), v);
  }
  void linkDamping(uint32_t l, float lin, float ang) {
    pl[l]->setLinearDamping(lin);
    pl[l]->setAngularDamping(ang);
    A::linkSetLinearDamping(*e, l, lin);
    A::linkSetAngularDamping(*e, l, ang);
  }
  void linkMaxVel(uint32_t l, float lin, float ang) {
    pl[l]->setMaxLinearVelocity(lin);
    pl[l]->setMaxAngularVelocity(ang);
    A::linkSetMaxLinearVelocity(*e, l, lin);
    A::linkSetMaxAngularVelocity(*e, l, ang);
  }
  void noGravity(uint32_t l) {
    pl[l]->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    A::linkSetDisableGravity(*e, l, true);
  }

  // PhysX 쪽 스텝 입력 (stepInputsEng 와 같은 순서·값)
  void applyInputsPx(int s, float dt, std::vector<float>& buf) {
    const float t = envTime(s, 0, dt);
    if (in.touched) {
      for (uint32_t d = 0; d < in.n; ++d) {
        J(in.link[d])->setDriveTarget(PxArticulationAxis::Enum(in.axis[d]), targetValue(in, d, t));
        if ((d % 2) == 1) J(in.link[d])->setDriveVelocity(PxArticulationAxis::Enum(in.axis[d]), targetVelValue(t));
      }
    }
    const uint32_t nd = e->dofs;
    buf.resize(64);
    if (s == 150 && nd) {
      px->copyInternalStateToCache(*cache, PxArticulationCacheFlag::ePOSITION);
      for (uint32_t i = 0; i < nd; ++i) cache->jointPosition[i] = cache->jointPosition[i] + 0.05f;
      px->applyCache(*cache, PxArticulationCacheFlag::ePOSITION);
    }
    if (s == 250 && nd) {
      for (uint32_t i = 0; i < nd; ++i) cache->jointVelocity[i] = 0.2f * sinE(float(i));
      px->applyCache(*cache, PxArticulationCacheFlag::eVELOCITY);
    }
    if (s == 300 && !(e->flags & A::AF_FIX_BASE)) {
      cache->rootLinkData->worldLinVel = PxVec3(0.3f, -0.1f, 0.5f);
      cache->rootLinkData->worldAngVel = PxVec3(0.2f, 0.4f, -0.3f);
      px->applyCache(*cache, PxArticulationCacheFlag::eROOT_VELOCITIES);
    }
    if (s == 350 && !(e->flags & A::AF_FIX_BASE)) {
      px->copyInternalStateToCache(*cache, PxArticulationCacheFlag::eROOT_TRANSFORM);
      PxTransform rt = cache->rootLinkData->transform;
      rt.p += PxVec3(0.0f, 0.0f, 1.0f);
      cache->rootLinkData->transform = rt;
      px->applyCache(*cache, PxArticulationCacheFlag::eROOT_TRANSFORM);
    }
    if (s >= 400 && s < 450 && nd) {
      for (uint32_t i = 0; i < nd; ++i) cache->jointForce[i] = 0.5f * sinE(float(i + uint32_t(s)));
      px->applyCache(*cache, PxArticulationCacheFlag::eFORCE);
    }
    if (s == 450 && nd) {
      for (uint32_t i = 0; i < nd; ++i) cache->jointForce[i] = 0.0f;
      px->applyCache(*cache, PxArticulationCacheFlag::eFORCE);
    }
  }
};

struct BuildOpts {
  int nArts = 12, nLinksMax = 9, seed = 1, posIt = 16, velIt = 1, spherical = 1, floatingOnly = 0, verbose = 0;
};

// 무작위 관절체 nArts 개를 만들어 장면에 넣는다 (PhysX + 엔진). 실패하면 false.
inline bool buildRandom(PxPhysics* phys, PxScene* scene, const BuildOpts& o, std::vector<Mirror>& arts) {
  std::mt19937 rng(o.seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { return PxVec3(s * U(rng), s * U(rng), s * U(rng)); };
  auto pick = [&](int n) { return int(P(rng) * float(n) * 0.9999f); };
  arts.clear();
  arts.resize(size_t(o.nArts));
  const PxArticulationAxis::Enum angAx[3] = {PxArticulationAxis::eTWIST, PxArticulationAxis::eSWING1, PxArticulationAxis::eSWING2};
  const PxArticulationAxis::Enum linAx[3] = {PxArticulationAxis::eX, PxArticulationAxis::eY, PxArticulationAxis::eZ};
  for (int ai = 0; ai < o.nArts; ++ai) {
    Mirror& m = arts[size_t(ai)];
    m.create(phys);
    const bool fixBase = o.floatingOnly ? false : (ai % 3) != 1;
    m.in.touched = (ai % 4) != 3;
    m.px->setSolverIterationCounts(PxU32(o.posIt), PxU32(o.velIt));
    A::artSetSolverIterationCounts(*m.e, uint32_t(o.posIt), uint32_t(o.velIt));
    m.px->setArticulationFlag(PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES, true);  // omni 는 늘 켠다 (UsdInterface.cpp:2252)
    A::artSetFlag(*m.e, A::AF_DRIVE_LIMITS_ARE_FORCES, true);
    if (fixBase) {
      m.px->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
      A::artSetFlag(*m.e, A::AF_FIX_BASE, true);
    }
    const int nl = 2 + pick(o.nLinksMax - 1);
    std::vector<PxTransform> poses;
    const PxTransform rootPose(PxVec3(float(ai) * 5.0f, 0.0f, 3.0f), rq());
    poses.push_back(rootPose);
    m.link(A::kNone, rootPose);
    for (int l = 1; l < nl; ++l) {
      const uint32_t parent = uint32_t(pick(l));
      const PxTransform pose = poses[parent] * PxTransform(rv(0.3f), rq());
      poses.push_back(pose);
      m.link(parent, pose);
    }
    std::vector<int> cmLater(size_t(nl), 0);
    for (int l = 0; l < nl; ++l) {
      m.mass(uint32_t(l), 0.2f + 5.0f * P(rng), PxVec3(0.002f + 0.1f * P(rng), 0.002f + 0.1f * P(rng), 0.002f + 0.1f * P(rng)));
      cmLater[size_t(l)] = P(rng) < 0.5f;
      if (!cmLater[size_t(l)]) m.cmass(uint32_t(l), PxTransform(rv(0.05f), rq()));
      if (P(rng) < 0.3f) m.linkDamping(uint32_t(l), 0.2f * P(rng), 0.5f * P(rng));
      if (P(rng) < 0.15f) m.linkMaxVel(uint32_t(l), 1.0f + 3.0f * P(rng), 1.0f + 3.0f * P(rng));
      if (P(rng) < 0.1f) m.noGravity(uint32_t(l));
    }
    std::vector<uint32_t> singleDof;
    std::vector<uint8_t> singleAx;
    for (int l = 1; l < nl; ++l) {
      const float r = P(rng);
      PxArticulationJointType::Enum t;
      if (r < 0.35f) t = PxArticulationJointType::eREVOLUTE_UNWRAPPED;
      else if (r < 0.5f) t = PxArticulationJointType::eREVOLUTE;
      else if (r < 0.75f) t = PxArticulationJointType::ePRISMATIC;
      else if (r < 0.87f || !o.spherical) t = PxArticulationJointType::eFIX;
      else t = PxArticulationJointType::eSPHERICAL;
      m.type(uint32_t(l), t);
      m.parentPose(uint32_t(l), PxTransform(rv(0.15f), rq()));
      m.childPose(uint32_t(l), PxTransform(rv(0.15f), rq()));
      if (P(rng) < 0.3f) m.frictionCoef(uint32_t(l), P(rng) < 0.5f ? 0.0f : 0.3f * P(rng));
      const bool sph = t == PxArticulationJointType::eSPHERICAL;
      auto setupAxis = [&](PxArticulationAxis::Enum ax, bool rotational, bool allowLimit, bool allowFree) {
        const bool limited = allowLimit && (!allowFree || P(rng) < 0.7f);
        m.motion(uint32_t(l), ax, limited ? PxArticulationMotion::eLIMITED : PxArticulationMotion::eFREE);
        if (limited) {
          const float s = rotational ? 1.2f : 0.2f;
          m.limit(uint32_t(l), ax, -s * (0.1f + P(rng)), s * (0.1f + P(rng)));
        }
        const float dr = P(rng);
        if (dr < 0.7f) {
          const auto dt = P(rng) < 0.75f ? PxArticulationDriveType::eFORCE : PxArticulationDriveType::eACCELERATION;
          if (dr < 0.1f)
            m.driveEnv(uint32_t(l), ax, 50.0f + 300.0f * P(rng), 1.0f + 20.0f * P(rng),
                       PxPerformanceEnvelope(5.0f + 50.0f * P(rng), 1.0f + 5.0f * P(rng), P(rng) < 0.5f ? 0.0f : 0.5f * P(rng),
                                             P(rng) < 0.5f ? 0.0f : 0.05f * P(rng)),
                       dt);
          else
            m.drive(uint32_t(l), ax, P(rng) < 0.2f ? 0.0f : 20.0f + 500.0f * P(rng), 0.5f + 30.0f * P(rng),
                    P(rng) < 0.3f ? 2.0f + 20.0f * P(rng) : PX_MAX_F32, dt);
          if (m.in.n < kMaxDriven) {
            m.in.link[m.in.n] = uint32_t(l);
            m.in.axis[m.in.n] = uint8_t(ax);
            m.in.spherical[m.in.n] = sph ? 1 : 0;
            m.in.n++;
          }
        }
        if (P(rng) < 0.3f) m.armature(uint32_t(l), ax, 0.02f * P(rng));
        if (P(rng) < 0.3f) {  // 정적 >= 동적 (NpArticulationJointReducedCoordinate.cpp:209 검사)
          const float dyn = 0.2f * P(rng);
          const float st = P(rng) < 0.3f ? dyn : dyn + 0.3f * P(rng);
          m.frictionParams(uint32_t(l), ax, st, dyn, P(rng) < 0.5f ? 0.0f : 0.1f * P(rng));
        }
        if (P(rng) < 0.2f) m.maxJointVel(uint32_t(l), ax, 0.5f + 3.0f * P(rng));
        if (P(rng) < 0.3f) {
          const float jp = (rotational ? 0.8f : 0.1f) * U(rng);
          m.jointPos(uint32_t(l), ax, limited ? jp * 0.5f : jp);
        }
        if (P(rng) < 0.3f) m.jointVel(uint32_t(l), ax, (rotational ? 1.0f : 0.2f) * U(rng));
      };
      if (t == PxArticulationJointType::eREVOLUTE_UNWRAPPED || t == PxArticulationJointType::eREVOLUTE) {
        const PxArticulationAxis::Enum ax = P(rng) < 0.8f ? PxArticulationAxis::eTWIST : angAx[pick(3)];
        setupAxis(ax, true, t == PxArticulationJointType::eREVOLUTE_UNWRAPPED, true);
        singleDof.push_back(uint32_t(l));
        singleAx.push_back(uint8_t(ax));
      } else if (t == PxArticulationJointType::ePRISMATIC) {
        const PxArticulationAxis::Enum ax = P(rng) < 0.8f ? PxArticulationAxis::eX : linAx[pick(3)];
        setupAxis(ax, false, true, true);
        singleDof.push_back(uint32_t(l));
        singleAx.push_back(uint8_t(ax));
      } else if (t == PxArticulationJointType::eSPHERICAL) {
        const int nd = 2 + pick(2);
        const int skip = nd == 2 ? pick(3) : -1;
        for (int k = 0; k < 3; ++k)
          if (k != skip) setupAxis(angAx[k], true, true, true);
      }
      if (cmLater[size_t(l)]) m.cmass(uint32_t(l), PxTransform(rv(0.05f), rq()));
    }
    if (cmLater[0]) m.cmass(0, PxTransform(rv(0.05f), rq()));
    if (singleDof.size() >= 2 && P(rng) < 0.7f) {
      const int nm = 1 + (singleDof.size() >= 4 && P(rng) < 0.5f ? 1 : 0);
      for (int k = 0; k < nm; ++k) {
        const size_t ia = size_t(2 * k), ib = size_t(2 * k + 1);
        const float gear = (P(rng) < 0.5f ? -1.0f : 1.0f) * (0.3f + P(rng));
        const float off = 0.1f * U(rng);
        const float nf = P(rng) < 0.5f ? 0.0f : 20.0f + 50.0f * P(rng);
        const float drt = P(rng) < 0.5f ? 0.0f : 0.5f + P(rng);
        m.px->createMimicJoint(*m.J(singleDof[ia]), PxArticulationAxis::Enum(singleAx[ia]), *m.J(singleDof[ib]),
                               PxArticulationAxis::Enum(singleAx[ib]), gear, off, nf, drt);
        A::createMimicJoint(*m.e, singleDof[ia], singleAx[ia], singleDof[ib], singleAx[ib], gear, off, nf, drt);
      }
    }
    scene->addArticulation(*m.px);
    if (!A::addToScene(*m.e)) return false;
    m.cache = m.px->createCache();
  }
  return true;
}

}  // namespace artest
#endif
