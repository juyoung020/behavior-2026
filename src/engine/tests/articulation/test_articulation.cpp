// 층 1 시험: 손으로 짠 관절체(core/articulation) = PhysX 5.6.1 (정답지) 비트 동일?
// 무작위 관절체(트리 모양·관절 종류·축·틀·질량·드라이브·한계·마찰·아머처·흉내 관절·고정/떠 있는 바닥)를 PhysX 와 우리 엔진에
// 같은 API 호출로 만들고, 매 스텝 같은 드라이브 목표·applyCache 를 넣은 뒤 simulate 마다 링크 자세·속도, 관절 위치·속도, 깸 카운터를 비트 비교.
// 모양(shape)은 없다 -> 접촉 없음 -> PhysX 는 "접촉 묶음 0개" 관절체 풀이 경로(DyTGSDynamics.cpp:2531)를 탄다.
//   test_articulation [--arts N] [--links N] [--steps N] [--seed S] [--pos P] [--vel V] [--extevery 0|1] [--contactlast 0|1]
//                     [--spherical 0|1] [--threads T] [--verbose 0|1]
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "core/articulation/art_step.h"

using namespace physx;
namespace A = eng::art;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ, 예외 가림)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
static eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

// 한 관절체를 양쪽에 똑같이 만드는 거울
struct Mirror {
  PxArticulationReducedCoordinate* px = nullptr;
  std::vector<PxArticulationLink*> pl;  // 생성 순서
  std::unique_ptr<A::Articulation> e;
  A::SceneScale sc;
  std::vector<std::pair<uint32_t, uint8_t>> driven;  // (링크 생성 번호, 축) 드라이브가 있는 dof
  std::vector<uint8_t> jtype;
  bool touched = true;  // 매 스텝 목표를 넣는지
  PxArticulationCache* cache = nullptr;

  void create(PxPhysics* phys) {
    px = phys->createArticulationReducedCoordinate();
    e.reset(new A::Articulation);
    A::createArticulation(*e, sc);
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
  void driveEnv(uint32_t l, PxArticulationAxis::Enum ax, float k, float d, const PxPerformanceEnvelope& env, PxArticulationDriveType::Enum t) {
    J(l)->setDriveParams(ax, PxArticulationDrive(k, d, env, t));
    A::jointSetDrive(*e, l, uint8_t(ax),
                     A::makeDriveEnvelope(k, d, A::Envelope{env.maxEffort, env.maxActuatorVelocity, env.velocityDependentResistance, env.speedEffortGradient},
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
  void target(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setDriveTarget(ax, v);
    A::jointSetDriveTarget(*e, l, uint8_t(ax), v);
  }
  void targetVel(uint32_t l, PxArticulationAxis::Enum ax, float v) {
    J(l)->setDriveVelocity(ax, v);
    A::jointSetDriveVelocity(*e, l, uint8_t(ax), v);
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
};

int main(int argc, char** argv) {
  int nArts = 12, nLinksMax = 9, steps = 600, seed = 1, posIt = 16, velIt = 1, extEvery = 0, contactLast = 0, spherical = 1, threads = 0,
      verbose = 0, floatingOnly = 0;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) nArts = atoi(argv[++i]);
    else if (arg("--links")) nLinksMax = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--seed")) seed = atoi(argv[++i]);
    else if (arg("--pos")) posIt = atoi(argv[++i]);
    else if (arg("--vel")) velIt = atoi(argv[++i]);
    else if (arg("--extevery")) extEvery = atoi(argv[++i]);
    else if (arg("--contactlast")) contactLast = atoi(argv[++i]);
    else if (arg("--spherical")) spherical = atoi(argv[++i]);
    else if (arg("--threads")) threads = atoi(argv[++i]);
    else if (arg("--verbose")) verbose = atoi(argv[++i]);
    else if (arg("--floating")) floatingOnly = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(PxU32(threads));
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS | PxSceneFlag::eENABLE_STABILIZATION;
  if (extEvery) sd.flags |= PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
  if (contactLast) sd.flags |= PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST;
  PxScene* scene = phys->createScene(sd);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { return PxVec3(s * U(rng), s * U(rng), s * U(rng)); };
  auto pick = [&](int n) { return int(P(rng) * float(n) * 0.9999f); };

  std::vector<Mirror> arts(nArts);
  const PxArticulationAxis::Enum angAx[3] = {PxArticulationAxis::eTWIST, PxArticulationAxis::eSWING1, PxArticulationAxis::eSWING2};
  const PxArticulationAxis::Enum linAx[3] = {PxArticulationAxis::eX, PxArticulationAxis::eY, PxArticulationAxis::eZ};
  for (int ai = 0; ai < nArts; ++ai) {
    Mirror& m = arts[ai];
    m.create(phys);
    const bool fixBase = floatingOnly ? false : (ai % 3) != 1;
    m.touched = (ai % 4) != 3;
    m.px->setSolverIterationCounts(PxU32(posIt), PxU32(velIt));
    A::artSetSolverIterationCounts(*m.e, uint32_t(posIt), uint32_t(velIt));
    m.px->setArticulationFlag(PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES, true);  // omni 는 늘 켠다 (UsdInterface.cpp:2252)
    A::artSetFlag(*m.e, A::AF_DRIVE_LIMITS_ARE_FORCES, true);
    if (fixBase) {
      m.px->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
      A::artSetFlag(*m.e, A::AF_FIX_BASE, true);
    }
    const int nl = 2 + pick(nLinksMax - 1);
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
    // 질량 (일부는 관절 틀 전에, 일부는 뒤에 질량중심 자세를 바꿈 — 틀 이동 공식 시험)
    std::vector<int> cmLater(nl, 0);
    for (int l = 0; l < nl; ++l) {
      m.mass(uint32_t(l), 0.2f + 5.0f * P(rng), PxVec3(0.002f + 0.1f * P(rng), 0.002f + 0.1f * P(rng), 0.002f + 0.1f * P(rng)));
      cmLater[l] = P(rng) < 0.5f;
      if (!cmLater[l]) m.cmass(uint32_t(l), PxTransform(rv(0.05f), rq()));
      if (P(rng) < 0.3f) m.linkDamping(uint32_t(l), 0.2f * P(rng), 0.5f * P(rng));
      if (P(rng) < 0.15f) m.linkMaxVel(uint32_t(l), 1.0f + 3.0f * P(rng), 1.0f + 3.0f * P(rng));
      if (P(rng) < 0.1f) m.noGravity(uint32_t(l));
    }
    std::vector<uint32_t> singleDof;  // 흉내 관절 후보 (링크, 축)
    std::vector<uint8_t> singleAx;
    for (int l = 1; l < nl; ++l) {
      const float r = P(rng);
      PxArticulationJointType::Enum t;
      if (r < 0.35f) t = PxArticulationJointType::eREVOLUTE_UNWRAPPED;
      else if (r < 0.5f) t = PxArticulationJointType::eREVOLUTE;
      else if (r < 0.75f) t = PxArticulationJointType::ePRISMATIC;
      else if (r < 0.87f || !spherical) t = PxArticulationJointType::eFIX;
      else t = PxArticulationJointType::eSPHERICAL;
      m.type(uint32_t(l), t);
      m.parentPose(uint32_t(l), PxTransform(rv(0.15f), rq()));
      m.childPose(uint32_t(l), PxTransform(rv(0.15f), rq()));
      if (P(rng) < 0.3f) m.frictionCoef(uint32_t(l), P(rng) < 0.5f ? 0.0f : 0.3f * P(rng));
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
          m.driven.push_back({uint32_t(l), uint8_t(ax)});
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
      if (cmLater[l]) m.cmass(uint32_t(l), PxTransform(rv(0.05f), rq()));
    }
    if (cmLater[0]) m.cmass(0, PxTransform(rv(0.05f), rq()));
    // 흉내 관절 (링크 둘 이상일 때 한두 개)
    if (singleDof.size() >= 2 && P(rng) < 0.7f) {
      const int nm = 1 + (singleDof.size() >= 4 && P(rng) < 0.5f ? 1 : 0);
      for (int k = 0; k < nm; ++k) {
        const size_t ia = size_t(2 * k), ib = size_t(2 * k + 1);
        const float gear = (P(rng) < 0.5f ? -1.0f : 1.0f) * (0.3f + P(rng));
        const float off = 0.1f * U(rng);
        const float nf = P(rng) < 0.5f ? 0.0f : 20.0f + 50.0f * P(rng);
        const float dr = P(rng) < 0.5f ? 0.0f : 0.5f + P(rng);
        m.px->createMimicJoint(*m.J(singleDof[ia]), PxArticulationAxis::Enum(singleAx[ia]), *m.J(singleDof[ib]),
                               PxArticulationAxis::Enum(singleAx[ib]), gear, off, nf, dr);
        A::createMimicJoint(*m.e, singleDof[ia], singleAx[ia], singleDof[ib], singleAx[ib], gear, off, nf, dr);
      }
    }
    scene->addArticulation(*m.px);
    if (!A::addToScene(*m.e)) {
      fprintf(stderr, "addToScene 실패 art %d err=%u\n", ai, m.e->err);
      return 1;
    }
    m.cache = m.px->createCache();
    if (verbose) printf("art %d: 링크 %d, dof %u, 고정 %d, 흉내 %u, 목표 %zu\n", ai, nl, m.e->dofs, int(fixBase), m.e->nMimic, m.driven.size());
  }

  const float dt = 1.0f / 120.0f;
  A::StepParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.dt = dt;
  sp.lengthScale = 1.0f;
  sp.externalForcesEveryTgsIteration = extEvery != 0;
  sp.solveArticulationContactLast = contactLast != 0;
  enum { K_POSE, K_LIN, K_ANG, K_JPOS, K_JVEL, K_WAKE, K_SLEEP, K_N };
  const char* names[K_N] = {"링크 자세", "링크 선속도", "링크 각속도", "관절 위치", "관절 속도", "깸 카운터", "잠 여부"};
  uint64_t cmp[K_N] = {0}, bad[K_N] = {0}, nanBoth[K_N] = {0};
  int64_t first[K_N];
  for (int k = 0; k < K_N; ++k) first[k] = -1;
  double maxd[K_N] = {0};
  int shown = 0;
  std::vector<float> jp(64), jv(64), jf(64);
  for (int s = 1; s <= steps; ++s) {
    const float t = float(s) * dt;
    for (int ai = 0; ai < nArts; ++ai) {
      Mirror& m = arts[ai];
      if (m.touched) {
        for (size_t d = 0; d < m.driven.size(); ++d) {
          const uint32_t l = m.driven[d].first;
          const auto ax = PxArticulationAxis::Enum(m.driven[d].second);
          const bool rot = ax < PxArticulationAxis::eX;
          const float amp = rot ? (m.jtype[l] == PxArticulationJointType::eSPHERICAL ? 0.6f : 1.0f) : 0.15f;
          m.target(l, ax, amp * sinf(0.7f * t * float(1 + d)));
          if ((d % 2) == 1) m.targetVel(l, ax, 0.3f * cosf(1.1f * t));
        }
      }
      // applyCache (텐서 API set_* 와 같은 호출)
      const uint32_t nd = m.e->dofs;
      if (s == 150 && nd) {  // 관절 위치 순간이동
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION);
        for (uint32_t i = 0; i < nd; ++i) jp[i] = m.cache->jointPosition[i] + 0.05f;
        memcpy(m.cache->jointPosition, jp.data(), nd * 4);
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::ePOSITION);
        A::CacheIn c{};
        c.jointPosition = jp.data();
        A::applyCache(*m.e, c, A::CF_POSITION);
      }
      if (s == 250 && nd) {  // 관절 속도
        for (uint32_t i = 0; i < nd; ++i) jv[i] = 0.2f * sinf(float(i));
        memcpy(m.cache->jointVelocity, jv.data(), nd * 4);
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::eVELOCITY);
        A::CacheIn c{};
        c.jointVelocity = jv.data();
        A::applyCache(*m.e, c, A::CF_VELOCITY);
      }
      if (s == 300 && !(m.e->flags & A::AF_FIX_BASE)) {  // 뿌리 속도
        m.cache->rootLinkData->worldLinVel = PxVec3(0.3f, -0.1f, 0.5f);
        m.cache->rootLinkData->worldAngVel = PxVec3(0.2f, 0.4f, -0.3f);
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::eROOT_VELOCITIES);
        A::CacheIn c{};
        c.rootLinVel = eng::V3{0.3f, -0.1f, 0.5f};
        c.rootAngVel = eng::V3{0.2f, 0.4f, -0.3f};
        A::applyCache(*m.e, c, A::CF_ROOT_VELOCITIES);
      }
      if (s == 350 && !(m.e->flags & A::AF_FIX_BASE)) {  // 뿌리 순간이동
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::eROOT_TRANSFORM);
        PxTransform rt = m.cache->rootLinkData->transform;
        rt.p += PxVec3(0.0f, 0.0f, 1.0f);
        m.cache->rootLinkData->transform = rt;
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::eROOT_TRANSFORM);
        A::CacheIn c{};
        c.rootTransform = toE(rt);
        A::applyCache(*m.e, c, A::CF_ROOT_TRANSFORM);
      }
      if (s >= 400 && s < 450 && nd) {  // 관절 힘
        for (uint32_t i = 0; i < nd; ++i) jf[i] = 0.5f * sinf(float(i + s));
        memcpy(m.cache->jointForce, jf.data(), nd * 4);
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::eFORCE);
        A::CacheIn c{};
        c.jointForce = jf.data();
        A::applyCache(*m.e, c, A::CF_FORCE);
      }
      if (s == 450 && nd) {  // 힘 끔
        for (uint32_t i = 0; i < nd; ++i) jf[i] = 0.0f;
        memcpy(m.cache->jointForce, jf.data(), nd * 4);
        m.px->applyCache(*m.cache, PxArticulationCacheFlag::eFORCE);
        A::CacheIn c{};
        c.jointForce = jf.data();
        A::applyCache(*m.e, c, A::CF_FORCE);
      }
    }
    scene->simulate(dt);
    scene->fetchResults(true);
    {
      FtzScope f;
      for (int ai = 0; ai < nArts; ++ai) A::stepAlone(*arts[ai].e, sp);
    }
    for (int ai = 0; ai < nArts; ++ai) {
      Mirror& m = arts[ai];
      A::Articulation& e = *m.e;
      auto check = [&](int k, const float* x, const float* y, int n, int link) {
        cmp[k]++;
        // 양쪽이 같은 칸에서 NaN (발산한 무작위 관절체) 이면 NaN 부호/페이로드는 따지지 않고 따로 센다
        bool same = true, anyNan = false;
        for (int j = 0; j < n; ++j) {
          const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
          if (nx || ny) {
            anyNan = true;
            if (nx != ny) same = false;
          } else if (memcmp(&x[j], &y[j], 4)) {
            same = false;
          }
        }
        if (same && anyNan) nanBoth[k]++;
        if (!same) {
          bad[k]++;
          if (first[k] < 0) first[k] = s;
          for (int j = 0; j < n; ++j) maxd[k] = std::fmax(maxd[k], std::fabs(double(x[j]) - double(y[j])));
          if (shown < 12) {
            shown++;
            printf("[다름] step %d art %d link %d %s\n  PhysX:", s, ai, link, names[k]);
            for (int j = 0; j < n; ++j) printf(" %.9g", x[j]);
            printf("\n  엔진 :");
            for (int j = 0; j < n; ++j) printf(" %.9g", y[j]);
            printf("\n");
          }
        }
      };
      for (uint32_t l = 0; l < m.pl.size(); ++l) {
        const PxTransform tp = m.pl[l]->getGlobalPose();
        const eng::Tf te = A::linkGlobalPose(e, l);
        check(K_POSE, &tp.q.x, &te.q.x, 7, int(l));
        const PxVec3 lv = m.pl[l]->getLinearVelocity(), av = m.pl[l]->getAngularVelocity();
        const A::LinkBody& b = e.bodies[e.ll[l]];
        check(K_LIN, &lv.x, &b.linVel.x, 3, int(l));
        check(K_ANG, &av.x, &b.angVel.x, 3, int(l));
      }
      const uint32_t nd = e.dofs;
      if (nd) {
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        check(K_JPOS, m.cache->jointPosition, e.jointPosition, int(nd), -1);
        check(K_JVEL, m.cache->jointVelocity, e.jointVelocity, int(nd), -1);
      }
      const float wp = m.px->getWakeCounter();
      check(K_WAKE, &wp, &e.wakeCounter, 1, -1);
      const float sp1 = m.px->isSleeping() ? 1.0f : 0.0f, se = e.awake ? 0.0f : 1.0f;
      check(K_SLEEP, &sp1, &se, 1, -1);
    }
  }
  int nSleep = 0;
  for (auto& m : arts) nSleep += m.px->isSleeping() ? 1 : 0;
  uint64_t total = 0;
  for (int k = 0; k < K_N; ++k) total += cmp[k];
  printf("\n관절체 %d 개 x %d 스텝 (위치반복 %d, 속도반복 %d, 매반복외력 %d, 접촉마지막 %d, 구면 %d, seed %d): 비교 %" PRIu64 ", 끝에 잠든 관절체 %d\n",
         nArts, steps, posIt, velIt, extEvery, contactLast, spherical, seed, total, nSleep);
  bool ok = true;
  for (int k = 0; k < K_N; ++k) {
    printf("  %-12s 비교 %8" PRIu64 "  비트 다름 %8" PRIu64 "  첫 다름 스텝 %5" PRId64 "  최대|차| %.3e  양쪽 NaN %" PRIu64 "\n", names[k], cmp[k],
           bad[k], first[k], maxd[k], nanBoth[k]);
    ok = ok && !bad[k];
  }
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  for (auto& m : arts) m.cache->release();
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
