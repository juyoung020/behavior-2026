// 층 1 시험: 손으로 짠 관절체(core/articulation) = PhysX 5.6.1 (정답지) 비트 동일?
// 무작위 관절체(트리 모양·관절 종류·축·틀·질량·드라이브·한계·마찰·아머처·흉내 관절·고정/떠 있는 바닥)를 PhysX 와 우리 엔진에
// 같은 API 호출로 만들고(random_art.h), 매 스텝 같은 드라이브 목표·applyCache 를 넣은 뒤 simulate 마다
// 링크 자세·속도, 관절 위치·속도, 깸 카운터, 잠 여부를 비트 비교한다.
// 모양(shape)은 없다 -> 접촉 없음 -> PhysX 는 "접촉 묶음 0개" 관절체 풀이 경로(DyTGSDynamics.cpp:2531)를 탄다.
//   test_articulation [--arts N] [--links N] [--steps N] [--seed S] [--pos P] [--vel V] [--extevery 0|1] [--contactlast 0|1]
//                     [--spherical 0|1] [--floating 0|1] [--threads T] [--verbose 0|1]
#include <xmmintrin.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include "random_art.h"

using namespace physx;
using namespace artest;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ, 예외 가림)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

int main(int argc, char** argv) {
  BuildOpts o;
  int steps = 600, extEvery = 0, contactLast = 0, threads = 0;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) o.nArts = atoi(argv[++i]);
    else if (arg("--links")) o.nLinksMax = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--seed")) o.seed = atoi(argv[++i]);
    else if (arg("--pos")) o.posIt = atoi(argv[++i]);
    else if (arg("--vel")) o.velIt = atoi(argv[++i]);
    else if (arg("--extevery")) extEvery = atoi(argv[++i]);
    else if (arg("--contactlast")) contactLast = atoi(argv[++i]);
    else if (arg("--spherical")) o.spherical = atoi(argv[++i]);
    else if (arg("--floating")) o.floatingOnly = atoi(argv[++i]);
    else if (arg("--threads")) threads = atoi(argv[++i]);
    else if (arg("--verbose")) o.verbose = atoi(argv[++i]);
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

  std::vector<Mirror> arts;
  if (!buildRandom(phys, scene, o, arts)) {
    fprintf(stderr, "addToScene 실패\n");
    return 1;
  }
  if (o.verbose)
    for (size_t ai = 0; ai < arts.size(); ++ai)
      printf("art %zu: 링크 %u, dof %u, 고정 %d, 흉내 %u, 목표 %u, 매스텝입력 %d\n", ai, arts[ai].e->nLinks, arts[ai].e->dofs,
             int(arts[ai].e->flags & A::AF_FIX_BASE), arts[ai].e->nMimic, arts[ai].in.n, int(arts[ai].in.touched));

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
  std::vector<float> buf;
  for (int s = 1; s <= steps; ++s) {
    for (auto& m : arts) {
      m.applyInputsPx(s, dt, buf);
      stepInputsEng(*m.e, m.in, s, 0, dt);
    }
    scene->simulate(dt);
    scene->fetchResults(true);
    {
      FtzScope f;
      for (auto& m : arts) A::stepAlone(*m.e, sp);
    }
    for (size_t ai = 0; ai < arts.size(); ++ai) {
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
            printf("[다름] step %d art %zu link %d %s\n  PhysX:", s, ai, link, names[k]);
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
         o.nArts, steps, o.posIt, o.velIt, extEvery, contactLast, o.spherical, o.seed, total, nSleep);
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
