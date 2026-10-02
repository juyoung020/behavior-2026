// 층 1 시험: 동적 강체 API 부수효과 (setGlobalPose·속도 설정·addForce/addTorque(4 방식)·setForceAndTorque·clear·
// wakeUp·putToSleep·setWakeCounter) = PhysX 5.6.1 비트 동일? 호출 직후 읽은 상태와, 그 뒤 simulate 결과를 둘 다 비교한다.
// simulate 쪽 자유 강체 적분은 solver 모듈 core/solver/free_body.h(stepFree) 를 쓰고, 힘 적용(beforeSolver)은 joints 의 applyForces.
// 섬은 몸체 하나짜리(접촉·조인트 없음, 50 m 간격): 활성 몸체만 적분, putToSleep 은 바로 비활성, 깸 카운터 0 이면 다음 simulate 에서 비활성.
//   test_rigid_api [--bodies N] [--steps N] [--seed S] [--ops P]
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "core/joints/rigid_api.h"
#include "core/solver/free_body.h"

using namespace physx;
namespace J = eng::jnt;

static PxDefaultAllocator gAlloc;
struct QuietErr : PxErrorCallback {
  int n = 0;
  void reportError(PxErrorCode::Enum, const char*, const char*, int) override { n++; }
};
static QuietErr gErr;

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
static eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

int main(int argc, char** argv) {
  int nb = 200, steps = 600, seed = 1, trace = -1;
  float opsPerStep = 3.0f;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ops") && i + 1 < argc) opsPerStep = float(atof(argv[++i]));
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(0);
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM;
  PxScene* scene = phys->createScene(sd);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.5f, 0.0f);
  const float resetValue = scene->getWakeCounterResetValue();

  eng::SceneParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.speedScale = 10.0f;
  sp.stabilization = false;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { const float x = s * U(rng), y = s * U(rng), z = s * U(rng); return PxVec3(x, y, z); };

  std::vector<PxRigidDynamic*> px(nb);
  std::vector<eng::Body> eb(nb);
  std::vector<J::BodySimState> es(nb);
  // 몸체 하나짜리 섬 모형 (PxsIslandSim.cpp:400 activateNode, :436 deactivateNode, :475 putNodeToSleep 규칙 그대로)
  //   nodeActive = 섬 풀이에 들어가는가, activating = 다음 섬 갱신 때 켜질 예정, readyForSleep = 다음 섬 갱신 때 꺼질 수 있음
  //   BodySim 의 활성(isSleeping 의 반대)은 es[i].active — API 가 바로 바꾼다. 둘이 다를 수 있다(깨운 뒤 같은 틈에 깸 카운터 0).
  std::vector<uint8_t> readyForSleep(nb, 0), nodeActive(nb, 1), activating(nb, 0);
  for (int i = 0; i < nb; ++i) {
    const PxTransform pose(PxVec3(float(i % 20) * 50.0f, float(i / 20) * 50.0f, 1000.0f), rq());
    const PxTransform cm(rv(0.05f), rq());
    const float mass = 0.1f + 20.0f * P(rng);
    const PxVec3 inertia(0.01f + P(rng), 0.01f + P(rng), 0.01f + P(rng));
    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    PxRigidActorExt::createExclusiveShape(*a, PxSphereGeometry(0.05f), *mat);
    a->setCMassLocalPose(cm);
    a->setMass(mass);
    a->setMassSpaceInertiaTensor(inertia);
    scene->addActor(*a);
    px[i] = a;
    eng::Body& b = eb[i];
    b = eng::createRigidDynamic(toE(pose), sp);
    eng::setCMassLocalPose(b, toE(cm));
    eng::setMass(b, mass);
    eng::setMassSpaceInertiaTensor(b, toE(inertia));
    es[i] = J::makeBodySimState();
  }

  const char* names[5] = {"자세", "선속도", "각속도", "깸 카운터", "잠 여부"};
  uint64_t cmpApi = 0, badApi = 0, cmpSim = 0, badSim = 0;
  int64_t firstApi = -1, firstSim = -1;
  int shown = 0;
  uint64_t opCount[16] = {0};
  auto compare = [&](int i, int step, bool afterApi, const char* what) {
    const PxTransform tp = px[i]->getGlobalPose();
    const eng::Tf te = eng::getGlobalPose(eb[i]);
    const PxVec3 lp = px[i]->getLinearVelocity(), ap = px[i]->getAngularVelocity();
    const float wp = px[i]->getWakeCounter();
    const uint32_t sPx = px[i]->isSleeping() ? 1u : 0u, sEn = es[i].active ? 0u : 1u;
    const void* pa[5] = {&tp, &lp, &ap, &wp, &sPx};
    const void* pe[5] = {&te, &eb[i].linVel, &eb[i].angVel, &eb[i].wakeCounter, &sEn};
    const int nf[5] = {7, 3, 3, 1, 1};
    bool ok = true;
    for (int k = 0; k < 5; ++k) {
      if (memcmp(pa[k], pe[k], 4 * nf[k])) {
        ok = false;
        if (shown < 8) {
          shown++;
          printf("[다름] step %d 몸체 %d %s (%s)\n  PhysX:", step, i, names[k], what);
          for (int j = 0; j < nf[k]; ++j) printf(" %.9g", static_cast<const float*>(pa[k])[j]);
          printf("\n  엔진 :");
          for (int j = 0; j < nf[k]; ++j) printf(" %.9g", static_cast<const float*>(pe[k])[j]);
          printf("\n");
        }
      }
    }
    if (afterApi) { cmpApi++; if (!ok) { badApi++; if (firstApi < 0) firstApi = step; } }
    else { cmpSim++; if (!ok) { badSim++; if (firstSim < 0) firstSim = step; } }
  };
  auto applyReq = [&](int i, uint32_t req) {  // 몸체 하나짜리 섬 관리자
    if (req == J::REQ_ACTIVATE) {  // activateNode
      if (!(nodeActive[i] || activating[i])) activating[i] = 1;
      readyForSleep[i] = 0;
    } else if (req == J::REQ_DEACTIVATE || req == J::REQ_SLEEP_NOW) {  // deactivateNode / putNodeToSleep
      if (activating[i]) activating[i] = 0;
      readyForSleep[i] = 1;
    }
  };
  const char* opNames[12] = {"setGlobalPose", "setLinearVelocity", "setAngularVelocity", "addForce", "addTorque", "setForceAndTorque",
                             "clearForce", "clearTorque", "wakeUp", "putToSleep", "setWakeCounter", "잠든 몸체에 힘"};

  const float dt = 1.0f / 120.0f;
  std::uniform_int_distribution<int> pickBody(0, nb - 1);
  for (int s = 1; s <= steps; ++s) {
    // ---- API 호출 (simulate 사이)
    const int nops = int(opsPerStep * 2.0f * P(rng) + 0.5f);
    for (int o = 0; o < nops; ++o) {
      const int i = pickBody(rng);
      const int op = int(rng() % 11u);
      const bool autowake = (rng() & 3u) != 0;
      const bool zero = (rng() % 6u) == 0;
      const PxForceMode::Enum mode = PxForceMode::Enum(rng() % 4u);
      opCount[op]++;
      uint32_t req = J::REQ_NONE;
      switch (op) {
        case 0: {
          const PxTransform cur = px[i]->getGlobalPose();
          const PxTransform np(cur.p + rv(0.5f), rq());
          px[i]->setGlobalPose(np, autowake);
          req = J::setGlobalPose(eb[i], es[i], toE(np), autowake, true, resetValue);
          break;
        }
        case 1: {
          const PxVec3 v = zero ? PxVec3(0.0f) : rv(3.0f);
          px[i]->setLinearVelocity(v, autowake);
          req = J::setLinearVelocity(eb[i], es[i], toE(v), autowake, true, resetValue);
          break;
        }
        case 2: {
          const PxVec3 v = zero ? PxVec3(0.0f) : rv(5.0f);
          px[i]->setAngularVelocity(v, autowake);
          req = J::setAngularVelocity(eb[i], es[i], toE(v), autowake, true, resetValue);
          break;
        }
        case 3: {
          const PxVec3 f = zero ? PxVec3(0.0f) : rv(20.0f);
          px[i]->addForce(f, mode, autowake);
          req = J::addForce(eb[i], es[i], toE(f), uint32_t(mode), autowake, resetValue);
          break;
        }
        case 4: {
          const PxVec3 t = zero ? PxVec3(0.0f) : rv(2.0f);
          px[i]->addTorque(t, mode, autowake);
          req = J::addTorque(eb[i], es[i], toE(t), uint32_t(mode), autowake, resetValue);
          break;
        }
        case 5: {
          const PxVec3 f = zero ? PxVec3(0.0f) : rv(20.0f), t = rv(2.0f);
          px[i]->setForceAndTorque(f, t, mode);
          req = J::setForceAndTorque(eb[i], es[i], toE(f), toE(t), uint32_t(mode), resetValue);
          break;
        }
        case 6: px[i]->clearForce(mode); J::clearSpatialForce(es[i], uint32_t(mode), true, false); break;
        case 7: px[i]->clearTorque(mode); J::clearSpatialForce(es[i], uint32_t(mode), false, true); break;
        case 8: px[i]->wakeUp(); req = J::wakeUp(eb[i], es[i], resetValue); break;
        case 9: px[i]->putToSleep(); req = J::putToSleep(eb[i], es[i]); break;
        case 10: {
          const float w = zero ? 0.0f : 0.8f * P(rng);
          px[i]->setWakeCounter(w);
          req = J::setWakeCounter(eb[i], es[i], w);
          break;
        }
      }
      applyReq(i, req);
      if (i == trace)
        printf("  [추적] step %d %s autowake=%d zero=%d mode=%d -> req %u, PhysX 잠=%d wc=%.9g v=%.3g | 엔진 활성=%d 잠준비=%d wc=%.9g\n", s, opNames[op],
               autowake, zero, int(mode), req, int(px[i]->isSleeping()), px[i]->getWakeCounter(), px[i]->getLinearVelocity().magnitude(),
               int(es[i].active), int(readyForSleep[i]), eb[i].wakeCounter);
      compare(i, s, true, opNames[op]);
    }
    // ---- simulate
    scene->simulate(dt);
    scene->fetchResults(true);
    {
      FtzScope ftz;
      for (int i = 0; i < nb; ++i) {
        // 섬 갱신: 켜질 노드는 켜고(BodySim 도 활성), 잠 준비된 활성 노드는 끈다(BodySim::deactivate: 속도 0)
        if (activating[i]) { activating[i] = 0; nodeActive[i] = 1; es[i].active = 1; }
        if (nodeActive[i] && readyForSleep[i]) { nodeActive[i] = 0; readyForSleep[i] = 0; J::onDeactivate(eb[i], es[i]); }
        if (!nodeActive[i]) continue;
        J::applyForces(eb[i], es[i], dt);   // beforeSolver (ScPipeline.cpp:1836)
        eng::stepFree(eb[i], sp, dt);       // 풀이·적분·잠 판정 (solver 모듈)
        if (eb[i].wakeCounter == 0.0f) readyForSleep[i] = 1;  // eDEACTIVATE_THIS_FRAME -> notifyReadyForSleeping
      }
    }
    if (trace >= 0)
      printf("  [추적] step %d simulate 뒤 PhysX 잠=%d wc=%.9g | 엔진 활성=%d 잠준비=%d wc=%.9g\n", s, int(px[trace]->isSleeping()),
             px[trace]->getWakeCounter(), int(es[trace].active), int(readyForSleep[trace]), eb[trace].wakeCounter);
    for (int i = 0; i < nb; ++i) compare(i, s, false, "simulate 뒤");
  }
  printf("\n강체 API 부수효과: 몸체 %d 개 x %d 스텝 (씨앗 %d)\n  호출 수:", nb, steps, seed);
  for (int k = 0; k < 11; ++k) printf(" %s %" PRIu64, opNames[k], opCount[k]);
  printf("\n  호출 직후 상태  비교 %8" PRIu64 "  비트 다름 %" PRIu64 "  첫 다름 스텝 %" PRId64 "\n", cmpApi, badApi, firstApi);
  printf("  simulate 뒤 상태 비교 %8" PRIu64 "  비트 다름 %" PRIu64 "  첫 다름 스텝 %" PRId64 "\n", cmpSim, badSim, firstSim);
  const bool ok = !badApi && !badSim;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
