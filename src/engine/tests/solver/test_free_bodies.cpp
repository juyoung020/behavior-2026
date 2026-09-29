// 층 1 시험 1: 손으로 짠 엔진(core/rigid.h) 의 자유 강체 = PhysX 5.6.1 (정답지) 비트 동일?
// 같은 API 입력(자세·질량중심 자세·질량·관성·감쇠·속도·잠금·자이로·중력 끔)을 양쪽에 넣고, 매 simulate 뒤
// 행위자 자세·선속도·각속도·깸 카운터를 비트 비교한다. 물체끼리 닿지 않게 멀리 떨어뜨린다(접촉 없음 경로).
//   test_free_bodies [--bodies N] [--steps N] [--seed S] [--stab 0|1]
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "core/solver/free_body.h"

using namespace physx;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
static eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

int main(int argc, char** argv) {
  int nb = 1000, steps = 600, seed = 1, stab = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stab") && i + 1 < argc) stab = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(4);
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  if (stab) sd.flags |= PxSceneFlag::eENABLE_STABILIZATION;
  PxScene* scene = phys->createScene(sd);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.5f, 0.0f);

  eng::SceneParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.speedScale = 10.0f;
  sp.stabilization = stab != 0;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  // 거의 단위인 쿼터니언 (double 로 정규화 후 float 로 -> 끝비트가 단위에서 조금 벗어남; API 의 isSane 검사는 통과)
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  std::vector<PxRigidDynamic*> px(nb);
  std::vector<eng::Body> eb(nb);
  for (int i = 0; i < nb; ++i) {
    // 서로 50 m 떨어진 격자: 5 초 동안 옆으로 최대 15 m 움직여도 경계 상자가 겹치지 않는다 (접촉이 하나라도 생기면
    // 그 묶음 전체가 위치 반복 경로로 바뀐다 — DyTGSDynamics.cpp:2527. 그 경로는 다음 단계에서 옮긴다)
    PxTransform pose(PxVec3(float(i % 40) * 50.0f, float((i / 40) % 40) * 50.0f, 100.0f + float(i / 1600) * 50.0f),
                     rq());
    PxTransform cm(PxVec3(0.05f * U(rng), 0.05f * U(rng), 0.05f * U(rng)), rq());
    const float mass = 0.1f + 20.0f * P(rng);
    const PxVec3 inertia(0.001f + P(rng), 0.001f + P(rng), 0.001f + P(rng));
    const PxVec3 lv(3.0f * U(rng), 3.0f * U(rng), 3.0f * U(rng));
    const PxVec3 av(8.0f * U(rng), 8.0f * U(rng), 8.0f * U(rng));
    const float ld = (i % 3 == 0) ? 0.5f * P(rng) : 0.0f, ad = (i % 4 == 0) ? 2.0f * P(rng) : 0.05f;
    const bool gyro = (i % 5 == 0), nograv = (i % 7 == 0);
    const PxU8 lock = (i % 11 == 0) ? PxU8(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X | PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) : PxU8(0);
    const float maxAng = (i % 13 == 0) ? 2.0f : -1.0f;

    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    if (!a) { fprintf(stderr, "createRigidDynamic 실패 %d\n", i); return 1; }
    PxRigidActorExt::createExclusiveShape(*a, PxSphereGeometry(0.05f), *mat);
    a->setCMassLocalPose(cm);
    a->setMass(mass);
    a->setMassSpaceInertiaTensor(inertia);
    if (ld != 0.0f) a->setLinearDamping(ld);
    a->setAngularDamping(ad);
    if (gyro) a->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
    if (nograv) a->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    if (lock) a->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(lock));
    if (maxAng > 0) a->setMaxAngularVelocity(maxAng);
    scene->addActor(*a);
    a->setLinearVelocity(lv);
    a->setAngularVelocity(av);
    px[i] = a;

    eng::Body& b = eb[i];
    b = eng::createRigidDynamic(toE(pose), sp);
    eng::setCMassLocalPose(b, toE(cm));
    eng::setMass(b, mass);
    eng::setMassSpaceInertiaTensor(b, toE(inertia));
    if (ld != 0.0f) b.linDamping = ld;
    b.angDamping = ad;
    b.gyroscopic = gyro;
    b.disableGravity = nograv;
    b.lockFlags = lock;
    if (maxAng > 0) b.maxAngVelSq = maxAng * maxAng;
    b.linVel = toE(lv);
    b.angVel = toE(av);
    // setLinearVelocity/setAngularVelocity 의 자동 깨우기: 이미 깨어 있고 깸 카운터 0.4 그대로 (NpRigidDynamic.cpp:599)
  }

  const float dt = 1.0f / 120.0f;
  uint64_t cmp = 0, bad[4] = {0, 0, 0, 0};
  int64_t first[4] = {-1, -1, -1, -1};
  double maxd[4] = {0, 0, 0, 0};
  const char* names[4] = {"행위자 자세", "선속도", "각속도", "깸 카운터"};
  int shown = 0;
  for (int s = 1; s <= steps; ++s) {
    scene->simulate(dt);
    scene->fetchResults(true);
    for (int i = 0; i < nb; ++i) eng::stepFree(eb[i], sp, dt);
    for (int i = 0; i < nb; ++i) {
      const PxTransform tp = px[i]->getGlobalPose();
      const eng::Tf te = eng::getGlobalPose(eb[i]);
      const PxVec3 lp = px[i]->getLinearVelocity(), ap = px[i]->getAngularVelocity();
      const float wp = px[i]->getWakeCounter();
      const void* pa[4] = {&tp, &lp, &ap, &wp};
      const void* pe[4] = {&te, &eb[i].linVel, &eb[i].angVel, &eb[i].wakeCounter};
      const int nf[4] = {7, 3, 3, 1};
      for (int k = 0; k < 4; ++k) {
        cmp++;
        if (memcmp(pa[k], pe[k], 4 * nf[k])) {
          bad[k]++;
          if (first[k] < 0) first[k] = s;
          const float* x = static_cast<const float*>(pa[k]);
          const float* y = static_cast<const float*>(pe[k]);
          for (int j = 0; j < nf[k]; ++j) maxd[k] = std::fmax(maxd[k], std::fabs(double(x[j]) - double(y[j])));
          if (shown < 6) {
            shown++;
            printf("[다름] step %d 몸체 %d %s\n  PhysX:", s, i, names[k]);
            for (int j = 0; j < nf[k]; ++j) printf(" %.9g", x[j]);
            printf("\n  엔진 :");
            for (int j = 0; j < nf[k]; ++j) printf(" %.9g", y[j]);
            printf("\n");
          }
        }
      }
    }
  }
  printf("\n자유 강체 %d 개 x %d 스텝 (안정화 %s): 비교 %" PRIu64 "\n", nb, steps, stab ? "켬" : "끔", cmp);
  for (int k = 0; k < 4; ++k)
    printf("  %-10s 비트 다름 %8" PRIu64 "  첫 다름 스텝 %5" PRId64 "  최대|차| %.3e\n", names[k], bad[k], first[k], maxd[k]);
  const bool ok = !bad[0] && !bad[1] && !bad[2] && !bad[3];
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
