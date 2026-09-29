// 층 2 시험 1: 자유 강체 CUDA 판(판 N 개 동시) = 층 1 C++ = PhysX 5.6.1, 비트 사슬 + 처리량.
//   test_free_bodies_gpu [--bodies N] [--steps N] [--seed S] [--stab 0|1] [--envs E] [--physx 0|1]
// 판(env) E 개 x 판마다 강체 N 개. 판 0 은 PhysX·층 1 과 비교하고, 판 전부는 층 1 과 비트 비교한다(같은 초기값을 복제한 판들).
// 처리량: GPU 커널 시간으로 (판 x 서브스텝)/초 와 (강체 x 서브스텝)/초 를 낸다. 대조: 같은 장면 PhysX CPU 단일 스레드.
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "core/rigid.h"

using namespace physx;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ, 예외 가림)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

__global__ void kStepFree(eng::Body* bodies, int n, eng::SceneParams sp, float dt, int substeps) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  eng::Body b = bodies[i];
  for (int s = 0; s < substeps; ++s) eng::stepFree(b, sp, dt);
  bodies[i] = b;
}

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;
static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
static eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

int main(int argc, char** argv) {
  int nb = 1000, steps = 600, seed = 1, stab = 0, envs = 1024, use_px = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stab") && i + 1 < argc) stab = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--physx") && i + 1 < argc) use_px = atoi(argv[++i]);
  }
  eng::SceneParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.speedScale = 10.0f;
  sp.stabilization = stab != 0;
  const float dt = 1.0f / 120.0f;

  // ---- 장면 만들기 (PhysX 와 층 1 에 같은 API 입력)
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(0);  // 0 = 호출 스레드에서 (단일 스레드 대조군)
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  if (stab) sd.flags |= PxSceneFlag::eENABLE_STABILIZATION;
  PxScene* scene = phys->createScene(sd);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.5f, 0.0f);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  std::vector<PxRigidDynamic*> px(nb);
  std::vector<eng::Body> e1(nb);
  for (int i = 0; i < nb; ++i) {
    PxTransform pose(PxVec3(float(i % 40) * 50.0f, float((i / 40) % 40) * 50.0f, 100.0f + float(i / 1600) * 50.0f), rq());
    PxTransform cm(PxVec3(0.05f * U(rng), 0.05f * U(rng), 0.05f * U(rng)), rq());
    const float mass = 0.1f + 20.0f * P(rng);
    const PxVec3 inertia(0.001f + P(rng), 0.001f + P(rng), 0.001f + P(rng));
    const PxVec3 lv(3.0f * U(rng), 3.0f * U(rng), 3.0f * U(rng)), av(8.0f * U(rng), 8.0f * U(rng), 8.0f * U(rng));
    const float ld = (i % 3 == 0) ? 0.5f * P(rng) : 0.0f, ad = (i % 4 == 0) ? 2.0f * P(rng) : 0.05f;
    const bool gyro = (i % 5 == 0), nograv = (i % 7 == 0);
    const PxU8 lock = (i % 11 == 0) ? PxU8(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X | PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) : PxU8(0);
    const float maxAng = (i % 13 == 0) ? 2.0f : -1.0f;
    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    PxRigidActorExt::createExclusiveShape(*a, PxSphereGeometry(0.05f), *mat);
    a->setCMassLocalPose(cm); a->setMass(mass); a->setMassSpaceInertiaTensor(inertia);
    if (ld != 0.0f) a->setLinearDamping(ld);
    a->setAngularDamping(ad);
    if (gyro) a->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
    if (nograv) a->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    if (lock) a->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(lock));
    if (maxAng > 0) a->setMaxAngularVelocity(maxAng);
    scene->addActor(*a);
    a->setLinearVelocity(lv); a->setAngularVelocity(av);
    px[i] = a;
    eng::Body& b = e1[i];
    b = eng::createRigidDynamic(toE(pose), sp);
    eng::setCMassLocalPose(b, toE(cm)); eng::setMass(b, mass); eng::setMassSpaceInertiaTensor(b, toE(inertia));
    if (ld != 0.0f) b.linDamping = ld;
    b.angDamping = ad; b.gyroscopic = gyro; b.disableGravity = nograv; b.lockFlags = lock;
    if (maxAng > 0) b.maxAngVelSq = maxAng * maxAng;
    b.linVel = toE(lv); b.angVel = toE(av);
  }
  const std::vector<eng::Body> init = e1;

  // ---- 층 0: PhysX (단일 스레드, 시간 잼)
  double px_sec = 0;
  if (use_px) {
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) { scene->simulate(dt); scene->fetchResults(true); }
    px_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  // ---- 층 1: 우리 C++ (한 판, FTZ/DAZ)
  double c1_sec;
  {
    FtzScope f;
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) for (int i = 0; i < nb; ++i) eng::stepFree(e1[i], sp, dt);
    c1_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  // ---- 층 2: 우리 CUDA (판 envs 개, 판마다 같은 초기값)
  const size_t total = size_t(nb) * envs;
  std::vector<eng::Body> h(total);
  for (int e = 0; e < envs; ++e) memcpy(&h[size_t(e) * nb], init.data(), sizeof(eng::Body) * nb);
  eng::Body* d = nullptr;
  CK(cudaMalloc(&d, sizeof(eng::Body) * total));
  CK(cudaMemcpy(d, h.data(), sizeof(eng::Body) * total, cudaMemcpyHostToDevice));
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
  const int bs = 128, gs = int((total + bs - 1) / bs);
  CK(cudaEventRecord(ev0));
  kStepFree<<<gs, bs>>>(d, int(total), sp, dt, steps);
  CK(cudaEventRecord(ev1));
  CK(cudaEventSynchronize(ev1));
  CK(cudaGetLastError());
  float gms = 0;
  CK(cudaEventElapsedTime(&gms, ev0, ev1));
  CK(cudaMemcpy(h.data(), d, sizeof(eng::Body) * total, cudaMemcpyDeviceToHost));
  CK(cudaFree(d));

  // ---- 비교: 층 0 vs 층 1 (판 0), 층 1 vs 층 2 (모든 판)
  uint64_t b01 = 0, b12 = 0;
  for (int i = 0; use_px && i < nb; ++i) {
    const PxTransform tp = px[i]->getGlobalPose();
    const eng::Tf te = eng::getGlobalPose(e1[i]);
    const PxVec3 lp = px[i]->getLinearVelocity(), ap = px[i]->getAngularVelocity();
    const float wp = px[i]->getWakeCounter();
    b01 += memcmp(&tp, &te, 28) != 0;
    b01 += memcmp(&lp, &e1[i].linVel, 12) != 0;
    b01 += memcmp(&ap, &e1[i].angVel, 12) != 0;
    b01 += memcmp(&wp, &e1[i].wakeCounter, 4) != 0;
  }
  int shown = 0;
  for (size_t k = 0; k < total; ++k) {
    const eng::Body& g = h[k];
    const eng::Body& c = e1[k % nb];
    const bool diff = memcmp(&g.body2World, &c.body2World, sizeof(eng::Tf)) || memcmp(&g.linVel, &c.linVel, 12) ||
                      memcmp(&g.angVel, &c.angVel, 12) || memcmp(&g.wakeCounter, &c.wakeCounter, 4);
    if (diff) {
      b12++;
      if (shown++ < 3)
        printf("[층1≠층2] 판 %zu 몸체 %zu: p %.9g %.9g %.9g / %.9g %.9g %.9g\n", k / nb, k % nb, g.body2World.p.x, g.body2World.p.y,
               g.body2World.p.z, c.body2World.p.x, c.body2World.p.y, c.body2World.p.z);
    }
  }
  printf("자유 강체 %d 개 x %d 서브스텝, 안정화 %s, GPU 판 %d 개 (강체 %zu 개)\n", nb, steps, stab ? "켬" : "끔", envs, total);
  if (use_px) printf("  층0 PhysX  vs 층1 C++  (판 0)  : 비트 다름 %" PRIu64 " / %d\n", b01, nb * 4);
  printf("  층1 C++    vs 층2 CUDA (전 판) : 비트 다름 %" PRIu64 " / %zu 강체\n", b12, total);
  const double sub_px = use_px ? steps / px_sec : 0, sub_c1 = steps / c1_sec;
  printf("  시간: PhysX 단일 스레드 %.3f s (%.0f 판·서브스텝/초), 층1 %.3f s (%.0f), 층2 %.3f s -> %.3g 판·서브스텝/초, %.3g 강체·서브스텝/초\n",
         px_sec, sub_px, c1_sec, sub_c1, gms / 1000.0, double(envs) * steps / (gms / 1000.0), double(total) * steps / (gms / 1000.0));
  scene->release(); phys->release(); fnd->release();
  return (b01 || b12) ? 3 : 0;
}
