// 층 1 함수 단위 시험: 관절체 결합점(접촉·조인트·solver 가 부르는 함수)이 PhysX 내부 함수와 비트 동일?
// 무작위 관절체(+ 선택: R1Pro 모양)를 양쪽에서 같은 입력으로 돌리다가, 정해진 스텝 뒤에 같은 무작위 충격으로
//   getImpulseResponse, getImpulseSelfResponse(부모-자식/먼 쌍), pxcFsApplyImpulse(s) -> pxcFsGetVelocity(관절 속도 포함),
//   pxcFsGetVelocities, getLinkMotionVector, getDeltaQ, getLinkMaxPenBias, getCfm, getMotionAcceleration
// 를 PhysX 의 Dy::FeatherstoneArticulation 에 직접 부른 결과와 비트 비교한다 (PhysX 내부 헤더 사용 — 시험 프로그램만).
//   test_articulation_coupling [--arts N] [--links N] [--steps N] [--seed S] [--trials T] [--r1pro urdf]
#include <xmmintrin.h>

#include <cinttypes>
#include <cstdlib>

#include "px_art_internal.h"
#include "random_art.h"

using namespace physx;
using namespace artest;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

static uint64_t gCmp[16], gBad[16];
static const char* gNames[16] = {"getImpulseResponse", "getImpulseSelfResponse(부모-자식)", "getImpulseSelfResponse(먼 쌍)", "pxcFsGetVelocity",
                                 "pxcFsGetVelocity 관절속도", "pxcFsGetVelocities", "getLinkMotionVector", "getDeltaQ", "getLinkMaxPenBias",
                                 "getCfm", "getMotionAcceleration", "", "", "", "", ""};
static int gShown = 0;
static void cmpv(int k, const float* x, const float* y, int n, int step) {
  gCmp[k]++;
  for (int j = 0; j < n; ++j) {
    const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
    if ((nx || ny) ? (nx != ny) : (memcmp(&x[j], &y[j], 4) != 0)) {
      gBad[k]++;
      if (gShown++ < 10) {
        printf("[다름] step %d %s 칸 %d: PhysX", step, gNames[k], j);
        for (int i = 0; i < n; ++i) printf(" %.9g", x[i]);
        printf(" / 엔진");
        for (int i = 0; i < n; ++i) printf(" %.9g", y[i]);
        printf("\n");
      }
      return;
    }
  }
}

int main(int argc, char** argv) {
  BuildOpts o;
  int steps = 400, trials = 20;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) o.nArts = atoi(argv[++i]);
    else if (arg("--links")) o.nLinksMax = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--seed")) o.seed = atoi(argv[++i]);
    else if (arg("--trials")) trials = atoi(argv[++i]);
    else if (arg("--r1pro")) o.r1pro = argv[++i];
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
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS | PxSceneFlag::eENABLE_STABILIZATION;
  PxScene* scene = phys->createScene(sd);
  std::vector<Mirror> arts;
  if (!buildRandom(phys, scene, o, arts)) return 1;
  const float dt = 1.0f / 120.0f;
  A::StepParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.dt = dt;
  std::mt19937 rng(uint32_t(o.seed) * 7919u + 1u);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  auto rv = [&](float s) { return PxVec3(s * U(rng), s * U(rng), s * U(rng)); };
  std::vector<float> buf;
  uint64_t stateMismatch = 0;
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
    // 상태가 같은지 먼저 (결합점 비교의 전제)
    for (auto& m : arts)
      for (uint32_t l = 0; l < m.pl.size(); ++l) {
        const PxTransform tp = m.pl[l]->getGlobalPose();
        const eng::Tf te = A::linkGlobalPose(*m.e, l);
        if (memcmp(&tp, &te, 28)) stateMismatch++;
      }
    if (!(s == 1 || s == 7 || s % 50 == 0 || s == 151 || s == 301 || s == 351)) continue;
    FtzScope f;  // 결합점 함수는 풀이 안(시뮬레이트 구간)에서 불린다
    for (auto& m : arts) {
      Dy::FeatherstoneArticulation* fa = llArticulation(m.px);
      A::Articulation& e = *m.e;
      const uint32_t n = e.nLinks;
      auto pickLink = [&]() { return uint32_t(std::min(float(n) - 1.0f, (U(rng) * 0.5f + 0.5f) * float(n))); };
      for (int t = 0; t < trials; ++t) {
        // 1) 링크 단위 응답
        {
          const uint32_t l = pickLink();
          const PxVec3 lin = rv(2.0f), ang = rv(1.0f);
          Cm::SpatialVector dv;
          fa->getImpulseResponse(l, Cm::SpatialVector(lin, ang), dv);
          eng::V3 dl, da;
          A::getImpulseResponse(e, l, toE(lin), toE(ang), dl, da);
          const float px6[6] = {dv.linear.x, dv.linear.y, dv.linear.z, dv.angular.x, dv.angular.y, dv.angular.z};
          const float en6[6] = {dl.x, dl.y, dl.z, da.x, da.y, da.z};
          cmpv(0, px6, en6, 6, s);
        }
        // 2) 자기 응답 (부모-자식 쌍, 먼 쌍)
        for (int kind = 0; kind < 2 && n > 1; ++kind) {
          uint32_t l1 = 1 + uint32_t(std::min(float(n) - 2.0f, (U(rng) * 0.5f + 0.5f) * float(n - 1)));
          uint32_t l0 = kind == 0 ? e.links[l1].parent : pickLink();
          if (l0 == l1) l0 = e.links[l1].parent;
          const bool fast = e.links[l1].parent == l0;
          const PxVec3 lin0 = rv(1.0f), ang0 = rv(1.0f), lin1 = rv(1.0f), ang1 = rv(1.0f);
          Cm::SpatialVector d0, d1;
          fa->getImpulseSelfResponse(l0, l1, Cm::SpatialVector(lin0, ang0), Cm::SpatialVector(lin1, ang1), d0, d1);
          eng::V3 a0l, a0a, a1l, a1a;
          A::getImpulseSelfResponse(e, l0, toE(lin0), toE(ang0), a0l, a0a, l1, toE(lin1), toE(ang1), a1l, a1a);
          const float px12[12] = {d0.linear.x, d0.linear.y, d0.linear.z, d0.angular.x, d0.angular.y, d0.angular.z,
                                  d1.linear.x, d1.linear.y, d1.linear.z, d1.angular.x, d1.angular.y, d1.angular.z};
          const float en12[12] = {a0l.x, a0l.y, a0l.z, a0a.x, a0a.y, a0a.z, a1l.x, a1l.y, a1l.z, a1a.x, a1a.y, a1a.z};
          cmpv(fast ? 1 : 2, px12, en12, 12, s);
        }
        // 3) 충격 적용 -> 속도 (지연 충격 누적 상태를 같이 끌고 간다)
        {
          const uint32_t l = pickLink();
          const PxVec3 lin = rv(0.5f), ang = rv(0.3f);
          float jimp[3] = {0.2f * U(rng), 0.2f * U(rng), 0.2f * U(rng)};
          fa->pxcFsApplyImpulse(l, aos::V3LoadU(lin), aos::V3LoadU(ang), jimp);
          A::pxcFsApplyImpulse(e, l, toE(lin), toE(ang), jimp);
          const uint32_t la = pickLink(), lb = pickLink();
          const PxVec3 linA = rv(0.5f), angA = rv(0.3f), linB = rv(0.5f), angB = rv(0.3f);
          float ja[3] = {0.1f * U(rng), 0.1f * U(rng), 0.1f * U(rng)}, jb[3] = {0.1f * U(rng), 0.1f * U(rng), 0.1f * U(rng)};
          fa->pxcFsApplyImpulses(la, aos::V3LoadU(linA), aos::V3LoadU(angA), ja, lb, aos::V3LoadU(linB), aos::V3LoadU(angB), jb);
          A::pxcFsApplyImpulses(e, la, toE(linA), toE(angA), ja, lb, toE(linB), toE(angB), jb);
          const uint32_t lq = pickLink();
          float sp3[3] = {0, 0, 0}, se3[3] = {0, 0, 0};
          const Cm::SpatialVectorV v = fa->pxcFsGetVelocity(lq, sp3);
          PxVec3 vl, va;
          aos::V3StoreU(v.linear, vl);
          aos::V3StoreU(v.angular, va);
          eng::V3 el, ea;
          A::pxcFsGetVelocity(e, lq, se3, el, ea);
          const float px6[6] = {vl.x, vl.y, vl.z, va.x, va.y, va.z}, en6[6] = {el.x, el.y, el.z, ea.x, ea.y, ea.z};
          cmpv(3, px6, en6, 6, s);
          cmpv(4, sp3, se3, int(e.jointData[lq].nbDof), s);
          const uint32_t lc = pickLink(), ld = pickLink();
          Cm::SpatialVectorV v0, v1;
          fa->pxcFsGetVelocities(lc, ld, v0, v1);
          PxVec3 p0l, p0a, p1l, p1a;
          aos::V3StoreU(v0.linear, p0l);
          aos::V3StoreU(v0.angular, p0a);
          aos::V3StoreU(v1.linear, p1l);
          aos::V3StoreU(v1.angular, p1a);
          eng::V3 e0l, e0a, e1l, e1a;
          A::pxcFsGetVelocities(e, lc, ld, e0l, e0a, e1l, e1a);
          const float px12[12] = {p0l.x, p0l.y, p0l.z, p0a.x, p0a.y, p0a.z, p1l.x, p1l.y, p1l.z, p1a.x, p1a.y, p1a.z};
          const float en12[12] = {e0l.x, e0l.y, e0l.z, e0a.x, e0a.y, e0a.z, e1l.x, e1l.y, e1l.z, e1a.x, e1a.y, e1a.z};
          cmpv(5, px12, en12, 12, s);
        }
      }
      // 4) 링크 조회
      for (uint32_t l = 0; l < n; ++l) {
        const Cm::SpatialVectorV mv = fa->getLinkMotionVector(l);
        PxVec3 ml, ma;
        aos::V3StoreU(mv.linear, ml);
        aos::V3StoreU(mv.angular, ma);
        eng::V3 el, ea;
        A::getLinkMotionVector(e, l, el, ea);
        const float px6[6] = {ml.x, ml.y, ml.z, ma.x, ma.y, ma.z}, en6[6] = {el.x, el.y, el.z, ea.x, ea.y, ea.z};
        cmpv(6, px6, en6, 6, s);
        const PxQuat& dq = fa->getDeltaQ(l);
        const eng::Q& eq = A::getDeltaQ(e, l);
        cmpv(7, &dq.x, &eq.x, 4, s);
        const float pb = fa->getLinkMaxPenBias(l), eb = A::getLinkMaxPenBias(e, l);
        cmpv(8, &pb, &eb, 1, s);
        const float pc = fa->getCfm(l), ec = A::getCfm(e, l);
        cmpv(9, &pc, &ec, 1, s);
        const Cm::SpatialVector acc = fa->getMotionAcceleration(l, false);
        eng::V3 al, aa;
        A::getMotionAcceleration(e, l, al, aa);
        const float px6b[6] = {acc.linear.x, acc.linear.y, acc.linear.z, acc.angular.x, acc.angular.y, acc.angular.z};
        const float en6b[6] = {al.x, al.y, al.z, aa.x, aa.y, aa.z};
        cmpv(10, px6b, en6b, 6, s);
      }
    }
  }
  printf("\n관절체 %zu 개, %d 스텝, 시도 %d (상태 자세 다름 %" PRIu64 ")\n", arts.size(), steps, trials, stateMismatch);
  bool ok = stateMismatch == 0;
  for (int k = 0; k < 11; ++k) {
    printf("  %-36s 비교 %8" PRIu64 "  비트 다름 %6" PRIu64 "\n", gNames[k], gCmp[k], gBad[k]);
    ok = ok && gBad[k] == 0;
  }
  printf("%s\n", ok ? "결과: 결합점 전부 비트 동일" : "결과: 불일치 있음");
  for (auto& m : arts) m.cache->release();
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
