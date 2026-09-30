// 층 1 시험: PhysX 관절체 상태를 아무 simulate 경계에서 우리 엔진으로 옮겨 담으면(art_from_px.h snapshotFromPx) 그 뒤로 PhysX 와 비트 동일?
// 통합(문서 15절 S5)에서 공식 OVD 재생 중 PhysX 관절체를 우리 것으로 바꿔 끼우는 자리의 검증이다.
// 무작위 관절체(+R1Pro 모양)를 PhysX 로 돌리다가 스냅숏 스텝마다 쌍둥이를 새로 만들고, 그 뒤 매 스텝 같은 입력으로 PhysX 와 비교한다.
// 스냅숏 스텝: 0(첫 simulate 전), 그리고 --snaps 로 준 스텝들(기본 1, 37, 149, 151, 201, 351, 420 — 순간이동·힘·잠 앞뒤).
//   test_articulation_snapshot [--arts N] [--links N] [--steps N] [--seed S] [--pos P] [--vel V] [--r1pro urdf] [--r1copies N]
//                              [--floating 0|1] [--snaps a,b,c] [--neg 1..4 (음성 대조)] [--every N (N 스텝마다 스냅숏 더)]
#include <xmmintrin.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "art_from_px.h"

using namespace physx;
using namespace artest;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

struct Twin {
  size_t art;
  int from;
  std::unique_ptr<A::Articulation> e;
};

int main(int argc, char** argv) {
  BuildOpts o;
  int steps = 600, neg = 0, every = 0;
  std::vector<int> snaps = {0, 1, 37, 149, 151, 201, 351, 420};
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) o.nArts = atoi(argv[++i]);
    else if (arg("--links")) o.nLinksMax = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--seed")) o.seed = atoi(argv[++i]);
    else if (arg("--pos")) o.posIt = atoi(argv[++i]);
    else if (arg("--vel")) o.velIt = atoi(argv[++i]);
    else if (arg("--r1pro")) o.r1pro = argv[++i];
    else if (arg("--r1copies")) o.r1copies = atoi(argv[++i]);
    else if (arg("--floating")) o.floatingOnly = atoi(argv[++i]);
    else if (arg("--neg")) neg = atoi(argv[++i]);
    else if (arg("--every")) every = atoi(argv[++i]);
    else if (arg("--snaps")) {
      snaps.clear();
      std::stringstream ss(argv[++i]);
      std::string t;
      while (std::getline(ss, t, ',')) snaps.push_back(atoi(t.c_str()));
    }
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
  if (!buildRandom(phys, scene, o, arts)) {
    fprintf(stderr, "addToScene 실패\n");
    return 1;
  }
  const float dt = 1.0f / 120.0f;
  A::StepParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.dt = dt;
  sp.lengthScale = 1.0f;

  std::vector<Twin> twins;
  uint64_t nPending = 0, nSnapFail = 0, cmp = 0, bad = 0, nanBoth = 0;
  int64_t firstBad = -1;
  int shown = 0;
  std::vector<float> buf;
  auto take = [&](int s) {
    for (size_t ai = 0; ai < arts.size(); ++ai) {
      Twin t{ai, s, std::unique_ptr<A::Articulation>(new A::Articulation)};
      if (!snapshotFromPx(arts[ai].px, *t.e, arts[ai].sc, arts[ai].pl)) {
        nSnapFail++;
        if (shown++ < 12) printf("[스냅숏 실패] step %d art %zu err %u\n", s, ai, t.e->err);
        continue;
      }
      // 음성 대조: 옮긴 값 하나를 망가뜨려 시험이 잡는지 (1 잠 누적값 0, 2 잠 대기 끔, 3 관절 속도 1 ulp, 4 지연 충격 표시 끔)
      A::Articulation& e = *t.e;
      if (neg == 1)
        for (uint32_t l = 0; l < e.nLinks; ++l) e.bodies[l].sleepLinVelAcc = e.bodies[l].sleepAngVelAcc = eng::V3{0, 0, 0};
      if (e.readyForSleep) nPending++;
      if (neg == 2) e.readyForSleep = 0;
      if (neg == 3 && e.dofs) e.jointVelocity[0] = std::nextafter(e.jointVelocity[0], 1e30f);
      if (neg == 4) e.jointDirty = 0;
      twins.push_back(std::move(t));
    }
  };
  if (std::find(snaps.begin(), snaps.end(), 0) != snaps.end()) take(0);
  for (int s = 1; s <= steps; ++s) {
    for (auto& m : arts) m.applyInputsPx(s, dt, buf);
    for (auto& t : twins) stepInputsEng(*t.e, arts[t.art].in, s, 0, dt);
    scene->simulate(dt);
    scene->fetchResults(true);
    {
      FtzScope f;
      for (auto& t : twins) A::stepAlone(*t.e, sp);
    }
    for (auto& t : twins) {
      Mirror& m = arts[t.art];
      A::Articulation& e = *t.e;
      // 링크 자세 7·선/각속도 6, 관절 위치·속도, 깸 카운터, 잠 여부
      std::vector<float> x, y;
      for (uint32_t l = 0; l < m.pl.size(); ++l) {
        const PxTransform tp = m.pl[l]->getGlobalPose();
        const eng::Tf te = A::linkGlobalPose(e, l);
        const PxVec3 lv = m.pl[l]->getLinearVelocity(), av = m.pl[l]->getAngularVelocity();
        const A::LinkBody& b = e.bodies[e.ll[l]];
        const float px7[13] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lv.x, lv.y, lv.z, av.x, av.y, av.z};
        const float en7[13] = {te.q.x, te.q.y, te.q.z, te.q.w, te.p.x, te.p.y, te.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z};
        x.insert(x.end(), px7, px7 + 13);
        y.insert(y.end(), en7, en7 + 13);
      }
      if (e.dofs) {
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        x.insert(x.end(), m.cache->jointPosition, m.cache->jointPosition + e.dofs);
        x.insert(x.end(), m.cache->jointVelocity, m.cache->jointVelocity + e.dofs);
        y.insert(y.end(), static_cast<const float*>(e.jointPosition), static_cast<const float*>(e.jointPosition) + e.dofs);
        y.insert(y.end(), static_cast<const float*>(e.jointVelocity), static_cast<const float*>(e.jointVelocity) + e.dofs);
      }
      x.push_back(m.px->getWakeCounter());
      y.push_back(e.wakeCounter);
      x.push_back(m.px->isSleeping() ? 1.0f : 0.0f);
      y.push_back(e.awake ? 0.0f : 1.0f);
      cmp++;
      bool same = true, anyNan = false;
      size_t firstJ = 0;
      for (size_t j = 0; j < x.size(); ++j) {
        const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
        if (nx || ny) {
          anyNan = true;
          if (nx != ny && same) same = false, firstJ = j;
        } else if (memcmp(&x[j], &y[j], 4) && same) {
          same = false;
          firstJ = j;
        }
      }
      if (same && anyNan) nanBoth++;
      if (!same) {
        bad++;
        if (firstBad < 0) firstBad = s;
        if (shown++ < 12)
          printf("[다름] step %d art %zu (스냅숏 %d) 칸 %zu/%zu: PhysX %.9g 엔진 %.9g\n", s, t.art, t.from, firstJ, x.size(), x[firstJ], y[firstJ]);
      }
    }
    if (std::find(snaps.begin(), snaps.end(), s) != snaps.end() || (every > 0 && s % every == 0)) take(s);
  }
  printf("\n관절체 %zu 개 x %d 스텝 (위치반복 %d, 속도반복 %d, seed %d), 스냅숏 %zu 번 -> 쌍둥이 %zu (실패 %" PRIu64 ")\n", arts.size(), steps, o.posIt,
         o.velIt, o.seed, snaps.size(), twins.size(), nSnapFail);
  printf("  잠 대기 중에 옮긴 쌍둥이 %" PRIu64 "\n", nPending);
  printf("  쌍둥이 = PhysX : 비교 %" PRIu64 " (쌍둥이 x 스텝, 관절체 상태 전체) 비트 다름 %" PRIu64 " 첫 다름 스텝 %" PRId64 " 양쪽 NaN %" PRIu64 "\n", cmp, bad,
         firstBad, nanBoth);
  const bool ok = bad == 0 && nSnapFail == 0;
  printf("%s\n", ok ? "결과: 옮겨 담은 뒤 전부 비트 동일" : "결과: 불일치 있음");
  for (auto& m : arts) m.cache->release();
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
