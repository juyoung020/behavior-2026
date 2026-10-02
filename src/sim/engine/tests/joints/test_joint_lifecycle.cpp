// 층 1 시험: 조인트 붙이기·떼기의 장면 쪽 부수효과(core/joints/joint_lifecycle.h) = PhysX 5.6.1.
// 무작위로 D6 조인트를 만들고(동적-동적, 동적-정적, 동적-세계, 같은 쌍 여러 개, eCOLLISION_ENABLED 섞음) 지우며 simulate/fetchResults 를 끼운다.
// 매 호출 뒤 비교:
//   - 살아 있는 조인트마다 제약 번호 (PxConstraint::getGPUIndex = Dy::Constraint::index = writeback 칸) — 미룬 반납·재사용 순서
//   - 동적 몸체마다 PxsBodyCore::numCountedInteractions (잠 판정 DySleep.cpp:69,142,195 에 들어감), BF_HAS_CONSTRAINTS
//   - 모든 행위자 쌍의 filterJointedBodies 결과 (ScFiltering.cpp:299 를 내부 함수로 다시 계산)
//   - wakeCounter·isSleeping 가 조인트 생성·해제 호출로 바뀌지 않는지 (깸은 섬 간선 처리 = solver 몫)
// 몸체에 모양을 달지 않는다(접촉 상호작용이 counted 를 올리지 않게).
//   test_joint_lifecycle [--frames F] [--seed S] [--bodies B]
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"

#define private public
#define protected public
#include "NpConstraint.h"
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "NpScene.h"
#include "ScBodySim.h"
#include "ScConstraintCore.h"
#include "ScScene.h"
#include "ScStaticSim.h"
#undef private
#undef protected

#include "core/joints/joint_lifecycle.h"

using namespace physx;
namespace J = eng::jnt;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

constexpr uint32_t MAXC = 256, MAXB = 64, MAXEV = 4096;

int main(int argc, char** argv) {
  int frames = 400, seed = 1, nb = 12;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
  }
  const int ns = 3;  // 정적 행위자 수. 행위자 번호: 0..nb-1 동적, nb..nb+ns-1 정적
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());
  PxSceneDesc sd(phys->getTolerancesScale());
  sd.gravity = PxVec3(0, 0, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(0);
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  PxScene* scene = phys->createScene(sd);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  std::vector<PxRigidActor*> actors;
  for (int i = 0; i < nb; ++i) {
    PxRigidDynamic* d = phys->createRigidDynamic(PxTransform(PxVec3(3.0f * i, 0, 0)));
    d->setMass(1.0f);
    d->setMassSpaceInertiaTensor(PxVec3(0.1f));
    if (i % 3 == 0) d->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    scene->addActor(*d);
    if (i % 4 == 1) d->putToSleep();
    actors.push_back(d);
  }
  for (int i = 0; i < ns; ++i) {
    PxRigidStatic* s = phys->createRigidStatic(PxTransform(PxVec3(0, 3.0f * i, 5)));
    scene->addActor(*s);
    actors.push_back(s);
  }
  const uint32_t nA = uint32_t(actors.size());
  auto* E = new J::JointScene<MAXC, MAXB, MAXEV>();
  for (auto& b : E->body) b = J::JointBodyState{0, 0};

  struct Live { PxD6Joint* px; uint32_t slot; };
  std::vector<Live> live;
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  auto simOf = [&](uint32_t a) -> Sc::ActorSim* {
    if (a < uint32_t(nb)) return static_cast<NpRigidDynamic*>(actors[a])->getCore().getSim();
    return static_cast<NpRigidStatic*>(actors[a])->getCore().getSim();
  };

  uint64_t cmpId = 0, badId = 0, cmpBody = 0, badBody = 0, cmpPair = 0, badPair = 0, cmpWake = 0, badWake = 0;
  int64_t firstBad = -1;
  uint64_t creates = 0, releases = 0, totalEv = 0;
  auto note = [&](int64_t t, const char* what) {
    if (firstBad < 0) { firstBad = t; printf("[첫 다름] 호출 %" PRId64 ": %s\n", t, what); }
  };
  int64_t call = 0;
  auto compareAll = [&]() {
    call++;
    for (const Live& l : live) {
      cmpId++;
      const uint32_t pid = l.px->getConstraint()->getGPUIndex();
      if (pid != E->slot[l.slot].id) { badId++; note(call, "제약 번호"); }
    }
    for (int b = 0; b < nb; ++b) {
      Sc::BodySim* bs = static_cast<Sc::BodySim*>(simOf(uint32_t(b)));
      cmpBody++;
      const uint32_t pc = bs->getLowLevelBody().getCore().numCountedInteractions;
      const bool pf = bs->readInternalFlag(Sc::ActorSim::BF_HAS_CONSTRAINTS) != 0;
      if (pc != E->body[b].numCountedInteractions || pf != E->hasConstraints(uint32_t(b))) { badBody++; note(call, "몸체 counted/flag"); }
    }
    for (uint32_t a = 0; a < nA; ++a)
      for (uint32_t b = a + 1; b < nA; ++b) {
        if (a >= uint32_t(nb) && b >= uint32_t(nb)) continue;  // 정적-정적 쌍은 거르개에 오지 않는다
        Sc::ActorSim* s0 = simOf(a);
        Sc::ActorSim* s1 = simOf(b);
        bool pxF = false;  // ScFiltering.cpp:299 filterJointedBodies
        if (s0->readInternalFlag(Sc::ActorSim::BF_HAS_CONSTRAINTS) || s1->readInternalFlag(Sc::ActorSim::BF_HAS_CONSTRAINTS)) {
          Sc::ConstraintCore* core = sc.findConstraintCore(s0, s1);
          pxF = core ? !(core->getFlags() & PxConstraintFlag::eCOLLISION_ENABLED) : false;
        }
        cmpPair++;
        if (pxF != E->jointedPairFiltered(a, a < uint32_t(nb), b, b < uint32_t(nb))) { badPair++; note(call, "쌍 거르개"); }
      }
  };

  for (int f = 0; f < frames; ++f) {
    const int nops = int(rng() % 4u);
    for (int o = 0; o < nops; ++o) {
      const bool doCreate = live.empty() || (rng() % 100u) < 55u || live.size() < 3;
      if (doCreate && live.size() < MAXC / 2) {
        // 행위자 고르기: 0 은 동적, 1 은 동적/정적/세계
        uint32_t a0 = uint32_t(rng() % uint32_t(nb));
        uint32_t a1;
        const uint32_t pick = rng() % 10u;
        if (pick < 5) { do a1 = uint32_t(rng() % uint32_t(nb)); while (a1 == a0); }
        else if (pick < 8) a1 = uint32_t(nb) + uint32_t(rng() % uint32_t(ns));
        else a1 = J::NO_ACTOR;
        // 같은 쌍 되풀이 (지도 insert/erase 규칙 확인)
        if (!live.empty() && rng() % 5u == 0) {
          const Live& l = live[rng() % live.size()];
          a0 = E->slot[l.slot].actor[0];
          a1 = E->slot[l.slot].actor[1];
          if (rng() % 2u && a1 != J::NO_ACTOR) std::swap(a0, a1);
          if (a0 == J::NO_ACTOR) std::swap(a0, a1);
        }
        const bool swap01 = a1 != J::NO_ACTOR && rng() % 3u == 0;
        uint32_t x0 = a0, x1 = a1;
        if (swap01) std::swap(x0, x1);
        // 깨우기 불변 확인용
        std::vector<float> wc0(nb);
        std::vector<int> sl0(nb);
        for (int b = 0; b < nb; ++b) {
          wc0[b] = static_cast<PxRigidDynamic*>(actors[b])->getWakeCounter();
          sl0[b] = static_cast<PxRigidDynamic*>(actors[b])->isSleeping();
        }
        PxRigidActor* p0 = x0 == J::NO_ACTOR ? nullptr : actors[x0];
        PxRigidActor* p1 = x1 == J::NO_ACTOR ? nullptr : actors[x1];
        PxD6Joint* j = PxD6JointCreate(*phys, p0, PxTransform(PxVec3(U(rng), U(rng), U(rng))), p1, PxTransform(PxVec3(U(rng), U(rng), U(rng))));
        const bool coll = rng() % 4u == 0;
        if (coll) j->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);
        const uint16_t flags = uint16_t(uint32_t(j->getConstraintFlags()));
        const uint32_t s = E->addConstraint(x0, x0 < uint32_t(nb), x1, x1 < uint32_t(nb), flags);
        live.push_back(Live{j, s});
        creates++;
        for (int b = 0; b < nb; ++b) {
          cmpWake++;
          const PxRigidDynamic* d = static_cast<PxRigidDynamic*>(actors[b]);
          if (d->getWakeCounter() != wc0[b] || int(d->isSleeping()) != sl0[b]) { badWake++; note(call + 1, "생성이 깸 상태를 바꿈"); }
        }
      } else if (!live.empty()) {
        const size_t k = rng() % live.size();
        std::vector<float> wc0(nb);
        for (int b = 0; b < nb; ++b) wc0[b] = static_cast<PxRigidDynamic*>(actors[b])->getWakeCounter();
        live[k].px->release();
        E->removeConstraint(live[k].slot);
        live.erase(live.begin() + long(k));
        releases++;
        for (int b = 0; b < nb; ++b) {
          cmpWake++;
          if (static_cast<PxRigidDynamic*>(actors[b])->getWakeCounter() != wc0[b]) { badWake++; note(call + 1, "해제가 깸 카운터를 바꿈"); }
        }
      }
      compareAll();
    }
    totalEv += E->nEv;
    E->nEv = 0;  // 섬 관리자(solver)가 simulate 앞에서 소비
    if (rng() % 3u) {
      scene->simulate(1.0f / 60.0f);
      scene->fetchResults(true);
      E->postReportsCleanup();
      compareAll();
    }
  }
  printf("\n조인트 붙이기·떼기 부수효과 (프레임 %d, 씨앗 %d, 동적 %d 정적 %d): 생성 %" PRIu64 " 해제 %" PRIu64 "\n", frames, seed, nb, ns,
         creates, releases);
  printf("  제약 번호(getGPUIndex)       비교 %8" PRIu64 "  다름 %" PRIu64 "\n", cmpId, badId);
  printf("  몸체 counted·BF_HAS_CONSTRAINTS 비교 %8" PRIu64 "  다름 %" PRIu64 "\n", cmpBody, badBody);
  printf("  쌍 거르개(filterJointedBodies) 비교 %8" PRIu64 "  다름 %" PRIu64 "\n", cmpPair, badPair);
  printf("  생성·해제 뒤 깸 상태 불변      비교 %8" PRIu64 "  다름 %" PRIu64 "\n", cmpWake, badWake);
  printf("  섬 간선 사건 %" PRIu64 " 개 (추가 = 생성, 제거 = 해제), 넘침 %d\n", totalEv + E->nEv, int(E->overflow || E->ids.overflow));
  const bool ok = badId + badBody + badPair + badWake == 0 && !E->overflow && !E->ids.overflow;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  for (Live& l : live) l.px->release();
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
