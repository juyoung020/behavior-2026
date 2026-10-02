// 층 1 시험 (contact, AABB 관리자): 우리 Bp::AABBManager(core/contact/px/aabb.h) = PhysX 5.6.1 Bp::AABBManager (+PABP)?
// Sc::Scene 이 부르는 순서(ScPipeline.cpp:351 updateBPFirstPass -> :396 updateBPSecondPass -> :426 postBroadPhase,
// 작업 경로 = continuation 있음, :583 바뀜 비트맵 비우기, :1366 freeBuffers)를 양쪽에 똑같이 하고,
// 겹침 생성·소멸 목록(eSHAPE·eTRIGGER)을 순서까지 비교한다. 묶음(aggregate, BEHAVIOR 관절체 23개가 씀)·자기 충돌·
// 묶음 안 모양 넣고 빼기·묶음 없애기·트리거 모양·정적/운동학 걸러내기 방식을 섞는다.
//   test_aabb_mgr [--shapes N] [--aggs A] [--steps S] [--seed X] [--workers W] [--kk keep|suppress|kill] [--sk ...]
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "BpAABBManager.h"
#include "BpBroadPhase.h"
#include "CmFlushPool.h"
#include "PxcScratchAllocator.h"

#include "core/contact/px/aabb.h"

using namespace physx;
namespace ep = eng::px;

struct DoneTask : public PxLightCpuTask {
  std::atomic<bool> done{false};
  void run() override { done.store(true); }
  const char* getName() const override { return "DoneTask"; }
};

static PxDefaultAllocator gAlloc;
struct HeapCb : public PxVirtualAllocatorCallback {  // PhysX 핀 메모리 할당기 자리 (Sc 는 GPU 없으면 힙)
  void* allocate(size_t size, int, const char*, int) override { return malloc(size); }
  void deallocate(void* p) override { free(p); }
};
static HeapCb gHeapCb;
static PxDefaultErrorCallback gErr;

static PxPairFilteringMode::Enum parseMode(const char* s) {
  if (!strcmp(s, "keep")) return PxPairFilteringMode::eKEEP;
  if (!strcmp(s, "kill")) return PxPairFilteringMode::eKILL;
  return PxPairFilteringMode::eSUPPRESS;
}

int main(int argc, char** argv) {
  int nShapes = 600, nAggs = 20, steps = 200, seed = 1, workers = 4;
  PxPairFilteringMode::Enum kk = PxPairFilteringMode::eDEFAULT, sk = PxPairFilteringMode::eDEFAULT;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--shapes") && i + 1 < argc) nShapes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--aggs") && i + 1 < argc) nAggs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--workers") && i + 1 < argc) workers = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--kk") && i + 1 < argc) kk = parseMode(argv[++i]);
    else if (!strcmp(argv[i], "--sk") && i + 1 < argc) sk = parseMode(argv[++i]);
  }
  PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxDefaultCpuDispatcher* disp = PxDefaultCpuDispatcherCreate(PxU32(workers));
  PxTaskManager* tm = PxTaskManager::createTaskManager(gErr, disp);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);

  // ---- 양쪽 관리자
  PxVirtualAllocator allocP(&gHeapCb);
  Bp::BroadPhase* bpP = Bp::BroadPhase::create(PxBroadPhaseType::ePABP, 0, 0, 0, 0, 0);
  Bp::BoundsArray boundsP(allocP);
  PxFloatArrayPinnedSafe distP(allocP.getCallback());
  Bp::AABBManager* mP = PX_NEW(Bp::AABBManager)(*bpP, boundsP, distP, 64, 64, allocP, 0, kk, sk);
  Cm::FlushPool poolP(4096);
  PxcScratchAllocator scratchP;

  ep::PxVirtualAllocator allocE;
  ep::Bp::BroadPhaseABP* bpE = new ep::Bp::BroadPhaseABP(0, 0, 0, 0, true);
  ep::Bp::BoundsArray boundsE(allocE);
  ep::PxFloatArrayPinnedSafe distE(allocE);
  auto* mE = new ep::Bp::AABBManager(*bpE, boundsE, distE, 64, 64, allocE, 0, ep::PxPairFilteringMode::Enum(kk), ep::PxPairFilteringMode::Enum(sk));
  ep::Cm::FlushPool poolE;
  ep::PxcScratchAllocator scratchE;

  const PxU32 cap = PxU32(nShapes + nAggs * 2 + 64);
  mP->getChangedAABBMgActorHandleMap().resize(cap);
  mE->getChangedAABBMgActorHandleMap().resize(cap);

  // ---- 우리 쪽 장면 기록
  enum Kind { FREED = -2, NONE = -1, STATIC = 0, DYN = 1, KIN = 2, AGGSHAPE = 3, AGG = 4 };
  std::vector<int> kind(cap, NONE), owner(cap, -1);  // owner: 묶음 번호(모양이 묶음에 들었으면)
  std::vector<PxBounds3> bounds(cap);
  std::vector<PxVec3> vel(cap);
  struct Agg { PxU32 index, handleP, handleE; std::vector<PxU32> shapes; PxVec3 vel; bool alive; };
  std::vector<Agg> aggs;
  PxU32 nextRigid = 0;
  auto freeIndex = [&]() -> PxU32 {
    for (PxU32 i = 0; i < cap; ++i)
      if (kind[i] == NONE) return i;
    return PxU32(-1);
  };
  auto setB = [&](PxU32 i, const PxBounds3& b) {
    bounds[i] = b;
    boundsP.setBounds(b, i);
    boundsE.setBounds(reinterpret_cast<const ep::PxBounds3&>(b), i);
  };
  auto markChanged = [&](PxU32 i) {
    mP->getChangedAABBMgActorHandleMap().growAndSet(i);
    mE->getChangedAABBMgActorHandleMap().growAndSet(i);
  };
  auto ud = [](PxU32 i) { return reinterpret_cast<void*>(size_t(i + 1) * 16); };
  auto addShape = [&](PxU32 i, const PxVec3& c, int k, PxU32 aggHandleP, PxU32 aggHandleE) {
    const PxVec3 e(0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng));
    boundsP.initEntry(i);
    boundsE.initEntry(i);
    setB(i, PxBounds3(c - e, c + e));
    const float cd = 0.01f + 0.04f * P(rng);
    Bp::FilterGroup::Enum g = k == STATIC ? Bp::FilterGroup::eSTATICS : Bp::getFilterGroup_Dynamics(nextRigid++ % 4000, k == KIN);
    const Bp::ElementType::Enum vt = P(rng) < 0.08f ? Bp::ElementType::eTRIGGER : Bp::ElementType::eSHAPE;
    mP->addBounds(i, cd, g, ud(i), aggHandleP, vt, PX_INVALID_U32);
    mE->addBounds(i, cd, ep::Bp::FilterGroup::Enum(g), ud(i), aggHandleE, ep::Bp::ElementType::Enum(vt), EPX_INVALID_U32);
    kind[i] = k;
  };
  auto newAgg = [&]() {
    const PxU32 idx = freeIndex();
    if (idx == PxU32(-1)) return;
    const float r = P(rng);
    const PxAggregateType::Enum t = r < 0.8f ? PxAggregateType::eGENERIC : (r < 0.9f ? PxAggregateType::eKINEMATIC : PxAggregateType::eSTATIC);
    const bool self = P(rng) < 0.6f;
    const PxAggregateFilterHint hint = PxGetAggregateFilterHint(t, self);
    boundsP.initEntry(idx);
    boundsE.initEntry(idx);
    kind[idx] = AGG;
    Agg a;
    a.index = idx;
    a.handleP = mP->createAggregate(idx, Bp::FilterGroup::eINVALID, ud(idx), 64, hint, PX_INVALID_U32);
    a.handleE = mE->createAggregate(idx, ep::Bp::FilterGroup::eINVALID, ud(idx), 64, ep::PxAggregateFilterHint(hint), EPX_INVALID_U32);
    a.vel = PxVec3(U(rng), U(rng), 0.3f * U(rng)) * (t == PxAggregateType::eSTATIC ? 0.0f : 0.15f);
    a.alive = true;
    const PxVec3 c(15.0f * U(rng), 15.0f * U(rng), 2.0f * P(rng));
    const int n = 2 + int(P(rng) * 9);
    for (int k = 0; k < n; ++k) {
      const PxU32 s = freeIndex();
      if (s == PxU32(-1)) break;
      addShape(s, c + PxVec3(U(rng), U(rng), U(rng)) * 0.5f, t == PxAggregateType::eSTATIC ? STATIC : (t == PxAggregateType::eKINEMATIC ? KIN : DYN), a.handleP, a.handleE);
      kind[s] = AGGSHAPE;
      owner[s] = int(aggs.size());
      a.shapes.push_back(s);
    }
    aggs.push_back(a);
  };

  uint64_t cmp[2][2] = {{0, 0}, {0, 0}}, bad = 0;
  int firstBadStep = -1;
  for (int s = 0; s < steps; ++s) {
    // ---- 장면 바꾸기
    if (s == 0) {
      for (int a = 0; a < nAggs; ++a) newAgg();
      for (int k = 0; k < nShapes * 3 / 4; ++k) {
        const PxU32 i = freeIndex();
        const float r = P(rng);
        const int kd = r < 0.3f ? STATIC : (r < 0.4f ? KIN : DYN);
        addShape(i, PxVec3(15.0f * U(rng), 15.0f * U(rng), 2.0f * P(rng)), kd, PX_INVALID_U32, EPX_INVALID_U32);
        vel[i] = kd == STATIC ? PxVec3(0.0f) : PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.15f;
      }
    } else {
      std::vector<bool> touched(cap, false);
      // 홀로 모양 넣기·빼기
      const int addN = int(P(rng) * 4);
      for (int k = 0; k < addN; ++k) {
        const PxU32 i = freeIndex();
        if (i == PxU32(-1)) break;
        const int kd = P(rng) < 0.3f ? STATIC : DYN;
        addShape(i, PxVec3(15.0f * U(rng), 15.0f * U(rng), 2.0f * P(rng)), kd, PX_INVALID_U32, EPX_INVALID_U32);
        vel[i] = kd == STATIC ? PxVec3(0.0f) : PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.15f;
        touched[i] = true;
      }
      const int remN = int(P(rng) * 4);
      for (int k = 0; k < remN; ++k) {
        const PxU32 i = PxU32(P(rng) * cap) % cap;
        if (kind[i] < STATIC || kind[i] > KIN || touched[i]) continue;
        mP->removeBounds(i);
        mE->removeBounds(i);
        kind[i] = FREED;
        touched[i] = true;
      }
      // 묶음: 가끔 모양 빼기/넣기, 드물게 통째로 없애고 새로 만들기
      for (size_t a = 0; a < aggs.size(); ++a) {
        Agg& g = aggs[a];
        if (!g.alive) continue;
        if (P(rng) < 0.01f) {  // 없애기: 모양 다 빼고 destroyAggregate
          for (PxU32 sh : g.shapes) { mP->removeBounds(sh); mE->removeBounds(sh); kind[sh] = FREED; touched[sh] = true; }
          g.shapes.clear();
          Bp::BoundsIndex iP; Bp::FilterGroup::Enum gP;
          ep::Bp::BoundsIndex iE; ep::Bp::FilterGroup::Enum gE;
          const bool okP = mP->destroyAggregate(iP, gP, g.handleP);
          const bool okE = mE->destroyAggregate(iE, gE, g.handleE);
          if (okP != okE || iP != iE || PxU32(gP) != PxU32(gE)) { printf("  [스텝 %d] destroyAggregate 다름\n", s); ++bad; }
          kind[g.index] = FREED;
          touched[g.index] = true;
          g.alive = false;
          continue;
        }
        if (P(rng) < 0.03f && g.shapes.size() > 1) {  // 모양 하나 빼기
          const size_t k = size_t(P(rng) * g.shapes.size()) % g.shapes.size();
          const PxU32 sh = g.shapes[k];
          mP->removeBounds(sh); mE->removeBounds(sh);
          kind[sh] = FREED; touched[sh] = true;
          g.shapes.erase(g.shapes.begin() + long(k));
        }
        if (P(rng) < 0.03f) {  // 모양 하나 넣기
          const PxU32 sh = freeIndex();
          if (sh != PxU32(-1) && !g.shapes.empty()) {
            const PxBounds3& b0 = bounds[g.shapes[0]];
            addShape(sh, b0.getCenter() + PxVec3(U(rng), U(rng), U(rng)) * 0.3f, DYN, g.handleP, g.handleE);
            kind[sh] = AGGSHAPE; owner[sh] = int(a); touched[sh] = true;
            g.shapes.push_back(sh);
          }
        }
      }
      if (P(rng) < 0.05f) newAgg();
      // 옮기기
      for (PxU32 i = 0; i < cap; ++i) {
        if (touched[i]) continue;
        if (kind[i] == DYN || kind[i] == KIN) {
          if (P(rng) < 0.2f) continue;
          PxBounds3 b = bounds[i];
          b.minimum += vel[i]; b.maximum += vel[i];
          if (P(rng) < 0.05f) vel[i] = PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.15f;
          setB(i, b);
          markChanged(i);
        } else if (kind[i] == AGGSHAPE) {
          const Agg& g = aggs[size_t(owner[i])];
          if (g.vel.isZero() || P(rng) < 0.3f) continue;
          PxBounds3 b = bounds[i];
          const PxVec3 d = g.vel + PxVec3(U(rng), U(rng), U(rng)) * 0.02f;
          b.minimum += d; b.maximum += d;
          setB(i, b);
          markChanged(i);
        }
      }
      for (Agg& g : aggs)
        if (g.alive && P(rng) < 0.05f) g.vel = PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.15f;
    }

    // ---- 한 스텝 (Sc 순서, 작업 경로)
    const bool tr = getenv("AABB_TRACE") != nullptr;
    if (tr) fprintf(stderr, "[%d] PhysX\n", s);
    {
      DoneTask d;
      d.setContinuation(*tm, NULL);
      mP->updateBPFirstPass(PxU32(workers), poolP, false, &d);
      d.removeReference();
      while (!d.done.load()) {}
    }
    {
      DoneTask d;
      d.setContinuation(*tm, NULL);
      mP->updateBPSecondPass(&scratchP, &d);
      d.removeReference();
      while (!d.done.load()) {}
    }
    {
      DoneTask d;
      d.setContinuation(*tm, NULL);
      mP->postBroadPhase(&d, poolP);
      d.removeReference();
      while (!d.done.load()) {}
    }
    if (tr) fprintf(stderr, "[%d] 우리 1\n", s);
    {
      ep::EndTask d;
      d.setContinuation(nullptr);
      mE->updateBPFirstPass(PxU32(workers), poolE, false, &d);
      d.removeReference();
    }
    if (tr) fprintf(stderr, "[%d] 우리 2\n", s);
    {
      ep::EndTask d;
      d.setContinuation(nullptr);
      mE->updateBPSecondPass(&scratchE, &d);
      d.removeReference();
    }
    if (tr) fprintf(stderr, "[%d] 우리 3\n", s);
    {
      ep::EndTask d;
      d.setContinuation(nullptr);
      mE->postBroadPhase(&d, poolE);
      d.removeReference();
    }

    if (tr) fprintf(stderr, "[%d] 비교\n", s);
    // ---- 비교
    for (int t = 0; t < 2; ++t)
      for (int w = 0; w < 2; ++w) {
        PxU32 nP = 0, nE = 0;
        const Bp::AABBOverlap* oP = w == 0 ? mP->getCreatedOverlaps(Bp::ElementType::Enum(t), nP) : mP->getDestroyedOverlaps(Bp::ElementType::Enum(t), nP);
        const ep::Bp::AABBOverlap* oE = w == 0 ? mE->getCreatedOverlaps(ep::Bp::ElementType::Enum(t), nE)
                                               : mE->getDestroyedOverlaps(ep::Bp::ElementType::Enum(t), nE);
        bool diff = nP != nE;
        for (PxU32 k = 0; k < PxMin(nP, nE); ++k) {
          ++cmp[t][w];
          diff |= oP[k].mUserData0 != oE[k].mUserData0 || oP[k].mUserData1 != oE[k].mUserData1;
        }
        if (diff) {
          if (!bad) {
            firstBadStep = s;
            printf("  [스텝 %d] %s %s: PhysX %u / 우리 %u\n", s, t ? "트리거" : "모양", w ? "소멸" : "생성", nP, nE);
            for (PxU32 k = 0; k < PxMax(nP, nE) && k < 10; ++k)
              printf("    %u: PhysX (%zu,%zu)  우리 (%zu,%zu)\n", k, k < nP ? size_t(oP[k].mUserData0) / 16 - 1 : 0, k < nP ? size_t(oP[k].mUserData1) / 16 - 1 : 0,
                     k < nE ? size_t(oE[k].mUserData0) / 16 - 1 : 0, k < nE ? size_t(oE[k].mUserData1) / 16 - 1 : 0);
          }
          ++bad;
        }
      }
    // ---- 스텝 끝 (ScPipeline.cpp:583, :1366)
    mP->getChangedAABBMgActorHandleMap().clear();
    mE->getChangedAABBMgActorHandleMap().clear();
    mP->freeBuffers();
    mE->freeBuffers();
    poolP.clear();
    poolE.clear();
    // 번호 되돌리기는 스텝이 끝난 뒤 (Sc 의 ElementIDPool 은 simulate 뒤에 풀어 준다: 같은 스텝에 빼고 다시 넣지 않음)
    for (PxU32 i = 0; i < cap; ++i)
      if (kind[i] == FREED) kind[i] = NONE;
  }
  int alive = 0;
  for (auto& g : aggs) alive += g.alive;
  printf("모양 %d, 묶음 %zu(끝에 %d), 스텝 %d, 일꾼 %d: 겹침 생성 %" PRIu64 "+%" PRIu64 ", 소멸 %" PRIu64 "+%" PRIu64 " (모양+트리거) 순서 비교, 다름 %" PRIu64,
         nShapes, aggs.size(), alive, steps, workers, cmp[0][0], cmp[1][0], cmp[0][1], cmp[1][1], bad);
  if (bad) printf(" 첫 다름 스텝 %d", firstBadStep);
  printf("\n%s\n", bad ? "결과: 다름 있음" : "결과: 겹침 생성·소멸 목록 순서까지 전부 같음");
  return bad ? 1 : 0;
}
