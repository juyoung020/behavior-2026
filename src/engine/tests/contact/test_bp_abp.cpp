// 층 1 시험 (contact, 넓은 단계): 우리 ABP(core/contact/px/bp.h) = PhysX 5.6.1 Bp::BroadPhaseABP(PABP 설정)?
// 같은 갱신 입력(새로 넣기·옮기기·빼기 목록, 경계 상자, 접촉 거리, 거르개 무리)을 스텝마다 양쪽에 넣고
// 새 쌍·사라진 쌍 목록을 "순서까지" 비교한다 (이 순서가 접촉 관리자·섬 간선 순서를 정한다).
//   test_bp_abp [--objects N] [--steps S] [--seed X]
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "BpBroadPhase.h"
#include "BpBroadPhaseUpdate.h"
#include "BpFiltering.h"
#include "PxcScratchAllocator.h"

#include "core/contact/px/bp.h"

using namespace physx;
namespace ep = eng::px;

struct DoneTask : public PxLightCpuTask {  // PhysX 쪽 끝 작업: 끝나면 표시
  std::atomic<bool> done{false};
  void run() override { done.store(true); }
  const char* getName() const override { return "DoneTask"; }
};

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

int main(int argc, char** argv) {
  int nObj = 800, steps = 200, seed = 1, mt = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--objects") && i + 1 < argc) nObj = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mt") && i + 1 < argc) mt = atoi(argv[++i]);
  }

  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  (void)fnd;
  // --mt N: PhysX 를 작업 경로(PABP + 이어질 작업)로, 일꾼 N 스레드 (AABB 관리자가 실제로 부르는 방식)
  PxDefaultCpuDispatcher* disp = mt ? PxDefaultCpuDispatcherCreate(PxU32(mt)) : NULL;
  PxTaskManager* tm = mt ? PxTaskManager::createTaskManager(gErr, disp) : NULL;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);

  const PxU32 cap = PxU32(nObj) + 64;
  std::vector<PxBounds3> bounds(cap);
  std::vector<PxReal> dist(cap, 0.0f);
  std::vector<Bp::FilterGroup::Enum> groups(cap, Bp::FilterGroup::eINVALID);
  std::vector<int> kind(cap, -1);  // -1 없음, 0 정적, 1 동적, 2 운동학
  std::vector<PxVec3> vel(cap);

  Bp::BroadPhase* bpP = Bp::BroadPhase::create(PxBroadPhaseType::ePABP, 0, 0, 0, 0, 0);
  ep::Bp::BroadPhaseABP* bpE = new ep::Bp::BroadPhaseABP(0, 0, 0, 0, true);
  const Bp::BpFilter filterP(true, true);
  const ep::Bp::BpFilter filterE(true, true);
  PxcScratchAllocator scratchP;
  ep::PxcScratchAllocator scratchE;

  auto randomBox = [&](PxU32 i) {
    const PxVec3 c(20.0f * U(rng), 20.0f * U(rng), 3.0f * P(rng));
    const PxVec3 e(0.05f + 0.6f * P(rng), 0.05f + 0.6f * P(rng), 0.05f + 0.6f * P(rng));
    bounds[i] = PxBounds3(c - e, c + e);
    dist[i] = 0.01f + 0.05f * P(rng);
  };
  PxU32 nextRigid = 0;
  uint64_t cmpCreated = 0, cmpDeleted = 0, bad = 0, totalCreated = 0, totalDeleted = 0;
  int firstBadStep = -1;
  std::vector<PxU32> alive;
  for (int s = 0; s < steps; ++s) {
    std::vector<PxU32> created, updated, removed;
    // 첫 스텝에 대부분 넣고, 그 뒤 몇 개씩 넣고 뺀다
    const int addN = s == 0 ? nObj * 3 / 4 : int(P(rng) * 6);
    for (int k = 0; k < addN; ++k) {
      PxU32 idx = PxU32(-1);
      for (PxU32 i = 0; i < PxU32(nObj); ++i)
        if (kind[i] < 0) { idx = i; break; }
      if (idx == PxU32(-1)) break;
      const float r = P(rng);
      kind[idx] = r < 0.25f ? 0 : (r < 0.35f ? 2 : 1);
      randomBox(idx);
      vel[idx] = PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.2f;
      groups[idx] = kind[idx] == 0 ? Bp::FilterGroup::eSTATICS : Bp::getFilterGroup_Dynamics(nextRigid++ % 1000, kind[idx] == 2);
      created.push_back(idx);
    }
    // 빼기 (가끔)
    if (s > 0) {
      const int remN = int(P(rng) * 5);
      for (int k = 0; k < remN; ++k) {
        const PxU32 idx = PxU32(P(rng) * nObj) % PxU32(nObj);
        if (kind[idx] < 0) continue;
        bool justCreated = false;
        for (PxU32 c : created) justCreated |= c == idx;
        if (justCreated) continue;
        bool already = false;
        for (PxU32 c : removed) already |= c == idx;
        if (already) continue;
        removed.push_back(idx);
      }
    }
    // 옮기기: 살아 있는 동적·운동학 물체 대부분
    if (s > 0)
      for (PxU32 i = 0; i < PxU32(nObj); ++i) {
        if (kind[i] <= 0) continue;
        bool skip = false;
        for (PxU32 c : created) skip |= c == i;
        for (PxU32 c : removed) skip |= c == i;
        if (skip || P(rng) < 0.2f) continue;  // 가끔 안 움직임
        bounds[i].minimum += vel[i];
        bounds[i].maximum += vel[i];
        if (P(rng) < 0.05f) vel[i] = PxVec3(U(rng), U(rng), 0.3f * U(rng)) * 0.2f;
        updated.push_back(i);
      }

    const Bp::BroadPhaseUpdateData udP(created.data(), PxU32(created.size()), updated.data(), PxU32(updated.size()), removed.data(),
                                       PxU32(removed.size()), bounds.data(), groups.data(), dist.data(), NULL, cap, filterP, true, false);
    const ep::Bp::BroadPhaseUpdateData udE(created.data(), PxU32(created.size()), updated.data(), PxU32(updated.size()), removed.data(),
                                           PxU32(removed.size()), reinterpret_cast<const ep::PxBounds3*>(bounds.data()),
                                           reinterpret_cast<const ep::Bp::FilterGroup::Enum*>(groups.data()), dist.data(), nullptr, cap, filterE,
                                           true, false);
    // 목록은 번호 오름차순이어야 한다 (BpBroadPhaseUpdate.cpp:60, AABB 관리자는 비트맵 순서로 넘김)
    std::sort(created.begin(), created.end());
    std::sort(updated.begin(), updated.end());
    std::sort(removed.begin(), removed.end());
    if (getenv("BP_DEBUG")) {  // PhysX 의 isValid (BpBroadPhaseABP.cpp:4290) 를 우리 쪽 자료로 흉내
      const auto* objs = bpE->mABP->mShared.mABP_Objects;
      const PxU32 capO = bpE->mABP->mShared.mABP_Objects_Capacity;
      for (PxU32 c : created) if (c < capO && objs[c].isValid()) printf("  [검사] 스텝 %d 넣기 %u 이미 있음\n", s, c);
      for (PxU32 c : updated) if (c >= capO || !objs[c].isValid()) printf("  [검사] 스텝 %d 옮기기 %u 없음 (용량 %u)\n", s, c, capO);
      for (PxU32 c : removed) if (c >= capO || !objs[c].isValid()) printf("  [검사] 스텝 %d 빼기 %u 없음 (용량 %u)\n", s, c, capO);
    }
    const char* only = getenv("BP_ONLY");
    if (tm) {
      if (!only || only[0] == 'P') {
      DoneTask doneP;
      doneP.setContinuation(*tm, NULL);
      bpP->update(&scratchP, udP, &doneP);
      doneP.removeReference();
      while (!doneP.done.load()) {}
      }
      if (!only || only[0] == 'E') {
      ep::EndTask doneE;
      doneE.setContinuation(nullptr);
      bpE->update(&scratchE, udE, &doneE);
      doneE.removeReference();
      }
    } else {
      bpP->update(&scratchP, udP, NULL);
      bpE->update(&scratchE, udE, nullptr);
    }
    PxU32 ncP, ndP, ncE, ndE;
    const Bp::BroadPhasePair* cP = bpP->getCreatedPairs(ncP);
    const Bp::BroadPhasePair* dP = bpP->getDeletedPairs(ndP);
    const ep::Bp::BroadPhasePair* cE = bpE->getCreatedPairs(ncE);
    const ep::Bp::BroadPhasePair* dE = bpE->getDeletedPairs(ndE);
    totalCreated += ncP;
    totalDeleted += ndP;
    auto mark = [&]() {
      if (!bad) {
        firstBadStep = s;
        printf("  [스텝 %d] 새 쌍 PhysX %u / 우리 %u, 사라진 쌍 %u / %u (넣기 %zu 옮기기 %zu 빼기 %zu)\n", s, ncP, ncE, ndP, ndE, created.size(),
               updated.size(), removed.size());
        for (PxU32 k = 0; k < PxMax(ncP, ncE) && k < 8; ++k)
          printf("    %u: PhysX (%u,%u)  우리 (%u,%u)\n", k, k < ncP ? cP[k].mVolA : 0u, k < ncP ? cP[k].mVolB : 0u, k < ncE ? cE[k].mVolA : 0u,
                 k < ncE ? cE[k].mVolB : 0u);
        // 무식하게 센 정답 (접촉 거리만큼 부풀린 상자 겹침, 정적-정적·같은 무리 제외)
        PxU32 brute = 0;
        for (PxU32 i = 0; i < cap; ++i)
          for (PxU32 j = i + 1; j < cap; ++j) {
            if (kind[i] < 0 || kind[j] < 0 || (kind[i] == 0 && kind[j] == 0) || groups[i] == groups[j]) continue;
            PxBounds3 a = bounds[i], b = bounds[j];
            a.fattenFast(dist[i]); b.fattenFast(dist[j]);
            if (a.intersects(b)) ++brute;
          }
        printf("    무식하게 센 겹침 쌍 %u\n", brute);
      }
      ++bad;
    };
    if (ncP != ncE) mark();
    if (ndP != ndE) mark();
    for (PxU32 k = 0; k < PxMin(ncP, ncE); ++k) {
      ++cmpCreated;
      if (cP[k].mVolA != cE[k].mVolA || cP[k].mVolB != cE[k].mVolB) mark();
    }
    for (PxU32 k = 0; k < PxMin(ndP, ndE); ++k) {
      ++cmpDeleted;
      if (dP[k].mVolA != dE[k].mVolA || dP[k].mVolB != dE[k].mVolB) mark();
    }
    bpP->freeBuffers();
    bpE->freeBuffers();
    for (PxU32 idx : removed) kind[idx] = -1;
  }
  printf("물체 %d, 스텝 %d: 새 쌍 %" PRIu64 ", 사라진 쌍 %" PRIu64 " (순서 비교 %" PRIu64 " + %" PRIu64 ") 다름 %" PRIu64, nObj, steps, totalCreated,
         totalDeleted, cmpCreated, cmpDeleted, bad);
  if (bad) printf(" 첫 다름 스텝 %d", firstBadStep);
  printf("\n%s\n", bad ? "결과: 다름 있음" : "결과: 새 쌍·사라진 쌍 목록 순서까지 전부 같음");
  return bad ? 1 : 0;
}
