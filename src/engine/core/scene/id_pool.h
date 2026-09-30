// Sc 층 번호 매김 (문서 15.3·20.4, 리드): PhysX Cm::IDPool + Sc::ObjectIDTracker 와 같은 규칙.
// 원본: common/src/CmIDPool.h:40 IDPoolBase (getNewID: 빈 번호 목록 끝에서 꺼냄, 없으면 mCurrentID++;
//        freeID: 맨 끝 번호(mCurrentID-1)면 mCurrentID 를 줄이고, 아니면 빈 번호 목록 끝에 넣음),
//       simulationcontroller/src/ScObjectIDTracker.h (releaseID: 지움 표시 + 대기 목록 끝에 넣음 — 바로 다시 쓰이지 않게;
//        processPendingReleases: 대기 목록 순서대로 freeID, Sc::Scene::postReportsCleanup ScScene.cpp:1878 에서 스텝 끝마다).
// 쓰는 곳: 요소 번호(모양·집합체 = 넓은 단계 칸·변환 캐시 칸, ScElementSim.h:111), 행위자 번호(ScActorSim.cpp:73), 조인트 번호(ScConstraintSim.cpp:63),
//          강체 번호(넓은 단계 무리, ScRigidSim.cpp:40). 판 도중 추가·삭제(자르기 등)의 번호가 PhysX 와 같아야 이후 모든 순서가 같다.
// 판 N 개(층 2)에서도 추가·삭제는 드문 사건이라 호스트에서 판마다 하나씩 둔다(추정: 장치 쪽이 필요해지면 고정 용량 판으로 옮김).
#pragma once
#include <cstdint>
#include <vector>

namespace eng {
namespace scene {

struct IdPool {
  uint32_t cur = 0;               // mCurrentID
  std::vector<uint32_t> freeIds;  // mFreeIDs (끝에서 꺼냄)

  uint32_t getNew() {
    if (!freeIds.empty()) {
      const uint32_t id = freeIds.back();
      freeIds.pop_back();
      return id;
    }
    return cur++;
  }
  void free(uint32_t id) {
    if (id == cur - 1)
      --cur;
    else
      freeIds.push_back(id);
  }
  uint32_t maxId() const { return cur; }
  uint32_t numUsed() const { return cur - uint32_t(freeIds.size()); }
};

struct IdTracker {
  IdPool pool;
  std::vector<uint32_t> pending;  // mPendingReleasedIDs (넣은 순서)
  std::vector<uint8_t> deleted;   // mDeletedIDsMap (번호별 1/0)

  uint32_t create() { return pool.getNew(); }
  void release(uint32_t id) {
    if (deleted.size() <= id) deleted.resize(id + 1, 0);
    deleted[id] = 1;
    pending.push_back(id);
  }
  bool isDeleted(uint32_t id) const { return id < deleted.size() && deleted[id]; }
  // 스텝 끝 (postReportsCleanup): 대기 목록 순서대로 풀고 지움 표시를 지운다
  void endStep() {
    for (uint32_t id : pending) pool.free(id);
    pending.clear();
    deleted.assign(deleted.size(), 0);
  }
  uint32_t maxId() const { return pool.maxId(); }
};

}  // namespace scene
}  // namespace eng
