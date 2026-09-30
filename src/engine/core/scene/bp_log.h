// 넓은 단계 상태 넘겨받기 (문서 15.3 v1-b, 리드): 우리 AABB 관리자(core/contact/px/aabb.h, PhysX Bp::AABBManager + PABP 기계 번역)는
// 내부 상태(정렬된 상자 배열·쌍 해시표·묶음 쌍 표 …)가 커서 PhysX 객체에서 통째로 옮기지 않고, **불러오기 구간의 입력 기록**을 장면 파일에 담아
// 적재 때 우리 관리자에 처음부터 다시 넣는다(입력이 같으면 상태가 같다 — 넓은 단계 그림자 시험으로 확인된 성질).
// 기록 = 구조 변경(넣기·빼기·묶음 만들기·없애기·바뀜 비트맵 크기) + 스텝마다 입력(경계 상자 배열·접촉 거리·바뀜 비트맵·접촉 거리 바뀜 표시).
// PhysX 없이 쓴다 (적재기·GPU 판 준비용).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/contact/px/aabb.h"

namespace eng {
namespace scene {

enum BpOpType : uint32_t { BP_ADD = 0, BP_REMOVE = 1, BP_CREATE_AGG = 2, BP_DESTROY_AGG = 3, BP_REALLOC = 4 };
struct BpOp {
  uint32_t type, frame;  // frame = 이 연산 앞에 지나간 스텝 수 (이 스텝 입력 전에 적용)
  uint32_t index, group, agg, volumeType, env, maxNum, hint, size;
  float contactDistance;
  uint32_t result;       // PhysX 반환값 (번호·성공)
  uint64_t userData;     // 기록 때 값 (적재 뒤 살아 있는 칸은 새 값으로 바꿈)
};
struct BpFrame {
  uint64_t boundsStart, distStart, wordsStart;
  uint32_t nBounds, nDist, nWords;
  uint8_t hasContactDistanceChanged, boundsChanged, pad[2];
};
struct BpLog {
  bool valid = false;
  uint32_t abpMaxOverlaps = 0, abpMaxStatic = 0, abpMaxDynamic = 0, abpMT = 1;
  uint64_t ctx = 0;
  uint32_t maxAggregates = 0, maxShapes = 0, kineKine = 0, staticKine = 0;
  std::vector<BpOp> ops;
  std::vector<BpFrame> frames;
  std::vector<float> bounds;  // 상자마다 6 (최소 xyz, 최대 xyz)
  std::vector<float> dist;
  std::vector<uint32_t> words;
};

namespace bpio {
template <class T>
inline bool w(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  return fwrite(&n, 8, 1, f) == 1 && (n == 0 || fwrite(v.data(), sizeof(T), n, f) == n);
}
template <class T>
inline bool r(FILE* f, std::vector<T>& v) {
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1 || n > (1ull << 34)) return false;
  v.resize(n);
  return n == 0 || fread(v.data(), sizeof(T), n, f) == n;
}
}  // namespace bpio
inline bool writeBpLog(FILE* f, const BpLog& b) {
  const uint32_t magic = 0x42504c31u;  // "BPL1"
  const uint32_t hdr[9] = {magic, b.abpMaxOverlaps, b.abpMaxStatic, b.abpMaxDynamic, b.abpMT, b.maxAggregates, b.maxShapes, b.kineKine, b.staticKine};
  return fwrite(hdr, 4, 9, f) == 9 && fwrite(&b.ctx, 8, 1, f) == 1 && bpio::w(f, b.ops) && bpio::w(f, b.frames) && bpio::w(f, b.bounds) &&
         bpio::w(f, b.dist) && bpio::w(f, b.words);
}
inline bool readBpLog(FILE* f, BpLog& b) {
  uint32_t hdr[9];
  if (fread(hdr, 4, 9, f) != 9 || hdr[0] != 0x42504c31u) return false;
  b.abpMaxOverlaps = hdr[1]; b.abpMaxStatic = hdr[2]; b.abpMaxDynamic = hdr[3]; b.abpMT = hdr[4];
  b.maxAggregates = hdr[5]; b.maxShapes = hdr[6]; b.kineKine = hdr[7]; b.staticKine = hdr[8];
  b.valid = fread(&b.ctx, 8, 1, f) == 1 && bpio::r(f, b.ops) && bpio::r(f, b.frames) && bpio::r(f, b.bounds) && bpio::r(f, b.dist) && bpio::r(f, b.words);
  return b.valid;
}

// 우리 AABB 관리자 한 벌 (+ 그 입력 배열·작업 풀). 그림자 시험(replay/g1_bp.cpp)과 적재기가 같이 쓴다.
struct BpRuntime {
  px::PxVirtualAllocator alloc;
  std::unique_ptr<px::Bp::BroadPhaseABP> bp;
  std::unique_ptr<px::Bp::BoundsArray> bounds;
  std::unique_ptr<px::PxFloatArrayPinnedSafe> dist;
  std::unique_ptr<px::Bp::AABBManager> m;
  px::Cm::FlushPool pool;
  px::PxcScratchAllocator scratch;
  std::unordered_map<uint32_t, uint32_t> aggMap;  // 기록(PhysX) 묶음 번호 -> 우리 묶음 번호
  uint32_t handleBad = 0;

  void create(uint32_t maxOverlaps, uint32_t maxStatic, uint32_t maxDynamic, uint64_t ctx, bool mt, uint32_t maxAgg, uint32_t maxShapes, uint32_t kk,
              uint32_t sk) {
    bp.reset(new px::Bp::BroadPhaseABP(maxOverlaps, maxStatic, maxDynamic, ctx, mt));
    bounds.reset(new px::Bp::BoundsArray(alloc));
    dist.reset(new px::PxFloatArrayPinnedSafe(alloc));
    m.reset(new px::Bp::AABBManager(*bp, *bounds, *dist, maxAgg, maxShapes, alloc, ctx, px::PxPairFilteringMode::Enum(kk), px::PxPairFilteringMode::Enum(sk)));
  }
  // 사용자 자료: 기록 값 그대로(그림자 시험) 또는 요소 번호로 (엔진: (번호+1)<<2 — 아래 2 비트는 AABB 관리자가 부피 종류로 씀, scene_step.h userOfElem)
  bool elemUserData = false;
  void* userData(const BpOp& o) const {
    return elemUserData ? reinterpret_cast<void*>((uintptr_t(o.index) + 1) << 2) : reinterpret_cast<void*>(uintptr_t(o.userData));
  }
  // 구조 변경 하나
  void apply(const BpOp& o) {
    switch (o.type) {
      case BP_ADD: {
        bounds->initEntry(o.index);
        px::Bp::AggregateHandle ea = EPX_INVALID_U32;
        if (o.agg != 0xffffffffu) {
          auto it = aggMap.find(o.agg);
          ea = it == aggMap.end() ? EPX_INVALID_U32 : it->second;
        }
        const bool r = m->addBounds(o.index, o.contactDistance, px::Bp::FilterGroup::Enum(o.group), userData(o), ea,
                                    px::Bp::ElementType::Enum(o.volumeType), o.env);
        if (uint32_t(r) != o.result) ++handleBad;
        break;
      }
      case BP_REMOVE:
        if (uint32_t(m->removeBounds(o.index)) != o.result) ++handleBad;
        break;
      case BP_CREATE_AGG: {
        bounds->initEntry(o.index);
        const px::Bp::AggregateHandle h = m->createAggregate(o.index, px::Bp::FilterGroup::Enum(o.group), userData(o), o.maxNum,
                                                            px::PxAggregateFilterHint(o.hint), o.env);
        aggMap[o.result] = h;
        if (h != o.result) ++handleBad;
        break;
      }
      case BP_DESTROY_AGG: {
        px::Bp::BoundsIndex ei;
        px::Bp::FilterGroup::Enum eg;
        auto it = aggMap.find(o.agg);
        const bool r = it != aggMap.end() && m->destroyAggregate(ei, eg, it->second);
        if (uint32_t(r) != o.result) ++handleBad;
        if (it != aggMap.end()) aggMap.erase(it);
        break;
      }
      case BP_REALLOC:
        m->reallocateChangedAABBMgActorHandleMap(o.size);
        break;
    }
  }
  // 한 스텝 입력 넣기 + 세 단계 (Sc 순서: updateBPFirstPass -> updateBPSecondPass -> postBroadPhase). 결과 목록은 freeBuffers 전까지 유효.
  void step(const px::PxBounds3* b, uint32_t nb, bool boundsChanged, const float* d, uint32_t nd, const uint32_t* w, uint32_t nw, bool hasCD) {
    if (nb) bounds->initEntry(nb - 1);
    memcpy(static_cast<void*>(bounds->begin()), b, sizeof(px::PxBounds3) * nb);
    if (boundsChanged) bounds->setChangedState();
    else bounds->resetChangedState();
    if (dist->capacity() < nd) dist->reserve(nd);
    dist->forceSize_Unsafe(nd);
    if (nd) memcpy(dist->begin(), d, sizeof(float) * nd);
    auto& ec = m->getChangedAABBMgActorHandleMap();
    if (ec.getWordCount() < nw) ec.resize(nw * 32);
    ec.clear();
    if (nw) memcpy(ec.getWords(), w, sizeof(uint32_t) * nw);
    {
      px::EndTask t;
      t.setContinuation(nullptr);
      m->updateBPFirstPass(1, pool, hasCD, &t);
      t.removeReference();
    }
    {
      px::EndTask t;
      t.setContinuation(nullptr);
      m->updateBPSecondPass(&scratch, &t);
      t.removeReference();
    }
    {
      px::EndTask t;
      t.setContinuation(nullptr);
      m->postBroadPhase(&t, pool);
      t.removeReference();
    }
  }
  void endStep() {
    m->freeBuffers();
    pool.clear();
  }
  // 기록을 처음부터 다시 넣어 경계 상태로
  bool replay(const BpLog& L, bool elemUD = false) {
    elemUserData = elemUD;
    create(L.abpMaxOverlaps, L.abpMaxStatic, L.abpMaxDynamic, L.ctx, L.abpMT != 0, L.maxAggregates, L.maxShapes, L.kineKine, L.staticKine);
    size_t op = 0;
    for (uint32_t f = 0; f <= L.frames.size(); ++f) {
      for (; op < L.ops.size() && L.ops[op].frame == f; ++op) apply(L.ops[op]);
      if (f == L.frames.size()) break;
      const BpFrame& F = L.frames[f];
      step(reinterpret_cast<const px::PxBounds3*>(L.bounds.data() + F.boundsStart), F.nBounds, F.boundsChanged != 0, L.dist.data() + F.distStart, F.nDist,
           L.words.data() + F.wordsStart, F.nWords, F.hasContactDistanceChanged != 0);
      endStep();
    }
    return op == L.ops.size();
  }
};

}  // namespace scene
}  // namespace eng
