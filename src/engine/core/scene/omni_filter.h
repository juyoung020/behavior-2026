// omni.physx 충돌 거르개를 PhysX 형 없이 (문서 15.3 적재기, 리드): 셰이더 + pair-found 콜백.
// replay/omni_filter.h (PhysX 형 판, 재생기가 PhysX 에 거는 것)와 같은 식. 원본: omni.physx/plugins/PhysXScene.cpp:48-161 (107.3-omni-and-physx-5.6.1).
// 쓰는 곳: 쌍 관리층(core/contact/sc_pairs.h ScPairs::filterShader·filterPairFound). 표(충돌 그룹 쌍·거른 쌍)는 판마다 장면 파일에 담는다.
// 검증: replay/g1_scene.cpp 가 이 거르개로 쌍 관리층을 돌려 PhysX 와 비교 (G1_SCENE_FILTER=core).
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/contact/sc_pairs.h"

namespace eng {
namespace scene {

// PxPairFlag / PxFilterFlag / PxFilterObjectFlag (include/PxFiltering.h)
namespace omnif {
constexpr uint32_t eSOLVE_CONTACT = 1u << 0, eMODIFY_CONTACTS = 1u << 1, eNOTIFY_TOUCH_FOUND = 1u << 2, eNOTIFY_TOUCH_PERSISTS = 1u << 3,
                   eNOTIFY_TOUCH_LOST = 1u << 4, eNOTIFY_CONTACT_POINTS = 1u << 9, eDETECT_DISCRETE_CONTACT = 1u << 10, eDETECT_CCD_CONTACT = 1u << 11;
constexpr uint32_t eCONTACT_DEFAULT = eSOLVE_CONTACT | eDETECT_DISCRETE_CONTACT;
constexpr uint32_t eTRIGGER_DEFAULT = eNOTIFY_TOUCH_FOUND | eNOTIFY_TOUCH_LOST | eDETECT_DISCRETE_CONTACT;
constexpr uint32_t fKILL = 1u << 0, fCALLBACK = 1u << 2, fDEFAULT = 0;
constexpr uint32_t oKINEMATIC = 1u << 4, oTRIGGER = 1u << 5, oTYPE_MASK = 15u, oRIGID_STATIC = 0;
// omni.physx Setup.h:37-38 (filter word3)
constexpr uint32_t CONTACT_MODIFY_SURFACE_VELOCITY = 1u << 1, CONTACT_SOLVE_DISABLE = 1u << 2;
}  // namespace omnif

// 판마다 다른 표 (omni CollisionGroupsPairsSet·FilteredPairsSet, 번호는 모양 filter word2·word1 과 같은 번호 공간)
struct OmniFilterSpec {
  std::vector<uint64_t> groupPairs;     // 오름차순 (작은 번호가 위 32 비트)
  std::vector<uint64_t> filteredPairs;  // 오름차순
  uint8_t invertedGroupFilter = 0, anyContactReport = 0, reportAll = 0, pad = 0;
  static uint64_t key(uint32_t a, uint32_t b) {  // omni Pair<uint32_t>: 작은 값이 앞
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    return (uint64_t(a) << 32) | b;
  }
  static bool has(const std::vector<uint64_t>& v, uint64_t k) { return std::binary_search(v.begin(), v.end(), k); }
  void sort() {
    std::sort(groupPairs.begin(), groupPairs.end());
    groupPairs.erase(std::unique(groupPairs.begin(), groupPairs.end()), groupPairs.end());
    std::sort(filteredPairs.begin(), filteredPairs.end());
    filteredPairs.erase(std::unique(filteredPairs.begin(), filteredPairs.end()), filteredPairs.end());
  }
};

// 거르개 자료 (ScPairs::filterShaderData·filterCallbackData 에 같은 것을 건다)
struct OmniFilterCtx {
  const OmniFilterSpec* spec = nullptr;
  const contact::sc::ScPairs* pairs = nullptr;  // pair-found 가 모양 트리거 표시를 본다
};

// OmniFilterShader (PhysXScene.cpp:48)
inline uint32_t omniFilterShader(uint32_t attr0, const contact::sc::FilterData& f0, uint32_t attr1, const contact::sc::FilterData& f1, uint32_t& pairFlags,
                                 const void* data) {
  using namespace omnif;
  pairFlags = eCONTACT_DEFAULT | eDETECT_CCD_CONTACT;
  if ((attr0 & oTRIGGER) || (attr1 & oTRIGGER)) {
    pairFlags = eTRIGGER_DEFAULT;
    return fDEFAULT;
  }
  const OmniFilterSpec& spec = *static_cast<const OmniFilterCtx*>(data)->spec;
  if (f0.word2 && f1.word2) {
    const bool found = OmniFilterSpec::has(spec.groupPairs, OmniFilterSpec::key(f0.word2, f1.word2));
    if (!spec.invertedGroupFilter) {
      if (found) return fKILL;
    } else if (!found) {
      return fKILL;
    }
  }
  if ((f0.word3 & CONTACT_MODIFY_SURFACE_VELOCITY) || (f1.word3 & CONTACT_MODIFY_SURFACE_VELOCITY)) pairFlags |= eMODIFY_CONTACTS;
  if ((f0.word3 & CONTACT_SOLVE_DISABLE) || (f1.word3 & CONTACT_SOLVE_DISABLE)) return fCALLBACK;
  if (f0.word1 && f1.word1 && OmniFilterSpec::has(spec.filteredPairs, OmniFilterSpec::key(f0.word1, f1.word1))) return fKILL;
  if (spec.anyContactReport) return fCALLBACK;
  return fDEFAULT;
}

// OmniFilterCallback::pairFound 중 동역학에 닿는 부분 (replay/omni_filter.h 와 같음; 행위자 종류 0xFFFF 검사는 강체만 있어 생략)
inline uint32_t omniPairFound(uint64_t, uint32_t attr0, const contact::sc::FilterData& f0, int32_t, int32_t shape0, uint32_t attr1,
                              const contact::sc::FilterData& f1, int32_t, int32_t shape1, uint32_t& pairFlags, const void* data) {
  using namespace omnif;
  const OmniFilterCtx& c = *static_cast<const OmniFilterCtx*>(data);
  if (c.pairs->shapes[size_t(shape0)].trigger || c.pairs->shapes[size_t(shape1)].trigger) pairFlags = eTRIGGER_DEFAULT;
  if ((f0.word3 & CONTACT_SOLVE_DISABLE) || (f1.word3 & CONTACT_SOLVE_DISABLE)) pairFlags &= ~eSOLVE_CONTACT;
  if (c.spec->reportAll) {
    const bool k0 = (attr0 & oKINEMATIC) != 0, k1 = (attr1 & oKINEMATIC) != 0;
    const bool st0 = (attr0 & oTYPE_MASK) == oRIGID_STATIC, st1 = (attr1 & oTYPE_MASK) == oRIGID_STATIC;
    pairFlags |= eNOTIFY_TOUCH_LOST | eNOTIFY_TOUCH_FOUND | eNOTIFY_TOUCH_PERSISTS | eNOTIFY_CONTACT_POINTS;
    if ((k0 || st0) && (k1 || st1)) {
      pairFlags &= ~eSOLVE_CONTACT;
      pairFlags |= eDETECT_DISCRETE_CONTACT;
    }
  }
  return fDEFAULT;
}

// 쌍 관리층에 걸기
inline void setOmniFilter(contact::sc::ScPairs& M, OmniFilterCtx& ctx) {
  ctx.pairs = &M;
  M.filterShader = omniFilterShader;
  M.filterShaderData = &ctx;
  M.filterPairFound = omniPairFound;
  M.filterCallbackData = &ctx;
}

}  // namespace scene
}  // namespace eng
