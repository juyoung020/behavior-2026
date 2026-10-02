// omni.physx 의 충돌 거르개(filter shader)를 그대로 옮긴 것.
// 원본: omni/extensions/runtime/source/omni.physx/plugins/PhysXScene.cpp:48-161 (태그 107.3-omni-and-physx-5.6.1)
// 원본은 omni.physx 내부 표(충돌 그룹 쌍, 거른 쌍)를 보는데 이 표는 OVD 에 없다 -> FilterSpec 으로 따로 받는다.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "PxPhysicsAPI.h"

namespace engine {

// omni.physx/plugins/PhysXTools.h / usdInterface 의 word 배치
//   word1 = 거른 쌍(FilteredPairsAPI) 번호, word2 = 충돌 그룹 번호, word3 = 접촉 수정/해 끄기 플래그
static const uint32_t CONTACT_MODIFY_SURFACE_VELOCITY = (1 << 1);  // omni.physx/plugins/Setup.h:37
static const uint32_t CONTACT_SOLVE_DISABLE = (1 << 2);            // omni.physx/plugins/Setup.h:38

inline uint64_t pair_key(uint32_t a, uint32_t b) {  // omni Pair<uint32_t>: 작은 값이 앞
  if (a > b) { uint32_t t = a; a = b; b = t; }
  return (uint64_t(a) << 32) | b;
}

struct FilterSpec {
  std::unordered_set<uint64_t> group_pairs;     // CollisionGroupsPairsSet
  std::unordered_set<uint64_t> filtered_pairs;  // FilteredPairsSet (개수 > 0 인 것)
  bool inverted_group_filter = false;           // OmniGibson: set_invert_collision_group_filter(False)
  bool any_contact_report = false;              // scene->getContactReport()->empty() 의 반대
  // 경로 기반 재료 (capture 가 USD 에서 뜬 것). 번호는 재생 중 OVD 이름으로 푼다 (ovd_replay resolve_filters)
  struct Group { std::string path; std::vector<std::string> filtered, includes; };
  std::vector<Group> groups;
  std::vector<std::pair<std::string, std::string>> rels;  // FilteredPairsAPI (prim, target)
};

inline physx::PxFilterFlags OmniFilterShader(physx::PxFilterObjectAttributes attributes0, physx::PxFilterData filterData0,
                                             physx::PxFilterObjectAttributes attributes1, physx::PxFilterData filterData1,
                                             physx::PxPairFlags& pairFlags, const void* constantBlock, physx::PxU32) {
  using namespace physx;
  pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eDETECT_CCD_CONTACT;
  if (PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1)) {
    pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    return PxFilterFlags();
  }
  const FilterSpec* spec = *(const FilterSpec* const*)constantBlock;
  if (filterData0.word2 && filterData1.word2) {
    const bool found = spec->group_pairs.count(pair_key(filterData0.word2, filterData1.word2)) != 0;
    if (!spec->inverted_group_filter) {
      if (found) return PxFilterFlag::eKILL;
    } else {
      if (!found) return PxFilterFlag::eKILL;
    }
  }
  if ((filterData0.word3 & CONTACT_MODIFY_SURFACE_VELOCITY) || (filterData1.word3 & CONTACT_MODIFY_SURFACE_VELOCITY))
    pairFlags = pairFlags | PxPairFlag::eMODIFY_CONTACTS;
  if ((filterData0.word3 & CONTACT_SOLVE_DISABLE) || (filterData1.word3 & CONTACT_SOLVE_DISABLE))
    return PxFilterFlag::eCALLBACK;
  if (filterData0.word1 && filterData1.word1) {
    if (spec->filtered_pairs.count(pair_key(filterData0.word1, filterData1.word1))) return PxFilterFlag::eKILL;
  }
  if (spec->any_contact_report) return PxFilterFlag::eCALLBACK;
  return PxFilterFlag::eDEFAULT;
}

// omni OmniFilterCallback::pairFound 중 동역학에 영향을 주는 부분만 옮김.
// 접촉 보고 대상 쌍(checkPair)은 알림 플래그만 붙이므로(동역학 무관) 여기서는 보고 대상이 없다고 본다.
// 단, 키네마틱/정적끼리 쌍의 eSOLVE_CONTACT 제거는 보고 대상일 때만 일어나며 두 쪽 다 무한 질량이라 결과에 영향 없음.
class OmniFilterCallback : public physx::PxSimulationFilterCallback {
 public:
  physx::PxFilterFlags pairFound(physx::PxU64, physx::PxFilterObjectAttributes attributes0, physx::PxFilterData filterData0,
                                 const physx::PxActor* a0, const physx::PxShape* s0, physx::PxFilterObjectAttributes attributes1,
                                 physx::PxFilterData filterData1, const physx::PxActor* a1, const physx::PxShape* s1,
                                 physx::PxPairFlags& pairFlags) override {
    using namespace physx;
    if (a0->getConcreteType() == 0xFFFF || a1->getConcreteType() == 0xFFFF) return PxFilterFlags();
    if (s0->getFlags() & PxShapeFlag::eTRIGGER_SHAPE || s1->getFlags() & PxShapeFlag::eTRIGGER_SHAPE)
      pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    if ((filterData0.word3 & CONTACT_SOLVE_DISABLE) || (filterData1.word3 & CONTACT_SOLVE_DISABLE))
      pairFlags &= ~PxPairFlag::eSOLVE_CONTACT;
    if (report_all) {  // omni PhysXScene.cpp:149 checkPair 가 참인 쌍 (radio: 접촉 보고 prim 416 개 ≈ 모든 몸체 -> 모든 쌍으로 근사)
      const bool k0 = PxFilterObjectIsKinematic(attributes0), k1 = PxFilterObjectIsKinematic(attributes1);
      const bool st0 = PxGetFilterObjectType(attributes0) == PxFilterObjectType::eRIGID_STATIC;
      const bool st1 = PxGetFilterObjectType(attributes1) == PxFilterObjectType::eRIGID_STATIC;
      pairFlags = pairFlags | PxPairFlag::eNOTIFY_TOUCH_LOST | PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                  PxPairFlag::eNOTIFY_CONTACT_POINTS;
      if ((k0 || st0) && (k1 || st1)) {
        pairFlags &= ~PxPairFlag::eSOLVE_CONTACT;
        pairFlags |= PxPairFlag::eDETECT_DISCRETE_CONTACT;
      }
      return PxFilterFlags();
    }
    if (!diag_sub.empty()) {  // 진단(ovd_replay --trace-obj): 알림 플래그만 더한다
      const char* n0 = a0->getName();
      const char* n1 = a1->getName();
      if ((n0 && std::string(n0).find(diag_sub) != std::string::npos) || (n1 && std::string(n1).find(diag_sub) != std::string::npos))
        pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
    }
    return PxFilterFlags();
  }
  std::string diag_sub;
  bool report_all = false;  // --contact-report-all
  void pairLost(physx::PxU64, physx::PxFilterObjectAttributes, physx::PxFilterData, physx::PxFilterObjectAttributes,
                physx::PxFilterData, bool) override {}
  bool statusChange(physx::PxU64&, physx::PxPairFlags&, physx::PxFilterFlags&) override { return false; }
};

}  // namespace engine
