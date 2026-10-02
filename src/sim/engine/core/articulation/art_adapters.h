// articulation 모듈: 다른 모듈이 관절체에 붙을 때 쓰는 접근자. PhysX FeatherstoneArticulation 의 같은 이름 함수와 같은 뜻·같은 연산.
// joints 의 Art 인자(core/joints/tgs_1d.h:225,597 NoArt 와 같은 모양) — setupSolverConstraintStep / solveExt1D(Step) 가 부른다.
// 링크 번호는 LL(BFS) 번호 (a.ll[생성 번호]).
#pragma once
#include <cstdint>

#include "art_step.h"

namespace eng {
namespace art {

struct ArtRef {
  Articulation* a;
  EHD float getCfm(uint32_t link) const { return art::getCfm(*a, link); }
  EHD void getImpulseResponse(uint32_t link, const V3& impLin, const V3& impAng, V3& dvLin, V3& dvAng) const {
    art::getImpulseResponse(*a, link, impLin, impAng, dvLin, dvAng);
  }
  EHD void getLinkVelocity(uint32_t link, V3& lin, V3& ang) const { art::getLinkVelocity(*a, link, lin, ang); }
  EHD void getVelocity(uint32_t link, V3& lin, V3& ang) const { art::pxcFsGetVelocity(*a, link, nullptr, lin, ang); }
  EHD void getVelocities(uint32_t l0, uint32_t l1, V3& lin0, V3& ang0, V3& lin1, V3& ang1) const {
    art::pxcFsGetVelocities(*a, l0, l1, lin0, ang0, lin1, ang1);
  }
  EHD void getMotionVector(uint32_t link, V3& lin, V3& ang) const { art::getLinkMotionVector(*a, link, lin, ang); }
  EHD Q getDeltaQ(uint32_t link) const { return art::getDeltaQ(*a, link); }
  EHD float getLinkMaxPenBias(uint32_t link) const { return art::getLinkMaxPenBias(*a, link); }
  EHD void applyImpulse(uint32_t link, const V3& lin, const V3& ang) const { art::pxcFsApplyImpulse(*a, link, lin, ang, nullptr); }
  EHD void applyImpulses(uint32_t l0, const V3& lin0, const V3& ang0, uint32_t l1, const V3& lin1, const V3& ang1) const {
    art::pxcFsApplyImpulses(*a, l0, lin0, ang0, nullptr, l1, lin1, ang1, nullptr);
  }
  // 같은 관절체 두 링크의 자기 응답 (getImpulseSelfResponse, 부모-자식이면 빠른 길, 아니면 공통 조상까지)
  EHD void getImpulseSelfResponse(uint32_t l0, const V3& lin0, const V3& ang0, V3& dv0Lin, V3& dv0Ang, uint32_t l1, const V3& lin1, const V3& ang1,
                                  V3& dv1Lin, V3& dv1Ang) const {
    art::getImpulseSelfResponse(*a, l0, lin0, ang0, dv0Lin, dv0Ang, l1, lin1, ang1, dv1Lin, dv1Ang);
  }
};

}  // namespace art
}  // namespace eng
