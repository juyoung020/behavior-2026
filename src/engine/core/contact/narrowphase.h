// 좁은 단계 한 쌍 처리 (호스트·GPU 공용): 모양 쌍 + 두 자세 + 지속 다양체 -> PhysX PCM 접촉 함수 결과.
// PhysX 흐름: PxcNpBatch.cpp:368 discreteNarrowPhase -> g_PCMContactMethodTable[type0][type1] (모양 번호 작은 쪽이 먼저,
// 뒤집히면 법선 부호를 바꿈 flipContacts:65). 여기서는 모양별 PCM 함수 호출과 결과 복사만 한다.
// 접촉 관리자(쌍 목록·생성 순서)·재질·패치 압축은 이 위 층(pipeline)에서.
#pragma once
#include "core/contact/px/gu.h"

namespace eng {
namespace contact {

static constexpr int kMaxOutContacts = 8;  // 볼록-볼록 PCM 은 다양체 4점까지(GU_MANIFOLD_CACHE_SIZE). 넘치면 overflow 표시

struct ConvexPair {  // 판끼리 공유하는 정적 부분
  px::PxConvexMeshGeometry g0, g1;
};
struct PairPose {
  px::PxTransform32 tf0, tf1;
};
struct PairResult {
  uint32_t ret;       // PCM 함수 반환값
  uint32_t count;     // 접촉 수 (PxContactBuffer::count)
  uint32_t overflow;  // count > kMaxOutContacts
  uint32_t pad;
  px::PxContactPoint c[kMaxOutContacts];
};

// 볼록-볼록 한 쌍. buf 는 호출자가 준비(256 칸 작업 공간 겸 출력). man 은 쌍마다 지속되는 다양체.
EHD void convexConvexPair(const ConvexPair& s, const PairPose& p, const px::Gu::NarrowPhaseParams& np,
                          px::Gu::LargePersistentContactManifold& man, px::PxContactBuffer& buf, PairResult& out) {
  px::Gu::Cache cache;
  cache.setManifold(&man);
  buf.reset();
  out.ret = px::Gu::pcmContactConvexConvex(s.g0, s.g1, p.tf0, p.tf1, np, cache, buf, nullptr) ? 1u : 0u;
  out.count = buf.count;
  out.overflow = buf.count > uint32_t(kMaxOutContacts) ? 1u : 0u;
  const uint32_t n = buf.count < uint32_t(kMaxOutContacts) ? buf.count : uint32_t(kMaxOutContacts);
  for (uint32_t i = 0; i < n; ++i) out.c[i] = buf.contacts[i];
}

}  // namespace contact
}  // namespace eng
