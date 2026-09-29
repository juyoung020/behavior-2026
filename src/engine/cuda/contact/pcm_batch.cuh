// 층 2: 좁은 단계 PCM 을 판(env) N 개 동시에. 판 하나 = CUDA 블록 하나, 블록 안 스레드가 그 판의 쌍들을 나눠 맡는다
// (쌍끼리는 서로 독립이라 순서가 결과를 바꾸지 않는다. 결과는 쌍 번호 자리에 쓴다 -> PhysX 쌍 순서 그대로).
// 같은 함수(core/contact/narrowphase.h)를 호스트에서 부르면 층 1 이다.
// 근사 명령(rcpps/rsqrtps)을 쓰는 모양(상자-상자 등)은 실행 전에 px::uploadApproxTables() (core/contact/px/approx.h) 를 불러야 한다.
#pragma once
#include <new>

#include "core/contact/narrowphase.h"

namespace eng {
namespace contact {

__global__ void kInitManifolds(px::Gu::LargePersistentContactManifold* man, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  px::Gu::LargePersistentContactManifold* m = new (&man[i]) px::Gu::LargePersistentContactManifold();
  m->clearManifold();
}

// poses/man/out: [env][pair] 순서. pairs: 판끼리 공유.
__global__ void kConvexConvexBatch(const ConvexPair* pairs, int nPairs, const PairPose* poses,
                                   px::Gu::LargePersistentContactManifold* man, PairResult* out, float contactDist,
                                   float meshMargin, float toleranceLength) {
  const int env = blockIdx.x;
  const px::Gu::NarrowPhaseParams np(contactDist, meshMargin, toleranceLength);
  px::PxContactBuffer buf;  // 스레드마다 256 칸 (PhysX 와 같은 작업 공간)
  for (int p = threadIdx.x; p < nPairs; p += blockDim.x) {
    const size_t k = size_t(env) * nPairs + p;
    convexConvexPair(pairs[p], poses[k], np, man[k], buf, out[k]);
  }
}

// 다양체 칸 초기화: 쌍 모양 종류에 따라 (PxsContext::createCache 와 같게)
__global__ void kInitSlots(const ShapePair* pairs, int nPairs, ManifoldSlot* slots, int nEnvs) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(nPairs) * nEnvs) return;
  const ShapePair& s = pairs[i % nPairs];
  const int t0 = s.s0.type < s.s1.type ? s.s0.type : s.s1.type;
  const int t1 = s.s0.type < s.s1.type ? s.s1.type : s.s0.type;
  initManifold(slots[i], t0, t1);
}

__global__ void kShapePairBatch(const ShapePair* pairs, int nPairs, const PairPose* poses, ManifoldSlot* slots, PairResult* out,
                                float contactDist, float meshMargin, float toleranceLength) {
  const int env = blockIdx.x;
  px::PxContactBuffer buf;
  for (int p = threadIdx.x; p < nPairs; p += blockDim.x) {
    const size_t k = size_t(env) * nPairs + p;
    shapePair(pairs[p], poses[k], contactDist, meshMargin, toleranceLength, slots[k], buf, out[k]);
  }
}

}  // namespace contact
}  // namespace eng
