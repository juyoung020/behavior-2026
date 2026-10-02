// articulation 층 2: 판 N 개를 한 커널로 (CUDA). core/articulation/ 의 층 1 함수(stepAlone 등)를 그대로 부른다.
// 배치: 스레드 하나 = (관절체 k, 판 e) 하나, 상태 배열은 arts[k * E + e] (같은 관절체의 이웃 판이 한 워프 -> 제어 흐름이 거의 같다).
// 관절체만 있는 섬(접촉·조인트 없음)은 서로 독립이라 판 안 순서가 결과에 영향이 없다. 접촉·조인트가 섞인 섬은 solver 가
// 판 = 블록 하나 안에서 PhysX 순서로 articulation 함수(DyArticulationPImpl 목록, 문서 16.4)를 부른다.
// 필요: cudaDeviceSetLimit(cudaLimitStackSize, 8192) (링크 트리 재귀), 컴파일 -fmad=false -prec-div=true -prec-sqrt=true -ftz=true.
#pragma once

#include "../../core/articulation/art_step.h"

namespace eng {
namespace art {

// in(a, k, e, s): 스텝 s 의 API 입력(드라이브 목표·applyCache 등)을 관절체 a 에 넣는다 (호스트·GPU 공용 함수 객체, 값으로 복사된다).
template <class Inputs>
__global__ void kStepAloneBatch(Articulation* arts, int K, int E, int s0, int nSteps, StepParams sp, Inputs in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * E) return;
  const int k = idx / E, e = idx % E;
  Articulation& a = arts[size_t(idx)];
  for (int s = s0; s < s0 + nSteps; ++s) {
    in(a, k, e, s);
    stepAlone(a, sp);
  }
}

// 입력 없음 (자유 낙하·드라이브 목표 고정)
struct NoInputs {
  EHD void operator()(Articulation&, int, int, int) const {}
};

template <class Inputs>
inline cudaError_t launchStepAloneBatch(Articulation* dArts, int K, int E, int s0, int nSteps, const StepParams& sp, const Inputs& in,
                                        int threads = 64, cudaStream_t st = 0) {
  const size_t n = size_t(K) * size_t(E);
  kStepAloneBatch<Inputs><<<unsigned((n + size_t(threads) - 1) / size_t(threads)), unsigned(threads), 0, st>>>(dArts, K, E, s0, nSteps, sp, in);
  return cudaGetLastError();
}

}  // namespace art
}  // namespace eng
