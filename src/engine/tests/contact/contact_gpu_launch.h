// contact GPU 시험: 커널 실행 감싸개 선언 (contact_gpu_launch.cu). 시간은 reps 번 평균(ms).
#pragma once
#include <cstdint>

#include "core/contact/narrowphase.h"

namespace cxt {
int gpuUploadApprox();  // rcpps/rsqrtps 표를 이 CPU 에서 떠 GPU 로 (cudaError_t 값)
void gpuMathSweep(uint32_t lo, uint32_t n, uint32_t* dRcp, uint32_t* dRsq, uint32_t* dAcos);
void gpuInitManifolds(eng::px::Gu::LargePersistentContactManifold* dMan, size_t n);
float gpuConvexConvex(int envs, int threads, const eng::contact::ConvexPair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                      eng::px::Gu::LargePersistentContactManifold* dMan, eng::contact::PairResult* dOut, float contactDist,
                      float meshMargin, float toleranceLength, int reps);
void gpuInitSlots(const eng::contact::ShapePair* dPairs, int nPairs, eng::contact::ManifoldSlot* dSlots, int envs);
float gpuShapePairs(int envs, int threads, const eng::contact::ShapePair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                    eng::contact::ManifoldSlot* dSlots, eng::contact::PairResult* dOut, float contactDist, float meshMargin,
                    float toleranceLength, int reps);
}  // namespace cxt
