// contact GPU 시험: 커널 실행 감싸개 선언 (pcm_gpu_launch.cu).
#pragma once
#include "core/contact/narrowphase.h"

namespace cxt {
void gpuInitManifolds(eng::px::Gu::LargePersistentContactManifold* dMan, size_t n);
// reps 번 실행한 평균 커널 시간(ms)
float gpuConvexConvex(int envs, int threads, const eng::contact::ConvexPair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                      eng::px::Gu::LargePersistentContactManifold* dMan, eng::contact::PairResult* dOut, float contactDist,
                      float meshMargin, float toleranceLength, int reps);
}  // namespace cxt
