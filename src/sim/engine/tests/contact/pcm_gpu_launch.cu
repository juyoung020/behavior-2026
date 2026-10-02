// contact GPU 시험용 커널 실행 감싸개 (nvcc 로만 컴파일, PhysX 헤더를 안 봄). 시험 본문(.cpp, clang)은 이 함수들만 부른다.
#include <cuda_runtime.h>

#include "cuda/contact/pcm_batch.cuh"
#include "pcm_gpu_launch.h"

namespace cxt {

void gpuInitManifolds(eng::px::Gu::LargePersistentContactManifold* dMan, size_t n) {
  eng::contact::kInitManifolds<<<unsigned((n + 255) / 256), 256>>>(dMan, int(n));
}

float gpuConvexConvex(int envs, int threads, const eng::contact::ConvexPair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                      eng::px::Gu::LargePersistentContactManifold* dMan, eng::contact::PairResult* dOut, float contactDist,
                      float meshMargin, float toleranceLength, int reps) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int r = 0; r < reps; ++r)
    eng::contact::kConvexConvexBatch<<<envs, threads>>>(dPairs, nPairs, dPoses, dMan, dOut, contactDist, meshMargin, toleranceLength);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / float(reps);
}

}  // namespace cxt
