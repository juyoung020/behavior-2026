// contact GPU 시험용 커널 실행 감싸개 (nvcc 로만 컴파일, PhysX 헤더를 안 봄). 시험 본문(.cpp, clang)은 이 함수들만 부른다.
#include <cuda_runtime.h>

#include "core/contact/px/approx.h"
#include "core/contact/px/glibc_acosf.h"
#include "cuda/contact/pcm_batch.cuh"
#include "contact_gpu_launch.h"

namespace cxt {

int gpuUploadApprox() { return int(eng::px::uploadApproxTables()); }

__global__ void kMathSweep(uint32_t lo, uint32_t n, uint32_t* rcp, uint32_t* rsq, uint32_t* acs) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = eng::px::em_u2f(lo + i);
  rcp[i] = eng::px::em_f2u(eng::px::em_rcp1(x));
  rsq[i] = eng::px::em_f2u(eng::px::em_rsqrt1(x));
  acs[i] = eng::px::em_f2u(eng::glibcx::acosf(x));
}

void gpuMathSweep(uint32_t lo, uint32_t n, uint32_t* dRcp, uint32_t* dRsq, uint32_t* dAcos) {
  kMathSweep<<<(n + 255) / 256, 256>>>(lo, n, dRcp, dRsq, dAcos);
  cudaDeviceSynchronize();
}

void gpuInitManifolds(eng::px::Gu::LargePersistentContactManifold* dMan, size_t n) {
  eng::contact::kInitManifolds<<<unsigned((n + 255) / 256), 256>>>(dMan, int(n));
}

static float timed(int reps, void (*launch)(void*), void* ctx) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int r = 0; r < reps; ++r) launch(ctx);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / float(reps);
}

struct CCtx {
  int envs, threads, nPairs;
  const eng::contact::ConvexPair* pairs;
  const eng::contact::PairPose* poses;
  eng::px::Gu::LargePersistentContactManifold* man;
  eng::contact::PairResult* out;
  float cd, mm, tl;
};
float gpuConvexConvex(int envs, int threads, const eng::contact::ConvexPair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                      eng::px::Gu::LargePersistentContactManifold* dMan, eng::contact::PairResult* dOut, float contactDist,
                      float meshMargin, float toleranceLength, int reps) {
  CCtx c{envs, threads, nPairs, dPairs, dPoses, dMan, dOut, contactDist, meshMargin, toleranceLength};
  return timed(reps, [](void* p) {
    CCtx& c = *static_cast<CCtx*>(p);
    eng::contact::kConvexConvexBatch<<<c.envs, c.threads>>>(c.pairs, c.nPairs, c.poses, c.man, c.out, c.cd, c.mm, c.tl);
  }, &c);
}

void gpuInitSlots(const eng::contact::ShapePair* dPairs, int nPairs, eng::contact::ManifoldSlot* dSlots, int envs) {
  const size_t n = size_t(nPairs) * envs;
  eng::contact::kInitSlots<<<unsigned((n + 255) / 256), 256>>>(dPairs, nPairs, dSlots, envs);
}

struct SCtx {
  int envs, threads, nPairs;
  const eng::contact::ShapePair* pairs;
  const eng::contact::PairPose* poses;
  eng::contact::ManifoldSlot* slots;
  eng::contact::PairResult* out;
  float cd, mm, tl;
};
float gpuShapePairs(int envs, int threads, const eng::contact::ShapePair* dPairs, int nPairs, const eng::contact::PairPose* dPoses,
                    eng::contact::ManifoldSlot* dSlots, eng::contact::PairResult* dOut, float contactDist, float meshMargin,
                    float toleranceLength, int reps) {
  SCtx c{envs, threads, nPairs, dPairs, dPoses, dSlots, dOut, contactDist, meshMargin, toleranceLength};
  return timed(reps, [](void* p) {
    SCtx& c = *static_cast<SCtx*>(p);
    eng::contact::kShapePairBatch<<<c.envs, c.threads>>>(c.pairs, c.nPairs, c.poses, c.slots, c.out, c.cd, c.mm, c.tl);
  }, &c);
}

}  // namespace cxt
