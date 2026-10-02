// 층 2 시험: 4개 묶음(SIMD 4칸) 조인트 판 CUDA = 층 1 C++ (같은 runEnv4 를 호스트에서), 비트 비교 + 처리량.
// 층 0(PhysX) == 층 1 은 test_joints_block4 가 증명한다. 판 하나 = 공식 radio 처럼 정적/세계-동적 조인트가 4개씩 묶이는 분할 한 개.
//   test_joints_block4_gpu [--envs E] [--seed S] [--reps R]
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda/joints/joint_island.cuh"
#include "joint_gen.h"

using namespace eng;
using namespace eng::jcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

int main(int argc, char** argv) {
  int envs = 32768, seed = 3, reps = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
  }
  std::vector<JEnv4In> in(envs);
  jgen::Gen g(static_cast<uint32_t>(seed));
  uint64_t njTotal = 0;
  const V3 grav{0, 0, -9.81f};
  for (int e = 0; e < envs; ++e) {
    JEnv4In& E = in[e];
    memset(&E, 0, sizeof E);
    E.nb = MAXB4 - 1;
    E.nj = g.i(4) == 0 ? 4u + uint32_t(g.i(4)) : uint32_t(MAXJ4);  // 가끔 끝 묶음이 4개 미만(하나씩)
    E.posIters = 1 + g.i(8);
    E.velIters = 1 + g.i(4);
    E.simDt = 1.0f / 120.0f;
    jtest::worldBody(E.v[0], E.t[0], E.d[0]);
    E.frame[0] = Tf{qid(), V3{0, 0, 0}};
    Tf com[MAXB4];
    for (uint32_t b = 1; b <= E.nb; ++b) {
      E.frame[b] = Tf{g.q(), g.v(2.0f)};
      com[b] = normalized(Tf{g.q(), g.v(0.1f)});
      V3 lv = g.v(2.0f), av = g.v(3.0f);
      const float invMass = 1.0f / (0.1f + 10.0f * g.p());
      const V3 invI{1.0f / (0.01f + g.p()), 1.0f / (0.01f + g.p()), 1.0f / (0.01f + g.p())};
      jtest::bodyCoreComputeUnconstrainedVelocity(grav, E.simDt, 0.0f, 0.05f, 1.0f, 1e32f, 1e4f, lv, av, false);
      jtest::copyToSolverBodyDataStep(lv, av, invMass, invI, E.frame[b], -1e32f, 1e32f, b, 3.4e38f, 1e4f, 0, false, E.v[b], E.t[b], E.d[b],
                                      E.simDt, g.i(3) == 0);
    }
    for (uint32_t k = 0; k < E.nj; ++k) {
      int kind = g.i(2) == 0 ? jgen::K_FIXED : g.i(jgen::K_COUNT);  // radio 는 전부 고정
      if (kind == jgen::K_SPHERE_CONE) kind = jgen::K_SPHERE;          // 원뿔 한계(tanf) 는 GPU 이식 전
      const uint32_t a = 1 + 2 * (k % 4);
      const uint32_t b = g.i(3) == 0 ? a + 1 : 0;                     // 대부분 세계(정적)-동적
      const Tf f0{g.q(), g.v(0.2f)}, f1{g.q(), g.v(0.2f)};
      jnt::D6Joint j = jnt::createD6(jnt::ACTOR_DYNAMIC, a, com[a], f0, b ? jnt::ACTOR_DYNAMIC : jnt::ACTOR_NONE, b,
                                     b ? com[b] : Tf{qid(), V3{0, 0, 0}}, f1, 1.0f);
      for (const jgen::Cmd& c : g.joint(kind)) jgen::applyEng(j, c);
      if (g.i(32) == 0) for (int ax = 0; ax < 6; ++ax) { jgen::Cmd c{jgen::MOTION}; c.a = ax; c.b = 2; jgen::applyEng(j, c); }  // 가끔 행 0 개
      if (j.data.mUseConeLimit) j.data.mUseConeLimit = false;
      E.data[k] = jnt::prepareData(j);
      E.flags[k] = j.constraintFlags;
      E.a[k] = uint8_t(a); E.b[k] = uint8_t(b);
      E.lb[k] = j.linBreakForce; E.ab[k] = j.angBreakForce; E.mrt[k] = j.minResponseThreshold;
      njTotal++;
    }
  }

  std::vector<JEnv4Out> o1(envs), o2(envs);
  memset(o1.data(), 0, sizeof(JEnv4Out) * size_t(envs));
  memset(o2.data(), 0, sizeof(JEnv4Out) * size_t(envs));
  double cpuSec;
  {
    FtzScope ftz;
    const auto t0 = std::chrono::steady_clock::now();
    for (int e = 0; e < envs; ++e) runEnv4(in[e], o1[e]);
    cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  JEnv4In* dIn;
  JEnv4Out* dOut;
  CK(cudaMalloc(&dIn, sizeof(JEnv4In) * size_t(envs)));
  CK(cudaMalloc(&dOut, sizeof(JEnv4Out) * size_t(envs)));
  CK(cudaMemcpy(dIn, in.data(), sizeof(JEnv4In) * size_t(envs), cudaMemcpyHostToDevice));
  const int bs = 64, gs = (envs + bs - 1) / bs;
  float bestMs = 1e30f;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(dOut, 0, sizeof(JEnv4Out) * size_t(envs)));
    CK(cudaEventRecord(e0));
    kJointIsland4<<<gs, bs>>>(dIn, dOut, envs);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < bestMs) bestMs = ms;
  }
  CK(cudaMemcpy(o2.data(), dOut, sizeof(JEnv4Out) * size_t(envs), cudaMemcpyDeviceToHost));

  uint64_t cmpB = 0, badB = 0, cmpC = 0, badC = 0, cmpW = 0, badW = 0, n4 = 0;
  int64_t firstEnv = -1;
  for (int e = 0; e < envs; ++e) {
    const JEnv4In& E = in[e];
    const JEnv4Out& A = o1[e];
    const JEnv4Out& B = o2[e];
    bool envBad = false;
    for (uint32_t b = 1; b <= E.nb; ++b) {
      cmpB++;
      if (memcmp(&A.v[b], &B.v[b], sizeof(TgsBodyVel)) || memcmp(&A.t[b], &B.t[b], sizeof(TgsTxInertia))) { badB++; envBad = true; }
    }
    for (uint32_t q = 0; q < (E.nj + 3) / 4; ++q) n4 += A.isBlock4[q];
    for (uint32_t k = 0; k < E.nj; ++k) {
      cmpC++;
      const bool lenBad = A.len[k] != B.len[k] || A.off[k] != B.off[k];
      // 4묶음 블록의 pad0[1..3] 은 원본도 안 쓰는 바이트지만 우리는 0 을 쓰므로 그대로 비교한다
      if (lenBad || (A.len[k] && memcmp(A.arena + A.off[k], B.arena + B.off[k], A.len[k]))) { badC++; envBad = true; }
      cmpW++;
      if (memcmp(&A.wb[k], &B.wb[k], sizeof(Writeback))) { badW++; envBad = true; }
    }
    if (envBad && firstEnv < 0) firstEnv = e;
  }
  printf("\n4개 묶음 조인트 판 층 1(C++) vs 층 2(CUDA): 판 %d 개, 조인트 %" PRIu64 " 개, 4묶음 %" PRIu64 " 개 (씨앗 %d)\n", envs, njTotal, n4, seed);
  printf("  몸체 속도·자세 비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpB, badB);
  printf("  풀이 제약 블록   비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpC, badC);
  printf("  되쓰기           비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpW, badW);
  if (firstEnv >= 0) printf("  첫 다름 판 %" PRId64 "\n", firstEnv);
  const double gpuSec = bestMs * 1e-3;
  printf("  처리량: GPU %.3f ms (판·simulate %.3g/초, 조인트·simulate %.3g/초), CPU 단일 스레드 %.3f s (판·simulate %.3g/초) -> %.0f 배\n", bestMs,
         envs / gpuSec, double(njTotal) / gpuSec, cpuSec, envs / cpuSec, cpuSec / gpuSec);
  const bool ok = !badB && !badC && !badW;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  cudaFree(dIn);
  cudaFree(dOut);
  return ok ? 0 : 3;
}
