// 층 2 시험: 조인트 섬(D6 조인트 준비·반복 풀이·적분·되쓰기) CUDA 판 N 개 = 층 1 C++ (같은 코드를 호스트에서), 비트 비교 + 처리량.
// 층 1 = PhysX 는 test_joints_prep 가 증명한다(같은 함수들). 여기서는 층 1 == 층 2.
//   test_joints_gpu [--envs E] [--seed S] [--cone 0|1] [--reps R]
//   --cone 1: 원뿔 한계(tanf 사용 — GPU 는 아직 glibc 이식본 없음)도 섞는다. 기본은 뺀다.
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

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

int main(int argc, char** argv) {
  int envs = 65536, seed = 3, cone = 0, reps = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cone") && i + 1 < argc) cone = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
  }
  // ---- 판 만들기 (호스트, 층 1 API)
  std::vector<JEnvIn> in(envs);
  std::vector<int> kinds(size_t(envs) * MAXJ, -1);
  jgen::Gen g(static_cast<uint32_t>(seed));
  uint64_t njTotal = 0;
  const V3 grav{0, 0, -9.81f};
  for (int e = 0; e < envs; ++e) {
    JEnvIn& E = in[e];
    memset(&E, 0, sizeof E);
    E.nb = 1 + g.i(MAXB - 1);
    E.nj = 1 + g.i(MAXJ);
    E.posIters = 1 + g.i(8);
    E.velIters = 1 + g.i(4);
    E.simDt = 1.0f / 120.0f;
    jtest::worldBody(E.v[0], E.t[0], E.d[0]);
    E.frame[0] = Tf{qid(), V3{0, 0, 0}};
    Tf com[MAXB];
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
      int kind = g.i(jgen::K_COUNT);
      if (!cone && kind == jgen::K_SPHERE_CONE) kind = jgen::K_SPHERE;
      kinds[size_t(e) * MAXJ + k] = kind;
      const uint32_t a = 1 + uint32_t(g.i(int(E.nb)));
      uint32_t b = uint32_t(g.i(int(E.nb) + 1));  // 0 = 세계
      if (b == a) b = 0;
      const Tf f0{g.q(), g.v(0.2f)}, f1{g.q(), g.v(0.2f)};
      jnt::D6Joint j = jnt::createD6(jnt::ACTOR_DYNAMIC, a, com[a], f0, b ? jnt::ACTOR_DYNAMIC : jnt::ACTOR_NONE, b,
                                     b ? com[b] : Tf{qid(), V3{0, 0, 0}}, f1, 1.0f);
      for (const jgen::Cmd& c : g.joint(kind)) jgen::applyEng(j, c);
      if (!cone && (j.data.mUseConeLimit)) j.data.mUseConeLimit = false;  // 원뿔 한계 빼기 (--cone 0)
      E.data[k] = jnt::prepareData(j);
      E.flags[k] = j.constraintFlags;
      E.a[k] = uint8_t(a); E.b[k] = uint8_t(b);
      E.lb[k] = j.linBreakForce; E.ab[k] = j.angBreakForce; E.mrt[k] = j.minResponseThreshold;
      njTotal++;
    }
  }

  // ---- 층 1 (호스트, FTZ/DAZ 켬 = PhysX simulate 안과 같게)
  std::vector<JEnvOut> o1(envs), o2(envs);
  memset(o1.data(), 0, sizeof(JEnvOut) * size_t(envs));
  memset(o2.data(), 0, sizeof(JEnvOut) * size_t(envs));
  double cpuSec;
  {
    FtzScope ftz;
    const auto t0 = std::chrono::steady_clock::now();
    for (int e = 0; e < envs; ++e) runEnv(in[e], o1[e]);
    cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // ---- 층 2 (GPU)
  JEnvIn* dIn;
  JEnvOut* dOut;
  CK(cudaMalloc(&dIn, sizeof(JEnvIn) * size_t(envs)));
  CK(cudaMalloc(&dOut, sizeof(JEnvOut) * size_t(envs)));
  CK(cudaMemcpy(dIn, in.data(), sizeof(JEnvIn) * size_t(envs), cudaMemcpyHostToDevice));
  const int bs = 64, gs = (envs + bs - 1) / bs;
  float bestMs = 1e30f;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(dOut, 0, sizeof(JEnvOut) * size_t(envs)));
    CK(cudaEventRecord(e0));
    kJointIsland<<<gs, bs>>>(dIn, dOut, envs);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < bestMs) bestMs = ms;
  }
  CK(cudaMemcpy(o2.data(), dOut, sizeof(JEnvOut) * size_t(envs), cudaMemcpyDeviceToHost));

  // ---- 비교
  uint64_t cmpB = 0, badB = 0, cmpC = 0, badC = 0, cmpW = 0, badW = 0;
  int64_t firstEnv = -1;
  int badKind[jgen::K_COUNT] = {0};
  for (int e = 0; e < envs; ++e) {
    const JEnvIn& E = in[e];
    bool envBad = false;
    for (uint32_t b = 1; b <= E.nb; ++b) {
      cmpB++;
      if (memcmp(&o1[e].v[b], &o2[e].v[b], sizeof(TgsBodyVel)) || memcmp(&o1[e].t[b], &o2[e].t[b], sizeof(TgsTxInertia))) { badB++; envBad = true; }
    }
    for (uint32_t k = 0; k < E.nj; ++k) {
      cmpC++;
      if (o1[e].len[k] != o2[e].len[k] || memcmp(o1[e].blk[k], o2[e].blk[k], o1[e].len[k])) {
        badC++;
        envBad = true;
        badKind[kinds[size_t(e) * MAXJ + k]]++;
      }
      cmpW++;
      if (memcmp(&o1[e].wb[k], &o2[e].wb[k], sizeof(Writeback))) { badW++; envBad = true; }
    }
    if (envBad && firstEnv < 0) firstEnv = e;
  }
  printf("\n조인트 섬 층 1(C++) vs 층 2(CUDA): 판 %d 개, 조인트 %" PRIu64 " 개 (씨앗 %d, 원뿔 %s)\n", envs, njTotal, seed, cone ? "포함" : "뺌");
  printf("  몸체 속도·자세 비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpB, badB);
  printf("  풀이 제약 블록   비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpC, badC);
  printf("  되쓰기           비교 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", cmpW, badW);
  if (firstEnv >= 0) {
    printf("  첫 다름 판 %" PRId64 ", 다른 조인트 종류:", firstEnv);
    for (int k = 0; k < jgen::K_COUNT; ++k) if (badKind[k]) printf(" %s %d", jgen::kKindNames[k], badKind[k]);
    printf("\n");
  }
  const double gpuSec = bestMs * 1e-3;
  printf("  처리량: GPU %.3f ms (판·simulate %.3g/초, 조인트·simulate %.3g/초), CPU 단일 스레드 %.3f s (판·simulate %.3g/초) -> %.0f 배\n", bestMs,
         envs / gpuSec, double(njTotal) / gpuSec, cpuSec, envs / cpuSec, cpuSec / gpuSec);
  const bool ok = !badB && !badC && !badW;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  cudaFree(dIn);
  cudaFree(dOut);
  return ok ? 0 : 3;
}
