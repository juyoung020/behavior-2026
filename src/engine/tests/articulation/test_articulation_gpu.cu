// 층 2 시험: 관절체 CUDA 판 N 개 = 층 1 C++ = PhysX 5.6.1 (비트 사슬) + 처리량.
// 무작위 관절체 K 개(random_art.h, 층 1 시험과 같은 생성기)를 판(env) E 개로 복제한다. 판 e 는 드라이브 목표 위상만 다르다(t + e*0.01).
//  1) 검증: 스텝마다 커널 1 번. 판 0..C-1 을 CPU 층 1 과 매 스텝 비트 비교, 판 0 의 CPU 층 1 은 PhysX 와 매 스텝 비트 비교.
//  2) 처리량: 판 E 개 x 스텝 S 를 커널 한 번에. 끝 상태 판 0..C-1 을 CPU 층 1 끝 상태와 비트 비교.
// GPU 배치: 스레드 하나 = (관절체, 판) 하나. 판 안 관절체들은 서로 독립(접촉 없는 섬)이라 순서가 결과에 영향이 없다.
//   같은 워프 = 같은 관절체의 이웃 판들 -> 제어 흐름이 거의 같다.
//   test_articulation_gpu [--arts K] [--links N] [--steps S] [--seed S] [--pos P] [--vel V] [--envs E] [--check C] [--benchenvs E2]
//                         [--spherical 0|1] (구면 관절: GPU 는 joints 의 glibc atan2f 이식본)
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#ifndef ENG_ART_MAX_LINKS
#define ENG_ART_MAX_LINKS 16
#endif
#ifndef ENG_ART_MAX_DOFS
#define ENG_ART_MAX_DOFS 32
#endif
#ifndef ENG_ART_MAX_PATH
#define ENG_ART_MAX_PATH 160
#endif
#ifndef ENG_ART_MAX_MIMIC
#define ENG_ART_MAX_MIMIC 4
#endif
#include "random_art.h"

using namespace physx;
using namespace artest;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));               \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

// 비교용 압축 상태: 링크마다 자세 7 + 선속도 3 + 각속도 3, dof 마다 위치·속도, 깸 카운터, 잠
constexpr uint32_t kRec = A::kMaxLinks * 13 + A::kMaxDofs * 2 + 2;
EHD void extract(const A::Articulation& a, float* r) {
  for (uint32_t i = 0; i < kRec; ++i) r[i] = 0.0f;
  for (uint32_t l = 0; l < a.nLinks; ++l) {
    const eng::Tf p = A::linkGlobalPose(a, l);
    const A::LinkBody& b = a.bodies[a.ll[l]];
    float* q = r + l * 13;
    q[0] = p.q.x; q[1] = p.q.y; q[2] = p.q.z; q[3] = p.q.w; q[4] = p.p.x; q[5] = p.p.y; q[6] = p.p.z;
    q[7] = b.linVel.x; q[8] = b.linVel.y; q[9] = b.linVel.z; q[10] = b.angVel.x; q[11] = b.angVel.y; q[12] = b.angVel.z;
  }
  float* j = r + A::kMaxLinks * 13;
  for (uint32_t d = 0; d < a.dofs; ++d) {
    j[d] = a.jointPosition[d];
    j[A::kMaxDofs + d] = a.jointVelocity[d];
  }
  r[kRec - 2] = a.wakeCounter;
  r[kRec - 1] = a.awake ? 0.0f : 1.0f;
}

__global__ void kStep(A::Articulation* arts, const ArtInputs* in, int K, int E, int s0, int nSteps, A::StepParams sp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * E) return;
  const int art = idx / E, env = idx % E;
  A::Articulation& a = arts[size_t(idx)];
  const ArtInputs& ai = in[art];
  for (int s = s0; s < s0 + nSteps; ++s) {
    stepInputsEng(a, ai, s, env, sp.dt);
    A::stepAlone(a, sp);
  }
}
__global__ void kExtract(const A::Articulation* arts, int K, int E, int C, float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // (관절체, 판 < C)
  if (idx >= K * C) return;
  const int art = idx / C, env = idx % C;
  extract(arts[size_t(art) * size_t(E) + size_t(env)], out + size_t(idx) * kRec);
}

static bool sameBits(const float* x, const float* y, uint32_t n, bool& anyNan) {
  bool same = true;
  anyNan = false;
  for (uint32_t j = 0; j < n; ++j) {
    const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
    if (nx || ny) {
      anyNan = true;
      if (nx != ny) same = false;
    } else if (memcmp(&x[j], &y[j], 4)) {
      same = false;
    }
  }
  return same;
}

int main(int argc, char** argv) {
  BuildOpts o;
  o.spherical = 1;
  o.nArts = 8;
  int steps = 600, E = 64, C = 8, benchEnvs = 2048, carveout = 0;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) o.nArts = atoi(argv[++i]);
    else if (arg("--links")) o.nLinksMax = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--seed")) o.seed = atoi(argv[++i]);
    else if (arg("--pos")) o.posIt = atoi(argv[++i]);
    else if (arg("--vel")) o.velIt = atoi(argv[++i]);
    else if (arg("--envs")) E = atoi(argv[++i]);
    else if (arg("--check")) C = atoi(argv[++i]);
    else if (arg("--benchenvs")) benchEnvs = atoi(argv[++i]);
    else if (arg("--spherical")) o.spherical = atoi(argv[++i]);
    else if (arg("--carveout")) carveout = atoi(argv[++i]);
    else if (arg("--r1pro")) o.r1pro = argv[++i];
    else if (arg("--r1copies")) o.r1copies = atoi(argv[++i]);
  }
  if (C > E) C = E;
  if (o.nLinksMax > int(A::kMaxLinks)) o.nLinksMax = int(A::kMaxLinks);
  const int K = o.nArts;
  const float dt = 1.0f / 120.0f;
  A::StepParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.dt = dt;
  sp.lengthScale = 1.0f;

  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(0);
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS | PxSceneFlag::eENABLE_STABILIZATION;
  PxScene* scene = phys->createScene(sd);
  std::vector<Mirror> arts;
  if (!buildRandom(phys, scene, o, arts)) {
    fprintf(stderr, "addToScene 실패\n");
    return 1;
  }
  uint32_t totLinks = 0, totDofs = 0;
  for (auto& m : arts) {
    totLinks += m.e->nLinks;
    totDofs += m.e->dofs;
  }
  printf("관절체 %d 개 (링크 %u, dof %u), sizeof(Articulation) = %zu B, 판 %d (검증 %d), 처리량 판 %d\n", K, totLinks, totDofs,
         sizeof(A::Articulation), E, C, benchEnvs);

  // 템플릿 (장면에 넣은 직후 상태) 과 입력
  std::vector<A::Articulation> tmpl(static_cast<size_t>(K));
  std::vector<ArtInputs> inputs(static_cast<size_t>(K));
  for (int k = 0; k < K; ++k) {
    tmpl[size_t(k)] = *arts[size_t(k)].e;
    inputs[size_t(k)] = arts[size_t(k)].in;
  }
  // CPU 층 1 판 0..C-1
  std::vector<A::Articulation> cpu(static_cast<size_t>(K) * static_cast<size_t>(C));
  for (int k = 0; k < K; ++k)
    for (int e = 0; e < C; ++e) cpu[size_t(k) * size_t(C) + size_t(e)] = tmpl[size_t(k)];

  CK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
  // 공유 메모리를 안 쓰므로 L1 을 최대로 (관절체 자료·스택이 L1 에 더 머문다). 점유율도 찍는다.
  CK(cudaFuncSetAttribute(kStep, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
  {
    int blocksPerSm = 0, nSm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kStep, 64, 0));
    CK(cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, 0));
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, kStep));
    printf("kStep: 레지스터 %d/스레드, 지역 메모리 %zu B, SM %d 개 x 블록(64) %d 개 = 동시 스레드 %d (L1 선호 carveout %d)\n", fa.numRegs,
           fa.localSizeBytes, nSm, blocksPerSm, nSm * blocksPerSm * 64, carveout);
  }
  A::Articulation* dArts = nullptr;
  ArtInputs* dIn = nullptr;
  float* dRec = nullptr;
  const size_t nV = size_t(K) * size_t(E);
  CK(cudaMalloc(&dArts, nV * sizeof(A::Articulation)));
  CK(cudaMalloc(&dIn, size_t(K) * sizeof(ArtInputs)));
  CK(cudaMalloc(&dRec, size_t(K) * size_t(C) * kRec * sizeof(float)));
  CK(cudaMemcpy(dIn, inputs.data(), size_t(K) * sizeof(ArtInputs), cudaMemcpyHostToDevice));
  for (int k = 0; k < K; ++k)
    for (int e = 0; e < E; ++e)
      CK(cudaMemcpy(dArts + size_t(k) * size_t(E) + size_t(e), &tmpl[size_t(k)], sizeof(A::Articulation), cudaMemcpyHostToDevice));

  std::vector<float> hRec(size_t(K) * size_t(C) * kRec), cRec(kRec), pRec(kRec);
  uint64_t cmpG = 0, badG = 0, nanG = 0, cmpP = 0, badP = 0, nanP = 0;
  int64_t firstG = -1, firstP = -1;
  std::vector<float> buf;
  const int threads = 64;
  double gpuMs = 0, cpuMs = 0, pxMs = 0;
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));
  for (int s = 1; s <= steps; ++s) {
    // 층 0: PhysX (판 0 입력)
    auto t0 = std::chrono::steady_clock::now();
    for (auto& m : arts) m.applyInputsPx(s, dt, buf);
    scene->simulate(dt);
    scene->fetchResults(true);
    auto t1 = std::chrono::steady_clock::now();
    pxMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    // 층 1: CPU 판 0..C-1
    for (int k = 0; k < K; ++k)
      for (int e = 0; e < C; ++e) stepInputsEng(cpu[size_t(k) * size_t(C) + size_t(e)], inputs[size_t(k)], s, e, dt);
    {
      FtzScope f;
      for (auto& a : cpu) A::stepAlone(a, sp);
    }
    auto t2 = std::chrono::steady_clock::now();
    cpuMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
    // 층 2: GPU 판 E 개
    CK(cudaEventRecord(ev0));
    kStep<<<unsigned((nV + threads - 1) / threads), threads>>>(dArts, dIn, K, E, s, 1, sp);
    CK(cudaEventRecord(ev1));
    CK(cudaGetLastError());
    CK(cudaEventSynchronize(ev1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, ev0, ev1));
    gpuMs += ms;
    kExtract<<<unsigned((K * C + threads - 1) / threads), threads>>>(dArts, K, E, C, dRec);
    CK(cudaMemcpy(hRec.data(), dRec, hRec.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < K; ++k) {
      for (int e = 0; e < C; ++e) {
        extract(cpu[size_t(k) * size_t(C) + size_t(e)], cRec.data());
        const float* g = &hRec[(size_t(k) * size_t(C) + size_t(e)) * kRec];
        bool anyNan;
        cmpG++;
        if (!sameBits(g, cRec.data(), kRec, anyNan)) {
          badG++;
          if (firstG < 0) {
            firstG = s;
            for (uint32_t i = 0; i < kRec; ++i)
              if (memcmp(&g[i], &cRec[i], 4)) {
                printf("[GPU!=CPU] step %d art %d env %d 칸 %u: GPU %.9g CPU %.9g\n", s, k, e, i, g[i], cRec[i]);
                break;
              }
          }
        } else if (anyNan) nanG++;
      }
      // 판 0 CPU vs PhysX
      Mirror& m = arts[size_t(k)];
      for (uint32_t i = 0; i < kRec; ++i) pRec[i] = 0.0f;
      for (uint32_t l = 0; l < m.pl.size(); ++l) {
        const PxTransform tp = m.pl[l]->getGlobalPose();
        const PxVec3 lv = m.pl[l]->getLinearVelocity(), av = m.pl[l]->getAngularVelocity();
        float* q = &pRec[l * 13];
        memcpy(q, &tp.q.x, 16);
        memcpy(q + 4, &tp.p.x, 12);
        memcpy(q + 7, &lv.x, 12);
        memcpy(q + 10, &av.x, 12);
      }
      const uint32_t nd = m.e->dofs;
      if (nd) {
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        memcpy(&pRec[A::kMaxLinks * 13], m.cache->jointPosition, nd * 4);
        memcpy(&pRec[A::kMaxLinks * 13 + A::kMaxDofs], m.cache->jointVelocity, nd * 4);
      }
      pRec[kRec - 2] = m.px->getWakeCounter();
      pRec[kRec - 1] = m.px->isSleeping() ? 1.0f : 0.0f;
      extract(cpu[size_t(k) * size_t(C)], cRec.data());
      bool anyNan;
      cmpP++;
      if (!sameBits(pRec.data(), cRec.data(), kRec, anyNan)) {
        badP++;
        if (firstP < 0) {
          firstP = s;
          for (uint32_t i = 0; i < kRec; ++i)
            if (memcmp(&pRec[i], &cRec[i], 4)) {
              printf("[PhysX!=CPU] step %d art %d 칸 %u: PhysX %.9g CPU %.9g\n", s, k, i, pRec[i], cRec[i]);
              break;
            }
        }
      } else if (anyNan) nanP++;
    }
  }
  printf("\n검증 (스텝 %d, 관절체 %d 개, 판 %d 개 GPU, 판 %d 개 비교):\n", steps, K, E, C);
  printf("  층2 GPU = 층1 CPU : 비교 %" PRIu64 " (관절체x판x스텝, 한 번에 %u 칸) 비트 다름 %" PRIu64 " 첫 다름 스텝 %" PRId64 " 양쪽 NaN %" PRIu64 "\n", cmpG,
         kRec, badG, firstG, nanG);
  printf("  층1 CPU = 층0 PhysX (판 0) : 비교 %" PRIu64 " 비트 다름 %" PRIu64 " 첫 다름 스텝 %" PRId64 " 양쪽 NaN %" PRIu64 "\n", cmpP, badP, firstP, nanP);
  printf("  시간: PhysX 판 1 개 %.1f ms, CPU 층1 판 %d 개 %.1f ms, GPU 판 %d 개 스텝마다 커널 %.1f ms\n", pxMs, C, cpuMs, E, gpuMs);
  CK(cudaFree(dArts));

  // ---- 처리량: 판 benchEnvs 개 x steps 를 커널 한 번에
  bool benchOk = true;
  if (benchEnvs > 0) {
    const size_t nB = size_t(K) * size_t(benchEnvs);
    A::Articulation* dB = nullptr;
    const size_t bytes = nB * sizeof(A::Articulation);
    CK(cudaMalloc(&dB, bytes));
    for (int k = 0; k < K; ++k) {
      CK(cudaMemcpy(dB + size_t(k) * size_t(benchEnvs), &tmpl[size_t(k)], sizeof(A::Articulation), cudaMemcpyHostToDevice));
      for (int filled = 1; filled < benchEnvs; filled *= 2) {  // 판 복제: 두 배씩
        const int n = std::min(filled, benchEnvs - filled);
        CK(cudaMemcpy(dB + size_t(k) * size_t(benchEnvs) + size_t(filled), dB + size_t(k) * size_t(benchEnvs),
                      size_t(n) * sizeof(A::Articulation), cudaMemcpyDeviceToDevice));
      }
    }
    CK(cudaEventRecord(ev0));
    kStep<<<unsigned((nB + threads - 1) / threads), threads>>>(dB, dIn, K, benchEnvs, 1, steps, sp);
    CK(cudaEventRecord(ev1));
    CK(cudaGetLastError());
    CK(cudaEventSynchronize(ev1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, ev0, ev1));
    const int Cb = std::min(C, benchEnvs);
    std::vector<float> bRec(size_t(K) * size_t(Cb) * kRec);
    float* dR2 = nullptr;
    CK(cudaMalloc(&dR2, bRec.size() * sizeof(float)));
    kExtract<<<unsigned((K * Cb + threads - 1) / threads), threads>>>(dB, K, benchEnvs, Cb, dR2);
    CK(cudaMemcpy(bRec.data(), dR2, bRec.size() * sizeof(float), cudaMemcpyDeviceToHost));
    uint64_t bBad = 0;
    for (int k = 0; k < K; ++k)
      for (int e = 0; e < Cb; ++e) {
        extract(cpu[size_t(k) * size_t(C) + size_t(e)], cRec.data());
        bool anyNan;
        if (!sameBits(&bRec[(size_t(k) * size_t(Cb) + size_t(e)) * kRec], cRec.data(), kRec, anyNan)) bBad++;
      }
    benchOk = bBad == 0;
    const double envSteps = double(benchEnvs) * double(steps);
    printf("\n처리량: 판 %d 개 x 스텝 %d (관절체 %d 개/판, 링크 %u/판) GPU %.1f ms -> 판·스텝/초 %.3g, 링크·스텝/초 %.3g (메모리 %.2f GB)\n",
           benchEnvs, steps, K, totLinks, ms, envSteps / (ms * 1e-3), envSteps * totLinks / (ms * 1e-3), double(bytes) / 1e9);
    printf("  대조: PhysX CPU 단일 스레드 판 1 개 %.1f ms -> 판·스텝/초 %.3g ; 끝 상태 판 %d 개 CPU 층1 과 비트 다름 %" PRIu64 "\n", pxMs,
           double(steps) / (pxMs * 1e-3), Cb, bBad);
    CK(cudaFree(dR2));
    CK(cudaFree(dB));
  }
  const bool ok = badG == 0 && badP == 0 && benchOk;
  printf("%s\n", ok ? "결과: 층0 = 층1 = 층2 전부 비트 동일" : "결과: 불일치 있음");
  for (auto& m : arts) m.cache->release();
  scene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
