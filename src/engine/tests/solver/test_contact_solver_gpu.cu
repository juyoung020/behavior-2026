// 층 2 시험: TGS 접촉 풀이 CUDA 판 N 개 = 층 1 C++ = PhysX, 비트 사슬 + 처리량.
// 입력: test_contact_solver --dump 가 쓴 입력 흐름 파일(스텝별 섬 순서·접촉 출력·Sc 층 입력 + PhysX 결과). PhysX 링크 없음.
//   test_contact_solver_gpu <file.svs> [--envs E] [--threads T]
// 판 하나 = CUDA 블록 하나(판 안 순서는 PhysX 와 같게 한 스레드가 차례로 — v1). 모든 판은 같은 장면·같은 입력 흐름을 받는다.
// 비교: (1) 층 1 CPU 재생 = PhysX (스텝마다), (2) GPU 판 0 = 층 1 (스텝마다), (3) GPU 판 전부의 마지막 상태 = 층 1.
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "solver_stream.h"

namespace sv = eng::sv;

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e = (x);                                                                          \
    if (e != cudaSuccess) {                                                                       \
      fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));                  \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

// 판별 작업 공간 용량
struct Caps {
  uint32_t nb, maxCMs, pool, desc, part, arena, fric;
};
// 판 b 의 조각 = base + b * cap
struct DevMem {
  eng::Body* bodies;
  sv::SolverCM* cms;
  sv::SBodyVel* vels;
  sv::SBodyTxI* txI;
  sv::SBodyData* datas;
  sv::SDesc *descs, *ordered, *temp;
  sv::BatchHeader* headers;
  uint32_t* partCounts;
  uint32_t* bodySolverIndex;
  uint8_t* arena;
  sv::FrictionPatch *fr0, *fr1;
  sv::CorrelationBuffer* corr;
  sv::ContactPoint* cbuf;
};

// 호스트·장치 공용: 판 하나 조립
__host__ __device__ inline sv::SolverBoard makeBoard(const DevMem& M, const Caps& C, uint32_t b) {
  sv::SolverBoard B{};
  B.bodies = M.bodies + size_t(b) * C.nb;
  B.nbBodies = C.nb;
  B.cms = M.cms + size_t(b) * C.maxCMs;
  B.nbCMs = C.maxCMs;
  B.vels = M.vels + size_t(b) * C.pool;
  B.txI = M.txI + size_t(b) * C.pool;
  B.datas = M.datas + size_t(b) * C.pool;
  B.poolCap = C.pool;
  B.descs = M.descs + size_t(b) * C.desc;
  B.ordered = M.ordered + size_t(b) * C.desc;
  B.temp = M.temp + size_t(b) * C.desc;
  B.headers = M.headers + size_t(b) * C.desc;
  B.descCap = C.desc;
  B.partitionCounts = M.partCounts + size_t(b) * C.part;
  B.partitionCap = C.part;
  B.bodySolverIndex = M.bodySolverIndex + size_t(b) * C.nb;
  B.constraints = sv::ByteArena{M.arena + size_t(b) * C.arena, 0, C.arena, 0};
  B.friction[0] = sv::FrictionArena{M.fr0 + size_t(b) * C.fric, 0, C.fric, 0};
  B.friction[1] = sv::FrictionArena{M.fr1 + size_t(b) * C.fric, 0, C.fric, 0};
  B.frictionCurIdx = 0;
  B.corr = M.corr + b;
  B.contactBuffer = M.cbuf + size_t(b) * sv::MAX_CONTACTS;
  return B;
}

__global__ void kRun(DevMem M, Caps C, const uint8_t* stream, const uint64_t* stepOff, uint32_t steps, sv::SolverParams prm, float* res0,
                     float* resFinal, uint32_t* errs) {
  const uint32_t b = blockIdx.x;
  if (threadIdx.x != 0) return;
  sv::SolverBoard B = makeBoard(M, C, b);
  for (uint32_t s = 0; s < steps; ++s) {
    const uint8_t* base = stream + stepOff[s];
    const svs::StepCounts c = *reinterpret_cast<const svs::StepCounts*>(base);
    const svs::StepView v = svs::makeView(base, c, C.nb);
    svs::runStep(B, prm, v);
    if (b == 0 && res0)
      for (uint32_t i = 0; i < C.nb; ++i) svs::bodyResult(B.bodies[i], res0 + (size_t(s) * C.nb + i) * svs::RES_FLOATS);
  }
  for (uint32_t i = 0; i < C.nb; ++i) svs::bodyResult(B.bodies[i], resFinal + (size_t(b) * C.nb + i) * svs::RES_FLOATS);
  errs[b] = B.error;
}

template <class T>
static T* dalloc(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "사용: %s <file.svs> [--envs E]\n", argv[0]);
    return 2;
  }
  int envs = 256;
  for (int i = 2; i < argc; ++i)
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
  svs::Stream S;
  if (!svs::readStream(argv[1], S)) {
    fprintf(stderr, "입력 흐름 읽기 실패: %s\n", argv[1]);
    return 2;
  }
  const uint32_t nb = S.h.nb, steps = S.h.steps;
  sv::SolverParams prm;
  prm.gravity = eng::V3{S.h.gravity[0], S.h.gravity[1], S.h.gravity[2]};
  prm.dt = S.h.dt;
  prm.enableStabilization = S.h.stab != 0;
  prm.bounceThreshold = S.h.bounce;
  prm.frictionOffsetThreshold = S.h.frictionOffset;
  prm.correlationDistance = S.h.correlation;
  prm.solverBatchSize = S.h.batchSize;
  prm.solverArticBatchSize = S.h.articBatchSize;
  Caps C;
  C.nb = nb;
  C.maxCMs = S.h.maxCMs;
  C.pool = nb + 2;
  C.desc = S.h.maxDescs + 8;
  C.part = 1024;
  C.arena = ((S.h.maxArena * 2 + 4096) + 15) & ~15u;
  C.fric = S.h.maxFriction * 2 + 64;

  // ---- 층 1 CPU 재생 (판 하나) = PhysX ?
  std::vector<eng::Body> cb = S.bodies0;
  std::vector<sv::SolverCM> ccms(C.maxCMs);
  for (auto& m : ccms) {
    m = sv::SolverCM{};
    m.frictionPtr = sv::NONE;
  }
  std::vector<sv::SBodyVel> cv(C.pool);
  std::vector<sv::SBodyTxI> ct(C.pool);
  std::vector<sv::SBodyData> cd(C.pool);
  std::vector<sv::SDesc> cds(C.desc), cor(C.desc), ctm(C.desc);
  std::vector<sv::BatchHeader> ch(C.desc);
  std::vector<uint32_t> cpart(C.part), cbsi(nb);
  std::vector<uint8_t> carena(C.arena);
  std::vector<sv::FrictionPatch> cf0(C.fric), cf1(C.fric);
  std::unique_ptr<sv::CorrelationBuffer> ccorr(new sv::CorrelationBuffer());
  std::vector<sv::ContactPoint> ccbuf(sv::MAX_CONTACTS);
  DevMem HM{cb.data(), ccms.data(), cv.data(), ct.data(), cd.data(), cds.data(), cor.data(), ctm.data(), ch.data(), cpart.data(), cbsi.data(),
            carena.data(), cf0.data(), cf1.data(), ccorr.get(), ccbuf.data()};
  sv::SolverBoard HB = makeBoard(HM, C, 0);
  std::vector<float> cpuRes(size_t(steps) * nb * svs::RES_FLOATS);
  uint64_t pxCmp = 0, pxBad = 0;
  int64_t pxFirst = -1;
  const auto tc0 = std::chrono::steady_clock::now();
  {
    FtzScope f;
    for (uint32_t s = 0; s < steps; ++s) {
      const uint8_t* base = S.data.data() + S.stepOffset[s];
      const svs::StepCounts c = *reinterpret_cast<const svs::StepCounts*>(base);
      svs::runStep(HB, prm, svs::makeView(base, c, nb));
      for (uint32_t i = 0; i < nb; ++i) svs::bodyResult(HB.bodies[i], cpuRes.data() + (size_t(s) * nb + i) * svs::RES_FLOATS);
    }
  }
  const double cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
  for (size_t k = 0; k < cpuRes.size(); k += svs::RES_FLOATS) {
    pxCmp++;
    if (memcmp(&cpuRes[k], &S.pxResults[k], 4 * svs::RES_FLOATS)) {
      pxBad++;
      if (pxFirst < 0) pxFirst = int64_t(k / svs::RES_FLOATS / nb) + 1;
    }
  }
  printf("층 1 CPU 재생 = PhysX: 몸체·스텝 %" PRIu64 " 비교, 비트 다름 %" PRIu64 " (첫 스텝 %" PRId64 "), 엔진 오류 0x%x\n", pxCmp, pxBad, pxFirst, HB.error);
  printf("  CPU 단일 스레드: %u 스텝 %.3f s (%.0f 스텝/초, 몸체 %u)\n", steps, cpuSec, steps / cpuSec, nb);

  // ---- 층 2 GPU 판 envs 개
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));
  DevMem D;
  D.bodies = dalloc<eng::Body>(size_t(envs) * C.nb);
  D.cms = dalloc<sv::SolverCM>(size_t(envs) * C.maxCMs);
  D.vels = dalloc<sv::SBodyVel>(size_t(envs) * C.pool);
  D.txI = dalloc<sv::SBodyTxI>(size_t(envs) * C.pool);
  D.datas = dalloc<sv::SBodyData>(size_t(envs) * C.pool);
  D.descs = dalloc<sv::SDesc>(size_t(envs) * C.desc);
  D.ordered = dalloc<sv::SDesc>(size_t(envs) * C.desc);
  D.temp = dalloc<sv::SDesc>(size_t(envs) * C.desc);
  D.headers = dalloc<sv::BatchHeader>(size_t(envs) * C.desc);
  D.partCounts = dalloc<uint32_t>(size_t(envs) * C.part);
  D.bodySolverIndex = dalloc<uint32_t>(size_t(envs) * C.nb);
  D.arena = dalloc<uint8_t>(size_t(envs) * C.arena);
  D.fr0 = dalloc<sv::FrictionPatch>(size_t(envs) * C.fric);
  D.fr1 = dalloc<sv::FrictionPatch>(size_t(envs) * C.fric);
  D.corr = dalloc<sv::CorrelationBuffer>(envs);
  D.cbuf = dalloc<sv::ContactPoint>(size_t(envs) * sv::MAX_CONTACTS);
  {  // 판마다 같은 초기 상태
    std::vector<eng::Body> rb(size_t(envs) * nb);
    for (int e = 0; e < envs; ++e) memcpy(&rb[size_t(e) * nb], S.bodies0.data(), nb * sizeof(eng::Body));
    CK(cudaMemcpy(D.bodies, rb.data(), rb.size() * sizeof(eng::Body), cudaMemcpyHostToDevice));
    std::vector<sv::SolverCM> rc(size_t(envs) * C.maxCMs);
    for (auto& m : rc) {
      m = sv::SolverCM{};
      m.frictionPtr = sv::NONE;
    }
    CK(cudaMemcpy(D.cms, rc.data(), rc.size() * sizeof(sv::SolverCM), cudaMemcpyHostToDevice));
  }
  uint8_t* dStream = dalloc<uint8_t>(S.data.size());
  CK(cudaMemcpy(dStream, S.data.data(), S.data.size(), cudaMemcpyHostToDevice));
  uint64_t* dOff = dalloc<uint64_t>(steps);
  CK(cudaMemcpy(dOff, S.stepOffset.data(), steps * sizeof(uint64_t), cudaMemcpyHostToDevice));
  float* dRes0 = dalloc<float>(size_t(steps) * nb * svs::RES_FLOATS);
  float* dFinal = dalloc<float>(size_t(envs) * nb * svs::RES_FLOATS);
  uint32_t* dErr = dalloc<uint32_t>(envs);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  kRun<<<envs, 32>>>(D, C, dStream, dOff, steps, prm, dRes0, dFinal, dErr);
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaEventSynchronize(e1));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<float> res0(size_t(steps) * nb * svs::RES_FLOATS), fin(size_t(envs) * nb * svs::RES_FLOATS);
  std::vector<uint32_t> errs(envs);
  CK(cudaMemcpy(res0.data(), dRes0, res0.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(fin.data(), dFinal, fin.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(errs.data(), dErr, envs * 4, cudaMemcpyDeviceToHost));
  uint64_t g0Cmp = 0, g0Bad = 0, gfCmp = 0, gfBad = 0;
  int64_t g0First = -1;
  double maxd = 0;
  for (size_t k = 0; k < res0.size(); k += svs::RES_FLOATS) {
    g0Cmp++;
    if (memcmp(&res0[k], &cpuRes[k], 4 * svs::RES_FLOATS)) {
      g0Bad++;
      if (g0First < 0) g0First = int64_t(k / svs::RES_FLOATS / nb) + 1;
      for (int j = 0; j < svs::RES_FLOATS; ++j) maxd = std::fmax(maxd, std::fabs(double(res0[k + j]) - double(cpuRes[k + j])));
    }
  }
  const float* cpuFinal = cpuRes.data() + size_t(steps - 1) * nb * svs::RES_FLOATS;
  for (int e = 0; e < envs; ++e)
    for (uint32_t i = 0; i < nb; ++i) {
      gfCmp++;
      if (memcmp(&fin[(size_t(e) * nb + i) * svs::RES_FLOATS], cpuFinal + size_t(i) * svs::RES_FLOATS, 4 * svs::RES_FLOATS)) gfBad++;
    }
  uint32_t anyErr = 0;
  for (uint32_t x : errs) anyErr |= x;
  printf("층 2 GPU 판 0 = 층 1 (스텝마다): 비교 %" PRIu64 ", 비트 다름 %" PRIu64 " (첫 스텝 %" PRId64 ", 최대|차| %.3e)\n", g0Cmp, g0Bad, g0First, maxd);
  printf("층 2 GPU 판 %d 개 마지막 상태 = 층 1: 비교 %" PRIu64 ", 비트 다름 %" PRIu64 ", 엔진 오류 0x%x\n", envs, gfCmp, gfBad, anyErr);
  const double sec = ms / 1000.0;
  printf("  GPU: 판 %d x %u 스텝 %.3f s -> 판·스텝/초 %.0f, 몸체·스텝/초 %.3g (CPU 단일 스레드 대비 %.1f 배)\n", envs, steps, sec, envs * double(steps) / sec,
         envs * double(steps) * nb / sec, (envs * double(steps) / sec) / (steps / cpuSec));
  const bool ok = !pxBad && !g0Bad && !gfBad && !anyErr && !HB.error;
  printf("%s\n", ok ? "결과: PhysX = 층 1 = 층 2 비트 동일" : "결과: 불일치 있음");
  return ok ? 0 : 3;
}
