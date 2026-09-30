// 층 2 시험: 관절체가 붙은 TGS 풀이(docs 17.6) CUDA 판 N 개 = 층 1 C++ = PhysX, 비트 사슬 + 처리량.
// 입력: test_art_solver --dump 가 쓴 흐름(art_stream.h). PhysX 링크 없음.
//   test_art_solver_gpu <file.asv> [--envs E] [--threads T]
// 판 하나 = CUDA 블록 하나. 판 안 스레드 T 개: 준비·되쓰기는 스레드 0, 분할 안 제약·강체 적분·관절체 내부 풀이는 나눠 푼다(solverStepPar).
// 모든 판은 같은 장면·같은 입력을 받는다(관절체 드라이브 입력도 판 0 과 같은 위상) -> 판 전부가 층 1 과 같아야 한다.
// 비교: (1) 층 1 CPU 재생 = PhysX (강체·관절체·1D 되쓰기, 스텝마다), (2) GPU 판 0 = 층 1 (스텝마다), (3) GPU 판 전부 마지막 상태 = 층 1.
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

#include "core/solver/tgs_solver.h"
#define ARTEST_NO_PHYSX
#include "tests/articulation/random_art.h"
#include "art_stream.h"

namespace sv = eng::sv;
namespace A = eng::art;

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e = (x);                                                         \
    if (e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

struct Caps {
  uint32_t nb, na, maxCMs, pool, desc, part, arena, fric, c1d, stat;
};
struct Mem {  // 판 b 의 조각 = base + b * cap
  eng::Body* bodies;
  A::Articulation* arts;
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
  eng::jnt::Writeback* wb;
  eng::jnt::Row* rows;
  A::StaticLists* lists;
  sv::SDesc *s1, *sc;
  uint32_t *n1, *nc, *batch;
  sv::ArtProgress* prog;
};

__host__ __device__ inline sv::SolverBoard makeBoard(const Mem& M, const Caps& C, uint32_t b) {
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
  B.bodySolverIndex = M.bodySolverIndex + size_t(b) * (C.nb + 1);
  B.constraints = sv::ByteArena{M.arena + size_t(b) * C.arena, 0, C.arena, 0};
  B.friction[0] = sv::FrictionArena{M.fr0 + size_t(b) * C.fric, 0, C.fric, 0};
  B.friction[1] = sv::FrictionArena{M.fr1 + size_t(b) * C.fric, 0, C.fric, 0};
  B.frictionCurIdx = 0;
  B.corr = M.corr + b;
  B.contactBuffer = M.cbuf + size_t(b) * sv::MAX_CONTACTS;
  B.writebacks = M.wb + size_t(b) * C.c1d;
  B.rowScratch = M.rows + size_t(b) * eng::jnt::MAX_CONSTRAINT_ROWS * 4;
  B.arts = M.arts + size_t(b) * C.na;
  B.nbArts = C.na;
  B.artLists = M.lists + size_t(b) * C.na;
  B.artStatic1D = M.s1 + size_t(b) * C.na * C.stat;
  B.artStaticContact = M.sc + size_t(b) * C.na * C.stat;
  B.artNbStatic1D = M.n1 + size_t(b) * C.na;
  B.artNbStaticContact = M.nc + size_t(b) * C.na;
  B.artStaticCap = C.stat;
  B.artBatchIndex = M.batch + size_t(b) * C.na;
  B.artProg = M.prog + size_t(b) * C.na;
  return B;
}

template <class T>
__host__ __device__ inline const T* at(const uint8_t* base, size_t off) {
  return reinterpret_cast<const T*>(base + off);
}

// 스텝 입력 넣기 (풀기 전): 관절체 드라이브 입력(시험 입력 계획), 강체 Sc 깨움·상호작용 수, 관리자 값(마찰 칸은 유지), 섬 배열
__host__ __device__ inline void applyPre(sv::SolverBoard& B, const ast::Header& h, const uint8_t* base, int s, const artest::ArtInputs* inputs) {
  const ast::Counts c = *at<ast::Counts>(base, 0);
  const ast::Layout L = ast::layout(c, h);
  for (uint32_t k = 0; k < h.na; ++k) artest::stepInputsEng(B.arts[k], inputs[k], s, 0, h.dt);
  const ast::Wake* w = at<ast::Wake>(base, L.wake);
  for (uint32_t k = 0; k < c.nWake; ++k)
    if (B.bodies[w[k].body].wakeCounter < w[k].value) B.bodies[w[k].body].wakeCounter = w[k].value;
  const uint32_t* nc = at<uint32_t>(base, L.numCounted);
  for (uint32_t i = 0; i < h.nb; ++i) B.bodies[i].numCountedInteractions = nc[i];
  const uint32_t* icm = at<uint32_t>(base, L.icm);
  const sv::SolverCM* cmIn = at<sv::SolverCM>(base, L.cmIn);
  for (uint32_t k = 0; k < c.nICM; ++k) {
    sv::SolverCM& m = B.cms[icm[k]];
    const uint32_t fp = m.frictionPtr, fc = m.frictionCount;
    m = cmIn[k];
    m.frictionPtr = fp;
    m.frictionCount = fc;
  }
  B.islands = at<sv::IslandIn>(base, L.islands);
  B.nbIslands = c.nIslands;
  B.islandBodies = at<uint32_t>(base, L.ib);
  B.islandCMs = icm;
  B.islandArts = at<uint32_t>(base, L.ia);
  B.activatedCMs = at<uint32_t>(base, L.act);
  B.nbActivatedCMs = c.nAct;
  B.resetCMs = at<uint32_t>(base, L.reset);
  B.nbResetCMs = c.nReset;
  B.patches = at<sv::ContactPatchIn>(base, L.patches);
  B.contacts = at<sv::ContactIn>(base, L.contacts);
  B.c1d = at<sv::Constraint1DIn>(base, L.c1d);
  B.nbC1D = c.nC1D;
  B.islandC1Ds = at<uint32_t>(base, L.ic1d);
  B.jointData = at<eng::jnt::D6Data>(base, L.jd);
}
// 풀이 뒤: 강체 afterIntegration, 관절체 Sc 입력(잠 판정 직전 값) + 잠, 섬 관리가 재운 강체 되돌리기
__host__ __device__ inline void applyPost(sv::SolverBoard& B, const ast::Header& h, const uint8_t* base) {
  const ast::Counts c = *at<ast::Counts>(base, 0);
  const ast::Layout L = ast::layout(c, h);
  sv::afterIntegration(B);
  if (c.lateValid) {
    const float* lw = at<float>(base, L.lateLinkWake);
    const uint32_t* lc = at<uint32_t>(base, L.lateLinkCounted);
    const float* aw = at<float>(base, L.lateArtWake);
    for (uint32_t k = 0; k < h.na; ++k) {
      A::Articulation& a = B.arts[k];
      for (uint32_t l = 0; l < a.nLinks; ++l) {
        a.bodies[l].numCountedInteractions = lc[k * h.maxLinks + l];
        if (lw[k * h.maxLinks + l] > a.bodies[l].wakeCounter) a.bodies[l].wakeCounter = lw[k * h.maxLinks + l];
      }
      if (aw[k] > a.wakeCounter) a.wakeCounter = aw[k];
    }
  }
  sv::afterIntegrationArts(B, h.dt, at<uint32_t>(base, L.deactArts), c.nDeactArts);
  sv::deactivateBodies(B, at<uint32_t>(base, L.deact), c.nDeact);
}
__host__ __device__ inline void bodyResult(const eng::Body& b, float* r) {
  const eng::Tf t = eng::getGlobalPose(b);
  const float x[ast::RES_FLOATS] = {t.q.x, t.q.y, t.q.z, t.q.w, t.p.x, t.p.y, t.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z,
                                    b.wakeCounter};
  for (int i = 0; i < ast::RES_FLOATS; ++i) r[i] = x[i];
}
__host__ __device__ inline void artResult(const A::Articulation& a, uint32_t maxLinks, float* x, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) x[i] = 0.0f;
  for (uint32_t l = 0; l < a.nLinks; ++l) {
    const eng::Tf te = A::linkGlobalPose(a, l);
    const A::LinkBody& b = a.bodies[a.ll[l]];
    const float e[14] = {te.q.x, te.q.y, te.q.z, te.q.w, te.p.x, te.p.y, te.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z,
                         b.wakeCounter};
    for (int j = 0; j < 14; ++j) x[l * 14 + j] = e[j];
  }
  for (uint32_t d = 0; d < a.dofs; ++d) {
    x[14 * maxLinks + d] = a.jointPosition[d];
    x[14 * maxLinks + A::kMaxDofs + d] = a.jointVelocity[d];
  }
  x[14 * maxLinks + 2 * A::kMaxDofs] = a.wakeCounter;
  x[14 * maxLinks + 2 * A::kMaxDofs + 1] = a.awake ? 0.0f : 1.0f;
}

__global__ void kRun(Mem M, Caps C, ast::Header h, const uint8_t* stream, const uint64_t* stepOff, const artest::ArtInputs* inputs, sv::SolverParams prm,
                     float* res0, float* art0, float* resFinal, float* artFinal, uint32_t* errs) {
  __shared__ sv::SolverBoard B;
  const uint32_t b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) B = makeBoard(M, C, b);
  __syncthreads();
  for (uint32_t s = 0; s < h.steps; ++s) {
    const uint8_t* base = stream + stepOff[s];
    if (tid == 0) applyPre(B, h, base, int(s + 1), inputs);
    __syncthreads();
    sv::solverStepPar(B, prm, tid, nt);
    __syncthreads();
    if (tid == 0) {
      applyPost(B, h, base);
      if (b == 0) {
        for (uint32_t i = 0; i < C.nb; ++i) bodyResult(B.bodies[i], res0 + (size_t(s) * C.nb + i) * ast::RES_FLOATS);
        for (uint32_t k = 0; k < C.na; ++k) artResult(B.arts[k], h.maxLinks, art0 + (size_t(s) * C.na + k) * h.artResFloats, h.artResFloats);
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (uint32_t i = 0; i < C.nb; ++i) bodyResult(B.bodies[i], resFinal + (size_t(b) * C.nb + i) * ast::RES_FLOATS);
    for (uint32_t k = 0; k < C.na; ++k) artResult(B.arts[k], h.maxLinks, artFinal + (size_t(b) * C.na + k) * h.artResFloats, h.artResFloats);
    errs[b] = B.error;
  }
}

template <class T>
static T* dalloc(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
  return p;
}

// 칸 비교 (NaN 은 양쪽 NaN 이면 같음)
static bool sameF(const float* a, const float* b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const bool na = std::isnan(a[i]), nb = std::isnan(b[i]);
    if (na || nb) {
      if (na != nb) return false;
    } else if (memcmp(&a[i], &b[i], 4)) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "사용: %s <file.asv> [--envs E] [--threads T]\n", argv[0]);
    return 2;
  }
  int envs = 64, threads = 32;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
  }
  ast::Stream S;
  if (!ast::readStream(argv[1], S)) {
    fprintf(stderr, "입력 흐름 읽기 실패: %s\n", argv[1]);
    return 2;
  }
  const ast::Header& h = S.h;
  if (h.sizeBody != sizeof(eng::Body) || h.sizeArt != sizeof(A::Articulation) || h.sizeCM != sizeof(sv::SolverCM) || h.sizeIsland != sizeof(sv::IslandIn) ||
      h.sizeC1D != sizeof(sv::Constraint1DIn) || h.sizeD6 != sizeof(eng::jnt::D6Data) || h.sizeInputs != sizeof(artest::ArtInputs) ||
      h.maxLinks != A::kMaxLinks) {
    fprintf(stderr, "자료형 크기가 흐름 파일과 다르다 (같은 빌드 선택으로 만들 것)\n");
    return 2;
  }
  const uint32_t nb = h.nb, na = h.na, steps = h.steps, AR = h.artResFloats;
  sv::SolverParams prm;
  prm.gravity = eng::V3{h.gravity[0], h.gravity[1], h.gravity[2]};
  prm.dt = h.dt;
  prm.enableStabilization = h.stab != 0;
  prm.bounceThreshold = h.bounce;
  prm.frictionOffsetThreshold = h.frictionOffset;
  prm.correlationDistance = h.correlation;
  prm.solverBatchSize = h.batchSize;
  prm.solverArticBatchSize = h.articBatchSize;
  prm.lengthScale = h.lengthScale;
  prm.solveArticulationContactLast = h.last != 0;
  Caps C;
  C.nb = nb;
  C.na = na;
  C.maxCMs = h.maxCMs;
  C.pool = nb + 2;
  C.desc = h.maxDescs + 8;
  C.part = 1024;
  C.arena = ((h.maxArena * 2 + 4096) + 15) & ~15u;
  C.fric = h.maxFriction * 2 + 64;
  C.c1d = h.maxC1D ? h.maxC1D : 1;
  C.stat = 256;
  const artest::ArtInputs* inputs = reinterpret_cast<const artest::ArtInputs*>(S.inputs.data());

  // ---- 층 1 CPU 재생 = PhysX
  std::vector<eng::Body> cb(nb);
  memcpy(cb.data(), S.bodies0.data(), S.bodies0.size());
  std::vector<A::Articulation> ca(na);
  if (na) memcpy(static_cast<void*>(ca.data()), S.arts0.data(), S.arts0.size());
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
  std::vector<uint32_t> cpart(C.part), cbsi(nb + 1);
  std::vector<uint8_t> carena(C.arena);
  std::vector<sv::FrictionPatch> cf0(C.fric), cf1(C.fric);
  std::unique_ptr<sv::CorrelationBuffer> ccorr(new sv::CorrelationBuffer());
  std::vector<sv::ContactPoint> ccbuf(sv::MAX_CONTACTS);
  std::vector<eng::jnt::Writeback> cwb(C.c1d);
  memset(cwb.data(), 0, cwb.size() * sizeof(eng::jnt::Writeback));
  std::vector<eng::jnt::Row> crows(eng::jnt::MAX_CONSTRAINT_ROWS * 4);
  std::vector<A::StaticLists> clists(na ? na : 1);
  std::vector<sv::SDesc> cs1(size_t(na) * C.stat + 1), csc(size_t(na) * C.stat + 1);
  std::vector<uint32_t> cn1(na + 1), cnc(na + 1), cbatch(na + 1);
  std::vector<sv::ArtProgress> cprog(na + 1);
  Mem HM{cb.data(), ca.data(), ccms.data(), cv.data(), ct.data(), cd.data(), cds.data(), cor.data(), ctm.data(), ch.data(), cpart.data(), cbsi.data(),
         carena.data(), cf0.data(), cf1.data(), ccorr.get(), ccbuf.data(), cwb.data(), crows.data(), clists.data(), cs1.data(), csc.data(), cn1.data(),
         cnc.data(), cbatch.data(), cprog.data()};
  sv::SolverBoard HB = makeBoard(HM, C, 0);
  std::vector<float> cpuRes(size_t(steps) * nb * ast::RES_FLOATS), cpuArt(size_t(steps) * na * AR);
  uint64_t pxCmp = 0, pxBad = 0, artCmp = 0, artBad = 0, wbCmp = 0, wbBad = 0;
  int64_t pxFirst = -1, artFirst = -1, wbFirst = -1;
  const auto tc0 = std::chrono::steady_clock::now();
  {
    FtzScope f;
    for (uint32_t s = 0; s < steps; ++s) {
      const uint8_t* base = S.data.data() + S.stepOffset[s];
      const ast::Counts c = *at<ast::Counts>(base, 0);
      const ast::Layout L = ast::layout(c, h);
      applyPre(HB, h, base, int(s + 1), inputs);
      sv::solverStep(HB, prm);
      applyPost(HB, h, base);
      for (uint32_t i = 0; i < nb; ++i) bodyResult(HB.bodies[i], cpuRes.data() + (size_t(s) * nb + i) * ast::RES_FLOATS);
      for (uint32_t k = 0; k < na; ++k) artResult(HB.arts[k], h.maxLinks, cpuArt.data() + (size_t(s) * na + k) * AR, AR);
      const float* pr = at<float>(base, L.pxRes);
      for (uint32_t i = 0; i < nb; ++i) {
        pxCmp++;
        if (!sameF(pr + size_t(i) * ast::RES_FLOATS, cpuRes.data() + (size_t(s) * nb + i) * ast::RES_FLOATS, ast::RES_FLOATS)) {
          pxBad++;
          if (pxFirst < 0) pxFirst = int64_t(s) + 1;
        }
      }
      const float* pa = at<float>(base, L.pxArt);
      for (uint32_t k = 0; k < na; ++k) {
        artCmp++;
        if (!sameF(pa + size_t(k) * AR, cpuArt.data() + (size_t(s) * na + k) * AR, AR)) {
          artBad++;
          if (artFirst < 0) artFirst = int64_t(s) + 1;
        }
      }
      const eng::jnt::Writeback* pw = at<eng::jnt::Writeback>(base, L.pxWb);
      const sv::Constraint1DIn* c1d = at<sv::Constraint1DIn>(base, L.c1d);
      for (uint32_t k = 0; k < c.nC1D; ++k) {
        wbCmp++;
        if (memcmp(&pw[k], &HB.writebacks[c1d[k].writeback], sizeof(eng::jnt::Writeback))) {
          wbBad++;
          if (wbFirst < 0) wbFirst = int64_t(s) + 1;
        }
      }
    }
  }
  const double cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
  printf("장면: 강체 %u 관절체 %u x %u 스텝 (접촉 마지막 %u, 안정화 %u)\n", nb, na, steps, h.last, h.stab);
  printf("층 1 CPU 재생 = PhysX: 강체 %" PRIu64 " 다름 %" PRIu64 " (첫 %" PRId64 ") | 관절체 %" PRIu64 " 다름 %" PRIu64 " (첫 %" PRId64 ") | 1D 되쓰기 %" PRIu64
         " 다름 %" PRIu64 " (첫 %" PRId64 ") | 엔진 오류 0x%x\n",
         pxCmp, pxBad, pxFirst, artCmp, artBad, artFirst, wbCmp, wbBad, wbFirst, HB.error);
  printf("  CPU 단일 스레드: %u 스텝 %.3f s (%.0f 스텝/초)\n", steps, cpuSec, steps / cpuSec);

  // ---- 층 2 GPU 판 envs 개
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));
  const size_t E = size_t(envs);
  Mem D;
  D.bodies = dalloc<eng::Body>(E * nb);
  D.arts = dalloc<A::Articulation>(E * na);
  D.cms = dalloc<sv::SolverCM>(E * C.maxCMs);
  D.vels = dalloc<sv::SBodyVel>(E * C.pool);
  D.txI = dalloc<sv::SBodyTxI>(E * C.pool);
  D.datas = dalloc<sv::SBodyData>(E * C.pool);
  D.descs = dalloc<sv::SDesc>(E * C.desc);
  D.ordered = dalloc<sv::SDesc>(E * C.desc);
  D.temp = dalloc<sv::SDesc>(E * C.desc);
  D.headers = dalloc<sv::BatchHeader>(E * C.desc);
  D.partCounts = dalloc<uint32_t>(E * C.part);
  D.bodySolverIndex = dalloc<uint32_t>(E * (nb + 1));
  D.arena = dalloc<uint8_t>(E * C.arena);
  D.fr0 = dalloc<sv::FrictionPatch>(E * C.fric);
  D.fr1 = dalloc<sv::FrictionPatch>(E * C.fric);
  D.corr = dalloc<sv::CorrelationBuffer>(E);
  D.cbuf = dalloc<sv::ContactPoint>(E * sv::MAX_CONTACTS);
  D.wb = dalloc<eng::jnt::Writeback>(E * C.c1d);
  CK(cudaMemset(D.wb, 0, E * C.c1d * sizeof(eng::jnt::Writeback)));
  D.rows = dalloc<eng::jnt::Row>(E * eng::jnt::MAX_CONSTRAINT_ROWS * 4);
  D.lists = dalloc<A::StaticLists>(E * na);
  D.s1 = dalloc<sv::SDesc>(E * na * C.stat);
  D.sc = dalloc<sv::SDesc>(E * na * C.stat);
  D.n1 = dalloc<uint32_t>(E * na);
  D.nc = dalloc<uint32_t>(E * na);
  D.batch = dalloc<uint32_t>(E * na);
  D.prog = dalloc<sv::ArtProgress>(E * na);
  for (size_t e = 0; e < E; ++e) {  // 판마다 같은 초기 상태
    CK(cudaMemcpy(D.bodies + e * nb, S.bodies0.data(), S.bodies0.size(), cudaMemcpyHostToDevice));
    if (na) CK(cudaMemcpy(D.arts + e * na, S.arts0.data(), S.arts0.size(), cudaMemcpyHostToDevice));
  }
  {
    std::vector<sv::SolverCM> rc(E * C.maxCMs);
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
  artest::ArtInputs* dIn = dalloc<artest::ArtInputs>(na);
  if (na) CK(cudaMemcpy(dIn, inputs, na * sizeof(artest::ArtInputs), cudaMemcpyHostToDevice));
  float* dRes0 = dalloc<float>(size_t(steps) * nb * ast::RES_FLOATS);
  float* dArt0 = dalloc<float>(size_t(steps) * na * AR);
  float* dResF = dalloc<float>(E * nb * ast::RES_FLOATS);
  float* dArtF = dalloc<float>(E * na * AR);
  uint32_t* dErr = dalloc<uint32_t>(E);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  kRun<<<envs, threads>>>(D, C, h, dStream, dOff, dIn, prm, dRes0, dArt0, dResF, dArtF, dErr);
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaEventSynchronize(e1));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<float> res0(size_t(steps) * nb * ast::RES_FLOATS), art0(size_t(steps) * na * AR), resF(E * nb * ast::RES_FLOATS), artF(E * na * AR);
  std::vector<uint32_t> errs(E);
  CK(cudaMemcpy(res0.data(), dRes0, res0.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(art0.data(), dArt0, art0.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(resF.data(), dResF, resF.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(artF.data(), dArtF, artF.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(errs.data(), dErr, E * 4, cudaMemcpyDeviceToHost));
  std::vector<eng::jnt::Writeback> wbAll(E * C.c1d);
  CK(cudaMemcpy(wbAll.data(), D.wb, wbAll.size() * sizeof(eng::jnt::Writeback), cudaMemcpyDeviceToHost));
  uint64_t g0Bad = 0, g0ArtBad = 0, gfBad = 0, gfArtBad = 0, gwBad = 0;
  int64_t g0First = -1;
  for (uint32_t s = 0; s < steps; ++s) {
    for (uint32_t i = 0; i < nb; ++i) {
      const size_t o = (size_t(s) * nb + i) * ast::RES_FLOATS;
      if (!sameF(&res0[o], &cpuRes[o], ast::RES_FLOATS)) {
        g0Bad++;
        if (g0First < 0) g0First = int64_t(s) + 1;
      }
    }
    for (uint32_t k = 0; k < na; ++k) {
      const size_t o = (size_t(s) * na + k) * AR;
      if (!sameF(&art0[o], &cpuArt[o], AR)) {
        g0ArtBad++;
        if (g0First < 0) g0First = int64_t(s) + 1;
      }
    }
  }
  const float* cpuFinal = cpuRes.data() + size_t(steps - 1) * nb * ast::RES_FLOATS;
  const float* cpuArtFinal = cpuArt.data() + size_t(steps - 1) * na * AR;
  for (size_t e = 0; e < E; ++e) {
    if (!sameF(&resF[e * nb * ast::RES_FLOATS], cpuFinal, size_t(nb) * ast::RES_FLOATS)) gfBad++;
    if (!sameF(&artF[e * na * AR], cpuArtFinal, size_t(na) * AR)) gfArtBad++;
    if (memcmp(&wbAll[e * C.c1d], cwb.data(), C.c1d * sizeof(eng::jnt::Writeback))) gwBad++;
  }
  uint32_t anyErr = 0;
  for (uint32_t x : errs) anyErr |= x;
  printf("층 2 GPU 판 0 = 층 1 (스텝마다): 강체 %" PRIu64 " 다름 %" PRIu64 ", 관절체 %" PRIu64 " 다름 %" PRIu64 " (첫 스텝 %" PRId64 ")\n", uint64_t(steps) * nb, g0Bad,
         uint64_t(steps) * na, g0ArtBad, g0First);
  printf("층 2 GPU 판 %d 개 마지막 상태 = 층 1: 강체 다른 판 %" PRIu64 ", 관절체 다른 판 %" PRIu64 ", 1D 되쓰기 표 다른 판 %" PRIu64 ", 엔진 오류 0x%x\n", envs, gfBad,
         gfArtBad, gwBad, anyErr);
  const double sec = ms / 1000.0;
  printf("  GPU 판 %d x %u 스텝, 판 안 스레드 %d: %.3f s -> 판·스텝/초 %.0f (CPU 단일 스레드 대비 %.1f 배)\n", envs, steps, threads, sec, E * double(steps) / sec,
         (E * double(steps) / sec) / (steps / cpuSec));
  const bool ok = !pxBad && !artBad && !wbBad && !HB.error && !g0Bad && !g0ArtBad && !gfBad && !gfArtBad && !gwBad && !anyErr;
  printf("%s\n", ok ? "결과: PhysX = 층 1 = 층 2 비트 동일" : "결과: 불일치 있음");
  return ok ? 0 : 3;
}
