// solver 판 N 개 GPU 진입 (solver_gpu.h 설명). 장치 메모리 하나에 판마다 조각을 잡는다:
//   [구역 A: 읽고 쓰는 것 — 판 구조체, 몸체, 접촉 관리자, 1D 되쓰기, 새 마찰 arena, 섬 관절체(딱 맞는 용량)]
//   [구역 B: 읽기만 — 풀이 인자, 섬 배열, 접촉 입력, 1D 입력, 지난 마찰 arena, 관절체 포인터 표]
//   [구역 C: 작업 공간 — 올리지 않음]
// 올리기 = 고정 버퍼에 A+B 를 모아 cudaMemcpyAsync 한 번, 되받기 = A 한 번.
// (판마다 따로 여러 번 pageable cudaMemcpy 하면 WSL2 + 드라이버 591.86 에서 커널이 뒤 조각을 0 으로 읽은 적이 있다 — 17.6. 한 번에 올려 피한다.)
#include "core/solver/solver_gpu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>

#include "core/solver/tgs_solver.h"

namespace eng {
namespace sv {
namespace {

#define GS_CK(x)                                                                              \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "gpuSolveBatch: %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));   \
      return false;                                                                           \
    }                                                                                         \
  } while (0)

__global__ void kSolveBoards(SolverBoard* boards, const SolverParams* prms) {
  __shared__ SolverBoard B;
  const uint32_t b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) B = boards[b];
  __syncthreads();
  solverStepPar(B, prms[b], tid, nt);
  __syncthreads();
  if (tid == 0) {
    afterIntegration(B);
    boards[b] = B;
  }
}

constexpr uint32_t kCapErrs = SV_ERR_POOL | SV_ERR_DESC | SV_ERR_PARTITION | SV_ERR_ARENA | SV_ERR_FRICTION;

inline size_t al256(size_t x) { return (x + 255) & ~size_t(255); }

// 판 하나의 장치 조각 (오프셋은 구역 안 -> 나중에 구역 시작을 더함)
struct BoardPlan {
  // 셈한 길이·용량
  uint32_t nIB = 0, nICM = 0, nIC1D = 0, nIA = 0, nPatch = 0, nContact = 0, nWb = 0, nJd = 0;
  uint32_t pool = 0, desc = 0, part = 0, arena = 0, fricNew = 0, fricPrev = 0, stat = 0;
  std::vector<uint32_t> arts;  // 옮길 관절체 번호
  std::vector<art::ArtCaps> artCaps;
  // A
  size_t oBoard = 0, oBodies = 0, oCms = 0, oWb = 0, oFricNew = 0, oBsi = 0;
  std::vector<size_t> oArt;
  // B
  size_t oPrm = 0, oIsl = 0, oIB = 0, oICM = 0, oIC1D = 0, oIA = 0, oPatch = 0, oContact = 0, oAct = 0, oReset = 0, oC1D = 0, oJd = 0, oFricPrev = 0,
         oArtPtr = 0;
  // C
  size_t oVels = 0, oTxI = 0, oDatas = 0, oDescs = 0, oOrd = 0, oTemp = 0, oHead = 0, oPart = 0, oArena = 0, oCorr = 0, oCbuf = 0, oRows = 0,
         oLists = 0, oS1 = 0, oSC = 0, oN1 = 0, oNC = 0, oBatch = 0, oProg = 0;
};

inline void take(size_t& cur, size_t& off, size_t bytes) {
  off = cur;
  cur += al256(bytes ? bytes : 1);
}

void planBoard(const SolverBoard& H, float grow, BoardPlan& P) {
  P = BoardPlan{};
  uint32_t sumBodies = 0, sumCMs = 0, sumC1D = 0;
  for (uint32_t i = 0; i < H.nbIslands; ++i) {
    const IslandIn& I = H.islands[i];
    P.nIB = I.bodyStart + I.bodyCount > P.nIB ? I.bodyStart + I.bodyCount : P.nIB;
    P.nICM = I.cmStart + I.cmCount > P.nICM ? I.cmStart + I.cmCount : P.nICM;
    P.nIC1D = I.c1dStart + I.c1dCount > P.nIC1D ? I.c1dStart + I.c1dCount : P.nIC1D;
    P.nIA = I.artStart + I.artCount > P.nIA ? I.artStart + I.artCount : P.nIA;
    sumBodies += I.bodyCount;
    sumCMs += I.cmCount;
    sumC1D += I.c1dCount;
  }
  std::vector<uint8_t> artUsed(H.nbArts, 0);
  auto useArt = [&](uint32_t link, uint32_t body) {
    if (link && body < H.nbArts) artUsed[body] = 1;
  };
  for (uint32_t k = 0; k < P.nIA; ++k)
    if (H.islandArts[k] < H.nbArts) artUsed[H.islandArts[k]] = 1;
  for (uint32_t k = 0; k < P.nICM; ++k) {
    const SolverCM& m = H.cms[H.islandCMs[k]];
    P.nPatch = m.patchStart + m.nbPatches > P.nPatch ? m.patchStart + m.nbPatches : P.nPatch;
    P.nContact = m.contactStart + m.nbContacts > P.nContact ? m.contactStart + m.nbContacts : P.nContact;
    useArt(m.artLink0, m.body0);
    useArt(m.artLink1, m.body1);
  }
  for (uint32_t k = 0; k < H.nbC1D; ++k) {
    const Constraint1DIn& c = H.c1d[k];
    P.nWb = c.writeback + 1 > P.nWb ? c.writeback + 1 : P.nWb;
    P.nJd = c.data + 1 > P.nJd ? c.data + 1 : P.nJd;
  }
  for (uint32_t k = 0; k < P.nIC1D; ++k) {
    const Constraint1DIn& c = H.c1d[H.islandC1Ds[k]];
    useArt(c.artLink0, c.body0);
    useArt(c.artLink1, c.body1);
  }
  for (uint32_t a = 0; a < H.nbArts; ++a)
    if (artUsed[a]) {
      P.arts.push_back(a);
      P.artCaps.push_back(art::artTightCaps(artAt(H, a)));
    }
  auto g = [&](size_t x) { return uint32_t(double(x) * grow + 0.5); };
  P.pool = g(sumBodies + H.nbBodies + 2);
  P.desc = g(sumCMs + sumC1D + 8);
  P.part = H.partitionCap;
  P.arena = uint32_t(al256(g(16384 + 768ull * P.nContact + 256ull * P.nPatch + 2048ull * sumC1D)));
  P.fricPrev = H.friction[H.frictionCurIdx].size;
  P.fricNew = g(P.nContact + 64);
  P.stat = g(sumCMs + sumC1D + 1);
}

}  // namespace

static GpuSolveCtx& defaultCtx() {
  static GpuSolveCtx c;
  return c;
}
bool gpuSolveBatch(SolverBoard* const* boards, const SolverParams* const* prms, int n, void* stream) {
  return gpuSolveBatch(defaultCtx(), boards, prms, n, stream);
}
const GpuSolveTimes& gpuSolveLastTimes() { return defaultCtx().last; }

void gpuSolveFree(GpuSolveCtx& ctx) {
  if (ctx.dev) cudaFree(ctx.dev);
  if (ctx.pin) cudaFreeHost(ctx.pin);
  for (void*& e : ctx.evt)
    if (e) {
      cudaEventDestroy(static_cast<cudaEvent_t>(e));
      e = nullptr;
    }
  ctx.dev = nullptr;
  ctx.pin = nullptr;
  ctx.devCap = ctx.pinCap = 0;
}

bool gpuSolveBatch(GpuSolveCtx& ctx, SolverBoard* const* boards, const SolverParams* const* prms, int n, void* streamPtr) {
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  GpuSolveTimes T;
  if (n <= 0) {
    ctx.last = T;
    return true;
  }
  cudaStream_t st = static_cast<cudaStream_t>(streamPtr);
  if (!ctx.evt[0])
    for (void*& e : ctx.evt) {
      cudaEvent_t x;
      GS_CK(cudaEventCreate(&x));
      e = x;
    }
  {
    size_t cur = 0;
    GS_CK(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    if (cur < ctx.stackBytes) GS_CK(cudaDeviceSetLimit(cudaLimitStackSize, ctx.stackBytes));
  }
  if (ctx.grow.size() < size_t(n)) ctx.grow.resize(size_t(n), 1.0f);
  std::vector<BoardPlan> P(static_cast<size_t>(n));
  std::vector<uint32_t> errIn(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) errIn[size_t(i)] = boards[i]->error;

  for (uint32_t attempt = 0;; ++attempt) {
    const auto tp = clk::now();
    // ---- 자리 잡기
    size_t a = 0, b = 0, c = 0;
    for (int i = 0; i < n; ++i) take(a, P[size_t(i)].oBoard, sizeof(SolverBoard));
    for (int i = 0; i < n; ++i) {
      const SolverBoard& H = *boards[i];
      BoardPlan& p = P[size_t(i)];
      planBoard(H, ctx.grow[size_t(i)], p);
      p.oBoard = size_t(i) * al256(sizeof(SolverBoard));
      take(a, p.oBodies, sizeof(Body) * H.nbBodies);
      take(a, p.oCms, sizeof(SolverCM) * H.nbCMs);
      take(a, p.oWb, sizeof(jnt::Writeback) * p.nWb);
      take(a, p.oFricNew, sizeof(FrictionPatch) * p.fricNew);
      take(a, p.oBsi, sizeof(uint32_t) * (H.nbBodies + 1));
      p.oArt.resize(p.arts.size());
      for (size_t k = 0; k < p.arts.size(); ++k) take(a, p.oArt[k], art::artBytes(p.artCaps[k]));
      take(b, p.oPrm, sizeof(SolverParams));
      take(b, p.oIsl, sizeof(IslandIn) * H.nbIslands);
      take(b, p.oIB, sizeof(uint32_t) * p.nIB);
      take(b, p.oICM, sizeof(uint32_t) * p.nICM);
      take(b, p.oIC1D, sizeof(uint32_t) * p.nIC1D);
      take(b, p.oIA, sizeof(uint32_t) * p.nIA);
      take(b, p.oPatch, sizeof(ContactPatchIn) * p.nPatch);
      take(b, p.oContact, sizeof(ContactIn) * p.nContact);
      take(b, p.oAct, sizeof(uint32_t) * H.nbActivatedCMs);
      take(b, p.oReset, sizeof(uint32_t) * H.nbResetCMs);
      take(b, p.oC1D, sizeof(Constraint1DIn) * H.nbC1D);
      take(b, p.oJd, sizeof(jnt::D6Data) * p.nJd);
      take(b, p.oFricPrev, sizeof(FrictionPatch) * (p.fricPrev ? p.fricPrev : 1));
      take(b, p.oArtPtr, sizeof(art::Articulation*) * H.nbArts);
      const size_t na = H.nbArts;
      take(c, p.oVels, sizeof(SBodyVel) * p.pool);
      take(c, p.oTxI, sizeof(SBodyTxI) * p.pool);
      take(c, p.oDatas, sizeof(SBodyData) * p.pool);
      take(c, p.oDescs, sizeof(SDesc) * p.desc);
      take(c, p.oOrd, sizeof(SDesc) * p.desc);
      take(c, p.oTemp, sizeof(SDesc) * p.desc);
      take(c, p.oHead, sizeof(BatchHeader) * p.desc);
      take(c, p.oPart, sizeof(uint32_t) * p.part);
      take(c, p.oArena, p.arena);
      take(c, p.oCorr, sizeof(CorrelationBuffer));
      take(c, p.oCbuf, sizeof(ContactPoint) * MAX_CONTACTS);
      take(c, p.oRows, sizeof(jnt::Row) * jnt::MAX_CONSTRAINT_ROWS * 4);
      take(c, p.oLists, sizeof(art::StaticLists) * na);
      take(c, p.oS1, sizeof(SDesc) * na * p.stat);
      take(c, p.oSC, sizeof(SDesc) * na * p.stat);
      take(c, p.oN1, sizeof(uint32_t) * na);
      take(c, p.oNC, sizeof(uint32_t) * na);
      take(c, p.oBatch, sizeof(uint32_t) * na);
      take(c, p.oProg, sizeof(ArtProgress) * na);
    }
    const size_t endA = a, endB = a + b, total = a + b + c;
    if (total > ctx.devCap) {
      if (ctx.dev) GS_CK(cudaFree(ctx.dev));
      ctx.dev = nullptr;
      const size_t cap = total + total / 4;
      GS_CK(cudaMalloc(&ctx.dev, cap));
      ctx.devCap = cap;
    }
    if (endB > ctx.pinCap) {
      if (ctx.pin) GS_CK(cudaFreeHost(ctx.pin));
      ctx.pin = nullptr;
      const size_t cap = endB + endB / 4;
      GS_CK(cudaHostAlloc(reinterpret_cast<void**>(&ctx.pin), cap, cudaHostAllocDefault));
      ctx.pinCap = cap;
    }
    uint8_t* const D = ctx.dev;
    uint8_t* const S = ctx.pin;
    // ---- 담기 (구역 B 오프셋에 endA 를 더한다)
    for (int i = 0; i < n; ++i) {
      const SolverBoard& H = *boards[i];
      BoardPlan& p = P[size_t(i)];
      auto put = [&](size_t off, const void* src, size_t bytes) {
        if (bytes) memcpy(S + off, src, bytes);
      };
      const size_t bB = endA, bC = endB;
      put(p.oBodies, H.bodies, sizeof(Body) * H.nbBodies);
      put(p.oCms, H.cms, sizeof(SolverCM) * H.nbCMs);
      put(p.oWb, H.writebacks, sizeof(jnt::Writeback) * p.nWb);
      put(p.oBsi, H.bodySolverIndex, sizeof(uint32_t) * (H.nbBodies + 1));
      art::Articulation** ptrs = reinterpret_cast<art::Articulation**>(S + bB + p.oArtPtr);
      for (uint32_t k = 0; k < H.nbArts; ++k) ptrs[k] = nullptr;
      for (size_t k = 0; k < p.arts.size(); ++k) {
        art::artRepack(*reinterpret_cast<art::Articulation*>(S + p.oArt[k]), artAt(H, p.arts[k]), p.artCaps[k]);
        ptrs[p.arts[k]] = reinterpret_cast<art::Articulation*>(D + p.oArt[k]);
      }
      put(bB + p.oPrm, prms[i], sizeof(SolverParams));
      put(bB + p.oIsl, H.islands, sizeof(IslandIn) * H.nbIslands);
      put(bB + p.oIB, H.islandBodies, sizeof(uint32_t) * p.nIB);
      put(bB + p.oICM, H.islandCMs, sizeof(uint32_t) * p.nICM);
      put(bB + p.oIC1D, H.islandC1Ds, sizeof(uint32_t) * p.nIC1D);
      put(bB + p.oIA, H.islandArts, sizeof(uint32_t) * p.nIA);
      put(bB + p.oPatch, H.patches, sizeof(ContactPatchIn) * p.nPatch);
      put(bB + p.oContact, H.contacts, sizeof(ContactIn) * p.nContact);
      put(bB + p.oAct, H.activatedCMs, sizeof(uint32_t) * H.nbActivatedCMs);
      put(bB + p.oReset, H.resetCMs, sizeof(uint32_t) * H.nbResetCMs);
      put(bB + p.oC1D, H.c1d, sizeof(Constraint1DIn) * H.nbC1D);
      put(bB + p.oJd, H.jointData, sizeof(jnt::D6Data) * p.nJd);
      put(bB + p.oFricPrev, H.friction[H.frictionCurIdx].data, sizeof(FrictionPatch) * p.fricPrev);
      // 판 구조체: 스칼라는 호스트 그대로, 포인터·용량은 장치 조각
      SolverBoard G = H;
      auto dp = [&](size_t off) { return D + off; };
      G.bodies = reinterpret_cast<Body*>(dp(p.oBodies));
      G.cms = reinterpret_cast<SolverCM*>(dp(p.oCms));
      G.patches = reinterpret_cast<const ContactPatchIn*>(dp(bB + p.oPatch));
      G.contacts = reinterpret_cast<const ContactIn*>(dp(bB + p.oContact));
      G.islands = reinterpret_cast<const IslandIn*>(dp(bB + p.oIsl));
      G.islandBodies = reinterpret_cast<const uint32_t*>(dp(bB + p.oIB));
      G.islandCMs = reinterpret_cast<const uint32_t*>(dp(bB + p.oICM));
      G.activatedCMs = reinterpret_cast<const uint32_t*>(dp(bB + p.oAct));
      G.resetCMs = reinterpret_cast<const uint32_t*>(dp(bB + p.oReset));
      G.c1d = reinterpret_cast<const Constraint1DIn*>(dp(bB + p.oC1D));
      G.islandC1Ds = reinterpret_cast<const uint32_t*>(dp(bB + p.oIC1D));
      G.jointData = reinterpret_cast<const jnt::D6Data*>(dp(bB + p.oJd));
      G.writebacks = reinterpret_cast<jnt::Writeback*>(dp(p.oWb));
      G.rowScratch = reinterpret_cast<jnt::Row*>(dp(bC + p.oRows));
      G.vels = reinterpret_cast<SBodyVel*>(dp(bC + p.oVels));
      G.txI = reinterpret_cast<SBodyTxI*>(dp(bC + p.oTxI));
      G.datas = reinterpret_cast<SBodyData*>(dp(bC + p.oDatas));
      G.poolCap = p.pool;
      G.descs = reinterpret_cast<SDesc*>(dp(bC + p.oDescs));
      G.ordered = reinterpret_cast<SDesc*>(dp(bC + p.oOrd));
      G.temp = reinterpret_cast<SDesc*>(dp(bC + p.oTemp));
      G.headers = reinterpret_cast<BatchHeader*>(dp(bC + p.oHead));
      G.descCap = p.desc;
      G.partitionCounts = reinterpret_cast<uint32_t*>(dp(bC + p.oPart));
      G.partitionCap = p.part;
      G.bodySolverIndex = reinterpret_cast<uint32_t*>(dp(p.oBsi));
      G.constraints.base = dp(bC + p.oArena);
      G.constraints.cap = p.arena;
      const uint32_t prevIdx = H.frictionCurIdx, newIdx = prevIdx ^ 1u;
      G.friction[prevIdx].data = reinterpret_cast<FrictionPatch*>(dp(bB + p.oFricPrev));
      G.friction[prevIdx].cap = p.fricPrev;
      G.friction[newIdx].data = reinterpret_cast<FrictionPatch*>(dp(p.oFricNew));
      G.friction[newIdx].cap = p.fricNew;
      G.friction[newIdx].size = 0;  // solverStepBegin 이 어차피 0 으로
      G.corr = reinterpret_cast<CorrelationBuffer*>(dp(bC + p.oCorr));
      G.contactBuffer = reinterpret_cast<ContactPoint*>(dp(bC + p.oCbuf));
      G.arts = nullptr;
      G.artPtrs = reinterpret_cast<art::Articulation* const*>(dp(bB + p.oArtPtr));
      G.artLists = reinterpret_cast<art::StaticLists*>(dp(bC + p.oLists));
      G.artStatic1D = reinterpret_cast<SDesc*>(dp(bC + p.oS1));
      G.artStaticContact = reinterpret_cast<SDesc*>(dp(bC + p.oSC));
      G.artNbStatic1D = reinterpret_cast<uint32_t*>(dp(bC + p.oN1));
      G.artNbStaticContact = reinterpret_cast<uint32_t*>(dp(bC + p.oNC));
      G.artStaticCap = p.stat;
      G.artBatchIndex = reinterpret_cast<uint32_t*>(dp(bC + p.oBatch));
      G.artProg = reinterpret_cast<ArtProgress*>(dp(bC + p.oProg));
      memcpy(S + p.oBoard, &G, sizeof(SolverBoard));
    }
    const auto tu = clk::now();
    T.packMs += std::chrono::duration<double, std::milli>(tu - tp).count();
    cudaEvent_t e0 = static_cast<cudaEvent_t>(ctx.evt[0]), e1 = static_cast<cudaEvent_t>(ctx.evt[1]), e2 = static_cast<cudaEvent_t>(ctx.evt[2]),
                e3 = static_cast<cudaEvent_t>(ctx.evt[3]);
    GS_CK(cudaEventRecord(e0, st));
    GS_CK(cudaMemcpyAsync(D, S, endB, cudaMemcpyHostToDevice, st));
    GS_CK(cudaEventRecord(e1, st));
    kSolveBoards<<<n, ctx.threads, 0, st>>>(reinterpret_cast<SolverBoard*>(D), reinterpret_cast<const SolverParams*>(D + endA));
    GS_CK(cudaGetLastError());
    GS_CK(cudaEventRecord(e2, st));
    GS_CK(cudaMemcpyAsync(S, D, endA, cudaMemcpyDeviceToHost, st));
    GS_CK(cudaEventRecord(e3, st));
    GS_CK(cudaEventSynchronize(e3));
    float up = 0, kr = 0, dn = 0;
    GS_CK(cudaEventElapsedTime(&up, e0, e1));
    GS_CK(cudaEventElapsedTime(&kr, e1, e2));
    GS_CK(cudaEventElapsedTime(&dn, e2, e3));
    T.upMs += up;
    T.kernelMs += kr;
    T.downMs += dn;
    T.upBytes += endB;
    T.downBytes += endA;
    T.devBytes = total;
    T.pinBytes = ctx.pinCap;
    // ---- 용량 넘침이면 그 판을 두 배로 늘려 다시 (호스트 상태는 아직 그대로)
    bool retry = false;
    for (int i = 0; i < n; ++i) {
      const SolverBoard* G = reinterpret_cast<const SolverBoard*>(S + P[size_t(i)].oBoard);
      if ((G->error & ~errIn[size_t(i)]) & kCapErrs) {
        ctx.grow[size_t(i)] *= 2.0f;
        retry = true;
      }
    }
    if (retry && attempt < ctx.maxRetries) {
      ++T.retries;
      continue;
    }
    // ---- 되받기
    const auto tk = clk::now();
    for (int i = 0; i < n; ++i) {
      SolverBoard& H = *boards[i];
      const BoardPlan& p = P[size_t(i)];
      const SolverBoard& G = *reinterpret_cast<const SolverBoard*>(S + p.oBoard);
      if (H.nbBodies) memcpy(H.bodies, S + p.oBodies, sizeof(Body) * H.nbBodies);
      if (H.nbCMs) memcpy(H.cms, S + p.oCms, sizeof(SolverCM) * H.nbCMs);
      if (p.nWb) memcpy(H.writebacks, S + p.oWb, sizeof(jnt::Writeback) * p.nWb);
      memcpy(H.bodySolverIndex, S + p.oBsi, sizeof(uint32_t) * (H.nbBodies + 1));
      for (size_t k = 0; k < p.arts.size(); ++k) {
        art::Articulation& ha = artAt(H, p.arts[k]);
        const art::ArtCaps hc = ha.cap;
        art::artRepack(ha, *reinterpret_cast<const art::Articulation*>(S + p.oArt[k]), hc);
      }
      H.error = G.error;
      const uint32_t newIdx = G.frictionCurIdx;
      FrictionArena& hf = H.friction[newIdx];
      const FrictionArena& gf = G.friction[newIdx];
      if (gf.size > hf.cap) {
        H.error |= SV_ERR_FRICTION;
      } else if (gf.size) {
        memcpy(hf.data, S + p.oFricNew, sizeof(FrictionPatch) * gf.size);
      }
      hf.size = gf.size;
      hf.overflow = gf.overflow;
      H.frictionCurIdx = G.frictionCurIdx;
      H.constraints.size = G.constraints.size;
      H.constraints.overflow = G.constraints.overflow;
      H.plan = G.plan;
      H.statBatches = G.statBatches;
      H.statBlock4 = G.statBlock4;
      H.statSingle = G.statSingle;
      H.statHeaders = G.statHeaders;
      H.statMaxPartitions = G.statMaxPartitions;
      H.statFreeBatches = G.statFreeBatches;
      H.stat1DBlock4 = G.stat1DBlock4;
      H.stat1DSingle = G.stat1DSingle;
      H.stat1DZeroRows = G.stat1DZeroRows;
      H.statMaxArena = G.statMaxArena;
      H.statMaxFriction = G.statMaxFriction;
      H.statMaxDescs = G.statMaxDescs;
      H.statArtExtContacts = G.statArtExtContacts;
      H.statArtStaticContacts = G.statArtStaticContacts;
      H.statArtExt1D = G.statArtExt1D;
      H.statArtStatic1D = G.statArtStatic1D;
    }
    T.unpackMs += std::chrono::duration<double, std::milli>(clk::now() - tk).count();
    break;
  }
  T.totalMs = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  ctx.last = T;
  return true;
}

}  // namespace sv
}  // namespace eng
