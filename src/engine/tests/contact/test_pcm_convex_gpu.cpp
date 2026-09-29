// 층 2 시험 (contact): PCM 볼록-볼록 CUDA 판(판 N 개 동시) = 층 1 C++ = PhysX 5.6.1, 비트 사슬 + 처리량.
//   test_pcm_convex_gpu [--envs E] [--pairs P] [--frames F] [--seed S] [--hulls H] [--bench-envs B]
// 판마다 같은 쌍 목록(볼록·척도 공유), 자세는 판·쌍마다 다르게 흔든다. 매 프레임 층 2 결과·다양체를 층 1 과, 층 1 을 PhysX 와 비교.
#include <cuda_runtime.h>

#include <chrono>

#include "px_bridge.h"
#include "core/contact/hull_pack.h"
#include "pcm_gpu_launch.h"

using namespace physx;
namespace ec = eng::contact;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct Walker {  // 판·쌍마다 자세를 흔드는 난수 (재현 가능)
  std::mt19937 rng;
  PxTransform tf0, tf1;
  float r, step;
};

int main(int argc, char** argv) {
  int envs = 32, pairs = 300, frames = 30, seed = 11, nhulls = 120, benchEnvs = 1024;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hulls") && i + 1 < argc) nhulls = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bench-envs") && i + 1 < argc) benchEnvs = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);

  // ---- 볼록 굽기 -> 한 덩어리로 포장해 GPU 로
  std::vector<PxConvexMesh*> hulls;
  for (int h = 0; h < nhulls; ++h) {
    const int n = (h % 3 == 0) ? 40 + int(P(rng) * 60) : 8 + int(P(rng) * 20);
    PxConvexMesh* m = cxt::cookHull(phys, cxt::randomCloud(rng, h % 4, n), (h % 5) < 2, 64);
    if (m) hulls.push_back(m);
  }
  std::vector<size_t> hoff(hulls.size());
  size_t total = 0;
  for (size_t h = 0; h < hulls.size(); ++h) { hoff[h] = total; total += ec::hullPackedBytes(*cxt::hullOf(hulls[h])); }
  std::vector<uint8_t> hpack(total);
  uint8_t* dpack = nullptr;
  CK(cudaMalloc(&dpack, total));
  for (size_t h = 0; h < hulls.size(); ++h) ec::packHull(*cxt::hullOf(hulls[h]), hpack.data() + hoff[h], uintptr_t(dpack) + hoff[h]);
  CK(cudaMemcpy(dpack, hpack.data(), total, cudaMemcpyHostToDevice));
  printf("볼록 %zu 개, 포장 %.1f KB\n", hulls.size(), total / 1024.0);

  // ---- 쌍 목록 (판끼리 공유)
  auto randScale = [&](int c) -> PxMeshScale {
    switch (c % 4) {
      case 0: return PxMeshScale(1.0f);
      case 1: return PxMeshScale(0.5f + 1.5f * P(rng));
      case 2: return PxMeshScale(PxVec3(0.4f + P(rng), 0.4f + P(rng), 0.4f + P(rng)));
      default: { PxQuat r(U(rng), U(rng), U(rng), U(rng)); r.normalize(); return PxMeshScale(PxVec3(0.4f + P(rng), 0.4f + P(rng), 0.4f + P(rng)), r); }
    }
  };
  std::vector<PxConvexMeshGeometry> g0(pairs), g1(pairs);
  std::vector<ec::ConvexPair> hostPairs(pairs), devPairs(pairs);
  std::vector<float> radius(pairs);
  for (int p = 0; p < pairs; ++p) {
    const size_t a = size_t(P(rng) * hulls.size()) % hulls.size(), b = size_t(P(rng) * hulls.size()) % hulls.size();
    g0[p] = PxConvexMeshGeometry(hulls[a], randScale(p));
    g1[p] = PxConvexMeshGeometry(hulls[b], randScale(p / 4));
    hostPairs[p].g0 = cxt::toE(g0[p]);
    hostPairs[p].g1 = cxt::toE(g1[p]);
    devPairs[p] = hostPairs[p];
    devPairs[p].g0.hullData = reinterpret_cast<const eng::px::Gu::ConvexHullData*>(dpack + hoff[a]);
    devPairs[p].g1.hullData = reinterpret_cast<const eng::px::Gu::ConvexHullData*>(dpack + hoff[b]);
    radius[p] = static_cast<Gu::ConvexMesh*>(hulls[a])->getLocalBoundsFast().mExtents.magnitude() * g0[p].scale.scale.maxElement() +
                static_cast<Gu::ConvexMesh*>(hulls[b])->getLocalBoundsFast().mExtents.magnitude() * g1[p].scale.scale.maxElement();
  }
  ec::ConvexPair* dPairs = nullptr;
  CK(cudaMalloc(&dPairs, sizeof(ec::ConvexPair) * pairs));
  CK(cudaMemcpy(dPairs, devPairs.data(), sizeof(ec::ConvexPair) * pairs, cudaMemcpyHostToDevice));

  const float contactDist = 0.04f, meshMargin = 0.01f * tol.length, tolLen = tol.length;
  const Gu::NarrowPhaseParams npP(contactDist, meshMargin, tolLen);
  const eng::px::Gu::NarrowPhaseParams npE(contactDist, meshMargin, tolLen);

  CK(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));
  const int threads = 64;

  // ---- 비교 실행: E 판 × P 쌍 × F 프레임
  const size_t N = size_t(envs) * pairs;
  std::vector<Walker> walk(N);
  for (int e = 0; e < envs; ++e)
    for (int p = 0; p < pairs; ++p) {
      Walker& w = walk[size_t(e) * pairs + p];
      w.rng.seed(uint32_t(seed * 7919 + e * 104729 + p));
      std::uniform_real_distribution<float> u(-1.0f, 1.0f), q(0.0f, 1.0f);
      PxQuat q0(u(w.rng), u(w.rng), u(w.rng), u(w.rng)), q1(u(w.rng), u(w.rng), u(w.rng), u(w.rng));
      q0.normalize(); q1.normalize();
      PxVec3 d(u(w.rng), u(w.rng), u(w.rng)); d.normalize();
      w.r = radius[p];
      w.step = 0.01f * w.r;
      w.tf0 = PxTransform(PxVec3(u(w.rng), u(w.rng), u(w.rng)), q0);
      w.tf1 = PxTransform(w.tf0.p + d * (w.r * (0.2f + 0.85f * q(w.rng))), q1);
    }
  std::vector<Gu::LargePersistentContactManifold> manP(N);
  std::vector<eng::px::Gu::LargePersistentContactManifold> manE(N);
  for (size_t k = 0; k < N; ++k) { manP[k].clearManifold(); manE[k].clearManifold(); }
  eng::px::Gu::LargePersistentContactManifold* dMan = nullptr;
  ec::PairPose* dPose = nullptr;
  ec::PairResult* dOut = nullptr;
  CK(cudaMalloc(&dMan, sizeof(eng::px::Gu::LargePersistentContactManifold) * N));
  CK(cudaMalloc(&dPose, sizeof(ec::PairPose) * N));
  CK(cudaMalloc(&dOut, sizeof(ec::PairResult) * N));
  cxt::gpuInitManifolds(dMan, N);
  CK(cudaGetLastError());
  std::vector<ec::PairPose> pose(N);
  std::vector<ec::PairResult> outG(N), outH(N);
  std::vector<eng::px::Gu::LargePersistentContactManifold> manG(N);
  static PxContactBuffer bufP;
  static eng::px::PxContactBuffer bufE;

  cxt::Tally t21("층2=층1 결과"), t21m("층2=층1 다양체"), t10("층1=PhysX 결과"), t10m("층1=PhysX 다양체");
  long long contacts = 0;
  double gpuMs = 0.0;
  auto cmpContacts = [](cxt::Tally& t, const auto& a, const auto& b, uint32_t n, long long c, long long f) {
    for (uint32_t i = 0; i < n; ++i) {
      t.f(a[i].normal.x, b[i].normal.x, c, f); t.f(a[i].normal.y, b[i].normal.y, c, f); t.f(a[i].normal.z, b[i].normal.z, c, f);
      t.f(a[i].separation, b[i].separation, c, f);
      t.f(a[i].point.x, b[i].point.x, c, f); t.f(a[i].point.y, b[i].point.y, c, f); t.f(a[i].point.z, b[i].point.z, c, f);
      t.u(a[i].internalFaceIndex1, b[i].internalFaceIndex1, c, f);
    }
  };
  auto cmpMan = [](cxt::Tally& t, const auto& a, const auto& b, long long c, long long f) {
    t.u(a.mNumContacts, b.mNumContacts, c, f);
    t.u(a.mNumWarmStartPoints, b.mNumWarmStartPoints, c, f);
    for (int k = 0; k < a.mNumWarmStartPoints && k < 4; ++k) { t.u(a.mAIndice[k], b.mAIndice[k], c, f); t.u(a.mBIndice[k], b.mBIndice[k], c, f); }
    const int nm = a.mNumContacts < b.mNumContacts ? a.mNumContacts : b.mNumContacts;
    for (int k = 0; k < nm; ++k) {
      float x[12], y[12];
      memcpy(x, &a.mContactPointsBuff[k], 48);
      memcpy(y, &b.mContactPointsBuff[k], 48);
      for (int q = 0; q < 12; ++q) t.f(x[q], y[q], c, f);
    }
    float x[12], y[12];
    memcpy(x, &a.mRelativeTransform, 32); memcpy(x + 8, &a.mQuatA, 16);
    memcpy(y, &b.mRelativeTransform, 32); memcpy(y + 8, &b.mQuatA, 16);
    for (int q = 0; q < 12; ++q) t.f(x[q], y[q], c, f);
  };

  for (int f = 0; f < frames; ++f) {
    for (size_t k = 0; k < N; ++k) {
      pose[k].tf0 = eng::px::PxTransform32(cxt::toE(walk[k].tf0));
      pose[k].tf1 = eng::px::PxTransform32(cxt::toE(walk[k].tf1));
    }
    CK(cudaMemcpy(dPose, pose.data(), sizeof(ec::PairPose) * N, cudaMemcpyHostToDevice));
    gpuMs += cxt::gpuConvexConvex(envs, threads, dPairs, pairs, dPose, dMan, dOut, contactDist, meshMargin, tolLen, 1);
    CK(cudaGetLastError());
    CK(cudaMemcpy(outG.data(), dOut, sizeof(ec::PairResult) * N, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(manG.data(), dMan, sizeof(eng::px::Gu::LargePersistentContactManifold) * N, cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < N; ++k) {
      const int p = int(k % pairs);
      bool rp;
      {
        cxt::FtzScope fz;
        ec::convexConvexPair(hostPairs[p], pose[k], npE, manE[k], bufE, outH[k]);
        bufP.reset();
        Gu::Cache cp; cp.setManifold(&manP[k]);
        rp = Gu::pcmContactConvexConvex(g0[p], g1[p], PxTransform32(walk[k].tf0), PxTransform32(walk[k].tf1), npP, cp, bufP, NULL);
      }
      const long long c = (long long)k;
      // 층 2 vs 층 1
      t21.u(outG[k].ret, outH[k].ret, c, f);
      t21.u(outG[k].count, outH[k].count, c, f);
      t21.u(outG[k].overflow, 0, c, f);
      const uint32_t n21 = outH[k].count < outG[k].count ? outH[k].count : outG[k].count;
      cmpContacts(t21, outG[k].c, outH[k].c, n21 < uint32_t(ec::kMaxOutContacts) ? n21 : uint32_t(ec::kMaxOutContacts), c, f);
      cmpMan(t21m, manG[k], manE[k], c, f);
      // 층 1 vs PhysX
      t10.u(rp, outH[k].ret, c, f);
      t10.u(bufP.count, outH[k].count, c, f);
      const uint32_t n10 = bufP.count < outH[k].count ? bufP.count : outH[k].count;
      cmpContacts(t10, bufP.contacts, outH[k].c, n10 < uint32_t(ec::kMaxOutContacts) ? n10 : uint32_t(ec::kMaxOutContacts), c, f);
      cmpMan(t10m, manP[k], manE[k], c, f);
      contacts += bufP.count;
      // 다음 자세
      Walker& w = walk[k];
      std::uniform_real_distribution<float> u(-1.0f, 1.0f), q(0.0f, 1.0f);
      const float s = q(w.rng) < 0.05f ? 10.0f : 1.0f;
      w.tf1.p += PxVec3(u(w.rng), u(w.rng), u(w.rng)) * (w.step * s);
      PxVec3 ax(u(w.rng), u(w.rng), u(w.rng));
      ax = ax.isZero() ? PxVec3(0, 0, 1) : ax.getNormalized();
      w.tf1.q = (PxQuat(0.02f * s * u(w.rng), ax) * w.tf1.q).getNormalized();
      w.tf0.p += PxVec3(u(w.rng), u(w.rng), u(w.rng)) * (0.3f * w.step);
      if ((w.tf1.p - w.tf0.p).magnitude() > 1.3f * w.r) w.tf1.p = w.tf0.p + (w.tf1.p - w.tf0.p) * 0.5f;
    }
  }
  printf("판 %d × 쌍 %d × 프레임 %d = 쌍·프레임 %zu, 접촉점 합 %lld, GPU 커널 합 %.1f ms\n", envs, pairs, frames, N * frames, contacts, gpuMs);
  uint64_t bad = 0;
  for (const cxt::Tally* t : {&t21, &t21m, &t10, &t10m}) { t->print(); bad += t->bad; }

  // ---- 처리량: 판 benchEnvs 개 (같은 쌍 목록), 한 프레임 커널 시간
  if (benchEnvs > 0) {
    const size_t NB = size_t(benchEnvs) * pairs;
    std::vector<ec::PairPose> bp(NB);
    for (size_t k = 0; k < NB; ++k) bp[k] = pose[k % N];
    ec::PairPose* dbp; eng::px::Gu::LargePersistentContactManifold* dbm; ec::PairResult* dbo;
    CK(cudaMalloc(&dbp, sizeof(ec::PairPose) * NB));
    CK(cudaMalloc(&dbm, sizeof(eng::px::Gu::LargePersistentContactManifold) * NB));
    CK(cudaMalloc(&dbo, sizeof(ec::PairResult) * NB));
    CK(cudaMemcpy(dbp, bp.data(), sizeof(ec::PairPose) * NB, cudaMemcpyHostToDevice));
    for (int threadsB : {32, 64, 128}) {
      cxt::gpuInitManifolds(dbm, NB);
      const float ms0 = cxt::gpuConvexConvex(benchEnvs, threadsB, dPairs, pairs, dbp, dbm, dbo, contactDist, meshMargin, tolLen, 1);  // 첫 프레임(다양체 비어 있음 -> 전부 GJK/EPA)
      // 같은 자세 반복 = 다양체 재사용 경로(실제 판에서 대부분)
      const float ms = cxt::gpuConvexConvex(benchEnvs, threadsB, dPairs, pairs, dbp, dbm, dbo, contactDist, meshMargin, tolLen, 5);
      CK(cudaGetLastError());
      printf("  첫 프레임(전체 생성) %.3f ms\n", ms0);
      printf("처리량 (블록당 스레드 %d): 판 %d × 쌍 %d = %zu 쌍, 프레임당 %.3f ms -> %.3g 쌍/초\n", threadsB, benchEnvs, pairs, NB, ms, NB / (ms * 1e-3));
    }
    // 대조: PhysX CPU 단일 스레드, 같은 쌍 한 프레임씩 (다양체 재사용 경로)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      size_t calls = 0;
      for (int r = 0; r < 3; ++r)
        for (size_t k = 0; k < N; ++k) {
          cxt::FtzScope fz;
          bufP.reset();
          Gu::Cache cp; cp.setManifold(&manP[k]);
          Gu::pcmContactConvexConvex(g0[k % pairs], g1[k % pairs], PxTransform32(walk[k].tf0), PxTransform32(walk[k].tf1), npP, cp, bufP, NULL);
          ++calls;
        }
      const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      printf("대조 PhysX CPU 1 스레드: %.3g 쌍/초\n", calls / s);
    }
  }
  printf(bad ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return bad ? 1 : 0;
}
