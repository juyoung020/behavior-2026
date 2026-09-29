// 층 2 시험 (contact): CUDA 판 N 개 = 층 1 C++ = PhysX 5.6.1, 비트 사슬 + 처리량.
//  1) 수학: GPU 의 rcpps·rsqrtps 표 흉내와 glibc acosf 이식이 호스트(= 진짜 명령·libm 과 이미 2^32 전수 일치)와 2^32 전수 비트 동일?
//  2) 모양 쌍 전부(구·평면·캡슐·상자·볼록, 뒤집힌 순서 포함): 판 E × 쌍 P × 프레임 F, 결과·접촉점·다양체를 층2=층1, 층1=PhysX 로 비교
//  3) 처리량: 판 B 개 한 프레임 커널 시간, 대조 PhysX CPU 1 스레드
//   test_contact_gpu [--envs E] [--pairs P] [--frames F] [--seed S] [--bench-envs B] [--no-math]
#include <cuda_runtime.h>

#include <chrono>
#include <thread>

#include "px_bridge.h"
#include "core/contact/hull_pack.h"
#include "core/contact/px/approx.h"
#include "core/common/glibc_trig.h"
#include "contact_gpu_launch.h"

using namespace physx;
namespace ec = eng::contact;
namespace ep = eng::px;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

static uint64_t mathSweep() {
  const uint32_t chunk = 1u << 24;
  uint32_t *dR, *dS, *dA;
  CK(cudaMalloc(&dR, chunk * 4ull)); CK(cudaMalloc(&dS, chunk * 4ull)); CK(cudaMalloc(&dA, chunk * 4ull));
  std::vector<uint32_t> hR(chunk), hS(chunk), hA(chunk);
  ep::ApproxTables tab;
  ep::buildApproxTables(tab);
  uint64_t bad[3] = {0, 0, 0};
  uint64_t first[3] = {~0ull, ~0ull, ~0ull};
  for (uint64_t lo = 0; lo < (1ull << 32); lo += chunk) {
    cxt::gpuMathSweep(uint32_t(lo), chunk, dR, dS, dA);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hR.data(), dR, chunk * 4ull, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hS.data(), dS, chunk * 4ull, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hA.data(), dA, chunk * 4ull, cudaMemcpyDeviceToHost));
    const int nt = 16;
    uint64_t b[nt][3] = {}, f[nt][3];
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));  // GPU -ftz=true 와 짝
        f[t][0] = f[t][1] = f[t][2] = ~0ull;
        for (uint32_t i = chunk / nt * t; i < chunk / nt * (t + 1); ++i) {
          const float x = ep::em_u2f(uint32_t(lo) + i);
          const uint32_t r = ep::em_f2u(ep::approxRcp(x, tab)), s = ep::em_f2u(ep::approxRsqrt(x, tab)),
                         a = ep::em_f2u(eng::glibc::acosf(x));
          if (r != hR[i]) { if (!b[t][0]) f[t][0] = lo + i; ++b[t][0]; }
          if (s != hS[i]) { if (!b[t][1]) f[t][1] = lo + i; ++b[t][1]; }
          if (a != hA[i]) { if (!b[t][2]) f[t][2] = lo + i; ++b[t][2]; }
        }
      });
    for (auto& x : th) x.join();
    for (int t = 0; t < nt; ++t)
      for (int k = 0; k < 3; ++k) { bad[k] += b[t][k]; if (f[t][k] < first[k]) first[k] = f[t][k]; }
  }
  const char* nm[3] = {"rcpps 흉내", "rsqrtps 흉내", "acosf"};
  for (int k = 0; k < 3; ++k) {
    printf("  GPU %-14s 입력 4294967296 개, 호스트와 비트 다름 %" PRIu64, nm[k], bad[k]);
    if (bad[k]) printf("  첫 다름 0x%08" PRIx64, first[k]);
    printf("\n");
  }
  cudaFree(dR); cudaFree(dS); cudaFree(dA);
  return bad[0] + bad[1] + bad[2];
}

// 다양체 칸 내용 비교 (GPU 에서 가져온 칸은 포인터가 장치 주소라 버퍼를 직접 읽는다)
static void cmpSlot(cxt::Tally& t, const ec::ManifoldSlot& a, const ec::ManifoldSlot& b, long long c, long long f) {
  t.u(a.kind, b.kind, c, f);
  if (a.kind == 0 || a.kind != b.kind) return;
  const ep::Gu::PersistentContactManifold& ma = *reinterpret_cast<const ep::Gu::PersistentContactManifold*>(a.storage);
  const ep::Gu::PersistentContactManifold& mb = *reinterpret_cast<const ep::Gu::PersistentContactManifold*>(b.storage);
  t.u(ma.mNumContacts, mb.mNumContacts, c, f);
  t.u(ma.mNumWarmStartPoints, mb.mNumWarmStartPoints, c, f);
  const ep::Gu::PersistentContact* pa = a.kind == 1 ? reinterpret_cast<const ep::Gu::SpherePersistentContactManifold*>(a.storage)->mContactPointsBuff
                                                    : reinterpret_cast<const ep::Gu::LargePersistentContactManifold*>(a.storage)->mContactPointsBuff;
  const ep::Gu::PersistentContact* pb = b.kind == 1 ? reinterpret_cast<const ep::Gu::SpherePersistentContactManifold*>(b.storage)->mContactPointsBuff
                                                    : reinterpret_cast<const ep::Gu::LargePersistentContactManifold*>(b.storage)->mContactPointsBuff;
  const int n = ma.mNumContacts < mb.mNumContacts ? ma.mNumContacts : mb.mNumContacts;
  for (int k = 0; k < n; ++k) {
    float x[12], y[12];
    memcpy(x, &pa[k], 48); memcpy(y, &pb[k], 48);
    for (int q = 0; q < 12; ++q) t.f(x[q], y[q], c, f);
  }
  float x[12], y[12];
  memcpy(x, &ma.mRelativeTransform, 32); memcpy(x + 8, &ma.mQuatA, 16);
  memcpy(y, &mb.mRelativeTransform, 32); memcpy(y + 8, &mb.mQuatA, 16);
  for (int q = 0; q < 12; ++q) t.f(x[q], y[q], c, f);
}
static void cmpPhysxSlot(cxt::Tally& t, cxt::PhysxSlot& a, const ec::ManifoldSlot& b, long long c, long long f) {
  t.u(a.kind, b.kind, c, f);
  if (a.kind == 0 || a.kind != b.kind) return;
  Gu::PersistentContactManifold& ma = *a.get();
  const ep::Gu::PersistentContactManifold& mb = *reinterpret_cast<const ep::Gu::PersistentContactManifold*>(b.storage);
  t.u(ma.mNumContacts, mb.mNumContacts, c, f);
  t.u(ma.mNumWarmStartPoints, mb.mNumWarmStartPoints, c, f);
  const int n = ma.mNumContacts < mb.mNumContacts ? ma.mNumContacts : mb.mNumContacts;
  for (int k = 0; k < n; ++k) {
    float x[12], y[12];
    memcpy(x, &ma.mContactPoints[k], 48); memcpy(y, &mb.mContactPoints[k], 48);
    for (int q = 0; q < 12; ++q) t.f(x[q], y[q], c, f);
  }
}
template <class A, class B>
static void cmpContacts(cxt::Tally& t, const A* a, const B* b, uint32_t n, long long c, long long f) {
  for (uint32_t i = 0; i < n; ++i) {
    t.f(a[i].normal.x, b[i].normal.x, c, f); t.f(a[i].normal.y, b[i].normal.y, c, f); t.f(a[i].normal.z, b[i].normal.z, c, f);
    t.f(a[i].separation, b[i].separation, c, f);
    t.f(a[i].point.x, b[i].point.x, c, f); t.f(a[i].point.y, b[i].point.y, c, f); t.f(a[i].point.z, b[i].point.z, c, f);
    t.u(a[i].internalFaceIndex1, b[i].internalFaceIndex1, c, f);
  }
}

int main(int argc, char** argv) {
  int envs = 16, pairs = 700, frames = 30, seed = 21, benchEnvs = 512;
  bool math = true;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bench-envs") && i + 1 < argc) benchEnvs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-math")) math = false;
  }
  uint64_t badAll = 0;
  CK(cudaSetDevice(0));
  CK(cudaError_t(cxt::gpuUploadApprox()));
  if (math) {
    printf("[1] 수학 함수 2^32 전수 (GPU vs 호스트)\n");
    badAll += mathSweep();
  }

  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);

  std::vector<PxConvexMesh*> hulls;
  for (int h = 0; h < 80; ++h) {
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
  auto devHull = [&](const ep::Gu::ConvexHullData* host) -> const ep::Gu::ConvexHullData* {
    for (size_t h = 0; h < hulls.size(); ++h)
      if (cxt::hullOf(hulls[h]) == host) return reinterpret_cast<const ep::Gu::ConvexHullData*>(dpack + hoff[h]);
    abort();
  };

  // ---- 쌍: 모양 종류 조합을 고르게 (평면-평면 빼고, 순서 뒤집힌 것 포함)
  const int types[5] = {0, 1, 2, 3, 5};
  std::vector<cxt::TestShape> sa(pairs), sb(pairs);
  std::vector<ec::ShapePair> hostPairs(pairs), devPairs(pairs);
  for (int p = 0; p < pairs; ++p) {
    int t0, t1;
    do { t0 = types[size_t(P(rng) * 5) % 5]; t1 = types[size_t(P(rng) * 5) % 5]; } while (t0 == 1 && t1 == 1);
    cxt::makeShape(t0, rng, hulls, sa[p]);
    cxt::makeShape(t1, rng, hulls, sb[p]);
    hostPairs[p].s0 = sa[p].e;
    hostPairs[p].s1 = sb[p].e;
    devPairs[p] = hostPairs[p];
    if (t0 == 5) devPairs[p].s0.convex.hullData = devHull(sa[p].e.convex.hullData);
    if (t1 == 5) devPairs[p].s1.convex.hullData = devHull(sb[p].e.convex.hullData);
  }
  ec::ShapePair* dPairs = nullptr;
  CK(cudaMalloc(&dPairs, sizeof(ec::ShapePair) * pairs));
  CK(cudaMemcpy(dPairs, devPairs.data(), sizeof(ec::ShapePair) * pairs, cudaMemcpyHostToDevice));

  const float contactDist = 0.04f, meshMargin = 0.01f, tolLen = 1.0f;
  CK(cudaDeviceSetLimit(cudaLimitStackSize, size_t(getenv("CX_STACK_KB") ? atoi(getenv("CX_STACK_KB")) : 64) * 1024));

  // ---- 자세 (판·쌍마다 따로 흔듦)
  const size_t N = size_t(envs) * pairs;
  std::vector<std::mt19937> wr(N);
  std::vector<PxTransform> tf0(N), tf1(N);
  auto randQuat = [](std::mt19937& g) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    PxQuat q(u(g), u(g), u(g), u(g)); q.normalize(); return q;
  };
  for (size_t k = 0; k < N; ++k) {
    const int p = int(k % pairs);
    wr[k].seed(uint32_t(seed * 7919 + k));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), q(0.0f, 1.0f);
    const float ra = sa[p].radius, rb = sb[p].radius;
    tf0[k] = PxTransform(PxVec3(u(wr[k]), u(wr[k]), u(wr[k])), randQuat(wr[k]));
    if (sa[p].e.type == 1) {
      const PxVec3 n = tf0[k].q.rotate(PxVec3(1, 0, 0));
      tf1[k] = PxTransform(tf0[k].p + n * (rb * (q(wr[k]) * 1.4f - 0.3f)) + tf0[k].q.rotate(PxVec3(0, u(wr[k]), u(wr[k]))), randQuat(wr[k]));
    } else if (sb[p].e.type == 1) {
      tf1[k] = tf0[k];
      tf0[k] = PxTransform(tf1[k].p + tf1[k].q.rotate(PxVec3(1, 0, 0)) * (ra * (q(wr[k]) * 1.4f - 0.3f)), randQuat(wr[k]));
    } else {
      PxVec3 d(u(wr[k]), u(wr[k]), u(wr[k])); d.normalize();
      tf1[k] = PxTransform(tf0[k].p + d * ((ra + rb) * (0.3f + 0.8f * q(wr[k]))), randQuat(wr[k]));
    }
  }

  std::vector<ec::ManifoldSlot> slotH(N), slotG(N);
  std::vector<cxt::PhysxSlot> slotP(N);
  for (size_t k = 0; k < N; ++k) {
    const int p = int(k % pairs);
    const int a = sa[p].e.type < sb[p].e.type ? sa[p].e.type : sb[p].e.type, b = sa[p].e.type < sb[p].e.type ? sb[p].e.type : sa[p].e.type;
    ec::initManifold(slotH[k], a, b);
    slotP[k].init(sa[p].e.type, sb[p].e.type);
  }
  ec::ManifoldSlot* dSlots; ec::PairPose* dPose; ec::PairResult* dOut;
  CK(cudaMalloc(&dSlots, sizeof(ec::ManifoldSlot) * N));
  CK(cudaMalloc(&dPose, sizeof(ec::PairPose) * N));
  CK(cudaMalloc(&dOut, sizeof(ec::PairResult) * N));
  cxt::gpuInitSlots(dPairs, pairs, dSlots, envs);
  CK(cudaGetLastError());
  std::vector<ec::PairPose> pose(N);
  std::vector<ec::PairResult> outG(N), outH(N);
  static PxContactBuffer bufP;
  static ep::PxContactBuffer bufE;
  cxt::Tally t21("층2=층1 결과·접촉점"), t21m("층2=층1 다양체"), t10("층1=PhysX 결과·접촉점"), t10m("층1=PhysX 다양체");
  long long contacts = 0;
  double gpuMs = 0.0;
  printf("[2] 모양 쌍 %d 개 × 판 %d × 프레임 %d\n", pairs, envs, frames);
  for (int f = 0; f < frames; ++f) {
    for (size_t k = 0; k < N; ++k) {
      pose[k].tf0 = ep::PxTransform32(cxt::toE(tf0[k]));
      pose[k].tf1 = ep::PxTransform32(cxt::toE(tf1[k]));
    }
    CK(cudaMemcpy(dPose, pose.data(), sizeof(ec::PairPose) * N, cudaMemcpyHostToDevice));
    if (getenv("CX_FIND")) {  // 진단: 쌍 하나씩 띄워서 GPU 오류를 내는 첫 쌍을 찾는다 (판 0)
      for (int p = 0; p < pairs; ++p) {
        cxt::gpuShapePairs(1, 1, dPairs + p, 1, dPose + p, dSlots + p, dOut + p, contactDist, meshMargin, tolLen, 1);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          printf("  [진단] 프레임 %d 쌍 %d (종류 %d-%d): %s\n", f, p, sa[p].e.type, sb[p].e.type, cudaGetErrorString(e));
          return 2;
        }
      }
    }
    gpuMs += cxt::gpuShapePairs(envs, 64, dPairs, pairs, dPose, dSlots, dOut, contactDist, meshMargin, tolLen, 1);
    CK(cudaGetLastError());
    CK(cudaMemcpy(outG.data(), dOut, sizeof(ec::PairResult) * N, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(slotG.data(), dSlots, sizeof(ec::ManifoldSlot) * N, cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < N; ++k) {
      const int p = int(k % pairs);
      bool rp;
      {
        cxt::FtzScope fz;
        ec::shapePair(hostPairs[p], pose[k], contactDist, meshMargin, tolLen, slotH[k], bufE, outH[k]);
        rp = cxt::physxPcmPair(sa[p], sb[p], tf0[k], tf1[k], contactDist, slotP[k], bufP);
      }
      const long long c = (long long)k;
      t21.u(outG[k].ret, outH[k].ret, c, f);
      t21.u(outG[k].count, outH[k].count, c, f);
      t21.u(outG[k].overflow + outH[k].overflow, 0, c, f);
      const uint32_t n21 = std::min(std::min(outG[k].count, outH[k].count), uint32_t(ec::kMaxOutContacts));
      cmpContacts(t21, outG[k].c, outH[k].c, n21, c, f);
      cmpSlot(t21m, slotG[k], slotH[k], c, f);
      t10.u(rp, outH[k].ret, c, f);
      t10.u(bufP.count, outH[k].count, c, f);
      const uint32_t n10 = std::min(std::min(bufP.count, outH[k].count), uint32_t(ec::kMaxOutContacts));
      cmpContacts(t10, bufP.contacts, outH[k].c, n10, c, f);
      cmpPhysxSlot(t10m, slotP[k], slotH[k], c, f);
      contacts += bufP.count;
      // 다음 자세
      std::uniform_real_distribution<float> u(-1.0f, 1.0f), q(0.0f, 1.0f);
      const float step = 0.01f * (sa[p].radius + sb[p].radius + 0.1f);
      const float s = q(wr[k]) < 0.05f ? 10.0f : 1.0f;
      tf1[k].p += PxVec3(u(wr[k]), u(wr[k]), u(wr[k])) * (step * s);
      PxVec3 ax(u(wr[k]), u(wr[k]), u(wr[k]));
      ax = ax.isZero() ? PxVec3(0, 0, 1) : ax.getNormalized();
      tf1[k].q = (PxQuat(0.02f * s * u(wr[k]), ax) * tf1[k].q).getNormalized();
      const float lim = 1.3f * (sa[p].radius + sb[p].radius) + 0.1f;
      if (sa[p].e.type != 1 && sb[p].e.type != 1 && (tf1[k].p - tf0[k].p).magnitude() > lim) tf1[k].p = tf0[k].p + (tf1[k].p - tf0[k].p) * 0.5f;
    }
  }
  printf("  쌍·프레임 %zu, 접촉점 합 %lld, GPU 커널 합 %.1f ms\n", N * frames, contacts, gpuMs);
  for (const cxt::Tally* t : {&t21, &t21m, &t10, &t10m}) { t->print(); badAll += t->bad; }

  // ---- 처리량
  if (benchEnvs > 0) {
    printf("[3] 처리량 (판 %d × 쌍 %d, 모양 섞임)\n", benchEnvs, pairs);
    const size_t NB = size_t(benchEnvs) * pairs;
    std::vector<ec::PairPose> bp(NB);
    for (size_t k = 0; k < NB; ++k) bp[k] = pose[k % N];
    ec::PairPose* dbp; ec::ManifoldSlot* dbs; ec::PairResult* dbo;
    CK(cudaMalloc(&dbp, sizeof(ec::PairPose) * NB));
    CK(cudaMalloc(&dbs, sizeof(ec::ManifoldSlot) * NB));
    CK(cudaMalloc(&dbo, sizeof(ec::PairResult) * NB));
    CK(cudaMemcpy(dbp, bp.data(), sizeof(ec::PairPose) * NB, cudaMemcpyHostToDevice));
    for (int threads : {32, 64, 128}) {
      cxt::gpuInitSlots(dPairs, pairs, dbs, benchEnvs);
      const float ms0 = cxt::gpuShapePairs(benchEnvs, threads, dPairs, pairs, dbp, dbs, dbo, contactDist, meshMargin, tolLen, 1);
      const float ms = cxt::gpuShapePairs(benchEnvs, threads, dPairs, pairs, dbp, dbs, dbo, contactDist, meshMargin, tolLen, 5);
      CK(cudaGetLastError());
      printf("  블록당 스레드 %3d: 첫 프레임(다양체 비어 전부 생성) %.2f ms, 다음 프레임 %.2f ms -> %.3g 쌍/초\n", threads, ms0, ms, NB / (ms * 1e-3));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t calls = 0;
    for (int r = 0; r < 3; ++r)
      for (size_t k = 0; k < N; ++k) {
        cxt::FtzScope fz;
        const int p = int(k % pairs);
        cxt::physxPcmPair(sa[p], sb[p], tf0[k], tf1[k], contactDist, slotP[k], bufP);
        ++calls;
      }
    const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("  대조 PhysX CPU 1 스레드: %.3g 쌍/초\n", calls / s);
  }
  printf(badAll ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return badAll ? 1 : 0;
}
