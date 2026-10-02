// 층 2 시험: 장면 광선 raycastClosest 의 CUDA 결과(cuda/joints/sq_kernels.cuh) = 층 1(같은 코드를 호스트에서, FTZ), 비트 비교 + 처리량.
// 층 1 = PhysX 는 test_scene_raycast·test_raycast_mesh·test_scene_query 가 증명한다. 장면은 PhysX 없이 만든다(평면·상자·구·캡슐·볼록 평면 묶음·삼각 격자).
// 판 E 개 = 같은 정적 모양 + 판마다 다른 동적 자세. 광선은 판마다 R 개.
//   test_scene_raycast_gpu [--envs E] [--rays R] [--seed S]
#include <cuda_runtime.h>
#include <xmmintrin.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/joints/sq_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using namespace eng;
namespace S = eng::sq;
using eng::jcuda::SqRayIn;
using eng::jcuda::SqRayOut;

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old); }
};

int main(int argc, char** argv) {
  int envs = 4096, raysPer = 64, seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rays") && i + 1 < argc) raysPer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
  }
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    const float a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const float n = std::sqrt(a * a + b * b + c * c + d * d);
    return Q{a / n, b / n, c / n, d / n};
  };
  auto rv = [&](float s) { const float x = s * U(rng), y = s * U(rng), z = s * U(rng); return V3{x, y, z}; };

  // 볼록 평면 묶음 4 개 (상자 6 면 + 모서리 깎는 면 몇 개)
  std::vector<std::vector<float>> cpl(4);
  for (auto& pl : cpl) {
    const V3 e{0.1f + 0.4f * P(rng), 0.1f + 0.4f * P(rng), 0.1f + 0.4f * P(rng)};
    const float f[6][4] = {{1, 0, 0, -e.x}, {-1, 0, 0, -e.x}, {0, 1, 0, -e.y}, {0, -1, 0, -e.y}, {0, 0, 1, -e.z}, {0, 0, -1, -e.z}};
    for (auto& r : f) pl.insert(pl.end(), r, r + 4);
    const int extra = int(rng() % 5u);
    for (int k = 0; k < extra; ++k) {
      V3 n = rv(1.0f);
      const float m = std::sqrt(magSq(n));
      n = V3{n.x / m, n.y / m, n.z / m};
      pl.insert(pl.end(), {n.x, n.y, n.z, -0.8f * (std::fabs(n.x) * e.x + std::fabs(n.y) * e.y + std::fabs(n.z) * e.z)});
    }
  }
  // 삼각 격자 1 개
  const int gn = 12;
  std::vector<V3> gv;
  std::vector<uint32_t> gt;
  for (int y = 0; y <= gn; ++y)
    for (int x = 0; x <= gn; ++x) gv.push_back(V3{-2.0f + 4.0f * x / gn, -2.0f + 4.0f * y / gn, 0.1f * U(rng)});
  for (int y = 0; y < gn; ++y)
    for (int x = 0; x < gn; ++x) {
      const uint32_t a = uint32_t(y * (gn + 1) + x), b = a + 1, c = a + uint32_t(gn + 1), d = c + 1;
      gt.insert(gt.end(), {a, b, d, a, d, c});
    }
  S::TriMeshData tm{nullptr, nullptr, uint32_t(gt.size() / 3), V3{0, 0, 0}, 2.83f, 0.0f};

  // 판 하나의 모양 틀: 0 평면, 1 삼각, 2..4 정적 상자, 나머지 동적 (자세는 판마다)
  const int nStatic = 5, nDyn = 24, per = nStatic + nDyn;
  std::vector<S::SqShape> tmpl(per);
  for (int i = 0; i < per; ++i) {
    S::SqShape& s = tmpl[i];
    s = S::SqShape{};
    s.scale = S::MeshScale{V3{1, 1, 1}, Q{0, 0, 0, 1}};
    if (i == 0) { s.type = S::SQ_PLANE; s.pose = Tf{Q{0.0f, -0.70710677f, 0.0f, 0.70710677f}, V3{0, 0, -1.5f}}; continue; }
    if (i == 1) {
      s.type = S::SQ_TRIMESH; s.doubleSided = 1; s.pose = Tf{rq(), rv(1.0f)};
      s.scale = S::MeshScale{V3{0.8f + P(rng), 0.8f + P(rng), 1.0f}, rq()};
      continue;
    }
    const int k = i < nStatic ? 3 : int(rng() % 4u);
    if (k == 3 || k == 0) { s.type = S::SQ_BOX; s.halfExtents = V3{0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng)}; }
    else if (k == 1) { s.type = S::SQ_SPHERE; s.radius = 0.05f + 0.4f * P(rng); }
    else if (k == 2) { s.type = S::SQ_CAPSULE; s.radius = 0.05f + 0.3f * P(rng); s.halfHeight = 0.4f * P(rng); }
    if (i >= nStatic && rng() % 3u == 0) {
      s.type = S::SQ_CONVEX;
      s.nPolys = uint32_t(rng() % 4u);  // 임시로 묶음 번호
      if (rng() % 2u) s.scale = S::MeshScale{V3{0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)}, rq()};
    }
    s.pose = Tf{rq(), rv(3.0f)};
  }
  // 판마다 동적 자세를 바꿔서 전체 배열
  const size_t nAll = size_t(envs) * per;
  std::vector<S::SqShape> hs(nAll);
  std::vector<int> convexIdx(nAll, -1);
  for (int e = 0; e < envs; ++e)
    for (int i = 0; i < per; ++i) {
      S::SqShape s = tmpl[i];
      if (i >= nStatic) s.pose = Tf{rq(), rv(3.0f)};
      if (s.type == S::SQ_CONVEX) convexIdx[size_t(e) * per + i] = int(tmpl[i].nPolys);
      s.id = uint32_t(i);
      hs[size_t(e) * per + i] = s;
    }
  // 광선
  const size_t nR = size_t(envs) * raysPer;
  std::vector<SqRayIn> hr(nR);
  for (size_t r = 0; r < nR; ++r) {
    const uint32_t e = uint32_t(r / raysPer);
    const V3 o = rv(4.0f);
    const V3 tp = hs[size_t(e) * per + rng() % per].pose.p;
    V3 d = (rng() % 3u == 0) ? rv(1.0f) : (tp + rv(0.3f)) - o;
    const float m = std::sqrt(magSq(d));
    d = V3{d.x / m, d.y / m, d.z / m};
    hr[r] = SqRayIn{o, d, 1.0f + 10.0f * P(rng), e};
  }

  // 장치 메모리: 볼록 평면·삼각 자료를 올리고 포인터를 장치 주소로 바꾼 사본
  std::vector<float*> dpl(cpl.size());
  for (size_t k = 0; k < cpl.size(); ++k) {
    CK(cudaMalloc(&dpl[k], cpl[k].size() * 4));
    CK(cudaMemcpy(dpl[k], cpl[k].data(), cpl[k].size() * 4, cudaMemcpyHostToDevice));
  }
  V3* dgv; uint32_t* dgt; S::TriMeshData* dtm;
  CK(cudaMalloc(&dgv, gv.size() * sizeof(V3)));
  CK(cudaMalloc(&dgt, gt.size() * 4));
  CK(cudaMemcpy(dgv, gv.data(), gv.size() * sizeof(V3), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dgt, gt.data(), gt.size() * 4, cudaMemcpyHostToDevice));
  S::TriMeshData tmD = tm;
  tmD.verts = dgv; tmD.tris = dgt;
  CK(cudaMalloc(&dtm, sizeof tmD));
  CK(cudaMemcpy(dtm, &tmD, sizeof tmD, cudaMemcpyHostToDevice));
  tm.verts = gv.data(); tm.tris = gt.data();
  std::vector<S::SqShape> ds(hs);
  for (size_t i = 0; i < nAll; ++i) {
    if (convexIdx[i] >= 0) {
      const int k = convexIdx[i];
      hs[i].planes = cpl[k].data(); hs[i].nPolys = uint32_t(cpl[k].size() / 4);
      ds[i].planes = dpl[k]; ds[i].nPolys = hs[i].nPolys;
    }
    if (hs[i].type == S::SQ_TRIMESH) { hs[i].mesh = &tm; ds[i].mesh = dtm; }
  }
  S::SqShape* dS; SqRayIn* dR; SqRayOut* dO;
  CK(cudaMalloc(&dS, nAll * sizeof(S::SqShape)));
  CK(cudaMalloc(&dR, nR * sizeof(SqRayIn)));
  CK(cudaMalloc(&dO, nR * sizeof(SqRayOut)));
  CK(cudaMemcpy(dS, ds.data(), nAll * sizeof(S::SqShape), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dR, hr.data(), nR * sizeof(SqRayIn), cudaMemcpyHostToDevice));

  float bestMs = 1e30f;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int rep = 0; rep < 5; ++rep) {
    cudaEventRecord(e0);
    eng::jcuda::kRaycastClosest<<<uint32_t((nR + 127) / 128), 128>>>(dS, uint32_t(per), dR, uint32_t(nR), S::HF_DEFAULT, dO);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    if (ms < bestMs) bestMs = ms;
  }
  std::vector<SqRayOut> go(nR), co(nR);
  CK(cudaMemcpy(go.data(), dO, nR * sizeof(SqRayOut), cudaMemcpyDeviceToHost));
  const auto t0 = std::chrono::steady_clock::now();
  {
    FtzScope f;
    for (size_t r = 0; r < nR; ++r) {
      co[r].hit = S::RayHit{};
      co[r].shape = S::raycastClosest(hs.data() + size_t(hr[r].env) * per, uint32_t(per), hr[r].origin, hr[r].dir, hr[r].maxDist, S::HF_DEFAULT, co[r].hit);
    }
  }
  const double cpuS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  uint64_t hits = 0, bad = 0;
  int64_t first = -1;
  for (size_t r = 0; r < nR; ++r) {
    bool ok = go[r].shape == co[r].shape;
    if (ok && co[r].shape >= 0) {
      hits++;
      const S::RayHit &a = go[r].hit, &b = co[r].hit;
      ok = !memcmp(&a.distance, &b.distance, 4) && !memcmp(&a.position, &b.position, 12) && !memcmp(&a.normal, &b.normal, 12) &&
           a.faceIndex == b.faceIndex && !memcmp(&a.u, &b.u, 4) && !memcmp(&a.v, &b.v, 4) && a.flags == b.flags;
    }
    if (!ok) { if (first < 0) first = int64_t(r); bad++; }
  }
  printf("\n장면 광선 층 1(C++) vs 층 2(CUDA): 판 %d 개 × 광선 %d = %zu, 판마다 모양 %d 개 (씨앗 %d)\n", envs, raysPer, nR, per, seed);
  printf("  맞음 %" PRIu64 "  비트 다름 %" PRIu64, hits, bad);
  if (first >= 0) printf("  첫 다름 광선 %" PRId64, first);
  printf("\n  처리량: GPU %.3f ms (광선 %.3g/초), CPU 단일 스레드 %.3f s (광선 %.3g/초) -> %.0f 배\n", bestMs, nR / (bestMs * 1e-3), cpuS, nR / cpuS,
         cpuS / (bestMs * 1e-3));
  printf("%s\n", bad == 0 ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  return bad == 0 ? 0 : 3;
}
