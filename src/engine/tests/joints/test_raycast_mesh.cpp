// 층 1 시험: 삼각 메시 광선(core/joints/scene_query.h raycastTriangleMesh) = PhysX 5.6.1 PxGeometryQuery::raycast(PxTriangleMeshGeometry), 비트 비교.
// PhysX 가 구운 메시(BVH34 = BV4)의 정점·내부 삼각형·나무 LocalBounds·geomEpsilon 을 그대로 입력으로 쓴다(굽기는 범위 밖).
// 메시: 무작위 삼각형 수프, 울퉁불퉁한 격자(바닥 같은 것), 상자 겉면. 척도: 항등·균일·비균일+회전·음수. 양면 여부·항등 자세도 섞는다.
//   test_raycast_mesh [--rays N] [--seed S]
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"

#define private public
#define protected public
#include "GuTriangleMeshBV4.h"
#undef private
#undef protected

#include "core/joints/scene_query.h"

using namespace physx;
namespace S = eng::sq;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;
struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};
static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }
static eng::V3 toE(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }

struct MeshRec {
  PxTriangleMesh* px;
  std::vector<eng::V3> v;
  std::vector<uint32_t> t;
  S::TriMeshData d;
};

int main(int argc, char** argv) {
  int rays = 200000, seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rays") && i + 1 < argc) rays = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxCookingParams cp(tol);
  cp.midphaseDesc = PxMeshMidPhase::eBVH34;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { const float x = s * U(rng), y = s * U(rng), z = s * U(rng); return PxVec3(x, y, z); };

  std::vector<MeshRec> meshes;
  for (int m = 0; m < 18; ++m) {
    std::vector<PxVec3> pts;
    std::vector<PxU32> idx;
    const int kind = m % 3;
    if (kind == 0) {  // 삼각형 수프
      const int nt = 20 + int(rng() % 200u);
      for (int i = 0; i < nt; ++i) {
        const PxVec3 c = rv(0.8f);
        for (int k = 0; k < 3; ++k) { idx.push_back(PxU32(pts.size())); pts.push_back(c + rv(0.2f)); }
      }
    } else if (kind == 1) {  // 격자 (바닥·지형)
      const int n = 4 + int(rng() % 20u);
      const float h = 0.05f * P(rng);
      for (int y = 0; y <= n; ++y)
        for (int x = 0; x <= n; ++x) pts.push_back(PxVec3(-1.0f + 2.0f * x / n, -1.0f + 2.0f * y / n, (rng() % 3u == 0) ? 0.0f : h * U(rng)));
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
          const PxU32 a = PxU32(y * (n + 1) + x), b = a + 1, c = a + PxU32(n + 1), d = c + 1;
          idx.insert(idx.end(), {a, b, d, a, d, c});
        }
    } else {  // 상자 겉면 (바깥을 보는 삼각형)
      const PxVec3 e(0.1f + 0.5f * P(rng), 0.1f + 0.5f * P(rng), 0.1f + 0.5f * P(rng));
      for (int i = 0; i < 8; ++i) pts.push_back(PxVec3((i & 1) ? e.x : -e.x, (i & 2) ? e.y : -e.y, (i & 4) ? e.z : -e.z));
      const PxU32 f[36] = {0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5};
      idx.assign(f, f + 36);
    }
    PxTriangleMeshDesc desc;
    desc.points.count = PxU32(pts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pts.data();
    desc.triangles.count = PxU32(idx.size() / 3);
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = idx.data();
    PxTriangleMesh* tm = PxCreateTriangleMesh(cp, desc, phys->getPhysicsInsertionCallback());
    if (!tm) continue;
    MeshRec r;
    r.px = tm;
    const PxVec3* vv = tm->getVertices();
    for (PxU32 i = 0; i < tm->getNbVertices(); ++i) r.v.push_back(toE(vv[i]));
    const bool b16 = tm->getTriangleMeshFlags() & PxTriangleMeshFlag::e16_BIT_INDICES;
    for (PxU32 i = 0; i < 3 * tm->getNbTriangles(); ++i)
      r.t.push_back(b16 ? uint32_t(static_cast<const PxU16*>(tm->getTriangles())[i]) : uint32_t(static_cast<const PxU32*>(tm->getTriangles())[i]));
    Gu::BV4TriangleMesh* bm = static_cast<Gu::BV4TriangleMesh*>(tm);
    const Gu::LocalBounds& lb = bm->getBV4Tree().mLocalBounds;
    r.d = S::TriMeshData{nullptr, nullptr, tm->getNbTriangles(), toE(lb.mCenter), lb.mExtentsMagnitude, bm->getGeomEpsilon()};
    meshes.push_back(std::move(r));
  }
  for (auto& r : meshes) { r.d.verts = r.v.data(); r.d.tris = r.t.data(); }

  const char* scaleNames[4] = {"항등", "균일", "비균일+회전", "음수"};
  uint64_t cmp[4] = {0}, hitsN[4] = {0}, bad[4] = {0};
  int shown = 0;
  for (int r = 0; r < rays; ++r) {
    MeshRec& M = meshes[rng() % meshes.size()];
    const int st = int(rng() % 4u);
    PxMeshScale ms(PxVec3(1.0f));
    if (st == 1) ms = PxMeshScale(0.3f + 2.0f * P(rng));
    else if (st == 2) ms = PxMeshScale(PxVec3(0.3f + 2.0f * P(rng), 0.3f + 2.0f * P(rng), 0.3f + 2.0f * P(rng)), rq());
    else if (st == 3) ms = PxMeshScale(PxVec3(-(0.3f + 2.0f * P(rng)), 0.3f + 2.0f * P(rng), 0.3f + 2.0f * P(rng)), rq());
    const PxTransform pose = (rng() % 8u == 0) ? PxTransform(PxIdentity) : ((rng() % 8u == 0) ? PxTransform(rv(1.0f)) : PxTransform(rv(1.0f), rq()));
    PxVec3 origin = pose.p + rv(2.0f);
    PxVec3 dir = (rng() % 4u == 0) ? rv(1.0f) : (pose.p + rv(0.6f)) - origin;
    dir.normalize();
    const float maxDist = (rng() % 5u == 0) ? 0.5f * P(rng) : 10.0f * P(rng) + 0.01f;
    const bool dbl = rng() % 3u == 0;
    PxHitFlags hf = PxHitFlag::eDEFAULT;
    uint32_t ehf = S::HF_DEFAULT;
    if (rng() % 4u == 0) { hf |= PxHitFlag::eMESH_BOTH_SIDES; ehf |= S::HF_MESH_BOTH_SIDES; }
    if (rng() % 6u == 0) { hf &= ~PxHitFlags(PxHitFlag::eNORMAL); ehf &= ~S::HF_NORMAL; }
    PxTriangleMeshGeometry g(M.px, ms, dbl ? PxMeshGeometryFlags(PxMeshGeometryFlag::eDOUBLE_SIDED) : PxMeshGeometryFlags());
    PxGeomRaycastHit ph;
    memset(&ph, 0, sizeof ph);
    const PxU32 pn = PxGeometryQuery::raycast(origin, dir, g, pose, maxDist, hf, 1, &ph);
    S::RayHit eh;
    memset(&eh, 0, sizeof eh);
    uint32_t en;
    {
      FtzScope f;
      const S::MeshScale es{toE(ms.scale), eng::Q{ms.rotation.x, ms.rotation.y, ms.rotation.z, ms.rotation.w}};
      en = S::raycastTriangleMesh(M.d, es, dbl, toE(pose), toE(origin), toE(dir), maxDist, ehf, eh);
    }
    cmp[st]++;
    bool ok = pn == en;
    if (ok && pn) {
      hitsN[st]++;
      const uint32_t pf = uint32_t(PxU16(ph.flags));
      ok = !memcmp(&ph.distance, &eh.distance, 4) && !memcmp(&ph.position, &eh.position, 12) && !memcmp(&ph.normal, &eh.normal, 12) &&
           ph.faceIndex == eh.faceIndex && !memcmp(&ph.u, &eh.u, 4) && !memcmp(&ph.v, &eh.v, 4) && pf == eh.flags;
    }
    if (!ok) {
      bad[st]++;
      if (shown < 10) {
        shown++;
        printf("[다름] 광선 %d 척도 %s 양면 %d: 맞음 %u/%u 거리 %.9g/%.9g 위치 (%.9g %.9g %.9g)/(%.9g %.9g %.9g) 법선 (%.9g %.9g %.9g)/(%.9g %.9g %.9g) 면 %u/%u uv %.9g,%.9g/%.9g,%.9g 플래그 %x/%x\n",
               r, scaleNames[st], int(dbl), pn, en, ph.distance, eh.distance, ph.position.x, ph.position.y, ph.position.z, eh.position.x,
               eh.position.y, eh.position.z, ph.normal.x, ph.normal.y, ph.normal.z, eh.normal.x, eh.normal.y, eh.normal.z, ph.faceIndex, eh.faceIndex,
               ph.u, ph.v, eh.u, eh.v, uint32_t(PxU16(ph.flags)), eh.flags);
      }
    }
  }
  printf("\n삼각 메시 광선 비교 (광선 %d, 씨앗 %d, 메시 %zu 개)\n", rays, seed, meshes.size());
  bool ok = true;
  for (int k = 0; k < 4; ++k) {
    printf("  척도 %-12s 광선 %8" PRIu64 "  맞음 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", scaleNames[k], cmp[k], hitsN[k], bad[k]);
    ok &= bad[k] == 0;
  }
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  for (auto& m : meshes) m.px->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
