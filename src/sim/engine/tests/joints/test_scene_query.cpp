// 층 1 시험: 광선 모양별 함수(상자·구·캡슐·평면·볼록 메시) = PhysX 5.6.1 PxGeometryQuery::raycast (같은 Gu::raycast_* 를 부름), 비트 비교.
// 비교: 맞음 여부, 거리·위치·법선·면 번호·u·v·플래그. PxGeometryQuery 는 기본 eSIMD_GUARD(FTZ/DAZ) 라 우리도 FTZ 안에서 부른다.
//   test_scene_query [--rays N] [--seed S]
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
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { const float x = s * U(rng), y = s * U(rng), z = s * U(rng); return PxVec3(x, y, z); };

  // 볼록 메시 몇 개 (무작위 점 구름을 PhysX 가 굽는다 — 굽기는 우리 범위 밖, 결과 다각형 평면만 쓴다)
  std::vector<PxConvexMesh*> meshes;
  std::vector<std::vector<float>> planes;
  for (int m = 0; m < 24; ++m) {
    std::vector<PxVec3> pts(8 + rng() % 40);
    const PxVec3 ext(0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng));
    for (auto& p : pts) p = PxVec3(ext.x * U(rng), ext.y * U(rng), ext.z * U(rng));
    PxConvexMeshDesc desc;
    desc.points.count = PxU32(pts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pts.data();
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMesh* cm = PxCreateConvexMesh(cp, desc, phys->getPhysicsInsertionCallback());
    if (!cm) continue;
    meshes.push_back(cm);
    std::vector<float> pl;
    for (PxU32 i = 0; i < cm->getNbPolygons(); ++i) {
      PxHullPolygon hp;
      cm->getPolygonData(i, hp);
      for (int k = 0; k < 4; ++k) pl.push_back(hp.mPlane[k]);
    }
    planes.push_back(pl);
  }

  const char* kinds[5] = {"상자", "구", "캡슐", "평면", "볼록"};
  uint64_t cmp[5] = {0}, hitsN[5] = {0}, bad[5] = {0};
  int shown = 0;
  for (int r = 0; r < rays; ++r) {
    const int kind = int(rng() % 5u);
    const PxTransform pose(rv(1.0f), rq());
    PxVec3 origin = pose.p + rv(2.0f);
    PxVec3 dir;
    const int aim = int(rng() % 4u);
    if (aim == 0) dir = rv(1.0f);                                  // 아무 방향
    else if (aim == 3) { origin = pose.p + rv(0.05f); dir = rv(1.0f); }  // 모양 안에서 시작
    else dir = (pose.p + rv(0.3f)) - origin;                       // 모양 쪽으로
    dir.normalize();                                               // omni 도 normalize (PhysXSceneQuery.cpp:344)
    const float maxDist = (rng() % 5u == 0) ? 0.5f * P(rng) : 10.0f * P(rng) + 0.01f;
    PxGeomRaycastHit ph;
    memset(&ph, 0, sizeof ph);
    PxU32 pn = 0;
    S::RayHit eh;
    memset(&eh, 0, sizeof eh);
    uint32_t en = 0;
    const eng::V3 eo = toE(origin), ed = toE(dir);
    const eng::Tf ep = toE(pose);
    switch (kind) {
      case 0: {
        const PxVec3 he(0.02f + 0.6f * P(rng), 0.02f + 0.6f * P(rng), 0.02f + 0.6f * P(rng));
        pn = PxGeometryQuery::raycast(origin, dir, PxBoxGeometry(he), pose, maxDist, PxHitFlag::eDEFAULT, 1, &ph);
        FtzScope f;
        en = S::raycastBox(toE(he), ep, eo, ed, maxDist, S::HF_DEFAULT, eh);
        break;
      }
      case 1: {
        const float rad = 0.02f + 0.6f * P(rng);
        pn = PxGeometryQuery::raycast(origin, dir, PxSphereGeometry(rad), pose, maxDist, PxHitFlag::eDEFAULT, 1, &ph);
        FtzScope f;
        en = S::raycastSphere(rad, ep, eo, ed, maxDist, S::HF_DEFAULT, eh);
        break;
      }
      case 2: {
        const float rad = 0.02f + 0.4f * P(rng), hh = 0.6f * P(rng);
        pn = PxGeometryQuery::raycast(origin, dir, PxCapsuleGeometry(rad, hh), pose, maxDist, PxHitFlag::eDEFAULT, 1, &ph);
        FtzScope f;
        en = S::raycastCapsule(rad, hh, ep, eo, ed, maxDist, S::HF_DEFAULT, eh);
        break;
      }
      case 3: {
        pn = PxGeometryQuery::raycast(origin, dir, PxPlaneGeometry(), pose, maxDist, PxHitFlag::eDEFAULT, 1, &ph);
        FtzScope f;
        en = S::raycastPlane(ep, eo, ed, maxDist, eh);
        break;
      }
      case 4: {
        const int m = int(rng() % meshes.size());
        PxMeshScale ms(PxVec3(1.0f));
        const int st = int(rng() % 3u);
        if (st == 1) ms = PxMeshScale(0.3f + 2.0f * P(rng));
        else if (st == 2) ms = PxMeshScale(PxVec3(0.3f + 2.0f * P(rng), 0.3f + 2.0f * P(rng), 0.3f + 2.0f * P(rng)), rq());
        pn = PxGeometryQuery::raycast(origin, dir, PxConvexMeshGeometry(meshes[m], ms), pose, maxDist, PxHitFlag::eDEFAULT, 1, &ph);
        FtzScope f;
        const S::MeshScale es{toE(ms.scale), eng::Q{ms.rotation.x, ms.rotation.y, ms.rotation.z, ms.rotation.w}};
        en = S::raycastConvex(planes[m].data(), uint32_t(planes[m].size() / 4), es, ep, eo, ed, maxDist, S::HF_DEFAULT, eh);
        break;
      }
    }
    cmp[kind]++;
    bool ok = pn == en;
    if (ok && pn) {
      hitsN[kind]++;
      const uint32_t pf = uint32_t(PxU16(ph.flags));
      ok = !memcmp(&ph.distance, &eh.distance, 4) && !memcmp(&ph.position, &eh.position, 12) && !memcmp(&ph.normal, &eh.normal, 12) &&
           ph.faceIndex == eh.faceIndex && !memcmp(&ph.u, &eh.u, 4) && !memcmp(&ph.v, &eh.v, 4) && pf == eh.flags;
    }
    if (!ok) {
      bad[kind]++;
      if (shown < 10) {
        shown++;
        printf("[다름] 광선 %d %s: 맞음 %u/%u  거리 %.9g/%.9g  위치 (%.9g %.9g %.9g)/(%.9g %.9g %.9g)  법선 (%.9g %.9g %.9g)/(%.9g %.9g %.9g)  면 %u/%u 플래그 %x/%x\n",
               r, kinds[kind], pn, en, ph.distance, eh.distance, ph.position.x, ph.position.y, ph.position.z, eh.position.x, eh.position.y,
               eh.position.z, ph.normal.x, ph.normal.y, ph.normal.z, eh.normal.x, eh.normal.y, eh.normal.z, ph.faceIndex, eh.faceIndex,
               uint32_t(PxU16(ph.flags)), eh.flags);
      }
    }
  }
  printf("\n광선 모양별 비교 (광선 %d, 씨앗 %d, 볼록 메시 %zu 개)\n", rays, seed, meshes.size());
  bool ok = true;
  for (int k = 0; k < 5; ++k) {
    printf("  %-6s 광선 %8" PRIu64 "  맞음 %8" PRIu64 "  비트 다름 %" PRIu64 "\n", kinds[k], cmp[k], hitsN[k], bad[k]);
    ok &= bad[k] == 0;
  }
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  for (auto* m : meshes) m->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
