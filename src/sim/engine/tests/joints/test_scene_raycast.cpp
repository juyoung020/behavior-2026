// 층 1 시험: 장면 단위 가장 가까운 광선(core/joints/scene_query.h raycastClosest) = PhysX 5.6.1 PxScene::raycast (막는 맞음 하나,
// 기본 플래그 ePOSITION|eNORMAL|eFACE_INDEX, 거르개 없음 — omni raycast_closest 와 같은 호출).
// 장면: 정적(평면·상자·삼각 메시)과 동적(상자·구·캡슐·볼록) 모양을 흩어 놓는다. 모양 지역 자세는 항등(장면 질의 자세 = 행위자 자세).
// 방문 순서가 결과를 바꾸는 광선(맞음 거리가 같거나 볼록 1e-5 안): 우리 함수를 정순·역순으로 돌려 다르면 "순서 민감"으로 따로 센다.
//   test_scene_raycast [--rays N] [--seed S] [--shapes K]
#include <xmmintrin.h>

#include <algorithm>
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

int main(int argc, char** argv) {
  int rays = 100000, seed = 1, nShapes = 40;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rays") && i + 1 < argc) rays = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) nShapes = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxCookingParams cp(tol);
  cp.midphaseDesc = PxMeshMidPhase::eBVH34;
  PxSceneDesc sd(tol);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(0);
  sd.filterShader = PxDefaultSimulationFilterShader;
  PxScene* scene = phys->createScene(sd);
  PxMaterial* mat = phys->createMaterial(0.5f, 0.5f, 0.1f);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double n = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / n), float(b / n), float(c / n), float(d / n));
  };
  auto rv = [&](float s) { const float x = s * U(rng), y = s * U(rng), z = s * U(rng); return PxVec3(x, y, z); };

  // 볼록 메시·삼각 메시 몇 개
  std::vector<PxConvexMesh*> cms;
  std::vector<std::vector<float>> cplanes;
  for (int m = 0; m < 6; ++m) {
    std::vector<PxVec3> pts(8 + rng() % 30);
    const PxVec3 ext(0.1f + 0.4f * P(rng), 0.1f + 0.4f * P(rng), 0.1f + 0.4f * P(rng));
    for (auto& p : pts) p = PxVec3(ext.x * U(rng), ext.y * U(rng), ext.z * U(rng));
    PxConvexMeshDesc desc;
    desc.points.count = PxU32(pts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pts.data();
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMesh* cm = PxCreateConvexMesh(cp, desc, phys->getPhysicsInsertionCallback());
    if (!cm) continue;
    std::vector<float> pl;
    for (PxU32 i = 0; i < cm->getNbPolygons(); ++i) {
      PxHullPolygon hp;
      cm->getPolygonData(i, hp);
      for (int k = 0; k < 4; ++k) pl.push_back(hp.mPlane[k]);
    }
    cms.push_back(cm);
    cplanes.push_back(pl);
  }
  struct TM { PxTriangleMesh* px; std::vector<eng::V3> v; std::vector<uint32_t> t; S::TriMeshData d; };
  std::vector<TM> tms(3);
  for (int m = 0; m < 3; ++m) {
    const int n = 6 + m * 5;
    std::vector<PxVec3> pts;
    std::vector<PxU32> idx;
    for (int y = 0; y <= n; ++y)
      for (int x = 0; x <= n; ++x) pts.push_back(PxVec3(-2.0f + 4.0f * x / n, -2.0f + 4.0f * y / n, 0.1f * U(rng)));
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const PxU32 a = PxU32(y * (n + 1) + x), b = a + 1, c = a + PxU32(n + 1), d = c + 1;
        idx.insert(idx.end(), {a, b, d, a, d, c});
      }
    PxTriangleMeshDesc desc;
    desc.points.count = PxU32(pts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pts.data();
    desc.triangles.count = PxU32(idx.size() / 3);
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = idx.data();
    TM& r = tms[m];
    r.px = PxCreateTriangleMesh(cp, desc, phys->getPhysicsInsertionCallback());
    const PxVec3* vv = r.px->getVertices();
    for (PxU32 i = 0; i < r.px->getNbVertices(); ++i) r.v.push_back(toE(vv[i]));
    const bool b16 = r.px->getTriangleMeshFlags() & PxTriangleMeshFlag::e16_BIT_INDICES;
    for (PxU32 i = 0; i < 3 * r.px->getNbTriangles(); ++i)
      r.t.push_back(b16 ? uint32_t(static_cast<const PxU16*>(r.px->getTriangles())[i]) : uint32_t(static_cast<const PxU32*>(r.px->getTriangles())[i]));
    Gu::BV4TriangleMesh* bm = static_cast<Gu::BV4TriangleMesh*>(r.px);
    const Gu::LocalBounds& lb = bm->getBV4Tree().mLocalBounds;
    r.d = S::TriMeshData{r.v.data(), r.t.data(), r.px->getNbTriangles(), toE(lb.mCenter), lb.mExtentsMagnitude, bm->getGeomEpsilon()};
  }

  // 장면 모양: 정적 먼저(PhysX 도 정적 나무를 먼저 본다), 동적 다음
  std::vector<S::SqShape> stat, dyn;
  std::vector<PxShape*> pxStat, pxDyn;
  auto addShape = [&](bool isStatic, const PxGeometry& g, S::SqShape s) {
    const PxTransform pose = isStatic && s.type == S::SQ_PLANE ? PxTransform(PxVec3(0, 0, -1.5f), PxQuat(-PxHalfPi, PxVec3(0, 1, 0)))
                                                                 : PxTransform(rv(3.0f), rq());
    PxRigidActor* a = isStatic ? static_cast<PxRigidActor*>(phys->createRigidStatic(pose)) : static_cast<PxRigidActor*>(phys->createRigidDynamic(pose));
    PxShape* sh = PxRigidActorExt::createExclusiveShape(*a, g, *mat);
    if (!isStatic) {
      static_cast<PxRigidDynamic*>(a)->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    }
    scene->addActor(*a);
    s.pose = toE(a->getGlobalPose());  // PhysX 는 생성 때 자세를 정규화해 둔다(PxTransform::getNormalized) — 장면 질의 자세는 그 값
    (isStatic ? stat : dyn).push_back(s);
    (isStatic ? pxStat : pxDyn).push_back(sh);
  };
  {
    S::SqShape s{};
    s.type = S::SQ_PLANE;
    addShape(true, PxPlaneGeometry(), s);
  }
  for (int i = 0; i < nShapes; ++i) {
    S::SqShape s{};
    s.scale = S::MeshScale{eng::V3{1, 1, 1}, eng::Q{0, 0, 0, 1}};
    const int k = int(rng() % 7u);
    if (k == 0) {  // 정적 상자
      s.type = S::SQ_BOX;
      const PxVec3 he(0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng));
      s.halfExtents = toE(he);
      addShape(true, PxBoxGeometry(he), s);
    } else if (k == 1) {  // 정적 삼각 메시
      const int m = int(rng() % tms.size());
      s.type = S::SQ_TRIMESH;
      s.mesh = &tms[m].d;
      PxMeshScale ms(PxVec3(1.0f));
      if (rng() % 2u) ms = PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)), rq());
      s.scale = S::MeshScale{toE(ms.scale), eng::Q{ms.rotation.x, ms.rotation.y, ms.rotation.z, ms.rotation.w}};
      s.doubleSided = uint8_t(rng() % 2u);
      addShape(true, PxTriangleMeshGeometry(tms[m].px, ms, s.doubleSided ? PxMeshGeometryFlags(PxMeshGeometryFlag::eDOUBLE_SIDED) : PxMeshGeometryFlags()), s);
    } else if (k == 2) {
      s.type = S::SQ_BOX;
      const PxVec3 he(0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng), 0.05f + 0.4f * P(rng));
      s.halfExtents = toE(he);
      addShape(false, PxBoxGeometry(he), s);
    } else if (k == 3) {
      s.type = S::SQ_SPHERE;
      s.radius = 0.05f + 0.4f * P(rng);
      addShape(false, PxSphereGeometry(s.radius), s);
    } else if (k == 4) {
      s.type = S::SQ_CAPSULE;
      s.radius = 0.05f + 0.3f * P(rng);
      s.halfHeight = 0.4f * P(rng);
      addShape(false, PxCapsuleGeometry(s.radius, s.halfHeight), s);
    } else {
      const int m = int(rng() % cms.size());
      s.type = S::SQ_CONVEX;
      s.planes = cplanes[m].data();
      s.nPolys = uint32_t(cplanes[m].size() / 4);
      PxMeshScale ms(PxVec3(1.0f));
      if (rng() % 2u) ms = PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)), rq());
      s.scale = S::MeshScale{toE(ms.scale), eng::Q{ms.rotation.x, ms.rotation.y, ms.rotation.z, ms.rotation.w}};
      addShape(false, PxConvexMeshGeometry(cms[m], ms), s);
    }
  }
  std::vector<S::SqShape> all(stat), rev;
  all.insert(all.end(), dyn.begin(), dyn.end());
  std::vector<PxShape*> pxAll(pxStat);
  pxAll.insert(pxAll.end(), pxDyn.begin(), pxDyn.end());
  for (uint32_t i = 0; i < all.size(); ++i) all[i].id = i;
  rev.assign(all.rbegin(), all.rend());

  uint64_t cmp = 0, hitsN = 0, bad = 0, orderSens = 0, typeHits[6] = {0};
  int shown = 0;
  for (int r = 0; r < rays; ++r) {
    PxVec3 origin = rv(4.0f);
    PxVec3 dir;
    const eng::V3 tp = all[rng() % all.size()].pose.p;
    dir = (rng() % 3u == 0) ? rv(1.0f) : (PxVec3(tp.x, tp.y, tp.z) + rv(0.3f) - origin);
    dir.normalize();
    const float maxDist = 1.0f + 10.0f * P(rng);
    PxRaycastBuffer buf;
    scene->raycast(origin, dir, maxDist, buf);
    S::RayHit eh{}, eh2{};
    int ei, ei2;
    {
      FtzScope f;
      ei = S::raycastClosest(all.data(), uint32_t(all.size()), toE(origin), toE(dir), maxDist, S::HF_DEFAULT, eh);
      ei2 = S::raycastClosest(rev.data(), uint32_t(rev.size()), toE(origin), toE(dir), maxDist, S::HF_DEFAULT, eh2);
    }
    const int id1 = ei >= 0 ? int(all[ei].id) : -1, id2 = ei2 >= 0 ? int(rev[ei2].id) : -1;
    if (id1 != id2 || (ei >= 0 && memcmp(&eh.distance, &eh2.distance, 4))) { orderSens++; continue; }
    cmp++;
    bool ok = buf.hasBlock == (ei >= 0);
    if (ok && buf.hasBlock) {
      hitsN++;
      typeHits[all[ei].type]++;
      const PxRaycastHit& ph = buf.block;
      ok = ph.shape == pxAll[ei] && !memcmp(&ph.distance, &eh.distance, 4) && !memcmp(&ph.position, &eh.position, 12) &&
           !memcmp(&ph.normal, &eh.normal, 12) && ph.faceIndex == eh.faceIndex && uint32_t(PxU16(ph.flags)) == eh.flags &&
           !memcmp(&ph.u, &eh.u, 4) && !memcmp(&ph.v, &eh.v, 4);
    }
    if (!ok) {
      bad++;
      if (shown < 10) {
        shown++;
        int pidx = -1;
        for (size_t i = 0; i < pxAll.size(); ++i) if (buf.hasBlock && buf.block.shape == pxAll[i]) pidx = int(i);
        if (pidx >= 0) {
          PxShape* sh = pxAll[pidx];
          const PxTransform gp = PxShapeExt::getGlobalPose(*sh, *sh->getActor());
          PxGeomRaycastHit gh;
          const PxU32 gn = PxGeometryQuery::raycast(origin, dir, sh->getGeometry(), gp, maxDist, PxHitFlag::eDEFAULT, 1, &gh);
          const eng::Tf& ep = all[pidx].pose;
          printf("  PxGeometryQuery(전역자세) 맞음 %u 거리 %.9g, 자세 q(%a %a %a %a) p(%a %a %a) / 우리 q(%a %a %a %a) p(%a %a %a)\n", gn, gh.distance, gp.q.x, gp.q.y,
                 gp.q.z, gp.q.w, gp.p.x, gp.p.y, gp.p.z, ep.q.x, ep.q.y, ep.q.z, ep.q.w, ep.p.x, ep.p.y, ep.p.z);
        }
        printf("[다름] 광선 %d: PhysX 모양 %d 거리 %.9g / 우리 모양 %d(종류 %d) 거리 %.9g\n", r, pidx, buf.hasBlock ? buf.block.distance : -1.0f, ei,
               ei >= 0 ? all[ei].type : -1, ei >= 0 ? eh.distance : -1.0f);
      }
    }
  }
  printf("\n장면 가장 가까운 광선 (광선 %d, 씨앗 %d, 모양 %zu 개: 정적 %zu 동적 %zu)\n", rays, seed, all.size(), stat.size(), dyn.size());
  printf("  비교 %" PRIu64 "  맞음 %" PRIu64 " (구 %" PRIu64 " 평면 %" PRIu64 " 캡슐 %" PRIu64 " 상자 %" PRIu64 " 볼록 %" PRIu64 " 삼각 %" PRIu64 ")  비트 다름 %" PRIu64 "\n", cmp,
         hitsN, typeHits[0], typeHits[1], typeHits[2], typeHits[3], typeHits[4], typeHits[5], bad);
  printf("  방문 순서 민감(비교에서 뺌) %" PRIu64 "\n", orderSens);
  printf("%s\n", bad == 0 ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  scene->release();
  phys->release();
  fnd->release();
  return bad == 0 ? 0 : 3;
}
