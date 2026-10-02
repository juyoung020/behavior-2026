// 층 1 시험 (contact, 경계 상자): 우리 Gu::computeBounds(core/contact/px/gu_bounds.inc) = PhysX 5.6.1 Gu::computeBounds?
// 구·평면(축 정렬 포함)·캡슐·상자·볼록(꼭 맞는 경계 eTIGHT_BOUNDS 켬/끔, 크기 늘임 있음/없음)을 무작위 자세·contactOffset 으로
// 양쪽에 넣고 최소·최대 6 값을 비트 비교한다. FTZ 켬(시뮬레이션 안 경계 갱신 작업)과 끔 둘 다.
//   test_bounds [--cases N] [--seed S]
#include "px_bridge.h"
#include "GuBounds.h"

using namespace physx;
namespace ep = eng::px;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

int main(int argc, char** argv) {
  int cases = 200000, seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--cases") && i + 1 < argc) cases = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  std::vector<PxConvexMesh*> hulls;
  for (int h = 0; h < 60; ++h) {
    const int n = (h % 3 == 0) ? 40 + int(P(rng) * 60) : 8 + int(P(rng) * 20);
    PxConvexMesh* m = cxt::cookHull(phys, cxt::randomCloud(rng, h % 4, n), (h % 5) < 2, 64);
    if (m) hulls.push_back(m);
  }
  auto randQuat = [&]() {
    if (P(rng) < 0.15f) {  // 축 정렬 (평면 경계가 좁아지는 길)
      static const PxQuat qs[] = {PxQuat(PxIdentity), PxQuat(PxHalfPi, PxVec3(0, 0, 1)), PxQuat(-PxHalfPi, PxVec3(0, 1, 0)), PxQuat(PxPi, PxVec3(0, 0, 1))};
      return qs[size_t(P(rng) * 4) % 4];
    }
    PxQuat q(U(rng), U(rng), U(rng), U(rng));
    q.normalize();
    return q;
  };
  const char* tname[] = {"구", "평면", "캡슐", "상자", "볼록"};
  cxt::Tally tal[5] = {cxt::Tally("구"), cxt::Tally("평면"), cxt::Tally("캡슐"), cxt::Tally("상자"), cxt::Tally("볼록")};
  for (int ftz = 0; ftz < 2; ++ftz) {
    for (int c = 0; c < cases; ++c) {
      const int t = c % 5;
      const PxTransform pose(PxVec3(U(rng), U(rng), U(rng)) * (P(rng) < 0.5f ? 3.0f : 300.0f), randQuat());
      const float co = P(rng) < 0.1f ? 0.0f : 0.001f + 0.05f * P(rng);
      const float infl = P(rng) < 0.8f ? 1.0f : 1.01f;
      PxGeometryHolder gp;
      ep::PxSphereGeometry es;
      ep::PxCapsuleGeometry ec;
      ep::PxBoxGeometry eb;
      ep::PxPlaneGeometry epl;
      ep::PxConvexMeshGeometry em;
      const ep::PxGeometry* ge = nullptr;
      switch (t) {
        case 0: { const float r = 0.02f + 0.4f * P(rng); gp.storeAny(PxSphereGeometry(r)); es = ep::PxSphereGeometry(r); ge = &es; break; }
        case 1: { gp.storeAny(PxPlaneGeometry()); ge = &epl; break; }
        case 2: {
          const float r = 0.02f + 0.3f * P(rng), hh = 0.01f + 0.5f * P(rng);
          gp.storeAny(PxCapsuleGeometry(r, hh)); ec = ep::PxCapsuleGeometry(r, hh); ge = &ec; break;
        }
        case 3: {
          const PxVec3 he(0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng));
          gp.storeAny(PxBoxGeometry(he)); eb = ep::PxBoxGeometry(he.x, he.y, he.z); ge = &eb; break;
        }
        default: {
          PxConvexMesh* m = hulls[size_t(P(rng) * hulls.size()) % hulls.size()];
          PxMeshScale sc = (P(rng) < 0.5f) ? PxMeshScale(1.0f) : PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)), randQuat());
          PxConvexMeshGeometry g(m, sc);
          g.meshFlags = P(rng) < 0.5f ? PxConvexMeshGeometryFlags(PxConvexMeshGeometryFlag::eTIGHT_BOUNDS) : PxConvexMeshGeometryFlags();
          gp.storeAny(g); em = cxt::toE(g); ge = &em;
        }
      }
      PxBounds3 bP;
      ep::PxBounds3 bE;
      {
        cxt::FtzScope* f = ftz ? new cxt::FtzScope() : nullptr;
        Gu::computeBounds(bP, gp.any(), pose, co, infl);
        ep::Gu::computeBounds(bE, *ge, cxt::toE(pose), co, infl);
        delete f;
      }
      tal[t].f(bP.minimum.x, bE.minimum.x, c, ftz);
      tal[t].f(bP.minimum.y, bE.minimum.y, c, ftz);
      tal[t].f(bP.minimum.z, bE.minimum.z, c, ftz);
      tal[t].f(bP.maximum.x, bE.maximum.x, c, ftz);
      tal[t].f(bP.maximum.y, bE.maximum.y, c, ftz);
      tal[t].f(bP.maximum.z, bE.maximum.z, c, ftz);
    }
  }
  (void)tname;
  uint64_t bad = 0;
  for (auto& x : tal) { x.print(); bad += x.bad; }
  printf("%s\n", bad ? "결과: 비트 다름 있음" : "결과: 전부 비트 동일");
  return bad ? 1 : 0;
}
