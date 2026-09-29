// 층 1 시험 (contact): PCM 기본 모양 쌍 전부 = PhysX 5.6.1 비트 동일?
// 구·평면·캡슐·상자·볼록의 모든 조합(평면-평면 제외, 모양 번호 작은 쪽이 앞 = PhysX 접촉 함수 표 순서)을
// 같은 자세 열로 양쪽에 넣고 매 프레임 결과·접촉점·다양체를 비교한다. 다양체 여부·종류는 PxsContext::createCache 와 같게
// (gEnablePCMCaching 표, PxsContext.cpp:89; 구가 끼면 1점 다양체, 아니면 4점).
//   test_pcm_prims [--pairs N] [--frames F] [--seed S]
#include "px_bridge.h"

using namespace physx;
namespace ep = eng::px;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

// PxsContext.cpp:89 gEnablePCMCaching (구·평면·캡슐·상자·볼록코어·볼록메시 부분)
static const bool kCache[6][6] = {
    {false, false, false, false, false, true},  // 구
    {false, false, true, true, false, true},    // 평면
    {false, true, false, true, false, true},    // 캡슐
    {false, true, true, true, false, true},     // 상자
    {false, false, false, false, false, false},  // 볼록코어 (안 씀)
    {true, true, true, true, false, true},      // 볼록메시
};

struct Shape {
  int type;
  PxGeometryHolder p;
  ep::PxSphereGeometry es;
  ep::PxCapsuleGeometry ec;
  ep::PxBoxGeometry eb;
  ep::PxPlaneGeometry epl;
  ep::PxConvexMeshGeometry em;
  float radius;  // 대략 크기
  const ep::PxGeometry& e() const {
    switch (type) {
      case 0: return es;
      case 1: return epl;
      case 2: return ec;
      case 3: return eb;
      default: return em;
    }
  }
};

static bool callP(int t0, int t1, const PxGeometry& a, const PxGeometry& b, const PxTransform32& ta, const PxTransform32& tb,
                  const Gu::NarrowPhaseParams& np, Gu::Cache& c, PxContactBuffer& buf) {
  const int k = t0 * 8 + t1;
  switch (k) {
    case 0 * 8 + 0: return Gu::pcmContactSphereSphere(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 8 + 1: return Gu::pcmContactSpherePlane(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 8 + 2: return Gu::pcmContactSphereCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 8 + 3: return Gu::pcmContactSphereBox(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 8 + 5: return Gu::pcmContactSphereConvex(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 8 + 2: return Gu::pcmContactPlaneCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 8 + 3: return Gu::pcmContactPlaneBox(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 8 + 5: return Gu::pcmContactPlaneConvex(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 8 + 2: return Gu::pcmContactCapsuleCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 8 + 3: return Gu::pcmContactCapsuleBox(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 8 + 5: return Gu::pcmContactCapsuleConvex(a, b, ta, tb, np, c, buf, NULL);
    case 3 * 8 + 3: return Gu::pcmContactBoxBox(a, b, ta, tb, np, c, buf, NULL);
    case 3 * 8 + 5: return Gu::pcmContactBoxConvex(a, b, ta, tb, np, c, buf, NULL);
    case 5 * 8 + 5: return Gu::pcmContactConvexConvex(a, b, ta, tb, np, c, buf, NULL);
  }
  abort();
}
static bool callE(int t0, int t1, const ep::PxGeometry& a, const ep::PxGeometry& b, const ep::PxTransform32& ta, const ep::PxTransform32& tb,
                  const ep::Gu::NarrowPhaseParams& np, ep::Gu::Cache& c, ep::PxContactBuffer& buf) {
  namespace G = ep::Gu;
  const int k = t0 * 8 + t1;
  switch (k) {
    case 0 * 8 + 0: return G::pcmContactSphereSphere(a, b, ta, tb, np, c, buf, nullptr);
    case 0 * 8 + 1: return G::pcmContactSpherePlane(a, b, ta, tb, np, c, buf, nullptr);
    case 0 * 8 + 2: return G::pcmContactSphereCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case 0 * 8 + 3: return G::pcmContactSphereBox(a, b, ta, tb, np, c, buf, nullptr);
    case 0 * 8 + 5: return G::pcmContactSphereConvex(a, b, ta, tb, np, c, buf, nullptr);
    case 1 * 8 + 2: return G::pcmContactPlaneCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case 1 * 8 + 3: return G::pcmContactPlaneBox(a, b, ta, tb, np, c, buf, nullptr);
    case 1 * 8 + 5: return G::pcmContactPlaneConvex(a, b, ta, tb, np, c, buf, nullptr);
    case 2 * 8 + 2: return G::pcmContactCapsuleCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case 2 * 8 + 3: return G::pcmContactCapsuleBox(a, b, ta, tb, np, c, buf, nullptr);
    case 2 * 8 + 5: return G::pcmContactCapsuleConvex(a, b, ta, tb, np, c, buf, nullptr);
    case 3 * 8 + 3: return G::pcmContactBoxBox(a, b, ta, tb, np, c, buf, nullptr);
    case 3 * 8 + 5: return G::pcmContactBoxConvex(a, b, ta, tb, np, c, buf, nullptr);
    case 5 * 8 + 5: return G::pcmContactConvexConvex(a, b, ta, tb, np, c, buf, nullptr);
  }
  abort();
}

int main(int argc, char** argv) {
  int pairs = 400, frames = 40, seed = 3;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
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
  auto randQuat = [&]() { PxQuat q(U(rng), U(rng), U(rng), U(rng)); q.normalize(); return q; };
  auto makeShape = [&](int type, Shape& s) {
    s.type = type;
    switch (type) {
      case 0: { const float r = 0.02f + 0.4f * P(rng); s.p.storeAny(PxSphereGeometry(r)); s.es = ep::PxSphereGeometry(r); s.radius = r; break; }
      case 1: { s.p.storeAny(PxPlaneGeometry()); s.radius = 0.0f; break; }
      case 2: {
        const float r = 0.02f + 0.3f * P(rng), hh = 0.01f + 0.5f * P(rng);
        s.p.storeAny(PxCapsuleGeometry(r, hh)); s.ec = ep::PxCapsuleGeometry(r, hh); s.radius = r + hh; break;
      }
      case 3: {
        const PxVec3 he(0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng));
        s.p.storeAny(PxBoxGeometry(he)); s.eb = ep::PxBoxGeometry(he.x, he.y, he.z); s.radius = he.magnitude(); break;
      }
      default: {
        PxConvexMesh* m = hulls[size_t(P(rng) * hulls.size()) % hulls.size()];
        PxMeshScale sc = (P(rng) < 0.5f) ? PxMeshScale(1.0f) : PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)));
        const PxConvexMeshGeometry g(m, sc);
        s.p.storeAny(g); s.em = cxt::toE(g);
        s.radius = static_cast<Gu::ConvexMesh*>(m)->getLocalBoundsFast().mExtents.magnitude() * sc.scale.maxElement();
      }
    }
  };

  const float contactDist = 0.04f;
  const Gu::NarrowPhaseParams npP(contactDist, 0.01f, 1.0f);
  const ep::Gu::NarrowPhaseParams npE(contactDist, 0.01f, 1.0f);
  const int combos[][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 5}, {1, 2}, {1, 3}, {1, 5}, {2, 2}, {2, 3}, {2, 5}, {3, 3}, {3, 5}, {5, 5}};
  const char* tname[] = {"구", "평면", "캡슐", "상자", "코어", "볼록"};
  static PxContactBuffer bufP;
  static ep::PxContactBuffer bufE;
  uint64_t badAll = 0;
  for (const auto& cb : combos) {
    const int t0 = cb[0], t1 = cb[1];
    cxt::Tally tr("결과·수"), tc("접촉점"), tm("다양체");
    long long contacts = 0;
    for (int c = 0; c < pairs; ++c) {
      Shape a, b;
      makeShape(t0, a);
      makeShape(t1, b);
      PxTransform tf0(PxVec3(U(rng), U(rng), U(rng)), randQuat()), tf1;
      if (t0 == 1) {  // 평면: x 축이 법선. 상대를 평면 근처에 둔다
        const PxVec3 n = tf0.q.rotate(PxVec3(1, 0, 0));
        tf1 = PxTransform(tf0.p + n * (b.radius * (P(rng) * 1.4f - 0.3f)) + tf0.q.rotate(PxVec3(0, U(rng), U(rng))), randQuat());
      } else if (t1 == 1) {
        const PxVec3 n = PxTransform(tf0).q.rotate(PxVec3(1, 0, 0));
        tf1 = tf0;
        tf0 = PxTransform(tf1.p + n * (a.radius * (P(rng) * 1.4f - 0.3f)), randQuat());
      } else {
        PxVec3 d(U(rng), U(rng), U(rng)); d.normalize();
        tf1 = PxTransform(tf0.p + d * ((a.radius + b.radius) * (0.3f + 0.8f * P(rng))), randQuat());
      }
      const bool cache = kCache[t0][t1];
      const bool sphere = t0 == 0 || t1 == 0;
      alignas(16) Gu::LargePersistentContactManifold lP; alignas(16) Gu::SpherePersistentContactManifold sP;
      alignas(16) ep::Gu::LargePersistentContactManifold lE; alignas(16) ep::Gu::SpherePersistentContactManifold sE;
      Gu::Cache cP; ep::Gu::Cache cE;
      Gu::PersistentContactManifold* mP = nullptr;
      ep::Gu::PersistentContactManifold* mE = nullptr;
      if (cache) {
        if (sphere) { mP = &sP; mE = &sE; } else { mP = &lP; mE = &lE; }
        cP.setManifold(mP); cP.getManifold().clearManifold();
        cE.setManifold(mE); cE.getManifold().clearManifold();
      }
      const float step = 0.01f * (a.radius + b.radius + 0.1f);
      for (int f = 0; f < frames; ++f) {
        bool rp, re;
        {
          cxt::FtzScope fz;
          bufP.reset();
          rp = callP(t0, t1, a.p.any(), b.p.any(), PxTransform32(tf0), PxTransform32(tf1), npP, cP, bufP);
          bufE.reset();
          re = callE(t0, t1, a.e(), b.e(), ep::PxTransform32(cxt::toE(tf0)), ep::PxTransform32(cxt::toE(tf1)), npE, cE, bufE);
        }
        tr.u(rp, re, c, f);
        tr.u(bufP.count, bufE.count, c, f);
        contacts += bufP.count;
        const PxU32 nc = bufP.count < bufE.count ? bufP.count : bufE.count;
        for (PxU32 i = 0; i < nc; ++i) {
          const PxContactPoint& x = bufP.contacts[i];
          const ep::PxContactPoint& y = bufE.contacts[i];
          tc.f(x.normal.x, y.normal.x, c, f); tc.f(x.normal.y, y.normal.y, c, f); tc.f(x.normal.z, y.normal.z, c, f);
          tc.f(x.separation, y.separation, c, f);
          tc.f(x.point.x, y.point.x, c, f); tc.f(x.point.y, y.point.y, c, f); tc.f(x.point.z, y.point.z, c, f);
          tc.u(x.internalFaceIndex1, y.internalFaceIndex1, c, f);
        }
        if (cache) {
          tm.u(mP->mNumContacts, mE->mNumContacts, c, f);
          tm.u(mP->mNumWarmStartPoints, mE->mNumWarmStartPoints, c, f);
          const int nm = mP->mNumContacts < mE->mNumContacts ? mP->mNumContacts : mE->mNumContacts;
          for (int k = 0; k < nm; ++k) {
            float x[12], y[12];
            memcpy(x, &mP->mContactPoints[k], 48);
            memcpy(y, &mE->mContactPoints[k], 48);
            for (int q = 0; q < 12; ++q) tm.f(x[q], y[q], c, f);
          }
          float x[12], y[12];
          memcpy(x, &mP->mRelativeTransform, 32); memcpy(x + 8, &mP->mQuatA, 16);
          memcpy(y, &mE->mRelativeTransform, 32); memcpy(y + 8, &mE->mQuatA, 16);
          for (int q = 0; q < 12; ++q) tm.f(x[q], y[q], c, f);
        }
        const float s = P(rng) < 0.05f ? 10.0f : 1.0f;
        tf1.p += PxVec3(U(rng), U(rng), U(rng)) * (step * s);
        PxVec3 ax(U(rng), U(rng), U(rng));
        ax = ax.isZero() ? PxVec3(0, 0, 1) : ax.getNormalized();
        tf1.q = (PxQuat(0.02f * s * U(rng), ax) * tf1.q).getNormalized();
        const float lim = 1.3f * (a.radius + b.radius) + 0.1f;
        if (t0 != 1 && t1 != 1 && (tf1.p - tf0.p).magnitude() > lim) tf1.p = tf0.p + (tf1.p - tf0.p) * 0.5f;
      }
    }
    const uint64_t bad = tr.bad + tc.bad + tm.bad;
    badAll += bad;
    printf("%s-%s: 쌍 %d × 프레임 %d, 접촉점 %lld, 비교 %" PRIu64 ", 비트 다름 %" PRIu64 "\n", tname[t0], tname[t1], pairs, frames, contacts,
           tr.cmp + tc.cmp + tm.cmp, bad);
    if (bad) { tr.print(); tc.print(); tm.print(); }
  }
  printf(badAll ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return badAll ? 1 : 0;
}
