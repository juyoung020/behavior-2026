// 층 1 시험 (contact): PCM 볼록-볼록 접촉 생성 = PhysX 5.6.1 Gu::pcmContactConvexConvex 비트 동일?
// 같은 두 볼록(PhysX 가 구운 같은 데이터), 같은 자세 열(프레임마다 조금씩 움직임 -> 지속 다양체 갱신·무효화·전체 재생성 경로를 다 탐),
// 같은 접촉 거리로 양쪽을 부르고, 매 프레임 결과(참/거짓, 접촉 수, 점·법선·분리·면 번호)와 다양체 내부 상태를 비트 비교한다.
//   test_pcm_convex [--pairs N] [--frames F] [--seed S] [--hulls H]
#include "px_bridge.h"

using namespace physx;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

static void v4(const aos::Vec3V& v, float* o) { memcpy(o, &v, 16); }

int main(int argc, char** argv) {
  int pairs = 2000, frames = 40, seed = 7, nhulls = 160;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hulls") && i + 1 < argc) nhulls = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);

  // ---- 볼록 준비: 작은 것(<32 꼭짓점, 전수 탐색 지지점)과 큰 것(>=32, 가우스 지도 언덕 오르기), GPU 데이터 있음/없음
  std::vector<PxConvexMesh*> hulls;
  int nBig = 0;
  for (int h = 0; h < nhulls; ++h) {
    const int kind = h % 4;
    const int n = (h % 3 == 0) ? 40 + int(P(rng) * 60) : 8 + int(P(rng) * 20);
    PxConvexMesh* m = cxt::cookHull(phys, cxt::randomCloud(rng, kind, n), (h % 5) < 2, 64);
    if (!m) continue;
    if (static_cast<Gu::ConvexMesh*>(m)->getBigConvexData()) ++nBig;
    hulls.push_back(m);
  }
  printf("볼록 %zu 개 (큰 볼록 %d 개)\n", hulls.size(), nBig);

  const float contactDist = 0.02f + 0.02f;  // contactOffset 0.02 두 개 (PxcNpBatch.cpp:362)
  const Gu::NarrowPhaseParams npP(contactDist, 0.01f * tol.length, tol.length);
  const eng::px::Gu::NarrowPhaseParams npE(contactDist, 0.01f * tol.length, tol.length);

  cxt::Tally tRet("반환(참/거짓)"), tCount("접촉 수"), tNorm("법선"), tSep("분리"), tPt("점"), tFace("면 번호"),
      tMan("다양체 점 수·웜스타트"), tManPt("다양체 점(국소 A/B·법선깊이)"), tRel("다양체 상대 자세");
  long long totalContacts = 0, framesWithContacts = 0;

  auto randScale = [&](int c) -> PxMeshScale {
    switch (c % 4) {
      case 0: return PxMeshScale(1.0f);
      case 1: return PxMeshScale(0.5f + 1.5f * P(rng));
      case 2: return PxMeshScale(PxVec3(0.4f + P(rng), 0.4f + P(rng), 0.4f + P(rng)));
      default: {
        PxQuat r(U(rng), U(rng), U(rng), U(rng));
        r.normalize();
        return PxMeshScale(PxVec3(0.4f + P(rng), 0.4f + P(rng), 0.4f + P(rng)), r);
      }
    }
  };
  auto randQuat = [&]() {
    PxQuat q(U(rng), U(rng), U(rng), U(rng));
    q.normalize();
    return q;
  };

  for (int c = 0; c < pairs; ++c) {
    PxConvexMesh* ma = hulls[size_t(P(rng) * hulls.size()) % hulls.size()];
    PxConvexMesh* mb = hulls[size_t(P(rng) * hulls.size()) % hulls.size()];
    const PxConvexMeshGeometry g0(ma, randScale(c)), g1(mb, randScale(c / 4));
    const eng::px::PxConvexMeshGeometry e0 = cxt::toE(g0), e1 = cxt::toE(g1);
    const float r0 = static_cast<Gu::ConvexMesh*>(ma)->getLocalBoundsFast().mExtents.magnitude() * g0.scale.scale.maxElement();
    const float r1 = static_cast<Gu::ConvexMesh*>(mb)->getLocalBoundsFast().mExtents.magnitude() * g1.scale.scale.maxElement();

    PxTransform tf0(PxVec3(U(rng), U(rng), U(rng)), randQuat());
    PxVec3 dir(U(rng), U(rng), U(rng));
    dir.normalize();
    PxTransform tf1(tf0.p + dir * ((r0 + r1) * (0.25f + 0.8f * P(rng))), randQuat());

    // 지속 다양체 (PxsContext::createCache 와 같게: 볼록끼리는 LargePersistentContactManifold)
    alignas(16) Gu::LargePersistentContactManifold manP;
    Gu::Cache cacheP;
    cacheP.setManifold(&manP);
    cacheP.getManifold().clearManifold();
    alignas(16) eng::px::Gu::LargePersistentContactManifold manE;
    eng::px::Gu::Cache cacheE;
    cacheE.setManifold(&manE);
    cacheE.getManifold().clearManifold();
    static PxContactBuffer bufP;
    static eng::px::PxContactBuffer bufE;

    const float step = 0.01f * (r0 + r1);
    for (int f = 0; f < frames; ++f) {
      bool rp, re;
      {
        cxt::FtzScope fz;
        bufP.reset();
        rp = Gu::pcmContactConvexConvex(g0, g1, PxTransform32(tf0), PxTransform32(tf1), npP, cacheP, bufP, NULL);
      }
      {
        cxt::FtzScope fz;
        bufE.reset();
        re = eng::px::Gu::pcmContactConvexConvex(e0, e1, eng::px::PxTransform32(cxt::toE(tf0)), eng::px::PxTransform32(cxt::toE(tf1)),
                                                 npE, cacheE, bufE, nullptr);
      }
      tRet.u(rp, re, c, f);
      tCount.u(bufP.count, bufE.count, c, f);
      if (bufP.count) ++framesWithContacts;
      totalContacts += bufP.count;
      const PxU32 nc = bufP.count < bufE.count ? bufP.count : bufE.count;
      for (PxU32 i = 0; i < nc; ++i) {
        const PxContactPoint& a = bufP.contacts[i];
        const eng::px::PxContactPoint& b = bufE.contacts[i];
        tNorm.f(a.normal.x, b.normal.x, c, f); tNorm.f(a.normal.y, b.normal.y, c, f); tNorm.f(a.normal.z, b.normal.z, c, f);
        tSep.f(a.separation, b.separation, c, f);
        tPt.f(a.point.x, b.point.x, c, f); tPt.f(a.point.y, b.point.y, c, f); tPt.f(a.point.z, b.point.z, c, f);
        tFace.u(a.internalFaceIndex1, b.internalFaceIndex1, c, f);
      }
      // 다양체 내부 상태 (다음 프레임 결과를 정하는 것 전부)
      tMan.u(manP.mNumContacts, manE.mNumContacts, c, f);
      tMan.u(manP.mNumWarmStartPoints, manE.mNumWarmStartPoints, c, f);
      for (int k = 0; k < manP.mNumWarmStartPoints && k < 4; ++k) {
        tMan.u(manP.mAIndice[k], manE.mAIndice[k], c, f);
        tMan.u(manP.mBIndice[k], manE.mBIndice[k], c, f);
      }
      const int nm = manP.mNumContacts < manE.mNumContacts ? manP.mNumContacts : manE.mNumContacts;
      for (int k = 0; k < nm; ++k) {
        float a[12], b[12];
        v4(manP.mContactPoints[k].mLocalPointA, a); v4(manP.mContactPoints[k].mLocalPointB, a + 4); v4(manP.mContactPoints[k].mLocalNormalPen, a + 8);
        memcpy(b, &manE.mContactPoints[k].mLocalPointA, 16); memcpy(b + 4, &manE.mContactPoints[k].mLocalPointB, 16);
        memcpy(b + 8, &manE.mContactPoints[k].mLocalNormalPen, 16);
        for (int q = 0; q < 12; ++q) tManPt.f(a[q], b[q], c, f);
      }
      {
        float a[12], b[12];
        memcpy(a, &manP.mRelativeTransform.p, 16); memcpy(a + 4, &manP.mRelativeTransform.q, 16); memcpy(a + 8, &manP.mQuatA, 16);
        memcpy(b, &manE.mRelativeTransform.p, 16); memcpy(b + 4, &manE.mRelativeTransform.q, 16); memcpy(b + 8, &manE.mQuatA, 16);
        for (int q = 0; q < 12; ++q) tRel.f(a[q], b[q], c, f);
      }

      // 다음 프레임 자세: 작은 흔들림(대부분) + 가끔 큰 이동/회전 (다양체 재생성 경로)
      const bool jump = P(rng) < 0.05f;
      const float s = jump ? 10.0f : 1.0f;
      tf1.p += PxVec3(U(rng), U(rng), U(rng)) * (step * s);
      const PxVec3 axis = PxVec3(U(rng), U(rng), U(rng)).getNormalized();
      tf1.q = (PxQuat(0.02f * s * U(rng), axis.isZero() ? PxVec3(0, 0, 1) : axis) * tf1.q).getNormalized();
      tf0.p += PxVec3(U(rng), U(rng), U(rng)) * (0.3f * step);
      // 두 물체가 멀어지면 다시 붙인다
      if ((tf1.p - tf0.p).magnitude() > 1.3f * (r0 + r1)) tf1.p = tf0.p + (tf1.p - tf0.p) * 0.5f;
    }
  }

  printf("쌍 %d × 프레임 %d, 접촉 있던 프레임 %lld, 접촉점 합 %lld\n", pairs, frames, framesWithContacts, totalContacts);
  const cxt::Tally* all[] = {&tRet, &tCount, &tNorm, &tSep, &tPt, &tFace, &tMan, &tManPt, &tRel};
  uint64_t bad = 0;
  for (const cxt::Tally* t : all) { t->print(); bad += t->bad; }
  printf(bad ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return bad ? 1 : 0;
}
