// 층 1 시험 (contact, 장면 단위): PhysX 장면을 실제로 돌리면서, 매 스텝 PhysX 가 만든 접촉 관리자 출력(패치 머리·접촉점)을
// 우리 좁은 단계 + 패치 만들기로 같은 입력에서 다시 만들어 비트 비교한다.
//  - 모양 월드 자세: 몸체 자세·질량중심 자세·모양 국소 자세 -> 우리 계산 = PhysX 변환 캐시 (Sc::ShapeSimBase::getAbsPoseAligned)
//  - 접촉 거리 = 두 모양 contactOffset 합, 모양 번호 순서 뒤집기, 재질 번호, 재질 섞기, 패치 나누기(PXC_SAME_NORMAL)
//  - 지속 다양체는 쌍마다 우리 쪽에 따로 두고 스텝마다 이어 간다 (PhysX 내부 다양체와 비교하지 않고 결과로만 비교)
// 잠·얼림은 끈다(잠든 쌍은 PhysX 가 좁은 단계를 건너뛰므로 그 규칙은 나중에 섬(solver)과 함께 옮긴다).
//   test_np_scene [--bodies N] [--steps S] [--seed X]
#include "px_bridge.h"
#include "core/contact/patches.h"

// PhysX 내부 (정답지 읽기 전용)
#include "NpScene.h"
#include "NpShape.h"
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "ScScene.h"
#include "ScShapeInteraction.h"
#include "PxsContext.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsTransformCache.h"
#include "PxvNphaseImplementationContext.h"

#include <map>
#include <set>

using namespace physx;
namespace ec = eng::contact;
namespace ep = eng::px;

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

struct OurShape {
  ec::ShapeGeom geom;
  cxt::TestShape ts;  // PhysX Gu 함수를 직접 부를 때 (진단용)
  uint16_t material;
  float contactOffset;
  // 자세 입력
  bool isStatic;
  const PxRigidActor* actor;
  ep::PxTransform shape2Actor;
  bool idtShape;
};

// Sc::ShapeSimBase::getAbsPoseAligned (ScShapeSimBase.cpp:217) 를 우리 수학으로
static ep::PxTransform shapeWorldPose(const OurShape& s, const ep::PxTransform& body2World, const ep::PxTransform& body2Actor, bool idtBody2Actor) {
  alignas(16) ep::PxTransform out;
  if (s.isStatic) {
    if (s.idtShape) return body2World;
    ep::Cm::getStaticGlobalPoseAligned(body2World, s.shape2Actor, out);
    return out;
  }
  if (!idtBody2Actor) {
    ep::Cm::getDynamicGlobalPoseAligned(body2World, s.shape2Actor, body2Actor, out);
    return out;
  }
  ep::Cm::getStaticGlobalPoseAligned(body2World, s.shape2Actor, out);
  return out;
}

int main(int argc, char** argv) {
  int nb = 120, steps = 300, seed = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bodies") && i + 1 < argc) nb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM;
  PxScene* scene = phys->createScene(sd);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  std::vector<PxConvexMesh*> hulls;
  for (int h = 0; h < 40; ++h) {
    const int n = (h % 3 == 0) ? 40 + int(P(rng) * 40) : 8 + int(P(rng) * 16);
    PxConvexMesh* m = cxt::cookHull(phys, cxt::randomCloud(rng, h % 4, n), (h % 5) < 2, 64);
    if (m) hulls.push_back(m);
  }

  // 재질 몇 개 (섞기 방식 다르게)
  std::vector<PxMaterial*> mats;
  for (int m = 0; m < 6; ++m) {
    PxMaterial* mt = phys->createMaterial(0.2f + P(rng), 0.1f + 0.5f * P(rng), 0.6f * P(rng));
    mt->setFrictionCombineMode(PxCombineMode::Enum(m % 4));
    mt->setRestitutionCombineMode(PxCombineMode::Enum((m + 1) % 4));
    if (m == 5) mt->setFlag(PxMaterialFlag::eDISABLE_STRONG_FRICTION, true);
    mats.push_back(mt);
  }
  std::vector<ec::MaterialData> ourMats(64);
  std::map<const PxsShapeCore*, int> shapeIndex;
  std::vector<OurShape> shapes;
  std::vector<PxRigidActor*> actors;

  auto addShape = [&](PxRigidActor* a, const PxGeometry& g, const ec::ShapeGeom& eg, const PxTransform& local, PxMaterial* m, bool isStatic) {
    PxShape* s = PxRigidActorExt::createExclusiveShape(*a, g, *m);
    s->setLocalPose(local);
    const float co = 0.01f + 0.03f * P(rng);
    s->setContactOffset(co);
    s->setRestOffset(0.0f);
    const PxsShapeCore& core = static_cast<NpShape*>(s)->getCore().getCore();
    OurShape o;
    o.geom = eg;
    o.ts.p.storeAny(g); o.ts.e = eg;
    o.material = core.mMaterialIndex;
    o.contactOffset = core.mContactOffset;
    o.isStatic = isStatic;
    o.actor = a;
    const PxTransform& t = core.getTransform();
    o.shape2Actor = cxt::toE(t);
    o.idtShape = core.mShapeCoreFlags.isSet(PxShapeCoreFlag::eIDT_TRANSFORM);
    ec::MaterialData& md = ourMats[core.mMaterialIndex];
    md.dynamicFriction = m->getDynamicFriction();
    md.staticFriction = m->getStaticFriction();
    md.restitution = m->getRestitution();
    md.damping = m->getDamping();
    md.flags = uint16_t(PxU16(m->getFlags()));
    md.fricCombineMode = uint8_t(m->getFrictionCombineMode());
    md.restCombineMode = uint8_t(m->getRestitutionCombineMode());
    md.dampingCombineMode = uint8_t(m->getDampingCombineMode());
    shapeIndex[&core] = int(shapes.size());
    shapes.push_back(o);
  };

  // 바닥 (정적 상자 + 정적 평면 조금 아래)
  {
    PxRigidStatic* g = phys->createRigidStatic(PxTransform(PxVec3(0, 0, -0.5f)));
    ec::ShapeGeom eg; eg.type = ec::eBOX; eg.box = ep::PxBoxGeometry(6.0f, 6.0f, 0.5f);
    addShape(g, PxBoxGeometry(6.0f, 6.0f, 0.5f), eg, PxTransform(PxIdentity), mats[0], true);
    scene->addActor(*g);
    actors.push_back(g);
    PxRigidStatic* pl = PxCreatePlane(*phys, PxPlane(0, 0, 1, 2.0f), *mats[1]);
    // PxCreatePlane 이 만든 모양을 우리 쪽에도
    PxShape* ps; pl->getShapes(&ps, 1);
    const PxsShapeCore& core = static_cast<NpShape*>(ps)->getCore().getCore();
    OurShape o; o.geom.type = ec::ePLANE; o.ts.p.storeAny(PxPlaneGeometry()); o.ts.e.type = ec::ePLANE; o.material = core.mMaterialIndex; o.contactOffset = core.mContactOffset; o.isStatic = true; o.actor = pl;
    o.shape2Actor = cxt::toE(core.getTransform()); o.idtShape = core.mShapeCoreFlags.isSet(PxShapeCoreFlag::eIDT_TRANSFORM);
    ec::MaterialData& md = ourMats[core.mMaterialIndex];
    md.dynamicFriction = mats[1]->getDynamicFriction(); md.staticFriction = mats[1]->getStaticFriction(); md.restitution = mats[1]->getRestitution();
    md.damping = mats[1]->getDamping(); md.flags = uint16_t(PxU16(mats[1]->getFlags()));
    md.fricCombineMode = uint8_t(mats[1]->getFrictionCombineMode()); md.restCombineMode = uint8_t(mats[1]->getRestitutionCombineMode());
    md.dampingCombineMode = uint8_t(mats[1]->getDampingCombineMode());
    shapeIndex[&core] = int(shapes.size());
    shapes.push_back(o);
    scene->addActor(*pl);
    actors.push_back(pl);
  }
  // 떨어뜨릴 몸체들 (모양 1~2 개씩)
  for (int i = 0; i < nb; ++i) {
    PxQuat q(U(rng), U(rng), U(rng), U(rng)); q.normalize();
    PxRigidDynamic* d = phys->createRigidDynamic(PxTransform(PxVec3(4.0f * U(rng), 4.0f * U(rng), 0.3f + 0.5f * float(i % 12) + P(rng)), q));
    const int ns = (i % 5 == 0) ? 2 : 1;
    for (int k = 0; k < ns; ++k) {
      cxt::TestShape ts;
      const int types[4] = {0, 2, 3, 5};
      cxt::makeShape(types[size_t(P(rng) * 4) % 4], rng, hulls, ts);
      PxQuat lq(U(rng), U(rng), U(rng), U(rng)); lq.normalize();
      const PxTransform local = (ns == 1 && P(rng) < 0.5f) ? PxTransform(PxIdentity) : PxTransform(PxVec3(0.1f * U(rng), 0.1f * U(rng), 0.1f * U(rng)), lq);
      addShape(d, ts.p.any(), ts.e, local, mats[size_t(P(rng) * mats.size()) % mats.size()], false);
    }
    PxRigidBodyExt::updateMassAndInertia(*d, 500.0f);
    d->setSleepThreshold(0.0f);  // 잠 끔
    d->setAngularVelocity(PxVec3(U(rng), U(rng), U(rng)));
    scene->addActor(*d);
    actors.push_back(d);
  }
  printf("몸체 %d, 모양 %zu, 재질 %zu\n", nb, shapes.size(), mats.size());
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  PxsContext* ctx = sc.getLowLevelContext();


  // 우리 쪽 쌍 상태: (모양 i, 모양 j) -> 다양체 + 이 쌍을 만든 PhysX 접촉 관리자 (바뀌면 새 쌍 = 다양체 새로)
  struct PairState {
    ec::ManifoldSlot slot;
    cxt::PhysxSlot pslot;  // 같은 쌍을 PhysX Gu 함수로 직접 (진단용)
    const PxsContactManager* cm = nullptr;
  };
  std::map<std::pair<int, int>, PairState> pairs;
  std::set<std::pair<int, int>> prevSeen, curSeen;
  static ep::PxContactBuffer buf;
  static ec::CompressedContacts<32, 256> out;
  ec::MaterialPair mp[256];

  cxt::Tally tPose("모양 월드 자세(변환 캐시)"), tCnt("패치·접촉 수·접촉 상태"), tPatch("패치 머리"), tPt("접촉점·분리");
  long long cmCount = 0, touching = 0, totalPts = 0;
  const float dt = 1.0f / 120.0f;
  for (int step = 0; step < steps; ++step) {
    // simulate 전 몸체 자세 기록 (좁은 단계는 이 자세로 돈다)
    std::map<const PxRigidActor*, std::pair<ep::PxTransform, std::pair<ep::PxTransform, bool>>> bodyPose;
    for (PxRigidActor* a : actors) {
      if (a->getType() == PxActorType::eRIGID_STATIC) {
        const PxsRigidCore& c = static_cast<NpRigidStatic*>(a)->getCore().getCore();
        bodyPose[a] = {cxt::toE(c.body2World), {ep::PxTransform(ep::PxIdentity), true}};
      } else {
        const PxsBodyCore& c = static_cast<NpRigidDynamic*>(a)->getCore().getCore();
        bodyPose[a] = {cxt::toE(c.body2World), {cxt::toE(c.getBody2Actor()), c.hasIdtBody2Actor()}};
      }
    }
    // 좁은 단계가 읽을 변환 캐시 (simulate 뒤에는 적분 후 값으로 바뀌므로 미리 떠 둔다)
    std::vector<PxTransform> cacheBefore;
    {
      PxsTransformCache& tc0 = ctx->getTransformCache();
      for (PxU32 k = 0; k < tc0.getTotalSize(); ++k) cacheBefore.push_back(tc0.getTransformCache(k).transform);
    }
    scene->simulate(dt);
    scene->fetchResults(true);

    prevSeen.swap(curSeen);
    curSeen.clear();
    PxsContactManagerOutputIterator outputs = ctx->getNphaseImplementationContext()->getContactManagerOutputs();
    const PxU32 nInter = sc.getNbInteractions(Sc::InteractionType::eOVERLAP);
    Sc::ElementSimInteraction** inter = sc.getInteractions(Sc::InteractionType::eOVERLAP);
    for (PxU32 ii = 0; ii < nInter; ++ii) {
      const Sc::ShapeInteraction* si = static_cast<const Sc::ShapeInteraction*>(inter[ii]);
      const PxsContactManager* cm = si->getContactManager();
      if (!cm) continue;
      const PxcNpWorkUnit& wu = cm->getWorkUnit();
      const PxsContactManagerOutput& po = outputs.getContactManagerOutput(wu.mNpIndex);
      const int a = shapeIndex.at(wu.getShapeCore0()), b = shapeIndex.at(wu.getShapeCore1());
      const OurShape& sa = shapes[a];
      const OurShape& sb = shapes[b];
      ++cmCount;
      // 모양 월드 자세 = PhysX 변환 캐시?
      const auto& pa = bodyPose.at(sa.actor);
      const auto& pb = bodyPose.at(sb.actor);
      const ep::PxTransform ta = shapeWorldPose(sa, pa.first, pa.second.first, pa.second.second);
      const ep::PxTransform tb = shapeWorldPose(sb, pb.first, pb.second.first, pb.second.second);
      const bool known = wu.mTransformCache0 < cacheBefore.size() && wu.mTransformCache1 < cacheBefore.size();
      const PxTransform ca = known ? cacheBefore[wu.mTransformCache0] : PxTransform(PxIdentity);
      const PxTransform cb = known ? cacheBefore[wu.mTransformCache1] : PxTransform(PxIdentity);
      const float xa[7] = {ta.p.x, ta.p.y, ta.p.z, ta.q.x, ta.q.y, ta.q.z, ta.q.w}, ya[7] = {ca.p.x, ca.p.y, ca.p.z, ca.q.x, ca.q.y, ca.q.z, ca.q.w};
      const float xb[7] = {tb.p.x, tb.p.y, tb.p.z, tb.q.x, tb.q.y, tb.q.z, tb.q.w}, yb[7] = {cb.p.x, cb.p.y, cb.p.z, cb.q.x, cb.q.y, cb.q.z, cb.q.w};
      if (known && step > 0)  // 첫 스텝은 캐시가 simulate 안에서 처음 채워짐
        for (int k = 0; k < 7; ++k) { tPose.f(xa[k], ya[k], cmCount, step); tPose.f(xb[k], yb[k], cmCount, step); }

      // 우리 좁은 단계 (PxcNpBatch::discreteNarrowPhase)
      // 새 쌍(지난 스텝에 없던 접촉 관리자) = PhysX 가 다양체를 새로 만듦 (PxsContext::createCache). 관리자 칸은 재사용되므로 포인터로 못 가림
      PairState& ps = pairs[{a, b}];
      curSeen.insert({a, b});
      if (ps.cm != cm || !prevSeen.count({a, b})) {
        const int t0 = sa.geom.type < sb.geom.type ? sa.geom.type : sb.geom.type, t1 = sa.geom.type < sb.geom.type ? sb.geom.type : sa.geom.type;
        ec::initManifold(ps.slot, t0, t1);
        ps.pslot.init(sa.geom.type, sb.geom.type);
        ps.cm = cm;
      }
      const float contactDist = sa.contactOffset + sb.contactOffset;  // PxcNpBatch.cpp:362 (mContactDistances = contactOffset)
      bool flipped;
      {
        cxt::FtzScope fz;
        ec::pcmPair(sa.geom, sb.geom, ep::PxTransform32(ta), ep::PxTransform32(tb), contactDist, 0.01f * tol.length, tol.length, ps.slot, buf, flipped);
        for (ep::PxU32 k = 0; k < buf.count; ++k) mp[k] = ec::MaterialPair{sa.material, sb.material};  // 모양-모양 재질 (뒤집기 되돌린 뒤)
        const int tmax = sa.geom.type > sb.geom.type ? sa.geom.type : sb.geom.type;
        ec::writeCompressedContact(buf.contacts, buf.count, mp, ourMats.data(), false, tmax > ec::eCONVEXMESH, out);
      }
      const uint8_t ourStatus = buf.count ? PxsContactManagerStatusFlag::eHAS_TOUCH : PxsContactManagerStatusFlag::eHAS_NO_TOUCH;
      tCnt.u(po.nbPatches, out.nbPatches, cmCount, step);
      tCnt.u(po.nbContacts, out.nbContacts, cmCount, step);
      tCnt.u(po.statusFlag & PxsContactManagerStatusFlag::eTOUCH_KNOWN, ourStatus, cmCount, step);
      if (buf.count) ++touching;
      totalPts += buf.count;
      const PxU32 np = PxMin<PxU32>(po.nbPatches, out.nbPatches);
      const PxContactPatch* pp = reinterpret_cast<const PxContactPatch*>(po.contactPatches);
      for (PxU32 k = 0; k < np; ++k) {
        const PxContactPatch& x = pp[k];
        const ec::ContactPatch& y = out.patches[k];
        tPatch.f(x.normal.x, y.normal.x, cmCount, step); tPatch.f(x.normal.y, y.normal.y, cmCount, step); tPatch.f(x.normal.z, y.normal.z, cmCount, step);
        tPatch.f(x.restitution, y.restitution, cmCount, step);
        tPatch.f(x.dynamicFriction, y.dynamicFriction, cmCount, step);
        tPatch.f(x.staticFriction, y.staticFriction, cmCount, step);
        tPatch.f(x.damping, y.damping, cmCount, step);
        tPatch.f(x.mMassModification.linear0, y.linear0, cmCount, step);
        tPatch.f(x.mMassModification.angular1, y.angular1, cmCount, step);
        tPatch.u(x.startContactIndex, y.startContactIndex, cmCount, step);
        tPatch.u(x.nbContacts, y.nbContacts, cmCount, step);
        tPatch.u(x.materialFlags, y.materialFlags, cmCount, step);
        tPatch.u(x.internalFlags, y.internalFlags, cmCount, step);
        tPatch.u(x.materialIndex0, y.materialIndex0, cmCount, step);
        tPatch.u(x.materialIndex1, y.materialIndex1, cmCount, step);
      }
      const PxU32 nc = PxMin<PxU32>(po.nbContacts, out.nbContacts);
      const PxContact* pc = reinterpret_cast<const PxContact*>(po.contactPoints);
      static PxContactBuffer dbuf;
      {
        cxt::FtzScope fz;
        cxt::physxPcmPair(sa.ts, sb.ts, ca, cb, contactDist, ps.pslot, dbuf);
      }
      if (getenv("NP_DEBUG")) {
        bool diff = false;
        for (PxU32 k = 0; k < nc; ++k)
          diff |= memcmp(&pc[k], &out.contacts[k], 16) != 0;
        static int shown = 0;
        if (diff && shown++ < 5)
          printf("  [다름] 스텝 %d 쌍(%d:%d, %d:%d) 접촉 %u 패치 %u flip %d 상태 %u cmFlags 0x%x | 파이프라인 %.9g 우리 %.9g Gu직접 %.9g (직접 %u개)\n",
                 step, a, sa.geom.type, b, sb.geom.type, po.nbContacts, po.nbPatches, int(flipped), unsigned(po.statusFlag), unsigned(wu.mFlags),
                 pc[0].separation, out.contacts[0].separation, dbuf.count ? dbuf.contacts[0].separation : -1.0f, dbuf.count);
      }
      for (PxU32 k = 0; k < nc; ++k) {
        tPt.f(pc[k].contact.x, out.contacts[k].contact.x, cmCount, step);
        tPt.f(pc[k].contact.y, out.contacts[k].contact.y, cmCount, step);
        tPt.f(pc[k].contact.z, out.contacts[k].contact.z, cmCount, step);
        tPt.f(pc[k].separation, out.contacts[k].separation, cmCount, step);
      }
    }
  }
  printf("스텝 %d, 접촉 관리자·스텝 %lld (닿음 %lld), 접촉점 %lld\n", steps, cmCount, touching, totalPts);
  uint64_t bad = 0;
  for (const cxt::Tally* t : {&tPose, &tCnt, &tPatch, &tPt}) { t->print(); bad += t->bad; }
  printf(bad ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return bad ? 1 : 0;
}
