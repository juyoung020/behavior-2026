// 층 1 시험: 손으로 짠 TGS 풀이(접촉 준비·분할·위치/속도 반복·마찰 패치·잠 판정) = PhysX 5.6.1 비트 동일?
// 방법: PhysX 를 작업 스레드 하나짜리 디스패처로 돌리고, "UpdateContinuationTask"(DyTGSDynamics.cpp:514, 풀이 묶음을 만드는
// updatePostKinematic 직전) 를 실행하기 직전에 내부 상태를 스냅샷한다:
//   활성 섬 순서, 섬마다 몸체 사슬·접촉 간선 사슬, 접촉 관리자 출력(패치·점), 작업 단위 값, 활성화된 간선, 섬 정적 닿음 수.
// 우리 엔진은 같은 입력(섬 순서 + 접촉)으로 자기 몸체 상태·자기 마찰 패치 상태를 써서 풀고, fetchResults 뒤 PhysX 와 비트 비교한다.
// 섬 관리(섬 순서 만들기)와 접촉 생성은 이 시험 범위 밖(각각 islands 시험·contact 모듈).
//   test_contact_solver [--scene boxes|pile] [--n N] [--steps S] [--seed K] [--stab 0|1] [--verbose 0|1]
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>

#include "px_internal.h"
#include "core/solver/tgs_solver.h"
#include "solver_stream.h"

using namespace physx;
namespace sv = eng::sv;

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

// ---------------- 스냅샷
struct SnapCM {
  const void* key;
  int body0, body1;  // body1 = -1 이면 정적
  PxTransform static1;
  uint16_t flags;
  float restDistance, torsional, minTorsional, offsetSlop;
  std::vector<sv::ContactPatchIn> patches;
  std::vector<sv::ContactIn> contacts;
  uint32_t pxFrictionCount;
  std::vector<sv::FrictionPatch> pxFriction;
  bool bad;
};
struct SnapIsland {
  std::vector<int> bodies;
  std::vector<int> cms;
  uint32_t staticTouch;
};
struct Snapshot {
  bool valid = false;
  std::vector<SnapIsland> islands;
  std::vector<SnapCM> cms;
  std::vector<const void*> activated;
  std::vector<uint32_t> numCounted;  // 몸체별 PxsBodyCore::numCountedInteractions (Sc 층 입력)
  std::vector<float> wakeCounter;    // 몸체별 풀이 직전 깸 카운터 (Sc 층 internalWakeUp 이 올린 값 포함)
  float bounce = 0, frictionOffset = 0, correlation = 0;
  uint32_t batchSize = 0, articBatchSize = 0, kinematics = 0;
  int errors = 0;
};

static std::unordered_map<const PxsBodyCore*, int> gCoreToBody;
static NpScene* gNpScene = nullptr;
static Snapshot gSnap;

static eng::V3 toV(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }

static void takeSnapshot() {
  Snapshot& S = gSnap;
  S = Snapshot();
  Sc::Scene& sc = gNpScene->getScScene();
  IG::SimpleIslandManager& im = *sc.getSimpleIslandManager();
  const IG::IslandSim& is = im.getAccurateIslandSim();
  Dy::DynamicsContextBase* dy = static_cast<Dy::DynamicsContextBase*>(sc.getDynamicsContext());
  S.bounce = dy->getBounceThreshold();
  S.frictionOffset = dy->getFrictionOffsetThreshold();
  S.correlation = dy->getCorrelationDistance();
  S.batchSize = dy->getSolverBatchSize();
  S.articBatchSize = dy->getSolverArticBatchSize();
  S.kinematics = is.getNbActiveKinematics();
  PxsContactManagerOutputIterator& outs = dy->mOutputIterator;
  std::unordered_map<const void*, int> cmIndex;
  const PxU32 nbIslands = is.getNbActiveIslands();
  const IG::IslandId* ids = is.getActiveIslands();
  for (PxU32 i = 0; i < nbIslands; ++i) {
    const IG::Island& island = is.getIsland(ids[i]);
    SnapIsland si;
    si.staticTouch = is.mIslandStaticTouchCount[ids[i]];
    PxNodeIndex cur = island.mRootNode;
    while (cur.isValid()) {
      const IG::Node& node = is.getNode(cur);
      if (node.getNodeType() != IG::Node::eRIGID_BODY_TYPE) {
        S.errors++;
      } else {
        PxsRigidBody* rb = reinterpret_cast<PxsRigidBody*>(node.mObject);
        auto it = gCoreToBody.find(&rb->getCore());
        if (it == gCoreToBody.end()) S.errors++;
        else si.bodies.push_back(it->second);
      }
      cur = node.mNextNode;
    }
    IG::EdgeIndex e = island.mFirstEdge[IG::Edge::eCONTACT_MANAGER];
    while (e != IG_INVALID_EDGE) {
      const IG::Edge& edge = is.getEdge(e);
      PxsContactManager* cm = im.getContactManager(e);
      if (cm) {
        SnapCM c{};
        c.key = cm;
        const PxNodeIndex n1 = is.mCpuData.getNodeIndex1(e);
        const PxNodeIndex n2 = is.mCpuData.getNodeIndex2(e);
        PxsRigidBody* rb0 = reinterpret_cast<PxsRigidBody*>(is.getNode(n1).mObject);
        auto it0 = gCoreToBody.find(&rb0->getCore());
        c.body0 = it0 == gCoreToBody.end() ? -1 : it0->second;
        if (c.body0 < 0) c.bad = true;
        const PxcNpWorkUnit& u = cm->getWorkUnit();
        if (static_cast<const void*>(u.mRigidCore0) != static_cast<const void*>(&rb0->getCore())) c.bad = true;  // node1 <-> core0
        if (n2.isStaticBody()) {
          c.body1 = -1;
          c.static1 = u.mRigidCore1->body2World;
        } else {
          PxsRigidBody* rb1 = reinterpret_cast<PxsRigidBody*>(is.getNode(n2).mObject);
          auto it1 = gCoreToBody.find(&rb1->getCore());
          c.body1 = it1 == gCoreToBody.end() ? -1 : it1->second;
          if (c.body1 < 0 || is.getNode(n2).isKinematic()) c.bad = true;
        }
        c.flags = u.mFlags;
        c.restDistance = u.mRestDistance;
        c.torsional = u.mTorsionalPatchRadius;
        c.minTorsional = u.mMinTorsionalPatchRadius;
        c.offsetSlop = u.mOffsetSlop;
        const PxsContactManagerOutput& o = outs.getContactManagerOutput(u.mNpIndex);
        const PxContactPatch* pp = reinterpret_cast<const PxContactPatch*>(o.contactPatches);
        const PxContact* pc = reinterpret_cast<const PxContact*>(o.contactPoints);
        if (o.nbPatches && (pp[0].internalFlags & (PxContactPatch::eMODIFIABLE | PxContactPatch::eCOMPRESSED_MODIFIED_CONTACT))) c.bad = true;
        for (PxU32 p = 0; p < o.nbPatches; ++p) {
          sv::ContactPatchIn q;
          q.invMassScale[0] = pp[p].mMassModification.linear0;
          q.invMassScale[1] = pp[p].mMassModification.angular0;
          q.invMassScale[2] = pp[p].mMassModification.linear1;
          q.invMassScale[3] = pp[p].mMassModification.angular1;
          q.normal = toV(pp[p].normal);
          q.restitution = pp[p].restitution;
          q.dynamicFriction = pp[p].dynamicFriction;
          q.staticFriction = pp[p].staticFriction;
          q.damping = pp[p].damping;
          q.startContactIndex = pp[p].startContactIndex;
          q.nbContacts = pp[p].nbContacts;
          q.materialFlags = pp[p].materialFlags;
          q.internalFlags = pp[p].internalFlags;
          q.materialIndex0 = pp[p].materialIndex0;
          q.materialIndex1 = pp[p].materialIndex1;
          c.patches.push_back(q);
        }
        for (PxU32 k = 0; k < o.nbContacts; ++k) c.contacts.push_back(sv::ContactIn{toV(pc[k].contact), pc[k].separation});
        c.pxFrictionCount = u.mFrictionPatchCount;
        const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
        for (PxU32 k = 0; fp && k < u.mFrictionPatchCount; ++k) {
          sv::FrictionPatch f;
          f.broken = fp[k].broken;
          f.materialFlags = fp[k].materialFlags;
          f.anchorCount = fp[k].anchorCount;
          f.restitution = fp[k].restitution;
          f.staticFriction = fp[k].staticFriction;
          f.dynamicFriction = fp[k].dynamicFriction;
          f.body0Normal = toV(fp[k].body0Normal);
          f.body1Normal = toV(fp[k].body1Normal);
          for (int a = 0; a < 2; ++a) {
            f.body0Anchors[a] = toV(fp[k].body0Anchors[a]);
            f.body1Anchors[a] = toV(fp[k].body1Anchors[a]);
          }
          f.relativeQuat = eng::Q{fp[k].relativeQuat.x, fp[k].relativeQuat.y, fp[k].relativeQuat.z, fp[k].relativeQuat.w};
          c.pxFriction.push_back(f);
        }
        if (c.bad) S.errors++;
        cmIndex[cm] = int(S.cms.size());
        si.cms.push_back(int(S.cms.size()));
        S.cms.push_back(std::move(c));
      }
      e = edge.mNextIslandEdge;
    }
    S.islands.push_back(std::move(si));
  }
  const PxU32 nbAct = is.getNbActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  const IG::EdgeIndex* act = is.getActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  for (PxU32 a = 0; a < nbAct; ++a) {
    PxsContactManager* cm = im.getContactManager(act[a]);
    if (cm) S.activated.push_back(cm);
  }
  S.numCounted.assign(gCoreToBody.size(), 0);
  S.wakeCounter.assign(gCoreToBody.size(), 0.0f);
  for (auto& kv : gCoreToBody) {
    S.numCounted[kv.second] = kv.first->numCountedInteractions;
    S.wakeCounter[kv.second] = kv.first->wakeCounter;
  }
  S.valid = true;
}

// ---------------- 작업 스레드 하나짜리 디스패처 (UpdateContinuationTask 직전에 스냅샷)
class HookDispatcher : public PxCpuDispatcher {
 public:
  HookDispatcher() : mThread([this] { loop(); }) {}
  ~HookDispatcher() override {
    {
      std::lock_guard<std::mutex> l(mM);
      mQuit = true;
    }
    mCv.notify_all();
    mThread.join();
  }
  void submitTask(PxBaseTask& task) override {
    {
      std::lock_guard<std::mutex> l(mM);
      mQ.push_back(&task);
    }
    mCv.notify_one();
  }
  PxU32 getWorkerCount() const override { return 1; }

 private:
  void loop() {
    for (;;) {
      PxBaseTask* t = nullptr;
      {
        std::unique_lock<std::mutex> l(mM);
        mCv.wait(l, [this] { return mQuit || !mQ.empty(); });
        if (mQ.empty()) return;
        t = mQ.front();
        mQ.pop_front();
      }
      if (!strcmp(t->getName(), "UpdateContinuationTask")) takeSnapshot();
      t->run();
      t->release();
    }
  }
  std::mutex mM;
  std::condition_variable mCv;
  std::deque<PxBaseTask*> mQ;
  bool mQuit = false;
  std::thread mThread;
};

static PxDefaultAllocator gAlloc;
static PxDefaultErrorCallback gErr;

int main(int argc, char** argv) {
  std::string scene = "boxes";
  int n = 64, steps = 600, seed = 1, stab = 0, verbose = 1, trace = -1, t0 = 0, t1 = 0;
  const char* dumpPath = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--scene") && i + 1 < argc) scene = argv[++i];
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stab") && i + 1 < argc) stab = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--verbose") && i + 1 < argc) verbose = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpPath = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 3 < argc) { trace = atoi(argv[++i]); t0 = atoi(argv[++i]); t1 = atoi(argv[++i]); }
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  HookDispatcher disp;
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = &disp;
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  if (stab) sd.flags |= PxSceneFlag::eENABLE_STABILIZATION;
  PxScene* pscene = phys->createScene(sd);
  gNpScene = static_cast<NpScene*>(pscene);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  std::vector<PxMaterial*> mats = {phys->createMaterial(0.5f, 0.5f, 0.0f), phys->createMaterial(0.9f, 0.6f, 0.0f),
                                   phys->createMaterial(0.2f, 0.1f, 0.3f), phys->createMaterial(0.6f, 0.4f, 0.6f)};
  PxRigidStatic* ground = PxCreatePlane(*phys, PxPlane(0, 0, 1, 0), *mats[0]);
  pscene->addActor(*ground);

  eng::SceneParams sp;
  sp.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  sp.speedScale = 10.0f;
  sp.stabilization = stab != 0;
  std::vector<PxRigidDynamic*> px;
  std::vector<eng::Body> eb;
  auto addBody = [&](const PxTransform& pose, const PxGeometry& g, PxMaterial* m, float density, const PxVec3& lv, const PxVec3& av) {
    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    PxRigidActorExt::createExclusiveShape(*a, g, *m);
    PxRigidBodyExt::updateMassAndInertia(*a, density);
    pscene->addActor(*a);
    a->setLinearVelocity(lv);
    a->setAngularVelocity(av);
    px.push_back(a);
    eng::Body b = eng::createRigidDynamic(toE(pose), sp);
    eng::setCMassLocalPose(b, toE(a->getCMassLocalPose()));
    eng::setMass(b, a->getMass());
    eng::setMassSpaceInertiaTensor(b, toV(a->getMassSpaceInertiaTensor()));
    b.linVel = toV(lv);
    b.angVel = toV(av);
    eb.push_back(b);
  };
  auto rq = [&]() {
    double a = U(rng), b = U(rng), c = U(rng), d = U(rng);
    const double nn = std::sqrt(a * a + b * b + c * c + d * d);
    return PxQuat(float(a / nn), float(b / nn), float(c / nn), float(d / nn));
  };
  if (scene == "boxes") {
    const int side = int(std::ceil(std::sqrt(double(n))));
    for (int i = 0; i < n; ++i) {
      const float x = float(i % side) * 1.6f, y = float(i / side) * 1.6f;
      const int kind = i % 5;
      PxMaterial* m = mats[i % mats.size()];
      if (kind == 3) {
        addBody(PxTransform(PxVec3(x, y, 0.35f + 0.3f * P(rng)), rq()), PxSphereGeometry(0.1f + 0.2f * P(rng)), m, 800.0f,
                PxVec3(U(rng), U(rng), 0.0f), PxVec3(3 * U(rng), 3 * U(rng), 3 * U(rng)));
      } else if (kind == 4) {
        addBody(PxTransform(PxVec3(x, y, 0.4f + 0.3f * P(rng)), rq()), PxCapsuleGeometry(0.08f + 0.1f * P(rng), 0.1f + 0.2f * P(rng)), m, 600.0f,
                PxVec3(U(rng), U(rng), 0.0f), PxVec3(2 * U(rng), 2 * U(rng), 2 * U(rng)));
      } else {
        const int stack = 1 + kind;  // 1~3 층
        float z = 0.0f;
        for (int s = 0; s < stack; ++s) {
          const PxVec3 he(0.1f + 0.2f * P(rng), 0.1f + 0.2f * P(rng), 0.08f + 0.12f * P(rng));
          z += he.z + 0.002f;
          const float yaw = 3.14159f * U(rng);
          PxQuat q(yaw, PxVec3(0, 0, 1));
          if (s == 0 && (i % 7) == 0) q = rq();  // 몇 개는 기울어진 채 떨어짐
          addBody(PxTransform(PxVec3(x + 0.02f * U(rng), y + 0.02f * U(rng), z + (s == 0 ? 0.05f * P(rng) : 0.0f)), q), PxBoxGeometry(he), m,
                  500.0f + 500.0f * P(rng), PxVec3(0.3f * U(rng), 0.3f * U(rng), 0.0f), PxVec3(0.5f * U(rng), 0.5f * U(rng), 0.5f * U(rng)));
          z += he.z + 0.002f;
        }
      }
    }
  } else {  // pile: 볼록 더미를 한 곳에 떨어뜨림
    PxConvexMeshDesc cd;
    std::vector<PxVec3> verts;
    for (int k = 0; k < 16; ++k) verts.push_back(PxVec3(0.15f * U(rng), 0.15f * U(rng), 0.15f * U(rng)));
    cd.points.count = PxU32(verts.size());
    cd.points.stride = sizeof(PxVec3);
    cd.points.data = verts.data();
    cd.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxCookingParams cp(tol);
    PxConvexMesh* cm = PxCreateConvexMesh(cp, cd, phys->getPhysicsInsertionCallback());
    for (int i = 0; i < n; ++i) {
      const PxVec3 pos(0.4f * U(rng), 0.4f * U(rng), 0.3f + 0.25f * float(i));
      addBody(PxTransform(pos, rq()), PxConvexMeshGeometry(cm), mats[i % mats.size()], 700.0f, PxVec3(0), PxVec3(0));
    }
  }
  const int nb = int(eb.size());
  for (int i = 0; i < nb; ++i) gCoreToBody[&static_cast<NpRigidDynamic*>(px[i])->getCore().getCore()] = i;

  // ---- 우리 엔진 판
  const uint32_t CM_CAP = 1u << 16, DESC_CAP = 1u << 16, POOL_CAP = uint32_t(nb) + 64;
  std::vector<sv::SolverCM> cms;
  cms.reserve(CM_CAP);
  std::unordered_map<const void*, uint32_t> cmMap;
  std::vector<sv::SBodyVel> vels(POOL_CAP);
  std::vector<sv::SBodyTxI> txis(POOL_CAP);
  std::vector<sv::SBodyData> datas(POOL_CAP);
  std::vector<sv::SDesc> descs(DESC_CAP), ordered(DESC_CAP), temp(DESC_CAP);
  std::vector<sv::BatchHeader> headers(DESC_CAP);
  std::vector<uint32_t> partCounts(4096), bodySolverIndex(nb);
  std::vector<uint8_t> arenaMem(64u << 20);
  std::vector<sv::FrictionPatch> fr0(1u << 18), fr1(1u << 18);
  std::unique_ptr<sv::CorrelationBuffer> corr(new sv::CorrelationBuffer());
  std::vector<sv::ContactPoint> cbuf(sv::MAX_CONTACTS);
  sv::SolverBoard B{};
  B.bodies = eb.data();
  B.nbBodies = nb;
  B.vels = vels.data();
  B.txI = txis.data();
  B.datas = datas.data();
  B.poolCap = POOL_CAP;
  B.descs = descs.data();
  B.ordered = ordered.data();
  B.temp = temp.data();
  B.headers = headers.data();
  B.descCap = DESC_CAP;
  B.partitionCounts = partCounts.data();
  B.partitionCap = uint32_t(partCounts.size());
  B.bodySolverIndex = bodySolverIndex.data();
  B.constraints = sv::ByteArena{arenaMem.data(), 0, uint32_t(arenaMem.size()), 0};
  B.friction[0] = sv::FrictionArena{fr0.data(), 0, uint32_t(fr0.size()), 0};
  B.friction[1] = sv::FrictionArena{fr1.data(), 0, uint32_t(fr1.size()), 0};
  B.frictionCurIdx = 0;
  B.corr = corr.get();
  B.contactBuffer = cbuf.data();

  const float dt = 1.0f / 120.0f;
  uint64_t cmp = 0, bad[4] = {0, 0, 0, 0};
  int64_t first[4] = {-1, -1, -1, -1};
  double maxd[4] = {0, 0, 0, 0};
  const char* names[4] = {"행위자 자세", "선속도", "각속도", "깸 카운터"};
  uint64_t fricCmp = 0, fricBad = 0, snapErr = 0, wakeEvents = 0, wakeBad = 0;
  int64_t fricFirst = -1;
  uint64_t totContacts = 0, totCMs = 0, maxIslands = 0;
  int shown = 0;
  std::vector<uint8_t> wasActive(nb, 1), isActive(nb, 0);
  std::vector<uint8_t> dumpBody;  // 스텝 블록들 (--dump)
  const std::vector<eng::Body> bodies0 = eb;
  for (int s = 1; s <= steps; ++s) {
    gSnap.valid = false;
    pscene->simulate(dt);
    pscene->fetchResults(true);
    Snapshot& S = gSnap;
    std::fill(isActive.begin(), isActive.end(), 0);
    std::vector<sv::IslandIn> islands;
    std::vector<uint32_t> ib, icm, act;
    std::vector<sv::ContactPatchIn> patches;
    std::vector<sv::ContactIn> contacts;
    std::vector<sv::SolverCM> cmIn;
    std::vector<svs::Wake> wakes;
    std::vector<uint32_t> numCounted(nb, 0);
    for (int i = 0; i < nb; ++i) numCounted[i] = eb[i].numCountedInteractions;
    if (S.valid) {
      snapErr += S.errors;
      for (auto& si : S.islands) {
        sv::IslandIn I{uint32_t(ib.size()), uint32_t(si.bodies.size()), uint32_t(icm.size()), uint32_t(si.cms.size()), si.staticTouch};
        for (int b : si.bodies) {
          ib.push_back(uint32_t(b));
          isActive[b] = 1;
        }
        for (int ci : si.cms) {
          const SnapCM& c = S.cms[ci];
          auto it = cmMap.find(c.key);
          uint32_t idx;
          if (it == cmMap.end()) {
            idx = uint32_t(cms.size());
            cmMap[c.key] = idx;
            sv::SolverCM z{};
            z.frictionPtr = sv::NONE;
            cms.push_back(z);
          } else
            idx = it->second;
          sv::SolverCM& m = cms[idx];
          m.body0 = uint32_t(c.body0);
          m.body1 = c.body1 < 0 ? sv::NONE : uint32_t(c.body1);
          m.staticPose1 = toE(c.static1);
          m.npFlags = c.flags;
          m.restDistance = c.restDistance;
          m.torsionalPatchRadius = c.torsional;
          m.minTorsionalPatchRadius = c.minTorsional;
          m.offsetSlop = c.offsetSlop;
          m.patchStart = uint32_t(patches.size());
          m.nbPatches = uint32_t(c.patches.size());
          m.contactStart = uint32_t(contacts.size());
          m.nbContacts = uint32_t(c.contacts.size());
          patches.insert(patches.end(), c.patches.begin(), c.patches.end());
          contacts.insert(contacts.end(), c.contacts.begin(), c.contacts.end());
          icm.push_back(idx);
          cmIn.push_back(m);
          totContacts += c.contacts.size();
        }
        islands.push_back(I);
      }
      for (const void* k : S.activated) {
        auto it = cmMap.find(k);
        if (it != cmMap.end()) act.push_back(it->second);
      }
      totCMs += icm.size();
      maxIslands = std::max<uint64_t>(maxIslands, islands.size());
      for (int i = 0; i < nb; ++i) {
        numCounted[i] = S.numCounted[i];
        // Sc 층 깨움(internalWakeUpBase: 깸 카운터를 올리기만 함, ScBodySim.cpp:541) — 닿음 잃음·새 닿음 이벤트가 만든 입력
        if (S.wakeCounter[i] != eb[i].wakeCounter) {
          if (S.wakeCounter[i] > eb[i].wakeCounter) {
            wakes.push_back(svs::Wake{uint32_t(i), S.wakeCounter[i]});
            wakeEvents++;
          } else
            wakeBad++;
        }
      }
    }
    B.islands = islands.data();
    B.nbIslands = uint32_t(islands.size());
    B.islandBodies = ib.data();
    B.islandCMs = icm.data();
    B.activatedCMs = act.data();
    B.nbActivatedCMs = uint32_t(act.size());
    B.cms = cms.data();
    B.nbCMs = uint32_t(cms.size());
    B.patches = patches.data();
    B.contacts = contacts.data();
    // 지난 스텝 마찰 패치 대조 (우리 상태 vs PhysX 상태, 풀기 전)
    if (S.valid) {
      const sv::FrictionArena& prevArena = B.friction[B.frictionCurIdx];  // 이번 스텝에서 prev 가 될 쪽
      std::vector<uint8_t> activated(cms.size(), 0);
      for (uint32_t a : act) activated[a] = 1;
      for (auto& si : S.islands)
        for (int ci : si.cms) {
          const SnapCM& c = S.cms[ci];
          const sv::SolverCM& m = cms[cmMap[c.key]];
          const uint32_t myCount = activated[cmMap[c.key]] ? 0u : m.frictionCount;
          fricCmp++;
          bool same = myCount == c.pxFrictionCount;
          for (uint32_t k = 0; same && k < myCount; ++k) {
            const sv::FrictionPatch& a = prevArena.data[m.frictionPtr + k];
            const sv::FrictionPatch& b = c.pxFriction[k];
            same = a.broken == b.broken && a.materialFlags == b.materialFlags && a.anchorCount == b.anchorCount &&
                   !memcmp(&a.restitution, &b.restitution, sizeof(float) * 3) && !memcmp(&a.body0Normal, &b.body0Normal, sizeof(float) * 6) &&
                   !memcmp(&a.relativeQuat, &b.relativeQuat, sizeof(float) * 4);
            for (uint32_t an = 0; same && an < a.anchorCount; ++an)
              same = !memcmp(&a.body0Anchors[an], &b.body0Anchors[an], 12) && !memcmp(&a.body1Anchors[an], &b.body1Anchors[an], 12);
          }
          if (!same) {
            fricBad++;
            if (fricFirst < 0) fricFirst = s;
          }
        }
    }
    sv::SolverParams prm;
    prm.gravity = eng::V3{0.0f, 0.0f, -9.81f};
    prm.dt = dt;
    prm.enableStabilization = stab != 0;
    prm.bounceThreshold = S.bounce;
    prm.frictionOffsetThreshold = S.frictionOffset;
    prm.correlationDistance = S.correlation;
    prm.solverBatchSize = S.batchSize ? S.batchSize : 128;
    prm.solverArticBatchSize = S.articBatchSize ? S.articBatchSize : 16;
    // 이번 스텝 섬 관리자가 재운 몸체 (fetchResults 뒤에도 accurate IslandSim 에 남아 있다, 다음 스텝 3차 섬 생성에서 지움)
    std::vector<uint32_t> deact;
    {
      const IG::IslandSim& is = gNpScene->getScScene().getSimpleIslandManager()->getAccurateIslandSim();
      const PxU32 nd = is.getNbNodesToDeactivate(IG::Node::eRIGID_BODY_TYPE);
      const PxNodeIndex* di = is.getNodesToDeactivate(IG::Node::eRIGID_BODY_TYPE);
      for (PxU32 k = 0; k < nd; ++k) {
        PxsRigidBody* rb = reinterpret_cast<PxsRigidBody*>(is.getNode(di[k]).mObject);
        auto it = gCoreToBody.find(&rb->getCore());
        if (it != gCoreToBody.end()) deact.push_back(uint32_t(it->second));
        else snapErr++;
      }
    }
    svs::StepView v;
    v.c = svs::StepCounts{uint32_t(islands.size()), uint32_t(ib.size()), uint32_t(icm.size()), uint32_t(act.size()), uint32_t(deact.size()),
                          uint32_t(wakes.size()), uint32_t(patches.size()), uint32_t(contacts.size())};
    v.islands = islands.data();
    v.ib = ib.data();
    v.icm = icm.data();
    v.cmIn = cmIn.data();
    v.act = act.data();
    v.deact = deact.data();
    v.wake = wakes.data();
    v.numCounted = numCounted.data();
    v.patches = patches.data();
    v.contacts = contacts.data();
    B.cms = cms.data();
    B.nbCMs = uint32_t(cms.size());
    {
      FtzScope f;
      svs::runStep(B, prm, v);
    }
    if (dumpPath) {
      auto put = [&](const void* p, size_t n) { const uint8_t* b = static_cast<const uint8_t*>(p); dumpBody.insert(dumpBody.end(), b, b + n); };
      put(&v.c, sizeof(v.c));
      put(islands.data(), islands.size() * sizeof(sv::IslandIn));
      put(ib.data(), ib.size() * 4);
      put(icm.data(), icm.size() * 4);
      put(cmIn.data(), cmIn.size() * sizeof(sv::SolverCM));
      put(act.data(), act.size() * 4);
      put(deact.data(), deact.size() * 4);
      put(wakes.data(), wakes.size() * sizeof(svs::Wake));
      put(numCounted.data(), size_t(nb) * 4);
      put(patches.data(), patches.size() * sizeof(sv::ContactPatchIn));
      put(contacts.data(), contacts.size() * sizeof(sv::ContactIn));
      for (int i = 0; i < nb; ++i) {  // PhysX 결과
        const PxTransform tp = px[i]->getGlobalPose();
        const PxVec3 lp = px[i]->getLinearVelocity(), ap = px[i]->getAngularVelocity();
        const float r[svs::RES_FLOATS] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lp.x, lp.y, lp.z, ap.x, ap.y, ap.z, px[i]->getWakeCounter()};
        put(r, sizeof(r));
      }
    }
    if (B.error && shown < 20) {
      printf("[엔진 오류] step %d error=0x%x\n", s, B.error);
      shown++;
    }
    wasActive = isActive;
    if (trace >= 0 && s >= t0 && s <= t1) {
      const PxTransform tp = px[trace]->getGlobalPose();
      const PxVec3 lp = px[trace]->getLinearVelocity();
      const eng::Tf te = eng::getGlobalPose(eb[trace]);
      printf("[추적] step %d 몸체 %d 활성 %d | PhysX p=(%.9g %.9g %.9g) v=(%.9g %.9g %.9g) wc=%.9g sleep=%d | 엔진 p=(%.9g %.9g %.9g) v=(%.9g %.9g %.9g) wc=%.9g flags=%u\n",
             s, trace, int(isActive[trace]), tp.p.x, tp.p.y, tp.p.z, lp.x, lp.y, lp.z, px[trace]->getWakeCounter(), int(px[trace]->isSleeping()),
             te.p.x, te.p.y, te.p.z, eb[trace].linVel.x, eb[trace].linVel.y, eb[trace].linVel.z, eb[trace].wakeCounter, unsigned(eb[trace].internalFlags));
    }
    for (int i = 0; i < nb; ++i) {
      const PxTransform tp = px[i]->getGlobalPose();
      const eng::Tf te = eng::getGlobalPose(eb[i]);
      const PxVec3 lp = px[i]->getLinearVelocity(), ap = px[i]->getAngularVelocity();
      const float wp = px[i]->getWakeCounter();
      const void* pa[4] = {&tp, &lp, &ap, &wp};
      const void* pe[4] = {&te, &eb[i].linVel, &eb[i].angVel, &eb[i].wakeCounter};
      const int nf[4] = {7, 3, 3, 1};
      for (int k = 0; k < 4; ++k) {
        cmp++;
        if (memcmp(pa[k], pe[k], 4 * nf[k])) {
          bad[k]++;
          if (first[k] < 0) first[k] = s;
          const float* x = static_cast<const float*>(pa[k]);
          const float* y = static_cast<const float*>(pe[k]);
          for (int j = 0; j < nf[k]; ++j) maxd[k] = std::fmax(maxd[k], std::fabs(double(x[j]) - double(y[j])));
          if (verbose && shown < 8) {
            shown++;
            printf("[다름] step %d 몸체 %d %s (활성 %d)\n  PhysX:", s, i, names[k], int(isActive[i]));
            for (int j = 0; j < nf[k]; ++j) printf(" %.9g", x[j]);
            printf("\n  엔진 :");
            for (int j = 0; j < nf[k]; ++j) printf(" %.9g", y[j]);
            printf("\n");
          }
          // 이어서 비교하려면 PhysX 값으로 맞춘다 (첫 다름 이후 누적 차이를 막음)
        }
      }
    }
  }
  printf("\n장면 %s: 몸체 %d 개 x %d 스텝 (안정화 %s), 접촉 관리자 누적 %" PRIu64 ", 접촉점 누적 %" PRIu64 ", 최대 활성 섬 %" PRIu64 "\n",
         scene.c_str(), nb, steps, stab ? "켬" : "끔", totCMs, totContacts, maxIslands);
  printf("몸체 비교 %" PRIu64 "\n", cmp);
  for (int k = 0; k < 4; ++k)
    printf("  %-10s 비트 다름 %8" PRIu64 "  첫 다름 스텝 %5" PRId64 "  최대|차| %.3e\n", names[k], bad[k], first[k], maxd[k]);
  printf("마찰 패치 상태 비교 %" PRIu64 " (접촉 관리자·스텝), 다름 %" PRIu64 ", 첫 다름 스텝 %" PRId64 "\n", fricCmp, fricBad, fricFirst);
  printf("풀이 묶음 %" PRIu64 " (제약 없는 묶음 %" PRIu64 "), 묶음 머리 %" PRIu64 ", 4개 묶음 준비 성공 %" PRIu64 ", 단일 준비 %" PRIu64 ", 최대 분할 수 %" PRIu64 "\n",
         B.statBatches, B.statFreeBatches, B.statHeaders, B.statBlock4, B.statSingle, B.statMaxPartitions);
  printf("Sc 층 깨움 입력 %" PRIu64 " 회, 풀이 직전 우리 깸 카운터가 PhysX 보다 큼(오류) %" PRIu64 "\n", wakeEvents, wakeBad);
  printf("스냅샷 오류 %" PRIu64 ", 엔진 오류 0x%x\n", snapErr, B.error);
  printf("작업 공간 최대: 제약 자료 %u B/묶음, 마찰 패치 %u 개/스텝, 제약 %u 개/묶음, 접촉 관리자 %zu 개\n", B.statMaxArena, B.statMaxFriction,
         B.statMaxDescs, cms.size());
  if (dumpPath) {
    svs::Header h{};
    memcpy(h.magic, "SVSTRM1", 8);
    h.version = 1;
    h.nb = uint32_t(nb);
    h.steps = uint32_t(steps);
    h.stab = uint32_t(stab);
    h.gravity[0] = 0.0f; h.gravity[1] = 0.0f; h.gravity[2] = -9.81f;
    h.dt = dt;
    h.bounce = gSnap.bounce;
    h.frictionOffset = gSnap.frictionOffset;
    h.correlation = gSnap.correlation;
    h.batchSize = gSnap.batchSize;
    h.articBatchSize = gSnap.articBatchSize;
    h.maxCMs = uint32_t(cms.size());
    h.bodySize = sizeof(eng::Body);
    h.cmSize = sizeof(sv::SolverCM);
    h.patchSize = sizeof(sv::ContactPatchIn);
    h.contactSize = sizeof(sv::ContactIn);
    h.islandSize = sizeof(sv::IslandIn);
    h.maxArena = B.statMaxArena;
    h.maxFriction = B.statMaxFriction;
    h.maxDescs = B.statMaxDescs;
    FILE* f = fopen(dumpPath, "wb");
    fwrite(&h, sizeof(h), 1, f);
    fwrite(bodies0.data(), sizeof(eng::Body), bodies0.size(), f);
    fwrite(dumpBody.data(), 1, dumpBody.size(), f);
    fclose(f);
    printf("입력 흐름 저장: %s (%.1f MB)\n", dumpPath, double(dumpBody.size()) / 1e6);
  }
  const bool ok = !bad[0] && !bad[1] && !bad[2] && !bad[3] && !fricBad && !snapErr && !B.error && !wakeBad;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  pscene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
