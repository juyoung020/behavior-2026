// 층 1 시험: 관절체를 TGS 반복 루프에 붙인 손 짠 풀이 = PhysX 5.6.1 비트 동일?  (docs 17.6, 관절체 결합)
// 장면: 바닥(정적 평면) + R1Pro 모양 관절체(바퀴 3 개 = 구, 바닥에 닿음, 가상 바닥 관절로 서 있음) + 무작위 관절체(링크마다 상자, 바닥·서로·자기에 닿음)
//       + 떨어지는 강체 상자(링크 위로: 링크-강체 ext 접촉) + D6 조인트(링크-강체 고정 = 보조 잡기 꼴, 링크-세계, 링크-다른 관절체 링크, 강체-세계).
// 방법: test_contact_solver 와 같다. PhysX 를 작업 스레드 하나로 돌려 UpdateContinuationTask 직전에 섬 순서(강체·관절체 노드)·접촉 관리자 출력·
//       1D 제약을 뜨고, 우리 엔진은 자기 몸체·관절체·마찰 패치 상태로 같은 섬을 풀어 fetchResults 뒤 비교한다.
// 관절체 입력(드라이브 목표·applyCache·순간이동)은 articulation 시험의 Mirror/stepInputsEng 그대로(양쪽 같은 API).
// Sc 층 깨움(새 닿음 등으로 링크 깸 카운터를 올림)은 PhysX 에서 받아 넣는다(강체 시험과 같음).
//   test_art_solver [--arts N] [--links N] [--seed K] [--steps S] [--pos P] [--vel V] [--r1pro urdf] [--r1copies N] [--boxes N]
//                   [--joints 0|1] [--selfcol 0|1] [--stab 0|1] [--last 0|1] [--neg 0|1|2] [--verbose 0|1]
//   --neg: 음성 대조 1 = 스텝 40 에 마지막 관절체의 가장 빠른 관절 속도 1 ulp, 2 = 스텝 40 접촉점 하나 분리 1 ulp
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <unistd.h>
#include <unordered_set>

#include "px_internal.h"
#define private public
#define protected public
#include "NpArticulationLink.h"
#include "NpArticulationReducedCoordinate.h"
#include "ScArticulationSim.h"
#include "DyFeatherstoneArticulation.h"
#undef private
#undef protected
#include "core/solver/solver_io.h"
#include "core/articulation/art_static.h"
#include "tests/articulation/random_art.h"
#include "art_stream.h"

using namespace physx;
namespace sv = eng::sv;
namespace A = eng::art;

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

static Dy::FeatherstoneArticulation* llArticulation(PxArticulationReducedCoordinate* a) {
  return static_cast<NpArticulationReducedCoordinate*>(a)->getCore().getSim()->getLowLevelArticulation();
}
static eng::V3 toV(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
static eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }

// ---------------- 스냅샷
struct Ref {  // 몸체 참조: 강체(body) 또는 관절체 링크(art, link = LL), 정적이면 body = -1
  int body = -1, art = -1;
  uint32_t link = 0;
};
struct SnapCM {
  const void* key;
  Ref r0, r1;
  PxTransform static1;
  uint32_t flags;
  float restDistance, torsional, minTorsional, offsetSlop;
  std::vector<sv::ContactPatchIn> patches;
  std::vector<sv::ContactIn> contacts;
  uint32_t pxFrictionCount;
  std::vector<sv::FrictionPatch> pxFriction;
};
struct SnapC1D {
  Ref r0, r1;
  uint32_t index;
  uint16_t flags;
  float linBreak, angBreak, minResp;
  eng::jnt::D6Data data;
};
struct SnapIsland {
  std::vector<int> bodies, arts, cms, c1ds;
  uint32_t staticTouch;
};
struct Snapshot {
  bool valid = false;
  std::vector<SnapIsland> islands;
  std::vector<SnapCM> cms;
  std::vector<SnapC1D> c1d;
  std::vector<const void*> activated;
  std::vector<float> wakeCounter;                // 강체 풀이 직전 깸 카운터
  std::vector<std::vector<float>> linkWake;      // 관절체 링크(LL) 풀이 직전 깸 카운터
  std::vector<std::vector<uint32_t>> linkCounted;  // 링크 PxsBodyCore::numCountedInteractions (Sc 층: 접촉·조인트 수, 잠 판정 입력)
  std::vector<float> artWake;                    // 관절체 코어 깸 카운터
  std::vector<uint32_t> numCounted;
  float bounce = 0, frictionOffset = 0, correlation = 0;
  uint32_t batchSize = 0, articBatchSize = 0;
  int errors = 0;
};

static std::unordered_map<const PxsBodyCore*, int> gCoreToBody;
static std::unordered_map<const void*, int> gFaToArt;                      // Dy::FeatherstoneArticulation* -> 관절체 번호
static std::unordered_map<const PxsRigidBody*, std::pair<int, uint32_t>> gLinkRb;  // 링크 PxsRigidBody* -> (관절체, LL)
static std::unordered_map<const PxsBodyCore*, std::pair<int, uint32_t>> gLinkCore;
static std::vector<A::Articulation*> gEngArts;
static std::vector<const Dy::ArticulationCore*> gArtCore;
static NpScene* gNpScene = nullptr;
static Snapshot gSnap;
static std::unordered_set<const void*> gFreshCM;

using SIM = IG::SimpleIslandManager;
extern "C" {
PxU32 __real__ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
    SIM*, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
PxU32 __wrap__ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
    SIM* s, PxsContactManager* cm, PxNodeIndex n1, PxNodeIndex n2, Sc::Interaction* it, IG::Edge::EdgeType t) {
  if (cm) gFreshCM.insert(cm);
  return __real__ZN5physx2IG19SimpleIslandManager17addContactManagerEPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
      s, cm, n1, n2, it, t);
}
bool __real__ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
    SIM*, PxU32, PxsContactManager*, PxNodeIndex, PxNodeIndex, Sc::Interaction*, IG::Edge::EdgeType);
bool __wrap__ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
    SIM* s, PxU32 h, PxsContactManager* cm, PxNodeIndex n1, PxNodeIndex n2, Sc::Interaction* it, IG::Edge::EdgeType t) {
  if (cm) gFreshCM.insert(cm);
  return __real__ZN5physx2IG19SimpleIslandManager29addPreallocatedContactManagerEjPNS_17PxsContactManagerENS_11PxNodeIndexES4_PNS_2Sc11InteractionENS0_4Edge8EdgeTypeE(
      s, h, cm, n1, n2, it, t);
}
void __real__ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE(SIM*, PxU32, PxsContactManager*);
void __wrap__ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE(SIM* s, PxU32 e, PxsContactManager* cm) {
  if (cm) gFreshCM.insert(cm);
  __real__ZN5physx2IG19SimpleIslandManager14setEdgeRigidCMEjPNS_17PxsContactManagerE(s, e, cm);
}
}

// 섬 노드 -> Ref
static Ref refOfNode(const IG::IslandSim& is, PxNodeIndex n, int& err) {
  Ref r;
  if (n.isStaticBody()) return r;
  const IG::Node& node = is.getNode(n);
  if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
    auto it = gFaToArt.find(node.mObject);
    if (it == gFaToArt.end()) err++;
    else {
      r.art = it->second;
      r.link = n.articulationLinkId();
    }
  } else {
    if (node.isKinematic()) err++;
    PxsRigidBody* rb = reinterpret_cast<PxsRigidBody*>(node.mObject);
    auto it = gCoreToBody.find(&rb->getCore());
    if (it == gCoreToBody.end()) err++;
    else r.body = it->second;
  }
  return r;
}
static Ref refOfRb(const PxsRigidBody* rb, int& err) {
  Ref r;
  if (!rb) return r;
  auto il = gLinkRb.find(rb);
  if (il != gLinkRb.end()) {
    r.art = il->second.first;
    r.link = il->second.second;
    return r;
  }
  auto it = gCoreToBody.find(&rb->getCore());
  if (it == gCoreToBody.end()) err++;
  else r.body = it->second;
  return r;
}

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
  if (is.getNbActiveKinematics()) S.errors++;
  PxsContactManagerOutputIterator& outs = dy->mOutputIterator;
  const PxU32 nbIslands = is.getNbActiveIslands();
  const IG::IslandId* ids = is.getActiveIslands();
  for (PxU32 i = 0; i < nbIslands; ++i) {
    const IG::Island& island = is.getIsland(ids[i]);
    SnapIsland si;
    si.staticTouch = is.mIslandStaticTouchCount[ids[i]];
    PxNodeIndex cur = island.mRootNode;
    while (cur.isValid()) {
      const IG::Node& node = is.getNode(cur);
      Ref r = refOfNode(is, cur, S.errors);
      if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
        if (r.art >= 0) si.arts.push_back(r.art);
      } else if (r.body >= 0) {
        si.bodies.push_back(r.body);
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
        c.r0 = refOfNode(is, n1, S.errors);
        c.r1 = refOfNode(is, n2, S.errors);
        const PxcNpWorkUnit& u = cm->getWorkUnit();
        // node1 <-> core0 확인 (링크면 링크 코어)
        if (c.r0.art >= 0) {
          auto il = gLinkCore.find(static_cast<const PxsBodyCore*>(u.mRigidCore0));
          if (il == gLinkCore.end() || il->second.first != c.r0.art || il->second.second != c.r0.link) S.errors++;
        } else if (c.r0.body < 0 || static_cast<const void*>(u.mRigidCore0) == nullptr) {
          S.errors++;
        }
        if (n2.isStaticBody()) c.static1 = u.mRigidCore1->body2World;
        c.flags = u.mFlags;
        c.restDistance = u.mRestDistance;
        c.torsional = u.mTorsionalPatchRadius;
        c.minTorsional = u.mMinTorsionalPatchRadius;
        c.offsetSlop = u.mOffsetSlop;
        const PxsContactManagerOutput& o = outs.getContactManagerOutput(u.mNpIndex);
        const PxContactPatch* pp = reinterpret_cast<const PxContactPatch*>(o.contactPatches);
        const PxContact* pc = reinterpret_cast<const PxContact*>(o.contactPoints);
        if (o.nbPatches && (pp[0].internalFlags & (PxContactPatch::eMODIFIABLE | PxContactPatch::eCOMPRESSED_MODIFIED_CONTACT))) S.errors++;
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
        si.cms.push_back(int(S.cms.size()));
        S.cms.push_back(std::move(c));
      }
      e = edge.mNextIslandEdge;
    }
    IG::EdgeIndex ce = island.mFirstEdge[IG::Edge::eCONSTRAINT];
    while (ce != IG_INVALID_EDGE) {
      const IG::Edge& edge = is.getEdge(ce);
      const Dy::Constraint* c = im.getConstraint(ce);
      SnapC1D x{};
      if (!c) {
        S.errors++;
      } else {
        x.r0 = refOfRb(c->body0, S.errors);
        x.r1 = refOfRb(c->body1, S.errors);
        if (is.mCpuData.getNodeIndex1(ce).isStaticBody() != (c->body0 == nullptr)) S.errors++;
        if (is.mCpuData.getNodeIndex2(ce).isStaticBody() != (c->body1 == nullptr)) S.errors++;
        x.index = c->index;
        x.flags = c->flags;
        x.linBreak = c->linBreakForce;
        x.angBreak = c->angBreakForce;
        x.minResp = c->minResponseThreshold;
        if (c->constantBlockSize != sizeof(eng::jnt::D6Data)) S.errors++;
        else memcpy(&x.data, c->constantBlock, sizeof(eng::jnt::D6Data));
        si.c1ds.push_back(int(S.c1d.size()));
        S.c1d.push_back(x);
      }
      ce = edge.mNextIslandEdge;
    }
    S.islands.push_back(std::move(si));
  }
  const PxU32 nbAct = is.getNbActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  const IG::EdgeIndex* act = is.getActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  for (PxU32 a = 0; a < nbAct; ++a) {
    PxsContactManager* cm = im.getContactManager(act[a]);
    if (cm) S.activated.push_back(cm);
  }
  S.wakeCounter.assign(gCoreToBody.size(), 0.0f);
  S.numCounted.assign(gCoreToBody.size(), 0);
  for (auto& kv : gCoreToBody) {
    S.wakeCounter[kv.second] = kv.first->wakeCounter;
    S.numCounted[kv.second] = kv.first->numCountedInteractions;
  }
  S.linkWake.assign(gEngArts.size(), {});
  S.artWake.assign(gEngArts.size(), 0.0f);
  S.linkCounted.assign(gEngArts.size(), {});
  for (auto& kv : gLinkCore) {
    auto& v = S.linkWake[size_t(kv.second.first)];
    auto& c = S.linkCounted[size_t(kv.second.first)];
    if (v.size() <= kv.second.second) v.resize(kv.second.second + 1), c.resize(kv.second.second + 1);
    v[kv.second.second] = kv.first->wakeCounter;
    c[kv.second.second] = kv.first->numCountedInteractions;
  }
  for (size_t k = 0; k < gArtCore.size(); ++k) S.artWake[k] = gArtCore[k]->wakeCounter;
  S.valid = true;
}

// 관절체 잠 판정 직전(ScScene.afterIntegration 작업 시작) 링크·코어 깸 카운터와 링크 상호작용 수.
// PhysX 는 좁은 단계 뒤처리(새 닿음 -> 깨움)를 풀이와 나란히 돌리므로, 관절체 sleepCheck(afterIntegration) 에 들어가는 값은 이 시점 값이다.
struct LateArt {
  bool valid = false;
  std::vector<std::vector<float>> linkWake;
  std::vector<std::vector<uint32_t>> linkCounted;
  std::vector<float> artWake;
};
static LateArt gLate;
static void takeLateSnapshot() {
  LateArt& L = gLate;
  L.linkWake.assign(gEngArts.size(), {});
  L.linkCounted.assign(gEngArts.size(), {});
  L.artWake.assign(gEngArts.size(), 0.0f);
  for (auto& kv : gLinkCore) {
    auto& v = L.linkWake[size_t(kv.second.first)];
    auto& c = L.linkCounted[size_t(kv.second.first)];
    if (v.size() <= kv.second.second) v.resize(kv.second.second + 1), c.resize(kv.second.second + 1);
    v[kv.second.second] = kv.first->wakeCounter;
    c[kv.second.second] = kv.first->numCountedInteractions;
  }
  for (size_t k = 0; k < gArtCore.size(); ++k) L.artWake[k] = gArtCore[k]->wakeCounter;
  L.valid = true;
}

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
      if (!strcmp(t->getName(), "ScScene.afterIntegration")) takeLateSnapshot();
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
  artest::BuildOpts o;
  o.nArts = 6;
  o.nLinksMax = 7;
  o.posIt = 8;
  o.velIt = 1;
  int steps = 300, nBoxes = 12, joints = 1, selfcol = 0, stab = 0, last = 0, neg = 0, verbose = 1;
  const char* dumpPath = nullptr;
  int tiers = 0;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* k) { return !strcmp(argv[i], k) && i + 1 < argc; };
    if (arg("--arts")) o.nArts = atoi(argv[++i]);
    else if (arg("--links")) o.nLinksMax = atoi(argv[++i]);
    else if (arg("--seed")) o.seed = atoi(argv[++i]);
    else if (arg("--steps")) steps = atoi(argv[++i]);
    else if (arg("--pos")) o.posIt = atoi(argv[++i]);
    else if (arg("--vel")) o.velIt = atoi(argv[++i]);
    else if (arg("--r1pro")) o.r1pro = argv[++i];
    else if (arg("--r1copies")) o.r1copies = atoi(argv[++i]);
    else if (arg("--boxes")) nBoxes = atoi(argv[++i]);
    else if (arg("--joints")) joints = atoi(argv[++i]);
    else if (arg("--selfcol")) selfcol = atoi(argv[++i]);
    else if (arg("--stab")) stab = atoi(argv[++i]);
    else if (arg("--last")) last = atoi(argv[++i]);
    else if (arg("--neg")) neg = atoi(argv[++i]);
    else if (arg("--verbose")) verbose = atoi(argv[++i]);
    else if (arg("--dump")) dumpPath = argv[++i];
    else if (arg("--tiers")) tiers = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);
  PxInitExtensions(*phys, nullptr);
  HookDispatcher disp;
  PxSceneDesc sd(tol);
  sd.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  sd.cpuDispatcher = &disp;
  sd.filterShader = PxDefaultSimulationFilterShader;
  sd.solverType = PxSolverType::eTGS;
  sd.broadPhaseType = PxBroadPhaseType::ePABP;
  sd.flags |= PxSceneFlag::eENABLE_PCM | PxSceneFlag::eENABLE_ACTIVE_ACTORS;
  if (stab) sd.flags |= PxSceneFlag::eENABLE_STABILIZATION;
  if (last) sd.flags |= PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST;
  PxScene* pscene = phys->createScene(sd);
  gNpScene = static_cast<NpScene*>(pscene);
  std::mt19937 rng(o.seed * 7 + 3);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  std::vector<PxMaterial*> mats = {phys->createMaterial(0.8f, 0.6f, 0.0f), phys->createMaterial(0.5f, 0.4f, 0.1f),
                                   phys->createMaterial(1.0f, 0.9f, 0.0f)};

  // ---- 관절체 (PhysX + 엔진 거울) — 장면에 넣은 뒤 링크에 모양을 붙이고 뿌리를 바닥 가까이로 옮긴다
  std::vector<artest::Mirror> mirrors;
  if (!artest::buildRandom(phys, pscene, o, mirrors)) {
    fprintf(stderr, "관절체 만들기 실패\n");
    return 1;
  }
  const int nRandom = o.nArts;
  float groundZ = 0.0f;
  std::vector<std::pair<int, uint32_t>> r1Wheels;  // (관절체, 생성 번호)
  for (size_t mi = size_t(nRandom); mi < mirrors.size(); ++mi) {  // R1Pro: 가장 낮은 링크 3 개 = 바퀴
    artest::Mirror& m = mirrors[mi];
    std::vector<std::pair<float, uint32_t>> zs;
    for (uint32_t l = 0; l < m.pl.size(); ++l) zs.push_back({m.pl[l]->getGlobalPose().p.z, l});
    std::sort(zs.begin(), zs.end());
    for (int k = 0; k < 3 && k < int(zs.size()); ++k) r1Wheels.push_back({int(mi), zs[size_t(k)].second});
    groundZ = zs.empty() ? 0.0f : zs[0].first - 0.07f + 0.002f;
  }
  PxRigidStatic* ground = PxCreatePlane(*phys, PxPlane(0, 0, 1, -groundZ), *mats[0]);
  pscene->addActor(*ground);
  for (auto& w : r1Wheels) PxRigidActorExt::createExclusiveShape(*mirrors[size_t(w.first)].pl[w.second], PxSphereGeometry(0.07f), *mats[0]);
  for (int ai = 0; ai < nRandom; ++ai) {
    artest::Mirror& m = mirrors[size_t(ai)];
    if (!selfcol) {
      m.px->setArticulationFlag(PxArticulationFlag::eDISABLE_SELF_COLLISION, true);
      A::artSetFlag(*m.e, A::AF_DISABLE_SELF_COLLISION, true);
    }
    for (uint32_t l = 0; l < m.pl.size(); ++l) {
      const PxVec3 he(0.04f + 0.06f * P(rng), 0.04f + 0.06f * P(rng), 0.04f + 0.06f * P(rng));
      PxRigidActorExt::createExclusiveShape(*m.pl[l], PxBoxGeometry(he), *mats[size_t(ai + l) % mats.size()]);
    }
    // 뿌리 자세: x = 1.6 + 1.6 ai 줄, 바닥 위 0.35~0.6
    PxTransform rt = m.px->getRootGlobalPose();
    rt.p = PxVec3(1.6f + 1.6f * float(ai), 0.3f * U(rng), groundZ + 0.35f + 0.25f * P(rng));
    m.px->setRootGlobalPose(rt);
    A::artSetRootGlobalPose(*m.e, toE(rt));
  }

  // ---- 강체 상자 (관절체 위에서 떨어짐 + 바닥)
  eng::SceneParams spar;
  spar.gravity = eng::V3{0.0f, 0.0f, -9.81f};
  spar.speedScale = 10.0f;
  spar.stabilization = stab != 0;
  std::vector<PxRigidDynamic*> px;
  std::vector<eng::Body> eb;
  auto addBody = [&](const PxTransform& pose, const PxGeometry& g, PxMaterial* mt, float density) {
    PxRigidDynamic* a = phys->createRigidDynamic(pose);
    PxRigidActorExt::createExclusiveShape(*a, g, *mt);
    PxRigidBodyExt::updateMassAndInertia(*a, density);
    pscene->addActor(*a);
    px.push_back(a);
    eng::Body b = eng::createRigidDynamic(toE(pose), spar);
    eng::setCMassLocalPose(b, toE(a->getCMassLocalPose()));
    eng::setMass(b, a->getMass());
    eng::setMassSpaceInertiaTensor(b, toV(a->getMassSpaceInertiaTensor()));
    eb.push_back(b);
    return a;
  };
  for (int i = 0; i < nBoxes; ++i) {
    const int ai = nRandom ? i % nRandom : 0;
    const float x = nRandom ? 1.6f + 1.6f * float(ai) : 0.5f * float(i);
    const PxVec3 pos(x + 0.15f * U(rng), 0.3f * U(rng), groundZ + 0.9f + 0.4f * float(i / (nRandom ? nRandom : 1)) + 0.1f * P(rng));
    PxQuat q(3.14159f * U(rng), PxVec3(0, 0, 1));
    addBody(PxTransform(pos, q), PxBoxGeometry(0.05f + 0.05f * P(rng), 0.05f + 0.05f * P(rng), 0.04f + 0.04f * P(rng)), mats[size_t(i) % mats.size()],
            400.0f + 400.0f * P(rng));
  }
  // ---- D6 조인트
  int nJoints = 0;
  if (joints) {
    // (1) 보조 잡기 꼴: R1Pro(있으면) 마지막 링크 또는 무작위 관절체 0 의 마지막 링크 <-> 새 상자, 고정
    artest::Mirror& g = mirrors.size() > size_t(nRandom) ? mirrors.back() : mirrors[0];
    PxArticulationLink* gl = g.pl.back();
    const PxTransform gp = gl->getGlobalPose();
    PxRigidDynamic* held = addBody(PxTransform(gp.p + PxVec3(0.0f, 0.0f, 0.06f), gp.q), PxBoxGeometry(0.03f, 0.03f, 0.03f), mats[1], 300.0f);
    PxD6Joint* j = PxD6JointCreate(*phys, gl, PxTransform(PxVec3(0.0f, 0.0f, 0.06f)), held, PxTransform(PxIdentity));
    (void)j;
    nJoints++;
    // (2) 무작위 관절체 1 의 뿌리 <-> 세계 (부드러운 거리 한계 + 회전 자유) = 관절체 정적 1D
    if (nRandom > 1) {
      PxArticulationLink* r = mirrors[1].pl[0];
      const PxTransform rp = r->getGlobalPose();
      PxD6Joint* w = PxD6JointCreate(*phys, r, PxTransform(PxIdentity), nullptr, PxTransform(rp.p + PxVec3(0, 0, 0.1f)));
      w->setMotion(PxD6Axis::eX, PxD6Motion::eLIMITED);
      w->setMotion(PxD6Axis::eY, PxD6Motion::eLIMITED);
      w->setMotion(PxD6Axis::eZ, PxD6Motion::eLIMITED);
      w->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
      w->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
      w->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
      w->setDistanceLimit(PxJointLinearLimit(0.15f, PxSpring(300.0f, 10.0f)));
      nJoints++;
    }
    // (3) 무작위 관절체 2 의 마지막 링크 <-> 관절체 3 의 마지막 링크 (둘 다 링크, 회전만 자유)
    if (nRandom > 3) {
      PxArticulationLink* a0 = mirrors[2].pl.back();
      PxArticulationLink* a1 = mirrors[3].pl.back();
      const PxTransform p0 = a0->getGlobalPose(), p1 = a1->getGlobalPose();
      const PxVec3 mid = (p0.p + p1.p) * 0.5f;
      PxD6Joint* ll = PxD6JointCreate(*phys, a0, p0.getInverse() * PxTransform(mid), a1, p1.getInverse() * PxTransform(mid));
      ll->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
      ll->setMotion(PxD6Axis::eX, PxD6Motion::eLIMITED);
      ll->setLinearLimit(PxD6Axis::eX, PxJointLinearLimitPair(-0.3f, 0.3f, PxSpring(200.0f, 5.0f)));
      nJoints++;
    }
    // (4) 강체 <-> 세계 (기존 강체 경로가 같은 섬에)
    if (nBoxes > 0) {
      PxD6Joint* bw = PxD6JointCreate(*phys, px[0], PxTransform(PxIdentity), nullptr, px[0]->getGlobalPose());
      bw->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
      bw->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
      nJoints++;
    }
  }

  // ---- 번호 표
  const int nb = int(eb.size());
  for (int i = 0; i < nb; ++i) gCoreToBody[&static_cast<NpRigidDynamic*>(px[i])->getCore().getCore()] = i;
  const int na = int(mirrors.size());
  std::vector<A::Articulation> arts(static_cast<size_t>(na));
  for (int k = 0; k < na; ++k) {
    arts[size_t(k)] = *mirrors[size_t(k)].e;
    gEngArts.push_back(&arts[size_t(k)]);
    gFaToArt[llArticulation(mirrors[size_t(k)].px)] = k;
    gArtCore.push_back(&static_cast<NpArticulationReducedCoordinate*>(mirrors[size_t(k)].px)->getCore().getCore());
    for (uint32_t l = 0; l < mirrors[size_t(k)].pl.size(); ++l) {
      NpArticulationLink* nl = static_cast<NpArticulationLink*>(mirrors[size_t(k)].pl[l]);
      const uint32_t ll = nl->getLinkIndex();
      if (ll != arts[size_t(k)].ll[l]) {
        fprintf(stderr, "LL 번호 불일치 art %d link %u\n", k, l);
        return 1;
      }
      gLinkRb[&nl->getCore().getSim()->getLowLevelBody()] = {k, ll};
      gLinkCore[&nl->getCore().getCore()] = {k, ll};
    }
  }

  // ---- 엔진 판
  const uint32_t CM_CAP = 1u << 15, DESC_CAP = 1u << 15, POOL_CAP = uint32_t(nb) + 64, STATIC_CAP = 2048;
  std::vector<sv::SolverCM> cms;
  cms.reserve(CM_CAP);
  std::unordered_map<const void*, uint32_t> cmMap;
  std::vector<sv::SBodyVel> vels(POOL_CAP);
  std::vector<sv::SBodyTxI> txis(POOL_CAP);
  std::vector<sv::SBodyData> datas(POOL_CAP);
  std::vector<sv::SDesc> descs(DESC_CAP), ordered(DESC_CAP), temp(DESC_CAP);
  std::vector<sv::BatchHeader> headers(DESC_CAP);
  std::vector<uint32_t> partCounts(4096), bodySolverIndex(size_t(nb) + 1);
  std::vector<uint8_t> arenaMem(64u << 20);
  std::vector<sv::FrictionPatch> fr0(1u << 16), fr1(1u << 16);
  std::unique_ptr<sv::CorrelationBuffer> corr(new sv::CorrelationBuffer());
  std::vector<sv::ContactPoint> cbuf(sv::MAX_CONTACTS);
  std::vector<A::StaticLists> artLists(static_cast<size_t>(na));
  std::vector<sv::SDesc> artS1(size_t(na) * STATIC_CAP), artSC(size_t(na) * STATIC_CAP);
  std::vector<uint32_t> artN1(static_cast<size_t>(na)), artNC(static_cast<size_t>(na)), artBatch(static_cast<size_t>(na));
  std::vector<sv::ArtProgress> artProg(static_cast<size_t>(na));
  sv::SolverBoard B{};
  B.bodies = eb.data();
  B.nbBodies = uint32_t(nb);
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
  B.corr = corr.get();
  B.contactBuffer = cbuf.data();
  const uint32_t C1D_CAP = 1u << 12;
  std::vector<eng::jnt::Writeback> wbs(C1D_CAP);
  memset(wbs.data(), 0, wbs.size() * sizeof(eng::jnt::Writeback));
  std::vector<eng::jnt::Row> rowScratch(eng::jnt::MAX_CONSTRAINT_ROWS * 4);
  B.writebacks = wbs.data();
  B.rowScratch = rowScratch.data();
  B.arts = arts.data();
  // --tiers 1: 엔진 관절체를 딱 맞는 용량 등급(articulation.h artRepack)으로 다시 담아 포인터 표로 푼다 — 결과가 같아야 한다
  std::vector<unsigned char> tierPool;
  std::vector<A::Articulation*> ap(static_cast<size_t>(na));
  for (int k = 0; k < na; ++k) ap[size_t(k)] = &arts[size_t(k)];
  if (tiers) {
    size_t total = 0;
    std::vector<size_t> offs;
    for (int k = 0; k < na; ++k) {
      offs.push_back(total);
      total += A::artBytes(A::artTightCaps(arts[size_t(k)]));
    }
    tierPool.assign(total + 16, 0);
    unsigned char* base = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(tierPool.data()) + 15) & ~uintptr_t(15));
    for (int k = 0; k < na; ++k) {
      ap[size_t(k)] = reinterpret_cast<A::Articulation*>(base + offs[size_t(k)]);
      A::artRepack(*ap[size_t(k)], arts[size_t(k)], A::artTightCaps(arts[size_t(k)]));
      gEngArts[size_t(k)] = ap[size_t(k)];
    }
    printf("용량 등급: 관절체 %d 개 %.1f KB (최대 용량이면 %.1f KB)\n", na, double(total) / 1024.0, double(na) * double(sizeof(A::Articulation)) / 1024.0);
    B.artPtrs = ap.data();
    dumpPath = nullptr;
  }
  B.nbArts = uint32_t(na);
  B.artLists = artLists.data();
  B.artStatic1D = artS1.data();
  B.artStaticContact = artSC.data();
  B.artNbStatic1D = artN1.data();
  B.artNbStaticContact = artNC.data();
  B.artStaticCap = STATIC_CAP;
  B.artBatchIndex = artBatch.data();
  B.artProg = artProg.data();

  const float dt = 1.0f / 120.0f;
  const char* names[4] = {"행위자 자세", "선속도", "각속도", "깸 카운터"};
  uint64_t cmp = 0, bad[4] = {0, 0, 0, 0};
  int64_t first[4] = {-1, -1, -1, -1};
  uint64_t artCmp = 0, artBad = 0, artNanBoth = 0;
  int64_t artFirst = -1;
  uint64_t fricCmp = 0, fricBad = 0, snapErr = 0, wakeEvents = 0, wakeBad = 0, artWakeEvents = 0, artWakeBad = 0;
  int64_t fricFirst = -1;
  uint64_t c1dCmp = 0, c1dBad = 0, c1dArt = 0;
  int64_t c1dFirst = -1;
  uint64_t totContacts = 0, totCMs = 0, artCMs = 0, artStaticCMs = 0, maxIslandArts = 0, artSleeps = 0, freshCMs = 0;
  int shown = 0;
  std::unordered_map<const void*, uint32_t> pxPostCount;
  std::vector<float> buf;
  // ---- 입력 흐름 (--dump, 층 2 시험용)
  const uint32_t ML = A::kMaxLinks;
  const uint32_t artResFloats = 14 * ML + 2 * A::kMaxDofs + 2;
  std::vector<uint8_t> dumpBody;
  const std::vector<eng::Body> bodies0 = eb;
  const std::vector<A::Articulation> arts0(arts);
  uint32_t maxC1D = 0;
  for (int s = 1; s <= steps; ++s) {
    std::vector<sv::SolverCM> cmIn;
    std::vector<ast::Wake> wakes;
    for (int k = 0; k < na; ++k) mirrors[size_t(k)].applyInputsPx(s, dt, buf);
    for (int k = 0; k < na; ++k) artest::stepInputsEng(*ap[size_t(k)], mirrors[size_t(k)].in, s, 0, dt);
    gSnap.valid = false;
    gLate.valid = false;
    pscene->simulate(dt);
    pscene->fetchResults(true);
    Snapshot& S = gSnap;
    std::vector<sv::IslandIn> islands;
    std::vector<uint32_t> ib, icm, ia, act, resetList, ic1d;
    std::unordered_set<const void*> actKeys(S.activated.begin(), S.activated.end());
    std::vector<sv::ContactPatchIn> patches;
    std::vector<sv::ContactIn> contacts;
    std::vector<sv::Constraint1DIn> c1dIn;
    std::vector<eng::jnt::D6Data> jd;
    if (S.valid) {
      snapErr += uint64_t(S.errors);
      for (auto& si : S.islands) {
        sv::IslandIn I{};
        I.bodyStart = uint32_t(ib.size());
        I.bodyCount = uint32_t(si.bodies.size());
        I.cmStart = uint32_t(icm.size());
        I.cmCount = uint32_t(si.cms.size());
        I.staticTouchCount = si.staticTouch;
        I.c1dStart = uint32_t(ic1d.size());
        I.c1dCount = uint32_t(si.c1ds.size());
        I.artStart = uint32_t(ia.size());
        I.artCount = uint32_t(si.arts.size());
        maxIslandArts = std::max<uint64_t>(maxIslandArts, si.arts.size());
        for (int a : si.arts) ia.push_back(uint32_t(a));
        for (int b : si.bodies) ib.push_back(uint32_t(b));
        for (int k : si.c1ds) {
          const SnapC1D& x = S.c1d[size_t(k)];
          sv::Constraint1DIn c{};
          auto setRef = [&](const Ref& r, uint32_t& body, uint32_t& artLink) {
            if (r.art >= 0) {
              body = uint32_t(r.art);
              artLink = r.link + 1;
              c1dArt++;
            } else {
              body = r.body < 0 ? sv::NONE : uint32_t(r.body);
              artLink = 0;
            }
          };
          setRef(x.r0, c.body0, c.artLink0);
          setRef(x.r1, c.body1, c.artLink1);
          c.index = x.index;
          c.data = uint32_t(jd.size());
          c.writeback = x.index;
          c.flags = x.flags;
          c.linBreakForce = x.linBreak;
          c.angBreakForce = x.angBreak;
          c.minResponseThreshold = x.minResp;
          if (x.index >= C1D_CAP) snapErr++;
          maxC1D = std::max(maxC1D, x.index + 1);
          ic1d.push_back(uint32_t(c1dIn.size()));
          c1dIn.push_back(c);
          jd.push_back(x.data);
        }
        for (int ci : si.cms) {
          SnapCM& c = S.cms[size_t(ci)];
          bool resetNow = false;
          if (gFreshCM.erase(c.key)) {
            cmMap.erase(c.key);
            pxPostCount.erase(c.key);
            freshCMs++;
          } else {
            auto pc = pxPostCount.find(c.key);
            if (pc != pxPostCount.end() && pc->second != 0 && c.pxFrictionCount == 0 && !actKeys.count(c.key)) resetNow = true;
          }
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
          if (neg == 2 && s == 40 && (c.r0.art >= 0 || c.r1.art >= 0))  // 관절체가 낀 관리자 중 가장 깊이 파고든 점 하나의 분리 1 ulp
            for (auto& ct : c.contacts)
              if (ct.separation < -1e-4f) {
                printf("[음성 대조] step %d 관리자(art %d/%d) 분리 %.9g 를 1 ulp\n", s, c.r0.art, c.r1.art, ct.separation);
                ct.separation = std::nextafter(ct.separation, 1e30f);
                neg = -2;
                break;
              }
          auto setRef = [&](const Ref& r, uint32_t& body, uint32_t& artLink) {
            if (r.art >= 0) {
              body = uint32_t(r.art);
              artLink = r.link + 1;
            } else {
              body = r.body < 0 ? sv::NONE : uint32_t(r.body);
              artLink = 0;
            }
          };
          setRef(c.r0, m.body0, m.artLink0);
          setRef(c.r1, m.body1, m.artLink1);
          if (c.r0.art >= 0 || c.r1.art >= 0) {
            artCMs++;
            if (c.r1.art < 0 && c.r1.body < 0) artStaticCMs++;
          }
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
          if (resetNow) resetList.push_back(idx);
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
      // Sc 층 깨움: 강체·관절체 링크 깸 카운터를 올리기만 (ScBodySim.cpp:541, ScArticulationSim internalWakeUp)
      for (int i = 0; i < nb; ++i) {
        eb[size_t(i)].numCountedInteractions = S.numCounted[size_t(i)];
        if (S.wakeCounter[size_t(i)] != eb[size_t(i)].wakeCounter) {
          if (S.wakeCounter[size_t(i)] > eb[size_t(i)].wakeCounter) {
            eb[size_t(i)].wakeCounter = S.wakeCounter[size_t(i)];
            wakes.push_back(ast::Wake{uint32_t(i), S.wakeCounter[size_t(i)]});
            wakeEvents++;
          } else
            wakeBad++;
        }
      }
      // 지난 스텝 마찰 패치 대조 (풀기 전)
      const sv::FrictionArena& prevArena = B.friction[B.frictionCurIdx];
      std::vector<uint8_t> activated(cms.size(), 0);
      for (uint32_t a : act) activated[a] = 1;
      for (uint32_t a : resetList) activated[a] = 1;
      for (auto& si : S.islands)
        for (int ci : si.cms) {
          const SnapCM& c = S.cms[size_t(ci)];
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
            if (verbose && shown < 10) {
              shown++;
              printf("[다름] step %d 마찰 패치: 관리자 %u (몸체 %d/art %d - %d/art %d) 우리 수 %u PhysX 수 %u\n", s, cmMap[c.key], c.r0.body, c.r0.art,
                     c.r1.body, c.r1.art, myCount, c.pxFrictionCount);
            }
          }
        }
    }
    if (neg == 1 && s == 40) {  // 마지막 관절체(R1Pro 가 있으면 R1Pro)의 가장 빠른 관절 속도 1 ulp
      A::Articulation& a = *ap.back();
      uint32_t best = 0;
      for (uint32_t d = 1; d < a.dofs; ++d)
        if (std::fabs(a.jointVelocity[d]) > std::fabs(a.jointVelocity[best])) best = d;
      if (a.dofs) {
        printf("[음성 대조] step %d 관절체 %d dof %u 속도 %.9g 를 1 ulp\n", s, na - 1, best, a.jointVelocity[best]);
        a.jointVelocity[best] = std::nextafter(a.jointVelocity[best], 1e30f);
      }
    }
    B.islands = islands.data();
    B.nbIslands = uint32_t(islands.size());
    B.islandBodies = ib.data();
    B.islandCMs = icm.data();
    B.islandArts = ia.data();
    B.activatedCMs = act.data();
    B.nbActivatedCMs = uint32_t(act.size());
    B.resetCMs = resetList.data();
    B.nbResetCMs = uint32_t(resetList.size());
    B.cms = cms.data();
    B.nbCMs = uint32_t(cms.size());
    B.patches = patches.data();
    B.contacts = contacts.data();
    B.c1d = c1dIn.data();
    B.nbC1D = uint32_t(c1dIn.size());
    B.islandC1Ds = ic1d.data();
    B.jointData = jd.data();
    sv::SolverParams prm;
    prm.gravity = eng::V3{0.0f, 0.0f, -9.81f};
    prm.dt = dt;
    prm.enableStabilization = stab != 0;
    prm.bounceThreshold = S.bounce;
    prm.frictionOffsetThreshold = S.frictionOffset;
    prm.correlationDistance = S.correlation;
    prm.solverBatchSize = S.batchSize ? S.batchSize : 128;
    prm.solverArticBatchSize = S.articBatchSize ? S.articBatchSize : 16;
    prm.lengthScale = tol.length;
    prm.solveArticulationContactLast = last != 0;
    std::vector<uint32_t> deact, deactArts;
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
      const PxU32 nda = is.getNbNodesToDeactivate(IG::Node::eARTICULATION_TYPE);
      const PxNodeIndex* dia = is.getNodesToDeactivate(IG::Node::eARTICULATION_TYPE);
      for (PxU32 k = 0; k < nda; ++k) {
        auto it = gFaToArt.find(is.getNode(dia[k]).mObject);
        if (it != gFaToArt.end()) deactArts.push_back(uint32_t(it->second));
        else snapErr++;
      }
      artSleeps += deactArts.size();
    }
    {
      FtzScope f;
      sv::solverStepHost(B, prm);
      sv::afterIntegrationHost(B);
    }
    // Sc 층 깨움(관절체 링크 깸 카운터를 올리기만, ScArticulationSim.cpp:501 internalWakeUp)과 링크 상호작용 수 — 잠 판정 직전 값
    if (gLate.valid) {
      for (int k = 0; k < na; ++k) {
        A::Articulation& a = *ap[size_t(k)];
        if (getenv("TRACE_ART") && atoi(getenv("TRACE_ART")) == k)
          for (uint32_t l = 0; l < a.nLinks && l < gLate.linkWake[size_t(k)].size(); ++l)
            printf("[추적] step %d art %d link %u 잠 판정 전 깸: PhysX %.9g 엔진 %.9g 상호작용 %u (코어 PhysX %.9g 엔진 %.9g)\n", s, k, l,
                   gLate.linkWake[size_t(k)][l], a.bodies[l].wakeCounter, gLate.linkCounted[size_t(k)][l], gLate.artWake[size_t(k)], a.wakeCounter);
        for (uint32_t l = 0; l < a.nLinks && l < gLate.linkWake[size_t(k)].size(); ++l) {
          a.bodies[l].numCountedInteractions = gLate.linkCounted[size_t(k)][l];
          const float w = gLate.linkWake[size_t(k)][l];
          if (w != a.bodies[l].wakeCounter) {
            if (w > a.bodies[l].wakeCounter) {
              a.bodies[l].wakeCounter = w;
              artWakeEvents++;
            } else
              artWakeBad++;
          }
        }
        if (gLate.artWake[size_t(k)] != a.wakeCounter) {
          if (gLate.artWake[size_t(k)] > a.wakeCounter) {
            a.wakeCounter = gLate.artWake[size_t(k)];
            artWakeEvents++;
          } else
            artWakeBad++;
        }
      }
    }
    {
      FtzScope f;
      sv::afterIntegrationArtsHost(B, dt, deactArts.data(), uint32_t(deactArts.size()));
      sv::deactivateBodiesHost(B, deact.data(), uint32_t(deact.size()));
    }
    if (S.valid)
      for (const SnapCM& c : S.cms) pxPostCount[c.key] = static_cast<const PxsContactManager*>(c.key)->getWorkUnit().mFrictionPatchCount;
    // 1D 되쓰기
    if (S.valid) {
      Dy::Context* ctx = static_cast<Dy::Context*>(gNpScene->getScScene().getDynamicsContext());
      const auto& pool = ctx->getConstraintWriteBackPool();
      for (size_t k = 0; k < c1dIn.size(); ++k) {
        const uint32_t idx = c1dIn[k].index;
        c1dCmp++;
        if (memcmp(&pool[idx], &wbs[idx], sizeof(eng::jnt::Writeback))) {
          c1dBad++;
          if (c1dFirst < 0) c1dFirst = s;
          if (verbose && shown < 10) {
            shown++;
            printf("[다름] step %d 조인트 %u 되쓰기 (링크 쪽 %u/%u)\n", s, idx, c1dIn[k].artLink0, c1dIn[k].artLink1);
          }
        }
      }
    }
    if (B.error && shown < 20) {
      printf("[엔진 오류] step %d error=0x%x\n", s, B.error);
      shown++;
    }
    // 강체 비교
    for (int i = 0; i < nb; ++i) {
      const PxTransform tp = px[size_t(i)]->getGlobalPose();
      const eng::Tf te = eng::getGlobalPose(eb[size_t(i)]);
      const PxVec3 lp = px[size_t(i)]->getLinearVelocity(), ap = px[size_t(i)]->getAngularVelocity();
      const float wp = px[size_t(i)]->getWakeCounter();
      const void* pa[4] = {&tp, &lp, &ap, &wp};
      const void* pe[4] = {&te, &eb[size_t(i)].linVel, &eb[size_t(i)].angVel, &eb[size_t(i)].wakeCounter};
      const int nf[4] = {7, 3, 3, 1};
      for (int k = 0; k < 4; ++k) {
        cmp++;
        if (memcmp(pa[k], pe[k], size_t(4 * nf[k]))) {
          bad[k]++;
          if (first[k] < 0) first[k] = s;
          if (verbose && shown < 10) {
            shown++;
            printf("[다름] step %d 몸체 %d %s\n", s, i, names[k]);
          }
        }
      }
    }
    // 관절체 비교 (링크 자세·속도, 관절 위치·속도, 깸 카운터, 잠)
    for (int k = 0; k < na; ++k) {
      artest::Mirror& m = mirrors[size_t(k)];
      A::Articulation& e = *ap[size_t(k)];
      std::vector<float> x, y;
      for (uint32_t l = 0; l < m.pl.size(); ++l) {
        const PxTransform tp = m.pl[l]->getGlobalPose();
        const eng::Tf te = A::linkGlobalPose(e, l);
        const PxVec3 lv = m.pl[l]->getLinearVelocity(), av = m.pl[l]->getAngularVelocity();
        const A::LinkBody& b = e.bodies[e.ll[l]];
        const float p13[13] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lv.x, lv.y, lv.z, av.x, av.y, av.z};
        const float e13[13] = {te.q.x, te.q.y, te.q.z, te.q.w, te.p.x, te.p.y, te.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z};
        x.insert(x.end(), p13, p13 + 13);
        y.insert(y.end(), e13, e13 + 13);
        x.push_back(static_cast<NpArticulationLink*>(m.pl[l])->getCore().getCore().wakeCounter);
        y.push_back(b.wakeCounter);
      }
      if (e.dofs) {
        m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        x.insert(x.end(), m.cache->jointPosition, m.cache->jointPosition + e.dofs);
        x.insert(x.end(), m.cache->jointVelocity, m.cache->jointVelocity + e.dofs);
        y.insert(y.end(), static_cast<const float*>(e.jointPosition), static_cast<const float*>(e.jointPosition) + e.dofs);
        y.insert(y.end(), static_cast<const float*>(e.jointVelocity), static_cast<const float*>(e.jointVelocity) + e.dofs);
      }
      x.push_back(m.px->getWakeCounter());
      y.push_back(e.wakeCounter);
      x.push_back(m.px->isSleeping() ? 1.0f : 0.0f);
      y.push_back(e.awake ? 0.0f : 1.0f);
      artCmp++;
      bool same = true, anyNan = false;
      size_t fj = 0;
      for (size_t j = 0; j < x.size(); ++j) {
        const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
        if (nx || ny) {
          anyNan = true;
          if (nx != ny && same) same = false, fj = j;
        } else if (memcmp(&x[j], &y[j], 4) && same) {
          same = false;
          fj = j;
        }
      }
      if (same && anyNan) artNanBoth++;
      if (!same) {
        artBad++;
        if (artFirst < 0) artFirst = s;
        if (verbose && shown < 10) {
          shown++;
          printf("[다름] step %d 관절체 %d 칸 %zu/%zu (링크 %zu 의 %zu): PhysX %.9g 엔진 %.9g\n", s, k, fj, x.size(), fj / 14, fj % 14, x[fj], y[fj]);
        }
      }
    }
    if (dumpPath && neg == 0) {  // 스텝 블록
      ast::Header hh{};
      hh.nb = uint32_t(nb);
      hh.na = uint32_t(na);
      hh.maxLinks = ML;
      hh.artResFloats = artResFloats;
      ast::Counts c{};
      c.nIslands = uint32_t(islands.size());
      c.nIB = uint32_t(ib.size());
      c.nICM = uint32_t(icm.size());
      c.nIA = uint32_t(ia.size());
      c.nAct = uint32_t(act.size());
      c.nReset = uint32_t(resetList.size());
      c.nC1D = uint32_t(c1dIn.size());
      c.nPatches = uint32_t(patches.size());
      c.nContacts = uint32_t(contacts.size());
      c.nDeact = uint32_t(deact.size());
      c.nDeactArts = uint32_t(deactArts.size());
      c.nWake = uint32_t(wakes.size());
      c.lateValid = gLate.valid ? 1u : 0u;
      const ast::Layout L = ast::layout(c, hh);
      std::vector<uint8_t> blk(L.total, 0);
      auto put = [&](size_t off, const void* src, size_t n) {
        if (n) memcpy(blk.data() + off, src, n);
      };
      put(0, &c, sizeof(c));
      put(L.islands, islands.data(), islands.size() * sizeof(sv::IslandIn));
      put(L.ib, ib.data(), ib.size() * 4);
      put(L.icm, icm.data(), icm.size() * 4);
      put(L.cmIn, cmIn.data(), cmIn.size() * sizeof(sv::SolverCM));
      put(L.ia, ia.data(), ia.size() * 4);
      put(L.act, act.data(), act.size() * 4);
      put(L.reset, resetList.data(), resetList.size() * 4);
      put(L.c1d, c1dIn.data(), c1dIn.size() * sizeof(sv::Constraint1DIn));
      put(L.ic1d, ic1d.data(), ic1d.size() * 4);
      put(L.jd, jd.data(), jd.size() * sizeof(eng::jnt::D6Data));
      put(L.patches, patches.data(), patches.size() * sizeof(sv::ContactPatchIn));
      put(L.contacts, contacts.data(), contacts.size() * sizeof(sv::ContactIn));
      put(L.deact, deact.data(), deact.size() * 4);
      put(L.deactArts, deactArts.data(), deactArts.size() * 4);
      put(L.wake, wakes.data(), wakes.size() * sizeof(ast::Wake));
      if (S.valid) put(L.numCounted, S.numCounted.data(), size_t(nb) * 4);
      if (gLate.valid)
        for (int k = 0; k < na; ++k) {
          for (size_t l = 0; l < gLate.linkWake[size_t(k)].size(); ++l) {
            put(L.lateLinkWake + (size_t(k) * ML + l) * 4, &gLate.linkWake[size_t(k)][l], 4);
            put(L.lateLinkCounted + (size_t(k) * ML + l) * 4, &gLate.linkCounted[size_t(k)][l], 4);
          }
          put(L.lateArtWake + size_t(k) * 4, &gLate.artWake[size_t(k)], 4);
        }
      for (int i = 0; i < nb; ++i) {
        const PxTransform tp = px[size_t(i)]->getGlobalPose();
        const PxVec3 lp = px[size_t(i)]->getLinearVelocity(), ap = px[size_t(i)]->getAngularVelocity();
        const float r[ast::RES_FLOATS] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lp.x, lp.y, lp.z, ap.x, ap.y, ap.z, px[size_t(i)]->getWakeCounter()};
        put(L.pxRes + size_t(i) * sizeof(r), r, sizeof(r));
      }
      for (int k = 0; k < na; ++k) {
        artest::Mirror& m = mirrors[size_t(k)];
        std::vector<float> x(artResFloats, 0.0f);
        for (uint32_t l = 0; l < m.pl.size(); ++l) {
          const PxTransform tp = m.pl[l]->getGlobalPose();
          const PxVec3 lv = m.pl[l]->getLinearVelocity(), av = m.pl[l]->getAngularVelocity();
          const float p14[14] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lv.x, lv.y, lv.z, av.x, av.y, av.z,
                                 static_cast<NpArticulationLink*>(m.pl[l])->getCore().getCore().wakeCounter};
          memcpy(&x[l * 14], p14, sizeof(p14));
        }
        const uint32_t dofs = ap[size_t(k)]->dofs;
        if (dofs) {
          m.px->copyInternalStateToCache(*m.cache, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
          memcpy(&x[14 * ML], m.cache->jointPosition, dofs * 4);
          memcpy(&x[14 * ML + A::kMaxDofs], m.cache->jointVelocity, dofs * 4);
        }
        x[14 * ML + 2 * A::kMaxDofs] = m.px->getWakeCounter();
        x[14 * ML + 2 * A::kMaxDofs + 1] = m.px->isSleeping() ? 1.0f : 0.0f;
        put(L.pxArt + size_t(k) * artResFloats * 4, x.data(), artResFloats * 4);
      }
      if (S.valid) {
        Dy::Context* ctx = static_cast<Dy::Context*>(gNpScene->getScScene().getDynamicsContext());
        const auto& pool = ctx->getConstraintWriteBackPool();
        for (size_t k = 0; k < c1dIn.size(); ++k) put(L.pxWb + k * sizeof(eng::jnt::Writeback), &pool[c1dIn[k].index], sizeof(eng::jnt::Writeback));
      }
      dumpBody.insert(dumpBody.end(), blk.begin(), blk.end());
    }
  }
  if (dumpPath && neg == 0) {
    ast::Header h{};
    memcpy(h.magic, "ARTSV1", 7);
    h.nb = uint32_t(nb);
    h.na = uint32_t(na);
    h.steps = uint32_t(steps);
    h.stab = uint32_t(stab);
    h.last = uint32_t(last);
    h.gravity[0] = 0.0f;
    h.gravity[1] = 0.0f;
    h.gravity[2] = -9.81f;
    h.dt = dt;
    h.bounce = gSnap.bounce;
    h.frictionOffset = gSnap.frictionOffset;
    h.correlation = gSnap.correlation;
    h.lengthScale = tol.length;
    h.batchSize = gSnap.batchSize;
    h.articBatchSize = gSnap.articBatchSize;
    h.maxCMs = uint32_t(cms.size());
    h.maxArena = B.statMaxArena;
    h.maxFriction = B.statMaxFriction;
    h.maxDescs = B.statMaxDescs;
    h.maxC1D = maxC1D;
    h.staticCap = STATIC_CAP;
    h.sizeBody = sizeof(eng::Body);
    h.sizeArt = sizeof(A::Articulation);
    h.sizeCM = sizeof(sv::SolverCM);
    h.sizeIsland = sizeof(sv::IslandIn);
    h.sizeC1D = sizeof(sv::Constraint1DIn);
    h.sizeD6 = sizeof(eng::jnt::D6Data);
    h.sizeInputs = sizeof(artest::ArtInputs);
    h.maxLinks = ML;
    h.artResFloats = artResFloats;
    FILE* f = fopen(dumpPath, "wb");
    fwrite(&h, sizeof(h), 1, f);
    fwrite(bodies0.data(), sizeof(eng::Body), bodies0.size(), f);
    fwrite(arts0.data(), sizeof(A::Articulation), arts0.size(), f);
    for (int k = 0; k < na; ++k) fwrite(&mirrors[size_t(k)].in, sizeof(artest::ArtInputs), 1, f);
    fwrite(dumpBody.data(), 1, dumpBody.size(), f);
    fclose(f);
    printf("입력 흐름 저장: %s (%.1f MB)\n", dumpPath, double(dumpBody.size() + arts0.size() * sizeof(A::Articulation)) / 1e6);
  }
  printf("\n장면: 관절체 %d (무작위 %d, R1Pro %d) 강체 %d 조인트 %d x %d 스텝 (위치 %d 속도 %d, 안정화 %s, 접촉 마지막 %s, 자기 충돌 %s)\n", na, nRandom,
         na - nRandom, nb, nJoints, steps, o.posIt, o.velIt, stab ? "켬" : "끔", last ? "켬" : "끔", selfcol ? "켬" : "끔");
  printf("접촉 관리자 누적 %" PRIu64 " (관절체 링크가 낀 것 %" PRIu64 ", 그중 링크-정적 %" PRIu64 "), 접촉점 %" PRIu64 ", 섬 하나 최대 관절체 %" PRIu64
         ", 새 관리자 %" PRIu64 "\n",
         totCMs, artCMs, artStaticCMs, totContacts, maxIslandArts, freshCMs);
  printf("엔진 경로 수: ext 접촉 %" PRIu64 ", 정적 접촉 %" PRIu64 ", ext 1D %" PRIu64 ", 정적 1D %" PRIu64 ", 1D 중 링크 쪽 %" PRIu64 "\n",
         B.statArtExtContacts, B.statArtStaticContacts, B.statArtExt1D, B.statArtStatic1D, c1dArt);
  printf("강체 비교 %" PRIu64 "\n", cmp);
  for (int k = 0; k < 4; ++k) printf("  %-10s 비트 다름 %8" PRIu64 "  첫 다름 스텝 %5" PRId64 "\n", names[k], bad[k], first[k]);
  printf("관절체 비교 %" PRIu64 " (관절체 x 스텝, 링크 자세·속도·깸 카운터 + 관절 위치·속도 + 잠) 비트 다름 %" PRIu64 " 첫 다름 스텝 %" PRId64
         " 양쪽 NaN %" PRIu64 "\n",
         artCmp, artBad, artFirst, artNanBoth);
  printf("마찰 패치 비교 %" PRIu64 " 다름 %" PRIu64 " 첫 %" PRId64 " | 1D 되쓰기 비교 %" PRIu64 " 다름 %" PRIu64 " 첫 %" PRId64 "\n", fricCmp, fricBad, fricFirst,
         c1dCmp, c1dBad, c1dFirst);
  printf("Sc 깨움 입력: 강체 %" PRIu64 " (오류 %" PRIu64 "), 관절체 %" PRIu64 " (오류 %" PRIu64 "), 관절체 재움 %" PRIu64 "\n", wakeEvents, wakeBad, artWakeEvents,
         artWakeBad, artSleeps);
  printf("스냅샷 오류 %" PRIu64 ", 엔진 오류 0x%x\n", snapErr, B.error);
  const bool ok = !bad[0] && !bad[1] && !bad[2] && !bad[3] && !artBad && !fricBad && !c1dBad && !snapErr && !B.error && !wakeBad && !artWakeBad;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  fflush(stdout);
  _exit(ok ? 0 : 3);  // PhysX 해제 순서 경고 생략
}
