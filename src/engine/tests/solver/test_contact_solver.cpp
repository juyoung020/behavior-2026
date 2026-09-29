// 층 1 시험: 손으로 짠 TGS 풀이(접촉 준비·분할·위치/속도 반복·마찰 패치·잠 판정) = PhysX 5.6.1 비트 동일?
// 방법: PhysX 를 작업 스레드 하나짜리 디스패처로 돌리고, "UpdateContinuationTask"(DyTGSDynamics.cpp:514, 풀이 묶음을 만드는
// updatePostKinematic 직전) 를 실행하기 직전에 내부 상태를 스냅샷한다:
//   활성 섬 순서, 섬마다 몸체 사슬·접촉 간선 사슬, 접촉 관리자 출력(패치·점), 작업 단위 값, 활성화된 간선, 섬 정적 닿음 수.
// 우리 엔진은 같은 입력(섬 순서 + 접촉)으로 자기 몸체 상태·자기 마찰 패치 상태를 써서 풀고, fetchResults 뒤 PhysX 와 비트 비교한다.
// 섬 관리(섬 순서 만들기)와 접촉 생성은 이 시험 범위 밖(각각 islands 시험·contact 모듈).
// 1D 제약(D6 조인트): 섬의 제약 간선 사슬(Dy::Constraint: 몸체·index·플래그·끊김 힘·D6 상수 블록)도 스냅샷해 넣고,
// 스텝 뒤 PhysX 되쓰기 칸(Dy::ConstraintWriteback: 선·각 충격, 끊김)과 비교한다.
//   test_contact_solver [--scene boxes|pile|joints] [--n N] [--steps S] [--seed K] [--stab 0|1] [--verbose 0|1] [--dump file]
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <unordered_set>

#include "px_internal.h"
#define SVS_HOST_API  // 풀이 본체(aos)는 core/solver/solver_host.cpp 번역 단위에 (PhysX 헤더와 분리)
#include "core/solver/solver_io.h"
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
struct SnapC1D {  // Dy::Constraint (DyConstraint.h)
  int body0, body1;  // -1 = 정적
  uint32_t index;
  uint16_t flags;
  float linBreak, angBreak, minResp;
  eng::jnt::D6Data data;
  bool bad;
};
struct SnapIsland {
  std::vector<int> bodies;
  std::vector<int> cms;
  std::vector<int> c1ds;  // Snapshot::c1d 번호 (섬 제약 사슬 순서)
  uint32_t staticTouch;
};
struct Snapshot {
  bool valid = false;
  std::vector<SnapIsland> islands;
  std::vector<SnapCM> cms;
  std::vector<SnapC1D> c1d;
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

// 새로 만들어진(또는 같은 주소에 다시 만들어진) 접촉 관리자: PhysX 는 새 관리자의 마찰 패치 수를 0 으로 시작한다(PxcNpWorkUnit::clear,
// PxcNpWorkUnit.h:204). 관리자를 만들고 없애는 일은 contact 몫이라, 시험은 섬 관리자에 관리자를 거는 호출을 가로채 "새 관리자" 표시만 받는다.
// 같은 주소가 재사용되면(풀) 포인터만으로는 모르므로 이 표시로 우리 쪽 관리자 칸을 새로 잡는다.
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

// ---- 진단(SV_DBG_STEP 스텝에서만): PhysX 접촉 준비가 마찰 패치 수를 어떻게 바꾸는지
namespace physx { namespace Dy { class ThreadContext; } }
static int gDbgStep = -1, gCurStep = 0;
static void* gDbgCM = nullptr;
extern "C" {
int __real__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
    PxsContactManagerOutput**, Dy::ThreadContext&, PxTGSSolverContactDesc*, float, float, float, float, float, float, float, float, PxConstraintAllocator&);
int __wrap__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
    PxsContactManagerOutput** o, Dy::ThreadContext& t, PxTGSSolverContactDesc* d, float a0, float a1, float a2, float a3, float a4, float a5, float a6,
    float a7, PxConstraintAllocator& al) {
  PxU8 in[4];
  for (int k = 0; k < 4; ++k) in[k] = d[k].frictionCount;
  const int r = __real__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
      o, t, d, a0, a1, a2, a3, a4, a5, a6, a7, al);
  if (gCurStep == gDbgStep)
    for (int k = 0; k < 4; ++k)
      if (getenv("SV_DBG_ALL") || (in[k] && !d[k].frictionCount))
        printf("[진단] 4개 묶음 준비 반환 %d 칸 %d 마찰 수 %u -> %u 점 %u\n", r, k, in[k], d[k].frictionCount, d[k].numContacts);
  return r;
}
bool __real__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
    PxTGSSolverContactDesc&, PxsContactManagerOutput&, Dy::ThreadContext&, float, float, float, float, float, float, float, float, PxConstraintAllocator&);
bool __wrap__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
    PxTGSSolverContactDesc& d, PxsContactManagerOutput& o, Dy::ThreadContext& t, float a0, float a1, float a2, float a3, float a4, float a5, float a6,
    float a7, PxConstraintAllocator& al) {
  const PxU8 in = d.frictionCount;
  const bool r = __real__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
      d, o, t, a0, a1, a2, a3, a4, a5, a6, a7, al);
  if (gCurStep == gDbgStep && (getenv("SV_DBG_ALL") || (in && !d.frictionCount))) printf("[진단] 단일 준비 반환 %d 마찰 수 %u -> %u 점 %u\n", int(r), in, d.frictionCount, d.numContacts);
  return r;
}
}

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
    IG::EdgeIndex ce = island.mFirstEdge[IG::Edge::eCONSTRAINT];
    while (ce != IG_INVALID_EDGE) {
      const IG::Edge& edge = is.getEdge(ce);
      const Dy::Constraint* c = im.getConstraint(ce);
      SnapC1D x{};
      if (!c) {
        S.errors++;
      } else {
        auto bodyOf = [&](const PxsRigidBody* rb) {
          if (!rb) return -1;
          auto it = gCoreToBody.find(&rb->getCore());
          return it == gCoreToBody.end() ? -2 : it->second;
        };
        x.body0 = bodyOf(c->body0);
        x.body1 = bodyOf(c->body1);
        if (x.body0 == -2 || x.body1 == -2) x.bad = true;
        // 섬 간선 node1/node2 와 constraint body0/body1 이 같은 쪽인지 (정적 = 무효 노드)
        if (is.mCpuData.getNodeIndex1(ce).isStaticBody() != (c->body0 == nullptr)) x.bad = true;
        if (is.mCpuData.getNodeIndex2(ce).isStaticBody() != (c->body1 == nullptr)) x.bad = true;
        x.index = c->index;
        x.flags = c->flags;
        x.linBreak = c->linBreakForce;
        x.angBreak = c->angBreakForce;
        x.minResp = c->minResponseThreshold;
        if (c->constantBlockSize != sizeof(eng::jnt::D6Data)) x.bad = true;
        else memcpy(&x.data, c->constantBlock, sizeof(eng::jnt::D6Data));
        if (x.bad) S.errors++;
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
      const int before = gDbgCM && gCurStep >= gDbgStep ? int(static_cast<PxsContactManager*>(gDbgCM)->getWorkUnit().mFrictionPatchCount) : -1;
      t->run();
      if (before >= 0) {
        const int after = int(static_cast<PxsContactManager*>(gDbgCM)->getWorkUnit().mFrictionPatchCount);
        if (after != before) printf("[진단] step %d 작업 %s: 마찰 수 %d -> %d\n", gCurStep, t->getName(), before, after);
      }
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
  int jointsPer = 5;  // joints 장면: 사슬 하나의 고리 수
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--scene") && i + 1 < argc) scene = argv[++i];
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stab") && i + 1 < argc) stab = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--verbose") && i + 1 < argc) verbose = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpPath = argv[++i];
    else if (!strcmp(argv[i], "--joints") && i + 1 < argc) jointsPer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace") && i + 3 < argc) { trace = atoi(argv[++i]); t0 = atoi(argv[++i]); t1 = atoi(argv[++i]); }
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
  } else if (scene == "joints") {
    // D6 조인트 사슬 n 개: 첫 고리는 세계(정적)에 매달고, 고리끼리 여러 종류의 D6 로 잇는다. 사슬이 흔들리다 바닥·옆 사슬에 닿고, 일부는 끊어진다.
    const int side = int(std::ceil(std::sqrt(double(n))));
    std::vector<PxD6Joint*> joints;
    auto configure = [&](PxD6Joint* j, int kind) {
      switch (kind % 7) {
        case 0:  // 고정 (전부 잠금)
          break;
        case 1:  // 회전 (twist 자유)
          j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
          break;
        case 2:  // 회전 (twist 한계, 단단함)
          j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLIMITED);
          j->setTwistLimit(PxJointAngularLimitPair(-0.6f, 0.5f));
          break;
        case 3:  // 구 (swing 원뿔 한계, twist 자유)
          j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLIMITED);
          j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLIMITED);
          j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
          j->setSwingLimit(PxJointLimitCone(0.5f, 0.4f));
          break;
        case 4: {  // 직선 (X 부드러운 한계)
          j->setMotion(PxD6Axis::eX, PxD6Motion::eLIMITED);
          PxJointLinearLimitPair l(-0.05f, 0.1f, PxSpring(800.0f, 20.0f));
          j->setLinearLimit(PxD6Axis::eX, l);
          break;
        }
        case 5:  // 각 자유 + slerp 드라이브(목표 자세)
          j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
          j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
          j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
          j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(60.0f, 4.0f, PX_MAX_F32, true));
          j->setDrivePosition(PxTransform(PxQuat(0.4f, PxVec3(0, 1, 0))));
          break;
        case 6:  // 거리 한계 (선 자유 + 거리)
          j->setMotion(PxD6Axis::eX, PxD6Motion::eLIMITED);
          j->setMotion(PxD6Axis::eY, PxD6Motion::eLIMITED);
          j->setMotion(PxD6Axis::eZ, PxD6Motion::eLIMITED);
          j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
          j->setDistanceLimit(PxJointLinearLimit(0.08f));
          j->setDrive(PxD6Drive::eX, PxD6JointDrive(200.0f, 10.0f, 50.0f, false));
          j->setDrivePosition(PxTransform(PxVec3(0.03f, 0, 0)));
          break;
      }
    };
    for (int c = 0; c < n; ++c) {
      const float x = float(c % side) * 0.9f, y = float(c / side) * 0.9f;
      const float top = 1.4f + 0.3f * P(rng);
      const PxVec3 he(0.12f, 0.05f, 0.05f);
      PxRigidActor* prev = nullptr;
      PxVec3 prevAnchorWorld(x, y, top);
      const float tilt = 0.8f * U(rng);
      for (int k = 0; k < jointsPer; ++k) {
        // 고리 중심: 앞 고리 끝에서 기울어진 방향으로 he.x 만큼
        const PxVec3 dir(std::cos(tilt), 0.3f * U(rng), -std::sin(std::fabs(tilt)) - 0.2f);
        const PxVec3 d = dir.getNormalized();
        const PxQuat q = PxShortestRotation(PxVec3(1, 0, 0), d);
        const PxVec3 center = prevAnchorWorld + d * he.x;
        const bool capsule = ((c + k) % 3) == 2;
        if (capsule)
          addBody(PxTransform(center, q), PxCapsuleGeometry(0.05f, 0.07f), mats[(c + k) % mats.size()], 700.0f, PxVec3(0), PxVec3(0));
        else
          addBody(PxTransform(center, q), PxBoxGeometry(he), mats[(c + k) % mats.size()], 500.0f + 300.0f * P(rng), PxVec3(0.2f * U(rng), 0, 0), PxVec3(0));
        PxRigidDynamic* cur = px.back();
        const PxTransform localA = prev ? PxTransform(PxVec3(he.x, 0, 0)) : PxTransform(prevAnchorWorld, q);  // 정적 쪽은 세계 틀
        PxD6Joint* j = PxD6JointCreate(*phys, prev, localA, cur, PxTransform(PxVec3(-he.x, 0, 0)));
        configure(j, c + k);
        if ((c + k) % 5 == 4) j->setBreakForce(250.0f + 200.0f * P(rng), 60.0f + 40.0f * P(rng));  // 끊어질 수 있는 조인트
        if ((c + k) % 11 == 3) j->setConstraintFlag(PxConstraintFlag::eENABLE_EXTENDED_LIMITS, true);
        joints.push_back(j);
        prev = cur;
        prevAnchorWorld = center + d * he.x;
      }
    }
    // 사슬 사이에 떨어지는 자유 상자 몇 개 (접촉과 조인트가 같은 섬에)
    for (int i = 0; i < n / 2; ++i)
      addBody(PxTransform(PxVec3(float(i % side) * 0.9f + 0.3f, float(i / side) * 0.9f, 2.2f + 0.3f * P(rng)), rq()), PxBoxGeometry(0.08f, 0.08f, 0.08f),
              mats[i % mats.size()], 600.0f, PxVec3(0), PxVec3(0));
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
  const uint32_t C1D_CAP = 1u << 16;
  std::vector<eng::jnt::Writeback> wbs(C1D_CAP);
  memset(wbs.data(), 0, wbs.size() * sizeof(eng::jnt::Writeback));  // Dy::ConstraintWriteback::initialize
  std::vector<eng::jnt::Row> rowScratch(eng::jnt::MAX_CONSTRAINT_ROWS * 4);
  B.writebacks = wbs.data();
  B.rowScratch = rowScratch.data();
  uint32_t maxC1D = 0;
  uint64_t c1dCmp = 0, c1dBad = 0, c1dTot = 0, brokenSeen = 0;
  int64_t c1dFirst = -1;

  const float dt = 1.0f / 120.0f;
  uint64_t cmp = 0, bad[4] = {0, 0, 0, 0};
  int64_t first[4] = {-1, -1, -1, -1};
  double maxd[4] = {0, 0, 0, 0};
  const char* names[4] = {"행위자 자세", "선속도", "각속도", "깸 카운터"};
  uint64_t fricCmp = 0, fricBad = 0, snapErr = 0, wakeEvents = 0, wakeBad = 0;
  int64_t fricFirst = -1;
  uint64_t totContacts = 0, totCMs = 0, maxIslands = 0;
  int shown = 0;
  uint64_t freshCMs = 0, resetEvents = 0;
  // PhysX 풀이가 끝난 뒤 관리자별 마찰 패치 수. 다음 스냅샷에서 활성화도 새 관리자도 아닌데 0 이 되어 있으면 Sc 층이 캐시 상태를 지운 것
  // (clearCachedState 말고는 풀이 밖에서 이 값을 바꾸는 곳이 없다: PxcNpWorkUnit.h:201, PxsContactManager.h:120, DyTGSDynamics.cpp:552,1178)
  std::unordered_map<const void*, uint32_t> pxPostCount;
  std::vector<uint8_t> wasActive(nb, 1), isActive(nb, 0);
  std::vector<uint8_t> dumpBody;  // 스텝 블록들 (--dump)
  const std::vector<eng::Body> bodies0 = eb;
  for (int s = 1; s <= steps; ++s) {
    gSnap.valid = false;
    gCurStep = s;
    if (getenv("SV_DBG_STEP")) gDbgStep = atoi(getenv("SV_DBG_STEP"));
    pscene->simulate(dt);
    pscene->fetchResults(true);
    Snapshot& S = gSnap;
    std::fill(isActive.begin(), isActive.end(), 0);
    std::vector<sv::IslandIn> islands;
    std::vector<uint32_t> ib, icm, act, resetList;
    std::unordered_set<const void*> actKeys(S.activated.begin(), S.activated.end());
    std::vector<sv::ContactPatchIn> patches;
    std::vector<sv::ContactIn> contacts;
    std::vector<sv::SolverCM> cmIn;
    std::vector<svs::Wake> wakes;
    std::vector<sv::Constraint1DIn> c1dIn;
    std::vector<uint32_t> ic1d;
    std::vector<eng::jnt::D6Data> jd;
    std::vector<uint32_t> numCounted(nb, 0);
    for (int i = 0; i < nb; ++i) numCounted[i] = eb[i].numCountedInteractions;
    if (S.valid) {
      snapErr += S.errors;
      for (auto& si : S.islands) {
        sv::IslandIn I{uint32_t(ib.size()), uint32_t(si.bodies.size()), uint32_t(icm.size()), uint32_t(si.cms.size()), si.staticTouch,
                       uint32_t(ic1d.size()), uint32_t(si.c1ds.size())};
        for (int k : si.c1ds) {
          const SnapC1D& x = S.c1d[k];
          sv::Constraint1DIn c{};
          c.body0 = x.body0 < 0 ? sv::NONE : uint32_t(x.body0);
          c.body1 = x.body1 < 0 ? sv::NONE : uint32_t(x.body1);
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
        for (int b : si.bodies) {
          ib.push_back(uint32_t(b));
          isActive[b] = 1;
        }
        for (int ci : si.cms) {
          const SnapCM& c = S.cms[ci];
          bool resetNow = false;
          if (gFreshCM.erase(c.key)) {
            cmMap.erase(c.key);  // 새 관리자: 마찰 패치 수 0 에서 시작
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
          if (getenv("SV_DBG_B0") && c.body0 == atoi(getenv("SV_DBG_B0")) && c.body1 == atoi(getenv("SV_DBG_B1")))
            gDbgCM = const_cast<void*>(c.key), printf("[추적] step %d 관리자 %u (%p) 패치 %zu 점 %zu 우리 지난 수 %u PhysX 수 %u flags %x\n", s, idx, c.key, c.patches.size(), c.contacts.size(), m.frictionCount,
                   c.pxFrictionCount, c.flags);
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
          if (resetNow) {
            resetList.push_back(idx);
            resetEvents++;
          }
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
      for (uint32_t a : resetList) activated[a] = 1;  // Sc 층 캐시 지움도 마찰 수 0
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
            if (verbose && shown < 8) {
              shown++;
              printf("[다름] step %d 마찰 패치: 관리자 %u (몸체 %d-%d) 활성화 %d, 우리 수 %u (지난 스텝 수 %u) PhysX 수 %u\n", s, cmMap[c.key], m.body0 == sv::NONE ? -1 : int(m.body0),
                     m.body1 == sv::NONE ? -1 : int(m.body1), int(activated[cmMap[c.key]]), myCount, m.frictionCount, c.pxFrictionCount);
            }
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
    prm.lengthScale = tol.length;
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
    v.c.nC1D = uint32_t(c1dIn.size());
    v.c1d = c1dIn.data();
    v.ic1d = ic1d.data();
    v.jd = jd.data();
    v.pxWb = nullptr;
    v.c.nReset = uint32_t(resetList.size());
    v.reset = resetList.data();
    B.cms = cms.data();
    B.nbCMs = uint32_t(cms.size());
    {
      FtzScope f;
      svs::runStep(B, prm, v);
    }
    if (S.valid)  // PhysX 풀이 뒤 마찰 패치 수 기록 (이번 스텝 섬의 관리자는 fetchResults 뒤에도 살아 있다)
      for (const SnapCM& c : S.cms) pxPostCount[c.key] = static_cast<const PxsContactManager*>(c.key)->getWorkUnit().mFrictionPatchCount;
    // 1D 제약 되쓰기 대조 (PhysX Dy::ConstraintWriteback 칸 = 우리 칸, 이번 스텝 섬에 있던 제약)
    std::vector<eng::jnt::Writeback> pxWb(c1dIn.size());
    if (S.valid) {
      Dy::Context* ctx = static_cast<Dy::Context*>(gNpScene->getScScene().getDynamicsContext());
      const auto& pool = ctx->getConstraintWriteBackPool();
      for (size_t k = 0; k < c1dIn.size(); ++k) {
        const uint32_t idx = c1dIn[k].index;
        memcpy(&pxWb[k], &pool[idx], sizeof(eng::jnt::Writeback));
        c1dCmp++;
        c1dTot++;
        if (pxWb[k].broken_residualPosIter & 0x80000000u) brokenSeen++;
        if (memcmp(&pxWb[k], &wbs[idx], sizeof(eng::jnt::Writeback))) {
          c1dBad++;
          if (c1dFirst < 0) c1dFirst = s;
          if (verbose && shown < 8) {
            shown++;
            printf("[다름] step %d 조인트 %u 되쓰기: PhysX lin (%.9g %.9g %.9g) ang (%.9g %.9g %.9g) %08x | 엔진 lin (%.9g %.9g %.9g) ang (%.9g %.9g %.9g) %08x\n", s,
                   idx, pxWb[k].linearImpulse.x, pxWb[k].linearImpulse.y, pxWb[k].linearImpulse.z, pxWb[k].angularImpulse.x, pxWb[k].angularImpulse.y,
                   pxWb[k].angularImpulse.z, pxWb[k].broken_residualPosIter, wbs[idx].linearImpulse.x, wbs[idx].linearImpulse.y, wbs[idx].linearImpulse.z,
                   wbs[idx].angularImpulse.x, wbs[idx].angularImpulse.y, wbs[idx].angularImpulse.z, wbs[idx].broken_residualPosIter);
          }
        }
      }
    }
    if (dumpPath) {
      auto put = [&](const void* p, size_t n) { const uint8_t* b = static_cast<const uint8_t*>(p); dumpBody.insert(dumpBody.end(), b, b + n); };
      svs::StepArrays A{v.c, islands.data(), ib.data(), icm.data(), cmIn.data(), act.data(), deact.data(), wakes.data(), numCounted.data(),
                        patches.data(), contacts.data(), c1dIn.data(), ic1d.data(), jd.data(), pxWb.data(), resetList.data()};
      const std::vector<uint8_t> blk = svs::packStep(A, uint32_t(nb));
      put(blk.data(), blk.size());
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
  printf("새로 만들어진 접촉 관리자(섬 간선에 건 것) %" PRIu64 ", Sc 층 캐시 지움(조인트 끊김 뒤 등) %" PRIu64 "\n", freshCMs, resetEvents);
  printf("스냅샷 오류 %" PRIu64 ", 엔진 오류 0x%x\n", snapErr, B.error);
  printf("1D 제약(D6 조인트) 되쓰기 비교 %" PRIu64 " (제약·스텝), 다름 %" PRIu64 ", 첫 다름 스텝 %" PRId64 ", 끊김 표시 %" PRIu64
         " | 4개 묶음 준비 %" PRIu64 ", 하나씩 %" PRIu64 " (행 0 %" PRIu64 ")\n",
         c1dCmp, c1dBad, c1dFirst, brokenSeen, B.stat1DBlock4, B.stat1DSingle, B.stat1DZeroRows);
  printf("작업 공간 최대: 제약 자료 %u B/묶음, 마찰 패치 %u 개/스텝, 제약 %u 개/묶음, 접촉 관리자 %zu 개\n", B.statMaxArena, B.statMaxFriction,
         B.statMaxDescs, cms.size());
  if (dumpPath) {
    svs::Header h{};
    memcpy(h.magic, "SVSTRM2", 8);
    h.version = 2;
    h.lengthScale = tol.length;
    h.maxC1D = maxC1D;
    h.c1dSize = sizeof(sv::Constraint1DIn);
    h.d6Size = sizeof(eng::jnt::D6Data);
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
  const bool ok = !bad[0] && !bad[1] && !bad[2] && !bad[3] && !fricBad && !snapErr && !B.error && !wakeBad && !c1dBad;
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  pscene->release();
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
