// G1 solver 그림자 시험 (문서 15절 G1): 공식 radio 장면을 PhysX 비계로 재생하면서, 매 simulate 마다 PhysX 풀이 직전 상태를 떠서
// 우리 TGS 풀이(core/solver, 층 1 호스트 진입 solverStepHost)로 같은 스텝을 한 번 더 풀고 fetchResults 뒤 PhysX 결과와 비트 비교한다.
// - 범위: 관절체가 없는 섬(강체·가구·조인트). radio 는 활성 섬 83 개가 묶음 2 개로 묶이고 둘 다 관절체를 품어서, 묶음 단위로는 풀 것이 없다.
//   그래서 PhysX 묶음 규칙(DyTGSDynamics.cpp:715-730: 몸체 수 < solverBatchSize, 관절체 < articBatchSize)으로 묶음을 다시 만들고,
//   묶음마다 관절체 없는 섬만 떼어 우리 풀이를 따로 한 번씩 부른다. 섬끼리는 몸체를 나누지 않아 계산이 서로 섞이지 않는다
//   (분할·순서는 몸체별). 묶음이 섬 사이에 주는 것은 반복 수 하나(묶음 안 강체·관절체의 최댓값, DyTGSDynamics.cpp:1006-1008,1690-1704)라서
//   뗀 섬의 몸체 반복 수 칸에 그 묶음 최댓값을 넣는다(반복 수 칸은 최댓값 계산에만 쓰인다). 관절체 노드·운동학 상대·수정 가능 접촉이 있는 섬은 뺀다.
//   관절체 섬은 관절체 결합(engine-solver-art 작업자, SolverBoard 뒤에 arts·artLink 추가)이 오면 같은 입력 형식으로 넣는다.
// - 스텝마다 다시 맞춤(한 스텝 그림자): 몸체 상태(PxsBodyCore + PxsRigidBody)·지난 마찰 패치·조인트 되쓰기 칸을 PhysX 에서 떠서 판에 넣는다.
//   그래서 차이가 누적되지 않고 스텝마다 "같은 입력 -> 같은 출력" 만 본다.
// - 뜨는 시점: 작업 스레드 하나짜리 디스패처로 돌리고 "UpdateContinuationTask"(DyTGSDynamics.cpp:514, 풀이 묶음을 만드는 updatePostKinematic 직전)
//   를 실행하기 직전 (tests/solver/test_contact_solver.cpp 와 같은 방법).
// - 입력 형식 = SolverBoard (core/solver/solver_io.h, 12.3 약속). 판은 값 초기화({})로 만들어 뒤에 붙는 필드(arts·artLink 등)는 비어 있다.
// 켜기: G1_SOLVER=1 (ovd_replay_g1). G1_SOLVER_SHOW=N 이면 다른 몸체 N 개를 자세히.
#include <xmmintrin.h>

#include <cinttypes>
#include <unordered_set>

#include "../tests/solver/px_internal.h"
#include "DyFeatherstoneArticulation.h"
#if defined(G1_SOLVER_IO)
#include G1_SOLVER_IO  // solver_host.cpp 와 같은 뿌리의 판 정의 (CMake G1_CORE_ROOT)
#else
#include "core/solver/solver_io.h"
#endif

using namespace physx;
namespace sv = eng::sv;
namespace jnt = eng::jnt;

namespace {

struct FtzScope {  // PhysX PxSIMDGuard 와 같은 MXCSR (FTZ + DAZ)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};
eng::V3 toV(const PxVec3& v) { return eng::V3{v.x, v.y, v.z}; }
eng::Tf toE(const PxTransform& t) { return eng::Tf{{t.q.x, t.q.y, t.q.z, t.q.w}, {t.p.x, t.p.y, t.p.z}}; }

eng::Body bodyFrom(const PxsRigidBody& rb) {
  const PxsBodyCore& c = rb.getCore();
  eng::Body b{};
  b.body2World = toE(c.body2World);
  b.body2Actor = toE(c.getBody2Actor());
  b.linVel = toV(c.linearVelocity);
  b.angVel = toV(c.angularVelocity);
  b.maxAngVelSq = c.maxAngularVelocitySq;
  b.maxLinVelSq = c.maxLinearVelocitySq;
  b.linDamping = c.linearDamping;
  b.angDamping = c.angularDamping;
  b.invInertia = toV(c.inverseInertia);
  b.invMass = c.inverseMass;
  b.maxContactImpulse = c.maxContactImpulse;
  b.maxPenBias = c.maxPenBias;
  b.sleepThreshold = c.sleepThreshold;
  b.freezeThreshold = c.freezeThreshold;
  b.wakeCounter = c.wakeCounter;
  b.solverWakeCounter = c.solverWakeCounter;
  b.numCountedInteractions = c.numCountedInteractions;
  b.lockFlags = uint16_t(PxU8(c.lockFlags));
  b.disableGravity = c.disableGravity;
  b.gyroscopic = (c.mFlags & PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES) ? 1 : 0;
  b.solverIterationCounts = c.solverIterationCounts;
  b.freezeCount = rb.mFreezeCount;
  b.accelScale = rb.mAccelScale;
  b.sleepLinVelAcc = toV(rb.mSleepLinVelAcc);
  b.sleepAngVelAcc = toV(rb.mSleepAngVelAcc);
  b.lastTransform = toE(rb.mLastTransform);
  b.internalFlags = rb.mInternalFlags;
  return b;
}

sv::FrictionPatch frictionFrom(const Dy::FrictionPatch& p) {
  sv::FrictionPatch f;
  f.broken = p.broken;
  f.materialFlags = p.materialFlags;
  f.anchorCount = p.anchorCount;
  f.restitution = p.restitution;
  f.staticFriction = p.staticFriction;
  f.dynamicFriction = p.dynamicFriction;
  f.body0Normal = toV(p.body0Normal);
  f.body1Normal = toV(p.body1Normal);
  for (int a = 0; a < 2; ++a) {
    f.body0Anchors[a] = toV(p.body0Anchors[a]);
    f.body1Anchors[a] = toV(p.body1Anchors[a]);
  }
  f.relativeQuat = eng::Q{p.relativeQuat.x, p.relativeQuat.y, p.relativeQuat.z, p.relativeQuat.w};
  return f;
}
bool sameFriction(const sv::FrictionPatch& a, const sv::FrictionPatch& b) {
  bool same = a.broken == b.broken && a.materialFlags == b.materialFlags && a.anchorCount == b.anchorCount &&
              !memcmp(&a.restitution, &b.restitution, sizeof(float) * 3) && !memcmp(&a.body0Normal, &b.body0Normal, sizeof(float) * 6) &&
              !memcmp(&a.relativeQuat, &b.relativeQuat, sizeof(float) * 4);
  for (uint32_t an = 0; same && an < a.anchorCount; ++an)
    same = !memcmp(&a.body0Anchors[an], &b.body0Anchors[an], 12) && !memcmp(&a.body1Anchors[an], &b.body1Anchors[an], 12);
  return same;
}

struct Tally {
  const char* name;
  uint64_t cmp = 0, bad = 0;
  long long first = -1;
  double maxd = 0;
  void f(const float* a, const float* b, int n, long long sim) {
    ++cmp;
    if (memcmp(a, b, 4 * n)) {
      if (!bad) first = sim;
      ++bad;
      for (int k = 0; k < n; ++k) maxd = std::fmax(maxd, std::fabs(double(a[k]) - double(b[k])));
    }
  }
  void u(bool same, long long sim) {
    ++cmp;
    if (!same) {
      if (!bad) first = sim;
      ++bad;
    }
  }
};

// 한 스텝 입력 (스냅샷이 채움)
struct Step {
  bool valid = false;
  PxScene* scene = nullptr;
  std::vector<eng::Body> bodies, bodies0;  // bodies0 = 풀이 전 (진단)
  std::vector<const PxsRigidBody*> rbs;  // 몸체 번호 -> PhysX 몸체 (비교용)
  std::unordered_map<const PxsRigidBody*, uint32_t> rbIndex;
  std::vector<sv::IslandIn> islands;
  std::vector<uint32_t> ib, icm, act, ic1d;
  std::vector<sv::SolverCM> cms;
  std::vector<const PxsContactManager*> cmKeys;
  std::unordered_map<const PxsContactManager*, uint32_t> cmIndex;
  std::vector<sv::ContactPatchIn> patches;
  std::vector<sv::ContactIn> contacts;
  std::vector<sv::FrictionPatch> friction;  // 지난 스텝 마찰 패치 (PhysX 값)
  std::vector<sv::Constraint1DIn> c1d;
  std::vector<jnt::D6Data> jd;
  std::vector<std::pair<uint32_t, jnt::Writeback>> wbSeed;  // (제약 번호, 풀이 전 PhysX 되쓰기 칸)
  sv::SolverParams prm{};
  struct Group { uint32_t islandStart, islandEnd; uint16_t iterWord; };  // PhysX 묶음 하나에서 뗀 섬들 + 그 묶음의 반복 수
  std::vector<Group> groups;
  // 통계
  uint32_t islandsAll = 0, islandsUsed = 0, bodiesSkipped = 0, batches = 0, batchesUsed = 0;
  uint32_t skipArt = 0, skipKin = 0, skipMod = 0, skipOther = 0;
};

struct SolverShadow {
  bool on = false, checked = false;
  PxScene* curScene = nullptr;
  Step st;
  // 판 작업 공간 (용량 고정, 스텝마다 재사용)
  std::vector<sv::SBodyVel> vels;
  std::vector<sv::SBodyTxI> txis;
  std::vector<sv::SBodyData> datas;
  std::vector<sv::SDesc> descs, ordered, temp;
  std::vector<sv::BatchHeader> headers;
  std::vector<uint32_t> partCounts, bodySolverIndex;
  std::vector<uint8_t> arenaMem;
  std::vector<sv::FrictionPatch> fr0, fr1;
  std::unique_ptr<sv::CorrelationBuffer> corr;
  std::vector<sv::ContactPoint> cbuf;
  std::vector<jnt::Writeback> wbs;
  std::vector<jnt::Row> rowScratch;
  // 결과
  Tally tPose{"몸체 자세"}, tLin{"선속도"}, tAng{"각속도"}, tWake{"깸 카운터"}, tSleep{"잠 누적·얼림"}, tFric{"마찰 패치(풀이 뒤)"}, tWb{"조인트 되쓰기"};
  uint64_t steps = 0, stepsNoSnap = 0, bodiesCmp = 0, cmsSolved = 0, contactsSolved = 0, c1dSolved = 0, engineErr = 0;
  uint64_t islandsAll = 0, islandsUsed = 0, bodiesSkipped = 0, batches = 0, batchesUsed = 0, skipArt = 0, skipKin = 0, skipMod = 0, skipOther = 0;
  int show = 0;

  void init() {
    const uint32_t POOL = 1u << 14, DESC = 1u << 16;
    vels.resize(POOL);
    txis.resize(POOL);
    datas.resize(POOL);
    descs.resize(DESC);
    ordered.resize(DESC);
    temp.resize(DESC);
    headers.resize(DESC);
    partCounts.resize(4096);
    arenaMem.resize(64u << 20);
    fr0.resize(1u << 18);
    fr1.resize(1u << 18);
    corr.reset(new sv::CorrelationBuffer());
    cbuf.resize(sv::MAX_CONTACTS);
    rowScratch.resize(jnt::MAX_CONSTRAINT_ROWS * 4);
    show = getenv("G1_SOLVER_SHOW") ? atoi(getenv("G1_SOLVER_SHOW")) : 0;
  }
} GS;

// ---------------- 스냅샷 (UpdateContinuationTask 직전, 작업 스레드)
void takeSnapshot() {
  Step& S = GS.st;
  S = Step();
  if (!GS.curScene) return;
  S.scene = GS.curScene;
  Sc::Scene& sc = static_cast<NpScene*>(GS.curScene)->getScScene();
  IG::SimpleIslandManager& im = *sc.getSimpleIslandManager();
  const IG::IslandSim& is = im.getAccurateIslandSim();
  Dy::Context* dy = static_cast<Dy::Context*>(sc.getDynamicsContext());
  Dy::DynamicsContextBase* dyb = static_cast<Dy::DynamicsContextBase*>(sc.getDynamicsContext());
  PxsContactManagerOutputIterator& outs = dyb->mOutputIterator;
  const auto& wbPool = dy->getConstraintWriteBackPool();
  sv::SolverParams& prm = S.prm;
  prm.gravity = toV(dy->getGravity());
  prm.dt = dy->getDt();
  prm.enableStabilization = (GS.curScene->getFlags() & PxSceneFlag::eENABLE_STABILIZATION);
  prm.bounceThreshold = dy->getBounceThreshold();
  prm.frictionOffsetThreshold = dy->getFrictionOffsetThreshold();
  prm.correlationDistance = dy->getCorrelationDistance();
  prm.solverBatchSize = dy->getSolverBatchSize();
  prm.solverArticBatchSize = dy->getSolverArticBatchSize();
  prm.lengthScale = GS.curScene->getPhysics().getTolerancesScale().length;

  const PxU32 nIsl = is.getNbActiveIslands();
  const IG::IslandId* ids = is.getActiveIslands();
  S.islandsAll = nIsl;
  PxU32 cur = 0;
  while (cur < nIsl) {
    // PhysX 묶음 하나 (DyTGSDynamics.cpp:720)
    PxU32 nbBodies = 0, nbArt = 0, maxPos = 0, maxVel = 0;
    const uint32_t gStart = uint32_t(S.islands.size());
    while (nbBodies < prm.solverBatchSize && cur < nIsl && nbArt < prm.solverArticBatchSize) {
      const IG::Island& island = is.getIsland(ids[cur]);
      nbBodies += island.mNodeCount[IG::Node::eRIGID_BODY_TYPE];
      nbArt += island.mNodeCount[IG::Node::eARTICULATION_TYPE];
      // 묶음 반복 수: 강체(preIntegrateBodies)·관절체(setupArticulations) 최댓값
      for (PxNodeIndex n = island.mRootNode; n.isValid();) {
        const IG::Node& node = is.getNode(n);
        PxU16 w = 0;
        if (node.getNodeType() == IG::Node::eRIGID_BODY_TYPE) w = reinterpret_cast<const PxsRigidBody*>(node.mObject)->getCore().solverIterationCounts;
        else if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) w = reinterpret_cast<const Dy::FeatherstoneArticulation*>(node.mObject)->getIterationCounts();
        maxPos = PxMax<PxU32>(maxPos, w & 0xff);
        maxVel = PxMax<PxU32>(maxVel, w >> 8);
        n = node.mNextNode;
      }
      // 섬 하나를 판 입력으로 (안 되면 되돌림)
      const size_t mB = S.bodies.size(), mIB = S.ib.size(), mICM = S.icm.size(), mCM = S.cms.size(), mP = S.patches.size(),
                   mC = S.contacts.size(), mF = S.friction.size(), mC1 = S.c1d.size(), mIC1 = S.ic1d.size(), mJD = S.jd.size(), mWB = S.wbSeed.size();
      int why = 0;  // 0 = 씀, 1 관절체, 2 운동학, 3 수정 가능 접촉, 4 기타
      auto fail = [&](int w) { if (!why) why = w; };
      if (island.mNodeCount[IG::Node::eARTICULATION_TYPE]) fail(1);
      sv::IslandIn I{};
      I.bodyStart = uint32_t(S.ib.size());
      I.cmStart = uint32_t(S.icm.size());
      I.c1dStart = uint32_t(S.ic1d.size());
      I.staticTouchCount = is.mIslandStaticTouchCount[ids[cur]];
      for (PxNodeIndex n = island.mRootNode; !why && n.isValid();) {
        const IG::Node& node = is.getNode(n);
        if (node.getNodeType() != IG::Node::eRIGID_BODY_TYPE) fail(1);
        else {
          const PxsRigidBody* rb = reinterpret_cast<const PxsRigidBody*>(node.mObject);
          const uint32_t bi = uint32_t(S.bodies.size());
          S.rbIndex[rb] = bi;
          S.rbs.push_back(rb);
          S.bodies.push_back(bodyFrom(*rb));
          S.ib.push_back(bi);
        }
        n = node.mNextNode;
      }
      I.bodyCount = uint32_t(S.ib.size()) - I.bodyStart;
      for (IG::EdgeIndex e = island.mFirstEdge[IG::Edge::eCONTACT_MANAGER]; !why && e != IG_INVALID_EDGE;) {
        const IG::Edge& edge = is.getEdge(e);
        PxsContactManager* cm = im.getContactManager(e);
        if (cm) {
          const PxNodeIndex n1 = is.mCpuData.getNodeIndex1(e), n2 = is.mCpuData.getNodeIndex2(e);
          sv::SolverCM m{};
          const PxcNpWorkUnit& u = cm->getWorkUnit();
          if (n1.isStaticBody() || is.getNode(n1).isKinematic() || (!n2.isStaticBody() && is.getNode(n2).isKinematic())) fail(2);
          else if (is.getNode(n1).getNodeType() != IG::Node::eRIGID_BODY_TYPE || (!n2.isStaticBody() && is.getNode(n2).getNodeType() != IG::Node::eRIGID_BODY_TYPE))
            fail(1);
          else {
            const PxsRigidBody* rb0 = reinterpret_cast<const PxsRigidBody*>(is.getNode(n1).mObject);
            auto i0 = S.rbIndex.find(rb0);
            if (i0 == S.rbIndex.end() || static_cast<const void*>(u.mRigidCore0) != static_cast<const void*>(&rb0->getCore())) fail(4);
            else m.body0 = i0->second;
            if (n2.isStaticBody()) {
              m.body1 = sv::NONE;
              m.staticPose1 = toE(u.mRigidCore1->body2World);
            } else {
              auto i1 = S.rbIndex.find(reinterpret_cast<const PxsRigidBody*>(is.getNode(n2).mObject));
              if (i1 == S.rbIndex.end()) fail(4);
              else m.body1 = i1->second;
            }
          }
          m.npFlags = u.mFlags;
          m.restDistance = u.mRestDistance;
          m.torsionalPatchRadius = u.mTorsionalPatchRadius;
          m.minTorsionalPatchRadius = u.mMinTorsionalPatchRadius;
          m.offsetSlop = u.mOffsetSlop;
          const PxsContactManagerOutput& o = outs.getContactManagerOutput(u.mNpIndex);
          const PxContactPatch* pp = reinterpret_cast<const PxContactPatch*>(o.contactPatches);
          const PxContact* pc = reinterpret_cast<const PxContact*>(o.contactPoints);
          if (o.nbPatches && (pp[0].internalFlags & (PxContactPatch::eMODIFIABLE | PxContactPatch::eCOMPRESSED_MODIFIED_CONTACT))) fail(3);
          m.patchStart = uint32_t(S.patches.size());
          m.nbPatches = o.nbPatches;
          m.contactStart = uint32_t(S.contacts.size());
          m.nbContacts = o.nbContacts;
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
            S.patches.push_back(q);
          }
          for (PxU32 k = 0; k < o.nbContacts; ++k) S.contacts.push_back(sv::ContactIn{toV(pc[k].contact), pc[k].separation});
          // 지난 스텝 마찰 패치 = PhysX 값 (스텝마다 다시 맞춤)
          m.frictionPtr = uint32_t(S.friction.size());
          m.frictionCount = u.mFrictionPatchCount;
          const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
          for (PxU32 k = 0; k < u.mFrictionPatchCount; ++k) {
            if (!fp) { fail(4); m.frictionCount = 0; break; }
            S.friction.push_back(frictionFrom(fp[k]));
          }
          const uint32_t ci = uint32_t(S.cms.size());
          S.cmIndex[cm] = ci;
          S.cmKeys.push_back(cm);
          S.cms.push_back(m);
          S.icm.push_back(ci);
        }
        e = edge.mNextIslandEdge;
      }
      I.cmCount = uint32_t(S.icm.size()) - I.cmStart;
      for (IG::EdgeIndex e = island.mFirstEdge[IG::Edge::eCONSTRAINT]; !why && e != IG_INVALID_EDGE;) {
        const IG::Edge& edge = is.getEdge(e);
        const Dy::Constraint* c = im.getConstraint(e);
        if (!c) fail(4);
        else {
          auto bodyOf = [&](const PxsRigidBody* rb) -> uint32_t {
            if (!rb) return sv::NONE;
            auto it = S.rbIndex.find(rb);
            if (it == S.rbIndex.end()) { fail(4); return sv::NONE; }
            return it->second;
          };
          sv::Constraint1DIn x{};
          x.body0 = bodyOf(c->body0);
          x.body1 = bodyOf(c->body1);
          if (is.mCpuData.getNodeIndex1(e).isStaticBody() != (c->body0 == nullptr)) fail(4);
          if (is.mCpuData.getNodeIndex2(e).isStaticBody() != (c->body1 == nullptr)) fail(4);
          x.index = c->index;
          x.data = uint32_t(S.jd.size());
          x.writeback = c->index;
          x.flags = c->flags;
          x.linBreakForce = c->linBreakForce;
          x.angBreakForce = c->angBreakForce;
          x.minResponseThreshold = c->minResponseThreshold;
          jnt::D6Data d{};
          if (c->constantBlockSize != sizeof(jnt::D6Data)) fail(4);
          else memcpy(&d, c->constantBlock, sizeof(jnt::D6Data));
          S.jd.push_back(d);
          jnt::Writeback w;
          memcpy(&w, &wbPool[c->index], sizeof(w));
          S.wbSeed.emplace_back(c->index, w);
          S.ic1d.push_back(uint32_t(S.c1d.size()));
          S.c1d.push_back(x);
        }
        e = edge.mNextIslandEdge;
      }
      I.c1dCount = uint32_t(S.ic1d.size()) - I.c1dStart;
      if (why) {  // 섬 되돌림
        S.bodiesSkipped += island.mNodeCount[IG::Node::eRIGID_BODY_TYPE];
        for (size_t k = mB; k < S.rbs.size(); ++k) S.rbIndex.erase(S.rbs[k]);
        for (size_t k = mCM; k < S.cmKeys.size(); ++k) S.cmIndex.erase(S.cmKeys[k]);
        S.bodies.resize(mB); S.rbs.resize(mB); S.ib.resize(mIB); S.icm.resize(mICM); S.cms.resize(mCM); S.cmKeys.resize(mCM);
        S.patches.resize(mP); S.contacts.resize(mC); S.friction.resize(mF); S.c1d.resize(mC1); S.ic1d.resize(mIC1); S.jd.resize(mJD); S.wbSeed.resize(mWB);
        ++(why == 1 ? S.skipArt : why == 2 ? S.skipKin : why == 3 ? S.skipMod : S.skipOther);
      } else {
        S.islands.push_back(I);
        ++S.islandsUsed;
      }
      ++cur;
    }
    ++S.batches;
    if (S.islands.size() > gStart) {
      ++S.batchesUsed;
      S.groups.push_back(Step::Group{gStart, uint32_t(S.islands.size()), uint16_t((maxVel << 8) | maxPos)});
    }
  }
  // 이번 스텝 활성화된 접촉 간선 (쓴 섬의 것만)
  const PxU32 nbAct = is.getNbActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  const IG::EdgeIndex* act = is.getActivatedEdges(IG::Edge::eCONTACT_MANAGER);
  for (PxU32 a = 0; a < nbAct; ++a) {
    auto it = S.cmIndex.find(im.getContactManager(act[a]));
    if (it != S.cmIndex.end()) S.act.push_back(it->second);
  }
  S.valid = true;
}

// ---------------- 진단: PhysX 가 접촉 준비를 4 개 묶음(SIMD)으로 했는지 하나씩 했는지 (몸체 틀 bodyFrame0 위치로 가림)
struct PrepRec { float x, y, z; int block; };
std::vector<PrepRec> gPrep;
}  // namespace
namespace physx { namespace Dy { class ThreadContext; } }
extern "C" {
int __real__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
    PxsContactManagerOutput**, Dy::ThreadContext&, PxTGSSolverContactDesc*, float, float, float, float, float, float, float, float, PxConstraintAllocator&);
int __wrap__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
    PxsContactManagerOutput** o, Dy::ThreadContext& t, PxTGSSolverContactDesc* d, float a0, float a1, float a2, float a3, float a4, float a5, float a6,
    float a7, PxConstraintAllocator& al) {
  const int r = __real__ZN5physx2Dy33createFinalizeSolverContacts4StepEPPNS_23PxsContactManagerOutputERNS0_13ThreadContextEPNS_22PxTGSSolverContactDescEffffffffRNS_21PxConstraintAllocatorE(
      o, t, d, a0, a1, a2, a3, a4, a5, a6, a7, al);
  for (int k = 0; k < 4; ++k) gPrep.push_back(PrepRec{d[k].bodyFrame0.p.x, d[k].bodyFrame0.p.y, d[k].bodyFrame0.p.z, r == 2 ? 4 : 40});  // SolverConstraintPrepState: 0 메모리 부족, 1 묶을 수 없음, 2 성공 (DySolverConstraintDesc.h:74)
  return r;
}
bool __real__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
    PxTGSSolverContactDesc&, PxsContactManagerOutput&, Dy::ThreadContext&, float, float, float, float, float, float, float, float, PxConstraintAllocator&);
bool __wrap__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
    PxTGSSolverContactDesc& d, PxsContactManagerOutput& o, Dy::ThreadContext& t, float a0, float a1, float a2, float a3, float a4, float a5, float a6,
    float a7, PxConstraintAllocator& al) {
  gPrep.push_back(PrepRec{d.bodyFrame0.p.x, d.bodyFrame0.p.y, d.bodyFrame0.p.z, 1});
  return __real__ZN5physx2Dy32createFinalizeSolverContactsStepERNS_22PxTGSSolverContactDescERNS_23PxsContactManagerOutputERNS0_13ThreadContextEffffffffRNS_21PxConstraintAllocatorE(
      d, o, t, a0, a1, a2, a3, a4, a5, a6, a7, al);
}
}
namespace {

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
      if (!strcmp(t->getName(), "UpdateContinuationTask")) {
        takeSnapshot();
        gPrep.clear();
      }
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

}  // namespace

PxCpuDispatcher* g1_dispatcher() {
  if (!getenv("G1_SOLVER")) return nullptr;
  static HookDispatcher* d = new HookDispatcher();  // 프로세스 끝까지 (PxPhysics 여러 개가 같이 씀)
  GS.on = true;
  return d;
}

// ovd_replay 가 simulate 바로 앞에서 (g1_shadow.cpp 의 g1_before_simulate 를 거쳐)
void g1_solver_before(PxScene* scene) {
  if (!GS.on) return;
  GS.curScene = scene;
  GS.st.valid = false;
}

// fetchResults 뒤: 같은 입력으로 우리 풀이를 돌려 PhysX 결과와 비교
void g1_solver_after(PxScene* scene, uint64_t sim) {
  if (!GS.on) return;
  if (!GS.checked) {
    GS.init();
    GS.checked = true;
  }
  Step& S = GS.st;
  ++GS.steps;
  if (!S.valid || S.scene != scene) {
    ++GS.stepsNoSnap;
    return;
  }
  GS.islandsAll += S.islandsAll;
  GS.islandsUsed += S.islandsUsed;
  GS.bodiesSkipped += S.bodiesSkipped;
  GS.batches += S.batches;
  GS.batchesUsed += S.batchesUsed;
  GS.skipArt += S.skipArt;
  GS.skipKin += S.skipKin;
  GS.skipMod += S.skipMod;
  GS.skipOther += S.skipOther;
  if (S.islands.empty()) return;
  S.bodies0 = S.bodies;
  const uint32_t nb = uint32_t(S.bodies.size());
  if (nb + 64 > GS.vels.size()) {
    GS.vels.resize(nb + 64);
    GS.txis.resize(nb + 64);
    GS.datas.resize(nb + 64);
  }
  GS.bodySolverIndex.assign(nb, 0);
  uint32_t maxWb = 0;
  for (auto& w : S.wbSeed) maxWb = std::max(maxWb, w.first + 1);
  if (maxWb > GS.wbs.size()) GS.wbs.resize(maxWb);
  for (auto& w : S.wbSeed) GS.wbs[w.first] = w.second;

  // 이번 스텝 섬 관리자가 재운 몸체 (fetchResults 뒤 accurate IslandSim 에 남아 있다)
  std::vector<uint8_t> deactFlag(nb, 0);
  {
    const IG::IslandSim& is = static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager()->getAccurateIslandSim();
    const PxU32 nd = is.getNbNodesToDeactivate(IG::Node::eRIGID_BODY_TYPE);
    const PxNodeIndex* di = is.getNodesToDeactivate(IG::Node::eRIGID_BODY_TYPE);
    for (PxU32 k = 0; k < nd; ++k) {
      auto it = S.rbIndex.find(reinterpret_cast<const PxsRigidBody*>(is.getNode(di[k]).mObject));
      if (it != S.rbIndex.end()) deactFlag[it->second] = 1;
    }
  }
  std::vector<uint32_t> cmGroup(S.cms.size(), 0);
  for (uint32_t g = 0; g < S.groups.size(); ++g)
    for (uint32_t i = S.groups[g].islandStart; i < S.groups[g].islandEnd; ++i)
      for (uint32_t k = 0; k < S.islands[i].cmCount; ++k) cmGroup[S.icm[S.islands[i].cmStart + k]] = g;
  if (S.friction.size() > GS.fr0.size()) GS.fr0.resize(S.friction.size());
  const auto& pool = static_cast<Dy::Context*>(static_cast<NpScene*>(scene)->getScScene().getDynamicsContext())->getConstraintWriteBackPool();

  // PhysX 묶음마다 뗀 섬들을 따로 한 번씩 푼다
  for (uint32_t g = 0; g < S.groups.size(); ++g) {
    const Step::Group& G = S.groups[g];
    std::vector<uint32_t> act, deact, gBodies, gCMs, gC1D;
    for (uint32_t i = G.islandStart; i < G.islandEnd; ++i) {
      const sv::IslandIn& I = S.islands[i];
      for (uint32_t k = 0; k < I.bodyCount; ++k) gBodies.push_back(S.ib[I.bodyStart + k]);
      for (uint32_t k = 0; k < I.cmCount; ++k) gCMs.push_back(S.icm[I.cmStart + k]);
      for (uint32_t k = 0; k < I.c1dCount; ++k) gC1D.push_back(S.ic1d[I.c1dStart + k]);
    }
    for (uint32_t b : gBodies) {
      S.bodies[b].solverIterationCounts = G.iterWord;  // 묶음 반복 수 (최댓값 계산에만 쓰임)
      if (deactFlag[b]) deact.push_back(b);
    }
    for (uint32_t a : S.act)
      if (cmGroup[a] == g) act.push_back(a);

    sv::SolverBoard B{};  // 값 초기화: 12.3 약속으로 뒤에 붙는 필드(관절체 arts·artLink 등)는 비어 있음
    B.bodies = S.bodies.data();
    B.nbBodies = nb;
    B.cms = S.cms.data();
    B.nbCMs = uint32_t(S.cms.size());
    B.patches = S.patches.data();
    B.contacts = S.contacts.data();
    B.islands = S.islands.data() + G.islandStart;
    B.nbIslands = G.islandEnd - G.islandStart;
    B.islandBodies = S.ib.data();  // IslandIn 의 범위는 전체 배열 기준
    B.islandCMs = S.icm.data();
    B.activatedCMs = act.data();
    B.nbActivatedCMs = uint32_t(act.size());
    B.resetCMs = nullptr;  // PhysX 마찰 수를 그대로 떴으므로 Sc 층 캐시 지움은 이미 반영됨
    B.nbResetCMs = 0;
    B.c1d = S.c1d.data();
    B.nbC1D = uint32_t(S.c1d.size());
    B.islandC1Ds = S.ic1d.data();
    B.jointData = S.jd.data();
    B.writebacks = GS.wbs.data();
    B.rowScratch = GS.rowScratch.data();
    B.vels = GS.vels.data();
    B.txI = GS.txis.data();
    B.datas = GS.datas.data();
    B.poolCap = uint32_t(GS.vels.size());
    B.descs = GS.descs.data();
    B.ordered = GS.ordered.data();
    B.temp = GS.temp.data();
    B.headers = GS.headers.data();
    B.descCap = uint32_t(GS.descs.size());
    B.partitionCounts = GS.partCounts.data();
    B.partitionCap = uint32_t(GS.partCounts.size());
    B.bodySolverIndex = GS.bodySolverIndex.data();
    B.constraints = sv::ByteArena{GS.arenaMem.data(), 0, uint32_t(GS.arenaMem.size()), 0};
    // 지난 마찰 패치: solverStepBegin 이 교대하므로 frictionCurIdx 쪽에 넣으면 prev 가 된다
    B.frictionCurIdx = 0;
    std::copy(S.friction.begin(), S.friction.end(), GS.fr0.begin());
    B.friction[0] = sv::FrictionArena{GS.fr0.data(), uint32_t(S.friction.size()), uint32_t(GS.fr0.size()), 0};
    B.friction[1] = sv::FrictionArena{GS.fr1.data(), 0, uint32_t(GS.fr1.size()), 0};
    B.corr = GS.corr.get();
    B.contactBuffer = GS.cbuf.data();
    {
      FtzScope f;
      sv::solverStepHost(B, S.prm);
      sv::afterIntegrationHost(B);
      sv::deactivateBodiesHost(B, deact.data(), uint32_t(deact.size()));
    }
    if (B.error) ++GS.engineErr;
    GS.cmsSolved += gCMs.size();
    GS.c1dSolved += gC1D.size();
    for (uint32_t ci : gCMs) GS.contactsSolved += S.cms[ci].nbContacts;
    // 비교: 몸체
    for (uint32_t i : gBodies) {
      const PxsRigidBody& rb = *S.rbs[i];
      const PxsBodyCore& c = rb.getCore();
      const eng::Body& e = S.bodies[i];
      ++GS.bodiesCmp;
      const uint64_t b0 = GS.tPose.bad + GS.tLin.bad + GS.tAng.bad + GS.tWake.bad;
      GS.tPose.f(&c.body2World.q.x, &e.body2World.q.x, 7, sim);
      GS.tLin.f(&c.linearVelocity.x, &e.linVel.x, 3, sim);
      GS.tAng.f(&c.angularVelocity.x, &e.angVel.x, 3, sim);
      GS.tWake.f(&c.wakeCounter, &e.wakeCounter, 1, sim);
      {
        const float a[8] = {rb.mSleepLinVelAcc.x, rb.mSleepLinVelAcc.y, rb.mSleepLinVelAcc.z, rb.mSleepAngVelAcc.x, rb.mSleepAngVelAcc.y, rb.mSleepAngVelAcc.z,
                            rb.mFreezeCount, rb.mAccelScale};
        const float b[8] = {e.sleepLinVelAcc.x, e.sleepLinVelAcc.y, e.sleepLinVelAcc.z, e.sleepAngVelAcc.x, e.sleepAngVelAcc.y, e.sleepAngVelAcc.z,
                            e.freezeCount, e.accelScale};
        GS.tSleep.f(a, b, 8, sim);
      }
      if (getenv("G1_SOLVER_LIST") && sim == uint64_t(atoi(getenv("G1_SOLVER_LIST")))) {  // 진단: 몸체별 성질과 같음 여부
        uint32_t nStatic = 0, nDyn = 0, nPts = 0;
        for (uint32_t ci : gCMs)
          if (S.cms[ci].body0 == i || S.cms[ci].body1 == i) (S.cms[ci].body1 == sv::NONE ? nStatic : nDyn)++, nPts += S.cms[ci].nbContacts;
        // PhysX 준비 방식: 이 몸체의 풀이 전 자세(bodyFrame0 = body2World)와 같은 기록
        int nb4 = 0, nb1 = 0, nbf = 0;
        const eng::Tf& p0 = S.bodies0[i].body2World;
        for (const PrepRec& r : gPrep)
          if (r.x == p0.p.x && r.y == p0.p.y && r.z == p0.p.z) (r.block == 4 ? nb4 : r.block == 1 ? nb1 : nbf)++;
        fprintf(stderr, "[목록] 몸체 %u 같음 %d gyro %u 정적관리자 %u 동적관리자 %u 점 %u invMass %g | PhysX 준비: 4묶음 %d 하나씩 %d 4묶음실패 %d\n", i,
                int(GS.tPose.bad + GS.tLin.bad + GS.tAng.bad + GS.tWake.bad == b0), unsigned(e.gyroscopic), nStatic, nDyn, nPts, e.invMass, nb4, nb1, nbf);
      }
      if (GS.show > 0 && GS.tPose.bad + GS.tLin.bad + GS.tAng.bad + GS.tWake.bad != b0) {
        --GS.show;
        fprintf(stderr, "[g1 solver 다름] sim %llu 묶음 %u 몸체 %u (반복 %04x, 잠 %d)\n  PhysX p (%.9g %.9g %.9g) v (%.9g %.9g %.9g) w (%.9g %.9g %.9g) wc %.9g\n  우리  p (%.9g %.9g %.9g) v (%.9g %.9g %.9g) w (%.9g %.9g %.9g) wc %.9g\n",
                (unsigned long long)sim, g, i, unsigned(e.solverIterationCounts), int(deactFlag[i]), c.body2World.p.x, c.body2World.p.y, c.body2World.p.z,
                c.linearVelocity.x, c.linearVelocity.y, c.linearVelocity.z, c.angularVelocity.x, c.angularVelocity.y, c.angularVelocity.z, c.wakeCounter,
                e.body2World.p.x, e.body2World.p.y, e.body2World.p.z, e.linVel.x, e.linVel.y, e.linVel.z, e.angVel.x, e.angVel.y, e.angVel.z, e.wakeCounter);
        for (uint32_t ii = G.islandStart; ii < G.islandEnd; ++ii) {
          const sv::IslandIn& I = S.islands[ii];
          bool has = false;
          for (uint32_t k = 0; k < I.bodyCount; ++k) has |= S.ib[I.bodyStart + k] == i;
          if (!has) continue;
          fprintf(stderr, "  섬 %u: 몸체 %u 관리자 %u 조인트 %u 정적닿음 %u | 몸체: lock %u noG %u gyro %u linDamp %g angDamp %g maxPenBias %g maxCI %g invMass %g flags %x\n", ii,
                  I.bodyCount, I.cmCount, I.c1dCount, I.staticTouchCount, unsigned(e.lockFlags), unsigned(e.disableGravity), unsigned(e.gyroscopic),
                  e.linDamping, e.angDamping, e.maxPenBias, e.maxContactImpulse, e.invMass, unsigned(e.internalFlags));
          for (uint32_t k = 0; k < I.cmCount; ++k) {
            const sv::SolverCM& m = S.cms[S.icm[I.cmStart + k]];
            fprintf(stderr, "    관리자 몸체 %d-%d 패치 %u 점 %u npFlags %x restDist %g 마찰수 %u", m.body0 == sv::NONE ? -1 : int(m.body0),
                    m.body1 == sv::NONE ? -1 : int(m.body1), m.nbPatches, m.nbContacts, m.npFlags, m.restDistance, m.frictionCount);
            for (uint32_t q = 0; q < m.nbPatches; ++q)
              fprintf(stderr, " [mat %x int %x ims %g %g %g %g]", unsigned(S.patches[m.patchStart + q].materialFlags), unsigned(S.patches[m.patchStart + q].internalFlags),
                      S.patches[m.patchStart + q].invMassScale[0], S.patches[m.patchStart + q].invMassScale[1], S.patches[m.patchStart + q].invMassScale[2],
                      S.patches[m.patchStart + q].invMassScale[3]);
            fprintf(stderr, "\n");
          }
        }
      }
    }
    // 비교: 풀이 뒤 마찰 패치 (PhysX 작업 단위 = 우리 관리자 칸)
    const sv::FrictionArena& fa = B.friction[B.frictionCurIdx];
    for (uint32_t k : gCMs) {
      const PxcNpWorkUnit& u = S.cmKeys[k]->getWorkUnit();
      const sv::SolverCM& m = S.cms[k];
      bool same = u.mFrictionPatchCount == m.frictionCount;
      const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
      for (uint32_t q = 0; same && q < m.frictionCount; ++q) same = fp && sameFriction(fa.data[m.frictionPtr + q], frictionFrom(fp[q]));
      GS.tFric.u(same, sim);
    }
    // 비교: 조인트 되쓰기
    for (uint32_t k : gC1D) GS.tWb.u(!memcmp(&pool[S.c1d[k].index], &GS.wbs[S.c1d[k].index], sizeof(jnt::Writeback)), sim);
  }
}

void g1_solver_report() {
  if (!GS.on) return;
  printf("G1 solver 그림자 (관절체 없는 섬, 스텝마다 PhysX 상태로 다시 맞춤): simulate %" PRIu64 " (스냅샷 없음 %" PRIu64 "), 엔진 오류 스텝 %" PRIu64 "\n",
         GS.steps, GS.stepsNoSnap, GS.engineErr);
  printf("  활성 섬 %" PRIu64 " 중 푼 섬 %" PRIu64 " (PhysX 묶음 %" PRIu64 " 중 %" PRIu64 "), 뺀 섬: 관절체 묶음 %" PRIu64 ", 운동학 %" PRIu64 ", 수정 가능 접촉 %" PRIu64 ", 기타 %" PRIu64
         ", 뺀 묶음의 강체 %" PRIu64 "\n",
         GS.islandsAll, GS.islandsUsed, GS.batches, GS.batchesUsed, GS.skipArt, GS.skipKin, GS.skipMod, GS.skipOther, GS.bodiesSkipped);
  printf("  푼 것 누적: 몸체 %" PRIu64 ", 접촉 관리자 %" PRIu64 ", 접촉점 %" PRIu64 ", 조인트 %" PRIu64 "\n", GS.bodiesCmp, GS.cmsSolved, GS.contactsSolved, GS.c1dSolved);
  for (const Tally* t : {&GS.tPose, &GS.tLin, &GS.tAng, &GS.tWake, &GS.tSleep, &GS.tFric, &GS.tWb})
    printf("  %-20s 비교 %10" PRIu64 "  비트 다름 %8" PRIu64 "  최대|차| %.3e%s\n", t->name, t->cmp, t->bad, t->maxd,
           t->bad ? ("  첫 다름 simulate " + std::to_string(t->first)).c_str() : "");
}
