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
#include "../tests/articulation/px_art_internal.h"  // Dy::FeatherstoneArticulation 내부 (관절체 결합 그림자)
#include "DyFeatherstoneArticulation.h"
#include "core/articulation/art_static.h"
#include "g1_hooks.h"
#include "core/solver/islands.h"
#include "g1_px.h"
#include "core/scene/scene_file.h"
#include <map>
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
using g1px::toV;
using g1px::toE;
using g1px::bodyFrom;

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
  // 관절체 결합 (전체 그림자): 섬의 관절체 노드 -> 판 관절체 번호
  std::vector<const void*> artFa;                    // 판 관절체 번호 -> Dy::FeatherstoneArticulation*
  std::unordered_map<const void*, uint32_t> artIndex;
  std::vector<uint32_t> ia;                          // 섬별 관절체 목록 (IslandIn.artStart/artCount)
  bool fullOk = true;
  int failWhy = 0;                                // 전체 모드에서 한 섬이라도 못 넣으면 이 스텝은 비교 안 함
  // 잠 판정 직전(ScScene.afterIntegration) 관절체 링크 깸·상호작용 수, 관절체 깸 (엔진 결합 작업자 지적: 풀이 직전이 아님)
  bool lateValid = false;
  std::vector<std::vector<float>> lateLinkWake;
  std::vector<std::vector<uint32_t>> lateLinkCounted;
  std::vector<float> lateArtWake;
  // 풀이 직전 관절체 깸 (simulate 안에서 깨어난 관절체: 쌍둥이는 잠든 상태로 옮겨졌으므로 깨움을 반영)
  std::vector<float> preArtWake;
  std::vector<std::vector<float>> preLinkWake;
  struct Group { uint32_t islandStart, islandEnd; uint16_t iterWord; };  // PhysX 묶음 하나에서 뗀 섬들 + 그 묶음의 반복 수
  std::vector<Group> groups;
  // 통계
  uint32_t islandsAll = 0, islandsUsed = 0, bodiesSkipped = 0, batches = 0, batchesUsed = 0;
  uint32_t skipArt = 0, skipKin = 0, skipMod = 0, skipOther = 0;
};

G1StepInfo gInfo;  // 관절체 단독 그림자(g1_art.cpp)에 넘기는 것

struct SolverShadow {
  bool on = false, cmpOn = false, checked = false, full = true;
  PxScene* curScene = nullptr;
  uint64_t stepSim = 0;
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
  Tally tPose{"몸체 자세"}, tLin{"선속도"}, tAng{"각속도"}, tWake{"깸 카운터"}, tSleep{"잠 누적·얼림"}, tFric{"마찰 패치(풀이 뒤)"}, tWb{"조인트 되쓰기"}, tArt{"관절체(링크·관절·잠)"};
  uint64_t fullSteps = 0, fullSkipped = 0, noLate = 0, artWoken = 0;
  uint64_t steps = 0, stepsNoSnap = 0, bodiesCmp = 0, cmsSolved = 0, contactsSolved = 0, c1dSolved = 0, engineErr = 0;
  uint64_t islandsAll = 0, islandsUsed = 0, bodiesSkipped = 0, batches = 0, batchesUsed = 0, skipArt = 0, skipKin = 0, skipMod = 0, skipOther = 0;
  int show = 0;
  // 닫힌 고리 (2b) 지속 모드: 몸체·마찰 패치·조인트 되쓰기를 우리 결과로 들고 간다
  bool persist = false;
  std::unordered_map<const PxsRigidBody*, eng::Body> pBodies;
  std::map<std::pair<const void*, const void*>, std::vector<sv::FrictionPatch>> pFric;  // (모양 핵 0, 1) -> 지난 마찰 패치
  std::unordered_set<uint32_t> pWbSeen;
  uint64_t lcUsed = 0, lcDiff = 0, lcMissing = 0;  // 2단 접촉 입력
  uint64_t lIslSteps = 0;                          // 3단: 섬 목록을 우리 섬 관리에서 읽은 스텝
  uint64_t pBodyUsed = 0, pBodyResync = 0, pBodyNew = 0, pBodyDiff = 0, pFricUsed = 0, pFricDiff = 0, pFricReset = 0, pFricNew = 0, pWbUsed = 0, pWbDiff = 0;
  long long pFirstBody = -1, pFirstFric = -1, pFirstWb = -1;

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


// ---------------- 섬 보기 (닫힌 고리 3단): 풀이 입력의 섬 목록·사슬·간선을 PhysX 정확 섬 시뮬 또는 우리 섬 관리(core/solver/islands.h)에서 읽는다.
// 노드·간선 -> 몸체·관리자·제약 대응(식별 표)은 번호가 같으므로 PhysX 표를 쓴다 (상태가 아니라 이름표).
struct IsView {
  const IG::IslandSim& p;
  IG::SimpleIslandManager& im;
  const eng::ig::IslandSim* o;  // 우리 (없으면 PhysX)
  PxU32 nbActive() const { return o ? o->activeIslands.size : p.getNbActiveIslands(); }
  uint32_t activeId(PxU32 k) const { return o ? o->activeIslands.d[k] : p.getActiveIslands()[k]; }
  uint32_t nodeCount(uint32_t id, int t) const { return o ? o->islands.d[id].nodeCount[t] : p.getIsland(id).mNodeCount[t]; }
  static PxNodeIndex pn(uint32_t id, uint32_t linkRaw) {  // 우리 NodeIndex.linkId = PhysX mLinkID 원값 ((링크 << 1) | 관절체)
    return id == eng::ig::INVALID_NODE ? PxNodeIndex() : PxNodeIndex(PxU64(id) | (PxU64(linkRaw) << 32));
  }
  PxNodeIndex root(uint32_t id) const { return o ? pn(o->islands.d[id].rootNode, 0) : p.getIsland(id).mRootNode; }
  uint32_t firstEdge(uint32_t id, int t) const { return o ? o->islands.d[id].firstEdge[t] : p.getIsland(id).mFirstEdge[t]; }
  uint32_t staticTouch(uint32_t id) const { return o ? o->islandStaticTouchCount[id] : p.mIslandStaticTouchCount[id]; }
  int type(PxNodeIndex n) const { return o ? int(o->nodes.d[n.index()].type) : int(p.getNode(n).getNodeType()); }
  bool kin(PxNodeIndex n) const { return o ? (o->nodes.d[n.index()].flags & eng::ig::N_KINEMATIC) != 0 : p.getNode(n).isKinematic(); }
  void* obj(PxNodeIndex n) const { return p.getNode(n).mObject; }  // 식별 표
  PxNodeIndex next(PxNodeIndex n) const { return o ? pn(o->nodes.d[n.index()].nextNode, 0) : p.getNode(n).mNextNode; }
  uint32_t nextEdge(uint32_t e) const { return o ? o->edges.d[e].nextIslandEdge : p.getEdge(e).mNextIslandEdge; }
  PxNodeIndex n1(uint32_t e) const { return o ? pn(o->cpu->edgeNodeIndices[2 * e].id, o->cpu->edgeNodeIndices[2 * e].linkId) : p.mCpuData.getNodeIndex1(e); }
  PxNodeIndex n2(uint32_t e) const { return o ? pn(o->cpu->edgeNodeIndices[2 * e + 1].id, o->cpu->edgeNodeIndices[2 * e + 1].linkId) : p.mCpuData.getNodeIndex2(e); }
  PxU32 nbAct() const { return o ? o->activatedEdges[0].size : p.getNbActivatedEdges(IG::Edge::eCONTACT_MANAGER); }
  uint32_t act(PxU32 k) const { return o ? o->activatedEdges[0].d[k] : p.getActivatedEdges(IG::Edge::eCONTACT_MANAGER)[k]; }
  PxU32 nbDeact(int t) const { return o ? o->nodesToPutToSleep[t].size : p.getNbNodesToDeactivate(IG::Node::NodeType(t)); }
  PxNodeIndex deact(int t, PxU32 k) const { return o ? pn(o->nodesToPutToSleep[t].d[k], 0) : p.getNodesToDeactivate(IG::Node::NodeType(t))[k]; }
};
const eng::ig::IslandSim* loopIslands() {
  static const bool on = getenv("G1_LOOP_ISLANDS") != nullptr;
  if (!on || !GS.persist) return nullptr;
  const eng::ig::IslandManager* M = g1_islands_ours();
  return M ? &M->accurate : nullptr;
}

// ---------------- 늦은 스냅샷 (ScScene.afterIntegration 직전): 관절체 잠 판정에 들어가는 링크 깸·상호작용 수, 관절체 깸
void takeLateSnapshot() {
  Step& S = GS.st;
  if (!S.valid) return;
  S.lateLinkWake.assign(S.artFa.size(), {});
  S.lateLinkCounted.assign(S.artFa.size(), {});
  S.lateArtWake.assign(S.artFa.size(), 0.0f);
  for (size_t k = 0; k < S.artFa.size(); ++k) {
    const Dy::FeatherstoneArticulation* fa = static_cast<const Dy::FeatherstoneArticulation*>(S.artFa[k]);
    const Dy::ArticulationData& d = fa->mArticulationData;
    const PxU32 n = d.getLinkCount();
    for (PxU32 l = 0; l < n; ++l) {
      S.lateLinkWake[k].push_back(d.mLinks[l].bodyCore->wakeCounter);
      S.lateLinkCounted[k].push_back(d.mLinks[l].bodyCore->numCountedInteractions);
    }
    S.lateArtWake[k] = fa->mSolverDesc.core->wakeCounter;
  }
  S.lateValid = true;
}

// ---------------- 스냅샷 (UpdateContinuationTask 직전, 작업 스레드)
void takeSnapshot() {
  Step& S = GS.st;
  S = Step();
  gInfo = G1StepInfo();
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

  gInfo.dt = prm.dt;
  gInfo.gravity[0] = prm.gravity.x;
  gInfo.gravity[1] = prm.gravity.y;
  gInfo.gravity[2] = prm.gravity.z;
  const IsView V{is, im, loopIslands()};
  if (V.o) ++GS.lIslSteps;
  const PxU32 nIsl = V.nbActive();
  S.islandsAll = nIsl;
  PxU32 cur = 0;
  while (cur < nIsl) {
    // PhysX 묶음 하나 (DyTGSDynamics.cpp:720)
    PxU32 nbBodies = 0, nbArt = 0, maxPos = 0, maxVel = 0;
    const uint32_t gStart = uint32_t(S.islands.size());
    const size_t aStart = gInfo.artAlone.size();
    while (nbBodies < prm.solverBatchSize && cur < nIsl && nbArt < prm.solverArticBatchSize) {
      const uint32_t iid = V.activeId(cur);
      nbBodies += V.nodeCount(iid, IG::Node::eRIGID_BODY_TYPE);
      nbArt += V.nodeCount(iid, IG::Node::eARTICULATION_TYPE);
      // 묶음 반복 수: 강체(preIntegrateBodies)·관절체(setupArticulations) 최댓값
      for (PxNodeIndex n = V.root(iid); n.isValid();) {
        PxU16 w = 0;
        if (V.type(n) == IG::Node::eRIGID_BODY_TYPE) w = reinterpret_cast<const PxsRigidBody*>(V.obj(n))->getCore().solverIterationCounts;
        else if (V.type(n) == IG::Node::eARTICULATION_TYPE) w = reinterpret_cast<const Dy::FeatherstoneArticulation*>(V.obj(n))->getIterationCounts();
        maxPos = PxMax<PxU32>(maxPos, w & 0xff);
        maxVel = PxMax<PxU32>(maxVel, w >> 8);
        n = V.next(n);
      }
      // 관절체 단독 그림자 대상: 노드가 관절체 하나뿐, 접촉·조인트 간선 없음
      if (V.nodeCount(iid, IG::Node::eARTICULATION_TYPE) == 1 && V.nodeCount(iid, IG::Node::eRIGID_BODY_TYPE) == 0 &&
          V.firstEdge(iid, IG::Edge::eCONTACT_MANAGER) == IG_INVALID_EDGE && V.firstEdge(iid, IG::Edge::eCONSTRAINT) == IG_INVALID_EDGE)
        gInfo.artAlone.push_back(G1ArtAlone{V.obj(V.root(iid)), 0});
      // 섬 하나를 판 입력으로 (안 되면 되돌림)
      const size_t mB = S.bodies.size(), mIB = S.ib.size(), mICM = S.icm.size(), mCM = S.cms.size(), mP = S.patches.size(),
                   mC = S.contacts.size(), mF = S.friction.size(), mC1 = S.c1d.size(), mIC1 = S.ic1d.size(), mJD = S.jd.size(), mWB = S.wbSeed.size();
      int why = 0;  // 0 = 씀, 1 관절체, 2 운동학, 3 수정 가능 접촉, 4 기타
      auto fail = [&](int w) { if (!why) why = w; };
      if (!GS.full && V.nodeCount(iid, IG::Node::eARTICULATION_TYPE)) fail(1);
      sv::IslandIn I{};
      I.artStart = uint32_t(S.ia.size());
      I.bodyStart = uint32_t(S.ib.size());
      I.cmStart = uint32_t(S.icm.size());
      I.c1dStart = uint32_t(S.ic1d.size());
      I.staticTouchCount = V.staticTouch(iid);
      for (PxNodeIndex n = V.root(iid); !why && n.isValid();) {
        void* nobj = V.obj(n);
        if (V.type(n) == IG::Node::eARTICULATION_TYPE && GS.full) {
          if (!S.artIndex.count(nobj)) {
            S.artIndex[nobj] = uint32_t(S.artFa.size());
            S.artFa.push_back(nobj);
          }
          S.ia.push_back(S.artIndex[nobj]);
        } else if (V.type(n) != IG::Node::eRIGID_BODY_TYPE) fail(1);
        else {
          const PxsRigidBody* rb = reinterpret_cast<const PxsRigidBody*>(nobj);
          const uint32_t bi = uint32_t(S.bodies.size());
          S.rbIndex[rb] = bi;
          S.rbs.push_back(rb);
          S.bodies.push_back(bodyFrom(*rb));
          if (GS.persist) {  // 우리 상태 = 지난 스텝 우리 결과 (옮기지 않은 API 로 건드렸으면 PhysX 로 다시 맞춤)
            const Sc::BodySim* bs = reinterpret_cast<const Sc::BodySim*>(reinterpret_cast<const PxU8*>(rb) - Sc::BodySim::getRigidBodyOffset());
            auto pit = GS.pBodies.find(rb);
            if (pit == GS.pBodies.end()) ++GS.pBodyNew;
            else if (g1_loop_touched(bs->getPxActor())) { ++GS.pBodyResync; GS.pBodies.erase(pit); }
            else {
              if (memcmp(&pit->second, &S.bodies.back(), sizeof(eng::Body))) {
                ++GS.pBodyDiff;
                if (GS.pFirstBody < 0) GS.pFirstBody = (long long)GS.stepSim;
              }
              S.bodies.back() = pit->second;
              ++GS.pBodyUsed;
            }
          }
          S.ib.push_back(bi);
        }
        n = V.next(n);
      }
      I.bodyCount = uint32_t(S.ib.size()) - I.bodyStart;
      I.artCount = uint32_t(S.ia.size()) - I.artStart;
      for (IG::EdgeIndex e = V.firstEdge(iid, IG::Edge::eCONTACT_MANAGER); !why && e != IG_INVALID_EDGE;) {
        PxsContactManager* cm = im.getContactManager(e);
        if (cm) {
          const PxNodeIndex n1 = V.n1(e), n2 = V.n2(e);
          sv::SolverCM m{};
          const PxcNpWorkUnit& u = cm->getWorkUnit();
          auto isArt = [&](PxNodeIndex n) { return !n.isStaticBody() && V.type(n) == IG::Node::eARTICULATION_TYPE; };
          if (n1.isStaticBody() || V.kin(n1) || (!n2.isStaticBody() && V.kin(n2))) fail(2);
          else if (!GS.full && (V.type(n1) != IG::Node::eRIGID_BODY_TYPE ||
                                (!n2.isStaticBody() && V.type(n2) != IG::Node::eRIGID_BODY_TYPE)))
            fail(1);
          else if (GS.full && (isArt(n1) || isArt(n2))) {  // 관절체가 낀 관리자: body = 관절체 번호, artLink = LL 링크 + 1
            auto ref = [&](PxNodeIndex n, uint32_t& body, uint32_t& artLink) {
              if (n.isStaticBody()) { body = sv::NONE; artLink = 0; return; }
              if (isArt(n)) {
                auto it = S.artIndex.find(V.obj(n));
                if (it == S.artIndex.end()) { fail(4); return; }
                body = it->second;
                artLink = n.articulationLinkId() + 1;
              } else {
                auto it = S.rbIndex.find(reinterpret_cast<const PxsRigidBody*>(V.obj(n)));
                if (it == S.rbIndex.end()) { fail(4); return; }
                body = it->second;
                artLink = 0;
              }
            };
            ref(n1, m.body0, m.artLink0);
            ref(n2, m.body1, m.artLink1);
            if (n2.isStaticBody()) m.staticPose1 = toE(u.mRigidCore1->body2World);
          } else {
            const PxsRigidBody* rb0 = reinterpret_cast<const PxsRigidBody*>(V.obj(n1));
            auto i0 = S.rbIndex.find(rb0);
            if (i0 == S.rbIndex.end() || static_cast<const void*>(u.mRigidCore0) != static_cast<const void*>(&rb0->getCore())) fail(4);
            else m.body0 = i0->second;
            if (n2.isStaticBody()) {
              m.body1 = sv::NONE;
              m.staticPose1 = toE(u.mRigidCore1->body2World);
            } else {
              auto i1 = S.rbIndex.find(reinterpret_cast<const PxsRigidBody*>(V.obj(n2)));
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
          if (GS.persist) {
            auto pf = GS.pFric.find({u.getShapeCore0(), u.getShapeCore1()});
            if (pf == GS.pFric.end()) ++GS.pFricNew;
            else {
              bool eq = pf->second.size() == m.frictionCount;
              for (uint32_t q = 0; eq && q < m.frictionCount; ++q) eq = sameFriction(pf->second[q], S.friction[m.frictionPtr + q]);
              if (!eq) {
                if (m.frictionCount == 0) ++GS.pFricReset;  // PhysX 가 지움 (Sc 층: 새로 활성화·관리자 다시 등록 -> 마찰 패치 0, 17.2 규칙)
                else {
                  ++GS.pFricDiff;
                  if (GS.pFirstFric < 0) GS.pFirstFric = (long long)GS.stepSim;
                }
              }
              if (pf->second.size() == m.frictionCount) {  // 수가 같으면 우리 값으로 (다르면 Sc 층 캐시 지움 = 섬·쌍 모듈 몫, 아직 PhysX 값)
                for (uint32_t q = 0; q < m.frictionCount; ++q) S.friction[m.frictionPtr + q] = pf->second[q];
                ++GS.pFricUsed;
              }
            }
          }
          const uint32_t ci = uint32_t(S.cms.size());
          S.cmIndex[cm] = ci;
          S.cmKeys.push_back(cm);
          S.cms.push_back(m);
          S.icm.push_back(ci);
        }
        e = V.nextEdge(e);
      }
      I.cmCount = uint32_t(S.icm.size()) - I.cmStart;
      for (IG::EdgeIndex e = V.firstEdge(iid, IG::Edge::eCONSTRAINT); !why && e != IG_INVALID_EDGE;) {
        const Dy::Constraint* c = im.getConstraint(e);
        if (!c) fail(4);
        else {
          auto bodyOf = [&](const PxsRigidBody* rb, uint32_t& artLink) -> uint32_t {
            artLink = 0;
            if (!rb) return sv::NONE;
            auto it = S.rbIndex.find(rb);
            if (it != S.rbIndex.end()) return it->second;
            const void* fa = nullptr;
            uint32_t ll = 0;
            if (GS.full && g1_art_link_of_rb(rb, &fa, &ll)) {  // 링크 쪽 (보조 잡기 고정 등)
              auto ai = S.artIndex.find(fa);
              if (ai != S.artIndex.end()) {
                artLink = ll + 1;
                return ai->second;
              }
            }
            fail(4);
            return sv::NONE;
          };
          sv::Constraint1DIn x{};
          x.body0 = bodyOf(c->body0, x.artLink0);
          x.body1 = bodyOf(c->body1, x.artLink1);
          if (V.n1(e).isStaticBody() != (c->body0 == nullptr)) fail(4);
          if (V.n2(e).isStaticBody() != (c->body1 == nullptr)) fail(4);
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
        e = V.nextEdge(e);
      }
      I.c1dCount = uint32_t(S.ic1d.size()) - I.c1dStart;
      if (why && GS.full) { S.fullOk = false; S.failWhy = why; }  // 전체 모드: 섬을 빼면 묶음이 달라지므로 이 스텝은 비교하지 않음
      if (why) {  // 섬 되돌림
        S.ia.resize(I.artStart);
        S.bodiesSkipped += V.nodeCount(iid, IG::Node::eRIGID_BODY_TYPE);
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
    for (size_t k = aStart; k < gInfo.artAlone.size(); ++k) gInfo.artAlone[k].iterWord = uint16_t((maxVel << 8) | maxPos);
    if (S.islands.size() > gStart) {
      ++S.batchesUsed;
      S.groups.push_back(Step::Group{gStart, uint32_t(S.islands.size()), uint16_t((maxVel << 8) | maxPos)});
    }
  }
  // 이번 스텝 활성화된 접촉 간선 (쓴 섬의 것만)
  const PxU32 nbAct = V.nbAct();
  for (PxU32 a = 0; a < nbAct; ++a) {
    auto it = S.cmIndex.find(im.getContactManager(V.act(a)));
    if (it != S.cmIndex.end()) S.act.push_back(it->second);
  }
  for (const void* f : S.artFa) {
    const Dy::FeatherstoneArticulation* fa = static_cast<const Dy::FeatherstoneArticulation*>(f);
    S.preArtWake.push_back(fa->mSolverDesc.core->wakeCounter);
    std::vector<float> lw;
    for (PxU32 l = 0; l < fa->mArticulationData.getLinkCount(); ++l) lw.push_back(fa->mArticulationData.mLinks[l].bodyCore->wakeCounter);
    S.preLinkWake.push_back(lw);
  }
  S.valid = true;
  gInfo.valid = true;
  // v1 넘겨받기 확인 (G1_SOLVER_FROM=<엔진 장면 파일>): 파일 경계 simulate 에서, 풀이가 읽는 지난 마찰 패치(PhysX, 풀이 직전)가
  // 파일(다른 실행, simulate 앞에서 뜬 값)과 같은지. 다를 수 있는 것 = simulate 안에서 PhysX 가 지운 관리자(활성화·다시 등록) — 그건 우리 엔진도 스스로 지운다.
  if (const char* fp = getenv("G1_SOLVER_FROM")) {
    static eng::scene::SceneFile ff;
    static int state = 0;  // 0 안 읽음, 1 읽음, 2 실패
    if (state == 0) state = eng::scene::readScene(fp, ff) ? 1 : 2;
    if (state == 1 && GS.stepSim == ff.h.sim) {
      const std::vector<const void*> cores = g1_shape_cores(GS.curScene);
      std::unordered_map<const void*, uint32_t> idx;
      for (uint32_t k = 0; k < cores.size(); ++k) idx[cores[k]] = k;
      std::map<std::pair<uint32_t, uint32_t>, uint32_t> fileCM;
      for (uint32_t k = 0; k < ff.cms.size(); ++k) fileCM[{ff.cms[k].shape0, ff.cms[k].shape1}] = k;
      std::unordered_set<uint32_t> actSet(S.act.begin(), S.act.end());
      uint64_t same = 0, diffReset = 0, diffOther = 0, miss = 0;
      for (uint32_t k = 0; k < S.cms.size(); ++k) {
        const PxcNpWorkUnit& u = S.cmKeys[k]->getWorkUnit();
        auto a = idx.find(u.getShapeCore0()), b = idx.find(u.getShapeCore1());
        auto f = (a == idx.end() || b == idx.end()) ? fileCM.end() : fileCM.find({a->second, b->second});
        if (f == fileCM.end()) { ++miss; continue; }
        const eng::scene::SceneCM& c = ff.cms[f->second];
        const sv::SolverCM& m = S.cms[k];
        bool eq = c.frictionCount == m.frictionCount;
        for (uint32_t q = 0; eq && q < m.frictionCount; ++q) eq = sameFriction(ff.friction[c.frictionStart + q], S.friction[m.frictionPtr + q]);
        if (eq) ++same;
        else if (m.frictionCount == 0 || actSet.count(k)) ++diffReset;
        else ++diffOther;
      }
      // 조인트 되쓰기 칸: 풀이 직전 PhysX 칸(wbSeed) = 파일 칸 (조인트 번호로 맞춤)
      uint64_t wbSame = 0, wbDiff = 0, wbMiss = 0;
      {
        std::unordered_map<uint32_t, uint32_t> byIndex;
        for (uint32_t k = 0; k < ff.joints.size(); ++k) byIndex[ff.joints[k].index] = k;
        for (auto& w : S.wbSeed) {
          auto it = byIndex.find(w.first);
          if (it == byIndex.end() || it->second >= ff.jointWritebacks.size()) { ++wbMiss; continue; }
          (memcmp(&ff.jointWritebacks[it->second], &w.second, sizeof(w.second)) ? wbDiff : wbSame)++;
        }
      }
      printf("G1 solver 넘겨받기 확인: 조인트 되쓰기 칸 같음 %" PRIu64 ", 다름 %" PRIu64 ", 파일에 없음 %" PRIu64 "\n", wbSame, wbDiff, wbMiss);
      printf("G1 solver 넘겨받기 확인: simulate %llu 풀이 입력 관리자 %zu — 파일 마찰 패치와 같음 %" PRIu64 ", 다름(PhysX 가 simulate 안에서 지움) %" PRIu64
             ", 다름(그 밖) %" PRIu64 ", 파일에 없음 %" PRIu64 "\n",
             (unsigned long long)GS.stepSim, S.cms.size(), same, diffReset, diffOther, miss);
    }
  }
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

// ---------------- env 진단 (G1_ENV_FRIC): 풀이 직전 PhysX 접촉 관리자별 지난 마찰 패치 (모양 요소 쌍 -> 패치들)
std::map<uint64_t, std::vector<sv::FrictionPatch>> gEnvFric;
PxScene* gEnvFricScene = nullptr;
void envFricCapture() {
  gEnvFric.clear();
  if (!gEnvFricScene) return;
  Sc::Scene& sc = static_cast<NpScene*>(gEnvFricScene)->getScScene();
  const PxU32 nInter = sc.getNbInteractions(Sc::InteractionType::eOVERLAP);
  Sc::ElementSimInteraction** inter = sc.getInteractions(Sc::InteractionType::eOVERLAP);
  for (PxU32 ii = 0; ii < nInter; ++ii) {
    const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
    if (!cm) continue;
    const PxcNpWorkUnit& u = cm->getWorkUnit();
    const Sc::ElementSimInteraction* ei = inter[ii];
    // 작업 단위 모양 차례 (u 의 모양 0/1) 로 요소 번호를 찾는다
    const Sc::ShapeSimBase& s0 = static_cast<const Sc::ShapeSimBase&>(ei->getElement0());
    const Sc::ShapeSimBase& s1 = static_cast<const Sc::ShapeSimBase&>(ei->getElement1());
    uint32_t e0 = s0.getElementID(), e1 = s1.getElementID();
    if (u.getShapeCore0() != &s0.getCore().getCore()) std::swap(e0, e1);
    std::vector<sv::FrictionPatch>& v = gEnvFric[(uint64_t(e0) << 32) | e1];
    const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
    for (PxU32 k = 0; k < u.mFrictionPatchCount && fp; ++k) v.push_back(frictionFrom(fp[k]));
  }
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
      g1_islands_task(t->getName());
      g1_bp_task(t->getName());
      g1_sc_task(t->getName());
      g1_scene_task(t->getName());
      if (getenv("G1_ENV_FRIC") && !strcmp(t->getName(), "UpdateContinuationTask")) envFricCapture();
      if (getenv("G1_TASKS")) fprintf(stderr, "[task] %s\n", t->getName());
      if (!strcmp(t->getName(), "ScScene.afterIntegration")) takeLateSnapshot();
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
  if (!getenv("G1_SOLVER") && !getenv("G1_ART") && !getenv("G1_ISLANDS") && !getenv("G1_BP") && !getenv("G1_SC") && !getenv("G1_SCENE")) return nullptr;
  static HookDispatcher* d = new HookDispatcher();  // 프로세스 끝까지 (PxPhysics 여러 개가 같이 씀)
  GS.on = true;
  GS.cmpOn = getenv("G1_SOLVER") != nullptr;
  GS.full = getenv("G1_SOLVER_RIGID") == nullptr;
  GS.persist = GS.cmpOn && GS.full && g1_loop_persist();  // 기본 = 관절체 결합 전체 그림자, G1_SOLVER_RIGID=1 이면 예전 관절체 없는 섬만
  return d;
}

const G1StepInfo& g1_step_info() { return gInfo; }
// env 진단: 풀이 직전 PhysX 마찰 패치 표 (G1_ENV_FRIC)
void g1_env_fric_scene(PxScene* s) { gEnvFricScene = s; }
const std::map<uint64_t, std::vector<sv::FrictionPatch>>& g1_env_px_fric() { return gEnvFric; }
bool g1_env_same_fric(const sv::FrictionPatch& a, const sv::FrictionPatch& b) { return sameFriction(a, b); }

// 장면 뜨기(g1_dump.cpp): 풀이 매개변수 (takeSnapshot 과 같은 값)
bool g1_solver_params(PxScene* scene, eng::sv::SolverParams& prm) {
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  Dy::Context* dy = static_cast<Dy::Context*>(sc.getDynamicsContext());
  if (!dy) return false;
  prm = sv::SolverParams{};
  prm.gravity = toV(dy->getGravity());
  prm.dt = dy->getDt();
  prm.enableStabilization = (scene->getFlags() & PxSceneFlag::eENABLE_STABILIZATION);
  prm.bounceThreshold = dy->getBounceThreshold();
  prm.frictionOffsetThreshold = dy->getFrictionOffsetThreshold();
  prm.correlationDistance = dy->getCorrelationDistance();
  prm.solverBatchSize = dy->getSolverBatchSize();
  prm.solverArticBatchSize = dy->getSolverArticBatchSize();
  prm.lengthScale = scene->getPhysics().getTolerancesScale().length;
  prm.solveArticulationContactLast = (scene->getFlags() & PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST);
  return true;
}

// ovd_replay 가 simulate 바로 앞에서 (g1_shadow.cpp 의 g1_before_simulate 를 거쳐)
void g1_solver_before(PxScene* scene, uint64_t sim) {
  if (!GS.on) return;
  GS.curScene = scene;
  GS.stepSim = sim;
  GS.st.valid = false;
  gInfo.valid = false;
}

// fetchResults 뒤: 같은 입력으로 우리 풀이를 돌려 PhysX 결과와 비교
void g1_solver_after(PxScene* scene, uint64_t sim) {
  if (!GS.on || !GS.cmpOn) return;
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
  // 닫힌 고리 2단 (G1_LOOP_CONTACT=1): 접촉 입력(패치·접촉점·작업 단위 값)을 우리 contact 장면 단위(g1_scene.cpp)에서
  if (GS.persist && getenv("G1_LOOP_CONTACT")) {
    std::vector<sv::ContactPatchIn> np;
    std::vector<sv::ContactIn> nc;
    for (size_t k = 0; k < S.cms.size(); ++k) {
      sv::SolverCM& m = S.cms[k];
      const sv::SolverCM* o = nullptr;
      const sv::ContactPatchIn* op = nullptr;
      const sv::ContactIn* oc = nullptr;
      if (!g1_scene_solver_input(S.cmKeys[k]->getIndex(), &o, &op, &oc)) {
        ++GS.lcMissing;
        np.insert(np.end(), S.patches.begin() + m.patchStart, S.patches.begin() + m.patchStart + m.nbPatches);
        nc.insert(nc.end(), S.contacts.begin() + m.contactStart, S.contacts.begin() + m.contactStart + m.nbContacts);
        m.patchStart = uint32_t(np.size() - m.nbPatches);
        m.contactStart = uint32_t(nc.size() - m.nbContacts);
        continue;
      }
      bool same = o->npFlags == m.npFlags && o->nbPatches == m.nbPatches && o->nbContacts == m.nbContacts && !memcmp(&o->restDistance, &m.restDistance, 16);
      for (uint32_t q = 0; same && q < m.nbPatches; ++q)
        same = !memcmp(&op[o->patchStart + q], &S.patches[m.patchStart + q], offsetof(sv::ContactPatchIn, materialIndex1) + 2);
      if (same && m.nbContacts) same = !memcmp(&oc[o->contactStart], &S.contacts[m.contactStart], sizeof(sv::ContactIn) * m.nbContacts);
      if (!same) ++GS.lcDiff;
      ++GS.lcUsed;
      m.npFlags = o->npFlags;
      m.restDistance = o->restDistance;
      m.torsionalPatchRadius = o->torsionalPatchRadius;
      m.minTorsionalPatchRadius = o->minTorsionalPatchRadius;
      m.offsetSlop = o->offsetSlop;
      m.nbPatches = o->nbPatches;
      m.nbContacts = o->nbContacts;
      m.patchStart = uint32_t(np.size());
      m.contactStart = uint32_t(nc.size());
      np.insert(np.end(), op + o->patchStart, op + o->patchStart + o->nbPatches);
      nc.insert(nc.end(), oc + o->contactStart, oc + o->contactStart + o->nbContacts);
    }
    S.patches.swap(np);
    S.contacts.swap(nc);
  }
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
  for (auto& w : S.wbSeed) {
    if (GS.persist && GS.pWbSeen.count(w.first)) {  // 지난 스텝 우리 되쓰기 칸 그대로
      ++GS.pWbUsed;
      if (memcmp(&GS.wbs[w.first], &w.second, sizeof(w.second))) {
        ++GS.pWbDiff;
        if (GS.pFirstWb < 0) GS.pFirstWb = (long long)sim;
      }
      continue;
    }
    GS.wbs[w.first] = w.second;
    if (GS.persist) GS.pWbSeen.insert(w.first);
  }

  // 이번 스텝 섬 관리자가 재운 몸체 (fetchResults 뒤 accurate IslandSim 에 남아 있다)
  std::vector<uint8_t> deactFlag(nb, 0);
  {
    const IG::IslandSim& is = static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager()->getAccurateIslandSim();
    IG::SimpleIslandManager& im2 = *static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager();
    const IsView V{is, im2, loopIslands()};
    const PxU32 nd = V.nbDeact(IG::Node::eRIGID_BODY_TYPE);
    for (PxU32 k = 0; k < nd; ++k) {
      auto it = S.rbIndex.find(reinterpret_cast<const PxsRigidBody*>(V.obj(V.deact(IG::Node::eRIGID_BODY_TYPE, k))));
      if (it != S.rbIndex.end()) deactFlag[it->second] = 1;
    }
  }
  std::vector<uint32_t> cmGroup(S.cms.size(), 0);
  for (uint32_t g = 0; g < S.groups.size(); ++g)
    for (uint32_t i = S.groups[g].islandStart; i < S.groups[g].islandEnd; ++i)
      for (uint32_t k = 0; k < S.islands[i].cmCount; ++k) cmGroup[S.icm[S.islands[i].cmStart + k]] = g;
  if (S.friction.size() > GS.fr0.size()) GS.fr0.resize(S.friction.size());
  const auto& pool = static_cast<Dy::Context*>(static_cast<NpScene*>(scene)->getScScene().getDynamicsContext())->getConstraintWriteBackPool();

  // ---- 전체 그림자 (관절체 결합): 섬 전부를 판 하나로, 관절체는 simulate 앞에 옮겨 담은 쌍둥이(g1_art.cpp)
  if (GS.full) {
    if (!S.fullOk) {
      ++GS.fullSkipped;
      fprintf(stderr, "[g1 solver] sim %llu 전체 그림자 건너뜀: 섬 못 넣음 (이유 %d: 1 관절체 2 운동학 3 수정 가능 접촉 4 기타)\n", (unsigned long long)sim, S.failWhy);
      return;
    }
    const uint32_t na = uint32_t(S.artFa.size());
    std::vector<eng::art::Articulation> arts(na);
    for (uint32_t k = 0; k < na; ++k) {
      eng::art::Articulation* tw = g1_art_twin(S.artFa[k]);
      if (!tw) {
        ++GS.fullSkipped;
        fprintf(stderr, "[g1 solver] sim %llu 전체 그림자 건너뜀: 관절체 쌍둥이 없음 (%s)\n", (unsigned long long)sim, g1_art_name(S.artFa[k]));
        return;
      }
      arts[k] = *tw;
      if (!arts[k].awake) {  // simulate 안에서 깨어남 (Sc 활성화): 깸 카운터는 풀이 직전 PhysX 값
        arts[k].awake = 1;
        arts[k].readyForSleep = 0;
        arts[k].wakeCounter = S.preArtWake[k];
        for (uint32_t l = 0; l < arts[k].nLinks && l < S.preLinkWake[k].size(); ++l) arts[k].bodies[l].wakeCounter = S.preLinkWake[k][l];
        ++GS.artWoken;
      }
    }
    const uint32_t STATIC_CAP = 2048;
    std::vector<eng::art::StaticLists> artLists(na);
    std::vector<sv::SDesc> artS1(size_t(na) * STATIC_CAP), artSC(size_t(na) * STATIC_CAP);
    std::vector<uint32_t> artN1(na), artNC(na), artBatch(na);
    std::vector<sv::ArtProgress> artProg(na);
    std::vector<uint32_t> deact, deactArts;
    for (uint32_t b = 0; b < nb; ++b)
      if (deactFlag[b]) deact.push_back(b);
    {
      const IG::IslandSim& is = static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager()->getAccurateIslandSim();
      IG::SimpleIslandManager& im3 = *static_cast<NpScene*>(scene)->getScScene().getSimpleIslandManager();
      const IsView V{is, im3, loopIslands()};
      const PxU32 nda = V.nbDeact(IG::Node::eARTICULATION_TYPE);
      for (PxU32 k = 0; k < nda; ++k) {
        auto it = S.artIndex.find(V.obj(V.deact(IG::Node::eARTICULATION_TYPE, k)));
        if (it != S.artIndex.end()) deactArts.push_back(it->second);
      }
    }
    sv::SolverBoard B{};
    B.bodies = S.bodies.data();
    B.nbBodies = nb;
    B.cms = S.cms.data();
    B.nbCMs = uint32_t(S.cms.size());
    B.patches = S.patches.data();
    B.contacts = S.contacts.data();
    B.islands = S.islands.data();
    B.nbIslands = uint32_t(S.islands.size());
    B.islandBodies = S.ib.data();
    B.islandCMs = S.icm.data();
    B.activatedCMs = S.act.data();
    B.nbActivatedCMs = uint32_t(S.act.size());
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
    B.frictionCurIdx = 0;
    std::copy(S.friction.begin(), S.friction.end(), GS.fr0.begin());
    B.friction[0] = sv::FrictionArena{GS.fr0.data(), uint32_t(S.friction.size()), uint32_t(GS.fr0.size()), 0};
    B.friction[1] = sv::FrictionArena{GS.fr1.data(), 0, uint32_t(GS.fr1.size()), 0};
    B.corr = GS.corr.get();
    B.contactBuffer = GS.cbuf.data();
    B.arts = arts.data();
    B.nbArts = na;
    B.islandArts = S.ia.data();
    B.artLists = artLists.data();
    B.artStatic1D = artS1.data();
    B.artStaticContact = artSC.data();
    B.artNbStatic1D = artN1.data();
    B.artNbStaticContact = artNC.data();
    B.artStaticCap = STATIC_CAP;
    B.artBatchIndex = artBatch.data();
    B.artProg = artProg.data();
    sv::SolverParams prm = S.prm;
    prm.solveArticulationContactLast = (scene->getFlags() & PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST);
    {
      FtzScope f;
      sv::solverStepHost(B, prm);
      sv::afterIntegrationHost(B);
    }
    // Sc 층 입력 (잠 판정 직전 값): 링크 상호작용 수, 링크·관절체 깸 카운터 올리기
    if (S.lateValid)
      for (uint32_t k = 0; k < na; ++k) {
        eng::art::Articulation& a = arts[k];
        for (uint32_t l = 0; l < a.nLinks && l < S.lateLinkWake[k].size(); ++l) {
          a.bodies[l].numCountedInteractions = S.lateLinkCounted[k][l];
          if (S.lateLinkWake[k][l] > a.bodies[l].wakeCounter) a.bodies[l].wakeCounter = S.lateLinkWake[k][l];
        }
        if (S.lateArtWake[k] > a.wakeCounter) a.wakeCounter = S.lateArtWake[k];
      }
    else
      ++GS.noLate;
    {
      FtzScope f;
      sv::afterIntegrationArtsHost(B, prm.dt, deactArts.data(), uint32_t(deactArts.size()));
      sv::deactivateBodiesHost(B, deact.data(), uint32_t(deact.size()));
    }
    if (B.error) ++GS.engineErr;
    if (g1_loop_persist()) {
      for (uint32_t k = 0; k < na; ++k) g1_art_persist(S.artFa[k], arts[k], sim);
      for (uint32_t i = 0; i < nb; ++i) GS.pBodies[S.rbs[i]] = S.bodies[i];
      if (g1_sc_loop_on()) {  // 2단: 적분 뒤 Sc 칸 갱신 (ScScene.cpp afterIntegration — 얼린 몸체는 안 고침, 잠든 몸체는 되돌린 자세로)
        for (uint32_t i = 0; i < nb; ++i) {
          const eng::Body& b = S.bodies[i];
          const bool frozen = (b.internalFlags & 1u) != 0;  // PxsRigidBody::eFROZEN
          if (frozen) continue;  // updateCached 는 얼린 몸체를 건너뜀 (ScBodySim.cpp:166, ScScene.cpp:180)
          const Sc::BodySim* bs = reinterpret_cast<const Sc::BodySim*>(reinterpret_cast<const PxU8*>(S.rbs[i]) - Sc::BodySim::getRigidBodyOffset());
          g1_sc_update_actor(static_cast<const Sc::ActorSim*>(bs), b.body2World, b.body2Actor, frozen);
        }
        std::unordered_set<uint32_t> artDeact(deactArts.begin(), deactArts.end());
        for (uint32_t k = 0; k < na; ++k) {
          if (artDeact.count(k) || !arts[k].awake) continue;  // 이번에 잠든 관절체는 안 고침 (putToSleep)
          const eng::art::Articulation& a = arts[k];
          for (uint32_t l = 0; l < a.nLinks; ++l) {
            const eng::art::LinkBody& lb = a.bodies[a.ll[l]];
            g1_sc_update_actor(g1_art_link_sim(S.artFa[k], l), lb.body2World, lb.body2Actor, false);
          }
        }
      }
      const sv::FrictionArena& fa2 = B.friction[B.frictionCurIdx];
      for (size_t k = 0; k < S.cms.size(); ++k) {
        const PxcNpWorkUnit& u = S.cmKeys[k]->getWorkUnit();
        const sv::SolverCM& m = S.cms[k];
        GS.pFric[{u.getShapeCore0(), u.getShapeCore1()}].assign(fa2.data + m.frictionPtr, fa2.data + m.frictionPtr + m.frictionCount);
      }
    }
    ++GS.fullSteps;
    GS.cmsSolved += S.cms.size();
    GS.c1dSolved += S.c1d.size();
    for (const sv::SolverCM& m : S.cms) GS.contactsSolved += m.nbContacts;
    for (uint32_t i = 0; i < nb; ++i) {
      const PxsRigidBody& rb = *S.rbs[i];
      const PxsBodyCore& c = rb.getCore();
      const eng::Body& e = S.bodies[i];
      ++GS.bodiesCmp;
      GS.tPose.f(&c.body2World.q.x, &e.body2World.q.x, 7, sim);
      GS.tLin.f(&c.linearVelocity.x, &e.linVel.x, 3, sim);
      GS.tAng.f(&c.angularVelocity.x, &e.angVel.x, 3, sim);
      GS.tWake.f(&c.wakeCounter, &e.wakeCounter, 1, sim);
    }
    for (uint32_t k = 0; k < na; ++k) {
      size_t fj = 0, nf = 0;
      float pv = 0, ev = 0;
      const bool diff = g1_art_diff(S.artFa[k], arts[k], &fj, &nf, &pv, &ev);
      GS.tArt.u(!diff, sim);
      if (diff && GS.show > 0) {
        --GS.show;
        fprintf(stderr, "[g1 solver 관절체 다름] sim %llu %s 칸 %zu/%zu: PhysX %.9g 우리 %.9g\n", (unsigned long long)sim, g1_art_name(S.artFa[k]), fj, nf, pv, ev);
      }
    }
    const sv::FrictionArena& fa = B.friction[B.frictionCurIdx];
    for (size_t k = 0; k < S.cms.size(); ++k) {
      const PxcNpWorkUnit& u = S.cmKeys[k]->getWorkUnit();
      const sv::SolverCM& m = S.cms[k];
      bool same = u.mFrictionPatchCount == m.frictionCount;
      const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
      for (uint32_t q = 0; same && q < m.frictionCount; ++q) same = fp && sameFriction(fa.data[m.frictionPtr + q], frictionFrom(fp[q]));
      GS.tFric.u(same, sim);
    }
    for (const sv::Constraint1DIn& x : S.c1d) GS.tWb.u(!memcmp(&pool[x.index], &GS.wbs[x.index], sizeof(jnt::Writeback)), sim);
    return;
  }

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
  if (!GS.on || !GS.cmpOn) return;
  printf("G1 solver 그림자 (%s, 스텝마다 PhysX 상태로 다시 맞춤): simulate %" PRIu64 " (스냅샷 없음 %" PRIu64 "), 엔진 오류 스텝 %" PRIu64 "\n",
         GS.full ? "관절체 결합 포함 섬 전부" : "관절체 없는 섬", GS.steps, GS.stepsNoSnap, GS.engineErr);
  printf("  활성 섬 %" PRIu64 " 중 푼 섬 %" PRIu64 " (PhysX 묶음 %" PRIu64 " 중 %" PRIu64 "), 뺀 섬: 관절체 묶음 %" PRIu64 ", 운동학 %" PRIu64 ", 수정 가능 접촉 %" PRIu64 ", 기타 %" PRIu64
         ", 뺀 묶음의 강체 %" PRIu64 "\n",
         GS.islandsAll, GS.islandsUsed, GS.batches, GS.batchesUsed, GS.skipArt, GS.skipKin, GS.skipMod, GS.skipOther, GS.bodiesSkipped);
  printf("  푼 것 누적: 몸체 %" PRIu64 ", 접촉 관리자 %" PRIu64 ", 접촉점 %" PRIu64 ", 조인트 %" PRIu64 "\n", GS.bodiesCmp, GS.cmsSolved, GS.contactsSolved, GS.c1dSolved);
  if (GS.full) printf("  전체 그림자(관절체 결합): 푼 스텝 %" PRIu64 ", 못 넣어 건너뛴 스텝 %" PRIu64 ", 늦은 스냅샷 없음 %" PRIu64 "\n", GS.fullSteps, GS.fullSkipped, GS.noLate);
  if (GS.full) printf("  simulate 안에서 깨어난 관절체 %" PRIu64 " 번\n", GS.artWoken);
  if (GS.persist)
    printf("  닫힌 고리 (2b): 우리 상태로 푼 몸체 %" PRIu64 " (처음 %" PRIu64 ", 다시 맞춤 %" PRIu64 ", 우리 != PhysX simulate 안 풀이 직전 %" PRIu64 "%s), 마찰 패치 관리자 %" PRIu64
           " (처음 %" PRIu64 ", Sc 지움 %" PRIu64 ", 그 밖 다름 %" PRIu64 "%s), 조인트 되쓰기 %" PRIu64 " (다름 %" PRIu64 "%s)\n",
           GS.pBodyUsed, GS.pBodyNew, GS.pBodyResync, GS.pBodyDiff, GS.pFirstBody >= 0 ? (" 첫 " + std::to_string(GS.pFirstBody)).c_str() : "", GS.pFricUsed,
           GS.pFricNew, GS.pFricReset, GS.pFricDiff, GS.pFirstFric >= 0 ? (" 첫 " + std::to_string(GS.pFirstFric)).c_str() : "", GS.pWbUsed, GS.pWbDiff,
           GS.pFirstWb >= 0 ? (" 첫 " + std::to_string(GS.pFirstWb)).c_str() : "");
  if (GS.persist && getenv("G1_LOOP_ISLANDS"))
    printf("  닫힌 고리 3단 섬 입력(우리 섬 관리 — 활성 섬·사슬·간선·활성화 간선·잠들 노드): 스텝 %" PRIu64 "%c", GS.lIslSteps, 10);
  if (GS.persist && getenv("G1_LOOP_CONTACT"))
    printf("  닫힌 고리 2단 접촉 입력(우리 contact 장면 단위): 관리자 %" PRIu64 " (PhysX 와 다름 %" PRIu64 ", 우리 쪽에 없음 %" PRIu64 ")\n", GS.lcUsed, GS.lcDiff, GS.lcMissing);
  for (const Tally* t : {&GS.tPose, &GS.tLin, &GS.tAng, &GS.tWake, &GS.tSleep, &GS.tFric, &GS.tWb, &GS.tArt})
    printf("  %-20s 비교 %10" PRIu64 "  비트 다름 %8" PRIu64 "  최대|차| %.3e%s\n", t->name, t->cmp, t->bad, t->maxd,
           t->bad ? ("  첫 다름 simulate " + std::to_string(t->first)).c_str() : "");
}
