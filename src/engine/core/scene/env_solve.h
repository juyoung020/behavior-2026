// env 풀이 자리 (문서 12.3 EnvSolve, 닫힌 고리 4단 E2, 리드): 우리 섬 관리·contact 장면 단위·깸 카운터 표를 solver·articulation 판(SolverBoard)으로 옮겨 푼다.
// replay/g1_solver.cpp takeSnapshot(PhysX 섬 → 판)과 같은 일을 PhysX 없이 우리 상태에서:
//   섬 = 정확 섬 시뮬의 활성 섬 차례, 섬 안 노드 사슬(강체·관절체), 접촉 간선 사슬(간선 → 우리 접촉 관리자 풀 번호), 제약 간선 사슬(→ 장면 조인트)
//   접촉 입력 = contactSolverInput (좁은 단계 칸), 지난 마찰 패치 = 모양 쌍 키로 들고 감(새 관리자는 0)
//   깸 카운터 = 깸 카운터 표(Sc 층 값)를 풀이 앞에 몸체·관절체 칸으로, 풀이 뒤 값을 post 로
//   적분 뒤 = afterIntegrationHost → Sc 칸 갱신(얼린 몸체 빼고) → 섬이 재운 몸체 되돌리기(+Sc 칸) → 관절체 잠 판정/재우기 → 깨어 있는 링크 Sc 칸
// 아직: 운동학 몸체(섬에 있으면 알림만), 판 도중 새 조인트(보조 잡기 — 간선 객체 번호가 장면 조인트 번호가 아님), resetCMs(조인트 끊김).
// 호스트 전용.
#pragma once
#include <xmmintrin.h>

#include <cstdio>
#include <cstdlib>

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/articulation/art_static.h"
#include "core/scene/env_step.h"
#include "core/scene/scene_file.h"
#include "core/solver/solver_io.h"

namespace eng {
namespace scene {

struct EnvFtz {  // PhysX PxSIMDGuard (FTZ·DAZ)
  unsigned old;
  EnvFtz() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~EnvFtz() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

struct EnvSolveImpl : public EnvSolve {
  // ---- 상태 (장면 파일 번호)
  std::vector<Body> bodies;
  std::vector<art::Articulation> arts;
  std::vector<SceneJoint> joints;
  std::vector<SceneActor> sceneActors;
  std::vector<jnt::Writeback> wbs;  // Dy 제약 번호별
  std::unordered_map<uint64_t, std::vector<sv::FrictionPatch>> fric;  // (요소0<<32 | 요소1) -> 지난 마찰 패치
  std::vector<uint64_t> cmPair;  // 풀 번호 -> 지난 스텝 모양 쌍 (~0 = 없음)
  std::vector<uint32_t> cmEpoch;  // 풀 번호 -> 지난 스텝 캐시 지움 세대 (ScPairs ContactManager::cachedStateEpoch: 다르면 마찰 패치 0)
  std::vector<int32_t> bodyOfNode, artOfNode;  // 섬 노드 번호 -> 몸체 / 관절체 (-1)
  // 조인트가 붙은 수 (BodySim::onConstraintAttach -> registerCountedInteraction, ScBodySim.h:159): 몸체별, 관절체 링크별(관절체*kMaxLinks+LL)
  std::vector<uint32_t> jointsOnBody, jointsOnLink;
  std::vector<uint8_t> jointDead;  // 판 도중 지운 조인트 (번호는 그대로 둔다)
  sv::SolverParams prm{};
  // ---- 작업 공간
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
  std::vector<jnt::Row> rowScratch;
  static constexpr uint32_t kStaticCap = 2048;
  std::vector<art::StaticLists> artLists;
  std::vector<sv::SDesc> artS1, artSC;
  std::vector<uint32_t> artN1, artNC, artBatch;
  std::vector<sv::ArtProgress> artProg;
  // ---- 스텝 안 (풀이 → 적분 뒤)
  std::vector<sv::IslandIn> islands;
  std::vector<uint32_t> ib, icm, act, ic1d, ia;
  std::vector<sv::SolverCM> cms;
  std::vector<sv::Constraint1DIn> c1d;
  std::vector<jnt::D6Data> jd;
  contact::SolverInputOut sin;
  std::vector<uint64_t> cmKey;  // 판 관리자(풀 번호) -> 모양 쌍
  sv::SolverBoard B{};
  // ---- 진단: 이번 스텝 풀이에 넣은 지난 마찰 패치 (모양 쌍 -> 패치), G1_ENV_FRIC
  bool keepIn = false;
  std::unordered_map<uint64_t, std::vector<sv::FrictionPatch>> lastIn;
  // ---- 알림
  uint64_t steps = 0, kinInIsland = 0, unknownEdge = 0, unknownNode = 0, engineErr = 0;

  static uint64_t pairKey(int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }
  // 적재 직후: 지금 쌍 관리층 좁은 단계 목록의 관리자를 "지난 스텝부터 있던 것"으로 (파일 마찰 패치를 이어 씀)
  void seedPairs(const ss::ScPairs& P) {
    for (uint32_t s2 = 0; s2 < P.npMain.size(); ++s2) {
      const int32_t ci = P.npMain.cms[s2];
      if (cmPair.size() <= size_t(ci)) cmPair.resize(size_t(ci) + 1, ~0ull);
      const ss::ContactManager& cm = P.cmsData[size_t(ci)];
      cmPair[size_t(ci)] = pairKey(cm.shape0, cm.shape1);
      if (cmEpoch.size() <= size_t(ci)) cmEpoch.resize(size_t(ci) + 1, 0);
      cmEpoch[size_t(ci)] = cm.cachedStateEpoch;
    }
  }

  bool load(const SceneFile& f) {
    bodies = f.bodies;
    arts = f.arts;
    joints = f.joints;
    sceneActors = f.actors;
    uint32_t maxWb = 0;
    for (const SceneJoint& j : f.joints) maxWb = j.index + 1 > maxWb ? j.index + 1 : maxWb;
    wbs.assign(maxWb, jnt::Writeback{});
    for (size_t k = 0; k < f.joints.size() && k < f.jointWritebacks.size(); ++k) wbs[f.joints[k].index] = f.jointWritebacks[k];
    fric.clear();
    for (const SceneCM& c : f.cms) {
      if (c.shape0 >= f.shapeElems.size() || c.shape1 >= f.shapeElems.size()) continue;
      std::vector<sv::FrictionPatch>& v = fric[pairKey(int32_t(f.shapeElems[c.shape0]), int32_t(f.shapeElems[c.shape1]))];
      v.assign(f.friction.begin() + c.frictionStart, f.friction.begin() + c.frictionStart + c.frictionCount);
    }
    jointsOnBody.assign(bodies.size(), 0);
    jointsOnLink.assign(arts.size() * art::kMaxLinks, 0);
    jointDead.assign(joints.size(), 0);
    for (const SceneJoint& j : joints) countJoint(j, 1);
    const IslandSimState& acc = f.islands.accurate;
    bodyOfNode.assign(acc.nodes.size(), -1);
    artOfNode.assign(acc.nodes.size(), -1);
    for (uint32_t id = 0; id < acc.nodes.size(); ++id) {
      const ig::Node& n = acc.nodes[id];
      if (n.flags & ig::N_DELETED) continue;
      if (n.type == ig::eRIGID_BODY_TYPE && n.object < f.actors.size() && f.actors[n.object].kind == kDynamic) bodyOfNode[id] = int32_t(f.actors[n.object].body);
      else if (n.type == ig::eARTICULATION_TYPE && (n.object & 0x80000000u)) artOfNode[id] = int32_t(n.object & 0x7fffffffu);
    }
    if (f.hasSolverPrm) {
      prm = f.solverPrm;
    } else {
      prm = sv::SolverParams{};
      prm.gravity = V3{f.h.gravity[0], f.h.gravity[1], f.h.gravity[2]};
      prm.dt = f.h.dt;
      prm.solverBatchSize = 128;
      prm.solverArticBatchSize = 16;
      prm.lengthScale = f.h.lengthScale;
    }
    // 작업 공간 (g1_solver.cpp 와 같은 용량)
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
    const size_t na = arts.size();
    artLists.assign(na, art::StaticLists{});
    artS1.assign(na * kStaticCap, sv::SDesc{});
    artSC.assign(na * kStaticCap, sv::SDesc{});
    artN1.assign(na, 0);
    artNC.assign(na, 0);
    artBatch.assign(na, 0);
    artProg.assign(na, sv::ArtProgress{});
    return true;
  }

  // 몸체별 붙은 조인트 수 (onConstraintAttach / onConstraintDetach, ScBodySim.cpp:722)
  void countJoint(const SceneJoint& j, int d) {
    for (uint32_t side = 0; side < 2; ++side) {
      uint32_t b = 0, l = 0;
      sceneRef(side ? j.actor1 : j.actor0, b, l);
      if (b == sv::NONE) continue;
      uint32_t& c = l == 0 ? jointsOnBody[b] : jointsOnLink[size_t(b) * art::kMaxLinks + (l - 1)];
      c = uint32_t(int(c) + d);
    }
  }
  // 판 도중 새 조인트 (보조 잡기 등 — joint_lifecycle.h 생성의 풀이 쪽): joints 뒤에 붙이고 번호를 돌려준다. 섬 간선 객체 = 0x80000000 | 번호.
  // 되쓰기 칸은 ConstraintWriteback::initialize (ScConstraintSim.cpp:57) = 0.
  uint32_t addJoint(const SceneJoint& j) {
    const uint32_t k = uint32_t(joints.size());
    joints.push_back(j);
    jointDead.resize(joints.size(), 0);
    if (wbs.size() <= j.index) wbs.resize(size_t(j.index) + 1, jnt::Writeback{});
    wbs[j.index] = jnt::Writeback{};
    countJoint(j, 1);
    return k;
  }
  // 조인트 해제 (섬 간선 removeConnection 과 같이)
  void removeJoint(uint32_t k) {
    if (k >= joints.size() || jointDead[k]) return;
    jointDead[k] = 1;
    countJoint(joints[k], -1);
  }
  int32_t bodyOf(uint32_t node) const { return node < bodyOfNode.size() ? bodyOfNode[node] : -1; }
  int32_t artOf(uint32_t node) const { return node < artOfNode.size() ? artOfNode[node] : -1; }

  // 장면 행위자 -> (판 몸체 번호, LL 링크 + 1)
  void sceneRef(uint32_t actor, uint32_t& body, uint32_t& artLink) const {
    body = sv::NONE;
    artLink = 0;
    if (actor == kNone || actor >= sceneActors.size()) return;
    const SceneActor& a = sceneActors[actor];
    if (a.kind == kDynamic) body = a.body;
    else if (a.kind == kLink && a.body < arts.size()) {
      body = a.body;
      artLink = art::slot(arts[a.body], a.link) + 1;
    }
  }

  void solve(EnvStep& E, HostWake& post) override {
    contact::ContactScene& S = *E.C->S;
    const ss::ScPairs& P = S.pairs;
    ig::IslandManager& M = E.isl->M;
    const ig::IslandSim& A = M.accurate;
    ScScene& sc = *E.sc;
    ++steps;
    // 1. 행위자 -> 판 몸체 표, 상호작용 수, 깸 카운터 (Sc 층 값 -> 몸체 칸)
    const size_t nPa = P.actors.size();
    std::vector<uint32_t> actorBody(nPa, sv::NONE), actorLinkP1(nPa, 0);
    std::vector<px::PxTransform> staticPose(nPa, px::PxTransform(px::PxIdentity));
    std::unordered_map<uint32_t, int32_t> scByActorID;
    for (size_t h = 0; h < sc.actors.size(); ++h)
      if (sc.actors[h].alive) scByActorID[sc.actors[h].actorID] = int32_t(h);
    for (size_t a = 0; a < nPa; ++a) {
      const ss::Actor& X = P.actors[a];
      if (X.isStatic()) {
        auto it = scByActorID.find(X.actorID);
        if (it != scByActorID.end()) staticPose[a] = sc.actors[size_t(it->second)].pose;
        continue;
      }
      const uint32_t node = uint32_t(X.nodeIndex & 0xffffffffu);
      if (X.articulation >= 0) {
        const int32_t k = artOf(node);
        if (k < 0) { ++unknownNode; continue; }
        actorBody[a] = uint32_t(k);
        actorLinkP1[a] = X.linkId + 1;
        art::Articulation& AR = arts[size_t(k)];
        if (X.linkId < AR.nLinks) AR.bodies[X.linkId].numCountedInteractions = X.countedInteractions + jointsOnLink[size_t(k) * art::kMaxLinks + X.linkId];
      } else {
        const int32_t b = bodyOf(node);
        if (b < 0) { ++unknownNode; continue; }
        actorBody[a] = uint32_t(b);
        bodies[size_t(b)].numCountedInteractions = X.countedInteractions + jointsOnBody[size_t(b)];
      }
    }
    for (const HostBodyWake& w : E.wake.bodies) {
      const uint32_t node = uint32_t(w.node & 0xffffffffu);
      if (!w.link) {
        const int32_t b = bodyOf(node);
        if (b >= 0) bodies[size_t(b)].wakeCounter = w.solveWc;
      } else {
        const int32_t k = artOf(node);
        const uint32_t ll = uint32_t(w.node >> 33);
        if (k >= 0 && ll < arts[size_t(k)].nLinks) arts[size_t(k)].bodies[ll].wakeCounter = w.solveWc;
      }
    }
    // 2. 접촉 입력 (풀 번호 칸 배열)
    uint32_t cap = 0;
    for (uint32_t s = 0; s < P.npMain.size(); ++s) cap = uint32_t(P.npMain.cms[s]) + 1 > cap ? uint32_t(P.npMain.cms[s]) + 1 : cap;
    cms.assign(cap, sv::SolverCM{});
    contact::contactSolverInput(S, actorBody.data(), actorLinkP1.data(), staticPose.data(), cms.data(), cap, sin);
    cmKey.assign(cap, ~0ull);
    if (cmPair.size() < cap) cmPair.resize(cap, ~0ull);
    uint32_t nFr = 0;
    // 이번 스텝 새로 만든 관리자 (섬 호출에 실린 관리자 번호: 추가·미리 받은 추가·강체 관리자 표시) -> 마찰 패치 0 (새 PxsContactManager)
    std::vector<uint8_t> fresh(cap, 0);
    for (const IslOp& r : E.live.out) {
      uint32_t c = ig::INVALID_EDGE;
      if (r.op == ISL_ADD_CM || r.op == ISL_SET_RIGID_CM) c = r.b;
      else if (r.op == ISL_ADD_PREALLOC_CM) c = r.c;
      if (c < cap) fresh[c] = 1;
    }
    for (uint32_t s = 0; s < P.npMain.size(); ++s) {
      const int32_t ci = P.npMain.cms[s];
      const ss::ContactManager& cm = P.cmsData[size_t(ci)];
      const uint64_t key = pairKey(cm.shape0, cm.shape1);
      cmKey[size_t(ci)] = key;
      sv::SolverCM& m = cms[size_t(ci)];
      m.frictionPtr = nFr;
      m.frictionCount = 0;
      if (cmEpoch.size() < cap) cmEpoch.resize(cap, 0);
      const bool sameEpoch = cmEpoch[size_t(ci)] == cm.cachedStateEpoch;
      if (cmPair[size_t(ci)] == key && !fresh[size_t(ci)] && sameEpoch) {  // 같은 관리자, 캐시 안 지움 (새 관리자·clearCachedState 는 마찰 패치 0)
        auto it = fric.find(key);
        if (it != fric.end())
          for (const sv::FrictionPatch& fp : it->second) {
            if (nFr < fr0.size()) fr0[nFr++] = fp;
            ++m.frictionCount;
          }
      }
    }
    if (keepIn) {
      lastIn.clear();
      for (uint32_t s2 = 0; s2 < P.npMain.size(); ++s2) {
        const int32_t ci = P.npMain.cms[s2];
        const sv::SolverCM& m = cms[size_t(ci)];
        lastIn[cmKey[size_t(ci)]].assign(fr0.begin() + m.frictionPtr, fr0.begin() + m.frictionPtr + m.frictionCount);
      }
    }
    // 3. 섬 (정확 섬 시뮬 활성 섬 차례)
    islands.clear();
    ib.clear();
    icm.clear();
    ic1d.clear();
    ia.clear();
    c1d.clear();
    jd.clear();
    for (uint32_t k = 0; k < A.activeIslands.size; ++k) {
      const uint32_t iid = A.activeIslands.d[k];
      const ig::Island& isl = A.islands.d[iid];
      sv::IslandIn I{};
      I.bodyStart = uint32_t(ib.size());
      I.artStart = uint32_t(ia.size());
      I.cmStart = uint32_t(icm.size());
      I.c1dStart = uint32_t(ic1d.size());
      I.staticTouchCount = A.islandStaticTouchCount[iid];
      uint32_t guard = 0;
      for (uint32_t n = isl.rootNode; n != ig::INVALID_NODE && n < A.nodes.size && guard++ <= A.nodes.size; n = A.nodes.d[n].nextNode) {
        const ig::Node& nd = A.nodes.d[n];
        if (nd.flags & ig::N_KINEMATIC) { ++kinInIsland; continue; }
        if (nd.type == ig::eARTICULATION_TYPE) {
          const int32_t a = artOf(n);
          if (a < 0) { ++unknownNode; continue; }
          art::Articulation& AR = arts[size_t(a)];
          if (!AR.awake) {  // simulate 안에서 깨어남 (Sc 활성화): 깸 카운터는 표 값 (링크 칸은 위에서)
            AR.awake = 1;
            AR.readyForSleep = 0;
            if (const HostArtWake* w = E.wake.art(n)) AR.wakeCounter = w->wc;
          }
          ia.push_back(uint32_t(a));
        } else {
          const int32_t b = bodyOf(n);
          if (b < 0) { ++unknownNode; continue; }
          ib.push_back(uint32_t(b));
        }
      }
      guard = 0;
      for (uint32_t e = isl.firstEdge[ig::eCONTACT_MANAGER]; e != ig::INVALID_EDGE && e < A.edges.size && guard++ <= A.edges.size; e = A.edges.d[e].nextIslandEdge) {
        const uint32_t ci = M.constraintOrCm[e];
        if (ci == ig::INVALID_EDGE || ci >= cap || cmKey[ci] == ~0ull) {
          if (!unknownEdge && getenv("G1_ENV_TRACE")) fprintf(stderr, "[env solve] 스텝 %llu 모르는 접촉 간선 %u 객체 %x (칸 수 %u)\n", (unsigned long long)steps, e, ci, cap);
          ++unknownEdge;
          continue;
        }
        icm.push_back(ci);
      }
      guard = 0;
      for (uint32_t e = isl.firstEdge[ig::eCONSTRAINT]; e != ig::INVALID_EDGE && e < A.edges.size && guard++ <= A.edges.size; e = A.edges.d[e].nextIslandEdge) {
        const uint32_t o = M.constraintOrCm[e];
        if (o == ig::INVALID_EDGE || !(o & 0x80000000u) || (o & 0x7fffffffu) >= joints.size()) {
          if (!unknownEdge && getenv("G1_ENV_TRACE")) fprintf(stderr, "[env solve] 스텝 %llu 모르는 제약 간선 %u 객체 %x\n", (unsigned long long)steps, e, o);
          ++unknownEdge;
          continue;
        }
        const SceneJoint& j = joints[o & 0x7fffffffu];
        sv::Constraint1DIn x{};
        sceneRef(j.actor0, x.body0, x.artLink0);
        sceneRef(j.actor1, x.body1, x.artLink1);
        x.index = j.index;
        x.data = uint32_t(jd.size());
        x.writeback = j.index;
        x.flags = j.flags;
        x.linBreakForce = j.linBreakForce;
        x.angBreakForce = j.angBreakForce;
        x.minResponseThreshold = j.minResponseThreshold;
        jd.push_back(j.data);
        ic1d.push_back(uint32_t(c1d.size()));
        c1d.push_back(x);
      }
      I.bodyCount = uint32_t(ib.size()) - I.bodyStart;
      I.artCount = uint32_t(ia.size()) - I.artStart;
      I.cmCount = uint32_t(icm.size()) - I.cmStart;
      I.c1dCount = uint32_t(ic1d.size()) - I.c1dStart;
      islands.push_back(I);
    }
    act.clear();
    {
      const ig::Arr<uint32_t>& L = A.activatedEdges[ig::eCONTACT_MANAGER];
      for (uint32_t k = 0; k < L.size; ++k) {
        const uint32_t ci = M.constraintOrCm[L.d[k]];
        if (ci != ig::INVALID_EDGE && ci < cap && cmKey[ci] != ~0ull) act.push_back(ci);
      }
    }
    // 4. 판 + 풀이 + 활성 강체 깸 카운터 확정
    const uint32_t nb = uint32_t(bodies.size());
    bodySolverIndex.assign(nb, 0);
    B = sv::SolverBoard{};
    B.bodies = bodies.data();
    B.nbBodies = nb;
    B.cms = cms.data();
    B.nbCMs = cap;
    B.patches = sin.patches.data();
    B.contacts = sin.contacts.data();
    B.islands = islands.data();
    B.nbIslands = uint32_t(islands.size());
    B.islandBodies = ib.data();
    B.islandCMs = icm.data();
    B.activatedCMs = act.data();
    B.nbActivatedCMs = uint32_t(act.size());
    B.c1d = c1d.data();
    B.nbC1D = uint32_t(c1d.size());
    B.islandC1Ds = ic1d.data();
    B.jointData = jd.data();
    B.writebacks = wbs.data();
    B.rowScratch = rowScratch.data();
    B.vels = vels.data();
    B.txI = txis.data();
    B.datas = datas.data();
    B.poolCap = uint32_t(vels.size());
    B.descs = descs.data();
    B.ordered = ordered.data();
    B.temp = temp.data();
    B.headers = headers.data();
    B.descCap = uint32_t(descs.size());
    B.partitionCounts = partCounts.data();
    B.partitionCap = uint32_t(partCounts.size());
    B.bodySolverIndex = bodySolverIndex.data();
    B.constraints = sv::ByteArena{arenaMem.data(), 0, uint32_t(arenaMem.size()), 0};
    B.frictionCurIdx = 0;
    B.friction[0] = sv::FrictionArena{fr0.data(), nFr, uint32_t(fr0.size()), 0};
    B.friction[1] = sv::FrictionArena{fr1.data(), 0, uint32_t(fr1.size()), 0};
    B.corr = corr.get();
    B.contactBuffer = cbuf.data();
    B.arts = arts.data();
    B.nbArts = uint32_t(arts.size());
    B.islandArts = ia.data();
    B.artLists = artLists.data();
    B.artStatic1D = artS1.data();
    B.artStaticContact = artSC.data();
    B.artNbStatic1D = artN1.data();
    B.artNbStaticContact = artNC.data();
    B.artStaticCap = kStaticCap;
    B.artBatchIndex = artBatch.data();
    B.artProg = artProg.data();
    {
      EnvTimer tm(E.times ? &E.times->solveCore : nullptr);
      EnvFtz f;
      sv::solverStepHost(B, prm);
      sv::afterIntegrationHost(B);
    }
    if (B.error) ++engineErr;
    // 5. 마찰 패치 들고 가기, 풀이 뒤 깸 카운터
    const sv::FrictionArena& fa = B.friction[B.frictionCurIdx];
    for (uint32_t k = 0; k < icm.size(); ++k) {
      const uint32_t ci = icm[k];
      const sv::SolverCM& m = cms[ci];
      fric[cmKey[ci]].assign(fa.data + m.frictionPtr, fa.data + m.frictionPtr + m.frictionCount);
    }
    for (uint32_t ci = 0; ci < cap; ++ci) {
      cmPair[ci] = cmKey[ci];
      if (cmKey[ci] != ~0ull) cmEpoch[ci] = P.cmsData[ci].cachedStateEpoch;
    }
    for (HostBodyWake& w : post.bodies) {
      if (w.link) continue;
      const int32_t b = bodyOf(uint32_t(w.node & 0xffffffffu));
      if (b >= 0) w.solverWc = bodies[size_t(b)].solverWakeCounter;
    }
  }

  void afterIntegration(EnvStep& E, HostWake& post) override {
    const ig::IslandSim& A = E.isl->M.accurate;
    ScScene& sc = *E.sc;
    const ss::ScPairs& P = E.C->S->pairs;
    std::unordered_map<uint64_t, int32_t> scByNode;
    for (size_t h = 0; h < sc.actors.size(); ++h)
      if (sc.actors[h].alive && sc.actors[h].kind != 0) scByNode[sc.actors[h].node] = int32_t(h);
    auto update = [&](uint64_t node, const Tf& b2w, const Tf& b2a, bool frozen) {
      auto it = scByNode.find(node);
      if (it == scByNode.end()) return;
      sc.updateActorCached(it->second, px::PxTransform(px::PxVec3(b2w.p.x, b2w.p.y, b2w.p.z), px::PxQuat(b2w.q.x, b2w.q.y, b2w.q.z, b2w.q.w)),
                           px::PxTransform(px::PxVec3(b2a.p.x, b2a.p.y, b2a.p.z), px::PxQuat(b2a.q.x, b2a.q.y, b2a.q.z, b2a.q.w)), frozen);
    };
    // 강체: 적분 뒤 Sc 칸 (얼린 몸체 빼고), 섬이 재운 몸체는 되돌린 자세로 한 번 더
    std::unordered_map<uint32_t, uint32_t> nodeOfBody;
    for (uint32_t n = 0; n < bodyOfNode.size(); ++n)
      if (bodyOfNode[n] >= 0) nodeOfBody[uint32_t(bodyOfNode[n])] = n;
    for (uint32_t b : ib) {
      const Body& x = bodies[b];
      if (x.internalFlags & 1u) continue;  // PxsRigidBody::eFROZEN
      update(nodeOfBody[b], x.body2World, x.body2Actor, false);
    }
    std::vector<uint32_t> deact;
    {
      const ig::Arr<uint32_t>& D = A.nodesToPutToSleep[ig::eRIGID_BODY_TYPE];
      for (uint32_t k = 0; k < D.size; ++k) {
        const int32_t b = bodyOf(D.d[k]);
        if (b >= 0) deact.push_back(uint32_t(b));
      }
    }
    std::vector<uint32_t> deactArts;
    {
      const ig::Arr<uint32_t>& D = A.nodesToPutToSleep[ig::eARTICULATION_TYPE];
      for (uint32_t k = 0; k < D.size; ++k) {
        const int32_t a = artOf(D.d[k]);
        if (a >= 0) deactArts.push_back(uint32_t(a));
      }
    }
    // 관절체 잠 판정 직전 값: 링크 깸 카운터(Sc 층 깨움이 올린 값)·상호작용 수, 관절체 깸 카운터
    for (const HostBodyWake& w : E.wake.bodies) {
      if (!w.link) continue;
      const int32_t k = artOf(uint32_t(w.node & 0xffffffffu));
      const uint32_t ll = uint32_t(w.node >> 33);
      if (k >= 0 && ll < arts[size_t(k)].nLinks) arts[size_t(k)].bodies[ll].wakeCounter = w.wc;
    }
    for (const HostArtWake& w : E.wake.arts) {
      const int32_t k = artOf(w.node);
      if (k >= 0) arts[size_t(k)].wakeCounter = w.wc;
    }
    for (const ss::Actor& X : P.actors)
      if (X.articulation >= 0) {
        const int32_t k = artOf(uint32_t(X.nodeIndex & 0xffffffffu));
        if (k >= 0 && X.linkId < arts[size_t(k)].nLinks)
          arts[size_t(k)].bodies[X.linkId].numCountedInteractions = X.countedInteractions + jointsOnLink[size_t(k) * art::kMaxLinks + X.linkId];
      }
    {
      EnvFtz f;
      sv::afterIntegrationArtsHost(B, prm.dt, deactArts.data(), uint32_t(deactArts.size()));
      sv::deactivateBodiesHost(B, deact.data(), uint32_t(deact.size()));
    }
    for (uint32_t b : deact) update(nodeOfBody[b], bodies[b].body2World, bodies[b].body2Actor, false);
    // 깨어 있는 관절체 링크 Sc 칸 (이번에 잠든 것은 안 고침)
    for (uint32_t a : ia) {
      bool sleeping = false;
      for (uint32_t d : deactArts) sleeping = sleeping || d == a;
      const art::Articulation& AR = arts[a];
      if (sleeping || !AR.awake) continue;
      uint32_t artNode = ig::INVALID_NODE;
      for (uint32_t n = 0; n < artOfNode.size(); ++n)
        if (artOfNode[n] == int32_t(a)) { artNode = n; break; }
      for (uint32_t l = 0; l < AR.nLinks; ++l) {
        const art::LinkBody& lb = AR.bodies[l];
        update(uint64_t(artNode) | (uint64_t((l << 1) | 1u) << 32), lb.body2World, lb.body2Actor, false);
      }
    }
    // post: 링크·관절체 깸 카운터 (잠 판정 뒤)
    for (HostBodyWake& w : post.bodies) {
      if (!w.link) continue;
      const int32_t k = artOf(uint32_t(w.node & 0xffffffffu));
      const uint32_t ll = uint32_t(w.node >> 33);
      if (k >= 0 && ll < arts[size_t(k)].nLinks) w.wc = arts[size_t(k)].bodies[ll].wakeCounter;
    }
    for (HostArtWake& w : post.arts) {
      const int32_t k = artOf(w.node);
      if (k >= 0) w.wc = arts[size_t(k)].wakeCounter;
    }
  }
};

}  // namespace scene
}  // namespace eng
