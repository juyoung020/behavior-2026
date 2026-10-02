// G1 contact 장면 단위 그림자 (문서 15.3·19.2, 리드, 비계 전용): core/contact/scene_step.h 를 한 줄로 이어 돌린다.
//   우리 넓은 단계(BpRuntime = AABB 관리자 번역본, 첫 simulate 앞 장면 짓기부터의 구조 변경을 PhysX 와 같은 순서로 받음)
//   -> 우리 쌍 관리층(ScPairs, 우리 넓은 단계의 새·사라진 겹침으로)
//   -> 우리 좁은 단계(np_step.h, 지속 다양체는 우리 칸 캐시에서 이어 감)
//   -> solver 입력(12.3) -> 풀이 뒤 정리(우리 사라진 겹침으로).
// 바깥에서 받는 것(다른 모듈 몫): Sc 입력 조각(경계 상자·접촉 거리·바뀜 비트맵 = PhysX 배열, g1_sc 가 우리 식과 비트 같음을 보임),
//   변환 캐시(fetchCollision 뒤 PhysX 값 = g1_shadow), 섬 관리자가 돌려주는 값과 활성화 사건(g1_pairs 가 받아 둔 PhysX 값), 쌍 관리층 앞 입력(g1_pairs).
// 비교: 넓은 단계 새·사라진 겹침 목록, 좁은 단계 목록(관리자 번호·모양), 칸별 출력(상태·패치·접촉점 비트), 닿음 사건, 스텝 끝 좁은 단계 목록.
// 켜기: G1_SCENE=1 (G1_CONTACT·G1_BP·G1_PAIRS 도 켜야 함 — 입력을 그쪽에서 받는다).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <xmmintrin.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include "PxPhysicsAPI.h"
#define private public
#define protected public
#include "GuConvexMesh.h"
#include "NpScene.h"
#include "NpShape.h"
#include "ScScene.h"
#include "ScShapeSim.h"
#include "ScShapeInteraction.h"
#include "PxsContext.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsTransformCache.h"
#include "PxvNphaseImplementationContext.h"
#include "PxsNphaseImplementationContext.h"
#include "BpAABBManager.h"

#include "core/contact/scene_step.h"
#include "core/scene/bp_log.h"
#include "core/scene/pairs_log.h"
#include "core/scene/omni_filter.h"
#include "core/scene/env_runtime.h"
#include "core/scene/env_step.h"  // 한 env 스텝 함수 (G1_ENV)
#include "core/scene/sc_scene.h"
#include "omni_filter.h"
#include "g1_hooks.h"

using namespace physx;
namespace ep = eng::px;
namespace ec = eng::contact;
namespace es = eng::scene;
namespace ss = eng::contact::sc;
namespace sv = eng::sv;

namespace {

struct FtzScope {  // PhysX PxSIMDGuard (FTZ·DAZ)
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};
uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

struct CapOut {
  uint8_t statusFlag = 0, nbPatches = 0;
  uint16_t nbContacts = 0;
  std::vector<uint8_t> patches, contacts;
  // 작업 단위 (solver 입력 대조용)
  uint32_t npFlags = 0;
  float restDistance = 0, torsional = 0, minTorsional = 0, offsetSlop = 0;
  PxTransform pose1;  // mRigidCore1->body2World (정적 쪽 자세)
};
struct SceneShadow {
  bool inited = false, on = false, started = false;
  PxScene* scene = nullptr;
  uint64_t sim = 0;
  // 넓은 단계: 장면 짓기부터 구조 변경을 받는다 (첫 simulate 앞에도)
  std::unique_ptr<es::BpRuntime> rt;
  std::vector<es::BpOp> pendingOps;
  // 넓은 단계 입력 (작업 ScScene.broadPhaseFirstPass 직전 PhysX 배열 복사)
  std::vector<float> bounds, dist;
  std::vector<uint32_t> words;
  bool hasCD = false, boundsChanged = false, haveInput = false, havePxOverlaps = false;
  std::vector<int32_t> pxOverlaps[2][2];
  std::unordered_map<uint32_t, CapOut> pxOut;  // 관리자 번호 -> PhysX 칸 출력 (관리자 없애기 전)  // [모양/트리거][생성/소멸] PhysX AABB 관리자 결과
  // 장면
  std::unique_ptr<ec::ContactScene> S;
  es::PairsHooks H;
  es::PairsHooks* Hp = &H;  // 파일 넘겨받기 뒤에는 env 의 것
  std::unique_ptr<es::EnvContact> env;  // G1_SCENE_FROM 넘겨받기 (거르개·갈고리가 여기 산다)
  std::unique_ptr<es::SceneShared> envShared;
  bool fromDone = false;
  uint32_t fromElems = 0;
  es::OmniFilterSpec spec;
  std::vector<uint32_t> actorBody, actorLinkP1;
  std::vector<ep::PxTransform> actorStaticPose;
  std::vector<sv::SolverCM> solverCms;
  ec::SolverInputOut solverIn;
  std::vector<uint8_t> ourNpStatus, ourNpPatches;  // 순서기 그림자용
  uint64_t ourStep = ~0ull;
  uint64_t cmpSolver = 0, badSolver = 0;
  // 닫힌 고리 2단: Sc 입력 조각을 우리 Sc 장면(g1_sc)에서
  uint64_t lsSteps = 0, lsCells = 0, lsResync = 0, lsBoundsDiff = 0, lsWordDiff = 0, lsCacheDiff = 0, lsDistDiff = 0, lsNonShape = 0, lsAggOurs = 0, lsAggDiff = 0;
  long long lsFirst = -1;
  std::string lsFirstWhat;
  es::OmniFilterCtx fctx;
  bool coreFilter = false;
  // 통계
  uint64_t steps = 0, bad = 0, cmpOverlap = 0, cmpList = 0, cmpSlot = 0, cmpVal = 0, cmpEvents = 0, badOverlap = 0, badList = 0, badSlot = 0, badEvents = 0,
           badEnd = 0, actBad = 0, handleBad = 0, npRan = 0, goneCm = 0;
  long long firstBad = -1;
  std::string firstWhat;
  int show = 0;
} SC;

void fail(const char* what, uint64_t& counter) {
  ++counter;
  ++SC.bad;
  if (SC.firstBad < 0) {
    SC.firstBad = (long long)SC.sim;
    SC.firstWhat = what;
  }
  if (SC.show > 0) {
    --SC.show;
    fprintf(stderr, "[g1 scene 다름] sim %llu %s\n", (unsigned long long)SC.sim, what);
  }
}

void segv(int) {
  void* bt[32];
  const int n = backtrace(bt, 32);
  backtrace_symbols_fd(bt, n, 2);
  _exit(3);
}
void init() {
  if (SC.inited) return;
  SC.inited = true;
  SC.on = getenv("G1_SCENE") != nullptr;
  if (SC.on && getenv("G1_SCENE_SEGV")) signal(SIGSEGV, segv);
  SC.show = getenv("G1_SCENE_SHOW") ? atoi(getenv("G1_SCENE_SHOW")) : 5;
}

// 좁은 단계 모양 자료 (요소 번호별) — 기하는 PhysX 볼록 덩어리를 그대로 가리킴 (g1_shadow 와 같음)
bool geomOf(const PxsShapeCore* c, ec::ShapeGeom& o) {
  memset(static_cast<void*>(&o), 0, sizeof(o));
  const PxGeometry& pg = c->mGeometry.getGeometry();
  switch (pg.getType()) {
    case PxGeometryType::eSPHERE: o.type = ec::eSPHERE; o.sphere = ep::PxSphereGeometry(static_cast<const PxSphereGeometry&>(pg).radius); return true;
    case PxGeometryType::ePLANE: o.type = ec::ePLANE; return true;
    case PxGeometryType::eCAPSULE: {
      const auto& g = static_cast<const PxCapsuleGeometry&>(pg);
      o.type = ec::eCAPSULE; o.capsule = ep::PxCapsuleGeometry(g.radius, g.halfHeight); return true;
    }
    case PxGeometryType::eBOX: {
      const auto& g = static_cast<const PxBoxGeometry&>(pg);
      o.type = ec::eBOX; o.box = ep::PxBoxGeometry(g.halfExtents.x, g.halfExtents.y, g.halfExtents.z); return true;
    }
    case PxGeometryType::eCONVEXMESH: {
      const auto& g = static_cast<const PxConvexMeshGeometry&>(pg);
      ep::PxMeshScale s;
      s.scale = ep::PxVec3(g.scale.scale.x, g.scale.scale.y, g.scale.scale.z);
      s.rotation = ep::PxQuat(g.scale.rotation.x, g.scale.rotation.y, g.scale.rotation.z, g.scale.rotation.w);
      const auto* hull = reinterpret_cast<const ep::Gu::ConvexHullData*>(&static_cast<const Gu::ConvexMesh*>(g.convexMesh)->getHull());
      o.type = ec::eCONVEXMESH; o.convex = ep::PxConvexMeshGeometry(hull, s, g.meshFlags); return true;
    }
    default: return false;
  }
}

void startScene(PxScene* scene) {
  SC.started = true;
  SC.scene = scene;
  SC.S.reset(new ec::ContactScene);
  ec::ContactScene& S = *SC.S;
  const es::PairsLog* L = g1_pairs_log();
  if (!L || !SC.rt) {
    fprintf(stderr, "[g1 scene] G1_PAIRS·G1_BP 가 켜져 있어야 한다\n");
    SC.on = false;
    return;
  }
  if (getenv("G1_SCENE_FILTER") && !strcmp(getenv("G1_SCENE_FILTER"), "core")) {
    // PhysX 형 없는 omni 거르개 (core/scene/omni_filter.h): 표는 재생기가 PhysX 에 준 것에서 옮김
    Sc::Scene& sc0 = static_cast<NpScene*>(scene)->getScScene();
    const engine::FilterSpec* fs = *static_cast<const engine::FilterSpec* const*>(scene->getFilterShaderData());
    SC.spec.groupPairs.assign(fs->group_pairs.begin(), fs->group_pairs.end());
    SC.spec.filteredPairs.assign(fs->filtered_pairs.begin(), fs->filtered_pairs.end());
    SC.spec.invertedGroupFilter = fs->inverted_group_filter;
    SC.spec.anyContactReport = fs->any_contact_report;
    PxSimulationFilterCallback* cb = sc0.getFilterCallbackFast();
    SC.spec.reportAll = cb && static_cast<engine::OmniFilterCallback*>(cb)->report_all;
    SC.spec.sort();
    SC.fctx.spec = &SC.spec;
    es::setOmniFilter(S.pairs, SC.fctx);
    SC.coreFilter = true;
  } else {
    g1_pairs_filters(S.pairs);
  }
  SC.H.m = &S.pairs;
  S.pairs.islands = &SC.H;
  es::pairsStart(S.pairs, *L);
  S.aabb = SC.rt->m.get();
  size_t nm = 0;
  const ec::MaterialData* mats = g1_contact_mats(&nm);
  S.materials.assign(mats, mats + nm);
  S.npParams.toleranceLength = g1_contact_tol();
  S.npParams.meshContactMargin = 0.01f * S.npParams.toleranceLength;
  S.npParams.createAveragePoint = scene->getFlags().isSet(PxSceneFlag::eENABLE_AVERAGE_POINT);
}

// 활성화 사건 재생 (pairs_log.h pairsPost 와 같음: PhysX 섬이 알려 준 값으로)
void replayActs(const es::PairsStep& st, bool afterFill) {
  ss::ScPairs& M = SC.S->pairs;
  for (const es::PairsAct& a : st.acts) {
    if ((a.afterFill != 0) != afterFill) continue;
    const int32_t it = M.findInteraction(a.e0, a.e1);
    if (it < 0) { ++SC.actBad; continue; }
    bool r;
    if (a.activate) {
      SC.Hp->forced = a.result ? 1 : 0;
      r = M.activateInteraction(it);
    } else {
      SC.Hp->forcedDeact = a.result ? 1 : 0;
      r = M.deactivateInteraction(it);
    }
    SC.Hp->forced = -1;
    SC.Hp->forcedDeact = -1;
    if (r != (a.result != 0)) ++SC.actBad;
  }
}

void runStep(PxScene* scene) {
  ec::ContactScene& S = *SC.S;
  ss::ScPairs& M = S.pairs;
  const es::PairsStep* stp = g1_pairs_step();
  if (!stp) return;
  const es::PairsStep& st = *stp;
  ++SC.steps;
  // 1. 앞 입력 (행위자·모양·조인트·장면 변경 연산)
  es::pairsPre(M, *SC.Hp, st);
  // 넓은 단계 구조 변경 (이번 창)
  for (const es::BpOp& o : SC.pendingOps) SC.rt->apply(o);
  SC.pendingOps.clear();
  // 2. 넓은 단계 (PhysX 입력 배열 그대로)
  if (!SC.haveInput) return;
  SC.rt->step(reinterpret_cast<const ep::PxBounds3*>(SC.bounds.data()), uint32_t(SC.bounds.size() / 6), SC.boundsChanged, SC.dist.data(), uint32_t(SC.dist.size()),
              SC.words.data(), uint32_t(SC.words.size()), SC.hasCD);
  auto grab = [&](bool created, ep::Bp::ElementType::Enum type, std::vector<int32_t>& out) {
    out.clear();
    ep::PxU32 n = 0;
    const ep::Bp::AABBOverlap* o = created ? S.aabb->getCreatedOverlaps(type, n) : S.aabb->getDestroyedOverlaps(type, n);
    for (ep::PxU32 i = 0; i < n; ++i) {
      out.push_back(ec::elemOfUser(o[i].mUserData0));
      out.push_back(ec::elemOfUser(o[i].mUserData1));
    }
  };
  grab(true, ep::Bp::ElementType::eSHAPE, S.createdShape);
  grab(true, ep::Bp::ElementType::eTRIGGER, S.createdTrigger);
  grab(false, ep::Bp::ElementType::eSHAPE, S.destroyedShape);
  grab(false, ep::Bp::ElementType::eTRIGGER, S.destroyedTrigger);
  SC.cmpOverlap += S.createdShape.size() / 2 + S.destroyedShape.size() / 2;
  const std::vector<int32_t>& pxC = SC.havePxOverlaps ? SC.pxOverlaps[0][0] : st.created;
  if (S.createdShape != pxC || S.createdTrigger != SC.pxOverlaps[1][0] || S.destroyedShape != SC.pxOverlaps[0][1] || S.destroyedTrigger != SC.pxOverlaps[1][1]) {
    if (SC.show > 0) {
      size_t k = 0;
      while (k < S.createdShape.size() && k < pxC.size() && S.createdShape[k] == pxC[k]) ++k;
      fprintf(stderr, "  새 겹침 우리 %zu / PhysX %zu (트리거 %zu / %zu), 첫 다른 자리 %zu: 우리 %d PhysX %d\n", S.createdShape.size() / 2, pxC.size() / 2,
              S.createdTrigger.size() / 2, st.createdTrigger.size() / 2, k / 2, k < S.createdShape.size() ? S.createdShape[k] : -9,
              k < pxC.size() ? pxC[k] : -9);
      for (size_t q = 0; q < 16 && q < S.createdShape.size() && q < pxC.size(); q += 2)
        fprintf(stderr, "    %zu: 우리 (%d,%d) PhysX (%d,%d)\n", q / 2, S.createdShape[q], S.createdShape[q + 1], pxC[q], pxC[q + 1]);
    }
    fail("넓은 단계 겹침 목록 (생성·소멸, AABB 관리자 출력 순서)", SC.badOverlap);
  }
  {
    std::vector<int32_t> ours = S.destroyedShape;
    ours.insert(ours.end(), S.destroyedTrigger.begin(), S.destroyedTrigger.end());
    std::multiset<std::pair<int32_t, int32_t>> a, b;
    for (size_t i = 0; i + 1 < ours.size(); i += 2) a.insert({ours[i], ours[i + 1]});
    for (size_t i = 0; i + 1 < st.removedPairs.size(); i += 2) b.insert({st.removedPairs[i], st.removedPairs[i + 1]});
    if (a != b) fail("넓은 단계 사라진 겹침", SC.badOverlap);
  }
  // 3. 쌍 관리 (섬이 돌려줄 값 = PhysX 기록)
  SC.Hp->prealloc = &st.preallocHandles;
  SC.Hp->addCm = &st.addCmEdges;
  SC.Hp->preallocPos = SC.Hp->addCmPos = 0;
  ec::contactPairs(S);
  replayActs(st, false);
  // 4. 좁은 단계 (모양 자료·변환 캐시·접촉 거리)
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  size_t ncache = 0;
  const PxsCachedTransform* cache = g1_contact_cache(&ncache);
  // 칸 배치가 달라(PhysX PxTransform 28 바이트, 우리 PxTransform32 32 바이트) 옮겨 담는다
  static std::vector<ec::CachedTransform> tc;
  tc.resize(ncache);
  for (size_t k = 0; k < ncache; ++k) {
    const PxTransform& t = cache[k].transform;
    tc[k].transform = ep::PxTransform32(ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)));
    tc[k].flags = cache[k].flags;
  }
  if (es::ScScene* E = g1_sc_scene()) {  // 2단: 좁은 단계 변환 캐시도 우리 Sc 칸 (모양 칸만)
    for (size_t k = 0; k < ncache && k < E->shapes.size(); ++k) {
      if (!E->shapes[k].alive) continue;
      const ep::PxTransform& t = E->cache[k];
      const float a7[7] = {t.q.x, t.q.y, t.q.z, t.q.w, t.p.x, t.p.y, t.p.z};
      const float b7[7] = {cache[k].transform.q.x, cache[k].transform.q.y, cache[k].transform.q.z, cache[k].transform.q.w, cache[k].transform.p.x,
                           cache[k].transform.p.y, cache[k].transform.p.z};
      if (memcmp(a7, b7, 28) || E->cacheFlags[k] != cache[k].flags) {
        ++SC.lsCacheDiff;
        if (SC.lsFirst < 0) { SC.lsFirst = (long long)SC.sim; SC.lsFirstWhat = "변환 캐시"; }
      }
      tc[k].transform = ep::PxTransform32(t);
      tc[k].flags = E->cacheFlags[k];
    }
  }
  {  // 재질 표 (g1_shadow 가 첫 simulate 앞에서 채움 -> 스텝마다 옮겨 담음, 판 도중 새 재질 대비)
    size_t nm = 0;
    const ec::MaterialData* mats = g1_contact_mats(&nm);
    if (!SC.env) S.materials.assign(mats, mats + nm);
  }
  if (S.npShapes.size() < ncache) S.npShapes.resize(ncache);
  for (size_t e = 0; e < M.shapes.size() && e < ncache; ++e) {
    if (e < SC.fromElems) continue;  // 파일 넘겨받기: 파일 기하 그대로 (뒤에 새로 생긴 모양만 PhysX 에서)
    Sc::ShapeSim* ss2 = g1_elem_sim(int32_t(e));
    if (!ss2) continue;
    const PxsShapeCore& pc = ss2->getCore().getCore();
    geomOf(&pc, S.npShapes[e].geom);
    S.npShapes[e].material = pc.mMaterialIndex;
    if (pc.mMaterialIndex >= S.materials.size()) fail("재질 번호가 표 밖", SC.badSlot);
  }
  {
    // 입력 크기 점검 (칸 캐시·모양·변환 캐시·접촉 거리가 목록이 가리키는 번호를 덮는지)
    uint32_t maxShape = 0;
    for (int nl = 0; nl < 2; ++nl) {
      const ss::NpList& L = nl ? M.npNew : M.npMain;
      if (S.caches.L[nl].size() < L.size()) {
        fail("칸 캐시가 목록보다 작음", SC.badSlot);
        return;
      }
      for (uint32_t k = 0; k < L.size(); ++k) {
        const ss::ContactManager& cm = M.cmsData[size_t(L.cms[k])];
        maxShape = std::max(maxShape, uint32_t(std::max(cm.shape0, cm.shape1)));
      }
    }
    if (maxShape >= tc.size() || maxShape >= SC.dist.size() || maxShape >= S.npShapes.size()) {
      if (SC.show > 0)
        fprintf(stderr, "  모양 번호 %u, 변환 캐시 %zu, 접촉 거리 %zu, 모양 자료 %zu\n", maxShape, tc.size(), SC.dist.size(), S.npShapes.size());
      fail("좁은 단계 입력 크기", SC.badSlot);
      return;
    }
    FtzScope fz;
    ec::contactNarrowPhase(S, tc.data(), SC.dist.data());
    // 순서기 그림자(g1_host)가 쓰는 우리 좁은 단계 결과: 합친 뒤 목록 칸 순서 (PhysX fillManagerTouchEvents 때 뜬 것과 같은 자리)
    SC.ourNpStatus.clear();
    SC.ourNpPatches.clear();
    for (uint32_t slot = 0; slot < M.npMain.size(); ++slot) {
      SC.ourNpStatus.push_back(S.caches.L[0][slot].out.statusFlag);
      SC.ourNpPatches.push_back(S.caches.L[0][slot].out.nbPatches);
    }
    SC.ourStep = SC.sim;
  }
  // 닿음 사건·좁은 단계 목록·칸 출력 비교 (PhysX 관리자 번호 -> 출력)
  {
    // PhysX 칸 출력: 관리자 없애기 전(작업 ScScene.processLostContact 직전)에 떠 둔 것
    const std::unordered_map<uint32_t, CapOut>& pxOut = SC.pxOut;
    for (uint32_t slot = 0; slot < M.npMain.size(); ++slot) {
      ++SC.cmpSlot;
      const int32_t ci = M.npMain.cms[slot];
      auto it = pxOut.find(uint32_t(ci));
      if (it == pxOut.end()) { ++SC.goneCm; continue; }  // PhysX 에 없는 관리자
      const CapOut& po = it->second;
      const ec::NpSlotOutput<32, 256>& o = S.caches.L[0][slot].out;
      bool same = po.nbPatches == o.nbPatches && po.nbContacts == o.nbContacts &&
                  (po.statusFlag & PxsContactManagerStatusFlag::eTOUCH_KNOWN) == (o.statusFlag & ec::NpStatus::eTOUCH_KNOWN);
      if (same && po.nbPatches) {
        ++SC.cmpVal;
        // 패치는 채움(pad) 5 칸을 뺀 앞 54 바이트 (PhysX 는 채움을 안 씀), 접촉점은 통째로
        const uint8_t* pp = po.patches.data();
        for (uint32_t k = 0; same && k < po.nbPatches; ++k)
          same = !memcmp(pp + 64 * k, &o.stream.patches[k], offsetof(ec::ContactPatch, pad));
        same = same && !memcmp(po.contacts.data(), &o.stream.contacts[0], sizeof(ec::Contact) * po.nbContacts);
      }
      if (!same && SC.show > 0) {
        const ss::ContactManager& cmd = M.cmsData[size_t(ci)];
        fprintf(stderr, "  칸 %u 관리자 %d 모양 (%d,%d) 상태 PhysX %x 우리 %x 패치 %u/%u 점 %u/%u np %x 플래그 %x\n", slot, ci, cmd.shape0, cmd.shape1, po.statusFlag,
                o.statusFlag, po.nbPatches, o.nbPatches, po.nbContacts, o.nbContacts, cmd.npIndex, cmd.wuFlags);
        if (po.nbContacts && o.nbContacts) {
          const PxContact* pc2 = reinterpret_cast<const PxContact*>(po.contacts.data());
          fprintf(stderr, "    첫 점 PhysX %.9g %.9g %.9g s %.9g / 우리 %.9g %.9g %.9g s %.9g\n", pc2[0].contact.x, pc2[0].contact.y, pc2[0].contact.z, pc2[0].separation,
                  o.stream.contacts[0].contact.x, o.stream.contacts[0].contact.y, o.stream.contacts[0].contact.z, o.stream.contacts[0].separation);
        }
      }
      if (!same) fail("좁은 단계 칸 출력", SC.badSlot);
    }
  }
  // 4b. solver 입력 조립 (contactSolverInput, 12.3) — PhysX 작업 단위·출력과 대조
  {
    const size_t na = M.actors.size();
    SC.actorBody.assign(na, sv::NONE);
    SC.actorLinkP1.assign(na, 0);
    SC.actorStaticPose.assign(na, ep::PxTransform(ep::PxIdentity));
    for (size_t a = 0; a < na; ++a) {
      const ss::Actor& A = M.actors[a];
      if (A.isStatic()) {
        if (const PxActor* px = g1_pairs_actor(int32_t(a))) {
          const PxTransform t = static_cast<const PxRigidActor*>(px)->getGlobalPose();
          SC.actorStaticPose[a] = ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w));
        }
        continue;
      }
      SC.actorBody[a] = A.articulation >= 0 ? uint32_t(A.articulation) : uint32_t(a);
      SC.actorLinkP1[a] = A.articulation >= 0 ? A.linkId + 1 : 0;
    }
    uint32_t cap = 0;
    for (uint32_t slot = 0; slot < M.npMain.size(); ++slot) cap = std::max(cap, uint32_t(M.npMain.cms[slot]) + 1);
    SC.solverCms.assign(cap, sv::SolverCM{});
    ec::contactSolverInput(S, SC.actorBody.data(), SC.actorLinkP1.data(), SC.actorStaticPose.data(), SC.solverCms.data(), cap, SC.solverIn);
    for (uint32_t slot = 0; slot < M.npMain.size(); ++slot) {
      const int32_t ci = M.npMain.cms[slot];
      auto it = SC.pxOut.find(uint32_t(ci));
      if (it == SC.pxOut.end()) continue;
      const CapOut& po = it->second;
      const sv::SolverCM& m = SC.solverCms[size_t(ci)];
      ++SC.cmpSolver;
      const bool touch = (po.statusFlag & PxsContactManagerStatusFlag::eHAS_TOUCH) != 0;
      bool same = m.npFlags == po.npFlags && fb(m.restDistance) == fb(po.restDistance) && fb(m.torsionalPatchRadius) == fb(po.torsional) &&
                  fb(m.minTorsionalPatchRadius) == fb(po.minTorsional) && fb(m.offsetSlop) == fb(po.offsetSlop) &&
                  m.nbPatches == (touch ? po.nbPatches : 0u) && m.nbContacts == (touch ? po.nbContacts : 0u);
      if (same && m.body1 == sv::NONE) {
        const float a7[7] = {m.staticPose1.q.x, m.staticPose1.q.y, m.staticPose1.q.z, m.staticPose1.q.w, m.staticPose1.p.x, m.staticPose1.p.y, m.staticPose1.p.z};
        const float b7[7] = {po.pose1.q.x, po.pose1.q.y, po.pose1.q.z, po.pose1.q.w, po.pose1.p.x, po.pose1.p.y, po.pose1.p.z};
        same = !memcmp(a7, b7, 28);
      }
      if (same) {
        for (uint32_t k = 0; same && k < m.nbPatches; ++k)
          same = !memcmp(&SC.solverIn.patches[m.patchStart + k], po.patches.data() + 64 * k, offsetof(sv::ContactPatchIn, materialIndex1) + 2);  // 끝 채움 2 바이트 빼고
        same = same && (m.nbContacts == 0 || !memcmp(&SC.solverIn.contacts[m.contactStart], po.contacts.data(), sizeof(sv::ContactIn) * m.nbContacts));
      }
      if (!same && SC.show > 0) {
        fprintf(stderr, "  solver 입력 관리자 %d: 플래그 %x/%x 쉼 %g/%g 비틀 %g/%g 최소 %g/%g 느슨 %g/%g 패치 %u/%u 점 %u/%u 몸체1 %u\n", ci, m.npFlags, po.npFlags,
                m.restDistance, po.restDistance, m.torsionalPatchRadius, po.torsional, m.minTorsionalPatchRadius, po.minTorsional, m.offsetSlop, po.offsetSlop,
                m.nbPatches, po.nbPatches, m.nbContacts, po.nbContacts, m.body1);
      }
      if (!same) fail("solver 입력 (contactSolverInput)", SC.badSolver);
    }
  }
  // 5. 풀이 뒤 정리 (섬 3차 자리는 비움: 섬 모듈 몫) + 6. 활성화(채운 뒤)
  ec::contactPostSolve(S, [] {});
  replayActs(st, true);
  // 스텝 끝 좁은 단계 목록 = PhysX
  {
    PxsContext* ctx = sc.getLowLevelContext();
    const PxsContactManagers& L = static_cast<PxsNphaseImplementationContext*>(ctx->getNphaseImplementationContext())->mNarrowPhasePairs;
    bool same = L.mContactManagerMapping.size() == M.npMain.size();
    for (PxU32 i = 0; same && i < M.npMain.size(); ++i) same = int32_t(L.mContactManagerMapping[i]->getIndex()) == M.npMain.cms[i];
    ++SC.cmpList;
    if (!same) fail("스텝 끝 좁은 단계 목록", SC.badEnd);
  }
  SC.rt->endStep();
  S.pool.clear();
}

}  // namespace

// ---- 넓은 단계 구조 변경 (g1_bp.cpp 가로채기가 모든 연산을 넘김). 사용자 자료 = userOfElem(요소 번호) (scene_step.h 규약)
void g1_scene_bp_created(const eng::scene::BpLog& hdr) {
  init();
  if (!SC.on) return;
  SC.rt.reset(new es::BpRuntime);
  SC.rt->create(hdr.abpMaxOverlaps, hdr.abpMaxStatic, hdr.abpMaxDynamic, hdr.ctx, hdr.abpMT != 0, hdr.maxAggregates, hdr.maxShapes, hdr.kineKine, hdr.staticKine);
  SC.pendingOps.clear();
  SC.started = false;
}
void g1_scene_note_bpop(const eng::scene::BpOp& o) {
  init();
  if (!SC.on || !SC.rt) return;
  es::BpOp x = o;
  if (x.type == es::BP_ADD || x.type == es::BP_CREATE_AGG) x.userData = uint64_t(uintptr_t(ec::userOfElem(x.index)));
  if (!SC.started) {
    SC.rt->apply(x);  // 장면 짓기 (첫 simulate 앞)
    return;
  }
  SC.pendingOps.push_back(x);
}
void g1_scene_task(const char* name) {
  if (!SC.on || !SC.started || !SC.scene) return;
  Sc::Scene& sc = static_cast<NpScene*>(SC.scene)->getScScene();
  if (!strcmp(name, "ScScene.postBroadPhaseCont")) {  // PhysX 넓은 단계 결과 (사용자 자료 = ShapeSim -> 요소 번호)
    Bp::AABBManager* am = static_cast<Bp::AABBManager*>(sc.getAABBManager());
    for (int t = 0; t < 2; ++t)
      for (int w = 0; w < 2; ++w) {
        std::vector<int32_t>& out = SC.pxOverlaps[t][w];
        out.clear();
        PxU32 n = 0;
        const Bp::AABBOverlap* o = w == 0 ? am->getCreatedOverlaps(Bp::ElementType::Enum(t), n) : am->getDestroyedOverlaps(Bp::ElementType::Enum(t), n);
        for (PxU32 k = 0; k < n; ++k) {
          out.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(o[k].mUserData0)->getElementID()));
          out.push_back(int32_t(reinterpret_cast<const Sc::ElementSim*>(o[k].mUserData1)->getElementID()));
        }
      }
    SC.havePxOverlaps = true;
    return;
  }
  if (!strcmp(name, "ScScene.processLostContact")) {  // 좁은 단계 결과 (관리자 없애기 전)
    SC.pxOut.clear();
    PxsContactManagerOutputIterator outputs = sc.getLowLevelContext()->getNphaseImplementationContext()->getContactManagerOutputs();
    const PxU32 nInter = sc.getNbInteractions(Sc::InteractionType::eOVERLAP);
    Sc::ElementSimInteraction** inter = sc.getInteractions(Sc::InteractionType::eOVERLAP);
    for (PxU32 ii = 0; ii < nInter; ++ii) {
      const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
      if (!cm) continue;
      const PxsContactManagerOutput& po = outputs.getContactManagerOutput(cm->getWorkUnit().mNpIndex);
      CapOut& c = SC.pxOut[cm->getIndex()];
      c.statusFlag = po.statusFlag;
      c.nbPatches = po.nbPatches;
      c.nbContacts = po.nbContacts;
      c.patches.assign(po.contactPatches, po.contactPatches + 64 * size_t(po.nbPatches));
      c.contacts.assign(po.contactPoints, po.contactPoints + 16 * size_t(po.nbContacts));
      const PxcNpWorkUnit& u = cm->getWorkUnit();
      c.npFlags = u.mFlags;
      c.restDistance = u.mRestDistance;
      c.torsional = u.mTorsionalPatchRadius;
      c.minTorsional = u.mMinTorsionalPatchRadius;
      c.offsetSlop = u.mOffsetSlop;
      c.pose1 = u.mRigidCore1 ? u.mRigidCore1->body2World : PxTransform(PxIdentity);
    }
    return;
  }
  if (strcmp(name, "ScScene.broadPhaseFirstPass")) return;
  Bp::BoundsArray& pb = sc.getBoundsArray();
  PxFloatArrayPinnedSafe& pd = *sc.mContactDistance;
  PxBitMapPinned& pc = sc.getAABBManager()->getChangedAABBMgActorHandleMap();
  SC.bounds.assign(reinterpret_cast<const float*>(pb.begin()), reinterpret_cast<const float*>(pb.begin()) + 6 * size_t(pb.size()));
  SC.dist.assign(pd.begin(), pd.begin() + pd.size());
  SC.words.assign(pc.getWords(), pc.getWords() + pc.getWordCount());
  SC.hasCD = sc.mHasContactDistanceChanged;
  SC.boundsChanged = pb.hasChanged();
  SC.haveInput = true;
  if (es::ScScene* E = g1_sc_scene()) {  // 2단: 우리 Sc 칸으로 (건드린 행위자 모양만 PhysX 로 다시 맞춤), PhysX 와 비교
    ++SC.lsSteps;
    auto lsBad = [&](const char* w) {
      if (SC.lsFirst < 0) { SC.lsFirst = (long long)SC.sim; SC.lsFirstWhat = w; }
    };
    PxsTransformCache& ptc = sc.getLowLevelContext()->getTransformCache();
    const size_t nb = pb.size();
    std::vector<uint32_t> w(SC.words.size(), 0u);
    for (size_t e = 0; e < nb; ++e) {
      const bool pxBit = e / 32 < SC.words.size() && (SC.words[e / 32] >> (e & 31)) & 1u;
      const bool isShape = e < E->shapes.size() && E->shapes[e].alive;
      if (!isShape) {  // 집합체 칸 (Sc 모양이 아님): AABB 관리자가 스스로 셈 -> 우리 넓은 단계가 지난 스텝에 든 값 (G1_LOOP_AGG_PX=1 이면 PhysX 값)
        ++SC.lsNonShape;
        if (!getenv("G1_LOOP_AGG_PX") && SC.rt && e < SC.rt->bounds->size()) {
          const float* ob = reinterpret_cast<const float*>(&SC.rt->bounds->begin()[e]);
          if (memcmp(ob, &SC.bounds[6 * e], 24)) ++SC.lsAggDiff;
          memcpy(&SC.bounds[6 * e], ob, 24);
          ++SC.lsAggOurs;
        }
        if (pxBit) w[e / 32] |= 1u << (e & 31);
        continue;
      }
      ++SC.lsCells;
      const PxActor* ax = g1_sc_actor_px(E->shapes[e].actor);
      bool touched = ax && g1_loop_touched(ax);
      if (ax && !touched)
        if (const PxArticulationLink* lk = ax->is<PxArticulationLink>()) touched = g1_loop_touched(&lk->getArticulation());
      const bool ourBit = e / 32 < E->changed.size() && (E->changed[e / 32] >> (e & 31)) & 1u;
      if (touched || SC.lsSteps == 1) {  // 첫 스텝 = 넘겨받기 (장면 짓기 뒤 더럽힘 처리까지 끝난 PhysX 칸)  // 옮기지 않은 API (자세 set 등): 모양 더럽힘 -> PhysX 가 simulate 안에서 다시 계산한 값
        if (touched) ++SC.lsResync;
        memcpy(&E->bounds[e], &SC.bounds[6 * e], 24);
        if (e < SC.dist.size()) E->contactDist[e] = SC.dist[e];
        const PxTransform& t = ptc.getTransformCache(PxU32(e)).transform;
        E->cache[e] = ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w));
        E->cacheFlags[e] = ptc.getTransformCache(PxU32(e)).flags;
        if (pxBit) w[e / 32] |= 1u << (e & 31);
        continue;
      }
      if (memcmp(&E->bounds[e], &SC.bounds[6 * e], 24)) { ++SC.lsBoundsDiff; lsBad("경계 상자"); }
      if (ourBit != pxBit) { ++SC.lsWordDiff; lsBad("바뀜 비트"); }
      if (e < SC.dist.size() && memcmp(&E->contactDist[e], &SC.dist[e], 4)) { ++SC.lsDistDiff; lsBad("접촉 거리"); }
      memcpy(&SC.bounds[6 * e], &E->bounds[e], 24);
      if (e < SC.dist.size()) SC.dist[e] = E->contactDist[e];
      if (ourBit) w[e / 32] |= 1u << (e & 31);
    }
    SC.words.swap(w);
    E->changed.assign(E->changed.size(), 0u);  // 넓은 단계가 씀 -> 비움
  }
}
void g1_scene_before(PxScene* scene, uint64_t sim) {
  init();
  if (!SC.on) return;
  SC.sim = sim;
  SC.haveInput = false;
  SC.havePxOverlaps = false;
  if (!SC.started) startScene(scene);
  // 파일 넘겨받기 (G1_SCENE_FROM=<엔진 장면 파일>): 경계 simulate 에서 판 런타임(env_runtime.h)으로 모든 contact 층 상태를 다시 세워 이어 간다
  if (SC.on && SC.started && !SC.fromDone) {
    static es::SceneFile ff;
    static int state = 0;
    if (state == 0) {
      state = 2;
      if (const char* p = getenv("G1_SCENE_FROM")) {
        std::string err;
        state = es::readScene(p, ff, &err) ? 1 : 2;
        if (state == 2) fprintf(stderr, "[g1 scene] 파일 읽기 실패: %s\n", err.c_str());
      }
    }
    if (state == 1 && sim == ff.h.sim) {
      SC.fromDone = true;
      SC.envShared = es::makeShared(ff);
      SC.env.reset(new es::EnvContact);
      std::string err;
      if (!SC.env->load(ff, *SC.envShared, &err)) {
        fprintf(stderr, "[g1 scene] 판 런타임 세우기 실패: %s\n", err.c_str());
        SC.on = false;
        return;
      }
      if (const char* neg = getenv("G1_SCENE_FROM_NEG")) {  // 음성 대조: 1 = 지속 다양체 비움, 2 = 거르개 표 비움 (이러면 달라져야 한다)
        ec::ContactScene& C = *SC.env->S;
        if (atoi(neg) == 1) {
          for (int nl = 0; nl < 2; ++nl) {
            const ss::NpList& L = nl ? C.pairs.npNew : C.pairs.npMain;
            for (uint32_t slot = 0; slot < L.size(); ++slot) {
              const ss::ContactManager& cm = C.pairs.cmsData[size_t(L.cms[slot])];
              const int g0 = cm.geomType0, g1 = cm.geomType1;
              ec::initManifold(C.caches.L[nl][slot].man, g0 < g1 ? g0 : g1, g0 < g1 ? g1 : g0);
            }
          }
        } else {
          SC.env->spec.groupPairs.clear();
          SC.env->spec.filteredPairs.clear();
        }
      }
      SC.S = std::move(SC.env->S);
      SC.rt = std::move(SC.env->bp);
      SC.Hp = &SC.env->H;
      SC.fromElems = uint32_t(SC.S->npShapes.size());
      SC.pendingOps.clear();  // 이 창의 구조 변경은 파일 기록에 이미 있다
      SC.cmpSolver = SC.badSolver = 0;
      SC.steps = SC.bad = SC.cmpOverlap = SC.cmpList = SC.cmpSlot = SC.cmpVal = SC.badOverlap = SC.badSlot = SC.badEnd = SC.actBad = SC.goneCm = 0;
      SC.firstBad = -1;
      printf("G1 contact 장면 넘겨받기: simulate %llu 에서 파일로 판 런타임을 세움 (쌍 기록 스텝 %zu, 넓은 단계 구조 변경 %zu, 다양체 %u 넣음 / 못 찾음 %u, 활성화 재생 어긋남 %u) — 이 뒤로만 비교\n",
             (unsigned long long)sim, ff.pairs.steps.size(), ff.bp.ops.size(), SC.env->manifoldsIn, SC.env->manifoldsMiss, SC.env->actBad);
    }
  }
}
void g1_scene_after(PxScene* scene, uint64_t sim) {
  if (!SC.on || !SC.started || scene != SC.scene) return;
  (void)sim;
  runStep(scene);
}
void g1_scene_report() {
  if (!SC.on) return;
  if (SC.lsSteps)
    printf("G1 닫힌 고리 2단 Sc 입력(우리 Sc 장면): 스텝 %" PRIu64 ", 모양 칸 %" PRIu64 " (건드림 다시 맞춤 %" PRIu64 ", Sc 모양 아닌 칸 %" PRIu64 ") — PhysX 와 다름: 경계 상자 %" PRIu64
           ", 바뀜 비트 %" PRIu64 ", 접촉 거리 %" PRIu64 ", 변환 캐시 %" PRIu64 "%s\n",
           SC.lsSteps, SC.lsCells, SC.lsResync, SC.lsNonShape, SC.lsBoundsDiff, SC.lsWordDiff, SC.lsDistDiff, SC.lsCacheDiff,
           SC.lsFirst >= 0 ? ("  첫 simulate " + std::to_string(SC.lsFirst) + " " + SC.lsFirstWhat).c_str() : "");
  if (SC.lsSteps) printf("  집합체 칸: 우리 넓은 단계 값으로 %" PRIu64 " (PhysX 입력과 달랐던 것 %" PRIu64 ")\n", SC.lsAggOurs, SC.lsAggDiff);
  printf("G1 contact 장면 단위 그림자 (scene_step.h 한 줄: 우리 넓은 단계 -> 쌍 관리 -> 좁은 단계 -> 풀이 뒤 정리, 거르개 %s): 스텝 %" PRIu64 "\n", SC.coreFilter ? "core/scene/omni_filter.h" : "재생기 PhysX 콜백", SC.steps);
  printf("  넓은 단계 겹침 %" PRIu64 " (다름 %" PRIu64 "), 좁은 단계 칸 %" PRIu64 " (값 비교 %" PRIu64 ", 다름 %" PRIu64 "), 스텝 끝 목록 %" PRIu64 " (다름 %" PRIu64
         "), solver 입력 관리자 %" PRIu64 " (다름 %" PRIu64 "), 활성화 재생 어긋남 %" PRIu64 ", PhysX 에 없는 칸 %" PRIu64 " — 다름 합 %" PRIu64 "%s\n",
         SC.cmpOverlap, SC.badOverlap, SC.cmpSlot, SC.cmpVal, SC.badSlot, SC.cmpList, SC.badEnd, SC.cmpSolver, SC.badSolver, SC.actBad, SC.goneCm, SC.bad,
         SC.firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(SC.firstBad) + " " + SC.firstWhat).c_str() : "");
}

// 순서기 그림자(g1_host.cpp): 이번 스텝 우리 넓은 단계 겹침(생성·소멸, AABB 관리자 출력 순서)과 우리 좁은 단계 칸 결과. 없으면 false
bool g1_scene_ours(uint64_t sim, std::vector<int32_t>& created, std::vector<int32_t>& createdTrigger, std::vector<int32_t>& removed,
                   std::vector<uint8_t>& npStatus, std::vector<uint8_t>& npPatches) {
  if (!SC.on || !SC.started || !SC.S || SC.ourStep != sim) return false;
  created = SC.S->createdShape;
  createdTrigger = SC.S->createdTrigger;
  removed = SC.S->destroyedShape;
  removed.insert(removed.end(), SC.S->destroyedTrigger.begin(), SC.S->destroyedTrigger.end());
  npStatus = SC.ourNpStatus;
  npPatches = SC.ourNpPatches;
  return true;
}

// 닫힌 고리 2단: 이번 스텝 우리 solver 입력 (관리자 번호 = PhysX 접촉 관리자 풀 번호). 없으면 false
bool g1_scene_solver_input(uint32_t cmIndex, const eng::sv::SolverCM** m, const eng::sv::ContactPatchIn** patches, const eng::sv::ContactIn** contacts) {
  if (!SC.on || !SC.started || cmIndex >= SC.solverCms.size()) return false;
  *m = &SC.solverCms[cmIndex];
  *patches = SC.solverIn.patches.data();
  *contacts = SC.solverIn.contacts.data();
  return true;
}
