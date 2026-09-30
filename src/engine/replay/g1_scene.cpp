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
#include "g1_hooks.h"

using namespace physx;
namespace ep = eng::px;
namespace ec = eng::contact;
namespace es = eng::scene;
namespace ss = eng::contact::sc;

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
  g1_pairs_filters(S.pairs);
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
      SC.H.forced = a.result ? 1 : 0;
      r = M.activateInteraction(it);
    } else {
      SC.H.forcedDeact = a.result ? 1 : 0;
      r = M.deactivateInteraction(it);
    }
    SC.H.forced = -1;
    SC.H.forcedDeact = -1;
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
  es::pairsPre(M, SC.H, st);
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
  SC.H.prealloc = &st.preallocHandles;
  SC.H.addCm = &st.addCmEdges;
  SC.H.preallocPos = SC.H.addCmPos = 0;
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
  {  // 재질 표 (g1_shadow 가 첫 simulate 앞에서 채움 -> 스텝마다 옮겨 담음, 판 도중 새 재질 대비)
    size_t nm = 0;
    const ec::MaterialData* mats = g1_contact_mats(&nm);
    S.materials.assign(mats, mats + nm);
  }
  if (S.npShapes.size() < ncache) S.npShapes.resize(ncache);
  for (size_t e = 0; e < M.shapes.size() && e < ncache; ++e) {
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
}
void g1_scene_before(PxScene* scene, uint64_t sim) {
  init();
  if (!SC.on) return;
  SC.sim = sim;
  SC.haveInput = false;
  SC.havePxOverlaps = false;
  if (!SC.started) startScene(scene);
}
void g1_scene_after(PxScene* scene, uint64_t sim) {
  if (!SC.on || !SC.started || scene != SC.scene) return;
  (void)sim;
  runStep(scene);
}
void g1_scene_report() {
  if (!SC.on) return;
  printf("G1 contact 장면 단위 그림자 (scene_step.h 한 줄: 우리 넓은 단계 -> 쌍 관리 -> 좁은 단계 -> 풀이 뒤 정리): 스텝 %" PRIu64 "\n", SC.steps);
  printf("  넓은 단계 겹침 %" PRIu64 " (다름 %" PRIu64 "), 좁은 단계 칸 %" PRIu64 " (값 비교 %" PRIu64 ", 다름 %" PRIu64 "), 스텝 끝 목록 %" PRIu64 " (다름 %" PRIu64
         "), 활성화 재생 어긋남 %" PRIu64 ", PhysX 에 없는 칸 %" PRIu64 " — 다름 합 %" PRIu64 "%s\n",
         SC.cmpOverlap, SC.badOverlap, SC.cmpSlot, SC.cmpVal, SC.badSlot, SC.cmpList, SC.badEnd, SC.actBad, SC.goneCm, SC.bad,
         SC.firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(SC.firstBad) + " " + SC.firstWhat).c_str() : "");
}
