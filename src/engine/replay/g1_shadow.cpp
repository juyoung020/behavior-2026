// G1a 그림자 시험 (문서 15절 G1): 공식 radio 장면을 PhysX 비계로 재생하면서, 매 simulate 마다 우리 모듈로 같은 단계를 다시 해 비트 비교한다.
// ovd_replay 의 약한 갈고리(g1_before_simulate / g1_after_simulate)를 이 번역 단위가 채운다 -> 실행 파일 ovd_replay_g1.
// 지금 단계: contact 좁은 단계 + 압축 접촉 스트림 (test_np_scene 의 장면 판을 radio 장면에). 입력 = fetchCollision 직후 PhysX 변환 캐시(모양 세계 자세, g1_simulate)·
//   접촉 거리(contactOffset 합)·재질, 쌍마다 우리 지속 다양체(관리자가 새로 생기거나 다시 등록되면 새로). PhysX 가 좁은 단계를 건너뛴 쌍(얼어붙은 두 몸체, 더럽지 않은 관리자)은
//   우리도 건너뛴다(PxcNpBatch.cpp checkContactsMustBeGenerated 규칙).
// 비교 = PhysX PxsContactManagerOutput (패치 수·점 수·상태, 패치 머리, 점 위치·분리).
#include <xmmintrin.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "PxPhysicsAPI.h"
#include "GuConvexMesh.h"
#include "NpScene.h"
#include "NpShape.h"
#include "NpMaterial.h"
#include "ScScene.h"
#include "ScShapeInteraction.h"
#include "PxsContext.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsTransformCache.h"
#include "PxvNphaseImplementationContext.h"
#include "geomutils/PxContactBuffer.h"

#include "core/contact/narrowphase.h"
#include "core/contact/patches.h"
#include "core/scene/scene_file.h"

using namespace physx;
namespace ec = eng::contact;
namespace ep = eng::px;

namespace {

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};
uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

struct Tally {
  const char* name;
  uint64_t cmp = 0, bad = 0;
  long long first = -1;
  void f(float a, float b, long long sim) { ++cmp; if (fb(a) != fb(b) && !(a != a && b != b)) { if (!bad) first = sim; ++bad; } }
  void u(uint64_t a, uint64_t b, long long sim) { ++cmp; if (a != b) { if (!bad) first = sim; ++bad; } }
};

ep::PxTransform toE(const PxTransform& t) { return ep::PxTransform(ep::PxVec3(t.p.x, t.p.y, t.p.z), ep::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w)); }
ep::PxMeshScale toE(const PxMeshScale& s) {
  ep::PxMeshScale o;
  o.scale = ep::PxVec3(s.scale.x, s.scale.y, s.scale.z);
  o.rotation = ep::PxQuat(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w);
  return o;
}

struct Shadow {
  bool on = false, inited = false;
  std::unordered_map<const PxsShapeCore*, ec::ShapeGeom> geom;
  std::vector<ec::MaterialData> mats;
  struct Pair { ec::ManifoldSlot slot; };
  std::unordered_map<const PxsContactManager*, Pair> pairs;
  std::set<const PxsContactManager*> seen, refreshed;  // 지난 simulate 에 있던 관리자
  std::vector<PxsCachedTransform> cacheBefore;
  std::unordered_map<const PxsContactManager*, uint8_t> statusBefore;
  float tolLength = 1.0f;
  Tally tCnt{"패치·점 수·상태"}, tPatch{"패치 머리"}, tPt{"접촉점·분리"};
  uint64_t nCM = 0, nRun = 0, nSkip = 0, nUnsup = 0, nOverflow = 0, nNewAfter = 0, nRefresh = 0;
  long long lastSim = 0;

  bool shapeGeom(const PxsShapeCore* c, ec::ShapeGeom& g) {
    // 모양 핵 주소로 기억해 두지 않는다: 판 도중 삭제 뒤 새 모양이 같은 주소를 다시 쓴다(chop_slice0 simulate 354, 반쪽 통나무)
    ec::ShapeGeom o;
    o.type = -1;
    const PxGeometry& pg = c->mGeometry.getGeometry();
    switch (pg.getType()) {
      case PxGeometryType::eSPHERE: o.type = ec::eSPHERE; o.sphere = ep::PxSphereGeometry(static_cast<const PxSphereGeometry&>(pg).radius); break;
      case PxGeometryType::ePLANE: o.type = ec::ePLANE; break;
      case PxGeometryType::eCAPSULE: {
        const auto& cg = static_cast<const PxCapsuleGeometry&>(pg);
        o.type = ec::eCAPSULE; o.capsule = ep::PxCapsuleGeometry(cg.radius, cg.halfHeight); break;
      }
      case PxGeometryType::eBOX: {
        const auto& bg = static_cast<const PxBoxGeometry&>(pg);
        o.type = ec::eBOX; o.box = ep::PxBoxGeometry(bg.halfExtents.x, bg.halfExtents.y, bg.halfExtents.z); break;
      }
      case PxGeometryType::eCONVEXMESH: {
        const auto& cg = static_cast<const PxConvexMeshGeometry&>(pg);
        static_assert(sizeof(ep::Gu::ConvexHullData) == sizeof(Gu::ConvexHullData), "ConvexHullData 배치");
        const auto* hull = reinterpret_cast<const ep::Gu::ConvexHullData*>(&static_cast<const Gu::ConvexMesh*>(cg.convexMesh)->getHull());
        o.type = ec::eCONVEXMESH; o.convex = ep::PxConvexMeshGeometry(hull, toE(cg.scale), cg.meshFlags); break;
      }
      default: o.type = -1;
    }
    geom[c] = o;
    g = o;
    return o.type >= 0;
  }
  void initMaterials(PxPhysics& phys) {
    const PxU32 n = phys.getNbMaterials();
    std::vector<PxMaterial*> m(n);
    phys.getMaterials(m.data(), n);
    for (PxMaterial* x : m) {
      const PxU16 idx = static_cast<NpMaterial*>(x)->mMaterial.mMaterialIndex;
      if (idx >= mats.size()) mats.resize(idx + 1);
      ec::MaterialData& md = mats[idx];
      md.dynamicFriction = x->getDynamicFriction();
      md.staticFriction = x->getStaticFriction();
      md.restitution = x->getRestitution();
      md.damping = x->getDamping();
      md.flags = uint16_t(PxU16(x->getFlags()));
      md.fricCombineMode = uint8_t(x->getFrictionCombineMode());
      md.restCombineMode = uint8_t(x->getRestitutionCombineMode());
      md.dampingCombineMode = uint8_t(x->getDampingCombineMode());
    }
  }
} G;

}  // namespace

#include "g1_hooks.h"  // solver·관절체 단독 그림자 (g1_solver.cpp G1_SOLVER=1, g1_art.cpp G1_ART=1)

// ovd_replay 가 simulate 바로 앞에서 부른다
void g1_before_simulate(PxScene* scene, PxPhysics* phys, uint64_t sim) {
  g1_solver_before(scene, sim);
  g1_art_before(scene);
  g1_dump_before(scene, sim);
  g1_env_before(scene, sim);  // env 적재 대조: 파일 경계에서 (쌍 관리층 창 연산 전 — 파일의 쌍 기록도 그 앞까지)
  g1_islands_before(scene, sim);
  g1_bp_before(scene, sim);
  g1_pairs_before(scene, sim);
  g1_loop_before(scene, sim);
  g1_sc_before(scene, sim);
  g1_scene_before(scene, sim);
  g1_host_before(scene, sim);  // 순서기: 섬·쌍 관리층 넘겨받기가 끝난 뒤
  if (!G.on) G.on = getenv("G1_CONTACT") != nullptr;
  if (!G.on) return;
  (void)sim;
  const bool dbg = (getenv("G1_DEBUG") && *getenv("G1_DEBUG"));
  if (dbg) fprintf(stderr, "[g1] before %llu\n", (unsigned long long)sim);
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  PxsContext* ctx = sc.getLowLevelContext();
  // 변환 캐시는 여기서 뜨지 않는다 — g1_simulate 의 fetchCollision 직후가 좁은 단계 입력
  G.statusBefore.clear();
  G.refreshed.clear();
  PxsContactManagerOutputIterator outputs = ctx->getNphaseImplementationContext()->getContactManagerOutputs();
  const PxU32 nInter = sc.getNbInteractions(Sc::InteractionType::eOVERLAP);
  Sc::ElementSimInteraction** inter = sc.getInteractions(Sc::InteractionType::eOVERLAP);
  for (PxU32 ii = 0; ii < nInter; ++ii) {
    const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
    if (!cm) continue;
    // 새 표시가 붙은 관리자 = 새로 생겼거나 refreshContactManager(사용자 자세 set -> ScShapeSimBase.cpp:361 resetManagerCachedState) 로
    // 다시 등록된 것. 다시 등록되면 좁은 단계 캐시(다양체)가 새로 시작한다(PxsNphaseImplementationContext.cpp:727-752).
    if (cm->getWorkUnit().mNpIndex & PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK) { G.refreshed.insert(cm); continue; }
    G.statusBefore[cm] = outputs.getContactManagerOutput(cm->getWorkUnit().mNpIndex).statusFlag;
  }
  if (dbg) fprintf(stderr, "[g1] cache %zu, 관리자 %zu\n", G.cacheBefore.size(), G.statusBefore.size());
  // v1 넘겨받기 시험 (G1_CONTACT_FROM=<엔진 장면 파일>): 파일의 경계 simulate 에서 우리 다양체를 전부 버리고 파일의 것(다른 실행에서 뜬 PhysX
  // 다양체 바이트)으로 채운 뒤, 그 simulate 부터만 비교한다 -> "넘겨받은 상태에서 이어 가도 PhysX 와 같은가"
  {
    static bool fromInit = false;
    static eng::scene::SceneFile fromFile;
    static bool fromOk = false;
    if (!fromInit) {
      fromInit = true;
      if (const char* p = getenv("G1_CONTACT_FROM")) {
        std::string err;
        fromOk = eng::scene::readScene(p, fromFile, &err);
        if (!fromOk) fprintf(stderr, "[g1] G1_CONTACT_FROM 읽기 실패: %s\n", err.c_str());
      }
    }
    if (fromOk && sim == fromFile.h.sim) {
      const std::vector<const void*> cores = g1_shape_cores(scene);
      std::unordered_map<const void*, uint32_t> idx;
      for (uint32_t k = 0; k < cores.size(); ++k) idx[cores[k]] = k;
      std::map<std::pair<uint32_t, uint32_t>, uint32_t> fileCM;
      for (uint32_t k = 0; k < fromFile.cms.size(); ++k) fileCM[{fromFile.cms[k].shape0, fromFile.cms[k].shape1}] = k;
      G.pairs.clear();
      G.seen.clear();
      uint64_t got = 0, miss = 0, noMani = 0;
      for (PxU32 ii = 0; ii < nInter; ++ii) {
        const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
        if (!cm) continue;
        const PxcNpWorkUnit& wu = cm->getWorkUnit();
        auto a = idx.find(wu.getShapeCore0()), b = idx.find(wu.getShapeCore1());
        auto f = (a == idx.end() || b == idx.end()) ? fileCM.end() : fileCM.find({a->second, b->second});
        if (f == fileCM.end()) { ++miss; continue; }
        const eng::scene::SceneCM& c = fromFile.cms[f->second];
        G.seen.insert(cm);
        if (c.manifold == eng::scene::kNone) { ++noMani; G.pairs[cm].slot = ec::ManifoldSlot{}; continue; }
        G.pairs[cm].slot = fromFile.manifolds[c.manifold];  // 대입이 자기 버퍼 포인터를 다시 건다
        if (getenv("G1_CONTACT_FROM_NEG")) {  // 음성 대조: 다양체를 새로 비운 것으로 (이러면 달라져야 한다)
          const int t0 = fromFile.shapes[c.shape0].geom.type, t1 = fromFile.shapes[c.shape1].geom.type;
          ec::initManifold(G.pairs[cm].slot, t0 < t1 ? t0 : t1, t0 < t1 ? t1 : t0);
        }
        ++got;
      }
      G.tCnt = Tally{"패치·점 수·상태"};
      G.tPatch = Tally{"패치 머리"};
      G.tPt = Tally{"접촉점·분리"};
      G.nCM = G.nRun = G.nSkip = G.nUnsup = G.nOverflow = G.nNewAfter = G.nRefresh = 0;
      printf("G1 contact 넘겨받기: simulate %llu 에서 파일 관리자 %zu 중 다양체 %" PRIu64 " 개를 넣음 (다양체 없는 관리자 %" PRIu64 ", 파일에 없는 관리자 %" PRIu64 ") — 이 뒤로만 비교\n",
             (unsigned long long)sim, fromFile.cms.size(), got, noMani, miss);
    }
  }
  G.initMaterials(*phys);  // 스텝마다 (판 도중 새 물체가 재질을 더하거나 바꾼다 — 양파 658 "재질 번호가 표 밖")
  if (!G.inited) {
    if (dbg) fprintf(stderr, "[g1] 재질 %zu\n", G.mats.size());
    G.tolLength = phys->getTolerancesScale().length;
    G.inited = true;
  }
}

// simulate 를 collide → fetchCollision → advance → fetchResults 로 나눠 돈다 (NpScene::simulateOrCollide 같은 길).
// 좁은 단계 첫 단계가 읽은 변환 캐시는 fetchCollision 직후 값이다 (사용자 set·키네마틱 목표는 simulate 안에서 캐시에 들어가므로
// simulate 앞에서 읽으면 틀림). 적분은 advance 의 풀이기 뒤라 이때까지 캐시는 그대로.
bool g1_simulate(PxScene* scene, float dt, uint64_t sim) {
  if (!G.on) return false;
  (void)sim;
  scene->collide(dt);
  scene->fetchCollision(true);
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  PxsTransformCache& tc = sc.getLowLevelContext()->getTransformCache();
  G.cacheBefore.resize(tc.getTotalSize());
  for (PxU32 k = 0; k < tc.getTotalSize(); ++k) G.cacheBefore[k] = tc.getTransformCache(k);
  scene->advance();
  scene->fetchResults(true);
  return true;
}

// ovd_replay 가 fetchResults 뒤에 부른다
void g1_after_simulate(PxScene* scene, PxPhysics* phys, uint64_t sim) {
  g1_pairs_after(scene, sim);
  g1_scene_after(scene, sim);  // 우리 contact 한 스텝 (닫힌 고리 2단: solver 가 이 접촉 입력을 쓴다 -> 먼저)
  g1_host_after(scene, sim);  // 순서기: 쌍 관리층 입력·섬 기록·우리 넓은/좁은 단계 결과가 다 모인 뒤, 다른 그림자가 PhysX 를 건드리기 전
  g1_islands_after(scene, sim);  // 우리 섬 스텝 끝 상태 (3단: solver 가 잠들 노드를 여기서 읽는다 -> 먼저)
  g1_solver_after(scene, sim);
  g1_art_after(scene, sim);
  g1_sc_after(scene, sim);
  g1_loop_after(scene, sim);
  if (!G.on) return;
  (void)phys;
  G.lastSim = (long long)sim;
  Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
  PxsContext* ctx = sc.getLowLevelContext();
  PxsContactManagerOutputIterator outputs = ctx->getNphaseImplementationContext()->getContactManagerOutputs();
  const PxU32 nInter = sc.getNbInteractions(Sc::InteractionType::eOVERLAP);
  Sc::ElementSimInteraction** inter = sc.getInteractions(Sc::InteractionType::eOVERLAP);
  static ep::PxContactBuffer buf;
  static ec::CompressedContacts<64, 256> out;
  static ec::MaterialPair mp[256];
  std::set<const PxsContactManager*> now;
  static const bool dbg = (getenv("G1_DEBUG") && *getenv("G1_DEBUG"));
  if (dbg) fprintf(stderr, "[g1] after %llu 상호작용 %u\n", (unsigned long long)sim, nInter);
  for (PxU32 ii = 0; ii < nInter; ++ii) {
    const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
    if (!cm) continue;
    const PxcNpWorkUnit& wu = cm->getWorkUnit();
    if (wu.mNpIndex & PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK) { ++G.nNewAfter; continue; }  // 새 관리자 표시(0x80000000)는 새 출력 버퍼 쪽 — simulate 뒤엔 없어야 함
    now.insert(cm);
    const PxsContactManagerOutput& po = outputs.getContactManagerOutput(wu.mNpIndex);
    ec::ShapeGeom g0, g1;
    const PxsShapeCore* s0 = wu.getShapeCore0();
    const PxsShapeCore* s1 = wu.getShapeCore1();
    if (!G.shapeGeom(s0, g0) || !G.shapeGeom(s1, g1)) { ++G.nUnsup; continue; }
    const bool isNew = !G.seen.count(cm) || G.refreshed.count(cm);
    if (isNew && G.seen.count(cm)) ++G.nRefresh;
    Shadow::Pair& ps = G.pairs[cm];
    if (isNew) {
      const int t0 = g0.type < g1.type ? g0.type : g1.type, t1 = g0.type < g1.type ? g1.type : g0.type;
      ec::initManifold(ps.slot, t0, t1);
    }
    ++G.nCM;
    if (wu.mTransformCache0 >= G.cacheBefore.size() || wu.mTransformCache1 >= G.cacheBefore.size()) { ++G.nUnsup; continue; }
    const PxsCachedTransform& c0 = G.cacheBefore[wu.mTransformCache0];
    const PxsCachedTransform& c1 = G.cacheBefore[wu.mTransformCache1];
    // checkContactsMustBeGenerated: 발견 끔이면 안 함, 더럽지 않고 두 쪽 다 얼었거나 정적이면 지난 결과 그대로
    if (!(wu.mFlags & PxcNpWorkUnitFlag::eDETECT_DISCRETE_CONTACT)) { ++G.nSkip; continue; }
    auto sb = G.statusBefore.find(cm);
    const bool dirty = isNew || sb == G.statusBefore.end() || (sb->second & PxcNpWorkUnitStatusFlag::eDIRTY_MANAGER);
    if (!dirty && !(wu.mFlags & PxcNpWorkUnitFlag::eMODIFIABLE_CONTACT)) {
      const bool dyn0 = wu.mFlags & (PxcNpWorkUnitFlag::eDYNAMIC_BODY0 | PxcNpWorkUnitFlag::eARTICULATION_BODY0);
      const bool dyn1 = wu.mFlags & (PxcNpWorkUnitFlag::eDYNAMIC_BODY1 | PxcNpWorkUnitFlag::eARTICULATION_BODY1);
      if (!((dyn0 && !c0.isFrozen()) || (dyn1 && !c1.isFrozen()))) { ++G.nSkip; continue; }
    }
    const float contactDist = s0->mContactOffset + s1->mContactOffset;
    bool flipped;
    {
      FtzScope fz;
      ec::pcmPair(g0, g1, ep::PxTransform32(toE(c0.transform)), ep::PxTransform32(toE(c1.transform)), contactDist, 0.01f * G.tolLength, G.tolLength,
                  ps.slot, buf, flipped);
      for (ep::PxU32 k = 0; k < buf.count; ++k) mp[k] = ec::MaterialPair{s0->mMaterialIndex, s1->mMaterialIndex};
      const int tmax = g0.type > g1.type ? g0.type : g1.type;
      ec::writeCompressedContact(buf.contacts, buf.count, mp, G.mats.data(), false, tmax > ec::eCONVEXMESH, out);
    }
    ++G.nRun;
    if (out.overflow) ++G.nOverflow;
    const uint8_t ourStatus = buf.count ? PxsContactManagerStatusFlag::eHAS_TOUCH : PxsContactManagerStatusFlag::eHAS_NO_TOUCH;
    G.tCnt.u(po.nbPatches, out.nbPatches, sim);
    G.tCnt.u(po.nbContacts, out.nbContacts, sim);
    G.tCnt.u(po.statusFlag & PxsContactManagerStatusFlag::eTOUCH_KNOWN, ourStatus, sim);
    const PxU32 np = PxMin<PxU32>(po.nbPatches, out.nbPatches);
    const PxContactPatch* pp = reinterpret_cast<const PxContactPatch*>(po.contactPatches);
    for (PxU32 k = 0; k < np; ++k) {
      const PxContactPatch& x = pp[k];
      const ec::ContactPatch& y = out.patches[k];
      G.tPatch.f(x.normal.x, y.normal.x, sim); G.tPatch.f(x.normal.y, y.normal.y, sim); G.tPatch.f(x.normal.z, y.normal.z, sim);
      G.tPatch.f(x.restitution, y.restitution, sim); G.tPatch.f(x.dynamicFriction, y.dynamicFriction, sim);
      G.tPatch.f(x.staticFriction, y.staticFriction, sim); G.tPatch.f(x.damping, y.damping, sim);
      G.tPatch.u(x.startContactIndex, y.startContactIndex, sim); G.tPatch.u(x.nbContacts, y.nbContacts, sim);
      G.tPatch.u(x.materialFlags, y.materialFlags, sim); G.tPatch.u(x.internalFlags, y.internalFlags, sim);
      G.tPatch.u(x.materialIndex0, y.materialIndex0, sim); G.tPatch.u(x.materialIndex1, y.materialIndex1, sim);
    }
    const PxU32 nc = PxMin<PxU32>(po.nbContacts, out.nbContacts);
    const PxContact* pc = reinterpret_cast<const PxContact*>(po.contactPoints);
    for (PxU32 k = 0; k < nc; ++k) {
      G.tPt.f(pc[k].contact.x, out.contacts[k].contact.x, sim); G.tPt.f(pc[k].contact.y, out.contacts[k].contact.y, sim);
      G.tPt.f(pc[k].contact.z, out.contacts[k].contact.z, sim); G.tPt.f(pc[k].separation, out.contacts[k].separation, sim);
    }
    // 다른 쌍 몇 개를 자세히 (G1_SHOW=개수)
    static int show = getenv("G1_SHOW") ? atoi(getenv("G1_SHOW")) : 0;
    if (show > 0) {
      bool bad = po.nbContacts != out.nbContacts;
      float mx = 0;
      for (PxU32 k = 0; k < nc; ++k) {
        const float d[4] = {pc[k].contact.x - out.contacts[k].contact.x, pc[k].contact.y - out.contacts[k].contact.y,
                            pc[k].contact.z - out.contacts[k].contact.z, pc[k].separation - out.contacts[k].separation};
        for (float v : d) { if (v != 0) bad = true; mx = PxMax(mx, PxAbs(v)); }
      }
      if (bad) {
        --show;
        fprintf(stderr, "[g1 다름] sim %llu 쌍 %p 모양 %d/%d 새것 %d 점 %u/%u 최대|차| %.3e 캐시 %u/%u 동적 %d%d\n", (unsigned long long)sim, (const void*)cm,
                g0.type, g1.type, int(isNew), po.nbContacts, out.nbContacts, mx, wu.mTransformCache0, wu.mTransformCache1,
                int(bool(wu.mFlags & (PxcNpWorkUnitFlag::eDYNAMIC_BODY0 | PxcNpWorkUnitFlag::eARTICULATION_BODY0))),
                int(bool(wu.mFlags & (PxcNpWorkUnitFlag::eDYNAMIC_BODY1 | PxcNpWorkUnitFlag::eARTICULATION_BODY1))));
        for (PxU32 k = 0; k < nc && k < 4; ++k)
          fprintf(stderr, "   공식 %.9g %.9g %.9g s %.9g | 우리 %.9g %.9g %.9g s %.9g\n", pc[k].contact.x, pc[k].contact.y, pc[k].contact.z, pc[k].separation,
                  out.contacts[k].contact.x, out.contacts[k].contact.y, out.contacts[k].contact.z, out.contacts[k].separation);
      }
    }
  }
  // 사라진 관리자의 다양체는 지운다 (칸 재사용 대비)
  for (auto it = G.pairs.begin(); it != G.pairs.end();) it = now.count(it->first) ? std::next(it) : G.pairs.erase(it);
  G.seen.swap(now);
}

void g1_report() {
  g1_solver_report();
  g1_art_report();
  g1_islands_report();
  g1_bp_report();
  g1_pairs_report();
  g1_sc_report();
  g1_scene_report();
  g1_loop_report();
  g1_host_report();
  if (!G.on) return;
  printf("G1 contact 그림자: 관리자·simulate %" PRIu64 " (좁은 단계 돈 것 %" PRIu64 ", 건너뜀 %" PRIu64 ", 못 옮긴 모양 %" PRIu64 ", 넘침 %" PRIu64 ", 뒤에도 새 표시 %" PRIu64 ", 다시 등록 %" PRIu64 ")\n",
         G.nCM, G.nRun, G.nSkip, G.nUnsup, G.nOverflow, G.nNewAfter, G.nRefresh);
  for (const Tally* t : {&G.tCnt, &G.tPatch, &G.tPt})
    printf("  %-20s 비교 %12" PRIu64 "  비트 다름 %10" PRIu64 "%s\n", t->name, t->cmp, t->bad,
           t->bad ? ("  첫 다름 simulate " + std::to_string(t->first)).c_str() : "");
}

const physx::PxsCachedTransform* g1_contact_cache(size_t* n) {
  *n = G.cacheBefore.size();
  return G.cacheBefore.data();
}
const eng::contact::MaterialData* g1_contact_mats(size_t* n) {
  *n = G.mats.size();
  return G.mats.data();
}
float g1_contact_tol() { return G.tolLength; }
