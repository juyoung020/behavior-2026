// 엔진 장면 파일 뜨기 (문서 15.3, 리드, 비계 전용): 공식 기록을 재생하다 G1_DUMP_AT 번째 simulate 바로 앞에서 PhysX 장면을
// core/scene/scene_file.h 형식으로 쓴다 (틀 = 재질·행위자·모양·볼록, 상태 = 동적 몸체·관절체·D6 조인트).
// 쓴 뒤 바로 판 N 개 배치(core/scene/batch.h)에 다시 읽어 넣고, 판마다 값을 **PhysX 공개 API 로 다시 읽은 값**과 비트 비교한다
// (옮겨 담기 규칙과 파일·배치 왕복을 한꺼번에 확인).
//   G1_DUMP_AT=<simulate 번호> G1_DUMP_OUT=<파일> [G1_DUMP_ENVS=3] ovd_replay_g1 ...
#include <cinttypes>
#include <unordered_map>

#include "../tests/solver/px_internal.h"
#include "../tests/articulation/art_from_px.h"
#include "GuConvexMesh.h"
#include "NpMaterial.h"
#include "NpShape.h"
#include "NpConstraint.h"
#include "ScConstraintSim.h"
#include "ScShapeInteraction.h"
#include "PxsNphaseImplementationContext.h"
#include "PxsContactManagerState.h"
#include "DyFrictionPatch.h"
#include "core/contact/hull_pack.h"
#include "core/scene/batch.h"
#include "g1_hooks.h"
#include "g1_px.h"

using namespace physx;
namespace sc = eng::scene;
namespace ec = eng::contact;
namespace ep = eng::px;

namespace {

// 채움 바이트를 0 으로 둔 채 만든다 (임시 객체를 대입하면 채움 바이트가 쓰레기로 복사돼 같은 틀의 해시가 갈린다)
template <class T, class... Args>
void zput(T& dst, Args... args) {
  alignas(T) unsigned char buf[sizeof(T)] = {};
  new (buf) T(args...);
  memcpy(static_cast<void*>(&dst), buf, sizeof(T));
}

bool geomFrom(const PxGeometry& pg, ec::ShapeGeom& o) {
  o.type = -1;
  switch (pg.getType()) {
    case PxGeometryType::eSPHERE: o.type = ec::eSPHERE; zput(o.sphere, static_cast<const PxSphereGeometry&>(pg).radius); return true;
    case PxGeometryType::ePLANE: o.type = ec::ePLANE; return true;
    case PxGeometryType::eCAPSULE: {
      const auto& g = static_cast<const PxCapsuleGeometry&>(pg);
      o.type = ec::eCAPSULE; zput(o.capsule, g.radius, g.halfHeight); return true;
    }
    case PxGeometryType::eBOX: {
      const auto& g = static_cast<const PxBoxGeometry&>(pg);
      o.type = ec::eBOX; zput(o.box, g.halfExtents.x, g.halfExtents.y, g.halfExtents.z); return true;
    }
    case PxGeometryType::eCONVEXMESH: {
      const auto& g = static_cast<const PxConvexMeshGeometry&>(pg);
      ep::PxMeshScale s;
      s.scale = ep::PxVec3(g.scale.scale.x, g.scale.scale.y, g.scale.scale.z);
      s.rotation = ep::PxQuat(g.scale.rotation.x, g.scale.rotation.y, g.scale.rotation.z, g.scale.rotation.w);
      o.type = ec::eCONVEXMESH; zput(o.convex, nullptr, s, g.meshFlags); return true;
    }
    default: return false;
  }
}

// PhysX 마찰 패치 -> 우리 것 (g1_solver.cpp frictionFrom 과 같은 규칙)
eng::sv::FrictionPatch frictionOf(const Dy::FrictionPatch& p) {
  eng::sv::FrictionPatch f;
  f.broken = p.broken;
  f.materialFlags = p.materialFlags;
  f.anchorCount = p.anchorCount;
  f.restitution = p.restitution;
  f.staticFriction = p.staticFriction;
  f.dynamicFriction = p.dynamicFriction;
  f.body0Normal = g1px::toV(p.body0Normal);
  f.body1Normal = g1px::toV(p.body1Normal);
  for (int a = 0; a < 2; ++a) {
    f.body0Anchors[a] = g1px::toV(p.body0Anchors[a]);
    f.body1Anchors[a] = g1px::toV(p.body1Anchors[a]);
  }
  f.relativeQuat = eng::Q{p.relativeQuat.x, p.relativeQuat.y, p.relativeQuat.z, p.relativeQuat.w};
  return f;
}

// 관리자의 Gu::Cache (좁은 단계 목록 칸; 새 표시면 새 목록)
Gu::Cache* cacheOf(PxsNphaseImplementationContext* np, PxU32 npIndex) {
  const bool isNew = (npIndex & PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK) != 0;
  PxsContactManagers& L = isNew ? np->mNewNarrowPhasePairs : np->mNarrowPhasePairs;
  const PxU32 idx = PxsContactManagerBase::computeIndexFromId(npIndex & ~PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK);
  return idx < L.mCaches.size() ? &L.mCaches[idx] : nullptr;
}
const PxsContactManagerOutput* outputOf(PxsNphaseImplementationContext* np, PxU32 npIndex) {
  const bool isNew = (npIndex & PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK) != 0;
  PxsContactManagers& L = isNew ? np->mNewNarrowPhasePairs : np->mNarrowPhasePairs;
  const PxU32 idx = PxsContactManagerBase::computeIndexFromId(npIndex & ~PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK);
  return idx < L.mOutputContactManagers.size() ? &L.mOutputContactManagers[idx] : nullptr;
}

// 조인트 되쓰기 칸 (Dy::ConstraintWriteback: 선·각 충격, 끊김·남은 위치 반복) — 풀이가 끝에 덮어쓰지만 끊김 비트는 이어진다
void jnt_wb_push(sc::SceneFile& F, Dy::Context* dy, PxU32 index) {
  eng::jnt::Writeback w;
  static_assert(sizeof(w) == sizeof(Dy::ConstraintWriteback), "되쓰기 칸 배치");
  memcpy(&w, &dy->getConstraintWriteBackPool()[index], sizeof(w));
  F.jointWritebacks.push_back(w);
}

struct Dumper {
  bool inited = false, done = false;
  long long at = -1;
  std::string out;
  uint32_t envs = 3;
  // 확인 결과
  uint64_t cmp = 0, bad = 0, unsup = 0;
  std::string firstBad;
  void chk(bool same, const std::string& what) {
    ++cmp;
    if (!same) {
      if (!bad) firstBad = what;
      ++bad;
    }
  }
  template <class A, class B>
  void chkBits(const A& a, const B& b, size_t bytes, const std::string& what) { chk(!memcmp(&a, &b, bytes), what); }
} D;

// 머리·재질 (장면 파일·새 행위자 틀 파일 공용)
void fillHeader(sc::SceneFile& F, PxScene* scene, uint64_t sim) {
  PxPhysics& phys = scene->getPhysics();
  Sc::Scene& scs = static_cast<NpScene*>(scene)->getScScene();
  Dy::Context* dy = static_cast<Dy::Context*>(scs.getDynamicsContext());
  const PxVec3 g = scene->getGravity();
  F.h.gravity[0] = g.x; F.h.gravity[1] = g.y; F.h.gravity[2] = g.z;
  F.h.dt = dy->getDt();
  F.h.lengthScale = phys.getTolerancesScale().length;
  F.h.speedScale = phys.getTolerancesScale().speed;
  F.h.sceneFlags = uint32_t(scene->getFlags());
  F.h.solverType = uint32_t(scene->getSolverType());
  F.h.sim = sim;
  if (const char* t = getenv("G1_DUMP_TASK")) strncpy(F.h.task, t, sizeof(F.h.task) - 1);
  if (const char* t = getenv("G1_DUMP_INSTANCE")) strncpy(F.h.instance, t, sizeof(F.h.instance) - 1);
  const PxU32 n = phys.getNbMaterials();
  std::vector<PxMaterial*> m(n);
  phys.getMaterials(m.data(), n);
  for (PxMaterial* x : m) {
    const PxU16 idx = static_cast<NpMaterial*>(x)->mMaterial.mMaterialIndex;
    if (idx >= F.materials.size()) F.materials.resize(idx + 1);
    ec::MaterialData& md = F.materials[idx];
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

// 모양 담기 (행위자별 PxRigidActor::getShapes 순서; 볼록 덩어리는 파일 안에 한 번씩)
struct ShapeAdder {
  sc::SceneFile& F;
  std::unordered_map<const void*, uint32_t> hullIdx;       // Gu::ConvexMesh* -> 덩어리 번호
  std::unordered_map<const PxsShapeCore*, uint32_t> shapeOfCore;  // 좁은 단계 모양 -> SceneShape 번호
  explicit ShapeAdder(sc::SceneFile& f) : F(f) {}
  void add(PxRigidActor* a, sc::SceneActor& sa, uint32_t actorIdx) {
    sa.shapeStart = uint32_t(F.shapes.size());
    const PxU32 ns = a->getNbShapes();
    std::vector<PxShape*> shs(ns);
    a->getShapes(shs.data(), ns);
    for (PxShape* s : shs) {
      sc::SceneShape o;
      memset(static_cast<void*>(&o), 0, sizeof(o));  // 채움 바이트까지 0 (틀 해시가 같은 틀에서 같게)
      o.geom.type = -1;
      o.actor = actorIdx;
      o.hull = sc::kNone;
      if (!geomFrom(s->getGeometry(), o.geom)) ++D.unsup;
      if (s->getGeometry().getType() == PxGeometryType::eCONVEXMESH) {
        const PxConvexMesh* cm = static_cast<const PxConvexMeshGeometry&>(s->getGeometry()).convexMesh;
        auto it = hullIdx.find(cm);
        if (it == hullIdx.end()) {
          static_assert(sizeof(ep::Gu::ConvexHullData) == sizeof(Gu::ConvexHullData), "ConvexHullData 배치");
          const auto& h = *reinterpret_cast<const ep::Gu::ConvexHullData*>(&static_cast<const Gu::ConvexMesh*>(cm)->getHull());
          const uint64_t off = ec::align16(F.hulls.size());
          F.hulls.resize(off + ec::hullPackedBytes(h));
          ec::packHull(h, F.hulls.data() + off, uintptr_t(off));  // 파일 안에서는 덩어리 위치 기준(읽을 때 주소를 더함)
          it = hullIdx.emplace(cm, uint32_t(F.hullOffsets.size())).first;
          F.hullOffsets.push_back(off);
        }
        o.hull = it->second;
      }
      o.localPose = g1px::toE(s->getLocalPose());
      o.contactOffset = s->getContactOffset();
      o.restOffset = s->getRestOffset();
      o.torsionalPatchRadius = s->getTorsionalPatchRadius();
      o.minTorsionalPatchRadius = s->getMinTorsionalPatchRadius();
      const PxFilterData fd = s->getSimulationFilterData();
      F.shapeFilters.push_back(sc::ShapeFilter{{fd.word0, fd.word1, fd.word2, fd.word3}});
      o.shapeFlags = uint32_t(PxU8(s->getFlags()));
      PxMaterial* m0 = nullptr;
      if (s->getNbMaterials()) s->getMaterials(&m0, 1);
      o.material = m0 ? static_cast<NpMaterial*>(m0)->mMaterial.mMaterialIndex : 0xffff;
      shapeOfCore[&static_cast<NpShape*>(s)->getCore().getCore()] = uint32_t(F.shapes.size());
      F.shapes.push_back(o);
    }
    sa.shapeCount = uint32_t(F.shapes.size()) - sa.shapeStart;
  }
};

// 강체 행위자 하나 (정적·동적)
uint32_t addRigid(sc::SceneFile& F, ShapeAdder& SA, PxActor* a, std::unordered_map<const PxsRigidBody*, uint32_t>* actorOfLL) {
  sc::SceneActor sa{};
  const uint32_t ai = uint32_t(F.actors.size());
  sa.name = F.addName(a->getName());
  sa.actorFlags = uint32_t(PxU8(a->getActorFlags()));
  sa.dominance = a->getDominanceGroup();
  sa.body = sc::kNone;
  sa.link = sc::kNone;
  if (a->getType() == PxActorType::eRIGID_STATIC) {
    sa.kind = sc::kStatic;
    sa.staticPose = g1px::toE(static_cast<PxRigidStatic*>(a)->getGlobalPose());
  } else {
    sa.kind = sc::kDynamic;
    auto* rd = static_cast<PxRigidDynamic*>(a);
    sa.rigidFlags = uint32_t(PxU16(rd->getRigidBodyFlags()));
    const PxsRigidBody& ll = static_cast<NpRigidDynamic*>(rd)->getCore().getSim()->getLowLevelBody();
    sa.body = uint32_t(F.bodies.size());
    F.bodies.push_back(g1px::bodyFrom(ll));
    if (actorOfLL) (*actorOfLL)[&ll] = ai;
  }
  SA.add(static_cast<PxRigidActor*>(a), sa, ai);
  F.actors.push_back(sa);
  return ai;
}

void dumpScene(PxScene* scene, uint64_t sim) {
  sc::SceneFile F;
  Sc::Scene& scs = static_cast<NpScene*>(scene)->getScScene();
  Dy::Context* dy = static_cast<Dy::Context*>(scs.getDynamicsContext());
  fillHeader(F, scene, sim);
  ShapeAdder SA(F);
  std::unordered_map<const PxsShapeCore*, uint32_t>& shapeOfCore = SA.shapeOfCore;
  std::unordered_map<const PxsRigidBody*, uint32_t> actorOfLL;  // 조인트 몸체 -> 행위자
  auto addShapes = [&](PxRigidActor* a, sc::SceneActor& sa, uint32_t actorIdx) { SA.add(a, sa, actorIdx); };
  // 강체 행위자 (장면 순서)
  const PxU32 na = scene->getNbActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC);
  std::vector<PxActor*> acts(na);
  scene->getActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC, acts.data(), na);
  for (PxActor* a : acts) addRigid(F, SA, a, &actorOfLL);
  // 관절체 (장면 순서) + 링크 행위자 (생성 순서)
  const PxU32 nArt = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(nArt);
  scene->getArticulations(arts.data(), nArt);
  artest::A::SceneScale ss;
  ss.length = F.h.lengthScale;
  ss.speed = F.h.speedScale;
  for (PxArticulationReducedCoordinate* a : arts) {
    const uint32_t artIdx = uint32_t(F.arts.size());
    F.arts.emplace_back();
    std::vector<PxArticulationLink*> links(a->getNbLinks());
    a->getLinks(links.data(), a->getNbLinks());
    if (links.size() > eng::art::kMaxLinks || !artest::snapshotFromPx(a, F.arts.back(), ss, links)) {
      ++D.unsup;
      fprintf(stderr, "[장면 뜨기] 관절체 옮겨 담기 실패: %s (링크 %zu)\n", a->getName() ? a->getName() : "?", links.size());
    }
    F.artName.push_back(F.addName(a->getName()));
    for (uint32_t l = 0; l < links.size(); ++l) {
      sc::SceneActor sa{};
      const uint32_t ai = uint32_t(F.actors.size());
      sa.kind = sc::kLink;
      sa.body = artIdx;
      sa.link = l;
      sa.name = F.addName(links[l]->getName());
      sa.actorFlags = uint32_t(PxU8(links[l]->getActorFlags()));
      sa.dominance = links[l]->getDominanceGroup();
      sa.rigidFlags = uint32_t(PxU16(links[l]->getRigidBodyFlags()));
      const PxsRigidBody& ll = static_cast<NpArticulationLink*>(links[l])->getCore().getSim()->getLowLevelBody();
      actorOfLL[&ll] = ai;
      addShapes(links[l], sa, ai);
      F.actors.push_back(sa);
    }
  }
  // D6 조인트 (PxScene::getConstraints 순서)
  {
    const PxU32 nc = scene->getNbConstraints();
    std::vector<PxConstraint*> cs(nc);
    scene->getConstraints(cs.data(), nc);
    for (PxConstraint* c : cs) {
      Sc::ConstraintSim* sim = static_cast<NpConstraint*>(c)->getCore().getSim();
      if (!sim) { ++D.unsup; continue; }
      const Dy::Constraint& dc = sim->getLowLevelConstraint();
      sc::SceneJoint j{};
      auto actorOf = [&](const PxsRigidBody* b) -> uint32_t {
        if (!b) return sc::kNone;
        auto it = actorOfLL.find(b);
        if (it == actorOfLL.end()) { ++D.unsup; return sc::kNone; }
        return it->second;
      };
      j.actor0 = actorOf(dc.body0);
      j.actor1 = actorOf(dc.body1);
      j.index = dc.index;
      j.flags = dc.flags;
      j.linBreakForce = dc.linBreakForce;
      j.angBreakForce = dc.angBreakForce;
      j.minResponseThreshold = dc.minResponseThreshold;
      PxU32 tid = 0;
      void* ext = c->getExternalReference(tid);
      j.name = F.addName(ext && tid == PxConstraintExtIDs::eJOINT ? static_cast<PxJoint*>(ext)->getName() : nullptr);
      if (dc.constantBlockSize != sizeof(eng::jnt::D6Data)) ++D.unsup;
      else memcpy(&j.data, dc.constantBlock, sizeof(eng::jnt::D6Data));
      F.joints.push_back(j);
      jnt_wb_push(F, dy, dc.index);
    }
  }
  // 접촉 관리자 (v1): 겹침 상호작용 순서
  uint64_t nNoCache = 0;
  {
    auto* np = static_cast<PxsNphaseImplementationContext*>(scs.getLowLevelContext()->getNphaseImplementationContext());
    const PxU32 nInter = scs.getNbInteractions(Sc::InteractionType::eOVERLAP);
    Sc::ElementSimInteraction** inter = scs.getInteractions(Sc::InteractionType::eOVERLAP);
    for (PxU32 ii = 0; ii < nInter; ++ii) {
      const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
      if (!cm) continue;  // 관리자 없는 상호작용(잠든 쌍 등)은 넘길 것이 없다
      const PxcNpWorkUnit& u = cm->getWorkUnit();
      sc::SceneCM c{};
      auto s0 = shapeOfCore.find(u.getShapeCore0()), s1 = shapeOfCore.find(u.getShapeCore1());
      if (s0 == shapeOfCore.end() || s1 == shapeOfCore.end()) { ++D.unsup; continue; }
      c.shape0 = s0->second;
      c.shape1 = s1->second;
      c.npIndex = u.mNpIndex;
      c.npFlags = u.mFlags;
      c.restDistance = u.mRestDistance;
      c.torsionalPatchRadius = u.mTorsionalPatchRadius;
      c.minTorsionalPatchRadius = u.mMinTorsionalPatchRadius;
      c.offsetSlop = u.mOffsetSlop;
      c.manifold = sc::kNone;
      if (const PxsContactManagerOutput* o = outputOf(np, u.mNpIndex)) c.statusFlag = o->statusFlag;
      Gu::Cache* gc = cacheOf(np, u.mNpIndex);
      if (!gc) ++nNoCache;
      else {
        c.pairData = gc->mPairData;
        c.manifoldFlags = gc->mManifoldFlags;
        if (gc->isManifold() && !gc->isMultiManifold()) {
          const int t0 = F.shapes[c.shape0].geom.type, t1 = F.shapes[c.shape1].geom.type;
          ec::ManifoldSlot m;
          ec::initManifold(m, t0 < t1 ? t0 : t1, t0 < t1 ? t1 : t0);  // 종류(구 1점 / 큰 4점)와 자리
          const size_t bytes = m.kind == 1 ? sizeof(Gu::SpherePersistentContactManifold) : sizeof(Gu::LargePersistentContactManifold);
          static_assert(sizeof(Gu::LargePersistentContactManifold) == sizeof(eng::px::Gu::LargePersistentContactManifold), "다양체 배치");
          if (m.kind) {
            memcpy(m.storage, gc->mCachedData, bytes);
            ec::relocateManifold(m);
            c.manifold = uint32_t(F.manifolds.size());
            F.manifolds.push_back(m);
          } else ++D.unsup;
        } else if (gc->isMultiManifold()) ++D.unsup;
      }
      c.frictionStart = uint32_t(F.friction.size());
      const Dy::FrictionPatch* fp = reinterpret_cast<const Dy::FrictionPatch*>(u.mFrictionDataPtr);
      for (PxU32 k = 0; fp && k < u.mFrictionPatchCount; ++k) F.friction.push_back(frictionOf(fp[k]));
      c.frictionCount = uint32_t(F.friction.size()) - c.frictionStart;
      F.cms.push_back(c);
    }
  }
  // 섬 관리자 (v1-b): 노드 객체 -> 행위자 번호(강체) / 0x80000000 | 관절체 번호, 간선 객체 -> 접촉 관리자 번호 / 0x80000000 | 조인트 번호
  {
    struct Maps {
      std::unordered_map<const void*, uint32_t> node, edge;
    } maps;
    for (auto& kv : actorOfLL) maps.node[kv.first] = kv.second;
    for (uint32_t k = 0; k < arts.size(); ++k) maps.node[llArticulation(arts[k])] = 0x80000000u | k;
    {
      const PxU32 nInter = scs.getNbInteractions(Sc::InteractionType::eOVERLAP);
      Sc::ElementSimInteraction** inter = scs.getInteractions(Sc::InteractionType::eOVERLAP);
      uint32_t ci = 0;
      for (PxU32 ii = 0; ii < nInter; ++ii) {
        const PxsContactManager* cm = static_cast<const Sc::ShapeInteraction*>(inter[ii])->getContactManager();
        if (cm) maps.edge[cm] = ci++;
      }
      const PxU32 nc = scene->getNbConstraints();
      std::vector<PxConstraint*> cs(nc);
      scene->getConstraints(cs.data(), nc);
      for (PxU32 k = 0; k < nc; ++k)
        if (Sc::ConstraintSim* sim = static_cast<NpConstraint*>(cs[k])->getCore().getSim()) maps.edge[&sim->getLowLevelConstraint()] = 0x80000000u | k;
    }
    auto nodeId = [](const void* o, uint32_t, void* u) -> uint32_t {
      auto& m = static_cast<Maps*>(u)->node;
      auto it = m.find(o);
      return it == m.end() ? 0xffffffffu : it->second;
    };
    auto edgeId = [](const void* o, void* u) -> uint32_t {
      auto& m = static_cast<Maps*>(u)->edge;
      auto it = m.find(o);
      return it == m.end() ? 0xfffffffeu : it->second;
    };
    g1_islands_capture(scene, F.islands, nodeId, edgeId, &maps);
  }
  if (const sc::BpLog* bl = g1_bp_log()) F.bp = *bl;
  if (const sc::PairsLog* pl = g1_pairs_log()) F.pairs = *pl;  // 쌍 관리층 입력 기록 (G1_PAIRS)  // 넓은 단계 입력 기록 (G1_BP 로 모은 것, 이 simulate 앞까지)
  if (!sc::writeScene(D.out.c_str(), F)) {
    fprintf(stderr, "[장면 뜨기] 쓰기 실패: %s\n", D.out.c_str());
    return;
  }

  // ---- 확인: 파일을 다시 읽어 판 N 개 배치에 넣고, 판마다 PhysX 공개 API 값과 비교
  sc::SceneFile R;
  std::string err;
  if (!sc::readScene(D.out.c_str(), R, &err)) {
    fprintf(stderr, "[장면 뜨기] 다시 읽기 실패: %s\n", err.c_str());
    return;
  }
  sc::Batch B;
  B.init(D.envs, sc::BatchCaps{uint32_t(R.bodies.size()), uint32_t(R.arts.size()), uint32_t(R.joints.size()), uint32_t(R.shapes.size()), uint32_t(R.cms.size()), uint32_t(R.friction.size())});
  for (uint32_t e = 0; e < D.envs; ++e)
    if (!B.load(e, R, &err)) fprintf(stderr, "[장면 뜨기] 판 %u 넣기 실패: %s\n", e, err.c_str());
  for (uint32_t e = 0; e < D.envs; ++e) {
    const sc::SceneShared& S = *B.shared(e);
    const eng::Body* bodies = B.envBodies(e);
    const eng::art::Articulation* A = B.envArts(e);
    const sc::SceneJoint* J = B.envJoints(e);
    const sc::ShapeFilter* FL = B.envFilters(e);
    const std::string env = "판 " + std::to_string(e) + " ";
    uint32_t ai = 0;
    auto chkShapes = [&](PxRigidActor* a, const sc::SceneActor& sa) {
      const PxU32 ns = a->getNbShapes();
      std::vector<PxShape*> shs(ns);
      a->getShapes(shs.data(), ns);
      D.chk(ns == sa.shapeCount, env + "모양 수 " + std::string(S.names.data() + sa.name));
      for (PxU32 k = 0; k < ns && k < sa.shapeCount; ++k) {
        const sc::SceneShape& o = S.shapes[sa.shapeStart + k];
        const PxTransform lp = shs[k]->getLocalPose();
        D.chkBits(lp, o.localPose, 28, env + "모양 자세");
        const float off[2] = {shs[k]->getContactOffset(), shs[k]->getRestOffset()};
        D.chkBits(off, o.contactOffset, 8, env + "모양 거리");
        const PxGeometry& pg = shs[k]->getGeometry();
        if (pg.getType() == PxGeometryType::eBOX) D.chkBits(static_cast<const PxBoxGeometry&>(pg).halfExtents, o.geom.box.halfExtents, 12, env + "상자");
        if (pg.getType() == PxGeometryType::eSPHERE) D.chkBits(static_cast<const PxSphereGeometry&>(pg).radius, o.geom.sphere.radius, 4, env + "구");
        if (pg.getType() == PxGeometryType::eCONVEXMESH) {
          const auto& cg = static_cast<const PxConvexMeshGeometry&>(pg);
          D.chkBits(cg.scale.scale, o.geom.convex.scale.scale, 12, env + "볼록 배율");
          const Gu::ConvexHullData& ph = static_cast<const Gu::ConvexMesh*>(cg.convexMesh)->getHull();
          const auto* oh = reinterpret_cast<const Gu::ConvexHullData*>(o.geom.convex.hullData);
          bool same = oh && ph.mNbPolygons == oh->mNbPolygons && ph.mNbHullVertices == oh->mNbHullVertices &&
                      !memcmp(ph.getHullVertices(), oh->getHullVertices(), ph.mNbHullVertices * 12) &&
                      !memcmp(ph.mPolygons, oh->mPolygons, ph.mNbPolygons * sizeof(Gu::HullPolygonData));
          D.chk(same, env + "볼록 덩어리");
        }
        const PxFilterData fd = shs[k]->getSimulationFilterData();
        D.chkBits(fd, FL[sa.shapeStart + k], 16, env + "거르기");
        PxMaterial* m0 = nullptr;
        if (shs[k]->getNbMaterials()) shs[k]->getMaterials(&m0, 1);
        if (m0) {
          const ec::MaterialData& md = S.materials[o.material];
          const float mf[3] = {m0->getDynamicFriction(), m0->getStaticFriction(), m0->getRestitution()};
          D.chkBits(mf, md, 12, env + "재질");
        }
      }
    };
    for (PxActor* a : acts) {
      const sc::SceneActor& sa = S.actors[ai++];
      if (sa.kind == sc::kStatic) {
        D.chkBits(static_cast<PxRigidStatic*>(a)->getGlobalPose(), sa.staticPose, 28, env + "정적 자세");
      } else {
        auto* rd = static_cast<PxRigidDynamic*>(a);
        const eng::Body& b = bodies[sa.body];
        const eng::Tf gp = eng::getGlobalPose(b);
        D.chkBits(rd->getGlobalPose(), gp, 28, env + "동적 자세 " + std::string(S.names.data() + sa.name));
        D.chkBits(rd->getLinearVelocity(), b.linVel, 12, env + "동적 선속도");
        D.chkBits(rd->getAngularVelocity(), b.angVel, 12, env + "동적 각속도");
        const float wc = rd->getWakeCounter();
        D.chkBits(wc, b.wakeCounter, 4, env + "동적 깸");
        const float im = rd->getInvMass();
        D.chkBits(im, b.invMass, 4, env + "동적 질량");
        D.chkBits(rd->getMassSpaceInvInertiaTensor(), b.invInertia, 12, env + "동적 관성");
        D.chkBits(rd->getCMassLocalPose(), b.body2Actor, 28, env + "동적 질량중심");
      }
      chkShapes(static_cast<PxRigidActor*>(a), sa);
    }
    for (uint32_t k = 0; k < arts.size(); ++k) {
      PxArticulationReducedCoordinate* a = arts[k];
      const eng::art::Articulation& ea = A[k];
      std::vector<PxArticulationLink*> links(a->getNbLinks());
      a->getLinks(links.data(), a->getNbLinks());
      for (uint32_t l = 0; l < links.size(); ++l) {
        const sc::SceneActor& sa = S.actors[ai++];
        const eng::Tf te = eng::art::linkGlobalPose(ea, l);
        D.chkBits(links[l]->getGlobalPose(), te, 28, env + "링크 자세 " + std::string(S.names.data() + sa.name));
        const eng::art::LinkBody& lb = ea.bodies[ea.ll[l]];
        D.chkBits(links[l]->getLinearVelocity(), lb.linVel, 12, env + "링크 선속도");
        D.chkBits(links[l]->getAngularVelocity(), lb.angVel, 12, env + "링크 각속도");
        chkShapes(links[l], sa);
      }
      if (ea.dofs) {
        PxArticulationCache* c = a->createCache();
        a->copyInternalStateToCache(*c, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        D.chk(!memcmp(c->jointPosition, ea.jointPosition, 4 * ea.dofs), env + "관절 위치 " + (a->getName() ? a->getName() : "?"));
        D.chk(!memcmp(c->jointVelocity, ea.jointVelocity, 4 * ea.dofs), env + "관절 속도");
        c->release();
      }
      const float wc = a->getWakeCounter();
      D.chkBits(wc, ea.wakeCounter, 4, env + "관절체 깸");
    }
    {
      const PxU32 nc = scene->getNbConstraints();
      std::vector<PxConstraint*> cs(nc);
      scene->getConstraints(cs.data(), nc);
      for (PxU32 k = 0; k < nc && k < B.nJoints[e]; ++k) {
        PxReal lf, af;
        cs[k]->getBreakForce(lf, af);
        D.chk(!memcmp(&lf, &J[k].linBreakForce, 4) && !memcmp(&af, &J[k].angBreakForce, 4), env + "조인트 끊김 힘");
        // 플래그: 우리 칸은 풀이가 쓰는 Dy::Constraint::flags(내부 표현, 공개 PxConstraintFlags 와 다름) — 진단으로 첫 몇 개만 보여 준다
        const uint16_t pub = uint16_t(PxU16(cs[k]->getFlags()));
        if (e == 0 && k < 3) printf("  조인트 %u 플래그: 공개 PxConstraintFlags %04x, Dy::Constraint %04x\n", k, pub, J[k].flags);
      }
    }
  }
  printf("엔진 장면 파일: %s (simulate %" PRIu64 " 앞) — 행위자 %zu (동적 몸체 %zu, 관절체 %zu, 링크 포함), 모양 %zu, 볼록 %zu (%.1f MB), 재질 %zu, D6 조인트 %zu, 못 옮긴 것 %" PRIu64 "\n",
         D.out.c_str(), sim, R.actors.size(), R.bodies.size(), R.arts.size(), R.shapes.size(), R.hullOffsets.size(), double(R.hulls.size()) / 1e6,
         R.materials.size(), R.joints.size(), D.unsup);
  printf("  판 %u 개 배치에 다시 넣음: 틀 %zu 벌 공유, 판당 상태 %.2f MB (몸체 %zu B, 관절체 %zu B/개, 조인트 %zu B/개)\n", D.envs, B.shareds.size(),
         double(B.stateBytes()) / D.envs / 1e6, sizeof(eng::Body), sizeof(eng::art::Articulation), sizeof(sc::SceneJoint));
  printf("  접촉 관리자 %zu (다양체 %zu, 마찰 패치 %zu, 캐시 못 찾음 %" PRIu64 ")\n", R.cms.size(), R.manifolds.size(), R.friction.size(), nNoCache);
  printf("  쌍 관리층 기록: %s (스텝 %zu)\n", R.pairs.valid ? "있음" : "없음(G1_PAIRS 로 떠야 함)", R.pairs.steps.size());
  printf("  넓은 단계 기록: %s (구조 변경 %zu, 스텝 %zu, %.1f MB)\n", R.bp.valid ? "있음" : "없음(G1_BP 로 떠야 함)", R.bp.ops.size(), R.bp.frames.size(),
         double(R.bp.bounds.size() * 4 + R.bp.dist.size() * 4 + R.bp.words.size() * 4) / 1e6);
  printf("  섬 관리자: 노드 %zu, 간선 %zu, 섬 %zu (활성 %zu)\n", R.islands.accurate.nodes.size(), R.islands.edgeNodeIndices.size() / 2, R.islands.accurate.islands.size(),
         R.islands.accurate.activeIslands.size());
  printf("  PhysX 공개 API 값과 비교 %" PRIu64 ", 비트 다름 %" PRIu64 "%s\n", D.cmp, D.bad, D.bad ? ("  첫 다름: " + D.firstBad).c_str() : "");
}

}  // namespace

// 장면 파일 모양 순서 = 정적·동적 행위자(PxScene::getActors) 순서, 그다음 관절체(getArticulations)의 링크(getLinks) 순서, 행위자 안은 getShapes 순서
// 새 행위자 틀 파일 (문서 20.4, particles 수확): 판 도중 넣은 강체들만 장면 파일 형식으로 (머리·재질 + 행위자·모양·볼록·몸체·거르개).
// 상태 쪽(접촉 관리자·섬·넓은 단계·쌍 기록)은 비움. g1_sc.cpp 가 편집 창 끝에 부른다.
bool g1_dump_actors(PxScene* scene, const std::vector<PxActor*>& actors, uint64_t sim, const char* path) {
  sc::SceneFile F;
  fillHeader(F, scene, sim);
  ShapeAdder SA(F);
  for (PxActor* a : actors) addRigid(F, SA, a, nullptr);
  return sc::writeScene(path, F);
}

std::vector<const void*> g1_shape_cores(PxScene* scene) {
  std::vector<const void*> out;
  auto add = [&](PxRigidActor* a) {
    const PxU32 ns = a->getNbShapes();
    std::vector<PxShape*> shs(ns);
    a->getShapes(shs.data(), ns);
    for (PxShape* s : shs) out.push_back(&static_cast<NpShape*>(s)->getCore().getCore());
  };
  const PxU32 na = scene->getNbActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC);
  std::vector<PxActor*> acts(na);
  scene->getActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC, acts.data(), na);
  for (PxActor* a : acts) add(static_cast<PxRigidActor*>(a));
  const PxU32 nArt = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(nArt);
  scene->getArticulations(arts.data(), nArt);
  for (PxArticulationReducedCoordinate* a : arts) {
    std::vector<PxArticulationLink*> links(a->getNbLinks());
    a->getLinks(links.data(), a->getNbLinks());
    for (PxArticulationLink* l : links) add(l);
  }
  return out;
}

// g1_shadow.cpp 의 g1_before_simulate 가 부른다 (simulate 바로 앞)
void g1_dump_before(PxScene* scene, uint64_t sim) {
  if (!D.inited) {
    D.inited = true;
    if (const char* a = getenv("G1_DUMP_AT")) D.at = atoll(a);
    if (const char* o = getenv("G1_DUMP_OUT")) D.out = o;
    if (const char* n = getenv("G1_DUMP_ENVS")) D.envs = uint32_t(atoi(n));
  }
  if (D.done || D.at < 0 || (long long)sim != D.at || D.out.empty()) return;
  D.done = true;
  dumpScene(scene, sim);
}
