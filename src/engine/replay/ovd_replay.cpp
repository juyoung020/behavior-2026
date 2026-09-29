// OVD 재생기 (층 0 정답지): 공식 평가기에서 뜬 OmniPVD 기록을 우리 PhysX 5.6.1 빌드로 처음부터 다시 만들고
// 같은 API 호출을 같은 순서로 넣은 뒤, simulate 마다 공식이 기록한 결과(자세·속도·관절값)와 비트 단위로 비교한다.
//   ovd_replay <file.ovd> [--convex convex.bin] [--filters filters.txt] [--sidelog sidelog.bin] [--threads N]
//              [--csv per_frame.csv] [--max-frames N] [--verbose]
// 원리 (PhysX 소스 기준)
//   - 객체는 만들어질 때 "create + 그 순간 속성 전부" 가 기록되고(OmniPvdPxSampler::onObjectAdd), 이어서
//     PxPhysics.<목록> add 가 온다 -> 그 시점에 우리 쪽 객체를 같은 인자로 만든다.
//   - 그 뒤의 set 은 API 호출이다 -> 같은 setter 를 부른다.
//   - simulate() 는 PxScene.elapsedTime set 으로 표시된다(NpScene.cpp:2839). 그 뒤 다음 stopFrame 까지는
//     fetchResults 가 적은 결과(NpSceneFetchResults.cpp:277-470) -> 적용하지 않고 비교만 한다.
//   - OVD 에 안 남는 호출(applyCache 계열, wakeUp/putToSleep)은 capture 의 sidelog 로 채운다.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "OmniPvdDefines.h"
#include "PxPhysicsAPI.h"
#include "omnipvd/PxOmniPvd.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdWriter.h"
#include "omni_filter.h"
#include "ovd.h"
#include "sidecar.h"

using namespace physx;

static PxDefaultAllocator gAlloc;
struct ErrCb : PxErrorCallback {
  int n = 0;
  void reportError(PxErrorCode::Enum code, const char* msg, const char* file, int line) override {
    if (++n <= 40) fprintf(stderr, "[PhysX %d] %s (%s:%d)\n", int(code), msg, file, line);
  }
} gErr;

// ------------------------------------------------------------------------------------------------ 정규화의 원상(原像)
// PhysX 의 몇몇 setter 는 입력 자세를 getNormalized() 한 뒤 저장하고, OVD 에는 정규화된 값이 남는다
// (setGlobalPose NpRigidDynamic.cpp:106, setCMassLocalPose :185, PxShape::setLocalPose NpShape.cpp:345,
//  createRigid* NpPhysics.cpp:454/522, createLink NpArticulationReducedCoordinate.cpp:987, PxJoint::setLocalPose ExtJoint.h:239).
// 기록값을 그대로 다시 넣으면 한 번 더 정규화되어 마지막 비트가 바뀔 수 있다. 그래서 getNormalized() 결과가
// 기록값과 비트까지 같은 입력을 찾아 넣는다 (내부 상태는 정규화된 값에만 의존하므로 이러면 원본과 같다).
static uint64_t g_prenorm_calls = 0, g_prenorm_fail = 0;
static bool qeq(const PxQuat& a, const PxQuat& b) { return memcmp(&a, &b, sizeof(PxQuat)) == 0; }
static PxQuat prenorm(const PxQuat& r) {
  g_prenorm_calls++;
  if (qeq(r.getNormalized(), r)) return r;
  for (int k = 1; k <= 64; ++k)
    for (int sg = -1; sg <= 1; sg += 2) {
      const float s = 1.0f + float(sg * k) * 5.9604645e-08f;  // 2^-24 단위로 전체 크기를 조금씩
      PxQuat q(r.x * s, r.y * s, r.z * s, r.w * s);
      if (qeq(q.getNormalized(), r)) return q;
    }
  float c[4] = {r.x, r.y, r.z, r.w};
  for (int a = -2; a <= 2; ++a)
    for (int b = -2; b <= 2; ++b)
      for (int d = -2; d <= 2; ++d)
        for (int e = -2; e <= 2; ++e) {
          int st[4] = {a, b, d, e};
          float t[4];
          for (int i = 0; i < 4; ++i) {
            t[i] = c[i];
            for (int j = 0; j < std::abs(st[i]); ++j) t[i] = std::nextafter(t[i], st[i] > 0 ? INFINITY : -INFINITY);
          }
          PxQuat q(t[0], t[1], t[2], t[3]);
          if (qeq(q.getNormalized(), r)) return q;
        }
  g_prenorm_fail++;
  return r;
}
static PxTransform prenorm(const PxTransform& t) { return PxTransform(t.p, prenorm(t.q)); }

// ------------------------------------------------------------------------------------------------ 재생기
struct Obj {
  uint32_t cls = 0;
  std::string name;
  bool done = false;
  PxBase* px = nullptr;
  PxScene* scene = nullptr;
  bool is_geom = false;
  std::unordered_map<uint32_t, std::vector<uint8_t>> pend;  // 만들 때 값
};

struct Stat {
  uint64_t compared = 0, bitdiff = 0;
  double maxdiff = 0;
  int64_t first_bit = -1, first_over = -1;
};

class Replayer {
 public:
  ovd::File& F;
  PxFoundation* fnd = nullptr;
  PxPhysics* phys = nullptr;
  PxDefaultCpuDispatcher* disp = nullptr;
  PxCookingParams* cook = nullptr;
  engine::FilterSpec spec;
  const engine::FilterSpec* specp = &spec;
  engine::OmniFilterCallback filter_cb;
  std::unordered_map<uint64_t, engine::ConvexData> convex;
  std::vector<engine::SideView> sviews;
  std::vector<engine::SideCall> scalls;
  size_t scall_next = 0;
  std::unordered_map<uint64_t, Obj> objs;
  std::unordered_map<std::string, uint32_t> A;  // "Class.attr" -> handle
  std::unordered_map<uint64_t, uint64_t> link_parent;  // link -> parent link (prescan)
  std::unordered_map<uint64_t, uint64_t> joint_child;  // art joint -> child link
  std::map<std::string, uint64_t> unsupported;
  std::map<std::string, uint64_t> applied;
  std::map<std::string, Stat> stats;  // 항목별 비교
  uint64_t sims = 0;
  int64_t max_frames = -1;
  bool verbose = false;
  FILE* csv = nullptr;
  double frame_max = 0;
  uint64_t frame_bitdiff = 0, frame_cmp = 0;
  int64_t first_div_frame = -1;
  std::string first_div_what;
  uint64_t convex_exact = 0, convex_approx = 0;
  std::unordered_map<std::string, PxArticulationReducedCoordinate*> art_by_name;
  std::unordered_map<std::string, PxRigidActor*> actor_by_name;
  std::unordered_map<PxArticulationReducedCoordinate*, PxArticulationCache*> caches;

  explicit Replayer(ovd::File& f) : F(f) {}

  uint32_t attr(const char* c, const char* a) {
    auto it = A.find(std::string(c) + "." + a);
    return it == A.end() ? 0xffffffffu : it->second;
  }
  const std::vector<uint8_t>* pv(Obj& o, const char* c, const char* a) {
    auto it = o.pend.find(attr(c, a));
    return it == o.pend.end() ? nullptr : &it->second;
  }
  template <class T> bool pget(Obj& o, const char* c, const char* a, T& out) {
    auto v = pv(o, c, a);
    if (!v || v->size() < sizeof(T)) return false;
    memcpy(&out, v->data(), sizeof(T));
    return true;
  }
  const char* cname(uint32_t c) { return F.classes[c].name.c_str(); }
  bool isa(const Obj& o, const char* c) { return F.is_a(o.cls, F.cls(c)); }

  std::string record_path;
  PxOmniPvd* rec_pvd = nullptr;
  bool init(int threads) {
    for (auto& kv : F.attrs) A[F.classes[kv.second.cls].name + "." + kv.second.name] = kv.first;
    fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    // 척도: OVD 의 PxPhysics.tolerancesScale (length, speed)
    PxTolerancesScale tol;
    for (auto& e : F.events)
      if (e.cmd == ovd::kSet && F.attr_name(e.attr) == "PxPhysics.tolerancesScale" && e.data_len >= 8) {
        memcpy(&tol.length, F.data(e), 4);
        memcpy(&tol.speed, F.data(e) + 4, 4);
        break;
      }
    // --record: 재생 쪽도 OVD 로 남겨 원본 OVD 와 명령 단위로 비교한다 (ovd_diff)
    PxOmniPvd* opvd = nullptr;
    if (!record_path.empty()) {
      opvd = PxCreateOmniPvd(*fnd);
      if (opvd) {
        OmniPvdFileWriteStream* fs = opvd->getFileWriteStream();
        fs->setFileName(record_path.c_str());
        opvd->getWriter()->setWriteStream(static_cast<OmniPvdWriteStream&>(*fs));
      }
    }
    phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol, true, nullptr, opvd);
    if (opvd) opvd->startSampling();
    rec_pvd = opvd;
    PxInitExtensions(*phys, nullptr);
    disp = PxDefaultCpuDispatcherCreate(threads);
    cook = new PxCookingParams(tol);
    cook->buildGPUData = false;            // omni.physx/plugins/Setup.cpp:869
    cook->buildTriangleAdjacencies = true;
    prescan();
    return phys != nullptr;
  }

  void prescan() {
    const uint32_t a_parent = attr("PxArticulationJointReducedCoordinate", "parentLink");
    const uint32_t a_child = attr("PxArticulationJointReducedCoordinate", "childLink");
    std::unordered_map<uint64_t, uint64_t> jparent;
    for (auto& e : F.events) {
      if (e.cmd != ovd::kSet) continue;
      if (e.attr == a_child && e.data_len == 8) { uint64_t v; memcpy(&v, F.data(e), 8); if (v && !joint_child.count(e.obj)) joint_child[e.obj] = v; }
      if (e.attr == a_parent && e.data_len == 8) { uint64_t v; memcpy(&v, F.data(e), 8); if (v && !jparent.count(e.obj)) jparent[e.obj] = v; }
    }
    for (auto& kv : joint_child) {
      auto it = jparent.find(kv.first);
      if (it != jparent.end()) link_parent[kv.second] = it->second;
    }
  }

  // ------------------------------------------------------------------ 객체 만들기
  PxGeometryHolder geometry(uint64_t h, bool& ok) {
    ok = false;
    auto it = objs.find(h);
    if (it == objs.end()) return PxGeometryHolder();
    Obj& g = it->second;
    std::string c = cname(g.cls);
    if (c == "PxBoxGeometry") {
      PxVec3 he; if (!pget(g, "PxBoxGeometry", "halfExtents", he)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxBoxGeometry(he));
    }
    if (c == "PxSphereGeometry") {
      float r; if (!pget(g, "PxSphereGeometry", "radius", r)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxSphereGeometry(r));
    }
    if (c == "PxCapsuleGeometry") {
      float r, hh; if (!pget(g, "PxCapsuleGeometry", "radius", r) || !pget(g, "PxCapsuleGeometry", "halfHeight", hh)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxCapsuleGeometry(r, hh));
    }
    if (c == "PxPlaneGeometry") { ok = true; return PxGeometryHolder(PxPlaneGeometry()); }
    if (c == "PxConvexMeshGeometry") {
      PxVec3 s; uint64_t m;
      if (!pget(g, "PxConvexMeshGeometry", "scale", s) || !pget(g, "PxConvexMeshGeometry", "convexMesh", m)) return PxGeometryHolder();
      auto im = objs.find(m);
      if (im == objs.end() || !im->second.px) return PxGeometryHolder();
      ok = true;  // OVD 에 척도 회전·geometry flags 가 없다 -> 항등 회전, flags 0 (추정; worldBounds 비교로 확인)
      return PxGeometryHolder(PxConvexMeshGeometry(static_cast<PxConvexMesh*>(im->second.px), PxMeshScale(s)));
    }
    if (c == "PxTriangleMeshGeometry") {
      PxVec3 s; uint64_t m;
      if (!pget(g, "PxTriangleMeshGeometry", "scale", s) || !pget(g, "PxTriangleMeshGeometry", "triangleMesh", m)) return PxGeometryHolder();
      auto im = objs.find(m);
      if (im == objs.end() || !im->second.px) return PxGeometryHolder();
      ok = true;
      return PxGeometryHolder(PxTriangleMeshGeometry(static_cast<PxTriangleMesh*>(im->second.px), PxMeshScale(s)));
    }
    unsupported["geometry:" + c]++;
    return PxGeometryHolder();
  }

  void make_scene(Obj& o) {
    PxTolerancesScale tol = phys->getTolerancesScale();
    { float ts[2]; if (pget(o, "PxScene", "tolerancesScale", ts)) { tol.length = ts[0]; tol.speed = ts[1]; } }
    PxSceneDesc sd(tol);
    uint32_t u32; float f; PxVec3 v;
    if (pget(o, "PxScene", "flags", u32)) sd.flags = PxSceneFlags(u32);
    sd.flags &= ~PxSceneFlags(PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_DIRECT_GPU_API);
    if (pget(o, "PxScene", "solverType", u32)) sd.solverType = PxSolverType::Enum(u32);
    if (pget(o, "PxScene", "broadPhaseType", u32)) sd.broadPhaseType = PxBroadPhaseType::Enum(u32);
    if (sd.broadPhaseType == PxBroadPhaseType::eGPU) sd.broadPhaseType = PxBroadPhaseType::ePABP;
    if (pget(o, "PxScene", "kineKineFilteringMode", u32)) sd.kineKineFilteringMode = PxPairFilteringMode::Enum(u32);
    if (pget(o, "PxScene", "staticKineFilteringMode", u32)) sd.staticKineFilteringMode = PxPairFilteringMode::Enum(u32);
    if (pget(o, "PxScene", "gravity", v)) sd.gravity = v;
    if (pget(o, "PxScene", "bounceThresholdVelocity", f)) sd.bounceThresholdVelocity = f;
    if (pget(o, "PxScene", "frictionOffsetThreshold", f)) sd.frictionOffsetThreshold = f;
    if (pget(o, "PxScene", "frictionCorrelationDistance", f)) sd.frictionCorrelationDistance = f;
    if (pget(o, "PxScene", "solverBatchSize", u32)) sd.solverBatchSize = u32;
    if (pget(o, "PxScene", "solverArticulationBatchSize", u32)) sd.solverArticulationBatchSize = u32;
    // nbContactDataBlocks / maxNbContactDataBlocks 는 OVD 에 "지금 쓰는 블록 수"(처음 0)로 남는다 -> 설명값이 아니라 기본값 유지
    if (pget(o, "PxScene", "maxBiasCoefficient", f)) sd.maxBiasCoefficient = f;
    if (pget(o, "PxScene", "contactReportStreamBufferSize", u32)) sd.contactReportStreamBufferSize = u32;
    if (pget(o, "PxScene", "ccdMaxPasses", u32)) sd.ccdMaxPasses = u32;
    if (pget(o, "PxScene", "ccdThreshold", f)) sd.ccdThreshold = f;
    if (pget(o, "PxScene", "ccdMaxSeparation", f)) sd.ccdMaxSeparation = f;
    if (pget(o, "PxScene", "wakeCounterResetValue", f)) sd.wakeCounterResetValue = f;
    if (pget(o, "PxScene", "limitsMaxNbActors", u32)) sd.limits.maxNbActors = u32;
    if (pget(o, "PxScene", "limitsMaxNbBodies", u32)) sd.limits.maxNbBodies = u32;
    if (pget(o, "PxScene", "limitsMaxNbStaticShapes", u32)) sd.limits.maxNbStaticShapes = u32;
    if (pget(o, "PxScene", "limitsMaxNbDynamicShapes", u32)) sd.limits.maxNbDynamicShapes = u32;
    if (pget(o, "PxScene", "limitsMaxNbAggregates", u32)) sd.limits.maxNbAggregates = u32;
    if (pget(o, "PxScene", "limitsMaxNbConstraints", u32)) sd.limits.maxNbConstraints = u32;
    if (pget(o, "PxScene", "limitsMaxNbRegions", u32)) sd.limits.maxNbRegions = u32;
    if (pget(o, "PxScene", "limitsMaxNbBroadPhaseOverlaps", u32)) sd.limits.maxNbBroadPhaseOverlaps = u32;
    PxBounds3 b;
    if (pget(o, "PxScene", "sanityBounds", b)) sd.sanityBounds = b;
    if (pget(o, "PxScene", "contactPairSlabSize", u32)) sd.contactPairSlabSize = u32;
    sd.cpuDispatcher = disp;
    sd.filterShader = engine::OmniFilterShader;
    sd.filterShaderData = &specp;
    sd.filterShaderDataSize = sizeof(specp);
    sd.filterCallback = &filter_cb;
    o.scene = phys->createScene(sd);
    o.px = nullptr;
    if (verbose) fprintf(stderr, "[scene] flags=0x%x solver=%d bp=%d\n", uint32_t(sd.flags), int(sd.solverType), int(sd.broadPhaseType));
  }

  void make_material(Obj& o) {
    float sf = 0.5f, df = 0.5f, r = 0.5f;
    pget(o, "PxMaterial", "staticFriction", sf);
    pget(o, "PxMaterial", "dynamicFriction", df);
    pget(o, "PxMaterial", "restitution", r);
    PxMaterial* m = phys->createMaterial(sf, df, r);
    // 만든 순간 값이 생성 기본값과 다를 때만 setter (원본에 없던 호출을 끼우지 않는다)
    uint32_t u; float f;
    if (auto fv = pv(o, "PxMaterial", "flags")) {
      PxMaterialFlags fl(PxU16(flagv(fv->data(), uint32_t(fv->size()))));
      if (fl != m->getFlags()) m->setFlags(fl);
    }
    if (pget(o, "PxMaterial", "frictionCombineMode", u) && PxCombineMode::Enum(u) != m->getFrictionCombineMode()) m->setFrictionCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "restitutionCombineMode", u) && PxCombineMode::Enum(u) != m->getRestitutionCombineMode()) m->setRestitutionCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "dampingCombineMode", u) && PxCombineMode::Enum(u) != m->getDampingCombineMode()) m->setDampingCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "damping", f)) { float c = m->getDamping(); if (memcmp(&c, &f, 4)) m->setDamping(f); }
    o.px = m;
  }

  void make_convex(Obj& o) {
    auto v = pv(o, "PxConvexMesh", "verts");
    if (!v) { unsupported["convex:no verts"]++; return; }
    uint64_t h = engine::hash_bytes(v->data(), v->size());
    auto it = convex.find(h);
    if (it != convex.end()) {
      const engine::ConvexData& c = it->second;
      std::vector<PxHullPolygon> polys(c.polys.size());
      for (size_t i = 0; i < polys.size(); ++i) {
        memcpy(polys[i].mPlane, c.polys[i].plane, 16);
        polys[i].mNbVerts = c.polys[i].nverts;
        polys[i].mIndexBase = c.polys[i].base;
      }
      std::vector<PxU32> idx(c.indices.begin(), c.indices.end());
      PxConvexMeshDesc d;
      d.points.count = PxU32(c.verts.size() / 3); d.points.stride = 12; d.points.data = c.verts.data();
      d.polygons.count = PxU32(polys.size()); d.polygons.stride = sizeof(PxHullPolygon); d.polygons.data = polys.data();
      d.indices.count = PxU32(idx.size()); d.indices.stride = 4; d.indices.data = idx.data();
      o.px = PxCreateConvexMesh(*cook, d, phys->getPhysicsInsertionCallback());
      if (o.px) { convex_exact++; return; }
    }
    // 보조 파일에 없으면 꼭짓점에서 다시 볼록 껍질 계산 (근사: 면 평면식이 원본과 비트 같다는 보장 없음)
    PxConvexMeshDesc d;
    d.points.count = PxU32(v->size() / 12); d.points.stride = 12; d.points.data = v->data();
    d.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    d.vertexLimit = 255;
    o.px = PxCreateConvexMesh(*cook, d, phys->getPhysicsInsertionCallback());
    convex_approx++;
  }

  void make_trimesh(Obj& o) {
    auto v = pv(o, "PxTriangleMesh", "verts");
    auto t = pv(o, "PxTriangleMesh", "tris");
    if (!v || !t) { unsupported["trimesh:no data"]++; return; }
    PxTriangleMeshDesc d;
    d.points.count = PxU32(v->size() / 12); d.points.stride = 12; d.points.data = v->data();
    d.triangles.count = PxU32(t->size() / 12); d.triangles.stride = 12; d.triangles.data = t->data();
    o.px = PxCreateTriangleMesh(*cook, d, phys->getPhysicsInsertionCallback());
  }

  void make_shape(Obj& o) {
    uint64_t gh = 0;
    if (!pget(o, "PxShape", "geom", gh)) { unsupported["shape:no geom"]++; return; }
    bool ok;
    PxGeometryHolder g = geometry(gh, ok);
    if (!ok) { unsupported["shape:geom unsupported"]++; return; }
    auto mv = pv(o, "PxShape", "materials");
    std::vector<PxMaterial*> mats;
    if (mv)
      for (size_t i = 0; i + 8 <= mv->size(); i += 8) {
        uint64_t mh; memcpy(&mh, mv->data() + i, 8);
        auto im = objs.find(mh);
        if (im != objs.end() && im->second.px) mats.push_back(static_cast<PxMaterial*>(im->second.px));
      }
    if (mats.empty()) { unsupported["shape:no material"]++; return; }
    uint8_t excl = 1; pget(o, "PxShape", "isExclusive", excl);
    uint32_t flags = uint32_t(PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE);
    { auto fv = pv(o, "PxShape", "shapeFlags"); if (fv) { flags = 0; memcpy(&flags, fv->data(), fv->size() < 4 ? fv->size() : 4); } }
    PxShape* s = phys->createShape(g.any(), mats.data(), PxU16(mats.size()), excl != 0, PxShapeFlags(PxU8(flags)));
    o.px = s;
    if (!s) return;
    // 생성 순간 값이 기본값과 다를 수 있는 것들 (createShape 인자에 없는 것)
    PxTransform lp; float f; PxFilterData fd;
    auto differs = [](const void* a, const void* b, size_t n) { return memcmp(a, b, n) != 0; };
    if (pget(o, "PxShape", "localPose", lp)) { PxTransform c = s->getLocalPose(); if (differs(&c, &lp, sizeof lp)) s->setLocalPose(prenorm(lp)); }
    if (pget(o, "PxShape", "contactOffset", f)) { float c = s->getContactOffset(); if (differs(&c, &f, 4)) s->setContactOffset(f); }
    if (pget(o, "PxShape", "restOffset", f)) { float c = s->getRestOffset(); if (differs(&c, &f, 4)) s->setRestOffset(f); }
    if (pget(o, "PxShape", "simulationFilterData", fd)) { PxFilterData c = s->getSimulationFilterData(); if (differs(&c, &fd, sizeof fd)) s->setSimulationFilterData(fd); }
    if (pget(o, "PxShape", "queryFilterData", fd)) { PxFilterData c = s->getQueryFilterData(); if (differs(&c, &fd, sizeof fd)) s->setQueryFilterData(fd); }
  }

  PxTransform pose_of(Obj& o) {
    PxTransform p(PxIdentity);
    pget(o, "PxRigidActor", "globalPose", p);
    return p;
  }

  void make_actor(Obj& o) {
    std::string c = cname(o.cls);
    if (c == "PxRigidStatic") o.px = phys->createRigidStatic(prenorm(pose_of(o)));
    else if (c == "PxRigidDynamic") o.px = phys->createRigidDynamic(prenorm(pose_of(o)));
  }

  void make_aggregate(Obj& o) {
    uint32_t maxShapes = 0; uint8_t self = 0; pget(o, "PxAggregate", "maxNbShapes", maxShapes); pget(o, "PxAggregate", "selfCollision", self);
    uint32_t maxActors = 256;  // OVD 에 없음 (용량일 뿐 결과와 무관)
    o.px = phys->createAggregate(maxActors, maxShapes ? maxShapes : 256, PxGetAggregateFilterHint(PxAggregateType::eGENERIC, self != 0));
  }

  void make_articulation(Obj& o) { o.px = phys->createArticulationReducedCoordinate(); }

  void make_link(uint64_t art_h, uint64_t link_h) {
    Obj& l = objs[link_h];
    auto ia = objs.find(art_h);
    if (ia == objs.end() || !ia->second.px) { unsupported["link:no articulation"]++; return; }
    auto* art = static_cast<PxArticulationReducedCoordinate*>(ia->second.px);
    PxArticulationLink* parent = nullptr;
    auto ip = link_parent.find(link_h);
    if (ip != link_parent.end()) {
      auto ol = objs.find(ip->second);
      if (ol != objs.end()) parent = static_cast<PxArticulationLink*>(ol->second.px);
    }
    l.px = art->createLink(parent, prenorm(pose_of(l)));
    l.done = true;
  }

  PxJoint* make_ext_joint(Obj& o) {
    std::string c = cname(o.cls);
    uint64_t a0 = 0, a1 = 0;
    PxTransform p0(PxIdentity), p1(PxIdentity);
    pget(o, "PxJoint", "actor0", a0); pget(o, "PxJoint", "actor1", a1);
    pget(o, "PxJoint", "actor0LocalPose", p0); pget(o, "PxJoint", "actor1LocalPose", p1);
    p0 = prenorm(p0); p1 = prenorm(p1);
    auto act = [&](uint64_t h) -> PxRigidActor* {
      auto it = objs.find(h);
      return (h && it != objs.end()) ? static_cast<PxRigidActor*>(it->second.px) : nullptr;
    };
    PxJoint* j = nullptr;
    if (c == "PxFixedJoint") j = PxFixedJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxD6Joint") j = PxD6JointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxRevoluteJoint") j = PxRevoluteJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxPrismaticJoint") j = PxPrismaticJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxSphericalJoint") j = PxSphericalJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxDistanceJoint") j = PxDistanceJointCreate(*phys, act(a0), p0, act(a1), p1);
    else unsupported["joint:" + c]++;
    return j;
  }

  // 만들 때 값 중 생성 인자에 없는 것을 setter 로 (관절 조인트·조인트·흉내 관절)
  void close_cluster(uint64_t h) {
    auto it = objs.find(h);
    if (it == objs.end() || it->second.done) return;
    Obj& o = it->second;
    std::string c = cname(o.cls);
    if (c == "PxArticulationJointReducedCoordinate") {
      auto ic = joint_child.find(h);
      if (ic != joint_child.end()) {
        auto ol = objs.find(ic->second);
        if (ol != objs.end() && ol->second.px) {
          o.px = static_cast<PxArticulationLink*>(ol->second.px)->getInboundJoint();
          o.done = true;
        }
      }
      return;
    }
    if (c == "PxArticulationMimicJoint") {
      uint64_t ah, ja, jb; uint32_t axa, axb; float gear = 0, off = 0, nf = 0, dr = 0;
      if (pget(o, c.c_str(), "articulation", ah) && pget(o, c.c_str(), "jointA", ja) && pget(o, c.c_str(), "jointB", jb) &&
          pget(o, c.c_str(), "axisA", axa) && pget(o, c.c_str(), "axisB", axb)) {
        pget(o, c.c_str(), "gearRatio", gear); pget(o, c.c_str(), "offset", off);
        pget(o, c.c_str(), "naturalFrequency", nf); pget(o, c.c_str(), "dampingRatio", dr);
        auto* art = static_cast<PxArticulationReducedCoordinate*>(objs[ah].px);
        auto* JA = static_cast<PxArticulationJointReducedCoordinate*>(objs[ja].px);
        auto* JB = static_cast<PxArticulationJointReducedCoordinate*>(objs[jb].px);
        if (art && JA && JB)
          o.px = art->createMimicJoint(*JA, PxArticulationAxis::Enum(axa), *JB, PxArticulationAxis::Enum(axb), gear, off, nf, dr);
        if (!o.px) unsupported["mimic joint create failed"]++;
        else if (verbose) fprintf(stderr, "[mimic] 만듦 art=%p A=%p B=%p\n", (void*)art, (void*)JA, (void*)JB);
      } else unsupported["mimic joint missing attrs"]++;
      o.done = true;
      return;
    }
    if (F.is_a(o.cls, F.cls("PxJoint"))) {
      PxJoint* j = make_ext_joint(o);
      o.px = j;
      o.done = true;
      if (j) {
        uint64_t ch = 0;
        if (pget(o, "PxJoint", "constraint", ch)) { Obj& co = objs[ch]; co.px = j->getConstraint(); co.done = true; }
        // 만든 순간 값(cluster)은 생성 기본값 + 생성 인자뿐이다. 기본값이 아닌 것은 뒤따르는 set(API 호출)으로 온다.
      }
      return;
    }
    if (c == "PxD6JointDrive") { o.done = true; return; }  // 값 객체: 조인트 setter 에서 읽는다
  }

  void apply_joint_cluster(Obj& o, PxJoint* j) {
    // 생성 인자에 없는 속성을 한 번에 (기본값이면 같은 값을 다시 넣는 것이라 무해)
    for (auto& kv : o.pend) {
      const std::string an = F.attrs[kv.first].name;
      if (an == "actor0" || an == "actor1" || an == "actor0LocalPose" || an == "actor1LocalPose" || an == "constraint" ||
          an == "type" || an == "concreteTypeName")
        continue;
      apply_set(o, kv.first, kv.second.data(), uint32_t(kv.second.size()), j);
    }
  }

  // ------------------------------------------------------------------ setter (API 호출 재현)
  template <class T> static T as(const uint8_t* p) { T v; memcpy(&v, p, sizeof(T)); return v; }
  // 플래그 속성은 타입마다 1/2/4 바이트 -> 길이만큼만 읽는다
  static uint32_t flagv(const uint8_t* p, uint32_t n) { uint32_t v = 0; memcpy(&v, p, n < 4 ? n : 4); return v; }
  static std::string cstr(const uint8_t* p, uint32_t n) { while (n && p[n - 1] == 0) --n; return std::string(reinterpret_cast<const char*>(p), n); }

  void apply_set(Obj& o, uint32_t ah, const uint8_t* p, uint32_t n, PxBase* target = nullptr) {
    const ovd::AttrInfo& ai = F.attrs[ah];
    const std::string cl = F.classes[ai.cls].name, an = ai.name;
    const std::string key = cl + "." + an;
    PxBase* b = target ? target : o.px;
    bool ok = true;
    if (cl == "PxScene") {
      PxScene* s = o.scene;
      if (!s) ok = false;
      else if (an == "gravity") s->setGravity(as<PxVec3>(p));
      else if (an == "bounceThresholdVelocity") s->setBounceThresholdVelocity(as<float>(p));
      else if (an == "flags") {
        PxSceneFlags want(flagv(p, n)), have = s->getFlags();
        for (uint32_t bit = 1; bit; bit <<= 1)
          if ((uint32_t(want) ^ uint32_t(have)) & bit) s->setFlag(PxSceneFlag::Enum(bit), (uint32_t(want) & bit) != 0);
      } else if (an == "frictionOffsetThreshold") s->setFrictionOffsetThreshold(as<float>(p));
      else if (an == "frictionCorrelationDistance") s->setFrictionCorrelationDistance(as<float>(p));
      else if (an == "ccdMaxPasses") s->setCCDMaxPasses(as<uint32_t>(p));
      else if (an == "ccdThreshold") s->setCCDThreshold(as<float>(p));
      else if (an == "ccdMaxSeparation") s->setCCDMaxSeparation(as<float>(p));
      else if (an == "maxBiasCoefficient") s->setMaxBiasCoefficient(as<float>(p));
      else ok = false;
    } else if (cl == "PxActor") {
      auto* a = b ? b->is<PxActor>() : nullptr;
      if (!a) ok = false;
      else if (an == "name") { o.name = cstr(p, n); a->setName(o.name.c_str()); register_name(o, a); }
      else if (an == "flags") a->setActorFlags(PxActorFlags(PxU8(flagv(p, n))));
      else if (an == "dominance") a->setDominanceGroup(p[0]);
      else if (an == "environmentID") a->setEnvironmentID(as<uint32_t>(p));
      else if (an == "worldBounds" || an == "type" || an == "ownerClient") {}
      else ok = false;
    } else if (cl == "PxRigidActor") {
      auto* a = b ? b->is<PxRigidActor>() : nullptr;
      if (!a) ok = false;
      else if (an == "globalPose") {
        if (a->is<PxArticulationLink>()) ok = false;  // 링크는 setGlobalPose 가 없다 (루트 자세는 applyCache/sidelog)
        else a->setGlobalPose(prenorm(as<PxTransform>(p)));
      } else ok = false;
    } else if (cl == "PxRigidBody") {
      auto* rb = b ? b->is<PxRigidBody>() : nullptr;
      auto* rd = b ? b->is<PxRigidDynamic>() : nullptr;
      if (!rb) ok = false;
      else if (an == "cMassLocalPose") rb->setCMassLocalPose(prenorm(as<PxTransform>(p)));
      else if (an == "mass") rb->setMass(as<float>(p));
      else if (an == "massSpaceInertiaTensor") rb->setMassSpaceInertiaTensor(as<PxVec3>(p));
      else if (an == "linearDamping") rb->setLinearDamping(as<float>(p));
      else if (an == "angularDamping") rb->setAngularDamping(as<float>(p));
      else if (an == "linearVelocity") { if (rd) rd->setLinearVelocity(as<PxVec3>(p)); else ok = false; }
      else if (an == "angularVelocity") { if (rd) rd->setAngularVelocity(as<PxVec3>(p)); else ok = false; }
      else if (an == "maxLinearVelocity") rb->setMaxLinearVelocity(as<float>(p));
      else if (an == "maxAngularVelocity") rb->setMaxAngularVelocity(as<float>(p));
      else if (an == "rigidBodyFlags") rb->setRigidBodyFlags(PxRigidBodyFlags(PxU16(flagv(p, n))));
      else if (an == "minAdvancedCCDCoefficient") rb->setMinCCDAdvanceCoefficient(as<float>(p));
      else if (an == "maxDepenetrationVelocity") rb->setMaxDepenetrationVelocity(as<float>(p));
      else if (an == "maxContactImpulse") rb->setMaxContactImpulse(as<float>(p));
      else if (an == "contactSlopCoefficient") rb->setContactSlopCoefficient(as<float>(p));
      else if (an == "force" || an == "torque") {}  // fetchResults 뒤 0 으로 되돌리는 기록일 뿐 (addForce 는 OVD 에 안 남음)
      else ok = false;
    } else if (cl == "PxRigidDynamic") {
      auto* rd = b ? b->is<PxRigidDynamic>() : nullptr;
      if (!rd) ok = false;
      else if (an == "sleepThreshold") rd->setSleepThreshold(as<float>(p));
      else if (an == "stabilizationThreshold") rd->setStabilizationThreshold(as<float>(p));
      else if (an == "rigidDynamicLockFlags") rd->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(PxU8(flagv(p, n))));
      else if (an == "wakeCounter") rd->setWakeCounter(as<float>(p));
      else if (an == "positionIterations") { PxU32 a0, v0; rd->getSolverIterationCounts(a0, v0); rd->setSolverIterationCounts(as<uint32_t>(p), v0); }
      else if (an == "velocityIterations") { PxU32 a0, v0; rd->getSolverIterationCounts(a0, v0); rd->setSolverIterationCounts(a0, as<uint32_t>(p)); }
      else if (an == "contactReportThreshold") rd->setContactReportThreshold(as<float>(p));
      else if (an == "isSleeping") {}
      else ok = false;
    } else if (cl == "PxArticulationLink") {
      auto* l = b ? b->is<PxArticulationLink>() : nullptr;
      if (!l) ok = false;
      else if (an == "CFMScale") l->setCfmScale(as<float>(p));
      else if (an == "inboundJoint" || an == "articulation" || an == "inboundJointDOF") {}
      else ok = false;
    } else if (cl == "PxArticulationReducedCoordinate") {
      auto* a = b ? b->is<PxArticulationReducedCoordinate>() : nullptr;
      if (!a) ok = false;
      else if (an == "name") { o.name = cstr(p, n); a->setName(o.name.c_str()); art_by_name[o.name] = a; }
      else if (an == "positionIterations") { PxU32 a0, v0; a->getSolverIterationCounts(a0, v0); a->setSolverIterationCounts(as<uint32_t>(p), v0); }
      else if (an == "velocityIterations") { PxU32 a0, v0; a->getSolverIterationCounts(a0, v0); a->setSolverIterationCounts(a0, as<uint32_t>(p)); }
      else if (an == "sleepThreshold") a->setSleepThreshold(as<float>(p));
      else if (an == "stabilizationThreshold") a->setStabilizationThreshold(as<float>(p));
      else if (an == "wakeCounter") a->setWakeCounter(as<float>(p));
      else if (an == "articulationFlags") a->setArticulationFlags(PxArticulationFlags(PxU8(flagv(p, n))));
      else if (an == "isSleeping" || an == "worldBounds" || an == "dofs") {}
      else ok = false;
    } else if (cl == "PxArticulationJointReducedCoordinate") {
      auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
      if (!j) ok = false;
      else ok = apply_art_joint(j, an, p, n);
    } else if (cl == "PxShape") {
      auto* s = b ? b->is<PxShape>() : nullptr;
      if (!s) ok = false;
      else if (an == "localPose") s->setLocalPose(prenorm(as<PxTransform>(p)));
      else if (an == "contactOffset") s->setContactOffset(as<float>(p));
      else if (an == "restOffset") s->setRestOffset(as<float>(p));
      else if (an == "torsionalPatchRadius") s->setTorsionalPatchRadius(as<float>(p));
      else if (an == "minTorsionalPatchRadius") s->setMinTorsionalPatchRadius(as<float>(p));
      else if (an == "shapeFlags") s->setFlags(PxShapeFlags(PxU8(flagv(p, n))));
      else if (an == "simulationFilterData") s->setSimulationFilterData(as<PxFilterData>(p));
      else if (an == "queryFilterData") s->setQueryFilterData(as<PxFilterData>(p));
      else if (an == "materials") {
        std::vector<PxMaterial*> mats;
        for (uint32_t i = 0; i + 8 <= n; i += 8) { auto im = objs.find(as<uint64_t>(p + i)); if (im != objs.end() && im->second.px) mats.push_back(static_cast<PxMaterial*>(im->second.px)); }
        if (!mats.empty()) s->setMaterials(mats.data(), PxU16(mats.size()));
      } else if (an == "densityForFluid" || an == "isExclusive") {}
      else if (an == "geom") {
        bool gok; PxGeometryHolder g = geometry(as<uint64_t>(p), gok);
        if (gok) s->setGeometry(g.any()); else ok = false;
      } else ok = false;
    } else if (cl == "PxMaterial") {
      auto* m = static_cast<PxMaterial*>(b);
      if (!m) ok = false;
      else if (an == "staticFriction") m->setStaticFriction(as<float>(p));
      else if (an == "dynamicFriction") m->setDynamicFriction(as<float>(p));
      else if (an == "restitution") m->setRestitution(as<float>(p));
      else if (an == "damping") m->setDamping(as<float>(p));
      else if (an == "flags") m->setFlags(PxMaterialFlags(PxU16(flagv(p, n))));
      else if (an == "frictionCombineMode") m->setFrictionCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else if (an == "restitutionCombineMode") m->setRestitutionCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else if (an == "dampingCombineMode") m->setDampingCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else ok = false;
    } else if (cl == "PxJoint" || F.is_a(ai.cls, F.cls("PxJoint"))) {
      ok = apply_ext_joint(o, static_cast<PxJoint*>(b), cl, an, p, n);
    } else if (cl == "PxAggregate") {
      if (an == "selfCollision" || an == "environmentID" || an == "maxNbShapes" || an == "scene") {} else ok = false;
    } else if (cl == "PxArticulationMimicJoint") {
      auto* m = static_cast<PxArticulationMimicJoint*>(b);
      if (!m) ok = false;
      else if (an == "gearRatio") m->setGearRatio(as<float>(p));
      else if (an == "offset") m->setOffset(as<float>(p));
      else if (an == "naturalFrequency") m->setNaturalFrequency(as<float>(p));
      else if (an == "dampingRatio") m->setDampingRatio(as<float>(p));
      else ok = false;
    } else ok = false;
    if (ok) applied[key]++;
    else unsupported["set:" + key]++;
  }

  bool apply_art_joint(PxArticulationJointReducedCoordinate* j, const std::string& an, const uint8_t* p, uint32_t n) {
    const int na = int(n / 4);
    auto f = [&](int i) { return as<float>(p + 4 * i); };
    auto u = [&](int i) { return as<uint32_t>(p + 4 * i); };
    auto AX = [](int i) { return PxArticulationAxis::Enum(i); };
    if (an == "type") { if (j->getJointType() != PxArticulationJointType::Enum(u(0))) j->setJointType(PxArticulationJointType::Enum(u(0))); return true; }
    if (an == "motion") { for (int i = 0; i < na && i < 6; ++i) if (j->getMotion(AX(i)) != PxArticulationMotion::Enum(u(i))) j->setMotion(AX(i), PxArticulationMotion::Enum(u(i))); return true; }
    if (an == "parentTranslation") { PxTransform t = j->getParentPose(); t.p = as<PxVec3>(p); j->setParentPose(t); return true; }
    if (an == "parentRotation") { PxTransform t = j->getParentPose(); t.q = as<PxQuat>(p); j->setParentPose(t); return true; }
    if (an == "childTranslation") { PxTransform t = j->getChildPose(); t.p = as<PxVec3>(p); j->setChildPose(t); return true; }
    if (an == "childRotation") { PxTransform t = j->getChildPose(); t.q = as<PxQuat>(p); j->setChildPose(t); return true; }
    if (an == "frictionCoefficient") { j->setFrictionCoefficient(f(0)); return true; }
    if (an == "maxJointVelocity") { j->setMaxJointVelocity(f(0)); return true; }
    if (an == "name") { static std::vector<std::unique_ptr<std::string>> keep; keep.emplace_back(new std::string(reinterpret_cast<const char*>(p), n)); j->setName(keep.back()->c_str()); return true; }
    if (an == "concreteTypeName" || an == "parentLink" || an == "childLink" || an == "jointForce") return true;
    // 축 6 개짜리 배열: 값이 바뀐 축만 setter 로 (원본은 축 하나씩 부른다)
    auto changed = [&](int i, float cur) { float v = f(i); return memcmp(&v, &cur, 4) != 0; };
    if (an == "armature") { for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getArmature(AX(i)))) j->setArmature(AX(i), f(i)); return true; }
    if (an == "jointPosition") { for (int i = 0; i < na && i < 6; ++i) if (j->getMotion(AX(i)) != PxArticulationMotion::eLOCKED && changed(i, j->getJointPosition(AX(i)))) j->setJointPosition(AX(i), f(i)); return true; }
    if (an == "jointVelocity") { for (int i = 0; i < na && i < 6; ++i) if (j->getMotion(AX(i)) != PxArticulationMotion::eLOCKED && changed(i, j->getJointVelocity(AX(i)))) j->setJointVelocity(AX(i), f(i)); return true; }
    if (an == "limitLow" || an == "limitHigh") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxArticulationLimit l = j->getLimitParams(AX(i));
        float& d = (an == "limitLow") ? l.low : l.high;
        if (memcmp(&d, p + 4 * i, 4)) { d = f(i); j->setLimitParams(AX(i), l); }
      }
      return true;
    }
    if (an == "driveStiffness" || an == "driveDamping" || an == "driveMaxForce" || an == "driveType" || an == "driveMaxEffort" ||
        an == "driveMaxActuatorVelocity" || an == "driveVelocityDependentResistance" || an == "driveSpeedEffortGradient") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxArticulationDrive d = j->getDriveParams(AX(i));
        bool ch = false;
        auto setf = [&](float& dst) { if (memcmp(&dst, p + 4 * i, 4)) { dst = f(i); ch = true; } };
        if (an == "driveStiffness") setf(d.stiffness);
        else if (an == "driveDamping") setf(d.damping);
        else if (an == "driveMaxForce") setf(d.maxForce);
        else if (an == "driveMaxEffort") setf(d.envelope.maxEffort);
        else if (an == "driveMaxActuatorVelocity") setf(d.envelope.maxActuatorVelocity);
        else if (an == "driveVelocityDependentResistance") setf(d.envelope.velocityDependentResistance);
        else if (an == "driveSpeedEffortGradient") setf(d.envelope.speedEffortGradient);
        else if (an == "driveType") { if (uint32_t(d.driveType) != u(i)) { d.driveType = PxArticulationDriveType::Enum(u(i)); ch = true; } }
        if (ch) j->setDriveParams(AX(i), d);
      }
      return true;
    }
    // 값이 같아도 원본은 setter 를 불렀다(자동 깨우기 autoWakeInternal 이 걸린다, NpArticulationJointReducedCoordinate.cpp:400)
    // -> 바뀐 축이 없으면 축 0 에 같은 값을 다시 넣어 부수효과를 똑같이 낸다.
    if (an == "driveTarget") {
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getDriveTarget(AX(i)))) { j->setDriveTarget(AX(i), f(i)); any = true; }
      if (!any) j->setDriveTarget(AX(0), j->getDriveTarget(AX(0)));
      return true;
    }
    if (an == "driveVelocity") {
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getDriveVelocity(AX(i)))) { j->setDriveVelocity(AX(i), f(i)); any = true; }
      if (!any) j->setDriveVelocity(AX(0), j->getDriveVelocity(AX(0)));
      return true;
    }
    if (an == "staticFrictionEffort" || an == "dynamicFrictionEffort" || an == "viscousFrictionCoefficient") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxJointFrictionParams fp = j->getFrictionParams(AX(i));
        float& d = an == "staticFrictionEffort" ? fp.staticFrictionEffort : (an == "dynamicFrictionEffort" ? fp.dynamicFrictionEffort : fp.viscousFrictionCoefficient);
        if (memcmp(&d, p + 4 * i, 4)) { d = f(i); j->setFrictionParams(AX(i), fp); }
      }
      return true;
    }
    if (an == "maxJointDofVelocity") { for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getMaxJointVelocity(AX(i)))) j->setMaxJointVelocity(AX(i), f(i)); return true; }
    return false;
  }

  // ------------------------------------------------------------------ 묶음 setter
  // 원본 API 한 번이 속성 여러 개를 남기는 것들 (PhysX 소스에서 확인):
  //   setSolverIterationCounts -> positionIterations, velocityIterations (NpRigidDynamic.cpp:543, NpArticulationReducedCoordinate.cpp:1010)
  //   PxJoint::setLocalPose -> actor0LocalPose, actor1LocalPose (ExtJoint.h:245), setBreakForce -> breakForce, breakTorque
  //   setParentPose / setChildPose -> *Translation, *Rotation (NpArticulationJointReducedCoordinate.cpp:498)
  //   setLimitParams -> limitLow, limitHigh (:307), setFrictionParams -> 3 개 (:220), setDriveParams -> 8 개 (:345~)
  //   PxD6Joint::setDriveVelocity -> driveLinVelocity, driveAngVelocity
  const std::vector<std::string>* group_of(Obj& o, uint32_t ah) {
    static const std::vector<std::string> iters = {"positionIterations", "velocityIterations"};
    static const std::vector<std::string> jlocal = {"actor0LocalPose", "actor1LocalPose"};
    static const std::vector<std::string> jbreak = {"breakForce", "breakTorque"};
    static const std::vector<std::string> ppose = {"parentTranslation", "parentRotation"};
    static const std::vector<std::string> cpose = {"childTranslation", "childRotation"};
    static const std::vector<std::string> limits = {"limitLow", "limitHigh"};
    static const std::vector<std::string> fric = {"staticFrictionEffort", "dynamicFrictionEffort", "viscousFrictionCoefficient"};
    static const std::vector<std::string> drive = {"driveStiffness", "driveDamping", "driveMaxForce", "driveMaxEffort",
                                                   "driveMaxActuatorVelocity", "driveVelocityDependentResistance",
                                                   "driveSpeedEffortGradient", "driveType"};
    static const std::vector<std::string> d6vel = {"driveLinVelocity", "driveAngVelocity"};
    const ovd::AttrInfo& ai = F.attrs[ah];
    const std::string& cl = F.classes[ai.cls].name;
    const std::string& an = ai.name;
    auto in = [&](const std::vector<std::string>& g) { for (auto& x : g) if (x == an) return true; return false; };
    if ((cl == "PxRigidDynamic" || cl == "PxArticulationReducedCoordinate") && in(iters)) return &iters;
    if (cl == "PxJoint" && in(jlocal)) return &jlocal;
    if (cl == "PxJoint" && in(jbreak)) return &jbreak;
    if (cl == "PxD6Joint" && in(d6vel)) return &d6vel;
    if (cl == "PxArticulationJointReducedCoordinate") {
      if (in(ppose)) return &ppose;
      if (in(cpose)) return &cpose;
      if (in(limits)) return &limits;
      if (in(fric)) return &fric;
      if (in(drive)) return &drive;
    }
    (void)o;
    return nullptr;
  }

  void apply_group(Obj& o, const std::vector<std::string>& g, std::unordered_map<std::string, const ovd::Event*>& v) {
    auto has = [&](const char* n) { return v.count(n) != 0; };
    auto dat = [&](const char* n) { return F.data(*v[n]); };
    auto AX = [](int i) { return PxArticulationAxis::Enum(i); };
    const std::string first = g[0];
    PxBase* b = o.px;
    if (!b) { unsupported["group:no object " + first]++; return; }
    if (first == "positionIterations") {
      PxU32 pi, vi;
      if (auto* rd = b->is<PxRigidDynamic>()) {
        rd->getSolverIterationCounts(pi, vi);
        if (has("positionIterations")) pi = as<uint32_t>(dat("positionIterations"));
        if (has("velocityIterations")) vi = as<uint32_t>(dat("velocityIterations"));
        rd->setSolverIterationCounts(pi, vi);
      } else if (auto* ar = b->is<PxArticulationReducedCoordinate>()) {
        ar->getSolverIterationCounts(pi, vi);
        if (has("positionIterations")) pi = as<uint32_t>(dat("positionIterations"));
        if (has("velocityIterations")) vi = as<uint32_t>(dat("velocityIterations"));
        ar->setSolverIterationCounts(pi, vi);
      }
      applied["group:iterations"]++;
      return;
    }
    if (first == "actor0LocalPose") {
      auto* j = static_cast<PxJoint*>(b);
      bool any = false;
      for (int a = 0; a < 2; ++a) {
        const char* n = a ? "actor1LocalPose" : "actor0LocalPose";
        if (!has(n)) continue;
        PxTransform want = as<PxTransform>(dat(n)), cur = j->getLocalPose(PxJointActorIndex::Enum(a));
        if (memcmp(&want, &cur, sizeof want)) { j->setLocalPose(PxJointActorIndex::Enum(a), prenorm(want)); any = true; }
      }
      if (!any) j->setLocalPose(PxJointActorIndex::eACTOR0, j->getLocalPose(PxJointActorIndex::eACTOR0));
      applied["group:jointLocalPose"]++;
      return;
    }
    if (first == "breakForce") {
      auto* j = static_cast<PxJoint*>(b);
      float f0, t0; j->getBreakForce(f0, t0);
      if (has("breakForce")) f0 = as<float>(dat("breakForce"));
      if (has("breakTorque")) t0 = as<float>(dat("breakTorque"));
      j->setBreakForce(f0, t0);
      applied["group:breakForce"]++;
      return;
    }
    if (first == "driveLinVelocity") {
      auto* d = static_cast<PxJoint*>(b)->is<PxD6Joint>();
      if (!d) return;
      PxVec3 l, a; d->getDriveVelocity(l, a);
      if (has("driveLinVelocity")) l = as<PxVec3>(dat("driveLinVelocity"));
      if (has("driveAngVelocity")) a = as<PxVec3>(dat("driveAngVelocity"));
      d->setDriveVelocity(l, a);
      applied["group:d6DriveVelocity"]++;
      return;
    }
    auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
    if (first == "parentTranslation" || first == "childTranslation") {
      const bool par = first == "parentTranslation";
      PxTransform t = par ? j->getParentPose() : j->getChildPose();
      if (has(par ? "parentTranslation" : "childTranslation")) t.p = as<PxVec3>(dat(par ? "parentTranslation" : "childTranslation"));
      if (has(par ? "parentRotation" : "childRotation")) t.q = as<PxQuat>(dat(par ? "parentRotation" : "childRotation"));
      if (par) j->setParentPose(t); else j->setChildPose(t);  // 입력 그대로 기록되므로 원상 찾기 불필요
      applied[par ? "group:parentPose" : "group:childPose"]++;
      return;
    }
    auto fa = [&](const char* n, int i) { return as<float>(dat(n) + 4 * i); };
    auto same = [](float a, float b) { return memcmp(&a, &b, 4) == 0; };
    if (first == "limitLow") {
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxArticulationLimit l = j->getLimitParams(AX(i));
        PxArticulationLimit w = l;
        if (has("limitLow")) w.low = fa("limitLow", i);
        if (has("limitHigh")) w.high = fa("limitHigh", i);
        if (!same(w.low, l.low) || !same(w.high, l.high)) { j->setLimitParams(AX(i), w); any = true; }
      }
      if (!any) j->setLimitParams(AX(0), j->getLimitParams(AX(0)));
      applied["group:limitParams"]++;
      return;
    }
    if (first == "staticFrictionEffort") {
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxJointFrictionParams c = j->getFrictionParams(AX(i)), w = c;
        if (has("staticFrictionEffort")) w.staticFrictionEffort = fa("staticFrictionEffort", i);
        if (has("dynamicFrictionEffort")) w.dynamicFrictionEffort = fa("dynamicFrictionEffort", i);
        if (has("viscousFrictionCoefficient")) w.viscousFrictionCoefficient = fa("viscousFrictionCoefficient", i);
        if (memcmp(&c, &w, sizeof c)) { j->setFrictionParams(AX(i), w); any = true; }
      }
      if (!any) j->setFrictionParams(AX(0), j->getFrictionParams(AX(0)));
      applied["group:frictionParams"]++;
      return;
    }
    if (first == "driveStiffness") {
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxArticulationDrive c = j->getDriveParams(AX(i)), w = c;
        if (has("driveStiffness")) w.stiffness = fa("driveStiffness", i);
        if (has("driveDamping")) w.damping = fa("driveDamping", i);
        if (has("driveMaxForce")) w.maxForce = fa("driveMaxForce", i);
        if (has("driveMaxEffort")) w.envelope.maxEffort = fa("driveMaxEffort", i);
        if (has("driveMaxActuatorVelocity")) w.envelope.maxActuatorVelocity = fa("driveMaxActuatorVelocity", i);
        if (has("driveVelocityDependentResistance")) w.envelope.velocityDependentResistance = fa("driveVelocityDependentResistance", i);
        if (has("driveSpeedEffortGradient")) w.envelope.speedEffortGradient = fa("driveSpeedEffortGradient", i);
        if (has("driveType")) w.driveType = PxArticulationDriveType::Enum(as<uint32_t>(dat("driveType") + 4 * i));
        const bool ch = !same(c.stiffness, w.stiffness) || !same(c.damping, w.damping) || !same(c.maxForce, w.maxForce) ||
                        !same(c.envelope.maxEffort, w.envelope.maxEffort) ||
                        !same(c.envelope.maxActuatorVelocity, w.envelope.maxActuatorVelocity) ||
                        !same(c.envelope.velocityDependentResistance, w.envelope.velocityDependentResistance) ||
                        !same(c.envelope.speedEffortGradient, w.envelope.speedEffortGradient) || c.driveType != w.driveType;
        if (ch) { j->setDriveParams(AX(i), w); any = true; }
      }
      if (!any) j->setDriveParams(AX(0), j->getDriveParams(AX(0)));
      applied["group:driveParams"]++;
      return;
    }
    unsupported["group:" + first]++;
  }

  bool apply_ext_joint(Obj& o, PxJoint* j, const std::string& cl, const std::string& an, const uint8_t* p, uint32_t n) {
    if (!j) return false;
    if (cl == "PxJoint") {
      if (an == "actor0LocalPose") { j->setLocalPose(PxJointActorIndex::eACTOR0, prenorm(as<PxTransform>(p))); return true; }
      if (an == "actor1LocalPose") { j->setLocalPose(PxJointActorIndex::eACTOR1, prenorm(as<PxTransform>(p))); return true; }
      if (an == "breakForce") { float f0, t0; j->getBreakForce(f0, t0); j->setBreakForce(as<float>(p), t0); return true; }
      if (an == "breakTorque") { float f0, t0; j->getBreakForce(f0, t0); j->setBreakForce(f0, as<float>(p)); return true; }
      if (an == "constraintFlags") { j->setConstraintFlags(PxConstraintFlags(PxU16(flagv(p, n)))); return true; }
      if (an == "invMassScale0") { j->setInvMassScale0(as<float>(p)); return true; }
      if (an == "invInertiaScale0") { j->setInvInertiaScale0(as<float>(p)); return true; }
      if (an == "invMassScale1") { j->setInvMassScale1(as<float>(p)); return true; }
      if (an == "invInertiaScale1") { j->setInvInertiaScale1(as<float>(p)); return true; }
      if (an == "name") { o.name = cstr(p, n); j->setName(o.name.c_str()); return true; }
      if (an == "actor0" || an == "actor1" || an == "constraint" || an == "type" || an == "concreteTypeName") return true;
      return false;
    }
    if (cl == "PxFixedJoint") return true;
    if (cl == "PxD6Joint") {
      auto* d = j->is<PxD6Joint>();
      if (!d) return false;
      if (an == "motions") { for (uint32_t i = 0; i < n / 4 && i < 6; ++i) d->setMotion(PxD6Axis::Enum(i), PxD6Motion::Enum(as<uint32_t>(p + 4 * i))); return true; }
      if (an == "drivePosition") { d->setDrivePosition(prenorm(as<PxTransform>(p))); return true; }
      if (an == "driveLinVelocity") { PxVec3 l, a; d->getDriveVelocity(l, a); d->setDriveVelocity(as<PxVec3>(p), a); return true; }
      if (an == "driveAngVelocity") { PxVec3 l, a; d->getDriveVelocity(l, a); d->setDriveVelocity(l, as<PxVec3>(p)); return true; }
      if (an == "twistAngle" || an == "swingYAngle" || an == "swingZAngle") return true;  // 결과값
      // 한계·드라이브: 첫 버전에서는 기록만 (BEHAVIOR 에서 D6 를 쓰는지 OVD 로 확인 후 채운다)
      return false;
    }
    if (cl == "PxRevoluteJoint" || cl == "PxPrismaticJoint" || cl == "PxSphericalJoint") {
      if (an == "angle" || an == "velocity" || an == "position" || an == "swingYAngle" || an == "swingZAngle") return true;  // 결과값
      return false;
    }
    return false;
  }

  void register_name(Obj& o, PxActor* a) {
    if (auto* ra = a->is<PxRigidActor>()) actor_by_name[o.name] = ra;
  }

  // ------------------------------------------------------------------ 목록 add/remove (구조 변경)
  void list_op(const ovd::Event& e, bool add) {
    const ovd::AttrInfo& ai = F.attrs[e.attr];
    const std::string key = F.classes[ai.cls].name + "." + ai.name;
    uint64_t item = 0;
    if (e.data_len >= 8) memcpy(&item, F.data(e), 8);
    Obj& owner = objs[e.obj];
    auto itm = objs.find(item);
    if (key.rfind("PxPhysics.", 0) == 0) {
      if (!add || itm == objs.end()) return;
      Obj& o = itm->second;
      if (o.done) return;
      const std::string list = ai.name;
      if (list == "scenes") make_scene(o);
      else if (list == "materials") make_material(o);
      else if (list == "convexMeshes") make_convex(o);
      else if (list == "triangleMeshes") make_trimesh(o);
      else if (list == "shapes") make_shape(o);
      else if (list == "rigidDynamics" || list == "rigidStatics") make_actor(o);
      else if (list == "aggregates") make_aggregate(o);
      else if (list == "articulations") make_articulation(o);
      else if (list == "constraints") { return; }  // 조인트가 만든다
      else { unsupported["physics list:" + list]++; }
      o.done = true;
      return;
    }
    if (key == "PxArticulationReducedCoordinate.links") { if (add) make_link(e.obj, item); return; }
    if (itm == objs.end()) {
      if (key == "PxScene.constraints") return;  // 조인트가 지워진 뒤 오는 제거 기록
      unsupported["list item unknown:" + key]++;
      return;
    }
    Obj& o = itm->second;
    if (key == "PxRigidActor.shapes") {
      auto* a = owner.px ? owner.px->is<PxRigidActor>() : nullptr;
      auto* s = o.px ? o.px->is<PxShape>() : nullptr;
      if (!a || !s) { unsupported["attach missing"]++; return; }
      if (add) a->attachShape(*s); else a->detachShape(*s);
      return;
    }
    if (key == "PxScene.actors") {
      PxScene* sc = owner.scene;
      auto* a = o.px ? o.px->is<PxActor>() : nullptr;
      if (!sc && !add) return;  // 장면을 지운 뒤 따라오는 제거 기록
      if (!sc || !a) { unsupported[std::string("scene actor missing:") + (sc ? "" : "noscene ") + (a ? "" : cname(o.cls))]++; return; }
      if (add) { if (!a->getScene() && !a->getAggregate()) sc->addActor(*a); }
      else if (a->getScene() && !a->getAggregate()) sc->removeActor(*a);
      return;
    }
    if (key == "PxScene.articulations") {
      auto* a = o.px ? o.px->is<PxArticulationReducedCoordinate>() : nullptr;
      if (!owner.scene || !a) return;
      if (add) { if (!a->getScene() && !a->getAggregate()) owner.scene->addArticulation(*a); }
      else if (a->getScene() && !a->getAggregate()) owner.scene->removeArticulation(*a);
      return;
    }
    if (key == "PxScene.aggregates") {
      auto* g = static_cast<PxAggregate*>(o.px);
      if (!owner.scene || !g) return;
      if (add) owner.scene->addAggregate(*g); else owner.scene->removeAggregate(*g);
      return;
    }
    if (key == "PxAggregate.actors") {
      auto* g = static_cast<PxAggregate*>(owner.px);
      if (!g || !o.px) return;
      if (auto* art = o.px->is<PxArticulationReducedCoordinate>()) { if (add) g->addArticulation(*art); else g->removeArticulation(*art); return; }
      if (auto* l = o.px->is<PxArticulationLink>()) {  // 관절체를 묶음에 넣으면 링크마다 기록된다 -> 첫 링크에서 관절체째로
        PxArticulationReducedCoordinate& art = l->getArticulation();
        if (add && !art.getAggregate()) g->addArticulation(art);
        if (!add && art.getAggregate()) g->removeArticulation(art);
        return;
      }
      if (auto* a = o.px->is<PxActor>()) { if (add) g->addActor(*a); else g->removeActor(*a); }
      return;
    }
    if (key == "PxScene.constraints") return;  // 조인트 액터가 장면에 들어가면 자동
    unsupported["list:" + key]++;
  }

  void destroy(uint64_t h) {
    auto it = objs.find(h);
    if (it == objs.end()) return;
    Obj& o = it->second;
    std::string c = cname(o.cls);
    if (o.px) {
      if (c == "PxArticulationLink" || c == "PxArticulationJointReducedCoordinate" || c == "PxConstraint") {}  // 주인이 지운다
      else if (F.is_a(o.cls, F.cls("PxJoint"))) static_cast<PxJoint*>(o.px)->release();
      else if (c == "PxArticulationMimicJoint") static_cast<PxArticulationMimicJoint*>(o.px)->release();
      else o.px->release();
    }
    if (o.scene) o.scene->release();
    objs.erase(it);
  }

  // ------------------------------------------------------------------ sidelog 적용 (OVD 에 없는 호출)
  PxArticulationCache* cache_of(PxArticulationReducedCoordinate* a) {
    auto it = caches.find(a);
    if (it != caches.end()) return it->second;
    PxArticulationCache* c = a->createCache();
    caches[a] = c;
    return c;
  }

  void apply_side(const engine::SideCall& c) {
    const engine::SideView& v = sviews[c.view];
    const std::string& m = c.method;
    auto arti = [&](uint32_t i) -> PxArticulationReducedCoordinate* {
      if (i >= v.prims.size()) return nullptr;
      auto ia = art_by_name.find(v.prims[i]);
      if (ia != art_by_name.end()) return ia->second;
      auto il = actor_by_name.find(v.prims[i]);  // 강체 뷰의 루트 링크
      if (il != actor_by_name.end()) if (auto* l = il->second->is<PxArticulationLink>()) return &l->getArticulation();
      return nullptr;
    };
    std::vector<uint32_t> idx = c.idx;
    if (idx.empty()) for (uint32_t i = 0; i < v.prims.size(); ++i) idx.push_back(i);
    for (uint32_t i : idx) {
      if (m == "set_dof_positions" || m == "set_dof_velocities" || m == "set_dof_actuation_forces") {
        auto* a = arti(i);
        if (!a) { unsupported["side:no art " + (i < v.prims.size() ? v.prims[i] : std::string("?"))]++; continue; }
        PxArticulationCache* ca = cache_of(a);
        const PxU32 nd = a->getDofs();
        PxReal* dst = m == "set_dof_positions" ? ca->jointPosition : (m == "set_dof_velocities" ? ca->jointVelocity : ca->jointForce);
        for (PxU32 j = 0; j < nd; ++j) {
          float s = (i < v.signs.size() && j < v.signs[i].size() && v.signs[i][j] < 0) ? -1.0f : 1.0f;
          float x = c.data[size_t(i) * v.max_dofs + j];
          dst[j] = s < 0 ? -x : x;
        }
        a->applyCache(*ca, m == "set_dof_positions" ? PxArticulationCacheFlag::ePOSITION
                                                   : (m == "set_dof_velocities" ? PxArticulationCacheFlag::eVELOCITY : PxArticulationCacheFlag::eFORCE));
        applied["side:" + m]++;
      } else if (m == "set_root_transforms" || (m == "set_transforms" && v.kind == 1)) {
        const float* s = &c.data[size_t(i) * 7];
        PxTransform t(PxVec3(s[0], s[1], s[2]), PxQuat(s[3], s[4], s[5], s[6]));
        auto* a = arti(i);
        if (a) { PxArticulationCache* ca = cache_of(a); ca->rootLinkData->transform = t; a->applyCache(*ca, PxArticulationCacheFlag::eROOT_TRANSFORM); applied["side:" + m]++; }
        // 강체 동체는 setGlobalPose 로 OVD 에 이미 남는다
      } else if (m == "set_root_velocities" || (m == "set_velocities" && v.kind == 1)) {
        const float* s = &c.data[size_t(i) * 6];
        auto* a = arti(i);
        if (a) {
          PxArticulationCache* ca = cache_of(a);
          ca->rootLinkData->worldLinVel = PxVec3(s[0], s[1], s[2]);
          ca->rootLinkData->worldAngVel = PxVec3(s[3], s[4], s[5]);
          a->applyCache(*ca, PxArticulationCacheFlag::eROOT_VELOCITIES);
          applied["side:" + m]++;
        }
      } else if (m == "add_force") {  // 강체 동체에 힘 (PxForceMode::eFORCE, 자동 깨우기)
        auto ib = actor_by_name.find(v.prims[i]);
        if (ib != actor_by_name.end()) if (auto* rd = ib->second->is<PxRigidDynamic>()) {
          const float* s3 = &c.data[size_t(i) * 3];
          rd->addForce(PxVec3(s3[0], s3[1], s3[2]));
          applied["side:add_force"]++;
        }
      } else if (m == "wake_up" || m == "put_to_sleep") {
        const std::string& path = v.prims[i];
        auto ia = art_by_name.find(path);
        if (ia != art_by_name.end()) { if (m == "wake_up") ia->second->wakeUp(); else ia->second->putToSleep(); applied["side:" + m]++; continue; }
        auto ib = actor_by_name.find(path);
        if (ib != actor_by_name.end()) if (auto* rd = ib->second->is<PxRigidDynamic>()) { if (m == "wake_up") rd->wakeUp(); else rd->putToSleep(); applied["side:" + m]++; }
      } else {
        // 목표값(드라이브)·질량 등은 OVD 에 이미 남는다
        applied["side-skip:" + m]++;
      }
    }
  }

  void apply_side_until(uint64_t after) {
    while (scall_next < scalls.size() && scalls[scall_next].after <= after) apply_side(scalls[scall_next++]);
  }

  // ------------------------------------------------------------------ 비교
  int shown = 0;
  std::string cur_obj;
  void cmp(const std::string& what, const float* ours, const uint8_t* rec, int n) {
    Stat& s = stats[what];
    if (verbose && shown < 12 && memcmp(ours, rec, 4 * n)) {
      shown++;
      fprintf(stderr, "[다름] sim %" PRIu64 " %s %s\n   우리:", sims, what.c_str(), cur_obj.c_str());
      for (int i = 0; i < n; ++i) fprintf(stderr, " %.9g", ours[i]);
      fprintf(stderr, "\n   공식:");
      for (int i = 0; i < n; ++i) { float r; memcpy(&r, rec + 4 * i, 4); fprintf(stderr, " %.9g", r); }
      fprintf(stderr, "\n");
    }
    double m = 0;
    bool bit = memcmp(ours, rec, 4 * n) != 0;
    for (int i = 0; i < n; ++i) { float r; memcpy(&r, rec + 4 * i, 4); double d = std::fabs(double(ours[i]) - double(r)); if (d > m || std::isnan(d)) m = std::isnan(d) ? INFINITY : d; }
    s.compared++;
    frame_cmp++;
    if (bit) {
      s.bitdiff++;
      frame_bitdiff++;
      if (s.first_bit < 0) s.first_bit = int64_t(sims);
      if (first_div_frame < 0) { first_div_frame = int64_t(sims); first_div_what = what; }
    }
    if (m > s.maxdiff) s.maxdiff = m;
    if (m > frame_max) frame_max = m;
  }

  void compare_output(const ovd::Event& e) {
    const ovd::AttrInfo& ai = F.attrs[e.attr];
    const std::string& an = ai.name;
    auto it = objs.find(e.obj);
    if (it == objs.end() || !it->second.px) return;
    PxBase* b = it->second.px;
    const std::string cl = cname(it->second.cls);
    cur_obj = it->second.name;
    if (an == "globalPose" && e.data_len == 28) {
      if (auto* a = b->is<PxRigidActor>()) { PxTransform t = a->getGlobalPose(); cmp(cl + ".pose", &t.q.x, F.data(e), 7); }
    } else if ((an == "linearVelocity" || an == "angularVelocity") && e.data_len == 12) {
      if (auto* rb = b->is<PxRigidBody>()) { PxVec3 v = an == "linearVelocity" ? rb->getLinearVelocity() : rb->getAngularVelocity(); cmp(cl + "." + an, &v.x, F.data(e), 3); }
    } else if ((an == "jointPosition" || an == "jointVelocity") && e.data_len == 24) {
      auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
      float v[6];
      for (int i = 0; i < 6; ++i) v[i] = an == "jointPosition" ? j->getJointPosition(PxArticulationAxis::Enum(i)) : j->getJointVelocity(PxArticulationAxis::Enum(i));
      cmp("joint." + an, v, F.data(e), 6);
    } else if (an == "wakeCounter" && e.data_len == 4) {
      float w;
      if (auto* rd = b->is<PxRigidDynamic>()) w = rd->getWakeCounter();
      else if (auto* a = b->is<PxArticulationReducedCoordinate>()) w = a->getWakeCounter();
      else return;
      cmp(cl + ".wakeCounter", &w, F.data(e), 1);
    } else if (an == "worldBounds" && e.data_len == 24) {
      PxBounds3 bb;
      if (auto* a = b->is<PxActor>()) bb = a->getWorldBounds();
      else if (auto* ar = b->is<PxArticulationReducedCoordinate>()) bb = ar->getWorldBounds();
      else return;
      cmp("worldBounds", &bb.minimum.x, F.data(e), 6);
    }
  }

  void end_frame() {
    if (csv) fprintf(csv, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.9g\n", sims, frame_cmp, frame_bitdiff, frame_max);
    frame_max = 0; frame_bitdiff = 0; frame_cmp = 0;
  }

  // ------------------------------------------------------------------ 본 반복
  void run() {
    uint64_t open = 0;         // 만들 때 값을 모으는 중인 객체
    bool out_block = false;    // simulate 뒤 결과 구간
    uint64_t out_ctx = 0;
    const uint32_t a_elapsed = attr("PxScene", "elapsedTime");
    for (size_t i = 0; i < F.events.size(); ++i) {
      const ovd::Event& e = F.events[i];
      // 만든 순간 값 묶음(cluster): create 바로 뒤, 같은 객체의 set 이 이어지는 동안. 같은 속성이 두 번 나오면
      // 두 번째부터는 API 호출이다 (생성 기록은 속성마다 한 번씩만 쓴다: OmniPvdPxSampler::stream*).
      if (e.cmd == ovd::kSet && e.obj == open && !objs[e.obj].done && !objs[e.obj].pend.count(e.attr)) {
        objs[e.obj].pend[e.attr].assign(F.data(e), F.data(e) + e.data_len);
        continue;
      }
      if (open) { close_cluster(open); open = 0; }
      switch (e.cmd) {
        case ovd::kCreate: {
          Obj o;
          o.cls = e.cls;
          o.name = F.str(e);
          objs[e.obj] = o;
          open = e.obj;
          break;
        }
        case ovd::kDestroy:
          destroy(e.obj);
          break;
        case ovd::kAddToList:
        case ovd::kRemoveFromList:
          if (out_block) break;
          list_op(e, e.cmd == ovd::kAddToList);
          break;
        case ovd::kSet: {
          if (e.attr == a_elapsed) {
            Obj& so = objs[e.obj];
            if (!so.scene) { unsupported["simulate without scene"]++; break; }
            apply_side_until(sims);
            float dt; memcpy(&dt, F.data(e), 4);
            so.scene->simulate(dt);
            so.scene->fetchResults(true);
            sims++;
            out_block = true;
            out_ctx = e.obj;  // 프레임 명령의 문맥 = 장면 핸들 (OmniPvdPxSampler.cpp:101)
            break;
          }
          if (out_block) { compare_output(e); break; }
          auto it = objs.find(e.obj);
          if (it == objs.end()) { unsupported["set unknown obj"]++; break; }
          if (!it->second.done) { it->second.pend[e.attr].assign(F.data(e), F.data(e) + e.data_len); break; }
          // API 한 번이 속성 여러 개를 남기는 경우(묶음)는 한 번의 호출로 되돌린다
          const std::vector<std::string>* grp = group_of(it->second, e.attr);
          if (grp) {
            std::unordered_map<std::string, const ovd::Event*> vals;
            size_t k = i;
            while (k < F.events.size()) {
              const ovd::Event& x = F.events[k];
              if (x.cmd != ovd::kSet || x.obj != e.obj) break;
              const std::string& xn = F.attrs[x.attr].name;
              bool in = false;
              for (auto& g : *grp) in |= (g == xn);
              if (!in || vals.count(xn)) break;
              vals[xn] = &x;
              ++k;
            }
            apply_group(it->second, *grp, vals);
            i = k - 1;
            break;
          }
          apply_set(it->second, e.attr, F.data(e), e.data_len);
          break;
        }
        case ovd::kStopFrame:
          if (out_block && e.ctx == out_ctx) {
            out_block = false;
            end_frame();
            if (max_frames >= 0 && int64_t(sims) >= max_frames) return;
          }
          break;
        default:
          break;
      }
    }
  }

  void report() {
    printf("\n== 재생 결과 ==\nsimulate %" PRIu64 " 번, 볼록 메시 원본 %" PRIu64 " / 근사 %" PRIu64 ", 정규화 원상 %" PRIu64
           " 번 중 못 찾음 %" PRIu64 "\n", sims, convex_exact, convex_approx, g_prenorm_calls, g_prenorm_fail);
    printf("첫 비트 불일치: %s\n", first_div_frame < 0 ? "없음 (전 구간 비트 동일)" : (std::to_string(first_div_frame) + " 번째 simulate, " + first_div_what).c_str());
    printf("%-44s %10s %10s %12s %10s\n", "항목", "비교 수", "비트 다름", "최대|차|", "첫 다름");
    for (auto& kv : stats)
      printf("%-44s %10" PRIu64 " %10" PRIu64 " %12.3e %10" PRId64 "\n", kv.first.c_str(), kv.second.compared, kv.second.bitdiff, kv.second.maxdiff, kv.second.first_bit);
    if (!unsupported.empty()) {
      printf("\n지원 안 한 것 (재생에서 빠짐 -> 불일치 원인 후보)\n");
      for (auto& kv : unsupported) printf("  %8" PRIu64 "  %s\n", kv.second, kv.first.c_str());
    }
    if (verbose) {
      printf("\n적용한 호출\n");
      for (auto& kv : applied) printf("  %8" PRIu64 "  %s\n", kv.second, kv.first.c_str());
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ovd_replay <file.ovd> [--convex f] [--filters f] [--sidelog f] [--threads N] [--csv f] [--max-frames N] [--verbose]\n");
    return 2;
  }
  ovd::File F;
  std::string err;
  if (!ovd::load(argv[1], F, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Replayer R(F);
  int threads = 4;
  std::string convex, filters, sidelog, csv;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--convex" && i + 1 < argc) convex = argv[++i];
    else if (a == "--filters" && i + 1 < argc) filters = argv[++i];
    else if (a == "--sidelog" && i + 1 < argc) sidelog = argv[++i];
    else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
    else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--max-frames" && i + 1 < argc) R.max_frames = atoll(argv[++i]);
    else if (a == "--verbose") R.verbose = true;
    else if (a == "--record" && i + 1 < argc) R.record_path = argv[++i];
  }
  if (!convex.empty() && !engine::read_convex_bin(convex, R.convex)) fprintf(stderr, "convex 보조 파일을 못 읽음: %s\n", convex.c_str());
  if (!filters.empty() && !engine::read_filters(filters, R.spec)) fprintf(stderr, "filters 파일을 못 읽음: %s\n", filters.c_str());
  if (!sidelog.empty() && !engine::read_sidelog(sidelog, R.sviews, R.scalls)) fprintf(stderr, "sidelog 를 못 읽음: %s\n", sidelog.c_str());
  if (!csv.empty()) { R.csv = fopen(csv.c_str(), "w"); if (R.csv) fprintf(R.csv, "simulate,compared,bitdiff,maxabs\n"); }
  if (!R.init(threads)) return 1;
  R.run();
  R.report();
  if (R.csv) fclose(R.csv);
  if (R.rec_pvd) {  // 기록 마무리 (파일 닫기)
    for (auto& kv : R.objs) if (kv.second.scene) { kv.second.scene->release(); kv.second.scene = nullptr; }
    R.phys->release();
    R.rec_pvd->release();
  }
  printf("PhysX 오류/경고 %d 건\n", gErr.n);
  return R.first_div_frame < 0 ? 0 : 3;
}
