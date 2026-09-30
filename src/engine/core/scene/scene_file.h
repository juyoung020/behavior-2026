// 엔진 장면 파일 (문서 15.3, 리드): 판 하나를 PhysX 없이 우리 모듈 판에 채우는 데 필요한 것 전부.
// "틀"(판끼리 같을 수 있는 것: 재질·모양·볼록 덩어리·행위자 구성)과 "상태"(판마다 다른 것: 몸체·관절체·조인트 값)를 나눠 둔다.
// 만드는 곳: replay/g1_dump.cpp (공식 기록을 재생하다 에피소드 시작 경계에서 PhysX 에서 뜸, 시험·비계 전용).
// 읽는 곳: core/scene/batch.h (판 N 개 배치에 넣기). 이 헤더는 PhysX 없이 쓴다.
// 형식: 머리 + 절(section) 목록. 절마다 (종류, 원소 크기, 개수, 바이트 위치). 원소 크기를 적어 두어 구조체가 바뀌면 읽기가 거절한다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/articulation/articulation.h"
#include "core/common/body.h"
#include "core/contact/narrowphase.h"
#include "core/contact/patches.h"
#include "core/joints/joint_types.h"
#include "core/solver/solver_io.h"  // FrictionPatch (지난 스텝 마찰 패치)

namespace eng {
namespace scene {

constexpr uint32_t kNone = 0xffffffffu;

// 행위자 (PhysX 장면 순서: 정적·동적 강체는 PxScene::getActors 순서, 관절체 링크는 관절체 순서·생성 순서)
enum ActorKind : uint32_t { kStatic = 0, kDynamic = 1, kLink = 2 };
struct SceneActor {
  uint32_t kind;
  uint32_t body;        // 동적: bodies 번호 / 링크: 관절체 번호
  uint32_t link;        // 링크: 생성 순서 번호 (Articulation::creation 기준)
  uint32_t shapeStart, shapeCount;
  uint32_t name;        // names 안 바이트 위치
  Tf staticPose;        // 정적: 전역 자세 (동적·링크는 상태 쪽)
  uint32_t rigidFlags;  // PxRigidBodyFlags (동적·링크), 정적 0
  uint32_t actorFlags;  // PxActorFlags
  uint8_t dominance, pad[3];
};

// 모양 (행위자별 PxRigidActor::getShapes 순서)
struct SceneShape {
  uint32_t actor;
  uint32_t pad0;            // 채움을 드러내 둠 (복사 때 암묵 채움 바이트가 쓰레기가 되어 틀 해시가 갈리지 않게)
  contact::ShapeGeom geom;  // 볼록이면 geom.convex.hullData 는 hulls 덩어리 안 바이트 위치(읽을 때 주소로 바꿈)
  uint32_t hull;            // 볼록 덩어리 번호 (hullOffsets), 아니면 kNone
  Tf localPose;
  float contactOffset, restOffset, torsionalPatchRadius, minTorsionalPatchRadius;
  uint32_t shapeFlags;      // PxShapeFlags
  uint16_t material, pad;   // 첫 재질의 PhysX 재질 번호 (materials 칸)
};

// 모양 거르기 자료 (시뮬레이션 filter data word0..3). omni 가 판(인스턴스)마다 다른 번호를 매겨서 틀이 아니라 상태 쪽에 둔다
// (radio 판 0·1: 모양 1,789 중 1,253 칸이 다름 — 나머지 틀은 바이트 같음)
struct ShapeFilter {
  uint32_t w[4];
};

static_assert(sizeof(SceneShape) == 184, "SceneShape 에 암묵 채움이 없어야 한다");

// D6 조인트 (Dy::Constraint 에서 풀이가 쓰는 것 + 두 행위자)
struct SceneJoint {
  uint32_t actor0, actor1;  // SceneActor 번호, kNone = 세계
  uint32_t index;           // Dy::Constraint::index (되쓰기 칸)
  uint16_t flags, pad;      // PxConstraintFlags
  float linBreakForce, angBreakForce, minResponseThreshold;
  uint32_t name;
  jnt::D6Data data;
};

// 접촉 관리자 (v1: 내부 상태 넘겨받기). 순서 = Sc::Scene 의 겹침 상호작용 배열 순서(getInteractions(eOVERLAP)).
// 좁은 단계·풀이가 스텝 사이에 들고 가는 것: 지속 다양체(PhysX Gu::Cache 가 가리키는 PersistentContactManifold 바이트),
// Gu::Cache::mPairData, 지난 스텝 출력 상태(statusFlag: 더러움 판정), 지난 스텝 마찰 패치(PxcNpWorkUnit::mFrictionDataPtr).
struct SceneCM {
  uint32_t shape0, shape1;  // SceneShape 번호 (작업 단위 모양 0·1 순서 그대로)
  uint32_t npIndex;         // PxcNpWorkUnit::mNpIndex (새 표시 0x80000000 포함, 순서 확인용)
  uint16_t npFlags;         // PxcNpWorkUnitFlag
  uint8_t statusFlag;       // 지난 PxsContactManagerOutput::statusFlag (새 관리자면 0)
  uint8_t pairData;         // Gu::Cache::mPairData
  float restDistance, torsionalPatchRadius, minTorsionalPatchRadius, offsetSlop;
  uint32_t manifold;        // manifolds 번호, kNone = 다양체 없음
  uint32_t frictionStart, frictionCount;  // friction 안 범위
  uint8_t manifoldFlags, pad[3];          // Gu::Cache::mManifoldFlags
};

struct SceneHeader {
  char magic[8];  // "ENGSCN1"
  uint32_t version;
  uint32_t sizeActor, sizeShape, sizeJoint, sizeBody, sizeArt, sizeMaterial, sizeGeom;
  uint32_t artMaxLinks, artMaxDofs;
  uint32_t nActors, nShapes, nJoints, nBodies, nArts, nMaterials, nHulls;
  uint32_t nCMs, nManifolds, nFriction, sizeCM, sizeManifold, sizeFriction;
  uint64_t hullBytes, nameBytes;
  float gravity[3], dt, lengthScale, speedScale;
  uint32_t sceneFlags, solverType, posIters, velIters;
  uint64_t sim;  // 뜬 경계 (이 simulate 바로 앞)
  char task[64], instance[32];
};

struct SceneFile {
  SceneHeader h{};
  // 틀
  std::vector<contact::MaterialData> materials;
  std::vector<SceneActor> actors;
  std::vector<SceneShape> shapes;
  std::vector<uint64_t> hullOffsets;  // hulls 안 덩어리 시작 (16 정렬), 덩어리 = contact::packHull(base = 그 위치)
  std::vector<uint8_t> hulls;
  std::vector<char> names;
  // 상태
  std::vector<Body> bodies;
  std::vector<art::Articulation> arts;
  std::vector<SceneJoint> joints;
  std::vector<uint32_t> artName;  // 관절체별 이름 위치
  std::vector<ShapeFilter> shapeFilters;  // 모양별 (shapes 와 같은 순서)
  std::vector<SceneCM> cms;
  std::vector<contact::ManifoldSlot> manifolds;  // 파일에는 바이트 그대로 (읽은 뒤 relocateManifold)
  std::vector<sv::FrictionPatch> friction;

  uint32_t addName(const char* s) {
    const uint32_t at = uint32_t(names.size());
    const size_t n = s ? strlen(s) : 0;
    names.insert(names.end(), s, s + n);
    names.push_back(0);
    return at;
  }
  const char* name(uint32_t at) const { return at < names.size() ? names.data() + at : ""; }
};

namespace detail {
template <class T>
inline bool wr(FILE* f, const std::vector<T>& v) { return v.empty() || fwrite(v.data(), sizeof(T), v.size(), f) == v.size(); }
template <class T>
inline bool rd(FILE* f, std::vector<T>& v, size_t n) { v.resize(n); return n == 0 || fread(v.data(), sizeof(T), n, f) == n; }
}  // namespace detail

inline void fillSizes(SceneHeader& h) {
  memcpy(h.magic, "ENGSCN1", 8);
  h.version = 3;
  h.sizeCM = sizeof(SceneCM);
  h.sizeManifold = sizeof(contact::ManifoldSlot);
  h.sizeFriction = sizeof(sv::FrictionPatch);
  h.sizeActor = sizeof(SceneActor);
  h.sizeShape = sizeof(SceneShape);
  h.sizeJoint = sizeof(SceneJoint);
  h.sizeBody = sizeof(Body);
  h.sizeArt = sizeof(art::Articulation);
  h.sizeMaterial = sizeof(contact::MaterialData);
  h.sizeGeom = sizeof(contact::ShapeGeom);
  h.artMaxLinks = art::kMaxLinks;
  h.artMaxDofs = art::kMaxDofs;
}

inline bool writeScene(const char* path, SceneFile& s) {
  fillSizes(s.h);
  s.h.nActors = uint32_t(s.actors.size());
  s.h.nShapes = uint32_t(s.shapes.size());
  s.h.nJoints = uint32_t(s.joints.size());
  s.h.nBodies = uint32_t(s.bodies.size());
  s.h.nArts = uint32_t(s.arts.size());
  s.h.nMaterials = uint32_t(s.materials.size());
  s.h.nHulls = uint32_t(s.hullOffsets.size());
  s.h.nCMs = uint32_t(s.cms.size());
  s.h.nManifolds = uint32_t(s.manifolds.size());
  s.h.nFriction = uint32_t(s.friction.size());
  s.h.hullBytes = s.hulls.size();
  s.h.nameBytes = s.names.size();
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  using detail::wr;
  bool ok = fwrite(&s.h, sizeof(s.h), 1, f) == 1 && wr(f, s.materials) && wr(f, s.actors) && wr(f, s.shapes) && wr(f, s.hullOffsets) && wr(f, s.hulls) &&
            wr(f, s.names) && wr(f, s.bodies) && wr(f, s.arts) && wr(f, s.joints) && wr(f, s.artName) && wr(f, s.shapeFilters) && wr(f, s.cms) &&
            (s.manifolds.empty() || fwrite(static_cast<const void*>(s.manifolds.data()), sizeof(contact::ManifoldSlot), s.manifolds.size(), f) == s.manifolds.size()) &&
            wr(f, s.friction);
  ok = fclose(f) == 0 && ok;
  return ok;
}

// 읽기: 구조체 크기가 이 빌드와 다르면 거절 (err 에 이유)
inline bool readScene(const char* path, SceneFile& s, std::string* err = nullptr) {
  auto fail = [&](const char* why) { if (err) *err = why; return false; };
  FILE* f = fopen(path, "rb");
  if (!f) return fail("파일을 못 엶");
  SceneHeader want{};
  fillSizes(want);
  if (fread(&s.h, sizeof(s.h), 1, f) != 1 || memcmp(s.h.magic, want.magic, 8)) { fclose(f); return fail("머리 틀림"); }
  if (s.h.version != want.version || s.h.sizeActor != want.sizeActor || s.h.sizeShape != want.sizeShape || s.h.sizeJoint != want.sizeJoint ||
      s.h.sizeBody != want.sizeBody || s.h.sizeArt != want.sizeArt || s.h.sizeMaterial != want.sizeMaterial || s.h.sizeGeom != want.sizeGeom ||
      s.h.artMaxLinks != want.artMaxLinks || s.h.artMaxDofs != want.artMaxDofs || s.h.sizeCM != want.sizeCM || s.h.sizeManifold != want.sizeManifold ||
      s.h.sizeFriction != want.sizeFriction) {
    fclose(f);
    return fail("구조체 크기·판이 이 빌드와 다름 (ENG_ART_MAX_LINKS 등)");
  }
  using detail::rd;
  bool ok = rd(f, s.materials, s.h.nMaterials) && rd(f, s.actors, s.h.nActors) && rd(f, s.shapes, s.h.nShapes) && rd(f, s.hullOffsets, s.h.nHulls) &&
            rd(f, s.hulls, s.h.hullBytes) && rd(f, s.names, s.h.nameBytes) && rd(f, s.bodies, s.h.nBodies) && rd(f, s.arts, s.h.nArts) &&
            rd(f, s.joints, s.h.nJoints) && rd(f, s.artName, s.h.nArts) && rd(f, s.shapeFilters, s.h.nShapes) &&
            rd(f, s.cms, s.h.nCMs);
  if (ok) {
    s.manifolds.resize(s.h.nManifolds);
    ok = s.h.nManifolds == 0 || fread(static_cast<void*>(s.manifolds.data()), sizeof(contact::ManifoldSlot), s.h.nManifolds, f) == s.h.nManifolds;
    for (contact::ManifoldSlot& m : s.manifolds) contact::relocateManifold(m);  // 자기 버퍼 포인터를 새 자리로
  }
  ok = ok && rd(f, s.friction, s.h.nFriction);
  fclose(f);
  return ok ? true : fail("짧음");
}

}  // namespace scene
}  // namespace eng
