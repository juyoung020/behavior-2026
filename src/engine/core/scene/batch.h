// 판 N 개 배치 (문서 15.3 [스펙 결정], 리드): 장면 파일(scene_file.h)을 판 칸에 채운다. 처음부터 층 2(GPU, 판 = 블록 하나) 배치를 전제로 짠다.
// 배치 규칙:
//  - 틀(재질·모양·볼록 덩어리·행위자)은 장면별 한 벌(SceneShared)을 판들이 가리킨다. 같은 틀(바이트 같음)이면 다시 만들지 않는다.
//  - 상태는 모듈의 원래 구조체 그대로 "판 순서로 이어 붙인 배열"(env-major, 칸 수 = 용량 고정)에 둔다:
//      bodies[e * capBodies + i], arts[e * capArts + a], joints[e * capJoints + j]
//    모듈 판(SolverBoard·관절체 판 등)은 판 e 의 시작 주소만 받으면 되므로 모듈 코드를 바꾸지 않는다.
//    필드 단위 SoA(자세 배열·속도 배열 따로)는 모듈 커널을 모두 바꿔야 해서 하지 않는다 — 판 = 블록 하나 설계에서는
//    한 블록이 한 판의 구조체를 차례로 읽으므로 판 순서 배열이 이미 연속 읽기다.
//  - GPU 로 옮길 때는 같은 배열을 그대로 복사하고, 모양의 볼록 포인터만 장치 주소로 다시 건다(makeShared 의 base 를 장치 주소로 — G2).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/scene/scene_file.h"

namespace eng {
namespace scene {

// 틀 한 벌 (호스트). hulls 의 포인터는 host 주소로 걸려 있다.
struct SceneShared {
  std::vector<contact::MaterialData> materials;
  std::vector<SceneActor> actors;
  std::vector<SceneShape> shapes;  // geom.convex.hullData = hulls 안 호스트 주소
  std::vector<uint8_t> hulls;
  std::vector<uint64_t> hullOffsets;
  std::vector<char> names;
  uint64_t key = 0;  // 틀 바이트 해시
};

inline uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// 파일의 틀 -> SceneShared. 볼록 덩어리는 파일 안에서 base = 덩어리 시작 위치(0 기준)로 포장돼 있으므로 호스트 주소만큼 더한다.
inline std::unique_ptr<SceneShared> makeShared(const SceneFile& f) {
  std::unique_ptr<SceneShared> s(new SceneShared);
  s->materials = f.materials;
  s->actors = f.actors;
  s->shapes = f.shapes;
  s->hulls = f.hulls;
  s->hullOffsets = f.hullOffsets;
  s->names = f.names;
  uint64_t k = fnv1a(f.materials.data(), f.materials.size() * sizeof(contact::MaterialData));
  k = fnv1a(f.actors.data(), f.actors.size() * sizeof(SceneActor), k);
  k = fnv1a(f.shapes.data(), f.shapes.size() * sizeof(SceneShape), k);
  k = fnv1a(f.hulls.data(), f.hulls.size(), k);
  s->key = k;
  const uintptr_t base = reinterpret_cast<uintptr_t>(s->hulls.data());
  for (uint64_t off : s->hullOffsets) {
    auto* ch = reinterpret_cast<px::Gu::ConvexHullData*>(s->hulls.data() + off);
    ch->mPolygons = reinterpret_cast<px::Gu::HullPolygonData*>(reinterpret_cast<uintptr_t>(ch->mPolygons) + base);
    if (ch->mBigConvexRawData) {
      auto* bb = reinterpret_cast<px::Gu::BigConvexRawData*>(s->hulls.data() + reinterpret_cast<uintptr_t>(ch->mBigConvexRawData));
      ch->mBigConvexRawData = bb;
      bb->mSamples = reinterpret_cast<px::PxU8*>(reinterpret_cast<uintptr_t>(bb->mSamples) + base);
      bb->mValencies = reinterpret_cast<px::Gu::Valency*>(reinterpret_cast<uintptr_t>(bb->mValencies) + base);
      bb->mAdjacentVerts = reinterpret_cast<px::PxU8*>(reinterpret_cast<uintptr_t>(bb->mAdjacentVerts) + base);
    }
  }
  for (SceneShape& sh : s->shapes)
    if (sh.hull != kNone) sh.geom.convex.hullData = reinterpret_cast<const px::Gu::ConvexHullData*>(s->hulls.data() + s->hullOffsets[sh.hull]);
  return s;
}

struct BatchCaps {
  uint32_t bodies = 0, arts = 0, joints = 0, shapes = 0;
};

struct Batch {
  uint32_t n = 0;
  BatchCaps cap;
  std::vector<std::unique_ptr<SceneShared>> shareds;
  std::vector<uint32_t> sharedOf;  // 판 -> shareds 번호 (kNone = 빈 판)
  std::vector<uint32_t> nBodies, nArts, nJoints;
  std::vector<Body> bodies;             // n * cap.bodies
  std::vector<art::Articulation> arts;  // n * cap.arts
  std::vector<SceneJoint> joints;       // n * cap.joints
  std::vector<ShapeFilter> filters;     // n * cap.shapes (모양 거르기 자료, 판마다 다름)
  std::vector<uint64_t> sim;            // 판별 뜬 경계

  void init(uint32_t nEnvs, const BatchCaps& c) {
    n = nEnvs;
    cap = c;
    sharedOf.assign(n, kNone);
    nBodies.assign(n, 0);
    nArts.assign(n, 0);
    nJoints.assign(n, 0);
    sim.assign(n, 0);
    bodies.assign(size_t(n) * cap.bodies, Body{});
    arts.resize(size_t(n) * cap.arts);
    joints.assign(size_t(n) * cap.joints, SceneJoint{});
    filters.assign(size_t(n) * cap.shapes, ShapeFilter{});
  }
  Body* envBodies(uint32_t e) { return bodies.data() + size_t(e) * cap.bodies; }
  art::Articulation* envArts(uint32_t e) { return arts.data() + size_t(e) * cap.arts; }
  SceneJoint* envJoints(uint32_t e) { return joints.data() + size_t(e) * cap.joints; }
  ShapeFilter* envFilters(uint32_t e) { return filters.data() + size_t(e) * cap.shapes; }
  const SceneShared* shared(uint32_t e) const { return sharedOf[e] == kNone ? nullptr : shareds[sharedOf[e]].get(); }
  size_t stateBytes() const {
    return bodies.size() * sizeof(Body) + arts.size() * sizeof(art::Articulation) + joints.size() * sizeof(SceneJoint) + filters.size() * sizeof(ShapeFilter);
  }

  // 파일 하나를 판 e 에 넣는다. 용량을 넘으면 false.
  bool load(uint32_t e, const SceneFile& f, std::string* err = nullptr) {
    auto fail = [&](const char* w) { if (err) *err = w; return false; };
    if (e >= n) return fail("판 번호 넘침");
    if (f.bodies.size() > cap.bodies || f.arts.size() > cap.arts || f.joints.size() > cap.joints || f.shapeFilters.size() > cap.shapes) return fail("용량 넘침");
    std::unique_ptr<SceneShared> s = makeShared(f);
    uint32_t si = kNone;
    for (uint32_t k = 0; k < shareds.size(); ++k)
      if (shareds[k]->key == s->key) si = k;
    if (si == kNone) {
      si = uint32_t(shareds.size());
      shareds.push_back(std::move(s));
    }
    sharedOf[e] = si;
    nBodies[e] = uint32_t(f.bodies.size());
    nArts[e] = uint32_t(f.arts.size());
    nJoints[e] = uint32_t(f.joints.size());
    sim[e] = f.h.sim;
    std::copy(f.bodies.begin(), f.bodies.end(), envBodies(e));
    std::copy(f.arts.begin(), f.arts.end(), envArts(e));
    std::copy(f.joints.begin(), f.joints.end(), envJoints(e));
    std::copy(f.shapeFilters.begin(), f.shapeFilters.end(), envFilters(e));
    return true;
  }
};

}  // namespace scene
}  // namespace eng
