// 장면 파일 여러 개를 판 N 개 배치 하나에 넣어 본다 (PhysX 없음, 문서 15.3).
// 판 i 에 파일 (i mod 파일 수) 를 넣고: 틀 공유 수, 용량, 판당 메모리, 판마다 상태가 파일과 바이트 같은지, 볼록 포인터가 제 틀 안을 가리키는지.
//   scene_batch_check <판 수> <파일...>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include "core/scene/batch.h"

namespace sc = eng::scene;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: scene_batch_check <판 수> <장면 파일...>\n");
    return 2;
  }
  const uint32_t n = uint32_t(atoi(argv[1]));
  std::vector<sc::SceneFile> files(argc - 2);
  sc::BatchCaps cap;
  for (size_t k = 0; k < files.size(); ++k) {
    std::string err;
    if (!sc::readScene(argv[2 + k], files[k], &err)) {
      fprintf(stderr, "%s: %s\n", argv[2 + k], err.c_str());
      return 1;
    }
    const sc::SceneFile& f = files[k];
    printf("파일 %zu %s: 과제 %s 판 %s simulate %" PRIu64 " — 행위자 %u 모양 %u 볼록 %u 몸체 %u 관절체 %u 조인트 %u\n", k, argv[2 + k], f.h.task, f.h.instance,
           f.h.sim, f.h.nActors, f.h.nShapes, f.h.nHulls, f.h.nBodies, f.h.nArts, f.h.nJoints);
    cap.bodies = std::max<uint32_t>(cap.bodies, f.h.nBodies);
    cap.arts = std::max<uint32_t>(cap.arts, f.h.nArts);
    cap.joints = std::max<uint32_t>(cap.joints, f.h.nJoints);
    cap.shapes = std::max<uint32_t>(cap.shapes, f.h.nShapes);
  }
  sc::Batch B;
  B.init(n, cap);
  uint64_t bad = 0;
  for (uint32_t e = 0; e < n; ++e) {
    std::string err;
    if (!B.load(e, files[e % files.size()], &err)) {
      fprintf(stderr, "판 %u: %s\n", e, err.c_str());
      return 1;
    }
  }
  for (uint32_t e = 0; e < n; ++e) {
    const sc::SceneFile& f = files[e % files.size()];
    if (memcmp(B.envBodies(e), f.bodies.data(), f.bodies.size() * sizeof(eng::Body))) ++bad;
    if (memcmp(static_cast<const void*>(B.envArts(e)), f.arts.data(), f.arts.size() * sizeof(eng::art::Articulation))) ++bad;
    if (memcmp(B.envJoints(e), f.joints.data(), f.joints.size() * sizeof(sc::SceneJoint))) ++bad;
    if (memcmp(B.envFilters(e), f.shapeFilters.data(), f.shapeFilters.size() * sizeof(sc::ShapeFilter))) ++bad;
    const sc::SceneShared& S = *B.shared(e);
    const uint8_t* lo = S.hulls.data();
    const uint8_t* hi = lo + S.hulls.size();
    for (const sc::SceneShape& sh : S.shapes)
      if (sh.hull != sc::kNone) {
        const auto* h = reinterpret_cast<const uint8_t*>(sh.geom.convex.hullData);
        const auto* poly = reinterpret_cast<const uint8_t*>(sh.geom.convex.hullData->mPolygons);
        if (h < lo || h >= hi || poly < lo || poly >= hi) ++bad;
      }
  }
  // 틀이 갈린 이유 (파일 0 과 나머지): 처음 다른 행위자·모양·볼록
  for (size_t k = 1; k < files.size(); ++k) {
    const sc::SceneFile &a = files[0], &b = files[k];
    int shown = 0;
    for (size_t i = 0; i < std::min(a.actors.size(), b.actors.size()) && shown < 3; ++i)
      if (memcmp(&a.actors[i], &b.actors[i], sizeof(sc::SceneActor))) {
        const sc::SceneActor &x = a.actors[i], &y = b.actors[i];
        printf("  틀 차이 파일 0~%zu 행위자 %zu %s / %s: 종류 %u/%u 모양 %u+%u / %u+%u 이름위치 %u/%u 정적자세 %s\n", k, i, a.name(x.name), b.name(y.name), x.kind, y.kind,
               x.shapeStart, x.shapeCount, y.shapeStart, y.shapeCount, x.name, y.name, memcmp(&x.staticPose, &y.staticPose, sizeof(x.staticPose)) ? "다름" : "같음");
        ++shown;
      }
    size_t ds = 0, dh = 0, dGeom = 0, dPose = 0, dOff = 0, dFilt = 0, dRest = 0;
    for (size_t i = 0; i < std::min(a.shapes.size(), b.shapes.size()); ++i) {
      const sc::SceneShape &x = a.shapes[i], &y = b.shapes[i];
      if (memcmp(&x, &y, sizeof(sc::SceneShape))) {
        ++ds;
        const uint8_t *px = reinterpret_cast<const uint8_t*>(&x), *py = reinterpret_cast<const uint8_t*>(&y);
        size_t o = 0;
        while (px[o] == py[o]) ++o;
        if (ds <= 3) printf("  모양 %zu 첫 다른 바이트 %zu/%zu (geom 은 %zu~%zu, 종류 %d)\n", i, o, sizeof(sc::SceneShape), offsetof(sc::SceneShape, geom),
                            offsetof(sc::SceneShape, geom) + sizeof(x.geom), x.geom.type);
      }
      dGeom += memcmp(&x.geom, &y.geom, sizeof(x.geom)) != 0;
      dPose += memcmp(&x.localPose, &y.localPose, sizeof(x.localPose)) != 0;
      dOff += memcmp(&x.contactOffset, &y.contactOffset, 16) != 0;
      dFilt += memcmp(&a.shapeFilters[i], &b.shapeFilters[i], 16) != 0;
      dRest += x.actor != y.actor || x.hull != y.hull || x.shapeFlags != y.shapeFlags || x.material != y.material;
    }
    printf("  모양 칸 차이: geom %zu 자세 %zu 거리 %zu 거르기 %zu 나머지 %zu\n", dGeom, dPose, dOff, dFilt, dRest);
    dh = a.hulls.size() != b.hulls.size() || memcmp(a.hulls.data(), b.hulls.data(), a.hulls.size());
    printf("  틀 차이 파일 0~%zu: 모양 다른 칸 %zu, 볼록 덩어리 %s, 이름 %s\n", k, ds, dh ? "다름" : "같음",
           a.names.size() == b.names.size() && !memcmp(a.names.data(), b.names.data(), a.names.size()) ? "같음" : "다름");
  }
  size_t sharedBytes = 0;
  for (auto& s : B.shareds) sharedBytes += s->hulls.size() + s->shapes.size() * sizeof(sc::SceneShape) + s->actors.size() * sizeof(sc::SceneActor);
  printf("판 %u 개: 용량 몸체 %u 관절체 %u 조인트 %u, 틀 %zu 벌 (%.1f MB), 상태 %.1f MB (판당 %.2f MB), 틀린 곳 %" PRIu64 "\n", n, cap.bodies, cap.arts,
         cap.joints, B.shareds.size(), double(sharedBytes) / 1e6, double(B.stateBytes()) / 1e6, double(B.stateBytes()) / n / 1e6, bad);
  return bad ? 3 : 0;
}
