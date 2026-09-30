// G1 env 적재 대조 (G1_ENV_FROM, 문서 15.3 닫힌 고리 4단 E1, 리드): 장면 파일 하나로 env 한 스텝 함수 상태(env_load.h)를 PhysX 없이 세우고,
// 파일을 뜬 경계 simulate 앞에서 살아 있는 PhysX(와 PhysX 와 맞춰 둔 그림자들)와 칸마다 비교한다.
//   섬 관리(두 섬 시뮬·번호 관리) = PhysX, 쌍 관리층 = g1_pairs 의 것(PhysX 와 다름 0), Sc 칸 = 지금 PhysX 에서 뜬 Sc 장면,
//   깸 카운터 표 = PhysX 몸체·관절체 값, 행위자 활성 = PhysX ActorSim::isActive.
// 켜기: G1_ENV_FROM=<장면 파일> G1_ISLANDS=1 G1_PAIRS=1 (파일은 G1_DUMP_AT/G1_DUMP_OUT 로, 같은 simulate 경계)
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/scene/env_load.h"
namespace physx { class PxActor; }
#include "g1_hooks.h"

namespace ss = eng::contact::sc;
namespace sc2 = eng::scene;

void g1_islands_compare_store(const sc2::IslandStore& O, uint64_t* n, uint64_t* bad, std::string* first);
ss::ScPairs* g1_pairs_M();
void g1_pairs_actor_active(std::vector<int8_t>& out);
void g1_pairs_wake(sc2::HostWake& out);

namespace {
struct EnvCheck {
  bool inited = false, on = false, done = false;
  std::string from;
  sc2::SceneFile f;
  std::unique_ptr<sc2::SceneShared> sh;
  std::unique_ptr<sc2::EnvOwned> env;
} EC;

struct Tally {
  const char* name;
  uint64_t n = 0, bad = 0;
  std::string first;
  void chk(bool ok, const std::string& what) {
    ++n;
    if (!ok) {
      ++bad;
      if (first.empty()) first = what;
    }
  }
  void print() const { printf("  %-28s: 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", name, n, bad, first.c_str()); }
};

void compareSc(const sc2::ScScene& A, const sc2::ScScene& B, Tally& t) {
  t.chk(A.elementIds.pool.cur == B.elementIds.pool.cur && A.elementIds.pool.freeIds == B.elementIds.pool.freeIds && A.elementIds.pending == B.elementIds.pending,
        "요소 번호 표");
  t.chk(A.actorIds.pool.cur == B.actorIds.pool.cur && A.actorIds.pool.freeIds == B.actorIds.pool.freeIds && A.actorIds.pending == B.actorIds.pending, "행위자 번호 표");
  t.chk(A.actors.size() == B.actors.size(), "행위자 수");
  for (size_t h = 0; h < A.actors.size() && h < B.actors.size(); ++h) {
    const sc2::ScActorRec& x = A.actors[h];
    const sc2::ScActorRec& y = B.actors[h];
    t.chk(x.kind == y.kind && x.actorID == y.actorID && x.node == y.node && x.alive == y.alive && x.kinematic == y.kinematic && x.elements == y.elements &&
              !memcmp(&x.pose, &y.pose, 28) && !memcmp(&x.body2Actor, &y.body2Actor, 28) && x.idtBody2Actor == y.idtBody2Actor,
          "행위자 #" + std::to_string(h));
  }
  t.chk(A.shapes.size() == B.shapes.size(), "요소 수");
  for (size_t e = 0; e < A.shapes.size() && e < B.shapes.size(); ++e) {
    const sc2::ScShapeRec& x = A.shapes[e];
    const sc2::ScShapeRec& y = B.shapes[e];
    if (!x.alive && !y.alive) continue;
    const bool geomSame = !memcmp(&x.in.geom, &y.in.geom, offsetof(eng::contact::ShapeGeom, convex)) || x.in.geom.type == y.in.geom.type;
    t.chk(x.alive == y.alive && x.actor == y.actor && x.inBp == y.inBp && x.in.shapeFlags == y.in.shapeFlags && x.in.contactOffset == y.in.contactOffset &&
              !memcmp(&x.in.localPose, &y.in.localPose, 28) && geomSame,
          "요소 기록 #" + std::to_string(e));
    t.chk(!memcmp(&A.cache[e], &B.cache[e], 28) && !memcmp(&A.bounds[e], &B.bounds[e], 24) && A.contactDist[e] == B.contactDist[e] && A.cacheFlags[e] == B.cacheFlags[e],
          "요소 칸 #" + std::to_string(e));
  }
}

}  // namespace

void g1_env_before(physx::PxScene* scene, uint64_t sim) {
  if (!EC.inited) {
    EC.inited = true;
    if (const char* p = getenv("G1_ENV_FROM")) {
      EC.from = p;
      std::string err;
      if (!sc2::readScene(p, EC.f, &err)) {
        printf("G1 env 적재 대조: 파일 읽기 실패 %s (%s)\n", p, err.c_str());
      } else {
        EC.on = true;
        EC.sh = sc2::makeShared(EC.f);
      }
    }
  }
  if (!EC.on || EC.done || sim != EC.f.h.sim) return;
  EC.done = true;
  EC.env.reset(new sc2::EnvOwned);
  std::string err;
  if (!sc2::envLoad(*EC.env, EC.f, *EC.sh, &err)) {
    printf("G1 env 적재 대조: 적재 실패 — %s\n", err.c_str());
    return;
  }
  sc2::EnvOwned& o = *EC.env;
  Tally tIsl{"섬 관리 (두 섬 시뮬·번호)"}, tPairs{"쌍 관리층 (g1_pairs 와)"}, tSc{"Sc 칸·번호 표 (PhysX 에서 뜬 것과)"}, tWake{"깸 카운터 표 (PhysX)"},
      tAct{"행위자 활성 (PhysX)"};
  {
    uint64_t n = 0, bad = 0;
    std::string first;
    g1_islands_compare_store(o.isl, &n, &bad, &first);
    tIsl.n = n;
    tIsl.bad = bad;
    tIsl.first = first;
  }
  if (const ss::ScPairs* ref = g1_pairs_M()) {
    const ss::ScPairs& A = o.C.S->pairs;
    tPairs.chk(A.inters.size() == ref->inters.size(), "상호작용 수");
    for (uint32_t i = 0; i < A.inters.size() && i < ref->inters.size(); ++i) {
      const ss::Interaction& x = A.inters[i];
      const ss::Interaction& y = ref->inters[i];
      tPairs.chk(x.alive == y.alive && (!x.alive || (x.type == y.type && x.iflags == y.iflags && x.siFlags == y.siFlags && x.elem0 == y.elem0 &&
                                                        x.elem1 == y.elem1 && x.cm == y.cm && x.edge == y.edge)),
                 "상호작용 #" + std::to_string(i));
    }
    bool np = A.npMain.size() == ref->npMain.size();
    for (uint32_t i = 0; np && i < A.npMain.size(); ++i) np = A.npMain.cms[i] == ref->npMain.cms[i];
    tPairs.chk(np, "좁은 단계 목록");
    bool fl = A.cmPool.freeList.size() == ref->cmPool.freeList.size();
    for (uint32_t i = 0; fl && i < A.cmPool.freeList.size(); ++i) fl = A.cmPool.freeList[i] == ref->cmPool.freeList[i];
    tPairs.chk(fl, "관리자 풀 빈 칸");
  }
  {
    sc2::ScScene cap;
    g1_sc_capture(scene, cap);
    compareSc(o.sc, cap, tSc);
  }
  {
    sc2::HostWake px;
    g1_pairs_wake(px);
    // PhysX 쪽은 장면의 모든 동적 몸체(모양 없는 것 포함), 우리는 섬 노드가 있는 것 — 우리 칸마다 비교
    for (const sc2::HostBodyWake& b : o.E.wake.bodies) {
      const sc2::HostBodyWake* q = px.body(b.node);
      tWake.chk(q && q->wc == b.wc && q->kinematic == b.kinematic, "몸체 노드 " + std::to_string(b.node));
    }
    for (const sc2::HostArtWake& a : o.E.wake.arts) {
      const sc2::HostArtWake* q = px.art(a.node);
      tWake.chk(q && q->wc == a.wc && q->links == a.links, "관절체 노드 " + std::to_string(a.node));
    }
    tWake.chk(px.bodies.size() == o.E.wake.bodies.size(), "몸체 수 (PhysX " + std::to_string(px.bodies.size()) + " / 우리 " + std::to_string(o.E.wake.bodies.size()) + ")");
  }
  {
    std::vector<int8_t> px;
    g1_pairs_actor_active(px);
    for (size_t k = 0; k < px.size() && k < o.E.active.size(); ++k)
      if (px[k] >= 0) tAct.chk(o.E.active[k] == uint8_t(px[k]), "행위자 #" + std::to_string(k));
  }
  printf("G1 env 적재 대조 (simulate %llu, 파일 %s): PhysX 없이 세운 env 상태 = 살아 있는 PhysX\n", (unsigned long long)sim, EC.from.c_str());
  tIsl.print();
  tPairs.print();
  tSc.print();
  tWake.print();
  tAct.print();
}
