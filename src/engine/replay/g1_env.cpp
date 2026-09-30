// G1 env 적재 대조 (G1_ENV_FROM, 문서 15.3 닫힌 고리 4단 E1, 리드): 장면 파일 하나로 env 한 스텝 함수 상태(env_load.h)를 PhysX 없이 세우고,
// 파일을 뜬 경계 simulate 앞에서 살아 있는 PhysX(와 PhysX 와 맞춰 둔 그림자들)와 칸마다 비교한다.
//   섬 관리(두 섬 시뮬·번호 관리) = PhysX, 쌍 관리층 = g1_pairs 의 것(PhysX 와 다름 0), Sc 칸 = 지금 PhysX 에서 뜬 Sc 장면,
//   깸 카운터 표 = PhysX 몸체·관절체 값, 행위자 활성 = PhysX ActorSim::isActive.
// 켜기: G1_ENV_FROM=<장면 파일> G1_ISLANDS=1 G1_PAIRS=1 (파일은 G1_DUMP_AT/G1_DUMP_OUT 로, 같은 simulate 경계)
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/scene/env_load.h"
#include "core/scene/env_solve.h"
#include "core/scene/env_body_api.h"
namespace physx { class PxActor; }
#include "g1_hooks.h"

namespace ss = eng::contact::sc;
namespace sc2 = eng::scene;

void g1_islands_compare_store(const sc2::IslandStore& O, uint64_t* n, uint64_t* bad, std::string* first);
ss::ScPairs* g1_pairs_M();
void g1_pairs_actor_active(std::vector<int8_t>& out);
void g1_pairs_wake(sc2::HostWake& out);
// particles 편집 창 자리 (약한 기호 — particles 쪽 번역 단위가 채운다): 이 simulate 앞 창을 엔진 편집 창으로 대신하면 1(창 처리함 — 기록 창 입력·건드림 다시 맞춤 안 함),
// 편집 창 안에서 물리 스텝을 이미 돌렸으면 2 를 더한다(이번 simulate 의 envStep 을 건너뜀). 0 = 평소대로 기록 창.
int g1_env_edit_hook(uint64_t sim, eng::scene::EnvStep& E, eng::scene::EnvSolveImpl& S, eng::scene::EnvBodyApi& api) __attribute__((weak));
size_t g1_islands_rec_count();
void g1_env_fric_scene(physx::PxScene* s);
const std::map<uint64_t, std::vector<eng::sv::FrictionPatch>>& g1_env_px_fric();
bool g1_env_same_fric(const eng::sv::FrictionPatch& a, const eng::sv::FrictionPatch& b);
bool g1_islands_rec(size_t i, sc2::IslOp& o);
bool g1_env_px_joint(physx::PxScene* scene, const void* dyc, uint32_t kind[2], uint32_t body[2], uint32_t link[2], sc2::SceneJoint& j);  // g1_dump.cpp

namespace {
struct EnvCheck {
  bool inited = false, on = false, done = false;
  std::string from;
  sc2::SceneFile f;
  std::unique_ptr<sc2::SceneShared> sh;
  std::unique_ptr<sc2::EnvOwned> env;
  // ---- 닫힌 고리 (G1_ENV_RUN): 경계 뒤로 env 가 스스로 스텝을 돈다. 창 입력만 재생기에서 (옮긴 API = 관절체 드라이브·깨움, 그 밖은 건드림 = PhysX 로 다시 맞춤)
  bool run = false, running = false, first = true;
  sc2::EnvSolveImpl solver;
  std::unique_ptr<sc2::EnvBodyApi> api;
  uint64_t editWindows = 0, editSkips = 0;
  std::vector<const void*> pxArts;  // 파일 관절체 번호 차례
  size_t cursor = 0, boundary = 0;
  std::vector<G1BodyState> pre;     // 창 뒤 PhysX 몸체 상태 (건드린 것 다시 맞춤용)
  std::vector<std::vector<G1ArtOp>> artOps;
  std::vector<uint8_t> artTouched;
  std::vector<std::unique_ptr<eng::art::Articulation>> artSnap;
  sc2::HostWake preWake;
  std::vector<int8_t> preActive;
  uint64_t steps = 0, resyncBodies = 0, resyncArts = 0, artOpsN = 0, winExt = 0, winOurs = 0, winMismatch = 0, bpNot = 0;
  // 판 도중 조인트 (보조 잡기): 우리 조인트 번호 -> PhysX 제약 주소 (상수 블록을 창 입력으로 다시 뜸 — 옮기지 않은 API)
  std::map<uint32_t, const void*> rtJoints;
  uint64_t jointsAdded = 0, jointsRemoved = 0, jointsBad = 0;
  // 대조
  uint64_t cmpBody = 0, badBody = 0, cmpArt = 0, badArt = 0, cmpWake = 0, badWake = 0;
  long long firstBad = -1;
  std::string firstWhat;
  int show = 5;
  physx::PxScene* scene = nullptr;
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

void g1_env_capture_window(physx::PxScene* scene);
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
  if (EC.running && sim > EC.f.h.sim) {
    EC.first = false;
    g1_env_capture_window(scene);
    return;
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
  // 쌍 관리층 입력 칸을 우리 공식(EnvModules: Sc 칸 -> 쌍 관리층 행위자·모양 칸)으로 다시 만들어 적재된 칸(= PhysX)과 비교 — 판 도중 새 행위자에 쓰는 공식의 검증
  Tally tRow{"쌍 관리층 입력 칸 (우리 공식)"};
  {
    ss::ScPairs& P = o.C.S->pairs;
    for (size_t h = 0; h < o.sc.actors.size(); ++h) {
      const sc2::ScActorRec& r = o.sc.actors[h];
      const int32_t pa = h < o.E.pairsOfSc.size() ? o.E.pairsOfSc[h] : -1;
      if (!r.alive || pa < 0 || r.kind == 2) continue;  // 링크는 판 도중 안 넣음
      const ss::Actor& A = P.actors[size_t(pa)];
      const bool dyn = r.kind != 0;
      const uint32_t fa = dyn ? (uint32_t(ss::FilterObj::eTYPE_RIGID_DYNAMIC) | ss::FilterObj::eEX_RIGID_DYNAMIC | (r.kinematic ? uint32_t(ss::FilterObj::eKINEMATIC) : 0u))
                              : (uint32_t(ss::FilterObj::eTYPE_RIGID_STATIC) | ss::FilterObj::eEX_RIGID_STATIC);
      tRow.chk(A.type == (dyn ? ss::eRIGID_DYNAMIC : ss::eRIGID_STATIC) && A.filterAttr == fa && A.actorID == r.actorID &&
                   A.nodeIndex == (dyn ? r.node : ss::INVALID_NODE),
               "행위자 칸 #" + std::to_string(h) + " 속성 " + std::to_string(A.filterAttr) + "/" + std::to_string(fa));
      for (uint32_t e : r.elements) {
        if (e >= P.shapes.size()) continue;
        const ss::Shape saved = P.shapes[e];
        sc2::envPairsShapeRow(o.E, int32_t(h), e);
        const ss::Shape& n = P.shapes[e];
        tRow.chk(n.valid == saved.valid && n.actor == saved.actor && n.geomType == saved.geomType && n.trigger == saved.trigger && n.fd.word0 == saved.fd.word0 &&
                     n.fd.word1 == saved.fd.word1 && n.fd.word2 == saved.fd.word2 && n.fd.word3 == saved.fd.word3 && n.restOffset == saved.restOffset &&
                     n.torsionalPatchRadius == saved.torsionalPatchRadius && n.minTorsionalPatchRadius == saved.minTorsionalPatchRadius &&
                     n.transformCacheId == saved.transformCacheId,
                 "모양 칸 #" + std::to_string(e));
        P.shapes[e] = saved;
      }
    }
  }
  printf("G1 env 적재 대조 (simulate %llu, 파일 %s): PhysX 없이 세운 env 상태 = 살아 있는 PhysX\n", (unsigned long long)sim, EC.from.c_str());
  tRow.print();
  tIsl.print();
  tPairs.print();
  tSc.print();
  tWake.print();
  tAct.print();
  // 닫힌 고리 준비
  EC.run = getenv("G1_ENV_RUN") != nullptr;
  if (EC.run) {
    EC.solver.load(EC.f);
    EC.solver.seedPairs(o.C.S->pairs);
    o.E.solver = &EC.solver;
    EC.api.reset(new sc2::EnvBodyApi(o.E, EC.solver));
    EC.solver.keepIn = getenv("G1_ENV_FRIC") != nullptr;
    g1_env_fric_scene(scene);
    EC.pxArts = g1_env_px_arts(scene);
    EC.running = true;
    EC.first = true;
    EC.cursor = EC.boundary = g1_islands_rec_count();
    if (EC.pxArts.size() != EC.solver.arts.size()) printf("G1 env 닫힌 고리: 관절체 수가 다름 (PhysX %zu / 파일 %zu)\n", EC.pxArts.size(), EC.solver.arts.size());
    // 경계 simulate 의 창 입력도 아래 before 몸통에서 뜬다
  }
  if (!EC.running) return;
  EC.first = true;
  g1_env_capture_window(scene);
}

// 창 뒤(simulate 앞) PhysX 에서 뜨는 것: 옮긴 관절체 호출, 건드린 몸체·관절체의 지금 상태, 깸 카운터·활성 (API 는 아직 바깥)
void g1_env_capture_window(physx::PxScene* scene) {
  EC.scene = scene;
  EC.boundary = g1_islands_rec_count();
  g1_pairs_body_states(EC.pre);
  g1_pairs_wake(EC.preWake);
  g1_pairs_actor_active(EC.preActive);
  const size_t na = EC.pxArts.size();
  EC.artOps.assign(na, {});
  EC.artTouched.assign(na, 0);
  EC.artSnap.clear();
  EC.artSnap.resize(na);
  for (size_t k = 0; k < na; ++k) {
    g1_loop_take_art_ops(EC.pxArts[k], EC.artOps[k]);
    bool sleepOp = false;
    for (const G1ArtOp& op : EC.artOps[k]) sleepOp = sleepOp || op.type == 3;
    if (g1_loop_touched(EC.pxArts[k]) || sleepOp) {  // 경계 스텝은 파일 = PhysX 라 다시 맞추지 않는다 (Sc 칸을 우리 공식으로 새로 계산하면 잠든 링크 칸이 PhysX 가 예전에 적어 둔 값과 갈림)
      EC.artTouched[k] = 1;
      EC.artSnap[k].reset(new eng::art::Articulation);
      if (!g1_env_art_snapshot(scene, EC.pxArts[k], *EC.artSnap[k])) EC.artSnap[k].reset();
    }
  }
}

namespace {
void envBad(const char* what, uint64_t sim, uint64_t& counter) {
  ++counter;
  if (EC.firstBad < 0) {
    EC.firstBad = (long long)sim;
    EC.firstWhat = what;
  }
}
eng::Tf tfOf(const float* v) { return eng::Tf{eng::Q{v[0], v[1], v[2], v[3]}, eng::V3{v[4], v[5], v[6]}}; }
}  // namespace

// fetchResults 뒤: 창 입력을 우리 env 에 넣고 한 스텝, PhysX 스텝 끝 공개 상태와 비교
void g1_env_after(physx::PxScene*, uint64_t sim) {
  if (!EC.running) return;
  sc2::EnvOwned& o = *EC.env;
  sc2::EnvStep& E = o.E;
  ss::ScPairs& P = o.C.S->pairs;
  eng::ig::IslandManager& M = o.isl.M;
  sc2::LiveIslands& L = E.live;
  const sc2::PairsStep* Sp = g1_pairs_step();
  if (!Sp) return;
  ++EC.steps;
  // 0. particles 편집 창 (있으면)
  const int editFlags = g1_env_edit_hook ? g1_env_edit_hook(sim, E, EC.solver, *EC.api) : 0;
  if (editFlags) ++EC.editWindows;
  // 1. 창: 쌍 관리층 앞 연산 (섬 호출은 모아 둠), 섬 바깥 호출은 기록 차례대로, 우리 쌍 호출은 기록의 쌍 호출 자리에
  if (!(editFlags & 1)) {  // 기록 창 (편집 창이 대신하지 않은 simulate)
  L.clearStep();
  L.defer = true;
  sc2::pairsPreOps(P, *Sp);
  L.defer = false;
  size_t gi = 0;
  if (!EC.first) {
    for (size_t i = EC.cursor; i < EC.boundary; ++i) {
      sc2::IslOp r;
      g1_islands_rec(i, r);
      if (r.op == sc2::ISL_ADD_CONSTRAINT && r.list.size() == 2) {  // 판 도중 새 조인트: PhysX 제약에서 조인트 칸을 떠 우리 조인트 표에 붙임
        const void* dyc = reinterpret_cast<const void*>(uintptr_t(r.list[0]) | (uintptr_t(r.list[1]) << 32));
        uint32_t kind[2], body[2], link[2];
        sc2::SceneJoint j{};
        uint32_t obj = 0;
        if (g1_env_px_joint(EC.scene, dyc, kind, body, link, j)) {
          uint32_t* act[2] = {&j.actor0, &j.actor1};
          bool ok = true;
          for (int s = 0; s < 2; ++s) {
            if (kind[s] == 0xffffffffu) continue;
            bool found = false;
            for (size_t a = 0; a < EC.solver.sceneActors.size() && !found; ++a) {
              const sc2::SceneActor& A = EC.solver.sceneActors[a];
              if (A.kind == kind[s] && A.body == body[s] && (kind[s] != sc2::kLink || A.link == link[s])) {
                *act[s] = uint32_t(a);
                found = true;
              }
            }
            ok = ok && found;
          }
          if (ok) {
            const uint32_t k = EC.solver.addJoint(j);
            EC.rtJoints[k] = dyc;
            obj = 0x80000000u | k;
            ++EC.jointsAdded;
            if (getenv("G1_ENV_TRACE"))
              fprintf(stderr, "[g1 env] sim %llu 새 조인트 %u (제약 번호 %u, 행위자 %u/%u)\n", (unsigned long long)sim, k, j.index, j.actor0, j.actor1);
          } else {
            ++EC.jointsBad;
          }
        } else {
          ++EC.jointsBad;
        }
        if (eng::ig::addConstraint(M, obj, sc2::islNode(r.p), sc2::islNode(r.q)) != r.result) ++EC.winMismatch;
        ++EC.winExt;
      } else if (sc2::islExternalOp(r)) {
        if (r.op == sc2::ISL_REMOVE_CONN && 2 * r.a + 1 < M.cpu.cap) {  // 조인트 간선 끊기 = 조인트 해제
          const uint32_t o = M.constraintOrCm[r.a];
          if (o != eng::ig::INVALID_EDGE && (o & 0x80000000u)) {
            EC.solver.removeJoint(o & 0x7fffffffu);
            EC.rtJoints.erase(o & 0x7fffffffu);
            ++EC.jointsRemoved;
          }
        }
        sc2::islApplyExternal(M, r);
        ++EC.winExt;
      } else if (sc2::islPairsOp(r)) {
        if (gi < L.out.size()) {
          if (!sc2::islSame(L.out[gi], r)) ++EC.winMismatch;
          L.apply(gi++);
          ++EC.winOurs;
        } else {
          ++EC.winMismatch;
        }
      }
    }
    for (; gi < L.out.size(); ++gi) {
      L.apply(gi);
      ++EC.winMismatch;
    }
  }  // 경계 simulate: 파일 섬 상태가 이미 창을 담았다 -> 우리 쌍 호출은 버림
  L.clearStep();
  EC.cursor = g1_islands_rec_count();
  // 2. 깸 카운터·활성 = 창 뒤 PhysX (API 는 바깥)
  E.wake = EC.preWake;
  if (E.active.size() < P.actors.size()) E.active.resize(P.actors.size(), 0);
  for (size_t k = 0; k < EC.preActive.size() && k < E.active.size(); ++k)
    if (EC.preActive[k] >= 0) E.active[k] = uint8_t(EC.preActive[k]);
  // 3. 몸체: 건드린 것은 창 뒤 PhysX 값으로 (자세·속도·깸) + Sc 칸
  std::unordered_map<uint64_t, int32_t> scByNode;
  for (size_t h = 0; h < o.sc.actors.size(); ++h)
    if (o.sc.actors[h].alive && o.sc.actors[h].kind != 0) scByNode[o.sc.actors[h].node] = int32_t(h);
  for (const G1BodyState& b : EC.pre) {
    if (b.link || !b.touched) continue;
    const int32_t bi = EC.solver.bodyOf(uint32_t(b.node & 0xffffffffu));
    if (bi < 0) continue;
    eng::Body& x = EC.solver.bodies[size_t(bi)];
    x.body2World = tfOf(b.b2w);
    x.body2Actor = tfOf(b.b2a);
    x.linVel = eng::V3{b.lin[0], b.lin[1], b.lin[2]};
    x.angVel = eng::V3{b.ang[0], b.ang[1], b.ang[2]};
    x.wakeCounter = b.wc;
    ++EC.resyncBodies;
    if (getenv("G1_ENV_TRACE") && !EC.first) fprintf(stderr, "[g1 env] sim %llu 다시 맞춤 몸체 노드 %llx\n", (unsigned long long)sim, (unsigned long long)b.node);
    auto it = scByNode.find(b.node);
    if (it != scByNode.end()) {
      eng::px::PxTransform t;
      memcpy(&t, b.b2w, 28);
      eng::px::PxTransform a;
      memcpy(&a, b.b2a, 28);
      o.sc.updateActorCached(it->second, t, a, false);
    }
  }
  // 4. 관절체: 옮긴 호출, 건드린 것은 창 뒤 PhysX 값 (+ 링크 Sc 칸)
  for (size_t k = 0; k < EC.pxArts.size() && k < EC.solver.arts.size(); ++k) {
    if (EC.artTouched[k] && EC.artSnap[k]) {
      EC.solver.arts[k] = *EC.artSnap[k];
      ++EC.resyncArts;
      if (getenv("G1_ENV_TRACE") && !EC.first) fprintf(stderr, "[g1 env] sim %llu 다시 맞춤 관절체 %zu\n", (unsigned long long)sim, k);
      for (const G1BodyState& b : EC.pre) {
        if (!b.link || EC.solver.artOf(uint32_t(b.node & 0xffffffffu)) != int32_t(k)) continue;
        auto it = scByNode.find(b.node);
        if (it == scByNode.end()) continue;
        eng::px::PxTransform t, a;
        memcpy(&t, b.b2w, 28);
        memcpy(&a, b.b2a, 28);
        o.sc.updateActorCached(it->second, t, a, false);
      }
    } else if (!EC.artOps[k].empty()) {
      g1_env_art_apply_ops(EC.solver.arts[k], EC.artOps[k]);
      EC.artOpsN += EC.artOps[k].size();
    }
  }
  // 판 도중 조인트의 상수 블록 = 이번 simulate 에 PhysX 가 쓴 값 (조인트 자세 등 창 API 는 아직 옮기지 않음)
  for (const auto& kv : EC.rtJoints) {
    uint32_t kind[2], body[2], link[2];
    sc2::SceneJoint j{};
    if (!g1_env_px_joint(EC.scene, kv.second, kind, body, link, j)) continue;
    sc2::SceneJoint& J = EC.solver.joints[kv.first];
    J.flags = j.flags;
    J.linBreakForce = j.linBreakForce;
    J.angBreakForce = j.angBreakForce;
    J.minResponseThreshold = j.minResponseThreshold;
    J.data = j.data;
  }
  }  // 기록 창 끝
  EC.cursor = g1_islands_rec_count();
  // 5. 한 스텝
  if (editFlags & 2) ++EC.editSkips;
  if (getenv("G1_ENV_TRACE"))
    fprintf(stderr, "[g1 env] sim %llu 앞: 활성 섬 %u, 몸체 %zu, 관리자 목록 %u, 상호작용 %u\n", (unsigned long long)sim, M.accurate.activeIslands.size,
            EC.solver.bodies.size(), P.npMain.size(), P.inters.size());
  if (!(editFlags & 2)) sc2::envStep(E);
  if (getenv("G1_ENV_TRACE"))
    fprintf(stderr, "[g1 env] sim %llu 뒤: 판 섬 %zu 몸체 %zu 관리자 %zu 1D %zu 관절체 %zu\n", (unsigned long long)sim, EC.solver.islands.size(), EC.solver.ib.size(),
            EC.solver.icm.size(), EC.solver.c1d.size(), EC.solver.ia.size());
  if (EC.solver.keepIn) {  // 진단: 풀이에 넣은 지난 마찰 패치 = PhysX 풀이 직전 값
    const auto& px = g1_env_px_fric();
    for (const auto& kv : EC.solver.lastIn) {
      auto it = px.find(kv.first);
      bool same = it != px.end() && it->second.size() == kv.second.size();
      for (size_t q = 0; same && q < kv.second.size(); ++q) same = g1_env_same_fric(kv.second[q], it->second[q]);
      if (!same && EC.show > 0) {
        --EC.show;
        fprintf(stderr, "[g1 env] sim %llu 마찰 패치 다름 요소쌍 %llx: 우리 %zu 개 / PhysX %zu 개\n", (unsigned long long)sim, (unsigned long long)kv.first,
                kv.second.size(), it == px.end() ? size_t(9999) : it->second.size());
      }
    }
  }
  // 6. PhysX 스텝 끝과 비교: 몸체·링크 자세·속도, 관절체 전체, 깸 카운터 표
  std::vector<G1BodyState> now;
  g1_pairs_body_states(now);
  for (const G1BodyState& b : now) {
    const uint32_t node = uint32_t(b.node & 0xffffffffu);
    if (!b.link) {
      const int32_t bi = EC.solver.bodyOf(node);
      if (bi < 0) continue;
      const eng::Body& x = EC.solver.bodies[size_t(bi)];
      ++EC.cmpBody;
      const bool same = !memcmp(&x.body2World, b.b2w, 28) && !memcmp(&x.linVel, b.lin, 12) && !memcmp(&x.angVel, b.ang, 12);
      if (!same) {
        envBad("몸체 자세·속도", sim, EC.badBody);
        if (EC.show > 0) {
          --EC.show;
          fprintf(stderr, "[g1 env] sim %llu 몸체 노드 %u: 우리 q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g v %.9g %.9g %.9g w %.9g %.9g %.9g / PhysX q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g v %.9g %.9g %.9g w %.9g %.9g %.9g\n",
                  (unsigned long long)sim, node, x.body2World.q.x, x.body2World.q.y, x.body2World.q.z, x.body2World.q.w, x.body2World.p.x, x.body2World.p.y,
                  x.body2World.p.z, x.linVel.x, x.linVel.y, x.linVel.z, x.angVel.x, x.angVel.y, x.angVel.z, b.b2w[0], b.b2w[1], b.b2w[2], b.b2w[3], b.b2w[4], b.b2w[5],
                  b.b2w[6], b.lin[0], b.lin[1], b.lin[2], b.ang[0], b.ang[1], b.ang[2]);
        }
      }
    }
  }
  for (size_t k = 0; k < EC.pxArts.size() && k < EC.solver.arts.size(); ++k) {
    size_t fj = 0, nf = 0;
    float pv = 0, ev = 0;
    ++EC.cmpArt;
    if (g1_env_art_diff(EC.pxArts[k], EC.solver.arts[k], &fj, &nf, &pv, &ev)) {
      envBad("관절체", sim, EC.badArt);
      if (EC.show > 0) {
        --EC.show;
        fprintf(stderr, "[g1 env] sim %llu 관절체 %zu 칸 %zu/%zu: PhysX %.9g 우리 %.9g\n", (unsigned long long)sim, k, fj, nf, pv, ev);
      }
    }
  }
  if (getenv("G1_ENV_SCCMP")) {  // 진단: 스텝 끝 우리 Sc 칸 = PhysX (변환 캐시·경계 상자)
    static physx::PxScene* sceneKeep = nullptr;
    (void)sceneKeep;
    sc2::ScScene cap;
    g1_sc_capture(EC.scene, cap);
    uint32_t bad = 0, firstE = ~0u;
    for (size_t e = 0; e < cap.shapes.size() && e < o.sc.shapes.size(); ++e) {
      if (!cap.shapes[e].alive || !o.sc.shapes[e].alive) continue;
      if (memcmp(&cap.cache[e], &o.sc.cache[e], 28) || memcmp(&cap.bounds[e], &o.sc.bounds[e], 24)) {
        if (firstE == ~0u) firstE = uint32_t(e);
        ++bad;
      }
    }
    static uint32_t lastBad = 0;
    const bool changedBad = bad != lastBad;
    lastBad = bad;
    if (changedBad && getenv("G1_ENV_SCCMP_CHG")) fprintf(stderr, "[g1 env] sim %llu Sc 칸 다름 수 바뀜 -> %u (첫 요소 %u)\n", (unsigned long long)sim, bad, firstE);
    if (bad && EC.show > 0) {
      --EC.show;
      const sc2::ScActorRec& ar = o.sc.actors[size_t(o.sc.shapes[firstE].actor)];
      const float* c0 = reinterpret_cast<const float*>(&o.sc.cache[firstE]);
      const float* c1 = reinterpret_cast<const float*>(&cap.cache[firstE]);
      const float* b0 = reinterpret_cast<const float*>(&o.sc.bounds[firstE]);
      const float* b1 = reinterpret_cast<const float*>(&cap.bounds[firstE]);
      {  // 그 링크의 몸체 자세: 우리 관절체 칸 vs PhysX
        std::vector<G1BodyState> bs;
        g1_pairs_body_states(bs);
        for (const G1BodyState& q : bs)
          if (q.node == ar.node) {
            const int32_t k = EC.solver.artOf(uint32_t(q.node & 0xffffffffu));
            const uint32_t ll = uint32_t(q.node >> 33);
            if (k >= 0 && ll < EC.solver.arts[size_t(k)].nLinks) {
              const eng::Tf& t = EC.solver.arts[size_t(k)].bodies[ll].body2World;
              fprintf(stderr, "  링크 몸체 우리 %.9g %.9g %.9g %.9g %.9g %.9g %.9g / PhysX %.9g %.9g %.9g %.9g %.9g %.9g %.9g (관절체 %d %s LL %u 깨어 %d)\n", t.q.x, t.q.y,
                      t.q.z, t.q.w, t.p.x, t.p.y, t.p.z, q.b2w[0], q.b2w[1], q.b2w[2], q.b2w[3], q.b2w[4], q.b2w[5], q.b2w[6], k,
                      size_t(k) < EC.f.artName.size() ? EC.f.name(EC.f.artName[size_t(k)]) : "?", ll,
                      int(EC.solver.arts[size_t(k)].awake));
              {
                std::vector<int8_t> pxa;
                g1_pairs_actor_active(pxa);
                const int32_t pa = size_t(o.sc.shapes[firstE].actor) < E.pairsOfSc.size() ? E.pairsOfSc[size_t(o.sc.shapes[firstE].actor)] : -1;
                const uint32_t nid = uint32_t(q.node & 0xffffffffu);
                const eng::ig::IslandSim& A2 = o.isl.M.accurate;
                fprintf(stderr, "  활성: 우리 행위자 %d / PhysX %d, 우리 섬 노드 플래그 %x, 깸 표 %.9g / PhysX %.9g, body2Actor q %.9g %.9g %.9g %.9g\n",
                        pa >= 0 ? int(E.active[size_t(pa)]) : -1, pa >= 0 && size_t(pa) < pxa.size() ? int(pxa[size_t(pa)]) : -9,
                        nid < A2.nodes.size ? A2.nodes.d[nid].flags : 0xff, E.wake.body(q.node) ? E.wake.body(q.node)->wc : -1.f, q.wc, q.b2a[0], q.b2a[1], q.b2a[2],
                        q.b2a[3]);
              }
            }
          }
      }
      {
        const sc2::ScShapeRec& x = o.sc.shapes[firstE];
        const sc2::ScShapeRec& y = cap.shapes[firstE];
        const sc2::ScActorRec& ya = cap.actors[size_t(y.actor)];
        fprintf(stderr, "  모양 국소 우리 q %.9g %.9g %.9g %.9g idt %u 몸체-행위자 idt %u / PhysX q %.9g %.9g %.9g %.9g idt %u 몸체-행위자 idt %u b2a %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                x.in.localPose.q.x, x.in.localPose.q.y, x.in.localPose.q.z, x.in.localPose.q.w, x.in.idtShape, ar.idtBody2Actor, y.in.localPose.q.x, y.in.localPose.q.y,
                y.in.localPose.q.z, y.in.localPose.q.w, y.in.idtShape, ya.idtBody2Actor, ya.body2Actor.q.x, ya.body2Actor.q.y, ya.body2Actor.q.z, ya.body2Actor.q.w,
                ya.body2Actor.p.x, ya.body2Actor.p.y, ya.body2Actor.p.z);
      }
      fprintf(stderr, "[g1 env] sim %llu Sc 칸 다름 %u (첫 요소 %u, 행위자 손잡이 %d 종류 %u 노드 %llx 얼림 %u/%u)\n  캐시 우리 %.9g %.9g %.9g %.9g %.9g %.9g %.9g / PhysX %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n  상자 우리 %.9g %.9g %.9g %.9g %.9g %.9g / PhysX %.9g %.9g %.9g %.9g %.9g %.9g\n",
              (unsigned long long)sim, bad, firstE, o.sc.shapes[firstE].actor, ar.kind, (unsigned long long)ar.node, o.sc.cacheFlags[firstE], cap.cacheFlags[firstE],
              c0[0], c0[1], c0[2], c0[3], c0[4], c0[5], c0[6], c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], b0[0], b0[1], b0[2], b0[3], b0[4], b0[5], b1[0], b1[1],
              b1[2], b1[3], b1[4], b1[5]);
    }
  }
  {
    sc2::HostWake px;
    g1_pairs_wake(px);
    for (const sc2::HostBodyWake& w : E.wake.bodies) {
      const sc2::HostBodyWake* q = px.body(w.node);
      if (!q) continue;
      ++EC.cmpWake;
      if (q->wc != w.wc) {
        envBad("깸 카운터 표", sim, EC.badWake);
        if (EC.show > 0) {
          --EC.show;
          fprintf(stderr, "[g1 env] sim %llu 깸 카운터 노드 %llx (링크 %d): 우리 %.9g / PhysX %.9g\n", (unsigned long long)sim, (unsigned long long)w.node, int(w.link),
                  w.wc, q->wc);
        }
      }
    }
  }
  EC.first = false;
}

void g1_env_report() {
  if (!EC.run) return;
  printf("G1 env 닫힌 고리 (장면 파일에서 세운 env 가 스스로 %" PRIu64 " 스텝, 창 입력만 재생기) — 몸체 %" PRIu64 " 다름 %" PRIu64 ", 관절체 %" PRIu64 " 다름 %" PRIu64
         ", 깸 카운터 %" PRIu64 " 다름 %" PRIu64 "%s\n",
         EC.steps, EC.cmpBody, EC.badBody, EC.cmpArt, EC.badArt, EC.cmpWake, EC.badWake,
         EC.firstBad >= 0 ? ("  첫 다름 simulate " + std::to_string(EC.firstBad) + " " + EC.firstWhat).c_str() : "");
  printf("  창: 바깥 섬 호출 %" PRIu64 ", 우리 쌍 호출 %" PRIu64 " (어긋남 %" PRIu64 "), 옮긴 관절체 호출 %" PRIu64 ", 다시 맞춤 몸체 %" PRIu64 " 관절체 %" PRIu64
         "; 풀이: 섬 안 운동학 %" PRIu64 ", 모르는 간선 %" PRIu64 ", 모르는 노드 %" PRIu64 ", 판 오류 %" PRIu64 "\n",
         EC.winExt, EC.winOurs, EC.winMismatch, EC.artOpsN, EC.resyncBodies, EC.resyncArts, EC.solver.kinInIsland, EC.solver.unknownEdge, EC.solver.unknownNode,
         EC.solver.engineErr);
  if (EC.jointsAdded || EC.jointsRemoved || EC.jointsBad)
    printf("  판 도중 조인트: 새로 %" PRIu64 ", 해제 %" PRIu64 ", 못 뜬 것 %" PRIu64 "\n", EC.jointsAdded, EC.jointsRemoved, EC.jointsBad);
  if (g1_env_edit_hook)
    printf("  편집 창(particles): %" PRIu64 " 번, 그 안에서 스텝을 돌려 건너뛴 simulate %" PRIu64 ", 몸체 API 요청 깨움 %" PRIu64 " 잠 준비 %" PRIu64 " 바로 재움 %" PRIu64
           " 모르는 손잡이 %" PRIu64 "\n",
           EC.editWindows, EC.editSkips, EC.api ? EC.api->reqActivate : 0, EC.api ? EC.api->reqDeactivate : 0, EC.api ? EC.api->reqSleep : 0,
           EC.api ? EC.api->unknown : 0);
}
