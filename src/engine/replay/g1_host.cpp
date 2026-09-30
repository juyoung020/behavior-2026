// G1 순서기 그림자 (G1_HOST, 문서 15.3 닫힌 고리 4단 ①, 리드): core/scene/step_host.h 를 공식 재생 위에서 PhysX 와 비교한다.
// 섬 관리자 상태 넘겨받기(G1_ISLANDS_AT) simulate 에서 우리 섬 저장소(PhysX 살아 있는 상태) + 우리 쌍 관리층(g1_pairs 의 것 복사)을 떼어 내
// 그 뒤로는 둘을 **직접** 잇는다: 쌍 관리층이 부르는 섬 호출은 우리 섬 관리에 바로 들어가고(돌려받는 번호도 우리 것),
// 상호작용 활성/비활성은 우리 섬 상태로 우리가 몬다 (activateEdgesInternal·postThird). PhysX 기록에서 오는 것은
//   바깥 호출(노드·조인트 간선·API/풀이 쪽 깨우기·재우기), 넓은 단계 새/사라진 겹침, 좁은 단계 결과, 창 앞 행위자 활성 표시 뿐.
// 스텝마다 비교: (1) 우리가 낸 쌍 관리층 섬 호출 = PhysX 호출 (차례·번호), (2) 스텝 끝 두 섬 시뮬·번호 관리 = PhysX,
//   (3) 상호작용 활성/비활성 호출·결과 = PhysX, (4) 행위자 활성 표시 = PhysX, (5) 쌍 관리층 상태 = g1_pairs 의 것(PhysX 와 0 다름 확인된 쪽).
// 켜기: G1_HOST=1 G1_ISLANDS=1 G1_ISLANDS_AT=<sim> G1_PAIRS=1 (G1_HOST_SHOW=1 이면 첫 다름을 자세히)
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/scene/island_state.h"
#include "core/scene/step_host.h"
namespace physx { class PxActor; }
#include "g1_hooks.h"

namespace ss = eng::contact::sc;
namespace sc2 = eng::scene;

size_t g1_islands_rec_count();
bool g1_islands_rec(size_t i, sc2::IslOp& o);
void g1_islands_compare_store(const sc2::IslandStore& O, uint64_t* n, uint64_t* bad, std::string* first);
ss::ScPairs* g1_pairs_M();
void g1_pairs_actor_active(std::vector<int8_t>& out);
void g1_pairs_wake(sc2::HostWake& out);

namespace {

struct Stat {
  uint64_t n = 0, bad = 0;
  std::string first;
  void chk(bool ok, const std::string& what) {
    ++n;
    if (!ok) {
      ++bad;
      if (first.empty()) first = what;
    }
  }
};

struct Host {
  bool inited = false, on = false, running = false, skipPre = false, show = false;
  long long at = -1;
  uint64_t curSim = 0, steps = 0;
  sc2::IslandStore store;
  std::unique_ptr<ss::ScPairs> P;
  sc2::LiveIslands live;
  std::vector<uint8_t> active;
  sc2::HostWake wake;  // 우리 깸 카운터 표 (창 앞마다 PhysX 값으로 = API 는 아직 바깥)
  size_t cursor = 0, boundary = 0;
  Stat ops, ext, isl, acts, act, pairs, win, wk;
  int neg = 0;  // G1_HOST_NEG: 1 = 활성화 몰이 빼기, 2 = 재우기 몰이 빼기, 3 = 풀이 뒤 깸/잠 요청 빼기 (비교가 살아 있나)
  uint64_t nWakeOps = 0, winActMatched = 0, extInSim = 0;
  uint64_t wakeReq = 0, woken = 0, slept = 0, nActs = 0, winChanged = 0;
} H;

uint32_t ptrLow(const void* o, uint32_t, void*) { return uint32_t(uintptr_t(o) & 0xffffffffu); }
uint32_t ptrLowE(const void* o, void*) { return uint32_t(uintptr_t(o) & 0xffffffffu); }

std::string opStr(const sc2::IslOp& r) {
  char b[160];
  snprintf(b, sizeof b, "%s(a=%u b=%u sim=%u n=%zu r=%u)", sc2::islOpName(r.op), r.a, r.b, r.sim, r.list.size(), r.result);
  return b;
}
std::string at(const char* what) {
  char b[64];
  snprintf(b, sizeof b, " @sim %llu", (unsigned long long)H.curSim);
  return std::string(what) + b;
}

// 창 앞: 행위자 활성 표시는 PhysX 에서 (API 깨우기·재우기·새 행위자 = 아직 바깥)
void refreshActive(bool count) {
  std::vector<int8_t> px;
  g1_pairs_actor_active(px);
  if (H.active.size() < H.P->actors.size()) H.active.resize(H.P->actors.size(), 0);
  for (size_t k = 0; k < px.size() && k < H.active.size(); ++k)
    if (px[k] >= 0 && H.active[k] != uint8_t(px[k])) {
      if (count) ++H.winChanged;
      H.active[k] = uint8_t(px[k]);
    }
}

void comparePairs(const ss::ScPairs& A, const ss::ScPairs& B) {
  bool same = A.inters.size() == B.inters.size();
  for (uint32_t i = 0; same && i < A.inters.size(); ++i) {
    const ss::Interaction& x = A.inters[i];
    const ss::Interaction& y = B.inters[i];
    same = x.alive == y.alive && (!x.alive || (x.type == y.type && x.iflags == y.iflags && x.siFlags == y.siFlags && x.elem0 == y.elem0 &&
                                               x.elem1 == y.elem1 && x.cm == y.cm && x.edge == y.edge));
    if (!same) H.pairs.chk(false, at(("상호작용 #" + std::to_string(i)).c_str()));
  }
  if (same) H.pairs.chk(true, "");
  else if (A.inters.size() != B.inters.size()) H.pairs.chk(false, at("상호작용 수"));
  bool np = A.npMain.size() == B.npMain.size();
  for (uint32_t i = 0; np && i < A.npMain.size(); ++i) np = A.npMain.cms[i] == B.npMain.cms[i];
  H.pairs.chk(np, at("좁은 단계 목록"));
  bool fl = A.cmPool.freeList.size() == B.cmPool.freeList.size();
  for (uint32_t i = 0; fl && i < A.cmPool.freeList.size(); ++i) fl = A.cmPool.freeList[i] == B.cmPool.freeList[i];
  H.pairs.chk(fl, at("관리자 풀 빈 칸"));
}

}  // namespace

void g1_host_before(physx::PxScene* scene, uint64_t sim) {
  if (!H.inited) {
    H.inited = true;
    H.on = getenv("G1_HOST") != nullptr;
    H.show = getenv("G1_HOST_SHOW") != nullptr;
    if (const char* n = getenv("G1_HOST_NEG")) H.neg = atoi(n);
    if (const char* a = getenv("G1_ISLANDS_AT")) H.at = atoll(a);
    if (H.on && (H.at < 0 || !getenv("G1_ISLANDS") || !getenv("G1_PAIRS"))) {
      printf("G1 순서기: G1_ISLANDS=1 G1_ISLANDS_AT=<sim> G1_PAIRS=1 이 함께 있어야 한다 — 끔\n");
      H.on = false;
    }
  }
  if (!H.on) return;
  H.curSim = sim;
  if (!H.running) {
    if ((long long)sim != H.at) return;
    ss::ScPairs* src = g1_pairs_M();
    if (!src) {
      printf("G1 순서기: 쌍 관리층 그림자가 아직 안 돌아 넘겨받지 못함 — 끔\n");
      H.on = false;
      return;
    }
    sc2::IslandMgrState st;
    g1_islands_capture(scene, st, ptrLow, ptrLowE, nullptr);
    const bool ok = H.store.load(st);
    H.P.reset(new ss::ScPairs(*src));
    H.live.M = &H.store.M;
    H.live.P = H.P.get();
    H.live.active = &H.active;
    H.live.wake = &H.wake;
    H.P->islands = &H.live;
    H.active.assign(H.P->actors.size(), 0);
    refreshActive(false);
    g1_pairs_wake(H.wake);
    H.running = true;
    H.skipPre = true;  // 이 simulate 의 창은 이미 두 쪽 다 들어감 (g1_pairs 가 pairsPre 를 한 뒤 복사)
    H.cursor = H.boundary = g1_islands_rec_count();
    printf("G1 순서기 넘겨받기: simulate %llu (섬 적재 %s, 행위자 %u, 상호작용 %u) — 이 뒤로 쌍 관리층 섬 호출·활성 몰이는 우리 것\n",
           (unsigned long long)sim, ok ? "됨" : "실패", H.P->actors.size(), H.P->inters.size());
    return;
  }
  H.boundary = g1_islands_rec_count();
  refreshActive(true);
  g1_pairs_wake(H.wake);  // 창의 API 깨우기·재우기가 바꾼 값 (바깥)
}

void g1_host_after(physx::PxScene*, uint64_t sim) {
  if (!H.on || !H.running) return;
  const sc2::PairsStep* Sp = g1_pairs_step();
  if (!Sp) return;
  const sc2::PairsStep& S = *Sp;
  ss::ScPairs& P = *H.P;
  eng::ig::IslandManager& M = H.store.M;
  sc2::LiveIslands& L = H.live;
  ++H.steps;
  const size_t end = g1_islands_rec_count();
  std::vector<sc2::IslOp> win, simr;
  for (size_t i = H.cursor; i < end; ++i) {
    sc2::IslOp o;
    g1_islands_rec(i, o);
    (i < H.boundary ? win : simr).push_back(std::move(o));
  }
  H.cursor = end;
  L.clearStep();
  std::vector<sc2::PairsAct> acts;
  size_t gi = 0;  // 다음에 맞출 우리 호출
  // 우리 마디 (simulate 안): PhysX 기록의 섬 마디가 허락한 것까지만 앞당겨 돈다
  enum { PH_BP, PH_NP, PH_LOST, PH_LOST3, PH_SCPOST, PH_AFTER, PH_DONE };
  sc2::HostWake post;  // 스텝 끝 PhysX 깸 카운터 (풀이·관절체 모듈 몫의 수치)
  g1_pairs_wake(post);
  struct Ctx {
    int next = PH_BP;
    bool spec = false, second2 = false, third = false, post = false;
    ss::ScPairs* P;
    const sc2::PairsStep* S;
    eng::ig::IslandManager* M;
    std::vector<sc2::PairsAct>* acts;
    sc2::LiveIslands* L;
    const sc2::HostWake* postW;
  } C{PH_BP, false, false, false, false, &P, &S, &M, &acts, &L, &post};
  auto allowed = [](Ctx& c) {
    switch (c.next) {
      case PH_BP: return true;
      case PH_NP: return c.spec;
      case PH_LOST: return c.second2;
      case PH_LOST3: return c.third;
      case PH_SCPOST: return c.post;
      case PH_AFTER: return c.post;
      default: return false;
    }
  };
  auto run = [](Ctx& c) {
    switch (c.next) {
      case PH_BP: sc2::hostPairsBP(*c.P, *c.S, c.L); break;
      case PH_NP: sc2::hostPairsNP(*c.P, *c.S); break;
      case PH_LOST: sc2::hostPairsLost(*c.P, *c.S); break;
      case PH_LOST3: sc2::hostPairsLost3(*c.P); break;
      case PH_SCPOST: {
        std::vector<uint32_t> ch;
        if (H.neg != 2) sc2::hostSetActiveFromIslands(*c.M, *c.P, H.active, false, &ch);  // 음성 대조 2: 재우기 몰이를 뺀다
        H.slept += ch.size();
        sc2::hostDeactivateEdges(*c.M, *c.P, c.acts);
        break;
      }
      case PH_AFTER: {
        const size_t n0 = c.L->out.size();
        if (H.neg != 3) sc2::hostAfterIntegration(*c.M, *c.P, *c.L, H.wake, *c.postW);  // 음성 대조 3: 풀이 뒤 깸/잠 요청 빼기
        H.nWakeOps += c.L->out.size() - n0;
        break;
      }
    }
    ++c.next;
  };
  auto force = [&](int upto) {
    while (C.next <= upto) run(C);
  };
  auto tryAdvance = [&]() -> bool {
    if (C.next >= PH_DONE || !allowed(C)) return false;
    run(C);
    return true;
  };
  // 우리 호출 하나를 PhysX 기록과 맞춘다 (창에서 모아 둔 것은 이 차례에 넣는다). 모자라면 허락된 다음 마디를 돈다
  auto matchOne = [&](const sc2::IslOp& r, bool canAdvance) {
    while (gi >= L.out.size() && canAdvance && tryAdvance()) {
    }
    if (gi >= L.out.size()) {
      H.ops.chk(false, at(("PhysX 만 부름: " + opStr(r)).c_str()));
      return;
    }
    const bool same = sc2::islSame(L.out[gi], r);
    H.ops.chk(same, at(("우리 " + opStr(L.out[gi]) + " / PhysX " + opStr(r)).c_str()));
    L.apply(gi);
    ++gi;
  };
  // ---- 창 (simulate 밖): 쌍 관리층 앞 연산은 모아 두고 PhysX 차례대로 넣는다
  if (!H.skipPre) {
    L.defer = true;
    sc2::pairsPreOps(P, S);
    L.defer = false;
  }
  H.skipPre = false;
  if (H.active.size() < P.actors.size()) H.active.resize(P.actors.size(), 0);
  for (const sc2::IslOp& r : win) {
    if (r.op == sc2::ISL_ACTIVATE && gi < L.out.size() && L.out[gi].op == sc2::ISL_ACTIVATE && L.out[gi].a == r.a) {
      ++H.winActMatched;
      matchOne(r, false);
    } else if (sc2::islExternalOp(r)) {
      H.ext.chk(sc2::islApplyExternal(M, r), at(("바깥 호출 결과 " + opStr(r)).c_str()));
    } else if (sc2::islPairsOp(r)) {
      matchOne(r, false);
    }
    else H.win.chk(false, at(("창에 섬 마디 " + opStr(r)).c_str()));
  }
  for (; gi < L.out.size(); ++gi) {  // 창에서 우리만 부른 것
    H.ops.chk(false, at(("우리만 부름(창): " + opStr(L.out[gi])).c_str()));
    L.apply(gi);
  }
  H.win.chk(L.deferBad == 0, at("창에서 돌려받을 값이 있는 호출"));
  L.deferBad = 0;
  // ---- simulate: PhysX 기록의 마디에 맞춰 우리 마디를 돈다
  for (const sc2::IslOp& r : simr) {
    if (sc2::islPairsOp(r) || r.op == sc2::ISL_ACTIVATE || r.op == sc2::ISL_DEACTIVATE) {
      matchOne(r, true);
      continue;
    }
    if (sc2::islExternalOp(r)) {
      ++H.extInSim;
      if (C.post) force(PH_AFTER);  // afterIntegration 까지 끝난 뒤의 바깥 호출
      H.ext.chk(sc2::islApplyExternal(M, r), at(("바깥 호출 결과 " + opStr(r)).c_str()));
      continue;
    }
    switch (r.op) {
      case sc2::ISL_FIRST_PASS: force(PH_BP); eng::ig::firstPassIslandGen(M); break;
      case sc2::ISL_ADD_SPEC_ACT:
        eng::ig::additionalSpeculativeActivation(M);
        if (H.neg != 1) sc2::hostActivateEdges(M, P, &acts);  // 음성 대조 1: 활성화 몰이를 뺀다
        C.spec = true;
        break;
      case sc2::ISL_SECOND1: force(PH_NP); eng::ig::secondPassIslandGenPart1(M); break;
      case sc2::ISL_SECOND2:
      case sc2::ISL_SECOND: {
        if (r.op == sc2::ISL_SECOND) {
          force(PH_NP);
          eng::ig::secondPassIslandGenPart1(M);
        }
        eng::ig::secondPassIslandGenPart2(M);
        sc2::hostSnapshotSolveWake(H.wake);  // 풀이가 읽는 깸 카운터
        std::vector<uint32_t> ch;
        sc2::hostSetActiveFromIslands(M, P, H.active, true, &ch);
        H.woken += ch.size();
        C.second2 = true;
        break;
      }
      case sc2::ISL_THIRD: force(PH_LOST); sc2::hostThirdBegin(M); C.third = true; break;
      case sc2::ISL_SIM_REMOVE_DESTROYED: eng::ig::removeDestroyedEdges(r.sim == 1 ? M.accurate : M.speculative); break;
      case sc2::ISL_SIM_PROCESS_LOST: {
        std::vector<uint32_t> ours(M.destroyedNodes.d, M.destroyedNodes.d + M.destroyedNodes.size);
        H.win.chk(ours == r.list, at("processLostEdges 지운 노드 목록"));
        eng::ig::processLostEdges(r.sim == 1 ? M.accurate : M.speculative, M.destroyedNodes.d, M.destroyedNodes.size, r.a != 0, r.b != 0);
        break;
      }
      case sc2::ISL_POST_THIRD: sc2::hostPostThird(M); C.post = true; break;
      default: H.win.chk(false, at(("모르는 마디 " + opStr(r)).c_str()));
    }
  }
  C.spec = C.second2 = C.third = C.post = true;
  force(PH_AFTER);
  for (; gi < L.out.size(); ++gi) {
    H.ops.chk(false, at(("우리만 부름: " + opStr(L.out[gi])).c_str()));
    L.apply(gi);
  }
  H.wakeReq += L.wakeReq.size();
  if (const char* d = getenv("G1_HOST_DUMP")) {  // 진단: 이 simulate 의 PhysX 기록과 우리 호출
    if ((long long)H.curSim == atoll(d)) {
      for (const sc2::IslOp& r : win) printf("  [px 창] %s p=%llx\n", opStr(r).c_str(), (unsigned long long)r.p);
      for (const sc2::IslOp& r : simr) printf("  [px] %s p=%llx\n", opStr(r).c_str(), (unsigned long long)r.p);
      for (const sc2::IslOp& r : L.out) printf("  [우리] %s p=%llx\n", opStr(r).c_str(), (unsigned long long)r.p);
    }
  }
  // ---- 비교
  {
    uint64_t n = 0, bad = 0;
    std::string first;
    g1_islands_compare_store(H.store, &n, &bad, &first);
    H.isl.n += n;
    H.isl.bad += bad;
    if (bad && H.isl.first.empty()) H.isl.first = at(first.c_str());
  }
  {
    bool same = acts.size() == S.acts.size();
    for (size_t i = 0; same && i < acts.size(); ++i) {
      const sc2::PairsAct& x = acts[i];
      const sc2::PairsAct& y = S.acts[i];
      same = x.activate == y.activate && x.result == y.result && x.afterFill == y.afterFill && x.e0 == y.e0 && x.e1 == y.e1;
    }
    char b[96];
    snprintf(b, sizeof b, "활성화 호출 (우리 %zu / PhysX %zu)", acts.size(), S.acts.size());
    H.acts.chk(same, at(b));
    H.nActs += acts.size();
  }
  {
    std::vector<int8_t> px;
    g1_pairs_actor_active(px);
    for (size_t k = 0; k < px.size() && k < H.active.size(); ++k)
      if (px[k] >= 0) H.act.chk(H.active[k] == uint8_t(px[k]), at(("행위자 활성 #" + std::to_string(k)).c_str()));
  }
  if (const ss::ScPairs* ref = g1_pairs_M()) comparePairs(P, *ref);
  {
    bool same = H.wake.bodies.size() == post.bodies.size();
    for (size_t i = 0; same && i < post.bodies.size(); ++i) {
      same = H.wake.bodies[i].node == post.bodies[i].node && H.wake.bodies[i].wc == post.bodies[i].wc;
      if (!same && H.show && H.wk.bad == 0)
        printf("    깸 카운터 다름 @sim %llu 노드 %llx: 우리 %.9g / PhysX %.9g (풀이 %.9g)\n", (unsigned long long)H.curSim,
               (unsigned long long)post.bodies[i].node, H.wake.bodies[i].wc, post.bodies[i].wc, post.bodies[i].solverWc);
    }
    H.wk.chk(same, at("몸체 깸 카운터 표"));
    bool sa = H.wake.arts.size() == post.arts.size();
    for (size_t i = 0; sa && i < post.arts.size(); ++i) sa = H.wake.arts[i].node == post.arts[i].node && H.wake.arts[i].wc == post.arts[i].wc;
    H.wk.chk(sa, at("관절체 깸 카운터"));
  }
  (void)sim;
}

void g1_host_report() {
  if (!H.on || !H.running) return;
  printf("G1 순서기 그림자 (넘겨받은 뒤 %" PRIu64 " simulate) — 쌍 관리층↔섬 관리 직접 연결 + Sc 활성 몰이\n", H.steps);
  auto line = [](const char* name, const Stat& s) { printf("  %-34s: 비교 %" PRIu64 ", 다름 %" PRIu64 " %s\n", name, s.n, s.bad, s.first.c_str()); };
  line("우리가 낸 섬 호출 (쌍·깸/잠, 차례·번호)", H.ops);
  line("바깥 호출 결과 번호", H.ext);
  line("섬 시뮬·번호 관리 (스텝 끝)", H.isl);
  line("상호작용 활성/비활성 (호출·결과)", H.acts);
  line("행위자 활성 표시 (스텝 끝)", H.act);
  line("쌍 관리층 상태 (g1_pairs 와)", H.pairs);
  line("순서 점검 (창·마디)", H.win);
  line("깸 카운터 표 (스텝 끝)", H.wk);
  printf("  깨움/잠 요청: afterIntegration %" PRIu64 ", 창에서 우리 것과 맞춘 activateNode %" PRIu64 ", simulate 안 바깥 호출 %" PRIu64 "\n", H.nWakeOps,
         H.winActMatched, H.extInSim);
  printf("  몰이 횟수: 활성화 호출 %" PRIu64 ", 깨움 노드 %" PRIu64 ", 재운 노드 %" PRIu64 ", internalWakeUp %" PRIu64 ", 창에서 바뀐 활성 표시 %" PRIu64 "\n", H.nActs,
         H.woken, H.slept, H.wakeReq, H.winChanged);
}
