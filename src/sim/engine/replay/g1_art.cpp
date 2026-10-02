// G1 관절체 단독 그림자 시험 (문서 15절 G1): 공식 radio 장면 재생 중, 접촉·조인트 없이 관절체 하나만 든 활성 섬(문·서랍 등)을
// 우리 관절체 모듈(core/articulation stepAlone, 층 1)로 PhysX 와 같은 스텝만큼 돌려 비트 비교한다.
// - 스텝마다 다시 맞춤: simulate 바로 앞(이번 스텝 입력이 다 들어간 뒤)에 깨어 있는 관절체를 전부 옮겨 담고(tests/articulation/art_from_px.h
//   snapshotFromPx, S5 비계용 — 층 1 시험에서 아무 경계든 옮긴 뒤 비트 동일 확인됨), 풀이 직전 스냅샷(g1_solver.cpp)이 알려 준
//   "관절체 하나뿐인 섬"만 한 스텝 돌린다. 반복 수는 그 섬이 든 PhysX 풀이 묶음의 최댓값(강체·관절체, DyTGSDynamics.cpp:1006,1690).
// - 비교: 링크 자세·선/각속도, 관절 위치·속도(텐서 캐시와 같은 길), 깸 카운터, 잠 여부 (test_articulation_snapshot 과 같은 항목).
// 켜기: G1_ART=1 (디스패처가 필요해 g1_solver.cpp 의 스냅샷을 같이 씀). G1_ART_SHOW=N 이면 다른 것 N 개를 자세히.
#include <xmmintrin.h>

#include <cinttypes>

#include "../tests/articulation/art_from_px.h"
#include "g1_hooks.h"

using namespace physx;
using namespace artest;

namespace {

struct FtzScope {
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

struct Twin {
  PxArticulationReducedCoordinate* px = nullptr;
  std::vector<PxArticulationLink*> links;
  std::unique_ptr<A::Articulation> e;
  bool ok = false;
};

struct ArtShadow {
  bool on = false, snap = false, inited = false;
  std::unordered_map<const void*, std::pair<const void*, uint32_t>> linkOfRb;  // 링크 PxsRigidBody* -> (FeatherstoneArticulation*, LL)
  std::unordered_map<const void*, Twin> twins;  // Dy::FeatherstoneArticulation* -> 이번 스텝 쌍둥이
  std::unordered_map<PxArticulationReducedCoordinate*, PxArticulationCache*> caches;
  uint64_t steps = 0, stepsNoInfo = 0, awakeSnap = 0, snapFail = 0, aloneSeen = 0, noTwin = 0, cmp = 0, bad = 0, nanBoth = 0, fields = 0;
  long long firstBad = -1;
  int show = 0;
  std::map<std::string, uint64_t> badByName;
  // 닫힌 고리 (2b) 지속 모드: 관절체별 우리 상태 (지난 스텝 우리 결과 + 이번 창 API 호출)
  std::unordered_map<const void*, std::unique_ptr<A::Articulation>> persisted;  // PxArticulationReducedCoordinate* -> 상태
  uint64_t pUsed = 0, pResync = 0, pNew = 0, pDiff = 0, pOps = 0, pCmp = 0;
  int arrShown = 0;
  long long pFirst = -1;
  std::map<std::string, uint64_t> pDiffByName;
  uint64_t curSim = 0;
} AS;

bool diffTwin(Twin& t, const eng::art::Articulation& e, size_t* firstJ, size_t* nFields, float* pxv, float* ev);

// 이번 창에 옮긴 API 호출을 우리 상태에 (core/articulation/articulation.h 의 API 판)
void applyOps(A::Articulation& e, const std::vector<G1ArtOp>& ops) {
  for (const G1ArtOp& o : ops) {
    switch (o.type) {
      case 0: A::jointSetDriveTarget(e, o.link, o.axis, o.v, true); break;
      case 1: A::jointSetDriveVelocity(e, o.link, o.axis, o.v, true); break;
      case 2: {  // NpArticulationReducedCoordinate::wakeUp (:1161): 링크마다 깸 카운터 = 재설정 값, 관절체도
        const float reset = 20.0f * 0.02f;
        for (uint32_t i = 0; i < e.nLinks; ++i) e.bodies[i].wakeCounter = reset;
        e.wakeCounter = reset;
        e.awake = 1;
        e.readyForSleep = 0;
        break;
      }
      default: break;  // putToSleep: 아직 (건드림으로 다시 맞춤)
    }
  }
}

}  // namespace

// simulate 바로 앞 (이번 스텝 입력이 다 들어간 뒤): 깨어 있는 관절체를 전부 옮겨 담는다
void g1_art_before(PxScene* scene) {
  if (g1_loop_persist() && !AS.inited) AS.show = getenv("G1_ART_SHOW") ? atoi(getenv("G1_ART_SHOW")) : 5;
  if (!AS.inited) {
    AS.inited = true;
    AS.on = getenv("G1_ART") != nullptr;
    AS.snap = AS.on || getenv("G1_SOLVER") != nullptr;
    if (g1_loop_persist() && !getenv("G1_ART_SHOW")) AS.show = 5;  // solver 전체 그림자(관절체 결합)도 쌍둥이를 쓴다
    AS.show = getenv("G1_ART_SHOW") ? atoi(getenv("G1_ART_SHOW")) : 0;
  }
  if (!AS.snap) return;
  AS.twins.clear();
  AS.linkOfRb.clear();
  const PxU32 n = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(n);
  scene->getArticulations(arts.data(), n);
  const PxTolerancesScale& tol = scene->getPhysics().getTolerancesScale();
  A::SceneScale sc;
  sc.length = tol.length;
  sc.speed = tol.speed;
  for (PxArticulationReducedCoordinate* a : arts) {
    if (a->isSleeping() && AS.on && !getenv("G1_SOLVER")) continue;  // solver 전체 그림자는 잠든 것도 (simulate 안에서 깨어 섬에 들 수 있음)
    Dy::FeatherstoneArticulation* fa = llArticulation(a);
    if (!fa) continue;
    Twin t;
    t.px = a;
    t.links.resize(a->getNbLinks());
    a->getLinks(t.links.data(), a->getNbLinks());
    t.e.reset(new A::Articulation);
    ++AS.awakeSnap;
    t.ok = t.links.size() <= A::kMaxLinks && snapshotFromPx(a, *t.e, sc, t.links);
    for (PxArticulationLink* l : t.links)
      AS.linkOfRb[&static_cast<NpArticulationLink*>(l)->getCore().getSim()->getLowLevelBody()] = {fa, l->getLinkIndex()};
    if (!t.ok) ++AS.snapFail;
    if (t.ok && g1_loop_persist()) {
      std::vector<G1ArtOp> ops;
      g1_loop_take_art_ops(a, ops);
      auto pit = AS.persisted.find(a);
      bool sleepOp = false;
      for (const G1ArtOp& o : ops) sleepOp |= o.type == 3;
      if (pit == AS.persisted.end()) {
        ++AS.pNew;
      } else if (g1_loop_touched(a) || sleepOp) {
        ++AS.pResync;  // 옮기지 않은 API 로 건드림 -> 이번 스텝은 PhysX 로 다시 맞춤
        AS.persisted.erase(pit);
      } else {
        A::Articulation& pe = *pit->second;
        AS.pOps += ops.size();
        applyOps(pe, ops);
        size_t fj = 0, nf = 0;
        float pv = 0, ev = 0;
        ++AS.pCmp;
        if (diffTwin(t, pe, &fj, &nf, &pv, &ev)) {  // 우리 상태(지난 결과 + API) != PhysX simulate 앞 상태
          ++AS.pDiff;
          if (AS.pFirst < 0) AS.pFirst = (long long)AS.curSim;
          AS.pDiffByName[a->getName() ? a->getName() : "?"]++;
          if (AS.show > 0) {
            --AS.show;
            fprintf(stderr, "[g1 loop 관절체 다름] %s 칸 %zu/%zu: PhysX %.9g 우리 %.9g\n", a->getName() ? a->getName() : "?", fj, nf, pv, ev);
          }
        }
        if (getenv("G1_LOOP_ARRDIFF") && AS.arrShown < 80 && (!getenv("G1_LOOP_ARRDIFF_NAME") || strstr(a->getName() ? a->getName() : "", getenv("G1_LOOP_ARRDIFF_NAME")))) {  // 진단: 우리 상태와 새 스냅샷의 배열별 차이 (공개 값은 같아도 안쪽이 다른 곳)
          const A::Articulation& fe = *t.e;
          const A::ArtCaps c{pe.nLinks, pe.dofs, pe.nPath, pe.nMimic};  // 쓰는 칸만
          using namespace eng;
          using namespace eng::art;
          auto cmpArr = [&](const char* name, const void* xp, const void* yp, size_t bytes, size_t esz) {
            const unsigned char* x = static_cast<const unsigned char*>(xp);
            const unsigned char* y = static_cast<const unsigned char*>(yp);
            size_t k = 0;
            while (k < bytes && x[k] == y[k]) ++k;
            if (k < bytes && AS.arrShown < 80) {
              ++AS.arrShown;
              fprintf(stderr, "[g1 loop 배열 다름] sim %llu %s: %s 바이트 %zu/%zu (원소 %zu)\n", (unsigned long long)AS.curSim, a->getName() ? a->getName() : "?", name, k,
                      bytes, k / esz);
            }
          };
#define ENG_ART_CMP(T, n, cnt) cmpArr(#n, static_cast<const T*>(pe.n), static_cast<const T*>(fe.n), sizeof(T) * size_t(cnt), sizeof(T));
          ENG_ART_ARRAYS(ENG_ART_CMP)
#undef ENG_ART_CMP
          cmpArr("머리", &pe, &fe, offsetof(A::Articulation, ll), 1);
        }
        *t.e = pe;  // 닫힌 고리: 우리 상태로 푼다
        ++AS.pUsed;
      }
    }
    AS.twins[fa] = std::move(t);
  }
}

// fetchResults 뒤: 관절체 하나뿐인 섬을 우리 모듈로 한 스텝 돌려 비교
void g1_art_after(PxScene* scene, uint64_t sim) {
  if (!AS.on) return;
  ++AS.steps;
  const G1StepInfo& inf = g1_step_info();
  if (!inf.valid) {
    ++AS.stepsNoInfo;
    return;
  }
  const PxSceneFlags sf = scene->getFlags();
  for (const G1ArtAlone& al : inf.artAlone) {
    ++AS.aloneSeen;
    auto it = AS.twins.find(al.fa);
    if (it == AS.twins.end() || !it->second.ok) {
      ++AS.noTwin;
      continue;
    }
    Twin& t = it->second;
    A::Articulation& e = *t.e;
    A::StepParams sp;
    sp.gravity = eng::V3{inf.gravity[0], inf.gravity[1], inf.gravity[2]};
    sp.dt = inf.dt;
    sp.lengthScale = scene->getPhysics().getTolerancesScale().length;
    sp.posIters = al.iterWord & 0xff;
    sp.velIters = al.iterWord >> 8;
    sp.externalForcesEveryTgsIteration = sf & PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
    sp.solveArticulationContactLast = sf & PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST;
    {
      FtzScope f;
      A::stepAlone(e, sp);
    }
    // 비교 (test_articulation_snapshot.cpp 와 같은 항목)
    std::vector<float> x, y;
    for (uint32_t l = 0; l < t.links.size(); ++l) {
      const PxTransform tp = t.links[l]->getGlobalPose();
      const eng::Tf te = A::linkGlobalPose(e, l);
      const PxVec3 lv = t.links[l]->getLinearVelocity(), av = t.links[l]->getAngularVelocity();
      const A::LinkBody& b = e.bodies[e.ll[l]];
      const float p13[13] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lv.x, lv.y, lv.z, av.x, av.y, av.z};
      const float e13[13] = {te.q.x, te.q.y, te.q.z, te.q.w, te.p.x, te.p.y, te.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z};
      x.insert(x.end(), p13, p13 + 13);
      y.insert(y.end(), e13, e13 + 13);
    }
    if (e.dofs) {
      PxArticulationCache*& c = AS.caches[t.px];
      if (!c) c = t.px->createCache();
      t.px->copyInternalStateToCache(*c, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
      x.insert(x.end(), c->jointPosition, c->jointPosition + e.dofs);
      x.insert(x.end(), c->jointVelocity, c->jointVelocity + e.dofs);
      const float* jp = e.jointPosition;  // 고정 배열·RelArr 둘 다 (용량 등급 작업과 함께 빌드되게)
      const float* jv = e.jointVelocity;
      y.insert(y.end(), jp, jp + e.dofs);
      y.insert(y.end(), jv, jv + e.dofs);
    }
    x.push_back(t.px->getWakeCounter());
    y.push_back(e.wakeCounter);
    x.push_back(t.px->isSleeping() ? 1.0f : 0.0f);
    y.push_back(e.awake ? 0.0f : 1.0f);
    ++AS.cmp;
    AS.fields += x.size();
    bool same = true, anyNan = false;
    size_t firstJ = 0;
    for (size_t j = 0; j < x.size(); ++j) {
      const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
      if (nx || ny) {
        anyNan = true;
        if (nx != ny && same) same = false, firstJ = j;
      } else if (memcmp(&x[j], &y[j], 4) && same) {
        same = false;
        firstJ = j;
      }
    }
    if (same && anyNan) ++AS.nanBoth;
    if (!same) {
      ++AS.bad;
      if (AS.firstBad < 0) AS.firstBad = (long long)sim;
      const char* nm = t.px->getName();
      AS.badByName[nm ? nm : "?"]++;
      if (AS.show > 0) {
        --AS.show;
        fprintf(stderr, "[g1 art 다름] sim %llu %s 링크 %zu dof %u 반복 %04x 칸 %zu/%zu: PhysX %.9g 우리 %.9g\n", (unsigned long long)sim, nm ? nm : "?",
                t.links.size(), e.dofs, unsigned(al.iterWord), firstJ, x.size(), x[firstJ], y[firstJ]);
      }
    }
  }
}

// ---- solver 전체 그림자(g1_solver.cpp)가 쓰는 것
eng::art::Articulation* g1_art_twin(const void* fa) {
  auto it = AS.twins.find(fa);
  return it == AS.twins.end() || !it->second.ok ? nullptr : it->second.e.get();
}
bool g1_art_link_of_rb(const void* rb, const void** fa, uint32_t* ll) {
  auto it = AS.linkOfRb.find(rb);
  if (it == AS.linkOfRb.end()) return false;
  *fa = it->second.first;
  *ll = it->second.second;
  return true;
}
// 우리 관절체 e 와 PhysX(fetchResults 뒤) 비교: 링크 자세·속도·깸, 관절 위치·속도, 관절체 깸·잠. 다르면 첫 칸과 두 값
bool g1_art_diff(const void* fa, const eng::art::Articulation& e, size_t* firstJ, size_t* nFields, float* pxv, float* ev) {
  auto it = AS.twins.find(fa);
  if (it == AS.twins.end()) return true;
  return diffTwin(it->second, e, firstJ, nFields, pxv, ev);
}
namespace {
bool diffTwin(Twin& t, const eng::art::Articulation& e, size_t* firstJ, size_t* nFields, float* pxv, float* ev) {
  std::vector<float> x, y;
  for (uint32_t l = 0; l < t.links.size(); ++l) {
    const PxTransform tp = t.links[l]->getGlobalPose();
    const eng::Tf te = A::linkGlobalPose(e, l);
    const PxVec3 lv = t.links[l]->getLinearVelocity(), av = t.links[l]->getAngularVelocity();
    const A::LinkBody& b = e.bodies[e.ll[l]];
    const float p13[13] = {tp.q.x, tp.q.y, tp.q.z, tp.q.w, tp.p.x, tp.p.y, tp.p.z, lv.x, lv.y, lv.z, av.x, av.y, av.z};
    const float e13[13] = {te.q.x, te.q.y, te.q.z, te.q.w, te.p.x, te.p.y, te.p.z, b.linVel.x, b.linVel.y, b.linVel.z, b.angVel.x, b.angVel.y, b.angVel.z};
    x.insert(x.end(), p13, p13 + 13);
    y.insert(y.end(), e13, e13 + 13);
    x.push_back(static_cast<NpArticulationLink*>(t.links[l])->getCore().getCore().wakeCounter);
    y.push_back(b.wakeCounter);
  }
  if (e.dofs) {
    PxArticulationCache*& c = AS.caches[t.px];
    if (!c) c = t.px->createCache();
    t.px->copyInternalStateToCache(*c, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
    x.insert(x.end(), c->jointPosition, c->jointPosition + e.dofs);
    x.insert(x.end(), c->jointVelocity, c->jointVelocity + e.dofs);
    const float* jp = e.jointPosition;
    const float* jv = e.jointVelocity;
    y.insert(y.end(), jp, jp + e.dofs);
    y.insert(y.end(), jv, jv + e.dofs);
  }
  x.push_back(t.px->getWakeCounter());
  y.push_back(e.wakeCounter);
  x.push_back(t.px->isSleeping() ? 1.0f : 0.0f);
  y.push_back(e.awake ? 0.0f : 1.0f);
  *nFields = x.size();
  for (size_t j = 0; j < x.size(); ++j) {
    const bool nx = std::isnan(x[j]), ny = std::isnan(y[j]);
    if ((nx || ny) ? (nx != ny) : memcmp(&x[j], &y[j], 4) != 0) {
      *firstJ = j;
      *pxv = x[j];
      *ev = y[j];
      return true;
    }
  }
  return false;
}
}  // namespace
// 풀이 뒤 우리 결과를 다음 스텝으로 (지속 모드)
void g1_art_persist(const void* fa, const eng::art::Articulation& e, uint64_t sim) {
  auto it = AS.twins.find(fa);
  if (it == AS.twins.end()) return;
  std::unique_ptr<A::Articulation>& p = AS.persisted[it->second.px];
  if (!p) p.reset(new A::Articulation);
  *p = e;
  AS.curSim = sim + 1;
}
// env 닫힌 고리(g1_env.cpp): 장면 관절체 차례(getArticulations = 장면 뜨기 차례)로 PhysX 관절체 -> 엔진 관절체 뜨기, 우리 것과 비교, 이번 창 옮긴 호출
std::vector<const void*> g1_env_px_arts(PxScene* scene) {
  const PxU32 n = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(n);
  scene->getArticulations(arts.data(), n);
  return std::vector<const void*>(arts.begin(), arts.end());
}
bool g1_env_art_snapshot(PxScene* scene, const void* pa, eng::art::Articulation& out) {
  PxArticulationReducedCoordinate* a = static_cast<PxArticulationReducedCoordinate*>(const_cast<void*>(pa));
  std::vector<PxArticulationLink*> links(a->getNbLinks());
  a->getLinks(links.data(), a->getNbLinks());
  const PxTolerancesScale& tol = scene->getPhysics().getTolerancesScale();
  A::SceneScale sc;
  sc.length = tol.length;
  sc.speed = tol.speed;
  return links.size() <= A::kMaxLinks && snapshotFromPx(a, out, sc, links);
}
bool g1_env_art_diff(const void* pa, const eng::art::Articulation& e, size_t* firstJ, size_t* nFields, float* pxv, float* ev) {
  Twin t;
  t.px = static_cast<PxArticulationReducedCoordinate*>(const_cast<void*>(pa));
  t.links.resize(t.px->getNbLinks());
  t.px->getLinks(t.links.data(), t.px->getNbLinks());
  return diffTwin(t, e, firstJ, nFields, pxv, ev);
}
void g1_env_art_apply_ops(eng::art::Articulation& e, const std::vector<G1ArtOp>& ops) { applyOps(e, ops); }
const char* g1_art_name(const void* fa) {
  auto it = AS.twins.find(fa);
  return it == AS.twins.end() || !it->second.px->getName() ? "?" : it->second.px->getName();
}

void g1_art_report() {
  if (g1_loop_persist()) {
    printf("G1 닫힌 고리 (2b) 관절체: 우리 상태로 푼 관절체·스텝 %" PRIu64 " (API 호출 %" PRIu64 " 넣음), 처음 %" PRIu64 ", 다시 맞춤(옮기지 않은 API) %" PRIu64
           " — 우리 상태 != PhysX simulate 앞 %" PRIu64 "/%" PRIu64 "%s\n",
           AS.pUsed, AS.pOps, AS.pNew, AS.pResync, AS.pDiff, AS.pCmp, AS.pDiff ? ("  첫 simulate " + std::to_string(AS.pFirst)).c_str() : "");
    int k = 0;
    for (auto& kv : AS.pDiffByName)
      if (k++ < 8) printf("    다름 %6" PRIu64 "  %s\n", kv.second, kv.first.c_str());
  }
  if (!AS.on) return;
  printf("G1 관절체 단독 그림자 (접촉·조인트 없는 관절체 섬, 스텝마다 PhysX 에서 옮겨 담음): simulate %" PRIu64 " (풀이 스냅샷 없음 %" PRIu64 ")\n", AS.steps,
         AS.stepsNoInfo);
  printf("  깨어 있는 관절체 옮겨 담기 %" PRIu64 " (실패 %" PRIu64 "), 단독 섬 %" PRIu64 " (쌍둥이 없음 %" PRIu64 ")\n", AS.awakeSnap, AS.snapFail, AS.aloneSeen, AS.noTwin);
  printf("  관절체·스텝 비교 %" PRIu64 " (값 %" PRIu64 " 개) 비트 다름 %" PRIu64 "%s, 양쪽 NaN %" PRIu64 "\n", AS.cmp, AS.fields, AS.bad,
         AS.bad ? ("  첫 다름 simulate " + std::to_string(AS.firstBad)).c_str() : "", AS.nanBoth);
  int k = 0;
  for (auto& kv : AS.badByName)
    if (k++ < 10) printf("    다름 %6" PRIu64 "  %s\n", kv.second, kv.first.c_str());
}

const void* g1_art_link_sim(const void* fa, uint32_t creationIdx) {
  auto it = AS.twins.find(fa);
  if (it == AS.twins.end() || creationIdx >= it->second.links.size()) return nullptr;
  return static_cast<NpArticulationLink*>(it->second.links[creationIdx])->getCore().getSim();
}
