// 닫힌 고리 대조 (particles): 리드 대조기(replay/g1_env.cpp)의 약한 기호 g1_env_edit_hook 를 채워, 전이 창을 엔진 편집 창으로 돈다.
//   PARTICLES_PLAN=<build_edit_plan.py 결과> G1_ENV_FROM=<장면 파일> G1_ENV_RUN=1 G1_BP=1 G1_PAIRS=1 G1_ISLANDS=1 G1_LOOP=1 G1_LOOP_PERSIST=1 ovd_replay_g1 ...
// 공식 removing_objects 창 나눔 그대로:
//   simulate K(= step_physics) 앞: 규칙 안 입자 생성 → 상태 뜨기(계 → 물체) → 무덤 순간이동            → 5 (기록 입력은 로봇 제어만)
//   simulate K+1 앞: 지우기 → 상태 되돌리기(계 → 물체 = base 창) → 새 물체 넣기(생성 → USD 동기화) → 동기화 → 5
// 관절체 쪽 base 창 입력(BaseArtApi)은 engine-solver-art 가 채우기 전까지 비어 있다(EnvBaseApi.artMissing 로 셈).
// PARTICLES_PLAN 이 없으면 아무것도 안 함 (0).
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/particles/base_state.h"
#include "core/particles/edit_window.h"
#include "core/particles/env_base_api.h"
#include "core/scene/env_body_api.h"
#include "core/scene/scene_file.h"

using namespace eng;
using namespace eng::particles;

namespace {
struct Half {
  std::string tmpl, name;
  uint32_t actor = 0;
  int part = 0;
  HalfSpec h{};
  float usd[3] = {1, 1, 1};
};
struct ObjLine {
  std::string name, root;
  int kind = 0;
  unsigned nj = 0;
  std::vector<std::string> links;
};
struct Trans {
  uint64_t simRemove = 0;  // K+1
  std::string src, srcRoot;
  float srcScale[3] = {1, 1, 1};
  std::vector<Half> halves;
  std::vector<ObjLine> objs;
};

struct Resolver : SpawnResolver {
  Trans* t = nullptr;
  std::map<std::string, SpawnTemplate>* cache = nullptr;
  bool half(int32_t src, int32_t part, SpawnSource& out) override {
    if (!t || src != 0 || part < 0 || size_t(part) >= t->halves.size()) return false;
    Half& h = t->halves[size_t(part)];
    auto it = cache->find(h.tmpl);
    if (it == cache->end()) {
      std::string err;
      if (!(*cache)[h.tmpl].load(h.tmpl.c_str(), &err)) {
        fprintf(stderr, "[particles 편집] 틀 읽기 실패 %s: %s\n", h.tmpl.c_str(), err.c_str());
        cache->erase(h.tmpl);
        return false;
      }
      it = cache->find(h.tmpl);
    }
    out.T = &it->second;
    out.actor = h.actor;
    memcpy(out.usd_scale, h.usd, 12);
    out.raw_q = false;
    return true;
  }
  bool particle(int32_t, SpawnSource&) override { return false; }
};

struct Hook {
  bool inited = false, on = false;
  std::vector<Trans> T;
  scene::SceneFile f;
  std::map<std::string, int32_t> handleOf;  // 행위자 이름 -> ScScene 손잡이
  std::map<std::string, int32_t> artOfLink;  // 링크 행위자 이름 -> 관절체 번호
  std::map<std::string, SpawnTemplate> cache;
  // 지금 창
  int cur = -1;
  Resolver res;
  std::vector<BaseObject> bobjs;
  std::vector<ObjectRt> objs;
  std::vector<ParticleSystemRt> systems;
  std::vector<Edit> edits;
  std::unique_ptr<EnvBaseApi> BA;
  BaseStateWindow B;
  TransitionEditWindow W;
  size_t removedIdx = size_t(-1);
  uint64_t windows = 0, missingNames = 0, failed = 0;

  ~Hook() {
    if (on)
      fprintf(stderr, "[particles 편집] 전이 %zu, 편집 창 %llu, 이름 못 찾음 %llu, 편집 실패 %llu, 관절체 입력 없음 %llu\n", T.size(), (unsigned long long)windows,
              (unsigned long long)missingNames, (unsigned long long)failed, (unsigned long long)(BA ? BA->artMissing : 0));
  }
  bool load(const char* plan, const char* scenePath) {
    std::string err;
    if (!scene::readScene(scenePath, f, &err)) {
      fprintf(stderr, "[particles 편집] 장면 파일 읽기 실패 %s: %s\n", scenePath, err.c_str());
      return false;
    }
    for (size_t h = 0; h < f.sc.actors.size(); ++h) {  // ScScene 손잡이 -> 첫 요소 -> 장면 모양 -> 장면 행위자 이름
      const scene::ScStateActor& r = f.sc.actors[h];
      if (!r.alive || r.elemCount == 0) continue;
      const uint32_t e = f.sc.elems[r.elemStart];
      if (e >= f.sc.shapes.size()) continue;
      const uint32_t ss = f.sc.shapes[e].sceneShape;
      if (ss >= f.shapes.size()) continue;
      handleOf[f.name(f.actors[f.shapes[ss].actor].name)] = int32_t(h);
    }
    for (const scene::SceneActor& a : f.actors)
      if (a.kind == scene::kLink) artOfLink[f.name(a.name)] = int32_t(a.body);
    FILE* p = fopen(plan, "r");
    if (!p) return false;
    char buf[1 << 16];
    auto F = [](std::istringstream& is) {
      std::string t;
      is >> t;
      return strtof(t.c_str(), nullptr);
    };
    while (fgets(buf, sizeof buf, p)) {
      std::istringstream is(buf);
      std::string k;
      is >> k;
      if (k == "T") {
        T.emplace_back();
        Trans& t = T.back();
        is >> t.simRemove >> t.src >> t.srcRoot;
        for (float& x : t.srcScale) x = F(is);
      } else if (k == "H") {
        Half h;
        is >> h.tmpl >> h.actor >> h.name >> h.part;
        for (float& x : h.h.bb_pos) x = F(is);
        for (float& x : h.h.bb_orn) x = F(is);
        for (float& x : h.h.bb_size) x = F(is);
        for (float& x : h.h.native_bb) x = F(is);
        for (float& x : h.h.base_link_offset) x = F(is);
        T.back().halves.push_back(h);
      } else if (k == "O") {
        ObjLine o;
        unsigned nl;
        is >> o.name >> o.kind >> o.nj >> o.root >> nl;
        for (unsigned i = 0; i < nl; ++i) {
          std::string l;
          is >> l;
          o.links.push_back(l);
        }
        T.back().objs.push_back(o);
      }
    }
    fclose(p);
    fprintf(stderr, "[particles 편집] 계획 %s: 전이 %zu, 장면 이름 %zu (관절체 링크 %zu)\n", plan, T.size(), handleOf.size(), artOfLink.size());
    return true;
  }
  int32_t handle(const std::string& n) {
    auto it = handleOf.find(n);
    if (it == handleOf.end()) {
      ++missingNames;
      return -1;
    }
    return it->second;
  }
  // simulate K 앞: 뜨기·무덤
  void begin(scene::EnvStep& E, scene::EnvBodyApi& api, int ti) {
    cur = ti;
    Trans& t = T[size_t(ti)];
    bobjs.clear();
    removedIdx = size_t(-1);
    for (size_t i = 0; i < t.objs.size(); ++i) {
      const ObjLine& o = t.objs[i];
      BaseObject b;
      b.nJoints = o.nj;
      if (o.kind == 1) {
        auto it = artOfLink.find(o.root);
        if (it == artOfLink.end()) ++missingNames;
        else b.kind = BASE_ART, b.art = it->second;
      } else {
        b.root = handle(o.root);
        b.kind = b.root < 0 ? BASE_NONE : (o.kind == 2 ? BASE_KINEMATIC : BASE_RIGID);
        for (const std::string& l : o.links) {
          const int32_t h = handle(l);
          if (h >= 0) b.links.push_back(h);
        }
      }
      if (o.name == t.src) removedIdx = i;
      bobjs.push_back(b);
    }
    objs.assign(1 + t.halves.size(), ObjectRt());
    const int32_t sh = handle(t.srcRoot);
    if (sh >= 0) objs[0].actors.push_back(sh);
    edits.clear();
    Edit rb;
    rb.kind = EDIT_REMOVE_BEGIN;
    rb.poses.push_back(Pose7{{100.0f, 100.0f, 100.0f}, {0.0f, 0.0f, 0.0f, 1.0f}});  // 지울 물체 하나: 무덤 첫 자리
    edits.push_back(rb);
    Edit ro;
    ro.kind = EDIT_REMOVE_OBJECT;
    ro.object = 0;
    edits.push_back(ro);
    // 반쪽 자세: 규칙이 본 원본 자세(= 지난 simulate 결과의 뿌리 링크 자세)로 (slicing.h, 공식과 비트 동일 확인된 사슬)
    const Tf sp = sh >= 0 ? api.actorPose(sh) : Tf{qid(), V3{0, 0, 0}};
    const float spos[3] = {sp.p.x, sp.p.y, sp.p.z}, sorn[4] = {sp.q.x, sp.q.y, sp.q.z, sp.q.w};
    for (size_t i = 0; i < t.halves.size(); ++i) {
      Half& h = t.halves[i];
      Edit ad;
      ad.kind = EDIT_ADD_OBJECT;
      ad.object = int32_t(1 + i);
      ad.src = 0;
      ad.part = int32_t(i);
      ad.poses.push_back(half_add_pose(spos, sorn, t.srcScale, h.h));
      float bp[3], bo[4], bb[3];
      slice_part_bbox(spos, sorn, t.srcScale, h.h.bb_pos, h.h.bb_orn, h.h.bb_size, bp, bo, bb);
      half_scale(bb, h.h.native_bb, h.usd);
      edits.push_back(ad);
    }
    res.t = &t;
    res.cache = &cache;
    if (!BA) BA.reset(new EnvBaseApi(api));
    B.api = BA.get();
    B.objects = &bobjs;
    W.base = &B;
    W.body = &api;
    W.spawn = &res;
    W.objects = &objs;
    W.systems = &systems;
    W.edits = &edits;
    W.pending.clear();
    const bool tr = getenv("PARTICLES_HOOK_TRACE") != nullptr;
    if (tr) fprintf(stderr, "[particles 편집] 창 %d 시작: 물체 %zu, 원본 손잡이 %d, 반쪽 %zu\n", ti, bobjs.size(), sh, t.halves.size());
    W.runRuleEdits(E);
    W.dumpState(E);
    if (tr) fprintf(stderr, "[particles 편집] 뜨기 끝\n");
    W.teleportToGrave(E);
    if (tr) fprintf(stderr, "[particles 편집] 무덤 끝\n");
  }
  // simulate K+1 앞: 지우기·되돌리기·넣기·동기화
  void finish(scene::EnvStep& E) {
    Trans& t = T[size_t(cur)];
    const bool tr = getenv("PARTICLES_HOOK_TRACE") != nullptr;
    if (removedIdx != size_t(-1)) B.markRemoved(removedIdx);
    W.edit(E);  // 지운 요소의 바뀜·dirty 칸은 sc_scene.h removeActor 가 지움 (리드 72c1a98)
    if (tr) fprintf(stderr, "[particles 편집] 지우기 끝\n");
    W.loadState(E);
    if (tr) fprintf(stderr, "[particles 편집] 되돌리기·넣기 끝 (실패 %d)\n", W.failed);
    W.flushSync();
    W.pending.clear();
    handleOf.erase(t.srcRoot);
    for (size_t i = 0; i < t.halves.size(); ++i)  // 다음 창 물체 표가 이름으로 찾도록
      if (!objs[1 + i].actors.empty()) handleOf["/World/scene_0/" + t.halves[i].name + "/base_link"] = objs[1 + i].actors[0];
    failed += uint64_t(W.failed);
    W.failed = 0;
    ++windows;
    cur = -1;
  }
} G;
}  // namespace

int g1_env_edit_hook(uint64_t sim, eng::scene::EnvStep& E, eng::scene::EnvSolveImpl& S, eng::scene::EnvBodyApi& api) {
  (void)S;
  if (!G.inited) {
    G.inited = true;
    if (getenv("PARTICLES_PLAN")) {  // 세그폴트 때 스택만 찍고 코어 덤프 없이 끝냄 (WSL 충돌 덤프 460 MB 방지)
      struct sigaction sa;
      memset(&sa, 0, sizeof sa);
      sa.sa_handler = [](int) {
        void* bt[64];
        const int n = backtrace(bt, 64);
        const char m[] = "[particles 편집] SIGSEGV 스택:\n";
        (void)!write(2, m, sizeof m - 1);
        backtrace_symbols_fd(bt, n, 2);
        _exit(86);
      };
      sigaction(SIGSEGV, &sa, nullptr);
      sigaction(SIGBUS, &sa, nullptr);
      sigaction(SIGABRT, &sa, nullptr);
    }
    const char* plan = getenv("PARTICLES_PLAN");
    const char* from = getenv("G1_ENV_FROM");
    G.on = plan && from && G.load(plan, from);
  }
  if (!G.on) return 0;
  for (size_t i = 0; i < G.T.size(); ++i) {
    if (sim + 1 == G.T[i].simRemove) {
      G.begin(E, api, int(i));
      return 5;
    }
    if (sim == G.T[i].simRemove && G.cur == int(i)) {
      G.finish(E);
      return 5;
    }
  }
  return 0;
}
