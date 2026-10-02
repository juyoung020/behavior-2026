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
#include "core/scene/env_art_api.h"
#include "core/scene/scene_file.h"
#include "core/particles/trng.h"
#include "tests/omni/npy.h"

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
  // 다지기 (P 줄): 입자 계 틀·틀 안 행위자(새 입자 차례)·계 틀 prim 행위자(첫 다지기만, 없으면 -1)·다지기 기록 폴더(dice_events_to_npy ev_NNN)
  bool dice = false;
  std::string system, tmpl, diceDir;
  int32_t tmplTemplateActor = -1;
  std::vector<uint32_t> particleActors;
  std::map<std::string, std::vector<float>> scale;  // 물체 이름 -> 척도 (C 줄)
  std::map<std::string, std::vector<double>> actorScale;
  std::map<std::string, Tf> xform;  // X 줄: 운동학 물체 dump XForm 자세  // 행위자 prim -> 세계 척도 (K 줄, 있으면 우선)
  std::map<int, std::vector<std::pair<std::string, int>>> sweepOrder;  // 벌 -> (행위자 이름, 정적?) 공식 차례 (PARTICLES_SYNC_ORDER)
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
  bool particle(int32_t system, int32_t idx, SpawnSource& out) override {
    if (!t || system != 0 || idx < 0 || size_t(idx) >= t->particleActors.size()) return false;
    auto it = cache->find(t->tmpl);
    if (it == cache->end()) {
      std::string err;
      if (!(*cache)[t->tmpl].load(t->tmpl.c_str(), &err)) {
        fprintf(stderr, "[particles 편집] 틀 읽기 실패 %s: %s\n", t->tmpl.c_str(), err.c_str());
        cache->erase(t->tmpl);
        return false;
      }
      it = cache->find(t->tmpl);
    }
    out.T = &it->second;
    out.actor = t->particleActors[size_t(idx)];
    const auto& A = it->second.shared->actors[out.actor];
    const auto& g = it->second.shared->shapes[A.shapeStart].geom;  // 입자 prim 척도 = 볼록 척도 (B10)
    out.usd_scale[0] = g.convex.scale.scale.x, out.usd_scale[1] = g.convex.scale.scale.y, out.usd_scale[2] = g.convex.scale.scale.z;
    out.raw_q = true;
    return true;
  }
};

struct Hook;
// 동기화 벌 차례: PARTICLES_SYNC_ORDER 가 있으면 공식 차례(이름), 없으면 등록부 차례(정적 뿌리 → 운동학 물체의 동적 링크 → 동적 뿌리 → 새 물체)
struct SyncFromPlan : TransitionEditWindow::SyncOrder {
  Hook* H = nullptr;
  void order(int sweep, std::vector<TransitionEditWindow::SyncActor>& out) override;
};

struct Hook {
  bool inited = false, on = false;
  SyncFromPlan syncp;
  uint64_t syncMissing = 0;
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
  std::vector<uint32_t> particleTmplIdx;  // systems[0].actors 와 같은 차례: 입자의 틀 안 행위자·틀 파일 (벌 척도 = 볼록 척도)
  std::vector<std::string> particleTmpl;
  std::vector<Edit> edits;
  std::unique_ptr<EnvBaseApi> BA;
  std::unique_ptr<scene::EnvArtApi> AA;
  scene::EnvSolveImpl* S = nullptr;
  BaseStateWindow B;
  TransitionEditWindow W;
  size_t removedIdx = size_t(-1);
  int32_t systemTemplateH = -1;
  uint64_t windows = 0, missingNames = 0, failed = 0;

  ~Hook() {
    if (on)
      fprintf(stderr, "[particles 편집] 전이 %zu, 편집 창 %llu, 이름 못 찾음 %llu, 편집 실패 %llu, 관절체 입력 없음 %llu, 벌 행위자 못 찾음 %llu, 정적 쓰기 건너뜀 %llu\n", T.size(),
              (unsigned long long)windows, (unsigned long long)missingNames, (unsigned long long)failed, (unsigned long long)(BA ? BA->artMissing : 0),
              (unsigned long long)syncMissing, (unsigned long long)W.staticSkipped);
  }
  bool load(const char* plan, const char* scenePath) {
    std::string err;
    if (!scene::readScene(scenePath, f, &err)) {
      fprintf(stderr, "[particles 편집] 장면 파일 읽기 실패 %s: %s\n", scenePath, err.c_str());
      return false;
    }
    size_t identity = 0, notIdentity = 0;
    for (size_t h = 0; h < f.sc.actors.size(); ++h) {  // ScScene 손잡이 -> 첫 요소 -> 장면 모양 -> 장면 행위자 이름
      const scene::ScStateActor& r = f.sc.actors[h];
      if (!r.alive || r.elemCount == 0) continue;
      const uint32_t e = f.sc.elems[r.elemStart];
      if (e >= f.sc.shapes.size()) continue;
      const uint32_t ss = f.sc.shapes[e].sceneShape;
      if (ss >= f.shapes.size()) continue;
      handleOf[f.name(f.actors[f.shapes[ss].actor].name)] = int32_t(h);
    }
    // 모양 없는 행위자(메타 링크 등): Sc 손잡이가 "링크 뺀 장면 행위자 차례" 와 모양 있는 것에서 전부 같으면 같은 규칙으로
    std::vector<uint32_t> rigid;  // 링크 뺀 장면 행위자 번호
    for (uint32_t a = 0; a < f.actors.size(); ++a)
      if (f.actors[a].kind != scene::kLink) rigid.push_back(a);
    for (size_t h = 0; h < f.sc.actors.size(); ++h) {
      const scene::ScStateActor& r = f.sc.actors[h];
      if (!r.alive || r.elemCount == 0) continue;
      const uint32_t e = f.sc.elems[r.elemStart];
      if (e >= f.sc.shapes.size() || f.sc.shapes[e].sceneShape >= f.shapes.size()) continue;
      (h < rigid.size() && rigid[h] == f.shapes[f.sc.shapes[e].sceneShape].actor ? identity : notIdentity)++;
    }
    if (notIdentity == 0 && f.sc.actors.size() == rigid.size())
      for (size_t h = 0; h < f.sc.actors.size(); ++h)
        if (f.sc.actors[h].alive && f.sc.actors[h].elemCount == 0) handleOf.emplace(f.name(f.actors[rigid[h]].name), int32_t(h));
    // 동적 행위자는 섬 노드로도 찾는다: Sc 손잡이 -> 섬 노드 -> 노드의 장면 행위자 (env_solve.h 와 같은 규칙) — 모양 없는 메타 링크
    size_t byNode = 0;
    const scene::IslandSimState& acc = f.islands.accurate;
    for (size_t h = 0; h < f.sc.actors.size(); ++h) {
      const scene::ScStateActor& r = f.sc.actors[h];
      if (!r.alive || r.node == ~0ull) continue;
      const uint32_t id = uint32_t(r.node & 0xffffffffu);
      if (id >= acc.nodes.size()) continue;
      const ig::Node& n = acc.nodes[id];
      if ((n.flags & ig::N_DELETED) || n.type != ig::eRIGID_BODY_TYPE || n.object >= f.actors.size()) continue;
      byNode += handleOf.emplace(f.name(f.actors[n.object].name), int32_t(h)).second;
    }
    fprintf(stderr, "[particles 편집] 섬 노드로 더 찾은 행위자 %zu\n", byNode);
    fprintf(stderr, "[particles 편집] 손잡이=링크 뺀 장면 행위자 차례: %zu 같음 %zu 다름 (Sc 행위자 %zu, 링크 뺀 장면 행위자 %zu)\n", identity, notIdentity, f.sc.actors.size(),
            rigid.size());
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
      } else if (k == "P") {
        T.emplace_back();
        Trans& t = T.back();
        t.dice = true;
        unsigned n;
        is >> t.simRemove >> t.src >> t.srcRoot >> t.system >> t.tmpl >> t.tmplTemplateActor >> n;
        for (unsigned i = 0; i < n; ++i) {
          uint32_t a;
          is >> a;
          t.particleActors.push_back(a);
        }
        is >> t.diceDir;
      } else if (k == "X") {
        std::string n;
        is >> n;
        float v[7];
        for (float& x : v) x = F(is);
        T.back().xform[n] = Tf{Q{v[3], v[4], v[5], v[6]}, V3{v[0], v[1], v[2]}};
      } else if (k == "K") {
        std::string n;
        is >> n;
        std::vector<double> v(3);
        for (double& x : v) {
          std::string tk;
          is >> tk;
          x = strtod(tk.c_str(), nullptr);
        }
        T.back().actorScale[n] = v;
      } else if (k == "C") {
        std::string n;
        is >> n;
        std::vector<float> v(3);
        for (float& x : v) x = F(is);
        T.back().scale[n] = v;
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
    if (const char* so = getenv("PARTICLES_SYNC_ORDER")) {
      if (FILE* q = fopen(so, "r")) {
        char b2[4096];
        while (fgets(b2, sizeof b2, q)) {
          std::istringstream is(b2);
          std::string k, n, kind;
          uint64_t simRemove;
          int sw;
          is >> k >> simRemove >> sw >> n >> kind;
          if (k != "Y") continue;
          for (Trans& t : T)
            if (t.simRemove == simRemove) t.sweepOrder[sw].push_back({n, kind == "static" ? 1 : 0});
        }
        fclose(q);
      }
    }
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
        b.kind = b.root == -1 ? BASE_NONE : (o.kind == 2 ? BASE_KINEMATIC : BASE_RIGID);
        auto xf = t.xform.find(o.name);
        if (xf != t.xform.end()) b.hasXform = 1, b.xform = xf->second;
        for (const std::string& l : o.links) {
          const int32_t h = handle(l);
          if (h != -1) b.links.push_back(h);  // 모양 없는 링크는 몸체 번호 손잡이(-2 이하)
        }
      }
      if (o.name == t.src) removedIdx = i;
      bobjs.push_back(b);
    }
    objs.assign(1 + t.halves.size(), ObjectRt());
    const int32_t sh = handle(t.srcRoot);
    if (sh >= 0) objs[0].actors.push_back(sh);
    edits.clear();
    if (t.dice) {  // 다지기: 규칙 안 입자 생성 (창 밖 ①) — 중심은 기록(harvest_dice), 방향은 기록한 난수 상태에서 T.random_quaternion (dice.h·trng.h 대조 완료)
      Npy cen, rng, off;
      const std::string d = t.diceDir + "/";
      if (npy_load(d + "centers.npy", cen) && npy_load(d + "rng.npy", rng) && npy_load(d + "off.npy", off)) {
        const int n = int(cen.shape[0]);
        TorchMT m;
        torch_mt_from_bytes(rng.as<uint8_t>(), int(rng.count()), m);
        std::vector<float> q(size_t(4 * n));
        random_quaternion(m, n, q.data());
        Edit pa;
        pa.kind = EDIT_PARTICLES_ADD;
        pa.system = 0;
        for (int i = 0; i < n; ++i) pa.poses.push_back(particle_frame_from_center(cen.as<float>() + 3 * i, &q[size_t(4 * i)], off.as<float>()));
        edits.push_back(pa);
        if (systems.empty()) systems.resize(1);
        memcpy(systems[0].off, off.as<float>(), 12);
        for (int i = 0; i < n && size_t(i) < t.particleActors.size(); ++i) particleTmplIdx.push_back(t.particleActors[size_t(i)]), particleTmpl.push_back(t.tmpl);
      } else {
        ++failed;
      }
    }
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
    if (!AA) AA.reset(new scene::EnvArtApi(E, *S));
    BA->art = getenv("PARTICLES_NO_ART") ? nullptr : AA.get();
    B.api = BA.get();
    B.objects = &bobjs;
    W.base = &B;
    W.body = &api;
    W.spawn = &res;
    W.objects = &objs;
    W.systems = &systems;
    W.edits = &edits;
    W.pending.clear();
    W.sweeps = 0;
    syncp.H = this;
    W.sync = getenv("PARTICLES_SYNC_OLD") ? nullptr : &syncp;
    const bool tr = getenv("PARTICLES_HOOK_TRACE") != nullptr;
    if (tr) fprintf(stderr, "[particles 편집] 창 %d 시작: 물체 %zu, 원본 손잡이 %d, 반쪽 %zu\n", ti, bobjs.size(), sh, t.halves.size());
    if (t.dice && t.tmplTemplateActor >= 0) {  // 입자 계 틀 prim (계를 처음 만들 때 강체로 한 번 들어감 — 리드 틀 줄의 첫 행위자)
      SpawnSource src;
      auto it = cache.find(t.tmpl);
      if (it == cache.end()) {
        std::string err;
        if (cache[t.tmpl].load(t.tmpl.c_str(), &err)) it = cache.find(t.tmpl);
      }
      if (it != cache.end()) {
        src.T = &it->second;
        src.actor = uint32_t(t.tmplTemplateActor);
        const auto& A = it->second.shared->actors[src.actor];
        if (getenv("PARTICLES_HOOK_TRACE")) fprintf(stderr, "[particles 편집] 계 틀 prim 행위자 종류 %u\n", A.kind);
        const int32_t h = A.kind == scene::kDynamic ? W.spawnOne(E, src, Pose7{{0, 0, 0}, {0, 0, 0, 1}})
                                                    : spawn_add_static(*E.sc, E.modules(), it->second, src.actor);  // 정적 틀 prim
        if (h >= 0) systemTemplateH = h;
        else ++failed;
      }
    }
    W.runRuleEdits(E);
    if (t.dice && !systems.empty())  // 다음 창 물체 표·벌 차례가 이름으로 찾도록 (입자 prim 이름 = 계 이름 + 전역 번호)
      for (size_t i = 0; i < systems[0].actors.size(); ++i)
        handleOf["/World/scene_0/" + t.system + "/particles/" + t.system + "Particle" + std::to_string(i)] = systems[0].actors[i];
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
    const char* dbgH = getenv("PARTICLES_DEBUG_H");  // 진단: 이 손잡이 자세를 되돌리기·벌 앞뒤로
    Tf dbg0{};
    if (dbgH) dbg0 = W.body->actorPose(atoi(dbgH));
    W.loadState(E);
    if (dbgH) {
      const Tf d1 = W.body->actorPose(atoi(dbgH));
      fprintf(stderr, "[particles 편집] 손잡이 %s: 앞 q %.9g %.9g %.9g %.9g / 뒤 q %.9g %.9g %.9g %.9g\n", dbgH, dbg0.q.x, dbg0.q.y, dbg0.q.z, dbg0.q.w, d1.q.x, d1.q.y,
              d1.q.z, d1.q.w);
      for (auto& kv : handleOf)
        if (kv.second == atoi(dbgH)) {
          auto as = t.actorScale.find(kv.first);
          double sc[3] = {1, 1, 1};
          if (as != t.actorScale.end()) memcpy(sc, as->second.data(), 24);
          const Pose7 raw{{dbg0.p.x, dbg0.p.y, dbg0.p.z}, {dbg0.q.x, dbg0.q.y, dbg0.q.z, dbg0.q.w}};
          const Q g = usd_roundtrip_quat_d(raw, dbg0, sc, false);
          fprintf(stderr, "[particles 편집]   %s 척도 %s %.9g %.9g %.9g → 왕복 q %.9g %.9g %.9g %.9g\n", kv.first.c_str(), as != t.actorScale.end() ? "K" : "없음", sc[0], sc[1],
                  sc[2], g.x, g.y, g.z, g.w);
        }
    }
    if (tr) fprintf(stderr, "[particles 편집] 되돌리기·넣기 끝 (실패 %d)\n", W.failed);
    if (!W.sync) W.flushSync();
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

void SyncFromPlan::order(int sweep, std::vector<TransitionEditWindow::SyncActor>& out) {
  Trans& t = H->T[size_t(H->cur)];
  auto scaleOf = [&](const std::string& obj, double s[3]) {
    auto it = t.scale.find(obj);
    for (int k = 0; k < 3; ++k) s[k] = it != t.scale.end() ? it->second[size_t(k)] : 1.0f;
  };
  // 새 반쪽 (이미 넣은 것만): 이름 -> 손잡이·척도
  std::map<std::string, std::pair<int32_t, const Half*>> fresh;
  for (size_t i = 0; i < t.halves.size(); ++i)
    if (!H->objs[1 + i].actors.empty()) fresh["/World/scene_0/" + t.halves[i].name + "/base_link"] = {H->objs[1 + i].actors[0], &t.halves[i]};
  std::map<std::string, std::string> objOfActor;
  for (const ObjLine& o : t.objs) {
    objOfActor[o.root] = o.name;
    for (const std::string& l : o.links) objOfActor[l] = o.name;
  }
  auto push = [&](const std::string& actor, int isStatic) {
    TransitionEditWindow::SyncActor a;
    a.isStatic = uint8_t(isStatic);
    auto f = fresh.find(actor);
    if (f != fresh.end()) {
      a.h = f->second.first;
      for (int k = 0; k < 3; ++k) a.scale[k] = f->second.second->usd[k];
    } else {
      auto it = H->handleOf.find(actor);
      if (it == H->handleOf.end()) {
        if (H->syncMissing++ < 12 && getenv("PARTICLES_HOOK_TRACE")) fprintf(stderr, "[particles 편집] 벌 행위자 이름 못 찾음: %s\n", actor.c_str());
        return;
      }
      a.h = it->second;
      scaleOf(objOfActor.count(actor) ? objOfActor[actor] : std::string(), a.scale);
      auto as = t.actorScale.find(actor);
      if (as != t.actorScale.end()) memcpy(a.scale, as->second.data(), 24);
    }
    out.push_back(a);
  };
  auto so = t.sweepOrder.find(sweep);
  if (so != t.sweepOrder.end()) {
    for (auto& e : so->second) push(e.first, e.second);
    if (getenv("PARTICLES_HOOK_TRACE")) {
      size_t st = 0, neg = 0;
      for (auto& a : out) st += a.isStatic, neg += a.h < 0;
      fprintf(stderr, "[particles 편집] 벌 %d: 행위자 %zu (정적 %zu, 몸체 손잡이 %zu), 공식 목록 %zu\n", sweep, out.size(), st, neg, so->second.size());
    }
    return;
  }
  for (const ObjLine& o : t.objs)
    if (o.kind == 2 && o.name != t.src) push(o.root, 1);
  for (const ObjLine& o : t.objs)
    if (o.kind == 2 && o.name != t.src)
      for (const std::string& l : o.links)
        if (l != o.root) push(l, 0);
  for (const ObjLine& o : t.objs)
    if (o.kind == 0 && o.name != t.src) push(o.root, 0);
  for (auto& kv : fresh) push(kv.first, 0);
  if (!H->systems.empty()) {  // 입자 (공식 onion: 벌 끝쪽)
    const auto& acts = H->systems[0].actors;
    for (size_t i = 0; i < acts.size(); ++i) {
      TransitionEditWindow::SyncActor a;
      a.h = acts[i];
      a.isStatic = 0;
      a.scale[0] = a.scale[1] = a.scale[2] = 1.0;
      if (i < H->particleTmplIdx.size()) {
        auto it = H->cache.find(H->particleTmpl[i]);
        if (it != H->cache.end()) {
          const auto& A = it->second.shared->actors[H->particleTmplIdx[i]];
          const auto& g = it->second.shared->shapes[A.shapeStart].geom;
          a.scale[0] = g.convex.scale.scale.x, a.scale[1] = g.convex.scale.scale.y, a.scale[2] = g.convex.scale.scale.z;
        }
      }
      out.push_back(a);
    }
  }
}
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
    if (G.on) {
      G.S = &S;
      std::map<uint64_t, int32_t> hOfNode;
      for (size_t h = 0; h < E.sc->actors.size(); ++h)
        if (E.sc->actors[h].alive && E.sc->actors[h].node != ~0ull) hOfNode[E.sc->actors[h].node & 0xffffffffu] = int32_t(h);
      std::map<int32_t, uint32_t> nodeOfBody;
      for (uint32_t id = 0; id < S.bodyOfNode.size(); ++id)
        if (S.bodyOfNode[id] >= 0) nodeOfBody[S.bodyOfNode[id]] = id;
      size_t more = 0;
      for (const scene::SceneActor& a : G.f.actors) {
        if (a.kind != scene::kDynamic) continue;
        auto nb = nodeOfBody.find(int32_t(a.body));
        if (nb == nodeOfBody.end()) continue;
        auto hh = hOfNode.find(nb->second);
        if (hh != hOfNode.end()) more += G.handleOf.emplace(G.f.name(a.name), hh->second).second;
      }
      size_t shapeless = 0;  // 모양 없는 동적 행위자 (meta 링크 등): 리드 EnvBodyApi::bodyHandle (3aec382)
      for (const scene::SceneActor& a : G.f.actors)
        if (a.kind == scene::kDynamic) shapeless += G.handleOf.emplace(G.f.name(a.name), api.bodyHandle(a.body)).second;
      fprintf(stderr, "[particles 편집] 몸체 노드로 더 찾은 동적 행위자 %zu, 몸체 손잡이로 %zu\n", more, shapeless);
      if (const char* dn = getenv("PARTICLES_NODE_NAMES")) {  // 진단: 섬 노드 번호 -> 이름
        std::map<int32_t, std::string> nameOfH;
        for (auto& kv : G.handleOf) nameOfH[kv.second] = kv.first;
        std::stringstream ss(dn);
        std::string t;
        while (std::getline(ss, t, ',')) {
          const uint32_t id = uint32_t(atoll(t.c_str()));
          auto it = hOfNode.find(id);
          std::string byBody = "?";
          if (id < S.bodyOfNode.size() && S.bodyOfNode[id] >= 0)
            for (const scene::SceneActor& a : G.f.actors)
              if (a.kind == scene::kDynamic && int32_t(a.body) == S.bodyOfNode[id]) byBody = G.f.name(a.name);
          fprintf(stderr, "[particles 편집] 노드 %s = %s (몸체로 %s, 손잡이 %d)\n", t.c_str(), it != hOfNode.end() && nameOfH.count(it->second) ? nameOfH[it->second].c_str() : "?",
                  byBody.c_str(), G.handleOf.count(byBody) ? G.handleOf[byBody] : -99999);
        }
      }
    }
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
