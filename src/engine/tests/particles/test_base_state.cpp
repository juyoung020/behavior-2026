// 층 1 시험 (base 창): base_state.h BaseStateWindow 가 공식 removing_objects 의 상태 뜨기·되돌리기와 같은 호출 열을 내는가.
// 정답 = 공식 평가기에서 가로챈 호출 열(harvest_spawn.py HARVEST_BASE=1 → base_events_to_txt.py). 가짜 텐서 뷰 층이 뜬 값(D 줄)과
// 되돌리기 때 읽은 지금 자세(정답의 read, 차례대로)를 돌려주고, 쓰기·잠·깨움 호출을 같은 꼴 글로 적어 비트 비교한다.
//   ./test_base_state <base.txt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/particles/base_state.h"

using namespace eng;
using namespace eng::particles;

static std::string hx(float v) {
  uint32_t u;
  memcpy(&u, &v, 4);
  char b[16];
  snprintf(b, sizeof b, "%08x", u);
  return b;
}

struct Win {
  std::vector<std::string> names, roots;
  std::vector<BaseObject> objs;
  std::vector<std::string> linkPath;  // 링크 손잡이 -> 경로
  std::map<std::string, BaseObjectState> dump;
  std::vector<std::string> removed;
  std::vector<std::string> want;  // 정답 호출 (정규형)
  std::map<std::string, std::deque<Tf>> reads;  // 경로 -> 되돌리기 때 읽은 자세 (차례)
};

struct Fake : BaseStateApi {
  Win* W = nullptr;
  bool loading = false;
  std::vector<std::string> got;
  uint64_t missingRead = 0;
  const std::string& nameOf(const BaseObject& o) { return W->names[size_t(&o - W->objs.data())]; }
  const std::string& rootOf(const BaseObject& o) { return W->roots[size_t(&o - W->objs.data())]; }
  std::string objPath(const BaseObject& o) {
    const std::string& r = rootOf(o);
    return r.substr(0, r.rfind('/'));
  }
  const BaseObjectState& D(const BaseObject& o) { return W->dump[nameOf(o)]; }
  static std::string tfs(const Tf& t) {
    return hx(t.p.x) + " " + hx(t.p.y) + " " + hx(t.p.z) + " " + hx(t.q.x) + " " + hx(t.q.y) + " " + hx(t.q.z) + " " + hx(t.q.w);
  }
  static std::string v3(const V3& v) { return hx(v.x) + " " + hx(v.y) + " " + hx(v.z); }
  static std::string vec(const std::vector<float>& v) {
    std::string s = std::to_string(v.size());
    for (float x : v) s += " " + hx(x);
    return s;
  }
  bool rigidSleeping(int32_t h) override { return D(W->objs[size_t(h)]).asleep; }
  bool artSleeping(int32_t a) override { return D(W->objs[size_t(a)]).asleep; }
  Tf rootPose(const BaseObject& o) override {
    if (!loading) return D(o).pose;
    auto& q = W->reads[rootOf(o)];
    if (q.empty()) {  // 정답이 뿌리 링크(RigidDynamicPrim) 읽기를 안 남긴 경우 — 관절체는 관절체 뷰로 읽음
      ++missingRead;
      return D(o).pose;
    }
    const Tf t = q.front();
    q.pop_front();
    got.push_back("read " + rootOf(o) + " " + tfs(t));
    return t;
  }
  void rootVelocity(const BaseObject& o, V3& l, V3& a) override { l = D(o).lin, a = D(o).ang; }
  void jointState(const BaseObject& o, std::vector<float>& p, std::vector<float>& v) override { p = D(o).jpos, v = D(o).jvel; }
  void setRootPose(const BaseObject& o, const Tf& t) override {  // 운동학 전용 뿌리(RigidKinematicPrim)는 XForm 쓰기 — 정답 가로채기 밖
    if (o.kind != BASE_KINEMATIC) got.push_back("set_position_orientation " + rootOf(o) + " " + tfs(t));
  }
  void setRootLinVel(const BaseObject& o, const V3& v) override { got.push_back("set_linear_velocity " + rootOf(o) + " " + v3(v)); }
  void setRootAngVel(const BaseObject& o, const V3& v) override { got.push_back("set_angular_velocity " + rootOf(o) + " " + v3(v)); }
  void setJointPositions(const BaseObject& o, const std::vector<float>& v) override { got.push_back("set_joint_positions " + objPath(o) + " " + vec(v)); }
  void setJointVelocities(const BaseObject& o, const std::vector<float>& v) override { got.push_back("set_joint_velocities " + objPath(o) + " " + vec(v)); }
  void rigidSleep(int32_t h) override { got.push_back("rigid_sleep " + W->linkPath[size_t(h)]); }
  void rigidWake(int32_t h) override { got.push_back("rigid_wake " + W->linkPath[size_t(h)]); }
  void artSleep(int32_t a) override { got.push_back("art_sleep " + W->names[size_t(a)]); }
  void artWake(int32_t a) override { got.push_back("art_wake " + W->names[size_t(a)]); }
  void loadExtra(size_t i) override { (void)i; }
};

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  FILE* f = fopen(argv[1], "r");
  if (!f) return 1;
  std::vector<Win> wins;
  char buf[1 << 16];
  auto F = [](std::istringstream& is) {
    std::string t;
    is >> t;
    return strtof(t.c_str(), nullptr);
  };
  while (fgets(buf, sizeof buf, f)) {
    std::istringstream is(buf);
    std::string k;
    is >> k;
    if (k == "W") {
      wins.emplace_back();
    } else if (k == "O") {
      Win& W = wins.back();
      std::string name, root;
      int kind;
      unsigned nj, nl;
      is >> name >> kind >> nj >> root >> nl;
      BaseObject o;
      o.kind = uint8_t(kind);
      o.nJoints = nj;
      o.root = o.art = int32_t(W.objs.size());
      for (unsigned i = 0; i < nl; ++i) {
        std::string lp;
        is >> lp;
        o.links.push_back(int32_t(W.linkPath.size()));
        W.linkPath.push_back(lp);
      }
      W.names.push_back(name);
      W.roots.push_back(root);
      W.objs.push_back(o);
    } else if (k == "R") {
      std::string n;
      is >> n;
      wins.back().removed.push_back(n);
    } else if (k == "D") {
      std::string name;
      int asleep;
      is >> name >> asleep;
      BaseObjectState s;
      s.asleep = uint8_t(asleep);
      float v[13];
      for (float& x : v) x = F(is);
      s.pose = Tf{Q{v[3], v[4], v[5], v[6]}, V3{v[0], v[1], v[2]}};
      s.lin = V3{v[7], v[8], v[9]}, s.ang = V3{v[10], v[11], v[12]};
      unsigned n;
      is >> n;
      for (unsigned i = 0; i < n; ++i) s.jpos.push_back(F(is));
      is >> n;
      for (unsigned i = 0; i < n; ++i) s.jvel.push_back(F(is));
      wins.back().dump[name] = s;
    } else if (k == "C") {
      Win& W = wins.back();
      std::string c, who;
      is >> c >> who;
      if (c == "obj") continue;  // 물체 경계 표시 (호출 아님)
      std::string line = c + " " + who;
      if (c == "read" || c == "set_position_orientation") {
        float v[7];
        for (float& x : v) x = F(is);
        const Tf t{Q{v[3], v[4], v[5], v[6]}, V3{v[0], v[1], v[2]}};
        line += " " + Fake::tfs(t);
        if (c == "read") W.reads[who].push_back(t);
      } else if (c == "set_linear_velocity" || c == "set_angular_velocity") {
        V3 v{F(is), F(is), F(is)};
        line += " " + Fake::v3(v);
      } else if (c == "set_joint_positions" || c == "set_joint_velocities") {
        unsigned n;
        is >> n;
        std::vector<float> v;
        for (unsigned i = 0; i < n; ++i) v.push_back(F(is));
        line += " " + Fake::vec(v);
      }
      W.want.push_back(line);
    }
  }
  fclose(f);
  long bad = 0, calls = 0, objsN = 0;
  for (size_t w = 0; w < wins.size(); ++w) {
    Win& W = wins[w];
    Fake api;
    api.W = &W;
    BaseStateWindow B;
    B.api = &api;
    B.objects = &W.objs;
    scene::EnvStep E;
    B.dumpState(E);
    for (size_t i = 0; i < W.objs.size(); ++i)
      for (const std::string& r : W.removed)
        if (W.names[i] == r) B.markRemoved(i);
    api.loading = true;
    B.loadState(E);
    size_t first = 0;
    while (first < W.want.size() && first < api.got.size() && W.want[first] == api.got[first]) ++first;
    const bool ok = W.want.size() == api.got.size() && first == W.want.size();
    printf("  창 %zu: 물체 %zu (지움 %zu) 정답 호출 %zu / 우리 %zu, 읽기 모자람 %llu — %s\n", w, W.objs.size(), W.removed.size(), W.want.size(), api.got.size(),
           (unsigned long long)api.missingRead, ok ? "같음" : "다름");
    if (!ok) {
      printf("    첫 다름 %zu:\n      정답 %s\n      우리 %s\n", first, first < W.want.size() ? W.want[first].c_str() : "(끝)",
             first < api.got.size() ? api.got[first].c_str() : "(끝)");
      ++bad;
    }
    calls += long(W.want.size());
    objsN += long(W.objs.size());
  }
  printf("base 창 되돌리기 호출 열: 창 %zu, 물체 %ld, 호출 %ld, 다른 창 %ld\n", wins.size(), objsN, calls, bad);
  const bool ok = !wins.empty() && !bad;
  printf(ok ? "비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
