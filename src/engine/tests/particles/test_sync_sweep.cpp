// 층 1 시험 (편집 창 동기화 벌): 공식 K+1 창에서 omni.physx 가 물체 넣기마다 모든 강체에 다시 쓰는 자세(USD 왕복)·속도 0 의
// 차례와 값을 우리 식(물체 등록부 차례, spawn.h usd_roundtrip 계열 — 척도 = 물체 척도, pxr·double)이 내는지.
//   python3 sync_sweep_to_txt.py <기록> <새 물체> <txt>;  ./test_sync_sweep <txt>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/omni/gfmat.h"
#include "core/particles/spawn.h"

using namespace eng;
using namespace eng::particles;
namespace gf = eng::omni::gf;

// USD 왕복 (pxr RemoveScaleShear → ExtractRotationQuat → float) 후 PhysX getNormalized
static Q sweepQuat(const float q[4], const float p[3], const float s[3]) {
  gf::M4 M = gf::from_physx_pose(p, q);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) M.m[r][c] *= (double)s[r];
  double o[4];
  gf::extract_rotation_quat(gf::remove_scale_shear(M), o);
  return normalized(Q{(float)o[0], (float)o[1], (float)o[2], (float)o[3]});
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  FILE* f = fopen(argv[1], "r");
  if (!f) return 1;
  struct Obj {
    std::string name, root;
    float s[3];
    std::vector<std::string> links;
  };
  std::vector<Obj> objs;
  std::map<std::string, std::vector<float>> input;                 // 벌 앞 자세 (q4 p3)
  std::map<int, std::vector<std::pair<std::string, std::vector<float>>>> sweeps;  // 벌 -> (행위자, q4p3)
  std::map<int, std::vector<std::pair<std::string, std::vector<float>>>> vels;
  char buf[1 << 16];
  while (fgets(buf, sizeof buf, f)) {
    std::istringstream is(buf);
    std::string k;
    is >> k;
    auto rd = [&](int n) {
      std::vector<float> v((size_t)n);
      for (float& x : v) {
        std::string t;
        is >> t;
        x = strtof(t.c_str(), nullptr);
      }
      return v;
    };
    if (k == "O") {
      Obj o;
      unsigned nl;
      is >> o.name >> o.root;
      for (float& x : o.s) {
        std::string t;
        is >> t;
        x = strtof(t.c_str(), nullptr);
      }
      is >> nl;
      for (unsigned i = 0; i < nl; ++i) {
        std::string l;
        is >> l;
        o.links.push_back(l);
      }
      objs.push_back(o);
    } else if (k == "L") {
      std::string n;
      is >> n;
      input[n] = rd(7);
    } else if (k == "S") {
      int s;
      std::string n, kind;
      is >> s >> n >> kind;
      sweeps[s].push_back({n, rd(7)});
    } else if (k == "V") {
      int s;
      std::string n;
      is >> s >> n;
      vels[s].push_back({n, rd(6)});
    }
  }
  fclose(f);
  std::map<std::string, const Obj*> objOfActor;
  for (const Obj& o : objs) {
    objOfActor[o.root] = &o;
    for (const std::string& l : o.links) objOfActor[l] = &o;
  }
  // 차례: 물체 등록부 차례 → 물체 안은 뿌리 먼저, 그다음 동적 링크 (등록부에 없는 새 물체는 끝, 넣은 차례)
  long badOrder = 0, badPose = 0, cmpPose = 0, badVel = 0, nV = 0;
  std::map<std::string, std::vector<float>> cur = input;
  for (auto& kv : sweeps) {
    const int s = kv.first;
    std::vector<std::string> want;
    for (auto& e : kv.second) want.push_back(e.first);
    std::vector<std::string> ours;
    for (const Obj& o : objs) {
      std::vector<std::string> acts{o.root};
      for (const std::string& l : o.links)
        if (l != o.root) acts.push_back(l);
      for (const std::string& a : acts)
        for (const std::string& w : want)
          if (w == a) {
            ours.push_back(a);
            break;
          }
    }
    for (const std::string& w : want)
      if (!objOfActor.count(w)) ours.push_back(w);  // 새 물체
    size_t first = 0;
    while (first < want.size() && first < ours.size() && want[first] == ours[first]) ++first;
    if (first != want.size() || ours.size() != want.size()) {
      ++badOrder;
      printf("  벌 %d 차례 다름: %zu 번째 정답 %s / 우리 %s (정답 %zu 개, 우리 %zu 개)\n", s, first, first < want.size() ? want[first].c_str() : "-",
             first < ours.size() ? ours[first].c_str() : "-", want.size(), ours.size());
    }
    for (auto& e : kv.second) {
      auto in = cur.find(e.first);
      auto ob = objOfActor.find(e.first);
      if (in != cur.end()) {
        const float one[3] = {1, 1, 1};
        const float* sc = ob != objOfActor.end() ? ob->second->s : one;
        const Q g = sweepQuat(in->second.data(), in->second.data() + 4, sc);
        const bool okq = memcmp(&g, e.second.data(), 16) == 0, okp = memcmp(in->second.data() + 4, e.second.data() + 4, 12) == 0;
        ++cmpPose;
        if (!(okq && okp)) {
          if (badPose < 8)
            printf("  벌 %d %s 자세 다름: 우리 q %.9g %.9g %.9g %.9g / 정답 %.9g %.9g %.9g %.9g (위치 %s, 척도 %g %g %g)\n", s, e.first.c_str(), g.x, g.y, g.z, g.w,
                   e.second[0], e.second[1], e.second[2], e.second[3], okp ? "같음" : "다름", sc[0], sc[1], sc[2]);
          ++badPose;
        }
      }
      cur[e.first] = e.second;  // 다음 벌 입력 = 이번 벌 결과
    }
    for (auto& v : vels[s]) {
      ++nV;
      for (float x : v.second) badVel += x != 0.0f;
    }
  }
  printf("동기화 벌: %zu 벌, 차례 다른 벌 %ld, 자세 비교 %ld 다름 %ld, 속도 쓰기 %ld (0 아닌 칸 %ld)\n", sweeps.size(), badOrder, cmpPose, badPose, nV, badVel);
  const bool ok = !badOrder && !badPose && !badVel && cmpPose > 0;
  printf(ok ? "비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
