// 두 OVD 를 명령 단위로 나란히 비교한다. 객체 핸들(원래 프로세스의 포인터)은 실행마다 다르므로
// "만들어진 순서"로 짝을 짓는다. 원본 기록과 ovd_replay --record 로 뜬 재생 기록을 비교하면
// 재생이 빠뜨리거나 다르게 넣은 API 호출(입력)과, 처음 달라진 결과(출력)를 정확히 짚는다.
//   ovd_diff <a.ovd> <b.ovd> [--max N] [--inputs-only]
#include <cinttypes>
#include <functional>
#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "OmniPvdDefines.h"
#include "ovd.h"

using namespace ovd;

struct Walker {
  File& f;
  std::unordered_map<uint64_t, uint64_t> ord;  // 핸들 -> 만든 순번
  uint64_t next = 1;
  size_t i = 0;
  uint64_t sims = 0;
  bool out_block = false;
  uint32_t a_elapsed = 0;
  explicit Walker(File& ff) : f(ff) {
    for (auto& kv : f.attrs) if (kv.second.name == "elapsedTime") a_elapsed = kv.first;
  }
  // 의미 있는 다음 명령 (등록·프레임 명령은 건너뛴다)
  const Event* step() {
    while (i < f.events.size()) {
      const Event& e = f.events[i++];
      if (e.cmd == kCreate) ord[e.obj] = next++;
      if (e.cmd == kStopFrame) { out_block = false; continue; }
      if (e.cmd == kSet && e.attr == a_elapsed) { sims++; out_block = true; }
      if (e.cmd == kCreate || e.cmd == kDestroy || e.cmd == kSet || e.cmd == kAddToList || e.cmd == kRemoveFromList) return &e;
    }
    return nullptr;
  }
  uint64_t o(uint64_t h) const { auto it = ord.find(h); return it == ord.end() ? 0 : it->second; }
  std::string attr(const Event& e) const { return e.cmd == kCreate ? f.classes.at(e.cls).name : f.attr_name(e.attr); }
};

static std::string hex(const uint8_t* p, uint32_t n, const AttrInfo* a) {
  std::string s;
  char b[48];
  if (a && a->type == OmniPvdDataType::eFLOAT32) {
    for (uint32_t k = 0; k + 4 <= n && k < 64; k += 4) { float v; memcpy(&v, p + k, 4); snprintf(b, sizeof b, "%s%.9g", k ? " " : "", v); s += b; }
    return s;
  }
  for (uint32_t k = 0; k < n && k < 32; ++k) { snprintf(b, sizeof b, "%02x", p[k]); s += b; }
  return s;
}

// --state: 첫 simulate 직전의 "객체별 마지막 속성 값"을 비교한다 (명령 순서가 조금 달라도 상태가 같으면 같음).
//   객체 짝 = 만든 순번 (--skip-class 로 준 클래스는 순번에서 뺀다: 재생기가 안 만드는 것, 예: 변형체·PBD 재질).
struct StateSnap {
  std::vector<std::string> cls;                                  // 순번 -> 클래스
  std::unordered_map<uint64_t, std::string> names;               // 순번 -> 이름(PxActor.name 등)
  std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>> val;  // 순번 -> 속성 -> 값(핸들은 순번으로)
};
static std::string cstr(const uint8_t* p, uint32_t n) { while (n && p[n - 1] == 0) --n; return std::string(reinterpret_cast<const char*>(p), n); }
// 정체(identity) 짝짓기: 만든 순서 대신 "무엇인가"로 짝을 짓는다.
//   이름 있는 것(액터·관절체) = 클래스+이름, 모양 = 붙은 액터 이름+몇 번째, 관절체 조인트 = 자식 링크 이름,
//   형상 객체 = 그것을 쓰는 모양, 메시·재질 = 만든 순간 값의 해시. 핸들 값도 정체로 바꿔 비교한다.
struct IdSnap {
  std::map<std::string, std::map<std::string, std::string>> val;  // 정체 -> 속성 -> 값(문자열, 핸들은 정체)
};
static uint64_t fnv(const uint8_t* p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }
static IdSnap id_snap_first_sim(File& f, const std::vector<std::string>& skip) {
  uint32_t a_elapsed = 0;
  for (auto& kv : f.attrs) if (kv.second.name == "elapsedTime") a_elapsed = kv.first;
  size_t end = f.events.size();
  for (size_t i = 0; i < f.events.size(); ++i) if (f.events[i].cmd == kSet && f.events[i].attr == a_elapsed) { end = i; break; }
  // 1) 객체별 클래스·마지막 값 (핸들 그대로), 생성 순간 값 해시, 붙이기 관계
  std::unordered_map<uint64_t, std::string> cls;
  std::unordered_map<uint64_t, std::map<std::string, std::vector<uint8_t>>> last;
  std::unordered_map<uint64_t, uint64_t> create_hash;
  std::unordered_map<uint64_t, std::pair<uint64_t, int>> shape_owner;  // 모양 -> (액터, 몇 번째)
  std::unordered_map<uint64_t, int> nshapes;
  std::unordered_map<uint64_t, bool> skipped;
  uint64_t creating = 0;
  for (size_t i = 0; i < end; ++i) {
    const Event& e = f.events[i];
    if (e.cmd == kCreate) {
      const std::string& c = f.classes.at(e.cls).name;
      bool sk = false;
      for (auto& x : skip) sk |= (c == x);
      skipped[e.obj] = sk;
      cls[e.obj] = c;
      last[e.obj].clear();
      create_hash[e.obj] = fnv(reinterpret_cast<const uint8_t*>(c.data()), c.size());
      creating = e.obj;
      continue;
    }
    if (e.cmd == kAddToList && f.attrs[e.attr].name == "shapes" && e.data_len >= 8) {
      uint64_t s;
      memcpy(&s, f.data(e), 8);
      if (cls[e.obj] != "PxPhysics") shape_owner[s] = {e.obj, nshapes[e.obj]++};
      continue;
    }
    if (e.cmd != kSet) { creating = 0; continue; }
    const AttrInfo& ai = f.attrs[e.attr];
    std::vector<uint8_t> d(f.data(e), f.data(e) + e.data_len);
    if (e.obj == creating && ai.type != OmniPvdDataType::eOBJECT_HANDLE) {  // 생성 순간 값으로 내용 해시 (메시 꼭짓점 등)
      uint64_t h = create_hash[e.obj];
      h ^= fnv(d.data(), d.size()) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      create_hash[e.obj] = h;
    }
    last[e.obj][f.classes.at(ai.cls).name + "." + ai.name] = d;
  }
  // 2) 정체 풀기 (재귀: 조인트 -> 자식 링크 이름, 형상 -> 모양 -> 액터)
  std::unordered_map<uint64_t, std::string> idc;
  std::unordered_map<uint64_t, uint64_t> geom_user;  // 형상 객체 -> 모양
  for (auto& kv : last) {
    auto it = kv.second.find("PxShape.geom");
    if (it != kv.second.end() && it->second.size() >= 8) { uint64_t g; memcpy(&g, it->second.data(), 8); geom_user[g] = kv.first; }
  }
  std::function<std::string(uint64_t, int)> ident = [&](uint64_t h, int depth) -> std::string {
    if (!h) return "0";
    auto ic = idc.find(h);
    if (ic != idc.end()) return ic->second;
    if (depth > 6 || !cls.count(h)) return "?";
    const std::string& c = cls[h];
    auto& L = last[h];
    std::string id;
    for (auto& kv : L) if (kv.first.size() > 5 && kv.first.compare(kv.first.size() - 5, 5, ".name") == 0) { id = c + ":" + cstr(kv.second.data(), uint32_t(kv.second.size())); break; }
    if (id.empty() && c == "PxArticulationJointReducedCoordinate") {
      auto it = L.find("PxArticulationJointReducedCoordinate.childLink");
      if (it != L.end() && it->second.size() >= 8) { uint64_t ch; memcpy(&ch, it->second.data(), 8); id = "joint->" + ident(ch, depth + 1); }
    }
    if (id.empty() && shape_owner.count(h)) id = "shape@" + ident(shape_owner[h].first, depth + 1) + "#" + std::to_string(shape_owner[h].second);
    if (id.empty() && geom_user.count(h)) id = c + "@" + ident(geom_user[h], depth + 1);
    if (id.empty()) { char b[40]; snprintf(b, sizeof b, "%016llx", (unsigned long long)create_hash[h]); id = c + "#" + b; }
    idc[h] = id;
    return id;
  };
  IdSnap s;
  for (auto& kv : last) {
    if (skipped[kv.first]) continue;
    const std::string id = ident(kv.first, 0);
    auto& dst = s.val[id];
    for (auto& av : kv.second) {
      const AttrInfo* ai = nullptr;
      for (auto& x : f.attrs) if (f.classes.at(x.second.cls).name + "." + x.second.name == av.first) { ai = &x.second; break; }
      std::string v;
      if (ai && ai->type == OmniPvdDataType::eOBJECT_HANDLE) {
        for (size_t k = 0; k + 8 <= av.second.size(); k += 8) { uint64_t h; memcpy(&h, &av.second[k], 8); v += (k ? "," : "") + ident(h, 0); }
      } else {
        v = hex(av.second.data(), uint32_t(av.second.size()), ai);
      }
      dst[av.first] = v;
    }
  }
  return s;
}
static int id_state_diff(File& A, File& B, const std::vector<std::string>& skip, int maxd) {
  IdSnap a = id_snap_first_sim(A, skip), b = id_snap_first_sim(B, skip);
  printf("첫 simulate 직전 정체 수: A %zu, B %zu\n", a.val.size(), b.val.size());
  int nd = 0, only_a = 0, only_b = 0;
  for (auto& kv : a.val) {
    auto it = b.val.find(kv.first);
    if (it == b.val.end()) { if (only_a++ < 8) printf("[A 에만] %s\n", kv.first.c_str()); continue; }
    for (auto& av : kv.second) {
      auto jt = it->second.find(av.first);
      if (jt != it->second.end() && jt->second == av.second) continue;
      if (nd++ < maxd) printf("[상태 차이] %s  %s\n  A: %s\n  B: %s\n", kv.first.c_str(), av.first.c_str(), av.second.c_str(), jt == it->second.end() ? "(없음)" : jt->second.c_str());
    }
  }
  for (auto& kv : b.val) if (!a.val.count(kv.first)) { if (only_b++ < 8) printf("[B 에만] %s\n", kv.first.c_str()); }
  printf("상태 차이 %d 건, A 에만 %d, B 에만 %d\n", nd, only_a, only_b);
  return nd ? 3 : 0;
}

static StateSnap snap_first_sim(File& f, const std::vector<std::string>& skip) {
  StateSnap s;
  std::unordered_map<uint64_t, uint64_t> ord;
  std::unordered_map<uint64_t, bool> skipped;
  uint32_t a_elapsed = 0;
  for (auto& kv : f.attrs) if (kv.second.name == "elapsedTime") a_elapsed = kv.first;
  s.cls.push_back("");
  for (const Event& e : f.events) {
    if (e.cmd == kSet && e.attr == a_elapsed) break;
    if (e.cmd == kCreate) {
      const std::string& c = f.classes.at(e.cls).name;
      bool sk = false;
      for (auto& x : skip) sk |= (c == x);
      skipped[e.obj] = sk;
      if (sk) continue;
      ord[e.obj] = s.cls.size();
      s.cls.push_back(c);
      continue;
    }
    if (e.cmd != kSet) continue;
    auto io = ord.find(e.obj);
    if (io == ord.end() || skipped[e.obj]) continue;
    const AttrInfo& ai = f.attrs[e.attr];
    std::vector<uint8_t> d(f.data(e), f.data(e) + e.data_len);
    if (ai.type == OmniPvdDataType::eOBJECT_HANDLE) {
      for (size_t k = 0; k + 8 <= d.size(); k += 8) { uint64_t h; memcpy(&h, &d[k], 8); auto it = ord.find(h); uint64_t o = it == ord.end() ? 0 : it->second; memcpy(&d[k], &o, 8); }
    }
    if (ai.name == "name") s.names[io->second] = cstr(d.data(), uint32_t(d.size()));
    s.val[io->second][f.classes.at(ai.cls).name + "." + ai.name] = d;
  }
  return s;
}
static int state_diff(File& A, File& B, const std::vector<std::string>& skip, int maxd) {
  StateSnap a = snap_first_sim(A, skip), b = snap_first_sim(B, skip);
  printf("첫 simulate 직전 객체 수: A %zu, B %zu\n", a.cls.size() - 1, b.cls.size() - 1);
  int nd = 0;
  const size_t n = std::min(a.cls.size(), b.cls.size());
  for (size_t o = 1; o < n && nd < maxd; ++o) {
    if (a.cls[o] != b.cls[o]) { printf("[짝 어긋남] 순번 %zu: A %s / B %s — 이후 비교 중단\n", o, a.cls[o].c_str(), b.cls[o].c_str()); return 3; }
    auto& va = a.val[o];
    auto& vb = b.val[o];
    for (auto& kv : va) {
      auto it = vb.find(kv.first);
      const bool same = it != vb.end() && it->second == kv.second;
      if (same) continue;
      const std::string nm = a.names.count(o) ? a.names[o] : "";
      const AttrInfo* ai = nullptr;
      for (auto& x : A.attrs) if (A.classes.at(x.second.cls).name + "." + x.second.name == kv.first) { ai = &x.second; break; }
      printf("[상태 차이 %d] #%zu %s %s %s\n  A: %s\n  B: %s\n", ++nd, o, a.cls[o].c_str(), nm.c_str(), kv.first.c_str(),
             hex(kv.second.data(), uint32_t(kv.second.size()), ai).c_str(),
             it == vb.end() ? "(없음)" : hex(it->second.data(), uint32_t(it->second.size()), ai).c_str());
      if (nd >= maxd) break;
    }
  }
  printf("상태 차이 %d 건 (최대 %d 까지 봄)\n", nd, maxd);
  return nd ? 3 : 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ovd_diff <a.ovd> <b.ovd> [--max N] [--inputs-only] [--state [--skip-class C1,C2]]\n"); return 2; }
  File A, B;
  std::string err;
  if (!load(argv[1], A, err) || !load(argv[2], B, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  int maxd = 10;
  bool inputs_only = false, state = false;
  std::vector<std::string> skip;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--max") && i + 1 < argc) maxd = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--inputs-only")) inputs_only = true;
    else if (!strcmp(argv[i], "--state")) state = true;
    else if (!strcmp(argv[i], "--skip-class") && i + 1 < argc) {
      std::string l = argv[++i];
      size_t p = 0;
      while (p <= l.size()) { size_t c = l.find(',', p); skip.push_back(l.substr(p, c == std::string::npos ? std::string::npos : c - p)); if (c == std::string::npos) break; p = c + 1; }
    }
  }
  if (state) return id_state_diff(A, B, skip, maxd);
  if (false) return state_diff(A, B, skip, maxd);
  Walker wa(A), wb(B);
  int nd = 0;
  uint64_t same = 0;
  for (;;) {
    const Event* a = wa.step();
    const Event* b = wb.step();
    if (!a || !b) { printf("끝: A %s, B %s (같은 명령 %" PRIu64 ")\n", a ? "남음" : "끝", b ? "남음" : "끝", same); break; }
    if (inputs_only && wa.out_block && wb.out_block) { same++; continue; }
    bool eq = a->cmd == b->cmd && wa.attr(*a) == wb.attr(*b) && wa.o(a->obj) == wb.o(b->obj);
    // 데이터: 핸들 값은 순번으로 바꿔 비교
    if (eq && a->cmd != kCreate && a->cmd != kDestroy) {
      const AttrInfo* ai = &A.attrs[a->attr];
      if (ai->type == OmniPvdDataType::eOBJECT_HANDLE || ai->is_list) {
        eq = a->data_len == b->data_len;
        for (uint32_t k = 0; eq && k + 8 <= a->data_len; k += 8) {
          uint64_t x, y;
          memcpy(&x, A.data(*a) + k, 8); memcpy(&y, B.data(*b) + k, 8);
          eq = wa.o(x) == wb.o(y);
        }
      } else {
        eq = a->data_len == b->data_len && !memcmp(A.data(*a), B.data(*b), a->data_len);
      }
    }
    if (eq) { same++; continue; }
    nd++;
    const char* cn[] = {"?", "regClass", "regEnum", "regAttr", "regClassAttr", "regList", "set", "add", "remove", "create", "destroy", "start", "stop", "msg"};
    printf("[차이 %d] sim A=%" PRIu64 " B=%" PRIu64 " (%s)\n", nd, wa.sims, wb.sims, wa.out_block ? "결과 구간" : "입력 구간");
    const AttrInfo* aa = (a->cmd == kSet) ? &A.attrs[a->attr] : nullptr;
    const AttrInfo* bb = (b->cmd == kSet) ? &B.attrs[b->attr] : nullptr;
    printf("  A #%zu %s %s obj#%" PRIu64 " %s\n", wa.i - 1, cn[a->cmd], wa.attr(*a).c_str(), wa.o(a->obj), hex(A.data(*a), a->data_len, aa).c_str());
    printf("  B #%zu %s %s obj#%" PRIu64 " %s\n", wb.i - 1, cn[b->cmd], wb.attr(*b).c_str(), wb.o(b->obj), hex(B.data(*b), b->data_len, bb).c_str());
    if (nd >= maxd) break;
  }
  return nd ? 3 : 0;
}
