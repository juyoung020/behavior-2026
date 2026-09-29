// 두 OVD 를 명령 단위로 나란히 비교한다. 객체 핸들(원래 프로세스의 포인터)은 실행마다 다르므로
// "만들어진 순서"로 짝을 짓는다. 원본 기록과 ovd_replay --record 로 뜬 재생 기록을 비교하면
// 재생이 빠뜨리거나 다르게 넣은 API 호출(입력)과, 처음 달라진 결과(출력)를 정확히 짚는다.
//   ovd_diff <a.ovd> <b.ovd> [--max N] [--inputs-only]
#include <cinttypes>
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

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ovd_diff <a.ovd> <b.ovd> [--max N] [--inputs-only]\n"); return 2; }
  File A, B;
  std::string err;
  if (!load(argv[1], A, err) || !load(argv[2], B, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  int maxd = 10;
  bool inputs_only = false;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--max") && i + 1 < argc) maxd = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--inputs-only")) inputs_only = true;
  }
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
