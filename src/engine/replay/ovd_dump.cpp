// OVD 요약: 어떤 PhysX 객체·기능이 쓰였나 (엔진 자체 구현에서 옮길 범위를 정하는 근거).
//   ovd_dump <file.ovd> [--attrs] [--objects CLASS] [--messages]
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "OmniPvdDefines.h"
#include "ovd.h"

using namespace ovd;

static std::string fmt_value(const File& f, const AttrInfo& a, const uint8_t* p, uint32_t len, size_t max_elems = 12) {
  char buf[64];
  std::string s;
  size_t esz = 0;
  switch (a.type) {
    case OmniPvdDataType::eINT8: case OmniPvdDataType::eUINT8: esz = 1; break;
    case OmniPvdDataType::eINT16: case OmniPvdDataType::eUINT16: esz = 2; break;
    case OmniPvdDataType::eINT32: case OmniPvdDataType::eUINT32: case OmniPvdDataType::eFLOAT32:
    case OmniPvdDataType::eENUM_VALUE: case OmniPvdDataType::eFLAGS_WORD: esz = 4; break;
    case OmniPvdDataType::eINT64: case OmniPvdDataType::eUINT64: case OmniPvdDataType::eFLOAT64:
    case OmniPvdDataType::eOBJECT_HANDLE: esz = 8; break;
    case OmniPvdDataType::eSTRING: return "\"" + std::string(reinterpret_cast<const char*>(p), len) + "\"";
    default: esz = 1;
  }
  if (a.type == OmniPvdDataType::eFLAGS_WORD && len == 1) esz = 1;
  if (a.type == OmniPvdDataType::eFLAGS_WORD && len == 2) esz = 2;
  size_t n = esz ? len / esz : 0;
  s += "[";
  for (size_t i = 0; i < n && i < max_elems; ++i) {
    const uint8_t* q = p + i * esz;
    switch (a.type) {
      case OmniPvdDataType::eFLOAT32: { float v; memcpy(&v, q, 4); snprintf(buf, sizeof buf, "%.9g", v); break; }
      case OmniPvdDataType::eFLOAT64: { double v; memcpy(&v, q, 8); snprintf(buf, sizeof buf, "%.17g", v); break; }
      case OmniPvdDataType::eINT32: { int32_t v; memcpy(&v, q, 4); snprintf(buf, sizeof buf, "%d", v); break; }
      case OmniPvdDataType::eOBJECT_HANDLE: case OmniPvdDataType::eUINT64: case OmniPvdDataType::eINT64: {
        uint64_t v; memcpy(&v, q, 8); snprintf(buf, sizeof buf, "0x%" PRIx64, v); break; }
      default: {
        uint32_t v = 0; memcpy(&v, q, esz); snprintf(buf, sizeof buf, "%u", v);
      }
    }
    if (i) s += " ";
    s += buf;
  }
  if (n > max_elems) s += " ...(" + std::to_string(n) + ")";
  s += "]";
  return s;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ovd_dump <file.ovd> [--attrs] [--objects CLASS] [--messages] [--values CLASS]\n");
    return 2;
  }
  File f;
  std::string err;
  if (!load(argv[1], f, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  bool show_attrs = false, show_msgs = false;
  std::string obj_class, values_class, trace_name;
  size_t trace_limit = 60;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--attrs")) show_attrs = true;
    else if (!strcmp(argv[i], "--messages")) show_msgs = true;
    else if (!strcmp(argv[i], "--objects") && i + 1 < argc) obj_class = argv[++i];
    else if (!strcmp(argv[i], "--values") && i + 1 < argc) values_class = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_name = argv[++i];
    else if (!strcmp(argv[i], "--limit") && i + 1 < argc) trace_limit = strtoull(argv[++i], nullptr, 10);
  }

  printf("OVD %u.%u.%u  명령 %zu  데이터 %.1f MB  클래스 %zu  속성 %zu\n", f.ver_major, f.ver_minor, f.ver_patch,
         f.events.size(), f.blob.size() / 1048576.0, f.classes.size(), f.attrs.size());

  std::map<uint64_t, uint32_t> obj_class_of;       // 살아 있는 객체 -> 클래스
  std::map<uint32_t, uint64_t> created, destroyed;  // 클래스별 개수
  std::map<uint32_t, uint64_t> set_count;           // 속성별 set 수
  std::map<uint64_t, uint64_t> frames_per_ctx;
  std::map<std::string, uint64_t> cmd_count;
  uint64_t msgs = 0;
  const char* cmd_names[] = {"invalid", "regClass", "regEnum", "regAttr", "regClassAttr", "regList", "set", "addToList",
                             "removeFromList", "create", "destroy", "startFrame", "stopFrame", "message"};
  // 마지막 값 (객체, 속성) -> 이벤트 번호
  std::map<std::pair<uint64_t, uint32_t>, size_t> last_set, first_set;
  for (size_t i = 0; i < f.events.size(); ++i) {
    const Event& e = f.events[i];
    cmd_count[cmd_names[e.cmd]]++;
    switch (e.cmd) {
      case kCreate: created[e.cls]++; obj_class_of[e.obj] = e.cls; break;
      case kDestroy: {
        auto it = obj_class_of.find(e.obj);
        if (it != obj_class_of.end()) { destroyed[it->second]++; obj_class_of.erase(it); }
        break;
      }
      case kSet: case kAddToList: case kRemoveFromList:
        set_count[e.attr]++;
        last_set[{e.obj, e.attr}] = i;
        first_set.emplace(std::make_pair(e.obj, e.attr), i);
        break;
      case kStopFrame: frames_per_ctx[e.ctx]++; break;
      case kMessage:
        msgs++;
        if (show_msgs) printf("  [메시지] %s\n", f.str(e).c_str());
        break;
      default: break;
    }
  }
  printf("\n명령 종류별 수:");
  for (auto& kv : cmd_count) printf(" %s=%" PRIu64, kv.first.c_str(), kv.second);
  printf("\n프레임(=simulate 한 번) 수, 문맥(장면)별:");
  for (auto& kv : frames_per_ctx) printf(" 0x%" PRIx64 "=%" PRIu64, kv.first, kv.second);
  printf("\n메시지 %" PRIu64 " 건\n", msgs);

  printf("\n클래스별 객체 수 (만든 수 / 지운 수 / 끝에 남은 수)\n");
  std::map<uint32_t, uint64_t> alive;
  for (auto& kv : obj_class_of) alive[kv.second]++;
  std::vector<std::pair<std::string, uint32_t>> cls_sorted;
  for (auto& kv : created) cls_sorted.push_back({f.classes[kv.first].name, kv.first});
  std::sort(cls_sorted.begin(), cls_sorted.end());
  for (auto& kv : cls_sorted)
    printf("  %-44s %8" PRIu64 " %8" PRIu64 " %8" PRIu64 "\n", kv.first.c_str(), created[kv.second],
           destroyed[kv.second], alive[kv.second]);

  if (show_attrs) {
    printf("\n속성별 set 수 (많은 순)\n");
    std::vector<std::pair<uint64_t, uint32_t>> v;
    for (auto& kv : set_count) v.push_back({kv.second, kv.first});
    std::sort(v.rbegin(), v.rend());
    for (auto& kv : v) printf("  %10" PRIu64 "  %s\n", kv.first, f.attr_name(kv.second).c_str());
  }

  // 특정 클래스 객체들의 속성 값 분포 (마지막 값 기준, 서로 다른 값별 개수)
  if (!values_class.empty()) {
    uint32_t c = f.cls(values_class);
    printf("\n[%s] 객체들의 속성 값 분포 (마지막 값, 서로 다른 값 상위 8개)\n", values_class.c_str());
    std::map<uint32_t, std::map<std::string, uint64_t>> dist;
    for (auto& kv : last_set) {
      auto oc = obj_class_of.find(kv.first.first);
      uint32_t oclass = 0;
      if (oc != obj_class_of.end()) oclass = oc->second;
      else {
        // 지워진 객체도 포함하려면 created 기록이 필요 -> 생성 이벤트에서 찾는다
        continue;
      }
      if (!f.is_a(oclass, c)) continue;
      const Event& e = f.events[kv.second];
      const AttrInfo& a = f.attrs[e.attr];
      dist[e.attr][fmt_value(f, a, f.data(e), e.data_len, 8)]++;
    }
    for (auto& d : dist) {
      std::vector<std::pair<uint64_t, std::string>> vv;
      for (auto& kv : d.second) vv.push_back({kv.second, kv.first});
      std::sort(vv.rbegin(), vv.rend());
      printf("  %s (서로 다른 값 %zu)\n", f.attr_name(d.first).c_str(), vv.size());
      for (size_t i = 0; i < vv.size() && i < 8; ++i) printf("      %8" PRIu64 "  %s\n", vv[i].first, vv[i].second.c_str());
    }
  }

  // 한 객체(이름 또는 16진 핸들)의 모든 명령을 순서대로: simulate 번호와 함께. "#a-b" 면 명령 번호 구간 전체
  if (!trace_name.empty()) {
    uint64_t target = 0;
    if (trace_name[0] == '#') {
      size_t a = strtoull(trace_name.c_str() + 1, nullptr, 10), b = a;
      const char* dash = strchr(trace_name.c_str(), '-');
      if (dash) b = strtoull(dash + 1, nullptr, 10);
      for (size_t i = a; i <= b && i < f.events.size(); ++i) {
        const Event& e = f.events[i];
        printf("  #%zu %s obj=0x%" PRIx64 " %s", i, cmd_names[e.cmd], e.obj,
               (e.cmd == kCreate ? f.classes[e.cls].name.c_str() : (e.cmd == kSet || e.cmd == kAddToList || e.cmd == kRemoveFromList) ? f.attr_name(e.attr).c_str() : ""));
        if (e.cmd == kSet || e.cmd == kAddToList || e.cmd == kRemoveFromList) printf(" %s", fmt_value(f, f.attrs[e.attr], f.data(e), e.data_len).c_str());
        printf("\n");
      }
      return 0;
    }
    if (trace_name.rfind("0x", 0) == 0) target = strtoull(trace_name.c_str(), nullptr, 16);
    const uint32_t a_name = [&] { for (auto& kv : f.attrs) if (kv.second.name == "name") return kv.first; return 0u; }();
    (void)a_name;
    uint64_t sims = 0;
    size_t shown = 0;
    for (size_t i = 0; i < f.events.size() && shown < trace_limit; ++i) {
      const Event& e = f.events[i];
      if (e.cmd == kSet && f.attrs[e.attr].name == "elapsedTime") sims++;
      if (!target && e.cmd == kSet && f.attrs[e.attr].name == "name" && f.str(e) == trace_name) target = e.obj;
      if (!target) continue;
      bool mine = e.obj == target;
      if (!mine && (e.cmd == kAddToList || e.cmd == kRemoveFromList) && e.data_len == 8) { uint64_t v; memcpy(&v, f.data(e), 8); mine = v == target; }
      if (!mine) continue;
      shown++;
      printf("  #%zu sim=%" PRIu64 " %s %s", i, sims, cmd_names[e.cmd], (e.cmd == kCreate ? f.classes[e.cls].name.c_str() : f.attr_name(e.attr).c_str()));
      if (e.cmd == kSet) printf(" %s", fmt_value(f, f.attrs[e.attr], f.data(e), e.data_len).c_str());
      if (e.cmd == kAddToList || e.cmd == kRemoveFromList) printf(" owner=0x%" PRIx64, e.obj);
      printf("\n");
    }
  }

  if (!obj_class.empty()) {
    uint32_t c = f.cls(obj_class);
    printf("\n[%s] 객체 목록 (이름, 처음 설정값)\n", obj_class.c_str());
    for (size_t i = 0; i < f.events.size(); ++i) {
      const Event& e = f.events[i];
      if (e.cmd != kCreate || !f.is_a(e.cls, c)) continue;
      printf("  0x%" PRIx64 " %s \"%s\"\n", e.obj, f.classes[e.cls].name.c_str(), f.str(e).c_str());
      for (auto it = first_set.lower_bound({e.obj, 0}); it != first_set.end() && it->first.first == e.obj; ++it) {
        const Event& s = f.events[it->second];
        const AttrInfo& a = f.attrs[s.attr];
        printf("      %-40s %s\n", f.attr_name(s.attr).c_str(), fmt_value(f, a, f.data(s), s.data_len).c_str());
      }
    }
  }
  return 0;
}
