// OVD 한 프레임의 접촉 쌍 목록 (PxScene.pairsActors / pairsContactCounts / pairsContactSeparations / pairsContactImpulses).
// 공식 기록과 ovd_replay --record 로 뜬 재생 기록을 같은 프레임에서 나란히 보면 어떤 접촉이 한쪽에만 있는지 바로 보인다.
//   ovd_pairs <file.ovd> <프레임 번호(1 부터)> [이름 필터]
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ovd.h"

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ovd_pairs <file.ovd> <frame> [filter]\n"); return 2; }
  ovd::File F;
  std::string err;
  if (!ovd::load(argv[1], F, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  const long want = atol(argv[2]);
  const std::string filt = argc > 3 ? argv[3] : "";
  auto attr = [&](const char* c, const char* a) -> uint32_t {
    for (auto& kv : F.attrs) if (kv.second.name == a && F.classes[kv.second.cls].name == c) return kv.first;
    return 0;
  };
  const uint32_t a_el = attr("PxScene", "elapsedTime"), a_name = attr("PxActor", "name"), a_pa = attr("PxScene", "pairsActors"),
                 a_cc = attr("PxScene", "pairsContactCounts"), a_sep = attr("PxScene", "pairsContactSeparations"),
                 a_imp = attr("PxScene", "pairsContactImpulses");
  std::unordered_map<uint64_t, std::string> name;
  long sims = 0;
  std::vector<uint64_t> actors;
  std::vector<uint32_t> counts;
  std::vector<float> seps, imps;
  for (const ovd::Event& e : F.events) {
    if (e.cmd == ovd::kCreate) { name.erase(e.obj); continue; }
    if (e.cmd != ovd::kSet) continue;
    if (e.attr == a_name) { name[e.obj] = F.str(e); continue; }
    if (e.attr == a_el) { if (++sims > want) break; continue; }
    if (sims != want) continue;
    const uint8_t* p = F.data(e);
    if (e.attr == a_pa) { actors.resize(e.data_len / 8); memcpy(actors.data(), p, e.data_len); }
    else if (e.attr == a_cc) { counts.resize(e.data_len / 4); memcpy(counts.data(), p, e.data_len); }
    else if (e.attr == a_sep) { seps.resize(e.data_len / 4); memcpy(seps.data(), p, e.data_len); }
    else if (e.attr == a_imp) { imps.resize(e.data_len / 4); memcpy(imps.data(), p, e.data_len); }
  }
  printf("프레임 %ld: 쌍 %zu, 점 %zu\n", want, actors.size() / 2, seps.size());
  size_t off = 0;
  for (size_t i = 0; i + 1 < actors.size(); i += 2) {
    const uint32_t n = i / 2 < counts.size() ? counts[i / 2] : 0;
    std::string a = name.count(actors[i]) ? name[actors[i]] : "?", b = name.count(actors[i + 1]) ? name[actors[i + 1]] : "?";
    float mn = 1e30f, imp = 0;
    for (uint32_t k = 0; k < n && off + k < seps.size(); ++k) { if (seps[off + k] < mn) mn = seps[off + k]; if (off + k < imps.size()) imp += imps[off + k]; }
    off += n;
    if (!filt.empty() && a.find(filt) == std::string::npos && b.find(filt) == std::string::npos) continue;
    if (a > b) std::swap(a, b);
    printf("  %s <-> %s  점 %u 최소분리 %.9g 충격합 %.9g\n", a.c_str(), b.c_str(), n, mn, imp);
  }
  return 0;
}
