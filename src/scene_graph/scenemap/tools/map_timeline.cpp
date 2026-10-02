// 팀 벤치마크(dynamic-object-mapping-benchmark) 형식의 지도 시간표: ep_<ep>.bin(proprio·slam2d keyframe 깊이) +
// ep_<ep>_det.bin(검출 — objmap_eval 과 같은 '완벽한 검출')을 C ABI 로만 넣고(sm_push_proprio 매 프레임,
// keyframe 은 sm_push_image 검출 없이, 검출 프레임은 그 깊이 + 검출로), 물체 지도가 바뀔 때마다 그 프레임의 지도 전체를
//   <out>/map_timeline.csv : frame,obj_id,x,y,z,label,moving
// 로 쓴다(METRICS.md: 같은 frame 의 줄이 그때의 지도 전체, 바뀔 때만, 빈 지도는 "frame,,,,,,"). 사라짐 상태 물체는 뺀다,
// moving = 들고 있음(SM_HELD). 구조물(벽·바닥·문 …)은 scenemap 이 노드로 만들지 않는다(sm_set_kind_names 기본 표).
//
//   map_timeline <ep.bin> <det.bin> <out dir> [--min-cells N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "scenemap.h"

#pragma pack(push, 1)
struct Row {
  float qvel[3], eefL[3], eefR[3], gl, gr;
  double gt[3];
  float prop[61];
};
#pragma pack(pop)

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: map_timeline <ep.bin> <det.bin> <out dir> [--min-cells N]\n");
    return 2;
  }
  int min_cells = 8;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--min-cells") && i + 1 < argc) min_cells = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  // ep.bin
  FILE* f = std::fopen(argv[1], "rb");
  char magic[4];
  uint32_t hdr[5];
  float K[4];
  if (!f || std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5 || std::fread(K, 4, 4, f) != 4 || hdr[0] != 2) {
    std::fprintf(stderr, "ep.bin\n");
    return 1;
  }
  const uint32_t n = hdr[1], w = hdr[3], h = hdr[4];
  std::vector<Row> rows(n);
  if (std::fread(rows.data(), sizeof(Row), n, f) != n) return 1;
  uint32_t m;
  if (std::fread(&m, 4, 1, f) != 1) return 1;
  std::vector<uint16_t> depth(size_t(w) * h);
  float Tbc[12];
  uint32_t next = 0, done = 0;
  auto readKf = [&]() -> bool {
    if (done >= m) return false;
    if (std::fread(&next, 4, 1, f) != 1 || std::fread(Tbc, 4, 12, f) != 12 || std::fread(depth.data(), 2, depth.size(), f) != depth.size())
      return false;
    ++done;
    return true;
  };
  // det.bin
  FILE* g = std::fopen(argv[2], "rb");
  uint32_t dh[3];
  float DK[4];
  if (!g || std::fread(magic, 1, 4, g) != 4 || std::memcmp(magic, "SMDT", 4) || std::fread(dh, 4, 3, g) != 3 ||
      std::fread(DK, 4, 4, g) != 4) {
    std::fprintf(stderr, "det.bin\n");
    return 1;
  }
  const int DW = int(dh[1]), DH = int(dh[2]);
  uint32_t n_obj;
  if (std::fread(&n_obj, 4, 1, g) != 1) return 1;
  std::vector<std::string> oname(n_obj), ocat(n_obj);
  std::vector<uint8_t> ostruct(n_obj);
  auto rstr = [&](std::string* s) {
    uint16_t L;
    if (std::fread(&L, 2, 1, g) != 1) return false;
    s->resize(L);
    return L == 0 || std::fread(&(*s)[0], 1, L, g) == L;
  };
  for (uint32_t k = 0; k < n_obj; ++k)
    if (!rstr(&oname[k]) || !rstr(&ocat[k]) || std::fread(&ostruct[k], 1, 1, g) != 1) return 1;
  std::map<std::string, int> cat_id;
  for (uint32_t k = 0; k < n_obj; ++k)
    if (!ostruct[k]) cat_id.emplace(ocat[k], 0);
  std::vector<std::string> cats;
  for (auto& kv : cat_id) { kv.second = int(cats.size()); cats.push_back(kv.first); }
  uint32_t dm;
  if (std::fread(&dm, 4, 1, g) != 1) return 1;
  std::vector<uint16_t> ddepth(size_t(DW) * DH), dlab(size_t(DW) * DH);
  float dT[12];
  uint32_t dnext = 0, ddone = 0;
  auto readDet = [&]() -> bool {
    if (ddone >= dm) return false;
    if (std::fread(&dnext, 4, 1, g) != 1 || std::fread(dT, 4, 12, g) != 12 || std::fread(ddepth.data(), 2, ddepth.size(), g) != ddepth.size() ||
        std::fread(dlab.data(), 2, dlab.size(), g) != dlab.size())
      return false;
    ++ddone;
    return true;
  };

  sm_ctx* c = sm_create(nullptr);
  {
    // 프롬프트 표 = 정답 범주 이름(검출 cls 가 이 번호)
    std::vector<const char*> lab;
    for (auto& s : cats) lab.push_back(s.c_str());
    sm_set_labels(c, lab.data(), int(lab.size()));
  }
  std::filesystem::create_directories(argv[3]);
  const std::string out = (std::filesystem::path(argv[3]) / "map_timeline.csv").string();
  FILE* o = std::fopen(out.c_str(), "w");
  if (!o) return 1;
  std::fprintf(o, "frame,obj_id,x,y,z,label,moving\n");
  bool have = readKf(), dhave = readDet();
  const size_t words = (size_t(DW) * DH + 31) / 32;
  std::vector<int32_t> cls;
  std::vector<float> score, box, dmf(ddepth.size()), kmf(depth.size());
  std::vector<uint32_t> bits;
  std::string prev = "\x01";   // 지난번 쓴 지도(같으면 안 씀)
  int n_snap = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const double t = i / 30.0;
    sm_proprio p{t, rows[i].prop, 61};
    sm_push_proprio(c, &p);
    if (have && next == i) {   // slam2d keyframe(검출 없음)
      for (size_t k = 0; k < depth.size(); ++k) kmf[k] = depth[k] * 1e-3f;
      sm_image im{t, 0, int(w), int(h), nullptr, kmf.data(), K[0], K[1], K[2], K[3]};
      sm_push_image(c, &im, nullptr);
      have = readKf();
    }
    if (!(dhave && dnext == i)) continue;
    // 라벨 → 검출(물체마다 마스크 한 장, objmap_eval 과 같음)
    std::map<int, std::vector<int>> cells;
    for (int k = 0; k < DW * DH; ++k)
      if (dlab[k] != 0xFFFF && dlab[k] < n_obj && !ostruct[dlab[k]]) cells[dlab[k]].push_back(k);
    cls.clear(); score.clear(); box.clear(); bits.clear();
    for (auto& [ob, v] : cells) {
      if (int(v.size()) < min_cells) continue;
      const size_t base = bits.size();
      bits.resize(base + words, 0);
      int x0 = DW, y0 = DH, x1 = 0, y1 = 0;
      for (int k : v) {
        bits[base + (k >> 5)] |= 1u << (k & 31);
        const int x = k % DW, y = k / DW;
        x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
      }
      cls.push_back(cat_id[ocat[ob]]);
      score.push_back(1.f);
      box.insert(box.end(), {float(x0), float(y0), float(x1 + 1), float(y1 + 1)});
    }
    sm_detections D{};
    D.stamp = t; D.cam = 0; D.img_w = DW; D.img_h = DH; D.n = int(cls.size());
    D.cls = cls.data(); D.score = score.data(); D.box = box.data();
    D.mask_w = DW; D.mask_h = DH; D.mask_sx = 1; D.mask_sy = 1; D.mask_ox = 0; D.mask_oy = 0; D.mask_bits = bits.data();
    for (size_t k = 0; k < ddepth.size(); ++k) dmf[k] = ddepth[k] * 1e-3f;
    sm_image im{t, 0, DW, DH, nullptr, dmf.data(), DK[0], DK[1], DK[2], DK[3]};
    sm_push_image(c, &im, &D);
    dhave = readDet();
    // 지도 전체(사라짐 뺌)
    sm_snapshot_t* s = nullptr;
    sm_snapshot(c, &s);
    const sm_object* ob = nullptr;
    const int m = sm_snap_objects(s, &ob);
    std::string cur;
    char line[512];
    for (int k = 0; k < m; ++k) {
      if (ob[k].state == SM_GONE) continue;
      std::string lab = ob[k].name ? ob[k].name : "";
      std::replace(lab.begin(), lab.end(), ',', ' ');
      std::snprintf(line, sizeof(line), "%u,%u,%.3f,%.3f,%.3f,%s,%d\n", i, ob[k].id, ob[k].pos[0], ob[k].pos[1], ob[k].pos[2], lab.c_str(),
                    int(ob[k].state == SM_HELD));
      cur += line;
    }
    sm_snapshot_release(s);
    if (cur.empty()) {
      std::snprintf(line, sizeof(line), "%u,,,,,,\n", i);
      cur = line;
    }
    // 프레임 번호를 뺀 내용이 같으면 쓰지 않음(0차 유지)
    auto body = [](const std::string& x) {
      std::string b;
      size_t a = 0;
      while (a < x.size()) {
        const size_t e = x.find('\n', a), comma = x.find(',', a);
        b.append(x, comma, e - comma + 1);
        a = e + 1;
      }
      return b;
    };
    const std::string cb = body(cur);
    if (cb != prev) {
      std::fputs(cur.c_str(), o);
      prev = cb;
      ++n_snap;
    }
  }
  std::fclose(o);
  sm_destroy(c);
  std::printf("%s: frames %u, snapshots %d\n", out.c_str(), n, n_snap);
  return 0;
}
