// scenemap 저장 구현(include/scenemap/dsg_save.hpp).
#include "scenemap/dsg_save.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "scenemap/png.hpp"

#ifdef SM_HAVE_SPARK_DSG
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#endif

namespace scenemap {
namespace {

namespace fs = std::filesystem;

const char* stateName(int s) {
  switch (s) {
    case SM_SEEN: return "seen";
    case SM_GONE: return "gone";
    case SM_MOVED: return "moved";
    case SM_HELD: return "held";
  }
  return "?";
}

const char* eventName(int k) {
  static const char* n[] = {"candidate", "confirmed", "moved", "gone", "picked", "placed", "seen_again"};
  return k >= 0 && k < 7 ? n[k] : "?";
}

// JSON 문자열 이스케이프(이름은 프롬프트 표 — 따옴표·역슬래시만 막으면 된다)
std::string esc(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if (static_cast<unsigned char>(ch) >= 0x20) o += ch;
  }
  return o;
}

// 임시 파일에 쓰고 rename(같은 디렉터리라 원자적)
bool writeAtomic(const fs::path& path, const std::string& data) {
  const fs::path tmp = path.string() + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), std::streamsize(data.size()));
    if (!f) return false;
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  return !ec;
}

std::string objPath(uint32_t id, const char* what) { return "objects/O" + std::to_string(id) + "_" + what + ".png"; }

bool hasView(const SaveInput& in, const std::vector<uint8_t>& ok, int i) { return i < int(ok.size()) && ok[i] && in.views[i]; }

std::string viewJson(const SaveInput& in, const std::vector<uint8_t>& ok) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(3);
  o << "{\"stamp\":" << in.stamp << ",\"pose\":[" << in.pose[0] << "," << in.pose[1] << "," << in.pose[2] << "],";
  o << "\"grid\":{\"resolution\":" << in.grid_res << ",\"origin\":[" << in.grid_ox << "," << in.grid_oy << "],\"width\":" << in.grid_w
    << ",\"height\":" << in.grid_h << "},\"objects\":[";
  for (int i = 0; i < in.n_objs; ++i) {
    const sm_object& b = in.objs[i];
    o << (i ? "," : "") << "{\"id\":" << b.id << ",\"name\":\"" << esc(b.name ? b.name : "") << "\",\"state\":\"" << stateName(b.state)
      << "\",\"pos\":[" << b.pos[0] << "," << b.pos[1] << "," << b.pos[2] << "],\"extent\":[" << b.extent[0] << "," << b.extent[1] << ","
      << b.extent[2] << "],\"first_pos\":[" << b.first_pos[0] << "," << b.first_pos[1] << "," << b.first_pos[2] << "],\"n_obs\":" << b.n_obs
      << ",\"last_seen\":" << b.last_seen << ",\"score\":" << b.score << ",\"structural\":" << (b.structural ? "true" : "false");
    if (hasView(in, ok, i)) o << ",\"rgbd\":{\"rgb\":\"" << objPath(b.id, "rgb") << "\",\"depth\":\"" << objPath(b.id, "depth") << "\"}";
    o << "}";
  }
  o << "],\"events\":[";
  for (size_t i = 0; i < in.events.size(); ++i) {
    const ObjEvent& e = in.events[i];
    o << (i ? "," : "") << "{\"t\":" << e.t << ",\"id\":" << e.id << ",\"kind\":\"" << eventName(e.kind) << "\",\"pos\":[" << e.pos[0] << ","
      << e.pos[1] << "," << e.pos[2] << "]}";
  }
  o << "]}\n";
  return o.str();
}

// ROS map_server 형식(위가 +y — 격자 행을 뒤집어 쓴다)
std::string pgm(const SaveInput& in) {
  std::string o = "P5\n" + std::to_string(in.grid_w) + " " + std::to_string(in.grid_h) + "\n255\n";
  o.reserve(o.size() + size_t(in.grid_w) * in.grid_h);
  for (int y = in.grid_h - 1; y >= 0; --y)
    for (int x = 0; x < in.grid_w; ++x) {
      const int v = in.cells[size_t(y) * in.grid_w + x];
      o += char(v < 0 ? 205 : (v >= 65 ? 0 : (v <= 25 ? 254 : 254 - (v * 254) / 100)));
    }
  return o;
}

std::string yaml(const SaveInput& in) {
  std::ostringstream o;
  o << "image: map.pgm\nresolution: " << in.grid_res << "\norigin: [" << in.grid_ox << ", " << in.grid_oy
    << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n";
  return o.str();
}

#ifdef SM_HAVE_SPARK_DSG
bool sceneDsg(const SaveInput& in, const std::vector<uint8_t>& ok, const fs::path& path) {
  using namespace spark_dsg;
  DynamicSceneGraph g;
  for (int i = 0; i < in.n_objs; ++i) {
    const sm_object& b = in.objs[i];
    auto a = std::make_unique<ObjectNodeAttributes>();
    a->position = Eigen::Vector3d(b.pos[0], b.pos[1], b.pos[2]);
    a->name = b.name ? b.name : "";
    a->bounding_box = BoundingBox(Eigen::Vector3f(float(b.extent[0]), float(b.extent[1]), float(b.extent[2])), a->position.cast<float>());
    a->last_update_time_ns = uint64_t(std::max(0.0, b.last_seen) * 1e9);
    a->is_active = b.state != SM_GONE;
    a->metadata.add({{"state", stateName(b.state)},
                     {"n_obs", b.n_obs},
                     {"score", b.score},
                     {"first_pos", {b.first_pos[0], b.first_pos[1], b.first_pos[2]}},
                     {"structural", bool(b.structural)},
                     {"handled", bool(b.handled)}});
    if (hasView(in, ok, i)) {
      const BestView& v = *in.views[i];
      std::vector<double> T(v.cam_T, v.cam_T + 12);
      a->metadata.add({{"rgbd",
                        {{"rgb", objPath(b.id, "rgb")},
                         {"depth", objPath(b.id, "depth")},
                         {"stamp", v.stamp},
                         {"box_px", {v.box[0], v.box[1], v.box[2], v.box[3]}},
                         {"det_box_px", {v.det_box[0], v.det_box[1], v.det_box[2], v.det_box[3]}},
                         {"mask_area", v.mask_area},
                         {"depth_m", v.depth_m},
                         {"score", v.score},
                         {"cam_T", T}}}});
    }
    g.emplaceNode(DsgLayers::OBJECTS, NodeSymbol('O', b.id), std::move(a));
  }
  g.metadata.add({{"stamp", in.stamp}, {"robot_pose", {in.pose[0], in.pose[1], in.pose[2]}}, {"grid", "map.pgm"}});
  const fs::path tmp = path.string() + ".tmp.json";
  g.save(tmp, false);
  std::error_code ec;
  fs::rename(tmp, path, ec);
  return !ec;
}
#endif

// objects/ 의 best view PNG: 바뀐 것·없는 것만 쓰고, ok[i] = 파일 있음
bool savePngs(const SaveInput& in, const fs::path& od, std::vector<uint8_t>& ok, SaveOut& out) {
  const auto t0 = std::chrono::steady_clock::now();
  ok.assign(in.n_objs, 0);
  std::error_code ec;
  bool any = false;
  for (int i = 0; i < in.n_objs && i < int(in.views.size()); ++i) any = any || in.views[i];
  if (!any && !in.clean_objects) return true;
  fs::create_directories(od, ec);
  if (in.clean_objects) {   // 지금 물체가 아닌 O<id>_rgb/depth.png 지우기
    std::vector<std::string> keep;
    for (int i = 0; i < in.n_objs && i < int(in.views.size()); ++i)
      if (in.views[i]) {
        keep.push_back("O" + std::to_string(in.objs[i].id) + "_rgb.png");
        keep.push_back("O" + std::to_string(in.objs[i].id) + "_depth.png");
      }
    for (const auto& e : fs::directory_iterator(od, ec)) {
      const std::string n = e.path().filename().string();
      const bool ours = n.size() > 9 && n[0] == 'O' && n.compare(n.size() - 4, 4, ".png") == 0 &&
                        (n.find("_rgb.png") != std::string::npos || n.find("_depth.png") != std::string::npos);
      if (ours && std::find(keep.begin(), keep.end(), n) == keep.end()) fs::remove(e.path(), ec);
    }
  }
  bool good = true;
  for (int i = 0; i < in.n_objs && i < int(in.views.size()); ++i) {
    const BestView* v = in.views[i].get();
    if (!v || v->rgb.empty() || v->depth.empty()) continue;
    const fs::path pr = od.parent_path() / objPath(in.objs[i].id, "rgb"), pd = od.parent_path() / objPath(in.objs[i].id, "depth");
    const bool dirty = i < int(in.png_dirty.size()) && in.png_dirty[i];
    bool ok_i = true;
    if (dirty || !fs::exists(pr, ec)) {
      const std::string s = pngRgb8(v->rgb.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pr, s);
      out.n_png += ok_i;
    }
    if (ok_i && (dirty || !fs::exists(pd, ec))) {
      const std::string s = pngGray16(v->depth.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pd, s);
      out.n_png += ok_i;
    }
    ok[i] = ok_i;
    good = good && ok_i;
  }
  out.png_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return good;
}

}  // namespace

int saveScene(const SaveInput& in, const std::string& dir, SaveOut* out_) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path d(dir);
  SaveOut local;
  SaveOut& out = out_ ? *out_ : local;
  out = SaveOut{};
  bool ok = savePngs(in, d / "objects", out.png_ok, out);
  if (in.grid_w > 0 && in.cells) {
    ok &= writeAtomic(d / "map.pgm", pgm(in));
    ok &= writeAtomic(d / "map.yaml", yaml(in));
  }
#ifdef SM_HAVE_SPARK_DSG
  ok &= sceneDsg(in, out.png_ok, d / "scene.json");
#endif
  ok &= writeAtomic(d / "view.json", viewJson(in, out.png_ok));   // 마지막에: 뷰어는 view.json 이 바뀌면 다시 읽는다
  return ok ? 0 : -1;
}

}  // namespace scenemap
