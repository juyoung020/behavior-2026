// scenemap 저장 구현(include/scenemap/dsg_save.hpp).
#include "scenemap/dsg_save.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
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

std::string plyPath(uint32_t id) { return "objects/O" + std::to_string(id) + "_points.ply"; }

bool hasView(const SaveInput& in, const SaveOut& out, int i) { return i < int(out.png_ok.size()) && out.png_ok[i] && in.views[i]; }
bool hasPly(const SaveOut& out, int i) { return i < int(out.ply_ok.size()) && out.ply_ok[i]; }

// 점 구름 → binary_little_endian PLY(머리 다음 점마다 float x,y,z(map) + uchar r,g,b = 15 바이트)
std::string plyBytes(const ObjCloud& c) {
  const size_t n = c.size();
  std::string o = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(n) +
                  "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                  "property uchar blue\nend_header\n";
  const size_t h = o.size();
  o.resize(h + n * 15);
  char* p = &o[h];
  for (size_t i = 0; i < n; ++i, p += 15) {
    const CloudPt& q = c.data->pts[i];
    const float xyz[3] = {float(c.org[0] + q.x), float(c.org[1] + q.y), float(c.org[2] + q.z)};
    std::memcpy(p, xyz, 12);   // x86 은 리틀 엔디언
    p[12] = char(q.r);
    p[13] = char(q.g);
    p[14] = char(q.b);
  }
  return o;
}

std::string viewJson(const SaveInput& in, const SaveOut& ok) {
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
      << ",\"last_seen\":" << b.last_seen << ",\"score\":" << b.score << ",\"structural\":" << (b.structural ? "true" : "false")
      << ",\"movable\":" << (i >= int(in.movable.size()) || in.movable[i] ? "true" : "false");
    if (hasView(in, ok, i))
      o << ",\"rgbd\":{\"rgb\":\"" << objPath(b.id, "rgb") << "\",\"depth\":\"" << objPath(b.id, "depth") << "\",\"mask\":\""
        << objPath(b.id, "mask") << "\"}";
    if (hasPly(ok, i))
      o << ",\"points\":{\"path\":\"" << plyPath(b.id) << "\",\"n\":" << in.clouds[i].size() << ",\"voxel\":" << in.voxel
        << ",\"stamp\":" << in.clouds[i].stamp << "}";
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
bool sceneDsg(const SaveInput& in, const SaveOut& ok, const fs::path& path) {
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
                     {"handled", bool(b.handled)},
                     {"movable", i >= int(in.movable.size()) || in.movable[i] != 0}});
    if (hasView(in, ok, i)) {
      const BestView& v = *in.views[i];
      std::vector<double> T(v.cam_T, v.cam_T + 12);
      a->metadata.add({{"rgbd",
                        {{"rgb", objPath(b.id, "rgb")},
                         {"depth", objPath(b.id, "depth")},
                         {"mask", objPath(b.id, "mask")},
                         {"stamp", v.stamp},
                         {"box_px", {v.box[0], v.box[1], v.box[2], v.box[3]}},
                         {"det_box_px", {v.det_box[0], v.det_box[1], v.det_box[2], v.det_box[3]}},
                         {"mask_area", v.mask_area},
                         {"depth_m", v.depth_m},
                         {"score", v.score},
                         {"cam_T", T}}}});
    }
    if (hasPly(ok, i))
      a->metadata.add({{"points",
                        {{"path", plyPath(b.id)}, {"n", in.clouds[i].size()}, {"voxel", in.voxel}, {"stamp", in.clouds[i].stamp}}}});
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

bool hasCloud(const SaveInput& in, int i) { return i < int(in.clouds.size()) && in.clouds[i].size() > 0; }

// objects/ 의 best view PNG·구름 PLY: 바뀐 것·없는 것만 쓰고, png_ok·ply_ok[i] = 파일 있음
bool savePngs(const SaveInput& in, const fs::path& od, SaveOut& out) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<uint8_t>& ok = out.png_ok;
  ok.assign(in.n_objs, 0);
  out.ply_ok.assign(in.n_objs, 0);
  std::error_code ec;
  bool any = false;
  for (int i = 0; i < in.n_objs; ++i) any = any || (i < int(in.views.size()) && in.views[i]) || hasCloud(in, i);
  if (!any && !in.clean_objects) return true;
  fs::create_directories(od, ec);
  if (in.clean_objects) {   // 지금 물체가 아닌 O<id>_{rgb,depth,mask}.png · O<id>_points.ply 지우기
    std::vector<std::string> keep;
    for (int i = 0; i < in.n_objs; ++i) {
      const std::string b = "O" + std::to_string(in.objs[i].id);
      if (i < int(in.views.size()) && in.views[i])
        for (const char* w : {"_rgb.png", "_depth.png", "_mask.png"}) keep.push_back(b + w);
      if (hasCloud(in, i)) keep.push_back(b + "_points.ply");
    }
    auto endsWith = [](const std::string& n, const char* e) {
      const size_t m = std::strlen(e);
      return n.size() > m && n.compare(n.size() - m, m, e) == 0;
    };
    for (const auto& e : fs::directory_iterator(od, ec)) {
      const std::string n = e.path().filename().string();
      const bool ours = n.size() > 1 && n[0] == 'O' && std::isdigit(static_cast<unsigned char>(n[1])) &&
                        (endsWith(n, "_rgb.png") || endsWith(n, "_depth.png") || endsWith(n, "_mask.png") || endsWith(n, "_points.ply"));
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
    const fs::path pm = od.parent_path() / objPath(in.objs[i].id, "mask");
    if (ok_i && !v->mask.empty() && (dirty || !fs::exists(pm, ec))) {
      const std::string s = pngGray8(v->mask.data(), v->w, v->h);
      ok_i = !s.empty() && writeAtomic(pm, s);
      out.n_png += ok_i;
    }
    ok[i] = ok_i;
    good = good && ok_i;
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.png_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (int i = 0; i < in.n_objs; ++i) {
    if (!hasCloud(in, i)) continue;
    const fs::path pp = od.parent_path() / plyPath(in.objs[i].id);
    const bool dirty = i < int(in.ply_dirty.size()) && in.ply_dirty[i];
    bool ok_i = true;
    if (dirty || !fs::exists(pp, ec)) {
      ok_i = writeAtomic(pp, plyBytes(in.clouds[i]));
      out.n_ply += ok_i;
    }
    out.ply_ok[i] = ok_i;
    good = good && ok_i;
  }
  out.ply_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
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
  bool ok = savePngs(in, d / "objects", out);
  if (in.grid_w > 0 && in.cells) {
    ok &= writeAtomic(d / "map.pgm", pgm(in));
    ok &= writeAtomic(d / "map.yaml", yaml(in));
  }
#ifdef SM_HAVE_SPARK_DSG
  ok &= sceneDsg(in, out, d / "scene.json");
#endif
  ok &= writeAtomic(d / "view.json", viewJson(in, out));   // 마지막에: 뷰어는 view.json 이 바뀌면 다시 읽는다
  return ok ? 0 : -1;
}

}  // namespace scenemap
