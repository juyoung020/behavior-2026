// sgrt 기록(SGRT_RECORD) 재생 — C ABI 로만 scenemap 을 굴려 자세 모드 비교(떠밀림)·단계별 µs 를 잰다.
//
//   sm_bench <rec.bin> [--pose slam|odom|gt] [--lag 0|1] [--policy 0|1] [--no-dets] [--snap-every N] [--save-every S]
//            [--save DIR] [--traj out.csv] [--labels names.txt] [--frames N] [--loops K]
//
// 재생은 sgrt_step 과 같은 순서: 스텝마다 (외부 자세) → proprio, keyframe 이면 영상(stamp = 직전 스텝, --lag 1) + 검출.
// --snap-every N: N 스텝마다 sm_take_dirty + sm_snapshot(탐색 쪽 sgrt_map 흉내). --save-every S: 시뮬 S 초마다 sm_save_dsg.
// --loops K: 같은 기록을 K 번(사이에 sm_reset, 시간은 합침 — 막대그래프 표본 늘리기).
// 끝에 단계 표(n, 평균, p50, p99, 최대 µs)와 자세 진단(외부 자세가 있으면: 첫 keyframe 에서 맞춘 뒤 떠밀림)을 찍는다.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "scenemap.h"

namespace {
struct Rec {
  char tag;
  double stamp;
  std::vector<float> f;       // P: proprio, I: 깊이
  double g[3];                // G
  int w = 0, h = 0;
  double K[4];
  std::vector<uint8_t> rgba;  // I: RGBA(호스트 자르기용)
  // 검출
  int n = 0, img_w = 0, img_h = 0, mask_w = 0, mask_h = 0;
  float msx = 0, msy = 0, mox = 0, moy = 0;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
};

template <class T>
bool rd(FILE* f, T* v) { return std::fread(v, sizeof(T), 1, f) == 1; }

bool load(const char* path, std::vector<Rec>* out, int max_frames) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t ver;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "SGRC", 4) || !rd(f, &ver) || ver != 1) return false;
  int frames = 0;
  for (;;) {
    const int t = std::fgetc(f);
    if (t == EOF) break;
    Rec r;
    r.tag = char(t);
    if (!rd(f, &r.stamp)) break;
    if (t == 'P') {
      int32_t n;
      if (!rd(f, &n)) break;
      r.f.resize(n);
      if (std::fread(r.f.data(), 4, n, f) != size_t(n)) break;
      if (max_frames > 0 && frames >= max_frames) break;
      ++frames;
    } else if (t == 'G') {
      if (std::fread(r.g, 8, 3, f) != 3) break;
    } else if (t == 'I') {
      int32_t w, h;
      if (!rd(f, &w) || !rd(f, &h) || std::fread(r.K, 8, 4, f) != 4) break;
      r.w = w; r.h = h;
      r.f.resize(size_t(w) * h);
      if (std::fread(r.f.data(), 4, r.f.size(), f) != r.f.size()) break;
      const int has = std::fgetc(f);
      if (has == 1) {
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        if (std::fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) break;
        r.rgba.resize(size_t(w) * h * 4);
        for (size_t i = 0; i < size_t(w) * h; ++i) {
          r.rgba[4 * i] = rgb[3 * i]; r.rgba[4 * i + 1] = rgb[3 * i + 1]; r.rgba[4 * i + 2] = rgb[3 * i + 2]; r.rgba[4 * i + 3] = 255;
        }
      }
      int32_t n, iw, ih, mw, mh;
      if (!rd(f, &n) || !rd(f, &iw) || !rd(f, &ih) || !rd(f, &mw) || !rd(f, &mh) || !rd(f, &r.msx) || !rd(f, &r.msy) ||
          !rd(f, &r.mox) || !rd(f, &r.moy))
        break;
      r.n = n; r.img_w = iw; r.img_h = ih; r.mask_w = mw; r.mask_h = mh;
      if (n) {
        const size_t words = (size_t(mw) * mh + 31) / 32;
        r.cls.resize(n); r.score.resize(n); r.box.resize(4 * size_t(n)); r.bits.resize(words * n);
        if (std::fread(r.cls.data(), 4, n, f) != size_t(n) || std::fread(r.score.data(), 4, n, f) != size_t(n) ||
            std::fread(r.box.data(), 4, r.box.size(), f) != r.box.size() || std::fread(r.bits.data(), 4, r.bits.size(), f) != r.bits.size())
          break;
      }
    } else {
      std::fprintf(stderr, "bad tag %d\n", t);
      break;
    }
    out->push_back(std::move(r));
  }
  std::fclose(f);
  return !out->empty();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: sm_bench <rec.bin> [--pose slam|odom|gt] [--lag 0|1] [--policy 0|1] [--no-dets] [--snap-every N] "
                         "[--save-every S] [--save DIR] [--traj out.csv] [--labels names.txt] [--frames N] [--loops K]\n");
    return 2;
  }
  std::string pose = "slam", save_dir, traj, labels_path;
  int lag = 1, policy = 1, snap_every = 6, frames = 0, loops = 1;
  double save_every = 0;
  bool dets_on = true;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--pose") pose = nx();
    else if (a == "--lag") lag = std::atoi(nx());
    else if (a == "--policy") policy = std::atoi(nx());
    else if (a == "--no-dets") dets_on = false;
    else if (a == "--snap-every") snap_every = std::atoi(nx());
    else if (a == "--save-every") save_every = std::atof(nx());
    else if (a == "--save") save_dir = nx();
    else if (a == "--traj") traj = nx();
    else if (a == "--labels") labels_path = nx();
    else if (a == "--frames") frames = std::atoi(nx());
    else if (a == "--loops") loops = std::atoi(nx());
  }
  std::vector<Rec> recs;
  if (!load(argv[1], &recs, frames)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  size_t nP = 0, nI = 0, nG = 0, nD = 0;
  for (const Rec& r : recs) { nP += r.tag == 'P'; nI += r.tag == 'I'; nG += r.tag == 'G'; nD += r.tag == 'I' ? r.n : 0; }
  std::printf("record: %zu steps, %zu keyframes (%zu detections), %zu external poses\n", nP, nI, nD, nG);

  sm_ctx* c = sm_create(nullptr);
  std::vector<std::string> names;
  if (!labels_path.empty()) {
    std::ifstream in(labels_path);
    for (std::string l; std::getline(in, l);) names.push_back(l);
  }
  int maxcls = 0;
  for (const Rec& r : recs) for (int k : r.cls) maxcls = std::max(maxcls, k + 1);
  for (int k = int(names.size()); k < maxcls; ++k) names.push_back("cls" + std::to_string(k));
  auto begin = [&]() {
    sm_reset(c);
    std::vector<const char*> np;
    for (auto& n : names) np.push_back(n.c_str());
    sm_set_labels(c, np.data(), int(np.size()));
    sm_set_pose_mode(c, pose == "gt" ? SM_POSE_GT : pose == "odom" ? SM_POSE_ODOM : SM_POSE_SLAM);
    sm_set_map_update(c, policy, 0);
  };
  FILE* tf = traj.empty() ? nullptr : std::fopen(traj.c_str(), "w");
  if (tf) std::fprintf(tf, "stamp,est_x,est_y,est_yaw,ref_x,ref_y,ref_yaw,err_xy,err_yaw\n");
  double wall_us = 0;
  sm_pose_diag diag{};
  for (int loop = 0; loop < loops; ++loop) {
    begin();
    if (loop == 0) sm_reset_timing(c);
    double cur = -1, before = -1, last_save = -1e9;
    int step = 0;
    const auto w0 = std::chrono::steady_clock::now();
    for (const Rec& r : recs) {
      if (r.tag == 'G') {
        const sm_pose2 p{r.stamp, r.g[0], r.g[1], r.g[2]};
        sm_push_pose(c, &p);
      } else if (r.tag == 'P') {
        before = cur;
        cur = r.stamp;
        sm_proprio p{r.stamp, r.f.data(), int(r.f.size())};
        sm_push_proprio(c, &p);
        if (snap_every > 0 && step % snap_every == 0) {
          int32_t box[4];
          uint64_t ver;
          sm_take_dirty(c, box, &ver);
          sm_snapshot_t* s = nullptr;
          sm_snapshot(c, &s);
          sm_scan2 sc;
          sm_snap_scan(s, &sc);
          sm_snapshot_release(s);
        }
        if (save_every > 0 && !save_dir.empty() && r.stamp - last_save >= save_every) {
          sm_save_dsg(c, save_dir.c_str());
          last_save = r.stamp;
        }
        ++step;
      } else if (r.tag == 'I') {
        const double st = lag && before >= 0 && before < cur ? before : cur;
        sm_image im{st, 0, r.w, r.h, r.rgba.empty() ? nullptr : r.rgba.data(), r.f.data(), r.K[0], r.K[1], r.K[2], r.K[3]};
        sm_detections d{};
        d.stamp = st;
        d.img_w = r.img_w; d.img_h = r.img_h; d.n = r.n;
        d.cls = r.cls.data(); d.score = r.score.data(); d.box = r.box.data();
        d.mask_w = r.mask_w; d.mask_h = r.mask_h; d.mask_sx = r.msx; d.mask_sy = r.msy; d.mask_ox = r.mox; d.mask_oy = r.moy;
        d.mask_bits = r.bits.data();
        sm_push_image_rgb(c, &im, dets_on && r.img_w > 0 ? &d : nullptr, nullptr);
        if (tf && loop == 0) {
          sm_get_pose_diag(c, &diag);
          if (diag.n && diag.stamp == st)
            std::fprintf(tf, "%.4f,%.5f,%.5f,%.6f,%.5f,%.5f,%.6f,%.5f,%.6f\n", st, diag.est[0], diag.est[1], diag.est[2], diag.ref[0],
                         diag.ref[1], diag.ref[2], diag.last_xy, diag.last_yaw);
        }
      }
    }
    wall_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
    if (loop == 0) sm_get_pose_diag(c, &diag);
  }
  if (tf) std::fclose(tf);
  if (!save_dir.empty()) {
    sm_save_stats ss{};
    sm_save_dsg_ex(c, save_dir.c_str(), &ss);
    std::printf("saved %s: %d objects (%.2f ms)\n", save_dir.c_str(), ss.n_objects, ss.total_ms);
  }
  sm_snapshot_t* s = nullptr;
  sm_snapshot(c, &s);
  const sm_status stt = sm_snap_status(s);
  sm_grid g{};
  sm_snap_map(s, &g);
  const sm_pose2 fp = sm_snap_pose(s);
  std::printf("pose %s lag %d policy %d dets %d: objects %d, grid %dx%d, final pose (%.3f, %.3f, %.1f deg)\n", pose.c_str(), lag, policy,
              int(dets_on), stt.n_objects, g.width, g.height, fp.x, fp.y, fp.yaw * 180 / M_PI);
  sm_snapshot_release(s);
  if (diag.n)
    std::printf("drift vs external pose (%d keyframes): max %.1f cm / %.2f deg, rms %.1f cm / %.2f deg, last %.1f cm / %.2f deg\n", diag.n,
                diag.max_xy * 100, diag.max_yaw * 180 / M_PI, diag.rms_xy * 100, diag.rms_yaw * 180 / M_PI, diag.last_xy * 100,
                diag.last_yaw * 180 / M_PI);
  sm_stage_timing T[64];
  const int nt = sm_get_timing(c, T, 64);
  std::printf("%-13s %8s %9s %9s %9s %9s %11s\n", "stage", "n", "mean_us", "p50_us", "p99_us", "max_us", "us/step");
  const double steps = double(nP) * loops;
  for (int k = 0; k < nt; ++k)
    if (T[k].n)
      std::printf("%-13s %8lld %9.2f %9.2f %9.2f %9.1f %11.3f\n", T[k].name, (long long)T[k].n, T[k].mean_us, T[k].p50_us, T[k].p99_us,
                  T[k].max_us, T[k].total_us / steps);
  std::printf("wall: %.1f ms total, %.2f us/step, %.1f us/keyframe (all calls)\n", wall_us / 1e3, wall_us / steps,
              wall_us / std::max(1.0, double(nI) * loops));
  sm_destroy(c);
  return 0;
}
