// sgs_play — 미리 돌려 기록한 scenemap sgview 스트림(stream.sgs, realbag_run --sg · og2sg 가 씀)을 벽시계에 맞춰 sgview(--ingest)로 다시 보낸다.
// 파이프라인(검출·SLAM·물체 기억)은 이미 끝났으므로 재생 속도가 검출에 묶이지 않는다. 자세는 기록된 자세(바퀴 오도메트리 + 맞추기,
// 오도메트리 메시지마다)를 시각으로 보간해 pose_hz(기본 60 Hz)로 보낸다 — 화면 보간용일 뿐 지도·물체는 기록 그대로.
//
//   sgs_play <stream.sgs> <host:port> [--rate 1] [--pose-hz 60] [--loop] [--hold 3] [--from S]
//
// 형식: 파일 "SGS1" 다음 [f64 t][u32 len][u8 type][payload] … (type 1 POSE f64 stamp,x,y,yaw · 2 MAP_RECT · 3 VIEW · 4 JOINTS),
// 선(wire) = [u32 len][u8 type][payload](scenemap stream.hpp). 5 s 마다 보낸 프레임 주기(종류별 /s)를 stderr 에.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct Fr { double t; uint8_t type; std::vector<uint8_t> pl; };
struct Pose { double t, x, y, yaw; };

static bool sendAll(int fd, const void* p, size_t n) {
  const uint8_t* q = static_cast<const uint8_t*>(p);
  while (n) {
    const ssize_t k = send(fd, q, n, MSG_NOSIGNAL);
    if (k <= 0) return false;
    q += k;
    n -= size_t(k);
  }
  return true;
}
static bool sendFrame(int fd, uint8_t type, const uint8_t* pl, uint32_t len) {
  uint8_t hd[5];
  std::memcpy(hd, &len, 4);
  hd[4] = type;
  return sendAll(fd, hd, 5) && sendAll(fd, pl, len);
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: sgs_play <stream.sgs> <host:port> [--rate 1] [--pose-hz 60] [--loop] [--hold 3] [--from S]\n"); return 2; }
  double rate = 1, pose_hz = 60, hold = 3, from = 0;
  bool loop = false;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rate" && i + 1 < argc) rate = std::stod(argv[++i]);
    else if (a == "--pose-hz" && i + 1 < argc) pose_hz = std::stod(argv[++i]);
    else if (a == "--hold" && i + 1 < argc) hold = std::stod(argv[++i]);
    else if (a == "--from" && i + 1 < argc) from = std::stod(argv[++i]);
    else if (a == "--loop") loop = true;
    else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
  }
  FILE* f = std::fopen(argv[1], "rb");
  char magic[4];
  if (!f || std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "SGS1", 4)) { std::fprintf(stderr, "not an SGS1 file: %s\n", argv[1]); return 1; }
  std::vector<Fr> frames;   // 자세 말고 전부
  std::vector<Pose> poses;
  for (;;) {
    double t;
    uint8_t hd[5];
    if (std::fread(&t, 8, 1, f) != 1 || std::fread(hd, 1, 5, f) != 5) break;
    uint32_t len;
    std::memcpy(&len, hd, 4);
    Fr fr{t, hd[4], std::vector<uint8_t>(len)};
    if (len && std::fread(fr.pl.data(), 1, len, f) != len) break;
    if (fr.type == 1 && len >= 32) {
      Pose p;
      std::memcpy(&p, fr.pl.data(), 32);
      if (poses.empty() || p.t > poses.back().t + 1e-6) poses.push_back(p);
      else poses.back() = p;   // 같은 시각이면 나중 것(맞춘 자세)
    } else {
      frames.push_back(std::move(fr));
    }
  }
  std::fclose(f);
  if (frames.empty() && poses.empty()) { std::fprintf(stderr, "empty stream\n"); return 1; }
  double t0 = 1e300, t1 = -1e300;
  for (auto& fr : frames) { t0 = std::min(t0, fr.t); t1 = std::max(t1, fr.t); }
  if (!poses.empty()) { t0 = std::min(t0, poses.front().t); t1 = std::max(t1, poses.back().t); }
  std::fprintf(stderr, "sgs_play: %zu frames + %zu poses, %.1f..%.1f s, rate %.2fx, pose %.0f Hz\n", frames.size(), poses.size(), t0, t1, rate, pose_hz);
  // 첫 자세 앞의 기록 프레임(지도·요약) 순서는 그대로, 자세 보간
  auto poseAt = [&](double t) {
    if (t <= poses.front().t) return poses.front();
    if (t >= poses.back().t) return poses.back();
    auto it = std::lower_bound(poses.begin(), poses.end(), t, [](const Pose& p, double tt) { return p.t < tt; });
    const Pose& b = *it;
    const Pose& a = *(it - 1);
    const double w = (t - a.t) / std::max(1e-9, b.t - a.t);
    const double dy = std::atan2(std::sin(b.yaw - a.yaw), std::cos(b.yaw - a.yaw));
    return Pose{t, a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.yaw + dy * w};
  };
  const std::string hp = argv[2];
  const size_t colon = hp.rfind(':');
  do {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(std::stoi(hp.substr(colon + 1))));
    inet_pton(AF_INET, hp.substr(0, colon).c_str(), &a.sin_addr);
    if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a)) { std::perror("connect"); return 1; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    const auto w0 = std::chrono::steady_clock::now();
    const double ts = std::max(t0, from);
    size_t fi = 0;
    // --from: 그 앞 기록 프레임(지도·요약)은 한꺼번에 보냄
    while (fi < frames.size() && frames[fi].t < ts) { sendFrame(fd, frames[fi].type, frames[fi].pl.data(), uint32_t(frames[fi].pl.size())); ++fi; }
    double next_pose = ts, last_rep = 0;   // 다음 자세를 보낼 자료 시각
    uint64_t cnt[8] = {}, cnt_rep[8] = {};
    bool ok = true;
    while (ok) {
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
      const double now = ts + wall * rate;
      while (ok && fi < frames.size() && frames[fi].t <= now) {
        ok = sendFrame(fd, frames[fi].type, frames[fi].pl.data(), uint32_t(frames[fi].pl.size()));
        ++cnt[frames[fi].type & 7];
        ++fi;
      }
      if (!poses.empty() && now >= next_pose) {
        const Pose p = poseAt(std::min(now, t1));
        ok = ok && sendFrame(fd, 1, reinterpret_cast<const uint8_t*>(&p), 32);
        ++cnt[1];
        next_pose += rate / pose_hz;   // 벽시계 1/pose_hz 마다(밀리면 따라잡지 않고 지금부터)
        if (next_pose < now) next_pose = now + rate / pose_hz;
      }
      if (wall - last_rep >= 5.0) {
        const double dt = wall - last_rep;
        std::fprintf(stderr, "[sgs_play] t %.1f/%.1f s  sent/s (wall): pose %.1f  map %.1f  view %.1f  joints %.1f\n", now, t1,
                     (cnt[1] - cnt_rep[1]) / dt, (cnt[2] - cnt_rep[2]) / dt, (cnt[3] - cnt_rep[3]) / dt, (cnt[4] - cnt_rep[4]) / dt);
        std::copy(cnt, cnt + 8, cnt_rep);
        last_rep = wall;
      }
      if (now >= t1 && fi >= frames.size()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    std::fprintf(stderr, "[sgs_play] done: %.1f s wall, pose %llu (%.1f/s), map %llu (%.1f/s), view %llu (%.1f/s)\n", wall, (unsigned long long)cnt[1],
                 cnt[1] / wall, (unsigned long long)cnt[2], cnt[2] / wall, (unsigned long long)cnt[3], cnt[3] / wall);
    std::this_thread::sleep_for(std::chrono::duration<double>(hold));
    close(fd);
  } while (loop);
  return 0;
}
