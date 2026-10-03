// 벽 2D 포팅 시험: 파이썬(viewer/walls2d.py)이 같은 격자에서 낸 값과 비교 + 속도.
// 사용: test_walls <cells.bin> <ref.json>  (인자 없으면 합성 격자로 속도만)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "scenemap/walls.hpp"
using namespace scenemap;

int main(int argc, char** argv) {
  int w = 500, h = 400;
  if (argc >= 5) { w = std::atoi(argv[3]); h = std::atoi(argv[4]); }
  std::vector<int8_t> cells(size_t(w) * h, 0);
  std::vector<double> ref_segs, ref_vec;
  double pose[3] = {4.0, 5.0, 0.7};
  if (argc >= 8) { pose[0] = std::atof(argv[5]); pose[1] = std::atof(argv[6]); pose[2] = std::atof(argv[7]); }
  if (argc >= 3) {
    std::ifstream f(argv[1], std::ios::binary);
    f.read(reinterpret_cast<char*>(cells.data()), cells.size());
    std::ifstream j(argv[2]);
    std::stringstream ss; ss << j.rdbuf();
    std::string s = ss.str();
    auto nums = [&](const std::string& key, std::vector<double>& out) {
      size_t p = s.find("\"" + key + "\""); p = s.find('[', p);
      int depth = 0; size_t q = p;
      for (; q < s.size(); ++q) { if (s[q] == '[') ++depth; if (s[q] == ']' && --depth == 0) break; }
      std::string body = s.substr(p, q - p + 1);
      for (char& c : body) if (c == '[' || c == ']' || c == ',') c = ' ';
      std::stringstream bs(body); double v; while (bs >> v) out.push_back(v);
    };
    nums("segs", ref_segs); nums("vec", ref_vec);
  }
  WallGrid g{cells.data(), w, h, 0.05, 0.0, 0.0};
  auto segs = wallSegments(g);
  float vec[kStateLen];
  wallStateVector(g, segs, pose, vec);
  int bad = 0;
  if (!ref_segs.empty()) {
    if (segs.size() * 4 != ref_segs.size()) { std::printf("FAIL segment count %zu vs %zu\n", segs.size(), ref_segs.size() / 4); ++bad; }
    else for (size_t i = 0; i < segs.size(); ++i) {
      const double a[4] = {segs[i].ax, segs[i].ay, segs[i].bx, segs[i].by};
      for (int q = 0; q < 4; ++q) if (std::abs(a[q] - ref_segs[4 * i + q]) > 1e-6) { std::printf("FAIL seg %zu[%d] %f vs %f\n", i, q, a[q], ref_segs[4 * i + q]); ++bad; }
    }
    // 광선: 파이썬은 res/2 간격 샘플링, 여기는 DDA — 칸 하나(0.05 m) 안에서 같으면 통과
    for (int i = 0; i < kStateLen; ++i) {
      double tol = i < kSectors ? (0.06 / kMaxRange) : 1e-4;
      if (std::abs(vec[i] - ref_vec[i]) > tol) { std::printf("FAIL vec[%d] %f vs %f\n", i, vec[i], ref_vec[i]); ++bad; }
    }
  }
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now(); int N = 50;
  for (int i = 0; i < N; ++i) segs = wallSegments(g);
  auto t1 = clk::now();
  int M = 20000;
  for (int i = 0; i < M; ++i) { pose[0] = 4.0 + (i % 100) * 0.01; wallStateVector(g, segs, pose, vec); }
  auto t2 = clk::now();
  std::printf("grid %dx%d: wallSegments %.3f ms, wallStateVector %.2f us (segments %zu)\n", w, h,
              std::chrono::duration<double, std::milli>(t1 - t0).count() / N,
              std::chrono::duration<double, std::micro>(t2 - t1).count() / M, segs.size());

  // 증분: 몇 칸을 바꾸고 바뀐 행만 알려 줘도 처음부터 계산한 것과 같아야 한다 + 속도
  {
    WallExtractor ex;
    ex.update(g, 0, -1);
    std::vector<int8_t> c2 = cells;
    WallGrid g2{c2.data(), w, h, 0.05, 0.0, 0.0};
    unsigned seed = 12345;
    for (int round = 0; round < 200 && !bad; ++round) {
      seed = seed * 1664525u + 1013904223u;
      int y0 = (seed >> 8) % (h - 12), x0 = (seed >> 3) % (w - 40);
      int8_t val = (round & 1) ? 100 : 0;
      for (int y = y0; y < y0 + 3 + int(seed % 8); ++y)
        for (int x = x0; x < x0 + 30; ++x) c2[size_t(y) * w + x] = val;
      auto inc = ex.update(g2, y0, y0 + 10);   // 복사: 다음 update 가 덮어쓴다
      auto full = wallSegments(g2);
      bool same = inc.size() == full.size();
      for (size_t i = 0; same && i < inc.size(); ++i)
        same = inc[i].ax == full[i].ax && inc[i].ay == full[i].ay && inc[i].bx == full[i].bx && inc[i].by == full[i].by;
      if (!same) { std::printf("FAIL incremental round %d (%zu vs %zu)\n", round, inc.size(), full.size()); ++bad; }
    }
    c2 = cells;   // 속도는 깨끗한 지도에서(위 시험이 흔적을 많이 남겼다)
    ex.update(g2, 0, -1);
    auto ti0 = clk::now();
    const int K = 20000;
    for (int i = 0; i < K; ++i) { int y0 = (i * 7) % (h - 12); c2[size_t(y0) * w + 5] ^= 1; ex.update(g2, y0, y0 + 10); }
    auto ti1 = clk::now();
    std::printf("incremental update (11 dirty rows): %.2f us\n", std::chrono::duration<double, std::micro>(ti1 - ti0).count() / K);
  }
  std::printf(bad ? "FAIL (%d)\n" : "OK\n", bad);
  return bad ? 1 : 0;
}
