// 층 3 렌더 비교 도구: 변환기 출력(export/frame_<k>/) 의 공식 RTX 영상과 우리 렌더(같은 이름 규칙) 을 쌍으로 비교한다.
//   render_compare <export 폴더> [--ours <우리 렌더 폴더>] [--pairs off:ours,off:noise0,noise0:noise1,...]
//                  [--roles head,left_wrist,right_wrist] [--csv <파일>] [--bmp <폴더>]
// 파일 이름: frame_<k>/<이름>_rgb_<역할>.npy (uint8 H×W×3), frame_<k>/<이름>_depth_<역할>.npy (float32 H×W).
//   공식: off(공식 한 번), noise0/noise1(같은 상태 다시 그림), ref224_<실행>(같은 행동열 다른 실행의 224 영상).
//   우리: ours(공식과 같은 해상도), ours224(224). 우리 렌더 폴더에 없으면 export 폴더에서 찾는다.
// 검은 프레임(RGB 평균 < 2)이 쌍의 어느 쪽이든 있으면 그 쌍은 빼고 수를 센다.
// 기준 주의: 지금 공식 영상은 Windows RTX 로 뜬 것이다(임시 기준). 대회 환경 Linux GPU 장비에서 다시 떠야 한다.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "tests/render/compare/compare.h"
#include "tests/render/io/npy.h"

namespace fs = std::filesystem;

static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> o;
  size_t p = 0;
  while (p <= s.size()) {
    const size_t e = s.find(c, p);
    o.push_back(s.substr(p, e == std::string::npos ? std::string::npos : e - p));
    if (e == std::string::npos) break;
    p = e + 1;
  }
  return o;
}

static bool find_file(const std::vector<fs::path>& dirs, const std::string& name, fs::path& out) {
  for (auto& d : dirs) {
    if (fs::exists(d / name)) { out = d / name; return true; }
  }
  return false;
}

static void write_bmp(const std::string& path, const std::vector<uint8_t>& rgb, int W, int H) {
  const int row = (W * 3 + 3) & ~3;
  const uint32_t size = 54 + uint32_t(row) * H;
  uint8_t hd[54] = {'B', 'M'};
  auto u32 = [&](int o, uint32_t v) { memcpy(hd + o, &v, 4); };
  auto u16 = [&](int o, uint16_t v) { memcpy(hd + o, &v, 2); };
  u32(2, size); u32(10, 54); u32(14, 40); u32(18, uint32_t(W)); u32(22, uint32_t(H)); u16(26, 1); u16(28, 24); u32(34, uint32_t(row) * H);
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return;
  fwrite(hd, 1, 54, f);
  std::vector<uint8_t> line(size_t(row), 0);
  for (int y = H - 1; y >= 0; --y) {
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 3; ++c) line[size_t(x) * 3 + c] = rgb[(size_t(y) * W + x) * 3 + (2 - c)];
    fwrite(line.data(), 1, size_t(row), f);
  }
  fclose(f);
}

struct Acc {
  int n = 0, black = 0;
  double dmean = 0, emd = 0, ssim = 0, psnr = 0, mad = 0;
  int nd = 0;
  double d_med = 0, d_p99 = 0, d_1cm = 0, d_1mm = 0, d_mismatch = 0;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: render_compare <export> [--ours dir] [--pairs a:b,...] [--roles r,...] [--csv f] [--bmp dir]\n");
    return 2;
  }
  fs::path exp = argv[1], ours;
  std::string pairs_s = "off:ours,off:noise0,noise0:noise1", roles_s = "head,left_wrist,right_wrist", csv_path, bmp_dir;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--ours") && i + 1 < argc) ours = argv[++i];
    else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs_s = argv[++i];
    else if (!strcmp(argv[i], "--roles") && i + 1 < argc) roles_s = argv[++i];
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--bmp") && i + 1 < argc) bmp_dir = argv[++i];
  }
  const auto roles = split(roles_s, ',');
  std::vector<std::pair<std::string, std::string>> pairs;
  for (auto& p : split(pairs_s, ',')) {
    auto ab = split(p, ':');
    if (ab.size() == 2) pairs.push_back({ab[0], ab[1]});
  }
  std::vector<fs::path> frames;
  for (auto& e : fs::directory_iterator(exp))
    if (e.is_directory() && e.path().filename().string().rfind("frame_", 0) == 0) frames.push_back(e.path());
  std::sort(frames.begin(), frames.end());
  if (!bmp_dir.empty()) fs::create_directories(bmp_dir);
  FILE* csv = csv_path.empty() ? nullptr : fopen(csv_path.c_str(), "w");
  if (csv)
    fprintf(csv, "frame,role,a,b,W,H,black,mean_a,mean_b,dmean_r,dmean_g,dmean_b,emd_r,emd_g,emd_b,ssim,psnr,mad,"
                 "d_both,d_a_only,d_b_only,d_bit_equal,d_med,d_p90,d_p99,d_max,d_rel_med,d_rel_p99,d_1mm,d_1cm,d_1pct\n");
  std::map<std::string, Acc> acc;
  printf("%-10s %-11s %-22s | %-7s %-7s %-6s %-5s %-6s | %-8s %-8s %-8s %-7s\n", "frame", "role", "pair", "meanA", "meanB", "EMD",
         "SSIM", "PSNR", "d_med_mm", "d_p99_mm", "d_1cm%", "d_miss");
  for (auto& fd : frames) {
    const std::string fk = fd.filename().string().substr(6);
    std::vector<fs::path> dirs = {fd};
    if (!ours.empty()) dirs.insert(dirs.begin(), ours / fd.filename());
    for (auto& role : roles)
      for (auto& pr : pairs) {
        fs::path pa, pb, da, db;
        const bool ha = find_file(dirs, pr.first + "_rgb_" + role + ".npy", pa), hb = find_file(dirs, pr.second + "_rgb_" + role + ".npy", pb);
        if (!ha || !hb) continue;
        npy::Array A = npy::load(pa.string()), B = npy::load(pb.string());
        int W = int(A.shape[1]), H = int(A.shape[0]);
        std::vector<uint8_t> brs;
        const uint8_t* bp = B.as<uint8_t>();
        if (B.shape[0] != A.shape[0] || B.shape[1] != A.shape[1]) {
          brs = rcmp::resize_nearest(B.as<uint8_t>(), int(B.shape[1]), int(B.shape[0]), W, H);
          bp = brs.data();
        }
        const std::string key = role + " " + pr.first + ":" + pr.second;
        Acc& ac = acc[key];
        const bool black = rcmp::is_black(A.as<uint8_t>(), int64_t(W) * H) || rcmp::is_black(bp, int64_t(W) * H);
        if (black) {
          ac.black++;
          printf("%-10s %-11s %-22s | 검은 프레임 -> 뺌\n", fk.c_str(), role.c_str(), (pr.first + ":" + pr.second).c_str());
          if (csv) fprintf(csv, "%s,%s,%s,%s,%d,%d,1\n", fk.c_str(), role.c_str(), pr.first.c_str(), pr.second.c_str(), W, H);
          continue;
        }
        const rcmp::RgbStats r = rcmp::rgb_compare(A.as<uint8_t>(), bp, W, H);
        rcmp::DepthStats d;
        bool hd = find_file(dirs, pr.first + "_depth_" + role + ".npy", da) && find_file(dirs, pr.second + "_depth_" + role + ".npy", db);
        if (hd) {
          npy::Array DA = npy::load(da.string()), DB = npy::load(db.string());
          if (DA.size() == DB.size()) d = rcmp::depth_compare(DA.as<float>(), DB.as<float>(), DA.size());
          else hd = false;
        }
        const double ma = (r.mean_a[0] + r.mean_a[1] + r.mean_a[2]) / 3, mb = (r.mean_b[0] + r.mean_b[1] + r.mean_b[2]) / 3;
        const double emd = (r.emd[0] + r.emd[1] + r.emd[2]) / 3;
        printf("%-10s %-11s %-22s | %7.2f %7.2f %6.2f %5.3f %6.2f", fk.c_str(), role.c_str(), (pr.first + ":" + pr.second).c_str(), ma, mb,
               emd, r.ssim, r.psnr);
        if (hd)
          printf(" | %8.3f %8.3f %7.2f %7lld", d.med * 1e3, d.p99 * 1e3, d.within_1cm * 100, (long long)(d.a_only + d.b_only));
        printf("\n");
        ac.n++;
        ac.dmean += std::fabs(ma - mb);
        ac.emd += emd;
        ac.ssim += r.ssim;
        ac.psnr += r.psnr;
        ac.mad += r.mad;
        if (hd) {
          ac.nd++;
          ac.d_med += d.med;
          ac.d_p99 += d.p99;
          ac.d_1cm += d.within_1cm;
          ac.d_1mm += d.within_1mm;
          ac.d_mismatch += double(d.a_only + d.b_only) / double(d.n);
        }
        if (csv) {
          fprintf(csv, "%s,%s,%s,%s,%d,%d,0,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.3f,%.4f", fk.c_str(), role.c_str(),
                  pr.first.c_str(), pr.second.c_str(), W, H, ma, mb, r.mean_a[0] - r.mean_b[0], r.mean_a[1] - r.mean_b[1],
                  r.mean_a[2] - r.mean_b[2], r.emd[0], r.emd[1], r.emd[2], r.ssim, r.psnr, r.mad);
          if (hd)
            fprintf(csv, ",%lld,%lld,%lld,%lld,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.5f,%.5f,%.5f\n", (long long)d.both, (long long)d.a_only,
                    (long long)d.b_only, (long long)d.bit_equal, d.med, d.p90, d.p99, d.maxd, d.rel_med, d.rel_p99, d.within_1mm,
                    d.within_1cm, d.within_1pct);
          else
            fprintf(csv, "\n");
        }
        if (!bmp_dir.empty()) {
          std::vector<uint8_t> img(size_t(W) * 3 * H * 3);
          for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
              for (int c = 0; c < 3; ++c) {
                const size_t s = (size_t(y) * W + x) * 3 + c;
                const int p = A.as<uint8_t>()[s], q = bp[s];
                img[(size_t(y) * W * 3 + x) * 3 + c] = uint8_t(p);
                img[(size_t(y) * W * 3 + W + x) * 3 + c] = uint8_t(q);
                const int dv = 128 + 2 * (p - q);
                img[(size_t(y) * W * 3 + 2 * W + x) * 3 + c] = uint8_t(dv < 0 ? 0 : (dv > 255 ? 255 : dv));
              }
          write_bmp(bmp_dir + "/" + fk + "_" + role + "_" + pr.first + "_" + pr.second + ".bmp", img, W * 3, H);
        }
      }
  }
  printf("\n== 요약 (검은 프레임 뺀 평균). 기준: Windows RTX(임시) -- Linux GPU 장비에서 다시 떠야 함\n");
  printf("%-40s %4s %5s | %-7s %-6s %-6s %-6s %-6s | %-8s %-8s %-7s %-7s %-7s\n", "role pair", "n", "black", "|dmean|", "EMD", "SSIM", "PSNR",
         "MAD", "d_med_mm", "d_p99_mm", "d_1mm%", "d_1cm%", "miss%");
  for (auto& kv : acc) {
    const Acc& a = kv.second;
    if (!a.n) {
      printf("%-40s %4d %5d | (전부 검음)\n", kv.first.c_str(), a.n, a.black);
      continue;
    }
    printf("%-40s %4d %5d | %7.3f %6.3f %6.4f %6.2f %6.3f", kv.first.c_str(), a.n, a.black, a.dmean / a.n, a.emd / a.n, a.ssim / a.n,
           a.psnr / a.n, a.mad / a.n);
    if (a.nd)
      printf(" | %8.3f %8.3f %7.2f %7.2f %7.3f", a.d_med / a.nd * 1e3, a.d_p99 / a.nd * 1e3, a.d_1mm / a.nd * 100, a.d_1cm / a.nd * 100,
             a.d_mismatch / a.nd * 100);
    printf("\n");
  }
  if (csv) fclose(csv);
  return 0;
}
