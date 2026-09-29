// 렌더 시험 — 공식 기록 장면 (render_capture.py -> convert_scene.py 결과).
//   test_render_scene <폴더(scene.rsc, frame_*.rfr)> [--gpu 0|1] [--envs E] [--check K] [--reps R] [--res N(모든 카메라 N×N)]
//                     [--spp S] [--bounces B] [--exposure X] [--ambient a] [--ao 거리] [--tonemap T] [--white W] [--lights 0|1] [--out 폴더 [--ppm]] [--official 0(공식 비교 건너뜀)] [--fit(노출 맞추기)] [--dome-tex r,g,b(하늘 텍스처 평균)] [--frames 0,10]
// 1) 층 1 vs 공식 RTX: depth 는 픽셀마다 차이 분포(공식 depth_linear), RGB 는 채널 평균·히스토그램·SSIM 을 공식 자신의
//    잡음(같은 상태에서 다시 그린 장)과 나란히.
// 2) 층 1 = 층 2: 판 e 가 프레임 (e mod 프레임 수) 를 그린다. 판 0..K-1 을 층 1 로 그려 depth 비트·RGB 바이트 전부 비교.
// 3) 처리량: 판 E × 카메라 3 대 GPU 시간.
// 4) 카메라를 링크에 붙여(RenderBatch, 포팅 평가기 입구) 층1=층2 와 처리량.
#include <cuda_runtime.h>
#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/render/render_batch.cuh"
#include "tests/render/io/npy.h"
#include <sys/stat.h>
#include <xmmintrin.h>

using namespace eng;
using namespace eng::rnd;

static const char* kRoles[3] = {"head", "left_wrist", "right_wrist"};

// ---- 영상 통계 ----
static void luma(const uint8_t* rgb, int stride, int n, std::vector<double>& y) {
  y.resize(n);
  for (int i = 0; i < n; ++i) y[i] = 0.299 * rgb[stride * i] + 0.587 * rgb[stride * i + 1] + 0.114 * rgb[stride * i + 2];
}
// SSIM (Wang 2004, 가우시안 11×11 σ=1.5, 밝기 0..255)
static double ssim(const std::vector<double>& a, const std::vector<double>& b, int w, int h) {
  const int R = 5;
  double g[11], gs = 0;
  for (int i = -R; i <= R; ++i) gs += (g[i + R] = std::exp(-(i * i) / (2 * 1.5 * 1.5)));
  for (double& x : g) x /= gs;
  auto blur = [&](const std::vector<double>& in) {
    std::vector<double> t(in.size()), o(in.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double s = 0;
        for (int k = -R; k <= R; ++k) s += g[k + R] * in[y * w + std::min(std::max(x + k, 0), w - 1)];
        t[y * w + x] = s;
      }
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double s = 0;
        for (int k = -R; k <= R; ++k) s += g[k + R] * t[std::min(std::max(y + k, 0), h - 1) * w + x];
        o[y * w + x] = s;
      }
    return o;
  };
  const size_t n = size_t(w) * h;
  std::vector<double> aa(n), bb(n), ab(n);
  for (size_t i = 0; i < n; ++i) { aa[i] = a[i] * a[i]; bb[i] = b[i] * b[i]; ab[i] = a[i] * b[i]; }
  auto ma = blur(a), mb = blur(b), saa = blur(aa), sbb = blur(bb), sab = blur(ab);
  const double C1 = 6.5025, C2 = 58.5225;
  double acc = 0;
  for (size_t i = 0; i < n; ++i) {
    const double va = saa[i] - ma[i] * ma[i], vb = sbb[i] - mb[i] * mb[i], cv = sab[i] - ma[i] * mb[i];
    acc += ((2 * ma[i] * mb[i] + C1) * (2 * cv + C2)) / ((ma[i] * ma[i] + mb[i] * mb[i] + C1) * (va + vb + C2));
  }
  return acc / n;
}
struct RgbStat {
  double mean[3];
  double hist[3][32];
};
static RgbStat rgb_stat(const uint8_t* rgb, int stride, int n) {
  RgbStat s{};
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) {
      s.mean[k] += rgb[stride * i + k];
      s.hist[k][rgb[stride * i + k] >> 3] += 1;
    }
  for (int k = 0; k < 3; ++k) {
    s.mean[k] /= n;
    for (double& x : s.hist[k]) x /= n;
  }
  return s;
}
// 히스토그램 거리: 누적분포 차의 L1 (1차원 EMD, 칸 = 8 단계) × 8 = 밝기 단위
static double hist_emd(const RgbStat& a, const RgbStat& b) {
  double t = 0;
  for (int k = 0; k < 3; ++k) {
    double ca = 0, cb = 0;
    for (int i = 0; i < 32; ++i) { ca += a.hist[k][i]; cb += b.hist[k][i]; t += std::fabs(ca - cb) * 8; }
  }
  return t / 3;
}
static double mean_abs(const uint8_t* a, int sa, const uint8_t* b, int sb, int n) {
  double t = 0;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) t += std::fabs(double(a[sa * i + k]) - double(b[sb * i + k]));
  return t / (3.0 * n);
}
static void write_ppm(const std::string& fn, const uint8_t* rgb, int stride, int w, int h) {
  FILE* f = fopen(fn.c_str(), "wb");
  if (!f) return;
  fprintf(f, "P6 %d %d 255\n", w, h);
  for (int i = 0; i < w * h; ++i) fwrite(rgb + stride * i, 1, 3, f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "사용: test_render_scene <폴더> ...\n"); return 2; }
  const std::string dir = argv[1];
  int use_gpu = 1, envs = 64, check = 3, reps = 5, res = 0, spp = -1, bounces = -1, tonemap = -1, lights = 1, ppm = 0, official = 1, fit = 0;
  float exposure = -1, ambient = -1, ao = -1, white = -1, dome_tex[3] = {1.0f, 1.0f, 1.0f};
  std::string out;
  std::vector<int> only_frames;
  for (int i = 2; i < argc; ++i) {
    auto nx = [&]() { return argv[++i]; };
    if (!strcmp(argv[i], "--gpu") && i + 1 < argc) use_gpu = atoi(nx());
    else if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(nx());
    else if (!strcmp(argv[i], "--check") && i + 1 < argc) check = atoi(nx());
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(nx());
    else if (!strcmp(argv[i], "--res") && i + 1 < argc) res = atoi(nx());
    else if (!strcmp(argv[i], "--spp") && i + 1 < argc) spp = atoi(nx());
    else if (!strcmp(argv[i], "--bounces") && i + 1 < argc) bounces = atoi(nx());
    else if (!strcmp(argv[i], "--tonemap") && i + 1 < argc) tonemap = atoi(nx());
    else if (!strcmp(argv[i], "--exposure") && i + 1 < argc) exposure = float(atof(nx()));
    else if (!strcmp(argv[i], "--ambient") && i + 1 < argc) ambient = float(atof(nx()));
    else if (!strcmp(argv[i], "--lights") && i + 1 < argc) lights = atoi(nx());
    else if (!strcmp(argv[i], "--ao") && i + 1 < argc) ao = float(atof(nx()));
    else if (!strcmp(argv[i], "--white") && i + 1 < argc) white = float(atof(nx()));
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = nx();
    else if (!strcmp(argv[i], "--ppm")) ppm = 1;
    else if (!strcmp(argv[i], "--official") && i + 1 < argc) official = atoi(nx());
    else if (!strcmp(argv[i], "--fit")) fit = 1;
    else if (!strcmp(argv[i], "--dome-tex") && i + 1 < argc) sscanf(nx(), "%f,%f,%f", &dome_tex[0], &dome_tex[1], &dome_tex[2]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
      char* s = nx();
      for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) only_frames.push_back(atoi(t));
    }
  }
  auto t0 = std::chrono::steady_clock::now();
  HostScene H;
  if (!load_scene(dir + "/scene.rsc", H)) { fprintf(stderr, "장면 못 읽음\n"); return 1; }
  if (spp >= 0) H.sp.spp = spp;
  if (bounces >= 0) H.sp.bounces = bounces;
  if (tonemap >= 0) H.sp.tonemap = tonemap;
  if (exposure >= 0) H.sp.exposure = exposure;
  if (ambient >= 0) H.sp.ambient[0] = H.sp.ambient[1] = H.sp.ambient[2] = ambient;
  if (ao >= 0) H.sp.ao_range = ao;
  if (white >= 0) H.sp.white_scale = white;
  for (int k = 0; k < 3; ++k) H.sp.dome[k] *= dome_tex[k];  // 하늘 텍스처 평균(선형) — sky.jpg 위 반구 평균 0.247,0.350,0.778
  if (!lights) H.lights.clear();
  const double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const SceneView SV = H.view();
  printf("장면: 인스턴스 %d, 기하 %zu, 삼각형 %zu, BLAS 노드 %zu, 기준 prim %d, 재질 %zu, 텍스처 %zu (%.0f MiB), 조명 %zu | "
         "읽기+BLAS 굽기 %.2f s\n",
         SV.n_inst, H.geoms.size(), H.tris.size(), H.blas_nodes.size(), SV.n_anchor, H.mats.size(), H.texs.size(),
         H.texels.size() * 4.0 / 1048576.0, H.lights.size(), t_load);
  printf("음영: spp %d, 튕김 %d, 그림자 광선 %d, 노출 %g, 주변광 %g, AO %g m, 톤매핑 %d (흰색 %g)\n", H.sp.spp, H.sp.bounces,
         H.sp.shadow_lights, H.sp.exposure, H.sp.ambient[0], H.sp.ao_range, H.sp.tonemap, H.sp.white_scale);
  // 프레임
  std::vector<std::string> fnames;
  if (DIR* dp = opendir(dir.c_str())) {
    while (dirent* de = readdir(dp)) {
      std::string n = de->d_name;
      if (n.rfind("frame_", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".rfr") fnames.push_back(n);
    }
    closedir(dp);
  }
  std::sort(fnames.begin(), fnames.end());
  std::vector<HostFrame> frames;
  for (auto& n : fnames) {
    HostFrame F;
    if (!load_frame(dir + "/" + n, F)) continue;
    if (!only_frames.empty() && std::find(only_frames.begin(), only_frames.end(), int(F.step)) == only_frames.end()) continue;
    if (res > 0)
      for (auto& c : F.cams) { c.w = res; c.h = res; }
    frames.push_back(std::move(F));
  }
  printf("프레임 %zu 개\n", frames.size());
  if (frames.empty()) return 1;

  // ---- 1) 층 1 vs 공식
  printf("\n== 층 1 vs 공식 RTX (depth: |차| 분포, RGB: 채널 평균 차 / 히스토그램 EMD / SSIM / 평균 |차|; 괄호 = 공식 자신(다시 그림) 잡음) ==\n");
  double sum_t1 = 0;
  uint64_t n_pix_t1 = 0;
  for (auto& F : frames) {
    if (!official) break;
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < SV.n_anchor && a < int(F.anchor.size()); ++a) HE.anchor[a] = F.anchor[a];
    for (size_t w = 0; w < HE.vis.size() && w < F.vis.size(); ++w) HE.vis[w] = F.vis[w];
    HE.build(SV);
    for (int c = 0; c < 3 && c < int(F.cams.size()); ++c) {
      const Camera& cm = F.cams[c];
      const int n = cm.w * cm.h;
      std::vector<float> dep(n);
      std::vector<uint8_t> rgb(size_t(n) * 3);
      auto ta = std::chrono::steady_clock::now();
      render_host(SV, HE, cm, 0, c, int(F.step), dep.data(), rgb.data());
      sum_t1 += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
      n_pix_t1 += n;
      const std::string role = kRoles[c];
      const Image* od = F.find("img::" + role + "::depth_linear");
      const Image* orgb = F.find("img::" + role + "::rgb");
      const Image* n0 = F.find("noise0::" + role + "::rgb");
      printf("스텝 %4" PRId64 " %-11s %dx%d", F.step, kRoles[c], cm.w, cm.h);
      if (od && od->h == cm.h && od->w == cm.w) {
        const float* o = od->f32();
        std::vector<double> diffs;
        uint64_t both = 0, only_off = 0, only_our = 0, none = 0, within1mm = 0, within1cm = 0, rel1e4 = 0;
        for (int i = 0; i < n; ++i) {
          const float ov = o[i * od->c];
          const bool oh = std::isfinite(ov) && ov > 0.0f && ov < cm.zfar;
          const bool uh = dep[i] > 0.0f;
          if (oh && uh) {
            ++both;
            const double d = std::fabs(double(dep[i]) - double(ov));
            diffs.push_back(d);
            within1mm += d <= 1e-3;
            within1cm += d <= 1e-2;
            rel1e4 += d <= 1e-4 * ov;
          } else if (oh) ++only_off;
          else if (uh) ++only_our;
          else ++none;
        }
        std::sort(diffs.begin(), diffs.end());
        auto q = [&](double p) { return diffs.empty() ? 0.0 : diffs[std::min(diffs.size() - 1, size_t(p * diffs.size()))]; };
        printf(" | depth 둘다 %.1f%% 공식만 %.2f%% 우리만 %.2f%% | |차| 중앙 %.2e p90 %.2e p99 %.2e 최대 %.2e | 상대1e-4 %.1f%% 1mm %.1f%% 1cm %.1f%%",
               100.0 * both / n, 100.0 * only_off / n, 100.0 * only_our / n, q(0.5), q(0.9), q(0.99),
               diffs.empty() ? 0.0 : diffs.back(), 100.0 * rel1e4 / std::max<uint64_t>(both, 1),
               100.0 * within1mm / std::max<uint64_t>(both, 1), 100.0 * within1cm / std::max<uint64_t>(both, 1));
      }
      if (orgb && orgb->h == cm.h && orgb->w == cm.w) {
        const int oc = int(orgb->c);
        const RgbStat so = rgb_stat(orgb->data.data(), oc, n), su = rgb_stat(rgb.data(), 3, n);
        std::vector<double> yo, yu;
        luma(orgb->data.data(), oc, n, yo);
        luma(rgb.data(), 3, n, yu);
        const bool black = so.mean[0] + so.mean[1] + so.mean[2] < 3.0;
        printf("\n        RGB 평균 공식 %.1f/%.1f/%.1f 우리 %.1f/%.1f/%.1f%s | EMD %.1f | SSIM %.3f | 평균|차| %.1f", so.mean[0],
               so.mean[1], so.mean[2], su.mean[0], su.mean[1], su.mean[2], black ? " (공식 검은 프레임)" : "",
               hist_emd(so, su), ssim(yo, yu, cm.w, cm.h), mean_abs(orgb->data.data(), oc, rgb.data(), 3, n));
        if (n0 && n0->h == cm.h) {
          const RgbStat s0 = rgb_stat(n0->data.data(), int(n0->c), n);
          std::vector<double> y0;
          luma(n0->data.data(), int(n0->c), n, y0);
          printf(" (공식 잡음: 평균 차 %.2f/%.2f/%.2f EMD %.2f SSIM %.3f 평균|차| %.2f)", s0.mean[0] - so.mean[0],
                 s0.mean[1] - so.mean[1], s0.mean[2] - so.mean[2], hist_emd(so, s0), ssim(yo, y0, cm.w, cm.h),
                 mean_abs(orgb->data.data(), oc, n0->data.data(), int(n0->c), n));
        }
      }
      printf("\n");
      if (!out.empty()) {  // docs 14.4 약속: <out>/frame_<k>/ours[224]_rgb_<역할>.npy (u8 H×W×3), ours[224]_depth_<역할>.npy (f32 H×W, 못 맞춤 0)
        char dn[512], fn[600];
        snprintf(dn, sizeof dn, "%s/frame_%04d", out.c_str(), int(F.step));
        mkdir(out.c_str(), 0755);
        mkdir(dn, 0755);
        const char* tag = (res == 224) ? "ours224" : "ours";
        snprintf(fn, sizeof fn, "%s/%s_rgb_%s.npy", dn, tag, kRoles[c]);
        npy::save(fn, "|u1", {cm.h, cm.w, 3}, rgb.data());
        snprintf(fn, sizeof fn, "%s/%s_depth_%s.npy", dn, tag, kRoles[c]);
        npy::save(fn, "<f4", {cm.h, cm.w}, dep.data());
        if (ppm) {
          snprintf(fn, sizeof fn, "%s/%s_rgb_%s.ppm", dn, tag, kRoles[c]);
          write_ppm(fn, rgb.data(), 3, cm.w, cm.h);
          if (orgb && orgb->h == cm.h) {
            snprintf(fn, sizeof fn, "%s/off_rgb_%s.ppm", dn, kRoles[c]);
            write_ppm(fn, orgb->data.data(), int(orgb->c), cm.w, cm.h);
          }
        }
      }
    }
  }
  printf("층 1 CPU: %.2f s, %.2f M픽셀/초 (%u 스레드)\n", sum_t1, n_pix_t1 / sum_t1 * 1e-6, std::thread::hardware_concurrency());
  if (fit) {
    // 노출 맞추기: 선형 휘도를 한 번 그려 두고, 톤매핑 연산자마다 노출을 훑어 공식과 히스토그램 EMD 가 가장 작은 값을 찾는다.
    // (장면 하나에 노출 스칼라 하나 = 보정값. 프레임마다 맞추지 않는다.) 검은 공식 프레임은 뺀다.
    struct Buf { std::vector<float> L; const Image* off; int n; };
    std::vector<Buf> bufs;
    for (auto& F : frames) {
      HostEnv HE;
      HE.resize(SV);
      for (int a = 0; a < SV.n_anchor && a < int(F.anchor.size()); ++a) HE.anchor[a] = F.anchor[a];
      for (size_t w = 0; w < HE.vis.size() && w < F.vis.size(); ++w) HE.vis[w] = F.vis[w];
      HE.build(SV);
      const EnvView ev = HE.view();
      for (int c = 0; c < 3 && c < int(F.cams.size()); ++c) {
        const Camera& cm = F.cams[c];
        const Image* orgb = F.find(std::string("img::") + kRoles[c] + "::rgb");
        if (!orgb || orgb->h != cm.h || orgb->w != cm.w) continue;
        const int n = cm.w * cm.h;
        const RgbStat so = rgb_stat(orgb->data.data(), int(orgb->c), n);
        if (so.mean[0] + so.mean[1] + so.mean[2] < 6.0) continue;  // 검은 프레임
        Buf b{std::vector<float>(size_t(n) * 3), orgb, n};
        std::vector<std::thread> th;
        const int nt = int(std::thread::hardware_concurrency());
        for (int t = 0; t < nt; ++t)
          th.emplace_back([&, t] {
            const unsigned old = _mm_getcsr();
            _mm_setcsr(old | 0x8040u);
            for (int py = t; py < cm.h; py += nt)
              for (int px = 0; px < cm.w; ++px) {
                float d;
                V3 L;
                shade_pixel(SV, ev, cm, px, py, pixel_seed(0, c, px, py, int(F.step)), d, L);
                float* o = &b.L[3 * (size_t(py) * cm.w + px)];
                o[0] = L.x; o[1] = L.y; o[2] = L.z;
              }
            _mm_setcsr(old);
          });
        for (auto& x : th) x.join();
        bufs.push_back(std::move(b));
      }
    }
    printf("\n== 노출 맞추기 (영상 %zu 장, 검은 프레임 뺌) ==\n", bufs.size());
    const char* tm_name[6] = {"클램프", "sRGB", "ACES(N)", "ACES(Hill)", "Reinhard", "Hable"};
    for (int tm = 1; tm <= 5; ++tm) {
      double best = 1e30, best_x = 0, best_md = 0;
      for (int k = -60; k <= 60; ++k) {
        const double x = std::pow(2.0, k * 0.25) * H.sp.exposure;
        ShadeParams sp = H.sp;
        sp.tonemap = tm;
        sp.exposure = float(x);
        double emd = 0, md = 0;
        for (auto& b : bufs) {
          std::vector<uint8_t> rgb(size_t(b.n) * 3);
          for (int i = 0; i < b.n; ++i) eng::rnd::tonemap(sp, V3{b.L[3 * i], b.L[3 * i + 1], b.L[3 * i + 2]}, &rgb[3 * i]);
          const RgbStat su = rgb_stat(rgb.data(), 3, b.n), so = rgb_stat(b.off->data.data(), int(b.off->c), b.n);
          emd += hist_emd(so, su);
          md += (su.mean[0] + su.mean[1] + su.mean[2] - so.mean[0] - so.mean[1] - so.mean[2]) / 3;
        }
        emd /= bufs.size();
        md /= bufs.size();
        if (emd < best) { best = emd; best_x = x; best_md = md; }
      }
      printf("  %-10s 노출 %.4g : EMD %.2f, 평균 차 %+.2f\n", tm_name[tm], best_x, best, best_md);
    }
    return 0;
  }
  if (!use_gpu) return 0;

  // ---- 2) 층 2
  const int nf = int(frames.size());
  if (check > envs) check = envs;
  gpu::DevScene DS;
  DS.init(H);
  gpu::Batch B = gpu::make_batch(DS.view, envs);
  std::vector<Aff> anchors(size_t(envs) * B.A);
  std::vector<uint32_t> vis(size_t(envs) * B.W, 0xFFFFFFFFu);
  std::vector<Camera> cams(size_t(envs) * 3);
  for (int e = 0; e < envs; ++e) {
    const HostFrame& F = frames[e % nf];
    for (int a = 0; a < SV.n_anchor && a < int(F.anchor.size()); ++a) anchors[size_t(e) * B.A + a] = F.anchor[a];
    for (int w = 0; w < B.W && w < int(F.vis.size()); ++w) vis[size_t(e) * B.W + w] = F.vis[w];
    for (int c = 0; c < 3; ++c) cams[size_t(e) * 3 + c] = F.cams[c];
  }
  RCK(cudaMemcpy(B.anchor, anchors.data(), anchors.size() * sizeof(Aff), cudaMemcpyHostToDevice));
  RCK(cudaMemcpy(B.vis, vis.data(), vis.size() * 4, cudaMemcpyHostToDevice));
  Camera* dcams;
  RCK(cudaMalloc(&dcams, cams.size() * sizeof(Camera)));
  RCK(cudaMemcpy(dcams, cams.data(), cams.size() * sizeof(Camera), cudaMemcpyHostToDevice));
  int cw[3], chh[3];
  float* ddep[3];
  uint8_t* drgb[3];
  size_t out_bytes = 0;
  for (int c = 0; c < 3; ++c) {
    cw[c] = frames[0].cams[c].w;
    chh[c] = frames[0].cams[c].h;
    const size_t hw = size_t(cw[c]) * chh[c];
    RCK(cudaMalloc(&ddep[c], hw * envs * 4));
    RCK(cudaMalloc(&drgb[c], hw * envs * 3));
    out_bytes += hw * envs * 7;
  }
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  printf("\n== 층 2 (판 %d) ==\nGPU 메모리: 정적 장면 %.0f MiB, 출력 %.0f MiB, 남음 %.0f MiB / %.0f MiB\n", envs, DS.bytes / 1048576.0,
         out_bytes / 1048576.0, fr / 1048576.0, tot / 1048576.0);
  RCK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  auto run = [&]() {
    cudaEventRecord(e0);
    gpu::kBuild<<<envs, gpu::kBuildThreads>>>(DS.view, B);
    RCK(cudaGetLastError());
    cudaEventRecord(e1);
    for (int c = 0; c < 3; ++c) {
      dim3 bs(16, 8), gs((cw[c] + 15) / 16, (chh[c] + 7) / 8, envs);
      // 프레임 번호 = 판이 그리는 기록 스텝 (층 1 과 같은 난수 씨앗) -> 판마다 다르므로 커널 인자 대신 0 으로 두고 판 번호로 구분
      gpu::kRender<<<gs, bs>>>(DS.view, B, dcams, 3, c, 0, ddep[c], drgb[c]);
      RCK(cudaGetLastError());
    }
    cudaEventRecord(e2);
    RCK(cudaEventSynchronize(e2));
  };
  run();
  float build_ms = 0, total_ms = 0;
  for (int r = 0; r < reps; ++r) {
    run();
    float a, b;
    cudaEventElapsedTime(&a, e0, e1);
    cudaEventElapsedTime(&b, e0, e2);
    build_ms += a;
    total_ms += b;
  }
  build_ms /= reps;
  total_ms /= reps;
  double pix = 0;
  for (int c = 0; c < 3; ++c) pix += double(cw[c]) * chh[c];
  printf("처리량: 판 %d × 카메라 3 (%dx%d, %dx%d, %dx%d) = %.2f ms (TLAS %.3f ms) -> 판·프레임 %.0f/초, 1차 광선 %.2f G/초\n", envs,
         cw[0], chh[0], cw[1], chh[1], cw[2], chh[2], total_ms, build_ms, envs / (total_ms * 1e-3),
         envs * pix / (total_ms * 1e-3) * 1e-9);
  uint64_t n_dep = 0, bad_dep = 0, n_rgb = 0, bad_rgb = 0;
  int fe = -1, fc = -1, fp = -1;
  for (int e = 0; e < check; ++e) {
    const HostFrame& F = frames[e % nf];
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < SV.n_anchor && a < int(F.anchor.size()); ++a) HE.anchor[a] = F.anchor[a];
    for (size_t w = 0; w < HE.vis.size() && w < F.vis.size(); ++w) HE.vis[w] = F.vis[w];
    HE.build(SV);
    for (int c = 0; c < 3; ++c) {
      const Camera& cm = F.cams[c];
      const size_t hw = size_t(cm.w) * cm.h;
      std::vector<float> hd(hw), gd(hw);
      std::vector<uint8_t> hr(hw * 3), gr(hw * 3);
      render_host(SV, HE, cm, e, c, 0, hd.data(), hr.data());
      RCK(cudaMemcpy(gd.data(), ddep[c] + e * hw, hw * 4, cudaMemcpyDeviceToHost));
      RCK(cudaMemcpy(gr.data(), drgb[c] + e * hw * 3, hw * 3, cudaMemcpyDeviceToHost));
      for (size_t p = 0; p < hw; ++p) {
        ++n_dep;
        if (memcmp(&hd[p], &gd[p], 4)) { if (fe < 0) { fe = e; fc = c; fp = int(p); } ++bad_dep; }
        for (int k = 0; k < 3; ++k) {
          ++n_rgb;
          if (hr[3 * p + k] != gr[3 * p + k]) { if (fe < 0) { fe = e; fc = c; fp = int(p); } ++bad_rgb; }
        }
      }
    }
  }
  printf("층 1 = 층 2 (판 %d 개 × 카메라 3): depth %" PRIu64 " 개 중 비트 다름 %" PRIu64 ", RGB %" PRIu64 " 개 중 다름 %" PRIu64
         " | 첫 다름 판 %d 카메라 %d 픽셀 %d\n", check, n_dep, bad_dep, n_rgb, bad_rgb, fe, fc, fp);
  gpu::free_batch(B);
  for (int c = 0; c < 3; ++c) { cudaFree(ddep[c]); cudaFree(drgb[c]); }
  cudaFree(dcams);

  // ---- 4) 카메라를 링크(기준 prim)에 붙여 따라가게 (포팅 평가기 경로: 물리 자세만 넘기면 됨) — RenderBatch
  // 카메라마다 프레임 사이 상대 변환이 가장 일정한 기준 prim 을 찾는다(double). 그 rel 로 만든 카메라와 뜬 카메라의 차이도 적는다.
  auto inv_d = [](const Aff& a, double* o) {  // 3x4 역행렬 (double)
    double m[12];
    for (int k = 0; k < 12; ++k) m[k] = a.m[k];
    const double c00 = m[5] * m[10] - m[6] * m[9], c01 = m[6] * m[8] - m[4] * m[10], c02 = m[4] * m[9] - m[5] * m[8];
    const double id = 1.0 / (m[0] * c00 + m[1] * c01 + m[2] * c02);
    o[0] = c00 * id; o[1] = (m[2] * m[9] - m[1] * m[10]) * id; o[2] = (m[1] * m[6] - m[2] * m[5]) * id;
    o[4] = c01 * id; o[5] = (m[0] * m[10] - m[2] * m[8]) * id; o[6] = (m[2] * m[4] - m[0] * m[6]) * id;
    o[8] = c02 * id; o[9] = (m[1] * m[8] - m[0] * m[9]) * id; o[10] = (m[0] * m[5] - m[1] * m[4]) * id;
    for (int r = 0; r < 3; ++r) o[r * 4 + 3] = -(o[r * 4] * m[3] + o[r * 4 + 1] * m[7] + o[r * 4 + 2] * m[11]);
  };
  auto mul_d = [](const double* a, const Aff& b, double* o) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) o[i * 4 + j] = a[i * 4] * b.m[j] + a[i * 4 + 1] * b.m[4 + j] + a[i * 4 + 2] * b.m[8 + j];
      o[i * 4 + 3] += a[i * 4 + 3];
    }
  };
  std::vector<CamRig> rigs(3);
  for (int c = 0; c < 3; ++c) {
    int best = -1;
    double best_dev = 1e30, rel0[12] = {0};
    for (int a = 0; a < SV.n_anchor; ++a) {
      double r0[12], dev = 0;
      for (int f = 0; f < nf; ++f) {
        double iv[12], r[12];
        inv_d(frames[f].anchor[a], iv);
        mul_d(iv, frames[f].cams[c].world, r);
        if (f == 0) memcpy(r0, r, sizeof r0);
        for (int k = 0; k < 12; ++k) dev = std::max(dev, std::fabs(r[k] - r0[k]));
      }
      if (dev < best_dev) { best_dev = dev; best = a; memcpy(rel0, r0, sizeof rel0); }
    }
    CamRig& g = rigs[c];
    g.anchor = best;
    for (int k = 0; k < 12; ++k) g.rel.m[k] = float(rel0[k]);
    const Camera& c0 = frames[0].cams[c];
    g.w = c0.w; g.h = c0.h; g.tanx = c0.tanx; g.tany = c0.tany; g.znear = c0.znear; g.zfar = c0.zfar;
    double perr = 0;
    for (int f = 0; f < nf; ++f) {
      const Camera cr = camera_from_rig(g, frames[f].anchor.data());
      for (int k = 0; k < 12; ++k) perr = std::max(perr, double(std::fabs(cr.world.m[k] - frames[f].cams[c].world.m[k])));
    }
    printf("카메라 %-11s 붙은 기준 prim %d (프레임 사이 rel 최대 흔들림 %.2e), rel 로 만든 카메라와 뜬 카메라 행렬 최대 차 %.2e\n", kRoles[c],
           best, best_dev, perr);
  }
  gpu::RenderBatch rb;
  rb.init(H, envs, rigs);
  Aff* dan;
  RCK(cudaMalloc(&dan, anchors.size() * sizeof(Aff)));
  RCK(cudaMemcpy(dan, anchors.data(), anchors.size() * sizeof(Aff), cudaMemcpyHostToDevice));
  uint32_t* dvis;
  RCK(cudaMalloc(&dvis, vis.size() * 4));
  RCK(cudaMemcpy(dvis, vis.data(), vis.size() * 4, cudaMemcpyHostToDevice));
  rb.set_anchors(dan);
  rb.set_visibility(dvis);
  rb.render(7);
  RCK(cudaDeviceSynchronize());
  cudaEventRecord(e0);
  for (int r = 0; r < reps; ++r) rb.render(7);
  cudaEventRecord(e2);
  RCK(cudaEventSynchronize(e2));
  float rb_ms;
  cudaEventElapsedTime(&rb_ms, e0, e2);
  rb_ms /= reps;
  uint64_t rn = 0, rbad = 0;
  for (int e = 0; e < check; ++e) {
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < SV.n_anchor; ++a) HE.anchor[a] = anchors[size_t(e) * B.A + a];
    for (size_t w = 0; w < HE.vis.size(); ++w) HE.vis[w] = vis[size_t(e) * B.W + w];
    HE.build(SV);
    for (int c = 0; c < 3; ++c) {
      const Camera cm = camera_from_rig(rigs[c], HE.anchor.data());
      const size_t hw = size_t(cm.w) * cm.h;
      std::vector<float> hd(hw), gd(hw);
      std::vector<uint8_t> hr(hw * 3), gr(hw * 3);
      render_host(SV, HE, cm, e, c, 7, hd.data(), hr.data());
      RCK(cudaMemcpy(gd.data(), rb.depth(c) + e * hw, hw * 4, cudaMemcpyDeviceToHost));
      RCK(cudaMemcpy(gr.data(), rb.rgb(c) + e * hw * 3, hw * 3, cudaMemcpyDeviceToHost));
      for (size_t p = 0; p < hw; ++p) {
        rn += 4;
        rbad += memcmp(&hd[p], &gd[p], 4) != 0;
        for (int k = 0; k < 3; ++k) rbad += hr[3 * p + k] != gr[3 * p + k];
      }
    }
  }
  printf("RenderBatch(링크에 붙은 카메라): 판 %d × 카메라 3 = %.2f ms/프레임 (판·프레임 %.0f/초) | 층 1 과 비교 %" PRIu64
         " 개 중 다름 %" PRIu64 "\n", envs, rb_ms, envs / (rb_ms * 1e-3), rn, rbad);
  cudaFree(dan);
  cudaFree(dvis);
  return (bad_dep || bad_rgb || rbad) ? 1 : 0;
}
