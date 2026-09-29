// 순회 통계 진단 (호스트 한 스레드): 1차 광선과 코사인 튕김 광선 하나가 TLAS·BLAS 노드·삼각형을 몇 번 도는지, 광선 하나 시간.
//   render_stats <rsc 폴더> [프레임 번호=100] [해상도=224]
// 빌드: tests/render/compare/build.sh (-DRENDER_STATS). 렌더 결과에는 영향 없음(통계 매크로는 이 도구에서만 켠다).
#include <chrono>
#include <cstdio>
#include <string>

#include "core/render/render_host.h"
#include "core/render/rsc_io.h"

using namespace eng::rnd;

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "사용: render_stats <rsc 폴더> [프레임] [해상도]\n"); return 2; }
  const std::string dir = argv[1];
  const int step = argc > 2 ? std::atoi(argv[2]) : 100;
  const int res = argc > 3 ? std::atoi(argv[3]) : 224;
  HostScene H;
  if (!load_scene(dir + "/scene.rsc", H)) return 1;
  char fn[64];
  std::snprintf(fn, sizeof fn, "/frame_%04d.rfr", step);
  HostFrame F;
  if (!load_frame(dir + fn, F)) return 1;
  const SceneView SV = H.view();
  HostEnv HE;
  HE.resize(SV);
  for (int a = 0; a < SV.n_anchor && a < int(F.anchor.size()); ++a) HE.anchor[a] = F.anchor[a];
  for (size_t w = 0; w < HE.vis.size() && w < F.vis.size(); ++w) HE.vis[w] = F.vis[w];
  HE.build(SV);
  const EnvView ev = HE.view();
  std::printf("장면: 인스턴스 %d, 삼각형 %zu, BLAS 노드 %zu | 프레임 %d, 해상도 %d\n", SV.n_inst, H.tris.size(), H.blas_nodes.size(), step, res);
  const char* roles[3] = {"head", "left_wrist", "right_wrist"};
  for (int c = 0; c < 3 && c < int(F.cams.size()); ++c) {
    Camera cm = F.cams[c];
    cm.w = cm.h = res;
    for (int kind = 0; kind < 2; ++kind) {
      g_rstats = RStats{};
      uint64_t rays = 0, hits = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int py = 0; py < cm.h; ++py)
        for (int px = 0; px < cm.w; ++px) {
          const Ray r0 = camera_ray(cm, px + 0.5f, py + 0.5f);
          if (kind == 0) {
            const Hit h = trace(SV, ev, r0, cm.znear, cm.zfar);
            ++rays;
            hits += h.inst >= 0;
          } else {
            RStats keep = g_rstats;  // 1차 광선은 세지 않는다
            const Hit h0 = trace(SV, ev, r0, cm.znear, cm.zfar);
            g_rstats = keep;
            if (h0.inst < 0) continue;
            const Surf s = surface(SV, ev, r0, h0, 0.0f);
            uint32_t rs = pixel_seed(0, c, px, py, 0);
            const float b1 = rnd01(rs), b2 = rnd01(rs);
            const Hit h = trace(SV, ev, make_ray(s.p + s.ng * 1e-4f, cosine_dir(s.ns, b1, b2)), 1e-4f, 1e30f);
            ++rays;
            hits += h.inst >= 0;
          }
        }
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double n = double(rays > 0 ? rays : 1);
      std::printf("%-11s %-6s 광선 %7llu 맞음 %5.1f%% | 광선당 TLAS 노드 %6.1f 인스턴스 잎 %5.1f BLAS 노드 %7.1f 삼각형 %7.1f | 스택 최대 %llu | %.0f ns/광선 (한 스레드)\n",
                  roles[c], kind == 0 ? "1차" : "튕김", (unsigned long long)rays, 100.0 * hits / n, g_rstats.tlas_nodes / n,
                  g_rstats.inst_leaves / n, g_rstats.blas_nodes / n, g_rstats.tri_tests / n, (unsigned long long)g_rstats.max_sp, dt * 1e9 / n);
    }
  }
  return 0;
}
