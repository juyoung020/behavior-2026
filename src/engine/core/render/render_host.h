// 층 1: 호스트(C++) 렌더. 판 하나 × 카메라 여러 대. 픽셀마다 같은 EHD 함수(shade.h)를 부른다 -> 층 2 와 비트 비교.
// 픽셀끼리 독립이라 스레드 수는 결과에 영향 없다.
#pragma once
#include <cstdint>
#include <thread>
#include <vector>
#include <xmmintrin.h>

#include "core/render/shade.h"
#include "core/render/tlas.h"

namespace eng {
namespace rnd {

EHD uint32_t pixel_seed(int32_t env, int32_t cam, int32_t px, int32_t py, int32_t frame) {
  return pcg(uint32_t(frame) * 0x9E3779B1u ^ pcg(uint32_t(env) * 0x85EBCA77u ^ pcg(uint32_t(cam) * 0xC2B2AE3Du ^
             pcg(uint32_t(py) * 0x27D4EB2Fu ^ uint32_t(px)))));
}

// 픽셀 하나: depth(f32) 와 RGB(u8 ×3). 층 1·2 가 똑같이 부르는 입구.
EHD void render_pixel(const SceneView& S, const EnvView& E, const Camera& c, int32_t env, int32_t cam, int32_t frame,
                      int px, int py, float* depth_out, uint8_t* rgb_out) {
  float d;
  V3 L;
  shade_pixel(S, E, c, px, py, pixel_seed(env, cam, px, py, frame), d, L);
  depth_out[py * c.w + px] = d;
  tonemap(S.sp, L, rgb_out + 3 * (py * c.w + px));
}

struct HostEnv {
  std::vector<Aff> anchor, light_world, inst_world, inst_inv;
  std::vector<float> inst_box;
  std::vector<Node2> tlas;
  std::vector<int32_t> order;
  std::vector<uint32_t> vis;
  std::vector<uint64_t> keys;
  std::vector<float> nodebox;
  std::vector<int32_t> kids;
  void resize(const SceneView& S) {
    anchor.resize(S.n_anchor);
    light_world.resize(S.n_lights > 0 ? S.n_lights : 1);
    inst_world.resize(S.n_inst);
    inst_inv.resize(S.n_inst);
    inst_box.resize(6 * size_t(S.n_inst));
    tlas.assign(S.n_inst > 1 ? S.n_inst - 1 : 1, Node2{});
    order.resize(S.n_inst > 0 ? S.n_inst : 1);
    vis.assign((S.n_inst + 31) / 32, 0xFFFFFFFFu);
  }
  EnvView view() {
    return EnvView{anchor.data(), vis.data(), light_world.data(), inst_world.data(), inst_inv.data(), inst_box.data(),
                   tlas.data(), order.data()};
  }
  void build(const SceneView& S) {
    EnvView v = view();
    const unsigned old = _mm_getcsr();  // GPU -ftz=true 와 같게 FTZ+DAZ
    _mm_setcsr(old | 0x8040u);
    tlas_build_host(S, v, keys, nodebox, kids);
    _mm_setcsr(old);
  }
};

inline void render_host(const SceneView& S, HostEnv& E, const Camera& c, int32_t env, int32_t cam, int32_t frame,
                        float* depth, uint8_t* rgb, int nthreads = 0) {
  if (nthreads <= 0) nthreads = int(std::thread::hardware_concurrency());
  const EnvView ev = E.view();
  std::vector<std::thread> th;
  for (int t = 0; t < nthreads; ++t)
    th.emplace_back([&, t] {
      // GPU -ftz=true 와 같게: 이 스레드의 MXCSR 에 FTZ(0x8000)+DAZ(0x40)
      const unsigned old = _mm_getcsr();
      _mm_setcsr(old | 0x8040u);
      for (int py = t; py < c.h; py += nthreads)
        for (int px = 0; px < c.w; ++px) render_pixel(S, ev, c, env, cam, frame, px, py, depth, rgb);
      _mm_setcsr(old);
    });
  for (auto& x : th) x.join();
}

}  // namespace rnd
}  // namespace eng
