// 렌더 모듈(core/render, 층 1 C++)의 파이썬 입구 (포팅 평가기 관측 영상, docs/엔진_자체구현.md 14·15절). ctypes 로 부른다: render_core.py
//
// 정적 장면 = scene.rsc (render 작업자 A 의 convert_scene.py, 공식 기록에서 — 에셋 파생물, git 밖). 기준 prim(anchor) 중 PhysX 몸체인 것은
// 매 스텝 물리 자세로 바꾸고(rr_set_anchor_pose, 축척은 기준 프레임 행렬의 열 길이), 나머지(정적·조명)는 기준 프레임 값 그대로 둔다.
// 카메라 = 엔진이 계산한 카메라 prim 세계 자세(obs_engine.camera_world_gf 와 같은 값) + 기준 프레임의 tan·znear·zfar, 해상도는 부르는 쪽이 정함.
// 관측 시점 규칙(10.1): 스텝 k 영상은 스텝 k-1 끝 자세로 그린다 — 부르는 쪽(backend_engine)이 물리 스텝 전에 부른다.
// 빌드(WSL): g++ -O2 -ffp-contract=off -fno-fast-math -std=c++17 -fPIC -shared -I/mnt/c/behavior-2026/src/engine \n//   /mnt/c/behavior-2026/src/engine/eval/render_capi.cpp -o ~/engine-build/render-capi/librender_capi.so -pthread
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/render/render_host.h"
#include "core/render/rig.h"
#include "core/render/rsc_io.h"

using namespace eng::rnd;

namespace {
struct R {
  HostScene H;
  SceneView SV{};
  HostFrame F;
  HostEnv E;
  std::vector<float> scale;  // n_anchor x 3
  std::vector<Camera> cams;  // 기준 프레임 카메라 (tan 등)
  int threads = 0;
};
}  // namespace

extern "C" {

void* rr_open(const char* rsc_dir, const char* frame_file, int spp, int threads) {
  auto* r = new R();
  const std::string d = rsc_dir;
  if (!load_scene(d + "/scene.rsc", r->H)) {
    fprintf(stderr, "rr_open: scene.rsc 를 못 읽음 (%s)\n", d.c_str());
    delete r;
    return nullptr;
  }
  if (!load_frame(frame_file, r->F)) {
    fprintf(stderr, "rr_open: 프레임을 못 읽음 (%s)\n", frame_file);
    delete r;
    return nullptr;
  }
  r->SV = r->H.view();
  if (spp > 0) r->SV.sp.spp = spp;
  r->threads = threads;
  r->E.resize(r->SV);
  for (int a = 0; a < r->SV.n_anchor && a < int(r->F.anchor.size()); ++a) r->E.anchor[a] = r->F.anchor[a];
  for (size_t w = 0; w < r->E.vis.size() && w < r->F.vis.size(); ++w) r->E.vis[w] = r->F.vis[w];
  r->scale.resize(size_t(r->SV.n_anchor) * 3);
  for (int a = 0; a < r->SV.n_anchor; ++a)
    for (int c = 0; c < 3; ++c) {
      const Aff& A = r->E.anchor[a];
      const float x = A.m[c], y = A.m[4 + c], z = A.m[8 + c];
      r->scale[size_t(a) * 3 + c] = std::sqrt(x * x + y * y + z * z);
    }
  r->cams = r->F.cams;
  return r;
}

int rr_n_anchor(void* h) { return static_cast<R*>(h)->SV.n_anchor; }
int rr_n_cam(void* h) { return int(static_cast<R*>(h)->cams.size()); }

// 기준 prim a 를 물리 자세로 (q = x,y,z,w; p = x,y,z)
void rr_set_anchor_pose(void* h, int a, const float* q, const float* p) {
  R& r = *static_cast<R*>(h);
  if (a < 0 || a >= r.SV.n_anchor) return;
  r.E.anchor[a] = aff_from_pose(q, p, &r.scale[size_t(a) * 3]);
}

// 카메라 c 를 세계 자세(q, p)로 그린다 (해상도 w x hgt). out_rgb = w*hgt*3 u8 (행 0 = 위), out_depth = w*hgt f32 (NULL 가능)
// frame = 결정적 난수 씨앗 (스텝 번호)
void rr_render(void* h, int c, const float* q, const float* p, int w, int hgt, int frame, uint8_t* out_rgb, float* out_depth) {
  R& r = *static_cast<R*>(h);
  if (c < 0 || c >= int(r.cams.size())) return;
  Camera cm = r.cams[c];
  const float one[3] = {1.0f, 1.0f, 1.0f};
  const Aff W = aff_from_pose(q, p, one);
  cm.world = W;
  cm.w = w;
  cm.h = hgt;
  std::vector<float> dep(size_t(w) * hgt);
  render_host(r.SV, r.E, cm, 0, c, frame, dep.data(), out_rgb, r.threads);
  if (out_depth) std::memcpy(out_depth, dep.data(), dep.size() * 4);
}

// 자세를 다 바꾼 뒤 한 번 (TLAS 다시 짓기)
void rr_build(void* h) {
  R& r = *static_cast<R*>(h);
  r.E.build(r.SV);
}

void rr_close(void* h) { delete static_cast<R*>(h); }

}  // extern "C"
