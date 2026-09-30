// 영상만 바꿔 π0.5 행동 청크가 얼마나 달라지나 (엔진 렌더 영상 vs 공식 영상, docs/엔진_자체구현.md 15절).
//   pi05_imgdiff --weights W.pi05w --in samples.bin --out actions.bin --prompt "..."
// samples.bin: "IMGD0001", i32 N, i32 H, i32 W, i32 P, 그 뒤 noise f32[horizon*32], 표본마다 영상 3 장(u8 H*W*4, 카메라 순서 = 엔진 cam_keys)·proprio f32[P]
// actions.bin: f64 [N][horizon][action_dim]. 시작 잡음은 모든 표본에 같은 것을 준다 -> 차이는 영상(과 proprio) 때문만.
// 빌드(WSL): nvcc -std=c++20 -O2 -o ~/pi05_native_build/pi05_imgdiff /mnt/c/behavior-2026/src/engine/eval/pi05_imgdiff.cpp ~/pi05_native_build/libpi05.a
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../pi05_native/include/pi05_native.h"

int main(int argc, char** argv) {
  std::string wpath, in, out, prompt;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--weights") wpath = argv[++i];
    else if (a == "--in") in = argv[++i];
    else if (a == "--out") out = argv[++i];
    else if (a == "--prompt") prompt = argv[++i];
  }
  char err[512] = {0};
  Pi05Engine* e = pi05_create(wpath.c_str(), 0, err, sizeof err);
  if (!e) { fprintf(stderr, "pi05_create: %s\n", err); return 1; }
  Pi05Info info;
  pi05_info(e, &info);
  FILE* f = fopen(in.c_str(), "rb");
  if (!f) { fprintf(stderr, "입력 못 엶\n"); return 1; }
  char mg[8];
  int32_t hdr[4];
  if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "IMGD0001", 8) || fread(hdr, 4, 4, f) != 4) { fprintf(stderr, "형식\n"); return 1; }
  const int N = hdr[0], H = hdr[1], W = hdr[2], P = hdr[3];
  std::vector<float> noise(size_t(info.action_horizon) * 32);
  if (fread(noise.data(), 4, noise.size(), f) != noise.size()) return 1;
  std::vector<double> acts(size_t(N) * info.action_horizon * info.action_dim);
  std::vector<uint8_t> img(size_t(3) * H * W * 4);
  std::vector<float> prop(P);
  for (int n = 0; n < N; ++n) {
    if (fread(img.data(), 1, img.size(), f) != img.size() || fread(prop.data(), 4, P, f) != size_t(P)) { fprintf(stderr, "표본 %d\n", n); return 1; }
    Pi05Image ims[3];
    for (int c = 0; c < 3; ++c) {
      ims[c].data = img.data() + size_t(c) * H * W * 4;
      ims[c].h = H; ims[c].w = W; ims[c].row_stride = int64_t(W) * 4; ims[c].pix_stride = 4; ims[c].on_device = 0;
    }
    Pi05Timing t;
    if (pi05_infer(e, ims, prop.data(), P, prompt.c_str(), noise.data(), &acts[size_t(n) * info.action_horizon * info.action_dim], &t) != 0) {
      fprintf(stderr, "infer %d: %s\n", n, pi05_last_error(e));
      return 1;
    }
  }
  fclose(f);
  FILE* g = fopen(out.c_str(), "wb");
  fwrite(acts.data(), 8, acts.size(), g);
  fclose(g);
  printf("표본 %d, 청크 %d x %d, 카메라 키 %s | %s | %s\n", N, info.action_horizon, info.action_dim, info.cam_keys[0], info.cam_keys[1], info.cam_keys[2]);
  pi05_destroy(e);
  return 0;
}
