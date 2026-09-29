// 변환기 출력(scene.rsc, frame_<k>.rfr) 을 렌더러 불러오기(core/render/rsc_io.h)로 읽어 개수·범위를 찍는다 (형식 맞물림 검사).
//   rsc_check <scene.rsc> [frame.rfr ...]
#include <cstdio>

#include "core/render/rsc_io.h"

using namespace eng::rnd;

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: rsc_check scene.rsc [frame.rfr...]\n"); return 2; }
  HostScene S;
  if (!load_scene(argv[1], S)) { std::fprintf(stderr, "장면 읽기 실패\n"); return 1; }
  size_t bad_slot = 0, bad_tex = 0, bad_geom = 0;
  for (auto& in : S.insts) {
    if (in.geom < 0 || in.geom >= int32_t(S.geoms.size())) ++bad_geom;
    if (in.slot_base < 0 || in.slot_base >= int32_t(S.slot_mat.size())) ++bad_slot;
  }
  for (auto& m : S.mats)
    if (m.tex_albedo >= int32_t(S.texs.size())) ++bad_tex;
  std::printf("기하 %zu, BLAS 마디 %zu, 삼각형 %zu, 인스턴스 %zu, 기준 prim %d, 재질 %zu, 텍스처 %zu (텍셀 %zu), 조명 %zu | 잘못된 기하 %zu 칸 %zu 텍스처 %zu\n",
              S.geoms.size(), S.blas_nodes.size(), S.tris.size(), S.insts.size(), S.n_anchor, S.mats.size(), S.texs.size(),
              S.texels.size(), S.lights.size(), bad_geom, bad_slot, bad_tex);
  for (int i = 2; i < argc; ++i) {
    HostFrame F;
    if (!load_frame(argv[i], F)) { std::fprintf(stderr, "프레임 읽기 실패 %s\n", argv[i]); return 1; }
    std::printf("프레임 %lld: 기준 prim %zu, 보임 단어 %zu, 카메라 %zu, 영상 %zu\n", (long long)F.step, F.anchor.size(), F.vis.size(),
                F.cams.size(), F.imgs.size());
    for (auto& c : F.cams) std::printf("  카메라 %dx%d tan %.4f %.4f 위치 %.3f %.3f %.3f\n", c.w, c.h, c.tanx, c.tany, c.world.m[3], c.world.m[7], c.world.m[11]);
    for (auto& im : F.imgs) std::printf("  영상 %-40s %lldx%lldx%lld %s\n", im.name.c_str(), (long long)im.h, (long long)im.w, (long long)im.c, im.dtype ? "f32" : "u8");
  }
  return 0;
}
