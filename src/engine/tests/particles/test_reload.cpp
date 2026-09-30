// 층 1 시험 (물리 결합): 거시 물리 입자 재적재 왕복 (EDIT_PARTICLES_RESET) vs 공식 dump_state → load_state.
//   spawn.h particle_reload: 원점 → 중심 → 장면 좌표 → 세계 → 원점 = 중심 (공식은 offset 을 다시 빼지 않음), 방향은 mat2quat 두 번.
//   ./test_reload ~/engine-data/particles/reload
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/particles/spawn.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(std::string(argv[1]) + "/" + k + ".npy", a)) {
      fprintf(stderr, "없음 %s/%s.npy\n", argv[1], k);
      exit(1);
    }
    return a;
  };
  Npy nn = L("n"), tfs = L("tfs"), off = L("off"), out = L("out"), sp = L("scene_pose");
  const int nc = (int)nn.shape[0];
  const int32_t* n = nn.as<int32_t>();
  const float* tf = tfs.as<float>();
  const float* o = off.as<float>();
  const float* w = out.as<float>();
  const float* P = sp.as<float>();  // 사례마다 [pose, pose_inv] 4x4
  long k = 0, bad = 0, moved = 0;
  for (int c = 0; c < nc; ++c)
    for (int i = 0; i < n[c]; ++i, ++k) {
      const Pose7 r = particle_reload(tf + 7 * k, o + 3 * c, P + 32 * c, P + 32 * c + 16);
      float got[7] = {r.p[0], r.p[1], r.p[2], r.q[0], r.q[1], r.q[2], r.q[3]};
      moved += memcmp(tf + 7 * k, w + 7 * k, 28) != 0;
      if (memcmp(got, w + 7 * k, 28)) {
        if (bad < (getenv("SHOW_N") ? atoi(getenv("SHOW_N")) : 1)) {
          printf("  다름 사례 %d 입자 %d:", c, i);
          for (int j = 0; j < 7; ++j)
            if (memcmp(got + j, w + 7 * k + j, 4)) printf(" [%d] 우리 %.9g 공식 %.9g", j, got[j], w[7 * k + j]);
          printf("  (입력 q %.9g %.9g %.9g %.9g)\n", tf[7 * k + 3], tf[7 * k + 4], tf[7 * k + 5], tf[7 * k + 6]);
        }
        ++bad;
      }
    }
  printf("재적재 왕복: 사례 %d 입자 %ld  다름 %ld  (재적재로 자세가 바뀐 입자 %ld)\n", nc, k, bad, moved);
  printf(bad || !k ? "실패\n" : "비트 동일\n");
  return bad || !k;
}
