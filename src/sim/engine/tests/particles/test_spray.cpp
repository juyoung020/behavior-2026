// 층 1 시험 (호스트, MKL dlopen): 뿌리기 표본 sample_cuboid_on_object vs 공식 (가짜 광선 신탁 기록 재생).
//   ENGINE_MKL_LIB=.../torch/lib/libtorch_cpu.so ./test_spray ~/engine-data/particles/spray
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/spray.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  Npy RG, SE, RO, RY, HO, HT, OU;
  if (!npy_load(d + "rng.npy", RG) || !npy_load(d + "se.npy", SE) || !npy_load(d + "ray_off.npy", RO) || !npy_load(d + "rays.npy", RY) ||
      !npy_load(d + "hit_off.npy", HO) || !npy_load(d + "hits.npy", HT) || !npy_load(d + "outs.npy", OU))
    return 1;
  const int N = (int)RG.shape[0], B = (int)RG.shape[1];
  long bad_req = 0, n_req = 0, bad_ok = 0, bad_c = 0, bad_n = 0, bad_r = 0, bad_b = 0, n_ok = 0;
  for (int c = 0; c < N; ++c) {
    TorchMT m;
    torch_mt_from_bytes(RG.as<uint8_t>() + (size_t)c * B, B, m);
    const float* se = SE.as<float>() + c * 18;
    long r = RO.as<int64_t>()[c];
    for (int s = 0; s < 2; ++s) {
      auto rayfn = [&](const float* o, const float* dir, float dist, std::vector<RayHitIn>& out) {
        const double* q = RY.as<double>() + r * 7;
        ++n_req;
        bad_req += !((float)q[0] == o[0] && (float)q[1] == o[1] && (float)q[2] == o[2] && (float)q[3] == dir[0] && (float)q[4] == dir[1] &&
                     (float)q[5] == dir[2] && (float)q[6] == dist);
        for (int64_t h = HO.as<int64_t>()[r]; h < HO.as<int64_t>()[r + 1]; ++h) {
          const double* x = HT.as<double>() + h * 8;
          RayHitIn hi;
          for (int k = 0; k < 3; ++k) hi.pos[k] = (float)x[k], hi.nrm[k] = (float)x[3 + k];
          hi.dist = x[6];
          hi.body = (int)x[7];
          out.push_back(hi);
        }
        ++r;
      };
      CuboidResult R = sample_cuboid_one(m, se + 3 * s, se + 6 + 3 * s, se + 12 + 3 * s, rayfn, [](int) { return false; });
      const double* o = OU.as<double>() + ((size_t)c * 2 + s) * 12;
      const bool want = o[0] != 0;
      n_ok += want;
      if (R.ok != want) {
        if (!bad_ok) printf("  첫 성공 여부 다름: 사례 %d 표본 %d 우리 %d\n", c, s, R.ok);
        ++bad_ok;
        continue;
      }
      if (!want) continue;
      bool okc = true, okn = true, okr = true;
      for (int k = 0; k < 3; ++k) okc &= R.centroid[k] == (float)o[1 + k], okn &= R.normal[k] == (float)o[4 + k];
      for (int k = 0; k < 4; ++k) okr &= R.rot[k] == (float)o[7 + k];
      if (!okr && !bad_r) printf("  첫 회전 다름: 사례 %d 표본 %d  %g %g %g %g / %g %g %g %g\n", c, s, R.rot[0], R.rot[1], R.rot[2], R.rot[3], o[7], o[8], o[9], o[10]);
      bad_c += !okc, bad_n += !okn, bad_r += !okr, bad_b += R.body != (int)o[11];
    }
  }
  printf("뿌리기 표본 층 1 vs 공식: 사례 %d, 광선 요청 %ld 다름 %ld | 성공 %ld, 성공 여부 다름 %ld, 중심 다름 %ld, 법선 다름 %ld, 회전 다름 %ld, 몸체 다름 %ld\n", N,
         n_req, bad_req, n_ok, bad_ok, bad_c, bad_n, bad_r, bad_b);
  const bool ok = !(bad_req || bad_ok || bad_c || bad_n || bad_r || bad_b);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
