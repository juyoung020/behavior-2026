// 층 1 시험 (호스트, MKL dlopen): fit_plane 중심·법선 vs 공식.  ENGINE_MKL_LIB=.../torch/lib/libtorch_cpu.so ./test_fitplane ~/engine-data/particles/fitplane
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/fitplane.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  Npy P, K, O;
  if (!npy_load(d + "pts.npy", P) || !npy_load(d + "k.npy", K) || !npy_load(d + "out.npy", O)) return 1;
  const int N = (int)P.shape[0], KM = (int)P.shape[1], W = (int)O.shape[1];
  long bad_c = 0, bad_n = 0, big = 0;
  for (int i = 0; i < N; ++i) {
    const int k = K.as<int32_t>()[i];
    big += k > 64;
    float c[3], n[3];
    fit_plane(P.as<float>() + (size_t)i * KM * 3, k, c, n);
    const float* o = O.as<float>() + (size_t)i * W;
    bad_c += memcmp(c, o, 12) != 0;
    bad_n += memcmp(n, o + 3, 12) != 0;
  }
  printf("fit_plane 층 1(MKL dlopen) vs 공식: 사례 %d (k>64 %ld)  중심 다름 %ld  법선 다름 %ld\n", N, big, bad_c, bad_n);
  printf(bad_c || bad_n ? "실패\n" : "전부 비트 동일\n");
  return bad_c || bad_n;
}
