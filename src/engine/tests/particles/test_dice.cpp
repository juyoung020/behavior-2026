// 층 1 시험: 다지기 입자 자리 vs 공식 generate_particles_from_link.   ./test_dice ~/engine-data/particles/dice [사례 수]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/dice.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const int nc = argc > 2 ? atoi(argv[2]) : 300;
  long bad_n = 0, bad_p = 0, tot = 0;
  for (int c = 0; c < nc; ++c) {
    char d[512];
    snprintf(d, sizeof d, "%s/case_%03d/", argv[1], c);
    auto L = [&](const std::string& k) {
      Npy a;
      if (!npy_load(std::string(d) + k + ".npy", a)) exit(1);
      return a;
    };
    Npy lo = L("lo"), hi = L("hi"), r = L("r"), mn = L("mesh_n"), mt = L("mesh_tf"), pos = L("pos");
    const int nm = (int)mn.shape[0];
    std::vector<Npy> keep;
    std::vector<DiceMesh> ms(nm);
    for (int m = 0; m < nm; ++m) {
      Npy T = L("dl" + std::to_string(m) + "_transform"), N = L("dl" + std::to_string(m) + "_neighbors"),
          E = L("dl" + std::to_string(m) + "_equations"), M = L("dl" + std::to_string(m) + "_misc");
      keep.push_back(T), keep.push_back(N), keep.push_back(E), keep.push_back(M);
      const size_t b = keep.size() - 4;
      const double* mi = keep[b + 3].as<double>();
      memcpy(ms[m].tf, mt.as<float>() + 16 * m, 64);
      ms[m].dl = Delaunay3{(int32_t)keep[b].shape[0], keep[b].as<double>(), keep[b + 1].as<int32_t>(), keep[b + 2].as<double>(), mi[0], mi[1],
                           {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
    }
    const int64_t want = pos.shape[0];
    std::vector<float> out((size_t)(want + 100000) * 3);
    const int64_t got = dice_grid(lo.as<float>(), hi.as<float>(), r.as<float>()[0], ms.data(), nm, out.data(), want + 100000);
    tot += want;
    if (got != want) {
      if (!bad_n) printf("  첫 개수 다름: 사례 %d 우리 %ld 공식 %ld\n", c, (long)got, (long)want);
      ++bad_n;
      continue;
    }
    if (memcmp(out.data(), pos.as<float>(), (size_t)want * 12) != 0) {
      if (!bad_p) printf("  첫 위치 다름: 사례 %d\n", c);
      ++bad_p;
    }
  }
  printf("다지기 격자 층 1 vs 공식: 사례 %d, 입자 %ld, 개수 다름 %ld, 위치 다름 %ld\n", nc, tot, bad_n, bad_p);
  printf(bad_n || bad_p ? "실패\n" : "전부 비트 동일\n");
  return bad_n || bad_p;
}
