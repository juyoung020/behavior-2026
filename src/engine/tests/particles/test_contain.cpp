// 층 1 시험: 물리 입자 중심·Contains·Filled vs 공식.   ./test_contain ~/engine-data/particles/contain [사례 수]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/contain.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const int nc = argc > 2 ? atoi(argv[2]) : 500;
  long n_c = 0, bad_c = 0, bad_in = 0, bad_cnt = 0, bad_f = 0;
  for (int c = 0; c < nc; ++c) {
    char d[512];
    snprintf(d, sizeof d, "%s/case_%03d/", argv[1], c);
    auto L = [&](const std::string& k) {
      Npy a;
      if (!npy_load(std::string(d) + k + ".npy", a)) {
        fprintf(stderr, "없음 %s%s\n", d, k.c_str());
        exit(1);
      }
      return a;
    };
    Npy tfs = L("tfs"), off = L("off"), mtf = L("mesh_tf"), cen = L("centers"), inv = L("in_volume"), scal = L("scal"), kind = L("kind");
    const int n = (int)tfs.shape[0];
    std::vector<float> ce(3 * n);
    for (int i = 0; i < n; ++i) {
      physical_center(tfs.as<float>() + 7 * i, off.as<float>(), &ce[3 * i]);
      ++n_c;
      bad_c += memcmp(&ce[3 * i], cen.as<float>() + 3 * i, 12) != 0;
    }
    const int k = kind.as<int32_t>()[0];
    ContainerMesh m{};
    static const int map[5] = {5, CYLINDER, CUBE, SPHERE, CONE};
    m.kind = map[k];
    memcpy(m.tf, mtf.as<float>(), 64);
    m.attr[0] = 0.5, m.attr[1] = 1.0, m.attr[2] = 1.0;
    Npy T, N, E, M;
    if (m.kind == 5) {
      T = L("dl_transform"), N = L("dl_neighbors"), E = L("dl_equations"), M = L("dl_misc");
      const double* mi = M.as<double>();
      m.dl = Delaunay3{(int32_t)T.shape[0], T.as<double>(), N.as<int32_t>(), E.as<double>(), mi[0], mi[1], {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
    }
    std::vector<uint8_t> in(n);
    // 판정은 공식 중심으로 (중심 비교와 따로)
    const int cnt = contained_count(&m, 1, cen.as<float>(), n, in.data());
    bad_in += memcmp(in.data(), inv.as<uint8_t>(), n) != 0;
    bad_cnt += cnt != kind.as<int32_t>()[1];
    bad_f += (int)filled_value((float)scal.as<double>()[0], cnt, scal.as<double>()[1]) != kind.as<int32_t>()[2];
  }
  printf("Contains/Filled 층 1 vs 공식 입자 중심 %ld 다름 %ld | 사례 %d: 안 표시 다름 %ld, 개수 다름 %ld, Filled 다름 %ld\n", n_c,
         bad_c, nc, bad_in, bad_cnt, bad_f);
  const bool ok = !(bad_c || bad_in || bad_cnt || bad_f);
  printf(ok ? "전부 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
