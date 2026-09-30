// 층 1 시험: find_simplex 옮김 vs scipy (단체 번호까지 같아야 함).
//   ./test_delaunay ~/engine-data/particles/delaunay [메시 수]
#include <cstdio>
#include <string>

#include "core/particles/delaunay.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const int nm = argc > 2 ? atoi(argv[2]) : 200;
  long n = 0, bad = 0, bad_in = 0, inside = 0;
  for (int m = 0; m < nm; ++m) {
    char d[512];
    snprintf(d, sizeof d, "%s/mesh_%03d/", argv[1], m);
    Npy T, N, E, M, Q, R;
    if (!npy_load(std::string(d) + "transform.npy", T) || !npy_load(std::string(d) + "neighbors.npy", N) ||
        !npy_load(std::string(d) + "equations.npy", E) || !npy_load(std::string(d) + "misc.npy", M) ||
        !npy_load(std::string(d) + "q.npy", Q) || !npy_load(std::string(d) + "res.npy", R))
      return 1;
    const double* mi = M.as<double>();
    Delaunay3 dl{(int32_t)T.shape[0], T.as<double>(), N.as<int32_t>(), E.as<double>(), mi[0], mi[1], {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
    int start = 0;
    for (long i = 0; i < (long)Q.shape[0]; ++i) {
      const int s = dl_find_simplex(dl, Q.as<double>() + 3 * i, &start);
      const int want = R.as<int32_t>()[i];
      if (s != want && bad++ == 0) printf("  첫 다름: 메시 %d 점 %ld 우리 %d 공식 %d\n", m, i, s, want);
      bad_in += (s >= 0) != (want >= 0);
      inside += want >= 0;
      ++n;
    }
  }
  printf("find_simplex 층 1 vs scipy: 점 %ld (안 %ld)  단체 번호 다름 %ld  안/밖 다름 %ld\n", n, inside, bad, bad_in);
  printf(bad ? "실패\n" : "전부 동일\n");
  return bad ? 1 : 0;
}
