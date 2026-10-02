// omni 시험 5: 보조 잡기 관절 틀 (core/omni/agframe.h) = 공식 T.relative_pose_transform (torch.compile, gen_agframe_ref.py)
//   test_agframe <~/engine-data/omni/agframe>
// 행 = contact3 eef_pos3 eef_q4 scale3 | pos3 q4 (float32). mm 순서 후보마다 비트 비교해 가장 맞는 것을 보고한다.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/omni/agframe.h"

using namespace eng::omni;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_agframe <dir>\n");
    return 2;
  }
  std::ifstream f(std::string(argv[1]) + "/agframe.bin", std::ios::binary);
  std::vector<float> d((std::istreambuf_iterator<char>(f)), {});
  std::vector<float> v(d.size() / 1);
  {
    std::ifstream g(std::string(argv[1]) + "/agframe.bin", std::ios::binary | std::ios::ate);
    const size_t n = (size_t)g.tellg();
    g.seekg(0);
    v.resize(n / 4);
    g.read((char*)v.data(), n);
  }
  const int W = 20, N = (int)(v.size() / W);
  int best = -1;
  long long best_bad = -1;
  for (int order = 0; order < 3; ++order) {
    long long bad = 0, cmp = 0;
    int first = -1;
    for (int r = 0; r < N; ++r) {
      const float* x = &v[(size_t)r * W];
      float pos[3], q[4];
      agf::grasp_frame(x, x + 3, x + 6, x + 10, pos, q, order);
      for (int k = 0; k < 7; ++k) {
        const float o = k < 3 ? pos[k] : q[k - 3];
        ++cmp;
        if (memcmp(&o, &x[13 + k], 4) != 0) {
          if (first < 0) first = r;
          ++bad;
        }
      }
    }
    printf("mm 순서 %d: 비교 %lld 다름 %lld (첫 행 %d)\n", order, cmp, bad, first);
    if (best_bad < 0 || bad < best_bad) best_bad = bad, best = order;
  }
  printf("가장 맞는 mm 순서 %d, 다름 %lld\n", best, best_bad);
  return best_bad ? 1 : 0;
}
