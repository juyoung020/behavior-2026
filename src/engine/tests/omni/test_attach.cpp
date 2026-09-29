// omni 시험 6: AttachedTo 붙이기 판정 (core/omni/attach.h) = 공식 (gen_attach_ref.py)
//   test_attach <~/engine-data/omni/attach>
// 행 = child_pos3 child_q4 parent_pos3 parent_q4 | pos_diff orn_diff attached
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/omni/attach.h"

using namespace eng::omni;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::ifstream g(std::string(argv[1]) + "/attach.bin", std::ios::binary | std::ios::ate);
  const size_t nb = (size_t)g.tellg();
  g.seekg(0);
  std::vector<float> v(nb / 4);
  g.read((char*)v.data(), nb);
  const int W = 17, N = (int)(v.size() / W);
  long long bp = 0, bo = 0, bd = 0, bd64 = 0, near = 0;
  int fp = -1, fo = -1;
  for (int r = 0; r < N; ++r) {
    const float* x = &v[(size_t)r * W];
    const float pd = att::pos_diff(x, x + 7), od = att::orientation_diff(x + 3, x + 10);
    if (memcmp(&pd, &x[14], 4)) bp++, fp = fp < 0 ? r : fp;
    if (memcmp(&od, &x[15], 4)) bo++, fo = fo < 0 ? r : fo;
    const bool a = att::aligned(x, x + 3, x + 7, x + 10);
    const bool a64 = (double)x[14] < 0.05 && (double)x[15] < 0.2617993950843811;
    bd += a != (x[16] != 0.0f);
    bd64 += a64 != (x[16] != 0.0f);
    near += (a != a64);
  }
  printf("행 %d: pos_diff 다름 %lld (첫 %d), orn_diff 다름 %lld (첫 %d), 붙음 판정 다름 %lld (double 문턱으로 하면 %lld, 두 방식이 갈리는 행 %lld)\n",
         N, bp, fp, bo, fo, bd, bd64, near);
  return (bp || bo || bd) ? 1 : 0;
}
