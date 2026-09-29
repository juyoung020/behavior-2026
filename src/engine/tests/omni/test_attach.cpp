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
  // _attach 자세 맞춤
  std::ifstream h(std::string(argv[1]) + "/attach_pose.bin", std::ios::binary | std::ios::ate);
  long long bpose = 0, cpose = 0;
  if (h) {
    const size_t nh = (size_t)h.tellg();
    h.seekg(0);
    std::vector<float> u(nh / 4);
    h.read((char*)u.data(), nh);
    for (size_t r = 0; r + 28 <= u.size(); r += 28) {
      const float* x = &u[r];
      float np[3], nq[4];
      att::attach_root_pose(x, x + 3, x + 7, x + 10, x + 14, x + 17, np, nq);
      for (int k = 0; k < 7; ++k) {
        const float o = k < 3 ? np[k] : nq[k - 3];
        ++cpose;
        bpose += memcmp(&o, &x[21 + k], 4) != 0;
      }
    }
    printf("붙일 때 뿌리 자세: 비교 %lld 다름 %lld\n", cpose, bpose);
  }
  return (bp || bo || bd || bpose) ? 1 : 0;
}
