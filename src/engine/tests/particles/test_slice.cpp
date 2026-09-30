// 층 1 시험: 자르기 전이의 실수 부분(반쪽 bb 자세·크기, 반쪽 척도, 기준 링크 위치)과 상태 물려받기 — 공식과 비트 비교.
//   정답: python gen_slice_ref.py --out ~/engine-data/particles/slice
//   ./test_slice ~/engine-data/particles/slice
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/slicing.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

static inline bool same(float a, float b) {
  uint32_t x, y;
  memcpy(&x, &a, 4);
  memcpy(&y, &b, 4);
  return x == y;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  Npy in, out, late;
  if (!npy_load(d + "slice_in.npy", in) || !npy_load(d + "slice_out.npy", out) || !npy_load(d + "slice_late.npy", late)) return 1;
  const long n = (long)in.shape[0];
  const int wi = (int)in.shape[1], wo = (int)out.shape[1];
  long bad[5] = {0}, first[5] = {-1, -1, -1, -1, -1};
  for (long r = 0; r < n; ++r) {
    const float* x = in.as<float>() + r * wi;
    const float* y = out.as<float>() + r * wo;
    float bp[3], bo[4], bb[3], sc[3], base[3];
    slice_part_bbox(x, x + 3, x + 7, x + 10, x + 13, x + 17, bp, bo, bb);
    bool ok0 = same(bp[0], y[0]) && same(bp[1], y[1]) && same(bp[2], y[2]);
    bool ok1 = same(bo[0], y[3]) && same(bo[1], y[4]) && same(bo[2], y[5]) && same(bo[3], y[6]);
    bool ok2 = same(bb[0], y[7]) && same(bb[1], y[8]) && same(bb[2], y[9]);
    half_scale(y + 7, y + 10, sc);  // 공식 bb·nativeBB 로
    bool ok3 = same(sc[0], y[13]) && same(sc[1], y[14]) && same(sc[2], y[15]);
    bbox_center_to_base(y, y + 3, y + 13, y + 16, base);  // 공식 bb 자세·척도·offset 로
    bool ok4 = same(base[0], y[19]) && same(base[1], y[20]) && same(base[2], y[21]) && same(y[3], y[22]) && same(y[4], y[23]) &&
               same(y[5], y[24]) && same(y[6], y[25]);
    const bool ok[5] = {ok0, ok1, ok2, ok3, ok4};
    for (int k = 0; k < 5; ++k)
      if (!ok[k] && bad[k]++ == 0) first[k] = r;
  }
  long bad_late = 0;
  for (long r = 0; r < (long)late.shape[0]; ++r) {
    const int* l = late.as<int32_t>() + 3 * r;
    bad_late += l[1] != l[2] - 1;  // 물려받는 상태 = 마지막으로 자른 물체
  }
  const char* nm[5] = {"반쪽 bb 중심", "반쪽 bb 방향", "반쪽 bb 크기", "반쪽 척도", "기준 링크 위치"};
  printf("자르기 전이 층 1 vs 공식 (반쪽 %ld)\n", n);
  long tot = bad_late;
  for (int k = 0; k < 5; ++k) {
    printf("  %-18s 다름 %ld%s", nm[k], bad[k], bad[k] ? "" : "\n");
    if (bad[k]) printf("  첫 다름 행 %ld\n", first[k]);
    tot += bad[k];
  }
  printf("  상태 물려받기(마지막 물체) 다름 %ld / %ld\n", bad_late, (long)late.shape[0]);
  printf(tot ? "실패\n" : "전부 비트 동일\n");
  return tot ? 1 : 0;
}
