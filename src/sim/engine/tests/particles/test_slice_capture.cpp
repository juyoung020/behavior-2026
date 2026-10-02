// 층 1 시험 (실제 과제 기록): 공식 자르기 한 번의 반쪽마다 bb 자세·크기 → 척도 → 기준 링크 위치 → PhysX 에 들어간 자세.
//   python3 slice_events_to_npy.py <기록 폴더>; ./test_slice_capture <기록 폴더>
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/slicing.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

static bool same(const float* a, const float* b, int n) { return memcmp(a, b, 4 * n) == 0; }

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  Npy R;
  if (!npy_load(std::string(argv[1]) + "/slice_rows.npy", R)) return 1;
  const int n = (int)R.shape[0], w = (int)R.shape[1];
  long b_bb = 0, b_sc = 0, b_set = 0, b_after = 0;
  for (int i = 0; i < n; ++i) {
    const float* x = R.as<float>() + (size_t)i * w;
    const float *pos = x, *orn = x + 3, *scale = x + 7, *pbp = x + 10, *pbo = x + 13, *pbs = x + 17;
    const float *obp = x + 20, *obo = x + 23, *obb = x + 27, *nat = x + 30, *off = x + 33, *hsc = x + 36, *sbp = x + 39, *sbo = x + 42,
                *ap = x + 46, *ao = x + 49;
    float bp[3], bo[4], bb[3], sc[3], base[3];
    slice_part_bbox(pos, orn, scale, pbp, pbo, pbs, bp, bo, bb);
    b_bb += !(same(bp, obp, 3) && same(bo, obo, 4) && same(bb, obb, 3));
    half_scale(bb, nat, sc);
    b_sc += !same(sc, hsc, 3);
    b_set += !(same(bp, sbp, 3) && same(bo, sbo, 4));
    bbox_center_to_base(bp, bo, sc, off, base);
    const bool ok_after = same(base, ap, 3) && same(bo, ao, 4);
    if (!ok_after && !b_after)
      printf("  첫 자세 다름: 반쪽 %d  우리 %.9g %.9g %.9g / PhysX %.9g %.9g %.9g  방향 %.9g %.9g %.9g %.9g / %.9g %.9g %.9g %.9g\n", i, base[0], base[1], base[2],
             ap[0], ap[1], ap[2], bo[0], bo[1], bo[2], bo[3], ao[0], ao[1], ao[2], ao[3]);
    b_after += !ok_after;
  }
  printf("실제 자르기 재생: 반쪽 %d  bb 자세·크기 다름 %ld  척도 다름 %ld  set 인자 다름 %ld  PhysX 자세 다름 %ld\n", n, b_bb, b_sc, b_set, b_after);
  const bool ok = !(b_bb || b_sc || b_set || b_after);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
