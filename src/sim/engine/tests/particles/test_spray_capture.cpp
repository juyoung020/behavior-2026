// 층 1 시험 (실제 과제 기록, 호스트 MKL): 분무기 갱신마다 공식 입력·공식 광선 적중으로 같은 광선 원점·같은 새 입자 국소 행렬을 내는가.
//   python3 spray_events_to_npy.py <기록 폴더>; ENGINE_MKL_LIB=... ./test_spray_capture <기록 폴더>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/applier.h"
#include "core/particles/spray.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  Npy RG, MI, RO, RY, NO, NW;
  if (!npy_load(d + "ev_rng.npy", RG) || !npy_load(d + "ev_misc.npy", MI) || !npy_load(d + "ev_ray_off.npy", RO) || !npy_load(d + "ev_rays.npy", RY) ||
      !npy_load(d + "ev_new_off.npy", NO) || !npy_load(d + "ev_new.npy", NW))
    return 1;
  const int E = (int)RG.shape[0], B = (int)RG.shape[1];
  long n_ray = 0, bad_ray = 0, n_new = 0, bad_cnt = 0, bad_lm = 0;
  for (int e = 0; e < E; ++e) {
    const float* mi = MI.as<float>() + (size_t)e * MI.shape[1];
    const float *lp = mi, *lq = mi + 3, *osc = mi + 7, *ext = mi + 10, *tmpl = mi + 13, *gmn = mi + 16, *gmx = mi + 19;
    TorchMT m;
    torch_mt_from_bytes(RG.as<uint8_t>() + (size_t)e * B, B, m);
    float st[2][3], en[2][3], sc[6];
    applier_cone_rays(m, ext, lp, lq, osc, st, en);
    group_scales_mm(m, gmn, gmx, osc, 2, sc);
    const float avg = avg_scale3(osc);
    long r = RO.as<int64_t>()[e];
    std::vector<CuboidResult> res;
    for (int s = 0; s < 2; ++s) {
      float cd[3];
      for (int k = 0; k < 3; ++k) cd[k] = (sc[3 * s + k] * tmpl[k]) * avg;
      auto rayfn = [&](const float* o, const float*, float, std::vector<RayHitIn>& out) {
        const double* q = RY.as<double>() + r * 15;
        ++n_ray;
        bad_ray += !((float)q[0] == o[0] && (float)q[1] == o[1] && (float)q[2] == o[2]);
        if (q[6] != 0) {
          RayHitIn h;
          for (int k = 0; k < 3; ++k) h.pos[k] = (float)q[7 + k], h.nrm[k] = (float)q[10 + k];
          h.dist = q[13];
          h.body = (int)q[14];
          out.push_back(h);
        }
        ++r;
      };
      res.push_back(sample_cuboid_one(m, st[s], en[s], cd, rayfn, [](int) { return false; }));
    }
    // 새 입자: 성공한 표본을 무리(맞은 몸체) 첫 등장 순서로
    std::vector<int> order;
    for (int s = 0; s < 2; ++s)
      if (res[s].ok) order.push_back(s);
    if (order.size() == 2 && res[order[0]].body != res[order[1]].body) {
      // 서로 다른 무리: 첫 적중 무리 먼저 (이미 순서대로)
    }
    const long a0 = NO.as<int64_t>()[e], a1 = NO.as<int64_t>()[e + 1];
    n_new += a1 - a0;
    if ((long)order.size() != a1 - a0) {
      if (!bad_cnt) printf("  첫 개수 다름: 사건 %d 우리 %zu 공식 %ld\n", e, order.size(), a1 - a0);
      ++bad_cnt;
      continue;
    }
    for (size_t t = 0; t < order.size(); ++t) {
      const float* w = NW.as<float>() + (size_t)(a0 + t) * 33;
      float lm[16];
      attach_local_mat(w + 16, res[order[t]].centroid, res[order[t]].rot, lm);
      bad_lm += memcmp(lm, w, 64) != 0;
    }
  }
  printf("실제 과제 뿌리기 재생: 도포 사건 %d, 광선 %ld (원점 다름 %ld), 새 입자 %ld (개수 다른 사건 %ld, 국소 행렬 다름 %ld)\n", E, n_ray, bad_ray, n_new, bad_cnt,
         bad_lm);
  const bool ok = !(bad_ray || bad_cnt || bad_lm);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
