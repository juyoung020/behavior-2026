// 층 1 시험 (호스트, MKL dlopen): 분무기 원뿔 광선·무리 척도·붙은 입자 국소 행렬 vs 공식.
//   ENGINE_MKL_LIB=.../libtorch_cpu.so ./test_applier ~/engine-data/particles/applier
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/applier.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(d + k + ".npy", a)) exit(1);
    return a;
  };
  Npy st = L("state"), apS = L("ap_scale"), apP = L("ap_p"), apQ = L("ap_q"), ext = L("ext"), rel = L("rel"), aab = L("atom_aabb"), tm = L("tmpl"),
      hS = L("hit_scale"), hTF = L("hit_tf"), hP = L("hit_p"), hQ = L("hit_q"), hO = L("hit_obj"), sp = L("start"), ep = L("end"), sc = L("scales"),
      cd = L("cdims"), lm = L("lm");
  const int N = (int)st.shape[0], B = (int)st.shape[1];
  long bad_ray = 0, bad_sc = 0, bad_cd = 0, bad_lm = 0;
  auto F = [](const Npy& a, int i) { return a.as<float>() + (size_t)i * a.shape[1]; };
  for (int c = 0; c < N; ++c) {
    TorchMT m;
    torch_mt_from_bytes(st.as<uint8_t>() + (size_t)c * B, B, m);
    float s[2][3], e[2][3];
    applier_cone_rays(m, F(ext, c), F(apP, c), F(apQ, c), F(apS, c), s, e);
    bad_ray += memcmp(s, F(sp, c), 24) != 0 || memcmp(e, F(ep, c), 24) != 0;
    float scl[6];
    group_scales(m, F(rel, c)[0] != 0, F(aab, c), F(tm, c), F(apS, c), 2, scl);
    bad_sc += memcmp(scl, F(sc, c), 24) != 0;
    const float avg = avg_scale3(F(apS, c));
    float cdm[6];
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 3; ++k) cdm[3 * i + k] = (F(sc, c)[3 * i + k] * F(tm, c)[k]) * avg;
    bad_cd += memcmp(cdm, F(cd, c), 24) != 0;
    // 붙이기: 무리 첫 등장 순서
    int order[2] = {0, 1};
    const float* ho = F(hO, c);
    if (ho[0] != ho[1]) order[0] = 0, order[1] = 1;  // 다른 무리: 첫 적중 무리 먼저, 그다음
    for (int t = 0; t < 2; ++t) {
      const int h = order[t];
      const int ob = (int)ho[h];
      float out[16];
      attach_local_mat(F(hTF, c) + 16 * ob, F(hP, c) + 3 * h, F(hQ, c) + 4 * h, out);
      bad_lm += memcmp(out, F(lm, c) + 16 * t, 64) != 0;
    }
  }
  printf("분무기 앞뒤 층 1 vs 공식: 사례 %d  광선 다름 %ld  척도 다름 %ld  표본 크기 다름 %ld  국소 행렬 다름 %ld\n", N, bad_ray, bad_sc, bad_cd, bad_lm);
  const bool ok = !(bad_ray || bad_sc || bad_cd || bad_lm);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
