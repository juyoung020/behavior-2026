// 층 1 시험 (실제 과제 기록): capture_covered.py 가 공식 평가기에서 가로챈 입력으로 같은 결정을 내는지.
//   ./test_covered_capture ~/engine-data/particles/covered/<과제>_<인스턴스>
// 1) 불러오기: 국소 행렬 = local_mat(입력 위치·방향, 배치 크기 1?)  vs 공식 _particles_local_mat
// 2) 제거 사건마다: 그 순간 링크 행렬·제거기 행렬로 remove_visual → 공식 전후 생존·누적 수
// 3) 링크 scaled_transform(float32) 을 PhysX 자세(float32)·척도로 세우는 식 가설: Fabric 세계 행렬(double) = diag(척도) * R(q) * T(p)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/omni/gfmat.h"
#include "core/particles/visual.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

static inline uint32_t fb(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(d + k + ".npy", a)) {
      fprintf(stderr, "없음 %s%s\n", d.c_str(), k);
      exit(1);
    }
    return a;
  };
  Npy p_link = L("p_link"), p_group = L("p_group"), p_lm = L("p_lm"), in_pos = L("p_in_pos"), in_quat = L("p_in_quat"),
      in_batch = L("p_in_batch"), ev_step = L("ev_step"), ev_kind = L("ev_kind"), ev_tf = L("ev_tf"), ev_hull = L("ev_hull"),
      ev_nh = L("ev_nh"), ev_attr = L("ev_attr"), ev_mb = L("ev_mod_before"), ev_ma = L("ev_mod_after"), ev_lim = L("ev_limit"),
      ev_before = L("ev_before"), ev_after = L("ev_after"), ev_ptf = L("ev_ptf"), ev_has = L("ev_has_ptf"),
      s_tf = L("step_link_tf"), s_pose = L("step_link_pose"), s_fab = L("step_link_fabric"), s_scale = L("step_link_scale");
  const int N = (int)p_link.shape[0], E = (int)ev_step.shape[0], NL = (int)ev_ptf.shape[1], H = (int)ev_hull.shape[1];
  long bad_lm = 0, bad_ev = 0, bad_cnt = 0, n_rm = 0, skipped = 0;
  std::vector<float> lm(N * 16);
  for (int i = 0; i < N; ++i) {
    local_mat(in_pos.as<float>() + 3 * i, in_quat.as<float>() + 4 * i, in_batch.as<int32_t>()[i] == 1, &lm[16 * i]);
    for (int k = 0; k < 16; ++k)
      if (fb(lm[16 * i + k]) != fb(p_lm.as<float>()[16 * i + k])) {
        if (!bad_lm) printf("  국소 행렬 첫 다름: 입자 %d 칸 %d\n", i, k);
        ++bad_lm;
        break;
      }
  }
  // 공식 행렬을 그대로 써서 제거 결정만 따로 본다 (국소 행렬이 다르면 둘 다 보고)
  std::vector<uint8_t> alive(N);
  VisualSet vs{N, p_link.as<int32_t>(), p_group.as<int32_t>(), p_lm.as<float>(), alive.data()};
  for (int e = 0; e < E; ++e) {
    const int kind = ev_kind.as<int32_t>()[e];
    if (kind > 4) {
      ++skipped;
      continue;
    }
    memcpy(alive.data(), ev_before.as<uint8_t>() + (size_t)e * N, N);
    std::vector<float> ltf(NL * 16, 0.0f);
    for (int l = 0; l < NL; ++l)
      if (ev_has.as<uint8_t>()[e * NL + l]) memcpy(&ltf[16 * l], ev_ptf.as<float>() + ((size_t)e * NL + l) * 16, 64);
    RemoverGeom g{kind, ev_nh.as<int32_t>()[e], ev_hull.as<float>() + (size_t)e * H * 3,
                  {ev_attr.as<double>()[3 * e], ev_attr.as<double>()[3 * e + 1], ev_attr.as<double>()[3 * e + 2]}};
    int32_t mod = ev_mb.as<int32_t>()[e];
    n_rm += remove_visual(vs, ltf.data(), g, ev_tf.as<float>() + (size_t)e * 16, ev_lim.as<int32_t>()[e], mod);
    if (memcmp(alive.data(), ev_after.as<uint8_t>() + (size_t)e * N, N) != 0) {
      if (!bad_ev) printf("  제거 첫 다름: 사건 %d (스텝 %d)\n", e, ev_step.as<int32_t>()[e]);
      ++bad_ev;
    }
    bad_cnt += mod != ev_ma.as<int32_t>()[e];
  }
  // 3) Fabric 행렬 가설
  const int S = (int)s_tf.shape[0], SL = (int)s_tf.shape[1];
  long bad_fab = 0, bad_tf = 0, n_fab = 0;
  for (int s = 0; s < S; ++s)
    for (int l = 0; l < SL; ++l) {
      const float* pose = s_pose.as<float>() + ((size_t)s * SL + l) * 7;
      const float* sc = s_scale.as<float>() + ((size_t)s * SL + l) * 6;
      const double* fab = s_fab.as<double>() + ((size_t)s * SL + l) * 16;
      eng::omni::gf::M4 m = eng::omni::gf::from_physx_pose(pose, pose + 3);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m.m[r][c] *= (double)sc[r];
      bool okf = true;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) okf &= m.m[r][c] == fab[r * 4 + c];
      bad_fab += !okf;
      // scaled_transform = float32(fabric).T
      bool okt = true;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) okt &= fb((float)fab[c * 4 + r]) == fb(s_tf.as<float>()[((size_t)s * SL + l) * 16 + r * 4 + c]);
      bad_tf += !okt;
      if (!okf && bad_fab == 1) {
        printf("  Fabric 가설 첫 다름: 스텝 %d 링크 %d  척도 %g %g %g (물체 %g %g %g)\n", s, l, sc[0], sc[1], sc[2], sc[3], sc[4], sc[5]);
        for (int r = 0; r < 4; ++r) printf("    %.17g %.17g %.17g %.17g | %.17g %.17g %.17g %.17g\n", m.m[r][0], m.m[r][1], m.m[r][2], m.m[r][3],
                                           fab[r * 4], fab[r * 4 + 1], fab[r * 4 + 2], fab[r * 4 + 3]);
      }
      ++n_fab;
    }
  printf("실제 과제 기록 재생: 입자 %d, 제거 사건 %d (건너뜀 %ld: Mesh 투영), 우리 판단으로 지운 입자 %ld\n", N, E, skipped, n_rm);
  printf("  국소 행렬 다름 %ld / %d\n  사건 생존 다름 %ld / %ld\n  누적 수 다름 %ld\n", bad_lm, N, bad_ev, E - skipped, bad_cnt);
  printf("  scaled_transform = float32(Fabric)^T : 다름 %ld / %ld\n  Fabric = diag(척도) R(q) T(p) 가설 : 다름 %ld / %ld\n", bad_tf,
         n_fab, bad_fab, n_fab);
  return (bad_lm || bad_ev || bad_cnt) ? 1 : 0;
}
