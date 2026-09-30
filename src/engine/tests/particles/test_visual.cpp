// 층 1 시험: 시각 입자 국소 행렬·세계 위치, 제거기 AABB·투영 부피·역행렬, 여러 스텝 제거(조건·한도) — 공식 OmniGibson 정답과 비트 비교.
//   정답: python gen_visual_ref.py --out ~/engine-data/particles/visual
//   ./test_visual ~/engine-data/particles/visual [사례 수]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/visual.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

static inline uint32_t fb(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}
struct Tally {
  const char* name;
  long n = 0, bad = 0;
  std::string first;
  void add(bool ok, const char* where) {
    ++n;
    if (!ok && bad++ == 0) first = where;
  }
  void print() const { printf("  %-22s 비교 %9ld  다름 %ld%s%s\n", name, n, bad, bad ? "  첫 다름: " : "", bad ? first.c_str() : ""); }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_visual <dir> [ncase]\n");
    return 2;
  }
  const std::string root = argv[1];
  const int ncase = argc > 2 ? atoi(argv[2]) : 40;
  Tally t_lm{"국소 행렬"}, t_w{"세계 위치"}, t_ab{"제거기 AABB"}, t_inv{"linalg.inv"}, t_loc{"투영 국소점"}, t_in{"투영 부피 판정"},
      t_alive{"제거 뒤 생존"}, t_cnt{"누적 제거 수"}, t_cov{"Covered"};
  char where[256];
  for (int c = 0; c < ncase; ++c) {
    char d[512];
    snprintf(d, sizeof d, "%s/case_%03d/", root.c_str(), c);
    auto L = [&](const char* k) {
      Npy a;
      if (!npy_load(std::string(d) + k + ".npy", a)) {
        fprintf(stderr, "없음 %s%s\n", d, k);
        exit(1);
      }
      return a;
    };
    Npy p_link = L("p_link"), lpos = L("lpos"), lquat = L("lquat"), ref_lm = L("local_mat"), link_tf = L("link_tf"),
        rem_meth = L("rem_meth"), rem_limit = L("rem_limit"), rem_need = L("rem_need_toggle"), rem_hull = L("rem_hull"),
        rem_attr = L("rem_attr"), rem_tf = L("rem_tf"), toggle = L("toggle"), world = L("world"), aabb = L("aabb"),
        alive_after = L("alive_after"), count_after = L("count_after"), proj_pts = L("proj_pts"), proj_inv = L("proj_inv"),
        proj_loc = L("proj_loc"), proj_in = L("proj_in");
    const int N = (int)p_link.shape[0], S = (int)link_tf.shape[0], NL = (int)link_tf.shape[1], R = (int)rem_tf.shape[1],
              H = (int)rem_hull.shape[1], P = (int)proj_pts.shape[2];
    const int32_t* pl = p_link.as<int32_t>();
    std::vector<int> gsize(NL, 0);
    for (int i = 0; i < N; ++i) gsize[pl[i]]++;
    // 국소 행렬 (_load_state: 무리마다 한 번에 set → 배치 크기 = 무리 크기)
    std::vector<float> lm(N * 16);
    for (int i = 0; i < N; ++i) {
      local_mat(lpos.as<float>() + 3 * i, lquat.as<float>() + 4 * i, gsize[pl[i]] == 1, &lm[16 * i]);
      bool ok = true;
      for (int k = 0; k < 16; ++k) ok &= fb(lm[16 * i + k]) == fb(ref_lm.as<float>()[16 * i + k]);
      snprintf(where, sizeof where, "사례 %d 입자 %d", c, i);
      t_lm.add(ok, where);
    }
    std::vector<uint8_t> alive(N, 1);
    VisualSet vs{N, pl, pl, lm.data(), alive.data()};
    std::vector<int32_t> modified(R, 0);
    for (int s = 0; s < S; ++s) {
      const float* ltf = link_tf.as<float>() + (size_t)s * NL * 16;
      for (int i = 0; i < N; ++i) {
        const float* wp = world.as<float>() + ((size_t)s * N + i) * 3;
        if (!alive[i]) continue;
        float p[3];
        particle_world_pos(ltf + 16 * pl[i], &lm[16 * i], p);
        snprintf(where, sizeof where, "사례 %d 스텝 %d 입자 %d", c, s, i);
        t_w.add(fb(p[0]) == fb(wp[0]) && fb(p[1]) == fb(wp[1]) && fb(p[2]) == fb(wp[2]), where);
      }
      for (int r = 0; r < R; ++r) {
        const int kind = rem_meth.as<int32_t>()[r];
        const float* tf = rem_tf.as<float>() + ((size_t)s * R + r) * 16;
        RemoverGeom g{kind, H, rem_hull.as<float>() + (size_t)r * H * 3, {0, 0, 0}};
        for (int k = 0; k < 3; ++k) g.attr[k] = rem_attr.as<double>()[r * 3 + k];
        snprintf(where, sizeof where, "사례 %d 스텝 %d 제거기 %d", c, s, r);
        if (kind == ADJACENCY) {
          float lo[3], hi[3];
          visual_aabb(tf, g.hull, H, lo, hi);
          const float* ab = aabb.as<float>() + ((size_t)s * R + r) * 6;
          bool ok = true;
          for (int k = 0; k < 3; ++k) ok &= fb(lo[k]) == fb(ab[k]) && fb(hi[k]) == fb(ab[3 + k]);
          t_ab.add(ok, where);
        } else {
          float inv[16];
          inv4_mkl(tf, inv);
          const float* ri = proj_inv.as<float>() + ((size_t)s * R + r) * 16;
          bool ok = true;
          for (int k = 0; k < 16; ++k) ok &= fb(inv[k]) == fb(ri[k]);
          t_inv.add(ok, where);
          for (int q = 0; q < P; ++q) {
            const size_t o = ((size_t)s * R + r) * P + q;
            float Lc[3];
            proj_local(inv, proj_pts.as<float>() + 3 * o, Lc);
            const float* rl = proj_loc.as<float>() + 3 * o;
            t_loc.add(fb(Lc[0]) == fb(rl[0]) && fb(Lc[1]) == fb(rl[1]) && fb(Lc[2]) == fb(rl[2]), where);
            t_in.add(in_mesh_volume(kind, g.attr, rl) == (proj_in.as<uint8_t>()[o] != 0), where);
          }
        }
        // 공식 _update: 조건 [켜짐?] + 한도, 그다음 Saturated 재확인 → _modify_particles
        Cond conds[2];
        int nc = 0;
        if (rem_need.as<uint8_t>()[r]) conds[nc++] = Cond{COND_TOGGLED, 1};
        conds[nc++] = Cond{COND_LIMIT, 0};
        int n_alive = 0;
        for (int i = 0; i < N; ++i) n_alive += alive[i];
        const int32_t lim = rem_limit.as<int32_t>()[r];
        if (remover_may_modify(conds, nc, toggle.as<uint8_t>()[s * R + r] != 0, n_alive, lim, modified[r], [](int) { return false; }))
          remove_visual(vs, ltf, g, tf, lim, modified[r]);
        const uint8_t* wa = alive_after.as<uint8_t>() + ((size_t)s * R + r) * N;
        t_alive.add(memcmp(wa, alive.data(), N) == 0, where);
        t_cnt.add(modified[r] == count_after.as<int32_t>()[s * R + r], where);
        for (int gi = 0; gi < NL; ++gi) {
          bool want = false;
          for (int i = 0; i < N; ++i) want |= wa[i] && pl[i] == gi;
          t_cov.add(covered_visual(vs, gi) == want, where);
        }
      }
    }
  }
  // 부피 판정 함수만: 경계 몇 ulp 점 (volume/)
  Tally t_vol{"부피 판정(경계)"};
  {
    const std::string vd = root + "/volume/";
    Npy kinds, attr, pts, inside, bin;
    if (npy_load(vd + "kinds.npy", kinds) && npy_load(vd + "attr.npy", attr) && npy_load(vd + "pts.npy", pts) &&
        npy_load(vd + "inside.npy", inside) && npy_load(vd + "batch_inside.npy", bin)) {
      const long M = (long)kinds.shape[0];
      for (long i = 0; i < M; ++i) {
        snprintf(where, sizeof where, "점 %ld 종류 %d", i, kinds.as<int32_t>()[i]);
        t_vol.add(in_mesh_volume(kinds.as<int32_t>()[i], attr.as<double>() + 3 * i, pts.as<float>() + 3 * i) ==
                      (inside.as<uint8_t>()[i] != 0), where);
      }
      for (long i = 0; i < (long)bin.shape[0]; ++i) {
        snprintf(where, sizeof where, "큰 배치 점 %ld", i);
        t_vol.add(in_mesh_volume(CYLINDER, attr.as<double>(), pts.as<float>() + 3 * i) == (bin.as<uint8_t>()[i] != 0), where);
      }
    }
  }
  printf("시각 입자·제거기 층 1 vs 공식 (%d 사례)\n", ncase);
  const Tally* all[] = {&t_lm, &t_w, &t_ab, &t_inv, &t_loc, &t_in, &t_alive, &t_cnt, &t_cov, &t_vol};
  long bad = 0;
  for (auto* t : all) {
    t->print();
    bad += t->bad;
  }
  printf(bad ? "실패\n" : "전부 비트 동일\n");
  return bad ? 1 : 0;
}
