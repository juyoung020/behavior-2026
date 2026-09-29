// omni 시험 3: 물체 상태 커널 — 우리 C++ (core/omni/states.h, warp_f32.h) = 공식 OmniGibson warp 커널 (CUDA, gen_states_ref.py).
//   test_states <정답 폴더(~/engine-data/omni/states)>
// 단계마다 공식 입력을 그대로 넣어 모듈별로 비교한다(앞 단계 오차가 뒤로 번지지 않게). 실수는 비트, 불은 완전 일치.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/omni/states.h"
#include "tests/omni/npy.h"

using namespace eng::omni;
using wf::M44;

struct Tally {
  long long n = 0, bad = 0;
  std::string first;
  void f(float ours, float ref, const std::string& what) {
    ++n;
    if (memcmp(&ours, &ref, 4) != 0) {
      if (!bad) {
        char b[256];
        snprintf(b, sizeof b, "%s 우리 %.9g 정답 %.9g", what.c_str(), ours, ref);
        first = b;
      }
      ++bad;
    }
  }
  void u(long long ours, long long ref, const std::string& what) {
    ++n;
    if (ours != ref) {
      if (!bad) first = what + " 우리 " + std::to_string(ours) + " 정답 " + std::to_string(ref);
      ++bad;
    }
  }
  void print(const char* name) const {
    printf("%-28s 비교 %10lld 다름 %6lld %s\n", name, n, bad, bad ? ("첫: " + first).c_str() : "");
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_states <ref dir>\n");
    return 2;
  }
  const std::string root = argv[1];
  std::ifstream idx(root + "/index.txt");
  std::string fr;
  Tally t_mat, t_aabb, t_open, t_cm, t_ccm, t_btf, t_touch, t_pre, t_invw, t_out, t_inside, t_adj, t_ontop, t_under,
      t_nextto, t_tg_contact, t_tg_overlap, t_tg_val, t_tg_time;
  int frames = 0;
  while (std::getline(idx, fr)) {
    if (fr.empty()) continue;
    ++frames;
    const std::string d = root + "/" + fr + "/";
    auto L_ = [&](const char* n) {
      Npy a;
      if (!npy_load(d + n + ".npy", a)) throw std::runtime_error(std::string("missing ") + n);
      return a;
    };
    auto has = [&](const char* n) { return std::ifstream(d + n + ".npy").good(); };
    const auto SOL = L_("S").vec<int32_t>();
    const int S = SOL[0], O = SOL[1], L = SOL[2];
    const auto poses = L_("poses").vec<float>();
    const auto mats_ref = L_("pose_matrices").vec<float>();
    const auto link_scene = L_("link_scene").vec<int32_t>();
    const auto link_obj = L_("link_obj").vec<int32_t>();
    const auto mpts = L_("mesh_pts").vec<float>();
    const auto mtri = L_("mesh_tri").vec<int32_t>();
    const auto poff = L_("pts_off").vec<int32_t>();
    const auto toff = L_("tri_off").vec<int32_t>();
    const auto has_mesh = L_("has_mesh").vec<uint8_t>();
    std::vector<M44> M(L);
    // 1. 자세 → 행렬
    for (int l = 0; l < L; ++l) {
      const M44 m = wf::pose_to_mat(&poses[l * 7]);
      for (int e = 0; e < 16; ++e) t_mat.f(m.m[e], mats_ref[l * 16 + e], fr + " 행렬 " + std::to_string(l));
      memcpy(M[l].m, &mats_ref[l * 16], 64);  // 다음 단계는 공식 행렬로
    }
    // 2. AABB
    {
      const auto aabb_ref = L_("aabb").vec<float>();
      const auto base_link = L_("aabb_base_link").vec<int32_t>();
      std::vector<float> aabb(S * O * 6);
      for (int r = 0; r < S * O; ++r) st::aabb_init(&aabb[r * 6]);
      for (int l = 0; l < L; ++l) {
        if (!has_mesh[l]) continue;
        const int row = link_scene[l] * O + link_obj[l];
        for (int v = 0; v < poff[l + 1] - poff[l]; ++v) {
          float w[3];
          st::aabb_vertex_world(M[l], &mpts[(poff[l] + v) * 3], w);
          st::aabb_accumulate(&aabb[row * 6], w);
        }
      }
      for (int b : base_link) st::aabb_fallback(&aabb[(link_scene[b] * O + link_obj[b]) * 6], &poses[b * 7]);
      for (size_t e = 0; e < aabb.size(); ++e) t_aabb.f(aabb[e], aabb_ref[e], fr + " aabb " + std::to_string(e));
    }
    const auto aabb = L_("aabb").vec<float>();  // (S*O,6) = (S,O,6)
    // 3. Open
    {
      const auto jp = L_("open_jp"), rows = L_("open_rows"), mask = L_("open_mask"), th1 = L_("open_th1"),
                 d1 = L_("open_d1"), th2 = L_("open_th2"), d2 = L_("open_d2"), both = L_("open_both"), out = L_("open_out");
      const int D = (int)jp.shape[1];
      for (int s = 0; s < S; ++s)
        for (int o = 0; o < O; ++o) {
          const int k = s * O + o;
          const uint8_t v = st::open_value(jp.as<float>() + (size_t)rows.as<int32_t>()[k] * D, D, mask.as<uint8_t>() + k * D,
                                           th1.as<float>() + k * D, d1.as<float>() + k * D, th2.as<float>() + k * D,
                                           d2.as<float>() + k * D, both.as<uint8_t>()[k]);
          t_open.u(v, out.as<uint8_t>()[k], fr + " open");
        }
    }
    // 4. 접촉 행렬 + 몸체 자세
    {
      const auto all_tf = L_("cm_all_tf"), prev = L_("cm_prev_tf"), net = L_("cm_net"), b2r = L_("cm_b2r"),
                 imp = L_("cm_imp"), rows = L_("cm_rows"), cols = L_("cm_cols"), ns = L_("cm_nsteps");
      auto cm = L_("cm_in").vec<uint8_t>();
      auto ccm = L_("ccm_in").vec<uint8_t>();
      auto btf = prev.vec<float>();
      const int B = (int)prev.shape[0], R = (int)rows.shape[0], C = (int)cols.shape[0];
      st::ContactIn in{all_tf.as<float>(), btf.data(), net.as<float>(), b2r.as<int32_t>(), B, R, 1e-6f, 1e-4f};
      const int n_steps = ns.as<int32_t>()[0];
      for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
          st::contact_update_rc(in, imp.as<float>(), C, rows.as<int32_t>(), cols.as<int32_t>(), n_steps, r, c, cm.data(),
                                ccm.data());
      // 공식: 몸체 커널이 prev_tf(=body_tf) 를 제자리로 쓴다. 스레드마다 자기 b 만 쓰고 읽는 것도 자기 b 뿐이라 순서 무관.
      for (int b = 0; b < B; ++b) st::body_transform_update(in, n_steps, b, btf.data());
      const auto cm_ref = L_("cm_out").vec<uint8_t>(), ccm_ref = L_("ccm_out").vec<uint8_t>();
      const auto btf_ref = L_("body_tf_out").vec<float>();
      for (size_t e = 0; e < cm.size(); ++e) t_cm.u(cm[e], cm_ref[e], fr + " cm");
      for (size_t e = 0; e < ccm.size(); ++e) t_ccm.u(ccm[e], ccm_ref[e], fr + " ccm");
      for (size_t e = 0; e < btf.size(); ++e) t_btf.f(btf[e], btf_ref[e], fr + " body_tf");
    }
    const int N = O;
    // 5. Touching
    {
      const auto ref = L_("touching").vec<uint8_t>();
      for (int s = 0; s < S; ++s) {
        const std::string ss = std::to_string(s);
        const auto rm = L_(("t_rows_" + ss).c_str()), cmk = L_(("t_cols_" + ss).c_str()), cmat = L_(("t_cm_" + ss).c_str());
        const int R = (int)rm.shape[1], C = (int)cmk.shape[1];
        std::vector<uint8_t> o2c((size_t)N * C);
        for (int i = 0; i < N; ++i)
          for (int c = 0; c < C; ++c) o2c[(size_t)i * C + c] = st::touching_obj_to_col(rm.as<uint8_t>(), cmat.as<uint8_t>(), R, C, i, c);
        std::vector<int32_t> pair((size_t)N * N, 0);
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j)
            for (int c = 0; c < C; ++c)
              if (o2c[(size_t)i * C + c] && cmk.as<uint8_t>()[(size_t)j * C + c]) pair[(size_t)i * N + j] = 1;
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j)
            t_touch.u(st::touching_finalize(pair.data(), N, i, j), ref[((size_t)s * N + i) * N + j], fr + " touching");
      }
    }
    // 6. Inside
    {
      const auto aidx = L_("in_aidx").vec<int32_t>();
      const int Mn = L_("in_M").vec<int32_t>()[0];
      const auto pre_ref = L_("in_prefilter").vec<int32_t>();
      std::vector<int32_t> pre((size_t)S * N * N);
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) {
            pre[((size_t)s * N + i) * N + j] = st::inside_prefilter(&aabb[(size_t)s * O * 6], aidx.data(), i, j);
            t_pre.u(pre[((size_t)s * N + i) * N + j], pre_ref[((size_t)s * N + i) * N + j], fr + " prefilter");
          }
      std::vector<int32_t> pair((size_t)S * N * N, 0);
      if (Mn > 0) {
        const auto mcont = L_("in_mesh_container").vec<int32_t>(), mscene = L_("in_mesh_scene").vec<int32_t>(),
                   mpar = L_("in_mesh_parent").vec<int32_t>(), f2m = L_("in_f2m").vec<int32_t>();
        const auto minv = L_("in_mesh_inv").vec<float>(), fc = L_("in_fc").vec<float>(), fn = L_("in_fn").vec<float>();
        const auto invw_ref = L_("in_inv_world").vec<float>();
        const auto out_ref = L_("in_outside").vec<int32_t>();
        std::vector<M44> invw(Mn);
        for (int m = 0; m < Mn; ++m) {
          M44 il;
          memcpy(il.m, &minv[m * 16], 64);
          const M44 w = st::inside_inv_world(M[mpar[m]], il);
          for (int e = 0; e < 16; ++e) t_invw.f(w.m[e], invw_ref[m * 16 + e], fr + " inv_world");
          memcpy(invw[m].m, &invw_ref[m * 16], 64);
        }
        const int F = (int)f2m.size();
        std::vector<int32_t> outside((size_t)S * N * Mn, 0);
        for (int s = 0; s < S; ++s)
          for (int i = 0; i < N; ++i)
            for (int f = 0; f < F; ++f) {
              const int m = f2m[f];
              const int cj = mcont[m];
              if (cj < 0 || mscene[m] != s || i == cj) continue;
              const int ai = aidx[i];
              if (ai < 0) continue;
              if (pre_ref[((size_t)s * N + i) * N + cj] == 0) continue;
              float c[3];
              st::aabb_center(&aabb[((size_t)s * O + ai) * 6], c);
              if (st::inside_face_outside(invw[m], c, &fc[f * 3], &fn[f * 3])) outside[((size_t)s * N + i) * Mn + m] = 1;
            }
        for (size_t e = 0; e < outside.size(); ++e) t_out.u(outside[e], out_ref[e], fr + " outside");
        for (int s = 0; s < S; ++s)
          for (int i = 0; i < N; ++i)
            for (int m = 0; m < Mn; ++m) {
              const int cj = mcont[m];
              if (cj < 0 || mscene[m] != s || i == cj || aidx[i] < 0) continue;
              if (pre_ref[((size_t)s * N + i) * N + cj] == 0) continue;
              if (out_ref[((size_t)s * N + i) * Mn + m] == 0) pair[((size_t)s * N + i) * N + cj] = 1;
            }
      }
      const auto ref = L_("inside").vec<uint8_t>();
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j)
            t_inside.u(st::inside_finalize(pair.data() + (size_t)s * N * N, N, i, j), ref[((size_t)s * N + i) * N + j],
                       fr + " inside");
    }
    // 7. Adjacency
    const auto adj_ref = L_("adjacency").vec<uint8_t>();
    {
      const auto dirs = L_("adj_dirs").vec<float>(), maxd = L_("adj_maxd").vec<float>();
      const auto l2o = L_("adj_l2o").vec<int32_t>(), l2s = L_("adj_l2s").vec<int32_t>();
      const auto aidx = L_("in_aidx").vec<int32_t>();
      const int K = (int)maxd.size();
      std::vector<uint8_t> adj((size_t)S * N * N * K, 0);
      for (int s = 0; s < S; ++s)
        for (int a = 0; a < N; ++a) {
          const int ai = aidx[a];
          if (ai < 0) continue;
          float o_w[3];
          st::aabb_center(&aabb[((size_t)s * O + ai) * 6], o_w);
          for (int l = 0; l < L; ++l) {
            if (!has_mesh[l] || l2s[l] != s) continue;
            const int b = l2o[l];
            if (b < 0 || b == a) continue;
            for (int k = 0; k < K; ++k) {
              float o_l[3], d_l[3];
              st::adjacency_ray_local(M[l], o_w, &dirs[k * 3], o_l, d_l);
              if (st::mesh_anyhit_bruteforce(&mpts[poff[l] * 3], &mtri[toff[l] * 3], toff[l + 1] - toff[l], o_l, d_l, maxd[k]))
                adj[(((size_t)s * N + a) * N + b) * K + k] = 1;
            }
          }
        }
      for (size_t e = 0; e < adj.size(); ++e) t_adj.u(adj[e], adj_ref[e], fr + " adjacency " + std::to_string(e));
      // 8. OnTop / Under / NextTo (공식 touching·adjacency·aabb 입력)
      const auto touch = L_("touching").vec<uint8_t>();
      const auto tidx = L_("st_tidx").vec<int32_t>(), jidx = L_("st_jdx").vec<int32_t>();
      const auto ot = L_("on_top").vec<uint8_t>(), un = L_("under").vec<uint8_t>(), nt = L_("next_to").vec<uint8_t>();
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) {
            const size_t e = ((size_t)s * N + i) * N + j;
            const uint8_t* adj_s = adj_ref.data() + (size_t)s * N * N * K;
            t_ontop.u(st::on_top_value(touch.data() + (size_t)s * N * N, N, adj_s, N, K, tidx.data(), jidx.data(), i, j), ot[e],
                      fr + " on_top");
            t_under.u(st::under_value(adj_s, N, K, jidx.data(), i, j), un[e], fr + " under");
            t_nextto.u(st::next_to_value(&aabb[(size_t)s * O * 6], adj_s, N, K, aidx.data(), jidx.data(), i, j, 2, K), nt[e],
                       fr + " next_to");
          }
    }
    // 9. ToggledOn (단계마다 공식 입력)
    if (has("tg_R")) {
      const auto RC = L_("tg_R").vec<int32_t>();
      const int Rt = RC[0], Ct = RC[1];
      const auto mref_c = L_("tg_mask_contact").vec<int32_t>();
      for (int s = 0; s < S; ++s) {
        const std::string ss = std::to_string(s);
        const auto q = L_(("tg_q_" + ss).c_str()).vec<uint8_t>(), cm = L_(("tg_cm_" + ss).c_str()).vec<uint8_t>(),
                   wm = L_(("tg_with_" + ss).c_str()).vec<uint8_t>();
        for (int o = 0; o < O; ++o)
          t_tg_contact.u(st::toggle_contact(q.data(), cm.data(), Rt, Ct, &wm[(size_t)o * Ct]), mref_c[s * O + o], fr + " tg_contact");
      }
      auto vals = L_("tg_vals0").vec<uint8_t>();
      auto time = L_("tg_time0").vec<float>();
      auto mask = mref_c;
      const auto rc_this = L_("tg_rc_this").vec<int32_t>(), rc_open = L_("tg_rc_open").vec<int32_t>();
      const auto openf = L_("tg_open").vec<uint8_t>();
      for (size_t i = 0; i < rc_this.size(); ++i)
        if (openf[rc_open[i]]) {
          vals[rc_this[i]] = 0;
          time[rc_this[i]] = 0.0f;
          mask[rc_this[i]] = 0;
        }
      const auto mkp = L_("tg_mk_parent").vec<int32_t>();
      const auto mko = L_("tg_mk_off").vec<float>();
      const auto mkr = L_("tg_mk_rad").vec<float>();
      const auto pairs = L_("tg_pairs").vec<int32_t>();
      for (size_t p = 0; p * 2 + 1 < pairs.size(); ++p) {
        const int k = pairs[p * 2], fg = pairs[p * 2 + 1];
        if (mask[k] != 1 || !has_mesh[fg]) continue;
        if (st::toggle_marker_overlap(M[mkp[k]], &mko[k * 3], mkr[k], M[fg], &mpts[poff[fg] * 3], &mtri[toff[fg] * 3],
                                      toff[fg + 1] - toff[fg]))
          mask[k] = 2;
      }
      const auto mref_s = L_("tg_mask_set").vec<int32_t>();
      for (int k = 0; k < S * O; ++k) t_tg_overlap.u(mask[k], mref_s[k], fr + " tg_overlap " + std::to_string(k));
      for (int k = 0; k < S * O; ++k) st::toggle_set_value(&vals[k], &mask[k], &time[k], 0.15f, (float)(1.0 / 30.0));
      const auto vref = L_("tg_vals").vec<uint8_t>();
      const auto tref = L_("tg_time").vec<float>();
      for (int k = 0; k < S * O; ++k) {
        t_tg_val.u(vals[k], vref[k], fr + " tg_val");
        t_tg_time.f(time[k], tref[k], fr + " tg_time");
      }
    }
  }
  printf("장면 %d 개\n", frames);
  t_tg_contact.print("Toggle 손가락 접촉 표시");
  t_tg_overlap.print("Toggle 표식 겹침 표시");
  t_tg_val.print("ToggledOn 값");
  t_tg_time.print("ToggledOn 누적 시간 (float)");
  t_mat.print("자세→행렬 (float)");
  t_aabb.print("AABB (float)");
  t_open.print("Open");
  t_cm.print("접촉 행렬 (구간 중 닿음)");
  t_ccm.print("접촉 행렬 (지금 닿음)");
  t_btf.print("몸체 기준 자세 (float)");
  t_touch.print("Touching");
  t_pre.print("Inside 사전 거르기");
  t_invw.print("Inside 역변환 (float)");
  t_out.print("Inside 면 밖 표시");
  t_inside.print("Inside");
  t_adj.print("Adjacency (광선 18방향)");
  t_ontop.print("OnTop");
  t_under.print("Under");
  t_nextto.print("NextTo");
  const long long bad = t_mat.bad + t_aabb.bad + t_open.bad + t_cm.bad + t_ccm.bad + t_btf.bad + t_touch.bad + t_pre.bad +
                        t_invw.bad + t_out.bad + t_inside.bad + t_adj.bad + t_ontop.bad + t_under.bad + t_nextto.bad +
                        t_tg_contact.bad + t_tg_overlap.bad + t_tg_val.bad + t_tg_time.bad;
  return bad ? 1 : 0;
}
