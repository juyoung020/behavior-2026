// omni 시험 4 (층 2): 판 N 개 CUDA = 층 1 C++ = 공식 파이썬 정답. 같은 입력으로 비트 비교 + 처리량.
//   test_omni_gpu <~/engine-data/omni> [판 수]
//   A. BDDL: 100과제, 과제마다 판 N 개(원자값 무작위) — GPU 판정·q_score 를 층 1 과 비교
//   B. 제어기: 공식 기록(ctrl.bin) 4 판을 판 N 개로 복제해 매 스텝 GPU 드라이브 목표를 공식 값과 비교
//   C. 물체 상태: 공식 warp 결과(states/frame_*) 의 자세→행렬, Adjacency 를 GPU 로 다시 계산해 비교
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda/omni/omni_kernels.cuh"
#include "tests/omni/npy.h"

using namespace eng::omni;

#define CK(x)                                                                           \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

template <class T>
static T* dput(const std::vector<T>& v) {
  T* p = nullptr;
  CK(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
  if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template <class T>
static std::vector<T> dget(const T* p, size_t n) {
  std::vector<T> v(n);
  if (n) CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string read_problem(const std::string& path) {
  std::ifstream f(path);
  std::string line, out;
  std::getline(f, line);
  std::getline(f, line);
  while (std::getline(f, line) && line != "PROBLEM_END") out += line + "\n";
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_omni_gpu <~/engine-data/omni> [boards]\n");
    return 2;
  }
  const std::string root = argv[1];
  const int boards = argc > 2 ? atoi(argv[2]) : 4096;
  const std::string parts = argc > 3 ? argv[3] : "ABC";
  long long total_bad = 0;
  std::mt19937 rng(5);

  // ---------------- A. BDDL ----------------
  if (parts.find('A') != std::string::npos) {
    std::ifstream idx(root + "/bddl/index.txt");
    std::string name;
    long long cmp = 0, bad = 0;
    double t_eval = 0, t_q = 0;
    long long evals = 0;
    while (std::getline(idx, name)) {
      if (name.empty()) continue;
      const auto c = bddl::compile_problem(bddl::parse_problem(read_problem(root + "/bddl/" + name + ".txt")));
      const int na = (int)c.atoms.size(), nn = (int)c.nodes.size(), nh = (int)c.heads.size();
      const int no = (int)c.opt_start.size() - 1;
      std::vector<uint8_t> now((size_t)boards * na), ini((size_t)boards * na);
      for (auto& v : now) v = rng() & 1;
      for (auto& v : ini) v = rng() & 1;
      auto* d_nodes = dput(c.nodes);
      auto* d_kids = dput(c.kids);
      auto* d_heads = dput(c.heads);
      auto* d_now = dput(now);
      auto* d_ini = dput(ini);
      auto* d_os = dput(c.opt_start);
      auto* d_lits = dput(c.opt_lits);
      uint8_t *d_scr, *d_hv, *d_ok;
      double* d_q;
      CK(cudaMalloc(&d_scr, (size_t)boards * std::max(nn, 1)));
      CK(cudaMalloc(&d_hv, (size_t)boards * std::max(nh, 1)));
      CK(cudaMalloc(&d_ok, boards));
      CK(cudaMalloc(&d_q, boards * sizeof(double)));
      CK(cudaDeviceSynchronize());
      double t0 = now_s();
      gpu::k_bddl_eval<<<(boards + 127) / 128, 128>>>(d_nodes, nn, d_kids, d_heads, nh, na, d_now, d_scr, d_hv, d_ok, boards);
      CK(cudaDeviceSynchronize());
      t_eval += now_s() - t0;
      t0 = now_s();
      gpu::k_bddl_qscore<<<boards, 256>>>(d_os, no, d_lits, na, d_now, d_ini, d_ok, d_q);
      CK(cudaDeviceSynchronize());
      t_q += now_s() - t0;
      evals += boards;
      const auto ok = dget(d_ok, boards);
      const auto hv = dget(d_hv, (size_t)boards * nh);
      const auto q = dget(d_q, boards);
      std::vector<uint8_t> scr(nn), h(nh);
      for (int b = 0; b < boards; ++b) {
        const bool s = bddl::eval_goal(c.nodes.data(), nn, c.kids.data(), c.heads.data(), nh, &now[(size_t)b * na], scr.data(),
                                       h.data());
        const double qc = bddl::q_score(s, c.opt_start.data(), no, c.opt_lits.data(), &now[(size_t)b * na], &ini[(size_t)b * na]);
        cmp += 2 + nh;
        bad += (ok[b] != (uint8_t)s);
        for (int k = 0; k < nh; ++k) bad += (hv[(size_t)b * nh + k] != h[k]);
        bad += memcmp(&qc, &q[b], 8) != 0;
      }
      cudaFree(d_nodes), cudaFree(d_kids), cudaFree(d_heads), cudaFree(d_now), cudaFree(d_ini), cudaFree(d_os);
      cudaFree(d_lits), cudaFree(d_scr), cudaFree(d_hv), cudaFree(d_ok), cudaFree(d_q);
    }
    printf("A. BDDL 판정 GPU=CPU   판 %d x 100과제, 비교 %lld, 다름 %lld | 판정 %.1f 만 판/초, q_score %.1f 만 판/초\n", boards, cmp,
           bad, evals / t_eval / 1e4, evals / t_q / 1e4);
    total_bad += bad;
  }

  // ---------------- B. 제어기 ----------------
  if (parts.find('B') != std::string::npos) {
    FILE* f = fopen((root + "/ctrl/config.bin").c_str(), "rb");
    ctrl::R1ProConfig cfg{};
    int lay[25];
    fread(&cfg.n_dof, 4, 1, f);
    fread(lay, 4, 25, f);
    int k = 0;
    for (int i = 0; i < 3; ++i) cfg.base_dof[i] = lay[k++];
    for (int i = 0; i < 4; ++i) cfg.trunk_dof[i] = lay[k++];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 7; ++i) cfg.arm_dof[a][i] = lay[k++];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 2; ++i) cfg.grip_dof[a][i] = lay[k++];
    fread(cfg.pos_lo, 4, cfg.n_dof, f), fread(cfg.pos_hi, 4, cfg.n_dof, f), fread(cfg.vel_lo, 4, cfg.n_dof, f);
    fread(cfg.vel_hi, 4, cfg.n_dof, f), fread(cfg.has_limit, 1, cfg.n_dof, f);
    fclose(f);
    const float in_lo[3] = {-1, -1, -1}, in_hi[3] = {1, 1, 1}, out_lo[3] = {-0.75f, -0.75f, -1}, out_hi[3] = {0.75f, 0.75f, 1};
    memcpy(cfg.base_in_lo, in_lo, 12), memcpy(cfg.base_in_hi, in_hi, 12), memcpy(cfg.base_out_lo, out_lo, 12);
    memcpy(cfg.base_out_hi, out_hi, 12);
    ctrl::init(cfg);
    f = fopen((root + "/ctrl/ctrl.bin").c_str(), "rb");
    int members, steps;
    fread(&members, 4, 1, f), fread(&steps, 4, 1, f);
    const int per = std::max(1, boards / members);
    const int NB = per * members;
    ctrl::R1ProConfig* d_cfg;
    CK(cudaMalloc(&d_cfg, sizeof(cfg)));
    CK(cudaMemcpy(d_cfg, &cfg, sizeof(cfg), cudaMemcpyHostToDevice));
    std::vector<ctrl::R1ProState> st(NB);
    for (auto& s : st) ctrl::reset(s);
    auto* d_st = dput(st);
    std::vector<uint8_t> has(NB);
    std::vector<float> act((size_t)NB * 23), bp((size_t)NB * 3), bq((size_t)NB * 4), rp((size_t)NB * 3), rq((size_t)NB * 4),
        q((size_t)NB * cfg.n_dof);
    uint8_t* d_has;
    float *d_act, *d_bp, *d_bq, *d_rp, *d_rq, *d_q;
    CK(cudaMalloc(&d_has, NB));
    CK(cudaMalloc(&d_act, act.size() * 4));
    CK(cudaMalloc(&d_bp, bp.size() * 4));
    CK(cudaMalloc(&d_bq, bq.size() * 4));
    CK(cudaMalloc(&d_rp, rp.size() * 4));
    CK(cudaMalloc(&d_rq, rq.size() * 4));
    CK(cudaMalloc(&d_q, q.size() * 4));
    ctrl::DriveTargets* d_out;
    CK(cudaMalloc(&d_out, (size_t)NB * sizeof(ctrl::DriveTargets)));
    long long cmp = 0, bad = 0, nan_payload = 0;  // NaN 끼리: GPU 는 표준 NaN(0x7fffffff), x86 은 입력 NaN 무늬를 보존 → 값 무늬만 다름
    double t_k = 0;
    std::vector<float> ref_pt((size_t)members * 28), ref_vt((size_t)members * 28);
    for (int s = 0; s < steps; ++s) {
      for (int m = 0; m < members; ++m) {
        uint8_t u;
        float a[23], b3[3], b4[4], r3[3], r4[4], qq[28];
        fread(&u, 1, 1, f), fread(a, 4, 23, f), fread(b3, 4, 3, f), fread(b4, 4, 4, f), fread(r3, 4, 3, f), fread(r4, 4, 4, f);
        fread(qq, 4, 28, f), fread(&ref_pt[m * 28], 4, 28, f), fread(&ref_vt[m * 28], 4, 28, f);
        for (int r = 0; r < per; ++r) {
          const int b = m * per + r;
          has[b] = u;
          memcpy(&act[(size_t)b * 23], a, 92), memcpy(&bp[(size_t)b * 3], b3, 12), memcpy(&bq[(size_t)b * 4], b4, 16);
          memcpy(&rp[(size_t)b * 3], r3, 12), memcpy(&rq[(size_t)b * 4], r4, 16), memcpy(&q[(size_t)b * 28], qq, 112);
        }
      }
      CK(cudaMemcpy(d_has, has.data(), NB, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_act, act.data(), act.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_bp, bp.data(), bp.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_bq, bq.data(), bq.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_rp, rp.data(), rp.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_rq, rq.data(), rq.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaDeviceSynchronize());
      const double t0 = now_s();
      gpu::k_ctrl_step<<<(NB + 127) / 128, 128>>>(d_cfg, d_st, d_has, d_act, d_bp, d_bq, d_rp, d_rq, d_q, d_out, NB);
      CK(cudaDeviceSynchronize());
      t_k += now_s() - t0;
      const auto out = dget(d_out, NB);
      for (int b = 0; b < NB; ++b) {
        const int m = b / per;
        for (int d = 0; d < cfg.n_dof; ++d) {
          if (out[b].set_pos[d]) {
            ++cmp;
            bool x = memcmp(&out[b].pos[d], &ref_pt[m * 28 + d], 4) != 0;
            if (x && std::isnan(out[b].pos[d]) && std::isnan(ref_pt[m * 28 + d])) x = false, ++nan_payload;
            if (x && !bad) printf("  첫 다름: 스텝 %d 판 %d dof %d 위치 GPU %.9g 정답 %.9g\n", s, b, d, out[b].pos[d], ref_pt[m * 28 + d]);
            bad += x;
          }
          if (out[b].set_vel[d]) {
            ++cmp;
            bool x = memcmp(&out[b].vel[d], &ref_vt[m * 28 + d], 4) != 0;
            if (x && std::isnan(out[b].vel[d]) && std::isnan(ref_vt[m * 28 + d])) x = false, ++nan_payload;
            if (x && !bad) printf("  첫 다름: 스텝 %d 판 %d dof %d 속도 GPU %.9g 정답 %.9g (has_action %d)\n", s, b, d, out[b].vel[d],
                                  ref_vt[m * 28 + d], has[b]);
            bad += x;
          }
        }
      }
    }
    fclose(f);
    printf("B. 제어기 GPU=공식       판 %d x %d 스텝, 드라이브 목표 비교 %lld, 다름 %lld | 커널 %.2f 억 판·스텝/초\n", NB, steps, cmp, bad,
           (double)NB * steps / t_k / 1e8);
    total_bad += bad;
  }

  // ---------------- C. 물체 상태 ----------------
  if (parts.find('C') != std::string::npos) {
    std::ifstream idx(root + "/states/index.txt");
    std::string fr;
    long long cmp_m = 0, bad_m = 0, cmp_a = 0, bad_a = 0;
    double t_adj = 0;
    long long rays = 0;
    while (std::getline(idx, fr)) {
      if (fr.empty()) continue;
      const std::string d = root + "/states/" + fr + "/";
      auto L_ = [&](const char* n) {
        Npy a;
        npy_load(d + n + ".npy", a);
        return a;
      };
      const auto SOL = L_("S").vec<int32_t>();
      const int S = SOL[0], O = SOL[1], L = SOL[2];
      const auto poses = L_("poses").vec<float>();
      const auto mref = L_("pose_matrices").vec<float>();
      auto* d_poses = dput(poses);
      wf::M44* d_m;
      CK(cudaMalloc(&d_m, (size_t)L * sizeof(wf::M44)));
      gpu::k_pose_to_mat<<<(L + 127) / 128, 128>>>(d_poses, d_m, L);
      CK(cudaDeviceSynchronize());
      const auto m = dget(d_m, L);
      for (int l = 0; l < L; ++l)
        for (int e = 0; e < 16; ++e) {
          ++cmp_m;
          bad_m += memcmp(&m[l].m[e], &mref[l * 16 + e], 4) != 0;
        }
      // Adjacency: 공식 행렬·AABB 입력
      std::vector<wf::M44> mo(L);
      for (int l = 0; l < L; ++l) memcpy(mo[l].m, &mref[l * 16], 64);
      auto* d_mo = dput(mo);
      auto* d_aabb = dput(L_("aabb").vec<float>());
      auto* d_aidx = dput(L_("in_aidx").vec<int32_t>());
      auto* d_hm = dput(L_("has_mesh").vec<uint8_t>());
      auto* d_l2o = dput(L_("adj_l2o").vec<int32_t>());
      auto* d_l2s = dput(L_("adj_l2s").vec<int32_t>());
      const auto dirs = L_("adj_dirs").vec<float>();
      const auto maxd = L_("adj_maxd").vec<float>();
      const int K = (int)maxd.size(), N = O;
      auto* d_dirs = dput(dirs);
      auto* d_maxd = dput(maxd);
      auto* d_pts = dput(L_("mesh_pts").vec<float>());
      auto* d_tri = dput(L_("mesh_tri").vec<int32_t>());
      auto* d_poff = dput(L_("pts_off").vec<int32_t>());
      auto* d_toff = dput(L_("tri_off").vec<int32_t>());
      uint8_t* d_adj;
      const size_t na = (size_t)S * N * N * K;
      CK(cudaMalloc(&d_adj, na));
      CK(cudaMemset(d_adj, 0, na));
      const long long total = (long long)S * N * L * K;
      CK(cudaDeviceSynchronize());
      const double t0 = now_s();
      gpu::k_adjacency<<<(unsigned)((total + 127) / 128), 128>>>(d_aabb, O, d_aidx, N, d_mo, d_hm, d_l2o, d_l2s, L, d_dirs,
                                                                   d_maxd, K, d_pts, d_tri, d_poff, d_toff, S, d_adj);
      CK(cudaDeviceSynchronize());
      t_adj += now_s() - t0;
      rays += total;
      const auto adj = dget(d_adj, na);
      const auto aref = L_("adjacency").vec<uint8_t>();
      for (size_t e = 0; e < na; ++e) {
        ++cmp_a;
        bad_a += adj[e] != aref[e];
      }
      cudaFree(d_poses), cudaFree(d_m), cudaFree(d_mo), cudaFree(d_aabb), cudaFree(d_aidx), cudaFree(d_hm), cudaFree(d_l2o);
      cudaFree(d_l2s), cudaFree(d_dirs), cudaFree(d_maxd), cudaFree(d_pts), cudaFree(d_tri), cudaFree(d_poff), cudaFree(d_toff);
      cudaFree(d_adj);
    }
    printf("C. 자세→행렬 GPU=warp    비교 %lld, 다름 %lld\n", cmp_m, bad_m);
    printf("C. Adjacency GPU=warp    비교 %lld, 다름 %lld | 광선 스레드 %.1f 백만/초 (삼각형 전수)\n", cmp_a, bad_a, rays / t_adj / 1e6);
    total_bad += bad_m + bad_a;
  }
  return total_bad ? 1 : 0;
}
