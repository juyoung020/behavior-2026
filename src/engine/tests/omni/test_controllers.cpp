// omni 시험 2: R1Pro 제어기 — 우리 C++ (core/omni/controllers.h, npf32.h) = 공식 OmniGibson 제어기 (gen_ctrl_ref.py 정답).
//   test_controllers <정답 폴더(~/engine-data/omni/ctrl)>
// 비교(비트): numba pose2mat·pose_inv, numpy 4x4 matmul, 제어기 드라이브 목표(위치·속도, 서브스텝마다), rz 감기.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/omni/controllers.h"

using namespace eng::omni;

static bool same(float a, float b) { return memcmp(&a, &b, 4) == 0; }

struct Reader {
  FILE* f;
  explicit Reader(const std::string& p) : f(fopen(p.c_str(), "rb")) {}
  ~Reader() {
    if (f) fclose(f);
  }
  template <class T>
  void get(T* x, size_t n = 1) {
    if (fread(x, sizeof(T), n, f) != n) throw std::runtime_error("short read");
  }
};

struct Tally {
  long long n = 0, bad = 0;
  std::string first;
  void cmp(float ours, float ref, const char* what, long long idx) {
    ++n;
    if (!same(ours, ref)) {
      if (!bad) {
        char buf[200];
        snprintf(buf, sizeof buf, "%s #%lld: 우리 %.9g 정답 %.9g", what, idx, ours, ref);
        first = buf;
      }
      ++bad;
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_controllers <ref dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  ctrl::R1ProConfig cfg{};
  {
    Reader r(dir + "/config.bin");
    r.get(&cfg.n_dof);
    int lay[3 + 4 + 14 + 4];
    r.get(lay, 25);
    int k = 0;
    for (int i = 0; i < 3; ++i) cfg.base_dof[i] = lay[k++];
    for (int i = 0; i < 4; ++i) cfg.trunk_dof[i] = lay[k++];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 7; ++i) cfg.arm_dof[a][i] = lay[k++];
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < 2; ++i) cfg.grip_dof[a][i] = lay[k++];
    r.get(cfg.pos_lo, cfg.n_dof);
    r.get(cfg.pos_hi, cfg.n_dof);
    r.get(cfg.vel_lo, cfg.n_dof);
    r.get(cfg.vel_hi, cfg.n_dof);
    r.get(cfg.has_limit, cfg.n_dof);
    const float in_lo[3] = {-1, -1, -1}, in_hi[3] = {1, 1, 1}, out_lo[3] = {-0.75f, -0.75f, -1}, out_hi[3] = {0.75f, 0.75f, 1};
    memcpy(cfg.base_in_lo, in_lo, 12);
    memcpy(cfg.base_in_hi, in_hi, 12);
    memcpy(cfg.base_out_lo, out_lo, 12);
    memcpy(cfg.base_out_hi, out_hi, 12);
    ctrl::init(cfg);
  }
  int rc = 0;
  // ---- 함수 단위
  {
    Reader r(dir + "/func.bin");
    int n;
    r.get(&n);
    Tally t2m, tinv, tmm;
    for (int i = 0; i < n; ++i) {
      float p1[3], q1[4], p2[3], q2[4], M1[16], M2[16], I1[16], P[16];
      r.get(p1, 3), r.get(q1, 4), r.get(p2, 3), r.get(q2, 4), r.get(M1, 16), r.get(M2, 16), r.get(I1, 16), r.get(P, 16);
      const M4 a = np_pose2mat(p1, q1), b = np_pose2mat(p2, q2);
      M4 rm1, ri1, rm2;
      memcpy(rm1.m, M1, 64);
      memcpy(ri1.m, I1, 64);
      memcpy(rm2.m, M2, 64);
      const M4 inv = np_pose_inv(rm1);
      const M4 pr = np_matmul4(ri1, rm2);
      for (int e = 0; e < 16; ++e) {
        t2m.cmp(a.m[e / 4][e % 4], M1[e], "pose2mat", i);
        t2m.cmp(b.m[e / 4][e % 4], M2[e], "pose2mat", i);
        tinv.cmp(inv.m[e / 4][e % 4], I1[e], "pose_inv", i);
        tmm.cmp(pr.m[e / 4][e % 4], P[e], "matmul4", i);
      }
    }
    printf("numba pose2mat      비교 %8lld 다름 %lld %s\n", t2m.n, t2m.bad, t2m.first.c_str());
    printf("numba pose_inv      비교 %8lld 다름 %lld %s\n", tinv.n, tinv.bad, tinv.first.c_str());
    printf("numpy 4x4 matmul    비교 %8lld 다름 %lld %s\n", tmm.n, tmm.bad, tmm.first.c_str());
    rc |= (t2m.bad || tinv.bad || tmm.bad);
  }
  // ---- 제어기
  {
    Reader r(dir + "/ctrl.bin");
    int members, steps;
    r.get(&members);
    r.get(&steps);
    std::vector<ctrl::R1ProState> st(members);
    for (auto& s : st) ctrl::reset(s);
    Tally tp, tv, tset;
    for (int s = 0; s < steps; ++s) {
      for (int m = 0; m < members; ++m) {
        uint8_t upd;
        float act[23], bp[3], bq[4], rp[3], rq[4], q[28], pt[28], vt[28];
        r.get(&upd);
        r.get(act, 23), r.get(bp, 3), r.get(bq, 4), r.get(rp, 3), r.get(rq, 4), r.get(q, 28), r.get(pt, 28), r.get(vt, 28);
        if (upd) ctrl::apply_action(cfg, st[m], act, bp, bq, rp, rq);
        ctrl::DriveTargets out;
        ctrl::step(cfg, st[m], q, out);
        const long long idx = (long long)s * members + m;
        for (int d = 0; d < cfg.n_dof; ++d) {
          if (out.set_pos[d]) tp.cmp(out.pos[d], pt[d], "위치 목표", idx * 100 + d);
          else { ++tset.n; if (!std::isnan(pt[d])) { ++tset.bad; } }
          if (out.set_vel[d]) tv.cmp(out.vel[d], vt[d], "속도 목표", idx * 100 + d);
          else { ++tset.n; if (!std::isnan(vt[d])) { ++tset.bad; } }
        }
      }
    }
    printf("제어기 위치 목표     비교 %8lld 다름 %lld %s\n", tp.n, tp.bad, tp.first.c_str());
    printf("제어기 속도 목표     비교 %8lld 다름 %lld %s\n", tv.n, tv.bad, tv.first.c_str());
    printf("쓰지 않는 자유도     비교 %8lld 다름 %lld\n", tset.n, tset.bad);
    rc |= (tp.bad || tv.bad || tset.bad);
  }
  // ---- rz 감기
  {
    Reader r(dir + "/wrap.bin");
    int n;
    r.get(&n);
    Tally tw;
    long long wflag_bad = 0;
    for (int i = 0; i < n; ++i) {
      float x, y;
      uint8_t w;
      r.get(&x), r.get(&w), r.get(&y);
      const bool ours_w = ctrl::rz_needs_wrap(x);
      if (ours_w != (w != 0)) ++wflag_bad;
      tw.cmp(ours_w ? ctrl::wrap_angle_f32(x) : x, y, "wrap", i);
    }
    printf("rz 감기             비교 %8lld 다름 %lld (감기 여부 다름 %lld) %s\n", tw.n, tw.bad, wflag_bad, tw.first.c_str());
    rc |= (tw.bad || wflag_bad);
  }
  return rc;
}
