// CPU check of the PiBehavior host side (pb_host.cpp) against the 1st-place Python code
// (cases from tools/make_pb_host_cases.py).  pb_host_test <pb weights .pi05w> <pb_host_cases.pi05d>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/pb_host.h"
#include "../src/weights.h"

using namespace pi05;

template <class T>
std::vector<T> rd(const WeightFile& f, const std::string& n) {
  const TensorInfo* t = f.find(n);
  std::vector<T> v(t ? t->nbytes / sizeof(T) : 0);
  std::string e;
  if (t) f.read(*t, v.data(), &e);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::string err;
  WeightFile wf, cf;
  if (!wf.open(argv[1], &err) || !cf.open(argv[2], &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  PbSpec s;
  if (!load_pb_spec(wf, &s, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  int bad = 0;
  auto report = [&](const char* what, double maxdiff, double tol, long mism) {
    const bool ok = maxdiff <= tol && mism == 0;
    if (!ok) ++bad;
    printf("%-34s max|diff| %.3e  mismatches %ld  %s\n", what, maxdiff, mism, ok ? "ok" : "FAIL");
  };
  // state extraction
  for (int legacy = 0; legacy < 2; ++legacy) {
    const int P = legacy ? 256 : 61;
    auto prop = rd<float>(cf, legacy ? "prop256" : "prop61");
    auto want = rd<float>(cf, legacy ? "state256" : "state61");
    const int n = (int)want.size() / 23;
    double md = 0;
    long mm = 0;
    for (int i = 0; i < n; ++i) {
      float st[23];
      pb_extract_state(&prop[(size_t)i * P], P, st);
      for (int d = 0; d < 23; ++d) {
        md = std::max(md, (double)std::fabs(st[d] - want[i * 23 + d]));
        mm += st[d] != want[i * 23 + d];
      }
    }
    report(legacy ? "state (256-d legacy proprio)" : "state (61-d proprio)", md, legacy ? 1e-6 : 0, legacy ? 0 : mm);
  }
  auto st61 = rd<float>(cf, "state61");
  const int n = (int)st61.size() / 23;
  // normalization + tokens
  {
    auto wn = rd<float>(cf, "state_norm");
    auto wt = rd<int32_t>(cf, "state_tokens");
    double md = 0;
    long mm = 0;
    for (int i = 0; i < n; ++i) {
      float nm[32];
      int tk[32];
      pb_state_tokens(s, &st61[i * 23], nm, tk);
      for (int d = 0; d < 32; ++d) {
        md = std::max(md, (double)std::fabs(nm[d] - wn[i * 32 + d]));
        mm += tk[d] != wt[i * 32 + d];
      }
    }
    report("normalized state / tokens", md, 0, mm);
  }
  // initial actions and output transform
  {
    auto kept = rd<double>(cf, "kept");
    auto wi = rd<float>(cf, "init_actions");
    auto raw = rd<float>(cf, "raw");
    auto wo = rd<double>(cf, "out_actions");
    auto wn = rd<float>(cf, "state_norm");
    double mi = 0, mo = 0;
    for (int i = 0; i < n; ++i) {
      float x0[4 * 32];
      pb_initial_actions(s, &kept[(size_t)i * 4 * 23], 4, &st61[i * 23], x0);
      for (int k = 0; k < 4 * 32; ++k) mi = std::max(mi, (double)std::fabs(x0[k] - wi[(size_t)i * 128 + k]));
      std::vector<double> o(30 * 23);
      pb_postprocess(s, &raw[(size_t)i * 30 * 32], &wn[i * 32], o.data());
      for (int k = 0; k < 30 * 23; ++k) mo = std::max(mo, std::fabs(o[k] - wo[(size_t)i * 30 * 23 + k]));
    }
    report("initial actions (model space)", mi, 0, 0);
    report("output actions (robot units)", mo, 1e-12, 0);
  }
  // cubic compression
  {
    auto in = rd<double>(cf, "cubic_in");
    auto want = rd<double>(cf, "cubic_out");
    const int m = (int)in.size() / (26 * 23);
    double md = 0;
    for (int i = 0; i < m; ++i) {
      std::vector<double> a(in.begin() + (size_t)i * 26 * 23, in.begin() + (size_t)(i + 1) * 26 * 23);
      auto c = pb_cubic_resample(a, 26, 20);
      for (int k = 0; k < 20 * 23; ++k) md = std::max(md, std::fabs(c[k] - want[(size_t)i * 20 * 23 + k]));
    }
    report("cubic 26 -> 20 (scipy interp1d)", md, 1e-9, 0);
  }
  // correction rules + gripper variation
  {
    auto task = rd<int32_t>(cf, "cr_task"), stage = rd<int32_t>(cf, "cr_stage"), ost = rd<int32_t>(cf, "cr_ostage"),
         gv = rd<int32_t>(cf, "gv");
    auto st = rd<float>(cf, "cr_state");
    auto act = rd<double>(cf, "cr_act"), out = rd<double>(cf, "cr_out");
    long mm = 0, mg = 0;
    double md = 0;
    for (size_t i = 0; i < task.size(); ++i) {
      std::vector<double> a(act.begin() + i * 30 * 23, act.begin() + (i + 1) * 30 * 23);
      const int ns = pb_correction_rules(task[i], stage[i], &st[i * 23], a, 30);
      mm += ns != ost[i];
      for (int k = 0; k < 30 * 23; ++k) md = std::max(md, std::fabs(a[k] - out[i * 30 * 23 + k]));
      mg += (int)pb_gripper_variation(a, 26) != gv[i];
    }
    report("correction rules (actions, stage)", md, 1e-7, mm);
    report("gripper variation check", 0, 0, mg);
  }
  // stage voting
  {
    auto in = rd<int32_t>(cf, "vote_in"), want = rd<int32_t>(cf, "vote_out");
    long mm = 0;
    PbWrapperCfg c;
    for (int q = 0; q < 200; ++q) {
      PbSlot sl;
      for (int k = 0; k < 40; ++k) {
        const int task = in[(q * 40 + k) * 2], p = in[(q * 40 + k) * 2 + 1];
        pb_vote(sl, s.stages[task] - 1, p, c);
        mm += sl.stage != want[q * 40 + k];
      }
    }
    report("stage voting (8000 steps)", 0, 0, mm);
  }
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
