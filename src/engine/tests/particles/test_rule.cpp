// 층 1 시험: 전이 규칙 논리 — SlicerActive(warp 커널 CPU 판과 value·delay 비트), SlicingRule 선택(공식 step), BDDL 범위 칸.
//   정답: python gen_rule_ref.py --out ~/engine-data/particles/rule
//   ./test_rule ~/engine-data/particles/rule
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/slicing.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(d + k + ".npy", a)) {
      fprintf(stderr, "없음 %s\n", k);
      exit(1);
    }
    return a;
  };
  // A. SlicerActive
  Npy touch = L("sa_touch"), dt = L("sa_dt"), val = L("sa_value"), dly = L("sa_delay");
  const int S = (int)touch.shape[0], O = (int)touch.shape[1];
  long bad_a = 0, reon = 0;
  for (int o = 0; o < O; ++o) {
    SlicerState st{1, 0, 0.0f};
    for (int s = 0; s < S; ++s) {
      const uint8_t v0 = st.value;
      slicer_active_update(st, touch.as<int32_t>()[s * O + o] != 0, dt.as<float>()[o]);
      reon += v0 == 0 && st.value == 1;
      uint32_t a, b;
      const float dref = dly.as<float>()[s * O + o];
      memcpy(&a, &st.delay, 4);
      memcpy(&b, &dref, 4);
      if (st.value != val.as<uint8_t>()[s * O + o] || a != b) {
        if (!bad_a) printf("  SlicerActive 첫 다름: 자르개 %d 스텝 %d\n", o, s);
        ++bad_a;
      }
    }
  }
  // B. SlicingRule 선택
  Npy rr = L("rule_rows");
  const long nb = (long)rr.shape[0];
  const int w = (int)rr.shape[1];
  long bad_b = 0, fired = 0;
  for (long r = 0; r < nb; ++r) {
    const int32_t* x = rr.as<int32_t>() + r * w;
    const int ns = x[0], nk = x[1];
    uint8_t act[4];
    for (int j = 0; j < 4; ++j) act[j] = (uint8_t)x[2 + j];
    const int32_t* tm = x + 6;
    int out[6];
    const int m = slicing_rule_select(ns, nk, act, [&](int i, int j) { return tm[i * 4 + j] != 0; }, out);
    uint8_t sel[6] = {0};
    for (int k = 0; k < m; ++k) sel[out[k]] = 1;
    bool ok = true;
    for (int i = 0; i < 6; ++i) ok &= sel[i] == x[30 + i];
    bad_b += !ok;
    fired += m > 0;
  }
  // C. 범위 칸
  Npy sl = L("scope_slots"), si = L("scope_init"), se = L("scope_evs"), sf = L("scope_final");
  const long nc = (long)sl.shape[0];
  const int MS = (int)sl.shape[1], ME = (int)se.shape[1];
  long bad_c = 0;
  for (long t = 0; t < nc; ++t) {
    const int32_t* slots = sl.as<int32_t>() + t * MS * 2;
    int n = 0;
    while (n < MS && slots[2 * n] >= 0) ++n;
    std::vector<uint8_t> is_sys(n);
    std::vector<int32_t> v(n);
    for (int j = 0; j < n; ++j) {
      is_sys[j] = (uint8_t)slots[2 * j + 1];
      v[j] = si.as<int32_t>()[t * MS + j];
    }
    for (int e = 0; e < ME; ++e) {
      const int32_t* ev = se.as<int32_t>() + (t * ME + e) * 2;
      if (ev[0] < 0) break;
      if (ev[0] == 0)
        scope_on_add(n, is_sys.data(), v.data(), [&](int j) { return slots[2 * j] == ev[1]; }, e);
      else if (ev[0] == 1)
        scope_on_remove(n, is_sys.data(), v.data(), ev[1]);
      else
        scope_on_system_init(n, is_sys.data(), v.data(), [&](int j) { return slots[2 * j] == ev[1]; }, 2000 + ev[1]);
    }
    bool ok = true;
    for (int j = 0; j < n; ++j) ok &= v[j] == sf.as<int32_t>()[t * MS + j];
    bad_c += !ok;
  }
  // D. 입자 익히기 선택 (gen_cook_ref.py, 없으면 건너뜀)
  long nd = 0, bad_d = 0;
  {
    Npy cr;
    if (npy_load(d + "../cook/cook_rows.npy", cr)) {
      const int MC = 5, MR = 6, MS = 8, wc = (int)cr.shape[1];
      nd = (long)cr.shape[0];
      for (long r = 0; r < nd; ++r) {
        const int32_t* x = cr.as<int32_t>() + r * wc;
        const int nc = x[0], nr = x[1];
        const int32_t *heat = x + 2, *cat = x + 2 + MC, *cont = x + 2 + 2 * MC, *rin = cont + MC * MS, *rn2 = rin + 2 * MR, *rfc = rn2 + MR,
                      *sel = rfc + MR;
        uint8_t h[MC];
        int32_t nin[MR], out[MC];
        for (int i = 0; i < MC; ++i) h[i] = (uint8_t)heat[i];
        for (int i = 0; i < MR; ++i) nin[i] = rn2[i] ? 2 : 1;
        cook_particles_select(nc, h, nr, nin, rin, [&](int c, int q) { return rfc[q] < 0 || rfc[q] == cat[c]; },
                              [&](int c, int s) { return cont[c * MS + s] != 0; }, out);
        bool ok = true;
        for (int i = 0; i < nc; ++i) ok &= out[i] == sel[i];
        bad_d += !ok;
      }
    }
  }
  printf("  입자 익히기 선택                 사례 %ld  다름 %ld\n", nd, bad_d);
  bad_c += bad_d;
  printf("전이 규칙 논리 층 1 vs 공식\n");
  printf("  SlicerActive (value·delay 비트)  자르개 %d x 스텝 %d  다름 %ld  (다시 켜짐 %ld 번)\n", O, S, bad_a, reon);
  printf("  SlicingRule 선택                 사례 %ld (발동 %ld)  다름 %ld\n", nb, fired, bad_b);
  printf("  BDDL 범위 칸                     사례 %ld  다름 %ld\n", nc, bad_c);
  const bool ok = !(bad_a || bad_b || bad_c);
  printf(ok ? "전부 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
