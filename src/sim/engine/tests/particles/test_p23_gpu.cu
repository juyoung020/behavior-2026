// 층 2 시험 (P2·P3): 자르기 반쪽 자세·척도·기준 위치, SlicerActive, torch 난수(random_quaternion·th.rand), 다지기 격자, Contains/Filled 를
// GPU 스레드(사례 하나 = 스레드 하나)로 돌려 공식 정답(= 층 1)과 비트 비교.
//   ./test_p23_gpu ~/engine-data/particles
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/contain.h"
#include "core/particles/dice.h"
#include "core/particles/slicing.h"
#include "core/particles/trng.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

template <class T>
T* up(const T* h, size_t n) {
  T* d;
  CK(cudaMalloc(&d, std::max<size_t>(1, n) * sizeof(T)));
  CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// ---- 자르기 ----
__global__ void k_slice(int n, const float* in, int wi, const float* ref, int wo, float* out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  const float* x = in + (size_t)r * wi;
  const float* y = ref + (size_t)r * wo;
  float* o = out + (size_t)r * 19;
  slice_part_bbox(x, x + 3, x + 7, x + 10, x + 13, x + 17, o, o + 3, o + 7);
  half_scale(y + 7, y + 10, o + 10);
  bbox_center_to_base(y, y + 3, y + 13, y + 16, o + 13);
}
// ---- SlicerActive ----
__global__ void k_slicer(int O, int S, const int32_t* touch, const float* dt, uint8_t* val, float* dly) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  SlicerState st{1, 0, 0.0f};
  for (int s = 0; s < S; ++s) {
    slicer_active_update(st, touch[s * O + o] != 0, dt[o]);
    val[s * O + o] = st.value;
    dly[s * O + o] = st.delay;
  }
}
// ---- torch 난수 ----
__global__ void k_trng(int N, const uint8_t* st, int B, const int32_t* nq, int M, float* q, const uint8_t* rst, const int32_t* rlen, int RW, float* rv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  TorchMT m;
  torch_mt_from_bytes(st + (size_t)i * B, B, m);
  random_quaternion(m, nq[i], q + (size_t)i * M * 4);
  TorchMT m2;
  torch_mt_from_bytes(rst + (size_t)i * B, B, m2);
  for (int k = 0; k < rlen[i]; ++k) rv[(size_t)i * RW + k] = torch_rand_float(m2);
}
// ---- 다지기 격자 (사례 하나 = 스레드 하나, 메시 ≤ 3) ----
struct DiceCase {
  float lo[3], hi[3], r;
  int nm;
  DiceMesh ms[3];
  long long out_off, cap;
};
__global__ void k_dice(int n, const DiceCase* cs, float* out, long long* cnt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const DiceCase& k = cs[c];
  cnt[c] = dice_grid(k.lo, k.hi, k.r, k.ms, k.nm, out + 3 * k.out_off, k.cap);
}
// ---- Contains/Filled ----
struct ContCase {
  int n;
  long long off;
  float off3[3];
  ContainerMesh m;
  float radius;
  double volume;
};
__global__ void k_contain(int n, const ContCase* cs, const float* tfs, float* cen, uint8_t* in, int* res) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const ContCase& k = cs[c];
  for (int i = 0; i < k.n; ++i) physical_center(tfs + 7 * (k.off + i), k.off3, cen + 3 * (k.off + i));
  const int cnt = contained_count(&k.m, 1, cen + 3 * k.off, k.n, in + k.off);
  res[2 * c] = cnt;
  res[2 * c + 1] = filled_value(k.radius, cnt, k.volume);
}

// ---- 규칙 선택 (SlicingRule 행: gen_rule_ref.py rule_rows) ----
__global__ void k_rule(int n, const int32_t* rows, int w, int32_t* sel) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  const int32_t* x = rows + (size_t)r * w;
  uint8_t act[4];
  for (int j = 0; j < 4; ++j) act[j] = (uint8_t)x[2 + j];
  const int32_t* tm = x + 6;
  int out[6];
  const int m = slicing_rule_select(x[0], x[1], act, [&](int i, int j) { return tm[i * 4 + j] != 0; }, out);
  for (int i = 0; i < 6; ++i) sel[r * 6 + i] = 0;
  for (int k = 0; k < m; ++k) sel[r * 6 + out[k]] = 1;
}
// ---- 익히기 선택 ----
__global__ void k_cook(int n, const int32_t* rows, int w, int32_t* sel) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  const int MC = 5, MR = 6, MS = 8;
  const int32_t* x = rows + (size_t)r * w;
  const int32_t *heat = x + 2, *cat = x + 2 + MC, *cont = x + 2 + 2 * MC, *rin = cont + MC * MS, *rn2 = rin + 2 * MR, *rfc = rn2 + MR;
  uint8_t h[MC];
  int32_t nin[MR];
  for (int i = 0; i < MC; ++i) h[i] = (uint8_t)heat[i];
  for (int i = 0; i < MR; ++i) nin[i] = rn2[i] ? 2 : 1;
  cook_particles_select(x[0], h, x[1], nin, rin, [&](int c, int q) { return rfc[q] < 0 || rfc[q] == cat[c]; },
                        [&](int c, int s) { return cont[c * MS + s] != 0; }, sel + r * MC);
}
// ---- find_simplex 단독 (메시 하나 = 블록 하나, 점은 순서대로 한 스레드 — start 가 이어지므로) ----
__global__ void k_dl(Delaunay3 dl, const double* q, int n, int32_t* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int start = 0;
  for (int i = 0; i < n; ++i) out[i] = dl_find_simplex(dl, q + 3 * i, &start);
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string root = std::string(argv[1]) + "/";
  long tot_bad = 0;
  auto L0 = [&](const std::string& p, Npy& a) { return npy_load(root + p, a); };
  {  // 규칙·익히기 선택
    Npy rr, cr;
    if (L0("rule/rule_rows.npy", rr) && L0("cook/cook_rows.npy", cr)) {
      const int n1 = (int)rr.shape[0], w1 = (int)rr.shape[1], n2 = (int)cr.shape[0], w2 = (int)cr.shape[1];
      int32_t *d1 = up(rr.as<int32_t>(), rr.count()), *d2 = up(cr.as<int32_t>(), cr.count()), *s1, *s2;
      CK(cudaMalloc(&s1, (size_t)n1 * 6 * 4));
      CK(cudaMalloc(&s2, (size_t)n2 * 5 * 4));
      k_rule<<<(n1 + 127) / 128, 128>>>(n1, d1, w1, s1);
      k_cook<<<(n2 + 127) / 128, 128>>>(n2, d2, w2, s2);
      CK(cudaDeviceSynchronize());
      auto h1 = down(s1, (size_t)n1 * 6);
      auto h2 = down(s2, (size_t)n2 * 5);
      long b1 = 0, b2 = 0;
      for (int r = 0; r < n1; ++r)
        for (int i = 0; i < 6; ++i) b1 += h1[r * 6 + i] != rr.as<int32_t>()[(size_t)r * w1 + 30 + i];
      for (int r = 0; r < n2; ++r) {
        const int32_t* x = cr.as<int32_t>() + (size_t)r * w2;
        const int32_t* sel = x + w2 - 5;
        for (int i = 0; i < x[0]; ++i) b2 += h2[r * 5 + i] != sel[i];
      }
      printf("  SlicingRule 선택·입자 익히기 선택      사례 %d·%d  다름 %ld·%ld\n", n1, n2, b1, b2);
      tot_bad += b1 + b2;
    }
  }
  {  // find_simplex
    long n = 0, bad = 0;
    for (int m = 0; m < 200; ++m) {
      char d[64];
      snprintf(d, sizeof d, "delaunay/mesh_%03d/", m);
      Npy T, N, E, MI, Q, R;
      if (!L0(std::string(d) + "transform.npy", T) || !L0(std::string(d) + "neighbors.npy", N) || !L0(std::string(d) + "equations.npy", E) ||
          !L0(std::string(d) + "misc.npy", MI) || !L0(std::string(d) + "q.npy", Q) || !L0(std::string(d) + "res.npy", R))
        break;
      const double* mi = MI.as<double>();
      Delaunay3 dl{(int32_t)T.shape[0], up(T.as<double>(), T.count()), up(N.as<int32_t>(), N.count()), up(E.as<double>(), E.count()), mi[0], mi[1],
                   {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
      const int nq = (int)Q.shape[0];
      double* dq = up(Q.as<double>(), Q.count());
      int32_t* dout;
      CK(cudaMalloc(&dout, (size_t)nq * 4));
      k_dl<<<1, 1>>>(dl, dq, nq, dout);
      CK(cudaDeviceSynchronize());
      auto h = down(dout, nq);
      for (int i = 0; i < nq; ++i) bad += h[i] != R.as<int32_t>()[i];
      n += nq;
      cudaFree(dq), cudaFree(dout), cudaFree((void*)dl.transform), cudaFree((void*)dl.neighbors), cudaFree((void*)dl.equations);
    }
    printf("  find_simplex (단체 번호)              점 %ld  다름 %ld\n", n, bad);
    tot_bad += bad;
  }
  auto L = [&](const std::string& p) {
    Npy a;
    if (!npy_load(root + p, a)) {
      fprintf(stderr, "없음 %s\n", p.c_str());
      exit(1);
    }
    return a;
  };
  {  // 자르기
    Npy in = L("slice/slice_in.npy"), ref = L("slice/slice_out.npy");
    const int n = (int)in.shape[0], wi = (int)in.shape[1], wo = (int)ref.shape[1];
    float *di = up(in.as<float>(), in.count()), *dr = up(ref.as<float>(), ref.count()), *dout;
    CK(cudaMalloc(&dout, (size_t)n * 19 * 4));
    k_slice<<<(n + 127) / 128, 128>>>(n, di, wi, dr, wo, dout);
    CK(cudaDeviceSynchronize());
    auto o = down(dout, (size_t)n * 19);
    long bad = 0;
    for (int r = 0; r < n; ++r) {
      const float* y = ref.as<float>() + (size_t)r * wo;
      const float* g = &o[(size_t)r * 19];
      bad += memcmp(g, y, 40) != 0 || memcmp(g + 10, y + 13, 12) != 0 || memcmp(g + 13, y + 19, 12) != 0;
    }
    printf("  자르기 반쪽 (bb 자세·크기·척도·기준)   사례 %d  다름 %ld\n", n, bad);
    tot_bad += bad;
  }
  {  // SlicerActive
    Npy t = L("rule/sa_touch.npy"), dt = L("rule/sa_dt.npy"), v = L("rule/sa_value.npy"), dl = L("rule/sa_delay.npy");
    const int S = (int)t.shape[0], O = (int)t.shape[1];
    int32_t* dtch = up(t.as<int32_t>(), t.count());
    float* ddt = up(dt.as<float>(), dt.count());
    uint8_t* dv;
    float* dd;
    CK(cudaMalloc(&dv, (size_t)S * O));
    CK(cudaMalloc(&dd, (size_t)S * O * 4));
    k_slicer<<<(O + 127) / 128, 128>>>(O, S, dtch, ddt, dv, dd);
    CK(cudaDeviceSynchronize());
    auto hv = down(dv, (size_t)S * O);
    auto hd = down(dd, (size_t)S * O);
    long bad = (memcmp(hv.data(), v.as<uint8_t>(), hv.size()) != 0) + (memcmp(hd.data(), dl.as<float>(), hd.size() * 4) != 0);
    printf("  SlicerActive value·delay              자르개 %d x 스텝 %d  다름 %ld\n", O, S, bad);
    tot_bad += bad;
  }
  {  // torch 난수
    Npy st = L("trng/state.npy"), nn = L("trng/n.npy"), qu = L("trng/quat.npy"), rs = L("trng/rand_state.npy"), rv = L("trng/rand_vals.npy"),
        rl = L("trng/rand_len.npy");
    const int N = (int)st.shape[0], B = (int)st.shape[1], M = (int)qu.shape[1], RW = (int)rv.shape[1];
    uint8_t *dst = up(st.as<uint8_t>(), st.count()), *drs = up(rs.as<uint8_t>(), rs.count());
    int32_t *dn = up(nn.as<int32_t>(), nn.count()), *drl = up(rl.as<int32_t>(), rl.count());
    float *dq, *dr;
    CK(cudaMalloc(&dq, (size_t)N * M * 16));
    CK(cudaMemset(dq, 0, (size_t)N * M * 16));
    CK(cudaMalloc(&dr, (size_t)N * RW * 4));
    CK(cudaMemset(dr, 0, (size_t)N * RW * 4));
    k_trng<<<(N + 63) / 64, 64>>>(N, dst, B, dn, M, dq, drs, drl, RW, dr);
    CK(cudaDeviceSynchronize());
    auto hq = down(dq, (size_t)N * M * 4);
    auto hr = down(dr, (size_t)N * RW);
    long bq = memcmp(hq.data(), qu.as<float>(), hq.size() * 4) != 0, br = memcmp(hr.data(), rv.as<float>(), hr.size() * 4) != 0;
    printf("  torch 난수 random_quaternion·th.rand  사례 %d  다름 %ld·%ld\n", N, bq, br);
    tot_bad += bq + br;
  }
  {  // 다지기 격자
    const int NC = 300;
    std::vector<DiceCase> cs(NC);
    std::vector<Npy> keep;
    std::vector<std::vector<float>> want(NC);
    long long off = 0;
    std::vector<double*> dptrs;
    for (int c = 0; c < NC; ++c) {
      char d[64];
      snprintf(d, sizeof d, "dice/case_%03d/", c);
      Npy lo = L(std::string(d) + "lo.npy"), hi = L(std::string(d) + "hi.npy"), r = L(std::string(d) + "r.npy"), mn = L(std::string(d) + "mesh_n.npy"),
          mt = L(std::string(d) + "mesh_tf.npy"), pos = L(std::string(d) + "pos.npy");
      DiceCase& k = cs[c];
      memcpy(k.lo, lo.as<float>(), 12);
      memcpy(k.hi, hi.as<float>(), 12);
      k.r = r.as<float>()[0];
      k.nm = (int)mn.shape[0];
      for (int m = 0; m < k.nm; ++m) {
        const std::string b = std::string(d) + "dl" + std::to_string(m) + "_";
        Npy T = L(b + "transform.npy"), N = L(b + "neighbors.npy"), E = L(b + "equations.npy"), MI = L(b + "misc.npy");
        const double* mi = MI.as<double>();
        memcpy(k.ms[m].tf, mt.as<float>() + 16 * m, 64);
        k.ms[m].dl = Delaunay3{(int32_t)T.shape[0], up(T.as<double>(), T.count()), up(N.as<int32_t>(), N.count()), up(E.as<double>(), E.count()), mi[0], mi[1],
                               {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
      }
      want[c].assign(pos.as<float>(), pos.as<float>() + pos.count());
      k.out_off = off;
      k.cap = (long long)pos.shape[0];
      off += k.cap;
    }
    DiceCase* dc = up(cs.data(), cs.size());
    float* dout;
    long long* dcnt;
    CK(cudaMalloc(&dout, (size_t)off * 12));
    CK(cudaMalloc(&dcnt, NC * 8));
    k_dice<<<(NC + 31) / 32, 32>>>(NC, dc, dout, dcnt);
    CK(cudaDeviceSynchronize());
    auto ho = down(dout, (size_t)off * 3);
    auto hc = down(dcnt, NC);
    long bad = 0;
    for (int c = 0; c < NC; ++c) bad += hc[c] != cs[c].cap || memcmp(&ho[3 * cs[c].out_off], want[c].data(), want[c].size() * 4) != 0;
    printf("  다지기 격자                           사례 %d, 입자 %lld  다름 %ld\n", NC, off, bad);
    tot_bad += bad;
  }
  {  // Contains/Filled
    const int NC = 500;
    std::vector<ContCase> cs(NC);
    std::vector<float> tfs, cen_want;
    std::vector<uint8_t> in_want;
    std::vector<int> res_want;
    static const int map[5] = {5, CYLINDER, CUBE, SPHERE, CONE};
    for (int c = 0; c < NC; ++c) {
      char d[64];
      snprintf(d, sizeof d, "contain/case_%03d/", c);
      Npy t = L(std::string(d) + "tfs.npy"), of = L(std::string(d) + "off.npy"), mt = L(std::string(d) + "mesh_tf.npy"), ce = L(std::string(d) + "centers.npy"),
          iv = L(std::string(d) + "in_volume.npy"), sc = L(std::string(d) + "scal.npy"), kd = L(std::string(d) + "kind.npy");
      ContCase& k = cs[c];
      k.n = (int)t.shape[0];
      k.off = (long long)tfs.size() / 7;
      memcpy(k.off3, of.as<float>(), 12);
      k.m = ContainerMesh{};
      k.m.kind = map[kd.as<int32_t>()[0]];
      memcpy(k.m.tf, mt.as<float>(), 64);
      k.m.attr[0] = 0.5, k.m.attr[1] = 1.0, k.m.attr[2] = 1.0;
      if (k.m.kind == 5) {
        Npy T = L(std::string(d) + "dl_transform.npy"), N = L(std::string(d) + "dl_neighbors.npy"), E = L(std::string(d) + "dl_equations.npy"),
            MI = L(std::string(d) + "dl_misc.npy");
        const double* mi = MI.as<double>();
        k.m.dl = Delaunay3{(int32_t)T.shape[0], up(T.as<double>(), T.count()), up(N.as<int32_t>(), N.count()), up(E.as<double>(), E.count()), mi[0], mi[1],
                           {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
      }
      k.radius = (float)sc.as<double>()[0];
      k.volume = sc.as<double>()[1];
      tfs.insert(tfs.end(), t.as<float>(), t.as<float>() + t.count());
      cen_want.insert(cen_want.end(), ce.as<float>(), ce.as<float>() + ce.count());
      in_want.insert(in_want.end(), iv.as<uint8_t>(), iv.as<uint8_t>() + iv.count());
      res_want.push_back(kd.as<int32_t>()[1]);
      res_want.push_back(kd.as<int32_t>()[2]);
    }
    const size_t np = tfs.size() / 7;
    ContCase* dc = up(cs.data(), cs.size());
    float* dt = up(tfs.data(), tfs.size());
    float* dcen;
    uint8_t* din;
    int* dres;
    CK(cudaMalloc(&dcen, np * 12));
    CK(cudaMalloc(&din, np));
    CK(cudaMalloc(&dres, NC * 8));
    k_contain<<<(NC + 63) / 64, 64>>>(NC, dc, dt, dcen, din, dres);
    CK(cudaDeviceSynchronize());
    auto hc = down(dcen, np * 3);
    auto hi = down(din, np);
    auto hr = down(dres, (size_t)NC * 2);
    const long b1 = memcmp(hc.data(), cen_want.data(), hc.size() * 4) != 0, b2 = memcmp(hi.data(), in_want.data(), np) != 0,
               b3 = memcmp(hr.data(), res_want.data(), hr.size() * 4) != 0;
    printf("  물리 입자 중심·안 판정·개수/Filled    입자 %zu  다름 %ld·%ld·%ld  (주의: 판정은 GPU 중심으로)\n", np, b1, b2, b3);
    tot_bad += b1 + b2 + b3;
  }
  printf(tot_bad ? "실패\n" : "P2·P3 층 2 전부 비트 동일\n");
  return tot_bad ? 1 : 0;
}
