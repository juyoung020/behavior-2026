// 층 2 시험: 시각 입자·제거기 한 판을 GPU 스레드 하나로 (판 N 개). 공식 정답(= 층 1)과 비트 비교 + 처리량.
//   ./test_visual_gpu ~/engine-data/particles/visual [사례 수] [복제 수]
// 판 e 는 사례 e % 사례수. 첫 벌(e < 사례수)은 스텝·제거기마다 생존 표를 적어 공식과 비교, 나머지는 끝 상태가 첫 벌과 같은지 본다.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/visual.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

#define CK(x)                                                                           \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));  \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

struct Case {  // 판 하나의 입력 (모든 배열은 한 덩어리 버퍼의 오프셋)
  int N, S, NL, R, H;
  long p_link, lpos, lquat, link_tf, rem_meth, rem_limit, rem_need, rem_hull, rem_attr, rem_tf, toggle;  // 입력
  long out_alive, out_world;  // 첫 벌 기록 위치 (바이트/float 단위 각각)
};

template <class T>
struct Buf {
  std::vector<T> h;
  long put(const T* p, size_t n) {
    long o = (long)h.size();
    h.insert(h.end(), p, p + n);
    return o;
  }
};

__global__ void run_envs(int n_env, int ncase, const Case* cs, const float* F, const int32_t* I, const uint8_t* U, const double* D,
                         float* lm_all, uint8_t* alive_all, int32_t* mod_all, long env_stride_lm, long env_stride_n, int max_r,
                         uint8_t* rec_alive, float* rec_world) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_env) return;
  const Case c = cs[e % ncase];
  const bool rec = e < ncase;
  float* lm = lm_all + e * env_stride_lm;
  uint8_t* alive = alive_all + e * env_stride_n;
  int32_t* mod = mod_all + e * max_r;
  const int32_t* pl = I + c.p_link;
  for (int i = 0; i < c.N; ++i) {
    int gs = 0;
    for (int j = 0; j < c.N; ++j) gs += pl[j] == pl[i];
    local_mat(F + c.lpos + 3 * i, F + c.lquat + 4 * i, gs == 1, lm + 16 * i);
    alive[i] = 1;
  }
  for (int r = 0; r < c.R; ++r) mod[r] = 0;
  VisualSet vs{c.N, pl, pl, lm, alive};
  for (int s = 0; s < c.S; ++s) {
    const float* ltf = F + c.link_tf + (long)s * c.NL * 16;
    if (rec)
      for (int i = 0; i < c.N; ++i) {
        float p[3] = {NAN, NAN, NAN};
        if (alive[i]) particle_world_pos(ltf + 16 * pl[i], lm + 16 * i, p);
        for (int k = 0; k < 3; ++k) rec_world[c.out_world + ((long)s * c.N + i) * 3 + k] = p[k];
      }
    for (int r = 0; r < c.R; ++r) {
      RemoverGeom g{I[c.rem_meth + r], c.H, F + c.rem_hull + (long)r * c.H * 3, {D[c.rem_attr + 3 * r], D[c.rem_attr + 3 * r + 1], D[c.rem_attr + 3 * r + 2]}};
      Cond conds[2];
      int nc = 0;
      if (U[c.rem_need + r]) conds[nc++] = Cond{COND_TOGGLED, 1};
      conds[nc++] = Cond{COND_LIMIT, 0};
      int n_alive = 0;
      for (int i = 0; i < c.N; ++i) n_alive += alive[i];
      const int32_t lim = I[c.rem_limit + r];
      const float* tf = F + c.rem_tf + ((long)s * c.R + r) * 16;
      if (remover_may_modify(conds, nc, U[c.toggle + s * c.R + r] != 0, n_alive, lim, mod[r], [](int) { return false; }))
        remove_visual(vs, ltf, g, tf, lim, mod[r]);
      if (rec)
        for (int i = 0; i < c.N; ++i) rec_alive[c.out_alive + ((long)s * c.R + r) * c.N + i] = alive[i];
    }
  }
}

int main(int argc, char** argv) {
  const std::string root = argv[1];
  const int ncase = argc > 2 ? atoi(argv[2]) : 40;
  const int reps = argc > 3 ? atoi(argv[3]) : 256;
  Buf<float> F;
  Buf<int32_t> I;
  Buf<uint8_t> U;
  Buf<double> D;
  std::vector<Case> cs(ncase);
  std::vector<Npy> ref_alive(ncase), ref_world(ncase), ref_count(ncase);
  long rec_a = 0, rec_w = 0;
  int maxN = 0, maxR = 0;
  for (int c = 0; c < ncase; ++c) {
    char d[512];
    snprintf(d, sizeof d, "%s/case_%03d/", root.c_str(), c);
    auto L = [&](const char* k) {
      Npy a;
      if (!npy_load(std::string(d) + k + ".npy", a)) exit(1);
      return a;
    };
    Npy p_link = L("p_link"), lpos = L("lpos"), lquat = L("lquat"), link_tf = L("link_tf"), rem_meth = L("rem_meth"),
        rem_limit = L("rem_limit"), rem_need = L("rem_need_toggle"), rem_hull = L("rem_hull"), rem_attr = L("rem_attr"),
        rem_tf = L("rem_tf"), toggle = L("toggle");
    ref_alive[c] = L("alive_after");
    ref_world[c] = L("world");
    ref_count[c] = L("count_after");
    Case& k = cs[c];
    k.N = (int)p_link.shape[0];
    k.S = (int)link_tf.shape[0];
    k.NL = (int)link_tf.shape[1];
    k.R = (int)rem_tf.shape[1];
    k.H = (int)rem_hull.shape[1];
    k.p_link = I.put(p_link.as<int32_t>(), p_link.count());
    k.lpos = F.put(lpos.as<float>(), lpos.count());
    k.lquat = F.put(lquat.as<float>(), lquat.count());
    k.link_tf = F.put(link_tf.as<float>(), link_tf.count());
    k.rem_meth = I.put(rem_meth.as<int32_t>(), rem_meth.count());
    k.rem_limit = I.put(rem_limit.as<int32_t>(), rem_limit.count());
    k.rem_need = U.put(rem_need.as<uint8_t>(), rem_need.count());
    k.rem_hull = F.put(rem_hull.as<float>(), rem_hull.count());
    k.rem_attr = D.put(rem_attr.as<double>(), rem_attr.count());
    k.rem_tf = F.put(rem_tf.as<float>(), rem_tf.count());
    k.toggle = U.put(toggle.as<uint8_t>(), toggle.count());
    k.out_alive = rec_a;
    rec_a += (long)k.S * k.R * k.N;
    k.out_world = rec_w;
    rec_w += (long)k.S * k.N * 3;
    maxN = std::max(maxN, k.N);
    maxR = std::max(maxR, k.R);
  }
  const int n_env = ncase * reps;
  auto up = [](auto& b) {
    using T = typename std::remove_reference<decltype(b.h[0])>::type;
    T* p;
    CK(cudaMalloc(&p, std::max<size_t>(1, b.h.size()) * sizeof(T)));
    CK(cudaMemcpy(p, b.h.data(), b.h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  };
  float* dF = up(F);
  int32_t* dI = up(I);
  uint8_t* dU = up(U);
  double* dD = up(D);
  Case* dC;
  CK(cudaMalloc(&dC, ncase * sizeof(Case)));
  CK(cudaMemcpy(dC, cs.data(), ncase * sizeof(Case), cudaMemcpyHostToDevice));
  float *lm, *rw;
  uint8_t *al, *ra;
  int32_t* md;
  CK(cudaMalloc(&lm, (size_t)n_env * maxN * 16 * sizeof(float)));
  CK(cudaMalloc(&al, (size_t)n_env * maxN));
  CK(cudaMalloc(&md, (size_t)n_env * maxR * sizeof(int32_t)));
  CK(cudaMalloc(&ra, rec_a));
  CK(cudaMalloc(&rw, rec_w * sizeof(float)));
  const int tb = 64;
  run_envs<<<(n_env + tb - 1) / tb, tb>>>(n_env, ncase, dC, dF, dI, dU, dD, lm, al, md, (long)maxN * 16, maxN, maxR, ra, rw);
  CK(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  const int iters = 5;
  for (int it = 0; it < iters; ++it)
    run_envs<<<(n_env + tb - 1) / tb, tb>>>(n_env, ncase, dC, dF, dI, dU, dD, lm, al, md, (long)maxN * 16, maxN, maxR, ra, rw);
  CK(cudaDeviceSynchronize());
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::vector<uint8_t> h_ra(rec_a), h_al((size_t)n_env * maxN);
  std::vector<float> h_rw(rec_w);
  std::vector<int32_t> h_md((size_t)n_env * maxR);
  CK(cudaMemcpy(h_ra.data(), ra, rec_a, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_rw.data(), rw, rec_w * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_al.data(), al, h_al.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_md.data(), md, h_md.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  long n_a = 0, bad_a = 0, n_w = 0, bad_w = 0, n_r = 0, bad_r = 0, n_c = 0, bad_c = 0;
  long steps = 0;
  for (int c = 0; c < ncase; ++c) {
    const Case& k = cs[c];
    steps += k.S;
    for (long i = 0; i < (long)k.S * k.R * k.N; ++i, ++n_a) bad_a += h_ra[k.out_alive + i] != ref_alive[c].as<uint8_t>()[i];
    for (long i = 0; i < (long)k.S * k.N * 3; ++i, ++n_w) {
      uint32_t a, b;
      memcpy(&a, &h_rw[k.out_world + i], 4);
      memcpy(&b, ref_world[c].as<float>() + i, 4);
      bad_w += a != b;
    }
    for (int r = 0; r < k.R; ++r, ++n_c) bad_c += h_md[(size_t)c * maxR + r] != ref_count[c].as<int32_t>()[((long)k.S - 1) * k.R + r];
    for (int rep = 1; rep < reps; ++rep) {
      const size_t e = (size_t)rep * ncase + c;
      bool ok = memcmp(&h_al[e * maxN], &h_al[(size_t)c * maxN], k.N) == 0 &&
                memcmp(&h_md[e * maxR], &h_md[(size_t)c * maxR], k.R * sizeof(int32_t)) == 0;
      ++n_r;
      bad_r += !ok;
    }
  }
  printf("시각 입자·제거기 층 2 (GPU) vs 공식, 판 %d (사례 %d x %d 벌)\n", n_env, ncase, reps);
  printf("  %-22s 비교 %9ld  다름 %ld\n", "스텝별 생존", n_a, bad_a);
  printf("  %-22s 비교 %9ld  다름 %ld\n", "세계 위치", n_w, bad_w);
  printf("  %-22s 비교 %9ld  다름 %ld\n", "끝 누적 제거 수", n_c, bad_c);
  printf("  %-22s 비교 %9ld  다름 %ld\n", "복제 판 끝 상태", n_r, bad_r);
  printf("  처리량: 커널 %.3f ms / 판 %d (판마다 평균 %.1f 스텝) = %.2f 백만 판-스텝/초\n", ms, n_env, (double)steps / ncase,
         (double)n_env * steps / ncase / (ms * 1e3));
  const bool ok = !(bad_a || bad_w || bad_c || bad_r);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
