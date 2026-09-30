// 층 2 시험 (P4 분무기): 원뿔 광선·무리 척도·표본 크기·붙은 입자 국소 행렬을 GPU 스레드(사례 = 스레드)로 돌려 공식 정답(= test_applier 자료)과 비트 비교.
// MKL VML cos/sin 만 호스트가 채운다: 커널 1(난수 → 반지름·각) → 호스트 cos/sin → 커널 2(광선 마무리 → 무리 척도(난수 이어서) → 국소 행렬).
// 뿌리기 표본(spray.h sample_cuboid_one: MKL sgemv·sdot·vmsAcos + 광선 신탁)과 fit_plane(MKL sgesdd)은 호스트 전용 — 5 스텝마다 몇 개라 결과만 넘긴다.
//   ENGINE_MKL_LIB=.../libtorch_cpu.so ./test_p4_gpu ~/engine-data/particles/applier
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/particles/applier.h"
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

struct In {  // 사례 하나의 입력 (층 1 시험과 같은 자료)
  float ext[3], ap_p[3], ap_q[4], ap_s[3], atom_aabb[3], tmpl[3], avg, rel;
  float hit_tf[2][16], hit_p[2][3], hit_q[2][4];
  int32_t hit_obj[2];
};
struct Out {
  float start[2][3], end[2][3], scales[6], cdims[6], lm[2][16];
};

__global__ void k_draw(int n, const uint8_t* st, int B, TorchMT* rng, const In* in, float* rr, float* th) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  torch_mt_from_bytes(st + (size_t)c * B, B, rng[c]);
  applier_cone_draw(rng[c], in[c].ext, rr + 2 * c, th + 2 * c);
}
struct NoOverlap {
  __host__ __device__ bool operator()(const float*, const float*) const { return false; }
};
__global__ void k_finish(int n, TorchMT* rng, const In* in, const float* rr, const float* cs, const float* sn, Out* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const In& x = in[c];
  Out& o = out[c];
  applier_cone_finish(x.ext, x.ap_p, x.ap_q, x.ap_s, rr + 2 * c, cs + 2 * c, sn + 2 * c, o.start, o.end);
  float mn[3], mx[3];
  group_scale_range(x.rel != 0.0f, x.atom_aabb, x.tmpl, mn, mx);
  group_scales_avg(rng[c], mn, mx, x.avg, 2, o.scales);
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 3; ++k) o.cdims[3 * i + k] = (o.scales[3 * i + k] * x.tmpl[k]) * x.avg;
  for (int t = 0; t < 2; ++t) attach_local_mat(x.hit_tf[x.hit_obj[t]], x.hit_p[t], x.hit_q[t], o.lm[t]);
  if (c == 0) {  // applier_step 도 장치에서 번역되는지 (값은 층 1 시험 몫)
    int32_t sc = 0;
    (void)applier_step(sc, false, 0, 1, x.hit_tf[0], x.ap_p, 1, NoOverlap{});
  }
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(d + k + ".npy", a)) {
      fprintf(stderr, "없음 %s%s.npy\n", d.c_str(), k);
      exit(1);
    }
    return a;
  };
  Npy st = L("state"), apS = L("ap_scale"), apP = L("ap_p"), apQ = L("ap_q"), ext = L("ext"), rel = L("rel"), aab = L("atom_aabb"), tm = L("tmpl"),
      hTF = L("hit_tf"), hP = L("hit_p"), hQ = L("hit_q"), hO = L("hit_obj"), sp = L("start"), ep = L("end"), sc = L("scales"), cd = L("cdims"), lm = L("lm");
  const int N = (int)st.shape[0], B = (int)st.shape[1];
  auto F = [](const Npy& a, int i) { return a.as<float>() + (size_t)i * a.shape[1]; };
  std::vector<In> in(N);
  for (int c = 0; c < N; ++c) {
    In& x = in[c];
    memcpy(x.ext, F(ext, c), 12), memcpy(x.ap_p, F(apP, c), 12), memcpy(x.ap_q, F(apQ, c), 16), memcpy(x.ap_s, F(apS, c), 12);
    memcpy(x.atom_aabb, F(aab, c), 12), memcpy(x.tmpl, F(tm, c), 12);
    x.rel = F(rel, c)[0];
    x.avg = avg_scale3(x.ap_s);  // 무리마다 고정 — 호스트 (double pow)
    const float* ho = F(hO, c);
    for (int t = 0; t < 2; ++t) {
      const int ob = (int)ho[t];
      memcpy(x.hit_tf[t], F(hTF, c) + 16 * ob, 64);
      memcpy(x.hit_p[t], F(hP, c) + 3 * t, 12), memcpy(x.hit_q[t], F(hQ, c) + 4 * t, 16);
      x.hit_obj[t] = t;
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  uint8_t* dst = up(st.as<uint8_t>(), (size_t)N * B);
  In* din = up(in.data(), N);
  TorchMT* drng;
  float *drr, *dth, *dcs, *dsn;
  Out* dout;
  CK(cudaMalloc(&drng, sizeof(TorchMT) * N));
  CK(cudaMalloc(&drr, 8 * N));
  CK(cudaMalloc(&dth, 8 * N));
  CK(cudaMalloc(&dout, sizeof(Out) * N));
  const int T = 128, G = (N + T - 1) / T;
  k_draw<<<G, T>>>(N, dst, B, drng, din, drr, dth);
  CK(cudaGetLastError());
  std::vector<float> th = down(dth, 2 * (size_t)N), cs(2 * (size_t)N), sn(2 * (size_t)N);
  auto& trig = eng::omni::mkl::trig();  // vmsCos/vmsSin 한꺼번에 (torch 도 텐서 통째로 부름)
  if (!trig.cos) return 1;
  trig.cos((int64_t)th.size(), th.data(), cs.data(), eng::omni::mkl::kTorchMode);
  trig.sin((int64_t)th.size(), th.data(), sn.data(), eng::omni::mkl::kTorchMode);
  dcs = up(cs.data(), cs.size()), dsn = up(sn.data(), sn.size());
  k_finish<<<G, T>>>(N, drng, din, drr, dcs, dsn, dout);
  CK(cudaGetLastError());
  std::vector<Out> out = down(dout, N);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  long bad_ray = 0, bad_sc = 0, bad_cd = 0, bad_lm = 0;
  for (int c = 0; c < N; ++c) {
    const Out& o = out[c];
    bad_ray += memcmp(o.start, F(sp, c), 24) != 0 || memcmp(o.end, F(ep, c), 24) != 0;
    bad_sc += memcmp(o.scales, F(sc, c), 24) != 0;
    bad_cd += memcmp(o.cdims, F(cd, c), 24) != 0;
    bad_lm += memcmp(o.lm, F(lm, c), 128) != 0;
  }
  printf("분무기 앞뒤 층 2 (GPU) vs 공식: 사례 %d  광선 다름 %ld  척도 다름 %ld  표본 크기 다름 %ld  국소 행렬 다름 %ld  (%.1f ms, 올리기·내리기·MKL 포함)\n", N, bad_ray,
         bad_sc, bad_cd, bad_lm, ms);
  const bool ok = !(bad_ray || bad_sc || bad_cd || bad_lm);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
