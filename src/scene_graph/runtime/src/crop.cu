// best view RGB 자르기(src/crop.hpp).
#include <cuda_runtime.h>

#include <cstring>

#include "crop.hpp"

namespace sgrt_crop {
namespace {

constexpr int kBatch = 32;
struct Req {
  int x0, y0, bw, bh, ow, oh;
  int64_t off;   // 출력 버퍼 안 바이트 위치
};
struct Batch {
  Req r[kBatch];
};

// 출력 화소 (i, j) = 원 화소 [x0 + i·bw/ow, x0 + (i+1)·bw/ow) × [..] 평균(최소 1 화소) — bestview.cpp cropRgbHost 와 같음
__global__ void cropKernel(const uint8_t* __restrict__ src, int64_t rs, int ps, Batch b, uint8_t* __restrict__ out) {
  const Req q = b.r[blockIdx.z];
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= q.ow || j >= q.oh) return;
  const int ya = q.y0 + j * q.bh / q.oh, yb = max(ya + 1, q.y0 + (j + 1) * q.bh / q.oh);
  const int xa = q.x0 + i * q.bw / q.ow, xb = max(xa + 1, q.x0 + (i + 1) * q.bw / q.ow);
  unsigned s0 = 0, s1 = 0, s2 = 0;
  for (int y = ya; y < yb; ++y) {
    const uint8_t* p = src + y * rs + int64_t(xa) * ps;
    for (int x = xa; x < xb; ++x, p += ps) { s0 += p[0]; s1 += p[1]; s2 += p[2]; }
  }
  const unsigned n = unsigned((yb - ya) * (xb - xa));
  uint8_t* d = out + q.off + (int64_t(j) * q.ow + i) * 3;
  d[0] = uint8_t((s0 + n / 2) / n);
  d[1] = uint8_t((s1 + n / 2) / n);
  d[2] = uint8_t((s2 + n / 2) / n);
}

}  // namespace

struct Gpu {
  cudaStream_t st = nullptr;
  uint8_t* d_out = nullptr;
  uint8_t* h_out = nullptr;   // 고정(pinned) 메모리
  size_t cap = 0;
};

Gpu* create() {
  auto* g = new Gpu();
  if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) {
    delete g;
    return nullptr;
  }
  return g;
}

void destroy(Gpu* g) {
  if (!g) return;
  cudaFree(g->d_out);
  cudaFreeHost(g->h_out);
  cudaStreamDestroy(g->st);
  delete g;
}

int run(Gpu* g, const uint8_t* src, int64_t rs, int ps, const sm_crop_req* reqs, int n) {
  if (!g || !src || n <= 0) return -1;
  size_t total = 0;
  for (int k = 0; k < n; ++k) total += size_t(reqs[k].out_w) * reqs[k].out_h * 3;
  if (total > g->cap) {
    cudaFree(g->d_out);
    cudaFreeHost(g->h_out);
    g->d_out = g->h_out = nullptr;
    g->cap = 0;
    const size_t cap = total + total / 2;
    if (cudaMalloc(&g->d_out, cap) != cudaSuccess || cudaMallocHost(&g->h_out, cap) != cudaSuccess) return -2;
    g->cap = cap;
  }
  size_t off = 0;
  for (int k0 = 0; k0 < n; k0 += kBatch) {
    Batch b{};
    const int m = n - k0 < kBatch ? n - k0 : kBatch;
    int mw = 1, mh = 1;
    for (int k = 0; k < m; ++k) {
      const sm_crop_req& r = reqs[k0 + k];
      b.r[k] = Req{r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, r.out_w, r.out_h, int64_t(off)};
      off += size_t(r.out_w) * r.out_h * 3;
      mw = r.out_w > mw ? r.out_w : mw;
      mh = r.out_h > mh ? r.out_h : mh;
    }
    const dim3 blk(16, 16), grd((mw + 15) / 16, (mh + 15) / 16, m);
    cropKernel<<<grd, blk, 0, g->st>>>(src, rs, ps, b, g->d_out);
  }
  cudaMemcpyAsync(g->h_out, g->d_out, total, cudaMemcpyDeviceToHost, g->st);
  if (cudaStreamSynchronize(g->st) != cudaSuccess) return -3;
  off = 0;
  for (int k = 0; k < n; ++k) {
    const size_t sz = size_t(reqs[k].out_w) * reqs[k].out_h * 3;
    std::memcpy(reqs[k].dst, g->h_out + off, sz);
    off += sz;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

}  // namespace sgrt_crop
