// openpi_client.image_tools.resize_with_pad on the GPU for device-resident views (same integer arithmetic as the
// host version in image.cpp, which reproduces Pillow's Resample.c: horizontal pass over the rows the vertical pass
// needs, 8-bit intermediate, vertical pass, then centered zero padding). Coefficients come from the same host code.
#include <cuda_runtime.h>

#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "common.cuh"
#include "image.h"

namespace pi05 {
namespace {
constexpr int PB = 32 - 8 - 2;

__device__ __forceinline__ uint8_t clip8d(int s) {
  s >>= PB;
  return s >= 255 ? 255 : (s <= 0 ? 0 : (uint8_t)s);
}

struct Plan {
  int in_w = 0, in_h = 0, rw = 0, rh = 0, ph = 0, pw = 0, ksh = 0, ksv = 0, yfirst = 0, rows = 0;
  bool need_h = false, need_v = false;
  int *bh = nullptr, *kh = nullptr, *bv = nullptr, *kv = nullptr;  // device
  uint8_t* tmp = nullptr;                                            // device [rows][tw][3]
};

std::mutex g_mu;
std::map<std::tuple<int, int, int, int>, Plan> g_plans;

const Plan& plan_for(int w, int h, int out_w, int out_h) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto key = std::make_tuple(w, h, out_w, out_h);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan p;
  p.in_w = w;
  p.in_h = h;
  const double ratio = std::fmax((double)w / out_w, (double)h / out_h);
  p.rh = (int)((double)h / ratio);
  p.rw = (int)((double)w / ratio);
  p.ph = std::max(0, (int)((out_h - p.rh) / 2.0));
  p.pw = std::max(0, (int)((out_w - p.rw) / 2.0));
  std::vector<int> bh, kh, bv, kv;
  p.ksh = pil_coeffs(w, p.rw, bh, kh);
  p.ksv = pil_coeffs(h, p.rh, bv, kv);
  p.need_h = p.rw != w;
  p.need_v = p.rh != h;
  const int yfirst = bv[0], ylast = bv[p.rh * 2 - 2] + bv[p.rh * 2 - 1];
  p.yfirst = p.need_v ? yfirst : 0;
  p.rows = p.need_v ? ylast - yfirst : h;
  if (p.need_v)
    for (int i = 0; i < p.rh; ++i) bv[i * 2] -= p.yfirst;
  auto up = [](const std::vector<int>& v, int** d) {
    PI05_CUDA(cudaMalloc(d, std::max<size_t>(v.size(), 1) * 4));
    PI05_CUDA(cudaMemcpy(*d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  };
  up(bh, &p.bh);
  up(kh, &p.kh);
  up(bv, &p.bv);
  up(kv, &p.kv);
  const int tw = p.need_h ? p.rw : w;
  PI05_CUDA(cudaMalloc(&p.tmp, (size_t)p.rows * tw * 3));
  return g_plans.emplace(key, p).first->second;
}

// horizontal pass (or plain copy of the needed rows when the width is unchanged)
__global__ void rs_h(const uint8_t* src, long long rs, int ps, Plan p, uint8_t* tmp) {
  const int tw = p.need_h ? p.rw : p.in_w;
  const long long total = (long long)p.rows * tw;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int y = (int)(i / tw), xx = (int)(i % tw);
    const uint8_t* row = src + (long long)(y + p.yfirst) * rs;
    uint8_t* o = tmp + i * 3;
    if (!p.need_h) {
      o[0] = row[(long long)xx * ps];
      o[1] = row[(long long)xx * ps + 1];
      o[2] = row[(long long)xx * ps + 2];
      continue;
    }
    const int xmin = p.bh[xx * 2], xmax = p.bh[xx * 2 + 1];
    const int* k = p.kh + (size_t)xx * p.ksh;
    int s0 = 1 << (PB - 1), s1 = s0, s2 = s0;
    for (int x = 0; x < xmax; ++x) {
      const uint8_t* q = row + (long long)(x + xmin) * ps;
      s0 += q[0] * k[x];
      s1 += q[1] * k[x];
      s2 += q[2] * k[x];
    }
    o[0] = clip8d(s0);
    o[1] = clip8d(s1);
    o[2] = clip8d(s2);
  }
}
// vertical pass + padding into out_h x out_w x 3
__global__ void rs_v(const uint8_t* tmp, Plan p, int out_h, int out_w, uint8_t* dst) {
  const int tw = p.need_h ? p.rw : p.in_w;
  const long long total = (long long)out_h * out_w;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int Y = (int)(i / out_w), X = (int)(i % out_w);
    uint8_t* o = dst + i * 3;
    const int yy = Y - p.ph, x = X - p.pw;
    if (yy < 0 || yy >= p.rh || x < 0 || x >= p.rw) {
      o[0] = o[1] = o[2] = 0;
      continue;
    }
    if (!p.need_v) {
      const uint8_t* q = tmp + ((size_t)yy * tw + x) * 3;
      o[0] = q[0];
      o[1] = q[1];
      o[2] = q[2];
      continue;
    }
    const int ymin = p.bv[yy * 2], ymax = p.bv[yy * 2 + 1];
    const int* k = p.kv + (size_t)yy * p.ksv;
    int s0 = 1 << (PB - 1), s1 = s0, s2 = s0;
    for (int y = 0; y < ymax; ++y) {
      const uint8_t* q = tmp + ((size_t)(y + ymin) * tw + x) * 3;
      s0 += q[0] * k[y];
      s1 += q[1] * k[y];
      s2 += q[2] * k[y];
    }
    o[0] = clip8d(s0);
    o[1] = clip8d(s1);
    o[2] = clip8d(s2);
  }
}
__global__ void rs_copy(const uint8_t* src, long long rs, int ps, int h, int w, uint8_t* dst) {
  const long long total = (long long)h * w;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int y = (int)(i / w), x = (int)(i % w);
    const uint8_t* q = src + (long long)y * rs + (long long)x * ps;
    dst[i * 3] = q[0];
    dst[i * 3 + 1] = q[1];
    dst[i * 3 + 2] = q[2];
  }
}
}  // namespace

void resize_with_pad_gpu(const uint8_t* src, int h, int w, long long row_stride, int pix_stride, int out_h, int out_w,
                         uint8_t* dst, void* stream) {
  cudaStream_t st = (cudaStream_t)stream;
  if (h == out_h && w == out_w) {
    rs_copy<<<256, 256, 0, st>>>(src, row_stride, pix_stride, h, w, dst);
    return;
  }
  const Plan& p = plan_for(w, h, out_w, out_h);
  rs_h<<<512, 256, 0, st>>>(src, row_stride, pix_stride, p, p.tmp);
  rs_v<<<256, 256, 0, st>>>(p.tmp, p, out_h, out_w, dst);
}

}  // namespace pi05
