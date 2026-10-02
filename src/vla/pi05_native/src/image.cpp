#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pi05 {
namespace {

constexpr int PRECISION_BITS = 32 - 8 - 2;

double bilinear(double x) {
  if (x < 0.0) x = -x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

// Resample.c precompute_coeffs + normalize_coeffs_8bpc
int precompute(int in_size, double in0, double in1, int out_size, std::vector<int>& bounds, std::vector<int>& kk) {
  double scale = (in1 - in0) / out_size;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  double support = 1.0 * filterscale;
  int ksize = (int)std::ceil(support) * 2 + 1;
  std::vector<double> k((size_t)out_size * ksize);
  bounds.assign((size_t)out_size * 2, 0);
  for (int xx = 0; xx < out_size; ++xx) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0, ss = 1.0 / filterscale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* kr = &k[(size_t)xx * ksize];
    int x = 0;
    for (; x < xmax; ++x) {
      double w = bilinear((x + xmin - center + 0.5) * ss);
      kr[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; ++x)
      if (ww != 0.0) kr[x] /= ww;
    for (; x < ksize; ++x) kr[x] = 0;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(k.size());
  for (size_t i = 0; i < k.size(); ++i)
    kk[i] = k[i] < 0 ? (int)(-0.5 + k[i] * (1 << PRECISION_BITS)) : (int)(0.5 + k[i] * (1 << PRECISION_BITS));
  return ksize;
}

inline uint8_t clip8(int in) {
  in >>= PRECISION_BITS;
  return in >= 255 ? 255 : (in <= 0 ? 0 : (uint8_t)in);
}

// PIL Image.resize((new_w, new_h), BILINEAR) for an RGB image held as 3 channels.
void pil_resize(const ImageView& src, int nw, int nh, std::vector<uint8_t>& out) {
  std::vector<int> bh, kh, bv, kv;
  const int ksh = precompute(src.w, 0.0, (double)src.w, nw, bh, kh);
  const int ksv = precompute(src.h, 0.0, (double)src.h, nh, bv, kv);
  const bool need_h = nw != src.w, need_v = nh != src.h;
  // horizontal pass over the source rows the vertical pass needs
  const int yfirst = bv[0], ylast = bv[nh * 2 - 2] + bv[nh * 2 - 1];
  std::vector<uint8_t> tmp;
  int th, tw;
  auto px = [&](int y, int x, int c) { return src.data[(long long)y * src.row_stride + (long long)x * src.pix_stride + c]; };
  if (need_h) {
    th = need_v ? ylast - yfirst : src.h;
    const int y0 = need_v ? yfirst : 0;
    tw = nw;
    tmp.resize((size_t)th * tw * 3);
    for (int y = 0; y < th; ++y)
      for (int xx = 0; xx < nw; ++xx) {
        const int xmin = bh[xx * 2], xmax = bh[xx * 2 + 1];
        const int* k = &kh[(size_t)xx * ksh];
        int s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < xmax; ++x) {
          s0 += px(y + y0, x + xmin, 0) * k[x];
          s1 += px(y + y0, x + xmin, 1) * k[x];
          s2 += px(y + y0, x + xmin, 2) * k[x];
        }
        uint8_t* o = &tmp[((size_t)y * tw + xx) * 3];
        o[0] = clip8(s0); o[1] = clip8(s1); o[2] = clip8(s2);
      }
    if (need_v)
      for (int i = 0; i < nh; ++i) bv[i * 2] -= yfirst;
  } else {
    th = src.h;
    tw = src.w;
    tmp.resize((size_t)th * tw * 3);
    for (int y = 0; y < th; ++y)
      for (int x = 0; x < tw; ++x)
        for (int c = 0; c < 3; ++c) tmp[((size_t)y * tw + x) * 3 + c] = px(y, x, c);
  }
  if (!need_v) { out = std::move(tmp); return; }
  out.resize((size_t)nh * tw * 3);
  for (int yy = 0; yy < nh; ++yy) {
    const int ymin = bv[yy * 2], ymax = bv[yy * 2 + 1];
    const int* k = &kv[(size_t)yy * ksv];
    for (int x = 0; x < tw; ++x) {
      int s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
      for (int y = 0; y < ymax; ++y) {
        const uint8_t* p = &tmp[((size_t)(y + ymin) * tw + x) * 3];
        s0 += p[0] * k[y];
        s1 += p[1] * k[y];
        s2 += p[2] * k[y];
      }
      uint8_t* o = &out[((size_t)yy * tw + x) * 3];
      o[0] = clip8(s0); o[1] = clip8(s1); o[2] = clip8(s2);
    }
  }
}

}  // namespace

void resize_with_pad(const ImageView& src, int out_h, int out_w, uint8_t* dst) {
  if (src.h == out_h && src.w == out_w) {
    for (int y = 0; y < out_h; ++y)
      for (int x = 0; x < out_w; ++x)
        for (int c = 0; c < 3; ++c)
          dst[((size_t)y * out_w + x) * 3 + c] = src.data[(long long)y * src.row_stride + (long long)x * src.pix_stride + c];
    return;
  }
  // image_tools._resize_with_pad_pil: ratio = max(cur_w / w, cur_h / h); resized = int(cur / ratio)
  const double ratio = std::fmax((double)src.w / out_w, (double)src.h / out_h);
  const int rh = (int)((double)src.h / ratio), rw = (int)((double)src.w / ratio);
  std::vector<uint8_t> r;
  pil_resize(src, rw, rh, r);
  memset(dst, 0, (size_t)out_h * out_w * 3);
  const int ph = std::max(0, (int)((out_h - rh) / 2.0)), pw = std::max(0, (int)((out_w - rw) / 2.0));
  for (int y = 0; y < rh && y + ph < out_h; ++y)
    memcpy(dst + ((size_t)(y + ph) * out_w + pw) * 3, &r[(size_t)y * rw * 3], (size_t)std::min(rw, out_w - pw) * 3);
}

}  // namespace pi05

namespace pi05 {
int pil_coeffs(int in_size, int out_size, std::vector<int>& bounds, std::vector<int>& kk) {
  return precompute(in_size, 0.0, (double)in_size, out_size, bounds, kk);
}
}  // namespace pi05
