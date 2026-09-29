// 렌더 비교 지표 (호스트 전용, 손으로 짬). 층 3 검증: 우리 렌더 vs 공식 RTX, 그리고 공식 자신의 잡음 폭(같은 상태 다시 그리기·다른 실행).
//  - depth(depth_linear = image plane 까지 거리): 둘 다 유효한 픽셀의 |차| 분포(중앙·p90·p99·최대), 상대 오차, 허용오차 안 비율, 비트 같음 수,
//    한쪽만 유효(안 맞음/맞음 불일치) 픽셀 수.
//  - RGB: 채널 평균·차, 채널 히스토그램 거리(EMD, 누적분포 L1, 단위 = 8 비트 단계), 밝기 SSIM(가우스 11x11, σ=1.5, Wang 2004), PSNR, 평균 |차|.
//  - 검은 프레임: RGB 평균 < 2 (tools/black_frame_check.py 와 같은 문턱) 이면 그 쌍은 비교에서 뺀다.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rcmp {

inline bool depth_valid(float d) { return std::isfinite(d) && d > 0.0f && d < 1.0e6f; }

struct DepthStats {
  int64_t n = 0, both = 0, a_only = 0, b_only = 0, neither = 0, bit_equal = 0;
  double med = 0, p90 = 0, p99 = 0, maxd = 0, mean = 0, rel_med = 0, rel_p99 = 0;
  double within_1mm = 0, within_1cm = 0, within_1pct = 0;  // both 중 비율
};

inline double quant(std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  const size_t k = std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

inline DepthStats depth_compare(const float* a, const float* b, int64_t n) {
  DepthStats s;
  s.n = n;
  std::vector<double> ad, rd;
  ad.reserve(size_t(n));
  rd.reserve(size_t(n));
  double sum = 0;
  int64_t w1mm = 0, w1cm = 0, w1pct = 0;
  for (int64_t i = 0; i < n; ++i) {
    const bool va = depth_valid(a[i]), vb = depth_valid(b[i]);
    if (va && vb) {
      ++s.both;
      if (a[i] == b[i]) ++s.bit_equal;
      const double d = std::fabs(double(a[i]) - double(b[i]));
      ad.push_back(d);
      rd.push_back(d / double(a[i]));
      sum += d;
      if (d > s.maxd) s.maxd = d;
      if (d <= 1e-3) ++w1mm;
      if (d <= 1e-2) ++w1cm;
      if (d <= 0.01 * double(a[i])) ++w1pct;
    } else if (va) {
      ++s.a_only;
    } else if (vb) {
      ++s.b_only;
    } else {
      ++s.neither;
    }
  }
  if (s.both) {
    s.mean = sum / double(s.both);
    s.within_1mm = double(w1mm) / double(s.both);
    s.within_1cm = double(w1cm) / double(s.both);
    s.within_1pct = double(w1pct) / double(s.both);
    s.med = quant(ad, 0.5);
    s.p90 = quant(ad, 0.9);
    s.p99 = quant(ad, 0.99);
    s.rel_med = quant(rd, 0.5);
    s.rel_p99 = quant(rd, 0.99);
  }
  return s;
}

struct RgbStats {
  double mean_a[3] = {0, 0, 0}, mean_b[3] = {0, 0, 0};
  double emd[3] = {0, 0, 0};
  double ssim = 0, psnr = 0, mad = 0;
};

inline double rgb_mean(const uint8_t* a, int64_t npx) {
  double s = 0;
  for (int64_t i = 0; i < npx * 3; ++i) s += a[i];
  return s / double(npx * 3);
}
inline bool is_black(const uint8_t* a, int64_t npx) { return rgb_mean(a, npx) < 2.0; }

// 가우스 창 SSIM (밝기 Y = 0.299R + 0.587G + 0.114B), 가장자리는 창이 다 들어가는 곳만
inline double ssim_luma(const uint8_t* a, const uint8_t* b, int W, int H) {
  const int R = 5;
  double g[2 * R + 1], gs = 0;
  for (int k = -R; k <= R; ++k) gs += (g[k + R] = std::exp(-double(k * k) / (2.0 * 1.5 * 1.5)));
  for (double& x : g) x /= gs;
  std::vector<double> ya(size_t(W) * H), yb(size_t(W) * H);
  for (int64_t i = 0; i < int64_t(W) * H; ++i) {
    ya[i] = 0.299 * a[3 * i] + 0.587 * a[3 * i + 1] + 0.114 * a[3 * i + 2];
    yb[i] = 0.299 * b[3 * i] + 0.587 * b[3 * i + 1] + 0.114 * b[3 * i + 2];
  }
  // 곱 5 개(x, y, x², y², xy)를 가로 → 세로로 거른다
  const int W2 = W - 2 * R, H2 = H - 2 * R;
  if (W2 <= 0 || H2 <= 0) return 0.0;
  std::vector<double> t(size_t(5) * W2 * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W2; ++x) {
      double s[5] = {0, 0, 0, 0, 0};
      for (int k = 0; k <= 2 * R; ++k) {
        const double p = ya[size_t(y) * W + x + k], q = yb[size_t(y) * W + x + k];
        s[0] += g[k] * p; s[1] += g[k] * q; s[2] += g[k] * p * p; s[3] += g[k] * q * q; s[4] += g[k] * p * q;
      }
      for (int c = 0; c < 5; ++c) t[(size_t(c) * H + y) * W2 + x] = s[c];
    }
  const double C1 = (0.01 * 255) * (0.01 * 255), C2 = (0.03 * 255) * (0.03 * 255);
  double acc = 0;
  for (int y = 0; y < H2; ++y)
    for (int x = 0; x < W2; ++x) {
      double s[5] = {0, 0, 0, 0, 0};
      for (int k = 0; k <= 2 * R; ++k)
        for (int c = 0; c < 5; ++c) s[c] += g[k] * t[(size_t(c) * H + y + k) * W2 + x];
      const double ma = s[0], mb = s[1], va = s[2] - ma * ma, vb = s[3] - mb * mb, cab = s[4] - ma * mb;
      acc += ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
    }
  return acc / (double(W2) * H2);
}

inline RgbStats rgb_compare(const uint8_t* a, const uint8_t* b, int W, int H) {
  RgbStats s;
  const int64_t n = int64_t(W) * H;
  int64_t ha[3][256] = {}, hb[3][256] = {};
  double se = 0, ad = 0;
  for (int64_t i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
      const int p = a[3 * i + c], q = b[3 * i + c];
      s.mean_a[c] += p;
      s.mean_b[c] += q;
      ha[c][p]++;
      hb[c][q]++;
      se += double(p - q) * double(p - q);
      ad += std::abs(p - q);
    }
  for (int c = 0; c < 3; ++c) {
    s.mean_a[c] /= double(n);
    s.mean_b[c] /= double(n);
    double ca = 0, cb = 0, e = 0;
    for (int k = 0; k < 256; ++k) {
      ca += double(ha[c][k]) / double(n);
      cb += double(hb[c][k]) / double(n);
      e += std::fabs(ca - cb);
    }
    s.emd[c] = e;
  }
  const double mse = se / double(n * 3);
  s.psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
  s.mad = ad / double(n * 3);
  s.ssim = ssim_luma(a, b, W, H);
  return s;
}

// 가까운 이웃으로 크기 맞추기 (해상도가 다른 두 영상을 비교할 때만; 공식 224 는 우리도 224 로 그려 비교하는 것이 원칙)
inline std::vector<uint8_t> resize_nearest(const uint8_t* a, int W, int H, int W2, int H2) {
  std::vector<uint8_t> o(size_t(W2) * H2 * 3);
  for (int y = 0; y < H2; ++y)
    for (int x = 0; x < W2; ++x) {
      const int sx = std::min(W - 1, int((x + 0.5) * W / W2)), sy = std::min(H - 1, int((y + 0.5) * H / H2));
      for (int c = 0; c < 3; ++c) o[(size_t(y) * W2 + x) * 3 + c] = a[(size_t(sy) * W + sx) * 3 + c];
    }
  return o;
}

}  // namespace rcmp
