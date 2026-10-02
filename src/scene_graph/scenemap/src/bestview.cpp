// 물체별 best view 자르기(include/scenemap/bestview.hpp). 장치 쪽 같은 계산은 runtime/src/crop.cu.
#include "scenemap/bestview.hpp"

#include <algorithm>
#include <cmath>

namespace scenemap {

bool cropGeometry(const float b[4], int img_w, int img_h, float margin, int max_side, int32_t box[4], int32_t* out_w,
                  int32_t* out_h) {
  const float mx = margin * (b[2] - b[0]), my = margin * (b[3] - b[1]);
  box[0] = std::clamp(int(std::floor(b[0] - mx)), 0, img_w);
  box[1] = std::clamp(int(std::floor(b[1] - my)), 0, img_h);
  box[2] = std::clamp(int(std::ceil(b[2] + mx)), 0, img_w);
  box[3] = std::clamp(int(std::ceil(b[3] + my)), 0, img_h);
  const int bw = box[2] - box[0], bh = box[3] - box[1];
  if (bw <= 0 || bh <= 0) return false;
  const double s = std::min(1.0, double(max_side) / std::max(bw, bh));
  *out_w = std::max(1, int(std::lround(bw * s)));
  *out_h = std::max(1, int(std::lround(bh * s)));
  return true;
}

// 출력 화소 i 가 덮는 원 화소 [a, b): a = x0 + i·bw/ow, b = x0 + (i+1)·bw/ow (최소 1 화소). crop.cu 와 같은 식.
void cropRgbHost(const uint8_t* src, int64_t rs, int ps, const sm_crop_req& r) {
  const int bw = r.x1 - r.x0, bh = r.y1 - r.y0;
  for (int j = 0; j < r.out_h; ++j) {
    const int ya = r.y0 + j * bh / r.out_h, yb = std::max(ya + 1, r.y0 + (j + 1) * bh / r.out_h);
    for (int i = 0; i < r.out_w; ++i) {
      const int xa = r.x0 + i * bw / r.out_w, xb = std::max(xa + 1, r.x0 + (i + 1) * bw / r.out_w);
      uint32_t s[3] = {0, 0, 0};
      for (int y = ya; y < yb; ++y) {
        const uint8_t* p = src + y * rs + int64_t(xa) * ps;
        for (int x = xa; x < xb; ++x, p += ps) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
      }
      const uint32_t n = uint32_t((yb - ya) * (xb - xa));
      uint8_t* d = r.dst + (size_t(j) * r.out_w + i) * 3;
      for (int c = 0; c < 3; ++c) d[c] = uint8_t((s[c] + n / 2) / n);
    }
  }
}

void cropDepthMm(const float* dm, int dw, int dh, int img_w, int img_h, const int32_t box[4], int ow, int oh, uint16_t* dst) {
  const double sx = img_w > 0 ? double(dw) / img_w : 1.0, sy = img_h > 0 ? double(dh) / img_h : 1.0;
  const double bw = box[2] - box[0], bh = box[3] - box[1];
  for (int j = 0; j < oh; ++j) {
    const int v = std::clamp(int((box[1] + (j + 0.5) * bh / oh) * sy), 0, dh - 1);
    for (int i = 0; i < ow; ++i) {
      const int u = std::clamp(int((box[0] + (i + 0.5) * bw / ow) * sx), 0, dw - 1);
      const float z = dm[size_t(v) * dw + u];
      dst[size_t(j) * ow + i] = (z > 0 && z < 65.535f) ? uint16_t(std::lround(z * 1000.f)) : 0;
    }
  }
}

}  // namespace scenemap
