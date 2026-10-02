#include "scenemap/grid.hpp"

#include <algorithm>
#include <cmath>

namespace scenemap {

void OccGrid::ensure(int ix0, int iy0, int ix1, int iy1) {
  if (w_ && ix0 >= x0_ && iy0 >= y0_ && ix1 < x0_ + w_ && iy1 < y0_ + h_) return;
  const int pad = 64;
  int nx0 = w_ ? std::min(x0_, ix0 - pad) : ix0 - pad;
  int ny0 = w_ ? std::min(y0_, iy0 - pad) : iy0 - pad;
  int nx1 = w_ ? std::max(x0_ + w_, ix1 + pad) : ix1 + pad;
  int ny1 = w_ ? std::max(y0_ + h_, iy1 + pad) : iy1 + pad;
  const int nw = nx1 - nx0, nh = ny1 - ny0;
  std::vector<float> L(size_t(nw) * nh, 0.f), sx(L.size(), 0.f), sy(L.size(), 0.f), nx(L.size(), 0.f), ny(L.size(), 0.f);
  std::vector<uint16_t> cntn(L.size(), 0);
  std::vector<uint8_t> seen(L.size(), 0);
  std::vector<uint16_t> cnt(L.size(), 0);
  std::vector<uint32_t> st(L.size(), 0);
  for (int y = 0; y < h_; ++y)
    for (int x = 0; x < w_; ++x) {
      const size_t o = size_t(y) * w_ + x, n = size_t(y + y0_ - ny0) * nw + (x + x0_ - nx0);
      L[n] = L_[o]; seen[n] = seen_[o]; sx[n] = sx_[o]; sy[n] = sy_[o]; cnt[n] = cnt_[o]; st[n] = stamp_[o];
      nx[n] = nx_[o]; ny[n] = ny_[o]; cntn[n] = cntn_[o];
    }
  L_.swap(L); seen_.swap(seen); sx_.swap(sx); sy_.swap(sy); cnt_.swap(cnt); stamp_.swap(st);
  nx_.swap(nx); ny_.swap(ny); cntn_.swap(cntn);
  x0_ = nx0; y0_ = ny0; w_ = nw; h_ = nh;
}

void OccGrid::touch(int ix, int iy, bool hit, float hx, float hy, float nx, float ny) {
  const size_t i = idx(ix, iy);
  if (stamp_[i] == scan_id_) return;
  stamp_[i] = scan_id_;
  seen_[i] = 1;
  float& l = L_[i];
  l = std::clamp(l + (hit ? p_.l_hit : p_.l_miss), p_.l_min, p_.l_max);
  if (hit) {
    if (cnt_[i] < 60000) {
      sx_[i] += hx; sy_[i] += hy; ++cnt_[i];
    }
    if ((nx != 0 || ny != 0) && cntn_[i] < 60000) {
      nx_[i] += nx; ny_[i] += ny; ++cntn_[i];
    }
  } else if (l <= 0.f) {
    sx_[i] = sy_[i] = 0.f; cnt_[i] = 0;
    nx_[i] = ny_[i] = 0.f; cntn_[i] = 0;
  }
}

void OccGrid::insert(const Scan2& s, const Pose2& pose) {
  ++scan_id_;
  const double c = std::cos(pose.th), sn = std::sin(pose.th);
  auto tf = [&](float x, float y, double* wx, double* wy) {
    *wx = pose.x + c * x - sn * y;
    *wy = pose.y + sn * x + c * y;
  };
  double ox, oy;
  tf(s.ox, s.oy, &ox, &oy);
  const int cx = cellOf(ox), cy = cellOf(oy);
  // 경계 먼저
  int bx0 = cx, by0 = cy, bx1 = cx, by1 = cy;
  auto grow = [&](const std::vector<float>& X, const std::vector<float>& Y) {
    for (size_t k = 0; k < X.size(); ++k) {
      double wx, wy;
      tf(X[k], Y[k], &wx, &wy);
      const int ix = cellOf(wx), iy = cellOf(wy);
      bx0 = std::min(bx0, ix); by0 = std::min(by0, iy); bx1 = std::max(bx1, ix); by1 = std::max(by1, iy);
    }
  };
  grow(s.hx, s.hy);
  grow(s.fx, s.fy);
  grow(s.mx, s.my);
  ensure(bx0, by0, bx1, by1);
  ++version_;
  if (!dirty_) { dx0_ = bx0; dy0_ = by0; dx1_ = bx1; dy1_ = by1; dirty_ = true; }
  else { dx0_ = std::min(dx0_, bx0); dy0_ = std::min(dy0_, by0); dx1_ = std::max(dx1_, bx1); dy1_ = std::max(dy1_, by1); }
  // 맞음 먼저(한 스캔에서 맞은 칸은 빈칸으로 덮이지 않게)
  // 맞추기 점 먼저(법선이 있는 쪽이 칸의 첫 맞음이 되게), 그다음 레이저 한 줄
  for (size_t k = 0; k < s.mx.size(); ++k) {
    double wx, wy;
    tf(s.mx[k], s.my[k], &wx, &wy);
    const float nx = k < s.mnx.size() ? float(c * s.mnx[k] - sn * s.mny[k]) : 0.f;
    const float ny = k < s.mnx.size() ? float(sn * s.mnx[k] + c * s.mny[k]) : 0.f;
    touch(cellOf(wx), cellOf(wy), true, float(wx), float(wy), nx, ny);
  }
  for (size_t k = 0; k < s.hx.size(); ++k) {
    double wx, wy;
    tf(s.hx[k], s.hy[k], &wx, &wy);
    touch(cellOf(wx), cellOf(wy), true, float(wx), float(wy));
  }

  auto ray = [&](int x1, int y1) {   // Bresenham, 끝 칸 제외
    int x = cx, y = cy;
    const int dx = std::abs(x1 - x), dy = -std::abs(y1 - y), sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1;
    int err = dx + dy;
    while (!(x == x1 && y == y1)) {
      touch(x, y, false, 0, 0);
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
  };
  for (size_t k = 0; k < s.hx.size(); ++k) {
    double wx, wy;
    tf(s.hx[k], s.hy[k], &wx, &wy);
    ray(cellOf(wx), cellOf(wy));
  }
  for (size_t k = 0; k < s.fx.size(); ++k) {
    double wx, wy;
    tf(s.fx[k], s.fy[k], &wx, &wy);
    const int ix = cellOf(wx), iy = cellOf(wy);
    ray(ix, iy);
    touch(ix, iy, false, 0, 0);
  }
}

bool OccGrid::normal(int ix, int iy, float* nx, float* ny) const {
  if (!inside(ix, iy)) return false;
  const size_t i = idx(ix, iy);
  if (!cntn_[i] || L_[i] <= 0.f) return false;
  const float n = std::sqrt(nx_[i] * nx_[i] + ny_[i] * ny_[i]);
  if (n < 0.5f * cntn_[i]) return false;
  *nx = nx_[i] / n;
  *ny = ny_[i] / n;
  return true;
}

bool OccGrid::mean(int ix, int iy, float* mx, float* my, int* n) const {
  if (!inside(ix, iy)) return false;
  const size_t i = idx(ix, iy);
  if (!cnt_[i] || L_[i] <= 0.f) return false;
  *mx = sx_[i] / cnt_[i];
  *my = sy_[i] / cnt_[i];
  if (n) *n = cnt_[i];
  return true;
}

float OccGrid::prob(int ix, int iy) const {
  if (!inside(ix, iy)) return pmin;
  const size_t i = idx(ix, iy);
  if (!seen_[i]) return pmin;
  const float pr = 1.f / (1.f + std::exp(-L_[i]));
  return std::clamp(pr, pmin, pmax);
}

bool OccGrid::takeDirty(int* ix0, int* iy0, int* ix1, int* iy1) {
  if (!dirty_) return false;
  *ix0 = dx0_; *iy0 = dy0_; *ix1 = dx1_; *iy1 = dy1_;
  dirty_ = false;
  return true;
}

std::vector<int8_t> OccGrid::export8() const {
  std::vector<int8_t> o(L_.size(), -1);
  for (size_t i = 0; i < L_.size(); ++i)
    if (seen_[i]) o[i] = int8_t(std::lround(100.f / (1.f + std::exp(-L_[i]))));
  return o;
}

}  // namespace scenemap
