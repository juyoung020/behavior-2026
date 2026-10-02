#include "scenemap/scan.hpp"

#include "scenemap/fk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace scenemap {

static inline float dist2(float ax, float ay, float az, const float* b) {
  const float dx = ax - b[0], dy = ay - b[1], dz = az - b[2];
  return dx * dx + dy * dy + dz * dz;
}

static inline float segDist2(float px, float py, float pz, const float* a, const float* b) {
  const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const float l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  float t = l2 > 1e-12f ? ((px - a[0]) * ab[0] + (py - a[1]) * ab[1] + (pz - a[2]) * ab[2]) / l2 : 0.f;
  t = std::fmin(std::fmax(t, 0.f), 1.f);
  const float c[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
  return dist2(px, py, pz, c);
}

BodyState bodyFromFk(const BodyFk& fk, const float eef[2][3], float arm_r, float hand_r, float torso_r) {
  BodyState b;
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 3; ++k) b.eef[s][k] = eef[s][k];
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k + 1 < BodyFk::kArmPts; ++k) {
      Capsule c;
      for (int i = 0; i < 3; ++i) { c.a[i] = fk.arm[s][k][i]; c.b[i] = fk.arm[s][k + 1][i]; }
      if (c.a[0] == c.b[0] && c.a[1] == c.b[1] && c.a[2] == c.b[2]) continue;
      c.r = k >= BodyFk::kArmPts - 3 ? hand_r : arm_r;
      b.caps.push_back(c);
    }
  for (int k = 1; k + 1 < 6; ++k) {
    Capsule c;
    for (int i = 0; i < 3; ++i) { c.a[i] = fk.torso[k][i]; c.b[i] = fk.torso[k + 1][i]; }
    c.r = torso_r;
    b.caps.push_back(c);
  }
  return b;
}

void AttachFilter::update(const std::vector<int64_t>& cur, const Pose2& odom) {
  if (!p_.enabled) return;
  if (!have_) {
    score_.clear();
    for (int64_t k : cur) score_[k] = 0;
    check_ = odom;
    have_ = true;
    return;
  }
  const double dxy = std::hypot(odom.x - check_.x, odom.y - check_.y), dth = std::fabs(wrapAngle(odom.th - check_.th));
  if (dxy < p_.move_xy && dth < p_.move_yaw) return;
  auto unpack = [](int64_t k, int* i, int* j, int* l) {
    auto sx = [](int64_t v) { return int((v ^ 0x80000) - 0x80000); };   // 20비트 부호 확장
    *i = sx((k >> 40) & 0xfffff); *j = sx((k >> 20) & 0xfffff); *l = sx(k & 0xfffff);
  };
  std::unordered_map<int64_t, int> next;
  next.reserve(cur.size());
  for (int64_t k : cur) {
    int i, j, l;
    unpack(k, &i, &j, &l);
    int best = -1;
    for (int a = -1; a <= 1; ++a)
      for (int b = -1; b <= 1; ++b)
        for (int c = -1; c <= 1; ++c) {
          auto it = score_.find(key3(i + a, j + b, l + c));
          if (it != score_.end() && it->second > best) best = it->second;
        }
    next[k] = best + 1;   // 지난번 근처에 있었으면 +1, 없었으면 0
  }
  score_.swap(next);
  check_ = odom;
}

size_t AttachFilter::nAttached() const {
  size_t n = 0;
  for (const auto& kv : score_) n += kv.second >= p_.min_score;
  return n;
}

void makeScan(const DepthView& d, const BodyState& b, const ScanParams& p, Scan2* out, const AttachFilter* att,
              std::vector<int64_t>* vox_out) {
  const int gw = (d.w + d.step - 1) / d.step, gh = (d.h + d.step - 1) / d.step;
  // 1. 격자 화소마다 베이스 기준 점(없으면 NaN)
  std::vector<float> P(size_t(gw) * gh * 3, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> Zo(size_t(gw) * gh, 0.f);   // 광학 z(가장자리 판정)
  const float* T = d.T_bc;
  for (int j = 0; j < gh; ++j) {
    const int v = j * d.step;
    const uint16_t* row = d.m ? nullptr : d.mm + size_t(v) * d.w;
    const float* rowf = d.m ? d.m + size_t(v) * d.w : nullptr;
    const float yn = (v - d.cy) / d.fy;
    for (int i = 0; i < gw; ++i) {
      const int u = i * d.step;
      const float z = rowf ? rowf[u] : row[u] * 1e-3f;
      if (!(z > 0.f)) continue;
      if (z < p.zmin || z > p.zmax) continue;
      const float X = (u - d.cx) / d.fx * z, Y = yn * z;
      float* q = &P[(size_t(j) * gw + i) * 3];
      q[0] = T[0] * X + T[1] * Y + T[2] * z + T[3];
      q[1] = T[4] * X + T[5] * Y + T[6] * z + T[7];
      q[2] = T[8] * X + T[9] * Y + T[10] * z + T[11];
      Zo[size_t(j) * gw + i] = z;
    }
  }
  // 2. 분류
  const int nb = p.bins;
  std::vector<float> hit_r(nb, std::numeric_limits<float>::infinity()), hx(nb), hy(nb), floor_r(nb, 0.f);
  const float bin_scale = nb / (2.f * float(M_PI));
  float ab[2][3], ab2[2];
  for (int s = 0; s < 2; ++s) {
    for (int k = 0; k < 3; ++k) ab[s][k] = b.eef[s][k] - p.shoulder[s][k];
    ab2[s] = std::max(ab[s][0] * ab[s][0] + ab[s][1] * ab[s][1] + ab[s][2] * ab[s][2], 1e-9f);
  }
  const float self2 = p.self_r * p.self_r, eef2 = p.eef_r * p.eef_r, arm2 = p.arm_r * p.arm_r;
  // 캡슐 전체를 감싼 상자(밖의 점은 캡슐 검사를 건너뜀)
  float bb0[3] = {1e9f, 1e9f, 1e9f}, bb1[3] = {-1e9f, -1e9f, -1e9f};
  for (const Capsule& c : b.caps)
    for (int i = 0; i < 3; ++i) {
      bb0[i] = std::fmin(bb0[i], std::fmin(c.a[i], c.b[i]) - c.r);
      bb1[i] = std::fmax(bb1[i], std::fmax(c.a[i], c.b[i]) + c.r);
    }
  struct Acc { float x, y, nx, ny; int n; };
  std::unordered_map<int64_t, Acc> cells;
  cells.reserve(4096);
  const float inv_mc = 1.f / p.match_cell;
  for (int j = 0; j < gh; ++j)
    for (int i = 0; i < gw; ++i) {
      const float* q = &P[(size_t(j) * gw + i) * 3];
      const float px = q[0], py = q[1], pz = q[2];
      if (std::isnan(px)) continue;
      const float r2 = px * px + py * py;
      if (r2 < self2) continue;
      bool self = false;
      if (px >= bb0[0] && px <= bb1[0] && py >= bb0[1] && py <= bb1[1] && pz >= bb0[2] && pz <= bb1[2])
        for (const Capsule& c : b.caps)
          if (segDist2(px, py, pz, c.a, c.b) < c.r * c.r) { self = true; break; }
      for (int s = 0; s < 2 && !self; ++s) {
        if (!b.caps.empty()) {
          if (dist2(px, py, pz, b.eef[s]) < eef2) self = true;
          continue;
        }
        if (dist2(px, py, pz, b.eef[s]) < eef2) { self = true; break; }
        const float* a = p.shoulder[s];
        float t = ((px - a[0]) * ab[s][0] + (py - a[1]) * ab[s][1] + (pz - a[2]) * ab[s][2]) / ab2[s];
        t = std::fmin(std::fmax(t, 0.f), 1.f);
        const float c[3] = {a[0] + t * ab[s][0], a[1] + t * ab[s][1], a[2] + t * ab[s][2]};
        if (dist2(px, py, pz, c) < arm2) self = true;
      }
      if (self) continue;
      if (pz >= p.band_lo && att && att->near(px, py)) {
        if (vox_out) vox_out->push_back(att->key(px, py, pz));
        if (att->attached(px, py, pz)) continue;
      }
      // 레이저 한 줄(띠 안 가장 가까운 것) · 바닥 빈칸
      if (pz <= p.band_hi) {
        const float rx = px - T[3], ry = py - T[7];
        int k = int((std::atan2(ry, rx) + float(M_PI)) * bin_scale);
        k = k < 0 ? 0 : (k >= nb ? nb - 1 : k);
        const float r = std::sqrt(rx * rx + ry * ry);
        if (pz < p.band_lo) {
          if (r > floor_r[k]) floor_r[k] = r;
        } else if (r < hit_r[k]) {
          hit_r[k] = r; hx[k] = px; hy[k] = py;
        }
      }
      // 맞추기 점: 수직면 쪽
      if (!p.dense || pz < p.band_lo || pz > p.match_hi || i == 0 || j == 0 || i == gw - 1 || j == gh - 1) continue;
      const size_t c0 = size_t(j) * gw + i;
      const float* l = &P[(c0 - 1) * 3];
      const float* rr = &P[(c0 + 1) * 3];
      const float* up = &P[(c0 - gw) * 3];
      const float* dn = &P[(c0 + gw) * 3];
      if (std::isnan(l[0]) || std::isnan(rr[0]) || std::isnan(up[0]) || std::isnan(dn[0])) continue;
      const float z = Zo[c0];
      if (std::fabs(Zo[c0 + 1] - Zo[c0 - 1]) >= 0.1f * z || std::fabs(Zo[c0 + gw] - Zo[c0 - gw]) >= 0.1f * z) continue;
      const float ax = rr[0] - l[0], ay = rr[1] - l[1], az = rr[2] - l[2];
      const float bx = dn[0] - up[0], by = dn[1] - up[1], bz = dn[2] - up[2];
      const float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const float nn = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (nn <= 0 || std::fabs(nz) >= p.vert_nz * nn) continue;
      const int64_t key = (int64_t(std::floor(px * inv_mc)) << 32) ^ (int64_t(std::floor(py * inv_mc)) & 0xffffffff);
      Acc& a = cells[key];
      float hx = nx, hy = ny;
      const float hn = std::sqrt(hx * hx + hy * hy);
      hx /= hn; hy /= hn;
      if (hx * (T[3] - px) + hy * (T[7] - py) < 0) { hx = -hx; hy = -hy; }   // 카메라 쪽
      a.x += px; a.y += py; a.nx += hx; a.ny += hy; ++a.n;
    }
  if (vox_out) {
    std::sort(vox_out->begin(), vox_out->end());
    vox_out->erase(std::unique(vox_out->begin(), vox_out->end()), vox_out->end());
  }
  out->ox = T[3];
  out->oy = T[7];
  out->hx.clear(); out->hy.clear(); out->fx.clear(); out->fy.clear(); out->mx.clear(); out->my.clear(); out->mnx.clear(); out->mny.clear();
  for (int k = 0; k < nb; ++k) {
    if (std::isfinite(hit_r[k])) {
      out->hx.push_back(hx[k]);
      out->hy.push_back(hy[k]);
    } else if (floor_r[k] > 0) {
      const float a = (k + 0.5f) / bin_scale - float(M_PI);
      out->fx.push_back(T[3] + floor_r[k] * std::cos(a));
      out->fy.push_back(T[7] + floor_r[k] * std::sin(a));
    }
  }
  out->mx.reserve(cells.size());
  out->my.reserve(cells.size());
  for (const auto& kv : cells) {
    out->mx.push_back(kv.second.x / kv.second.n);
    out->my.push_back(kv.second.y / kv.second.n);
    const float nn = std::sqrt(kv.second.nx * kv.second.nx + kv.second.ny * kv.second.ny);
    out->mnx.push_back(nn > 1e-6f ? kv.second.nx / nn : 0.f);
    out->mny.push_back(nn > 1e-6f ? kv.second.ny / nn : 0.f);
  }
}

}  // namespace scenemap
