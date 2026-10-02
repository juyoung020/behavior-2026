#include "scenemap/objmap.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace scenemap {
namespace {

inline bool maskBit(const sm_detections* d, int k, int i, int j) {
  if (i < 0 || j < 0 || i >= d->mask_w || j >= d->mask_h) return false;
  const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
  const size_t c = size_t(j) * d->mask_w + i;
  return (d->mask_bits[k * words + (c >> 5)] >> (c & 31)) & 1u;
}

double pct(std::vector<double>& v, double q) {
  const size_t k = std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

double dist3(const double* a, const double* b) {
  return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

struct Obs {
  int det;                        // 검출 번호
  double zmed;                    // 카메라 깊이 중앙값
  int cls;
  float score;
  double pos[3], ext[3], lo[3], hi[3];
  int n;
};

}  // namespace

void ObjectMap::event(double t, const MapObject& o, int kind) {
  ev_.push_back(ObjEvent{t, o.id, kind, {o.pos[0], o.pos[1], o.pos[2]}});
}

void ObjectMap::updateHands(double t, const double eef[2][3], const float grip[2], double yaw) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  for (int h = 0; h < 2; ++h) {
    const bool closed = grip[h] < p_.grip_closed;
    if (closed && !closed_[h]) {   // 잡기
      MapObject* best = nullptr;
      double bd = p_.grasp_r;
      for (auto& o : objs_) {
        if (!o.confirmed || o.held_by >= 0 || o.state == SM_GONE) continue;
        const double d = dist3(o.pos, eef[h]);
        if (d < bd) { bd = d; best = &o; }
      }
      if (best) {
        best->held_by = h;
        best->parent = 0;
        const double d[3] = {best->pos[0] - eef[h][0], best->pos[1] - eef[h][1], best->pos[2] - eef[h][2]};
        best->held_rel[0] = cy * d[0] + sy * d[1];
        best->held_rel[1] = -sy * d[0] + cy * d[1];
        best->held_rel[2] = d[2];
        for (int k = 0; k < 3; ++k) best->grasp_pos[k] = best->pos[k];
        best->state = SM_HELD;
        event(t, *best, 4);
      }
    } else if (!closed && closed_[h]) {   // 놓기
      for (auto& o : objs_) {
        if (o.held_by != h) continue;
        o.held_by = -1;
        const bool mv = dist3(o.pos, o.grasp_pos) > p_.moved_d;
        o.moved = o.moved || mv;
        o.state = o.moved ? SM_MOVED : SM_SEEN;
        o.misses = 0;
        // 놓은 자리가 다른 물체 위·안이면 붙이기(그 물체와 함께 움직임)
        const MapObject* best = nullptr;
        double bv = 1e18;
        for (const auto& p : objs_) {
          if (&p == &o || !p.confirmed || p.state == SM_GONE) continue;
          if (o.pos[0] < p.lo[0] - 0.1 || o.pos[0] > p.hi[0] + 0.1 || o.pos[1] < p.lo[1] - 0.1 || o.pos[1] > p.hi[1] + 0.1) continue;
          if (o.pos[2] < p.lo[2] - 0.05) continue;                  // 받침은 놓은 점 아래에서 시작
          const double v = o.pos[2] - std::min(p.hi[2], o.pos[2]);   // 떨어질 높이: 가장 높은 받침(바로 아래)
          if (v < bv) { bv = v; best = &p; }
        }
        if (best) {
          o.parent = best->id;
          for (int k = 0; k < 3; ++k) o.parent_rel[k] = o.pos[k] - best->pos[k];
        }
        event(t, o, 5);
      }
    }
    closed_[h] = closed;
  }
  for (auto& o : objs_)
    if (o.held_by >= 0)
      for (int k = 0; k < 3; ++k) {
        const double* r = o.held_rel;
        const double rel = k == 0 ? cy * r[0] - sy * r[1] : (k == 1 ? sy * r[0] + cy * r[1] : r[2]);
        const double np = eef[o.held_by][k] + rel, dk = np - o.pos[k];
        o.pos[k] = np;
        o.lo[k] += dk;
        o.hi[k] += dk;
      }
  // 붙은 물체는 받침을 따라간다(받침이 움직였으면 옮겨짐)
  for (auto& o : objs_) {
    if (!o.parent) continue;
    const MapObject* p = nullptr;
    for (const auto& q : objs_)
      if (q.id == o.parent) { p = &q; break; }
    if (!p) { o.parent = 0; continue; }
    for (int k = 0; k < 3; ++k) {
      const double np = p->pos[k] + o.parent_rel[k], dk = np - o.pos[k];
      if (std::fabs(dk) > 1e-9) o.moved = true;
      o.pos[k] = np;
      o.lo[k] += dk;
      o.hi[k] += dk;
    }
    if (o.moved && o.state == SM_SEEN) o.state = SM_MOVED;
  }
}

void ObjectMap::update(const ObjFrame& f) {
  updateHands(f.stamp, f.eef, f.grip, f.base_yaw);
  const double* T = f.T_mc;
  auto depthAt = [&](int u, int v) -> float {
    if (u < 0 || v < 0 || u >= f.w || v >= f.h) return 0.f;
    const size_t i = size_t(v) * f.w + u;
    return f.depth_m ? f.depth_m[i] : f.depth_mm[i] * 1e-3f;
  };
  // 1. 검출 → 관측
  std::vector<Obs> obs;
  const sm_detections* D = f.dets;
  assoc_.assign(D && D->n > 0 ? D->n : 0, DetAssoc{});
  const int st = std::max(1, p_.step);
  if (D && D->n > 0) {
    const float sxu = D->img_w > 0 ? float(f.w) / D->img_w : 1.f;   // 검출 영상 화소 ↔ 깊이 화소(크기가 다르면)
    const float syv = D->img_h > 0 ? float(f.h) / D->img_h : 1.f;
    std::vector<double> X, Y, Z, ZC;
    for (int k = 0; k < D->n; ++k) {
      X.clear(); Y.clear(); Z.clear(); ZC.clear();
      // 상자 안만 훑는다(검출 영상 화소 → 깊이 화소)
      const float* b = D->box + 4 * k;
      const int u0 = std::max(0, int(b[0] * sxu) - 1), u1 = std::min(f.w - 1, int(b[2] * sxu) + 1);
      const int v0 = std::max(0, int(b[1] * syv) - 1), v1 = std::min(f.h - 1, int(b[3] * syv) + 1);
      int near_hand = 0;
      for (int v = v0; v <= v1; v += st)
        for (int u = u0; u <= u1; u += st) {
          // 깊이 화소 중심 → 검출 영상 화소 → 마스크 칸, 1 칸 깎기(네 이웃도 마스크)
          const float xi = (u + 0.5f) / sxu, yi = (v + 0.5f) / syv;
          const int i = int(std::floor((xi - D->mask_ox) / D->mask_sx)), j = int(std::floor((yi - D->mask_oy) / D->mask_sy));
          if (!maskBit(D, k, i, j) || !maskBit(D, k, i - 1, j) || !maskBit(D, k, i + 1, j) || !maskBit(D, k, i, j - 1) ||
              !maskBit(D, k, i, j + 1))
            continue;
          const float z = depthAt(u, v);
          if (!(z > p_.zmin && z < p_.zmax)) continue;
          const double xc = (u - f.cx) / f.fx * z, yc = (v - f.cy) / f.fy * z;
          const double px = T[0] * xc + T[1] * yc + T[2] * z + T[3];
          const double py = T[4] * xc + T[5] * yc + T[6] * z + T[7];
          const double pz = T[8] * xc + T[9] * yc + T[10] * z + T[11];
          const double pp[3] = {px, py, pz};
          if (dist3(pp, f.eef[0]) < p_.hand_r || dist3(pp, f.eef[1]) < p_.hand_r) ++near_hand;
          X.push_back(px); Y.push_back(py); Z.push_back(pz); ZC.push_back(z);
        }
      const int np = int(X.size());
      if (np < p_.min_points) continue;
      if (near_hand >= p_.hand_frac * np) continue;   // 손에 든 것
      Obs o;
      o.det = k;
      o.zmed = pct(ZC, 0.5);
      o.cls = D->cls[k];
      o.score = D->score ? D->score[k] : 1.f;
      o.n = np;
      std::vector<double>* ax[3] = {&X, &Y, &Z};
      for (int a = 0; a < 3; ++a) {
        o.pos[a] = pct(*ax[a], 0.5);
        const double lo = pct(*ax[a], 0.1), hi = pct(*ax[a], 0.9);
        o.ext[a] = hi - lo;
        o.lo[a] = lo;
        o.hi[a] = hi;
      }
      obs.push_back(o);
    }
  }
  // 2. 같은 물체: 같은 이름 번호끼리 가까운 쌍부터 1:1
  std::vector<std::tuple<double, int, int>> pairs;
  for (int a = 0; a < int(obs.size()); ++a)
    for (int b = 0; b < int(objs_.size()); ++b) {
      const MapObject& m = objs_[b];
      if (m.cls != obs[a].cls || m.held_by >= 0) continue;
      const double e = std::max({obs[a].ext[0], obs[a].ext[1], obs[a].ext[2], m.ext[0], m.ext[1], m.ext[2]});
      const double thr = std::max(p_.da_min, p_.da_k * e);
      const double d = dist3(obs[a].pos, m.pos);
      double gap2 = 0;
      for (int k = 0; k < 3; ++k) {
        const double gk = std::max({0.0, obs[a].lo[k] - m.hi[k], m.lo[k] - obs[a].hi[k]});
        gap2 += gk * gk;
      }
      const double gap = std::sqrt(gap2);
      if (d < thr || gap < p_.da_gap) pairs.emplace_back(gap + 1e-3 * d, a, b);
    }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> obs_to(obs.size(), -1), obj_hit(objs_.size(), 0);
  for (auto& [d, a, b] : pairs) {
    if (obs_to[a] >= 0 || obj_hit[b]) continue;
    obs_to[a] = b;
    obj_hit[b] = 1;
  }
  // 3. 갱신
  for (int a = 0; a < int(obs.size()); ++a) {
    const Obs& o = obs[a];
    DetAssoc& as = assoc_[o.det];
    as.n_valid = o.n;
    as.area_px = float(o.n) * st * st;
    as.depth_med = float(o.zmed);
    if (obs_to[a] >= 0) {
      MapObject& m = objs_[obs_to[a]];
      as.obj_id = m.id;
      if (m.parent && dist3(m.pos, o.pos) > 0.3) m.parent = 0;
      if (m.last_kf != f.stamp) ++m.n_obs;
      m.last_kf = f.stamp;
      const double w = std::min<double>(m.n_obs, 20);
      const bool big = std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], o.ext[0], o.ext[1]}) > p_.big;
      for (int k = 0; k < 3; ++k) {
        if (big) {
          m.lo[k] = std::min(m.lo[k], o.lo[k]);
          m.hi[k] = std::max(m.hi[k], o.hi[k]);
          m.pos[k] = 0.5 * (m.lo[k] + m.hi[k]);
          m.ext[k] = m.hi[k] - m.lo[k];
        } else {
          m.pos[k] = (m.pos[k] * (w - 1) + o.pos[k]) / w;
          m.ext[k] = (m.ext[k] * (w - 1) + o.ext[k]) / w;
          m.lo[k] = (m.lo[k] * (w - 1) + o.lo[k]) / w;
          m.hi[k] = (m.hi[k] * (w - 1) + o.hi[k]) / w;
        }
      }
      m.score = std::max(m.score, o.score);
      m.last_seen = f.stamp;
      m.misses = 0;
      if (m.state == SM_GONE) {
        m.state = m.moved ? SM_MOVED : SM_SEEN;
        event(f.stamp, m, 6);
      }
      if (!m.confirmed && int(m.n_obs) >= p_.confirm) {
        m.confirmed = true;
        event(f.stamp, m, 1);
      }
      continue;
    }
    // 안 맞은 관측: 같은 이름의 확정 물체가 '사라짐'이면 그것이 옮겨진 것으로 잇는다(Khronos 식 이력)
    MapObject* moved_from = nullptr;
    double best = 1e9;
    for (auto& m : objs_) {
      if (m.cls != o.cls || !m.confirmed || m.held_by >= 0 || obj_hit[&m - objs_.data()]) continue;
      if (m.state != SM_GONE) continue;   // 사라짐으로 판정된 것만(같은 이름이 새로 하나 더 생긴 것과 헷갈리지 않게)
      const double d = dist3(o.pos, m.pos);
      if (d < best) { best = d; moved_from = &m; }
    }
    if (moved_from) {
      MapObject& m = *moved_from;
      as.obj_id = m.id;
      for (int k = 0; k < 3; ++k) { m.pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k]; }
      m.moved = true;
      m.state = SM_MOVED;
      m.misses = 0;
      m.last_seen = f.stamp;
      m.last_kf = f.stamp;
      ++m.n_obs;
      obj_hit[moved_from - objs_.data()] = 1;
      event(f.stamp, m, 2);
      continue;
    }
    MapObject m;
    m.id = next_id_++;
    m.cls = o.cls;
    for (int k = 0; k < 3; ++k) { m.pos[k] = m.first_pos[k] = o.pos[k]; m.ext[k] = o.ext[k]; m.lo[k] = o.lo[k]; m.hi[k] = o.hi[k]; }
    m.n_obs = 1;
    m.first_seen = m.last_seen = m.last_kf = f.stamp;
    m.score = o.score;
    m.confirmed = p_.confirm <= 1;
    as.obj_id = m.id;
    objs_.push_back(m);
    obj_hit.push_back(1);
    event(f.stamp, objs_.back(), 0);
  }
  // 4. 부재 확인(확정·안 든 것·이번에 안 맞은 것)
  if (f.depth_m || f.depth_mm) {
    for (size_t b = 0; b < objs_.size(); ++b) {
      MapObject& m = objs_[b];
      if (!m.confirmed || m.held_by >= 0 || obj_hit[b] || m.state == SM_GONE) continue;
      if (m.parent) continue;   // 통 안에 넣은 것은 안 보여도 그대로 있다고 본다
      // 큰 가구(한 변 > big)는 사라짐 판정을 하지 않는다: 부분만 보이고 중심 한 점으로 가림을 판단하기 어렵고, 과제 중 없어지지 않는다
      if (std::max({m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]}) > p_.big) continue;
      // 손 가까이는 판단하지 않는다: 손에 든 관측은 거르므로(위) 잡으러 다가가는 동안 '안 보임'으로 셈하면 안 됨
      if (dist3(m.pos, f.eef[0]) < p_.hand_r + 0.1 || dist3(m.pos, f.eef[1]) < p_.hand_r + 0.1) continue;
      // map → 카메라
      const double d[3] = {m.pos[0] - T[3], m.pos[1] - T[7], m.pos[2] - T[11]};
      const double xc = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
      const double yc = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
      const double zc = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
      if (zc < 0.3 || zc > p_.zmax) continue;
      const int u = int(f.fx * xc / zc + f.cx), v = int(f.fy * yc / zc + f.cy);
      const double size_px = f.fx * std::max({m.ext[0], m.ext[1], m.ext[2]}) / zc;
      if (size_px < p_.min_px || u < 2 || v < 2 || u >= f.w - 2 || v >= f.h - 2) continue;
      // 3×3 깊이 중앙값이 물체보다 occl 넘게 가까우면 가려진 것
      float ds[9];
      int nd = 0;
      for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du) {
          const float z = depthAt(u + du, v + dv);
          if (z > 0) ds[nd++] = z;
        }
      if (nd < 5) continue;
      std::nth_element(ds, ds + nd / 2, ds + nd);
      if (ds[nd / 2] < zc - p_.occl) continue;
      if (++m.misses >= p_.gone_misses) {
        m.state = SM_GONE;
        event(f.stamp, m, 3);
      }
    }
  }
  // 5. 오래된 후보 버리기
  objs_.erase(std::remove_if(objs_.begin(), objs_.end(),
                             [&](const MapObject& m) { return !m.confirmed && f.stamp - m.last_seen > p_.prune_s; }),
              objs_.end());
}

}  // namespace scenemap
