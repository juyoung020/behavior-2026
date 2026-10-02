// scenemap — 2D 점유 격자(로그 오즈). 필요할 때 넓어진다.
//
// 칸마다: 로그 오즈 L, 본 적 있는지, 맞은 점 합(점-선 맞추기용 칸 평균). L 이 0 이하로 떨어지면 점 합을 비운다
// (움직인 물체의 흔적이 다시 점유될 때 옛 평균이 섞이지 않게).
#pragma once
#include <cstdint>
#include <vector>

#include "scenemap/geom.hpp"
#include "scenemap/scan.hpp"

namespace scenemap {

struct GridParams {
  float res = 0.05f;
  float l_hit = 0.85f, l_miss = -0.4f, l_min = -4.f, l_max = 4.f;
};

class OccGrid {
 public:
  explicit OccGrid(const GridParams& p = {}) : p_(p) {}

  // 스캔을 자세 pose 로 넣는다(맞은 칸 +, 광선 위 칸 −, 한 스캔에서 칸마다 한 번, 맞음 우선)
  void insert(const Scan2& s, const Pose2& pose);

  // 칸 번호(전역 정수 좌표) ↔ 값
  bool inside(int ix, int iy) const { return ix >= x0_ && iy >= y0_ && ix < x0_ + w_ && iy < y0_ + h_; }
  float logOdds(int ix, int iy) const { return inside(ix, iy) ? L_[idx(ix, iy)] : 0.f; }
  bool seen(int ix, int iy) const { return inside(ix, iy) && seen_[idx(ix, iy)]; }
  // 칸 평균 점(맞은 점이 없으면 false)
  bool mean(int ix, int iy, float* mx, float* my, int* n = nullptr) const;
  // 칸에 쌓인 수평 법선(맞추기 점에서) — 방향이 서로 맞지 않으면(|합| < 0.5·개수) false
  bool normal(int ix, int iy, float* nx, float* ny) const;
  // 점유 확률 [pmin, pmax], 본 적 없는 칸 = pmin(Cartographer 와 같이 '맞을 곳 아님')
  float prob(int ix, int iy) const;

  int cellOf(double v) const { return int(std::floor(v / p_.res)); }
  float res() const { return p_.res; }
  int x0() const { return x0_; }
  int y0() const { return y0_; }
  int width() const { return w_; }
  int height() const { return h_; }
  // 계획기용: −1 모름, 0..100 점유 확률(%)
  std::vector<int8_t> export8() const;
  // 바뀐 영역: 지난 takeDirty 뒤 insert 가 고친 칸의 경계 상자(전역 칸 좌표, 끝 포함). 없으면 false. 부를 때마다 비움.
  bool takeDirty(int* ix0, int* iy0, int* ix1, int* iy1);
  uint64_t version() const { return version_; }

  float pmin = 0.1f, pmax = 0.9f;

 private:
  size_t idx(int ix, int iy) const { return size_t(iy - y0_) * w_ + (ix - x0_); }
  void ensure(int ix0, int iy0, int ix1, int iy1);
  void touch(int ix, int iy, bool hit, float hx, float hy, float nx = 0, float ny = 0);

  GridParams p_;
  int x0_ = 0, y0_ = 0, w_ = 0, h_ = 0;
  std::vector<float> L_;
  std::vector<uint8_t> seen_;
  std::vector<float> sx_, sy_, nx_, ny_;
  std::vector<uint16_t> cntn_;
  std::vector<uint16_t> cnt_;
  std::vector<uint32_t> stamp_;   // 이 스캔에서 이미 고친 칸
  uint32_t scan_id_ = 0;
  uint64_t version_ = 0;          // insert 마다 +1
  bool dirty_ = false;
  int dx0_ = 0, dy0_ = 0, dx1_ = 0, dy1_ = 0;
};

}  // namespace scenemap
