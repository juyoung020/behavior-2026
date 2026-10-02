// scenemap ② objmap — 검출(마스크 + 이름 번호) + 깊이 + 자세 → 물체 3D 위치 → 같은 물체 판단 → 바뀐 부분만 갱신.
//
// docs/scenemap_설계.md 3.2. 규칙 요약
//   위치   : 마스크를 1 칸 깎은 안쪽 깊이 점(map)의 축별 중앙값, 크기 = 10~90 백분위 폭
//   거르기 : 점 min_points 미만 버림. 점의 hand_frac 이상이 팔 끝 hand_r 안이면(손에 든 것) 버림
//   같은 것: 같은 이름 번호끼리만. 중심 거리 < max(da_min, da_k·큰 쪽 크기) 이거나 map 축 상자 사이 틈 < da_gap.
//            틈(겹치면 0)·중심 거리 순으로 1:1(탐욕). 큰 가구(한 변 > big)는 상자를 합집합으로 키우고 위치 = 상자 중심
//   확정   : 서로 다른 keyframe 에서 confirm 번 보이면 물체. 후보가 prune_s 동안 다시 안 보이면 버림
//   옮겨짐 : 확정 물체가 moved_d 넘게 떨어진 자리에서 보이면 그 자리로 옮기고 이력을 남김
//   사라짐 : (작은 물체만, 한 변 ≤ big) 시야 안·가림 없음(깊이가 물체보다 occl 이상 가깝지 않음)·팔 끝 hand_r+0.1 밖인데
//            안 보이면 +1, gone_misses 번 연속이면 사라짐. 통 안에 붙은 것은 안 봄
//   들기   : 그리퍼가 닫히는 순간 팔 끝 grasp_r 안 가장 가까운 확정 물체를 든 것으로 — 드는 동안 팔 끝을 따라가고,
//            열리는 순간 그 자리에 놓는다(moved_d 넘게 옮겼으면 옮겨짐). 놓은 점 아래에 xy 가 겹치는(0.1 m 여유) 다른 물체
//            상자가 있으면 그중 윗면이 가장 높은 것(떨어져 닿을 받침)에 붙인다 — 들고 있는 쓰레기통에 넣은 캔이 통을 따라가게
#pragma once
#include <cstdint>
#include <vector>

#include "scenemap.h"

namespace scenemap {

struct ObjParams {
  int min_points = 20;
  float zmin = 0.15f, zmax = 5.0f;
  float hand_r = 0.40f, hand_frac = 0.5f;
  double da_min = 0.30, da_k = 0.5;
  double da_gap = 0.10;           // 상자끼리 이 안으로 붙어 있으면 같은 것(부분만 보이는 큰 가구)
  double big = 0.5;               // 상자 한 변이 이보다 크면 합집합으로 키움(작은 물체는 평균)
  int confirm = 2;
  double prune_s = 10.0;
  double moved_d = 0.15;
  int gone_misses = 3;
  double occl = 0.10;
  int min_px = 6;                 // 시야 안 판정: 물체가 이 화소보다 작게 보이면 부재 증거로 안 씀
  double grasp_r = 0.25;
  float grip_closed = 0.09f;      // 손가락 합이 이보다 작으면 닫힘(열림 0.1)
  int step = 1;                   // 깊이 화소 간격
};

struct ObjEvent {
  double t;
  uint32_t id;
  int kind;                       // 0 새 후보, 1 확정, 2 옮겨짐, 3 사라짐, 4 들기, 5 놓기, 6 다시 보임
  double pos[3];
};

struct MapObject {
  uint32_t id = 0;
  int cls = -1;
  double pos[3] = {0, 0, 0}, ext[3] = {0, 0, 0}, first_pos[3] = {0, 0, 0};
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // map 축 상자(관측 10~90 백분위). 큰 물체는 합집합으로 자란다
  uint32_t n_obs = 0;
  double first_seen = 0, last_seen = 0;
  int32_t state = SM_SEEN;
  bool confirmed = false;
  bool moved = false;
  int misses = 0;
  int held_by = -1;               // 0 왼손, 1 오른손
  double held_rel[3] = {0, 0, 0}, grasp_pos[3] = {0, 0, 0};   // held_rel: 팔 끝 → 물체, 베이스 축(로봇이 돌면 같이 돔)
  uint32_t parent = 0;            // 놓을 때 다른 물체(들고 있는 통·탁자) 위·안이면 그 물체에 붙어 같이 움직임
  double parent_rel[3] = {0, 0, 0};
  float score = 0;
  double last_kf = -1;
};

// 한 keyframe 의 검출 k → 물체(update 가 채움, lastAssoc()). obj_id 0 = 물체에 안 붙음(점 부족·손에 든 것 등)
struct DetAssoc {
  uint32_t obj_id = 0;
  int n_valid = 0;                // 마스크 안 유효 깊이 점(step 간격 표본)
  float area_px = 0;              // 유효 마스크 넓이(깊이 화소) = n_valid·step²
  float depth_med = 0;            // 마스크 안 깊이 중앙값 m(카메라 z)
};

// 한 keyframe 입력
struct ObjFrame {
  double stamp = 0;
  int w = 0, h = 0;
  const float* depth_m = nullptr;     // 또는
  const uint16_t* depth_mm = nullptr;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  double T_mc[12] = {0};              // map ← 카메라 광학(행 우선 3×4)
  const sm_detections* dets = nullptr;
  double eef[2][3] = {{0}};           // map 기준 팔 끝
  float grip[2] = {0.1f, 0.1f};       // 손가락 합
  double base_yaw = 0;                // map 기준 베이스 yaw
};

class ObjectMap {
 public:
  explicit ObjectMap(const ObjParams& p = {}) : p_(p) {}
  void update(const ObjFrame& f);
  // 팔 끝·그리퍼만 바뀐 스텝(영상 없음)에도 들고 있는 물체를 따라가게
  void updateHands(double stamp, const double eef[2][3], const float grip[2], double base_yaw);
  const std::vector<MapObject>& objects() const { return objs_; }
  const std::vector<ObjEvent>& events() const { return ev_; }
  // 마지막 update 의 검출별 짝(크기 = dets->n, 검출 순서)
  const std::vector<DetAssoc>& lastAssoc() const { return assoc_; }

 private:
  void event(double t, const MapObject& o, int kind);
  ObjParams p_;
  std::vector<MapObject> objs_;
  std::vector<ObjEvent> ev_;
  std::vector<DetAssoc> assoc_;
  uint32_t next_id_ = 1;
  bool closed_[2] = {false, false};
};

}  // namespace scenemap
