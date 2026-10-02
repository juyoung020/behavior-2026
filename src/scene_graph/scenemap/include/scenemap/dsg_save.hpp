// scenemap 저장 — 로봇 기억을 파일로(sm_save_dsg). docs/scenemap_설계.md 3절 "save".
//   scene.json : Spark-DSG DynamicSceneGraph. OBJECTS 층에 확정 물체 하나 = 노드 하나(NodeSymbol 'O', id).
//                위치 = map xyz, 상자 = 위치 ± 크기/2, 이름 = 프롬프트 이름, 메타데이터 = 상태·관측 수·처음 위치·마지막 시각.
//   view.json  : 계획기·뷰어용 요약(자세, 물체, 최근 사건). 의존 없는 손 JSON.
//   map.pgm/.yaml : 2D 점유 격자(ROS map_server 형식: 254 빈칸, 0 점유, 205 모름).
//   objects/O<id>_rgb.png · O<id>_depth.png : 물체별 best view(8 비트 RGB · 16 비트 회색 mm). 모습이 바뀐 것(png_dirty)
//                이나 파일이 없는 것만 다시 쓴다. scene.json 노드 metadata.rgbd · view.json objects[].rgbd 가 이 경로를 가리킨다.
//   순서: PNG → scene.json → view.json(뷰어는 view.json 이 바뀌면 다시 읽으니 그때는 가리키는 파일이 다 있다).
// 모든 파일은 임시 이름으로 쓰고 rename 으로 바꾼다(읽는 쪽이 반쯤 쓴 파일을 보지 않게).
#pragma once
#include <string>
#include <vector>

#include "scenemap.h"
#include "scenemap/bestview.hpp"
#include "scenemap/objmap.hpp"

namespace scenemap {

struct SaveInput {
  double stamp = 0;
  double pose[3] = {0, 0, 0};      // x, y, yaw (map)
  const sm_object* objs = nullptr;
  int n_objs = 0;
  std::vector<ObjEvent> events;    // 최근 사건(오래된 것부터)
  double grid_res = 0.05, grid_ox = 0, grid_oy = 0;
  int grid_w = 0, grid_h = 0;
  const int8_t* cells = nullptr;   // −1 모름, 0..100 점유 %
  // 물체별(objs 와 같은 순서, 비어 있어도 됨)
  std::vector<BestViewPtr> views;  // best view(없으면 null)
  std::vector<uint8_t> png_dirty;  // 1: 지난 저장 뒤 모습이 바뀜
  std::vector<uint8_t> movable;    // 1: 옮길 수 있는 물체, 0: 가구·가전·붙박이(비어 있으면 모두 1)
  bool clean_objects = false;      // objects/ 에서 지금 물체가 아닌 O<id>_*.png 지우기(새 판·새 디렉터리)
};

struct SaveOut {
  int n_png = 0;                   // 이번에 쓴 PNG 파일 수
  double png_ms = 0;               // PNG 만들기·쓰기 시간
  std::vector<uint8_t> png_ok;     // objs[i]: objects/ 에 지금 모습 파일이 있음(이번에 썼거나 그대로)
};

// 0 = 성공. Spark-DSG 없이 빌드하면 scene.json 만 빠진다.
int saveScene(const SaveInput& in, const std::string& dir, SaveOut* out = nullptr);

}  // namespace scenemap
