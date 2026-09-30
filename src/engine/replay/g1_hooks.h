// G1 그림자 시험 번역 단위끼리 주고받는 것 (g1_shadow.cpp · g1_solver.cpp · g1_art.cpp). PhysX 내부 헤더 없이 쓸 수 있게 둔다.
#pragma once
#include <cstdint>
#include <vector>

namespace physx {
class PxScene;
}

// 이번 simulate 에서 "관절체 하나뿐이고 접촉·조인트 간선이 없는" 활성 섬 (g1_solver.cpp 의 풀이 직전 스냅샷이 채움)
struct G1ArtAlone {
  const void* fa;     // Dy::FeatherstoneArticulation*
  uint16_t iterWord;  // 그 섬이 든 PhysX 풀이 묶음의 반복 수 (강체·관절체 최댓값, 위치 = 낮은 8 비트)
};
struct G1StepInfo {
  bool valid = false;
  float dt = 0, gravity[3] = {0, 0, 0};
  std::vector<G1ArtAlone> artAlone;
};
const G1StepInfo& g1_step_info();  // g1_solver.cpp (G1_SOLVER 또는 G1_ART 로 디스패처가 켜졌을 때만 채워짐)

void g1_solver_before(physx::PxScene* scene, uint64_t sim);
void g1_solver_after(physx::PxScene* scene, uint64_t sim);
void g1_solver_report();
void g1_art_before(physx::PxScene* scene);
void g1_art_after(physx::PxScene* scene, uint64_t sim);
void g1_art_report();
// 엔진 장면 파일의 모양 순서대로 PxsShapeCore* (g1_dump.cpp, 뜨기와 같은 열거)
std::vector<const void*> g1_shape_cores(physx::PxScene* scene);
void g1_dump_before(physx::PxScene* scene, uint64_t sim);  // g1_dump.cpp (G1_DUMP_AT·G1_DUMP_OUT)
