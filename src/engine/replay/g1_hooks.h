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
namespace eng { namespace art { struct Articulation; } }
eng::art::Articulation* g1_art_twin(const void* fa);  // simulate 앞에 옮겨 담은 쌍둥이 (없거나 실패면 null)
bool g1_art_link_of_rb(const void* rb, const void** fa, uint32_t* ll);
bool g1_art_diff(const void* fa, const eng::art::Articulation& e, size_t* firstJ, size_t* nFields, float* pxv, float* ev);
const char* g1_art_name(const void* fa);
// 엔진 장면 파일의 모양 순서대로 PxsShapeCore* (g1_dump.cpp, 뜨기와 같은 열거)
std::vector<const void*> g1_shape_cores(physx::PxScene* scene);
namespace eng { namespace scene { struct IslandMgrState; } }
// 섬 관리자 그림자·넘겨받기 (g1_islands.cpp, G1_ISLANDS)
void g1_islands_before(physx::PxScene* scene, uint64_t sim);
void g1_islands_after(physx::PxScene* scene, uint64_t sim);
void g1_islands_task(const char* name);  // 가로채기 디스패처가 작업을 돌리기 직전에
void g1_islands_report();
bool g1_islands_capture(physx::PxScene* scene, eng::scene::IslandMgrState& s, uint32_t (*objectId)(const void*, uint32_t, void*),
                        uint32_t (*edgeObject)(const void*, void*), void* user);
// 넓은 단계 그림자 (g1_bp.cpp, G1_BP)
void g1_bp_before(physx::PxScene* scene, uint64_t sim);
void g1_bp_task(const char* name);
void g1_bp_report();
namespace eng { namespace scene { struct BpLog; } }
const eng::scene::BpLog* g1_bp_log();
// 쌍 관리층 그림자 (g1_pairs.cpp, G1_PAIRS)
void g1_pairs_before(physx::PxScene* scene, uint64_t sim);
void g1_pairs_after(physx::PxScene* scene, uint64_t sim);
void g1_pairs_report();
namespace eng { namespace scene { struct PairsLog; } }
const eng::scene::PairsLog* g1_pairs_log();
void g1_dump_before(physx::PxScene* scene, uint64_t sim);  // g1_dump.cpp (G1_DUMP_AT·G1_DUMP_OUT)
