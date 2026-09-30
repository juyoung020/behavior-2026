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
// Sc 층 입력 그림자 (g1_sc.cpp, G1_SC)
void g1_sc_before(physx::PxScene* scene, uint64_t sim);
void g1_sc_task(const char* name);
void g1_sc_report();
void g1_sc_after(physx::PxScene* scene, uint64_t sim);
// Sc 편집 그림자(G1_SC_EDIT)가 PhysX 의 넓은 단계·섬 호출을 받는다 (g1_bp.cpp·g1_islands.cpp 가로채기에서 넘김)
void g1_sc_note_bp(uint32_t type, uint32_t index, uint32_t group, uint32_t agg, uint32_t vt, uint32_t env, float cd, uint32_t result);
void g1_sc_note_island(int op, uint64_t node, int a, int b);
// contact 장면 단위 그림자 (g1_scene.cpp, G1_SCENE)
namespace eng { namespace scene { struct BpOp; struct PairsStep; } namespace contact { struct MaterialData; namespace sc { struct ScPairs; } } }
namespace physx { struct PxsCachedTransform; namespace Sc { class ShapeSim; } }
void g1_scene_bp_created(const eng::scene::BpLog& hdr);
void g1_scene_note_bpop(const eng::scene::BpOp& o);
void g1_scene_task(const char* name);
void g1_scene_before(physx::PxScene* scene, uint64_t sim);
void g1_scene_after(physx::PxScene* scene, uint64_t sim);
void g1_scene_report();
// 닫힌 고리 2단 Sc 입력 조각 (g1_sc.cpp 의 우리 Sc 장면)
namespace eng { struct Tf; namespace scene { struct ScScene; } }
bool g1_sc_loop_on();
namespace eng { namespace ig { struct IslandManager; } }
const eng::ig::IslandManager* g1_islands_ours();  // g1_islands.cpp: 우리 섬 관리 (넘겨받은 뒤)
void g1_sc_update_actor(const void* actorSim, const eng::Tf& b2w, const eng::Tf& b2a, bool frozen);
eng::scene::ScScene* g1_sc_scene();
const physx::PxActor* g1_sc_actor_px(int32_t h);
const void* g1_art_link_sim(const void* fa, uint32_t creationIdx);  // g1_art.cpp: 관절체 링크 -> Sc::ActorSim*
namespace eng { namespace sv { struct SolverCM; struct ContactPatchIn; struct ContactIn; } }
bool g1_scene_solver_input(uint32_t cmIndex, const eng::sv::SolverCM** m, const eng::sv::ContactPatchIn** patches, const eng::sv::ContactIn** contacts);
// 닫힌 고리 (2a) API 창 (g1_loop.cpp, G1_LOOP)
void g1_loop_before(physx::PxScene* scene, uint64_t sim);
void g1_loop_after(physx::PxScene* scene, uint64_t sim);
void g1_loop_report();
// 닫힌 고리 (2b) 지속 모드 (G1_LOOP_PERSIST=1): 우리 상태를 스텝 사이에 들고 가고 창의 API 호출만 넣는다
struct G1ArtOp {
  uint8_t type;  // 0 드라이브 목표, 1 드라이브 목표 속도, 2 wakeUp, 3 putToSleep
  uint8_t axis;
  uint32_t link;  // 생성 순서 번호 (PxArticulationLink::getLinkIndex)
  float v;
};
bool g1_loop_persist();                                        // 지속 모드인가
bool g1_loop_touched(const void* obj);                          // 이번 창에 옮기지 않은 API 로 건드린 객체 (PxArticulationReducedCoordinate* / PxRigidActor*)
void g1_loop_take_art_ops(const void* art, std::vector<G1ArtOp>& out);  // 이번 창에 옮긴 관절체 호출 (부른 순서)
namespace eng { namespace art { struct Articulation; } }
void g1_art_persist(const void* fa, const eng::art::Articulation& e, uint64_t sim);  // g1_art.cpp: 풀이 뒤 우리 관절체 상태를 다음 스텝으로
// 다른 그림자가 내주는 것
const physx::PxsCachedTransform* g1_contact_cache(size_t* n);   // g1_shadow: fetchCollision 뒤 변환 캐시
const eng::contact::MaterialData* g1_contact_mats(size_t* n);    // g1_shadow: 재질 표
float g1_contact_tol();                                          // g1_shadow: 길이 눈금
void g1_pairs_filters(eng::contact::sc::ScPairs& M);             // g1_pairs: omni 거르개·pair-found
const eng::scene::PairsStep* g1_pairs_step();                    // g1_pairs: 이번 스텝 입력(뒤 부분까지 채운 것)
physx::Sc::ShapeSim* g1_elem_sim(int32_t e);
const physx::PxActor* g1_pairs_actor(int32_t a);                // g1_pairs: 쌍 관리층 행위자 번호 -> PhysX 행위자                     // g1_pairs: 요소 번호 -> ShapeSim
void g1_dump_before(physx::PxScene* scene, uint64_t sim);
bool g1_dump_actors(physx::PxScene* scene, const std::vector<physx::PxActor*>& actors, uint64_t sim, const char* path);  // g1_dump.cpp: 새 행위자 틀 파일  // g1_dump.cpp (G1_DUMP_AT·G1_DUMP_OUT)
