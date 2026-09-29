// solver 모듈 경계 자료형 (aos 없음 → PhysX 헤더와 같은 번역 단위에 넣어도 된다).
// 다른 모듈·통합 비계·시험이 solver 에 넘기는 입력(접촉 패치·점, 접촉 관리자, 섬 순서, 풀이 설정)과 판 작업 공간,
// 그리고 호스트 진입 함수(solver_host.cpp, 층 1)를 선언한다. 풀이 본체(aos 사용)는 tgs_solver.h.
// 필드 뜻·순서는 PhysX 5.6.1 과 같다 (원본 위치는 각 구조체 주석).
#pragma once
#include <cstdint>

#include "../common/body.h"
#include "../common/pmath.h"
#include "../joints/joint_types.h"  // D6Data·Writeback·Row (PhysX 배치, joints 모듈 소유, aos 없음)
#include "sv_hd.h"

namespace eng {
namespace sv {

constexpr uint32_t NONE = 0xffffffffu;
constexpr uint32_t RIGID_BODY = 0xffffu;  // PxSolverConstraintDesc::RIGID_BODY

// DySolverConstraintTypes.h
enum : uint8_t {
  SC_TYPE_NONE = 0,
  SC_TYPE_RB_CONTACT,
  SC_TYPE_RB_1D,
  SC_TYPE_EXT_CONTACT,
  SC_TYPE_EXT_1D,
  SC_TYPE_STATIC_CONTACT,
  SC_TYPE_BLOCK_RB_CONTACT,
  SC_TYPE_BLOCK_STATIC_RB_CONTACT,
  SC_TYPE_BLOCK_1D,
};

// PxMaterialFlag
enum : uint8_t {
  MAT_DISABLE_FRICTION = 1 << 0,
  MAT_DISABLE_STRONG_FRICTION = 1 << 1,
  MAT_COMPLIANT_ACCELERATION_SPRING = 1 << 4,  // PxMaterial.h:84
};
// PxContactPatch::PxContactPatchFlags
enum : uint16_t {
  PATCH_HAS_FACE_INDICES = 1,
  PATCH_MODIFIABLE = 2,
  PATCH_FORCE_NO_RESPONSE = 4,
  PATCH_HAS_MODIFIED_MASS_RATIOS = 8,
  PATCH_HAS_TARGET_VELOCITY = 16,
  PATCH_HAS_MAX_IMPULSE = 32,
  PATCH_REGENERATE_PATCHES = 64,
  PATCH_COMPRESSED_MODIFIED_CONTACT = 128,
};
// PxSolverContactDesc::BodyState
enum : uint8_t { BS_DYNAMIC = 1, BS_STATIC = 2, BS_KINEMATIC = 4, BS_ARTICULATION = 8 };

constexpr float PXC_SAME_NORMAL = 0.999f;  // PxcNpContactPrepShared.h:49

// ---------------- 풀이 몸체 (PxTGSSolverBodyVel / TxInertia / Data) — 필드 순서도 같게 둔다
struct alignas(16) SBodyVel {
  float lin[3];
  uint16_t maxDynamicPartition;   // 분할에서 PxSolverBody::maxSolverNormalProgress 로 씀
  uint16_t nbStaticInteractions;  // PxSolverBody::maxSolverFrictionProgress
  float ang[3];
  uint32_t partitionMask;         // PxSolverBody::solverProgress
  float deltaAngDt[3];
  float maxAngVel;
  float deltaLinDt[3];
  uint16_t lockFlags;
  uint8_t isKinematic;
  uint8_t pad;
};
struct SBodyTxI {
  Q deltaBody2WorldQ;
  V3 body2WorldP;
  M33 sqrtInvInertia;
};
struct alignas(16) SBodyData {
  V3 originalLinearVelocity;
  float maxContactImpulse;
  V3 originalAngularVelocity;
  float penBiasClamp;
  float invMass;
  uint32_t nodeIndex;
  float reportThreshold;
  uint32_t pad;
};

// ---------------- 접촉 입력 (좁은 단계 출력 = PxsContactManagerOutput 의 단순 스트림)
struct ContactPatchIn {  // PxContactPatch
  float invMassScale[4];  // mMassModification: linear0, angular0, linear1, angular1
  V3 normal;
  float restitution, dynamicFriction, staticFriction, damping;
  uint16_t startContactIndex;
  uint8_t nbContacts;
  uint8_t materialFlags;
  uint16_t internalFlags;
  uint16_t materialIndex0, materialIndex1;
};
struct ContactIn {  // PxContact
  V3 point;
  float separation;
};

// PxContactPoint (geomutils/PxContactPoint.h) — 풀이 준비용 펼친 접촉점
struct alignas(16) ContactPoint {
  V3 normal;
  float separation;
  V3 point;
  float maxImpulse;
  V3 targetVel;
  float staticFriction;
  uint8_t materialFlags;
  uint32_t internalFaceIndex1;
  float dynamicFriction;
  float restitution;
  float damping;
};
constexpr uint32_t MAX_CONTACTS = 256;  // PxContactBuffer::MAX_CONTACTS

// ---------------- 마찰 패치 (DyFrictionPatch.h) — 다음 스텝으로 이어지는 상태
struct FrictionPatch {
  uint8_t broken;
  uint8_t materialFlags;
  uint16_t anchorCount;
  float restitution, staticFriction, dynamicFriction;
  V3 body0Normal, body1Normal;
  V3 body0Anchors[2];
  V3 body1Anchors[2];
  Q relativeQuat;
};

struct Bounds3 { V3 minimum, maximum; };

// DyCorrelationBuffer.h
struct CorrelationBuffer {
  static constexpr uint32_t MAX_FRICTION_PATCHES = 32;
  static constexpr uint16_t LIST_END = 0xffff;
  struct ContactPatchData {
    Bounds3 patchBounds;
    uint32_t boundsPadding;
    float staticFriction, dynamicFriction, restitution;
    uint16_t start, next;
    uint8_t flags, count;
  };
  ContactPatchData contactPatches[MAX_CONTACTS];
  FrictionPatch frictionPatches[MAX_FRICTION_PATCHES];
  V3 frictionPatchWorldNormal[MAX_FRICTION_PATCHES];
  Bounds3 patchBounds[MAX_FRICTION_PATCHES];
  uint32_t frictionPatchContactCounts[MAX_FRICTION_PATCHES];
  uint32_t correlationListHeads[MAX_FRICTION_PATCHES + 1];
  uint16_t contactID[MAX_FRICTION_PATCHES][2];
  uint32_t contactPatchCount, frictionPatchCount;
};

// ---------------- 제약 기술자 (PxSolverConstraintDesc) / 묶음 머리 (PxConstraintBatchHeader)
struct SDesc {
  uint32_t bodyA, bodyB;                   // 풀이 몸체 풀 번호 (0 = 세계)
  uint32_t bodyADataIndex, bodyBDataIndex;
  uint32_t linkIndexA, linkIndexB;         // RIGID_BODY 또는 관절체 링크
  uint32_t constraint;                     // 제약 자료 바이트 위치(arena) 또는 NONE
  uint16_t constraintLengthOver16;
  uint8_t constraintType;                  // SC_TYPE_RB_CONTACT / SC_TYPE_RB_1D ...
  uint8_t pad;
  uint32_t source;                         // 접촉 관리자 번호 또는 1D 제약 번호
  uint32_t sortKey;                        // 1D 제약: Dy::Constraint::index (내림차순 정렬, DyTGSDynamics.cpp:916)
  uint16_t progressA, progressB;
};
struct BatchHeader {
  uint32_t startIndex;
  uint16_t stride;
  uint16_t constraintType;
};

// ---------------- 바이트 arena (제약 자료). 판마다 고정 용량, 넘치면 overflow 표시
struct ByteArena {
  uint8_t* base;
  uint32_t size, cap;
  uint32_t overflow;
};
SV_HD uint32_t arenaAlloc(ByteArena& a, uint32_t bytes) {  // 16 바이트 정렬
  const uint32_t off = (a.size + 15u) & ~15u;
  if (off + bytes > a.cap) {
    a.overflow = 1;
    return NONE;
  }
  a.size = off + bytes;
  return off;
}
template <class T>
SV_HD T* arenaPtr(const ByteArena& a, uint32_t off) { return reinterpret_cast<T*>(a.base + off); }

struct FrictionArena {
  FrictionPatch* data;
  uint32_t size, cap;
  uint32_t overflow;
};
SV_HD uint32_t frictionAlloc(FrictionArena& a, uint32_t n) {
  if (a.size + n > a.cap) {
    a.overflow = 1;
    return NONE;
  }
  const uint32_t off = a.size;
  a.size += n;
  return off;
}


// PxcNpWorkUnitFlag (PxcNpWorkUnit.h:51)
enum : uint32_t {
  NP_DISABLE_STRONG_FRICTION = 1 << 2,
  NP_ARTICULATION_BODY0 = 1 << 3,
  NP_ARTICULATION_BODY1 = 1 << 4,
  NP_DYNAMIC_BODY0 = 1 << 5,
  NP_DYNAMIC_BODY1 = 1 << 6,
  NP_FORCE_THRESHOLD = 1 << 9,
  NP_HAS_KINEMATIC_ACTOR = 1 << 11,
  NP_DOMINANCE_0 = 1 << 14,
  NP_DOMINANCE_1 = 1 << 15,
};

struct SolverParams {
  V3 gravity;
  float dt;
  bool enableStabilization;
  float bounceThreshold;          // Dy 문맥 값 = -bounceThresholdVelocity (ScScene.cpp:1002)
  float frictionOffsetThreshold;  // desc.frictionOffsetThreshold
  float correlationDistance;      // desc.frictionCorrelationDistance
  uint32_t solverBatchSize;       // desc.solverBatchSize (기본 128)
  uint32_t solverArticBatchSize;  // desc.solverArticulationBatchSize (기본 16)
  float lengthScale;              // PxTolerancesScale::length (1D 제약 준비, DynamicsTGSContext::mLengthScale)
};

// 접촉 관리자 (PxsContactManager + PxcNpWorkUnit 에서 풀이가 쓰는 것)
struct SolverCM {
  uint32_t body0, body1;  // Body 번호. body1 == NONE 이면 정적 (staticPose1 이 bodyFrame1)
  Tf staticPose1;
  uint32_t npFlags;       // PxcNpWorkUnitFlag
  float restDistance, torsionalPatchRadius, minTorsionalPatchRadius, offsetSlop;
  uint32_t patchStart, nbPatches, contactStart, nbContacts;  // 이번 스텝 좁은 단계 출력 (SolverBoard.patches / contacts)
  uint32_t frictionPtr, frictionCount;                       // 지난 스텝 마찰 패치 (frictionPrev arena)
};

// 이번 스텝 활성 섬 (IslandSim 의 활성 섬 순서, 섬 안 몸체·접촉 간선은 PhysX 사슬 순서)
struct IslandIn {
  uint32_t bodyStart, bodyCount;  // SolverBoard.islandBodies 안 범위
  uint32_t cmStart, cmCount;      // SolverBoard.islandCMs 안 범위
  uint32_t staticTouchCount;      // IslandSim::getIslandStaticTouchCount (잠 판정의 hasStaticTouch)
  uint32_t c1dStart, c1dCount;    // SolverBoard.islandC1Ds 안 범위 (섬의 제약 간선 사슬 순서, island.mFirstEdge[eCONSTRAINT])
};

// 1D 제약 (조인트) — Dy::Constraint (DyConstraint.h) 에서 풀이가 쓰는 것. 준비·풀이 식은 joints 모듈(core/joints/tgs_1d*.h).
// BEHAVIOR 의 조인트는 전부 D6 이라(12.3) 셰이더는 D6 하나다.
struct Constraint1DIn {
  uint32_t body0, body1;  // Body 번호, NONE = 정적(세계). 섬 간선의 node1/node2 와 같은 쪽
  uint32_t index;         // Dy::Constraint::index — 묶음 안 정렬 키(내림차순, DyTGSDynamics.cpp:916-922)
  uint32_t data;          // SolverBoard.jointData 번호 (D6 상수 블록 = Ext::D6JointData 바이트 그대로)
  uint32_t writeback;     // SolverBoard.writebacks 번호 (Dy::ConstraintWriteback, 풀린 스텝에만 쓴다)
  uint16_t flags;         // PxConstraintFlags (Dy::Constraint::flags)
  uint16_t pad;
  float linBreakForce, angBreakForce, minResponseThreshold;
};

struct SolverBoard {
  // 상태
  Body* bodies;
  uint32_t nbBodies;
  SolverCM* cms;
  uint32_t nbCMs;
  const ContactPatchIn* patches;
  const ContactIn* contacts;
  // 이번 스텝 섬
  const IslandIn* islands;
  uint32_t nbIslands;
  const uint32_t* islandBodies;
  const uint32_t* islandCMs;
  const uint32_t* activatedCMs;  // 이번 스텝 활성화된 접촉 간선 -> 마찰 패치 수 0 (DyTGSDynamics.cpp:548)
  uint32_t nbActivatedCMs;
  // Sc 층이 캐시 상태를 지운 접촉 관리자(PxcNpWorkUnit::clearCachedState, PxcNpWorkUnit.h:201) -> 마찰 패치 수 0.
  // 조인트가 끊기면 두 행위자 중 상호작용이 적은 쪽의 접촉 상호작용 전부가 거르기 상태 더러움 표시를 받고(ScConstraintBreakage.cpp:96-101),
  // 다음 스텝 Sc 층 갱신(ShapeInteraction::updateState -> resetManagerCachedState, ScShapeInteraction.cpp:872,194)에서 지워진다.
  const uint32_t* resetCMs;
  uint32_t nbResetCMs;
  // 1D 제약 (조인트)
  const Constraint1DIn* c1d;
  uint32_t nbC1D;
  const uint32_t* islandC1Ds;
  const jnt::D6Data* jointData;
  jnt::Writeback* writebacks;
  jnt::Row* rowScratch;  // jnt::MAX_CONSTRAINT_ROWS * 4 칸 (셰이더 행)
  // 작업 공간 (용량 고정)
  SBodyVel* vels;
  SBodyTxI* txI;
  SBodyData* datas;
  uint32_t poolCap;
  SDesc* descs;
  SDesc* ordered;
  SDesc* temp;
  BatchHeader* headers;
  uint32_t descCap;
  uint32_t* partitionCounts;
  uint32_t partitionCap;
  uint32_t* bodySolverIndex;  // 몸체 -> 풀 번호 (이번 스텝)
  ByteArena constraints;
  FrictionArena friction[2];
  uint32_t frictionCurIdx;
  CorrelationBuffer* corr;
  ContactPoint* contactBuffer;
  uint32_t error;  // 넘침 등 (0 = 정상)
  // 통계 (시험·보고용, 결과에 영향 없음)
  uint64_t statBatches, statBlock4, statSingle, statHeaders, statMaxPartitions, statFreeBatches;
  uint64_t stat1DBlock4, stat1DSingle, stat1DZeroRows;
  uint32_t statMaxArena, statMaxFriction, statMaxDescs;
};

enum : uint32_t {
  SV_ERR_POOL = 1,
  SV_ERR_DESC = 2,
  SV_ERR_PARTITION = 4,
  SV_ERR_ARENA = 8,
  SV_ERR_FRICTION = 16,
  SV_ERR_UNSUPPORTED = 32,
};

// ---------------- 호스트 진입 함수 (층 1, core/solver/solver_host.cpp). GPU 는 tgs_solver.h 의 같은 이름 함수를 직접 부른다.
// 한 스텝 순서: (Sc 층 깨움 반영) -> solverStepHost -> afterIntegrationHost -> deactivateBodiesHost(섬 관리가 재운 몸체)
void solverStepHost(SolverBoard& B, const SolverParams& prm);
void afterIntegrationHost(SolverBoard& B);
void deactivateBodiesHost(SolverBoard& B, const uint32_t* list, uint32_t n);

}  // namespace sv
}  // namespace eng
