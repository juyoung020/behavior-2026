// solver 모듈 공통 자료형 (손으로 짬). PhysX 5.6.1 의 TGS 풀이 자료를 같은 뜻·같은 필드로 옮긴 것.
// 포인터 대신 번호(index)·바이트 위치(offset)로 잇는다 → 같은 코드가 호스트(층 1)와 GPU(층 2)에서 돈다.
// 원본: physx/include/solver/PxSolverDefs.h (PxTGSSolverBody*, PxSolverConstraintDesc, PxConstraintBatchHeader),
//       include/PxContact.h (PxContactPatch, PxContact), include/geomutils/PxContactPoint.h,
//       source/lowleveldynamics/src/DyFrictionPatch.h, DyCorrelationBuffer.h, DySolverConstraint1DStep.h,
//       DyTGSContactPrepBlock.cpp:89-152 (블록 헤더·점·마찰), DySolverConstraintTypes.h
#pragma once
#include <cstdint>

#include "../common/pmath.h"
#include "saos.h"

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

// ---------------- 단일 경로 제약 자료 (DySolverConstraint1DStep.h:47-117)
struct alignas(16) SolverContactHeaderStep {
  uint8_t type, flags, numNormalConstr, numFrictionConstr;
  float angDom0, angDom1, invMass0;
  V4 staticFrictionX_dynamicFrictionY_dominance0Z_dominance1W;
  V3 normal;
  float maxPenBias, invMass1, minNormalForce;
  uint32_t broken;
  uint32_t frictionBrokenWriteback;  // 마찰 패치 번호(현 스텝 arena) 또는 NONE (원본은 바이트 포인터)
  uint32_t pad[2];
};
struct alignas(16) SolverContactPointStep {
  V3 raXnI;
  float separation;
  V3 rbXnI;
  float velMultiplier, targetVelocity, biasCoefficient, recipResponse, maxImpulse;
};
struct alignas(16) SolverContactFrictionStep {
  V4 normalXYZ_ErrorW;
  V4 raXnI_targetVelW;
  V4 rbXnI_velMultiplierW;
  float biasScale, appliedForce, frictionScale;
  uint32_t pad;
};

// ---------------- 4개 묶음 제약 자료 (DyTGSContactPrepBlock.cpp:89-152)
struct alignas(16) SolverContactHeaderStepBlock {
  enum { eHAS_MAX_IMPULSE = 1 << 0, eHAS_TARGET_VELOCITY = 1 << 1 };
  uint8_t type, numNormalConstr, numFrictionConstr, flag;
  uint8_t flags[4];
  uint8_t numNormalConstrs[4];
  uint8_t numFrictionConstrs[4];
  V4 staticFriction, dynamicFriction;
  V4 invMass0D0, invMass1D1, angDom0, angDom1;
  V4 normalX, normalY, normalZ;
  V4 maxPenBias;
  BV broken;
  uint32_t frictionBrokenWriteback[4];
};
struct alignas(16) SolverContactPointStepBlock {
  V4 raXnI[3];
  V4 rbXnI[3];
  V4 separation, velMultiplier, targetVelocity, biasCoefficient, recipResponse;
};
struct alignas(16) SolverContactFrictionStepBlock {
  V4 normal[3];
  V4 raXnI[3];
  V4 rbXnI[3];
  V4 error, velMultiplier, targetVel, biasCoefficient;
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

}  // namespace sv
}  // namespace eng
