// joints 모듈 공통 타입: PhysX 5.6.1 과 바이트 배치까지 같은 구조체 (CPU·CUDA 공용).
// 배치를 같게 둔 이유: 시험에서 PhysX 가 만든 메모리(조인트 상수 블록, 풀이 제약 블록)와 memcmp 로 비교하고,
// 나중에 공식 기록(OVD)의 값을 그대로 부어 넣기 위해서다.
// 원본(태그 107.3-omni-and-physx-5.6.1, physx/ 기준)
//   D6 데이터   : source/physxextensions/src/ExtD6Joint.h:44, ExtJointData.h:38, include/extensions/PxJointLimit.h, PxD6Joint.h:237
//   1D 행       : include/PxConstraintDesc.h:194 (Px1DConstraint), :57 (플래그), :79 (solveHint)
//   풀이 몸체   : include/solver/PxSolverDefs.h:564-613 (PxTGSSolverBodyVel/TxInertia/Data)
//   풀이 제약   : source/lowleveldynamics/src/DySolverConstraint1DStep.h:128-265, DySolverConstraintTypes.h:61
//   되쓰기      : source/lowleveldynamics/src/DyConstraintWriteBack.h:42
#pragma once
#include <cstdint>

#include "../common/pmath.h"

namespace eng {
namespace jnt {

// ---- 열거값 (PxD6Joint.h)
enum D6Axis : uint32_t { AX_X = 0, AX_Y = 1, AX_Z = 2, AX_TWIST = 3, AX_SWING1 = 4, AX_SWING2 = 5 };
enum D6Motion : uint32_t { M_LOCKED = 0, M_LIMITED = 1, M_FREE = 2 };
enum D6DriveType : uint32_t { DR_X = 0, DR_Y = 1, DR_Z = 2, DR_SWING = 3, DR_TWIST = 4, DR_SLERP = 5, DR_SWING1 = 6, DR_SWING2 = 7 };
enum D6AngularDriveConfig : uint8_t { ADC_SWING_TWIST = 0, ADC_SLERP = 1, ADC_LEGACY = 2 };
enum : uint32_t { DRIVE_FLAG_ACCELERATION = 1, DRIVE_FLAG_OUTPUT_FORCE = 2 };  // PxD6JointDriveFlag

// PxConstraintFlag (include/PxConstraint.h:56)
enum : uint16_t {
  CF_BROKEN = 1 << 0, CF_COLLISION_ENABLED = 1 << 3, CF_VISUALIZATION = 1 << 4, CF_DRIVE_LIMITS_ARE_FORCES = 1 << 5,
  CF_IMPROVED_SLERP = 1 << 7, CF_DISABLE_PREPROCESSING = 1 << 8, CF_ENABLE_EXTENDED_LIMITS = 1 << 9,
  CF_GPU_COMPATIBLE = 1 << 10, CF_ALWAYS_UPDATE = 1 << 11, CF_DISABLE_CONSTRAINT = 1 << 12
};
// Px1DConstraintFlag
enum : uint16_t {
  RF_SPRING = 1 << 0, RF_ACCELERATION_SPRING = 1 << 1, RF_RESTITUTION = 1 << 2, RF_KEEPBIAS = 1 << 3,
  RF_OUTPUT_FORCE = 1 << 4, RF_HAS_DRIVE_LIMIT = 1 << 5, RF_ANGULAR_CONSTRAINT = 1 << 6
};
// PxConstraintSolveHint
enum : uint16_t {
  SH_NONE = 0, SH_ACCELERATION1 = 256, SH_SLERP_SPRING = 258, SH_ACCELERATION2 = 512, SH_ACCELERATION3 = 768,
  SH_ROTATIONAL_EQUALITY = 1024, SH_ROTATIONAL_INEQUALITY = 1025, SH_EQUALITY = 2048, SH_INEQUALITY = 2049
};
// Dy::SolverConstraintFlags (DySolverConstraintTypes.h:61), 제약 종류
enum : uint32_t {
  SC_OUTPUT_FORCE = 1 << 1, SC_KEEP_BIAS = 1 << 2, SC_ROT_EQ = 1 << 3, SC_ORTHO_TARGET = 1 << 4, SC_SPRING = 1 << 5,
  SC_INEQUALITY = 1 << 6, SC_ACCELERATION_SPRING = 1 << 7
};
enum : uint8_t { SC_TYPE_RB_1D = 2, SC_TYPE_EXT_1D = 4, SC_TYPE_BLOCK_1D = 8 };

static const uint32_t MAX_CONSTRAINT_ROWS = 20;           // DyConstraintPrep.h:55
static const uint32_t RIGID_BODY = 0xffff;                // PxSolverConstraintDesc::RIGID_BODY
static const float MAX_F32 = 3.40282346638528859812e+38F; // PX_MAX_F32
static const float PI = 3.141592653589793f;               // PxPi (float 반올림값)

// ---- 한계·드라이브 (PxJointLimit.h: 기반 클래스 PxJointLimitParameters 의 4개 뒤에 파생 멤버)
struct LinearLimit { float restitution, bounceThreshold, stiffness, damping, value; };
struct LinearLimitPair { float restitution, bounceThreshold, stiffness, damping, upper, lower; };
struct AngularLimitPair { float restitution, bounceThreshold, stiffness, damping, upper, lower; };
struct LimitCone { float restitution, bounceThreshold, stiffness, damping, yAngle, zAngle; };
struct LimitPyramid { float restitution, bounceThreshold, stiffness, damping, yAngleMin, yAngleMax, zAngleMin, zAngleMax; };
struct Drive { float stiffness, damping, forceLimit; uint32_t flags; };  // PxD6JointDrive (PxSpring + forceLimit + flags)

// PxJointLimitParameters::isSoft
template <class L> EHD bool isSoft(const L& l) { return l.damping > 0 || l.stiffness > 0; }

struct alignas(16) InvMassScale { float linear0, angular0, linear1, angular1; };  // PxConstraintInvMassScale
struct alignas(16) Tf32 { Q q; V3 p; float pad; };                                  // PxTransform32

// Ext::JointData + Ext::D6JointData (ExtD6Joint.h:44). 조인트 셰이더의 "상수 블록" 그 자체.
struct alignas(16) D6Data {
  InvMassScale invMassScale;
  Tf32 c2b[2];                 // 질량중심 자세^-1 * 조인트 틀 (정적 몸체는 전역 자세 * 조인트 틀)
  uint32_t motion[6];
  LinearLimit distanceLimit;
  LinearLimitPair linearLimitX, linearLimitY, linearLimitZ;
  AngularLimitPair twistLimit;
  LimitCone swingLimit;
  LimitPyramid pyramidSwingLimit;
  Drive drive[6];
  Tf drivePosition;
  V3 driveLinearVelocity, driveAngularVelocity;
  uint32_t locked, limited, driving;
  float distanceMinDist;
  bool mUseDistanceLimit, mUseNewLinearLimits, mUseConeLimit, mUsePyramidLimits;
  uint8_t angularDriveConfig;
};

// Px1DConstraint (PxConstraintDesc.h:194) — 셰이더가 채우는 제약 한 행
struct alignas(16) Row {
  V3 linear0; float geometricError;
  V3 angular0; float velocityTarget;
  V3 linear1; float minImpulse;
  V3 angular1; float maxImpulse;
  float mod0, mod1;            // spring.stiffness/damping 또는 bounce.restitution/velocityThreshold (union)
  uint16_t flags, solveHint;
  uint32_t pad;
};

// ---- TGS 풀이 몸체 (PxSolverDefs.h:564)
struct alignas(16) TgsBodyVel {
  V3 linearVelocity; uint16_t maxDynamicPartition, nbStaticInteractions;
  V3 angularVelocity; uint32_t partitionMask;
  V3 deltaAngDt; float maxAngVel;
  V3 deltaLinDt; uint16_t lockFlags; uint8_t isKinematic, pad;
};
struct alignas(16) TgsTxInertia {  // PxTGSSolverBodyTxInertia
  Q deltaBody2WorldQ;
  V3 body2WorldP;
  M33 sqrtInvInertia;
};
struct alignas(16) TgsBodyData {  // PxTGSSolverBodyData
  V3 originalLinearVelocity; float maxContactImpulse;
  V3 originalAngularVelocity; float penBiasClamp;
  float invMass; uint32_t nodeIndex; float reportThreshold; uint32_t pad;
};

// ---- 풀이용 1D 제약 블록 (DySolverConstraint1DStep.h)
struct alignas(16) V4f { float x, y, z, w; };  // PxVec4
struct alignas(16) Sc1DHeader {                // SolverConstraint1DHeaderStep (176 B)
  uint8_t type, count, dominance, breakable;
  float linBreakImpulse, angBreakImpulse, invMass0D0;
  V3 body0WorldOffset; float invMass1D1;
  V3 rAWorld; float linearInvMassScale0;
  V3 rBWorld; float angularInvMassScale0;
  float linearInvMassScale1, angularInvMassScale1;
  uint32_t pad[2];
  V4f angOrthoAxis0_recipResponseW[3];
  V4f angOrthoAxis1_Error[3];
};
struct alignas(16) Sc1DRow {                   // SolverConstraint1DStep (96 B)
  V3 lin0; float error;
  V3 lin1; float biasScale;
  V3 ang0; float velMultiplier;
  V3 ang1; float velTarget;
  float minImpulse, maxImpulse, appliedForce, maxBias;
  uint32_t flags; float recipResponse; float residualVelIter;
  uint32_t useAngularError_residualPosIter;    // union { PxU32 useAngularError(부호 비트); PxReal residualPosIter; }
};
struct alignas(16) SpatialV4 { float lin[4]; float ang[4]; };  // Cm::SpatialVectorV (Vec3V 두 개)
struct alignas(16) Sc1DRowExt : Sc1DRow { SpatialV4 deltaVA, deltaVB; };  // SolverConstraint1DExtStep (160 B)

// Dy::ConstraintWriteback
struct alignas(16) Writeback {
  V3 linearImpulse; uint32_t broken_residualPosIter;
  V3 angularImpulse; float residual;
};

// 풀이 제약 서술자 (PxSolverConstraintDesc 의 우리 판: 포인터 대신 판 안 번호)
struct ConstraintDesc {
  uint32_t bodyA, bodyB;          // 풀이 몸체 배열 번호 (PhysX tgsBodyA/B 와 bodyA/BDataIndex 를 같은 번호로 둔다)
  uint32_t linkIndexA, linkIndexB; // RIGID_BODY 또는 관절체 링크 번호
  uint32_t offset;                 // 제약 블록 바이트 위치 (UINT32_MAX = 행 없음)
  uint32_t length;                 // 바이트 길이 (16 배수)
  uint32_t writeback;              // 되쓰기 칸 번호
};

// 셰이더 출력 (D6JointSolverPrep 의 참조 인자들)
struct PrepOut {
  uint32_t numRows;
  V3 body0WorldOffset;
  InvMassScale invMassScale;
  V3 cA2w, cB2w;
};

// TGS 1D 준비 입력 (PxTGSSolverConstraintPrepDesc 의 필요한 부분)
struct PrepIn {
  Row* rows;
  uint32_t numRows;
  InvMassScale invMassScales;
  V3 body0WorldOffset, cA2w, cB2w;
  Tf bodyFrame0, bodyFrame1;                   // 몸체 질량중심 자세 (정적·세계 = 항등)
  const TgsBodyVel* body0; const TgsBodyVel* body1;
  const TgsTxInertia* txI0; const TgsTxInertia* txI1;
  const TgsBodyData* data0; const TgsBodyData* data1;
  uint32_t linkIndexA, linkIndexB;             // RIGID_BODY 또는 링크 번호
  float linBreakForce, angBreakForce, minResponseThreshold;
  bool disablePreprocessing, improvedSlerp, driveLimitsAreForces, extendedLimits, disableConstraint;
};

EHD uint32_t blockLength(uint32_t numRows, bool isExtended) {
  return uint32_t(sizeof(Sc1DHeader)) + (isExtended ? uint32_t(sizeof(Sc1DRowExt)) : uint32_t(sizeof(Sc1DRow))) * numRows;
}

// 비트 헬퍼
EHD uint32_t f2u(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
EHD float u2f(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }

}  // namespace jnt
}  // namespace eng
