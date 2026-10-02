// joints 시험 전용: 엔진 쪽 함수를 PhysX 헤더와 다른 번역 단위(jeng_api.cpp)에서 부르기 위한 얇은 선언.
// 이유: common/aos.h(PhysX aos 생성물 + sse 흉내)와 PhysX 헤더는 같은 이름(PX_ALIGN, _mm_* 등)을 쓰므로 한 파일에 같이 못 넣는다
// (리드의 tests/common/aos_diff_{eng,px}.cpp 와 같은 나눔). 여기 선언은 aos 없는 평범한 형만 쓴다.
#pragma once
#include <cstdint>

#include "core/joints/d6_joint.h"

namespace jeng {
using namespace eng::jnt;

void setupConstraintRows(Row* rows, uint32_t n);
PrepOut d6SolverPrep(Row* rows, const D6Data& data, const eng::Tf& bA2w, const eng::Tf& bB2w, bool useExtendedLimits);
uint32_t setupSolverConstraintStep(PrepIn& p, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt, float lengthScale,
                                   float biasCoefficient);
uint32_t prepareD6Step(const D6Data& data, uint16_t constraintFlags, float linBreakForce, float angBreakForce, float minResponseThreshold,
                       const eng::Tf& bodyFrame0, const eng::Tf& bodyFrame1, const TgsBodyVel& b0, const TgsBodyVel& b1, const TgsTxInertia& t0,
                       const TgsTxInertia& t1, const TgsBodyData& d0, const TgsBodyData& d1, Row* rows, uint8_t* blk, float stepDt, float simDt,
                       float recipStepDt, float recipSimDt, float lengthScale, float biasCoefficient, uint32_t* outLength);
void solve1DStep(uint8_t* blk, TgsBodyVel& b0, TgsBodyVel& b1, const TgsTxInertia& t0, const TgsTxInertia& t1, float elapsed, bool residual,
                 bool isPositionIteration);
void conclude1DStep(uint8_t* blk);
void writeBack1DStep(const uint8_t* blk, Writeback* wb);
// 4개 묶음 경로 (tgs_1d4.h). 머리 pad0[1..3] 은 원본이 안 쓰는 바이트
uint32_t blockLength4(uint32_t maxRows);
uint32_t setupSolverConstraintStep4(PrepIn* p, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt, uint32_t maxRows,
                                    float lengthScale, float biasCoefficient);
uint32_t prepareD6Step4(const D6Data* const data[4], const uint16_t flags[4], const float linBreak[4], const float angBreak[4], const float minResp[4],
                        const eng::Tf* frame0[4], const eng::Tf* frame1[4], const TgsBodyVel* b0[4], const TgsBodyVel* b1[4], const TgsTxInertia* t0[4],
                        const TgsTxInertia* t1[4], const TgsBodyData* d0[4], const TgsBodyData* d1[4], Row* rows, uint8_t* blk, float stepDt,
                        float simDt, float recipStepDt, float recipSimDt, float lengthScale, float biasCoefficient);
void solve1DStep4(uint8_t* blk, TgsBodyVel* const b[4][2], const TgsTxInertia* const t[4][2], float elapsed);
void conclude1DStep4(uint8_t* blk);
void writeBack1D4(const uint8_t* blk, Writeback* const wb[4]);
// 시험 틀 (tgs_harness.h)
void bodyCoreComputeUnconstrainedVelocity(const eng::V3& gravity, float dt, float linearDamping, float angularDamping, float accelScale,
                                          float maxLinearVelocitySq, float maxAngularVelocitySq, eng::V3& lin, eng::V3& ang, bool disableGravity);
void copyToSolverBodyDataStep(const eng::V3& lv, const eng::V3& av, float invMass, const eng::V3& invInertia, const eng::Tf& globalPose,
                              float maxDepenetrationVelocity, float maxContactImpulse, uint32_t nodeIndex, float reportThreshold, float maxAngVelSq,
                              uint32_t lockFlags, bool isKinematic, TgsBodyVel& v, TgsTxInertia& t, TgsBodyData& d, float dt, bool gyro);
void integrateCoreStep(TgsBodyVel& v, TgsTxInertia& t, float dt);
void worldBody(TgsBodyVel& v, TgsTxInertia& t, TgsBodyData& d);

}  // namespace jeng
