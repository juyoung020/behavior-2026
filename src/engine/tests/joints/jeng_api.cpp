// joints 시험 전용: 엔진 쪽 번역 단위 (common/aos.h 포함, PhysX 헤더 없음). 선언은 jeng_api.h.
#include "jeng_api.h"

#include "tgs_harness.h"

namespace jeng {
namespace H = eng::jtest;

void setupConstraintRows(Row* rows, uint32_t n) { eng::jnt::setupConstraintRows(rows, n); }
PrepOut d6SolverPrep(Row* rows, const D6Data& data, const eng::Tf& bA2w, const eng::Tf& bB2w, bool ext) {
  return eng::jnt::d6SolverPrep(rows, data, bA2w, bB2w, ext);
}
uint32_t setupSolverConstraintStep(PrepIn& p, uint8_t* blk, float stepDt, float simDt, float recipStepDt, float recipSimDt, float lengthScale,
                                   float biasCoefficient) {
  return eng::jnt::setupSolverConstraintStep(p, blk, stepDt, simDt, recipStepDt, recipSimDt, lengthScale, biasCoefficient, NoArt(), NoArt());
}
uint32_t prepareD6Step(const D6Data& data, uint16_t flags, float lb, float ab, float mrt, const eng::Tf& f0, const eng::Tf& f1, const TgsBodyVel& b0,
                       const TgsBodyVel& b1, const TgsTxInertia& t0, const TgsTxInertia& t1, const TgsBodyData& d0, const TgsBodyData& d1, Row* rows,
                       uint8_t* blk, float stepDt, float simDt, float rsd, float rsim, float ls, float bc, uint32_t* outLength) {
  return eng::jnt::prepareD6Step(data, flags, lb, ab, mrt, f0, f1, b0, b1, t0, t1, d0, d1, RIGID_BODY, RIGID_BODY, rows, blk, stepDt, simDt, rsd, rsim,
                                 ls, bc, NoArt(), NoArt(), outLength);
}
void solve1DStep(uint8_t* blk, TgsBodyVel& b0, TgsBodyVel& b1, const TgsTxInertia& t0, const TgsTxInertia& t1, float elapsed, bool residual,
                 bool isPos) {
  eng::jnt::solve1DStep(blk, b0, b1, t0, t1, elapsed, residual, isPos);
}
void conclude1DStep(uint8_t* blk) { eng::jnt::conclude1DStep(blk); }
void writeBack1DStep(const uint8_t* blk, Writeback* wb) { eng::jnt::writeBack1DStep(blk, wb); }
void bodyCoreComputeUnconstrainedVelocity(const eng::V3& g, float dt, float ld, float ad, float as, float mls, float mas, eng::V3& l, eng::V3& a,
                                          bool dg) {
  H::bodyCoreComputeUnconstrainedVelocity(g, dt, ld, ad, as, mls, mas, l, a, dg);
}
void copyToSolverBodyDataStep(const eng::V3& lv, const eng::V3& av, float invMass, const eng::V3& invInertia, const eng::Tf& pose, float mdv,
                              float mci, uint32_t node, float rt, float mav, uint32_t lock, bool kin, TgsBodyVel& v, TgsTxInertia& t, TgsBodyData& d,
                              float dt, bool gyro) {
  H::copyToSolverBodyDataStep(lv, av, invMass, invInertia, pose, mdv, mci, node, rt, mav, lock, kin, v, t, d, dt, gyro);
}
void integrateCoreStep(TgsBodyVel& v, TgsTxInertia& t, float dt) { H::integrateCoreStep(v, t, dt); }
void worldBody(TgsBodyVel& v, TgsTxInertia& t, TgsBodyData& d) { H::worldBody(v, t, d); }

}  // namespace jeng
