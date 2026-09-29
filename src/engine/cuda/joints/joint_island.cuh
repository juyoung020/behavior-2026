// joints 층 2: 판(env) N 개의 "조인트 섬" 한 simulate 를 CUDA 로 (판 하나 = 스레드 하나, 판 안 순서는 PhysX 와 같게).
// 한 판 = 몸체 ≤ MAXB-1 개(+세계) + D6 조인트 ≤ MAXJ 개(강체-강체, 모두 stride 1 제약 묶음) — 보조 잡기 한 번 분량.
// 순서는 PhysX 섬 풀이 경로(DyTGSDynamics.cpp:2515 iterativeSolveIsland, 조인트가 있는 묶음)와 같다:
//   준비(SetupSolverConstraintStep) -> 위치 반복 1..n-1: 풀기, 적분 -> 마지막 위치 반복: 풀기+마무리, 적분 -> 속도 반복 -> 되쓰기.
// 진짜 엔진에서는 solver 가 섬 묶기·분할·반복 루프를 맡고 이 함수들(core/joints)을 부른다. 이 커널은 joints 함수의 층 2 시험·처리량용.
// 같은 코드(runEnv)를 호스트에서 돌리면 층 1 이다.
#pragma once
#include <cstdint>

#include "core/joints/tgs_1d.h"
#include "tests/joints/tgs_harness.h"

namespace eng {
namespace jcuda {
using namespace eng::jnt;

constexpr int MAXB = 4;  // 0 = 세계
constexpr int MAXJ = 3;
constexpr uint32_t BLK = uint32_t(sizeof(Sc1DHeader) + sizeof(Sc1DRow) * MAX_CONSTRAINT_ROWS);

struct JEnvIn {
  uint32_t nb, nj, posIters, velIters;
  float simDt;
  TgsBodyVel v[MAXB];
  TgsTxInertia t[MAXB];
  TgsBodyData d[MAXB];
  Tf frame[MAXB];  // 몸체 질량중심 자세(PxsBodyCore::body2World), 세계 = 항등
  D6Data data[MAXJ];
  uint16_t flags[MAXJ];
  uint8_t a[MAXJ], b[MAXJ];
  float lb[MAXJ], ab[MAXJ], mrt[MAXJ];
};
struct alignas(16) JEnvOut {
  TgsBodyVel v[MAXB];
  TgsTxInertia t[MAXB];
  Writeback wb[MAXJ];
  uint32_t len[MAXJ];
  uint32_t pad;
  alignas(16) uint8_t blk[MAXJ][BLK];
};

EHD void runEnv(const JEnvIn& in, JEnvOut& out) {
  const uint32_t nb = in.nb, nj = in.nj;
  for (uint32_t i = 0; i <= nb; ++i) { out.v[i] = in.v[i]; out.t[i] = in.t[i]; }
  const uint32_t posIters = in.posIters, velIters = in.velIters;
  const float simDt = in.simDt;
  const float stepDt = simDt / float(posIters);          // SetStepperTask (DyTGSDynamics.cpp:1898)
  const float invStepDt = 1.f / stepDt;
  const float biasCoefficient = 2.f * psqrt(1.f / float(posIters));
  const float invSimDt = 1.0f / simDt;
  Row rows[MAX_CONSTRAINT_ROWS];
  uint8_t* blk[MAXJ];
  for (uint32_t k = 0; k < nj; ++k) {
    const uint32_t a = in.a[k], b = in.b[k];
    uint32_t len = 0;
    const uint32_t n = prepareD6Step(in.data[k], in.flags[k], in.lb[k], in.ab[k], in.mrt[k], in.frame[a], in.frame[b], out.v[a], out.v[b], out.t[a],
                                     out.t[b], in.d[a], in.d[b], RIGID_BODY, RIGID_BODY, rows, out.blk[k], stepDt, simDt, invStepDt, invSimDt,
                                     1.0f, biasCoefficient, NoArt(), NoArt(), &len);
    out.len[k] = len;
    blk[k] = n ? out.blk[k] : nullptr;
  }
  float elapsed = 0.0f;
  for (uint32_t it = 1; it < posIters; ++it) {
    for (uint32_t k = 0; k < nj; ++k) solve1DStep(blk[k], out.v[in.a[k]], out.v[in.b[k]], out.t[in.a[k]], out.t[in.b[k]], elapsed);
    for (uint32_t i = 1; i <= nb; ++i) jtest::integrateCoreStep(out.v[i], out.t[i], stepDt);
    elapsed += stepDt;
  }
  for (uint32_t k = 0; k < nj; ++k) {
    solve1DStep(blk[k], out.v[in.a[k]], out.v[in.b[k]], out.t[in.a[k]], out.t[in.b[k]], elapsed);
    conclude1DStep(blk[k]);
  }
  elapsed += stepDt;
  for (uint32_t i = 1; i <= nb; ++i) jtest::integrateCoreStep(out.v[i], out.t[i], stepDt);
  for (uint32_t it = 0; it < velIters; ++it)
    for (uint32_t k = 0; k < nj; ++k) solve1DStep(blk[k], out.v[in.a[k]], out.v[in.b[k]], out.t[in.a[k]], out.t[in.b[k]], elapsed);
  for (uint32_t k = 0; k < nj; ++k) writeBack1DStep(blk[k], &out.wb[k]);
}

#if defined(__CUDACC__)
__global__ void kJointIsland(const JEnvIn* in, JEnvOut* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  runEnv(in[i], out[i]);
}
#endif

}  // namespace jcuda
}  // namespace eng
