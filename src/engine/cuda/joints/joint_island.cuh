// joints 층 2: 판(env) N 개의 "조인트 섬" 한 simulate 를 CUDA 로 (판 하나 = 스레드 하나, 판 안 순서는 PhysX 와 같게).
// 한 판 = 몸체 ≤ MAXB-1 개(+세계) + D6 조인트 ≤ MAXJ 개(강체-강체, 모두 stride 1 제약 묶음) — 보조 잡기 한 번 분량.
// 순서는 PhysX 섬 풀이 경로(DyTGSDynamics.cpp:2515 iterativeSolveIsland, 조인트가 있는 묶음)와 같다:
//   준비(SetupSolverConstraintStep) -> 위치 반복 1..n-1: 풀기, 적분 -> 마지막 위치 반복: 풀기+마무리, 적분 -> 속도 반복 -> 되쓰기.
// 진짜 엔진에서는 solver 가 섬 묶기·분할·반복 루프를 맡고 이 함수들(core/joints)을 부른다. 이 커널은 joints 함수의 층 2 시험·처리량용.
// 같은 코드(runEnv)를 호스트에서 돌리면 층 1 이다.
#pragma once
#include <cstdint>

#include "core/joints/tgs_1d.h"
#include "core/joints/tgs_1d4.h"
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

// ---- 4개 묶음 판 (공식 radio 처럼 독립 조인트가 한 분할에 모이는 경우). 조인트를 장면 순서대로 4개씩 묶어
// 넷 다 행이 있으면 SIMD 4칸 경로(tgs_1d4.h), 아니면 하나씩 (PhysX 묶기 규칙: DyTGSContactPrepBlock.cpp:1659, NpImmediateMode.cpp:1496).
// 묶음 안 동적 몸체는 겹치지 않게 만든다(분할이 보장하는 것). 세계는 여러 칸이 같이 쓴다.
constexpr int MAXB4 = 9;   // 0 = 세계
constexpr int MAXJ4 = 8;   // 묶음 2개
constexpr uint32_t BLK4 = uint32_t(sizeof(Sc1DHeader4) + sizeof(Sc1DRow4) * MAX_CONSTRAINT_ROWS) + 16u;
constexpr uint32_t ARENA4 = 2u * BLK4 > uint32_t(MAXJ4) * BLK ? 2u * BLK4 : uint32_t(MAXJ4) * BLK;

struct JEnv4In {
  uint32_t nb, nj, posIters, velIters;
  float simDt;
  TgsBodyVel v[MAXB4];
  TgsTxInertia t[MAXB4];
  TgsBodyData d[MAXB4];
  Tf frame[MAXB4];
  D6Data data[MAXJ4];
  uint16_t flags[MAXJ4];
  uint8_t a[MAXJ4], b[MAXJ4];
  float lb[MAXJ4], ab[MAXJ4], mrt[MAXJ4];
};
struct alignas(16) JEnv4Out {
  TgsBodyVel v[MAXB4];
  TgsTxInertia t[MAXB4];
  Writeback wb[MAXJ4];
  uint32_t off[MAXJ4];   // 조인트 k 의 블록 시작 (arena 안), 없으면 ~0
  uint32_t len[MAXJ4];   // 블록 길이 (4묶음이면 네 조인트가 같은 값)
  uint32_t isBlock4[MAXJ4 / 4];
  uint32_t pad[2];
  alignas(16) uint8_t arena[ARENA4];
};

EHD void runEnv4(const JEnv4In& in, JEnv4Out& out) {
  const uint32_t nb = in.nb, nj = in.nj, ng = (nj + 3) / 4;
  for (uint32_t i = 0; i <= nb; ++i) { out.v[i] = in.v[i]; out.t[i] = in.t[i]; }
  const uint32_t posIters = in.posIters, velIters = in.velIters;
  const float simDt = in.simDt;
  const float stepDt = simDt / float(posIters);
  const float invStepDt = 1.f / stepDt;
  const float biasCoefficient = 2.f * psqrt(1.f / float(posIters));
  const float invSimDt = 1.0f / simDt;
  Row rows[MAX_CONSTRAINT_ROWS * 4];
  uint8_t* blk[MAXJ4];
  uint32_t cur = 0;
  for (uint32_t q = 0; q < ng; ++q) {
    const uint32_t k0 = 4 * q;
    uint32_t len = 0;
    if (k0 + 4 <= nj) {
      const D6Data* dd[4]; uint16_t fl[4]; float lb[4], ab[4], mr[4];
      const Tf* f0[4]; const Tf* f1[4];
      const TgsBodyVel* b0[4]; const TgsBodyVel* b1[4]; const TgsTxInertia* t0[4]; const TgsTxInertia* t1[4];
      const TgsBodyData* d0[4]; const TgsBodyData* d1[4];
      for (int a = 0; a < 4; ++a) {
        const uint32_t k = k0 + a, A = in.a[k], B = in.b[k];
        dd[a] = &in.data[k]; fl[a] = in.flags[k]; lb[a] = in.lb[k]; ab[a] = in.ab[k]; mr[a] = in.mrt[k];
        f0[a] = &in.frame[A]; f1[a] = &in.frame[B];
        b0[a] = &out.v[A]; b1[a] = &out.v[B]; t0[a] = &out.t[A]; t1[a] = &out.t[B]; d0[a] = &in.d[A]; d1[a] = &in.d[B];
      }
      len = prepareD6Step4(dd, fl, lb, ab, mr, f0, f1, b0, b1, t0, t1, d0, d1, rows, out.arena + cur, stepDt, simDt, invStepDt, invSimDt, 1.0f,
                           biasCoefficient, false);
    }
    out.isBlock4[q] = len ? 1u : 0u;
    if (len) {
      for (int a = 0; a < 4; ++a) { blk[k0 + a] = out.arena + cur; out.off[k0 + a] = cur; out.len[k0 + a] = len; }
      cur += len + 16u;
      continue;
    }
    for (uint32_t k = k0; k < nj && k < k0 + 4; ++k) {
      const uint32_t A = in.a[k], B = in.b[k];
      uint32_t l = 0;
      const uint32_t n = prepareD6Step(in.data[k], in.flags[k], in.lb[k], in.ab[k], in.mrt[k], in.frame[A], in.frame[B], out.v[A], out.v[B], out.t[A],
                                       out.t[B], in.d[A], in.d[B], RIGID_BODY, RIGID_BODY, rows, out.arena + cur, stepDt, simDt, invStepDt, invSimDt,
                                       1.0f, biasCoefficient, NoArt(), NoArt(), &l);
      blk[k] = n ? out.arena + cur : nullptr;
      out.off[k] = n ? cur : ~0u;
      out.len[k] = n ? l : 0u;
      cur += n ? l : 0u;
    }
  }
  auto solveAll = [&](float elapsed, bool conclude) {
    for (uint32_t q = 0; q < ng; ++q) {
      const uint32_t k0 = 4 * q;
      if (out.isBlock4[q]) {
        TgsBodyVel* bb[4][2];
        const TgsTxInertia* tt[4][2];
        for (int a = 0; a < 4; ++a) {
          bb[a][0] = &out.v[in.a[k0 + a]]; bb[a][1] = &out.v[in.b[k0 + a]];
          tt[a][0] = &out.t[in.a[k0 + a]]; tt[a][1] = &out.t[in.b[k0 + a]];
        }
        solve1DStep4(blk[k0], bb, tt, elapsed, false, true);
        if (conclude) conclude1DStep4(blk[k0], false);
      } else {
        for (uint32_t k = k0; k < nj && k < k0 + 4; ++k) {
          solve1DStep(blk[k], out.v[in.a[k]], out.v[in.b[k]], out.t[in.a[k]], out.t[in.b[k]], elapsed);
          if (conclude) conclude1DStep(blk[k]);
        }
      }
    }
  };
  float elapsed = 0.0f;
  for (uint32_t it = 1; it < posIters; ++it) {
    solveAll(elapsed, false);
    for (uint32_t i = 1; i <= nb; ++i) jtest::integrateCoreStep(out.v[i], out.t[i], stepDt);
    elapsed += stepDt;
  }
  solveAll(elapsed, true);
  elapsed += stepDt;
  for (uint32_t i = 1; i <= nb; ++i) jtest::integrateCoreStep(out.v[i], out.t[i], stepDt);
  for (uint32_t it = 0; it < velIters; ++it) solveAll(elapsed, false);
  for (uint32_t q = 0; q < ng; ++q) {
    const uint32_t k0 = 4 * q;
    if (out.isBlock4[q]) {
      Writeback* w[4] = {&out.wb[k0], &out.wb[k0 + 1], &out.wb[k0 + 2], &out.wb[k0 + 3]};
      writeBack1D4(blk[k0], w, false);
    } else {
      for (uint32_t k = k0; k < nj && k < k0 + 4; ++k) writeBack1DStep(blk[k], &out.wb[k]);
    }
  }
}

#if defined(__CUDACC__)
__global__ void kJointIsland(const JEnvIn* in, JEnvOut* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  runEnv(in[i], out[i]);
}
#endif

#if defined(__CUDACC__)
__global__ void kJointIsland4(const JEnv4In* in, JEnv4Out* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  runEnv4(in[i], out[i]);
}
#endif

}  // namespace jcuda
}  // namespace eng
