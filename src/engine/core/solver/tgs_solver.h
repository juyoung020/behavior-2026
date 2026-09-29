// TGS 풀이 한 스텝 (손으로 짬, PhysX 5.6.1 과 비트 동일 목표): 활성 섬 -> 풀이 묶음 -> 분할(partition) -> 제약 준비 ->
// 위치 반복(반복마다 적분) -> 속도 반복 -> 되쓰기 -> 몸체 되쓰기·잠 판정.
// 섬 순서(어느 몸체·접촉 관리자가 어떤 순서로 오는지)는 입력으로 받는다(섬 관리 = islands.h, 따로 검증).
// 원본 (physx/source/lowleveldynamics/src/)
//   DyTGSDynamics.cpp:525-756    update / updatePostKinematic (묶음 나누기: 몸체 >= solverBatchSize 또는 관절체 >= 16 까지 섬을 모음)
//   DyTGSDynamics.cpp:758-1021   prepareBodiesAndConstraints / setupDescs / preIntegrateBodies
//   DyTGSDynamics.cpp:1023-1259  createSolverConstraints (4개 묶음 시도 -> 실패하면 하나씩)
//   DyTGSDynamics.cpp:1868-1901  SetStepperTask (stepDt = dt/위치반복, biasCoefficient = 2*sqrt(1/위치반복))
//   DyTGSDynamics.cpp:2128-2252  PartitionTask (묶음 머리: 같은 분할·같은 종류 최대 4개)
//   DyTGSDynamics.cpp:2306-2463  SolveIslandTask (길이 0 제약 빼기, 정적 묶음 종류 고치기)
//   DyTGSDynamics.cpp:2515-2793  iterativeSolveIsland (단일 스레드 경로; PhysX 는 스레드 수와 무관하게 같은 결과)
//   DyTGSDynamics.cpp:3610,1549  finishSolveIsland / copyBackBodies
//   DyConstraintPartition.cpp    partitionContactConstraints (관절체 없는 RigidBodyClassification)
//   DyTGSDynamics.cpp:916-966,1181-1259  1D 제약(조인트) 기술자·정렬·준비, :1262-1316 풀이·마무리·되쓰기 함수 표
// 1D 제약 준비·풀이 식은 joints 모듈(core/joints/tgs_1d.h, tgs_1d4.h)을 부른다(docs 12.3 경계). 어느 순서로 부를지는 여기.
// 아직 안 옮김: 관절체(articulation 모듈 함수 표로 붙일 자리 표시), 운동학 몸체, CCD,
//              externalForcesEveryTgsIteration(기본 꺼짐), 잔차 보고.
#pragma once
#include <cstring>

#include "../joints/tgs_1d.h"
#include "../joints/tgs_1d4.h"
#include "contact_prep4.h"
#include "contact_solve.h"
#include "tgs_body.h"

namespace eng {
namespace sv {

// 판 안 여러 스레드(GPU 블록 하나 = 판 하나)용: 장치에서는 블록 동기화, 호스트(스레드 하나)에서는 아무것도 안 함
#if defined(__CUDA_ARCH__)
#define SV_SYNC() __syncthreads()
SV_HD void svAtomicOr(uint32_t& dst, uint32_t v) { atomicOr(&dst, v); }
#else
#define SV_SYNC() ((void)0)
SV_HD void svAtomicOr(uint32_t& dst, uint32_t v) { dst |= v; }
#endif

// ---------------- 분할 (DyConstraintPartition.cpp, RigidBodyClassification)
struct PartitionView {
  SBodyVel* vels;
  uint32_t base, count;  // 활성 몸체 = 풀 번호 [base, base+count)
  SV_HD bool active(uint32_t poolIndex) const { return poolIndex - base < count; }  // uintptr 나눗셈과 같은 뜻(음수 -> 큰 수)
};

SV_HD bool computeAvailablePartition(uint32_t& availablePartition, uint32_t& partitionsA, uint32_t& partitionsB, bool activeA, bool activeB) {
  const uint32_t combinedMask = (~partitionsA & ~partitionsB);
  if (combinedMask == 0) {
    availablePartition = 32;
    return false;
  }
  uint32_t low = 0;
  while (!((combinedMask >> low) & 1u)) ++low;  // PxLowestSetBit
  availablePartition = low;
  const uint32_t partitionBit = (1u << availablePartition);
  if (activeA) partitionsA |= partitionBit;
  if (activeB) partitionsB |= partitionBit;
  return true;
}

// 반환: maxPartition. counts = 분할별 끝 위치(누적). 결과 순서는 ordered.
SV_HDN uint32_t partitionContactConstraints(const PartitionView& pv, const SDesc* descs, uint32_t numDescs, SDesc* ordered, SDesc* overflowTmp,
                                           uint32_t* counts, uint32_t& countsSize, uint32_t countsCap, uint32_t maxPartitions,
                                           uint32_t& numOverflows, uint32_t& numStaticConstraints, uint32_t& numOrdered, uint32_t& err) {
  const uint32_t MAX_NUM_PARTITIONS = 32;
  // zeroBodies
  for (uint32_t k = 0; k < pv.count; ++k) {
    SBodyVel& b = pv.vels[pv.base + k];
    b.partitionMask = 0;
    b.nbStaticInteractions = 0;
    b.maxDynamicPartition = 0;
  }
  // classifyConstraintDesc
  uint32_t numUnpartitioned = 0;
  countsSize = MAX_NUM_PARTITIONS;
  for (uint32_t a = 0; a < MAX_NUM_PARTITIONS; ++a) counts[a] = 0;
  for (uint32_t i = 0; i < numDescs; ++i) {
    const SDesc& d = descs[i];
    const bool activeA = pv.active(d.bodyA), activeB = pv.active(d.bodyB);
    uint32_t partitionsA = pv.vels[d.bodyA].partitionMask, partitionsB = pv.vels[d.bodyB].partitionMask;
    if (activeA && activeB) {
      uint32_t availablePartition;
      if (!computeAvailablePartition(availablePartition, partitionsA, partitionsB, activeA, activeB)) {
        overflowTmp[numUnpartitioned++] = d;
        continue;
      }
      counts[availablePartition]++;
      availablePartition++;
      SBodyVel& A = pv.vels[d.bodyA];
      SBodyVel& B = pv.vels[d.bodyB];
      A.partitionMask = partitionsA;
      A.maxDynamicPartition = uint16_t(pmaxu(A.maxDynamicPartition, availablePartition));
      B.partitionMask = partitionsB;
      B.maxDynamicPartition = uint16_t(pmaxu(B.maxDynamicPartition, availablePartition));
    } else {
      if (activeA) pv.vels[d.bodyA].nbStaticInteractions++;
      if (activeB) pv.vels[d.bodyB].nbStaticInteractions++;
    }
  }
  uint32_t partitionStartIndex = 0;
  while (numUnpartitioned > 0) {
    for (uint32_t k = 0; k < pv.count; ++k) pv.vels[pv.base + k].partitionMask = 0;  // clearState
    partitionStartIndex += MAX_NUM_PARTITIONS;
    if (maxPartitions <= partitionStartIndex) break;
    if (countsSize + MAX_NUM_PARTITIONS > countsCap) {
      err |= SV_ERR_PARTITION;
      return 0;
    }
    countsSize += MAX_NUM_PARTITIONS;
    for (uint32_t a = 0; a < MAX_NUM_PARTITIONS; ++a) counts[partitionStartIndex + a] = 0;
    uint32_t newNum = 0;
    for (uint32_t i = 0; i < numUnpartitioned; ++i) {
      const SDesc d = overflowTmp[i];
      const bool activeA = pv.active(d.bodyA), activeB = pv.active(d.bodyB);
      uint32_t partitionsA = pv.vels[d.bodyA].partitionMask, partitionsB = pv.vels[d.bodyB].partitionMask;
      uint32_t availablePartition;
      if (!computeAvailablePartition(availablePartition, partitionsA, partitionsB, activeA, activeB)) {
        overflowTmp[newNum++] = d;
        continue;
      }
      availablePartition += partitionStartIndex;
      counts[availablePartition]++;
      availablePartition++;
      SBodyVel& A = pv.vels[d.bodyA];
      SBodyVel& B = pv.vels[d.bodyB];
      A.partitionMask = partitionsA;
      A.maxDynamicPartition = uint16_t(pmaxu(A.maxDynamicPartition, availablePartition));
      B.partitionMask = partitionsB;
      B.maxDynamicPartition = uint16_t(pmaxu(B.maxDynamicPartition, availablePartition));
    }
    numUnpartitioned = newNum;
  }
  // reserveSpaceForStaticConstraints
  for (uint32_t k = 0; k < pv.count; ++k) {
    SBodyVel& b = pv.vels[pv.base + k];
    b.partitionMask = 0;
    const uint32_t requiredSize = uint32_t(b.maxDynamicPartition + b.nbStaticInteractions);
    if (requiredSize > countsSize) {
      if (requiredSize > countsCap) {
        err |= SV_ERR_PARTITION;
        return 0;
      }
      for (uint32_t a = countsSize; a < requiredSize; ++a) counts[a] = 0;  // PxArray::resize 는 0 으로 채움
      countsSize = requiredSize;
    }
    for (uint32_t bb = 0; bb < b.nbStaticInteractions; bb++) counts[b.maxDynamicPartition + bb]++;
  }
  numOverflows = numUnpartitioned;
  // 누적 (radix 정렬 오프셋)
  uint32_t accumulation = 0;
  for (uint32_t a = 0; a < countsSize; a++) {
    const uint32_t count = counts[a];
    counts[a] = accumulation;
    accumulation += count;
  }
  // afterClassification
  for (uint32_t k = 0; k < pv.count; ++k) {
    SBodyVel& b = pv.vels[pv.base + k];
    b.partitionMask = 0;
    b.nbStaticInteractions = 0;
  }
  // writeConstraintDesc
  numUnpartitioned = 0;
  numStaticConstraints = 0;
  for (uint32_t i = 0; i < numDescs; ++i) {
    const SDesc& d = descs[i];
    const bool activeA = pv.active(d.bodyA), activeB = pv.active(d.bodyB);
    uint32_t partitionsA = pv.vels[d.bodyA].partitionMask, partitionsB = pv.vels[d.bodyB].partitionMask;
    if (activeA && activeB) {
      uint32_t availablePartition;
      if (!computeAvailablePartition(availablePartition, partitionsA, partitionsB, activeA, activeB)) {
        overflowTmp[numUnpartitioned++] = d;
        continue;
      }
      SBodyVel& A = pv.vels[d.bodyA];
      SBodyVel& B = pv.vels[d.bodyB];
      A.partitionMask = partitionsA;
      A.maxDynamicPartition = uint16_t(pmaxu(A.maxDynamicPartition, availablePartition + 1));
      B.partitionMask = partitionsB;
      B.maxDynamicPartition = uint16_t(pmaxu(B.maxDynamicPartition, availablePartition + 1));
      ordered[numOverflows + counts[availablePartition]++] = d;
    } else {
      uint32_t index = NONE;
      if (activeA) {
        SBodyVel& A = pv.vels[d.bodyA];
        index = uint32_t(A.maxDynamicPartition + A.nbStaticInteractions++);
      } else if (activeB) {
        SBodyVel& B = pv.vels[d.bodyB];
        index = uint32_t(B.maxDynamicPartition + B.nbStaticInteractions++);
      }
      if (index != NONE)
        ordered[numOverflows + counts[index]++] = d;
      else
        numStaticConstraints++;
    }
  }
  partitionStartIndex = 0;
  while (numUnpartitioned > 0) {
    for (uint32_t k = 0; k < pv.count; ++k) pv.vels[pv.base + k].partitionMask = 0;
    partitionStartIndex += MAX_NUM_PARTITIONS;
    if (partitionStartIndex >= maxPartitions) break;
    uint32_t newNum = 0;
    for (uint32_t i = 0; i < numUnpartitioned; ++i) {
      const SDesc d = overflowTmp[i];
      const bool activeA = pv.active(d.bodyA), activeB = pv.active(d.bodyB);
      uint32_t partitionsA = pv.vels[d.bodyA].partitionMask, partitionsB = pv.vels[d.bodyB].partitionMask;
      uint32_t availablePartition;
      if (!computeAvailablePartition(availablePartition, partitionsA, partitionsB, activeA, activeB)) {
        overflowTmp[newNum++] = d;
        continue;
      }
      pv.vels[d.bodyA].partitionMask = partitionsA;  // storeProgress_
      pv.vels[d.bodyB].partitionMask = partitionsB;
      availablePartition += partitionStartIndex;
      ordered[numOverflows + counts[availablePartition]++] = d;
    }
    numUnpartitioned = newNum;
  }
  numOrdered = numDescs;  // RigidBodyClassification (extended = false)
  if (numOverflows) {     // outputOverflowConstraints
    if (countsSize + 1 > countsCap) {
      err |= SV_ERR_PARTITION;
      return 0;
    }
    countsSize += 1;
    uint32_t partitionCount = countsSize;
    counts[countsSize - 1] = 0;
    while (partitionCount-- > 1) counts[partitionCount] = counts[partitionCount - 1] + numOverflows;
    counts[0] = numOverflows;
    for (uint32_t i = 0; i < numOverflows; ++i) ordered[i] = overflowTmp[i];
  }
  uint32_t prevPartitionSize = 0, maxPartition = 0;
  for (uint32_t a = 0; a < countsSize; ++a, maxPartition++) {
    if (counts[a] == prevPartitionSize) break;
    prevPartitionSize = counts[a];
  }
  return maxPartition;
}

// ---------------- 1D 제약 (조인트): joints 모듈 함수에 넘길 때 풀이 몸체는 바이트 복사로 건넨다
// (SBodyVel/SBodyTxI/SBodyData 와 jnt::TgsBodyVel/TgsTxInertia/TgsBodyData 는 둘 다 PhysX 배치 — 형이 달라 포인터로 섞지 않는다)
static_assert(sizeof(SBodyVel) == sizeof(jnt::TgsBodyVel), "PxTGSSolverBodyVel 배치");
static_assert(sizeof(SBodyTxI) == sizeof(jnt::TgsTxInertia), "PxTGSSolverBodyTxInertia 배치");
static_assert(sizeof(SBodyData) == sizeof(jnt::TgsBodyData), "PxTGSSolverBodyData 배치");
template <class D, class S>
SV_HD D bitCopy(const S& s) {
  static_assert(sizeof(D) == sizeof(S), "");
  D d;
  memcpy(&d, &s, sizeof(D));
  return d;
}

// solve1DBlock / solveConclude1DBlock / solve1D4 / solveConclude1D4 (DyTGSContactPrep.cpp:3320,3361, DyTGSContactPrepBlock.cpp:3410,3628)
SV_HDN void solve1DHeader(const BatchHeader& h, const SDesc* ordered, SBodyVel* vels, const SBodyTxI* txI, ByteArena& arena, float elapsed,
                          bool conclude) {
  if (h.constraintType == SC_TYPE_RB_1D) {
    for (uint32_t i = h.startIndex, e = h.startIndex + h.stride; i < e; ++i) {
      const SDesc& d = ordered[i];
      if (d.constraint == NONE) continue;  // solve1DStep: bPtr == NULL 이면 끝
      uint8_t* blk = arenaPtr<uint8_t>(arena, d.constraint);
      jnt::TgsBodyVel b0 = bitCopy<jnt::TgsBodyVel>(vels[d.bodyA]);
      jnt::TgsBodyVel b1 = bitCopy<jnt::TgsBodyVel>(vels[d.bodyB]);
      const jnt::TgsTxInertia t0 = bitCopy<jnt::TgsTxInertia>(txI[d.bodyADataIndex]);
      const jnt::TgsTxInertia t1 = bitCopy<jnt::TgsTxInertia>(txI[d.bodyBDataIndex]);
      jnt::solve1DStep(blk, b0, b1, t0, t1, elapsed);
      vels[d.bodyA] = bitCopy<SBodyVel>(b0);  // 원본 저장 순서: b0 다음 b1
      vels[d.bodyB] = bitCopy<SBodyVel>(b1);
      if (conclude) jnt::conclude1DStep(blk);
    }
  } else {  // SC_TYPE_BLOCK_1D
    const SDesc* d = ordered + h.startIndex;
    uint8_t* blk = arenaPtr<uint8_t>(arena, d[0].constraint);
    jnt::TgsBodyVel bv[4][2];
    jnt::TgsTxInertia tv[4][2];
    jnt::TgsBodyVel* bp[4][2];
    const jnt::TgsTxInertia* tp[4][2];
    for (int a = 0; a < 4; ++a) {
      bv[a][0] = bitCopy<jnt::TgsBodyVel>(vels[d[a].bodyA]);
      bv[a][1] = bitCopy<jnt::TgsBodyVel>(vels[d[a].bodyB]);
      tv[a][0] = bitCopy<jnt::TgsTxInertia>(txI[d[a].bodyADataIndex]);
      tv[a][1] = bitCopy<jnt::TgsTxInertia>(txI[d[a].bodyBDataIndex]);
      bp[a][0] = &bv[a][0];
      bp[a][1] = &bv[a][1];
      tp[a][0] = &tv[a][0];
      tp[a][1] = &tv[a][1];
    }
    jnt::solve1DStep4(blk, bp, tp, elapsed, false, true);
    for (int a = 0; a < 4; ++a) vels[d[a].bodyA] = bitCopy<SBodyVel>(bv[a][0]);  // 원본 저장 순서: 네 칸의 몸체 0 다음 몸체 1
    for (int a = 0; a < 4; ++a) vels[d[a].bodyB] = bitCopy<SBodyVel>(bv[a][1]);
    if (conclude) jnt::conclude1DStep4(blk, false);
  }
}

// createSolverConstraints 의 1D 갈래 (DyTGSDynamics.cpp:1181-1259): 머리에 4개면 4개 묶음 준비(setupSolverConstraintStep4)를 먼저 해 보고,
// 안 되면(행 0 인 조인트가 있으면) 하나씩 SetupSolverConstraintStep. 셰이더 = D6 (joints 모듈 prepareD6Step*).
// 제약 자료는 원본처럼 길이 + 16 바이트를 잡는다(DyTGSContactPrep.cpp:1965, DyTGSContactPrepBlock.cpp:1760).
SV_HDN void prepare1DHeader(SolverBoard& B, const SolverParams& prm, const BatchHeader& hdr, float stepDt, float totalDt, float invStepDt,
                            float invTotalDt, float biasCoefficient) {
  const uint32_t startIdx = hdr.startIndex, endIdx = startIdx + hdr.stride;
  const Tf ident{qid(), V3{0.0f, 0.0f, 0.0f}};  // PxTransform(PxIdentity): 정적 쪽 몸체 틀
  const jnt::D6Data* data[4];
  uint16_t flags[4];
  float linBreak[4], angBreak[4], minResp[4];
  Tf frame0[4], frame1[4];
  jnt::TgsBodyVel bv0[4], bv1[4];
  jnt::TgsTxInertia t0[4], t1[4];
  jnt::TgsBodyData d0[4], d1[4];
  for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, ++i) {
    const SDesc& desc = B.ordered[a];
    const Constraint1DIn& c = B.c1d[desc.source];
    data[i] = &B.jointData[c.data];
    flags[i] = c.flags;
    linBreak[i] = c.linBreakForce;
    angBreak[i] = c.angBreakForce;
    minResp[i] = c.minResponseThreshold;
    frame0[i] = c.body0 == NONE ? ident : B.bodies[c.body0].body2World;  // constraint->body0->getPose()
    frame1[i] = c.body1 == NONE ? ident : B.bodies[c.body1].body2World;
    bv0[i] = bitCopy<jnt::TgsBodyVel>(B.vels[desc.bodyA]);
    bv1[i] = bitCopy<jnt::TgsBodyVel>(B.vels[desc.bodyB]);
    t0[i] = bitCopy<jnt::TgsTxInertia>(B.txI[desc.bodyADataIndex]);
    t1[i] = bitCopy<jnt::TgsTxInertia>(B.txI[desc.bodyBDataIndex]);
    d0[i] = bitCopy<jnt::TgsBodyData>(B.datas[desc.bodyADataIndex]);
    d1[i] = bitCopy<jnt::TgsBodyData>(B.datas[desc.bodyBDataIndex]);
  }
  if (hdr.stride == 4) {  // PX_USE_BLOCK_1D
    const uint32_t prev = B.constraints.size;
    const uint32_t off = arenaAlloc(B.constraints, jnt::blockLength4(jnt::MAX_CONSTRAINT_ROWS, false) + 16u);
    if (off == NONE) {
      B.error |= SV_ERR_ARENA;
      return;
    }
    const Tf* f0p[4] = {&frame0[0], &frame0[1], &frame0[2], &frame0[3]};
    const Tf* f1p[4] = {&frame1[0], &frame1[1], &frame1[2], &frame1[3]};
    const jnt::TgsBodyVel* b0p[4] = {&bv0[0], &bv0[1], &bv0[2], &bv0[3]};
    const jnt::TgsBodyVel* b1p[4] = {&bv1[0], &bv1[1], &bv1[2], &bv1[3]};
    const jnt::TgsTxInertia* t0p[4] = {&t0[0], &t0[1], &t0[2], &t0[3]};
    const jnt::TgsTxInertia* t1p[4] = {&t1[0], &t1[1], &t1[2], &t1[3]};
    const jnt::TgsBodyData* d0p[4] = {&d0[0], &d0[1], &d0[2], &d0[3]};
    const jnt::TgsBodyData* d1p[4] = {&d1[0], &d1[1], &d1[2], &d1[3]};
    const uint32_t len = jnt::prepareD6Step4(data, flags, linBreak, angBreak, minResp, f0p, f1p, b0p, b1p, t0p, t1p, d0p, d1p, B.rowScratch,
                                             arenaPtr<uint8_t>(B.constraints, off), stepDt, totalDt, invStepDt, invTotalDt, prm.lengthScale,
                                             biasCoefficient, false);
    if (len) {
      for (uint32_t a = startIdx; a < endIdx; ++a) {
        B.ordered[a].constraint = off;
        B.ordered[a].constraintLengthOver16 = uint16_t(len / 16);
      }
      B.constraints.size = off + len + 16u;
      B.stat1DBlock4++;
      return;
    }
    B.constraints.size = prev;  // eUNBATCHABLE: 하나씩
  }
  const jnt::NoArt na;
  for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, ++i) {
    SDesc& desc = B.ordered[a];
    const uint32_t prev = B.constraints.size;
    const uint32_t off = arenaAlloc(B.constraints, jnt::blockLength(jnt::MAX_CONSTRAINT_ROWS, false) + 16u);
    if (off == NONE) {
      B.error |= SV_ERR_ARENA;
      return;
    }
    uint32_t len = 0;
    const uint32_t n = jnt::prepareD6Step(*data[i], flags[i], linBreak[i], angBreak[i], minResp[i], frame0[i], frame1[i], bv0[i], bv1[i], t0[i],
                                          t1[i], d0[i], d1[i], jnt::RIGID_BODY, jnt::RIGID_BODY, B.rowScratch,
                                          arenaPtr<uint8_t>(B.constraints, off), stepDt, totalDt, invStepDt, invTotalDt, prm.lengthScale,
                                          biasCoefficient, na, na, &len);
    B.stat1DSingle++;
    if (n == 0) {  // 행 없음: constraint = NULL, 길이 0 -> SolveIslandTask 가 뺀다
      desc.constraint = NONE;
      desc.constraintLengthOver16 = 0;
      B.constraints.size = prev;
      B.stat1DZeroRows++;
    } else {
      desc.constraint = off;
      desc.constraintLengthOver16 = uint16_t(len / 16);
      B.constraints.size = off + len + 16u;
    }
  }
}

// ---------------- 한 풀이 묶음 (DyTGSDynamics.cpp:3646 solveIsland 의 작업 사슬을 차례로)
struct BatchRange {
  uint32_t islandStart, islandEnd;
  uint32_t bodyStart, nbBodies;  // islandBodies 안
  uint32_t cmStart, nbCMs;       // islandCMs 안
  uint32_t solverBodyOffset;     // 운동학 수 + 앞 묶음 몸체 수
};

SV_HDN void solveContactHeader(const BatchHeader& h, const SDesc* ordered, SBodyVel* vels, const SBodyTxI* txI, ByteArena& arena, float minPen,
                               float elapsed, bool conclude, uint32_t& err) {
  switch (h.constraintType) {
    case SC_TYPE_RB_1D:
    case SC_TYPE_BLOCK_1D:
      solve1DHeader(h, ordered, vels, txI, arena, elapsed, conclude);
      break;
    case SC_TYPE_RB_CONTACT:
    case SC_TYPE_STATIC_CONTACT:
      for (uint32_t i = h.startIndex, e = h.startIndex + h.stride; i < e; ++i) solveContact(ordered[i], vels, arena, true, minPen, elapsed);
      break;
    case SC_TYPE_BLOCK_RB_CONTACT:
    case SC_TYPE_BLOCK_STATIC_RB_CONTACT:
      solveContact4_Block(ordered + h.startIndex, vels, arena, minPen, elapsed);
      break;
    default:
      err |= SV_ERR_UNSUPPORTED;  // 관절체 접촉·1D 는 아직
  }
}

// 준비 (차례로): 기술자·분할·묶음 머리·제약 준비·길이 0 제약 빼기까지. 반복 계획을 B.plan 에 남긴다.
SV_HDN void solveBatchPrep(SolverBoard& B, const SolverParams& prm, const BatchRange& R, uint32_t kinematicCount) {
  (void)kinematicCount;
  B.plan.ok = 0;
  const float mDt = prm.dt;
  const uint32_t bodyOffset = R.solverBodyOffset;
  SBodyVel* vels = B.vels;
  if (bodyOffset + R.nbBodies + 1 > B.poolCap) {
    B.error |= SV_ERR_POOL;
    return;
  }
  // prepareBodiesAndConstraints: 몸체 풀 번호
  for (uint32_t k = 0; k < R.nbBodies; ++k) B.bodySolverIndex[B.islandBodies[R.bodyStart + k]] = bodyOffset + k + 1;
  // setupDescs: 1D 제약 먼저(섬 순서대로 모아 Dy::Constraint::index 내림차순 정렬, DyTGSDynamics.cpp:916-966), 다음 접촉
  uint32_t nbDescs = 0;
  uint32_t nbC1D = 0;
  for (uint32_t isl = R.islandStart; isl < R.islandEnd; ++isl) nbC1D += B.islands[isl].c1dCount;
  if (R.nbCMs + nbC1D > B.descCap) {
    B.error |= SV_ERR_DESC;
    return;
  }
  for (uint32_t isl = R.islandStart; isl < R.islandEnd; ++isl) {
    const IslandIn& I = B.islands[isl];
    for (uint32_t k = 0; k < I.c1dCount; ++k) {
      const uint32_t ci = B.islandC1Ds[I.c1dStart + k];
      const Constraint1DIn& c = B.c1d[ci];
      SDesc& d = B.descs[nbDescs++];
      d.bodyA = d.bodyADataIndex = c.body0 == NONE ? 0u : B.bodySolverIndex[c.body0];
      d.bodyB = d.bodyBDataIndex = c.body1 == NONE ? 0u : B.bodySolverIndex[c.body1];
      d.linkIndexA = d.linkIndexB = RIGID_BODY;
      d.constraint = NONE;
      d.constraintLengthOver16 = 0;
      d.constraintType = SC_TYPE_RB_1D;
      d.source = ci;
      d.sortKey = c.index;
      d.progressA = d.progressB = 0;
    }
  }
  // PxSort(ConstraintLess): index 는 제약마다 달라 정렬 결과가 하나뿐이다 -> 삽입 정렬로 같은 순서
  for (uint32_t i = 1; i < nbDescs; ++i) {
    const SDesc t = B.descs[i];
    uint32_t j = i;
    while (j > 0 && B.descs[j - 1].sortKey < t.sortKey) {
      B.descs[j] = B.descs[j - 1];
      --j;
    }
    B.descs[j] = t;
  }
  for (uint32_t a = 0; a < R.nbCMs; ++a) {
    const uint32_t cmi = B.islandCMs[R.cmStart + a];
    const SolverCM& cm = B.cms[cmi];
    SDesc& d = B.descs[nbDescs++];
    d.bodyA = d.bodyADataIndex = B.bodySolverIndex[cm.body0];
    d.bodyB = d.bodyBDataIndex = cm.body1 == NONE ? 0u : B.bodySolverIndex[cm.body1];
    d.linkIndexA = d.linkIndexB = RIGID_BODY;
    d.constraint = NONE;
    d.constraintLengthOver16 = 0;
    d.constraintType = SC_TYPE_RB_CONTACT;
    d.source = cmi;
    d.sortKey = 0;
    d.progressA = d.progressB = 0;
  }
  // preIntegrateBodies
  uint32_t posIters = 0, velIters = 0;
  for (uint32_t k = 0; k < R.nbBodies; ++k) {
    Body& b = B.bodies[B.islandBodies[R.bodyStart + k]];
    const uint16_t iterWord = b.solverIterationCounts;
    posIters = pmaxu(uint32_t(iterWord & 0xff), posIters);
    velIters = pmaxu(uint32_t(iterWord >> 8), velIters);
    bodyCoreComputeUnconstrainedVelocity(prm.gravity, mDt, b.linDamping, b.angDamping, b.accelScale, b.maxLinVelSq, b.maxAngVelSq, b.linVel,
                                         b.angVel, b.disableGravity != 0);
    copyToSolverBodyDataStep(b.linVel, b.angVel, b.invMass, b.invInertia, b.body2World, b.maxPenBias, b.maxContactImpulse, 0, kMaxReal,
                             b.maxAngVelSq, b.lockFlags, false, vels[bodyOffset + k + 1], B.txI[bodyOffset + k + 1], B.datas[bodyOffset + k + 1],
                             mDt, b.gyroscopic != 0);
  }
  // SetStepperTask
  const float stepDt = mDt / float(posIters);
  const float invStepDt = 1.f / stepDt;
  const float biasCoefficient = 2.f * psqrt(1.f / float(posIters));
  // PartitionTask
  PartitionView pv{vels, bodyOffset + 1, R.nbBodies};
  uint32_t countsSize = 0, numOverflows = 0, numStatic = 0, numOrdered = 0;
  const uint32_t maxPartitions = partitionContactConstraints(pv, B.descs, nbDescs, B.ordered, B.temp, B.partitionCounts, countsSize,
                                                             B.partitionCap, 64, numOverflows, numStatic, numOrdered, B.error);
  if (B.error) return;
  uint32_t* acc = B.partitionCounts;  // mConstraintsPerPartition
  const uint32_t descCount = numOrdered;
  uint32_t numHeaders = 0, currentPartition = 0, headersPerPartition = 0;
  uint32_t maxJ = descCount == 0 ? 0 : acc[0];
  uint32_t maxBatchSize = numOverflows == 0 ? 4u : 1u;
  for (uint32_t a = 0; a < descCount;) {
    const uint32_t loopMax = pminu(maxJ - a, maxBatchSize);
    uint32_t j = 0;
    if (loopMax > 0) {
      BatchHeader& header = B.headers[numHeaders++];
      j = 1;
      const SDesc& desc = B.ordered[a];
      if (desc.constraintType == SC_TYPE_RB_CONTACT || desc.constraintType == SC_TYPE_RB_1D)  // 관절체 제약 아님
        for (; j < loopMax && desc.constraintType == B.ordered[a + j].constraintType; ++j)
          ;
      header.startIndex = a;
      header.stride = uint16_t(j);
      header.constraintType = desc.constraintType;
      headersPerPartition++;
    }
    if (maxJ == (a + j) && maxJ != descCount) {
      acc[currentPartition] = headersPerPartition;
      headersPerPartition = 0;
      currentPartition++;
      maxJ = acc[currentPartition];
      maxBatchSize = 4u;
    }
    a += j;
  }
  if (descCount) acc[currentPartition] = headersPerPartition;
  const uint32_t nbPartitions = maxPartitions;  // forceSize_Unsafe(mMaxPartitions)
  const bool hasOverflowPartitions0 = numOverflows != 0;
  // SetupSolverConstraintsTask -> createSolverConstraints
  B.constraints.size = 0;
  PrepCtx P{vels, B.txI, B.datas, &B.constraints, &B.friction[B.frictionCurIdx], B.corr, B.contactBuffer, 0};
  const FrictionArena& fprev = B.friction[B.frictionCurIdx ^ 1u];
  const float totalDt = mDt;
  const float invTotalDt = 1.0f / totalDt;
  for (uint32_t h = 0; h < numHeaders; ++h) {
    BatchHeader& hdr = B.headers[h];
    const uint32_t startIdx = hdr.startIndex, endIdx = startIdx + hdr.stride;
    if (B.ordered[startIdx].constraintType == SC_TYPE_RB_1D) {
      prepare1DHeader(B, prm, hdr, stepDt, totalDt, invStepDt, invTotalDt, biasCoefficient);
      continue;
    }
    if (B.ordered[startIdx].constraintType != SC_TYPE_RB_CONTACT) {
      B.error |= SV_ERR_UNSUPPORTED;
      continue;
    }
    TGSContactDesc blockDescs[4];
    CMOutput outs[4];
    const CMOutput* outPtrs[4];
    for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, i++) {
      SDesc& desc = B.ordered[a];
      SolverCM& cm = B.cms[desc.source];
      TGSContactDesc& bd = blockDescs[i];
      outs[i] = CMOutput{B.patches + cm.patchStart, B.contacts + cm.contactStart, cm.nbPatches, cm.nbContacts};
      outPtrs[i] = &outs[i];
      bd.b0 = desc.bodyADataIndex;
      bd.b1 = desc.bodyBDataIndex;
      bd.bodyFrame0 = B.bodies[cm.body0].body2World;
      bd.bodyFrame1 = cm.body1 == NONE ? cm.staticPose1 : B.bodies[cm.body1].body2World;
      bd.desc = &desc;
      bd.hasForceThresholds = !!(cm.npFlags & NP_FORCE_THRESHOLD);
      bd.disableStrongFriction = !!(cm.npFlags & NP_DISABLE_STRONG_FRICTION);
      bd.bodyState0 = (cm.npFlags & NP_ARTICULATION_BODY0) ? BS_ARTICULATION : BS_DYNAMIC;
      bd.bodyState1 = (cm.npFlags & NP_HAS_KINEMATIC_ACTOR) ? BS_KINEMATIC : ((cm.npFlags & NP_DYNAMIC_BODY1) ? BS_DYNAMIC : BS_STATIC);
      const float maxImpulse0 = B.datas[bd.b0].maxContactImpulse;
      const float maxImpulse1 = B.datas[bd.b1].maxContactImpulse;
      const float dom0 = (cm.npFlags & NP_DOMINANCE_0) ? 0.0f : 1.0f;
      const float dom1 = (cm.npFlags & NP_DOMINANCE_1) ? 0.0f : 1.0f;
      bd.invMassScales[0] = bd.invMassScales[1] = dom0;
      bd.invMassScales[2] = bd.invMassScales[3] = dom1;
      bd.restDistance = cm.restDistance;
      bd.frictionPrev = cm.frictionCount ? fprev.data + cm.frictionPtr : nullptr;
      bd.frictionPrevCount = cm.frictionCount;
      bd.frictionPtr = NONE;
      bd.frictionCount = 0;
      bd.maxCCDSeparation = kMaxReal;
      bd.maxImpulse = pmin(maxImpulse0, maxImpulse1);
      bd.torsionalPatchRadius = cm.torsionalPatchRadius;
      bd.minTorsionalPatchRadius = cm.minTorsionalPatchRadius;
      bd.offsetSlop = cm.offsetSlop;
      bd.hasMaxImpulse = false;
      bd.axisConstraintCount = 0;
    }
    uint32_t buildState = PREP_UNBATCHABLE;
    B.statHeaders++;
    if (hdr.stride == 4)
      buildState = createFinalizeSolverContacts4Step(P, outPtrs, blockDescs, invStepDt, totalDt, invTotalDt, stepDt, prm.bounceThreshold,
                                                     prm.frictionOffsetThreshold, prm.correlationDistance, biasCoefficient);
    if (buildState == PREP_SUCCESS) B.statBlock4++;
    if (buildState != PREP_SUCCESS) {
      B.statSingle += hdr.stride;
      for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, i++)
        createFinalizeSolverContactsStep(P, blockDescs[i], *outPtrs[i], invStepDt, invTotalDt, totalDt, stepDt, prm.bounceThreshold,
                                         prm.frictionOffsetThreshold, prm.correlationDistance, biasCoefficient);
    }
    for (uint32_t i = 0; i < hdr.stride; ++i) {
      SolverCM& cm = B.cms[B.ordered[startIdx + i].source];
      cm.frictionPtr = blockDescs[i].frictionPtr;  // 다음 스텝에 읽을 패치 (이번 arena)
      cm.frictionCount = blockDescs[i].frictionCount;
    }
  }
  if (B.constraints.overflow) B.error |= SV_ERR_ARENA;
  if (B.constraints.size > B.statMaxArena) B.statMaxArena = B.constraints.size;
  if (B.friction[B.frictionCurIdx].size > B.statMaxFriction) B.statMaxFriction = B.friction[B.frictionCurIdx].size;
  if (nbDescs > B.statMaxDescs) B.statMaxDescs = nbDescs;
  if (B.friction[B.frictionCurIdx].overflow) B.error |= SV_ERR_FRICTION;
  // SolveIslandTask: 길이 0 인 제약 빼기
  uint32_t jj = 0, ii = 0, numBatches = 0, currIndex = 0, totalCount = 0, totalPartitions = 0;
  bool hasOverflowPartitions = hasOverflowPartitions0;
  SDesc* cdb = B.ordered;
  for (uint32_t a = 0; a < nbPartitions; ++a) {
    const uint32_t endIndex = currIndex + acc[a];
    uint32_t numBatchesInPartition = 0;
    for (uint32_t b = currIndex; b < endIndex; ++b) {
      BatchHeader& _header = B.headers[b];
      const uint16_t stride = _header.stride;
      uint16_t newStride = _header.stride;
      const uint32_t startIndex = jj;
      for (uint16_t c = 0; c < stride; ++c) {
        if (cdb[ii].constraintLengthOver16 == 0) {
          newStride--;
          ii++;
        } else {
          if (ii != jj) cdb[jj] = cdb[ii];
          ii++;
          jj++;
        }
      }
      if (newStride != 0) {
        B.headers[numBatches].startIndex = startIndex;
        B.headers[numBatches].stride = newStride;
        uint8_t type = *arenaPtr<uint8_t>(B.constraints, cdb[startIndex].constraint);
        if (type == SC_TYPE_STATIC_CONTACT) {
          for (uint32_t c = 1; c < newStride; ++c)
            if (*arenaPtr<uint8_t>(B.constraints, cdb[startIndex + c].constraint) == SC_TYPE_RB_CONTACT) type = SC_TYPE_RB_CONTACT;
        }
        B.headers[numBatches].constraintType = type;
        numBatches++;
        numBatchesInPartition++;
      }
    }
    currIndex += acc[a];
    acc[totalPartitions] = numBatchesInPartition;
    if (numBatchesInPartition)
      totalPartitions++;
    else if (a == 0)
      hasOverflowPartitions = false;
    totalCount += numBatchesInPartition;
  }
  (void)hasOverflowPartitions;
  const uint32_t numContactConstraintBatches = totalCount;
  B.statBatches++;
  if (numContactConstraintBatches == 0) B.statFreeBatches++;
  if (totalPartitions > B.statMaxPartitions) B.statMaxPartitions = totalPartitions;
  BatchPlan& pl = B.plan;
  pl.numBatches = numContactConstraintBatches;
  pl.totalPartitions = totalPartitions;
  // 넘침 제약은 첫 분할 앞에 들어간다(outputOverflowConstraints). 첫 분할이 비면 PhysX 도 넘침 없음으로 본다(위 hasOverflowPartitions).
  pl.firstSequential = (hasOverflowPartitions0 && totalPartitions && acc[0]) ? 1u : 0u;
  pl.bodyOffset = bodyOffset;
  pl.nbBodies = R.nbBodies;
  pl.posIters = posIters;
  pl.velIters = velIters;
  pl.dt = mDt;
  pl.invDt = 1.0f / prm.dt;
  pl.stepDt = stepDt;
  pl.ok = 1;
}

// 반복 한 번의 제약 풀기: 분할 순서대로, 분할 안 머리는 서로 몸체를 나누지 않으므로(세계 몸체 제외: 정적 쪽은 질량 0 이라 같은 값만 쓴다)
// 스레드 nt 개가 나눠 푼다. nt = 1 이면 PhysX 단일 스레드 순서 그대로.
SV_HD void solvePartitions(SolverBoard& B, const BatchPlan& pl, float minPen, float elapsed, bool conclude, uint32_t tid, uint32_t nt,
                           uint32_t& err) {
  uint32_t h0 = 0;
  for (uint32_t p = 0; p < pl.totalPartitions; ++p) {
    const uint32_t h1 = h0 + B.partitionCounts[p];
    if (nt == 1 || (p == 0 && pl.firstSequential)) {
      if (tid == 0)
        for (uint32_t h = h0; h < h1; ++h) solveContactHeader(B.headers[h], B.ordered, B.vels, B.txI, B.constraints, minPen, elapsed, conclude, err);
    } else {
      for (uint32_t h = h0 + tid; h < h1; h += nt) solveContactHeader(B.headers[h], B.ordered, B.vels, B.txI, B.constraints, minPen, elapsed, conclude, err);
    }
    SV_SYNC();
    h0 = h1;
  }
}
SV_HD void integrateBodies(SolverBoard& B, const BatchPlan& pl, float dt, uint32_t tid, uint32_t nt) {
  for (uint32_t k = tid; k < pl.nbBodies; k += nt) integrateCoreStep(B.vels[pl.bodyOffset + k + 1], B.txI[pl.bodyOffset + k + 1], dt);
  SV_SYNC();
}

// iterativeSolveIsland 의 반복 부분 (DyTGSDynamics.cpp:2515-2793). 판 안 스레드 전부가 부른다(nt = 1 이면 호스트 순서 그대로).
SV_HDN void solveBatchIterate(SolverBoard& B, uint32_t tid, uint32_t nt) {
  const BatchPlan pl = B.plan;
  if (!pl.ok) return;
  uint32_t err = 0;
  if (pl.numBatches == 0) {
    integrateBodies(B, pl, pl.dt, tid, nt);
  } else {
    float elapsedTime = 0.0f;
    for (uint32_t a = 1; a < pl.posIters; a++) {
      solvePartitions(B, pl, -kMaxReal, elapsedTime, false, tid, nt, err);
      integrateBodies(B, pl, pl.stepDt, tid, nt);
      elapsedTime += pl.stepDt;
    }
    {  // 마지막 위치 반복 (solveConclude: 접촉의 conclude 는 빈 함수, 1D 는 conclude1DStep)
      solvePartitions(B, pl, -kMaxReal, elapsedTime, true, tid, nt, err);
      elapsedTime += pl.stepDt;
      integrateBodies(B, pl, pl.stepDt, tid, nt);
    }
    for (uint32_t a = 0; a < pl.velIters; ++a) solvePartitions(B, pl, 0.f, elapsedTime, false, tid, nt, err);
  }
  if (err) svAtomicOr(B.error, err);
}

// 마무리 (차례로): 되쓰기, copyBackBodies
SV_HDN void solveBatchFinish(SolverBoard& B, const SolverParams& prm, const BatchRange& R) {
  const BatchPlan& pl = B.plan;
  if (!pl.ok) return;
  const uint32_t bodyOffset = pl.bodyOffset;
  const float mDt = pl.dt, mInvDt = pl.invDt;
  SBodyVel* vels = B.vels;
  SDesc* cdb = B.ordered;
  const uint32_t numContactConstraintBatches = pl.numBatches;
  if (numContactConstraintBatches != 0) {
    FrictionArena& fcur = B.friction[B.frictionCurIdx];
    for (uint32_t h = 0; h < numContactConstraintBatches; ++h) {
      const BatchHeader& hd = B.headers[h];
      if (hd.constraintType == SC_TYPE_BLOCK_RB_CONTACT || hd.constraintType == SC_TYPE_BLOCK_STATIC_RB_CONTACT)
        writeBackContact4_Block(cdb + hd.startIndex, B.constraints, fcur);
      else if (hd.constraintType == SC_TYPE_RB_1D)  // writeBack1D (DyTGSContactPrep.cpp:3353)
        for (uint32_t i = hd.startIndex, e = hd.startIndex + hd.stride; i < e; ++i)
          jnt::writeBack1DStep(arenaPtr<uint8_t>(B.constraints, cdb[i].constraint), &B.writebacks[B.c1d[cdb[i].source].writeback]);
      else if (hd.constraintType == SC_TYPE_BLOCK_1D) {  // writeBack1D4 (DyTGSContactPrepBlock.cpp:3418)
        jnt::Writeback* wb[4];
        for (int a = 0; a < 4; ++a) wb[a] = &B.writebacks[B.c1d[cdb[hd.startIndex + a].source].writeback];
        jnt::writeBack1D4(arenaPtr<uint8_t>(B.constraints, cdb[hd.startIndex].constraint), wb, false);
      } else
        for (uint32_t i = hd.startIndex, e = hd.startIndex + hd.stride; i < e; ++i) writeBackContact(cdb[i], B.constraints, fcur);
    }
  }
  // finishSolveIsland -> copyBackBodies (몸체 순서대로), hasStaticTouch = 몸체가 속한 섬의 정적 닿음 수
  uint32_t k = 0;
  for (uint32_t isl = R.islandStart; isl < R.islandEnd; ++isl) {
    const IslandIn& I = B.islands[isl];
    for (uint32_t m = 0; m < I.bodyCount; ++m, ++k) {
      Body& b = B.bodies[B.islandBodies[R.bodyStart + k]];
      copyBackBody(b, vels[bodyOffset + k + 1], B.txI[bodyOffset + k + 1], mInvDt, mDt, prm.enableStabilization, I.staticTouchCount != 0);
    }
  }
}

// 스텝 시작 (차례로, 판마다 한 번): 마찰 arena 교대, 마찰 수 0 되돌리기, 세계 몸체
SV_HDN void solverStepBegin(SolverBoard& B) {
  // 마찰 arena 교대: 지난 스텝 패치는 이제 prev
  B.frictionCurIdx ^= 1u;
  B.friction[B.frictionCurIdx].size = 0;
  B.friction[B.frictionCurIdx].overflow = 0;
  B.constraints.overflow = 0;
  for (uint32_t a = 0; a < B.nbResetCMs; ++a) B.cms[B.resetCMs[a]].frictionCount = 0;          // Sc 층 clearCachedState (풀이 전)
  for (uint32_t a = 0; a < B.nbActivatedCMs; ++a) B.cms[B.activatedCMs[a]].frictionCount = 0;  // resetFrictionPatchCount
  if (!B.nbIslands) return;
  // 세계 몸체 (풀 0): mWorldSolverBodyVel / TxInertia / Data2 (DyTGSDynamics.cpp:299-312, 596)
  {
    SBodyVel& w = B.vels[0];
    for (int c = 0; c < 3; ++c) w.lin[c] = w.ang[c] = w.deltaAngDt[c] = w.deltaLinDt[c] = 0.0f;
    w.maxDynamicPartition = w.nbStaticInteractions = 0;
    w.partitionMask = 0;
    w.maxAngVel = 0.0f;
    w.lockFlags = 0;
    w.isKinematic = 0;
    w.pad = 0;
    B.txI[0].sqrtInvInertia = M33{V3{0, 0, 0}, V3{0, 0, 0}, V3{0, 0, 0}};
    B.txI[0].body2WorldP = V3{0, 0, 0};
    B.txI[0].deltaBody2WorldQ = qid();
    SBodyData& d = B.datas[0];
    d.penBiasClamp = -kMaxReal;
    d.maxContactImpulse = kMaxReal;
    d.nodeIndex = NONE;
    d.invMass = 0;
    d.reportThreshold = kMaxReal;
    d.originalLinearVelocity = V3{0, 0, 0};
    d.originalAngularVelocity = V3{0, 0, 0};
  }
}

// 섬 묶음 나누기와 묶음별 준비·반복·마무리. 판 안 스레드 전부가 같은 순서로 부른다(묶음 나누기는 읽기만 해서 모두 같은 값을 얻는다).
SV_HDN void solverIslands(SolverBoard& B, const SolverParams& prm, uint32_t tid, uint32_t nt) {
  const uint32_t kinematicCount = 0;  // 운동학 몸체: 아직
  uint32_t currentIsland = 0, currentBodyIndex = 0, currentContact = 0;
  while (currentIsland < B.nbIslands) {
    BatchRange R;
    R.islandStart = currentIsland;
    R.bodyStart = B.islands[currentIsland].bodyStart;
    R.cmStart = B.islands[currentIsland].cmStart;
    uint32_t nbBodies = 0, nbArticulations = 0, nbCMs = 0;
    while (nbBodies < prm.solverBatchSize && currentIsland < B.nbIslands && nbArticulations < prm.solverArticBatchSize) {
      nbBodies += B.islands[currentIsland].bodyCount;
      nbCMs += B.islands[currentIsland].cmCount;
      currentIsland++;
    }
    R.islandEnd = currentIsland;
    R.nbBodies = nbBodies;
    R.nbCMs = nbCMs;
    R.solverBodyOffset = kinematicCount + currentBodyIndex;
    if (tid == 0) solveBatchPrep(B, prm, R, kinematicCount);
    SV_SYNC();
    solveBatchIterate(B, tid, nt);
    if (tid == 0) solveBatchFinish(B, prm, R);
    SV_SYNC();
    currentBodyIndex += nbBodies;
    currentContact += nbCMs;
  }
  (void)currentContact;
}

// DynamicsTGSContext::update + updatePostKinematic (호스트·GPU 한 스레드)
SV_HDN void solverStep(SolverBoard& B, const SolverParams& prm) {
  solverStepBegin(B);
  if (B.nbIslands) solverIslands(B, prm, 0, 1);
}

// 같은 스텝을 판 안 스레드 nt 개로 (GPU: B 는 공유 메모리의 판 하나, 스레드 전부가 부른다). 결과는 solverStep 과 비트 같다.
SV_HDN void solverStepPar(SolverBoard& B, const SolverParams& prm, uint32_t tid, uint32_t nt) {
  if (tid == 0) solverStepBegin(B);
  SV_SYNC();
  if (B.nbIslands) solverIslands(B, prm, tid, nt);
}

// Sc::Scene::afterIntegration (ScPipeline.cpp:2640-2690): 이번 스텝 섬 관리자가 재운 몸체는 풀이에서 적분됐더라도
// 스텝 시작 자세로 되돌리고 깸 카운터·속도를 0 으로 (섬 생성이 풀이보다 먼저 돌던 예전 동작을 흉내 내는 PhysX 규칙).
// deactivated = 이번 스텝 IslandSim::getNodesToDeactivate(eRIGID_BODY) 의 몸체 번호 (섬 관리가 만든다).
SV_HDN void deactivateBodies(SolverBoard& B, const uint32_t* deactivated, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    Body& b = B.bodies[deactivated[i]];
    b.body2World = b.lastTransform;  // rigid->setPose(rigid->getLastCCDTransform())
    b.wakeCounter = 0.0f;
    b.linVel = V3{0.0f, 0.0f, 0.0f};
    b.angVel = V3{0.0f, 0.0f, 0.0f};
    b.internalFlags &= uint16_t(~(RB_FREEZE_THIS_FRAME | RB_UNFREEZE_THIS_FRAME | RB_ACTIVATE_THIS_FRAME | RB_DEACTIVATE_THIS_FRAME));
  }
}

// ScAfterIntegrationTask (ScScene.cpp:176): 활성 몸체의 깸 카운터 확정 + 프레임 플래그 지우기
SV_HDN void afterIntegration(SolverBoard& B) {
  for (uint32_t i = 0; i < B.nbIslands; ++i) {
    const IslandIn& I = B.islands[i];
    for (uint32_t k = 0; k < I.bodyCount; ++k) {
      Body& b = B.bodies[B.islandBodies[I.bodyStart + k]];
      b.wakeCounter = b.solverWakeCounter;
      b.internalFlags &= uint16_t(~(RB_FREEZE_THIS_FRAME | RB_UNFREEZE_THIS_FRAME | RB_ACTIVATE_THIS_FRAME | RB_DEACTIVATE_THIS_FRAME));
    }
  }
}

}  // namespace sv
}  // namespace eng
