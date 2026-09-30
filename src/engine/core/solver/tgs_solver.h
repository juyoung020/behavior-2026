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
// 관절체(09-30): 섬의 관절체 노드·링크 접촉(ext·정적)·링크 조인트를 PhysX 단일 스레드 경로 순서로 (art_couple.h, docs 17.6).
// 아직 안 옮김: 운동학 몸체, CCD,
//              externalForcesEveryTgsIteration(기본 꺼짐), 잔차 보고.
#pragma once
#include <cstring>

#include "../joints/tgs_1d.h"
#include "../joints/tgs_1d4.h"
#include "art_couple.h"
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

// ---------------- 분할, 관절체가 있는 묶음 (DyConstraintPartition.cpp:291 ExtendedRigidBodyClassification, forceStaticCollisionsToSolver = false)
// 링크-정적(또는 운동학) 제약은 관절체 정적 목록으로 간다(storeStaticConstraint, DY_STATIC_CONTACTS_IN_INTERNAL_SOLVER = true).
struct ExtClass {
  SBodyVel* vels;
  uint32_t base, count;
  ArtProgress* prog;
  const uint32_t* batchIdx;
  SV_HD bool rigidActive(uint32_t i) const { return i - base < count; }
  // classifyConstraint (:367). 반환 = 정적 없음
  SV_HD bool classify(const SDesc& d, bool& activeA, bool& activeB, uint32_t& pA, uint32_t& pB) const {
    bool hasStatic = false;
    if (d.linkIndexA == RIGID_BODY) {
      activeA = rigidActive(d.bodyA);
      hasStatic = !activeA;
      pA = activeA ? vels[d.bodyA].partitionMask : 0;
    } else {
      pA = prog[batchIdx[d.bodyA]].partitionMask;
      activeA = true;
    }
    if (d.linkIndexB == RIGID_BODY) {
      activeB = rigidActive(d.bodyB);
      hasStatic = hasStatic || !activeB;
      pB = activeB ? vels[d.bodyB].partitionMask : 0;
    } else {
      activeB = true;
      pB = prog[batchIdx[d.bodyB]].partitionMask;
    }
    return !hasStatic;
  }
  SV_HD void storeProgress(const SDesc& d, uint32_t pA, uint32_t pB, uint16_t avail) {  // storeRigidBodyProgress / storeArticulationProgress
    if (d.linkIndexA == RIGID_BODY) {
      vels[d.bodyA].partitionMask = pA;
      vels[d.bodyA].maxDynamicPartition = uint16_t(pmaxu(vels[d.bodyA].maxDynamicPartition, avail));
    } else {
      ArtProgress& p = prog[batchIdx[d.bodyA]];
      p.partitionMask = pA;
      p.maxDynamicPartition = uint16_t(pmaxu(p.maxDynamicPartition, avail));
    }
    if (d.linkIndexB == RIGID_BODY) {
      vels[d.bodyB].partitionMask = pB;
      vels[d.bodyB].maxDynamicPartition = uint16_t(pmaxu(vels[d.bodyB].maxDynamicPartition, avail));
    } else {
      ArtProgress& p = prog[batchIdx[d.bodyB]];
      p.partitionMask = pB;
      p.maxDynamicPartition = uint16_t(pmaxu(p.maxDynamicPartition, avail));
    }
  }
  SV_HD void storeProgress_(const SDesc& d, uint32_t pA, uint32_t pB) {
    if (d.linkIndexA == RIGID_BODY) vels[d.bodyA].partitionMask = pA;
    else prog[batchIdx[d.bodyA]].partitionMask = pA;
    if (d.linkIndexB == RIGID_BODY) vels[d.bodyB].partitionMask = pB;
    else prog[batchIdx[d.bodyB]].partitionMask = pB;
  }
};

// 관절체 정적 목록에 넣기 (FeatherstoneArticulation::storeStaticConstraint, DyFeatherstoneArticulation.cpp:5337)
SV_HD bool artStoreStatic(SolverBoard& B, uint32_t artIdx, const SDesc& d) {
  const bool contact = d.constraintType == SC_TYPE_RB_CONTACT;
  uint32_t& n = contact ? B.artNbStaticContact[artIdx] : B.artNbStatic1D[artIdx];
  if (n >= B.artStaticCap) {
    B.error |= SV_ERR_DESC;
    return false;
  }
  (contact ? B.artStaticContact : B.artStatic1D)[artIdx * B.artStaticCap + n++] = d;
  return true;
}

SV_HDN uint32_t partitionContactConstraintsExt(SolverBoard& B, ExtClass& X, uint32_t nbArts, const SDesc* descs, uint32_t numDescs, SDesc* ordered,
                                              SDesc* overflowTmp, uint32_t* counts, uint32_t& countsSize, uint32_t countsCap, uint32_t maxPartitions,
                                              uint32_t& numOverflows, uint32_t& numStaticConstraints, uint32_t& numOrdered, uint32_t& err) {
  const uint32_t MAX_NUM_PARTITIONS = 32;
  auto clearState = [&]() {
    for (uint32_t k = 0; k < X.count; ++k) X.vels[X.base + k].partitionMask = 0;
    for (uint32_t a = 0; a < nbArts; ++a) X.prog[a].partitionMask = 0;
  };
  // zeroBodies
  for (uint32_t k = 0; k < X.count; ++k) {
    SBodyVel& b = X.vels[X.base + k];
    b.partitionMask = 0;
    b.nbStaticInteractions = 0;
    b.maxDynamicPartition = 0;
  }
  for (uint32_t a = 0; a < nbArts; ++a) X.prog[a] = ArtProgress{0, 0, 0};
  // classifyConstraintDesc
  uint32_t numUnpartitioned = 0;
  countsSize = MAX_NUM_PARTITIONS;
  for (uint32_t a = 0; a < MAX_NUM_PARTITIONS; ++a) counts[a] = 0;
  for (uint32_t i = 0; i < numDescs; ++i) {
    const SDesc& d = descs[i];
    bool activeA, activeB;
    uint32_t pA, pB;
    if (X.classify(d, activeA, activeB, pA, pB)) {
      uint32_t avail;
      if (!computeAvailablePartition(avail, pA, pB, activeA, activeB)) {
        overflowTmp[numUnpartitioned++] = d;
        continue;
      }
      counts[avail]++;
      avail++;
      X.storeProgress(d, pA, pB, uint16_t(avail));
    } else {  // recordStaticConstraint: 강체만 센다 (관절체는 willStoreStaticConstraint)
      if (activeA && d.linkIndexA == RIGID_BODY) X.vels[d.bodyA].nbStaticInteractions++;
      if (activeB && d.linkIndexB == RIGID_BODY) X.vels[d.bodyB].nbStaticInteractions++;
    }
  }
  uint32_t partitionStartIndex = 0;
  while (numUnpartitioned > 0) {
    clearState();
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
      bool activeA, activeB;
      uint32_t pA, pB;
      X.classify(d, activeA, activeB, pA, pB);
      uint32_t avail;
      if (!computeAvailablePartition(avail, pA, pB, activeA, activeB)) {
        overflowTmp[newNum++] = d;
        continue;
      }
      avail += partitionStartIndex;
      counts[avail]++;
      avail++;
      X.storeProgress(d, pA, pB, uint16_t(avail));
    }
    numUnpartitioned = newNum;
  }
  // reserveSpaceForStaticConstraints (강체 + 관절체: 관절체의 정적 수는 늘 0)
  for (uint32_t k = 0; k < X.count; ++k) {
    SBodyVel& b = X.vels[X.base + k];
    b.partitionMask = 0;
    const uint32_t requiredSize = uint32_t(b.maxDynamicPartition + b.nbStaticInteractions);
    if (requiredSize > countsSize) {
      if (requiredSize > countsCap) {
        err |= SV_ERR_PARTITION;
        return 0;
      }
      for (uint32_t a = countsSize; a < requiredSize; ++a) counts[a] = 0;
      countsSize = requiredSize;
    }
    for (uint32_t bb = 0; bb < b.nbStaticInteractions; bb++) counts[b.maxDynamicPartition + bb]++;
  }
  for (uint32_t a = 0; a < nbArts; ++a) {
    ArtProgress& p = X.prog[a];
    p.partitionMask = 0;
    const uint32_t requiredSize = uint32_t(p.maxDynamicPartition + p.nbStaticInteractions);
    if (requiredSize > countsSize) {
      if (requiredSize > countsCap) {
        err |= SV_ERR_PARTITION;
        return 0;
      }
      for (uint32_t c = countsSize; c < requiredSize; ++c) counts[c] = 0;
      countsSize = requiredSize;
    }
    for (uint32_t bb = 0; bb < p.nbStaticInteractions; bb++) counts[p.maxDynamicPartition + bb]++;
  }
  numOverflows = numUnpartitioned;
  uint32_t accumulation = 0;
  for (uint32_t a = 0; a < countsSize; a++) {
    const uint32_t count = counts[a];
    counts[a] = accumulation;
    accumulation += count;
  }
  // afterClassification
  for (uint32_t k = 0; k < X.count; ++k) {
    SBodyVel& b = X.vels[X.base + k];
    b.partitionMask = 0;
    b.nbStaticInteractions = 0;
  }
  for (uint32_t a = 0; a < nbArts; ++a) {
    X.prog[a].partitionMask = 0;
    X.prog[a].nbStaticInteractions = 0;
  }
  // writeConstraintDesc
  numUnpartitioned = 0;
  numStaticConstraints = 0;
  for (uint32_t i = 0; i < numDescs; ++i) {
    const SDesc& d = descs[i];
    bool activeA, activeB;
    uint32_t pA, pB;
    if (X.classify(d, activeA, activeB, pA, pB)) {
      uint32_t avail;
      if (!computeAvailablePartition(avail, pA, pB, activeA, activeB)) {
        overflowTmp[numUnpartitioned++] = d;
        continue;
      }
      X.storeProgress(d, pA, pB, uint16_t(avail + 1));
      ordered[numOverflows + counts[avail]++] = d;
    } else {  // getStaticContactWriteIndex
      uint32_t index = NONE;
      if (activeA) {
        if (d.linkIndexA == RIGID_BODY) {
          SBodyVel& A = X.vels[d.bodyA];
          index = uint32_t(A.maxDynamicPartition + A.nbStaticInteractions++);
        } else if (!artStoreStatic(B, d.bodyA, d)) {
          err |= SV_ERR_DESC;
        }
      } else if (activeB) {
        if (d.linkIndexB == RIGID_BODY) {
          SBodyVel& Bv = X.vels[d.bodyB];
          index = uint32_t(Bv.maxDynamicPartition + Bv.nbStaticInteractions++);
        } else if (!artStoreStatic(B, d.bodyB, d)) {
          err |= SV_ERR_DESC;
        }
      }
      if (index != NONE)
        ordered[numOverflows + counts[index]++] = d;
      else
        numStaticConstraints++;
    }
  }
  partitionStartIndex = 0;
  while (numUnpartitioned > 0) {
    clearState();
    partitionStartIndex += MAX_NUM_PARTITIONS;
    if (partitionStartIndex >= maxPartitions) break;
    uint32_t newNum = 0;
    for (uint32_t i = 0; i < numUnpartitioned; ++i) {
      const SDesc d = overflowTmp[i];
      bool activeA, activeB;
      uint32_t pA, pB;
      X.classify(d, activeA, activeB, pA, pB);
      uint32_t avail;
      if (!computeAvailablePartition(avail, pA, pB, activeA, activeB)) {
        overflowTmp[newNum++] = d;
        continue;
      }
      X.storeProgress_(d, pA, pB);
      avail += partitionStartIndex;
      ordered[numOverflows + counts[avail]++] = d;
    }
    numUnpartitioned = newNum;
  }
  numOrdered = numDescs - numStaticConstraints;  // extended
  if (numOverflows) {
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
// bitCopy 는 art_couple.h

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
// list = 기술자 배열(묶음 머리면 B.ordered, 관절체 정적 목록이면 그 목록), stride = 1..4
SV_HDN void prepare1DHeader(SolverBoard& B, const SolverParams& prm, SDesc* list, uint32_t startIdx, uint32_t stride, float stepDt, float totalDt,
                            float invStepDt, float invTotalDt, float biasCoefficient) {
  const uint32_t endIdx = startIdx + stride;
  const Tf ident{qid(), V3{0.0f, 0.0f, 0.0f}};  // PxTransform(PxIdentity): 정적 쪽 몸체 틀
  const jnt::D6Data* data[4];
  uint16_t flags[4];
  float linBreak[4], angBreak[4], minResp[4];
  Tf frame0[4], frame1[4];
  jnt::TgsBodyVel bv0[4], bv1[4];
  jnt::TgsTxInertia t0[4], t1[4];
  jnt::TgsBodyData d0[4], d1[4];
  for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, ++i) {
    const SDesc& desc = list[a];
    const Constraint1DIn& c = B.c1d[desc.source];
    data[i] = &B.jointData[c.data];
    flags[i] = c.flags;
    linBreak[i] = c.linBreakForce;
    angBreak[i] = c.angBreakForce;
    minResp[i] = c.minResponseThreshold;
    // constraint->body0->getPose(): 관절체 링크면 링크 몸체 자세(PxsBodyCore::body2World)
    frame0[i] = c.body0 == NONE ? ident : (c.artLink0 ? artAt(B, c.body0).bodies[c.artLink0 - 1].body2World : B.bodies[c.body0].body2World);
    frame1[i] = c.body1 == NONE ? ident : (c.artLink1 ? artAt(B, c.body1).bodies[c.artLink1 - 1].body2World : B.bodies[c.body1].body2World);
    // 관절체 쪽 desc.tgsBodyA 는 관절체 포인터(풀이 몸체 아님) — 강체 칸만 읽으므로 세계 몸체(0)를 넣는다
    bv0[i] = bitCopy<jnt::TgsBodyVel>(B.vels[desc.linkIndexA == RIGID_BODY ? desc.bodyA : 0u]);
    bv1[i] = bitCopy<jnt::TgsBodyVel>(B.vels[desc.linkIndexB == RIGID_BODY ? desc.bodyB : 0u]);
    t0[i] = bitCopy<jnt::TgsTxInertia>(B.txI[desc.bodyADataIndex]);
    t1[i] = bitCopy<jnt::TgsTxInertia>(B.txI[desc.bodyBDataIndex]);
    d0[i] = bitCopy<jnt::TgsBodyData>(B.datas[desc.bodyADataIndex]);
    d1[i] = bitCopy<jnt::TgsBodyData>(B.datas[desc.bodyBDataIndex]);
  }
  if (stride == 4) {  // PX_USE_BLOCK_1D
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
        list[a].constraint = off;
        list[a].constraintLengthOver16 = uint16_t(len / 16);
      }
      B.constraints.size = off + len + 16u;
      B.stat1DBlock4++;
      return;
    }
    B.constraints.size = prev;  // eUNBATCHABLE: 하나씩
  }
  const jnt::NoArt na;
  for (uint32_t a = startIdx, i = 0; a < endIdx; ++a, ++i) {
    SDesc& desc = list[a];
    const bool ext = isArtDesc(desc);
    const uint32_t prev = B.constraints.size;
    const uint32_t off = arenaAlloc(B.constraints, jnt::blockLength(jnt::MAX_CONSTRAINT_ROWS, ext) + 16u);
    if (off == NONE) {
      B.error |= SV_ERR_ARENA;
      return;
    }
    uint32_t len = 0;
    uint32_t n;
    if (!ext) {
      n = jnt::prepareD6Step(*data[i], flags[i], linBreak[i], angBreak[i], minResp[i], frame0[i], frame1[i], bv0[i], bv1[i], t0[i], t1[i], d0[i], d1[i],
                             jnt::RIGID_BODY, jnt::RIGID_BODY, B.rowScratch, arenaPtr<uint8_t>(B.constraints, off), stepDt, totalDt, invStepDt,
                             invTotalDt, prm.lengthScale, biasCoefficient, na, na, &len);
    } else {  // 관절체 링크가 낀 조인트: SolverExtBodyStep (응답은 articulation 모듈)
      const art::ArtRef ra{desc.linkIndexA == RIGID_BODY ? nullptr : &artAt(B, desc.bodyA)};
      const art::ArtRef rb{desc.linkIndexB == RIGID_BODY ? nullptr : &artAt(B, desc.bodyB)};
      n = jnt::prepareD6Step(*data[i], flags[i], linBreak[i], angBreak[i], minResp[i], frame0[i], frame1[i], bv0[i], bv1[i], t0[i], t1[i], d0[i], d1[i],
                             desc.linkIndexA, desc.linkIndexB, B.rowScratch, arenaPtr<uint8_t>(B.constraints, off), stepDt, totalDt, invStepDt,
                             invTotalDt, prm.lengthScale, biasCoefficient, ra, rb, &len);
      B.statArtExt1D++;
    }
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
  uint32_t artStart, nbArts;     // islandArts 안 (관절체, 섬 노드 사슬 순서)
};

SV_HDN void solveContactHeader(SolverBoard& B, const BatchHeader& h, float minPen, float elapsed, bool conclude, bool posIter, uint32_t& err) {
  const SDesc* ordered = B.ordered;
  SBodyVel* vels = B.vels;
  const SBodyTxI* txI = B.txI;
  ByteArena& arena = B.constraints;
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
    case SC_TYPE_EXT_CONTACT:  // solveExtContactBlock / solveConcludeContactExtBlock (concludeContactStep 은 빈 함수)
      for (uint32_t i = h.startIndex, e = h.startIndex + h.stride; i < e; ++i) solveExtContactDesc(ordered[i], vels, B, arena, minPen, elapsed);
      break;
    case SC_TYPE_EXT_1D:  // solveExt1DBlock / solveConclude1DBlockExt
      for (uint32_t i = h.startIndex, e = h.startIndex + h.stride; i < e; ++i)
        solveExt1DDesc(ordered[i], vels, txI, B, arena, elapsed, posIter, conclude);
      break;
    default:
      err |= SV_ERR_UNSUPPORTED;
  }
}

// 관절체 링크가 낀 접촉 하나 준비 (createSolverConstraints 의 RB_CONTACT 갈래 :1044-1180, prepareStaticConstraintsTGS :2175-2254 공통).
// 관절체 쪽 desc.tgsBody 는 관절체 포인터 -> 풀이 몸체 자리는 세계(0)로, 자세·최대 충격은 링크 몸체(PxsBodyCore)에서.
SV_HDN void prepareContactExtDesc(SolverBoard& B, const SolverParams& prm, SDesc& desc, PrepCtx& P, const FrictionArena& fprev, float stepDt,
                                  float totalDt, float invStepDt, float invTotalDt, float biasCoefficient) {
  SolverCM& cm = B.cms[desc.source];
  const CMOutput out{B.patches + cm.patchStart, B.contacts + cm.contactStart, cm.nbPatches, cm.nbContacts};
  art::Articulation* a0 = desc.linkIndexA == RIGID_BODY ? nullptr : &artAt(B, desc.bodyA);
  art::Articulation* a1 = desc.linkIndexB == RIGID_BODY ? nullptr : &artAt(B, desc.bodyB);
  const uint32_t v0 = a0 ? 0u : desc.bodyA, v1 = a1 ? 0u : desc.bodyB;
  TGSContactDesc bd;
  bd.b0 = desc.bodyADataIndex;
  bd.b1 = desc.bodyBDataIndex;
  bd.bodyFrame0 = a0 ? a0->bodies[desc.linkIndexA].body2World : B.bodies[cm.body0].body2World;  // unit.mRigidCore0->body2World
  bd.bodyFrame1 = a1 ? a1->bodies[desc.linkIndexB].body2World : (cm.body1 == NONE ? cm.staticPose1 : B.bodies[cm.body1].body2World);
  bd.desc = &desc;
  bd.hasForceThresholds = !!(cm.npFlags & NP_FORCE_THRESHOLD);
  bd.disableStrongFriction = !!(cm.npFlags & NP_DISABLE_STRONG_FRICTION);
  bd.bodyState0 = (cm.npFlags & NP_ARTICULATION_BODY0) ? BS_ARTICULATION : BS_DYNAMIC;
  bd.bodyState1 = (cm.npFlags & NP_ARTICULATION_BODY1)
                      ? BS_ARTICULATION
                      : ((cm.npFlags & NP_HAS_KINEMATIC_ACTOR) ? BS_KINEMATIC : ((cm.npFlags & NP_DYNAMIC_BODY1) ? BS_DYNAMIC : BS_STATIC));
  const float maxImpulse0 = (cm.npFlags & NP_ARTICULATION_BODY0) ? a0->bodies[desc.linkIndexA].maxContactImpulse : B.datas[bd.b0].maxContactImpulse;
  const float maxImpulse1 = (cm.npFlags & NP_ARTICULATION_BODY1) ? a1->bodies[desc.linkIndexB].maxContactImpulse : B.datas[bd.b1].maxContactImpulse;
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
  const ExtBody e0{a0, &B.vels[v0], &B.txI[bd.b0], &B.datas[bd.b0], desc.linkIndexA};
  const ExtBody e1{a1, &B.vels[v1], &B.txI[bd.b1], &B.datas[bd.b1], desc.linkIndexB};
  // P.vels[b0].isKinematic 은 관절체 쪽이면 읽지 않는다(bodyState 검사) — b0 = 0(세계)
  createFinalizeSolverContactsStepExt(P, bd, out, e0, e1, invStepDt, invTotalDt, totalDt, stepDt, prm.bounceThreshold, prm.frictionOffsetThreshold,
                                      prm.correlationDistance, biasCoefficient);
  cm.frictionPtr = bd.frictionPtr;
  cm.frictionCount = bd.frictionCount;
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
      if (c.artLink0) {  // setDescFromIndices_Constraints (DyTGSDynamics.cpp:383): articulationA, linkIndexA, bodyADataIndex = 0
        d.bodyA = c.body0;
        d.bodyADataIndex = 0;
        d.linkIndexA = c.artLink0 - 1;
      }
      if (c.artLink1) {
        d.bodyB = c.body1;
        d.bodyBDataIndex = 0;
        d.linkIndexB = c.artLink1 - 1;
      }
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
    d.linkIndexA = d.linkIndexB = RIGID_BODY;
    if (cm.artLink0) {  // setDescFromIndices_Contacts (DyTGSDynamics.cpp:330)
      d.bodyA = cm.body0;
      d.bodyADataIndex = 0;
      d.linkIndexA = cm.artLink0 - 1;
    } else {
      d.bodyA = d.bodyADataIndex = B.bodySolverIndex[cm.body0];
    }
    if (cm.artLink1) {
      d.bodyB = cm.body1;
      d.bodyBDataIndex = 0;
      d.linkIndexB = cm.artLink1 - 1;
    } else {
      d.bodyB = d.bodyBDataIndex = cm.body1 == NONE ? 0u : B.bodySolverIndex[cm.body1];
    }
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
  // SetupArticulationTask (DyTGSDynamics.cpp:1672, ArticulationTask :1645): 정적 목록 비우기(computeUnconstrainedVelocitiesInternal
  // ForwardDynamic.cpp:1811-1815) + 제약 없는 속도, 반복 수는 몸체·관절체 최댓값
  const float invLengthScale = 1.f / prm.lengthScale;
  for (uint32_t k = 0; k < R.nbArts; ++k) {
    const uint32_t ai = B.islandArts[R.artStart + k];
    art::Articulation& a = artAt(B, ai);
    a.awake = 1;              // 활성 섬에 있다 = 깨어 있음 (Sc 층 깨움은 호출자가 깸 카운터로 넣는다)
    B.artBatchIndex[ai] = k;  // mArticulationIndex
    const uint16_t iterWord = a.solverIterationCounts;
    velIters = pmaxu(uint32_t(iterWord >> 8), velIters);
    posIters = pmaxu(uint32_t(iterWord & 0xff), posIters);
    B.artNbStatic1D[ai] = 0;
    B.artNbStaticContact[ai] = 0;
    art::clearStaticLists(B.artLists[ai], a.nLinks);
    art::computeUnconstrainedVelocitiesTGS(a, mDt, prm.gravity, invLengthScale, false);
  }
  // SetStepperTask
  const float stepDt = mDt / float(posIters);
  const float invStepDt = 1.f / stepDt;
  const float biasCoefficient = 2.f * psqrt(1.f / float(posIters));
  // SetupArticulationInternalConstraintsTask (:1707): setupSolverInternalConstraintsTGS(desc, mStepDt, mInvStepDt, dt)
  for (uint32_t k = 0; k < R.nbArts; ++k) art::setupSolverConstraintsTGS(artAt(B, B.islandArts[R.artStart + k]), stepDt, invStepDt, mDt);
  // PartitionTask
  PartitionView pv{vels, bodyOffset + 1, R.nbBodies};
  uint32_t countsSize = 0, numOverflows = 0, numStatic = 0, numOrdered = 0;
  uint32_t maxPartitions;
  if (R.nbArts == 0) {
    maxPartitions = partitionContactConstraints(pv, B.descs, nbDescs, B.ordered, B.temp, B.partitionCounts, countsSize, B.partitionCap, 64,
                                                numOverflows, numStatic, numOrdered, B.error);
  } else {  // ExtendedRigidBodyClassification
    ExtClass X{vels, bodyOffset + 1, R.nbBodies, B.artProg, B.artBatchIndex};
    maxPartitions = partitionContactConstraintsExt(B, X, R.nbArts, B.descs, nbDescs, B.ordered, B.temp, B.partitionCounts, countsSize, B.partitionCap,
                                                   64, numOverflows, numStatic, numOrdered, B.error);
  }
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
      if (!isArtDesc(desc) && (desc.constraintType == SC_TYPE_RB_CONTACT || desc.constraintType == SC_TYPE_RB_1D))  // 관절체 제약은 하나씩
        for (; j < loopMax && desc.constraintType == B.ordered[a + j].constraintType && !isArtDesc(B.ordered[a + j]); ++j)
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
      prepare1DHeader(B, prm, B.ordered, hdr.startIndex, hdr.stride, stepDt, totalDt, invStepDt, invTotalDt, biasCoefficient);
      continue;
    }
    if (B.ordered[startIdx].constraintType != SC_TYPE_RB_CONTACT) {
      B.error |= SV_ERR_UNSUPPORTED;
      continue;
    }
    if (isArtDesc(B.ordered[startIdx])) {  // 관절체 링크가 낀 접촉 (머리 보폭 1)
      B.statHeaders++;
      B.statSingle++;
      B.statArtExtContacts++;
      prepareContactExtDesc(B, prm, B.ordered[startIdx], P, fprev, stepDt, totalDt, invStepDt, invTotalDt, biasCoefficient);
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
  // PxsCreateArticConstraintsSubTask -> prepareStaticConstraintsTGS (DyFeatherstoneArticulation.cpp:2100): 관절체마다 정적 1D·접촉 목록을
  // 링크 번호로 PxSort(불안정) 한 뒤 차례로 준비, 준비된 것만 남겨 링크별 개수·시작을 만든다. invStepDt = min(maxBiasCoefficient(=PX_MAX_F32), invStepDt)
  for (uint32_t k = 0; k < R.nbArts; ++k) {
    const uint32_t ai = B.islandArts[R.artStart + k];
    art::StaticLists& L = B.artLists[ai];
    auto linkA = [](const SDesc& d) { return d.linkIndexA; };
    auto linkB = [](const SDesc& d) { return d.linkIndexB; };
    SDesc* l1 = B.artStatic1D + ai * B.artStaticCap;
    B.artNbStatic1D[ai] = art::prepareStaticList(l1, B.artNbStatic1D[ai], L.nb1D, L.start1D, linkA, linkB, [&](SDesc& d) {
      B.statArtStatic1D++;
      prepare1DHeader(B, prm, &d, 0, 1, stepDt, totalDt, invStepDt, invTotalDt, biasCoefficient);
      return d.constraint != NONE;
    });
    SDesc* lc = B.artStaticContact + ai * B.artStaticCap;
    B.artNbStaticContact[ai] = art::prepareStaticList(lc, B.artNbStaticContact[ai], L.nbContact, L.startContact, linkA, linkB, [&](SDesc& d) {
      B.statArtStaticContacts++;
      prepareContactExtDesc(B, prm, d, P, fprev, stepDt, totalDt, invStepDt, invTotalDt, biasCoefficient);
      return d.constraint != NONE;
    });
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
  pl.artStart = R.artStart;
  pl.nbArts = R.nbArts;
  pl.biasCoefficient = biasCoefficient;
  pl.ok = 1;
}

// 반복 한 번의 제약 풀기: 분할 순서대로, 분할 안 머리는 서로 몸체를 나누지 않으므로(세계 몸체 제외: 정적 쪽은 질량 0 이라 같은 값만 쓴다)
// 스레드 nt 개가 나눠 푼다. nt = 1 이면 PhysX 단일 스레드 순서 그대로.
SV_HD void solvePartitions(SolverBoard& B, const BatchPlan& pl, float minPen, float elapsed, bool conclude, uint32_t tid, uint32_t nt,
                           uint32_t& err, bool posIter = true) {
  uint32_t h0 = 0;
  for (uint32_t p = 0; p < pl.totalPartitions; ++p) {
    const uint32_t h1 = h0 + B.partitionCounts[p];
    if (nt == 1 || (p == 0 && pl.firstSequential)) {
      if (tid == 0)
        for (uint32_t h = h0; h < h1; ++h) solveContactHeader(B, B.headers[h], minPen, elapsed, conclude, posIter, err);
    } else {
      for (uint32_t h = h0 + tid; h < h1; h += nt) solveContactHeader(B, B.headers[h], minPen, elapsed, conclude, posIter, err);
    }
    SV_SYNC();
    h0 = h1;
  }
}
SV_HD void integrateBodies(SolverBoard& B, const BatchPlan& pl, float dt, uint32_t tid, uint32_t nt) {
  for (uint32_t k = tid; k < pl.nbBodies; k += nt) integrateCoreStep(B.vels[pl.bodyOffset + k + 1], B.txI[pl.bodyOffset + k + 1], dt);
  SV_SYNC();
}

// ---------------- 관절체 내부 풀이의 정적 제약 자리 (art::solveInternalConstraints 의 Static 인자, docs 16.4)
// DyFeatherstoneArticulation.cpp:4323 solveStaticConstraint: 링크 속도를 A/B 자리에 넣고 solveExt1D / solveExtContactStep 을 부른다.
// minPenetration = 속도 반복이면 0, 위치 반복이면 -PX_MAX_F32 (:4744,4753). 1D 잔차는 위치 반복이면 위치 칸에(:4745).
struct ArtStaticSolve {
  SolverBoard* B;
  uint32_t ai;
  SV_HD uint32_t count1D(const art::Articulation&, uint32_t link) const { return B->artLists[ai].nb1D[link]; }
  SV_HD uint32_t countContact(const art::Articulation&, uint32_t link) const { return B->artLists[ai].nbContact[link]; }
  template <class Data>
  SV_HD void solve1D(art::Articulation& a, uint32_t link, uint32_t i, art::SV& linkV, art::SV& imp, art::SV& dv, const Data& data) const {
    const SDesc& d = B->artStatic1D[ai * B->artStaticCap + B->artLists[ai].start1D[link] + i];
    uint8_t* blk = arenaPtr<uint8_t>(B->constraints, d.constraint);
    art::solveStaticConstraint(d.linkIndexA != RIGID_BODY, linkV, imp, dv, a.deltaMotion[link], a.deltaQ[link],
                               [&](V3& lv0, V3& lv1, V3& av0, V3& av1, const V3& lm0, const V3& lm1, const V3& am0, const V3& am1, const Q& rA,
                                   const Q& rB, V3& li0, V3& li1, V3& ai0, V3& ai1) {
                                 V4 l0 = v3to4(lv0), l1 = v3to4(lv1), a0 = v3to4(av0), a1 = v3to4(av1);
                                 const float qa[4] = {rA.x, rA.y, rA.z, rA.w}, qb[4] = {rB.x, rB.y, rB.z, rB.w};
                                 V4 i0, i1, j0, j1;
                                 jnt::solveExt1D(blk, l0, l1, a0, a1, v3to4(lm0), v3to4(lm1), v3to4(am0), v3to4(am1), QuatVLoadU(qa), QuatVLoadU(qb),
                                                 data.elapsedTime, i0, i1, j0, j1, !data.isVelIter);
                                 lv0 = v4to3(l0); lv1 = v4to3(l1); av0 = v4to3(a0); av1 = v4to3(a1);
                                 li0 = v4to3(i0); li1 = v4to3(i1); ai0 = v4to3(j0); ai1 = v4to3(j1);
                               });
  }
  template <class Data>
  SV_HD void solveContact(art::Articulation& a, uint32_t link, uint32_t i, art::SV& linkV, art::SV& imp, art::SV& dv, const Data& data) const {
    const SDesc& d = B->artStaticContact[ai * B->artStaticCap + B->artLists[ai].startContact[link] + i];
    uint8_t* blk = arenaPtr<uint8_t>(B->constraints, d.constraint);
    const uint32_t len = d.constraintLengthOver16;
    art::solveStaticConstraint(d.linkIndexA != RIGID_BODY, linkV, imp, dv, a.deltaMotion[link], a.deltaQ[link],
                               [&](V3& lv0, V3& lv1, V3& av0, V3& av1, const V3& lm0, const V3& lm1, const V3& am0, const V3& am1, const Q&,
                                   const Q&, V3& li0, V3& li1, V3& ai0, V3& ai1) {
                                 V4 l0 = v3to4(lv0), l1 = v3to4(lv1), a0 = v3to4(av0), a1 = v3to4(av1);
                                 V4 i0 = v3to4(li0), i1 = v3to4(li1), j0 = v3to4(ai0), j1 = v3to4(ai1);
                                 solveExtContactStepCore(blk, len, l0, l1, a0, a1, v3to4(lm0), v3to4(lm1), v3to4(am0), v3to4(am1), i0, i1, j0, j1,
                                                         data.isVelIter ? 0.f : -kMaxReal, data.elapsedTime);
                                 lv0 = v4to3(l0); lv1 = v4to3(l1); av0 = v4to3(a0); av1 = v4to3(a1);
                                 li0 = v4to3(i0); li1 = v4to3(i1); ai0 = v4to3(j0); ai1 = v4to3(j1);
                               });
  }
};

// concludeInternalConstraints (:4477): 정적 1D 만 (접촉 conclude 는 빈 함수)
SV_HD void artConcludeInternal(SolverBoard& B, uint32_t ai) {
  for (uint32_t i = 0; i < B.artNbStatic1D[ai]; ++i) jnt::conclude1DStep(arenaPtr<uint8_t>(B.constraints, B.artStatic1D[ai * B.artStaticCap + i].constraint));
}
// writebackInternalConstraints (:4434)
SV_HD void artWritebackInternal(SolverBoard& B, uint32_t ai) {
  for (uint32_t i = 0; i < B.artNbStatic1D[ai]; ++i) {
    const SDesc& d = B.artStatic1D[ai * B.artStaticCap + i];
    jnt::writeBack1DStep(arenaPtr<uint8_t>(B.constraints, d.constraint), &B.writebacks[B.c1d[d.source].writeback]);
  }
  for (uint32_t i = 0; i < B.artNbStaticContact[ai]; ++i)
    writeBackContactAny(B.artStaticContact[ai * B.artStaticCap + i], B.constraints, B.friction[B.frictionCurIdx]);
}

// 관절체 내부 풀이 한 번 (묶음 안 관절체 전부, 스레드 nt 개가 나눠 — 관절체끼리 독립)
SV_HD void artSolveInternalAll(SolverBoard& B, const BatchPlan& pl, const art::ProcessConfig& cfg, bool velIter, float elapsed, bool conclude,
                               uint32_t tid, uint32_t nt) {
  const float recipStepDt = 1.0f / pl.stepDt;
  for (uint32_t k = tid; k < pl.nbArts; k += nt) {
    const uint32_t ai = B.islandArts[pl.artStart + k];
    art::solveInternalConstraints(artAt(B, ai), pl.dt, pl.stepDt, recipStepDt, velIter, true, cfg, elapsed, pl.biasCoefficient, false,
                                  ArtStaticSolve{&B, ai});
    if (conclude) artConcludeInternal(B, ai);
  }
  SV_SYNC();
}
// stepArticulations = updateDeltaMotion (DyTGSDynamics.cpp:1582)
SV_HD void artRecordAll(SolverBoard& B, const BatchPlan& pl, uint32_t tid, uint32_t nt) {
  for (uint32_t k = tid; k < pl.nbArts; k += nt) {
    art::SV scratch[art::kMaxLinks];
    art::recordDeltaMotion(artAt(B, B.islandArts[pl.artStart + k]), pl.stepDt, scratch);
  }
  SV_SYNC();
}

// iterativeSolveIsland (DyTGSDynamics.cpp:2515-2793) 관절체가 있는 묶음. solveArticulationContactLast 면 접촉 앞뒤 두 번(:2595).
SV_HDN void solveBatchIterateArt(SolverBoard& B, const SolverParams& prm, uint32_t tid, uint32_t nt) {
  const BatchPlan pl = B.plan;
  uint32_t err = 0;
  const art::ProcessConfig single = art::singlePassConfig(prm.solveArticulationContactLast);
  const art::ProcessConfig first = art::firstPassConfig();
  const art::ProcessConfig second = art::secondPassConfig();
  const bool last = prm.solveArticulationContactLast;
  if (pl.numBatches == 0) {  // :2531 관절체마다 전부 따로
    const float recipStepDt = 1.0f / pl.stepDt;
    for (uint32_t k = tid; k < pl.nbArts; k += nt) {
      const uint32_t ai = B.islandArts[pl.artStart + k];
      art::Articulation& a = artAt(B, ai);
      const ArtStaticSolve st{&B, ai};
      art::SV scratch[art::kMaxLinks];
      float elapsedTime = 0.0f;
      for (uint32_t it = 0; it < pl.posIters; it++) {
        art::solveInternalConstraints(a, pl.dt, pl.stepDt, recipStepDt, false, true, single, elapsedTime, pl.biasCoefficient, false, st);
        art::recordDeltaMotion(a, pl.stepDt, scratch);
        elapsedTime += pl.stepDt;
      }
      art::saveVelocityTGS(a, pl.invDt);
      artConcludeInternal(B, ai);
      for (uint32_t it = 0; it < pl.velIters; ++it)
        art::solveInternalConstraints(a, pl.dt, pl.stepDt, recipStepDt, true, true, single, elapsedTime, pl.biasCoefficient, false, st);
      artWritebackInternal(B, ai);
    }
    SV_SYNC();
    integrateBodies(B, pl, pl.dt, tid, nt);
    return;
  }
  float elapsedTime = 0.0f;
  for (uint32_t a = 1; a < pl.posIters; a++) {
    if (last) {
      artSolveInternalAll(B, pl, first, false, elapsedTime, false, tid, nt);
      solvePartitions(B, pl, -kMaxReal, elapsedTime, false, tid, nt, err, true);
      artSolveInternalAll(B, pl, second, false, elapsedTime, false, tid, nt);
    } else {
      solvePartitions(B, pl, -kMaxReal, elapsedTime, false, tid, nt, err, true);
      artSolveInternalAll(B, pl, single, false, elapsedTime, false, tid, nt);
    }
    integrateBodies(B, pl, pl.stepDt, tid, nt);
    artRecordAll(B, pl, tid, nt);
    elapsedTime += pl.stepDt;
  }
  {  // 마지막 위치 반복 (conclude)
    if (last) {
      artSolveInternalAll(B, pl, first, false, elapsedTime, false, tid, nt);
      solvePartitions(B, pl, -kMaxReal, elapsedTime, true, tid, nt, err, true);
      artSolveInternalAll(B, pl, second, false, elapsedTime, true, tid, nt);
    } else {
      solvePartitions(B, pl, -kMaxReal, elapsedTime, true, tid, nt, err, true);
      artSolveInternalAll(B, pl, single, false, elapsedTime, true, tid, nt);
    }
    elapsedTime += pl.stepDt;
    integrateBodies(B, pl, pl.stepDt, tid, nt);
    artRecordAll(B, pl, tid, nt);
  }
  for (uint32_t k = tid; k < pl.nbArts; k += nt) art::saveVelocityTGS(artAt(B, B.islandArts[pl.artStart + k]), pl.invDt);
  SV_SYNC();
  for (uint32_t a = 0; a < pl.velIters; ++a) {
    if (last) {
      artSolveInternalAll(B, pl, first, true, elapsedTime, false, tid, nt);
      solvePartitions(B, pl, 0.f, elapsedTime, false, tid, nt, err, false);
      artSolveInternalAll(B, pl, second, true, elapsedTime, false, tid, nt);
    } else {
      solvePartitions(B, pl, 0.f, elapsedTime, false, tid, nt, err, false);
      artSolveInternalAll(B, pl, single, true, elapsedTime, false, tid, nt);
    }
  }
  if (err) svAtomicOr(B.error, err);
}

// iterativeSolveIsland 의 반복 부분 (DyTGSDynamics.cpp:2515-2793). 판 안 스레드 전부가 부른다(nt = 1 이면 호스트 순서 그대로).
SV_HDN void solveBatchIterate(SolverBoard& B, const SolverParams& prm, uint32_t tid, uint32_t nt) {
  const BatchPlan pl = B.plan;
  if (!pl.ok) return;
  if (pl.nbArts) {
    solveBatchIterateArt(B, prm, tid, nt);
    return;
  }
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
      else if (hd.constraintType == SC_TYPE_RB_1D || hd.constraintType == SC_TYPE_EXT_1D)  // writeBack1D (DyTGSContactPrep.cpp:3353)
        for (uint32_t i = hd.startIndex, e = hd.startIndex + hd.stride; i < e; ++i)
          jnt::writeBack1DStep(arenaPtr<uint8_t>(B.constraints, cdb[i].constraint), &B.writebacks[B.c1d[cdb[i].source].writeback]);
      else if (hd.constraintType == SC_TYPE_BLOCK_1D) {  // writeBack1D4 (DyTGSContactPrepBlock.cpp:3418)
        jnt::Writeback* wb[4];
        for (int a = 0; a < 4; ++a) wb[a] = &B.writebacks[B.c1d[cdb[hd.startIndex + a].source].writeback];
        jnt::writeBack1D4(arenaPtr<uint8_t>(B.constraints, cdb[hd.startIndex].constraint), wb, false);
      } else if (hd.constraintType == SC_TYPE_EXT_CONTACT)
        for (uint32_t i = hd.startIndex, e = hd.startIndex + hd.stride; i < e; ++i) writeBackContactAny(cdb[i], B.constraints, fcur);
      else
        for (uint32_t i = hd.startIndex, e = hd.startIndex + hd.stride; i < e; ++i) writeBackContact(cdb[i], B.constraints, fcur);
    }
    for (uint32_t k = 0; k < pl.nbArts; ++k) artWritebackInternal(B, B.islandArts[pl.artStart + k]);  // :2788
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
  // UpdateArticTask -> updateArticulations -> updateBodiesTGS (DyTGSDynamics.cpp:1593)
  for (uint32_t a = 0; a < pl.nbArts; ++a) {
    art::SV scratch[art::kMaxLinks];
    art::updateBodiesTGS(artAt(B, B.islandArts[pl.artStart + a]), mDt, scratch);
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
    R.artStart = B.islands[currentIsland].artStart;
    uint32_t nbBodies = 0, nbArticulations = 0, nbCMs = 0;
    while (nbBodies < prm.solverBatchSize && currentIsland < B.nbIslands && nbArticulations < prm.solverArticBatchSize) {
      nbBodies += B.islands[currentIsland].bodyCount;
      nbArticulations += B.nbArts ? B.islands[currentIsland].artCount : 0u;  // island.mNodeCount[eARTICULATION_TYPE]
      nbCMs += B.islands[currentIsland].cmCount;
      currentIsland++;
    }
    R.nbArts = nbArticulations;
    R.islandEnd = currentIsland;
    R.nbBodies = nbBodies;
    R.nbCMs = nbCMs;
    R.solverBodyOffset = kinematicCount + currentBodyIndex;
    if (tid == 0) solveBatchPrep(B, prm, R, kinematicCount);
    SV_SYNC();
    solveBatchIterate(B, prm, tid, nt);
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

// 관절체 (ScPipeline.cpp:2694-2707): 이번 스텝 섬의 관절체 — 섬 관리가 재운 것은 putToSleep, 나머지는 sleepCheck
// (SimulationController::updateArticulationAfterIntegration -> ArticulationSim::sleepCheck, ScArticulationSim.cpp:432). deact = 관절체 번호.
SV_HDN void afterIntegrationArts(SolverBoard& B, float dt, const uint32_t* deact, uint32_t n) {
  for (uint32_t i = 0; i < B.nbIslands; ++i) {
    const IslandIn& I = B.islands[i];
    for (uint32_t k = 0; k < I.artCount; ++k) {
      const uint32_t ai = B.islandArts[I.artStart + k];
      bool sleeping = false;
      for (uint32_t d = 0; d < n; ++d) sleeping = sleeping || deact[d] == ai;
      if (!sleeping) art::sleepCheck(artAt(B, ai), dt);
    }
  }
  for (uint32_t d = 0; d < n; ++d) art::putToSleep(artAt(B, deact[d]));
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
