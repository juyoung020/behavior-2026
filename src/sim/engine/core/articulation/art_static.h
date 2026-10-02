// articulation 모듈: 관절체 링크와 정적(또는 운동학) 물체 사이 제약(접촉·1D 행)을 관절체 내부 풀이에서 푸는 자리.
// PhysX 는 제약 분할 때 이런 제약을 관절체에 넘기고(DyConstraintPartition.cpp:269 storeStaticConstraint), 준비 단계에서 링크 번호로
// 정렬해 링크별 개수·시작 번호를 만든 뒤(DyFeatherstoneArticulation.cpp:2100 prepareStaticConstraintsTGS), 내부 풀이에서
// 링크 차례가 올 때 푼다(:4323 solveStaticConstraint, :4729, :4919).
// 여기 있는 것 = 관절체 몫: PxSort 이식(불안정 정렬이라 같은 링크 안 순서까지 같아야 함), 목록 정리(준비 실패 항목 빼고 당기기),
// 링크별 개수·시작, 풀이 때 링크 속도 <-> 제약 풀이 함수 사이 변환. 제약 준비·풀이 식(createFinalizeSolverContactsStep,
// SetupSolverConstraintStep, solveExtContactStep, solveExt1DStep)은 solver·joints·contact 몫이라 함수 인자로 받는다.
#pragma once
#include <cstdint>

#include "art_step.h"

namespace eng {
namespace art {

// ---------------------------------------------------------------- PxSort (physx/include/foundation/PxSort.h:56, PxSortInternals.h)
template <class T>
EHD void pxSwap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}
template <class T, class Less>
EHD void pxMedian3(T* e, int32_t first, int32_t last, const Less& cmp) {
  const int32_t mid = (first + last) / 2;
  if (cmp(e[mid], e[first])) pxSwap(e[first], e[mid]);
  if (cmp(e[last], e[first])) pxSwap(e[first], e[last]);
  if (cmp(e[last], e[mid])) pxSwap(e[mid], e[last]);
  pxSwap(e[mid], e[last - 1]);
}
template <class T, class Less>
EHD int32_t pxPartition(T* e, int32_t first, int32_t last, const Less& cmp) {
  pxMedian3(e, first, last, cmp);
  int32_t i = first, j = last - 1;
  for (;;) {
    while (cmp(e[++i], e[last - 1])) {
    }
    while (cmp(e[last - 1], e[--j])) {
    }
    if (i >= j) break;
    pxSwap(e[i], e[j]);
  }
  pxSwap(e[i], e[last - 1]);
  return i;
}
template <class T, class Less>
EHD void pxSmallSort(T* e, int32_t first, int32_t last, const Less& cmp) {
  for (int32_t i = first; i < last; i++) {
    int32_t m = i;
    for (int32_t j = i + 1; j <= last; j++)
      if (cmp(e[j], e[m])) m = j;
    if (m != i) pxSwap(e[m], e[i]);
  }
}
// 스택은 고정 128 칸 (작은 쪽을 먼저 쌓으므로 깊이 <= 2 log2 n — 2^60 개까지)
template <class T, class Less>
EHD void pxSort(T* e, uint32_t count, const Less& cmp) {
  const uint32_t SMALL_SORT_CUTOFF = 5;
  int32_t stack[128];
  uint32_t sp = 0;
  int32_t first = 0, last = int32_t(count - 1);
  if (last > first) {
    for (;;) {
      while (last > first) {
        if (uint32_t(last - first) < SMALL_SORT_CUTOFF) {
          pxSmallSort(e, first, last, cmp);
          break;
        } else {
          const int32_t partIndex = pxPartition(e, first, last, cmp);
          if ((partIndex - first) < (last - partIndex)) {
            stack[sp++] = first;
            stack[sp++] = partIndex - 1;
            first = partIndex + 1;
          } else {
            stack[sp++] = partIndex + 1;
            stack[sp++] = last;
            last = partIndex - 1;
          }
        }
      }
      if (sp == 0) break;
      last = stack[--sp];
      first = stack[--sp];
    }
  }
}

// ---------------------------------------------------------------- 링크별 정적 제약 목록
constexpr uint32_t kRigidBody = 0xffff;  // PxSolverConstraintDesc::RIGID_BODY (PxSolverDefs.h:113)

struct StaticLists {  // ArticulationData::mNbStatic*Constraints / m*StartIndex (링크마다)
  uint32_t nb1D[kMaxLinks], start1D[kMaxLinks];
  uint32_t nbContact[kMaxLinks], startContact[kMaxLinks];
};
EHD void clearStaticLists(StaticLists& s, uint32_t nLinks) {  // computeUnconstrainedVelocitiesInternal (ForwardDynamic.cpp:1811-1815)
  for (uint32_t i = 0; i < nLinks; ++i) s.nb1D[i] = s.nbContact[i] = 0;
}

// prepareStaticConstraintsTGS (:2100) 의 관절체 몫. Desc 는 호출자 형식(PxSolverConstraintDesc 와 같은 정보),
//   linkA(d)/linkB(d) = linkIndexA/B (강체 쪽이면 kRigidBody), prep(d) = 준비(solver/joints/contact) 뒤 "제약이 남았는지"(desc.constraint != NULL).
// 반환: 남은 개수 (지운 항목은 PxArray::remove 처럼 뒤를 당긴다).
template <class Desc, class LinkA, class LinkB, class Prep>
EHD uint32_t prepareStaticList(Desc* list, uint32_t n, uint32_t* nbPerLink, uint32_t* startPerLink, const LinkA& linkA, const LinkB& linkB,
                               const Prep& prep) {
  auto key = [&](const Desc& d) { return linkA(d) != kRigidBody ? linkA(d) : linkB(d); };
  struct Less {  // ArticulationStaticConstraintSortPredicate (:2076)
    const decltype(key)& k;
    EHD bool operator()(const Desc& l, const Desc& r) const { return k(l) < k(r); }
  };
  pxSort(list, n, Less{key});
  for (uint32_t i = 0; i < n; ++i) {
    Desc& d = list[i];
    const uint32_t linkIndex = key(d);
    if (prep(d)) {
      if (nbPerLink[linkIndex] == 0) startPerLink[linkIndex] = i;
      nbPerLink[linkIndex]++;
    } else {
      for (uint32_t k = i + 1; k < n; ++k) list[k - 1] = list[k];  // remove(i)
      --n;
      --i;
    }
  }
  return n;
}

// solveStaticConstraint (:4323) 의 관절체 몫: 링크 속도를 A/B 자리에 넣고 풀이 함수(solveExt1DStep 또는 solveExtContactStep)를 부른 뒤
// deltaV += 새 속도 - 옛 속도, 링크 속도 갱신, 충격 누적(impulse -= newImp).
// solveFn(linVel0, linVel1, angVel0, angVel1, linMotion0, linMotion1, angMotion0, angMotion1, rotA, rotB, li0, li1, ai0, ai1)
//   — 값은 모두 V3/Q, 반대쪽(강체=정적)은 0 과 항등. 우리 SV 는 (top=각, bottom=선).
template <class SolveFn>
EHD void solveStaticConstraint(bool linkIsA, SV& linkV, SV& impulse, SV& deltaV, const SV& motion, const Q& rot, const SolveFn& solveFn) {
  const V3 linVel = linkV.bottom, angVel = linkV.top;
  const V3 z{0, 0, 0};
  const Q idt = qid();
  V3 linVel0, linVel1, angVel0, angVel1, linMotion0, linMotion1, angMotion0, angMotion1;
  Q rotA, rotB;
  V3 li0 = z, li1 = z, ai0 = z, ai1 = z;
  if (linkIsA) {
    linVel0 = linVel;
    angVel0 = angVel;
    linMotion0 = motion.bottom;
    angMotion0 = motion.top;
    rotA = rot;
    rotB = idt;
    linVel1 = angVel1 = linMotion1 = angMotion1 = z;
  } else {
    linVel1 = linVel;
    angVel1 = angVel;
    linMotion1 = motion.bottom;
    angMotion1 = motion.top;
    rotB = rot;
    rotA = idt;
    linVel0 = angVel0 = linMotion0 = angMotion0 = z;
  }
  solveFn(linVel0, linVel1, angVel0, angVel1, linMotion0, linMotion1, angMotion0, angMotion1, rotA, rotB, li0, li1, ai0, ai1);
  SV newVel, newImp;
  if (linkIsA) {
    newVel = SV{angVel0, linVel0};
    newImp = SV{li0, ai0};
  } else {
    newVel = SV{angVel1, linVel1};
    newImp = SV{li1, ai1};
  }
  deltaV.top += (newVel.top - linkV.top);
  deltaV.bottom += (newVel.bottom - linkV.bottom);
  linkV.top = newVel.top;
  linkV.bottom = newVel.bottom;
  impulse -= newImp;
}

// solveInternalConstraints 의 Static 인자 구현: 링크별 목록 + 호출자 풀이 함수.
//   solve1D(i, linkIsA?, ...) / solveContact(i, ...) 는 목록 번호 i 의 제약을 푸는 함수 (SolveFn 형식을 돌려주는 대신 직접 푼다):
//   Fn(uint32_t descIndex, bool isVelIter, float elapsedTime, V3& linVel0, V3& linVel1, ... , V3& ai1)
template <class Fn1D, class FnContact, class IsLinkA1D, class IsLinkAContact>
struct ArtStatic {
  const StaticLists* lists;
  Fn1D fn1D;
  FnContact fnContact;
  IsLinkA1D isA1D;           // isA1D(descIndex) -> 링크가 A 쪽인가
  IsLinkAContact isAContact;
  EHD uint32_t count1D(const Articulation&, uint32_t link) const { return lists->nb1D[link]; }
  EHD uint32_t countContact(const Articulation&, uint32_t link) const { return lists->nbContact[link]; }
  template <class Data>
  EHD void solve1D(Articulation& a, uint32_t link, uint32_t i, SV& linkV, SV& imp, SV& dv, const Data& data) const {
    const uint32_t idx = lists->start1D[link] + i;
    solveStaticConstraint(isA1D(idx), linkV, imp, dv, a.deltaMotion[link], a.deltaQ[link],
                          [&](V3& lv0, V3& lv1, V3& av0, V3& av1, const V3& lm0, const V3& lm1, const V3& am0, const V3& am1, const Q& rA,
                              const Q& rB, V3& li0, V3& li1, V3& ai0, V3& ai1) {
                            fn1D(idx, data.isVelIter, data.elapsedTime, lv0, lv1, av0, av1, lm0, lm1, am0, am1, rA, rB, li0, li1, ai0, ai1);
                          });
  }
  template <class Data>
  EHD void solveContact(Articulation& a, uint32_t link, uint32_t i, SV& linkV, SV& imp, SV& dv, const Data& data) const {
    const uint32_t idx = lists->startContact[link] + i;
    solveStaticConstraint(isAContact(idx), linkV, imp, dv, a.deltaMotion[link], a.deltaQ[link],
                          [&](V3& lv0, V3& lv1, V3& av0, V3& av1, const V3& lm0, const V3& lm1, const V3& am0, const V3& am1, const Q& rA,
                              const Q& rB, V3& li0, V3& li1, V3& ai0, V3& ai1) {
                            fnContact(idx, data.isVelIter, data.elapsedTime, lv0, lv1, av0, av1, lm0, lm1, am0, am1, rA, rB, li0, li1, ai0, ai1);
                          });
  }
};

}  // namespace art
}  // namespace eng
