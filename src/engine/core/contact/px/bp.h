// PhysX 넓은 단계 ABP(PABP 설정) 의 우리 판 (eng::px::Bp). PhysX 링크 없음. 층 1(호스트) 판: 동적 배열을 쓴다.
// 원본 lowlevelaabb/src/BpBroadPhaseABP.cpp 등을 gen/translate.py 로 기계 번역(gen_bp.inc). 원본은 ABP_MT2(BpBroadPhaseABP.h:37)로
// 작업 여러 개를 나눠 돌리지만 각 작업이 자기 버퍼에 쓰고 끝에 번호 순서로 합치므로(addDelayedPairs2) 결과 순서는 스레드 수와 무관하다.
// 우리 판은 작업을 한 스레드로 차례대로 돌린다(bp_support.inc PxLightCpuTask). 시험: tests/contact/test_bp_abp (--mt 0/1/4 모두 같음).
// 설정: USE_ABP_BUCKETS 512(물체 512 개 넘으면 5 칸 나눔), ABP_USE_INTEGER_XS2, ABP_SIMD_OVERLAP, ABP_BATCHING 256.
#pragma once
#include "core/contact/px/gu.h"

namespace eng {
#include "core/contact/px/bp_support.inc"
#include "core/contact/px/gen_bp.inc"
}  // namespace eng
