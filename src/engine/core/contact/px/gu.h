// PhysX geomutils 접촉 생성(PCM 볼록-볼록·GJK·EPA·지속 접촉 다양체) 의 우리 판 (eng::px::Gu). PhysX 링크 없음.
// gen/translate.py 가 원본 .h/.cpp 를 기계 번역한 gen_gu_convex.inc 를 담는다 (식·순서 그대로, 함수에 호스트·장치 표시만 붙음).
// 원본 목록은 gen/jobs.txt. 볼록 메시 데이터(ConvexHullData 바이트 배치)는 PhysX 가 구운 것을 그대로 쓴다(장면 추출 단계에서 한 번).
#pragma once
#include "core/contact/px/aos.h"
#include "core/contact/px/glibc_acosf.h"

#if defined(__CUDACC__)
#define EHDV __host__ __device__
#else
#define EHDV
#endif
#define EPX_NOALIAS
#define EPX_POP_PACK
#define EPX_PUSH_PACK_DEFAULT

namespace eng {
#include "core/contact/px/gu_support.inc"
#include "core/contact/px/gen_gu_convex.inc"
}  // namespace eng
