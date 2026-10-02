// PhysX aos (Vec3V·FloatV·BoolV·QuatV·Mat33V·PxTransformV) 의 우리 판 (eng::px::aos). PhysX 링크 없음.
// 원본 PxVecMath.h + PxVecMathSSE.h + unix/sse2/PxUnixSse2InlineAoS.h + PxVecQuat.h + PxVecTransform.h 를
// gen/translate.py 로 기계 번역(gen_aos.inc): SSE 명령만 sse_emu.h 의 em_* 로 바뀌고 식·섞기 순서는 원본 그대로다.
// 리눅스 clang 빌드 기준(SSE2 경로, __SSE4_2__ 없음): V3Dot 등 가로 연산은 _mm_dp_ps 가 아니라 섞기+더하기 판.
#pragma once
#include "core/contact/px/foundation.h"

namespace eng {
#include "core/contact/px/gen_aos.inc"
}  // namespace eng
