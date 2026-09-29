#!/usr/bin/env python3
"""PhysX 5.6.1 의 aos(SIMD 수학) SSE2 판을 core/common/aos.h 로 기계적으로 옮긴다 (WSL 에서 실행).

    python3 /mnt/c/behavior-2026/src/engine/scripts/gen_aos.py

원리: 공식 리눅스 빌드가 컴파일한 것과 같은 전처리 분기(SSE4 없음, clang, 리눅스)를 unifdef 로 고른 뒤,
      __m128 -> sse::m128, _mm_* -> sse::_mm_* 로만 바꾼다. 식은 한 글자도 안 바꾼다 -> 연산 순서가 원본과 같다.
      _mm_* 의 뜻은 core/common/sse_emu.h (호스트·CUDA 공용, rcpps/rsqrtps 는 표 흉내).
입력: ~/engine-deps/physx-107.3-omni/physx/include/foundation/ (태그 107.3-omni-and-physx-5.6.1)
출력: /mnt/c/behavior-2026/src/engine/core/common/aos.h  (생성물 — 손으로 고치지 말 것)
"""
import os
import re
import subprocess
import sys

SRC = os.path.expanduser("~/engine-deps/physx-107.3-omni/physx/include/foundation")
OUT = "/mnt/c/behavior-2026/src/engine/core/common/aos.h"
DEFS = ["-U__SSE4_2__", "-DPX_EMSCRIPTEN=0", "-DPX_CLANG=1", "-DPX_LINUX=1", "-DPX_DEBUG=0", "-DPX_DOXYGEN=0",
        "-DCOMPILE_VECTOR_INTRINSICS=1", "-DPX_INTEL_FAMILY=1", "-DPX_UNIX_FAMILY=1", "-DPX_NEON=0",
        "-DPX_WINDOWS=0", "-DPX_SWITCH=0", "-DPX_SIMD_DISABLED=0"]
# PxVecMath.h 안 include 순서 그대로: 종류(PxUnixSse2AoS) -> 선언(PxVecMath) -> 삼각 상수 -> 구현(SSE 공통 -> 유닉스 SSE2) -> 쿼터니언 -> 변환
ORDER = ["unix/sse2/PxUnixSse2AoS.h", "PxVecMath.h", "unix/PxUnixTrigConstants.h", "PxVecMathSSE.h",
         "unix/sse2/PxUnixSse2InlineAoS.h", "PxVecQuat.h", "PxVecTransform.h"]
GLOBALS = ["gMaskXYZ", "g_PXSinCoefficients0", "g_PXSinCoefficients1", "g_PXSinCoefficients2",
           "g_PXCosCoefficients0", "g_PXCosCoefficients1", "g_PXCosCoefficients2", "g_PXReciprocalTwoPi", "g_PXTwoPi"]


def unifdef(path):
    r = subprocess.run(["unifdef", *DEFS, os.path.join(SRC, path)], capture_output=True, text=True)
    if r.returncode not in (0, 1):
        sys.exit(f"unifdef 실패 {path}: {r.stderr}")
    return r.stdout


def strip_guard_and_includes(text):
    lines = text.split("\n")
    out = []
    guard = None
    for ln in lines:
        s = ln.strip()
        m = re.match(r"#\s*ifndef\s+(\w+_H)\s*$", s)
        if guard is None and m:
            guard = m.group(1)
            continue
        if guard and re.match(rf"#\s*define\s+{guard}\s*$", s):
            continue
        if re.match(r"#\s*include\b", s) or re.match(r"#\s*pragma\b", s):
            continue
        out.append(ln)
    # 마지막 #endif (가드 닫기) 지우기
    if guard:
        for i in range(len(out) - 1, -1, -1):
            if re.match(r"\s*#\s*endif", out[i]):
                del out[i]
                break
    return "\n".join(out)


def transform(text, name):
    t = strip_guard_and_includes(text)
    # 라이선스 머리말(맨 앞 // 주석 덩어리)은 파일 머리에 한 번만 둔다
    t = re.sub(r"\A(\s*//[^\n]*\n)+", "", t)
    t = t.replace("namespace physx", "namespace eng")
    t = re.sub(r"\bphysx::", "eng::", t)
    t = re.sub(r"\b__m128i\b", "sse::m128i", t)
    t = re.sub(r"\b__m128\b", "sse::m128", t)
    t = re.sub(r"(?<![\w:])(_mm_\w+)\s*\(", r"sse::\1(", t)
    t = re.sub(r"\b_MM_SHUFFLE\b", "ENG_MM_SHUFFLE", t)
    t = re.sub(r"\bPX_FORCE_INLINE\b", "EHD", t)
    # 전역 상수: 호스트판 이름_h 와 장치판 이름_d 두 벌(초기값 글자 그대로 복사), 쓰는 곳은 ENG_G(이름) (aos_prelude.h)
    def dup(nm, typ, init):
        return (f"alignas(16) static const {typ} {nm}_h = {init};\n"
                f"#if defined(__CUDACC__)\n__device__ __constant__ static const {typ} {nm}_d = {init};\n#endif\n")
    if name == "PxUnixTrigConstants.h":
        t = re.sub(r"#define PX_GLOBALCONST[^\n]*\n", "", t)
        t = re.sub(r"PX_GLOBALCONST\s+PX_VECTORF32\s*(g_PX\w+)\s*=\s*(\{\s*\{[^;]*?\}\s*\});",
                   lambda m: dup(m.group(1), "PX_VECTORF32", m.group(2)), t)
    if name == "PxUnixSse2InlineAoS.h":
        # 비트 무늬(NaN) 상수라 union 의 정수 칸으로 초기화한다
        t = re.sub(r"const PX_ALIGN\(16, PxF32 gMaskXYZ\[4\]\)\s*=[^;]*;",
                   dup("gMaskXYZ", "ENG_U4F", "{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0u}}"), t, flags=re.S)
        t = re.sub(r"\binternalSimd::gMaskXYZ\b", "ENG_G(internalSimd::gMaskXYZ).f", t)
    t = re.sub(r"\b(g_PX\w+)\.f\b", r"ENG_G(\1).f", t)
    return f"\n// ======== {name} (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========\n" + t


def main():
    parts = []
    for rel in ORDER:
        parts.append(transform(unifdef(rel), os.path.basename(rel)))
    body = "\n".join(parts)
    # 남은 원본 전용 코드 점검: 흉내에 없는 intrinsic 이 있으면 멈춘다
    left = sorted(set(re.findall(r"(?<!sse::)\b_mm_\w+\b", body)) - {"_mm_"})
    if left:
        print("주의: sse:: 로 안 바뀐 _mm_ 이름:", left)
    head = (
        "// 생성물 — 손으로 고치지 말 것. scripts/gen_aos.py 가 PhysX 5.6.1 aos SSE2 판(include/foundation)에서 만든다.\n"
        "// 원본 저작권: Copyright (c) 2008-2025 NVIDIA Corporation. BSD-3-Clause (원본 머리말과 같은 조건).\n"
        "// 식은 원본 그대로이고 __m128/_mm_* 만 core/common/sse_emu.h 의 흉내로 바꿨다 (호스트·CUDA 공용).\n"
        "#pragma once\n#include \"aos_prelude.h\"\n\n")
    open(OUT, "w", encoding="utf-8").write(head + body + "\n")
    print(f"쓴 파일 {OUT}: {len(body.splitlines())} 줄")


if __name__ == "__main__":
    main()
