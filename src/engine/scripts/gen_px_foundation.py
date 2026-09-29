#!/usr/bin/env python3
"""core/common/px_foundation_vec.inc 를 만든다 (문서 12.7 "Px 값 형 한 벌", 09-30 결정).

입력: contact 작업자의 기계 번역본 core/contact/px/gen_foundation_vec.inc (gen/translate.py 가 PhysX foundation 에서 만든 것).
식은 한 글자도 안 바꾸고, 우리 수학 형(pmath.h 의 V3·Q·Tf)과 오가는 생성자·형변환 두 줄씩만 끼운다
(joints·solver·articulation 코드가 PxVec3(V3) 처럼 쓰던 것을 그대로 두기 위해. 같은 개념의 Px 형은 eng::px 한 벌뿐).
contact 가 번역을 다시 하면 이 스크립트를 다시 돌린다 (나중에 translate.py 출력 위치를 common 으로 바꾸면 이 파일은 사라진다).

    python3 src/engine/scripts/gen_px_foundation.py
"""
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "core", "contact", "px", "gen_foundation_vec.inc")
OUT = os.path.join(ROOT, "core", "common", "px_foundation_vec.inc")

INTEROP = {
    "PxVec3T": "\t// 우리 수학 형과 오가기 (common 추가, 원본 없음)\n"
               "\tEHD PxVec3T(const ::eng::V3& v) : x(Type(v.x)), y(Type(v.y)), z(Type(v.z))\n\t{\n\t}\n"
               "\tEHD operator ::eng::V3() const\n\t{\n\t\treturn ::eng::V3{float(x), float(y), float(z)};\n\t}\n",
    "PxQuatT": "\t// 우리 수학 형과 오가기 (common 추가, 원본 없음)\n"
               "\tEHD PxQuatT(const ::eng::Q& v) : x(Type(v.x)), y(Type(v.y)), z(Type(v.z)), w(Type(v.w))\n\t{\n\t}\n"
               "\tEHD operator ::eng::Q() const\n\t{\n\t\treturn ::eng::Q{float(x), float(y), float(z), float(w)};\n\t}\n",
    "PxTransformT": "\t// 우리 수학 형과 오가기 (common 추가, 원본 없음)\n"
                    "\tEHD PxTransformT(const ::eng::Tf& t) : q(t.q), p(t.p)\n\t{\n\t}\n"
                    "\tEHD operator ::eng::Tf() const\n\t{\n\t\treturn ::eng::Tf{::eng::Q(q), ::eng::V3(p)};\n\t}\n",
}


def main():
    src = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    out = src
    for cls, add in INTEROP.items():
        # "class X\n{\n  public:\n" 바로 뒤 (PxTransformT 는 멤버 선언 뒤, 첫 생성자 앞)
        m = re.search(r"class " + cls + r"\n\{\n  public:\n", out)
        if not m:
            sys.exit(f"{cls} 머리를 못 찾음")
        pos = m.end()
        if cls == "PxTransformT":
            m2 = re.search(r"\tEHD PxTransformT\(\)", out[pos:])
            pos += m2.start()
        out = out[:pos] + add + "\n" + out[pos:]
    head = ("// 생성물 — 손으로 고치지 말 것. scripts/gen_px_foundation.py 가 core/contact/px/gen_foundation_vec.inc (contact 의 PhysX foundation\n"
            "// 기계 번역본) 에 우리 수학 형(V3·Q·Tf) 오가기만 더해 만든다. 원본 저작권: NVIDIA PhysX 5.6.1, BSD-3.\n")
    open(OUT, "w", encoding="utf-8", newline="\n").write(head + out)
    print(f"쓴 파일 {OUT}")


if __name__ == "__main__":
    main()
