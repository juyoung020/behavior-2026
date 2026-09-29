#!/usr/bin/env python3
"""aos 흉내(core/common/aos.h) = 진짜 PhysX aos(SSE 하드웨어) 인지 함수마다 비트 비교하는 시험을 만든다 (WSL 에서).

    python3 /mnt/c/behavior-2026/src/engine/scripts/gen_aos_test.py
    -> src/engine/tests/common/aos_diff_{px,eng}.cpp, aos_diff_main.cpp (생성물)

선언(aos.h 의 "EHD 반환 이름(인자);")을 읽어, 값 형(FloatV, Vec3V, Vec4V, BoolV, QuatV, VecU32V, VecI32V, Mat33V,
PxF32, PxU32, bool)만 쓰는 함수마다 두 벌의 감싸개를 만든다. 두 번역 단위로 나눠 이름 충돌을 피한다.
입력은 형별 규칙을 지킨 난수(FloatV 는 네 칸 같음, Vec3V 는 w=0, BoolV 는 칸마다 0 또는 전부 1).
"""
import re

AOS = "/mnt/c/behavior-2026/src/engine/core/common/aos.h"
OUTDIR = "/mnt/c/behavior-2026/src/engine/tests/common"
SIZES = {"FloatV": 4, "Vec3V": 4, "Vec4V": 4, "BoolV": 4, "QuatV": 4, "VecU32V": 4, "VecI32V": 4, "Mat33V": 12,
         "PxF32": 1, "PxU32": 1, "bool": 1}
KIND = {"FloatV": "F", "Vec3V": "V3", "Vec4V": "V4", "BoolV": "B", "QuatV": "Q", "VecU32V": "U", "VecI32V": "I",
        "Mat33V": "M33", "PxF32": "f", "PxU32": "u", "bool": "b"}
SKIP = {"V4LoadXYZW", "BLoad", "U4Load", "I4Load", "V4LoadU", "V4LoadA", "V3LoadU", "V3LoadA", "QuatVLoadU", "QuatVLoadA",
        "V4Load", "V3Load", "FLoad", "U4LoadXYZW", "U4LoadU", "U4LoadA", "I4LoadXYZW", "I4LoadU", "I4LoadA",
        "VecI32V_From_BoolV", "BGetBitMask", "V4U16CompareGt", "V4I16CompareGt", "V4U32Sel", "V4I32Sel",
        "VecU32V_ReinterpretFrom_Vec4V", "Vec4V_ReinterpretFrom_VecU32V", "VecI32V_ReinterpretFrom_Vec4V",
        "Vec4V_ReinterpretFrom_VecI32V", "V4U16Or", "V4U16And", "V4U16Andc", "V4I16Or", "V4I16And", "V4I16Andc",
        "V4I16Sub", "V4I16Add", "V4U16Sub", "V4U16Add", "V4U32SplatElement", "V4I32SplatElement"}


def norm_type(t):
    t = t.strip().replace("const ", "").replace("&", "").strip()
    t = {"FloatVArg": "FloatV", "Vec3VArg": "Vec3V", "Vec4VArg": "Vec4V", "BoolVArg": "BoolV", "QuatVArg": "QuatV",
         "VecU32VArg": "VecU32V", "VecI32VArg": "VecI32V"}.get(t, t)
    return t


def parse():
    txt = open(AOS, encoding="utf-8").read()
    funcs = {}
    for m in re.finditer(r"^EHD ([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\(([^)]*)\);", txt, re.M):
        ret, name, args = m.group(1), m.group(2), m.group(3).strip()
        if name in SKIP or name in funcs:
            continue
        if "template" in txt[max(0, m.start() - 40):m.start()]:
            continue
        argl = [a.strip() for a in args.split(",")] if args else []
        types = []
        ok = ret in SIZES or ret == "void"
        for a in argl:
            if "*" in a or re.search(r"[^t]&\s*\w+$", a.replace("const ", "")) and "const" not in a:
                ok = False
                break
            parts = a.rsplit(" ", 1)
            t = norm_type(parts[0] if len(parts) == 2 else a)
            if t not in SIZES:
                ok = False
                break
            types.append(t)
        if ok and ret != "void":
            funcs[name] = (ret, types)
    return funcs


def gen_side(funcs, ns, include):
    out = [include, "#include <cstring>", "#include <cstdint>"]
    out.append(f"using namespace {ns};")
    for name, (ret, types) in funcs.items():
        params = []
        off = 0
        for i, t in enumerate(types):
            n = SIZES[t]
            if t in ("PxF32",):
                params.append(f"in[{off}]")
            elif t == "PxU32":
                params.append(f"*(const uint32_t*)&in[{off}]")
            elif t == "bool":
                params.append(f"(in[{off}] != 0.0f)")
            else:
                params.append(f"*(const {t}*)&in[{off}]")
            off += 16 if n == 12 else 4  # 칸 맞춤: 네 칸씩 (Mat33V 는 16)
        call = f"{name}({', '.join(params)})"
        if ret == "PxF32":
            body = f"const float r = {call}; memcpy(out, &r, 4);"
        elif ret == "PxU32":
            body = f"const uint32_t r = {call}; memcpy(out, &r, 4);"
        elif ret == "bool":
            body = f"const float r = {call} ? 1.0f : 0.0f; memcpy(out, &r, 4);"
        else:
            body = f"const {ret} r = {call}; memcpy(out, &r, sizeof(r));"
        tag = "px" if "physx" in ns else "eng"
        out.append(f'extern "C" void {tag}_{name}(const float* in, float* out) {{ {body} }}')
    return "\n".join(out) + "\n"


def gen_main(funcs):
    lines = ['#include <cstdio>', '#include <cstring>', '#include <cstdint>', '#include <cmath>', '#include <random>',
             '#include <xmmintrin.h>', '#include <string>', '#include <vector>']
    for name in funcs:
        lines.append(f'extern "C" void px_{name}(const float*, float*); extern "C" void eng_{name}(const float*, float*);')
    lines.append("struct Fn { const char* name; void (*px)(const float*, float*); void (*eng)(const float*, float*); "
                 "const char* kinds; int outn; };")
    lines.append("static Fn kFns[] = {")
    for name, (ret, types) in funcs.items():
        kinds = ",".join(KIND[t] for t in types)
        outn = SIZES[ret] if ret in SIZES else 1
        lines.append(f'  {{"{name}", px_{name}, eng_{name}, "{kinds}", {outn}}},')
    lines.append("};")
    lines.append(r'''
static float fr(std::mt19937& g, int mode) {  // mode 0: 보통 값, 1: 큰/작은 값, 2: 특수값 섞기
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  if (mode == 0) return U(g) * 4.0f;
  if (mode == 1) { float e = U(g) * 30.0f; return U(g) * std::ldexp(1.0f, int(e)); }
  static const float sp[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1e-20f, 3.0e38f, 1.17549435e-38f, 0.70710678f};
  return (g() % 3 == 0) ? sp[g() % 9] : U(g);
}
static void fill(std::mt19937& g, const std::string& kinds, float* in, int mode) {
  memset(in, 0, 64 * sizeof(float));
  size_t pos = 0; int off = 0;
  while (pos <= kinds.size()) {
    size_t c = kinds.find(',', pos); std::string k = kinds.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    if (k.empty()) break;
    if (k == "F") { float v = fr(g, mode); for (int i = 0; i < 4; ++i) in[off + i] = v; off += 4; }
    else if (k == "V3") { for (int i = 0; i < 3; ++i) in[off + i] = fr(g, mode); in[off + 3] = 0.0f; off += 4; }
    else if (k == "V4" || k == "Q") { for (int i = 0; i < 4; ++i) in[off + i] = fr(g, mode); off += 4; }
    else if (k == "B") { for (int i = 0; i < 4; ++i) { uint32_t u = (g() & 1) ? 0xffffffffu : 0u; memcpy(&in[off + i], &u, 4); } off += 4; }
    else if (k == "U" || k == "I") { for (int i = 0; i < 4; ++i) { uint32_t u = uint32_t(g()) >> (g() % 24); memcpy(&in[off + i], &u, 4); } off += 4; }
    else if (k == "M33") { for (int c3 = 0; c3 < 3; ++c3) { for (int i = 0; i < 3; ++i) in[off + 4 * c3 + i] = fr(g, mode); in[off + 4 * c3 + 3] = 0.0f; } off += 16; }
    else if (k == "f") { in[off] = fr(g, mode); off += 4; }
    else if (k == "u") { uint32_t u = g() % 4; memcpy(&in[off], &u, 4); off += 4; }
    else if (k == "b") { in[off] = float(g() & 1); off += 4; }
    if (c == std::string::npos) break;
    pos = c + 1;
  }
}
int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 20000;
  const bool ftz = argc > 2 && atoi(argv[2]) != 0;
  if (ftz) _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));  // PhysX PxSIMDGuard 와 같은 FTZ+DAZ
  std::mt19937 g(12345);
  int nbad = 0, nfn = 0; long long total = 0;
  for (const Fn& f : kFns) {
    nfn++;
    int bad = 0; float in[64], a[64], b[64];
    for (int it = 0; it < iters; ++it) {
      fill(g, f.kinds, in, it % 3);
      memset(a, 0, sizeof a); memset(b, 0, sizeof b);
      f.px(in, a); f.eng(in, b);
      total++;
      if (memcmp(a, b, 4 * f.outn)) {
        bool bothnan = true;
        for (int i = 0; i < f.outn; ++i) { if (!(std::isnan(a[i]) && std::isnan(b[i])) && memcmp(&a[i], &b[i], 4)) bothnan = false; }
        if (bothnan) continue;
        if (bad++ < 2) { printf("[다름] %s 입력종류 %s:", f.name, f.kinds); for (int i = 0; i < f.outn; ++i) printf(" %.9g/%.9g", a[i], b[i]); printf("\n"); }
      }
    }
    if (bad) nbad++;
  }
  printf("aos 함수 %d 개 x %d 입력 (FTZ/DAZ %s): 다른 함수 %d, 비교 %lld\n", nfn, iters, ftz ? "켬" : "끔", nbad, total);
  return nbad ? 3 : 0;
}
''')
    return "\n".join(lines) + "\n"


def main():
    funcs = parse()
    open(f"{OUTDIR}/aos_diff_px.cpp", "w").write(
        "// 생성물 (gen_aos_test.py): 진짜 PhysX aos (SSE 하드웨어)\n" +
        gen_side(funcs, "physx::aos", '#include "foundation/PxVecMath.h"\n#include "foundation/PxVecTransform.h"'))
    open(f"{OUTDIR}/aos_diff_eng.cpp", "w").write(
        "// 생성물 (gen_aos_test.py): 엔진 aos (sse_emu 흉내)\n" +
        gen_side(funcs, "eng::aos", '#include "core/common/aos.h"'))
    open(f"{OUTDIR}/aos_diff_main.cpp", "w").write("// 생성물 (gen_aos_test.py)\n" + gen_main(funcs))
    print(f"함수 {len(funcs)} 개 시험 생성")


if __name__ == "__main__":
    main()
