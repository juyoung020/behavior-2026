#!/usr/bin/env python3
"""PhysX 5.6.1 CPU 소스 -> 우리 엔진(eng::px) 헤더 기계 번역기 (한 번 돌리는 도구).

왜: 접촉 생성(PCM·GJK·EPA)은 PhysX 의 SIMD(aos) 코드로 짜여 있고, 비트까지 같게 옮기려면 식·연산 순서를
한 글자도 바꾸지 않아야 한다. 손으로 옮겨 적으면 실수가 생기므로, 원본 문장은 그대로 두고
(1) _mm_* SSE 명령 -> em_* (sse_emu.h, 칸마다 같은 float 연산), (2) PX_ 매크로 -> 우리 매크로,
(3) 전처리 분기 -> 리눅스 clang SSE2 빌드에서 켜지는 쪽만, (4) 파일별 손질(rules) 만 한다.
결과물은 PhysX 에 링크하지 않고 호스트(C++)와 GPU(CUDA) 둘 다 컴파일된다.

사용: python3 translate.py <physx 루트> <출력 폴더>
"""
import re
import sys
from pathlib import Path

KNOWN = {  # 리눅스 carbonite checked 빌드 (clang 14, x86_64, SSE2) 에서의 값. PX_CHECKED 는 검사·경고만이라 0 으로 뺀다.
    'PX_DOXYGEN': 0, 'PX_EMSCRIPTEN': 0, 'PX_DEBUG': 0, 'PX_CHECKED': 0, 'PX_CLANG': 1, 'PX_LINUX': 1,
    'PX_GCC_FAMILY': 1, 'PX_VC': 0, 'PX_WINDOWS': 0, 'PX_WINDOWS_FAMILY': 0, 'PX_OSX': 0, 'PX_APPLE_FAMILY': 0,
    'PX_UNIX_FAMILY': 1, 'PX_LINUX_FAMILY': 1, 'PX_INTEL_FAMILY': 1, 'PX_ARM_FAMILY': 0, 'PX_NEON': 0,
    'PX_SWITCH': 0, 'PX_X64': 1, 'PX_X86': 0, 'PX_A64': 0, 'PX_P64_FAMILY': 1, 'COMPILE_VECTOR_INTRINSICS': 1,
    'PX_CUDA_COMPILER': 0, 'PX_SUPPORT_PVD': 0, 'PX_ENABLE_SIM_STATS': 0, 'PCM_LOW_LEVEL_DEBUG': 0,
    'PX_ENABLE_ASSERTS': 0, 'PX_ENABLE_DEBUG_VISUALIZATION': 0, 'PX_SUPPORT_OMNI_PVD': 0, 'PX_SUPPORT_GPU_PHYSX': 0,
    'PX_ENABLE_FEATURES_UNDER_CONSTRUCTION': 0, 'PX_GJK_USE_SIMD': 1, 'PX_PS4': 0, 'PX_PS5': 0,
    '__SSE4_2__': None, 'PX_SIMD_DISABLED': None, 'PX_SPU': 0, 'PX_PHYSX_STATIC_LIB': 1, 'PX_PPC': 0,
    'PX_DEBUG_GJK': 0, 'PX_INTEL_FAMILY_SSE': 1, 'PX_SSE2': 1, 'PX_ENABLE_INVARIANT_CHECKS': 0,
    'EPA_DEBUG': 0, 'GJK_DEBUG': 0, 'PX_GJK_EPA_DEBUG': 0, 'PCM_BOX_HULL_DEBUG': 0, 'PX_PROFILE': 0,
    'PX_CUDA_ARCH': 0, '__CUDA_ARCH__': None, '__CUDACC__': None, 'PX_ENABLE_PROFILE': 0, 'PX_NVTX': 0,
    'PX_SUPPORT_EXTERN_TEMPLATE': 0, 'PX_MAX_ALIGN': 16, 'PX_GJK_TEST_EPA': 0, '__EMSCRIPTEN__': None, '__SSE2__': 1, '__GNUC__': 4, '_DEBUG': None, '__clang__': 1, 'NDEBUG': 1, '_MSC_VER': None, '__linux__': 1, 'PX_GPU_BROADPHASE': 0, 'PX_ENABLE_GPU': 0, 'PX_ARM': 0, '__BIG_ENDIAN__': None, '_XBOX': None,
}


class CondError(Exception):
    pass


def lenient_unknown(name, path, lineno):
    """파일 안 설정 매크로(예: ABP_USE_INTEGER_XS)는 정의 안 된 것으로 본다. PX_/플랫폼 매크로는 엄격히 멈춘다."""
    if name.startswith('PX_') or name.startswith('__'):
        raise CondError(f'{path}:{lineno}: 모르는 매크로 {name}')
    sys.stderr.write(f'  (정의 안 됨으로 봄) {name} @ {path}:{lineno}\n')
    KNOWN[name] = None


def eval_cond(expr, path, lineno):
    e = expr.split('//')[0].strip()
    e = re.sub(r'defined\s*\(\s*(\w+)\s*\)', lambda m: 'DEF_' + m.group(1), e)
    e = re.sub(r'defined\s+(\w+)', lambda m: 'DEF_' + m.group(1), e)
    toks = re.findall(r'[A-Za-z_]\w*', e)
    env = {}
    for t in toks:
        if t.startswith('DEF_'):
            name = t[4:]
            if name not in KNOWN:
                raise CondError(f'{path}:{lineno}: 모르는 매크로 {name} in "{expr}"')
            env[t] = 1 if KNOWN[name] is not None else 0
        else:
            if t not in KNOWN:
                raise CondError(f'{path}:{lineno}: 모르는 매크로 {t} in "{expr}"')
            v = KNOWN[t]
            env[t] = 0 if v is None else v
    py = e.replace('&&', ' and ').replace('||', ' or ')
    py = re.sub(r'!(?!=)', ' not ', py)
    return bool(eval(py, {}, env))


def preprocess(text, path):
    """#if 분기를 KNOWN 으로 풀고 #include·include guard·#pragma 를 지운다."""
    lines = text.split('\n')
    out = []
    stack = []  # (active_parent, this_taken, currently_active)
    active = True
    guard = None
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        m = re.match(r'#\s*(\w+)\s*(.*)', s)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ('if', 'ifdef', 'ifndef'):
                if d == 'ifndef' and guard is None and not out_nonblank(out) and re.match(r'\w+_H\b|\w+_H_*$', rest.split()[0] if rest else ''):
                    guard = rest.split()[0]
                    stack.append((active, True, active, 'guard'))
                    continue
                if not active:
                    stack.append((active, True, False, ''))
                    continue
                if d == 'ifdef':
                    name = rest.split()[0]
                    if name not in KNOWN:
                        lenient_unknown(name, path, i)
                    c = KNOWN[name] is not None
                elif d == 'ifndef':
                    name = rest.split()[0]
                    nxt = lines[i].strip() if i < len(lines) else ''
                    if name not in KNOWN and re.match(r'#\s*define\s+' + name + r'(\s|$)', nxt):
                        KNOWN[name] = None  # "없으면 정의" 꼴
                    if name not in KNOWN:
                        lenient_unknown(name, path, i)
                    c = KNOWN[name] is None
                else:
                    c = eval_cond(rest, path, i)
                stack.append((active, c, active and c, ''))
                active = active and c
                continue
            if d == 'elif':
                parent, taken, cur, kind = stack.pop()
                if not parent or taken:
                    stack.append((parent, True, False, kind))
                    active = False
                else:
                    c = eval_cond(rest, path, i)
                    stack.append((parent, c, c, kind))
                    active = c
                continue
            if d == 'else':
                parent, taken, cur, kind = stack.pop()
                stack.append((parent, True, parent and not taken, kind))
                active = parent and not taken
                continue
            if d == 'endif':
                parent, taken, cur, kind = stack.pop()
                active = parent
                continue
            if not active:
                continue
            if d == 'define' and guard and rest.split()[0] == guard:
                continue
            if d == 'define':
                dm = re.match(r'(\w+)(\(?)\s*(.*)', rest)
                if dm and not dm.group(2) and (dm.group(1) not in KNOWN or (KNOWN[dm.group(1)] is None and not dm.group(1).startswith('PX_'))):
                    val = dm.group(3).strip()
                    KNOWN[dm.group(1)] = int(val) if re.fullmatch(r'\d+', val) else 1
            if d == 'undef':
                KNOWN[rest.split()[0]] = None
            if d == 'include':
                base = rest.strip().strip('"<>').split('/')[-1]
                if base in INLINE_INCLUDES:
                    out.append('@@INCLUDE ' + base + '@@')
                continue
            if d == 'pragma':
                continue
            out.append(raw)
            continue
        if active:
            out.append(raw)
    if stack:
        raise CondError(f'{path}: #if 짝이 안 맞음 ({len(stack)})')
    return '\n'.join(out)


INLINE_INCLUDES = set()
FILE_INDEX = {}


def expand_includes(text, root, seen):
    def rep(m):
        base = m.group(1)
        if base in seen:
            return ''
        seen.add(base)
        rel = FILE_INDEX[base]
        t = strip_license((root / rel).read_text())
        t = preprocess(t, rel)
        return f'// ----- (포함) physx/{rel}\n' + expand_includes(t, root, seen)
    return re.sub(r'@@INCLUDE (\S+)@@', rep, text)


def out_nonblank(out):
    for l in out:
        t = l.strip()
        if t and not t.startswith('//'):
            return True
    return False


def strip_license(text):
    # 파일 맨 앞 // 주석 덩어리(BSD 라이선스) 제거. 출처는 번역본 머리말에 적는다.
    lines = text.split('\n')
    i = 0
    while i < len(lines) and (lines[i].startswith('//') or not lines[i].strip()):
        i += 1
    return '\n'.join(lines[i:])


def replace_px_align(text):
    # PX_ALIGN(16, X) -> X  (em128 는 이미 16 정렬, 흉내에서는 정렬이 값에 영향 없음)
    out = []
    i = 0
    while True:
        j = text.find('PX_ALIGN(', i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len('PX_ALIGN(')
        depth = 1
        comma = None
        p = k
        while depth:
            c = text[p]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 1 and comma is None:
                comma = p
            p += 1
        align = text[k:comma].strip()
        content = text[comma + 1:p - 1].strip()
        # 줄 앞(공백만 앞에 있음)에서 "형 이름" 선언이면 alignas 로 정렬을 지킨다 (PxContactPoint 등 배치가 원본과 같아야 함).
        # 형 뒤에 붙은 꼴(Vec3V PX_ALIGN(16, col0))이나 형만 든 꼴(PX_ALIGN(16, PxF32))은 내용만 (em128 은 이미 16 정렬).
        line_start = text.rfind('\n', 0, j) + 1
        before = text[line_start:j]
        if before.strip() == '' and len(re.findall(r'[A-Za-z_]\w*', content.split('[')[0].split('=')[0])) >= 2:
            out.append(f'alignas({align}) {content}')
        else:
            out.append(content)
        i = p
    return ''.join(out)


TOKEN_RULES = [
    (r'\b_mm_(\w+)', r'em_\1'),
    (r'\b_MM_SHUFFLE\b', 'EM_SHUFFLE'),
    (r'\b__m128i\b', 'em128i'),
    (r'\b__m128\b', 'em128'),
    (r'\bPX_FORCE_INLINE\b', 'EHD'),
    (r'\bPX_CUDA_CALLABLE\s+', ''),
    (r'\bPX_CUDA_CALLABLE\b', ''),
    (r'\bPX_INLINE\b', 'EHDI'),
    (r'\bPX_NOINLINE\b', ''),
    (r'\bPX_RESTRICT\b', ''),
    (r'\bPX_ALIGN_PREFIX\(\s*\d+\s*\)', ''),
    (r'\bPX_ALIGN_SUFFIX\(\s*\d+\s*\)', ''),
    (r'\bPX_ASSERT\b', 'E_ASSERT'),
    (r'\bPX_ASSERT_WITH_MESSAGE\b', 'E_ASSERT2'),
    (r'\bPX_UNUSED\b', 'E_UNUSED'),
    (r'\bPX_COMPILE_TIME_ASSERT\b', 'E_STATIC_ASSERT'),
    (r'\bPX_SHARED_ASSERT\b', 'E_ASSERT'),
    (r'\bPX_INTEL_FAMILY\b', '1'),
    (r'\bASSERT_IS', 'E_ASSERT_IS'),
    (r'\bPX_TRANSPOSE_', 'E_TRANSPOSE_'),
    (r'\bPX_FPCLASS_', 'E_FPCLASS_'),
    (r'\bnamespace physx\b', 'namespace px'),
    (r'\bphysx::', 'px::'),
    (r'\bPX_PHYSX_COMMON_API\b', ''),
    (r'\bPX_PHYSX_CORE_API\b', ''),
    (r'\bPX_FOUNDATION_API\b', ''),
    (r'\bPX_PHYSX_GPU_API\b', ''),
    (r'\bPX_DEPRECATED\b', ''),
    (r'\bPX_OFFSET_OF\b', 'E_OFFSET_OF'),
    (r'\bPX_OFFSET_OF_RT\b', 'E_OFFSET_OF'),
    (r'\bPX_CATCH_UNDEFINED_ENABLE_SIM_STATS\b', ''),
    (r'\bPX_NOCOPY\b', 'E_NOCOPY'),
    (r'\bPX_PLACEMENT_NEW\b', 'E_PLACEMENT_NEW'),
    (r'\bPX_FREE\b', 'E_FREE'),
    (r'\bPX_DELETE\b', 'E_DELETE'),
]
# 이름만 바꿔 우리 foundation 에 정의해 둔 상수
CONST_RENAMES = ['PX_MAX_F32', 'PX_MAX_F64', 'PX_EPS_F32', 'PX_EPS_F64', 'PX_MAX_REAL', 'PX_EPS_REAL',
                 'PX_NORMALIZATION_EPSILON', 'PX_MAX_I8', 'PX_MIN_I8', 'PX_MAX_U8', 'PX_MIN_U8', 'PX_MAX_I16',
                 'PX_MIN_I16', 'PX_MAX_U16', 'PX_MIN_U16', 'PX_MAX_I32', 'PX_MIN_I32', 'PX_MAX_U32', 'PX_MIN_U32',
                 'PX_PI', 'PX_HALF_PI', 'PX_TWO_PI', 'PX_INV_PI', 'PX_INV_TWO_PI', 'PX_PIDIV2', 'PX_PIDIV4',
                 'PX_SQRT2', 'PX_SQRT3', 'PX_INV_SQRT2', 'PX_INV_SQRT3', 'PX_MAX_SWEEP_DISTANCE', 'PX_MIN_F32',
                 'PX_MAX_F32_BITS', 'PX_INVALID_U32', 'PX_INVALID_U16', 'PX_INVALID_NODE', 'PX_MAX_BOUNDS_EXTENTS',
                 'PX_GLOBALCONST', 'PX_VECTORF32', 'PX_SIGN_BITMASK']

LANE = {'x': '0', 'y': '1', 'z': '2', 'w': '3'}
# 번역 뒤 손질: GPU 에서 못 읽는 전역 상수 배열, 형 다른 참조로 칸 읽기(엄격 별칭 규칙 위반) 등을 같은 값의 식으로 바꾼다
POST_RULES = [
    (r'reinterpret_cast<(?:const )?PxVec[34]&>\((\w+)\)\.([xyzw])', lambda m: f'{m.group(1)}.f[{LANE[m.group(2)]}]'),
    (r'reinterpret_cast<PxVec3&>\((\w+)\) = (\w+);', r'\1.f[0] = \2.x; \1.f[1] = \2.y; \1.f[2] = \2.z;'),
    (r'em_load_ps\(minus1w\)', 'em_set_ps(-1.0f, 0.0f, 0.0f, 0.0f)'),
    (r'const PxF32 minus1w\[4\] = \{[^}]*\};', ''),
    (r'V4LoadA\(internalSimd::gMaskXYZ\)', 'em_castsi128_ps(em_set_epi32(0, -1, -1, -1))'),
    (r'const PxF32 gMaskXYZ\[4\] = \{[^}]*\};', ''),
    (r'reinterpret_cast<const Vec3V&>\(f\)', 'em_loadu_ps(&f.x)'),
    (r'#define EPX_GLOBALCONST extern const __attribute__\(\(weak\)\)', '#define EPX_GLOBALCONST static constexpr'),
    # 원본 .cpp 의 전역 이름공간 static 함수를 ::f 로 부르는 곳 (우리 번역은 전부 eng 안이라 :: 를 뗀다)
    (r'(?<![\w>])::(intersectSegmentAABB)\(', r'\1('),
    (r'(?<![\w>])::px::', 'px::'),  # 원본 ::physx:: (우리 px 는 eng 안)
    # GuConvexSupportTable.cpp:33 boxVertexTable (함수 호출로 초기화되는 전역 표) -> 같은 값을 번호로 만드는 함수
    (r'EPX_PHYSX_COMMON_API\s+extern const aos::BoolV boxVertexTable\[8\];|extern const aos::BoolV boxVertexTable\[8\];', ''),
    (r'\bboxVertexTable\[(\w+)\]', r'boxVertexTable_get(\1)'),
    # 스택 할당(alloca) -> 고정 크기 지역 배열 (GPU 에 alloca 없음). 개수 상한 E_ALLOCA_N (볼록 꼭짓점 255, 접촉 256)
    (r'(\w+)\*\s*(\w+)\s*=\s*reinterpret_cast<\w+\*>\(PxAlloca(?:Aligned)?\(sizeof\((\w+)\)\s*\*\s*([\w.]+)(?:\s*,\s*16)?\)\);',
     r'\1 \2_buf[E_ALLOCA_N]; \1* \2 = \2_buf;'),
]


def translate_text(text, path, root=None, seen=None):
    text = strip_license(text)
    text = preprocess(text, path)
    if root is not None:
        text = expand_includes(text, root, seen)
    text = replace_px_align(text)
    for pat, rep in TOKEN_RULES:
        text = re.sub(pat, rep, text)
    for c in CONST_RENAMES:
        text = re.sub(r'\b' + c + r'\b', 'E' + c, text)
    for pat, rep in POST_RULES:
        text = re.sub(pat, rep, text)
    # 남은 PX_ 매크로(번역본 안에서 정의되는 것들)는 EPX_ 로: 시험 프로그램에서 PhysX 헤더와 같이 써도 겹치지 않게
    text = re.sub(r'\bPX_(\w+)', r'EPX_\1', text)
    return text


def leftover_px(text):
    return sorted(set(re.findall(r'\bPX_\w+', text)))


GTABLE = re.compile(r'^[ \t]*(?:static\s+)?const\s+(\w+)\s+(\w+)\s*\[\s*(\w*)\s*\]\s*=\s*(\{.*?\})\s*;', re.S | re.M)


def ns_level_positions(text):
    """각 글자 위치가 이름공간 수준(함수·클래스 밖)인지: 괄호 짝으로 센다."""
    sh = shadow_code(text)
    flags = bytearray(len(sh))
    stack = []
    head_start = 0
    for i, c in enumerate(sh):
        flags[i] = 1 if all(k == 'ns' for k in stack) else 0
        if c == '{':
            top_ns = all(k == 'ns' for k in stack)
            stack.append(classify(sh[head_start:i]) if top_ns else 'blk')
            head_start = i + 1
        elif c == '}':
            if stack:
                stack.pop()
            head_start = i + 1
        elif c == ';':
            head_start = i + 1
    return flags


def device_tables(text):
    """이름공간 수준 상수 표(const T name[N] = {...};)를 호스트·장치 둘 다 읽을 수 있게 바꾼다.
    E_GTABLE 가 호스트 표 + __constant__ 표 + 고르는 함수 name_p() 를 만들고, 쓰는 곳 name[ 는 name_p()[ 로."""
    names = []
    holders = []  # 정의·extern 선언은 자리표로 빼 두었다가 쓰는 곳 이름을 바꾼 뒤 되돌린다
    nsflags = ns_level_positions(text)

    def rep(m):
        t, name, n, body = m.group(1), m.group(2), m.group(3), m.group(4)
        k = m.start(1)
        if not nsflags[k]:
            return m.group(0)  # 함수 안 지역 표는 그대로
        names.append(name)
        n = n or str(body.count(',') + 1)
        indent = re.match(r'[ \t]*', m.group(0)).group(0)
        holders.append(f'{indent}E_GTABLE({t}, {name}, {n}, {" ".join(body.split())})')
        return f'@@GT{len(holders) - 1}@@'
    text = GTABLE.sub(rep, text)
    for name in names:
        def rep_ext(m, name=name):
            holders.append(f'E_GTABLE_DECL({m.group(1)}, {name});')
            return f'@@GT{len(holders) - 1}@@'
        text = re.sub(r'(?:EHDV\s+)?extern\s+const\s+(\w+)\s+' + name + r'\s*\[\s*\w*\s*\]\s*;', rep_ext, text)
    for name in names:
        text = re.sub(r'\b' + name + r'\b', name + '_p()', text)
    for k, h in enumerate(holders):
        text = text.replace(f'@@GT{k}@@', h)
    return text


def shadow_code(text):
    """주석·문자열·전처리 줄을 공백으로 가린 사본 (길이 같음). 괄호 짝 세기에 쓴다."""
    out = list(text)
    i, n = 0, len(text)
    line_start = True
    while i < n:
        c = text[i]
        if line_start and c in ' \t':
            i += 1
            continue
        if line_start and c == '#':
            j = i
            while j < n and text[j] != '\n':
                if text[j] == '\\' and j + 1 < n and text[j + 1] == '\n':
                    j += 2
                    continue
                j += 1
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
            continue
        line_start = False
        if c == '\n':
            line_start = True
            i += 1
            continue
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
            continue
        if text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
            continue
        if c in '"\'':
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            for k in range(i + 1, min(j, n)):
                out[k] = ' '
            i = j + 1
            continue
        i += 1
    return ''.join(out)


ACCESS = re.compile(r'^\s*(?:public|private|protected)\s*:(?!:)')
MACRO_CALL = re.compile(r'^\s*E_[A-Z_]+\s*\([^()]*\)\s*')
TEMPLATE = re.compile(r'^\s*template\s*<')


def skip_prefix(head):
    """머리말 앞의 접근 지정자·매크로 호출을 건너뛴 위치."""
    off = 0
    while True:
        m = ACCESS.match(head[off:]) or MACRO_CALL.match(head[off:])
        if not m:
            break
        off += m.end()
    return off


def after_template(head, off):
    m = TEMPLATE.match(head[off:])
    if not m:
        return off
    p = off + m.end()
    depth = 1
    while p < len(head) and depth:
        if head[p] == '<':
            depth += 1
        elif head[p] == '>':
            depth -= 1
        p += 1
    return after_template(head, p)


def classify(head):
    """'ns' | 'cls' | 'fn' | 'other'"""
    h = head[after_template(head, 0):].strip()
    h = re.sub(r'alignas\s*\(\s*\w+\s*\)', '', h).strip()
    if h.startswith('E_'):
        return 'other'
    if re.match(r'(namespace\b|extern\s*$)', h) or h == 'namespace':
        return 'ns'
    pre = h.split('(')[0]
    if re.search(r'\b(class|struct|union)\b', pre) and not re.search(r'\boperator\b', pre):
        return 'cls'
    if re.search(r'\benum\b', pre):
        return 'other'
    if '(' in h and '=' not in pre and not re.match(r'(typedef|using|return)\b', h):
        return 'fn'
    return 'other'


def annotate_functions(text):
    """이름공간·클래스 수준의 함수 정의·선언 앞에 EHDI(정의)/EHDV(선언)를 붙인다 (CUDA 에서 호스트·장치 둘 다 되게)."""
    sh = shadow_code(text)
    inserts = []  # (pos, str)
    stack = []
    head_start = 0
    i, n = 0, len(sh)
    while i < n:
        c = sh[i]
        top = stack[-1] if stack else 'ns'
        at_decl_level = top in ('ns', 'cls')
        if c == '{':
            if at_decl_level:
                head = sh[head_start:i]
                off = skip_prefix(head)
                kind = classify(head[off:])
                if kind == 'fn':
                    body = head[off:]
                    if not re.search(r'\b(EHD|EHDI|EHDV|__host__|__device__|E?PX_\w*INLINE\w*)\b', body):
                        ins = after_template(head, off)
                        while ins < len(head) and head[ins] in ' \t\n':
                            ins += 1
                        inserts.append((head_start + ins, 'EHDI '))
                stack.append(kind)
            else:
                stack.append('blk')
            head_start = i + 1
        elif c == '}':
            if stack:
                stack.pop()
            head_start = i + 1
        elif c == ';' and at_decl_level:
            head = sh[head_start:i]
            off = skip_prefix(head)
            body = head[off:].strip()
            if body and classify(body) == 'fn' and not re.search(r'\b(EHD|EHDI|EHDV|__host__|__device__|friend|E?PX_\w*INLINE\w*)\b', body) \
                    and not re.match(r'E_', body):
                ins = after_template(head, off)
                while ins < len(head) and head[ins] in ' \t\n':
                    ins += 1
                inserts.append((head_start + ins, 'EHDV '))
            head_start = i + 1
        elif c == ':' and at_decl_level and ACCESS.match(sh[head_start:i + 1]):
            head_start = i + 1
        i += 1
    out = text
    for pos, s in sorted(inserts, reverse=True):
        out = out[:pos] + s + out[pos:]
    return out


def main():
    root = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    for f in root.rglob('*.h'):
        sf = str(f)
        if 'gpu' not in sf.lower() and '/windows' not in sf and '/neon' not in sf and '/compiler/' not in sf and '/install/' not in sf:
            FILE_INDEX.setdefault(f.name, str(f.relative_to(root)))
    jobs = []
    for l in Path(sys.argv[3]).read_text().split('\n'):
        if l.startswith('INLINE '):
            INLINE_INCLUDES.update(l.split()[1:])
        elif l.strip() and not l.startswith('#'):
            jobs.append(l.split())
    seen = set()  # 모든 조각에 걸쳐 한 번만 펼친다 (앞 조각에서 번역한 헤더는 다시 넣지 않음)
    for job in jobs:
        out_name, srcs = job[0], job[1:]
        annotate = False
        if srcs and srcs[0] == 'ANNOTATE':
            annotate, srcs = True, srcs[1:]
        body = []
        for s in srcs:
            if s.startswith('@'):  # 손으로 짠 조각을 이 자리에 끼움
                body.append(f'#include "core/contact/px/{s[1:]}"\n')
                continue
            p = root / s
            seen.add(p.name)
            t = translate_text(p.read_text(), s, root, seen)
            body.append(f'// ===== 원본: physx/{s}\n' + t.strip('\n') + '\n')
        text = '\n'.join(body)
        if annotate:
            text = annotate_functions(text)
            text = device_tables(text)
        (outdir / out_name).write_text(text)
        lo = leftover_px(text)
        print(f'{out_name}: {len(text.splitlines())} 줄, 남은 PX_ 토큰: {" ".join(lo)}')


if __name__ == '__main__':
    main()
