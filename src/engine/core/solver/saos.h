// solver 전용 aos 흉내 (손으로 짬). PhysX 5.6.1 리눅스 빌드(clang, SSE2 경로, __SSE4_2__ 없음)의 aos 함수를
// 칸(lane)마다 같은 float 연산·같은 순서로 옮겨, 호스트(층 1)와 GPU(층 2)에서 PhysX 와 비트까지 같은 결과를 낸다.
// 원본: physx/include/foundation/PxVecMathSSE.h, unix/sse2/PxUnixSse2InlineAoS.h, PxVecQuat.h, PxVecMath.h (BSD-3)
//
// 표현 (docs/엔진_자체구현.md solver 절 "aos 흉내"):
//  - FV  = FloatV. SSE 에서는 네 칸이지만 PhysX 는 FloatV 의 네 칸을 늘 같게 쓴다(FLoad = 복제, V3Dot/V3SumElems 결과도
//          셔플 순서상 네 칸이 같다 — 덧셈 교환법칙은 비트까지 성립). 그래서 한 칸만 들고 다닌다(GPU 에서 4배 절약).
//  - V4  = Vec3V / Vec4V / QuatV. 네 칸(x,y,z,w)을 정확히 추적한다. Vec3V 의 W 칸도 PhysX 와 같은 값을 들고 다닌다
//          (V3Dot 이 aw*bw 를 더하므로 W 가 0 이 아니면 결과가 달라진다).
//  - BS  = FloatV 비교 결과(네 칸이 같은 BoolV). BV = 칸마다 다른 BoolV(0 또는 0xffffffff).
// 규칙: FNeg/V3Neg/V4Neg 는 0-x (부호 반전이 아님, PxVecMathSSE.h:406), min/max 는 minps/maxps 규칙(a<b?a:b / a>b?a:b),
//       나눗셈·제곱근은 IEEE 한 번(-prec-div/-prec-sqrt, SSE divps/sqrtps). FMA 축약 금지.
// 근사(rcpps/rsqrtps)는 solver(TGS 접촉) 경로에서 쓰지 않는다(DyTGSContactPrep*.cpp 전수 확인: V4Rsqrt 는 1/sqrt 정확판).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SV_HD __host__ __device__ __forceinline__
#define SV_HDN static __host__ __device__ __noinline__  // 큰 함수: 장치 코드에서 펼치지 않음(컴파일 시간·레지스터)
#else
#define SV_HD inline
#define SV_HDN inline
#endif

namespace eng {
namespace sv {

SV_HD uint32_t f2u(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
#endif
}
SV_HD float u2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}
SV_HD float fsqrt(float x) {
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return std::sqrt(x);
#endif
}
SV_HD float fdiv(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
SV_HD float fmaxps(float a, float b) { return a > b ? a : b; }  // maxps: 같거나 NaN 이면 두 번째
SV_HD float fminps(float a, float b) { return a < b ? a : b; }  // minps
SV_HD uint32_t pmaxu(uint32_t a, uint32_t b) { return a < b ? b : a; }  // PxMax<PxU32> (PxMath.h:75)
SV_HD uint32_t pminu(uint32_t a, uint32_t b) { return a < b ? a : b; }

struct FV { float f; };
struct V4 { float x, y, z, w; };
struct BS { uint32_t m; };
struct BV { uint32_t x, y, z, w; };
struct M33V { V4 col0, col1, col2; };
struct VecCrossV { V4 mL1, mR1; };
typedef V4 Vec3V;
typedef V4 Vec4V;
typedef V4 QuatV;
typedef FV FloatV;

constexpr float kMaxReal = 3.4028234663852885981170418348452e+38F;  // PX_MAX_F32
constexpr float kEpsReal = 1.192092896e-07F;                          // PX_EPS_F32

// ---------------- FloatV
SV_HD FV FLoad(float f) { return FV{f}; }
SV_HD FV FZero() { return FV{0.0f}; }
SV_HD FV FOne() { return FV{1.0f}; }
SV_HD FV FMax() { return FV{kMaxReal}; }
SV_HD FV FEps() { return FV{kEpsReal}; }
SV_HD FV FNeg(FV a) { return FV{0.0f - a.f}; }
SV_HD FV FAdd(FV a, FV b) { return FV{a.f + b.f}; }
SV_HD FV FSub(FV a, FV b) { return FV{a.f - b.f}; }
SV_HD FV FMul(FV a, FV b) { return FV{a.f * b.f}; }
SV_HD FV FDiv(FV a, FV b) { return FV{fdiv(a.f, b.f)}; }
SV_HD FV FRecip(FV a) { return FV{fdiv(1.0f, a.f)}; }
SV_HD FV FSqrt(FV a) { return FV{fsqrt(a.f)}; }
SV_HD FV FRsqrt(FV a) { return FV{fdiv(1.0f, fsqrt(a.f))}; }
SV_HD FV FScaleAdd(FV a, FV b, FV c) { return FV{a.f * b.f + c.f}; }
SV_HD FV FNegScaleSub(FV a, FV b, FV c) { return FV{c.f - a.f * b.f}; }
SV_HD FV FSel(BS c, FV a, FV b) { return c.m ? a : b; }
SV_HD BS FIsGrtr(FV a, FV b) { return BS{a.f > b.f ? 0xffffffffu : 0u}; }
SV_HD BS FIsGrtrOrEq(FV a, FV b) { return BS{a.f >= b.f ? 0xffffffffu : 0u}; }
SV_HD BS FIsEq(FV a, FV b) { return BS{a.f == b.f ? 0xffffffffu : 0u}; }
SV_HD FV FMax(FV a, FV b) { return FV{fmaxps(a.f, b.f)}; }
SV_HD FV FMin(FV a, FV b) { return FV{fminps(a.f, b.f)}; }
SV_HD FV FClamp(FV a, FV mn, FV mx) { return FMax(FMin(a, mx), mn); }
SV_HD uint32_t FAllGrtr(FV a, FV b) { return a.f > b.f ? 1u : 0u; }        // comigt_ss (NaN -> 0)
SV_HD uint32_t FAllGrtrOrEq(FV a, FV b) { return a.f >= b.f ? 1u : 0u; }
SV_HD uint32_t FAllEq(FV a, FV b) { return a.f == b.f ? 1u : 0u; }
SV_HD FV FAbs(FV a) { return FV{u2f(f2u(a.f) & 0x7fffffffu)}; }
SV_HD void FStore(FV a, float* p) { *p = a.f; }

// ---------------- BoolV
SV_HD BS BLoad(bool b) { return BS{b ? 0xffffffffu : 0u}; }
SV_HD BV BLoad(const bool* b) { return BV{b[0] ? 0xffffffffu : 0u, b[1] ? 0xffffffffu : 0u, b[2] ? 0xffffffffu : 0u, b[3] ? 0xffffffffu : 0u}; }
SV_HD BS BFFFF_S() { return BS{0u}; }
SV_HD BS BTTTT_S() { return BS{0xffffffffu}; }
SV_HD BV BV4(bool x, bool y, bool z, bool w) { return BV{x ? 0xffffffffu : 0u, y ? 0xffffffffu : 0u, z ? 0xffffffffu : 0u, w ? 0xffffffffu : 0u}; }
SV_HD BV BFFFF() { return BV{0, 0, 0, 0}; }
SV_HD BV BTTTT() { return BV4(1, 1, 1, 1); }
SV_HD BV BSplat(BS a) { return BV{a.m, a.m, a.m, a.m}; }
SV_HD BS BAnd(BS a, BS b) { return BS{a.m & b.m}; }
SV_HD BS BOr(BS a, BS b) { return BS{a.m | b.m}; }
SV_HD BS BAndNot(BS a, BS b) { return BS{a.m & ~b.m}; }  // _mm_andnot_ps(b, a)
SV_HD BV BAnd(BV a, BV b) { return BV{a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w}; }
SV_HD BV BOr(BV a, BV b) { return BV{a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w}; }
SV_HD BV BAndNot(BV a, BV b) { return BV{a.x & ~b.x, a.y & ~b.y, a.z & ~b.z, a.w & ~b.w}; }
// BSetX(v, f) = V4Sel(BFTTT, v, f): x 칸만 f 에서
SV_HD BV BSetX(BV v, BV f) { return BV{f.x, v.y, v.z, v.w}; }
SV_HD BV BSetY(BV v, BV f) { return BV{v.x, f.y, v.z, v.w}; }
SV_HD BV BSetZ(BV v, BV f) { return BV{v.x, v.y, f.z, v.w}; }
SV_HD BV BSetW(BV v, BV f) { return BV{v.x, v.y, v.z, f.w}; }
SV_HD uint32_t BAllEqTTTT(BV a) { return ((a.x >> 31) & (a.y >> 31) & (a.z >> 31) & (a.w >> 31)); }  // movemask == 15
SV_HD uint32_t BAllEqFFFF(BV a) { return ((a.x >> 31) | (a.y >> 31) | (a.z >> 31) | (a.w >> 31)) == 0 ? 1u : 0u; }

// ---------------- 칸 연산 도우미
SV_HD float selbits(uint32_t c, float a, float b) { return u2f((c & f2u(a)) | (~c & f2u(b))); }
SV_HD V4 v4(float x, float y, float z, float w) { return V4{x, y, z, w}; }

// ---------------- Vec3V (W 칸 추적)
SV_HD V4 V3Zero() { return V4{0.0f, 0.0f, 0.0f, 0.0f}; }
SV_HD V4 V3Load(float f) { return V4{f, f, f, 0.0f}; }
SV_HD V4 V3LoadU(const float* p) { return V4{p[0], p[1], p[2], 0.0f}; }   // _mm_set_ps(0, z, y, x)
SV_HD V4 V3LoadA(const float* p) { return V4{p[0], p[1], p[2], 0.0f}; }   // and gMaskXYZ -> W = +0
SV_HD V4 V3LoadU_SafeReadW(const float* p) { return V4{p[0], p[1], p[2], 0.0f}; }
SV_HD V4 V4ClearW(V4 v) { return V4{v.x, v.y, v.z, 0.0f}; }
SV_HD V4 Vec3V_From_Vec4V(V4 v) { return V4ClearW(v); }
SV_HD V4 Vec3V_From_Vec4V_WUndefined(V4 v) { return v; }
SV_HD V4 Vec4V_From_Vec3V(V4 v) { return v; }
SV_HD V4 Vec3V_From_FloatV(FV f) { return V4{f.f, f.f, f.f, 0.0f}; }
SV_HD V4 V3Splat(FV f) { return V4{f.f, f.f, f.f, 0.0f}; }
SV_HD V4 V3Merge(FV x, FV y, FV z) { return V4{x.f, y.f, z.f, 0.0f}; }
SV_HD FV V3GetX(V4 v) { return FV{v.x}; }
SV_HD FV V3GetY(V4 v) { return FV{v.y}; }
SV_HD FV V3GetZ(V4 v) { return FV{v.z}; }
SV_HD V4 V3Neg(V4 a) { return V4{0.0f - a.x, 0.0f - a.y, 0.0f - a.z, 0.0f - a.w}; }
SV_HD V4 V3Add(V4 a, V4 b) { return V4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
SV_HD V4 V3Sub(V4 a, V4 b) { return V4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
SV_HD V4 V3Mul(V4 a, V4 b) { return V4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
SV_HD V4 V3Scale(V4 a, FV b) { return V4{a.x * b.f, a.y * b.f, a.z * b.f, a.w * b.f}; }
SV_HD V4 V3ScaleInv(V4 a, FV b) { return V4{fdiv(a.x, b.f), fdiv(a.y, b.f), fdiv(a.z, b.f), fdiv(a.w, b.f)}; }
SV_HD V4 V3ScaleAdd(V4 a, FV b, V4 c) { return V3Add(V3Scale(a, b), c); }
SV_HD V4 V3NegScaleSub(V4 a, FV b, V4 c) { return V3Sub(c, V3Scale(a, b)); }
SV_HD V4 V3MulAdd(V4 a, V4 b, V4 c) { return V3Add(V3Mul(a, b), c); }
SV_HD V4 V3NegMulSub(V4 a, V4 b, V4 c) { return V3Sub(c, V3Mul(a, b)); }
SV_HD V4 V3Max(V4 a, V4 b) { return V4{fmaxps(a.x, b.x), fmaxps(a.y, b.y), fmaxps(a.z, b.z), fmaxps(a.w, b.w)}; }
SV_HD V4 V3Min(V4 a, V4 b) { return V4{fminps(a.x, b.x), fminps(a.y, b.y), fminps(a.z, b.z), fminps(a.w, b.w)}; }
SV_HD V4 V3Abs(V4 a) { return V3Max(a, V3Neg(a)); }
// V3Dot SSE2 (PxVecMathSSE.h:972): t0=a*b, 결과 = (t0.y + t0.w) + (t0.x + t0.z)
SV_HD FV V3Dot(V4 a, V4 b) {
  const float tx = a.x * b.x, ty = a.y * b.y, tz = a.z * b.z, tw = a.w * b.w;
  return FV{(ty + tw) + (tx + tz)};
}
// V3SumElems SSE2 (PxVecMathSSE.h:1384): (x + y) + z
SV_HD FV V3SumElems(V4 a) { return FV{(a.x + a.y) + a.z}; }
// V3Cross (PxVecMathSSE.h:997): W = aw*bw - aw*bw
SV_HD V4 V3Cross(V4 a, V4 b) {
  return V4{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, a.w * b.w - a.w * b.w};
}
SV_HD VecCrossV V3PrepareCross(V4 a) {
  VecCrossV v;
  v.mR1 = V4{a.z, a.x, a.y, a.w};  // z,x,y,w
  v.mL1 = V4{a.y, a.z, a.x, a.w};  // y,z,x,w
  return v;
}
SV_HD V4 V3Cross(const VecCrossV& a, V4 b) {  // mL1*l2 - mR1*r2, l2=(z,x,y,w) r2=(y,z,x,w)
  return V4{a.mL1.x * b.z - a.mR1.x * b.y, a.mL1.y * b.x - a.mR1.y * b.z, a.mL1.z * b.y - a.mR1.z * b.x,
            a.mL1.w * b.w - a.mR1.w * b.w};
}
SV_HD V4 V3Cross(V4 a, const VecCrossV& b) {  // mR1*r2 - mL1*l2, r2=(ay,az,ax,aw) l2=(az,ax,ay,aw)
  return V4{b.mR1.x * a.y - b.mL1.x * a.z, b.mR1.y * a.z - b.mL1.y * a.x, b.mR1.z * a.x - b.mL1.z * a.y,
            b.mR1.w * a.w - b.mL1.w * a.w};
}
SV_HD V4 V3Cross(const VecCrossV& a, const VecCrossV& b) {
  return V4{a.mL1.x * b.mR1.x - a.mR1.x * b.mL1.x, a.mL1.y * b.mR1.y - a.mR1.y * b.mL1.y,
            a.mL1.z * b.mR1.z - a.mR1.z * b.mL1.z, a.mL1.w * b.mR1.w - a.mR1.w * b.mL1.w};
}
SV_HD FV V3LengthSq(V4 a) { return V3Dot(a, a); }
SV_HD FV V3Length(V4 a) { return FV{fsqrt(V3Dot(a, a).f)}; }
SV_HD V4 V3Normalize(V4 a) { return V3ScaleInv(a, FV{fsqrt(V3Dot(a, a).f)}); }
SV_HD V4 V3Sel(BS c, V4 a, V4 b) { return c.m ? a : b; }
SV_HD V4 V3Sel(BV c, V4 a, V4 b) {
  return V4{selbits(c.x, a.x, b.x), selbits(c.y, a.y, b.y), selbits(c.z, a.z, b.z), selbits(c.w, a.w, b.w)};
}
SV_HD BV V3IsGrtr(V4 a, V4 b) { return BV4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w); }
SV_HD void V3StoreA(V4 a, float* p) { p[0] = a.x; p[1] = a.y; p[2] = a.z; }
SV_HD void V3StoreU(V4 a, float* p) { p[0] = a.x; p[1] = a.y; p[2] = a.z; }

// ---------------- Vec4V
SV_HD V4 V4Load(float f) { return V4{f, f, f, f}; }
SV_HD V4 V4LoadA(const float* p) { return V4{p[0], p[1], p[2], p[3]}; }
SV_HD V4 V4LoadU(const float* p) { return V4{p[0], p[1], p[2], p[3]}; }
SV_HD V4 V4LoadXYZW(float x, float y, float z, float w) { return V4{x, y, z, w}; }
SV_HD V4 V4Zero() { return V4{0.0f, 0.0f, 0.0f, 0.0f}; }
SV_HD V4 V4One() { return V4{1.0f, 1.0f, 1.0f, 1.0f}; }
SV_HD V4 V4Splat(FV f) { return V4{f.f, f.f, f.f, f.f}; }
SV_HD V4 V4Merge(FV x, FV y, FV z, FV w) { return V4{x.f, y.f, z.f, w.f}; }
SV_HD FV V4GetX(V4 v) { return FV{v.x}; }
SV_HD FV V4GetY(V4 v) { return FV{v.y}; }
SV_HD FV V4GetZ(V4 v) { return FV{v.z}; }
SV_HD FV V4GetW(V4 v) { return FV{v.w}; }
SV_HD V4 V4SetX(V4 v, FV f) { return V4{f.f, v.y, v.z, v.w}; }
SV_HD V4 V4SetY(V4 v, FV f) { return V4{v.x, f.f, v.z, v.w}; }
SV_HD V4 V4SetZ(V4 v, FV f) { return V4{v.x, v.y, f.f, v.w}; }
SV_HD V4 V4SetW(V4 v, FV f) { return V4{v.x, v.y, v.z, f.f}; }
SV_HD V4 V4Neg(V4 a) { return V4{0.0f - a.x, 0.0f - a.y, 0.0f - a.z, 0.0f - a.w}; }
SV_HD V4 V4Add(V4 a, V4 b) { return V4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
SV_HD V4 V4Sub(V4 a, V4 b) { return V4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
SV_HD V4 V4Mul(V4 a, V4 b) { return V4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
SV_HD V4 V4Scale(V4 a, FV b) { return V4{a.x * b.f, a.y * b.f, a.z * b.f, a.w * b.f}; }
SV_HD V4 V4Div(V4 a, V4 b) { return V4{fdiv(a.x, b.x), fdiv(a.y, b.y), fdiv(a.z, b.z), fdiv(a.w, b.w)}; }
SV_HD V4 V4Recip(V4 a) { return V4{fdiv(1.0f, a.x), fdiv(1.0f, a.y), fdiv(1.0f, a.z), fdiv(1.0f, a.w)}; }
SV_HD V4 V4Sqrt(V4 a) { return V4{fsqrt(a.x), fsqrt(a.y), fsqrt(a.z), fsqrt(a.w)}; }
SV_HD V4 V4Rsqrt(V4 a) { return V4Recip(V4Sqrt(a)); }  // _mm_div_ps(1, _mm_sqrt_ps(a))
SV_HD V4 V4MulAdd(V4 a, V4 b, V4 c) { return V4Add(V4Mul(a, b), c); }
SV_HD V4 V4NegMulSub(V4 a, V4 b, V4 c) { return V4Sub(c, V4Mul(a, b)); }
SV_HD V4 V4ScaleAdd(V4 a, FV b, V4 c) { return V4Add(V4Scale(a, b), c); }
SV_HD V4 V4Max(V4 a, V4 b) { return V4{fmaxps(a.x, b.x), fmaxps(a.y, b.y), fmaxps(a.z, b.z), fmaxps(a.w, b.w)}; }
SV_HD V4 V4Min(V4 a, V4 b) { return V4{fminps(a.x, b.x), fminps(a.y, b.y), fminps(a.z, b.z), fminps(a.w, b.w)}; }
SV_HD V4 V4Abs(V4 a) { return V4Max(a, V4Neg(a)); }
SV_HD V4 V4Clamp(V4 a, V4 mn, V4 mx) { return V4Max(V4Min(a, mx), mn); }
SV_HD V4 V4Sel(BV c, V4 a, V4 b) {
  return V4{selbits(c.x, a.x, b.x), selbits(c.y, a.y, b.y), selbits(c.z, a.z, b.z), selbits(c.w, a.w, b.w)};
}
SV_HD V4 V4Sel(BS c, V4 a, V4 b) { return c.m ? a : b; }
SV_HD BV V4IsGrtr(V4 a, V4 b) { return BV4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w); }
SV_HD BV V4IsGrtrOrEq(V4 a, V4 b) { return BV4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w); }
SV_HD BV V4IsEq(V4 a, V4 b) { return BV4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w); }
SV_HD V4 V4Cross(V4 a, V4 b) { return V3Cross(a, b); }  // 같은 셔플 (PxVecMathSSE.h:1767)
// V4Dot3 SSE2 (PxVecMathSSE.h:1747): (x + y) + z
SV_HD FV V4Dot3(V4 a, V4 b) { return FV{(a.x * b.x + a.y * b.y) + a.z * b.z}; }
SV_HD void V4StoreA(V4 a, float* p) { p[0] = a.x; p[1] = a.y; p[2] = a.z; p[3] = a.w; }
SV_HD void V4StoreU(V4 a, float* p) { p[0] = a.x; p[1] = a.y; p[2] = a.z; p[3] = a.w; }
SV_HD V4 V4UnpackXY(V4 a, V4 b) { return V4{a.x, b.x, a.y, b.y}; }  // unpacklo
SV_HD V4 V4UnpackZW(V4 a, V4 b) { return V4{a.z, b.z, a.w, b.w}; }  // unpackhi

// PX_TRANSPOSE_44_34 (PxVecMath.h:1267) — 입력을 덮어쓰는 매크로와 같은 결과(출력 A,B,C = 각 입력의 x,y,z)
SV_HD void Transpose44_34(V4 inA, V4 inB, V4 inC, V4 inD, V4& outA, V4& outB, V4& outC) {
  outA = V4UnpackXY(inA, inC);
  inA = V4UnpackZW(inA, inC);
  inC = V4UnpackXY(inB, inD);
  inB = V4UnpackZW(inB, inD);
  outB = V4UnpackZW(outA, inC);
  outA = V4UnpackXY(outA, inC);
  outC = V4UnpackXY(inA, inB);
}

// ---------------- Mat33V / 쿼터니언
SV_HD V4 M33MulV3(const M33V& a, V4 b) {  // PxVecMathSSE.h:2305
  const V4 v0 = V3Scale(a.col0, V3GetX(b));
  const V4 v1 = V3Scale(a.col1, V3GetY(b));
  const V4 v2 = V3Scale(a.col2, V3GetZ(b));
  return V3Add(V3Add(v0, v1), v2);
}
SV_HD V4 QuatVLoadU(const float* p) { return V4LoadU(p); }
SV_HD V4 QuatVLoadA(const float* p) { return V4LoadA(p); }
SV_HD V4 QuatRotate(V4 q, V4 v) {  // PxVecQuat.h:185
  const FV two = FLoad(2.0f);
  const FV nhalf = FLoad(-0.5f);
  const V4 u = Vec3V_From_Vec4V(q);
  const FV w = V4GetW(q);
  const FV w2 = FScaleAdd(w, w, nhalf);
  const V4 a = V3Scale(v, w2);
  const V4 temp = V3ScaleAdd(V3Cross(u, v), w, a);
  return V3Scale(V3ScaleAdd(u, V3Dot(u, v), temp), two);
}
SV_HD V4 QuatRotate4V(V4 q, V4 v) {  // PxVecQuat.h:208 (W 를 지우지 않음, V4Dot3)
  const FV two = FLoad(2.0f);
  const FV nhalf = FLoad(-0.5f);
  const V4 u = q;
  const FV w = V4GetW(q);
  const FV w2 = FScaleAdd(w, w, nhalf);
  const V4 a = V4Scale(v, w2);
  const V4 temp = V4ScaleAdd(V4Cross(u, v), w, a);
  return V4Scale(V4ScaleAdd(u, V4Dot3(u, v), temp), two);
}

}  // namespace sv
}  // namespace eng
