// joints 모듈: PhysX SSE2 aos(Vec3V/Vec4V/FloatV) 연산을 칸(lane) 단위로 그대로 흉내 낸 최소 집합 (CPU·CUDA 공용).
// 왜 필요한가: PhysX 조인트 준비·풀이 코드는 SIMD 로 짜여 있고, 가로 합(내적·합)의 더하는 순서와 w 칸 값이 스칼라 식과
// 다르다. 예) V3Dot = (xx+zz)+(yy+ww) (스칼라 PxVec3::dot 은 (xx+yy)+zz), V4Dot3 = (xx+yy)+zz, FNeg = 0-x.
// 원본: physx/include/foundation/PxVecMathSSE.h, unix/sse2/PxUnixSse2InlineAoS.h, PxVecQuat.h, PxVecMath.h:1336
// 전제: 리눅스 x86_64 PhysX 는 -msse4.2 없이 빌드된다(physx/source/compiler/cmake/linux/CMakeLists.txt) -> __SSE4_2__ 경로가 아니라
//       SSE2 경로. FMA 없음. min/max 는 minps/maxps 의미(같거나 NaN 이면 두 번째 인자).
// common/aos.h(리드 작성 중)가 들어오면 그쪽으로 옮긴다 — 이름을 PhysX 와 같게 둬서 바꿔 끼우기 쉽게 했다.
#pragma once
#include <cstdint>

#include "../common/pmath.h"

namespace eng {
namespace js {

// Vec3V / Vec4V: 4칸 모두 들고 다닌다 (Vec3V 의 w 칸도 SSE 에서 실제로 계산되므로 그대로 따라간다).
struct L4 { float x, y, z, w; };
typedef float F;  // FloatV: SSE 에서는 네 칸이 같은 값(splat). 여기서는 한 칸만 둔다.

// ---- 적재·저장 (PxVecMathSSE.h V3LoadU/V3LoadA/V4LoadA, PxVecMath.h:1336 V3LoadU_SafeReadW)
EHD L4 V3Load(const V3& v) { return L4{v.x, v.y, v.z, 0.0f}; }            // w = +0 (마스크/0.0f)
EHD L4 V4Load(const float* p) { return L4{p[0], p[1], p[2], p[3]}; }
EHD L4 V4Load(float x, float y, float z, float w) { return L4{x, y, z, w}; }
EHD V3 V3Store(const L4& a) { return V3{a.x, a.y, a.z}; }
EHD L4 V4ClearW(const L4& a) { return L4{a.x, a.y, a.z, 0.0f}; }          // and 마스크 -> +0
EHD L4 V3FromV4(const L4& a) { return V4ClearW(a); }                      // Vec3V_From_Vec4V
EHD L4 V3Zero() { return L4{0.0f, 0.0f, 0.0f, 0.0f}; }
EHD L4 V4SetW(const L4& a, F w) { return L4{a.x, a.y, a.z, w}; }

// ---- 칸 연산
EHD L4 V4Add(const L4& a, const L4& b) { return L4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
EHD L4 V4Sub(const L4& a, const L4& b) { return L4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
EHD L4 V4Mul(const L4& a, const L4& b) { return L4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
EHD L4 V4Scale(const L4& a, F s) { return L4{a.x * s, a.y * s, a.z * s, a.w * s}; }
EHD L4 V4ScaleAdd(const L4& a, F b, const L4& c) { return V4Add(V4Scale(a, b), c); }
EHD L4 V4NegScaleSub(const L4& a, F b, const L4& c) { return V4Sub(c, V4Scale(a, b)); }
EHD L4 V4MulAdd(const L4& a, const L4& b, const L4& c) { return V4Add(V4Mul(a, b), c); }
EHD L4 V4Neg(const L4& a) { return L4{0.0f - a.x, 0.0f - a.y, 0.0f - a.z, 0.0f - a.w}; }  // _mm_sub_ps(0, a)
// Vec3V 연산은 같은 SSE 명령 (V3Add = _mm_add_ps …)
EHD L4 V3Add(const L4& a, const L4& b) { return V4Add(a, b); }
EHD L4 V3Sub(const L4& a, const L4& b) { return V4Sub(a, b); }
EHD L4 V3Mul(const L4& a, const L4& b) { return V4Mul(a, b); }
EHD L4 V3Scale(const L4& a, F s) { return V4Scale(a, s); }
EHD L4 V3ScaleAdd(const L4& a, F b, const L4& c) { return V4ScaleAdd(a, b, c); }
EHD L4 V3NegScaleSub(const L4& a, F b, const L4& c) { return V4NegScaleSub(a, b, c); }
EHD L4 V3MulAdd(const L4& a, const L4& b, const L4& c) { return V4MulAdd(a, b, c); }

// ---- 가로 연산 (SSE2 셔플 순서 그대로)
// V3Dot (PxVecMathSSE.h:965): t2 = (xx+zz, yy+ww, …), 결과 = (yy+ww)+(xx+zz)  [덧셈은 교환법칙 성립 -> (xx+zz)+(yy+ww)]
EHD F V3Dot(const L4& a, const L4& b) {
  const float xx = a.x * b.x, yy = a.y * b.y, zz = a.z * b.z, ww = a.w * b.w;
  const float t0 = xx + zz, t1 = yy + ww;
  return t1 + t0;
}
// V4Dot3 (PxVecMathSSE.h:1742): (xx + yy) + zz
EHD F V4Dot3(const L4& a, const L4& b) {
  const float xx = a.x * b.x, yy = a.y * b.y, zz = a.z * b.z;
  return (xx + yy) + zz;
}
// V3SumElems (SSE2): (x + y) + z
EHD F V3SumElems(const L4& a) { return (a.x + a.y) + a.z; }
// V3Cross / V4Cross (PxVecMathSSE.h:983, 1755): v = a*b0 - a1*b 를 섞어 되돌림. w 칸 = aw*bw - aw*bw
EHD L4 V3Cross(const L4& a, const L4& b) {
  // b0 = (b.y, b.z, b.x, b.w), a1 = (a.y, a.z, a.x, a.w); v = a*b0 - a1*b; res = (v.y, v.z, v.x, v.w)
  const float vx = a.x * b.y - a.y * b.x;
  const float vy = a.y * b.z - a.z * b.y;
  const float vz = a.z * b.x - a.x * b.z;
  const float vw = a.w * b.w - a.w * b.w;
  return L4{vy, vz, vx, vw};
}
EHD L4 V4Cross(const L4& a, const L4& b) { return V3Cross(a, b); }
// V3PrepareCross + V3Cross(VecCrossV, Vec3V) (PxVecMathSSE.h:1016): mL1*l2 - mR1*r2
struct CrossV { L4 l1, r1; };
EHD CrossV V3PrepareCross(const L4& a) { return CrossV{L4{a.y, a.z, a.x, a.w}, L4{a.z, a.x, a.y, a.w}}; }
EHD L4 V3Cross(const CrossV& a, const L4& b) {
  const L4 r2{b.y, b.z, b.x, b.w};
  const L4 l2{b.z, b.x, b.y, b.w};
  return V4Sub(V4Mul(a.l1, l2), V4Mul(a.r1, r2));
}

// ---- 스칼라(FloatV)
EHD F FNeg(F a) { return 0.0f - a; }                                        // _mm_sub_ps(0, a)
EHD F FScaleAdd(F a, F b, F c) { return a * b + c; }                        // FAdd(FMul(a,b),c)
EHD F FNegScaleSub(F a, F b, F c) { return c - a * b; }
EHD F FRecip(F a) { return 1.0f / a; }                                      // _mm_div_ps(1, a)
EHD F FMin(F a, F b) { return a < b ? a : b; }                              // minps
EHD F FMax(F a, F b) { return a > b ? a : b; }                              // maxps
EHD F FClamp(F a, F lo, F hi) { return FMax(FMin(a, hi), lo); }             // _mm_max_ps(_mm_min_ps(a, hi), lo)
EHD F FMaxReal() { return 3.40282346638528859812e+38F; }                    // FMax() = PX_MAX_REAL

// ---- 행렬·쿼터니언
struct M3V { L4 c0, c1, c2; };  // Mat33V
EHD M3V M33Load(const M33& m) { return M3V{V3Load(m.c0), V3Load(m.c1), V3Load(m.c2)}; }
// M33MulV3 (PxVecMathSSE.h): (c0*x + c1*y) + c2*z, w 칸도 같은 식
EHD L4 M33MulV3(const M3V& m, const L4& b) {
  const L4 v0 = V3Scale(m.c0, b.x), v1 = V3Scale(m.c1, b.y), v2 = V3Scale(m.c2, b.z);
  return V3Add(V3Add(v0, v1), v2);
}
// QuatRotate (PxVecQuat.h:185): ((v*w2 + cross(u,v)*w) + u*dot(u,v)) * 2, w2 = w*w + (-0.5)
EHD L4 QuatRotate(const L4& q, const L4& v) {
  const L4 u = V4ClearW(q);
  const F w = q.w;
  const F w2 = FScaleAdd(w, w, -0.5f);
  const L4 a = V3Scale(v, w2);
  const L4 temp = V3ScaleAdd(V3Cross(u, v), w, a);
  return V3Scale(V3ScaleAdd(u, V3Dot(u, v), temp), 2.0f);
}

// aos::transformMultiply<.,.> (PxSIMDHelpers.h:77 transformKernelVec4): out = a * b
EHD Tf transformMultiply(const Tf& a, const Tf& b) {
  const L4 aPos = L4{a.p.x, a.p.y, a.p.z, 0.0f};  // PxTransform32 의 p 뒤 패딩 = 0 (PxTransform32 생성자가 0 으로 둠, 추정 -> 시험으로 확인)
  const L4 aRot = L4{a.q.x, a.q.y, a.q.z, a.q.w};
  const L4 bPos = L4{b.p.x, b.p.y, b.p.z, 0.0f};
  const L4 bRot = L4{b.q.x, b.q.y, b.q.z, b.q.w};
  const F wa = aRot.w, wb = bRot.w;
  const L4 va = aRot, vb = bRot;
  const F wo = wa * wb - V4Dot3(va, vb);                                     // FSub(FMul(wa,wb), V4Dot3(va,vb))
  const L4 vo = V4ScaleAdd(va, wb, V4ScaleAdd(vb, wa, V4Cross(va, vb)));
  const L4 t1 = V4Scale(bPos, FScaleAdd(wa, wa, -0.5f));
  const L4 t2 = V4ScaleAdd(V4Cross(va, bPos), wa, t1);
  const L4 t3 = V4ScaleAdd(va, V4Dot3(va, bPos), t2);
  const L4 po = V4ScaleAdd(t3, 2.0f, aPos);
  return Tf{Q{vo.x, vo.y, vo.z, wo}, V3{po.x, po.y, po.z}};
}

}  // namespace js
}  // namespace eng
