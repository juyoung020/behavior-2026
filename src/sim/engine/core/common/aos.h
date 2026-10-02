// 생성물 — 손으로 고치지 말 것. scripts/gen_aos.py 가 PhysX 5.6.1 aos SSE2 판(include/foundation)에서 만든다.
// 원본 저작권: Copyright (c) 2008-2025 NVIDIA Corporation. BSD-3-Clause (원본 머리말과 같은 조건).
// 식은 원본 그대로이고 __m128/_mm_* 만 core/common/sse_emu.h 의 흉내로 바꿨다 (호스트·CUDA 공용).
#pragma once
#include "aos_prelude.h"


// ======== PxUnixSse2AoS.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========


namespace eng
{
namespace aos
{


typedef union UnionM128
{
	EHD UnionM128()
	{
	}
	EHD UnionM128(sse::m128 in)
	{
		m128 = in;
	}

	EHD UnionM128(sse::m128i in)
	{
		m128i = in;
	}

	EHD operator sse::m128()
	{
		return m128;
	}

	EHD operator sse::m128() const
	{
		return m128;
	}

	float m128_f32[4];
	__int8_t m128_i8[16];
	__int16_t m128_i16[8];
	__int32_t m128_i32[4];
	__int64_t m128_i64[2];
	__uint16_t m128_u16[8];
	__uint32_t m128_u32[4];
	__uint64_t m128_u64[2];
	sse::m128 m128;
	sse::m128i m128i;
} UnionM128;

#if defined(ENG_AOS_FLOATV_ONE_LANE)
typedef sse::FV1 FloatV;
#else
typedef sse::m128 FloatV;
#endif
typedef sse::m128 Vec3V;
typedef sse::m128 Vec4V;
typedef sse::m128 BoolV;
typedef sse::m128 QuatV;
typedef sse::m128i VecI32V;
typedef UnionM128 VecU32V;
typedef UnionM128 VecU16V;
typedef UnionM128 VecI16V;
typedef UnionM128 VecU8V;

#define FloatVArg FloatV &
#define Vec3VArg Vec3V &
#define Vec4VArg Vec4V &
#define BoolVArg BoolV &
#define VecU32VArg VecU32V &
#define VecI32VArg VecI32V &
#define VecU16VArg VecU16V &
#define VecI16VArg VecI16V &
#define VecU8VArg VecU8V &
#define QuatVArg QuatV &

// Optimization for situations in which you cross product multiple vectors with the same vector.
// Avoids 2X shuffles per product
struct VecCrossV
{
	Vec3V mL1;
	Vec3V mR1;
};

struct VecShiftV
{
	VecI32V shift;
};
#define VecShiftVArg VecShiftV &

PX_ALIGN_PREFIX(16)
struct Mat33V
{
	EHD Mat33V()
	{
	}
	EHD Mat33V(const Vec3V& c0, const Vec3V& c1, const Vec3V& c2) : col0(c0), col1(c1), col2(c2)
	{
	}
	Vec3V PX_ALIGN(16, col0);
	Vec3V PX_ALIGN(16, col1);
	Vec3V PX_ALIGN(16, col2);
} PX_ALIGN_SUFFIX(16);

PX_ALIGN_PREFIX(16)
struct Mat34V
{
	EHD Mat34V()
	{
	}
	EHD Mat34V(const Vec3V& c0, const Vec3V& c1, const Vec3V& c2, const Vec3V& c3) : col0(c0), col1(c1), col2(c2), col3(c3)
	{
	}
	Vec3V PX_ALIGN(16, col0);
	Vec3V PX_ALIGN(16, col1);
	Vec3V PX_ALIGN(16, col2);
	Vec3V PX_ALIGN(16, col3);
} PX_ALIGN_SUFFIX(16);

PX_ALIGN_PREFIX(16)
struct Mat43V
{
	Mat43V()
	{
	}
	Mat43V(const Vec4V& c0, const Vec4V& c1, const Vec4V& c2) : col0(c0), col1(c1), col2(c2)
	{
	}
	Vec4V PX_ALIGN(16, col0);
	Vec4V PX_ALIGN(16, col1);
	Vec4V PX_ALIGN(16, col2);
} PX_ALIGN_SUFFIX(16);

PX_ALIGN_PREFIX(16)
struct Mat44V
{
	EHD Mat44V()
	{
	}
	EHD Mat44V(const Vec4V& c0, const Vec4V& c1, const Vec4V& c2, const Vec4V& c3) : col0(c0), col1(c1), col2(c2), col3(c3)
	{
	}
	Vec4V PX_ALIGN(16, col0);
	Vec4V PX_ALIGN(16, col1);
	Vec4V PX_ALIGN(16, col2);
	Vec4V PX_ALIGN(16, col3);
} PX_ALIGN_SUFFIX(16);

} // namespace aos
} // namespace eng



// ======== PxVecMath.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========
	#define COMPILE_VECTOR_INTRINSICS 0

// only SSE2 compatible platforms should reach this


namespace eng
{
namespace aos
{

// Basic AoS types are
// FloatV	- 16-byte aligned representation of float.
// Vec3V		- 16-byte aligned representation of PxVec3 stored as (x y z 0).
// Vec4V		- 16-byte aligned representation of vector of 4 floats stored as (x y z w).
// BoolV		- 16-byte aligned representation of vector of 4 bools stored as (x y z w).
// VecU32V	- 16-byte aligned representation of 4 unsigned ints stored as (x y z w).
// VecI32V	- 16-byte aligned representation of 4 signed ints stored as (x y z w).
// Mat33V	- 16-byte aligned representation of any 3x3 matrix.
// Mat34V	- 16-byte aligned representation of transformation matrix (rotation in col1,col2,col3 and translation in
// col4).
// Mat44V	- 16-byte aligned representation of any 4x4 matrix.

//////////////////////////////////////////
// Construct a simd type from a scalar type
//////////////////////////////////////////

// FloatV
//(f,f,f,f)
EHD FloatV FLoad(const PxF32 f);

// Vec3V
//(f,f,f,0)
EHD Vec3V V3Load(const PxF32 f);
//(f.x,f.y,f.z,0)
EHD Vec3V V3LoadU(const PxVec3& f);
//(f.x,f.y,f.z,0), f must be 16-byte aligned
EHD Vec3V V3LoadA(const PxVec3& f);
//(f.x,f.y,f.z,w_undefined), f must be 16-byte aligned
EHD Vec3V V3LoadUnsafeA(const PxVec3& f);
//(f.x,f.y,f.z,0)
EHD Vec3V V3LoadU(const PxF32* f);
//(f.x,f.y,f.z,0), f must be 16-byte aligned
EHD Vec3V V3LoadA(const PxF32* f);

// Vec4V
//(f,f,f,f)
EHD Vec4V V4Load(const PxF32 f);
//(f[0],f[1],f[2],f[3])
EHD Vec4V V4LoadU(const PxF32* const f);
//(f[0],f[1],f[2],f[3]), f must be 16-byte aligned
EHD Vec4V V4LoadA(const PxF32* const f);
//(x,y,z,w)
EHD Vec4V V4LoadXYZW(const PxF32& x, const PxF32& y, const PxF32& z, const PxF32& w);

// BoolV
//(f,f,f,f)
EHD BoolV BLoad(const bool f);
//(f[0],f[1],f[2],f[3])
EHD BoolV BLoad(const bool* const f);

// VecU32V
//(f,f,f,f)
EHD VecU32V U4Load(const PxU32 f);
//(f[0],f[1],f[2],f[3])
EHD VecU32V U4LoadU(const PxU32* f);
//(f[0],f[1],f[2],f[3]), f must be 16-byte aligned
EHD VecU32V U4LoadA(const PxU32* f);
//((U32)x, (U32)y, (U32)z, (U32)w)
EHD VecU32V U4LoadXYZW(PxU32 x, PxU32 y, PxU32 z, PxU32 w);

// VecI32V
//(i,i,i,i)
EHD VecI32V I4Load(const PxI32 i);
//(i,i,i,i)
EHD VecI32V I4LoadU(const PxI32* i);
//(i,i,i,i)
EHD VecI32V I4LoadA(const PxI32* i);

// QuatV
//(x = v[0], y=v[1], z=v[2], w=v3[3]) and array don't need to aligned
EHD QuatV QuatVLoadU(const PxF32* v);
//(x = v[0], y=v[1], z=v[2], w=v3[3]) and array need to aligned, fast load
EHD QuatV QuatVLoadA(const PxF32* v);
//(x, y, z, w)
EHD QuatV QuatVLoadXYZW(const PxF32 x, const PxF32 y, const PxF32 z, const PxF32 w);

// not added to public api
EHD Vec4V Vec4V_From_PxVec3_WUndefined(const PxVec3& v);

///////////////////////////////////////////////////
// Construct a simd type from a different simd type
///////////////////////////////////////////////////

// Vec3V
//(v.x,v.y,v.z,0)
EHD Vec3V Vec3V_From_Vec4V(Vec4V v);
//(v.x,v.y,v.z,undefined) - be very careful with w!=0 because many functions require w==0 for correct operation eg V3Dot, V3Length, V3Cross etc etc.
EHD Vec3V Vec3V_From_Vec4V_WUndefined(const Vec4V v);

// Vec4V
//(f.x,f.y,f.z,f.w)
EHD Vec4V Vec4V_From_Vec3V(Vec3V f);
//((PxF32)f.x, (PxF32)f.y, (PxF32)f.z, (PxF32)f.w)
EHD Vec4V Vec4V_From_VecU32V(VecU32V a);
//((PxF32)f.x, (PxF32)f.y, (PxF32)f.z, (PxF32)f.w)
EHD Vec4V Vec4V_From_VecI32V(VecI32V a);
//(*(reinterpret_cast<PxF32*>(&f.x), (reinterpret_cast<PxF32*>(&f.y), (reinterpret_cast<PxF32*>(&f.z),
//(reinterpret_cast<PxF32*>(&f.w))
EHD Vec4V Vec4V_ReinterpretFrom_VecU32V(VecU32V a);
//(*(reinterpret_cast<PxF32*>(&f.x), (reinterpret_cast<PxF32*>(&f.y), (reinterpret_cast<PxF32*>(&f.z),
//(reinterpret_cast<PxF32*>(&f.w))
EHD Vec4V Vec4V_ReinterpretFrom_VecI32V(VecI32V a);

// VecU32V
//(*(reinterpret_cast<PxU32*>(&f.x), (reinterpret_cast<PxU32*>(&f.y), (reinterpret_cast<PxU32*>(&f.z),
//(reinterpret_cast<PxU32*>(&f.w))
EHD VecU32V VecU32V_ReinterpretFrom_Vec4V(Vec4V a);
//(b[0], b[1], b[2], b[3])
EHD VecU32V VecU32V_From_BoolV(const BoolVArg b);

// VecI32V
//(*(reinterpret_cast<PxI32*>(&f.x), (reinterpret_cast<PxI32*>(&f.y), (reinterpret_cast<PxI32*>(&f.z),
//(reinterpret_cast<PxI32*>(&f.w))
EHD VecI32V VecI32V_ReinterpretFrom_Vec4V(Vec4V a);
//((I32)a.x, (I32)a.y, (I32)a.z, (I32)a.w)
EHD VecI32V VecI32V_From_Vec4V(Vec4V a);
//((I32)b.x, (I32)b.y, (I32)b.z, (I32)b.w)
EHD VecI32V VecI32V_From_BoolV(const BoolVArg b);

///////////////////////////////////////////////////
// Convert from a simd type back to a scalar type
///////////////////////////////////////////////////

// FloatV
// a.x
EHD void FStore(const FloatV a, PxF32* PX_RESTRICT f);

// Vec3V
//(a.x,a.y,a.z)
EHD void V3StoreA(const Vec3V a, PxVec3& f);
//(a.x,a.y,a.z)
EHD void V3StoreU(const Vec3V a, PxVec3& f);

// Vec4V
EHD void V4StoreA(const Vec4V a, PxF32* f);
EHD void V4StoreU(const Vec4V a, PxF32* f);

// BoolV
EHD void BStoreA(const BoolV b, PxU32* f);

// VecU32V
EHD void U4StoreA(const VecU32V uv, PxU32* u);

// VecI32V
EHD void I4StoreA(const VecI32V iv, PxI32* i);

//////////////////////////////////////////////////////////////////
// Test that simd types have elements in the floating point range
//////////////////////////////////////////////////////////////////

// check for each component is valid ie in floating point range
EHD bool isFiniteFloatV(const FloatV a);
// check for each component is valid ie in floating point range
EHD bool isFiniteVec3V(const Vec3V a);
// check for each component is valid ie in floating point range
EHD bool isFiniteVec4V(const Vec4V a);

// Check that w-component is zero.
EHD bool isValidVec3V(const Vec3V a);

//////////////////////////////////////////////////////////////////
// Tests that all elements of two 16-byte types are completely equivalent.
// Use these tests for unit testing and asserts only.
//////////////////////////////////////////////////////////////////

namespace vecMathTests
{
EHD Vec3V getInvalidVec3V();
EHD bool allElementsEqualFloatV(const FloatV a, const FloatV b);
EHD bool allElementsEqualVec3V(const Vec3V a, const Vec3V b);
EHD bool allElementsEqualVec4V(const Vec4V a, const Vec4V b);
EHD bool allElementsEqualBoolV(const BoolV a, const BoolV b);
EHD bool allElementsEqualVecU32V(const VecU32V a, const VecU32V b);
EHD bool allElementsEqualVecI32V(const VecI32V a, const VecI32V b);

EHD bool allElementsEqualMat33V(const Mat33V& a, const Mat33V& b)
{
	return (allElementsEqualVec3V(a.col0, b.col0) && allElementsEqualVec3V(a.col1, b.col1) &&
	        allElementsEqualVec3V(a.col2, b.col2));
}
EHD bool allElementsEqualMat34V(const Mat34V& a, const Mat34V& b)
{
	return (allElementsEqualVec3V(a.col0, b.col0) && allElementsEqualVec3V(a.col1, b.col1) &&
	        allElementsEqualVec3V(a.col2, b.col2) && allElementsEqualVec3V(a.col3, b.col3));
}
EHD bool allElementsEqualMat44V(const Mat44V& a, const Mat44V& b)
{
	return (allElementsEqualVec4V(a.col0, b.col0) && allElementsEqualVec4V(a.col1, b.col1) &&
	        allElementsEqualVec4V(a.col2, b.col2) && allElementsEqualVec4V(a.col3, b.col3));
}

EHD bool allElementsNearEqualFloatV(const FloatV a, const FloatV b);
EHD bool allElementsNearEqualVec3V(const Vec3V a, const Vec3V b);
EHD bool allElementsNearEqualVec4V(const Vec4V a, const Vec4V b);
EHD bool allElementsNearEqualMat33V(const Mat33V& a, const Mat33V& b)
{
	return (allElementsNearEqualVec3V(a.col0, b.col0) && allElementsNearEqualVec3V(a.col1, b.col1) &&
	        allElementsNearEqualVec3V(a.col2, b.col2));
}
EHD bool allElementsNearEqualMat34V(const Mat34V& a, const Mat34V& b)
{
	return (allElementsNearEqualVec3V(a.col0, b.col0) && allElementsNearEqualVec3V(a.col1, b.col1) &&
	        allElementsNearEqualVec3V(a.col2, b.col2) && allElementsNearEqualVec3V(a.col3, b.col3));
}
EHD bool allElementsNearEqualMat44V(const Mat44V& a, const Mat44V& b)
{
	return (allElementsNearEqualVec4V(a.col0, b.col0) && allElementsNearEqualVec4V(a.col1, b.col1) &&
	        allElementsNearEqualVec4V(a.col2, b.col2) && allElementsNearEqualVec4V(a.col3, b.col3));
}
}

//////////////////////////////////////////////////////////////////
// Math operations on FloatV
//////////////////////////////////////////////////////////////////

//(0,0,0,0)
EHD FloatV FZero();
//(1,1,1,1)
EHD FloatV FOne();
//(0.5,0.5,0.5,0.5)
EHD FloatV FHalf();
//(PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL)
EHD FloatV FEps();
//! \cond
//(PX_MAX_REAL, PX_MAX_REAL, PX_MAX_REAL PX_MAX_REAL)
EHD FloatV FMax();
//! \endcond
//(-PX_MAX_REAL, -PX_MAX_REAL, -PX_MAX_REAL -PX_MAX_REAL)
EHD FloatV FNegMax();
//(1e-6f, 1e-6f, 1e-6f, 1e-6f)
EHD FloatV FEps6();
//((PxF32*)&1, (PxF32*)&1, (PxF32*)&1, (PxF32*)&1)

//-f (per component)
EHD FloatV FNeg(const FloatV f);
// a+b (per component)
EHD FloatV FAdd(const FloatV a, const FloatV b);
// a-b (per component)
EHD FloatV FSub(const FloatV a, const FloatV b);
// a*b (per component)
EHD FloatV FMul(const FloatV a, const FloatV b);
// a/b (per component)
EHD FloatV FDiv(const FloatV a, const FloatV b);
// a/b (per component)
EHD FloatV FDivFast(const FloatV a, const FloatV b);
// 1.0f/a
EHD FloatV FRecip(const FloatV a);
// 1.0f/a
EHD FloatV FRecipFast(const FloatV a);
// 1.0f/sqrt(a)
EHD FloatV FRsqrt(const FloatV a);
// 1.0f/sqrt(a)
EHD FloatV FRsqrtFast(const FloatV a);
// sqrt(a)
EHD FloatV FSqrt(const FloatV a);
// a*b+c
EHD FloatV FScaleAdd(const FloatV a, const FloatV b, const FloatV c);
// c-a*b
EHD FloatV FNegScaleSub(const FloatV a, const FloatV b, const FloatV c);
// fabs(a)
EHD FloatV FAbs(const FloatV a);
// c ? a : b (per component)
EHD FloatV FSel(const BoolV c, const FloatV a, const FloatV b);
// a>b (per component)
EHD BoolV FIsGrtr(const FloatV a, const FloatV b);
// a>=b (per component)
EHD BoolV FIsGrtrOrEq(const FloatV a, const FloatV b);
// a==b (per component)
EHD BoolV FIsEq(const FloatV a, const FloatV b);
// Max(a,b) (per component)
EHD FloatV FMax(const FloatV a, const FloatV b);
// Min(a,b) (per component)
EHD FloatV FMin(const FloatV a, const FloatV b);
// Clamp(a,b) (per component)
EHD FloatV FClamp(const FloatV a, const FloatV minV, const FloatV maxV);

// a.x>b.x
EHD PxU32 FAllGrtr(const FloatV a, const FloatV b);
// a.x>=b.x
EHD PxU32 FAllGrtrOrEq(const FloatV a, const FloatV b);
// a.x==b.x
EHD PxU32 FAllEq(const FloatV a, const FloatV b);
// a<min || a>max
EHD PxU32 FOutOfBounds(const FloatV a, const FloatV min, const FloatV max);
// a>=min && a<=max
EHD PxU32 FInBounds(const FloatV a, const FloatV min, const FloatV max);
// a<-bounds || a>bounds
EHD PxU32 FOutOfBounds(const FloatV a, const FloatV bounds);
// a>=-bounds && a<=bounds
EHD PxU32 FInBounds(const FloatV a, const FloatV bounds);

// round float a to the near int
EHD FloatV FRound(const FloatV a);
// calculate the sin of float a
EHD FloatV FSin(const FloatV a);
// calculate the cos of float b
EHD FloatV FCos(const FloatV a);

//////////////////////////////////////////////////////////////////
// Math operations on Vec3V
//////////////////////////////////////////////////////////////////

//(f,f,f,f)
EHD Vec3V V3Splat(const FloatV f);

//(x,y,z)
EHD Vec3V V3Merge(const FloatVArg x, const FloatVArg y, const FloatVArg z);

//(1,0,0,0)
EHD Vec3V V3UnitX();
//(0,1,0,0)
EHD Vec3V V3UnitY();
//(0,0,1,0)
EHD Vec3V V3UnitZ();

//(f.x,f.x,f.x,f.x)
EHD FloatV V3GetX(const Vec3V f);
//(f.y,f.y,f.y,f.y)
EHD FloatV V3GetY(const Vec3V f);
//(f.z,f.z,f.z,f.z)
EHD FloatV V3GetZ(const Vec3V f);

//(f,v.y,v.z,v.w)
EHD Vec3V V3SetX(const Vec3V v, const FloatV f);
//(v.x,f,v.z,v.w)
EHD Vec3V V3SetY(const Vec3V v, const FloatV f);
//(v.x,v.y,f,v.w)
EHD Vec3V V3SetZ(const Vec3V v, const FloatV f);

// v.x=f
EHD void V3WriteX(Vec3V& v, const PxF32 f);
// v.y=f
EHD void V3WriteY(Vec3V& v, const PxF32 f);
// v.z=f
EHD void V3WriteZ(Vec3V& v, const PxF32 f);
// v.x=f.x, v.y=f.y, v.z=f.z
EHD void V3WriteXYZ(Vec3V& v, const PxVec3& f);
// return v.x
EHD PxF32 V3ReadX(const Vec3V& v);
// return v.y
EHD PxF32 V3ReadY(const Vec3V& v);
// return v.y
EHD PxF32 V3ReadZ(const Vec3V& v);
// return (v.x,v.y,v.z)
EHD const PxVec3& V3ReadXYZ(const Vec3V& v);

//(a.x, b.x, c.x)
EHD Vec3V V3ColX(const Vec3V a, const Vec3V b, const Vec3V c);
//(a.y, b.y, c.y)
EHD Vec3V V3ColY(const Vec3V a, const Vec3V b, const Vec3V c);
//(a.z, b.z, c.z)
EHD Vec3V V3ColZ(const Vec3V a, const Vec3V b, const Vec3V c);

//(0,0,0,0)
EHD Vec3V V3Zero();
//(1,1,1,1)
EHD Vec3V V3One();
//(PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL)
EHD Vec3V V3Eps();
//-c (per component)
EHD Vec3V V3Neg(const Vec3V c);
// a+b (per component)
EHD Vec3V V3Add(const Vec3V a, const Vec3V b);
// a-b (per component)
EHD Vec3V V3Sub(const Vec3V a, const Vec3V b);
// a*b (per component)
EHD Vec3V V3Scale(const Vec3V a, const FloatV b);
// a*b (per component)
EHD Vec3V V3Mul(const Vec3V a, const Vec3V b);
// a/b (per component)
EHD Vec3V V3ScaleInv(const Vec3V a, const FloatV b);
// a/b (per component)
EHD Vec3V V3Div(const Vec3V a, const Vec3V b);
// a/b (per component)
EHD Vec3V V3ScaleInvFast(const Vec3V a, const FloatV b);
// a/b (per component)
EHD Vec3V V3DivFast(const Vec3V a, const Vec3V b);
// 1.0f/a
EHD Vec3V V3Recip(const Vec3V a);
// 1.0f/a
EHD Vec3V V3RecipFast(const Vec3V a);
// 1.0f/sqrt(a)
EHD Vec3V V3Rsqrt(const Vec3V a);
// 1.0f/sqrt(a)
EHD Vec3V V3RsqrtFast(const Vec3V a);
// a*b+c
EHD Vec3V V3ScaleAdd(const Vec3V a, const FloatV b, const Vec3V c);
// c-a*b
EHD Vec3V V3NegScaleSub(const Vec3V a, const FloatV b, const Vec3V c);
// a*b+c
EHD Vec3V V3MulAdd(const Vec3V a, const Vec3V b, const Vec3V c);
// c-a*b
EHD Vec3V V3NegMulSub(const Vec3V a, const Vec3V b, const Vec3V c);
// fabs(a)
EHD Vec3V V3Abs(const Vec3V a);

// a.b 
// Note: a.w and b.w must have value zero
EHD FloatV V3Dot(const Vec3V a, const Vec3V b);
// aXb
// Note: a.w and b.w must have value zero
EHD Vec3V V3Cross(const Vec3V a, const Vec3V b);
// |a.a|^1/2
// Note: a.w must have value zero
EHD FloatV V3Length(const Vec3V a);
// a.a
// Note: a.w must have value zero
EHD FloatV V3LengthSq(const Vec3V a);
// a*|a.a|^-1/2
// Note: a.w must have value zero
EHD Vec3V V3Normalize(const Vec3V a);
// a.a>0 ? a*|a.a|^-1/2 : (0,0,0,0)
// Note: a.w must have value zero
EHD FloatV V3Length(const Vec3V a);
// a.a>0 ? a*|a.a|^-1/2 : unsafeReturnValue 
// Note: a.w must have value zero
EHD Vec3V V3NormalizeSafe(const Vec3V a, const Vec3V unsafeReturnValue);
// a.x + a.y + a.z
// Note: a.w must have value zero
EHD FloatV V3SumElems(const Vec3V a);

// c ? a : b (per component)
EHD Vec3V V3Sel(const BoolV c, const Vec3V a, const Vec3V b);
// a>b (per component)
EHD BoolV V3IsGrtr(const Vec3V a, const Vec3V b);
// a>=b (per component)
EHD BoolV V3IsGrtrOrEq(const Vec3V a, const Vec3V b);
// a==b (per component)
EHD BoolV V3IsEq(const Vec3V a, const Vec3V b);
// Max(a,b) (per component)
EHD Vec3V V3Max(const Vec3V a, const Vec3V b);
// Min(a,b) (per component)
EHD Vec3V V3Min(const Vec3V a, const Vec3V b);

// Extract the maximum value from a
// Note: a.w must have value zero
EHD FloatV V3ExtractMax(const Vec3V a);

// Extract the minimum value from a
// Note: a.w must have value zero
EHD FloatV V3ExtractMin(const Vec3V a);

// Clamp(a,b) (per component)
EHD Vec3V V3Clamp(const Vec3V a, const Vec3V minV, const Vec3V maxV);

// Extract the sign for each component
EHD Vec3V V3Sign(const Vec3V a);

// Test all components.
// (a.x>b.x && a.y>b.y && a.z>b.z)
// Note: a.w and b.w must have value zero
EHD PxU32 V3AllGrtr(const Vec3V a, const Vec3V b);
// (a.x>=b.x && a.y>=b.y && a.z>=b.z)
// Note: a.w and b.w must have value zero
EHD PxU32 V3AllGrtrOrEq(const Vec3V a, const Vec3V b);
// (a.x==b.x && a.y==b.y && a.z==b.z)
// Note: a.w and b.w must have value zero
EHD PxU32 V3AllEq(const Vec3V a, const Vec3V b);
// a.x<min.x || a.y<min.y || a.z<min.z || a.x>max.x || a.y>max.y || a.z>max.z
// Note: a.w and min.w and max.w must have value zero
EHD PxU32 V3OutOfBounds(const Vec3V a, const Vec3V min, const Vec3V max);
// a.x>=min.x && a.y>=min.y && a.z>=min.z && a.x<=max.x && a.y<=max.y && a.z<=max.z
// Note: a.w and min.w and max.w must have value zero
EHD PxU32 V3InBounds(const Vec3V a, const Vec3V min, const Vec3V max);
// a.x<-bounds.x || a.y<=-bounds.y || a.z<bounds.z || a.x>bounds.x || a.y>bounds.y || a.z>bounds.z
// Note: a.w and bounds.w must have value zero
EHD PxU32 V3OutOfBounds(const Vec3V a, const Vec3V bounds);
// a.x>=-bounds.x && a.y>=-bounds.y && a.z>=-bounds.z && a.x<=bounds.x && a.y<=bounds.y && a.z<=bounds.z
// Note: a.w and bounds.w must have value zero
EHD PxU32 V3InBounds(const Vec3V a, const Vec3V bounds);

//(floor(a.x + 0.5f), floor(a.y + 0.5f), floor(a.z + 0.5f))
EHD Vec3V V3Round(const Vec3V a);

//(sinf(a.x), sinf(a.y), sinf(a.z))
EHD Vec3V V3Sin(const Vec3V a);
//(cosf(a.x), cosf(a.y), cosf(a.z))
EHD Vec3V V3Cos(const Vec3V a);

//(a.y,a.z,a.z)
EHD Vec3V V3PermYZZ(const Vec3V a);
//(a.x,a.y,a.x)
EHD Vec3V V3PermXYX(const Vec3V a);
//(a.y,a.z,a.x)
EHD Vec3V V3PermYZX(const Vec3V a);
//(a.z, a.x, a.y)
EHD Vec3V V3PermZXY(const Vec3V a);
//(a.z,a.z,a.y)
EHD Vec3V V3PermZZY(const Vec3V a);
//(a.y,a.x,a.x)
EHD Vec3V V3PermYXX(const Vec3V a);
//(0, v1.z, v0.y)
EHD Vec3V V3Perm_Zero_1Z_0Y(const Vec3V v0, const Vec3V v1);
//(v0.z, 0, v1.x)
EHD Vec3V V3Perm_0Z_Zero_1X(const Vec3V v0, const Vec3V v1);
//(v1.y, v0.x, 0)
EHD Vec3V V3Perm_1Y_0X_Zero(const Vec3V v0, const Vec3V v1);

// Transpose 3 Vec3Vs inplace. Sets the w component to zero
// [ x0, y0, z0, w0] [ x1, y1, z1, w1]  [ x2, y2, z2, w2]  -> [x0 x1 x2 0] [y0 y1 y2 0] [z0 z1 z2 0]
EHD void V3Transpose(Vec3V& col0, Vec3V& col1, Vec3V& col2);

//////////////////////////////////////////////////////////////////
// Math operations on Vec4V
//////////////////////////////////////////////////////////////////

//(f,f,f,f)
EHD Vec4V V4Splat(const FloatV f);

//(f[0],f[1],f[2],f[3])
EHD Vec4V V4Merge(const FloatV* const f);
//(x,y,z,w)
EHD Vec4V V4Merge(const FloatVArg x, const FloatVArg y, const FloatVArg z, const FloatVArg w);
//(x.w, y.w, z.w, w.w)
EHD Vec4V V4MergeW(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w);
//(x.z, y.z, z.z, w.z)
EHD Vec4V V4MergeZ(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w);
//(x.y, y.y, z.y, w.y)
EHD Vec4V V4MergeY(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w);
//(x.x, y.x, z.x, w.x)
EHD Vec4V V4MergeX(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w);

//(a.x, b.x, a.y, b.y)
EHD Vec4V V4UnpackXY(const Vec4VArg a, const Vec4VArg b);
//(a.z, b.z, a.w, b.w)
EHD Vec4V V4UnpackZW(const Vec4VArg a, const Vec4VArg b);

//(1,0,0,0)
EHD Vec4V V4UnitW();
//(0,1,0,0)
EHD Vec4V V4UnitY();
//(0,0,1,0)
EHD Vec4V V4UnitZ();
//(0,0,0,1)
EHD Vec4V V4UnitW();

//(f.x,f.x,f.x,f.x)
EHD FloatV V4GetX(const Vec4V f);
//(f.y,f.y,f.y,f.y)
EHD FloatV V4GetY(const Vec4V f);
//(f.z,f.z,f.z,f.z)
EHD FloatV V4GetZ(const Vec4V f);
//(f.w,f.w,f.w,f.w)
EHD FloatV V4GetW(const Vec4V f);

//(f,v.y,v.z,v.w)
EHD Vec4V V4SetX(const Vec4V v, const FloatV f);
//(v.x,f,v.z,v.w)
EHD Vec4V V4SetY(const Vec4V v, const FloatV f);
//(v.x,v.y,f,v.w)
EHD Vec4V V4SetZ(const Vec4V v, const FloatV f);
//(v.x,v.y,v.z,f)
EHD Vec4V V4SetW(const Vec4V v, const FloatV f);

//(v.x,v.y,v.z,0)
EHD Vec4V V4ClearW(const Vec4V v);

//(a[elementIndex], a[elementIndex], a[elementIndex], a[elementIndex])
template <int elementIndex>
EHD Vec4V V4SplatElement(Vec4V a);

// v.x=f
EHD void V4WriteX(Vec4V& v, const PxF32 f);
// v.y=f
EHD void V4WriteY(Vec4V& v, const PxF32 f);
// v.z=f
EHD void V4WriteZ(Vec4V& v, const PxF32 f);
// v.w=f
EHD void V4WriteW(Vec4V& v, const PxF32 f);
// v.x=f.x, v.y=f.y, v.z=f.z
EHD void V4WriteXYZ(Vec4V& v, const PxVec3& f);
// return v.x
EHD PxF32 V4ReadX(const Vec4V& v);
// return v.y
EHD PxF32 V4ReadY(const Vec4V& v);
// return v.z
EHD PxF32 V4ReadZ(const Vec4V& v);
// return v.w
EHD PxF32 V4ReadW(const Vec4V& v);
// return (v.x,v.y,v.z)
EHD const PxVec3& V4ReadXYZ(const Vec4V& v);

//(0,0,0,0)
EHD Vec4V V4Zero();
//(1,1,1,1)
EHD Vec4V V4One();
//(PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL,PX_EPS_REAL)
EHD Vec4V V4Eps();

//-c (per component)
EHD Vec4V V4Neg(const Vec4V c);
// a+b (per component)
EHD Vec4V V4Add(const Vec4V a, const Vec4V b);
// a-b (per component)
EHD Vec4V V4Sub(const Vec4V a, const Vec4V b);
// a*b (per component)
EHD Vec4V V4Scale(const Vec4V a, const FloatV b);
// a*b (per component)
EHD Vec4V V4Mul(const Vec4V a, const Vec4V b);
// a/b (per component)
EHD Vec4V V4ScaleInv(const Vec4V a, const FloatV b);
// a/b (per component)
EHD Vec4V V4Div(const Vec4V a, const Vec4V b);
// a/b (per component)
EHD Vec4V V4ScaleInvFast(const Vec4V a, const FloatV b);
// a/b (per component)
EHD Vec4V V4DivFast(const Vec4V a, const Vec4V b);
// 1.0f/a
EHD Vec4V V4Recip(const Vec4V a);
// 1.0f/a
EHD Vec4V V4RecipFast(const Vec4V a);
// 1.0f/sqrt(a)
EHD Vec4V V4Rsqrt(const Vec4V a);
// 1.0f/sqrt(a)
EHD Vec4V V4RsqrtFast(const Vec4V a);
// a*b+c
EHD Vec4V V4ScaleAdd(const Vec4V a, const FloatV b, const Vec4V c);
// c-a*b
EHD Vec4V V4NegScaleSub(const Vec4V a, const FloatV b, const Vec4V c);
// a*b+c
EHD Vec4V V4MulAdd(const Vec4V a, const Vec4V b, const Vec4V c);
// c-a*b
EHD Vec4V V4NegMulSub(const Vec4V a, const Vec4V b, const Vec4V c);

// fabs(a)
EHD Vec4V V4Abs(const Vec4V a);
// bitwise a & ~b
EHD Vec4V V4Andc(const Vec4V a, const VecU32V b);

// a.b (W is taken into account)
EHD FloatV V4Dot(const Vec4V a, const Vec4V b);
// a.b (same computation as V3Dot. W is ignored in input)
EHD FloatV V4Dot3(const Vec4V a, const Vec4V b);
// aXb (same computation as V3Cross. W is ignored in input and undefined in output)
EHD Vec4V V4Cross(const Vec4V a, const Vec4V b);

//|a.a|^1/2
EHD FloatV V4Length(const Vec4V a);
// a.a
EHD FloatV V4LengthSq(const Vec4V a);

// a*|a.a|^-1/2
EHD Vec4V V4Normalize(const Vec4V a);
// a.a>0 ? a*|a.a|^-1/2 : unsafeReturnValue 
EHD Vec4V V4NormalizeSafe(const Vec4V a, const Vec4V unsafeReturnValue);
// a*|a.a|^-1/2
EHD Vec4V V4NormalizeFast(const Vec4V a);

// c ? a : b (per component)
EHD Vec4V V4Sel(const BoolV c, const Vec4V a, const Vec4V b);
// a>b (per component)
EHD BoolV V4IsGrtr(const Vec4V a, const Vec4V b);
// a>=b (per component)
EHD BoolV V4IsGrtrOrEq(const Vec4V a, const Vec4V b);
// a==b (per component)
EHD BoolV V4IsEq(const Vec4V a, const Vec4V b);
// Max(a,b) (per component)
EHD Vec4V V4Max(const Vec4V a, const Vec4V b);
// Min(a,b) (per component)
EHD Vec4V V4Min(const Vec4V a, const Vec4V b);
// Get the maximum component from a
EHD FloatV V4ExtractMax(const Vec4V a);
// Get the minimum component from a
EHD FloatV V4ExtractMin(const Vec4V a);

// Clamp(a,b) (per component)
EHD Vec4V V4Clamp(const Vec4V a, const Vec4V minV, const Vec4V maxV);

// return 1 if all components of a are greater than all components of b.
EHD PxU32 V4AllGrtr(const Vec4V a, const Vec4V b);
// return 1 if all components of a are greater than or equal to all components of b
EHD PxU32 V4AllGrtrOrEq(const Vec4V a, const Vec4V b);
// return 1 if XYZ components of a are greater than or equal to XYZ components of b. W is ignored.
EHD PxU32 V4AllGrtrOrEq3(const Vec4V a, const Vec4V b);
// return 1 if all components of a are equal to all components of b
EHD PxU32 V4AllEq(const Vec4V a, const Vec4V b);
// return 1 if any XYZ component of a is greater than the corresponding component of b. W is ignored.
EHD PxU32 V4AnyGrtr3(const Vec4V a, const Vec4V b);

// round(a)(per component)
EHD Vec4V V4Round(const Vec4V a);
// sin(a) (per component)
EHD Vec4V V4Sin(const Vec4V a);
// cos(a) (per component)
EHD Vec4V V4Cos(const Vec4V a);

// Permute v into a new vec4v with YXWZ format
EHD Vec4V V4PermYXWZ(const Vec4V v);
// Permute v into a new vec4v with XZXZ format
EHD Vec4V V4PermXZXZ(const Vec4V v);
// Permute v into a new vec4v with YWYW format
EHD Vec4V V4PermYWYW(const Vec4V v);
// Permute v into a new vec4v with YZXW format
EHD Vec4V V4PermYZXW(const Vec4V v);
// Permute v into a new vec4v with ZWXY format - equivalent to a swap of the two 64bit parts of the vector
EHD Vec4V V4PermZWXY(const Vec4V a);

// Permute v into a new vec4v with format {a[x], a[y], a[z], a[w]}
// V4Perm<1,3,1,3> is equal to V4PermYWYW
// V4Perm<0,2,0,2> is equal to V4PermXZXZ
// V3Perm<1,0,3,2> is equal to V4PermYXWZ
template <PxU8 x, PxU8 y, PxU8 z, PxU8 w>
EHD Vec4V V4Perm(const Vec4V a);

// Transpose 4 Vec4Vs inplace.
// [ x0, y0, z0, w0] [ x1, y1, z1, w1] [ x2, y2, z2, w2] [ x3, y3, z3, w3] ->
// [ x0, x1, x2, x3] [ y0, y1, y2, y3] [ z0, z1, z2, z3] [ w0, w1, w2, w3]
EHD void V3Transpose(Vec3V& col0, Vec3V& col1, Vec3V& col2);

// q = cos(a/2) + u*sin(a/2)
EHD QuatV QuatV_From_RotationAxisAngle(const Vec3V u, const FloatV a);
// convert q to a unit quaternion
EHD QuatV QuatNormalize(const QuatV q);
//|q.q|^1/2
EHD FloatV QuatLength(const QuatV q);
// q.q
EHD FloatV QuatLengthSq(const QuatV q);
// a.b
EHD FloatV QuatDot(const QuatV a, const QuatV b);
//(-q.x, -q.y, -q.z, q.w)
EHD QuatV QuatConjugate(const QuatV q);
//(q.x, q.y, q.z)
EHD Vec3V QuatGetImaginaryPart(const QuatV q);
// convert quaternion to matrix 33
EHD Mat33V QuatGetMat33V(const QuatVArg q);
// convert quaternion to matrix 33
EHD void QuatGetMat33V(const QuatVArg q, Vec3V& column0, Vec3V& column1, Vec3V& column2);
// convert matrix 33 to quaternion
EHD QuatV Mat33GetQuatV(const Mat33V& a);
// brief computes rotation of x-axis
EHD Vec3V QuatGetBasisVector0(const QuatV q);
// brief computes rotation of y-axis
EHD Vec3V QuatGetBasisVector1(const QuatV q);
// brief computes rotation of z-axis
EHD Vec3V QuatGetBasisVector2(const QuatV q);
// calculate the rotation vector from q and v
EHD Vec3V QuatRotate(const QuatV q, const Vec3V v);
// calculate the rotation vector from the conjugate quaternion and v
EHD Vec3V QuatRotateInv(const QuatV q, const Vec3V v);
// quaternion multiplication
EHD QuatV QuatMul(const QuatV a, const QuatV b);
// quaternion add
EHD QuatV QuatAdd(const QuatV a, const QuatV b);
// (-q.x, -q.y, -q.z, -q.w)
EHD QuatV QuatNeg(const QuatV q);
// (a.x - b.x, a.y-b.y, a.z-b.z, a.w-b.w )
EHD QuatV QuatSub(const QuatV a, const QuatV b);
// (a.x*b, a.y*b, a.z*b, a.w*b)
EHD QuatV QuatScale(const QuatV a, const FloatV b);
// (x = v[0], y = v[1], z = v[2], w =v[3])
EHD QuatV QuatMerge(const FloatV* const v);
// (x = v[0], y = v[1], z = v[2], w =v[3])
EHD QuatV QuatMerge(const FloatVArg x, const FloatVArg y, const FloatVArg z, const FloatVArg w);
// (x = 0.f, y = 0.f, z = 0.f, w = 1.f)
EHD QuatV QuatIdentity();
// check for each component is valid
EHD bool isFiniteQuatV(const QuatV q);
// check for each component is valid
EHD bool isValidQuatV(const QuatV q);
// check for each component is valid
EHD bool isSaneQuatV(const QuatV q);

// Math operations on 16-byte aligned booleans.
// x=false	y=false		z=false		w=false
EHD BoolV BFFFF();
// x=false	y=false		z=false		w=true
EHD BoolV BFFFT();
// x=false	y=false		z=true		w=false
EHD BoolV BFFTF();
// x=false	y=false		z=true		w=true
EHD BoolV BFFTT();
// x=false	y=true		z=false		w=false
EHD BoolV BFTFF();
// x=false	y=true		z=false		w=true
EHD BoolV BFTFT();
// x=false	y=true		z=true		w=false
EHD BoolV BFTTF();
// x=false	y=true		z=true		w=true
EHD BoolV BFTTT();
// x=true	y=false		z=false		w=false
EHD BoolV BTFFF();
// x=true	y=false		z=false		w=true
EHD BoolV BTFFT();
// x=true	y=false		z=true		w=false
EHD BoolV BTFTF();
// x=true	y=false		z=true		w=true
EHD BoolV BTFTT();
// x=true	y=true		z=false		w=false
EHD BoolV BTTFF();
// x=true	y=true		z=false		w=true
EHD BoolV BTTFT();
// x=true	y=true		z=true		w=false
EHD BoolV BTTTF();
// x=true	y=true		z=true		w=true
EHD BoolV BTTTT();

// x=false	y=false		z=false		w=true
EHD BoolV BWMask();
// x=true	y=false		z=false		w=false
EHD BoolV BXMask();
// x=false	y=true		z=false		w=false
EHD BoolV BYMask();
// x=false	y=false		z=true		w=false
EHD BoolV BZMask();

// get x component
EHD BoolV BGetX(const BoolV f);
// get y component
EHD BoolV BGetY(const BoolV f);
// get z component
EHD BoolV BGetZ(const BoolV f);
// get w component
EHD BoolV BGetW(const BoolV f);

// Use elementIndex to splat xxxx or yyyy or zzzz or wwww
template <int elementIndex>
EHD BoolV BSplatElement(Vec4V a);

// component-wise && (AND)
EHD BoolV BAnd(const BoolV a, const BoolV b);
// component-wise || (OR)
EHD BoolV BOr(const BoolV a, const BoolV b);
// component-wise not
EHD BoolV BNot(const BoolV a);

// if all four components are true, return true, otherwise return false
EHD BoolV BAllTrue4(const BoolV a);

// if any four components is true, return true, otherwise return false
EHD BoolV BAnyTrue4(const BoolV a);

// if all three(0, 1, 2) components are true, return true, otherwise return false
EHD BoolV BAllTrue3(const BoolV a);

// if any three (0, 1, 2) components is true, return true, otherwise return false
EHD BoolV BAnyTrue3(const BoolV a);

// Return 1 if all components equal, zero otherwise.
EHD PxU32 BAllEq(const BoolV a, const BoolV b);

// Specialized/faster BAllEq function for b==TTTT
EHD PxU32 BAllEqTTTT(const BoolV a);
// Specialized/faster BAllEq function for b==FFFF
EHD PxU32 BAllEqFFFF(const BoolV a);

/// Get BoolV as bits set in an PxU32. A bit in the output is set if the element is 'true' in the input.
/// There is a bit for each element in a, with element 0s value held in bit0, element 1 in bit 1s and so forth.
/// If nothing is true in the input it will return 0, and if all are true if will return 0xf.
/// NOTE! That performance of the function varies considerably by platform, thus it is recommended to use
/// where your algorithm really needs a BoolV in an integer variable.
EHD PxU32 BGetBitMask(const BoolV a);

// VecI32V stuff

EHD VecI32V VecI32V_Zero();

EHD VecI32V VecI32V_One();

EHD VecI32V VecI32V_Two();

EHD VecI32V VecI32V_MinusOne();

// Compute a shift parameter for VecI32V_LeftShift and VecI32V_RightShift
// Each element of shift must be identical ie the vector must have form {count, count, count, count} with count>=0
EHD VecShiftV VecI32V_PrepareShift(const VecI32VArg shift);

// Shift each element of a leftwards by the same amount
// Compute shift with VecI32V_PrepareShift
//{a.x<<shift[0], a.y<<shift[0], a.z<<shift[0], a.w<<shift[0]}
EHD VecI32V VecI32V_LeftShift(const VecI32VArg a, const VecShiftVArg shift);

// Shift each element of a rightwards by the same amount
// Compute shift with VecI32V_PrepareShift
//{a.x>>shift[0], a.y>>shift[0], a.z>>shift[0], a.w>>shift[0]}
EHD VecI32V VecI32V_RightShift(const VecI32VArg a, const VecShiftVArg shift);

EHD VecI32V VecI32V_Add(const VecI32VArg a, const VecI32VArg b);

EHD VecI32V VecI32V_Or(const VecI32VArg a, const VecI32VArg b);

EHD VecI32V VecI32V_GetX(const VecI32VArg a);

EHD VecI32V VecI32V_GetY(const VecI32VArg a);

EHD VecI32V VecI32V_GetZ(const VecI32VArg a);

EHD VecI32V VecI32V_GetW(const VecI32VArg a);

EHD VecI32V VecI32V_Sub(const VecI32VArg a, const VecI32VArg b);

EHD BoolV VecI32V_IsGrtr(const VecI32VArg a, const VecI32VArg b);

EHD BoolV VecI32V_IsEq(const VecI32VArg a, const VecI32VArg b);

EHD VecI32V V4I32Sel(const BoolV c, const VecI32V a, const VecI32V b);

// VecU32V stuff

EHD VecU32V U4Zero();

EHD VecU32V U4One();

EHD VecU32V U4Two();

EHD BoolV V4IsEqU32(const VecU32V a, const VecU32V b);

EHD VecU32V V4U32Sel(const BoolV c, const VecU32V a, const VecU32V b);

EHD VecU32V V4U32or(VecU32V a, VecU32V b);

EHD VecU32V V4U32xor(VecU32V a, VecU32V b);

EHD VecU32V V4U32and(VecU32V a, VecU32V b);

EHD VecU32V V4U32Andc(VecU32V a, VecU32V b);

// VecU32 - why does this not return a bool?
EHD VecU32V V4IsGrtrV32u(const Vec4V a, const Vec4V b);

// Math operations on 16-byte aligned Mat33s (represents any 3x3 matrix)
EHD Mat33V M33Load(const PxMat33& m) 
{
	return Mat33V(Vec3V_From_Vec4V(V4LoadU(&m.column0.x)), 
	Vec3V_From_Vec4V(V4LoadU(&m.column1.x)), V3LoadU(m.column2)); 
}
// a*b
EHD Vec3V M33MulV3(const Mat33V& a, const Vec3V b);
// A*x + b
EHD Vec3V M33MulV3AddV3(const Mat33V& A, const Vec3V b, const Vec3V c);
// transpose(a) * b
EHD Vec3V M33TrnspsMulV3(const Mat33V& a, const Vec3V b);
// a*b
EHD Mat33V M33MulM33(const Mat33V& a, const Mat33V& b);
// a+b
EHD Mat33V M33Add(const Mat33V& a, const Mat33V& b);
// a+b
EHD Mat33V M33Sub(const Mat33V& a, const Mat33V& b);
//-a
EHD Mat33V M33Neg(const Mat33V& a);
// absolute value of the matrix
EHD Mat33V M33Abs(const Mat33V& a);
// inverse mat
EHD Mat33V M33Inverse(const Mat33V& a);
// transpose(a)
EHD Mat33V M33Trnsps(const Mat33V& a);
// create an identity matrix
EHD Mat33V M33Identity();

// create a vec3 to store the diagonal element of the M33
EHD Mat33V M33Diagonal(const Vec3VArg);

// Not implemented
// return 1 if all components of a are equal to all components of b
// EHD PxU32 V4U32AllEq(const VecU32V a, const VecU32V b);
// v.w=f
// EHD void V3WriteW(Vec3V& v, const PxF32 f);
// EHD PxF32 V3ReadW(const Vec3V& v);

// Not used
// EHD Vec4V V4LoadAligned(Vec4V* addr);
// EHD Vec4V V4LoadUnaligned(Vec4V* addr);
// floor(a)(per component)
// EHD Vec4V V4Floor(Vec4V a);
// ceil(a) (per component)
// EHD Vec4V V4Ceil(Vec4V a);
// EHD VecU32V V4ConvertToU32VSaturate(const Vec4V a, PxU32 power);

// Math operations on 16-byte aligned Mat34s (represents transformation matrix - rotation and translation).
// namespace _Mat34V
//{
//	//a*b
//	EHD Vec3V multiplyV(const Mat34V& a, const Vec3V b);
//	//a_rotation * b
//	EHD Vec3V multiply3X3V(const Mat34V& a, const Vec3V b);
//	//transpose(a_rotation)*b
//	EHD Vec3V multiplyTranspose3X3V(const Mat34V& a, const Vec3V b);
//	//a*b
//	EHD Mat34V multiplyV(const Mat34V& a, const Mat34V& b);
//	//a_rotation*b
//	EHD Mat33V multiply3X3V(const Mat34V& a, const Mat33V& b);
//	//a_rotation*b_rotation
//	EHD Mat33V multiply3X3V(const Mat34V& a, const Mat34V& b);
//	//a+b
//	EHD Mat34V addV(const Mat34V& a, const Mat34V& b);
//	//a^-1
//	EHD Mat34V getInverseV(const Mat34V& a);
//	//transpose(a_rotation)
//	EHD Mat33V getTranspose3X3(const Mat34V& a);
//}; //namespace _Mat34V

// a*b
//#define M34MulV3(a,b)			(M34MulV3(a,b))
////a_rotation * b
//#define M34Mul33V3(a,b)			(M34Mul33V3(a,b))
////transpose(a_rotation)*b
//#define M34TrnspsMul33V3(a,b)	(M34TrnspsMul33V3(a,b))
////a*b
//#define M34MulM34(a,b)			(_Mat34V::multiplyV(a,b))
// a_rotation*b
//#define M34MulM33(a,b)			(M34MulM33(a,b))
// a_rotation*b_rotation
//#define M34Mul33MM34(a,b)		(M34MulM33(a,b))
// a+b
//#define M34Add(a,b)				(M34Add(a,b))
////a^-1
//#define M34Inverse(a,b)			(M34Inverse(a))
// transpose(a_rotation)
//#define M34Trnsps33(a)			(M33Trnsps3X3(a))

// Math operations on 16-byte aligned Mat44s (represents any 4x4 matrix)
// namespace _Mat44V
//{
//	//a*b
//	EHD Vec4V multiplyV(const Mat44V& a, const Vec4V b);
//	//transpose(a)*b
//	EHD Vec4V multiplyTransposeV(const Mat44V& a, const Vec4V b);
//	//a*b
//	EHD Mat44V multiplyV(const Mat44V& a, const Mat44V& b);
//	//a+b
//	EHD Mat44V addV(const Mat44V& a, const Mat44V& b);
//	//a&-1
//	EHD Mat44V getInverseV(const Mat44V& a);
//	//transpose(a)
//	EHD Mat44V getTransposeV(const Mat44V& a);
//}; //namespace _Mat44V

// namespace _VecU32V
//{
//	// pack 8 U32s to 8 U16s with saturation
//	EHD VecU16V pack2U32VToU16VSaturate(VecU32V a, VecU32V b);
//	EHD VecU32V orV(VecU32V a, VecU32V b);
//	EHD VecU32V andV(VecU32V a, VecU32V b);
//	EHD VecU32V andcV(VecU32V a, VecU32V b);
//	// conversion from integer to float
//	EHD Vec4V convertToVec4V(VecU32V a);
//	// splat a[elementIndex] into all fields of a
//	template<int elementIndex>
//	EHD VecU32V splatElement(VecU32V a);
//	EHD void storeAligned(VecU32V a, VecU32V* address);
//};

// namespace _VecI32V
//{
//	template<int a> EHD VecI32V splatI32();
//};
//
// namespace _VecU16V
//{
//	EHD VecU16V orV(VecU16V a, VecU16V b);
//	EHD VecU16V andV(VecU16V a, VecU16V b);
//	EHD VecU16V andcV(VecU16V a, VecU16V b);
//	EHD void storeAligned(VecU16V val, VecU16V *address);
//	EHD VecU16V loadAligned(VecU16V* addr);
//	EHD VecU16V loadUnaligned(VecU16V* addr);
//	EHD VecU16V compareGt(VecU16V a, VecU16V b);
//	template<int elementIndex>
//	EHD VecU16V splatElement(VecU16V a);
//	EHD VecU16V subtractModulo(VecU16V a, VecU16V b);
//	EHD VecU16V addModulo(VecU16V a, VecU16V b);
//	EHD VecU32V getLo16(VecU16V a); // [0,2,4,6] 16-bit values to [0,1,2,3] 32-bit vector
//	EHD VecU32V getHi16(VecU16V a); // [1,3,5,7] 16-bit values to [0,1,2,3] 32-bit vector
//};
//
// namespace _VecI16V
//{
//	template <int val> EHD VecI16V splatImmediate();
//};
//
// namespace _VecU8V
//{
//};

// a*b
//#define M44MulV4(a,b)		(M44MulV4(a,b))
////transpose(a)*b
//#define M44TrnspsMulV4(a,b) (M44TrnspsMulV4(a,b))
////a*b
//#define M44MulM44(a,b)		(M44MulM44(a,b))
////a+b
//#define M44Add(a,b)			(M44Add(a,b))
////a&-1
//#define M44Inverse(a)		(M44Inverse(a))
////transpose(a)
//#define M44Trnsps(a)		(M44Trnsps(a))

// dsequeira: these used to be assert'd out in SIMD builds, but they're necessary if
// we want to be able to write some scalar functions which run using SIMD data structures

EHD void V3WriteX(Vec3V& v, const PxF32 f)
{
	reinterpret_cast<PxVec3&>(v).x = f;
}

EHD void V3WriteY(Vec3V& v, const PxF32 f)
{
	reinterpret_cast<PxVec3&>(v).y = f;
}

EHD void V3WriteZ(Vec3V& v, const PxF32 f)
{
	reinterpret_cast<PxVec3&>(v).z = f;
}

EHD void V3WriteXYZ(Vec3V& v, const PxVec3& f)
{
	reinterpret_cast<PxVec3&>(v) = f;
}

EHD PxF32 V3ReadX(const Vec3V& v)
{
	return reinterpret_cast<const PxVec3&>(v).x;
}

EHD PxF32 V3ReadY(const Vec3V& v)
{
	return reinterpret_cast<const PxVec3&>(v).y;
}

EHD PxF32 V3ReadZ(const Vec3V& v)
{
	return reinterpret_cast<const PxVec3&>(v).z;
}

EHD const PxVec3& V3ReadXYZ(const Vec3V& v)
{
	return reinterpret_cast<const PxVec3&>(v);
}

EHD void V4WriteX(Vec4V& v, const PxF32 f)
{
	reinterpret_cast<PxVec4&>(v).x = f;
}

EHD void V4WriteY(Vec4V& v, const PxF32 f)
{
	reinterpret_cast<PxVec4&>(v).y = f;
}

EHD void V4WriteZ(Vec4V& v, const PxF32 f)
{
	reinterpret_cast<PxVec4&>(v).z = f;
}

EHD void V4WriteW(Vec4V& v, const PxF32 f)
{
	reinterpret_cast<PxVec4&>(v).w = f;
}

EHD void V4WriteXYZ(Vec4V& v, const PxVec3& f)
{
	reinterpret_cast<PxVec3&>(v) = f;
}

EHD PxF32 V4ReadX(const Vec4V& v)
{
	return reinterpret_cast<const PxVec4&>(v).x;
}

EHD PxF32 V4ReadY(const Vec4V& v)
{
	return reinterpret_cast<const PxVec4&>(v).y;
}

EHD PxF32 V4ReadZ(const Vec4V& v)
{
	return reinterpret_cast<const PxVec4&>(v).z;
}

EHD PxF32 V4ReadW(const Vec4V& v)
{
	return reinterpret_cast<const PxVec4&>(v).w;
}

EHD const PxVec3& V4ReadXYZ(const Vec4V& v)
{
	return reinterpret_cast<const PxVec3&>(v);
}

// this macro transposes 4 Vec4V into 3 Vec4V (assuming that the W component can be ignored
//inA:  1   2   3   4
//inB:  5   6   7   8
//inC:  9  10  11  12
//inD:  13 14  15  16
//outA: 1   5   9  13
//outB: 2   6  10  14
//outC: 3   7  11  15
#define PX_TRANSPOSE_44_34(inA, inB, inC, inD, outA, outB, outC)	\
outA = V4UnpackXY(inA, inC);                                        \
inA = V4UnpackZW(inA, inC);                                         \
inC = V4UnpackXY(inB, inD);                                         \
inB = V4UnpackZW(inB, inD);                                         \
outB = V4UnpackZW(outA, inC);                                       \
outA = V4UnpackXY(outA, inC);                                       \
outC = V4UnpackXY(inA, inB);

// this macro transposes 3 Vec4V into 4 Vec4V (with W components as garbage!)
//inA:  1   2   3   4
//inB:  5   6   7   8
//inC:  9  10  11  12
//outA: 1   5   9  undefined
//outB: 2   6  10  undefined
//outC: 3   7  11  undefined
//outD: 4   8  12  undefined
#define PX_TRANSPOSE_34_44(inA, inB, inC, outA, outB, outC, outD)	\
	outA = V4UnpackXY(inA, inC);                                    \
	inA = V4UnpackZW(inA, inC);                                     \
	outC = V4UnpackXY(inB, inB);                                    \
	inC = V4UnpackZW(inB, inB);                                     \
	outB = V4UnpackZW(outA, outC);                                  \
	outA = V4UnpackXY(outA, outC);                                  \
	outC = V4UnpackXY(inA, inC);                                    \
	outD = V4UnpackZW(inA, inC);

//inA:  1   2   3   4
//inB:  5   6   7   8
//inC:  9  10  11  12
//inD:  13 14  15  16
//outA: 1   5   9  13
//outB: 2   6  10  14
//outC: 3   7  11  15
//outD: 4   8  12  16
#define PX_TRANSPOSE_44(inA, inB, inC, inD, outA, outB, outC, outD)	\
	outA = V4UnpackXY(inA, inC);                                    \
	inA = V4UnpackZW(inA, inC);                                     \
	inC = V4UnpackXY(inB, inD);                                     \
	inB = V4UnpackZW(inB, inD);                                     \
	outB = V4UnpackZW(outA, inC);                                   \
	outA = V4UnpackXY(outA, inC);                                   \
	outC = V4UnpackXY(inA, inB);                                    \
	outD = V4UnpackZW(inA, inB);

// This function returns a Vec4V, where each element is the dot product of one pair of Vec3Vs. On PC, each element in
// the result should be identical to the results if V3Dot was performed
// for each pair of Vec3V.
// However, on other platforms, the result might diverge by some small margin due to differences in FP rounding, e.g. if
// _mm_dp_ps was used or some other approximate dot product or fused madd operations
// were used.
// Where there does not exist a hw-accelerated dot-product operation, this approach should be the fastest way to compute
// the dot product of 4 vectors.
EHD Vec4V V3Dot4(const Vec3VArg a0, const Vec3VArg b0, const Vec3VArg a1, const Vec3VArg b1,
                             const Vec3VArg a2, const Vec3VArg b2, const Vec3VArg a3, const Vec3VArg b3)
{
	Vec4V a0b0 = Vec4V_From_Vec3V(V3Mul(a0, b0));
	Vec4V a1b1 = Vec4V_From_Vec3V(V3Mul(a1, b1));
	Vec4V a2b2 = Vec4V_From_Vec3V(V3Mul(a2, b2));
	Vec4V a3b3 = Vec4V_From_Vec3V(V3Mul(a3, b3));

	Vec4V aTrnsps, bTrnsps, cTrnsps;

	PX_TRANSPOSE_44_34(a0b0, a1b1, a2b2, a3b3, aTrnsps, bTrnsps, cTrnsps);

	return V4Add(V4Add(aTrnsps, bTrnsps), cTrnsps);
}

//(f.x,f.y,f.z,0) - Alternative/faster V3LoadU implementation when it is safe to read "W", i.e. the 32bits after the PxVec3.
EHD Vec3V V3LoadU_SafeReadW(const PxVec3& f)
{
	return Vec3V_From_Vec4V(V4LoadU(&f.x));
}

} // namespace aos
} // namespace eng

// Now for the cross-platform implementations of the 16-byte aligned maths functions (win32/360/ppu/spu etc).




// ======== PxUnixTrigConstants.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========



namespace eng
{
namespace aos
{



PX_ALIGN_PREFIX(16)
struct PX_VECTORF32
{
	float f[4];
} PX_ALIGN_SUFFIX(16);

alignas(16) static const PX_VECTORF32 g_PXSinCoefficients0_h = { { 1.0f, -0.166666667f, 8.333333333e-3f, -1.984126984e-4f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXSinCoefficients0_d = { { 1.0f, -0.166666667f, 8.333333333e-3f, -1.984126984e-4f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXSinCoefficients1_h = { { 2.755731922e-6f, -2.505210839e-8f, 1.605904384e-10f, -7.647163732e-13f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXSinCoefficients1_d = { { 2.755731922e-6f, -2.505210839e-8f, 1.605904384e-10f, -7.647163732e-13f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXSinCoefficients2_h = { { 2.811457254e-15f, -8.220635247e-18f, 1.957294106e-20f, -3.868170171e-23f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXSinCoefficients2_d = { { 2.811457254e-15f, -8.220635247e-18f, 1.957294106e-20f, -3.868170171e-23f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXCosCoefficients0_h = { { 1.0f, -0.5f, 4.166666667e-2f, -1.388888889e-3f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXCosCoefficients0_d = { { 1.0f, -0.5f, 4.166666667e-2f, -1.388888889e-3f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXCosCoefficients1_h = { { 2.480158730e-5f, -2.755731922e-7f, 2.087675699e-9f, -1.147074560e-11f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXCosCoefficients1_d = { { 2.480158730e-5f, -2.755731922e-7f, 2.087675699e-9f, -1.147074560e-11f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXCosCoefficients2_h = { { 4.779477332e-14f, -1.561920697e-16f, 4.110317623e-19f, -8.896791392e-22f } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXCosCoefficients2_d = { { 4.779477332e-14f, -1.561920697e-16f, 4.110317623e-19f, -8.896791392e-22f } };
#endif

alignas(16) static const PX_VECTORF32 g_PXReciprocalTwoPi_h = { { PxInvTwoPi, PxInvTwoPi, PxInvTwoPi, PxInvTwoPi } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXReciprocalTwoPi_d = { { PxInvTwoPi, PxInvTwoPi, PxInvTwoPi, PxInvTwoPi } };
#endif

alignas(16) static const PX_VECTORF32 g_PXTwoPi_h = { { PxTwoPi, PxTwoPi, PxTwoPi, PxTwoPi } };
#if defined(__CUDACC__)
__device__ __constant__ static const PX_VECTORF32 g_PXTwoPi_d = { { PxTwoPi, PxTwoPi, PxTwoPi, PxTwoPi } };
#endif


} // namespace aos
} // namespace eng



// ======== PxVecMathSSE.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========



namespace eng
{
namespace aos
{

namespace
{
	const PX_ALIGN(16, PxF32) minus1w[4] = { 0.0f, 0.0f, 0.0f, -1.0f };
}

EHD void QuatGetMat33V(const QuatVArg q, Vec3V& column0, Vec3V& column1, Vec3V& column2)
{
    const sse::m128 q2 = V4Add(q, q);
    const sse::m128 qw2 = V4MulAdd(q2, V4GetW(q), sse::_mm_load_ps(minus1w));			// (2wx, 2wy, 2wz, 2ww-1)
    const sse::m128 nw2 = Vec3V_From_Vec4V(V4Neg(qw2));							// (-2wx, -2wy, -2wz, 0)
    const sse::m128 v = Vec3V_From_Vec4V(q);

    const sse::m128 a0 = sse::_mm_shuffle_ps(qw2, nw2, ENG_MM_SHUFFLE(3, 1, 2, 3));		// (2ww-1, 2wz, -2wy, 0)
    column0 = V4MulAdd(v, V4GetX(q2), a0);

    const sse::m128 a1 = sse::_mm_shuffle_ps(qw2, nw2, ENG_MM_SHUFFLE(3, 2, 0, 3));		// (2ww-1, 2wx, -2wz, 0)
    column1 = V4MulAdd(v, V4GetY(q2), sse::_mm_shuffle_ps(a1, a1, ENG_MM_SHUFFLE(3, 1, 0, 2)));

    const sse::m128 a2 = sse::_mm_shuffle_ps(qw2, nw2, ENG_MM_SHUFFLE(3, 0, 1, 3));		// (2ww-1, 2wy, -2wx, 0)
    column2 = V4MulAdd(v, V4GetZ(q2), sse::_mm_shuffle_ps(a2, a2, ENG_MM_SHUFFLE(3, 0, 2, 1)));
}

//////////////////////////////////////////////////////////////////////
//Test that Vec3V and FloatV are legal
//////////////////////////////////////////////////////////////////////

EHD bool isValidVec3V(const Vec3V a)
{
	//using _mm_comieq_ss to do the comparison doesn't work for NaN.
	PX_ALIGN(16, PxF32 f[4]);
	V4StoreA(a, f);
	return f[3] == 0.0f;
}

EHD bool isFiniteLength(const Vec3V a)
{
	return !FAllEq(V4LengthSq(a), FZero());
}

EHD bool isAligned16(const void* a)
{
	return (0 == (size_t(a) & 0x0f));
}

//ASSERT_FINITELENGTH is deactivated because there is a lot of code that calls a simd normalisation function with zero length but then ignores the result.

	#define ASSERT_ISVALIDVEC3V(a)
	#define ASSERT_ISVALIDFLOATV(a) 
	#define ASSERT_ISALIGNED16(a)
	#define ASSERT_ISFINITELENGTH(a)

namespace internalSimd
{
EHD sse::m128 m128_I2F(sse::m128i n)
{
	return sse::_mm_castsi128_ps(n);
}

EHD sse::m128i m128_F2I(sse::m128 n)
{
	return sse::_mm_castps_si128(n);
}

EHD PxU32 BAllTrue4_R(const BoolV a)
{
	const PxI32 moveMask = sse::_mm_movemask_ps(a);
	return PxU32(moveMask == 0xf);
}

EHD PxU32 BAllTrue3_R(const BoolV a)
{
	const PxI32 moveMask = sse::_mm_movemask_ps(a);
	return PxU32((moveMask & 0x7) == 0x7);
}

EHD PxU32 BAnyTrue4_R(const BoolV a)
{
	const PxI32 moveMask = sse::_mm_movemask_ps(a);
	return PxU32(moveMask != 0x0);
}

EHD PxU32 BAnyTrue3_R(const BoolV a)
{
	const PxI32 moveMask = sse::_mm_movemask_ps(a);
	return PxU32((moveMask & 0x7) != 0x0);
}

EHD PxU32 FiniteTestEq(const Vec4V a, const Vec4V b)
{
	// This is a bit of a bodge.
	//_mm_comieq_ss returns 1 if either value is nan so we need to re-cast a and b with true encoded as a non-nan
	// number.
	// There must be a better way of doing this in sse.
	const BoolV one = FOne();
	const BoolV zero = FZero();
	const BoolV a1 = V4Sel(a, one, zero);
	const BoolV b1 = V4Sel(b, one, zero);
	return (PxU32(
	    sse::_mm_comieq_ss(a1, b1) &&
	    sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a1, a1, ENG_MM_SHUFFLE(1, 1, 1, 1)), sse::_mm_shuffle_ps(b1, b1, ENG_MM_SHUFFLE(1, 1, 1, 1))) &&
	    sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a1, a1, ENG_MM_SHUFFLE(2, 2, 2, 2)), sse::_mm_shuffle_ps(b1, b1, ENG_MM_SHUFFLE(2, 2, 2, 2))) &&
	    sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a1, a1, ENG_MM_SHUFFLE(3, 3, 3, 3)), sse::_mm_shuffle_ps(b1, b1, ENG_MM_SHUFFLE(3, 3, 3, 3)))));
}

EHD bool hasZeroElementInFloatV(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0)), FZero()) ? true : false;
}

EHD bool hasZeroElementInVec3V(const Vec3V a)
{
	return (sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0)), FZero()) ||
			sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 1, 1, 1)), FZero()) ||
			sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2)), FZero()));
}

EHD bool hasZeroElementInVec4V(const Vec4V a)
{
	return (sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0)), FZero()) ||
	        sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 1, 1, 1)), FZero()) ||
	        sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2)), FZero()) ||
	        sse::_mm_comieq_ss(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 3, 3, 3)), FZero()));
}

} //internalSimd

namespace vecMathTests
{
// PT: this function returns an invalid Vec3V (W!=0.0f) just for unit-testing 'isValidVec3V'
EHD Vec3V getInvalidVec3V()
{
	const float f = 1.0f;
	return sse::_mm_load1_ps(&f);
}

EHD bool allElementsEqualFloatV(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_comieq_ss(a, b) != 0;
}

EHD bool allElementsEqualVec3V(const Vec3V a, const Vec3V b)
{
	return V3AllEq(a, b) != 0;
}

EHD bool allElementsEqualVec4V(const Vec4V a, const Vec4V b)
{
	return V4AllEq(a, b) != 0;
}

EHD bool allElementsEqualVecU32V(const VecU32V a, const VecU32V b)
{
	return internalSimd::BAllTrue4_R(V4IsEqU32(a, b)) != 0;
}

} //vecMathTests

/////////////////////////////////////////////////////////////////////
////VECTORISED FUNCTION IMPLEMENTATIONS
/////////////////////////////////////////////////////////////////////

EHD BoolV BLoad(const bool f)
{
	const PxU32 i = PxU32(-PxI32(f));
	return sse::_mm_load1_ps(reinterpret_cast<const float*>(&i));
}

EHD FloatV FLoad(const PxF32 f)
{
	return sse::_mm_load1_ps(&f);
}

EHD Vec3V V3Load(const PxF32 f)
{
	return sse::_mm_set_ps(0.0f, f, f, f);
}

EHD Vec4V V4Load(const PxF32 f)
{
	return sse::_mm_load1_ps(&f);
}

EHD Vec3V V3LoadU(const PxVec3& f)
{
	return sse::_mm_set_ps(0.0f, f.z, f.y, f.x);
}

EHD Vec3V V3LoadU(const PxF32* const i)
{
	return sse::_mm_set_ps(0.0f, i[2], i[1], i[0]);
}

EHD Vec3V Vec3V_From_Vec4V(Vec4V v)
{
	return V4ClearW(v);
}

EHD Vec3V Vec3V_From_Vec4V_WUndefined(const Vec4V v)
{
	return v;
}

EHD Vec4V Vec4V_From_PxVec3_WUndefined(const PxVec3& f)
{
	return sse::_mm_set_ps(0.0f, f.z, f.y, f.x);
}

EHD Vec4V Vec4V_From_FloatV(FloatV f)
{
	return f;
}

EHD Vec4V Vec4V_From_Vec3V(Vec3V f)
{
	ASSERT_ISVALIDVEC3V(f);
	return f; // ok if it is implemented as the same type.
}

EHD Vec3V Vec3V_From_FloatV(FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return Vec3V_From_Vec4V(Vec4V_From_FloatV(f));
}

EHD Vec3V Vec3V_From_FloatV_WUndefined(FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return Vec3V_From_Vec4V_WUndefined(Vec4V_From_FloatV(f));
}

EHD Vec4V V4LoadA(const PxF32* const f)
{
	ASSERT_ISALIGNED16(f);
	return sse::_mm_load_ps(f);
}

EHD Vec4V V4LoadU(const PxF32* const f)
{
	return sse::_mm_loadu_ps(f);
}

EHD void V4StoreA(const Vec4V a, PxF32* f)
{
	ASSERT_ISALIGNED16(f);
	sse::_mm_store_ps(f, a);
}

EHD void V4StoreU(const Vec4V a, PxF32* f)
{
	sse::_mm_storeu_ps(f, a);
}

EHD void BStoreA(const BoolV a, PxU32* f)
{
	ASSERT_ISALIGNED16(f);
	sse::_mm_store_ps(reinterpret_cast<PxF32*>(f), a);
}

EHD void U4StoreA(const VecU32V uv, PxU32* u)
{
	ASSERT_ISALIGNED16(u);
	sse::_mm_store_ps(reinterpret_cast<float*>(u), uv);
}

EHD void FStore(const FloatV a, PxF32* PX_RESTRICT f)
{
	ASSERT_ISVALIDFLOATV(a);
	sse::_mm_store_ss(f, a);
}

EHD void Store_From_BoolV(const BoolV b, PxU32* b2)
{
	sse::_mm_store_ss(reinterpret_cast<PxF32*>(b2), b);
}

//////////////////////////////////
// FLOATV
//////////////////////////////////

EHD FloatV FZero()
{
	return sse::_mm_setzero_ps();
}

EHD FloatV FOne()
{
	return FLoad(1.0f);
}

EHD FloatV FHalf()
{
	return FLoad(0.5f);
}

EHD FloatV FEps()
{
	return FLoad(PX_EPS_REAL);
}

EHD FloatV FEps6()
{
	return FLoad(1e-6f);
}

EHD FloatV FMax()
{
	return FLoad(PX_MAX_REAL);
}

EHD FloatV FNegMax()
{
	return FLoad(-PX_MAX_REAL);
}

EHD FloatV IZero()
{
	const PxU32 zero = 0;
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&zero));
}

EHD FloatV IOne()
{
	const PxU32 one = 1;
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&one));
}

EHD FloatV ITwo()
{
	const PxU32 two = 2;
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&two));
}

EHD FloatV IThree()
{
	const PxU32 three = 3;
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&three));
}

EHD FloatV IFour()
{
	const PxU32 four = 4;
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&four));
}

EHD FloatV FNeg(const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return sse::_mm_sub_ps(sse::_mm_setzero_ps(), f);
}

EHD FloatV FAdd(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_add_ps(a, b);
}

EHD FloatV FSub(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_sub_ps(a, b);
}

EHD FloatV FMul(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_mul_ps(a, b);
}

EHD FloatV FDiv(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_div_ps(a, b);
}

EHD FloatV FDivFast(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_mul_ps(a, sse::_mm_rcp_ps(b));
}

EHD FloatV FRecip(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_div_ps(FOne(), a);
}

EHD FloatV FRecipFast(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_rcp_ps(a);
}

EHD FloatV FRsqrt(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_div_ps(FOne(), sse::_mm_sqrt_ps(a));
}

EHD FloatV FSqrt(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_sqrt_ps(a);
}

EHD FloatV FRsqrtFast(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	return sse::_mm_rsqrt_ps(a);
}

EHD FloatV FScaleAdd(const FloatV a, const FloatV b, const FloatV c)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	ASSERT_ISVALIDFLOATV(c);
	return FAdd(FMul(a, b), c);
}

EHD FloatV FNegScaleSub(const FloatV a, const FloatV b, const FloatV c)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	ASSERT_ISVALIDFLOATV(c);
	return FSub(c, FMul(a, b));
}

EHD FloatV FSel(const BoolV c, const FloatV a, const FloatV b)
{
	PX_ASSERT(vecMathTests::allElementsEqualBoolV(c, BTTTT()) ||
			  vecMathTests::allElementsEqualBoolV(c, BFFFF()));
	ASSERT_ISVALIDFLOATV(sse::_mm_or_ps(sse::_mm_andnot_ps(c, b), sse::_mm_and_ps(c, a)));
	return sse::_mm_or_ps(sse::_mm_andnot_ps(c, b), sse::_mm_and_ps(c, a));
}

EHD BoolV FIsGrtr(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_cmpgt_ps(a, b);
}

EHD BoolV FIsGrtrOrEq(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_cmpge_ps(a, b);
}

EHD BoolV FIsEq(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_cmpeq_ps(a, b);
}

EHD FloatV FMax(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_max_ps(a, b);
}

EHD FloatV FMin(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_min_ps(a, b);
}

EHD FloatV FClamp(const FloatV a, const FloatV minV, const FloatV maxV)
{
	ASSERT_ISVALIDFLOATV(minV);
	ASSERT_ISVALIDFLOATV(maxV);
	return sse::_mm_max_ps(sse::_mm_min_ps(a, maxV), minV);
}

EHD PxU32 FAllGrtr(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return PxU32(sse::_mm_comigt_ss(a, b));
}

EHD PxU32 FAllGrtrOrEq(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return PxU32(sse::_mm_comige_ss(a, b));
}

EHD PxU32 FAllEq(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	return PxU32(sse::_mm_comieq_ss(a, b));
}

EHD FloatV FRound(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	// return sse::_mm_round_ps(a, 0x0);
	const FloatV half = FLoad(0.5f);
	const sse::m128 signBit = sse::_mm_cvtepi32_ps(sse::_mm_srli_epi32(sse::_mm_cvtps_epi32(a), 31));
	const FloatV aRound = FSub(FAdd(a, half), signBit);
	sse::m128i tmp = sse::_mm_cvttps_epi32(aRound);
	return sse::_mm_cvtepi32_ps(tmp);
}

EHD FloatV FSin(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);

	// Modulo the range of the given angles such that -XM_2PI <= Angles < XM_2PI
	const FloatV recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const FloatV twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const FloatV tmp = FMul(a, recipTwoPi);
	const FloatV b = FRound(tmp);
	const FloatV V1 = FNegScaleSub(twoPi, b, a);

	// sin(V) ~= V - V^3 / 3! + V^5 / 5! - V^7 / 7! + V^9 / 9! - V^11 / 11! + V^13 / 13! -
	//           V^15 / 15! + V^17 / 17! - V^19 / 19! + V^21 / 21! - V^23 / 23! (for -PI <= V < PI)
	const FloatV V2 = FMul(V1, V1);
	const FloatV V3 = FMul(V2, V1);
	const FloatV V5 = FMul(V3, V2);
	const FloatV V7 = FMul(V5, V2);
	const FloatV V9 = FMul(V7, V2);
	const FloatV V11 = FMul(V9, V2);
	const FloatV V13 = FMul(V11, V2);
	const FloatV V15 = FMul(V13, V2);
	const FloatV V17 = FMul(V15, V2);
	const FloatV V19 = FMul(V17, V2);
	const FloatV V21 = FMul(V19, V2);
	const FloatV V23 = FMul(V21, V2);

	const Vec4V sinCoefficients0 = V4LoadA(ENG_G(g_PXSinCoefficients0).f);
	const Vec4V sinCoefficients1 = V4LoadA(ENG_G(g_PXSinCoefficients1).f);
	const Vec4V sinCoefficients2 = V4LoadA(ENG_G(g_PXSinCoefficients2).f);

	const FloatV S1 = V4GetY(sinCoefficients0);
	const FloatV S2 = V4GetZ(sinCoefficients0);
	const FloatV S3 = V4GetW(sinCoefficients0);
	const FloatV S4 = V4GetX(sinCoefficients1);
	const FloatV S5 = V4GetY(sinCoefficients1);
	const FloatV S6 = V4GetZ(sinCoefficients1);
	const FloatV S7 = V4GetW(sinCoefficients1);
	const FloatV S8 = V4GetX(sinCoefficients2);
	const FloatV S9 = V4GetY(sinCoefficients2);
	const FloatV S10 = V4GetZ(sinCoefficients2);
	const FloatV S11 = V4GetW(sinCoefficients2);

	FloatV Result;
	Result = FScaleAdd(S1, V3, V1);
	Result = FScaleAdd(S2, V5, Result);
	Result = FScaleAdd(S3, V7, Result);
	Result = FScaleAdd(S4, V9, Result);
	Result = FScaleAdd(S5, V11, Result);
	Result = FScaleAdd(S6, V13, Result);
	Result = FScaleAdd(S7, V15, Result);
	Result = FScaleAdd(S8, V17, Result);
	Result = FScaleAdd(S9, V19, Result);
	Result = FScaleAdd(S10, V21, Result);
	Result = FScaleAdd(S11, V23, Result);

	return Result;
}

EHD FloatV FCos(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);

	// Modulo the range of the given angles such that -XM_2PI <= Angles < XM_2PI
	const FloatV recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const FloatV twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const FloatV tmp = FMul(a, recipTwoPi);
	const FloatV b = FRound(tmp);
	const FloatV V1 = FNegScaleSub(twoPi, b, a);

	// cos(V) ~= 1 - V^2 / 2! + V^4 / 4! - V^6 / 6! + V^8 / 8! - V^10 / 10! + V^12 / 12! -
	//           V^14 / 14! + V^16 / 16! - V^18 / 18! + V^20 / 20! - V^22 / 22! (for -PI <= V < PI)
	const FloatV V2 = FMul(V1, V1);
	const FloatV V4 = FMul(V2, V2);
	const FloatV V6 = FMul(V4, V2);
	const FloatV V8 = FMul(V4, V4);
	const FloatV V10 = FMul(V6, V4);
	const FloatV V12 = FMul(V6, V6);
	const FloatV V14 = FMul(V8, V6);
	const FloatV V16 = FMul(V8, V8);
	const FloatV V18 = FMul(V10, V8);
	const FloatV V20 = FMul(V10, V10);
	const FloatV V22 = FMul(V12, V10);

	const Vec4V cosCoefficients0 = V4LoadA(ENG_G(g_PXCosCoefficients0).f);
	const Vec4V cosCoefficients1 = V4LoadA(ENG_G(g_PXCosCoefficients1).f);
	const Vec4V cosCoefficients2 = V4LoadA(ENG_G(g_PXCosCoefficients2).f);

	const FloatV C1 = V4GetY(cosCoefficients0);
	const FloatV C2 = V4GetZ(cosCoefficients0);
	const FloatV C3 = V4GetW(cosCoefficients0);
	const FloatV C4 = V4GetX(cosCoefficients1);
	const FloatV C5 = V4GetY(cosCoefficients1);
	const FloatV C6 = V4GetZ(cosCoefficients1);
	const FloatV C7 = V4GetW(cosCoefficients1);
	const FloatV C8 = V4GetX(cosCoefficients2);
	const FloatV C9 = V4GetY(cosCoefficients2);
	const FloatV C10 = V4GetZ(cosCoefficients2);
	const FloatV C11 = V4GetW(cosCoefficients2);

	FloatV Result;
	Result = FScaleAdd(C1, V2, V4One());
	Result = FScaleAdd(C2, V4, Result);
	Result = FScaleAdd(C3, V6, Result);
	Result = FScaleAdd(C4, V8, Result);
	Result = FScaleAdd(C5, V10, Result);
	Result = FScaleAdd(C6, V12, Result);
	Result = FScaleAdd(C7, V14, Result);
	Result = FScaleAdd(C8, V16, Result);
	Result = FScaleAdd(C9, V18, Result);
	Result = FScaleAdd(C10, V20, Result);
	Result = FScaleAdd(C11, V22, Result);

	return Result;
}

EHD PxU32 FOutOfBounds(const FloatV a, const FloatV min, const FloatV max)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(min);
	ASSERT_ISVALIDFLOATV(max);
	const BoolV c = BOr(FIsGrtr(a, max), FIsGrtr(min, a));
	return PxU32(!BAllEqFFFF(c));
}

EHD PxU32 FInBounds(const FloatV a, const FloatV min, const FloatV max)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(min);
	ASSERT_ISVALIDFLOATV(max);
	const BoolV c = BAnd(FIsGrtrOrEq(a, min), FIsGrtrOrEq(max, a));
	return BAllEqTTTT(c);
}

EHD PxU32 FOutOfBounds(const FloatV a, const FloatV bounds)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(bounds);
	return FOutOfBounds(a, FNeg(bounds), bounds);
}

EHD PxU32 FInBounds(const FloatV a, const FloatV bounds)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(bounds);
	return FInBounds(a, FNeg(bounds), bounds);
}

//////////////////////////////////
// VEC3V
//////////////////////////////////

EHD Vec3V V3Splat(const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	const sse::m128 zero = sse::_mm_setzero_ps();
	const sse::m128 fff0 = sse::_mm_move_ss(f, zero);
	return sse::_mm_shuffle_ps(fff0, fff0, ENG_MM_SHUFFLE(0, 1, 2, 3));
}

EHD Vec3V V3Merge(const FloatVArg x, const FloatVArg y, const FloatVArg z)
{
	ASSERT_ISVALIDFLOATV(x);
	ASSERT_ISVALIDFLOATV(y);
	ASSERT_ISVALIDFLOATV(z);
	// static on zero causes compiler crash on x64 debug_opt
	const sse::m128 zero = sse::_mm_setzero_ps();
	const sse::m128 xy = sse::_mm_move_ss(x, y);
	const sse::m128 z0 = sse::_mm_move_ss(zero, z);

	return sse::_mm_shuffle_ps(xy, z0, ENG_MM_SHUFFLE(1, 0, 0, 1));
}

EHD FloatV V3GetX(const Vec3V f)
{
	ASSERT_ISVALIDVEC3V(f);
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(0, 0, 0, 0));
}

EHD FloatV V3GetY(const Vec3V f)
{
	ASSERT_ISVALIDVEC3V(f);
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(1, 1, 1, 1));
}

EHD FloatV V3GetZ(const Vec3V f)
{
	ASSERT_ISVALIDVEC3V(f);
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(2, 2, 2, 2));
}

EHD Vec3V V3SetX(const Vec3V v, const FloatV f)
{
	ASSERT_ISVALIDVEC3V(v);
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BFTTT(), v, f);
}

EHD Vec3V V3SetY(const Vec3V v, const FloatV f)
{
	ASSERT_ISVALIDVEC3V(v);
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BTFTT(), v, f);
}

EHD Vec3V V3SetZ(const Vec3V v, const FloatV f)
{
	ASSERT_ISVALIDVEC3V(v);
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BTTFT(), v, f);
}

EHD Vec3V V3ColX(const Vec3V a, const Vec3V b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	ASSERT_ISVALIDVEC3V(c);
	const Vec3V r = sse::_mm_shuffle_ps(a, c, ENG_MM_SHUFFLE(3, 0, 3, 0));
	return V3SetY(r, V3GetX(b));
}

EHD Vec3V V3ColY(const Vec3V a, const Vec3V b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	ASSERT_ISVALIDVEC3V(c);
	const Vec3V r = sse::_mm_shuffle_ps(a, c, ENG_MM_SHUFFLE(3, 1, 3, 1));
	return V3SetY(r, V3GetY(b));
}

EHD Vec3V V3ColZ(const Vec3V a, const Vec3V b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	ASSERT_ISVALIDVEC3V(c);
	const Vec3V r = sse::_mm_shuffle_ps(a, c, ENG_MM_SHUFFLE(3, 2, 3, 2));
	return V3SetY(r, V3GetZ(b));
}

EHD Vec3V V3Zero()
{
	return sse::_mm_setzero_ps();
}

EHD Vec3V V3One()
{
	return V3Load(1.0f);
}

EHD Vec3V V3Eps()
{
	return V3Load(PX_EPS_REAL);
}

EHD Vec3V V3Neg(const Vec3V f)
{
	ASSERT_ISVALIDVEC3V(f);
	return sse::_mm_sub_ps(sse::_mm_setzero_ps(), f);
}

EHD Vec3V V3Add(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_add_ps(a, b);
}

EHD Vec3V V3Sub(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_sub_ps(a, b);
}

EHD Vec3V V3Scale(const Vec3V a, const FloatV b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_mul_ps(a, b);
}

EHD Vec3V V3Mul(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_mul_ps(a, b);
}

EHD Vec3V V3ScaleInv(const Vec3V a, const FloatV b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_div_ps(a, b);
}

EHD Vec3V V3Div(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return V4ClearW(sse::_mm_div_ps(a, b));
}

EHD Vec3V V3ScaleInvFast(const Vec3V a, const FloatV b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_mul_ps(a, sse::_mm_rcp_ps(b));
}

EHD Vec3V V3DivFast(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return V4ClearW(sse::_mm_mul_ps(a, sse::_mm_rcp_ps(b)));
}

EHD Vec3V V3Recip(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 zero = V3Zero();
	const sse::m128 tttf = BTTTF();
	const sse::m128 recipA = sse::_mm_div_ps(V3One(), a);
	return V4Sel(tttf, recipA, zero);
}

EHD Vec3V V3RecipFast(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 zero = V3Zero();
	const sse::m128 tttf = BTTTF();
	const sse::m128 recipA = sse::_mm_rcp_ps(a);
	return V4Sel(tttf, recipA, zero);
}

EHD Vec3V V3Rsqrt(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 zero = V3Zero();
	const sse::m128 tttf = BTTTF();
	const sse::m128 recipA = sse::_mm_div_ps(V3One(), sse::_mm_sqrt_ps(a));
	return V4Sel(tttf, recipA, zero);
}

EHD Vec3V V3RsqrtFast(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 zero = V3Zero();
	const sse::m128 tttf = BTTTF();
	const sse::m128 recipA = sse::_mm_rsqrt_ps(a);
	return V4Sel(tttf, recipA, zero);
}

EHD Vec3V V3ScaleAdd(const Vec3V a, const FloatV b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDFLOATV(b);
	ASSERT_ISVALIDVEC3V(c);
	return V3Add(V3Scale(a, b), c);
}

EHD Vec3V V3NegScaleSub(const Vec3V a, const FloatV b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDFLOATV(b);
	ASSERT_ISVALIDVEC3V(c);
	return V3Sub(c, V3Scale(a, b));
}

EHD Vec3V V3MulAdd(const Vec3V a, const Vec3V b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	ASSERT_ISVALIDVEC3V(c);
	return V3Add(V3Mul(a, b), c);
}

EHD Vec3V V3NegMulSub(const Vec3V a, const Vec3V b, const Vec3V c)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	ASSERT_ISVALIDVEC3V(c);
	return V3Sub(c, V3Mul(a, b));
}

EHD Vec3V V3Abs(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return V3Max(a, V3Neg(a));
}

EHD FloatV V3Dot(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	const sse::m128 t0 = sse::_mm_mul_ps(a, b);								//	aw*bw | az*bz | ay*by | ax*bx
	const sse::m128 t1 = sse::_mm_shuffle_ps(t0, t0, ENG_MM_SHUFFLE(1,0,3,2));	//	ay*by | ax*bx | aw*bw | az*bz
	const sse::m128 t2 = sse::_mm_add_ps(t0, t1);							//	ay*by + aw*bw | ax*bx + az*bz | aw*bw + ay*by | az*bz + ax*bx
	const sse::m128 t3 = sse::_mm_shuffle_ps(t2, t2, ENG_MM_SHUFFLE(2,3,0,1));	//	ax*bx + az*bz | ay*by + aw*bw | az*bz + ax*bx | aw*bw + ay*by
	return sse::_mm_add_ps(t3, t2);										//	ax*bx + az*bz + ay*by + aw*bw 
																	//	ay*by + aw*bw + ax*bx + az*bz
																	//	az*bz + ax*bx + aw*bw + ay*by
																	//	aw*bw + ay*by + az*bz + ax*bx
}

EHD Vec3V V3Cross(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
/*	if(0)
	{
		const sse::m128 r1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
		const sse::m128 r2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
		const sse::m128 l1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
		const sse::m128 l2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
		return sse::_mm_sub_ps(sse::_mm_mul_ps(l1, l2), sse::_mm_mul_ps(r1, r2));
	}
	else*/
	{
		const sse::m128 b0 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3,0,2,1));
		const sse::m128 a1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3,0,2,1));
		sse::m128 v = sse::_mm_mul_ps(a1, b);
		v = sse::_mm_sub_ps(sse::_mm_mul_ps(a, b0), v);
		sse::m128 res = sse::_mm_shuffle_ps(v, v, ENG_MM_SHUFFLE(3,0,2,1));
		ASSERT_ISVALIDVEC3V(res);
		return res;
	}
}

EHD VecCrossV V3PrepareCross(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	VecCrossV v;
	v.mR1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
	v.mL1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
	return v;
}

EHD Vec3V V3Cross(const VecCrossV& a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(b);
	const sse::m128 r2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
	const sse::m128 l2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
	return sse::_mm_sub_ps(sse::_mm_mul_ps(a.mL1, l2), sse::_mm_mul_ps(a.mR1, r2));
}

EHD Vec3V V3Cross(const Vec3V a, const VecCrossV& b)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 r2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
	const sse::m128 l2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
	return sse::_mm_sub_ps(sse::_mm_mul_ps(b.mR1, r2), sse::_mm_mul_ps(b.mL1, l2));
}

EHD Vec3V V3Cross(const VecCrossV& a, const VecCrossV& b)
{
	return sse::_mm_sub_ps(sse::_mm_mul_ps(a.mL1, b.mR1), sse::_mm_mul_ps(a.mR1, b.mL1));
}

EHD FloatV V3Length(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_sqrt_ps(V3Dot(a, a));
}

EHD FloatV V3LengthSq(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return V3Dot(a, a);
}

EHD Vec3V V3Normalize(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISFINITELENGTH(a);
	return V3ScaleInv(a, sse::_mm_sqrt_ps(V3Dot(a, a)));
}

EHD Vec3V V3NormalizeFast(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISFINITELENGTH(a);
	return V3Scale(a, sse::_mm_rsqrt_ps(V3Dot(a, a)));
}

EHD Vec3V V3NormalizeSafe(const Vec3V a, const Vec3V unsafeReturnValue)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 eps = FEps();
	const sse::m128 length = V3Length(a);
	const sse::m128 isGreaterThanZero = FIsGrtr(length, eps);
	return V3Sel(isGreaterThanZero, V3ScaleInv(a, length), unsafeReturnValue);
}

EHD Vec3V V3Sel(const BoolV c, const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(sse::_mm_or_ps(sse::_mm_andnot_ps(c, b), sse::_mm_and_ps(c, a)));
	return sse::_mm_or_ps(sse::_mm_andnot_ps(c, b), sse::_mm_and_ps(c, a));
}

EHD BoolV V3IsGrtr(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_cmpgt_ps(a, b);
}

EHD BoolV V3IsGrtrOrEq(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_cmpge_ps(a, b);
}

EHD BoolV V3IsEq(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_cmpeq_ps(a, b);
}

EHD Vec3V V3Max(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_max_ps(a, b);
}

EHD Vec3V V3Min(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return sse::_mm_min_ps(a, b);
}

EHD FloatV V3ExtractMax(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0));
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 1, 1, 1));
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2));
	return sse::_mm_max_ps(sse::_mm_max_ps(shuf1, shuf2), shuf3);
}

EHD FloatV V3ExtractMin(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0));
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 1, 1, 1));
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2));
	return sse::_mm_min_ps(sse::_mm_min_ps(shuf1, shuf2), shuf3);
}

//// if(a > 0.0f) return 1.0f; else if a == 0.f return 0.f, else return -1.f;
// EHD Vec3V V3MathSign(const Vec3V a)
//{
//	VECMATHAOS_ASSERT(isValidVec3V(a));
//
//	const sse::m128i ai = sse::_mm_cvtps_epi32(a);
//	const sse::m128i bi = sse::_mm_cvtps_epi32(V3Neg(a));
//	const sse::m128  aa = sse::_mm_cvtepi32_ps(sse::_mm_srai_epi32(ai, 31));
//	const sse::m128  bb = sse::_mm_cvtepi32_ps(sse::_mm_srai_epi32(bi, 31));
//	return sse::_mm_or_ps(aa, bb);
//}

// return (a >= 0.0f) ? 1.0f : -1.0f;
EHD Vec3V V3Sign(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 zero = V3Zero();
	const sse::m128 one = V3One();
	const sse::m128 none = V3Neg(one);
	return V3Sel(V3IsGrtrOrEq(a, zero), one, none);
}

EHD Vec3V V3Clamp(const Vec3V a, const Vec3V minV, const Vec3V maxV)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(minV);
	ASSERT_ISVALIDVEC3V(maxV);
	return V3Max(V3Min(a, maxV), minV);
}

EHD PxU32 V3AllGrtr(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return internalSimd::BAllTrue3_R(V4IsGrtr(a, b));
}

EHD PxU32 V3AllGrtrOrEq(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return internalSimd::BAllTrue3_R(V4IsGrtrOrEq(a, b));
}

EHD PxU32 V3AllEq(const Vec3V a, const Vec3V b)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(b);
	return internalSimd::BAllTrue3_R(V4IsEq(a, b));
}

EHD Vec3V V3Round(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	// return sse::_mm_round_ps(a, 0x0);
	const Vec3V half = V3Load(0.5f);
	const sse::m128 signBit = sse::_mm_cvtepi32_ps(sse::_mm_srli_epi32(sse::_mm_cvtps_epi32(a), 31));
	const Vec3V aRound = V3Sub(V3Add(a, half), signBit);
	sse::m128i tmp = sse::_mm_cvttps_epi32(aRound);
	return sse::_mm_cvtepi32_ps(tmp);
}

EHD Vec3V V3Sin(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);

	// Modulo the range of the given angles such that -XM_2PI <= Angles < XM_2PI
	const Vec4V recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const Vec4V twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const Vec3V tmp = V3Scale(a, recipTwoPi);
	const Vec3V b = V3Round(tmp);
	const Vec3V V1 = V3NegScaleSub(b, twoPi, a);

	// sin(V) ~= V - V^3 / 3! + V^5 / 5! - V^7 / 7! + V^9 / 9! - V^11 / 11! + V^13 / 13! -
	//           V^15 / 15! + V^17 / 17! - V^19 / 19! + V^21 / 21! - V^23 / 23! (for -PI <= V < PI)
	const Vec3V V2 = V3Mul(V1, V1);
	const Vec3V V3 = V3Mul(V2, V1);
	const Vec3V V5 = V3Mul(V3, V2);
	const Vec3V V7 = V3Mul(V5, V2);
	const Vec3V V9 = V3Mul(V7, V2);
	const Vec3V V11 = V3Mul(V9, V2);
	const Vec3V V13 = V3Mul(V11, V2);
	const Vec3V V15 = V3Mul(V13, V2);
	const Vec3V V17 = V3Mul(V15, V2);
	const Vec3V V19 = V3Mul(V17, V2);
	const Vec3V V21 = V3Mul(V19, V2);
	const Vec3V V23 = V3Mul(V21, V2);

	const Vec4V sinCoefficients0 = V4LoadA(ENG_G(g_PXSinCoefficients0).f);
	const Vec4V sinCoefficients1 = V4LoadA(ENG_G(g_PXSinCoefficients1).f);
	const Vec4V sinCoefficients2 = V4LoadA(ENG_G(g_PXSinCoefficients2).f);

	const FloatV S1 = V4GetY(sinCoefficients0);
	const FloatV S2 = V4GetZ(sinCoefficients0);
	const FloatV S3 = V4GetW(sinCoefficients0);
	const FloatV S4 = V4GetX(sinCoefficients1);
	const FloatV S5 = V4GetY(sinCoefficients1);
	const FloatV S6 = V4GetZ(sinCoefficients1);
	const FloatV S7 = V4GetW(sinCoefficients1);
	const FloatV S8 = V4GetX(sinCoefficients2);
	const FloatV S9 = V4GetY(sinCoefficients2);
	const FloatV S10 = V4GetZ(sinCoefficients2);
	const FloatV S11 = V4GetW(sinCoefficients2);

	Vec3V Result;
	Result = V3ScaleAdd(V3, S1, V1);
	Result = V3ScaleAdd(V5, S2, Result);
	Result = V3ScaleAdd(V7, S3, Result);
	Result = V3ScaleAdd(V9, S4, Result);
	Result = V3ScaleAdd(V11, S5, Result);
	Result = V3ScaleAdd(V13, S6, Result);
	Result = V3ScaleAdd(V15, S7, Result);
	Result = V3ScaleAdd(V17, S8, Result);
	Result = V3ScaleAdd(V19, S9, Result);
	Result = V3ScaleAdd(V21, S10, Result);
	Result = V3ScaleAdd(V23, S11, Result);

	ASSERT_ISVALIDVEC3V(Result);
	return Result;
}

EHD Vec3V V3Cos(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);

	// Modulo the range of the given angles such that -XM_2PI <= Angles < XM_2PI
	const Vec4V recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const Vec4V twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const Vec3V tmp = V3Scale(a, recipTwoPi);
	const Vec3V b = V3Round(tmp);
	const Vec3V V1 = V3NegScaleSub(b, twoPi, a);

	// cos(V) ~= 1 - V^2 / 2! + V^4 / 4! - V^6 / 6! + V^8 / 8! - V^10 / 10! + V^12 / 12! -
	//           V^14 / 14! + V^16 / 16! - V^18 / 18! + V^20 / 20! - V^22 / 22! (for -PI <= V < PI)
	const Vec3V V2 = V3Mul(V1, V1);
	const Vec3V V4 = V3Mul(V2, V2);
	const Vec3V V6 = V3Mul(V4, V2);
	const Vec3V V8 = V3Mul(V4, V4);
	const Vec3V V10 = V3Mul(V6, V4);
	const Vec3V V12 = V3Mul(V6, V6);
	const Vec3V V14 = V3Mul(V8, V6);
	const Vec3V V16 = V3Mul(V8, V8);
	const Vec3V V18 = V3Mul(V10, V8);
	const Vec3V V20 = V3Mul(V10, V10);
	const Vec3V V22 = V3Mul(V12, V10);

	const Vec4V cosCoefficients0 = V4LoadA(ENG_G(g_PXCosCoefficients0).f);
	const Vec4V cosCoefficients1 = V4LoadA(ENG_G(g_PXCosCoefficients1).f);
	const Vec4V cosCoefficients2 = V4LoadA(ENG_G(g_PXCosCoefficients2).f);

	const FloatV C1 = V4GetY(cosCoefficients0);
	const FloatV C2 = V4GetZ(cosCoefficients0);
	const FloatV C3 = V4GetW(cosCoefficients0);
	const FloatV C4 = V4GetX(cosCoefficients1);
	const FloatV C5 = V4GetY(cosCoefficients1);
	const FloatV C6 = V4GetZ(cosCoefficients1);
	const FloatV C7 = V4GetW(cosCoefficients1);
	const FloatV C8 = V4GetX(cosCoefficients2);
	const FloatV C9 = V4GetY(cosCoefficients2);
	const FloatV C10 = V4GetZ(cosCoefficients2);
	const FloatV C11 = V4GetW(cosCoefficients2);

	Vec3V Result;
	Result = V3ScaleAdd(V2, C1, V3One());
	Result = V3ScaleAdd(V4, C2, Result);
	Result = V3ScaleAdd(V6, C3, Result);
	Result = V3ScaleAdd(V8, C4, Result);
	Result = V3ScaleAdd(V10, C5, Result);
	Result = V3ScaleAdd(V12, C6, Result);
	Result = V3ScaleAdd(V14, C7, Result);
	Result = V3ScaleAdd(V16, C8, Result);
	Result = V3ScaleAdd(V18, C9, Result);
	Result = V3ScaleAdd(V20, C10, Result);
	Result = V3ScaleAdd(V22, C11, Result);

	ASSERT_ISVALIDVEC3V(Result); 
	return Result;
}

EHD Vec3V V3PermYZZ(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 2, 2, 1));
}

EHD Vec3V V3PermXYX(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 1, 0));
}

EHD Vec3V V3PermYZX(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1));
}

EHD Vec3V V3PermZXY(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 0, 2));
}

EHD Vec3V V3PermZZY(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 2, 2));
}

EHD Vec3V V3PermYXX(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 0, 1));
}

EHD Vec3V V3Perm_Zero_1Z_0Y(const Vec3V v0, const Vec3V v1)
{
	ASSERT_ISVALIDVEC3V(v0);
	ASSERT_ISVALIDVEC3V(v1);
	return sse::_mm_shuffle_ps(v1, v0, ENG_MM_SHUFFLE(3, 1, 2, 3));
}

EHD Vec3V V3Perm_0Z_Zero_1X(const Vec3V v0, const Vec3V v1)
{
	ASSERT_ISVALIDVEC3V(v0);
	ASSERT_ISVALIDVEC3V(v1);
	return sse::_mm_shuffle_ps(v0, v1, ENG_MM_SHUFFLE(3, 0, 3, 2));
}

EHD Vec3V V3Perm_1Y_0X_Zero(const Vec3V v0, const Vec3V v1)
{
	ASSERT_ISVALIDVEC3V(v0);
	ASSERT_ISVALIDVEC3V(v1);
	// There must be a better way to do this.
	Vec3V v2 = V3Zero();
	const FloatV y1 = V3GetY(v1);
	const FloatV x0 = V3GetX(v0);
	v2 = V3SetX(v2, y1);
	return V3SetY(v2, x0);
}

EHD FloatV V3SumElems(const Vec3V a)
{
	ASSERT_ISVALIDVEC3V(a);
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 0, 0, 0)); // z,y,x,w
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 1, 1, 1)); // y,x,w,z
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2)); // x,w,z,y
	return sse::_mm_add_ps(sse::_mm_add_ps(shuf1, shuf2), shuf3);
}

EHD PxU32 V3OutOfBounds(const Vec3V a, const Vec3V min, const Vec3V max)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(min);
	ASSERT_ISVALIDVEC3V(max);
	const BoolV c = BOr(V3IsGrtr(a, max), V3IsGrtr(min, a));
	return PxU32(!BAllEqFFFF(c));
}

EHD PxU32 V3InBounds(const Vec3V a, const Vec3V min, const Vec3V max)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(min);
	ASSERT_ISVALIDVEC3V(max);
	const BoolV c = BAnd(V3IsGrtrOrEq(a, min), V3IsGrtrOrEq(max, a));
	return BAllEqTTTT(c);
}

EHD PxU32 V3OutOfBounds(const Vec3V a, const Vec3V bounds)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(bounds);
	return V3OutOfBounds(a, V3Neg(bounds), bounds);
}

EHD PxU32 V3InBounds(const Vec3V a, const Vec3V bounds)
{
	ASSERT_ISVALIDVEC3V(a);
	ASSERT_ISVALIDVEC3V(bounds);
	return V3InBounds(a, V3Neg(bounds), bounds);
}

EHD void V3Transpose(Vec3V& col0, Vec3V& col1, Vec3V& col2)
{
	ASSERT_ISVALIDVEC3V(col0);
	ASSERT_ISVALIDVEC3V(col1);
	ASSERT_ISVALIDVEC3V(col2);
	const Vec3V col3 = sse::_mm_setzero_ps();
	const Vec3V tmp0 = sse::_mm_unpacklo_ps(col0, col1);
	const Vec3V tmp2 = sse::_mm_unpacklo_ps(col2, col3);
	const Vec3V tmp1 = sse::_mm_unpackhi_ps(col0, col1);
	const Vec3V tmp3 = sse::_mm_unpackhi_ps(col2, col3);
	col0 = sse::_mm_movelh_ps(tmp0, tmp2);
	col1 = sse::_mm_movehl_ps(tmp2, tmp0);
	col2 = sse::_mm_movelh_ps(tmp1, tmp3);
}

//////////////////////////////////
// VEC4V
//////////////////////////////////

EHD Vec4V V4Splat(const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	// return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(0,0,0,0));
	return f;
}

EHD Vec4V V4Merge(const FloatV* const floatVArray)
{
	ASSERT_ISVALIDFLOATV(floatVArray[0]);
	ASSERT_ISVALIDFLOATV(floatVArray[1]);
	ASSERT_ISVALIDFLOATV(floatVArray[2]);
	ASSERT_ISVALIDFLOATV(floatVArray[3]);
	const sse::m128 xw = sse::_mm_move_ss(floatVArray[1], floatVArray[0]); // y, y, y, x
	const sse::m128 yz = sse::_mm_move_ss(floatVArray[2], floatVArray[3]); // z, z, z, w
	return sse::_mm_shuffle_ps(xw, yz, ENG_MM_SHUFFLE(0, 2, 1, 0));
}

EHD Vec4V V4Merge(const FloatVArg x, const FloatVArg y, const FloatVArg z, const FloatVArg w)
{
	ASSERT_ISVALIDFLOATV(x);
	ASSERT_ISVALIDFLOATV(y);
	ASSERT_ISVALIDFLOATV(z);
	ASSERT_ISVALIDFLOATV(w);
	const sse::m128 xw = sse::_mm_move_ss(y, x); // y, y, y, x
	const sse::m128 yz = sse::_mm_move_ss(z, w); // z, z, z, w
	return sse::_mm_shuffle_ps(xw, yz, ENG_MM_SHUFFLE(0, 2, 1, 0));
}

EHD Vec4V V4MergeW(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w)
{
	const Vec4V xz = sse::_mm_unpackhi_ps(x, z);
	const Vec4V yw = sse::_mm_unpackhi_ps(y, w);
	return sse::_mm_unpackhi_ps(xz, yw);
}

EHD Vec4V V4MergeZ(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w)
{
	const Vec4V xz = sse::_mm_unpackhi_ps(x, z);
	const Vec4V yw = sse::_mm_unpackhi_ps(y, w);
	return sse::_mm_unpacklo_ps(xz, yw);
}

EHD Vec4V V4MergeY(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w)
{
	const Vec4V xz = sse::_mm_unpacklo_ps(x, z);
	const Vec4V yw = sse::_mm_unpacklo_ps(y, w);
	return sse::_mm_unpackhi_ps(xz, yw);
}

EHD Vec4V V4MergeX(const Vec4VArg x, const Vec4VArg y, const Vec4VArg z, const Vec4VArg w)
{
	const Vec4V xz = sse::_mm_unpacklo_ps(x, z);
	const Vec4V yw = sse::_mm_unpacklo_ps(y, w);
	return sse::_mm_unpacklo_ps(xz, yw);
}

EHD Vec4V V4UnpackXY(const Vec4VArg a, const Vec4VArg b)
{
	return sse::_mm_unpacklo_ps(a, b);
}

EHD Vec4V V4UnpackZW(const Vec4VArg a, const Vec4VArg b)
{
	return sse::_mm_unpackhi_ps(a, b);
}

EHD Vec4V V4PermYXWZ(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 3, 0, 1));
}

EHD Vec4V V4PermXZXZ(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 0, 2, 0));
}

EHD Vec4V V4PermYWYW(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 3, 1));
}

EHD Vec4V V4PermYZXW(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1));
}

EHD Vec4V V4PermZWXY(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 0, 3, 2));
}

template <PxU8 x, PxU8 y, PxU8 z, PxU8 w>
EHD Vec4V V4Perm(const Vec4V a)
{
	return sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(w, z, y, x));
}

EHD FloatV V4GetW(const Vec4V f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(3, 3, 3, 3));
}

EHD FloatV V4GetX(const Vec4V f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(0, 0, 0, 0));
}

EHD FloatV V4GetY(const Vec4V f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(1, 1, 1, 1));
}

EHD FloatV V4GetZ(const Vec4V f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(2, 2, 2, 2));
}

EHD Vec4V V4SetW(const Vec4V v, const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BTTTF(), v, f);
}

EHD Vec4V V4SetX(const Vec4V v, const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BFTTT(), v, f);
}

EHD Vec4V V4SetY(const Vec4V v, const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BTFTT(), v, f);
}

EHD Vec4V V4SetZ(const Vec4V v, const FloatV f)
{
	ASSERT_ISVALIDFLOATV(f);
	return V4Sel(BTTFT(), v, f);
}

EHD Vec4V V4Zero()
{
	return sse::_mm_setzero_ps();
}

EHD Vec4V V4One()
{
	return V4Load(1.0f);
}

EHD Vec4V V4Eps()
{
	return V4Load(PX_EPS_REAL);
}

EHD Vec4V V4Neg(const Vec4V f)
{
	return sse::_mm_sub_ps(sse::_mm_setzero_ps(), f);
}

EHD Vec4V V4Add(const Vec4V a, const Vec4V b)
{
	return sse::_mm_add_ps(a, b);
}

EHD Vec4V V4Sub(const Vec4V a, const Vec4V b)
{
	return sse::_mm_sub_ps(a, b);
}

EHD Vec4V V4Scale(const Vec4V a, const FloatV b)
{
	return sse::_mm_mul_ps(a, b);
}

EHD Vec4V V4Mul(const Vec4V a, const Vec4V b)
{
	return sse::_mm_mul_ps(a, b);
}

EHD Vec4V V4ScaleInv(const Vec4V a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_div_ps(a, b);
}

EHD Vec4V V4Div(const Vec4V a, const Vec4V b)
{
	return sse::_mm_div_ps(a, b);
}

EHD Vec4V V4ScaleInvFast(const Vec4V a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(b);
	return sse::_mm_mul_ps(a, sse::_mm_rcp_ps(b));
}

EHD Vec4V V4DivFast(const Vec4V a, const Vec4V b)
{
	return sse::_mm_mul_ps(a, sse::_mm_rcp_ps(b));
}

EHD Vec4V V4Recip(const Vec4V a)
{
	return sse::_mm_div_ps(V4One(), a);
}

EHD Vec4V V4RecipFast(const Vec4V a)
{
	return sse::_mm_rcp_ps(a);
}

EHD Vec4V V4Rsqrt(const Vec4V a)
{
	return sse::_mm_div_ps(V4One(), sse::_mm_sqrt_ps(a));
}

EHD Vec4V V4RsqrtFast(const Vec4V a)
{
	return sse::_mm_rsqrt_ps(a);
}

EHD Vec4V V4Sqrt(const Vec4V a)
{
	return sse::_mm_sqrt_ps(a);
}

EHD Vec4V V4ScaleAdd(const Vec4V a, const FloatV b, const Vec4V c)
{
	ASSERT_ISVALIDFLOATV(b);
	return V4Add(V4Scale(a, b), c);
}

EHD Vec4V V4NegScaleSub(const Vec4V a, const FloatV b, const Vec4V c)
{
	ASSERT_ISVALIDFLOATV(b);
	return V4Sub(c, V4Scale(a, b));
}

EHD Vec4V V4MulAdd(const Vec4V a, const Vec4V b, const Vec4V c)
{
	return V4Add(V4Mul(a, b), c);
}

EHD Vec4V V4NegMulSub(const Vec4V a, const Vec4V b, const Vec4V c)
{
	return V4Sub(c, V4Mul(a, b));
}

EHD Vec4V V4Abs(const Vec4V a)
{
	return V4Max(a, V4Neg(a));
}

EHD FloatV V4SumElements(const Vec4V a)
{
	const Vec4V xy = V4UnpackXY(a, a);	// x,x,y,y
	const Vec4V zw = V4UnpackZW(a, a);	// z,z,w,w
	const Vec4V xz_yw = V4Add(xy, zw);	// x+z,x+z,y+w,y+w
	const FloatV xz = V4GetX(xz_yw);	// x+z
	const FloatV yw = V4GetZ(xz_yw);	// y+w
	return FAdd(xz, yw);				// sum
}

EHD FloatV V4Dot(const Vec4V a, const Vec4V b)
{
	//const sse::m128 dot1 = sse::_mm_mul_ps(a, b);                                     // x,y,z,w
	//const sse::m128 shuf1 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(2, 1, 0, 3)); // w,x,y,z
	//const sse::m128 shuf2 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(1, 0, 3, 2)); // z,w,x,y
	//const sse::m128 shuf3 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(0, 3, 2, 1)); // y,z,w,x
	//return sse::_mm_add_ps(sse::_mm_add_ps(shuf2, shuf3), sse::_mm_add_ps(dot1, shuf1));

	// PT: this version has two less instructions but we should check its accuracy
	// aw*bw | az*bz | ay*by | ax*bx
	const sse::m128 t0 = sse::_mm_mul_ps(a, b);
	// ay*by | ax*bx | aw*bw | az*bz
	const sse::m128 t1 = sse::_mm_shuffle_ps(t0, t0, ENG_MM_SHUFFLE(1, 0, 3, 2));
	// ay*by + aw*bw | ax*bx + az*bz | aw*bw + ay*by | az*bz + ax*bx
	const sse::m128 t2 = sse::_mm_add_ps(t0, t1);
	// ax*bx + az*bz | ay*by + aw*bw | az*bz + ax*bx | aw*bw + ay*by
	const sse::m128 t3 = sse::_mm_shuffle_ps(t2, t2, ENG_MM_SHUFFLE(2, 3, 0, 1));
	// ax*bx + az*bz + ay*by + aw*bw
	return sse::_mm_add_ps(t3, t2);
	// ay*by + aw*bw + ax*bx + az*bz
	// az*bz + ax*bx + aw*bw + ay*by
	// aw*bw + ay*by + az*bz + ax*bx
}

EHD FloatV V4Dot3(const Vec4V a, const Vec4V b)
{
	const sse::m128 dot1 = sse::_mm_mul_ps(a, b);                                     // aw*bw | az*bz | ay*by | ax*bx
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(0, 0, 0, 0)); // ax*bx | ax*bx | ax*bx | ax*bx
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(1, 1, 1, 1)); // ay*by | ay*by | ay*by | ay*by
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(dot1, dot1, ENG_MM_SHUFFLE(2, 2, 2, 2)); // az*bz | az*bz | az*bz | az*bz
	return sse::_mm_add_ps(sse::_mm_add_ps(shuf1, shuf2), shuf3);                       // ax*bx + ay*by + az*bz in each component
}

EHD Vec4V V4Cross(const Vec4V a, const Vec4V b)
{
/*	if(0)
	{
		const sse::m128 r1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
		const sse::m128 r2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
		const sse::m128 l1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3, 0, 2, 1)); // y,z,x,w
		const sse::m128 l2 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3, 1, 0, 2)); // z,x,y,w
		return sse::_mm_sub_ps(sse::_mm_mul_ps(l1, l2), sse::_mm_mul_ps(r1, r2));
	}
	else*/
	{
		const sse::m128 b0 = sse::_mm_shuffle_ps(b, b, ENG_MM_SHUFFLE(3,0,2,1));
		const sse::m128 a1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(3,0,2,1));
		sse::m128 v = sse::_mm_mul_ps(a1, b);
		v = sse::_mm_sub_ps(sse::_mm_mul_ps(a, b0), v);
		return sse::_mm_shuffle_ps(v, v, ENG_MM_SHUFFLE(3,0,2,1));
	}
}

EHD FloatV V4Length(const Vec4V a)
{
	return sse::_mm_sqrt_ps(V4Dot(a, a));
}

EHD FloatV V4LengthSq(const Vec4V a)
{
	return V4Dot(a, a);
}

EHD Vec4V V4Normalize(const Vec4V a)
{
	ASSERT_ISFINITELENGTH(a);
	return V4ScaleInv(a, sse::_mm_sqrt_ps(V4Dot(a, a)));
}

EHD Vec4V V4NormalizeFast(const Vec4V a)
{
	ASSERT_ISFINITELENGTH(a);
	return V4ScaleInvFast(a, sse::_mm_sqrt_ps(V4Dot(a, a)));
}

EHD Vec4V V4NormalizeSafe(const Vec4V a, const Vec4V unsafeReturnValue)
{
	const sse::m128 eps = V3Eps();
	const sse::m128 length = V4Length(a);
	const sse::m128 isGreaterThanZero = V4IsGrtr(length, eps);
	return V4Sel(isGreaterThanZero, V4ScaleInv(a, length), unsafeReturnValue);
}

EHD Vec4V V4Sel(const BoolV c, const Vec4V a, const Vec4V b)
{
	return sse::_mm_or_ps(sse::_mm_andnot_ps(c, b), sse::_mm_and_ps(c, a));
}

EHD BoolV V4IsGrtr(const Vec4V a, const Vec4V b)
{
	return sse::_mm_cmpgt_ps(a, b);
}

EHD BoolV V4IsGrtrOrEq(const Vec4V a, const Vec4V b)
{
	return sse::_mm_cmpge_ps(a, b);
}

EHD BoolV V4IsEq(const Vec4V a, const Vec4V b)
{
	return sse::_mm_cmpeq_ps(a, b);
}

EHD Vec4V V4Max(const Vec4V a, const Vec4V b)
{
	return sse::_mm_max_ps(a, b);
}

EHD Vec4V V4Min(const Vec4V a, const Vec4V b)
{
	return sse::_mm_min_ps(a, b);
}

EHD FloatV V4ExtractMax(const Vec4V a)
{
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 1, 0, 3));
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 0, 3, 2));
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 3, 2, 1));

	return sse::_mm_max_ps(sse::_mm_max_ps(a, shuf1), sse::_mm_max_ps(shuf2, shuf3));
}

EHD FloatV V4ExtractMin(const Vec4V a)
{
	const sse::m128 shuf1 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 1, 0, 3));
	const sse::m128 shuf2 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(1, 0, 3, 2));
	const sse::m128 shuf3 = sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 3, 2, 1));

	return sse::_mm_min_ps(sse::_mm_min_ps(a, shuf1), sse::_mm_min_ps(shuf2, shuf3));
}

EHD Vec4V V4Clamp(const Vec4V a, const Vec4V minV, const Vec4V maxV)
{
	return V4Max(V4Min(a, maxV), minV);
}

EHD PxU32 V4AllGrtr(const Vec4V a, const Vec4V b)
{
	return internalSimd::BAllTrue4_R(V4IsGrtr(a, b));
}

EHD PxU32 V4AllGrtrOrEq(const Vec4V a, const Vec4V b)
{
	return internalSimd::BAllTrue4_R(V4IsGrtrOrEq(a, b));
}

EHD PxU32 V4AllGrtrOrEq3(const Vec4V a, const Vec4V b)
{
	return internalSimd::BAllTrue3_R(V4IsGrtrOrEq(a, b));
}

EHD PxU32 V4AllEq(const Vec4V a, const Vec4V b)
{
	return internalSimd::BAllTrue4_R(V4IsEq(a, b));
}

EHD PxU32 V4AnyGrtr3(const Vec4V a, const Vec4V b)
{
	return internalSimd::BAnyTrue3_R(V4IsGrtr(a, b));
}

EHD Vec4V V4Round(const Vec4V a)
{
	// return sse::_mm_round_ps(a, 0x0);
	const Vec4V half = V4Load(0.5f);
	const sse::m128 signBit = sse::_mm_cvtepi32_ps(sse::_mm_srli_epi32(sse::_mm_cvtps_epi32(a), 31));
	const Vec4V aRound = V4Sub(V4Add(a, half), signBit);
	const sse::m128i tmp = sse::_mm_cvttps_epi32(aRound);
	return sse::_mm_cvtepi32_ps(tmp);
}

EHD Vec4V V4Sin(const Vec4V a)
{
	// PT: TODO: these should be FLoads
	const Vec4V recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const Vec4V twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const Vec4V tmp = V4Mul(a, recipTwoPi);
	const Vec4V b = V4Round(tmp);
	const Vec4V V1 = V4NegMulSub(twoPi, b, a);

	// sin(V) ~= V - V^3 / 3! + V^5 / 5! - V^7 / 7! + V^9 / 9! - V^11 / 11! + V^13 / 13! -
	//           V^15 / 15! + V^17 / 17! - V^19 / 19! + V^21 / 21! - V^23 / 23! (for -PI <= V < PI)
	const Vec4V V2 = V4Mul(V1, V1);
	const Vec4V V3 = V4Mul(V2, V1);
	const Vec4V V5 = V4Mul(V3, V2);
	const Vec4V V7 = V4Mul(V5, V2);
	const Vec4V V9 = V4Mul(V7, V2);
	const Vec4V V11 = V4Mul(V9, V2);
	const Vec4V V13 = V4Mul(V11, V2);
	const Vec4V V15 = V4Mul(V13, V2);
	const Vec4V V17 = V4Mul(V15, V2);
	const Vec4V V19 = V4Mul(V17, V2);
	const Vec4V V21 = V4Mul(V19, V2);
	const Vec4V V23 = V4Mul(V21, V2);

	const Vec4V sinCoefficients0 = V4LoadA(ENG_G(g_PXSinCoefficients0).f);
	const Vec4V sinCoefficients1 = V4LoadA(ENG_G(g_PXSinCoefficients1).f);
	const Vec4V sinCoefficients2 = V4LoadA(ENG_G(g_PXSinCoefficients2).f);

	const FloatV S1 = V4GetY(sinCoefficients0);
	const FloatV S2 = V4GetZ(sinCoefficients0);
	const FloatV S3 = V4GetW(sinCoefficients0);
	const FloatV S4 = V4GetX(sinCoefficients1);
	const FloatV S5 = V4GetY(sinCoefficients1);
	const FloatV S6 = V4GetZ(sinCoefficients1);
	const FloatV S7 = V4GetW(sinCoefficients1);
	const FloatV S8 = V4GetX(sinCoefficients2);
	const FloatV S9 = V4GetY(sinCoefficients2);
	const FloatV S10 = V4GetZ(sinCoefficients2);
	const FloatV S11 = V4GetW(sinCoefficients2);

	Vec4V Result;
	Result = V4MulAdd(S1, V3, V1);
	Result = V4MulAdd(S2, V5, Result);
	Result = V4MulAdd(S3, V7, Result);
	Result = V4MulAdd(S4, V9, Result);
	Result = V4MulAdd(S5, V11, Result);
	Result = V4MulAdd(S6, V13, Result);
	Result = V4MulAdd(S7, V15, Result);
	Result = V4MulAdd(S8, V17, Result);
	Result = V4MulAdd(S9, V19, Result);
	Result = V4MulAdd(S10, V21, Result);
	Result = V4MulAdd(S11, V23, Result);

	return Result;
}

EHD Vec4V V4Cos(const Vec4V a)
{
	// PT: TODO: these should be FLoads
	const Vec4V recipTwoPi = V4LoadA(ENG_G(g_PXReciprocalTwoPi).f);
	const FloatV twoPi = V4LoadA(ENG_G(g_PXTwoPi).f);
	const Vec4V tmp = V4Mul(a, recipTwoPi);
	const Vec4V b = V4Round(tmp);
	const Vec4V V1 = V4NegMulSub(twoPi, b, a);

	// cos(V) ~= 1 - V^2 / 2! + V^4 / 4! - V^6 / 6! + V^8 / 8! - V^10 / 10! + V^12 / 12! -
	//           V^14 / 14! + V^16 / 16! - V^18 / 18! + V^20 / 20! - V^22 / 22! (for -PI <= V < PI)
	const Vec4V V2 = V4Mul(V1, V1);
	const Vec4V V4 = V4Mul(V2, V2);
	const Vec4V V6 = V4Mul(V4, V2);
	const Vec4V V8 = V4Mul(V4, V4);
	const Vec4V V10 = V4Mul(V6, V4);
	const Vec4V V12 = V4Mul(V6, V6);
	const Vec4V V14 = V4Mul(V8, V6);
	const Vec4V V16 = V4Mul(V8, V8);
	const Vec4V V18 = V4Mul(V10, V8);
	const Vec4V V20 = V4Mul(V10, V10);
	const Vec4V V22 = V4Mul(V12, V10);

	const Vec4V cosCoefficients0 = V4LoadA(ENG_G(g_PXCosCoefficients0).f);
	const Vec4V cosCoefficients1 = V4LoadA(ENG_G(g_PXCosCoefficients1).f);
	const Vec4V cosCoefficients2 = V4LoadA(ENG_G(g_PXCosCoefficients2).f);

	const FloatV C1 = V4GetY(cosCoefficients0);
	const FloatV C2 = V4GetZ(cosCoefficients0);
	const FloatV C3 = V4GetW(cosCoefficients0);
	const FloatV C4 = V4GetX(cosCoefficients1);
	const FloatV C5 = V4GetY(cosCoefficients1);
	const FloatV C6 = V4GetZ(cosCoefficients1);
	const FloatV C7 = V4GetW(cosCoefficients1);
	const FloatV C8 = V4GetX(cosCoefficients2);
	const FloatV C9 = V4GetY(cosCoefficients2);
	const FloatV C10 = V4GetZ(cosCoefficients2);
	const FloatV C11 = V4GetW(cosCoefficients2);

	Vec4V Result;
	Result = V4MulAdd(C1, V2, V4One());
	Result = V4MulAdd(C2, V4, Result);
	Result = V4MulAdd(C3, V6, Result);
	Result = V4MulAdd(C4, V8, Result);
	Result = V4MulAdd(C5, V10, Result);
	Result = V4MulAdd(C6, V12, Result);
	Result = V4MulAdd(C7, V14, Result);
	Result = V4MulAdd(C8, V16, Result);
	Result = V4MulAdd(C9, V18, Result);
	Result = V4MulAdd(C10, V20, Result);
	Result = V4MulAdd(C11, V22, Result);

	return Result;
}

EHD void V4Transpose(Vec4V& col0, Vec4V& col1, Vec4V& col2, Vec4V& col3)
{
	const Vec4V tmp0 = sse::_mm_unpacklo_ps(col0, col1);
	const Vec4V tmp2 = sse::_mm_unpacklo_ps(col2, col3);
	const Vec4V tmp1 = sse::_mm_unpackhi_ps(col0, col1);
	const Vec4V tmp3 = sse::_mm_unpackhi_ps(col2, col3);
	col0 = sse::_mm_movelh_ps(tmp0, tmp2);
	col1 = sse::_mm_movehl_ps(tmp2, tmp0);
	col2 = sse::_mm_movelh_ps(tmp1, tmp3);
	col3 = sse::_mm_movehl_ps(tmp3, tmp1);
}

EHD BoolV V4IsEqU32(const VecU32V a, const VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_cmpeq_epi32(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
}

//////////////////////////////////
// BoolV
//////////////////////////////////

EHD BoolV BFFFF()
{
	return sse::_mm_setzero_ps();
}

EHD BoolV BFFFT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0,0,0xFFFFFFFF};
	const sse::m128 ffft=sse::_mm_load_ps((float*)&f);
	return ffft;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, 0, 0, 0));
}

EHD BoolV BFFTF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0,0xFFFFFFFF,0};
	const sse::m128 fftf=sse::_mm_load_ps((float*)&f);
	return fftf;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, -1, 0, 0));
}

EHD BoolV BFFTT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0,0xFFFFFFFF,0xFFFFFFFF};
	const sse::m128 fftt=sse::_mm_load_ps((float*)&f);
	return fftt;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, -1, 0, 0));
}

EHD BoolV BFTFF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0xFFFFFFFF,0,0};
	const sse::m128 ftff=sse::_mm_load_ps((float*)&f);
	return ftff;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, 0, -1, 0));
}

EHD BoolV BFTFT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0xFFFFFFFF,0,0xFFFFFFFF};
	const sse::m128 ftft=sse::_mm_load_ps((float*)&f);
	return ftft;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, 0, -1, 0));
}

EHD BoolV BFTTF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0xFFFFFFFF,0xFFFFFFFF,0};
	const sse::m128 fttf=sse::_mm_load_ps((float*)&f);
	return fttf;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, -1, -1, 0));
}

EHD BoolV BFTTT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
	const sse::m128 fttt=sse::_mm_load_ps((float*)&f);
	return fttt;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, -1, -1, 0));
}

EHD BoolV BTFFF()
{
	// const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0,0,0};
	// const sse::m128 tfff=sse::_mm_load_ps((float*)&f);
	// return tfff;
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, 0, 0, -1));
}

EHD BoolV BTFFT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0,0,0xFFFFFFFF};
	const sse::m128 tfft=sse::_mm_load_ps((float*)&f);
	return tfft;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, 0, 0, -1));
}

EHD BoolV BTFTF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0,0xFFFFFFFF,0};
	const sse::m128 tftf=sse::_mm_load_ps((float*)&f);
	return tftf;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, -1, 0, -1));
}

EHD BoolV BTFTT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0,0xFFFFFFFF,0xFFFFFFFF};
	const sse::m128 tftt=sse::_mm_load_ps((float*)&f);
	return tftt;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, -1, 0, -1));
}

EHD BoolV BTTFF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0xFFFFFFFF,0,0};
	const sse::m128 ttff=sse::_mm_load_ps((float*)&f);
	return ttff;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, 0, -1, -1));
}

EHD BoolV BTTFT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0xFFFFFFFF,0,0xFFFFFFFF};
	const sse::m128 ttft=sse::_mm_load_ps((float*)&f);
	return ttft;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, 0, -1, -1));
}

EHD BoolV BTTTF()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0};
	const sse::m128 tttf=sse::_mm_load_ps((float*)&f);
	return tttf;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, -1, -1, -1));
}

EHD BoolV BTTTT()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
	const sse::m128 tttt=sse::_mm_load_ps((float*)&f);
	return tttt;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, -1, -1, -1));
}

EHD BoolV BXMask()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0xFFFFFFFF,0,0,0};
	const sse::m128 tfff=sse::_mm_load_ps((float*)&f);
	return tfff;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, 0, 0, -1));
}

EHD BoolV BYMask()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0xFFFFFFFF,0,0};
	const sse::m128 ftff=sse::_mm_load_ps((float*)&f);
	return ftff;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, 0, -1, 0));
}

EHD BoolV BZMask()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0,0xFFFFFFFF,0};
	const sse::m128 fftf=sse::_mm_load_ps((float*)&f);
	return fftf;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(0, -1, 0, 0));
}

EHD BoolV BWMask()
{
	/*const PX_ALIGN(16, PxU32 f[4])={0,0,0,0xFFFFFFFF};
	const sse::m128 ffft=sse::_mm_load_ps((float*)&f);
	return ffft;*/
	return internalSimd::m128_I2F(sse::_mm_set_epi32(-1, 0, 0, 0));
}

EHD BoolV BGetX(const BoolV f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(0, 0, 0, 0));
}

EHD BoolV BGetY(const BoolV f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(1, 1, 1, 1));
}

EHD BoolV BGetZ(const BoolV f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(2, 2, 2, 2));
}

EHD BoolV BGetW(const BoolV f)
{
	return sse::_mm_shuffle_ps(f, f, ENG_MM_SHUFFLE(3, 3, 3, 3));
}

EHD BoolV BSetX(const BoolV v, const BoolV f)
{
	return V4Sel(BFTTT(), v, f);
}

EHD BoolV BSetY(const BoolV v, const BoolV f)
{
	return V4Sel(BTFTT(), v, f);
}

EHD BoolV BSetZ(const BoolV v, const BoolV f)
{
	return V4Sel(BTTFT(), v, f);
}

EHD BoolV BSetW(const BoolV v, const BoolV f)
{
	return V4Sel(BTTTF(), v, f);
}

EHD BoolV BAnd(const BoolV a, const BoolV b)
{
	return sse::_mm_and_ps(a, b);
}

EHD BoolV BNot(const BoolV a)
{
	const BoolV bAllTrue(BTTTT());
	return sse::_mm_xor_ps(a, bAllTrue);
}

EHD BoolV BAndNot(const BoolV a, const BoolV b)
{
	return sse::_mm_andnot_ps(b, a);
}

EHD BoolV BOr(const BoolV a, const BoolV b)
{
	return sse::_mm_or_ps(a, b);
}

EHD BoolV BAllTrue4(const BoolV a)
{
	const BoolV bTmp = sse::_mm_and_ps(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 1, 0, 1)), sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 3, 2, 3)));
	return sse::_mm_and_ps(sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(0, 0, 0, 0)),
	                  sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(1, 1, 1, 1)));
}

EHD BoolV BAnyTrue4(const BoolV a)
{
	const BoolV bTmp = sse::_mm_or_ps(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 1, 0, 1)), sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 3, 2, 3)));
	return sse::_mm_or_ps(sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(0, 0, 0, 0)),
	                 sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(1, 1, 1, 1)));
}

EHD BoolV BAllTrue3(const BoolV a)
{
	const BoolV bTmp = sse::_mm_and_ps(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 1, 0, 1)), sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2)));
	return sse::_mm_and_ps(sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(0, 0, 0, 0)),
	                  sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(1, 1, 1, 1)));
}

EHD BoolV BAnyTrue3(const BoolV a)
{
	const BoolV bTmp = sse::_mm_or_ps(sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(0, 1, 0, 1)), sse::_mm_shuffle_ps(a, a, ENG_MM_SHUFFLE(2, 2, 2, 2)));
	return sse::_mm_or_ps(sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(0, 0, 0, 0)),
	                 sse::_mm_shuffle_ps(bTmp, bTmp, ENG_MM_SHUFFLE(1, 1, 1, 1)));
}

EHD PxU32 BAllEq(const BoolV a, const BoolV b)
{
	const BoolV bTest = internalSimd::m128_I2F(sse::_mm_cmpeq_epi32(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
	return internalSimd::BAllTrue4_R(bTest);
}

EHD PxU32 BAllEqTTTT(const BoolV a)
{
	return PxU32(sse::_mm_movemask_ps(a)==15);
}

EHD PxU32 BAllEqFFFF(const BoolV a)
{
	return PxU32(sse::_mm_movemask_ps(a)==0);
}

EHD PxU32 BGetBitMask(const BoolV a)
{
	return PxU32(sse::_mm_movemask_ps(a));
}

//////////////////////////////////
// MAT33V
//////////////////////////////////

EHD Mat33V M33Identity()
{
	return Mat33V(V3UnitX(), V3UnitY(), V3UnitZ());
}

EHD Vec3V M33MulV3(const Mat33V& a, const Vec3V b)
{
	const FloatV x = V3GetX(b);
	const FloatV y = V3GetY(b);
	const FloatV z = V3GetZ(b);
	const Vec3V v0 = V3Scale(a.col0, x);
	const Vec3V v1 = V3Scale(a.col1, y);
	const Vec3V v2 = V3Scale(a.col2, z);
	const Vec3V v0PlusV1 = V3Add(v0, v1);
	return V3Add(v0PlusV1, v2);
}

EHD Vec3V M33MulV3AddV3(const Mat33V& A, const Vec3V b, const Vec3V c)
{
	const FloatV x = V3GetX(b);
	const FloatV y = V3GetY(b);
	const FloatV z = V3GetZ(b);
	Vec3V result = V3ScaleAdd(A.col0, x, c);
	result = V3ScaleAdd(A.col1, y, result);
	return V3ScaleAdd(A.col2, z, result);
}

EHD Mat33V M33MulM33(const Mat33V& a, const Mat33V& b)
{
	return Mat33V(M33MulV3(a, b.col0), M33MulV3(a, b.col1), M33MulV3(a, b.col2));
}

EHD Mat33V M33Add(const Mat33V& a, const Mat33V& b)
{
	return Mat33V(V3Add(a.col0, b.col0), V3Add(a.col1, b.col1), V3Add(a.col2, b.col2));
}

EHD Mat33V M33Scale(const Mat33V& a, const FloatV& b)
{
	return Mat33V(V3Scale(a.col0, b), V3Scale(a.col1, b), V3Scale(a.col2, b));
}

EHD Mat33V M33Sub(const Mat33V& a, const Mat33V& b)
{
	return Mat33V(V3Sub(a.col0, b.col0), V3Sub(a.col1, b.col1), V3Sub(a.col2, b.col2));
}

EHD Mat33V M33Neg(const Mat33V& a)
{
	return Mat33V(V3Neg(a.col0), V3Neg(a.col1), V3Neg(a.col2));
}

EHD Mat33V M33Abs(const Mat33V& a)
{
	return Mat33V(V3Abs(a.col0), V3Abs(a.col1), V3Abs(a.col2));
}

EHD Mat33V M33Inverse(const Mat33V& a)
{
	const BoolV tfft = BTFFT();
	const BoolV tttf = BTTTF();
	const FloatV zero = FZero();
	const Vec3V cross01 = V3Cross(a.col0, a.col1);
	const Vec3V cross12 = V3Cross(a.col1, a.col2);
	const Vec3V cross20 = V3Cross(a.col2, a.col0);
	const FloatV dot = V3Dot(cross01, a.col2);
	const FloatV invDet = sse::_mm_rcp_ps(dot);
	const Vec3V mergeh = sse::_mm_unpacklo_ps(cross12, cross01);
	const Vec3V mergel = sse::_mm_unpackhi_ps(cross12, cross01);
	Vec3V colInv0 = sse::_mm_unpacklo_ps(mergeh, cross20);
	colInv0 = sse::_mm_or_ps(sse::_mm_andnot_ps(tttf, zero), sse::_mm_and_ps(tttf, colInv0));
	const Vec3V zppd = sse::_mm_shuffle_ps(mergeh, cross20, ENG_MM_SHUFFLE(3, 0, 0, 2));
	const Vec3V pbwp = sse::_mm_shuffle_ps(cross20, mergeh, ENG_MM_SHUFFLE(3, 3, 1, 0));
	const Vec3V colInv1 = sse::_mm_or_ps(sse::_mm_andnot_ps(BTFFT(), pbwp), sse::_mm_and_ps(BTFFT(), zppd));
	const Vec3V xppd = sse::_mm_shuffle_ps(mergel, cross20, ENG_MM_SHUFFLE(3, 0, 0, 0));
	const Vec3V pcyp = sse::_mm_shuffle_ps(cross20, mergel, ENG_MM_SHUFFLE(3, 1, 2, 0));
	const Vec3V colInv2 = sse::_mm_or_ps(sse::_mm_andnot_ps(tfft, pcyp), sse::_mm_and_ps(tfft, xppd));

	return Mat33V(sse::_mm_mul_ps(colInv0, invDet), sse::_mm_mul_ps(colInv1, invDet), sse::_mm_mul_ps(colInv2, invDet));
}

EHD Mat33V M33Diagonal(const Vec3VArg d)
{
	const FloatV x = V3Mul(V3UnitX(), d);
	const FloatV y = V3Mul(V3UnitY(), d);
	const FloatV z = V3Mul(V3UnitZ(), d);
	return Mat33V(x, y, z);
}

EHD Mat33V Mat33V_From_PxMat33(const PxMat33& m)
{
	return Mat33V(V3LoadU(m.column0), V3LoadU(m.column1), V3LoadU(m.column2));
}

EHD void PxMat33_From_Mat33V(const Mat33V& m, PxMat33& out)
{
	V3StoreU(m.col0, out.column0);
	V3StoreU(m.col1, out.column1);
	V3StoreU(m.col2, out.column2);
}

//////////////////////////////////
// MAT34V
//////////////////////////////////

EHD Vec3V M34MulV3(const Mat34V& a, const Vec3V b)
{
	const FloatV x = V3GetX(b);
	const FloatV y = V3GetY(b);
	const FloatV z = V3GetZ(b);
	const Vec3V v0 = V3Scale(a.col0, x);
	const Vec3V v1 = V3Scale(a.col1, y);
	const Vec3V v2 = V3Scale(a.col2, z);
	const Vec3V v0PlusV1 = V3Add(v0, v1);
	const Vec3V v0PlusV1Plusv2 = V3Add(v0PlusV1, v2);
	return V3Add(v0PlusV1Plusv2, a.col3);
}

EHD Vec3V M34Mul33V3(const Mat34V& a, const Vec3V b)
{
	const FloatV x = V3GetX(b);
	const FloatV y = V3GetY(b);
	const FloatV z = V3GetZ(b);
	const Vec3V v0 = V3Scale(a.col0, x);
	const Vec3V v1 = V3Scale(a.col1, y);
	const Vec3V v2 = V3Scale(a.col2, z);
	const Vec3V v0PlusV1 = V3Add(v0, v1);
	return V3Add(v0PlusV1, v2);
}

EHD Mat34V M34MulM34(const Mat34V& a, const Mat34V& b)
{
	return Mat34V(M34Mul33V3(a, b.col0), M34Mul33V3(a, b.col1), M34Mul33V3(a, b.col2), M34MulV3(a, b.col3));
}

EHD Mat33V M34MulM33(const Mat34V& a, const Mat33V& b)
{
	return Mat33V(M34Mul33V3(a, b.col0), M34Mul33V3(a, b.col1), M34Mul33V3(a, b.col2));
}

EHD Mat33V M34Mul33MM34(const Mat34V& a, const Mat34V& b)
{
	return Mat33V(M34Mul33V3(a, b.col0), M34Mul33V3(a, b.col1), M34Mul33V3(a, b.col2));
}

EHD Mat34V M34Add(const Mat34V& a, const Mat34V& b)
{
	return Mat34V(V3Add(a.col0, b.col0), V3Add(a.col1, b.col1), V3Add(a.col2, b.col2), V3Add(a.col3, b.col3));
}

//////////////////////////////////
// MAT44V
//////////////////////////////////

EHD Vec4V M44MulV4(const Mat44V& a, const Vec4V b)
{
	const FloatV x = V4GetX(b);
	const FloatV y = V4GetY(b);
	const FloatV z = V4GetZ(b);
	const FloatV w = V4GetW(b);

	const Vec4V v0 = V4Scale(a.col0, x);
	const Vec4V v1 = V4Scale(a.col1, y);
	const Vec4V v2 = V4Scale(a.col2, z);
	const Vec4V v3 = V4Scale(a.col3, w);
	const Vec4V v0PlusV1 = V4Add(v0, v1);
	const Vec4V v0PlusV1Plusv2 = V4Add(v0PlusV1, v2);
	return V4Add(v0PlusV1Plusv2, v3);
}

EHD Mat44V M44MulM44(const Mat44V& a, const Mat44V& b)
{
	return Mat44V(M44MulV4(a, b.col0), M44MulV4(a, b.col1), M44MulV4(a, b.col2), M44MulV4(a, b.col3));
}

EHD Mat44V M44Add(const Mat44V& a, const Mat44V& b)
{
	return Mat44V(V4Add(a.col0, b.col0), V4Add(a.col1, b.col1), V4Add(a.col2, b.col2), V4Add(a.col3, b.col3));
}

EHD Mat44V M44Trnsps(const Mat44V& a);

EHD Mat44V M44Inverse(const Mat44V& a)
{
	sse::m128 minor0, minor1, minor2, minor3;
	sse::m128 row0, row1, row2, row3;
	sse::m128 det, tmp1;

	tmp1 = V4Zero();
	row1 = V4Zero();
	row3 = V4Zero();

	row0 = a.col0;
	row1 = sse::_mm_shuffle_ps(a.col1, a.col1, ENG_MM_SHUFFLE(1, 0, 3, 2));
	row2 = a.col2;
	row3 = sse::_mm_shuffle_ps(a.col3, a.col3, ENG_MM_SHUFFLE(1, 0, 3, 2));

	tmp1 = sse::_mm_mul_ps(row2, row3);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	minor0 = sse::_mm_mul_ps(row1, tmp1);
	minor1 = sse::_mm_mul_ps(row0, tmp1);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor0 = sse::_mm_sub_ps(sse::_mm_mul_ps(row1, tmp1), minor0);
	minor1 = sse::_mm_sub_ps(sse::_mm_mul_ps(row0, tmp1), minor1);
	minor1 = sse::_mm_shuffle_ps(minor1, minor1, 0x4E);

	tmp1 = sse::_mm_mul_ps(row1, row2);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	minor0 = sse::_mm_add_ps(sse::_mm_mul_ps(row3, tmp1), minor0);
	minor3 = sse::_mm_mul_ps(row0, tmp1);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor0 = sse::_mm_sub_ps(minor0, sse::_mm_mul_ps(row3, tmp1));
	minor3 = sse::_mm_sub_ps(sse::_mm_mul_ps(row0, tmp1), minor3);
	minor3 = sse::_mm_shuffle_ps(minor3, minor3, 0x4E);

	tmp1 = sse::_mm_mul_ps(sse::_mm_shuffle_ps(row1, row1, 0x4E), row3);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	row2 = sse::_mm_shuffle_ps(row2, row2, 0x4E);
	minor0 = sse::_mm_add_ps(sse::_mm_mul_ps(row2, tmp1), minor0);
	minor2 = sse::_mm_mul_ps(row0, tmp1);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor0 = sse::_mm_sub_ps(minor0, sse::_mm_mul_ps(row2, tmp1));
	minor2 = sse::_mm_sub_ps(sse::_mm_mul_ps(row0, tmp1), minor2);
	minor2 = sse::_mm_shuffle_ps(minor2, minor2, 0x4E);

	tmp1 = sse::_mm_mul_ps(row0, row1);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	minor2 = sse::_mm_add_ps(sse::_mm_mul_ps(row3, tmp1), minor2);
	minor3 = sse::_mm_sub_ps(sse::_mm_mul_ps(row2, tmp1), minor3);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor2 = sse::_mm_sub_ps(sse::_mm_mul_ps(row3, tmp1), minor2);
	minor3 = sse::_mm_sub_ps(minor3, sse::_mm_mul_ps(row2, tmp1));

	tmp1 = sse::_mm_mul_ps(row0, row3);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	minor1 = sse::_mm_sub_ps(minor1, sse::_mm_mul_ps(row2, tmp1));
	minor2 = sse::_mm_add_ps(sse::_mm_mul_ps(row1, tmp1), minor2);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor1 = sse::_mm_add_ps(sse::_mm_mul_ps(row2, tmp1), minor1);
	minor2 = sse::_mm_sub_ps(minor2, sse::_mm_mul_ps(row1, tmp1));

	tmp1 = sse::_mm_mul_ps(row0, row2);
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0xB1);
	minor1 = sse::_mm_add_ps(sse::_mm_mul_ps(row3, tmp1), minor1);
	minor3 = sse::_mm_sub_ps(minor3, sse::_mm_mul_ps(row1, tmp1));
	tmp1 = sse::_mm_shuffle_ps(tmp1, tmp1, 0x4E);
	minor1 = sse::_mm_sub_ps(minor1, sse::_mm_mul_ps(row3, tmp1));
	minor3 = sse::_mm_add_ps(sse::_mm_mul_ps(row1, tmp1), minor3);

	det = sse::_mm_mul_ps(row0, minor0);
	det = sse::_mm_add_ps(sse::_mm_shuffle_ps(det, det, 0x4E), det);
	det = sse::_mm_add_ss(sse::_mm_shuffle_ps(det, det, 0xB1), det);
	tmp1 = sse::_mm_rcp_ss(det);
#if 0
	det = sse::_mm_sub_ss(sse::_mm_add_ss(tmp1, tmp1), sse::_mm_mul_ss(det, sse::_mm_mul_ss(tmp1, tmp1)));
	det = sse::_mm_shuffle_ps(det, det, 0x00);
#else
	det = sse::_mm_shuffle_ps(tmp1, tmp1, ENG_MM_SHUFFLE(0, 0, 0, 0));
#endif

	minor0 = sse::_mm_mul_ps(det, minor0);
	minor1 = sse::_mm_mul_ps(det, minor1);
	minor2 = sse::_mm_mul_ps(det, minor2);
	minor3 = sse::_mm_mul_ps(det, minor3);
	Mat44V invTrans(minor0, minor1, minor2, minor3);
	return M44Trnsps(invTrans);
}

//////////////////////////////////
// Misc
//////////////////////////////////

// PT: TODO: seems to be in the wrong section
EHD Vec4V V4LoadXYZW(const PxF32& x, const PxF32& y, const PxF32& z, const PxF32& w)
{
	return sse::_mm_set_ps(w, z, y, x);
}

EHD VecU32V V4U32Sel(const BoolV c, const VecU32V a, const VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_or_si128(sse::_mm_andnot_si128(internalSimd::m128_F2I(c), internalSimd::m128_F2I(b)), sse::_mm_and_si128(internalSimd::m128_F2I(c), internalSimd::m128_F2I(a))));
}

EHD VecU32V V4U32or(VecU32V a, VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_or_si128(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
}

EHD VecU32V V4U32xor(VecU32V a, VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_xor_si128(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
}

EHD VecU32V V4U32and(VecU32V a, VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_and_si128(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
}

EHD VecU32V V4U32Andc(VecU32V a, VecU32V b)
{
	return internalSimd::m128_I2F(sse::_mm_andnot_si128(internalSimd::m128_F2I(b), internalSimd::m128_F2I(a)));
}

EHD VecU32V U4Load(const PxU32 i)
{
	return sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&i));
}

EHD VecU32V U4LoadU(const PxU32* i)
{
	return sse::_mm_loadu_ps(reinterpret_cast<const PxF32*>(i));
}

EHD VecU32V U4LoadA(const PxU32* i)
{
	ASSERT_ISALIGNED16(i);
	return sse::_mm_load_ps(reinterpret_cast<const PxF32*>(i));
}

EHD VecI32V VecI32V_One()
{
	return I4Load(1);
}

EHD VecI32V VecI32V_Two()
{
	return I4Load(2);
}

EHD VecI32V VecI32V_MinusOne()
{
	return I4Load(-1);
}

EHD VecU32V U4Zero()
{
	return U4Load(0);
}

EHD VecU32V U4One()
{
	return U4Load(1);
}

EHD VecU32V U4Two()
{
	return U4Load(2);
}

EHD Vec4V V4Andc(const Vec4V a, const VecU32V b)
{
	VecU32V result32(a);
	result32 = V4U32Andc(result32, b);
	return Vec4V(result32);
}

EHD VecU32V V4IsGrtrV32u(const Vec4V a, const Vec4V b)
{
	return V4IsGrtr(a, b);
}

EHD VecU16V V4U16LoadAligned(const VecU16V* addr)
{
	return *addr;
}

EHD VecU16V V4U16LoadUnaligned(const VecU16V* addr)
{
	return *addr;
}

EHD VecU16V V4I16CompareGt(VecU16V a, VecU16V b)
{
	return internalSimd::m128_I2F(sse::_mm_cmpgt_epi16(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
}

// unsigned compares are not supported on x86
EHD VecU16V V4U16CompareGt(VecU16V a, VecU16V b)
{
	// _mm_cmpgt_epi16 doesn't work for unsigned values unfortunately
	// return m128_I2F(sse::_mm_cmpgt_epi16(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b)));
	VecU16V result;
	result.m128_u16[0] = PxU16((a).m128_u16[0] > (b).m128_u16[0]);
	result.m128_u16[1] = PxU16((a).m128_u16[1] > (b).m128_u16[1]);
	result.m128_u16[2] = PxU16((a).m128_u16[2] > (b).m128_u16[2]);
	result.m128_u16[3] = PxU16((a).m128_u16[3] > (b).m128_u16[3]);
	result.m128_u16[4] = PxU16((a).m128_u16[4] > (b).m128_u16[4]);
	result.m128_u16[5] = PxU16((a).m128_u16[5] > (b).m128_u16[5]);
	result.m128_u16[6] = PxU16((a).m128_u16[6] > (b).m128_u16[6]);
	result.m128_u16[7] = PxU16((a).m128_u16[7] > (b).m128_u16[7]);
	return result;
}

EHD Vec4V Vec4V_From_VecU32V(VecU32V a)
{
	Vec4V result = V4LoadXYZW(PxF32(a.m128_u32[0]), PxF32(a.m128_u32[1]), PxF32(a.m128_u32[2]), PxF32(a.m128_u32[3]));
	return result;
}

EHD VecU32V U4LoadXYZW(PxU32 x, PxU32 y, PxU32 z, PxU32 w)
{
	VecU32V result;
	result.m128_u32[0] = x;
	result.m128_u32[1] = y;
	result.m128_u32[2] = z;
	result.m128_u32[3] = w;
	return result;
}

} // namespace aos
} // namespace eng





// ======== PxUnixSse2InlineAoS.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========


namespace eng
{
namespace aos
{
//////////////////////////////////////////////////////////////////////
//Test that Vec3V and FloatV are legal
//////////////////////////////////////////////////////////////////////

#define FLOAT_COMPONENTS_EQUAL_THRESHOLD 0.01f
EHD static bool isValidFloatV(const FloatV a)
{
	const PxF32 x = V4ReadX(a);
	const PxF32 y = V4ReadY(a);
	const PxF32 z = V4ReadZ(a);
	const PxF32 w = V4ReadW(a);

	return (x == y && x == z && x == w);

 	/*if (
		(PxAbs(x - y) < FLOAT_COMPONENTS_EQUAL_THRESHOLD) &&
		(PxAbs(x - z) < FLOAT_COMPONENTS_EQUAL_THRESHOLD) &&
		(PxAbs(x - w) < FLOAT_COMPONENTS_EQUAL_THRESHOLD)
		)
	{
		return true;
	}

	if (
		(PxAbs((x - y) / x) < FLOAT_COMPONENTS_EQUAL_THRESHOLD) &&
		(PxAbs((x - z) / x) < FLOAT_COMPONENTS_EQUAL_THRESHOLD) &&
		(PxAbs((x - w) / x) < FLOAT_COMPONENTS_EQUAL_THRESHOLD)
		)
	{
		return true;
	}
	return false;*/
}

}
}


namespace eng
{
namespace aos
{

#define PX_FPCLASS_SNAN 0x0001 /* signaling NaN */
#define PX_FPCLASS_QNAN 0x0002 /* quiet NaN */
#define PX_FPCLASS_NINF 0x0004 /* negative infinity */
#define PX_FPCLASS_PINF 0x0200 /* positive infinity */

namespace internalSimd
{
alignas(16) static const ENG_U4F gMaskXYZ_h = {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0u}};
#if defined(__CUDACC__)
__device__ __constant__ static const ENG_U4F gMaskXYZ_d = {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0u}};
#endif

}

namespace vecMathTests
{
EHD bool allElementsEqualBoolV(const BoolV a, const BoolV b)
{
	return internalSimd::BAllTrue4_R(VecI32V_IsEq(internalSimd::m128_F2I(a), internalSimd::m128_F2I(b))) != 0;
}

EHD bool allElementsEqualVecI32V(const VecI32V a, const VecI32V b)
{
	BoolV c = internalSimd::m128_I2F(sse::_mm_cmpeq_epi32(a, b));
	return internalSimd::BAllTrue4_R(c) != 0;
}

#define VECMATH_AOS_EPSILON (1e-3f)

EHD bool allElementsNearEqualFloatV(const FloatV a, const FloatV b)
{
	ASSERT_ISVALIDFLOATV(a);
	ASSERT_ISVALIDFLOATV(b);
	const FloatV c = FSub(a, b);
	const FloatV minError = FLoad(-VECMATH_AOS_EPSILON);
	const FloatV maxError = FLoad(VECMATH_AOS_EPSILON);
	return sse::_mm_comigt_ss(c, minError) && sse::_mm_comilt_ss(c, maxError);
}

EHD bool allElementsNearEqualVec3V(const Vec3V a, const Vec3V b)
{
	const Vec3V c = V3Sub(a, b);
	const Vec3V minError = V3Load(-VECMATH_AOS_EPSILON);
	const Vec3V maxError = V3Load(VECMATH_AOS_EPSILON);
	return (sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(0, 0, 0, 0)), minError) &&
	 		sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(0, 0, 0, 0)), maxError) &&
	 		sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(1, 1, 1, 1)), minError) &&
	 		sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(1, 1, 1, 1)), maxError) &&
	 		sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(2, 2, 2, 2)), minError) &&
	 		sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(2, 2, 2, 2)), maxError));
}

EHD bool allElementsNearEqualVec4V(const Vec4V a, const Vec4V b)
{
	const Vec4V c = V4Sub(a, b);
	const Vec4V minError = V4Load(-VECMATH_AOS_EPSILON);
	const Vec4V maxError = V4Load(VECMATH_AOS_EPSILON);
	return (sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(0, 0, 0, 0)), minError) &&
	        sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(0, 0, 0, 0)), maxError) &&
	        sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(1, 1, 1, 1)), minError) &&
	        sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(1, 1, 1, 1)), maxError) &&
	        sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(2, 2, 2, 2)), minError) &&
	        sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(2, 2, 2, 2)), maxError) &&
	        sse::_mm_comigt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(3, 3, 3, 3)), minError) &&
	        sse::_mm_comilt_ss(sse::_mm_shuffle_ps(c, c, ENG_MM_SHUFFLE(3, 3, 3, 3)), maxError));
}
} //vecMathTests

/////////////////////////////////////////////////////////////////////
////FUNCTIONS USED ONLY FOR ASSERTS IN VECTORISED IMPLEMENTATIONS
/////////////////////////////////////////////////////////////////////

EHD bool isFiniteFloatV(const FloatV a)
{
	PxF32 badNumber =
	    eng::PxUnionCast<PxF32, PxU32>(PX_FPCLASS_SNAN | PX_FPCLASS_QNAN | PX_FPCLASS_NINF | PX_FPCLASS_PINF);
	const FloatV vBadNum = FLoad(badNumber);
	const BoolV vMask = BAnd(vBadNum, a);
	return internalSimd::FiniteTestEq(vMask, BFFFF()) == 1;
}

EHD bool isFiniteVec3V(const Vec3V a)
{
	PxF32 badNumber =
	    eng::PxUnionCast<PxF32, PxU32>(PX_FPCLASS_SNAN | PX_FPCLASS_QNAN | PX_FPCLASS_NINF | PX_FPCLASS_PINF);
	const Vec3V vBadNum = V3Load(badNumber);
	const BoolV vMask = BAnd(BAnd(vBadNum, a), BTTTF());
	return internalSimd::FiniteTestEq(vMask, BFFFF()) == 1;
}

EHD bool isFiniteVec4V(const Vec4V a)
{
	/*Vec4V a;
	PX_ALIGN(16, PxF32 f[4]);
	F32Array_Aligned_From_Vec4V(a, f);
	return PxIsFinite(f[0])
	        && PxIsFinite(f[1])
	        && PxIsFinite(f[2])
	        && PxIsFinite(f[3]);*/

	PxF32 badNumber =
	    eng::PxUnionCast<PxF32, PxU32>(PX_FPCLASS_SNAN | PX_FPCLASS_QNAN | PX_FPCLASS_NINF | PX_FPCLASS_PINF);
	const Vec4V vBadNum = V4Load(badNumber);
	const BoolV vMask = BAnd(vBadNum, a);

	return internalSimd::FiniteTestEq(vMask, BFFFF()) == 1;
}

/////////////////////////////////////////////////////////////////////
////VECTORISED FUNCTION IMPLEMENTATIONS
/////////////////////////////////////////////////////////////////////

EHD Vec3V V3LoadA(const PxVec3& f)
{
	ASSERT_ISALIGNED16(&f);
	return sse::_mm_and_ps(reinterpret_cast<const Vec3V&>(f), V4LoadA(ENG_G(internalSimd::gMaskXYZ).f));
}

EHD Vec3V V3LoadUnsafeA(const PxVec3& f)
{
	ASSERT_ISALIGNED16(&f);
	return sse::_mm_set_ps(0.0f, f.z, f.y, f.x);
}

EHD Vec3V V3LoadA(const PxF32* const f)
{
	ASSERT_ISALIGNED16(f);
	return sse::_mm_and_ps(V4LoadA(f), V4LoadA(ENG_G(internalSimd::gMaskXYZ).f));
}

EHD void I4StoreA(const VecI32V iv, PxI32* i)
{
	ASSERT_ISALIGNED16(i);
	sse::_mm_store_ps(reinterpret_cast<float*>(i), internalSimd::m128_I2F(iv));
}

EHD BoolV BLoad(const bool* const f)
{
	const PX_ALIGN(16, PxI32) b[4] = { -PxI32(f[0]), -PxI32(f[1]), -PxI32(f[2]), -PxI32(f[3]) };
	return sse::_mm_load_ps(reinterpret_cast<const float*>(&b));
}

EHD void V3StoreA(const Vec3V a, PxVec3& f)
{
	ASSERT_ISALIGNED16(&f);
	PX_ALIGN(16, PxF32) f2[4];
	sse::_mm_store_ps(f2, a);
	f = PxVec3(f2[0], f2[1], f2[2]);
}

EHD void V3StoreU(const Vec3V a, PxVec3& f)
{
	PX_ALIGN(16, PxF32) f2[4];
	sse::_mm_store_ps(f2, a);
	f = PxVec3(f2[0], f2[1], f2[2]);
}

//////////////////////////////////
// FLOATV
//////////////////////////////////

EHD FloatV FAbs(const FloatV a)
{
	ASSERT_ISVALIDFLOATV(a);
	PX_ALIGN(16, const PxU32) absMask[4] = { 0x7fFFffFF, 0x7fFFffFF, 0x7fFFffFF, 0x7fFFffFF };
	return sse::_mm_and_ps(a, sse::_mm_load_ps(reinterpret_cast<const PxF32*>(absMask)));
}

//////////////////////////////////
// VEC3V
//////////////////////////////////

EHD Vec3V V3UnitX()
{
	const PX_ALIGN(16, PxF32) x[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
	const sse::m128 x128 = sse::_mm_load_ps(x);
	return x128;
}

EHD Vec3V V3UnitY()
{
	const PX_ALIGN(16, PxF32) y[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
	const sse::m128 y128 = sse::_mm_load_ps(y);
	return y128;
}

EHD Vec3V V3UnitZ()
{
	const PX_ALIGN(16, PxF32) z[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
	const sse::m128 z128 = sse::_mm_load_ps(z);
	return z128;
}

//////////////////////////////////
// VEC4V
//////////////////////////////////

EHD Vec4V V4UnitW()
{
	const PX_ALIGN(16, PxF32) w[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const sse::m128 w128 = sse::_mm_load_ps(w);
	return w128;
}

EHD Vec4V V4UnitX()
{
	const PX_ALIGN(16, PxF32) x[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
	const sse::m128 x128 = sse::_mm_load_ps(x);
	return x128;
}

EHD Vec4V V4UnitY()
{
	const PX_ALIGN(16, PxF32) y[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
	const sse::m128 y128 = sse::_mm_load_ps(y);
	return y128;
}

EHD Vec4V V4UnitZ()
{
	const PX_ALIGN(16, PxF32) z[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
	const sse::m128 z128 = sse::_mm_load_ps(z);
	return z128;
}

EHD Vec4V V4ClearW(const Vec4V v)
{
	return sse::_mm_and_ps(v, V4LoadA(ENG_G(internalSimd::gMaskXYZ).f));
}

//////////////////////////////////
// BoolV
//////////////////////////////////

/*
template<int index> EHD BoolV BSplatElement(BoolV a)
{
    BoolV result;
    result[0] = result[1] = result[2] = result[3] = a[index];
    return result;
}
*/

template <int index>
EHD BoolV BSplatElement(BoolV a)
{
	float* data = reinterpret_cast<float*>(&a);
	return V4Load(data[index]);
}

//////////////////////////////////
// MAT33V
//////////////////////////////////

EHD Vec3V M33TrnspsMulV3(const Mat33V& a, const Vec3V b)
{
	const FloatV x = V3Dot(a.col0, b);
	const FloatV y = V3Dot(a.col1, b);
	const FloatV z = V3Dot(a.col2, b);
	return V3Merge(x, y, z);
}

EHD Mat33V M33Trnsps(const Mat33V& a)
{
	return Mat33V(V3Merge(V3GetX(a.col0), V3GetX(a.col1), V3GetX(a.col2)),
	              V3Merge(V3GetY(a.col0), V3GetY(a.col1), V3GetY(a.col2)),
	              V3Merge(V3GetZ(a.col0), V3GetZ(a.col1), V3GetZ(a.col2)));
}

/*EHD Mat33V PromoteVec3V(const Vec3V v)
{
	const BoolV bTFFF = BTFFF();
	const BoolV bFTFF = BFTFF();
	const BoolV bFFTF = BTFTF();

	const Vec3V zero = V3Zero();

	return Mat33V(V3Sel(bTFFF, v, zero), V3Sel(bFTFF, v, zero), V3Sel(bFFTF, v, zero));
}*/

//////////////////////////////////
// MAT34V
//////////////////////////////////

EHD Vec3V M34TrnspsMul33V3(const Mat34V& a, const Vec3V b)
{
	const FloatV x = V3Dot(a.col0, b);
	const FloatV y = V3Dot(a.col1, b);
	const FloatV z = V3Dot(a.col2, b);
	return V3Merge(x, y, z);
}

EHD Mat33V M34Trnsps33(const Mat34V& a)
{
	return Mat33V(V3Merge(V3GetX(a.col0), V3GetX(a.col1), V3GetX(a.col2)),
	              V3Merge(V3GetY(a.col0), V3GetY(a.col1), V3GetY(a.col2)),
	              V3Merge(V3GetZ(a.col0), V3GetZ(a.col1), V3GetZ(a.col2)));
}

//////////////////////////////////
// MAT44V
//////////////////////////////////

EHD Vec4V M44TrnspsMulV4(const Mat44V& a, const Vec4V b)
{
	PX_ALIGN(16, FloatV) dotProdArray[4] = { V4Dot(a.col0, b), V4Dot(a.col1, b), V4Dot(a.col2, b), V4Dot(a.col3, b) };
	return V4Merge(dotProdArray);
}

EHD Mat44V M44Trnsps(const Mat44V& a)
{
	const Vec4V v0 = sse::_mm_unpacklo_ps(a.col0, a.col2);
	const Vec4V v1 = sse::_mm_unpackhi_ps(a.col0, a.col2);
	const Vec4V v2 = sse::_mm_unpacklo_ps(a.col1, a.col3);
	const Vec4V v3 = sse::_mm_unpackhi_ps(a.col1, a.col3);
	return Mat44V(sse::_mm_unpacklo_ps(v0, v2), sse::_mm_unpackhi_ps(v0, v2), sse::_mm_unpacklo_ps(v1, v3), sse::_mm_unpackhi_ps(v1, v3));
}

//////////////////////////////////
// Misc
//////////////////////////////////

/*
// AP: work in progress - use proper SSE intrinsics where possible
EHD VecU16V V4U32PK(VecU32V a, VecU32V b)
{
    VecU16V result;
    result.m128_u16[0] = PxU16(PxClamp<PxU32>((a).m128_u32[0], 0, 0xFFFF));
    result.m128_u16[1] = PxU16(PxClamp<PxU32>((a).m128_u32[1], 0, 0xFFFF));
    result.m128_u16[2] = PxU16(PxClamp<PxU32>((a).m128_u32[2], 0, 0xFFFF));
    result.m128_u16[3] = PxU16(PxClamp<PxU32>((a).m128_u32[3], 0, 0xFFFF));
    result.m128_u16[4] = PxU16(PxClamp<PxU32>((b).m128_u32[0], 0, 0xFFFF));
    result.m128_u16[5] = PxU16(PxClamp<PxU32>((b).m128_u32[1], 0, 0xFFFF));
    result.m128_u16[6] = PxU16(PxClamp<PxU32>((b).m128_u32[2], 0, 0xFFFF));
    result.m128_u16[7] = PxU16(PxClamp<PxU32>((b).m128_u32[3], 0, 0xFFFF));
    return result;
}
*/

/*
EHD VecU16V V4U16Or(VecU16V a, VecU16V b)
{
    return m128_I2F(sse::_mm_or_si128(m128_F2I(a), m128_F2I(b)));
}
*/

/*
EHD VecU16V V4U16And(VecU16V a, VecU16V b)
{
    return m128_I2F(sse::_mm_and_si128(m128_F2I(a), m128_F2I(b)));
}
*/

/*
EHD VecU16V V4U16Andc(VecU16V a, VecU16V b)
{
    return m128_I2F(sse::_mm_andnot_si128(m128_F2I(b), m128_F2I(a)));
}
*/

EHD VecI32V I4LoadXYZW(const PxI32& x, const PxI32& y, const PxI32& z, const PxI32& w)
{
	return sse::_mm_set_epi32(w, z, y, x);
}

EHD VecI32V I4Load(const PxI32 i)
{
	return internalSimd::m128_F2I(sse::_mm_load1_ps(reinterpret_cast<const PxF32*>(&i)));
}

EHD VecI32V I4LoadU(const PxI32* i)
{
	return internalSimd::m128_F2I(sse::_mm_loadu_ps(reinterpret_cast<const PxF32*>(i)));
}

EHD VecI32V I4LoadA(const PxI32* i)
{
	ASSERT_ISALIGNED16(i);
	return internalSimd::m128_F2I(sse::_mm_load_ps(reinterpret_cast<const PxF32*>(i)));
}

EHD VecI32V VecI32V_Add(const VecI32VArg a, const VecI32VArg b)
{
	return sse::_mm_add_epi32(a, b);
}

EHD VecI32V VecI32V_Sub(const VecI32VArg a, const VecI32VArg b)
{
	return sse::_mm_sub_epi32(a, b);
}

EHD BoolV VecI32V_IsGrtr(const VecI32VArg a, const VecI32VArg b)
{
	return internalSimd::m128_I2F(sse::_mm_cmpgt_epi32(a, b));
}

EHD BoolV VecI32V_IsEq(const VecI32VArg a, const VecI32VArg b)
{
	return internalSimd::m128_I2F(sse::_mm_cmpeq_epi32(a, b));
}

EHD VecI32V V4I32Sel(const BoolV c, const VecI32V a, const VecI32V b)
{
	return sse::_mm_or_si128(sse::_mm_andnot_si128(internalSimd::m128_F2I(c), b), sse::_mm_and_si128(internalSimd::m128_F2I(c), a));
}

EHD VecI32V VecI32V_Zero()
{
	return sse::_mm_setzero_si128();
}

EHD VecI32V VecI32V_Sel(const BoolV c, const VecI32VArg a, const VecI32VArg b)
{
	return sse::_mm_or_si128(sse::_mm_andnot_si128(internalSimd::m128_F2I(c), b), sse::_mm_and_si128(internalSimd::m128_F2I(c), a));
}

EHD VecShiftV VecI32V_PrepareShift(const VecI32VArg shift)
{
	VecShiftV s;
	s.shift = VecI32V_Sel(BTFFF(), shift, VecI32V_Zero());
	return s;
}

EHD VecI32V VecI32V_LeftShift(const VecI32VArg a, const VecShiftVArg count)
{
	return sse::_mm_sll_epi32(a, count.shift);
}

EHD VecI32V VecI32V_RightShift(const VecI32VArg a, const VecShiftVArg count)
{
	return sse::_mm_srl_epi32(a, count.shift);
}

EHD VecI32V VecI32V_LeftShift(const VecI32VArg a, const PxU32 count)
{
	return sse::_mm_slli_epi32(a, PxI32(count));
}

EHD VecI32V VecI32V_RightShift(const VecI32VArg a, const PxU32 count)
{
	return sse::_mm_srai_epi32(a, PxI32(count));
}

EHD VecI32V VecI32V_And(const VecI32VArg a, const VecI32VArg b)
{
	return sse::_mm_and_si128(a, b);
}

EHD VecI32V VecI32V_Or(const VecI32VArg a, const VecI32VArg b)
{
	return sse::_mm_or_si128(a, b);
}

EHD VecI32V VecI32V_GetX(const VecI32VArg a)
{
	return internalSimd::m128_F2I(sse::_mm_shuffle_ps(internalSimd::m128_I2F(a), internalSimd::m128_I2F(a), ENG_MM_SHUFFLE(0, 0, 0, 0)));
}

EHD VecI32V VecI32V_GetY(const VecI32VArg a)
{
	return internalSimd::m128_F2I(sse::_mm_shuffle_ps(internalSimd::m128_I2F(a), internalSimd::m128_I2F(a), ENG_MM_SHUFFLE(1, 1, 1, 1)));
}

EHD VecI32V VecI32V_GetZ(const VecI32VArg a)
{
	return internalSimd::m128_F2I(sse::_mm_shuffle_ps(internalSimd::m128_I2F(a), internalSimd::m128_I2F(a), ENG_MM_SHUFFLE(2, 2, 2, 2)));
}

EHD VecI32V VecI32V_GetW(const VecI32VArg a)
{
	return internalSimd::m128_F2I(sse::_mm_shuffle_ps(internalSimd::m128_I2F(a), internalSimd::m128_I2F(a), ENG_MM_SHUFFLE(3, 3, 3, 3)));
}

EHD void PxI32_From_VecI32V(const VecI32VArg a, PxI32* i)
{
	sse::_mm_store_ss(reinterpret_cast<PxF32*>(i), internalSimd::m128_I2F(a));
}

EHD VecI32V VecI32V_From_BoolV(const BoolVArg a)
{
	return internalSimd::m128_F2I(a);
}

EHD VecU32V VecU32V_From_BoolV(const BoolVArg a)
{
	return a;
}

EHD VecI32V VecI32V_Merge(const VecI32VArg x, const VecI32VArg y, const VecI32VArg z, const VecI32VArg w)
{
	const sse::m128 xw = sse::_mm_move_ss(internalSimd::m128_I2F(y), internalSimd::m128_I2F(x)); // y, y, y, x
	const sse::m128 yz = sse::_mm_move_ss(internalSimd::m128_I2F(z), internalSimd::m128_I2F(w)); // z, z, z, w
	return internalSimd::m128_F2I(sse::_mm_shuffle_ps(xw, yz, ENG_MM_SHUFFLE(0, 2, 1, 0)));
}

/*
template<int a> EHD VecI32V V4ISplat()
{
    VecI32V result;
    result.m128_i32[0] = a;
    result.m128_i32[1] = a;
    result.m128_i32[2] = a;
    result.m128_i32[3] = a;
    return result;
}

template<PxU32 a> EHD VecU32V V4USplat()
{
    VecU32V result;
    result.m128_u32[0] = a;
    result.m128_u32[1] = a;
    result.m128_u32[2] = a;
    result.m128_u32[3] = a;
    return result;
}
*/

/*
EHD void V4U16StoreAligned(VecU16V val, VecU16V* address)
{
    *address = val;
}
*/

EHD void V4U32StoreAligned(VecU32V val, VecU32V* address)
{
	*address = val;
}

/*EHD Vec4V V4LoadAligned(Vec4V* addr)
{
	return *addr;
}

EHD Vec4V V4LoadUnaligned(Vec4V* addr)
{
	return V4LoadU(reinterpret_cast<float*>(addr));
}*/

EHD Vec4V Vec4V_From_VecI32V(VecI32V in)
{
	return sse::_mm_cvtepi32_ps(in);
}

EHD VecI32V VecI32V_From_Vec4V(Vec4V a)
{
	return sse::_mm_cvttps_epi32(a);
}

EHD Vec4V Vec4V_ReinterpretFrom_VecU32V(VecU32V a)
{
	return Vec4V(a);
}

EHD Vec4V Vec4V_ReinterpretFrom_VecI32V(VecI32V a)
{
	return internalSimd::m128_I2F(a);
}

EHD VecU32V VecU32V_ReinterpretFrom_Vec4V(Vec4V a)
{
	return VecU32V(a);
}

EHD VecI32V VecI32V_ReinterpretFrom_Vec4V(Vec4V a)
{
	return internalSimd::m128_F2I(a);
}

template <int index>
EHD VecU32V V4U32SplatElement(VecU32V a)
{
	VecU32V result;
	result.m128_u32[0] = result.m128_u32[1] = result.m128_u32[2] = result.m128_u32[3] = a.m128_u32[index];
	return result;
}

template <int index>
EHD Vec4V V4SplatElement(Vec4V a)
{
	float* data = reinterpret_cast<float*>(&a);
	return V4Load(data[index]);
}

/*EHD Vec4V V4Ceil(const Vec4V in)
{
	UnionM128 a(in);
	return V4LoadXYZW(PxCeil(a.m128_f32[0]), PxCeil(a.m128_f32[1]), PxCeil(a.m128_f32[2]), PxCeil(a.m128_f32[3]));
}

EHD Vec4V V4Floor(const Vec4V in)
{
	UnionM128 a(in);
	return V4LoadXYZW(PxFloor(a.m128_f32[0]), PxFloor(a.m128_f32[1]), PxFloor(a.m128_f32[2]), PxFloor(a.m128_f32[3]));
}

EHD VecU32V V4ConvertToU32VSaturate(const Vec4V in, PxU32 power)
{
	PX_ASSERT(power == 0 && "Non-zero power not supported in convertToU32VSaturate");
	PX_UNUSED(power); // prevent warning in release builds
	PxF32 ffffFFFFasFloat = PxF32(0xFFFF0000);
	UnionM128 a(in);
	VecU32V result;
	result.m128_u32[0] = PxU32(PxClamp<PxF32>((a).m128_f32[0], 0.0f, ffffFFFFasFloat));
	result.m128_u32[1] = PxU32(PxClamp<PxF32>((a).m128_f32[1], 0.0f, ffffFFFFasFloat));
	result.m128_u32[2] = PxU32(PxClamp<PxF32>((a).m128_f32[2], 0.0f, ffffFFFFasFloat));
	result.m128_u32[3] = PxU32(PxClamp<PxF32>((a).m128_f32[3], 0.0f, ffffFFFFasFloat));
	return result;
}*/

} // namespace aos
} // namespace eng



// ======== PxVecQuat.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========


namespace eng
{
namespace aos
{

#ifndef PX_PIDIV2
#define PX_PIDIV2 1.570796327f
#endif

//////////////////////////////////
// QuatV
//////////////////////////////////
EHD QuatV QuatVLoadXYZW(const PxF32 x, const PxF32 y, const PxF32 z, const PxF32 w)
{
	return V4LoadXYZW(x, y, z, w);
}

EHD QuatV QuatVLoadU(const PxF32* v)
{
	return V4LoadU(v);
}

EHD QuatV QuatVLoadA(const PxF32* v)
{
	return V4LoadA(v);
}

EHD QuatV QuatV_From_RotationAxisAngle(const Vec3V u, const FloatV a)
{
	// q = cos(a/2) + u*sin(a/2)
	const FloatV half = FLoad(0.5f);
	const FloatV hangle = FMul(a, half);
	const FloatV piByTwo(FLoad(PX_PIDIV2));
	const FloatV PiByTwoMinHangle(FSub(piByTwo, hangle));
	const Vec4V hangle2(Vec4V_From_Vec3V(V3Merge(hangle, PiByTwoMinHangle, hangle)));

	/*const FloatV sina = FSin(hangle);
	const FloatV cosa = FCos(hangle);*/

	const Vec4V _sina = V4Sin(hangle2);
	const FloatV sina = V4GetX(_sina);
	const FloatV cosa = V4GetY(_sina);

	const Vec3V v = V3Scale(u, sina);
	// return V4Sel(BTTTF(), Vec4V_From_Vec3V(v), V4Splat(cosa));
	return V4SetW(Vec4V_From_Vec3V(v), cosa);
}

// Normalize
EHD QuatV QuatNormalize(const QuatV q)
{
	return V4Normalize(q);
}

EHD FloatV QuatLength(const QuatV q)
{
	return V4Length(q);
}

EHD FloatV QuatLengthSq(const QuatV q)
{
	return V4LengthSq(q);
}

EHD FloatV QuatDot(const QuatV a, const QuatV b) // convert this PxQuat to a unit quaternion
{
	return V4Dot(a, b);
}

EHD QuatV QuatConjugate(const QuatV q)
{
	return V4SetW(V4Neg(q), V4GetW(q));
}

EHD Vec3V QuatGetImaginaryPart(const QuatV q)
{
	return Vec3V_From_Vec4V(q);
}

/** brief computes rotation of x-axis */
EHD Vec3V QuatGetBasisVector0(const QuatV q)
{
	/*const PxF32 x2 = x*2.0f;
	const PxF32 w2 = w*2.0f;
	return PxVec3(	(w * w2) - 1.0f + x*x2,
	                (z * w2)        + y*x2,
	                (-y * w2)       + z*x2);*/

	const FloatV two = FLoad(2.0f);
	const FloatV w = V4GetW(q);
	const Vec3V u = Vec3V_From_Vec4V(q);

	const FloatV x2 = FMul(V3GetX(u), two);
	const FloatV w2 = FMul(w, two);

	const Vec3V a = V3Scale(u, x2);
	const Vec3V tmp = V3Merge(w, V3GetZ(u), FNeg(V3GetY(u)));
	// const Vec3V b = V3Scale(tmp, w2);
	// const Vec3V ab = V3Add(a, b);
	const Vec3V ab = V3ScaleAdd(tmp, w2, a);
	return V3SetX(ab, FSub(V3GetX(ab), FOne()));
}

/** brief computes rotation of y-axis */
EHD Vec3V QuatGetBasisVector1(const QuatV q)
{
	/*const PxF32 y2 = y*2.0f;
	const PxF32 w2 = w*2.0f;
	return PxVec3(	(-z * w2)       + x*y2,
	                (w * w2) - 1.0f + y*y2,
	                (x * w2)        + z*y2);*/

	const FloatV two = FLoad(2.0f);
	const FloatV w = V4GetW(q);
	const Vec3V u = Vec3V_From_Vec4V(q);

	const FloatV y2 = FMul(V3GetY(u), two);
	const FloatV w2 = FMul(w, two);

	const Vec3V a = V3Scale(u, y2);
	const Vec3V tmp = V3Merge(FNeg(V3GetZ(u)), w, V3GetX(u));
	// const Vec3V b = V3Scale(tmp, w2);
	// const Vec3V ab = V3Add(a, b);
	const Vec3V ab = V3ScaleAdd(tmp, w2, a);
	return V3SetY(ab, FSub(V3GetY(ab), FOne()));
}

/** brief computes rotation of z-axis */
EHD Vec3V QuatGetBasisVector2(const QuatV q)
{
	/*const PxF32 z2 = z*2.0f;
	const PxF32 w2 = w*2.0f;
	return PxVec3(	(y * w2)        + x*z2,
	                (-x * w2)       + y*z2,
	                (w * w2) - 1.0f + z*z2);*/

	const FloatV two = FLoad(2.0f);
	const FloatV w = V4GetW(q);
	const Vec3V u = Vec3V_From_Vec4V(q);

	const FloatV z2 = FMul(V3GetZ(u), two);
	const FloatV w2 = FMul(w, two);

	const Vec3V a = V3Scale(u, z2);
	const Vec3V tmp = V3Merge(V3GetY(u), FNeg(V3GetX(u)), w);
	/*const Vec3V b = V3Scale(tmp, w2);
	const Vec3V ab = V3Add(a, b);*/
	const Vec3V ab = V3ScaleAdd(tmp, w2, a);
	return V3SetZ(ab, FSub(V3GetZ(ab), FOne()));
}

EHD Vec3V QuatRotate(const QuatV q, const Vec3V v)
{
	/*
	const PxVec3 qv(x,y,z);
	return (v*(w*w-0.5f) + (qv.cross(v))*w + qv*(qv.dot(v)))*2;
	*/

	const FloatV two = FLoad(2.0f);
	// const FloatV half = FloatV_From_F32(0.5f);
	const FloatV nhalf = FLoad(-0.5f);
	const Vec3V u = Vec3V_From_Vec4V(q);
	const FloatV w = V4GetW(q);
	// const FloatV w2 = FSub(FMul(w, w), half);
	const FloatV w2 = FScaleAdd(w, w, nhalf);
	const Vec3V a = V3Scale(v, w2);
	// const Vec3V b = V3Scale(V3Cross(u, v), w);
	// const Vec3V c = V3Scale(u, V3Dot(u, v));
	// return V3Scale(V3Add(V3Add(a, b), c), two);
	const Vec3V temp = V3ScaleAdd(V3Cross(u, v), w, a);
	return V3Scale(V3ScaleAdd(u, V3Dot(u, v), temp), two);
}

// PT: same as QuatRotate but operates on a Vec4V
EHD Vec4V QuatRotate4V(const QuatV q, const Vec4V v)
{
	const FloatV two = FLoad(2.0f);
	const FloatV nhalf = FLoad(-0.5f);
	const Vec4V u = q;	// PT: W not cleared here
	const FloatV w = V4GetW(q);
	const FloatV w2 = FScaleAdd(w, w, nhalf);
	const Vec4V a = V4Scale(v, w2);	// PT: W has non-zero data here
	const Vec4V temp = V4ScaleAdd(V4Cross(u, v), w, a);
	return V4Scale(V4ScaleAdd(u, V4Dot3(u, v), temp), two);	// PT: beware, V4Dot3 has one more instruction here
}

// PT: avoid some multiplies when immediately normalizing a rotated vector
EHD Vec3V QuatRotateAndNormalize(const QuatV q, const Vec3V v)
{
	const FloatV nhalf = FLoad(-0.5f);
	const Vec3V u = Vec3V_From_Vec4V(q);
	const FloatV w = V4GetW(q);
	const FloatV w2 = FScaleAdd(w, w, nhalf);
	const Vec3V a = V3Scale(v, w2);
	const Vec3V temp = V3ScaleAdd(V3Cross(u, v), w, a);
	return V3Normalize(V3ScaleAdd(u, V3Dot(u, v), temp));
}

EHD Vec3V QuatTransform(const QuatV q, const Vec3V p, const Vec3V v)
{
	// p + q.rotate(v)
	const FloatV two = FLoad(2.0f);
	// const FloatV half = FloatV_From_F32(0.5f);
	const FloatV nhalf = FLoad(-0.5f);
	const Vec3V u = Vec3V_From_Vec4V(q);
	const FloatV w = V4GetW(q);
	// const FloatV w2 = FSub(FMul(w, w), half);
	const FloatV w2 = FScaleAdd(w, w, nhalf);
	const Vec3V a = V3Scale(v, w2);
	/*const Vec3V b = V3Scale(V3Cross(u, v), w);
	const Vec3V c = V3Scale(u, V3Dot(u, v));
	return V3ScaleAdd(V3Add(V3Add(a, b), c), two, p);*/
	const Vec3V temp = V3ScaleAdd(V3Cross(u, v), w, a);
	const Vec3V z = V3ScaleAdd(u, V3Dot(u, v), temp);
	return V3ScaleAdd(z, two, p);
}

EHD Vec3V QuatRotateInv(const QuatV q, const Vec3V v)
{
	//	const PxVec3 qv(x,y,z);
	//	return (v*(w*w-0.5f) - (qv.cross(v))*w + qv*(qv.dot(v)))*2;

	const FloatV two = FLoad(2.0f);
	const FloatV nhalf = FLoad(-0.5f);
	const Vec3V u = Vec3V_From_Vec4V(q);
	const FloatV w = V4GetW(q);
	const FloatV w2 = FScaleAdd(w, w, nhalf);
	const Vec3V a = V3Scale(v, w2);
	/*const Vec3V b = V3Scale(V3Cross(u, v), w);
	const Vec3V c = V3Scale(u, V3Dot(u, v));
	return V3Scale(V3Add(V3Sub(a, b), c), two);*/
	const Vec3V temp = V3NegScaleSub(V3Cross(u, v), w, a);
	return V3Scale(V3ScaleAdd(u, V3Dot(u, v), temp), two);
}

EHD QuatV QuatMul(const QuatV a, const QuatV b)
{
	const Vec4V imagA = a;
	const Vec4V imagB = b;
	const FloatV rA = V4GetW(a);
	const FloatV rB = V4GetW(b);

	const FloatV real = FSub(FMul(rA, rB), V4Dot3(imagA, imagB));
	const Vec4V v0 = V4Scale(imagA, rB);
	const Vec4V v1 = V4Scale(imagB, rA);
	const Vec4V v2 = V4Cross(imagA, imagB);
	const Vec4V imag = V4Add(V4Add(v0, v1), v2);

	return V4SetW(imag, real);
}

EHD QuatV QuatAdd(const QuatV a, const QuatV b)
{
	return V4Add(a, b);
}

EHD QuatV QuatNeg(const QuatV q)
{
	return V4Neg(q);
}

EHD QuatV QuatSub(const QuatV a, const QuatV b)
{
	return V4Sub(a, b);
}

EHD QuatV QuatScale(const QuatV a, const FloatV b)
{
	return V4Scale(a, b);
}

EHD QuatV QuatMerge(const FloatV* const floatVArray)
{
	return V4Merge(floatVArray);
}

EHD QuatV QuatMerge(const FloatVArg x, const FloatVArg y, const FloatVArg z, const FloatVArg w)
{
	return V4Merge(x, y, z, w);
}

EHD QuatV QuatIdentity()
{
	return V4SetW(V4Zero(), FOne());
}

EHD bool isFiniteQuatV(const QuatV q)
{
	return isFiniteVec4V(q);
}


EHD bool isValidQuatV(const QuatV q)
{
	const FloatV unitTolerance = FLoad(1e-4f);
	const FloatV tmp = FAbs(FSub(QuatLength(q), FOne()));
	const BoolV con = FIsGrtr(unitTolerance, tmp);
	return isFiniteVec4V(q) & (BAllEqTTTT(con) == 1);
}

EHD bool isSaneQuatV(const QuatV q)
{
	const FloatV unitTolerance = FLoad(1e-2f);
	const FloatV tmp = FAbs(FSub(QuatLength(q), FOne()));
	const BoolV con = FIsGrtr(unitTolerance, tmp);
	return isFiniteVec4V(q) & (BAllEqTTTT(con) == 1);
}


EHD Mat33V QuatGetMat33V(const QuatVArg q)
{
	// const FloatV two = FloatV_From_F32(2.0f);
	// const FloatV one = FOne();

	// const FloatV x = V4GetX(q);
	// const FloatV y = V4GetY(q);
	// const FloatV z = V4GetZ(q);
	// const Vec4V _q = V4Mul(q, two);
	//
	////const FloatV w = V4GetW(q);

	// const Vec4V t0 = V4Mul(_q, x); // 2xx, 2xy, 2xz, 2xw
	// const Vec4V t1 = V4Mul(_q, y); // 2xy, 2yy, 2yz, 2yw
	// const Vec4V t2 = V4Mul(_q, z); // 2xz, 2yz, 2zz, 2zw
	////const Vec4V t3 = V4Mul(_q, w); // 2xw, 2yw, 2zw, 2ww

	// const FloatV xx2 = V4GetX(t0);
	// const FloatV xy2 = V4GetY(t0);
	// const FloatV xz2 = V4GetZ(t0);
	// const FloatV xw2 = V4GetW(t0);

	// const FloatV yy2 = V4GetY(t1);
	// const FloatV yz2 = V4GetZ(t1);
	// const FloatV yw2 = V4GetW(t1);

	// const FloatV zz2 = V4GetZ(t2);
	// const FloatV zw2 = V4GetW(t2);

	////const FloatV ww2 = V4GetW(t3);

	// const FloatV c00 = FSub(one, FAdd(yy2, zz2));
	// const FloatV c01 = FSub(xy2, zw2);
	// const FloatV c02 = FAdd(xz2, yw2);

	// const FloatV c10 = FAdd(xy2, zw2);
	// const FloatV c11 = FSub(one, FAdd(xx2, zz2));
	// const FloatV c12 = FSub(yz2, xw2);

	// const FloatV c20 = FSub(xz2, yw2);
	// const FloatV c21 = FAdd(yz2, xw2);
	// const FloatV c22 = FSub(one, FAdd(xx2, yy2));

	// const Vec3V c0 = V3Merge(c00, c10, c20);
	// const Vec3V c1 = V3Merge(c01, c11, c21);
	// const Vec3V c2 = V3Merge(c02, c12, c22);

	// return Mat33V(c0, c1, c2);

	const FloatV one = FOne();
	const FloatV x = V4GetX(q);
	const FloatV y = V4GetY(q);
	const FloatV z = V4GetZ(q);
	const FloatV w = V4GetW(q);

	const FloatV x2 = FAdd(x, x);
	const FloatV y2 = FAdd(y, y);
	const FloatV z2 = FAdd(z, z);

	const FloatV xx = FMul(x2, x);
	const FloatV yy = FMul(y2, y);
	const FloatV zz = FMul(z2, z);

	const FloatV xy = FMul(x2, y);
	const FloatV xz = FMul(x2, z);
	const FloatV xw = FMul(x2, w);

	const FloatV yz = FMul(y2, z);
	const FloatV yw = FMul(y2, w);
	const FloatV zw = FMul(z2, w);

	const FloatV v = FSub(one, xx);

	const Vec3V column0 = V3Merge(FSub(FSub(one, yy), zz), FAdd(xy, zw), FSub(xz, yw));
	const Vec3V column1 = V3Merge(FSub(xy, zw), FSub(v, zz), FAdd(yz, xw));
	const Vec3V column2 = V3Merge(FAdd(xz, yw), FSub(yz, xw), FSub(v, yy));
	return Mat33V(column0, column1, column2);
}

EHD QuatV Mat33GetQuatV(const Mat33V& a)
{
	const FloatV one = FOne();
	const FloatV zero = FZero();
	const FloatV half = FLoad(0.5f);
	const FloatV two = FLoad(2.0f);
	const FloatV scale = FLoad(0.25f);
	const FloatV a00 = V3GetX(a.col0);
	const FloatV a11 = V3GetY(a.col1);
	const FloatV a22 = V3GetZ(a.col2);

	const FloatV a21 = V3GetZ(a.col1); // row=2, col=1;
	const FloatV a12 = V3GetY(a.col2); // row=1, col=2;
	const FloatV a02 = V3GetX(a.col2); // row=0, col=2;
	const FloatV a20 = V3GetZ(a.col0); // row=2, col=0;
	const FloatV a10 = V3GetY(a.col0); // row=1, col=0;
	const FloatV a01 = V3GetX(a.col1); // row=0, col=1;

	const Vec3V vec0 = V3Merge(a21, a02, a10);
	const Vec3V vec1 = V3Merge(a12, a20, a01);
	const Vec3V v = V3Sub(vec0, vec1);
	const Vec3V g = V3Add(vec0, vec1);

	const FloatV trace = FAdd(a00, FAdd(a11, a22));

	if(FAllGrtrOrEq(trace, zero))
	{
		const FloatV h = FSqrt(FAdd(trace, one));
		const FloatV w = FMul(half, h);
		const FloatV s = FMul(half, FRecip(h));
		const Vec3V u = V3Scale(v, s);
		return V4SetW(Vec4V_From_Vec3V(u), w);
	}
	else
	{
		const FloatV ntrace = FNeg(trace);
		const Vec3V d = V3Merge(a00, a11, a22);
		const BoolV con0 = BAllTrue3(V3IsGrtrOrEq(V3Splat(a00), d));
		const BoolV con1 = BAllTrue3(V3IsGrtrOrEq(V3Splat(a11), d));

		const FloatV t0 = FAdd(one, FScaleAdd(a00, two, ntrace));
		const FloatV t1 = FAdd(one, FScaleAdd(a11, two, ntrace));
		const FloatV t2 = FAdd(one, FScaleAdd(a22, two, ntrace));

		const FloatV t = FSel(con0, t0, FSel(con1, t1, t2));

		const FloatV h = FMul(two, FSqrt(t));
		const FloatV s = FRecip(h);
		const FloatV g0 = FMul(scale, h);
		const Vec3V vs = V3Scale(v, s);
		const Vec3V gs = V3Scale(g, s);
		const FloatV gsx = V3GetX(gs);
		const FloatV gsy = V3GetY(gs);
		const FloatV gsz = V3GetZ(gs);
		// vs.x= (a21 - a12)*s; vs.y=(a02 - a20)*s; vs.z=(a10 - a01)*s;
		// gs.x= (a21 + a12)*s; gs.y=(a02 + a20)*s; gs.z=(a10 + a01)*s;
		const Vec4V v0 = V4Merge(g0, gsz, gsy, V3GetX(vs));
		const Vec4V v1 = V4Merge(gsz, g0, gsx, V3GetY(vs));
		const Vec4V v2 = V4Merge(gsy, gsx, g0, V3GetZ(vs));
		return V4Sel(con0, v0, V4Sel(con1, v1, v2));
	}
}

} // namespace aos
} // namespace eng



// ======== PxVecTransform.h (PhysX 5.6.1, 전처리 분기: SSE2·clang·linux) ========



namespace eng
{
namespace aos
{

class PxTransformV
{
  public:
	QuatV q;
	Vec3V p;

	EHD PxTransformV(const PxTransform& transform)
	{
		// PT: this is now similar to loadTransformU below.
		q = QuatVLoadU(&transform.q.x);
		p = V3LoadU(&transform.p.x);
	}

	EHD PxTransformV(const Vec3VArg p0 = V3Zero(), const QuatVArg q0 = QuatIdentity()) : q(q0), p(p0)
	{
		PX_ASSERT(isSaneQuatV(q0));
	}

	EHD PxTransformV operator*(const PxTransformV& x) const
	{
		PX_ASSERT(x.isSane());
		return transform(x);
	}

	EHD PxTransformV getInverse() const
	{
		PX_ASSERT(isFinite());
		// return PxTransform(q.rotateInv(-p),q.getConjugate());
		return PxTransformV(QuatRotateInv(q, V3Neg(p)), QuatConjugate(q));
	}

	EHD void invalidate()
	{
		p = V3Splat(FMax());
		q = QuatIdentity();
	}

	EHD Vec3V transform(const Vec3VArg input) const
	{
		PX_ASSERT(isFinite());
		// return q.rotate(input) + p;
		return QuatTransform(q, p, input);
	}

	EHD Vec3V transformInv(const Vec3VArg input) const
	{
		PX_ASSERT(isFinite());
		// return q.rotateInv(input-p);
		return QuatRotateInv(q, V3Sub(input, p));
	}

	EHD Vec3V rotate(const Vec3VArg input) const
	{
		PX_ASSERT(isFinite());
		// return q.rotate(input);
		return QuatRotate(q, input);
	}

	// PT: avoid some multiplies when immediately normalizing a rotated vector
	EHD Vec3V rotateAndNormalize(const Vec3VArg input) const
	{
		PX_ASSERT(isFinite());
		return QuatRotateAndNormalize(q, input);
	}

	EHD Vec3V rotateInv(const Vec3VArg input) const
	{
		PX_ASSERT(isFinite());
		// return q.rotateInv(input);
		return QuatRotateInv(q, input);
	}

	//! Transform transform to parent (returns compound transform: first src, then *this)
	EHD PxTransformV transform(const PxTransformV& src) const
	{
		PX_ASSERT(src.isSane());
		PX_ASSERT(isSane());
		// src = [srct, srcr] -> [r*srct + t, r*srcr]
		// return PxTransform(q.rotate(src.p) + p, q*src.q);
		return PxTransformV(V3Add(QuatRotate(q, src.p), p), QuatMul(q, src.q));
	}


	/**
	\brief returns true if finite and q is a unit quaternion
	*/
	EHD bool isValid() const
	{
		// return p.isFinite() && q.isFinite() && q.isValid();
		return isFiniteVec3V(p) & isFiniteQuatV(q) & isValidQuatV(q);
	}

	/**
	\brief returns true if finite and quat magnitude is reasonably close to unit to allow for some accumulation of error
	vs isValid
	*/

	EHD bool isSane() const
	{
		// return isFinite() && q.isSane();
		return isFinite() & isSaneQuatV(q);
	}

	/**
	\brief returns true if all elems are finite (not NAN or INF, etc.)
	*/
	EHD bool isFinite() const
	{
		// return p.isFinite() && q.isFinite();
		return isFiniteVec3V(p) & isFiniteQuatV(q);
	}


	//! Transform transform from parent (returns compound transform: first src, then this->inverse)
	EHD PxTransformV transformInv(const PxTransformV& src) const
	{
		PX_ASSERT(src.isSane());
		PX_ASSERT(isFinite());
		// src = [srct, srcr] -> [r^-1*(srct-t), r^-1*srcr]
		/*PxQuat qinv = q.getConjugate();
		return PxTransform(qinv.rotate(src.p - p), qinv*src.q);*/
		const QuatV qinv = QuatConjugate(q);
		const Vec3V v = QuatRotate(qinv, V3Sub(src.p, p));
		const QuatV rot = QuatMul(qinv, src.q);
		return PxTransformV(v, rot);
	}

	static EHD PxTransformV createIdentity()
	{
		return PxTransformV(V3Zero());
	}
};

EHD PxTransformV loadTransformA(const PxTransform& transform)
{
	const QuatV q0 = QuatVLoadA(&transform.q.x);
	const Vec3V p0 = V3LoadA(&transform.p.x);

	return PxTransformV(p0, q0);
}

EHD PxTransformV loadTransformU(const PxTransform& transform)
{
	const QuatV q0 = QuatVLoadU(&transform.q.x);
	const Vec3V p0 = V3LoadU(&transform.p.x);

	return PxTransformV(p0, q0);
}

class PxMatTransformV
{
  public:
	Mat33V rot;
	Vec3V p;

	EHD PxMatTransformV()
	{
		p = V3Zero();
		rot = M33Identity();
	}
	EHD PxMatTransformV(const Vec3VArg _p, const Mat33V& _rot)
	{
		p = _p;
		rot = _rot;
	}

	EHD PxMatTransformV(const PxTransformV& other)
	{
		p = other.p;
		QuatGetMat33V(other.q, rot.col0, rot.col1, rot.col2);
	}

	EHD PxMatTransformV(const Vec3VArg _p, const QuatV& quat)
	{
		p = _p;
		QuatGetMat33V(quat, rot.col0, rot.col1, rot.col2);
	}

	EHD Vec3V getCol0() const
	{
		return rot.col0;
	}

	EHD Vec3V getCol1() const
	{
		return rot.col1;
	}

	EHD Vec3V getCol2() const
	{
		return rot.col2;
	}

	EHD void setCol0(const Vec3VArg col0)
	{
		rot.col0 = col0;
	}

	EHD void setCol1(const Vec3VArg col1)
	{
		rot.col1 = col1;
	}

	EHD void setCol2(const Vec3VArg col2)
	{
		rot.col2 = col2;
	}

	EHD Vec3V transform(const Vec3VArg input) const
	{
		return V3Add(p, M33MulV3(rot, input));
	}

	EHD Vec3V transformInv(const Vec3VArg input) const
	{
		return M33TrnspsMulV3(rot, V3Sub(input, p)); // QuatRotateInv(q, V3Sub(input, p));
	}

	EHD Vec3V rotate(const Vec3VArg input) const
	{
		return M33MulV3(rot, input);
	}

	EHD Vec3V rotateInv(const Vec3VArg input) const
	{
		return M33TrnspsMulV3(rot, input);
	}

	EHD PxMatTransformV transformInv(const PxMatTransformV& src) const
	{

		const Vec3V v = M33TrnspsMulV3(rot, V3Sub(src.p, p));
		const Mat33V mat = M33MulM33(M33Trnsps(rot), src.rot);
		return PxMatTransformV(v, mat);
	}
};
}
} // namespace eng


