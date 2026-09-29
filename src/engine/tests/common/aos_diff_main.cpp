// 생성물 (gen_aos_test.py)
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <xmmintrin.h>
#include <string>
#include <vector>
extern "C" void px_QuatVLoadXYZW(const float*, float*); extern "C" void eng_QuatVLoadXYZW(const float*, float*);
extern "C" void px_Vec3V_From_Vec4V(const float*, float*); extern "C" void eng_Vec3V_From_Vec4V(const float*, float*);
extern "C" void px_Vec3V_From_Vec4V_WUndefined(const float*, float*); extern "C" void eng_Vec3V_From_Vec4V_WUndefined(const float*, float*);
extern "C" void px_Vec4V_From_Vec3V(const float*, float*); extern "C" void eng_Vec4V_From_Vec3V(const float*, float*);
extern "C" void px_Vec4V_From_VecU32V(const float*, float*); extern "C" void eng_Vec4V_From_VecU32V(const float*, float*);
extern "C" void px_Vec4V_From_VecI32V(const float*, float*); extern "C" void eng_Vec4V_From_VecI32V(const float*, float*);
extern "C" void px_VecU32V_From_BoolV(const float*, float*); extern "C" void eng_VecU32V_From_BoolV(const float*, float*);
extern "C" void px_VecI32V_From_Vec4V(const float*, float*); extern "C" void eng_VecI32V_From_Vec4V(const float*, float*);
extern "C" void px_isFiniteFloatV(const float*, float*); extern "C" void eng_isFiniteFloatV(const float*, float*);
extern "C" void px_isFiniteVec3V(const float*, float*); extern "C" void eng_isFiniteVec3V(const float*, float*);
extern "C" void px_isFiniteVec4V(const float*, float*); extern "C" void eng_isFiniteVec4V(const float*, float*);
extern "C" void px_isValidVec3V(const float*, float*); extern "C" void eng_isValidVec3V(const float*, float*);
extern "C" void px_FZero(const float*, float*); extern "C" void eng_FZero(const float*, float*);
extern "C" void px_FOne(const float*, float*); extern "C" void eng_FOne(const float*, float*);
extern "C" void px_FHalf(const float*, float*); extern "C" void eng_FHalf(const float*, float*);
extern "C" void px_FEps(const float*, float*); extern "C" void eng_FEps(const float*, float*);
extern "C" void px_FMax(const float*, float*); extern "C" void eng_FMax(const float*, float*);
extern "C" void px_FNegMax(const float*, float*); extern "C" void eng_FNegMax(const float*, float*);
extern "C" void px_FEps6(const float*, float*); extern "C" void eng_FEps6(const float*, float*);
extern "C" void px_FNeg(const float*, float*); extern "C" void eng_FNeg(const float*, float*);
extern "C" void px_FAdd(const float*, float*); extern "C" void eng_FAdd(const float*, float*);
extern "C" void px_FSub(const float*, float*); extern "C" void eng_FSub(const float*, float*);
extern "C" void px_FMul(const float*, float*); extern "C" void eng_FMul(const float*, float*);
extern "C" void px_FDiv(const float*, float*); extern "C" void eng_FDiv(const float*, float*);
extern "C" void px_FDivFast(const float*, float*); extern "C" void eng_FDivFast(const float*, float*);
extern "C" void px_FRecip(const float*, float*); extern "C" void eng_FRecip(const float*, float*);
extern "C" void px_FRecipFast(const float*, float*); extern "C" void eng_FRecipFast(const float*, float*);
extern "C" void px_FRsqrt(const float*, float*); extern "C" void eng_FRsqrt(const float*, float*);
extern "C" void px_FRsqrtFast(const float*, float*); extern "C" void eng_FRsqrtFast(const float*, float*);
extern "C" void px_FSqrt(const float*, float*); extern "C" void eng_FSqrt(const float*, float*);
extern "C" void px_FScaleAdd(const float*, float*); extern "C" void eng_FScaleAdd(const float*, float*);
extern "C" void px_FNegScaleSub(const float*, float*); extern "C" void eng_FNegScaleSub(const float*, float*);
extern "C" void px_FAbs(const float*, float*); extern "C" void eng_FAbs(const float*, float*);
extern "C" void px_FSel(const float*, float*); extern "C" void eng_FSel(const float*, float*);
extern "C" void px_FIsGrtr(const float*, float*); extern "C" void eng_FIsGrtr(const float*, float*);
extern "C" void px_FIsGrtrOrEq(const float*, float*); extern "C" void eng_FIsGrtrOrEq(const float*, float*);
extern "C" void px_FIsEq(const float*, float*); extern "C" void eng_FIsEq(const float*, float*);
extern "C" void px_FMin(const float*, float*); extern "C" void eng_FMin(const float*, float*);
extern "C" void px_FClamp(const float*, float*); extern "C" void eng_FClamp(const float*, float*);
extern "C" void px_FAllGrtr(const float*, float*); extern "C" void eng_FAllGrtr(const float*, float*);
extern "C" void px_FAllGrtrOrEq(const float*, float*); extern "C" void eng_FAllGrtrOrEq(const float*, float*);
extern "C" void px_FAllEq(const float*, float*); extern "C" void eng_FAllEq(const float*, float*);
extern "C" void px_FOutOfBounds(const float*, float*); extern "C" void eng_FOutOfBounds(const float*, float*);
extern "C" void px_FInBounds(const float*, float*); extern "C" void eng_FInBounds(const float*, float*);
extern "C" void px_FRound(const float*, float*); extern "C" void eng_FRound(const float*, float*);
extern "C" void px_FSin(const float*, float*); extern "C" void eng_FSin(const float*, float*);
extern "C" void px_FCos(const float*, float*); extern "C" void eng_FCos(const float*, float*);
extern "C" void px_V3Splat(const float*, float*); extern "C" void eng_V3Splat(const float*, float*);
extern "C" void px_V3Merge(const float*, float*); extern "C" void eng_V3Merge(const float*, float*);
extern "C" void px_V3UnitX(const float*, float*); extern "C" void eng_V3UnitX(const float*, float*);
extern "C" void px_V3UnitY(const float*, float*); extern "C" void eng_V3UnitY(const float*, float*);
extern "C" void px_V3UnitZ(const float*, float*); extern "C" void eng_V3UnitZ(const float*, float*);
extern "C" void px_V3GetX(const float*, float*); extern "C" void eng_V3GetX(const float*, float*);
extern "C" void px_V3GetY(const float*, float*); extern "C" void eng_V3GetY(const float*, float*);
extern "C" void px_V3GetZ(const float*, float*); extern "C" void eng_V3GetZ(const float*, float*);
extern "C" void px_V3SetX(const float*, float*); extern "C" void eng_V3SetX(const float*, float*);
extern "C" void px_V3SetY(const float*, float*); extern "C" void eng_V3SetY(const float*, float*);
extern "C" void px_V3SetZ(const float*, float*); extern "C" void eng_V3SetZ(const float*, float*);
extern "C" void px_V3ReadX(const float*, float*); extern "C" void eng_V3ReadX(const float*, float*);
extern "C" void px_V3ReadY(const float*, float*); extern "C" void eng_V3ReadY(const float*, float*);
extern "C" void px_V3ReadZ(const float*, float*); extern "C" void eng_V3ReadZ(const float*, float*);
extern "C" void px_V3ColX(const float*, float*); extern "C" void eng_V3ColX(const float*, float*);
extern "C" void px_V3ColY(const float*, float*); extern "C" void eng_V3ColY(const float*, float*);
extern "C" void px_V3ColZ(const float*, float*); extern "C" void eng_V3ColZ(const float*, float*);
extern "C" void px_V3Zero(const float*, float*); extern "C" void eng_V3Zero(const float*, float*);
extern "C" void px_V3One(const float*, float*); extern "C" void eng_V3One(const float*, float*);
extern "C" void px_V3Eps(const float*, float*); extern "C" void eng_V3Eps(const float*, float*);
extern "C" void px_V3Neg(const float*, float*); extern "C" void eng_V3Neg(const float*, float*);
extern "C" void px_V3Add(const float*, float*); extern "C" void eng_V3Add(const float*, float*);
extern "C" void px_V3Sub(const float*, float*); extern "C" void eng_V3Sub(const float*, float*);
extern "C" void px_V3Scale(const float*, float*); extern "C" void eng_V3Scale(const float*, float*);
extern "C" void px_V3Mul(const float*, float*); extern "C" void eng_V3Mul(const float*, float*);
extern "C" void px_V3ScaleInv(const float*, float*); extern "C" void eng_V3ScaleInv(const float*, float*);
extern "C" void px_V3Div(const float*, float*); extern "C" void eng_V3Div(const float*, float*);
extern "C" void px_V3ScaleInvFast(const float*, float*); extern "C" void eng_V3ScaleInvFast(const float*, float*);
extern "C" void px_V3DivFast(const float*, float*); extern "C" void eng_V3DivFast(const float*, float*);
extern "C" void px_V3Recip(const float*, float*); extern "C" void eng_V3Recip(const float*, float*);
extern "C" void px_V3RecipFast(const float*, float*); extern "C" void eng_V3RecipFast(const float*, float*);
extern "C" void px_V3Rsqrt(const float*, float*); extern "C" void eng_V3Rsqrt(const float*, float*);
extern "C" void px_V3RsqrtFast(const float*, float*); extern "C" void eng_V3RsqrtFast(const float*, float*);
extern "C" void px_V3ScaleAdd(const float*, float*); extern "C" void eng_V3ScaleAdd(const float*, float*);
extern "C" void px_V3NegScaleSub(const float*, float*); extern "C" void eng_V3NegScaleSub(const float*, float*);
extern "C" void px_V3MulAdd(const float*, float*); extern "C" void eng_V3MulAdd(const float*, float*);
extern "C" void px_V3NegMulSub(const float*, float*); extern "C" void eng_V3NegMulSub(const float*, float*);
extern "C" void px_V3Abs(const float*, float*); extern "C" void eng_V3Abs(const float*, float*);
extern "C" void px_V3Dot(const float*, float*); extern "C" void eng_V3Dot(const float*, float*);
extern "C" void px_V3Cross(const float*, float*); extern "C" void eng_V3Cross(const float*, float*);
extern "C" void px_V3Length(const float*, float*); extern "C" void eng_V3Length(const float*, float*);
extern "C" void px_V3LengthSq(const float*, float*); extern "C" void eng_V3LengthSq(const float*, float*);
extern "C" void px_V3Normalize(const float*, float*); extern "C" void eng_V3Normalize(const float*, float*);
extern "C" void px_V3NormalizeSafe(const float*, float*); extern "C" void eng_V3NormalizeSafe(const float*, float*);
extern "C" void px_V3SumElems(const float*, float*); extern "C" void eng_V3SumElems(const float*, float*);
extern "C" void px_V3Sel(const float*, float*); extern "C" void eng_V3Sel(const float*, float*);
extern "C" void px_V3IsGrtr(const float*, float*); extern "C" void eng_V3IsGrtr(const float*, float*);
extern "C" void px_V3IsGrtrOrEq(const float*, float*); extern "C" void eng_V3IsGrtrOrEq(const float*, float*);
extern "C" void px_V3IsEq(const float*, float*); extern "C" void eng_V3IsEq(const float*, float*);
extern "C" void px_V3Max(const float*, float*); extern "C" void eng_V3Max(const float*, float*);
extern "C" void px_V3Min(const float*, float*); extern "C" void eng_V3Min(const float*, float*);
extern "C" void px_V3ExtractMax(const float*, float*); extern "C" void eng_V3ExtractMax(const float*, float*);
extern "C" void px_V3ExtractMin(const float*, float*); extern "C" void eng_V3ExtractMin(const float*, float*);
extern "C" void px_V3Clamp(const float*, float*); extern "C" void eng_V3Clamp(const float*, float*);
extern "C" void px_V3Sign(const float*, float*); extern "C" void eng_V3Sign(const float*, float*);
extern "C" void px_V3AllGrtr(const float*, float*); extern "C" void eng_V3AllGrtr(const float*, float*);
extern "C" void px_V3AllGrtrOrEq(const float*, float*); extern "C" void eng_V3AllGrtrOrEq(const float*, float*);
extern "C" void px_V3AllEq(const float*, float*); extern "C" void eng_V3AllEq(const float*, float*);
extern "C" void px_V3OutOfBounds(const float*, float*); extern "C" void eng_V3OutOfBounds(const float*, float*);
extern "C" void px_V3InBounds(const float*, float*); extern "C" void eng_V3InBounds(const float*, float*);
extern "C" void px_V3Round(const float*, float*); extern "C" void eng_V3Round(const float*, float*);
extern "C" void px_V3Sin(const float*, float*); extern "C" void eng_V3Sin(const float*, float*);
extern "C" void px_V3Cos(const float*, float*); extern "C" void eng_V3Cos(const float*, float*);
extern "C" void px_V3PermYZZ(const float*, float*); extern "C" void eng_V3PermYZZ(const float*, float*);
extern "C" void px_V3PermXYX(const float*, float*); extern "C" void eng_V3PermXYX(const float*, float*);
extern "C" void px_V3PermYZX(const float*, float*); extern "C" void eng_V3PermYZX(const float*, float*);
extern "C" void px_V3PermZXY(const float*, float*); extern "C" void eng_V3PermZXY(const float*, float*);
extern "C" void px_V3PermZZY(const float*, float*); extern "C" void eng_V3PermZZY(const float*, float*);
extern "C" void px_V3PermYXX(const float*, float*); extern "C" void eng_V3PermYXX(const float*, float*);
extern "C" void px_V3Perm_Zero_1Z_0Y(const float*, float*); extern "C" void eng_V3Perm_Zero_1Z_0Y(const float*, float*);
extern "C" void px_V3Perm_0Z_Zero_1X(const float*, float*); extern "C" void eng_V3Perm_0Z_Zero_1X(const float*, float*);
extern "C" void px_V3Perm_1Y_0X_Zero(const float*, float*); extern "C" void eng_V3Perm_1Y_0X_Zero(const float*, float*);
extern "C" void px_V4Splat(const float*, float*); extern "C" void eng_V4Splat(const float*, float*);
extern "C" void px_V4Merge(const float*, float*); extern "C" void eng_V4Merge(const float*, float*);
extern "C" void px_V4MergeW(const float*, float*); extern "C" void eng_V4MergeW(const float*, float*);
extern "C" void px_V4MergeZ(const float*, float*); extern "C" void eng_V4MergeZ(const float*, float*);
extern "C" void px_V4MergeY(const float*, float*); extern "C" void eng_V4MergeY(const float*, float*);
extern "C" void px_V4MergeX(const float*, float*); extern "C" void eng_V4MergeX(const float*, float*);
extern "C" void px_V4UnpackXY(const float*, float*); extern "C" void eng_V4UnpackXY(const float*, float*);
extern "C" void px_V4UnpackZW(const float*, float*); extern "C" void eng_V4UnpackZW(const float*, float*);
extern "C" void px_V4UnitW(const float*, float*); extern "C" void eng_V4UnitW(const float*, float*);
extern "C" void px_V4UnitY(const float*, float*); extern "C" void eng_V4UnitY(const float*, float*);
extern "C" void px_V4UnitZ(const float*, float*); extern "C" void eng_V4UnitZ(const float*, float*);
extern "C" void px_V4GetX(const float*, float*); extern "C" void eng_V4GetX(const float*, float*);
extern "C" void px_V4GetY(const float*, float*); extern "C" void eng_V4GetY(const float*, float*);
extern "C" void px_V4GetZ(const float*, float*); extern "C" void eng_V4GetZ(const float*, float*);
extern "C" void px_V4GetW(const float*, float*); extern "C" void eng_V4GetW(const float*, float*);
extern "C" void px_V4SetX(const float*, float*); extern "C" void eng_V4SetX(const float*, float*);
extern "C" void px_V4SetY(const float*, float*); extern "C" void eng_V4SetY(const float*, float*);
extern "C" void px_V4SetZ(const float*, float*); extern "C" void eng_V4SetZ(const float*, float*);
extern "C" void px_V4SetW(const float*, float*); extern "C" void eng_V4SetW(const float*, float*);
extern "C" void px_V4ClearW(const float*, float*); extern "C" void eng_V4ClearW(const float*, float*);
extern "C" void px_V4ReadX(const float*, float*); extern "C" void eng_V4ReadX(const float*, float*);
extern "C" void px_V4ReadY(const float*, float*); extern "C" void eng_V4ReadY(const float*, float*);
extern "C" void px_V4ReadZ(const float*, float*); extern "C" void eng_V4ReadZ(const float*, float*);
extern "C" void px_V4ReadW(const float*, float*); extern "C" void eng_V4ReadW(const float*, float*);
extern "C" void px_V4Zero(const float*, float*); extern "C" void eng_V4Zero(const float*, float*);
extern "C" void px_V4One(const float*, float*); extern "C" void eng_V4One(const float*, float*);
extern "C" void px_V4Eps(const float*, float*); extern "C" void eng_V4Eps(const float*, float*);
extern "C" void px_V4Neg(const float*, float*); extern "C" void eng_V4Neg(const float*, float*);
extern "C" void px_V4Add(const float*, float*); extern "C" void eng_V4Add(const float*, float*);
extern "C" void px_V4Sub(const float*, float*); extern "C" void eng_V4Sub(const float*, float*);
extern "C" void px_V4Scale(const float*, float*); extern "C" void eng_V4Scale(const float*, float*);
extern "C" void px_V4Mul(const float*, float*); extern "C" void eng_V4Mul(const float*, float*);
extern "C" void px_V4ScaleInv(const float*, float*); extern "C" void eng_V4ScaleInv(const float*, float*);
extern "C" void px_V4Div(const float*, float*); extern "C" void eng_V4Div(const float*, float*);
extern "C" void px_V4ScaleInvFast(const float*, float*); extern "C" void eng_V4ScaleInvFast(const float*, float*);
extern "C" void px_V4DivFast(const float*, float*); extern "C" void eng_V4DivFast(const float*, float*);
extern "C" void px_V4Recip(const float*, float*); extern "C" void eng_V4Recip(const float*, float*);
extern "C" void px_V4RecipFast(const float*, float*); extern "C" void eng_V4RecipFast(const float*, float*);
extern "C" void px_V4Rsqrt(const float*, float*); extern "C" void eng_V4Rsqrt(const float*, float*);
extern "C" void px_V4RsqrtFast(const float*, float*); extern "C" void eng_V4RsqrtFast(const float*, float*);
extern "C" void px_V4ScaleAdd(const float*, float*); extern "C" void eng_V4ScaleAdd(const float*, float*);
extern "C" void px_V4NegScaleSub(const float*, float*); extern "C" void eng_V4NegScaleSub(const float*, float*);
extern "C" void px_V4MulAdd(const float*, float*); extern "C" void eng_V4MulAdd(const float*, float*);
extern "C" void px_V4NegMulSub(const float*, float*); extern "C" void eng_V4NegMulSub(const float*, float*);
extern "C" void px_V4Abs(const float*, float*); extern "C" void eng_V4Abs(const float*, float*);
extern "C" void px_V4Andc(const float*, float*); extern "C" void eng_V4Andc(const float*, float*);
extern "C" void px_V4Dot(const float*, float*); extern "C" void eng_V4Dot(const float*, float*);
extern "C" void px_V4Dot3(const float*, float*); extern "C" void eng_V4Dot3(const float*, float*);
extern "C" void px_V4Cross(const float*, float*); extern "C" void eng_V4Cross(const float*, float*);
extern "C" void px_V4Length(const float*, float*); extern "C" void eng_V4Length(const float*, float*);
extern "C" void px_V4LengthSq(const float*, float*); extern "C" void eng_V4LengthSq(const float*, float*);
extern "C" void px_V4Normalize(const float*, float*); extern "C" void eng_V4Normalize(const float*, float*);
extern "C" void px_V4NormalizeSafe(const float*, float*); extern "C" void eng_V4NormalizeSafe(const float*, float*);
extern "C" void px_V4NormalizeFast(const float*, float*); extern "C" void eng_V4NormalizeFast(const float*, float*);
extern "C" void px_V4Sel(const float*, float*); extern "C" void eng_V4Sel(const float*, float*);
extern "C" void px_V4IsGrtr(const float*, float*); extern "C" void eng_V4IsGrtr(const float*, float*);
extern "C" void px_V4IsGrtrOrEq(const float*, float*); extern "C" void eng_V4IsGrtrOrEq(const float*, float*);
extern "C" void px_V4IsEq(const float*, float*); extern "C" void eng_V4IsEq(const float*, float*);
extern "C" void px_V4Max(const float*, float*); extern "C" void eng_V4Max(const float*, float*);
extern "C" void px_V4Min(const float*, float*); extern "C" void eng_V4Min(const float*, float*);
extern "C" void px_V4ExtractMax(const float*, float*); extern "C" void eng_V4ExtractMax(const float*, float*);
extern "C" void px_V4ExtractMin(const float*, float*); extern "C" void eng_V4ExtractMin(const float*, float*);
extern "C" void px_V4Clamp(const float*, float*); extern "C" void eng_V4Clamp(const float*, float*);
extern "C" void px_V4AllGrtr(const float*, float*); extern "C" void eng_V4AllGrtr(const float*, float*);
extern "C" void px_V4AllGrtrOrEq(const float*, float*); extern "C" void eng_V4AllGrtrOrEq(const float*, float*);
extern "C" void px_V4AllGrtrOrEq3(const float*, float*); extern "C" void eng_V4AllGrtrOrEq3(const float*, float*);
extern "C" void px_V4AllEq(const float*, float*); extern "C" void eng_V4AllEq(const float*, float*);
extern "C" void px_V4AnyGrtr3(const float*, float*); extern "C" void eng_V4AnyGrtr3(const float*, float*);
extern "C" void px_V4Round(const float*, float*); extern "C" void eng_V4Round(const float*, float*);
extern "C" void px_V4Sin(const float*, float*); extern "C" void eng_V4Sin(const float*, float*);
extern "C" void px_V4Cos(const float*, float*); extern "C" void eng_V4Cos(const float*, float*);
extern "C" void px_V4PermYXWZ(const float*, float*); extern "C" void eng_V4PermYXWZ(const float*, float*);
extern "C" void px_V4PermXZXZ(const float*, float*); extern "C" void eng_V4PermXZXZ(const float*, float*);
extern "C" void px_V4PermYWYW(const float*, float*); extern "C" void eng_V4PermYWYW(const float*, float*);
extern "C" void px_V4PermYZXW(const float*, float*); extern "C" void eng_V4PermYZXW(const float*, float*);
extern "C" void px_V4PermZWXY(const float*, float*); extern "C" void eng_V4PermZWXY(const float*, float*);
extern "C" void px_QuatV_From_RotationAxisAngle(const float*, float*); extern "C" void eng_QuatV_From_RotationAxisAngle(const float*, float*);
extern "C" void px_QuatNormalize(const float*, float*); extern "C" void eng_QuatNormalize(const float*, float*);
extern "C" void px_QuatLength(const float*, float*); extern "C" void eng_QuatLength(const float*, float*);
extern "C" void px_QuatLengthSq(const float*, float*); extern "C" void eng_QuatLengthSq(const float*, float*);
extern "C" void px_QuatDot(const float*, float*); extern "C" void eng_QuatDot(const float*, float*);
extern "C" void px_QuatConjugate(const float*, float*); extern "C" void eng_QuatConjugate(const float*, float*);
extern "C" void px_QuatGetImaginaryPart(const float*, float*); extern "C" void eng_QuatGetImaginaryPart(const float*, float*);
extern "C" void px_QuatGetMat33V(const float*, float*); extern "C" void eng_QuatGetMat33V(const float*, float*);
extern "C" void px_Mat33GetQuatV(const float*, float*); extern "C" void eng_Mat33GetQuatV(const float*, float*);
extern "C" void px_QuatGetBasisVector0(const float*, float*); extern "C" void eng_QuatGetBasisVector0(const float*, float*);
extern "C" void px_QuatGetBasisVector1(const float*, float*); extern "C" void eng_QuatGetBasisVector1(const float*, float*);
extern "C" void px_QuatGetBasisVector2(const float*, float*); extern "C" void eng_QuatGetBasisVector2(const float*, float*);
extern "C" void px_QuatRotate(const float*, float*); extern "C" void eng_QuatRotate(const float*, float*);
extern "C" void px_QuatRotateInv(const float*, float*); extern "C" void eng_QuatRotateInv(const float*, float*);
extern "C" void px_QuatMul(const float*, float*); extern "C" void eng_QuatMul(const float*, float*);
extern "C" void px_QuatAdd(const float*, float*); extern "C" void eng_QuatAdd(const float*, float*);
extern "C" void px_QuatNeg(const float*, float*); extern "C" void eng_QuatNeg(const float*, float*);
extern "C" void px_QuatSub(const float*, float*); extern "C" void eng_QuatSub(const float*, float*);
extern "C" void px_QuatScale(const float*, float*); extern "C" void eng_QuatScale(const float*, float*);
extern "C" void px_QuatMerge(const float*, float*); extern "C" void eng_QuatMerge(const float*, float*);
extern "C" void px_QuatIdentity(const float*, float*); extern "C" void eng_QuatIdentity(const float*, float*);
extern "C" void px_isFiniteQuatV(const float*, float*); extern "C" void eng_isFiniteQuatV(const float*, float*);
extern "C" void px_isValidQuatV(const float*, float*); extern "C" void eng_isValidQuatV(const float*, float*);
extern "C" void px_isSaneQuatV(const float*, float*); extern "C" void eng_isSaneQuatV(const float*, float*);
extern "C" void px_BFFFF(const float*, float*); extern "C" void eng_BFFFF(const float*, float*);
extern "C" void px_BFFFT(const float*, float*); extern "C" void eng_BFFFT(const float*, float*);
extern "C" void px_BFFTF(const float*, float*); extern "C" void eng_BFFTF(const float*, float*);
extern "C" void px_BFFTT(const float*, float*); extern "C" void eng_BFFTT(const float*, float*);
extern "C" void px_BFTFF(const float*, float*); extern "C" void eng_BFTFF(const float*, float*);
extern "C" void px_BFTFT(const float*, float*); extern "C" void eng_BFTFT(const float*, float*);
extern "C" void px_BFTTF(const float*, float*); extern "C" void eng_BFTTF(const float*, float*);
extern "C" void px_BFTTT(const float*, float*); extern "C" void eng_BFTTT(const float*, float*);
extern "C" void px_BTFFF(const float*, float*); extern "C" void eng_BTFFF(const float*, float*);
extern "C" void px_BTFFT(const float*, float*); extern "C" void eng_BTFFT(const float*, float*);
extern "C" void px_BTFTF(const float*, float*); extern "C" void eng_BTFTF(const float*, float*);
extern "C" void px_BTFTT(const float*, float*); extern "C" void eng_BTFTT(const float*, float*);
extern "C" void px_BTTFF(const float*, float*); extern "C" void eng_BTTFF(const float*, float*);
extern "C" void px_BTTFT(const float*, float*); extern "C" void eng_BTTFT(const float*, float*);
extern "C" void px_BTTTF(const float*, float*); extern "C" void eng_BTTTF(const float*, float*);
extern "C" void px_BTTTT(const float*, float*); extern "C" void eng_BTTTT(const float*, float*);
extern "C" void px_BWMask(const float*, float*); extern "C" void eng_BWMask(const float*, float*);
extern "C" void px_BXMask(const float*, float*); extern "C" void eng_BXMask(const float*, float*);
extern "C" void px_BYMask(const float*, float*); extern "C" void eng_BYMask(const float*, float*);
extern "C" void px_BZMask(const float*, float*); extern "C" void eng_BZMask(const float*, float*);
extern "C" void px_BGetX(const float*, float*); extern "C" void eng_BGetX(const float*, float*);
extern "C" void px_BGetY(const float*, float*); extern "C" void eng_BGetY(const float*, float*);
extern "C" void px_BGetZ(const float*, float*); extern "C" void eng_BGetZ(const float*, float*);
extern "C" void px_BGetW(const float*, float*); extern "C" void eng_BGetW(const float*, float*);
extern "C" void px_BAnd(const float*, float*); extern "C" void eng_BAnd(const float*, float*);
extern "C" void px_BOr(const float*, float*); extern "C" void eng_BOr(const float*, float*);
extern "C" void px_BNot(const float*, float*); extern "C" void eng_BNot(const float*, float*);
extern "C" void px_BAllTrue4(const float*, float*); extern "C" void eng_BAllTrue4(const float*, float*);
extern "C" void px_BAnyTrue4(const float*, float*); extern "C" void eng_BAnyTrue4(const float*, float*);
extern "C" void px_BAllTrue3(const float*, float*); extern "C" void eng_BAllTrue3(const float*, float*);
extern "C" void px_BAnyTrue3(const float*, float*); extern "C" void eng_BAnyTrue3(const float*, float*);
extern "C" void px_BAllEq(const float*, float*); extern "C" void eng_BAllEq(const float*, float*);
extern "C" void px_BAllEqTTTT(const float*, float*); extern "C" void eng_BAllEqTTTT(const float*, float*);
extern "C" void px_BAllEqFFFF(const float*, float*); extern "C" void eng_BAllEqFFFF(const float*, float*);
extern "C" void px_VecI32V_Zero(const float*, float*); extern "C" void eng_VecI32V_Zero(const float*, float*);
extern "C" void px_VecI32V_One(const float*, float*); extern "C" void eng_VecI32V_One(const float*, float*);
extern "C" void px_VecI32V_Two(const float*, float*); extern "C" void eng_VecI32V_Two(const float*, float*);
extern "C" void px_VecI32V_MinusOne(const float*, float*); extern "C" void eng_VecI32V_MinusOne(const float*, float*);
extern "C" void px_VecI32V_Add(const float*, float*); extern "C" void eng_VecI32V_Add(const float*, float*);
extern "C" void px_VecI32V_Or(const float*, float*); extern "C" void eng_VecI32V_Or(const float*, float*);
extern "C" void px_VecI32V_GetX(const float*, float*); extern "C" void eng_VecI32V_GetX(const float*, float*);
extern "C" void px_VecI32V_GetY(const float*, float*); extern "C" void eng_VecI32V_GetY(const float*, float*);
extern "C" void px_VecI32V_GetZ(const float*, float*); extern "C" void eng_VecI32V_GetZ(const float*, float*);
extern "C" void px_VecI32V_GetW(const float*, float*); extern "C" void eng_VecI32V_GetW(const float*, float*);
extern "C" void px_VecI32V_Sub(const float*, float*); extern "C" void eng_VecI32V_Sub(const float*, float*);
extern "C" void px_VecI32V_IsGrtr(const float*, float*); extern "C" void eng_VecI32V_IsGrtr(const float*, float*);
extern "C" void px_VecI32V_IsEq(const float*, float*); extern "C" void eng_VecI32V_IsEq(const float*, float*);
extern "C" void px_U4Zero(const float*, float*); extern "C" void eng_U4Zero(const float*, float*);
extern "C" void px_U4One(const float*, float*); extern "C" void eng_U4One(const float*, float*);
extern "C" void px_U4Two(const float*, float*); extern "C" void eng_U4Two(const float*, float*);
extern "C" void px_V4IsEqU32(const float*, float*); extern "C" void eng_V4IsEqU32(const float*, float*);
extern "C" void px_V4U32or(const float*, float*); extern "C" void eng_V4U32or(const float*, float*);
extern "C" void px_V4U32xor(const float*, float*); extern "C" void eng_V4U32xor(const float*, float*);
extern "C" void px_V4U32and(const float*, float*); extern "C" void eng_V4U32and(const float*, float*);
extern "C" void px_V4U32Andc(const float*, float*); extern "C" void eng_V4U32Andc(const float*, float*);
extern "C" void px_V4IsGrtrV32u(const float*, float*); extern "C" void eng_V4IsGrtrV32u(const float*, float*);
extern "C" void px_M33MulV3(const float*, float*); extern "C" void eng_M33MulV3(const float*, float*);
extern "C" void px_M33MulV3AddV3(const float*, float*); extern "C" void eng_M33MulV3AddV3(const float*, float*);
extern "C" void px_M33TrnspsMulV3(const float*, float*); extern "C" void eng_M33TrnspsMulV3(const float*, float*);
extern "C" void px_M33MulM33(const float*, float*); extern "C" void eng_M33MulM33(const float*, float*);
extern "C" void px_M33Add(const float*, float*); extern "C" void eng_M33Add(const float*, float*);
extern "C" void px_M33Sub(const float*, float*); extern "C" void eng_M33Sub(const float*, float*);
extern "C" void px_M33Neg(const float*, float*); extern "C" void eng_M33Neg(const float*, float*);
extern "C" void px_M33Abs(const float*, float*); extern "C" void eng_M33Abs(const float*, float*);
extern "C" void px_M33Inverse(const float*, float*); extern "C" void eng_M33Inverse(const float*, float*);
extern "C" void px_M33Trnsps(const float*, float*); extern "C" void eng_M33Trnsps(const float*, float*);
extern "C" void px_M33Identity(const float*, float*); extern "C" void eng_M33Identity(const float*, float*);
struct Fn { const char* name; void (*px)(const float*, float*); void (*eng)(const float*, float*); const char* kinds; int outn; int splat; };
static Fn kFns[] = {
  {"QuatVLoadXYZW", px_QuatVLoadXYZW, eng_QuatVLoadXYZW, "f,f,f,f", 4, 0},
  {"Vec3V_From_Vec4V", px_Vec3V_From_Vec4V, eng_Vec3V_From_Vec4V, "V4", 4, 0},
  {"Vec3V_From_Vec4V_WUndefined", px_Vec3V_From_Vec4V_WUndefined, eng_Vec3V_From_Vec4V_WUndefined, "V4", 4, 0},
  {"Vec4V_From_Vec3V", px_Vec4V_From_Vec3V, eng_Vec4V_From_Vec3V, "V3", 4, 0},
  {"Vec4V_From_VecU32V", px_Vec4V_From_VecU32V, eng_Vec4V_From_VecU32V, "U", 4, 0},
  {"Vec4V_From_VecI32V", px_Vec4V_From_VecI32V, eng_Vec4V_From_VecI32V, "I", 4, 0},
  {"VecU32V_From_BoolV", px_VecU32V_From_BoolV, eng_VecU32V_From_BoolV, "B", 4, 0},
  {"VecI32V_From_Vec4V", px_VecI32V_From_Vec4V, eng_VecI32V_From_Vec4V, "V4", 4, 0},
  {"isFiniteFloatV", px_isFiniteFloatV, eng_isFiniteFloatV, "F", 1, 0},
  {"isFiniteVec3V", px_isFiniteVec3V, eng_isFiniteVec3V, "V3", 1, 0},
  {"isFiniteVec4V", px_isFiniteVec4V, eng_isFiniteVec4V, "V4", 1, 0},
  {"isValidVec3V", px_isValidVec3V, eng_isValidVec3V, "V3", 1, 0},
  {"FZero", px_FZero, eng_FZero, "", 4, 1},
  {"FOne", px_FOne, eng_FOne, "", 4, 1},
  {"FHalf", px_FHalf, eng_FHalf, "", 4, 1},
  {"FEps", px_FEps, eng_FEps, "", 4, 1},
  {"FMax", px_FMax, eng_FMax, "", 4, 1},
  {"FNegMax", px_FNegMax, eng_FNegMax, "", 4, 1},
  {"FEps6", px_FEps6, eng_FEps6, "", 4, 1},
  {"FNeg", px_FNeg, eng_FNeg, "F", 4, 1},
  {"FAdd", px_FAdd, eng_FAdd, "F,F", 4, 1},
  {"FSub", px_FSub, eng_FSub, "F,F", 4, 1},
  {"FMul", px_FMul, eng_FMul, "F,F", 4, 1},
  {"FDiv", px_FDiv, eng_FDiv, "F,F", 4, 1},
  {"FDivFast", px_FDivFast, eng_FDivFast, "F,F", 4, 1},
  {"FRecip", px_FRecip, eng_FRecip, "F", 4, 1},
  {"FRecipFast", px_FRecipFast, eng_FRecipFast, "F", 4, 1},
  {"FRsqrt", px_FRsqrt, eng_FRsqrt, "F", 4, 1},
  {"FRsqrtFast", px_FRsqrtFast, eng_FRsqrtFast, "F", 4, 1},
  {"FSqrt", px_FSqrt, eng_FSqrt, "F", 4, 1},
  {"FScaleAdd", px_FScaleAdd, eng_FScaleAdd, "F,F,F", 4, 1},
  {"FNegScaleSub", px_FNegScaleSub, eng_FNegScaleSub, "F,F,F", 4, 1},
  {"FAbs", px_FAbs, eng_FAbs, "F", 4, 1},
  {"FSel", px_FSel, eng_FSel, "BS,F,F", 4, 1},
  {"FIsGrtr", px_FIsGrtr, eng_FIsGrtr, "F,F", 4, 1},
  {"FIsGrtrOrEq", px_FIsGrtrOrEq, eng_FIsGrtrOrEq, "F,F", 4, 1},
  {"FIsEq", px_FIsEq, eng_FIsEq, "F,F", 4, 1},
  {"FMin", px_FMin, eng_FMin, "F,F", 4, 1},
  {"FClamp", px_FClamp, eng_FClamp, "F,F,F", 4, 1},
  {"FAllGrtr", px_FAllGrtr, eng_FAllGrtr, "F,F", 1, 0},
  {"FAllGrtrOrEq", px_FAllGrtrOrEq, eng_FAllGrtrOrEq, "F,F", 1, 0},
  {"FAllEq", px_FAllEq, eng_FAllEq, "F,F", 1, 0},
  {"FOutOfBounds", px_FOutOfBounds, eng_FOutOfBounds, "F,F,F", 1, 0},
  {"FInBounds", px_FInBounds, eng_FInBounds, "F,F,F", 1, 0},
  {"FRound", px_FRound, eng_FRound, "F", 4, 1},
  {"FSin", px_FSin, eng_FSin, "F", 4, 1},
  {"FCos", px_FCos, eng_FCos, "F", 4, 1},
  {"V3Splat", px_V3Splat, eng_V3Splat, "F", 4, 0},
  {"V3Merge", px_V3Merge, eng_V3Merge, "F,F,F", 4, 0},
  {"V3UnitX", px_V3UnitX, eng_V3UnitX, "", 4, 0},
  {"V3UnitY", px_V3UnitY, eng_V3UnitY, "", 4, 0},
  {"V3UnitZ", px_V3UnitZ, eng_V3UnitZ, "", 4, 0},
  {"V3GetX", px_V3GetX, eng_V3GetX, "V3", 4, 1},
  {"V3GetY", px_V3GetY, eng_V3GetY, "V3", 4, 1},
  {"V3GetZ", px_V3GetZ, eng_V3GetZ, "V3", 4, 1},
  {"V3SetX", px_V3SetX, eng_V3SetX, "V3,F", 4, 0},
  {"V3SetY", px_V3SetY, eng_V3SetY, "V3,F", 4, 0},
  {"V3SetZ", px_V3SetZ, eng_V3SetZ, "V3,F", 4, 0},
  {"V3ReadX", px_V3ReadX, eng_V3ReadX, "V3", 1, 0},
  {"V3ReadY", px_V3ReadY, eng_V3ReadY, "V3", 1, 0},
  {"V3ReadZ", px_V3ReadZ, eng_V3ReadZ, "V3", 1, 0},
  {"V3ColX", px_V3ColX, eng_V3ColX, "V3,V3,V3", 4, 0},
  {"V3ColY", px_V3ColY, eng_V3ColY, "V3,V3,V3", 4, 0},
  {"V3ColZ", px_V3ColZ, eng_V3ColZ, "V3,V3,V3", 4, 0},
  {"V3Zero", px_V3Zero, eng_V3Zero, "", 4, 0},
  {"V3One", px_V3One, eng_V3One, "", 4, 0},
  {"V3Eps", px_V3Eps, eng_V3Eps, "", 4, 0},
  {"V3Neg", px_V3Neg, eng_V3Neg, "V3", 4, 0},
  {"V3Add", px_V3Add, eng_V3Add, "V3,V3", 4, 0},
  {"V3Sub", px_V3Sub, eng_V3Sub, "V3,V3", 4, 0},
  {"V3Scale", px_V3Scale, eng_V3Scale, "V3,F", 4, 0},
  {"V3Mul", px_V3Mul, eng_V3Mul, "V3,V3", 4, 0},
  {"V3ScaleInv", px_V3ScaleInv, eng_V3ScaleInv, "V3,F", 4, 0},
  {"V3Div", px_V3Div, eng_V3Div, "V3,V3", 4, 0},
  {"V3ScaleInvFast", px_V3ScaleInvFast, eng_V3ScaleInvFast, "V3,F", 4, 0},
  {"V3DivFast", px_V3DivFast, eng_V3DivFast, "V3,V3", 4, 0},
  {"V3Recip", px_V3Recip, eng_V3Recip, "V3", 4, 0},
  {"V3RecipFast", px_V3RecipFast, eng_V3RecipFast, "V3", 4, 0},
  {"V3Rsqrt", px_V3Rsqrt, eng_V3Rsqrt, "V3", 4, 0},
  {"V3RsqrtFast", px_V3RsqrtFast, eng_V3RsqrtFast, "V3", 4, 0},
  {"V3ScaleAdd", px_V3ScaleAdd, eng_V3ScaleAdd, "V3,F,V3", 4, 0},
  {"V3NegScaleSub", px_V3NegScaleSub, eng_V3NegScaleSub, "V3,F,V3", 4, 0},
  {"V3MulAdd", px_V3MulAdd, eng_V3MulAdd, "V3,V3,V3", 4, 0},
  {"V3NegMulSub", px_V3NegMulSub, eng_V3NegMulSub, "V3,V3,V3", 4, 0},
  {"V3Abs", px_V3Abs, eng_V3Abs, "V3", 4, 0},
  {"V3Dot", px_V3Dot, eng_V3Dot, "V3,V3", 4, 1},
  {"V3Cross", px_V3Cross, eng_V3Cross, "V3,V3", 4, 0},
  {"V3Length", px_V3Length, eng_V3Length, "V3", 4, 1},
  {"V3LengthSq", px_V3LengthSq, eng_V3LengthSq, "V3", 4, 1},
  {"V3Normalize", px_V3Normalize, eng_V3Normalize, "V3", 4, 0},
  {"V3NormalizeSafe", px_V3NormalizeSafe, eng_V3NormalizeSafe, "V3,V3", 4, 0},
  {"V3SumElems", px_V3SumElems, eng_V3SumElems, "V3", 4, 1},
  {"V3Sel", px_V3Sel, eng_V3Sel, "B,V3,V3", 4, 0},
  {"V3IsGrtr", px_V3IsGrtr, eng_V3IsGrtr, "V3,V3", 4, 0},
  {"V3IsGrtrOrEq", px_V3IsGrtrOrEq, eng_V3IsGrtrOrEq, "V3,V3", 4, 0},
  {"V3IsEq", px_V3IsEq, eng_V3IsEq, "V3,V3", 4, 0},
  {"V3Max", px_V3Max, eng_V3Max, "V3,V3", 4, 0},
  {"V3Min", px_V3Min, eng_V3Min, "V3,V3", 4, 0},
  {"V3ExtractMax", px_V3ExtractMax, eng_V3ExtractMax, "V3", 4, 1},
  {"V3ExtractMin", px_V3ExtractMin, eng_V3ExtractMin, "V3", 4, 1},
  {"V3Clamp", px_V3Clamp, eng_V3Clamp, "V3,V3,V3", 4, 0},
  {"V3Sign", px_V3Sign, eng_V3Sign, "V3", 4, 0},
  {"V3AllGrtr", px_V3AllGrtr, eng_V3AllGrtr, "V3,V3", 1, 0},
  {"V3AllGrtrOrEq", px_V3AllGrtrOrEq, eng_V3AllGrtrOrEq, "V3,V3", 1, 0},
  {"V3AllEq", px_V3AllEq, eng_V3AllEq, "V3,V3", 1, 0},
  {"V3OutOfBounds", px_V3OutOfBounds, eng_V3OutOfBounds, "V3,V3,V3", 1, 0},
  {"V3InBounds", px_V3InBounds, eng_V3InBounds, "V3,V3,V3", 1, 0},
  {"V3Round", px_V3Round, eng_V3Round, "V3", 4, 0},
  {"V3Sin", px_V3Sin, eng_V3Sin, "V3", 4, 0},
  {"V3Cos", px_V3Cos, eng_V3Cos, "V3", 4, 0},
  {"V3PermYZZ", px_V3PermYZZ, eng_V3PermYZZ, "V3", 4, 0},
  {"V3PermXYX", px_V3PermXYX, eng_V3PermXYX, "V3", 4, 0},
  {"V3PermYZX", px_V3PermYZX, eng_V3PermYZX, "V3", 4, 0},
  {"V3PermZXY", px_V3PermZXY, eng_V3PermZXY, "V3", 4, 0},
  {"V3PermZZY", px_V3PermZZY, eng_V3PermZZY, "V3", 4, 0},
  {"V3PermYXX", px_V3PermYXX, eng_V3PermYXX, "V3", 4, 0},
  {"V3Perm_Zero_1Z_0Y", px_V3Perm_Zero_1Z_0Y, eng_V3Perm_Zero_1Z_0Y, "V3,V3", 4, 0},
  {"V3Perm_0Z_Zero_1X", px_V3Perm_0Z_Zero_1X, eng_V3Perm_0Z_Zero_1X, "V3,V3", 4, 0},
  {"V3Perm_1Y_0X_Zero", px_V3Perm_1Y_0X_Zero, eng_V3Perm_1Y_0X_Zero, "V3,V3", 4, 0},
  {"V4Splat", px_V4Splat, eng_V4Splat, "F", 4, 0},
  {"V4Merge", px_V4Merge, eng_V4Merge, "F,F,F,F", 4, 0},
  {"V4MergeW", px_V4MergeW, eng_V4MergeW, "V4,V4,V4,V4", 4, 0},
  {"V4MergeZ", px_V4MergeZ, eng_V4MergeZ, "V4,V4,V4,V4", 4, 0},
  {"V4MergeY", px_V4MergeY, eng_V4MergeY, "V4,V4,V4,V4", 4, 0},
  {"V4MergeX", px_V4MergeX, eng_V4MergeX, "V4,V4,V4,V4", 4, 0},
  {"V4UnpackXY", px_V4UnpackXY, eng_V4UnpackXY, "V4,V4", 4, 0},
  {"V4UnpackZW", px_V4UnpackZW, eng_V4UnpackZW, "V4,V4", 4, 0},
  {"V4UnitW", px_V4UnitW, eng_V4UnitW, "", 4, 0},
  {"V4UnitY", px_V4UnitY, eng_V4UnitY, "", 4, 0},
  {"V4UnitZ", px_V4UnitZ, eng_V4UnitZ, "", 4, 0},
  {"V4GetX", px_V4GetX, eng_V4GetX, "V4", 4, 1},
  {"V4GetY", px_V4GetY, eng_V4GetY, "V4", 4, 1},
  {"V4GetZ", px_V4GetZ, eng_V4GetZ, "V4", 4, 1},
  {"V4GetW", px_V4GetW, eng_V4GetW, "V4", 4, 1},
  {"V4SetX", px_V4SetX, eng_V4SetX, "V4,F", 4, 0},
  {"V4SetY", px_V4SetY, eng_V4SetY, "V4,F", 4, 0},
  {"V4SetZ", px_V4SetZ, eng_V4SetZ, "V4,F", 4, 0},
  {"V4SetW", px_V4SetW, eng_V4SetW, "V4,F", 4, 0},
  {"V4ClearW", px_V4ClearW, eng_V4ClearW, "V4", 4, 0},
  {"V4ReadX", px_V4ReadX, eng_V4ReadX, "V4", 1, 0},
  {"V4ReadY", px_V4ReadY, eng_V4ReadY, "V4", 1, 0},
  {"V4ReadZ", px_V4ReadZ, eng_V4ReadZ, "V4", 1, 0},
  {"V4ReadW", px_V4ReadW, eng_V4ReadW, "V4", 1, 0},
  {"V4Zero", px_V4Zero, eng_V4Zero, "", 4, 0},
  {"V4One", px_V4One, eng_V4One, "", 4, 0},
  {"V4Eps", px_V4Eps, eng_V4Eps, "", 4, 0},
  {"V4Neg", px_V4Neg, eng_V4Neg, "V4", 4, 0},
  {"V4Add", px_V4Add, eng_V4Add, "V4,V4", 4, 0},
  {"V4Sub", px_V4Sub, eng_V4Sub, "V4,V4", 4, 0},
  {"V4Scale", px_V4Scale, eng_V4Scale, "V4,F", 4, 0},
  {"V4Mul", px_V4Mul, eng_V4Mul, "V4,V4", 4, 0},
  {"V4ScaleInv", px_V4ScaleInv, eng_V4ScaleInv, "V4,F", 4, 0},
  {"V4Div", px_V4Div, eng_V4Div, "V4,V4", 4, 0},
  {"V4ScaleInvFast", px_V4ScaleInvFast, eng_V4ScaleInvFast, "V4,F", 4, 0},
  {"V4DivFast", px_V4DivFast, eng_V4DivFast, "V4,V4", 4, 0},
  {"V4Recip", px_V4Recip, eng_V4Recip, "V4", 4, 0},
  {"V4RecipFast", px_V4RecipFast, eng_V4RecipFast, "V4", 4, 0},
  {"V4Rsqrt", px_V4Rsqrt, eng_V4Rsqrt, "V4", 4, 0},
  {"V4RsqrtFast", px_V4RsqrtFast, eng_V4RsqrtFast, "V4", 4, 0},
  {"V4ScaleAdd", px_V4ScaleAdd, eng_V4ScaleAdd, "V4,F,V4", 4, 0},
  {"V4NegScaleSub", px_V4NegScaleSub, eng_V4NegScaleSub, "V4,F,V4", 4, 0},
  {"V4MulAdd", px_V4MulAdd, eng_V4MulAdd, "V4,V4,V4", 4, 0},
  {"V4NegMulSub", px_V4NegMulSub, eng_V4NegMulSub, "V4,V4,V4", 4, 0},
  {"V4Abs", px_V4Abs, eng_V4Abs, "V4", 4, 0},
  {"V4Andc", px_V4Andc, eng_V4Andc, "V4,U", 4, 0},
  {"V4Dot", px_V4Dot, eng_V4Dot, "V4,V4", 4, 1},
  {"V4Dot3", px_V4Dot3, eng_V4Dot3, "V4,V4", 4, 1},
  {"V4Cross", px_V4Cross, eng_V4Cross, "V4,V4", 4, 0},
  {"V4Length", px_V4Length, eng_V4Length, "V4", 4, 1},
  {"V4LengthSq", px_V4LengthSq, eng_V4LengthSq, "V4", 4, 1},
  {"V4Normalize", px_V4Normalize, eng_V4Normalize, "V4", 4, 0},
  {"V4NormalizeSafe", px_V4NormalizeSafe, eng_V4NormalizeSafe, "V4,V4", 4, 0},
  {"V4NormalizeFast", px_V4NormalizeFast, eng_V4NormalizeFast, "V4", 4, 0},
  {"V4Sel", px_V4Sel, eng_V4Sel, "B,V4,V4", 4, 0},
  {"V4IsGrtr", px_V4IsGrtr, eng_V4IsGrtr, "V4,V4", 4, 0},
  {"V4IsGrtrOrEq", px_V4IsGrtrOrEq, eng_V4IsGrtrOrEq, "V4,V4", 4, 0},
  {"V4IsEq", px_V4IsEq, eng_V4IsEq, "V4,V4", 4, 0},
  {"V4Max", px_V4Max, eng_V4Max, "V4,V4", 4, 0},
  {"V4Min", px_V4Min, eng_V4Min, "V4,V4", 4, 0},
  {"V4ExtractMax", px_V4ExtractMax, eng_V4ExtractMax, "V4", 4, 2},
  {"V4ExtractMin", px_V4ExtractMin, eng_V4ExtractMin, "V4", 4, 2},
  {"V4Clamp", px_V4Clamp, eng_V4Clamp, "V4,V4,V4", 4, 0},
  {"V4AllGrtr", px_V4AllGrtr, eng_V4AllGrtr, "V4,V4", 1, 0},
  {"V4AllGrtrOrEq", px_V4AllGrtrOrEq, eng_V4AllGrtrOrEq, "V4,V4", 1, 0},
  {"V4AllGrtrOrEq3", px_V4AllGrtrOrEq3, eng_V4AllGrtrOrEq3, "V4,V4", 1, 0},
  {"V4AllEq", px_V4AllEq, eng_V4AllEq, "V4,V4", 1, 0},
  {"V4AnyGrtr3", px_V4AnyGrtr3, eng_V4AnyGrtr3, "V4,V4", 1, 0},
  {"V4Round", px_V4Round, eng_V4Round, "V4", 4, 0},
  {"V4Sin", px_V4Sin, eng_V4Sin, "V4", 4, 0},
  {"V4Cos", px_V4Cos, eng_V4Cos, "V4", 4, 0},
  {"V4PermYXWZ", px_V4PermYXWZ, eng_V4PermYXWZ, "V4", 4, 0},
  {"V4PermXZXZ", px_V4PermXZXZ, eng_V4PermXZXZ, "V4", 4, 0},
  {"V4PermYWYW", px_V4PermYWYW, eng_V4PermYWYW, "V4", 4, 0},
  {"V4PermYZXW", px_V4PermYZXW, eng_V4PermYZXW, "V4", 4, 0},
  {"V4PermZWXY", px_V4PermZWXY, eng_V4PermZWXY, "V4", 4, 0},
  {"QuatV_From_RotationAxisAngle", px_QuatV_From_RotationAxisAngle, eng_QuatV_From_RotationAxisAngle, "V3,F", 4, 0},
  {"QuatNormalize", px_QuatNormalize, eng_QuatNormalize, "Q", 4, 0},
  {"QuatLength", px_QuatLength, eng_QuatLength, "Q", 4, 1},
  {"QuatLengthSq", px_QuatLengthSq, eng_QuatLengthSq, "Q", 4, 1},
  {"QuatDot", px_QuatDot, eng_QuatDot, "Q,Q", 4, 1},
  {"QuatConjugate", px_QuatConjugate, eng_QuatConjugate, "Q", 4, 0},
  {"QuatGetImaginaryPart", px_QuatGetImaginaryPart, eng_QuatGetImaginaryPart, "Q", 4, 0},
  {"QuatGetMat33V", px_QuatGetMat33V, eng_QuatGetMat33V, "Q", 12, 0},
  {"Mat33GetQuatV", px_Mat33GetQuatV, eng_Mat33GetQuatV, "M33", 4, 0},
  {"QuatGetBasisVector0", px_QuatGetBasisVector0, eng_QuatGetBasisVector0, "Q", 4, 0},
  {"QuatGetBasisVector1", px_QuatGetBasisVector1, eng_QuatGetBasisVector1, "Q", 4, 0},
  {"QuatGetBasisVector2", px_QuatGetBasisVector2, eng_QuatGetBasisVector2, "Q", 4, 0},
  {"QuatRotate", px_QuatRotate, eng_QuatRotate, "Q,V3", 4, 0},
  {"QuatRotateInv", px_QuatRotateInv, eng_QuatRotateInv, "Q,V3", 4, 0},
  {"QuatMul", px_QuatMul, eng_QuatMul, "Q,Q", 4, 0},
  {"QuatAdd", px_QuatAdd, eng_QuatAdd, "Q,Q", 4, 0},
  {"QuatNeg", px_QuatNeg, eng_QuatNeg, "Q", 4, 0},
  {"QuatSub", px_QuatSub, eng_QuatSub, "Q,Q", 4, 0},
  {"QuatScale", px_QuatScale, eng_QuatScale, "Q,F", 4, 0},
  {"QuatMerge", px_QuatMerge, eng_QuatMerge, "F,F,F,F", 4, 0},
  {"QuatIdentity", px_QuatIdentity, eng_QuatIdentity, "", 4, 0},
  {"isFiniteQuatV", px_isFiniteQuatV, eng_isFiniteQuatV, "Q", 1, 0},
  {"isValidQuatV", px_isValidQuatV, eng_isValidQuatV, "Q", 1, 0},
  {"isSaneQuatV", px_isSaneQuatV, eng_isSaneQuatV, "Q", 1, 0},
  {"BFFFF", px_BFFFF, eng_BFFFF, "", 4, 0},
  {"BFFFT", px_BFFFT, eng_BFFFT, "", 4, 0},
  {"BFFTF", px_BFFTF, eng_BFFTF, "", 4, 0},
  {"BFFTT", px_BFFTT, eng_BFFTT, "", 4, 0},
  {"BFTFF", px_BFTFF, eng_BFTFF, "", 4, 0},
  {"BFTFT", px_BFTFT, eng_BFTFT, "", 4, 0},
  {"BFTTF", px_BFTTF, eng_BFTTF, "", 4, 0},
  {"BFTTT", px_BFTTT, eng_BFTTT, "", 4, 0},
  {"BTFFF", px_BTFFF, eng_BTFFF, "", 4, 0},
  {"BTFFT", px_BTFFT, eng_BTFFT, "", 4, 0},
  {"BTFTF", px_BTFTF, eng_BTFTF, "", 4, 0},
  {"BTFTT", px_BTFTT, eng_BTFTT, "", 4, 0},
  {"BTTFF", px_BTTFF, eng_BTTFF, "", 4, 0},
  {"BTTFT", px_BTTFT, eng_BTTFT, "", 4, 0},
  {"BTTTF", px_BTTTF, eng_BTTTF, "", 4, 0},
  {"BTTTT", px_BTTTT, eng_BTTTT, "", 4, 0},
  {"BWMask", px_BWMask, eng_BWMask, "", 4, 0},
  {"BXMask", px_BXMask, eng_BXMask, "", 4, 0},
  {"BYMask", px_BYMask, eng_BYMask, "", 4, 0},
  {"BZMask", px_BZMask, eng_BZMask, "", 4, 0},
  {"BGetX", px_BGetX, eng_BGetX, "B", 4, 0},
  {"BGetY", px_BGetY, eng_BGetY, "B", 4, 0},
  {"BGetZ", px_BGetZ, eng_BGetZ, "B", 4, 0},
  {"BGetW", px_BGetW, eng_BGetW, "B", 4, 0},
  {"BAnd", px_BAnd, eng_BAnd, "B,B", 4, 0},
  {"BOr", px_BOr, eng_BOr, "B,B", 4, 0},
  {"BNot", px_BNot, eng_BNot, "B", 4, 0},
  {"BAllTrue4", px_BAllTrue4, eng_BAllTrue4, "B", 4, 0},
  {"BAnyTrue4", px_BAnyTrue4, eng_BAnyTrue4, "B", 4, 0},
  {"BAllTrue3", px_BAllTrue3, eng_BAllTrue3, "B", 4, 0},
  {"BAnyTrue3", px_BAnyTrue3, eng_BAnyTrue3, "B", 4, 0},
  {"BAllEq", px_BAllEq, eng_BAllEq, "B,B", 1, 0},
  {"BAllEqTTTT", px_BAllEqTTTT, eng_BAllEqTTTT, "B", 1, 0},
  {"BAllEqFFFF", px_BAllEqFFFF, eng_BAllEqFFFF, "B", 1, 0},
  {"VecI32V_Zero", px_VecI32V_Zero, eng_VecI32V_Zero, "", 4, 0},
  {"VecI32V_One", px_VecI32V_One, eng_VecI32V_One, "", 4, 0},
  {"VecI32V_Two", px_VecI32V_Two, eng_VecI32V_Two, "", 4, 0},
  {"VecI32V_MinusOne", px_VecI32V_MinusOne, eng_VecI32V_MinusOne, "", 4, 0},
  {"VecI32V_Add", px_VecI32V_Add, eng_VecI32V_Add, "I,I", 4, 0},
  {"VecI32V_Or", px_VecI32V_Or, eng_VecI32V_Or, "I,I", 4, 0},
  {"VecI32V_GetX", px_VecI32V_GetX, eng_VecI32V_GetX, "I", 4, 0},
  {"VecI32V_GetY", px_VecI32V_GetY, eng_VecI32V_GetY, "I", 4, 0},
  {"VecI32V_GetZ", px_VecI32V_GetZ, eng_VecI32V_GetZ, "I", 4, 0},
  {"VecI32V_GetW", px_VecI32V_GetW, eng_VecI32V_GetW, "I", 4, 0},
  {"VecI32V_Sub", px_VecI32V_Sub, eng_VecI32V_Sub, "I,I", 4, 0},
  {"VecI32V_IsGrtr", px_VecI32V_IsGrtr, eng_VecI32V_IsGrtr, "I,I", 4, 0},
  {"VecI32V_IsEq", px_VecI32V_IsEq, eng_VecI32V_IsEq, "I,I", 4, 0},
  {"U4Zero", px_U4Zero, eng_U4Zero, "", 4, 0},
  {"U4One", px_U4One, eng_U4One, "", 4, 0},
  {"U4Two", px_U4Two, eng_U4Two, "", 4, 0},
  {"V4IsEqU32", px_V4IsEqU32, eng_V4IsEqU32, "U,U", 4, 0},
  {"V4U32or", px_V4U32or, eng_V4U32or, "U,U", 4, 0},
  {"V4U32xor", px_V4U32xor, eng_V4U32xor, "U,U", 4, 0},
  {"V4U32and", px_V4U32and, eng_V4U32and, "U,U", 4, 0},
  {"V4U32Andc", px_V4U32Andc, eng_V4U32Andc, "U,U", 4, 0},
  {"V4IsGrtrV32u", px_V4IsGrtrV32u, eng_V4IsGrtrV32u, "V4,V4", 4, 0},
  {"M33MulV3", px_M33MulV3, eng_M33MulV3, "M33,V3", 4, 0},
  {"M33MulV3AddV3", px_M33MulV3AddV3, eng_M33MulV3AddV3, "M33,V3,V3", 4, 0},
  {"M33TrnspsMulV3", px_M33TrnspsMulV3, eng_M33TrnspsMulV3, "M33,V3", 4, 0},
  {"M33MulM33", px_M33MulM33, eng_M33MulM33, "M33,M33", 12, 0},
  {"M33Add", px_M33Add, eng_M33Add, "M33,M33", 12, 0},
  {"M33Sub", px_M33Sub, eng_M33Sub, "M33,M33", 12, 0},
  {"M33Neg", px_M33Neg, eng_M33Neg, "M33", 12, 0},
  {"M33Abs", px_M33Abs, eng_M33Abs, "M33", 12, 0},
  {"M33Inverse", px_M33Inverse, eng_M33Inverse, "M33", 12, 0},
  {"M33Trnsps", px_M33Trnsps, eng_M33Trnsps, "M33", 12, 0},
  {"M33Identity", px_M33Identity, eng_M33Identity, "", 12, 0},
};

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
    else if (k == "BS") { uint32_t u = (g() & 1) ? 0xffffffffu : 0u; for (int i = 0; i < 4; ++i) memcpy(&in[off + i], &u, 4); off += 4; }
    else if (k == "U" || k == "I") { for (int i = 0; i < 4; ++i) { uint32_t u = uint32_t(g()) >> (g() % 24); memcpy(&in[off + i], &u, 4); } off += 4; }
    else if (k == "M33") { for (int c3 = 0; c3 < 3; ++c3) { for (int i = 0; i < 3; ++i) in[off + 4 * c3 + i] = fr(g, mode); in[off + 4 * c3 + 3] = 0.0f; } off += 16; }
    else if (k == "f") { in[off] = fr(g, mode); off += 4; }
    else if (k == "u") { uint32_t u = g() % 4; memcpy(&in[off], &u, 4); off += 4; }
    else if (k == "b") { in[off] = float(g() & 1); off += 4; }
    if (c == std::string::npos) break;
    pos = c + 1;
  }
}
#ifdef AOS_DIFF_GPU
extern "C" int gpu_run(int fid, const float* in, float* out, int n);
#endif
int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 20000;
#ifdef AOS_DIFF_GPU
  {  // 층 2: 진짜 PhysX (CPU, FTZ+DAZ) vs 엔진 aos (GPU, -ftz=true)
    _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));
    std::mt19937 g(777);
    std::vector<float> in(64 * size_t(iters)), a(16 * size_t(iters)), b(16 * size_t(iters));
    int nbad = 0, fid = 0; long long total = 0;
    for (const Fn& f : kFns) {
      for (int it = 0; it < iters; ++it) { fill(g, f.kinds, &in[64 * size_t(it)], it % 3); float t[64] = {}; f.px(&in[64 * size_t(it)], t); memcpy(&a[16 * size_t(it)], t, 64); }
      if (gpu_run(fid++, in.data(), b.data(), iters)) { printf("CUDA 실패\n"); return 2; }
      int bad = 0;
      for (int it = 0; it < iters; ++it) {
        const float* x = &a[16 * size_t(it)]; const float* y = &b[16 * size_t(it)];
        total++;
        for (int i = 0; i < f.outn; ++i) {
          if (memcmp(&x[i], &y[i], 4) && !(std::isnan(x[i]) && std::isnan(y[i]))) {
            if (bad++ < 2) { printf("[다름 GPU] %s 입력종류 %s 칸 %d: %.9g/%.9g\n", f.name, f.kinds, i, x[i], y[i]); }
            break;
          }
        }
      }
      if (bad) nbad++;
    }
    printf("aos 함수 %d 개 x %d 입력 (PhysX CPU vs 엔진 GPU): 다른 함수 %d, 비교 %lld\n", fid, iters, nbad, total);
    return nbad ? 3 : 0;
  }
#endif
  const bool ftz = argc > 2 && atoi(argv[2]) != 0;
  if (ftz) _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));  // PhysX PxSIMDGuard 와 같은 FTZ+DAZ
  std::mt19937 g(12345);
  int nbad = 0, nfn = 0, nsplatfn = 0, nsplatbad = 0; long long total = 0;
  for (const Fn& f : kFns) {
    nfn++;
    nsplatfn += (f.splat != 0);
    int bad = 0, sbad = 0; float in[64], a[64], b[64];
    for (int it = 0; it < iters; ++it) {
      fill(g, f.kinds, in, it % 3);
      memset(a, 0, sizeof a); memset(b, 0, sizeof b);
      f.px(in, a); f.eng(in, b);
      total++;
      if (f.splat) {  // 진짜 PhysX 결과의 네 칸이 비트까지 같은가 (NaN 은 값 무늬 무시)
        float in2[64], s[64] = {};
        memcpy(in2, in, sizeof in2);
        if (f.splat == 2)  // 알려진 예외: PhysX 가 쓰는 조건(입력이 V4Abs 결과, -0 없음)에서만 본다
          for (int i = 0; i < 4; ++i) { uint32_t u; memcpy(&u, &in2[i], 4); u &= 0x7fffffffu; memcpy(&in2[i], &u, 4); }
        f.px(in2, s);
        for (int i = 1; i < 4; ++i)
          if (memcmp(&s[0], &s[i], 4) && !(std::isnan(s[0]) && std::isnan(s[i]))) {
            if (sbad++ < 2) printf("[네 칸 다름] %s: %.9g %.9g %.9g %.9g\n", f.name, s[0], s[1], s[2], s[3]);
            break;
          }
      }
      if (memcmp(a, b, 4 * f.outn)) {
        bool bothnan = true;
        for (int i = 0; i < f.outn; ++i) { if (!(std::isnan(a[i]) && std::isnan(b[i])) && memcmp(&a[i], &b[i], 4)) bothnan = false; }
        if (bothnan) continue;
        if (bad++ < 2) { printf("[다름] %s 입력종류 %s:", f.name, f.kinds); for (int i = 0; i < f.outn; ++i) printf(" %.9g/%.9g", a[i], b[i]); printf("\n"); }
      }
    }
    if (bad) nbad++;
    if (sbad) nsplatbad++;
  }
  printf("aos 함수 %d 개 x %d 입력 (FTZ/DAZ %s): 다른 함수 %d, 비교 %lld\n", nfn, iters, ftz ? "켬" : "끔", nbad, total);
  printf("FloatV 한 칸 표현 근거: 네 칸 같아야 할 함수 %d 개 중 PhysX 에서 네 칸이 갈린 함수 %d 개\n", nsplatfn, nsplatbad);
  return (nbad || nsplatbad) ? 3 : 0;
}

