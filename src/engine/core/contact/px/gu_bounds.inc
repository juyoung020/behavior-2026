// 모양 경계 상자 Gu::computeBounds 의 우리 판 — 원본 geomutils/src/GuBounds.cpp:136~490 을 손으로 옮김(식·순서 그대로, 쓰는 길만).
// BEHAVIOR 장면에 나오는 구·평면·캡슐·상자·볼록 메시만 (삼각 메시·높이장·사면체·사용자 모양·convex core 는 없음).
// 이 값 + contactOffset 이 넓은 단계 입력(Sc 가 BoundsArray 에 넣는 값)이다. 호스트·GPU 공용.
namespace px {
namespace Gu {

// GuBounds.cpp:143
EHD aos::Vec4V bounds_multiply3x3V(const aos::Vec4V p, const PxMat33Padded& mat_Padded) {
  using namespace aos;
  Vec4V ResV = V4Scale(V4LoadU(&mat_Padded.column0.x), V4GetX(p));
  ResV = V4Add(ResV, V4Scale(V4LoadU(&mat_Padded.column1.x), V4GetY(p)));
  ResV = V4Add(ResV, V4Scale(V4LoadU(&mat_Padded.column2.x), V4GetZ(p)));
  return ResV;
}

// GuBounds.cpp:151
EHD void bounds_transformNoEmptyTestV(PxVec3p& c, PxVec3p& ext, const PxMat33Padded& rot, const PxVec3& pos, const CenterExtentsPadded& bounds) {
  using namespace aos;
  const Vec4V boundsCenterV = V4LoadU(&bounds.mCenter.x);
  const Vec4V posV = Vec4V_From_Vec3V(V3LoadU(&pos.x));
  const Vec4V cV = V4Add(bounds_multiply3x3V(boundsCenterV, rot), posV);
  V4StoreU(cV, &c.x);
  const Vec4V boundsExtentsV = V4LoadU(&bounds.mExtents.x);
  const Vec4V c0V = V4Scale(V4LoadU(&rot.column0.x), V4GetX(boundsExtentsV));
  const Vec4V c1V = V4Scale(V4LoadU(&rot.column1.x), V4GetY(boundsExtentsV));
  const Vec4V c2V = V4Scale(V4LoadU(&rot.column2.x), V4GetZ(boundsExtentsV));
  Vec4V extentsV = V4Add(V4Abs(c0V), V4Abs(c1V));
  extentsV = V4Add(extentsV, V4Abs(c2V));
  V4StoreU(extentsV, &ext.x);
}

// GuBounds.cpp:174
EHD PxU32 bounds_isNonIdentity(const PxVec3& scale) {
  const PxU32 IEEE_1_0 = 0x3f800000;
  return (PxUnionCast<PxU32>(scale.x) - IEEE_1_0) | (PxUnionCast<PxU32>(scale.y) - IEEE_1_0) | (PxUnionCast<PxU32>(scale.z) - IEEE_1_0);
}

// GuBounds.cpp:182
EHD void bounds_computeScaledMatrix(PxMat33Padded& rot, const PxMeshScale& scale) { rot = rot * Cm::toMat33(scale); }

// GuBounds.cpp:187, :205 (computeMeshBounds -> transformNoEmptyTest(transform 판))
EHD void bounds_computeMeshBounds(const PxTransform& pose, const CenterExtentsPadded* localSpaceBounds, const PxMeshScale& meshScale, PxVec3p& origin,
                                  PxVec3p& extent) {
  PxMat33Padded rot(pose.q);
  if (bounds_isNonIdentity(meshScale.scale)) bounds_computeScaledMatrix(rot, meshScale);
  bounds_transformNoEmptyTestV(origin, extent, rot, pose.p, *localSpaceBounds);
}

// GuBounds.cpp:210
EHD void bounds_computePlaneBounds(PxBounds3& bounds, const PxTransform& pose, float contactOffset, float inflation) {
  const PxF32 bigValue = EPX_MAX_BOUNDS_EXTENTS;
  PxVec3 minPt = PxVec3(-bigValue, -bigValue, -bigValue);
  PxVec3 maxPt = PxVec3(bigValue, bigValue, bigValue);
  const PxVec3 planeNormal = pose.q.getBasisVector0();
  const PxPlane plane(pose.p, planeNormal);
  const float nx = PxAbs(planeNormal.x);
  const float ny = PxAbs(planeNormal.y);
  const float nz = PxAbs(planeNormal.z);
  const float epsilon = 1e-6f;
  const float oneMinusEpsilon = 1.0f - epsilon;
  if (nx > oneMinusEpsilon && ny < epsilon && nz < epsilon) {
    if (planeNormal.x > 0.0f) maxPt.x = -plane.d + contactOffset;
    else minPt.x = plane.d - contactOffset;
  } else if (nx < epsilon && ny > oneMinusEpsilon && nz < epsilon) {
    if (planeNormal.y > 0.0f) maxPt.y = -plane.d + contactOffset;
    else minPt.y = plane.d - contactOffset;
  } else if (nx < epsilon && ny < epsilon && nz > oneMinusEpsilon) {
    if (planeNormal.z > 0.0f) maxPt.z = -plane.d + contactOffset;
    else minPt.z = plane.d - contactOffset;
  }
  if (inflation != 1.0f) {
    const PxVec3 c = (maxPt + minPt) * 0.5f;
    const PxVec3 e = (maxPt - minPt) * 0.5f * inflation;
    minPt = c - e;
    maxPt = c + e;
  }
  bounds.minimum = minPt;
  bounds.maximum = maxPt;
}

// GuBounds.cpp:265
EHD void bounds_inflateBounds(PxBounds3& bounds, const PxVec3p& origin, const PxVec3p& extents, float contactOffset, float inflation) {
  using namespace aos;
  Vec4V extentsV = V4LoadU(&extents.x);
  extentsV = V4Add(extentsV, V4Load(contactOffset));
  extentsV = V4Scale(extentsV, FLoad(inflation));
  const Vec4V originV = V4LoadU(&origin.x);
  const Vec4V minV = V4Sub(originV, extentsV);
  const Vec4V maxV = V4Add(originV, extentsV);
  StoreBounds(bounds, minV, maxV);
}

// GuBounds.cpp:278
EHD aos::Vec4V bounds_basisExtentV(const PxMat33Padded& basis, const PxVec3& extent, float offset, float inflation) {
  using namespace aos;
  const Vec4V c0V = V4Scale(V4LoadU(&basis.column0.x), FLoad(extent.x));
  const Vec4V c1V = V4Scale(V4LoadU(&basis.column1.x), FLoad(extent.y));
  const Vec4V c2V = V4Scale(V4LoadU(&basis.column2.x), FLoad(extent.z));
  Vec4V extentsV = V4Add(V4Abs(c0V), V4Abs(c1V));
  extentsV = V4Add(extentsV, V4Abs(c2V));
  extentsV = V4Add(extentsV, V4Load(offset));
  extentsV = V4Scale(extentsV, FLoad(inflation));
  return extentsV;
}

// GuBounds.cpp:301 computeTightBounds (볼록 메시 기본값 eTIGHT_BOUNDS 가 이 길)
EHD void computeTightBounds(PxBounds3& bounds, PxU32 nb, const PxVec3* v, const PxTransform& pose, const PxMeshScale& scale, float contactOffset,
                            float inflation) {
  using namespace aos;
  if (!nb) {
    bounds.setEmpty();
    return;
  }
  PxMat33Padded rot(pose.q);
  if (bounds_isNonIdentity(scale.scale)) bounds_computeScaledMatrix(rot, scale);
  PxU32 nbSafe = nb - 1;
  const Vec4V lastVertexV = bounds_multiply3x3V(Vec4V_From_Vec3V(V3LoadU(&v[nbSafe].x)), rot);
  Vec4V minV = lastVertexV;
  Vec4V maxV = lastVertexV;
  while (nbSafe--) {
    // 원본은 V4LoadU(&v->x) 로 다음 꼭짓점 x 를 w 칸에 읽는다. w 칸은 결과에 안 쓰이므로(multiply3x3V 가 x·y·z 만 씀) V3 로 읽는다
    // (GPU·마지막 앞 꼭짓점에서 배열 밖 읽기 방지). 값 같음.
    const Vec4V vertexV = bounds_multiply3x3V(Vec4V_From_Vec3V(V3LoadU(&v->x)), rot);
    v++;
    minV = V4Min(minV, vertexV);
    maxV = V4Max(maxV, vertexV);
  }
  const Vec4V offsetV = V4Load(contactOffset);
  minV = V4Sub(minV, offsetV);
  maxV = V4Add(maxV, offsetV);
  const Vec4V posV = Vec4V_From_Vec3V(V3LoadU(&pose.p.x));
  maxV = V4Add(maxV, posV);
  minV = V4Add(minV, posV);
  {
    const Vec4V centerV = V4Scale(V4Add(maxV, minV), FLoad(0.5f));
    const Vec4V extentsV = V4Scale(V4Sub(maxV, minV), FLoad(0.5f * inflation));
    maxV = V4Add(centerV, extentsV);
    minV = V4Sub(centerV, extentsV);
  }
  StoreBounds(bounds, minV, maxV);
}

// GuBounds.cpp:354
EHD void computeBounds(PxBounds3& bounds, const PxGeometry& geometry, const PxTransform& pose, float contactOffset, float inflation) {
  using namespace aos;
  switch (geometry.getType()) {
    case PxGeometryType::eSPHERE: {
      const PxSphereGeometry& shape = static_cast<const PxSphereGeometry&>(geometry);
      const PxVec3 extents((shape.radius + contactOffset) * inflation);
      bounds.minimum = pose.p - extents;
      bounds.maximum = pose.p + extents;
    } break;
    case PxGeometryType::ePLANE:
      bounds_computePlaneBounds(bounds, pose, contactOffset, inflation);
      break;
    case PxGeometryType::eCAPSULE:
      computeCapsuleBounds(bounds, static_cast<const PxCapsuleGeometry&>(geometry), pose, contactOffset, inflation);
      break;
    case PxGeometryType::eBOX: {
      const PxBoxGeometry& shape = static_cast<const PxBoxGeometry&>(geometry);
      const PxVec3p origin(pose.p);
      const PxMat33Padded basis(pose.q);
      const Vec4V extentsV = bounds_basisExtentV(basis, shape.halfExtents, contactOffset, inflation);
      const Vec4V originV = V4LoadU(&origin.x);
      const Vec4V minV = V4Sub(originV, extentsV);
      const Vec4V maxV = V4Add(originV, extentsV);
      StoreBounds(bounds, minV, maxV);
    } break;
    case PxGeometryType::eCONVEXMESH: {
      const PxConvexMeshGeometry& shape = static_cast<const PxConvexMeshGeometry&>(geometry);
      const Gu::ConvexHullData& hullData = *shape.hullData;
      const bool useTightBounds = shape.meshFlags & 1;  // PxConvexMeshGeometryFlag::eTIGHT_BOUNDS
      if (useTightBounds)
        computeTightBounds(bounds, hullData.mNbHullVertices, hullData.getHullVertices(), pose, shape.scale, contactOffset, inflation);
      else {
        PxVec3p origin, extents;
        bounds_computeMeshBounds(pose, &hullData.getPaddedBounds(), shape.scale, origin, extents);
        bounds_inflateBounds(bounds, origin, extents, contactOffset, inflation);
      }
    } break;
    default:
      bounds.setEmpty();  // BEHAVIOR 장면에 없는 모양
  }
}

}  // namespace Gu
}  // namespace px
