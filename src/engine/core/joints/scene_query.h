// joints 모듈: 장면 질의 — 광선(raycast) 모양별 함수. PhysX 5.6.1 과 비트 동일 목표 (CPU·CUDA 공용).
// omni.physx 의 raycast_closest (omni/.../plugins/PhysXSceneQuery.cpp:336) = PxSceneQueryExt::raycastSingle(기본 hitFlags
// = ePOSITION|eNORMAL|eFACE_INDEX, 거르개 없음) — OmniGibson 보조 잡기(robots/robot.py:3265)가 손가락 점 사이 광선마다 부른다.
// 원본(physx/source/geomutils/src 기준)
//   GuRaycastTests.cpp:52 (raycast_box), :119 (raycast_sphere), :168 (raycast_capsule), :225 (raycast_plane), :268 (raycast_convexMesh)
//   intersection/GuIntersectionRayBox.cpp:153 (rayAABBIntersect2, RAYAABB_EPSILON 1e-5), GuIntersectionRaySphere.cpp:36,78,
//   GuIntersectionRayCapsule.h:43 / .cpp:36,57, GuIntersectionRayPlane.h:38, GuInternal.h:60-70 (캡슐 선분), GuInternal.cpp:72 (평면),
//   distance/GuDistancePointSegment.h:40, include/GuSegment.h:155, common/src/CmScaling.h:45,219 (메시 척도 행렬)
// 볼록 메시는 다각형 평면(HullPolygonData::mPlane = PxHullPolygon::mPlane, n·x + d)만 쓴다.
#pragma once
#include <cstdint>

#include "../common/pmath.h"

namespace eng {
namespace sq {

enum : uint32_t { HF_POSITION = 1 << 0, HF_NORMAL = 1 << 1, HF_UV = 1 << 3, HF_FACE_INDEX = 1 << 10 };  // PxHitFlag (값 확인은 시험)
enum : uint32_t { HF_DEFAULT = HF_POSITION | HF_NORMAL | HF_FACE_INDEX };

struct RayHit {
  float distance;
  V3 position, normal;
  uint32_t faceIndex;
  float u, v;
  uint32_t flags;
};

// ---- 스칼라 보조 (PxVec3/PxMat33 연산 순서)
EHD V3 sneg(const V3& a) { return V3{-a.x, -a.y, -a.z}; }
EHD float sdot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }  // PxVec3::dot
EHD V3 snormalize(V3 v) {  // PxVec3::normalize (m>0 이면 *= 1/m)
  const float m = psqrt(magSq(v));
  if (m > 0.0f) {
    const float f = 1.0f / m;
    v.x *= f; v.y *= f; v.z *= f;
  }
  return v;
}
EHD M33 mtranspose(const M33& m) {
  return M33{V3{m.c0.x, m.c1.x, m.c2.x}, V3{m.c0.y, m.c1.y, m.c2.y}, V3{m.c0.z, m.c1.z, m.c2.z}};
}
EHD M33 mmul(const M33& a, const M33& b) { return M33{a * b.c0, a * b.c1, a * b.c2}; }  // PxMat33::operator*(PxMat33)
EHD V3 mtransformT(const M33& m, const V3& v) { return V3{sdot(m.c0, v), sdot(m.c1, v), sdot(m.c2, v)}; }  // transformTranspose
EHD float comp(const V3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
EHD void setComp(V3& v, int i, float f) { if (i == 0) v.x = f; else if (i == 1) v.y = f; else v.z = f; }

// ---- 상자 (rayAABBIntersect2)
EHD uint32_t rayAABBIntersect2(const V3& minimum, const V3& maximum, const V3& origin, const V3& dir, V3& coord, float& t) {
  const float RAYAABB_EPSILON = 0.00001f;
  bool inside = true;
  V3 maxT{-1.0f, -1.0f, -1.0f};
  for (int i = 0; i < 3; i++) {
    const float o = comp(origin, i), mn = comp(minimum, i), mx = comp(maximum, i), d = comp(dir, i);
    union { float f; uint32_t u; } id;
    id.f = d;
    if (o < mn) {
      setComp(coord, i, mn);
      inside = false;
      if (id.u) setComp(maxT, i, (mn - o) / d);
    } else if (o > mx) {
      setComp(coord, i, mx);
      inside = false;
      if (id.u) setComp(maxT, i, (mx - o) / d);
    }
  }
  if (inside) {
    coord = origin;
    t = 0;
    return 1;
  }
  int whichPlane = 0;
  if (maxT.y > comp(maxT, whichPlane)) whichPlane = 1;
  if (maxT.z > comp(maxT, whichPlane)) whichPlane = 2;
  union { float f; uint32_t u; } tw;
  tw.f = comp(maxT, whichPlane);
  if (tw.u & 0x80000000u) return 0;
  for (int i = 0; i < 3; i++) {
    if (i != whichPlane) {
      const float c = comp(origin, i) + tw.f * comp(dir, i);
      setComp(coord, i, c);
      if (c < comp(minimum, i) - RAYAABB_EPSILON || c > comp(maximum, i) + RAYAABB_EPSILON) return 0;
    }
  }
  t = tw.f;
  return 1 + uint32_t(whichPlane);
}
EHD uint32_t raycastBox(const V3& halfExtents, const Tf& pose, const V3& rayOrigin, const V3& rayDir, float maxDist, uint32_t hitFlags,
                        RayHit& hit) {
  const V3 localOrigin = rotateInv(pose.q, rayOrigin - pose.p);
  const V3 localDir = rotateInv(pose.q, rayDir);
  V3 localImpact{0, 0, 0};
  float t = 0.0f;
  uint32_t rval = rayAABBIntersect2(sneg(halfExtents), halfExtents, localOrigin, localDir, localImpact, t);
  if (!rval) return 0;
  if (t > maxDist) return 0;
  hit.distance = t;
  hit.faceIndex = 0xffffffffu;
  hit.u = 0.0f;
  hit.v = 0.0f;
  uint32_t outFlags = 0;
  if (hitFlags & HF_POSITION) {
    outFlags |= HF_POSITION;
    if (t != 0.0f) hit.position = rotate(pose.q, localImpact) + pose.p;  // PxTransform::transform
    else hit.position = rayOrigin;
  }
  if (hitFlags & HF_NORMAL) {
    outFlags |= HF_NORMAL;
    if (t == 0) {
      hit.normal = sneg(rayDir);
    } else {
      rval--;
      V3 n{0.0f, 0.0f, 0.0f};
      setComp(n, int(rval), (comp(localImpact, int(rval)) > 0.0f) ? 1.0f : -1.0f);
      hit.normal = rotate(pose.q, n);
    }
  } else {
    hit.normal = V3{0.0f, 0.0f, 0.0f};
  }
  hit.flags = outFlags;
  return 1;
}

// ---- 구 (intersectRaySphere + Basic)
EHD bool intersectRaySphereBasic(const V3& origin, const V3& dir, float length, const V3& center, float radius, float& dist, V3* hitPos) {
  const V3 offset = center - origin;
  const float ray_dist = sdot(dir, offset);
  const float off2 = sdot(offset, offset);
  const float rad_2 = radius * radius;
  if (off2 <= rad_2) {
    if (hitPos) *hitPos = origin;
    dist = 0.0f;
    return true;
  }
  if (ray_dist <= 0 || (ray_dist - length) > radius) return false;
  const float d = rad_2 - (off2 - ray_dist * ray_dist);
  if (d < 0.0f) return false;
  dist = ray_dist - psqrt(d);
  if (dist > length) return false;
  if (hitPos) *hitPos = origin + dir * dist;
  return true;
}
EHD bool intersectRaySphere(const V3& origin, const V3& dir, float length, const V3& center, float radius, float& dist, V3* hitPos) {
  const V3 x = origin - center;
  float l = psqrt(sdot(x, x)) - radius - 10.0f;  // GU_RAY_SURFACE_OFFSET
  l = pmax(l, 0.0f);                             // selectMax
  const bool status = intersectRaySphereBasic(origin + dir * l, dir, length - l, center, radius, dist, hitPos);  // l*dir (float*vec)
  if (status) dist += l;
  return status;
}
EHD uint32_t raycastSphere(float radius, const Tf& pose, const V3& rayOrigin, const V3& rayDir, float maxDist, uint32_t hitFlags, RayHit& hit) {
  if (!intersectRaySphere(rayOrigin, rayDir, maxDist, pose.p, radius, hit.distance, &hit.position)) return 0;
  hit.faceIndex = 0xffffffffu;
  hit.u = 0.0f;
  hit.v = 0.0f;
  uint32_t outFlags = HF_POSITION;
  if (hitFlags & HF_NORMAL) {
    if (hit.distance == 0.0f) hit.normal = sneg(rayDir);
    else hit.normal = snormalize(hit.position - pose.p);
    outFlags |= HF_NORMAL;
  } else {
    hit.normal = V3{0.0f, 0.0f, 0.0f};
  }
  hit.flags = outFlags;
  return 1;
}

// ---- 캡슐 (x 축 방향, 반높이 halfHeight)
EHD float distancePointSegmentSquaredInternal(const V3& p0, const V3& dir, const V3& point, float* param) {
  V3 diff = point - p0;
  float fT = sdot(diff, dir);
  if (fT <= 0.0f) {
    fT = 0.0f;
  } else {
    const float sqrLen = magSq(dir);
    if (fT >= sqrLen) {
      fT = 1.0f;
      diff = diff - dir;
    } else {
      fT /= sqrLen;
      diff = diff - V3{fT * dir.x, fT * dir.y, fT * dir.z};
    }
  }
  if (param) *param = fT;
  return magSq(diff);
}
EHD bool capsuleSphere(const V3& rayOrigin, const V3& rayDir, const V3& c, float radius2, float& tmin, float& tmax) {
  const V3 CO = rayOrigin - c;
  const float a = sdot(rayDir, rayDir);
  const float b = 2.0f * sdot(CO, rayDir);
  const float cc = sdot(CO, CO) - radius2;
  const float discriminant = b * b - 4.0f * a * cc;
  if (discriminant < 0.0f) return false;
  const float OneOver2A = 1.0f / (2.0f * a);
  const float sqrtDet = psqrt(discriminant);
  tmin = (-b - sqrtDet) * OneOver2A;
  tmax = (-b + sqrtDet) * OneOver2A;
  if (tmin > tmax) { const float t = tmin; tmin = tmax; tmax = t; }
  return true;
}
EHD uint32_t intersectRayCapsuleInternal(const V3& rayOrigin, const V3& rayDir, const V3& p0, const V3& p1, float radius, float s[2]) {
  const float radius2 = radius * radius;
  const V3 AB = p1 - p0;
  const V3 AO = rayOrigin - p0;
  const float AB_dot_d = sdot(AB, rayDir);
  const float AB_dot_AO = sdot(AB, AO);
  const float AB_dot_AB = sdot(AB, AB);
  const float OneOverABDotAB = AB_dot_AB != 0.0f ? 1.0f / AB_dot_AB : 0.0f;
  const float m = AB_dot_d * OneOverABDotAB;
  const float n = AB_dot_AO * OneOverABDotAB;
  const V3 Q = rayDir - (AB * m);
  const V3 R = AO - (AB * n);
  const float a = sdot(Q, Q);
  const float b = 2.0f * sdot(Q, R);
  const float c = sdot(R, R) - radius2;
  if (a == 0.0f) {
    float atmin, atmax, btmin, btmax;
    if (!capsuleSphere(rayOrigin, rayDir, p0, radius2, atmin, atmax) || !capsuleSphere(rayOrigin, rayDir, p1, radius2, btmin, btmax)) return 0;
    s[0] = atmin < btmin ? atmin : btmin;
    return 1;
  }
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) return 0;
  const float OneOver2A = 1.0f / (2.0f * a);
  const float sqrtDet = psqrt(discriminant);
  float tmin = (-b - sqrtDet) * OneOver2A;
  float tmax = (-b + sqrtDet) * OneOver2A;
  if (tmin > tmax) { const float t = tmin; tmin = tmax; tmax = t; }
  const float t_k1 = tmin * m + n;
  if (t_k1 < 0.0f) {
    float stmin, stmax;
    if (capsuleSphere(rayOrigin, rayDir, p0, radius2, stmin, stmax)) s[0] = stmin;
    else return 0;
  } else if (t_k1 > 1.0f) {
    float stmin, stmax;
    if (capsuleSphere(rayOrigin, rayDir, p1, radius2, stmin, stmax)) s[0] = stmin;
    else return 0;
  } else {
    s[0] = tmin;
  }
  return 1;
}
EHD bool intersectRayCapsule(const V3& origin, const V3& dir, const V3& p0, const V3& p1, float radius, float& t) {
  float l = distancePointSegmentSquaredInternal(p0, p1 - p0, origin, nullptr);
  l = psqrt(l) - radius;
  if (l <= 0.0f) {
    t = 0.0f;
    return true;
  }
  if (l > 10.0f) l -= 10.0f;
  else l = 0.0f;
  float s[2];
  const uint32_t nbHits = intersectRayCapsuleInternal(origin + dir * l, dir, p0, p1, radius, s);
  if (!nbHits) return false;
  if (nbHits == 1) t = s[0];
  else t = (s[0] < s[1]) ? s[0] : s[1];
  t += l;
  return true;
}
EHD V3 basisVector0(const Q& q) {  // PxQuat::getBasisVector0
  const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
  return V3{(q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2};
}
EHD uint32_t raycastCapsule(float radius, float halfHeight, const Tf& pose, const V3& rayOrigin, const V3& rayDir, float maxDist, uint32_t hitFlags,
                            RayHit& hit) {
  const V3 tmp = basisVector0(pose.q) * halfHeight;
  const V3 p0 = pose.p + tmp, p1 = pose.p - tmp;
  float t = 0.0f;
  if (!intersectRayCapsule(rayOrigin, rayDir, p0, p1, radius, t)) return 0;
  if (t < 0.0f || t > maxDist) return 0;
  hit.position = rayOrigin + rayDir * t;
  hit.distance = t;
  hit.faceIndex = 0xffffffffu;
  hit.u = 0.0f;
  hit.v = 0.0f;
  uint32_t outFlags = HF_POSITION;
  if (hitFlags & HF_NORMAL) {
    outFlags |= HF_NORMAL;
    if (t == 0.0f) {
      hit.normal = sneg(rayDir);
    } else {
      float capsuleT;
      distancePointSegmentSquaredInternal(p0, p1 - p0, hit.position, &capsuleT);
      const V3 d = p1 - p0;
      const V3 pt = p0 + V3{capsuleT * d.x, capsuleT * d.y, capsuleT * d.z};  // computePoint: p0 + t*(p1-p0)
      hit.normal = snormalize(hit.position - pt);
    }
  } else {
    hit.normal = V3{0.0f, 0.0f, 0.0f};
  }
  hit.flags = outFlags;
  return 1;
}

// ---- 평면 (x 축이 법선)
EHD uint32_t raycastPlane(const Tf& pose, const V3& rayOrigin, const V3& rayDir, float maxDist, RayHit& hit) {
  const V3 n = basisVector0(pose.q);
  const float d = -sdot(pose.p, n);
  if (sdot(rayDir, n) >= 0.0f) return 0;
  const float dn = sdot(rayDir, n);
  if (-1E-7f < dn && dn < 1E-7f) return 0;
  const float distanceAlongLine = -(sdot(rayOrigin, n) + d) / dn;
  hit.position = rayOrigin + V3{distanceAlongLine * rayDir.x, distanceAlongLine * rayDir.y, distanceAlongLine * rayDir.z};
  if (distanceAlongLine < 0.0f) return 0;
  if (distanceAlongLine > maxDist) return 0;
  hit.distance = distanceAlongLine;
  hit.faceIndex = 0xffffffffu;
  hit.u = 0.0f;
  hit.v = 0.0f;
  hit.flags = HF_POSITION | HF_NORMAL;
  hit.normal = n;
  return 1;
}

// ---- 볼록 메시 (평면 방식)
struct MeshScale { V3 scale; Q rotation; };  // PxMeshScale
// Cm::toMat33 (CmScaling.h:45): rot^T * diag(scale) * rot
EHD M33 scaleToMat33(const MeshScale& s) {
  const M33 rot = mat_from_quat_simd(s.rotation);  // PxMat33Padded
  M33 trans = mtranspose(rot);
  trans.c0 *= s.scale.x;
  trans.c1 *= s.scale.y;
  trans.c2 *= s.scale.z;
  return mmul(trans, rot);
}
struct Mat34 { M33 m; V3 p; };
EHD V3 m34transform(const Mat34& a, const V3& v) { return a.m * v + a.p; }
EHD uint32_t raycastConvex(const float* planes /* nPolys x (nx,ny,nz,d) */, uint32_t nPolys, const MeshScale& scale, const Tf& pose,
                           const V3& rayOrigin, const V3& rayDir, float maxDist, uint32_t hitFlags, RayHit& hit) {
  // world2vertexSkew = scale.getInverse() * pose.getInverse()  (CmScaling.h:219)
  const MeshScale inv{V3{1.0f / scale.scale.x, 1.0f / scale.scale.y, 1.0f / scale.scale.z}, scale.rotation};
  const Tf poseInv{conj(pose.q), rotateInv(pose.q, sneg(pose.p))};  // PxTransform::getInverse
  const M33 scaleMat = scaleToMat33(inv);
  const M33 t = mat_from_quat_simd(poseInv.q);
  const Mat34 w2v{mmul(scaleMat, t), scaleMat * poseInv.p};
  const V3 vrayOrig = m34transform(w2v, rayOrigin);
  const V3 vrayDir = w2v.m * rayDir;
  bool originInsideAllPlanes = true;
  float latestEntry = -3.40282346638528859812e+38F;
  float earliestExit = 3.40282346638528859812e+38F;
  hit.faceIndex = 0xffffffffu;
  for (uint32_t i = 0; i < nPolys; i++) {
    const V3 n{planes[4 * i + 0], planes[4 * i + 1], planes[4 * i + 2]};
    const float pd = planes[4 * i + 3];
    const float distToPlane = sdot(vrayOrig, n) + pd;  // PxPlane::distance
    const float dn = sdot(n, vrayDir);
    const float distAlongRay = -distToPlane / dn;
    if (distToPlane > 0.0f) originInsideAllPlanes = false;
    if (dn > 1E-7f) {
      earliestExit = earliestExit < distAlongRay ? earliestExit : distAlongRay;  // selectMin
    } else if (dn < -1E-7f) {
      if (distAlongRay > latestEntry) {
        latestEntry = distAlongRay;
        hit.faceIndex = i;
      }
    } else {
      if (distToPlane > 0.0f) return 0;
    }
  }
  if (originInsideAllPlanes) {
    hit.distance = 0.0f;
    hit.faceIndex = 0xffffffffu;
    hit.u = 0.0f;
    hit.v = 0.0f;
    hit.position = rayOrigin;
    hit.normal = sneg(rayDir);
    hit.flags = HF_NORMAL | HF_POSITION;
    return 1;
  }
  if (latestEntry < earliestExit && latestEntry > 0.0f && latestEntry < maxDist - 1e-5f) {
    uint32_t outFlags = HF_FACE_INDEX;
    if (hitFlags & HF_POSITION) {
      outFlags |= HF_POSITION;
      const V3 pointOnPlane = vrayOrig + V3{latestEntry * vrayDir.x, latestEntry * vrayDir.y, latestEntry * vrayDir.z};
      hit.position = rotate(pose.q, scaleToMat33(scale) * pointOnPlane) + pose.p;
    }
    hit.distance = latestEntry;
    hit.u = 0.0f;
    hit.v = 0.0f;
    hit.normal = V3{0.0f, 0.0f, 0.0f};
    if (hitFlags & HF_NORMAL) {
      outFlags |= HF_NORMAL;
      const uint32_t f = hit.faceIndex;
      hit.normal = snormalize(mtransformT(w2v.m, V3{planes[4 * f + 0], planes[4 * f + 1], planes[4 * f + 2]}));  // rotateTranspose
    }
    hit.flags = outFlags;
    return 1;
  }
  return 0;
}

// ---- 삼각 메시 (BV4 midphase, 가장 가까운 한 개)
// 원본: GuMidphaseBV4.cpp:367 (raycast_triangleMesh_BV4), :240 (raycastVsMesh), :100,129 (setRotation·setupWorldMatrix), :306 (processLocalNormal)
//       mesh/GuBV4_Raycast.cpp:290 (RayTriOverlapT), :423 (LeafFunction_RaycastClosest: 더 가까울 때만 교체), :513,525 (computeImpactData),
//       :573,585 (clipRay·setupRayParams), GuBV4_Common.h:282-336 (inverseRotate·inverseTransform·computeLocalRay), GuBV4.h:288 (LocalBounds)
// BV4 나무는 "가장 가까운 삼각형"을 빨리 찾는 수단일 뿐이라 전부 훑어도 결과는 같다(교체 조건이 거리 '<' 라 거리가 똑같은 두 삼각형이면
// PhysX 는 나무 방문 순서상 먼저 만난 것, 우리는 번호가 작은 것 — 그때만 다를 수 있다). 노드 상자 판정은 보수적이라 맞을 삼각형을 버리지 않는다.
// 입력은 구운(cooked) 메시 그대로: 정점, 내부 순서 삼각형(PxTriangleMesh::getTriangles 순서 = faceIndex), 나무의 LocalBounds, geomEpsilon.
struct TriMeshData {
  const V3* verts;
  const uint32_t* tris;  // 3 x nTris (16비트 메시도 32비트로 넓혀서)
  uint32_t nTris;
  V3 boundsCenter;        // BV4Tree::mLocalBounds.mCenter
  float extentsMagnitude; // BV4Tree::mLocalBounds.mExtentsMagnitude
  float geomEpsilon;      // TriangleMesh::mGeomEpsilon
};
struct Mat44W { V3 c0, c1, c2, c3; };  // PxMat44 의 쓰는 칸 (w 칸은 0/1 고정)

EHD float sfabs(float x) { union { float f; uint32_t u; } c; c.f = x; c.u &= 0x7fffffffu; return c.f; }  // fabsf (부호 비트 지움)

EHD bool rayTriOverlap(const V3& vert0, const V3& vert1, const V3& vert2, const V3& localDir, const V3& origin, float geomEpsilon, bool backfaceCulling,
                       float& dist, float& uu, float& vv) {
  const float CULL = 1.1920928955078125e-07f * 1.1920928955078125e-07f;  // FLT_EPSILON*FLT_EPSILON
  const V3 edge1 = vert1 - vert0;
  const V3 edge2 = vert2 - vert0;
  const V3 pvec = cross(localDir, edge2);
  const float det = sdot(edge1, pvec);
  if (backfaceCulling) {
    if (det < CULL) return false;
    const V3 tvec = origin - vert0;
    const float u = sdot(tvec, pvec);
    const float enlargeCoeff = geomEpsilon * det;
    const float uvlimit = -enlargeCoeff;
    const float uvlimit2 = det + enlargeCoeff;
    if (u < uvlimit || u > uvlimit2) return false;
    const V3 qvec = cross(tvec, edge1);
    const float v = sdot(localDir, qvec);
    if (v < uvlimit || (u + v) > uvlimit2) return false;
    const float d = sdot(edge2, qvec);
    if (d < 0.0f) return false;
    const float OneOverDet = 1.0f / det;
    dist = d * OneOverDet;
    uu = u * OneOverDet;
    vv = v * OneOverDet;
  } else {
    if (sfabs(det) < CULL) return false;
    const float OneOverDet = 1.0f / det;
    const V3 tvec = origin - vert0;
    const float u = sdot(tvec, pvec) * OneOverDet;
    if (u < -geomEpsilon || u > 1.0f + geomEpsilon) return false;
    const V3 qvec = cross(tvec, edge1);
    const float v = sdot(localDir, qvec) * OneOverDet;
    if (v < -geomEpsilon || (u + v) > 1.0f + geomEpsilon) return false;
    const float d = sdot(edge2, qvec) * OneOverDet;
    if (d < 0.0f) return false;
    dist = d;
    uu = u;
    vv = v;
  }
  return true;
}

// BV4_RaycastSingle (closest). wm = nullptr 이면 항등. 성공 시 hit 의 distance/u/v/faceIndex/position/normal(정규화, 뒤집기 전)
EHD bool bv4RaycastSingle(const TriMeshData& m, const V3& origin, const V3& dir, const Mat44W* wm, float maxDist, bool bothSides, RayHit& hit) {
  V3 ld, lo;
  if (wm) {  // inverseRotate / inverseTransform (GuBV4_Common.h:282,301)
    ld = V3{wm->c0.x * dir.x + wm->c0.y * dir.y + wm->c0.z * dir.z, wm->c1.x * dir.x + wm->c1.y * dir.y + wm->c1.z * dir.z,
            wm->c2.x * dir.x + wm->c2.y * dir.y + wm->c2.z * dir.z};
    const V3 p = origin, t = wm->c3;
    lo = V3{wm->c0.x * p.x + wm->c0.y * p.y + wm->c0.z * p.z - (t.x * wm->c0.x + t.y * wm->c0.y + t.z * wm->c0.z),
            wm->c1.x * p.x + wm->c1.y * p.y + wm->c1.z * p.z - (t.x * wm->c1.x + t.y * wm->c1.y + t.z * wm->c1.z),
            wm->c2.x * p.x + wm->c2.y * p.y + wm->c2.z * p.z - (t.x * wm->c2.x + t.y * wm->c2.y + t.z * wm->c2.z)};
  } else {
    ld = dir;
    lo = origin;
  }
  {  // clipRay (GuBV4_Raycast.cpp:573)
    const float dpc = sdot(m.boundsCenter, ld);
    const float dpMin = dpc - m.extentsMagnitude;
    const float dpMax = dpc + m.extentsMagnitude;
    const float dpO = sdot(lo, ld);
    const float boxLength = m.extentsMagnitude * 2.0f;
    const float a = sfabs(dpMin - dpO), b = sfabs(dpMax - dpO);
    const float distToBox = a < b ? a : b;  // PxMin
    const float MaxDist = distToBox + boxLength * 2.0f;
    maxDist = maxDist < MaxDist ? maxDist : MaxDist;
  }
  const bool cull = !bothSides;
  float best = maxDist;
  uint32_t bestId = 0xffffffffu;
  float bu = 0.0f, bv = 0.0f;
  for (uint32_t i = 0; i < m.nTris; ++i) {
    float d, u, v;
    if (rayTriOverlap(m.verts[m.tris[3 * i]], m.verts[m.tris[3 * i + 1]], m.verts[m.tris[3 * i + 2]], ld, lo, m.geomEpsilon, cull, d, u, v) && d < best) {
      best = d; bu = u; bv = v; bestId = i;
    }
  }
  if (bestId == 0xffffffffu) return false;
  hit.u = bu;
  hit.v = bv;
  hit.distance = best;
  hit.faceIndex = bestId;
  const V3 P0 = m.verts[m.tris[3 * bestId]], P1 = m.verts[m.tris[3 * bestId + 1]], P2 = m.verts[m.tris[3 * bestId + 2]];
  const float w = 1.0f - bu - bv;
  V3 lp = P1 * bu;  // computeImpactData: P1*u + P2*v + P0*w
  lp = lp + P2 * bv;
  lp = lp + P0 * w;
  const V3 ln = cross(P0 - P1, P0 - P2);
  V3 n;
  if (wm) {  // multiply3x3V_Aligned: c0*x + c1*y + c2*z (+ c3)
    V3 r = wm->c0 * lp.x;
    r = r + wm->c1 * lp.y;
    r = r + wm->c2 * lp.z;
    hit.position = r + wm->c3;
    n = wm->c0 * ln.x;
    n = n + wm->c1 * ln.y;
    n = n + wm->c2 * ln.z;
  } else {
    hit.position = lp;
    n = ln;
  }
  hit.normal = snormalize(n);
  return true;
}

// PxGeometryQuery::raycast(PxTriangleMeshGeometry) 한 개 결과 (eMESH_MULTIPLE 없음). doubleSided = PxMeshGeometryFlag::eDOUBLE_SIDED,
// hitFlags 의 HF_MESH_BOTH_SIDES 도 받는다. scale 이 항등이 아니면 정점 공간으로 광선을 옮긴다.
enum : uint32_t { HF_MESH_BOTH_SIDES = 1 << 7 };
EHD uint32_t raycastTriangleMesh(const TriMeshData& m, const MeshScale& scale, bool doubleSided, const Tf& pose, const V3& rayOrigin, const V3& rayDir,
                                 float maxDist, uint32_t hitFlags, RayHit& hit) {
  const bool idtScale = scale.scale.x == 1.0f && scale.scale.y == 1.0f && scale.scale.z == 1.0f;
  const bool bothSides = doubleSided || (hitFlags & HF_MESH_BOTH_SIDES);
  if (idtScale) {
    // setupWorldMatrix: 회전이 (0,0,0,1) 비트가 아니면 setRotation, 위치가 0 비트가 아니면 넣음. 둘 다 항등이면 NULL
    union { float f; uint32_t u; } a, b, c, d;
    Mat44W W{V3{1, 0, 0}, V3{0, 1, 0}, V3{0, 0, 1}, V3{0, 0, 0}};
    bool isIdt = true;
    a.f = pose.q.x; b.f = pose.q.y; c.f = pose.q.z; d.f = pose.q.w;
    if (a.u != 0 || b.u != 0 || c.u != 0 || d.u != 0x3f800000u) {
      const float x = pose.q.x, y = pose.q.y, z = pose.q.z, ww = pose.q.w;
      const float x2 = x + x, y2 = y + y, z2 = z + z;
      const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
      const float xy = x2 * y, xz = x2 * z, xw = x2 * ww;
      const float yz = y2 * z, yw = y2 * ww, zw = z2 * ww;
      W.c0 = V3{1.0f - yy - zz, xy + zw, xz - yw};
      W.c1 = V3{xy - zw, 1.0f - xx - zz, yz + xw};
      W.c2 = V3{xz + yw, yz - xw, 1.0f - xx - yy};
      isIdt = false;
    }
    a.f = pose.p.x; b.f = pose.p.y; c.f = pose.p.z;
    if (a.u != 0 || b.u != 0 || c.u != 0) {
      W.c3 = pose.p;
      isIdt = false;
    }
    if (!bv4RaycastSingle(m, rayOrigin, rayDir, isIdt ? nullptr : &W, maxDist, bothSides, hit)) return 0;
    uint32_t dst = HF_POSITION | HF_UV | HF_FACE_INDEX;
    if (hitFlags & HF_NORMAL) {
      dst |= HF_NORMAL;
      if (doubleSided && sdot(hit.normal, rayDir) > 0.0f) hit.normal = sneg(hit.normal);
    } else {
      hit.normal = V3{0.0f, 0.0f, 0.0f};
    }
    hit.flags = dst;
    return 1;
  }
  // world2vertexSkew = scale.getInverse() * pose.getInverse() (CmScaling.h:219)
  const MeshScale inv{V3{1.0f / scale.scale.x, 1.0f / scale.scale.y, 1.0f / scale.scale.z}, scale.rotation};
  const Tf poseInv{conj(pose.q), rotateInv(pose.q, sneg(pose.p))};
  const M33 scaleMat = scaleToMat33(inv);
  const M33 t = mat_from_quat_simd(poseInv.q);
  const Mat34 w2v{mmul(scaleMat, t), scaleMat * poseInv.p};
  const V3 orig = m34transform(w2v, rayOrigin);
  V3 dir = w2v.m * rayDir;
  float distCoeff = psqrt(magSq(dir));  // PxVec3::normalize 가 돌려주는 크기
  if (distCoeff > 0.0f) {
    const float f = 1.0f / distCoeff;
    dir.x *= f; dir.y *= f; dir.z *= f;
  }
  maxDist *= distCoeff;
  maxDist += 1e-3f;
  distCoeff = 1.0f / distCoeff;
  if (!bv4RaycastSingle(m, orig, dir, nullptr, maxDist, bothSides, hit)) return 0;
  hit.distance *= distCoeff;
  // pose.transform(scale.transform(p)): scale.transform = rotation.rotateInv(scale.multiply(rotation.rotate(v)))
  const V3 r = rotate(scale.rotation, hit.position);
  const V3 sp = rotateInv(scale.rotation, V3{scale.scale.x * r.x, scale.scale.y * r.y, scale.scale.z * r.z});
  hit.position = rotate(pose.q, sp) + pose.p;
  uint32_t dst = HF_POSITION | HF_UV | HF_FACE_INDEX;
  if (scale.scale.x * scale.scale.y * scale.scale.z < 0.0f) {  // hasNegativeDeterminant: u,v 바꿈
    const float tmp = hit.u; hit.u = hit.v; hit.v = tmp;
  }
  if (hitFlags & HF_NORMAL) {
    dst |= HF_NORMAL;
    V3 n = snormalize(mtransformT(w2v.m, hit.normal));  // processLocalNormal: rotateTranspose -> normalize
    if (doubleSided && sdot(n, rayDir) > 0.0f) n = sneg(n);
    hit.normal = n;
  } else {
    hit.normal = V3{0.0f, 0.0f, 0.0f};
  }
  hit.flags = dst;
  return 1;
}

}  // namespace sq
}  // namespace eng
