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

}  // namespace sq
}  // namespace eng
