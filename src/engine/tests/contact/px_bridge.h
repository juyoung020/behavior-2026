// contact 시험 공용: PhysX(정답지) 쪽 객체 -> 우리 엔진(eng::px) 입력으로 옮기는 다리, FTZ 구간, 볼록 굽기, 비트 비교 집계.
// 시험 프로그램에서만 쓴다 (PhysX 링크). 우리 엔진 코드는 이 파일을 모른다.
#pragma once
#include <xmmintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "PxPhysicsAPI.h"
#include "GuConvexMesh.h"
#include "GuBigConvexData.h"
#include "GuContactMethodImpl.h"
#include "GuPersistentContactManifold.h"
#include "geomutils/PxContactBuffer.h"

#include "core/contact/px/gu.h"
#include "core/contact/narrowphase.h"

namespace cxt {

struct FtzScope {  // PhysX CmTask 의 PX_SIMD_GUARD 와 같은 MXCSR (FTZ + DAZ, 예외 가림) — 좁은 단계는 이 안에서 돈다
  unsigned old;
  FtzScope() { old = _mm_getcsr(); _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)); }
  ~FtzScope() { _mm_setcsr(old & ~unsigned(_MM_EXCEPT_MASK)); }
};

inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// 비트 비교 집계 (solver 시험과 같은 보고 형식: 비교 수 / 비트 다름 / 첫 다름 / 최대 |차|)
struct Tally {
  const char* name;
  uint64_t cmp = 0, bad = 0;
  long long firstCase = -1, firstFrame = -1;
  double maxd = 0.0;
  explicit Tally(const char* n) : name(n) {}
  void f(float a, float b, long long c, long long fr) {
    ++cmp;
    const bool same = fbits(a) == fbits(b) || (std::isnan(a) && std::isnan(b));
    if (!same) {
      if (!bad) { firstCase = c; firstFrame = fr; }
      ++bad;
      const double d = std::fabs(double(a) - double(b));
      if (d > maxd || std::isnan(d)) maxd = d;
    }
  }
  void u(uint64_t a, uint64_t b, long long c, long long fr) {
    ++cmp;
    if (a != b) {
      if (!bad) { firstCase = c; firstFrame = fr; }
      ++bad;
      const double d = std::fabs(double(a) - double(b));
      if (d > maxd) maxd = d;
    }
  }
  void print() const {
    printf("  %-28s 비교 %12" PRIu64 "  비트 다름 %10" PRIu64, name, cmp, bad);
    if (bad) printf("  첫 다름 (쌍 %lld, 프레임 %lld)  최대|차| %.3g", firstCase, firstFrame, maxd);
    printf("\n");
  }
};

// ---- 볼록 굽기 (PhysX cooking): 무작위 점 구름 -> PxConvexMesh
struct CookedHull {
  physx::PxConvexMesh* mesh = nullptr;
  int kind = 0;
};

inline physx::PxConvexMesh* cookHull(physx::PxPhysics* phys, const std::vector<physx::PxVec3>& pts, bool gpuData, physx::PxU16 vertexLimit) {
  using namespace physx;
  PxTolerancesScale tol(1.0f, 10.0f);
  PxCookingParams cp(tol);
  cp.buildGPUData = gpuData;
  PxConvexMeshDesc desc;
  desc.points.count = PxU32(pts.size());
  desc.points.stride = sizeof(PxVec3);
  desc.points.data = pts.data();
  desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
  desc.vertexLimit = vertexLimit;
  return PxCreateConvexMesh(cp, desc, phys->getPhysicsInsertionCallback());
}

// 여러 모양의 점 구름 (BEHAVIOR 물체 볼록 조각 흉내: 상자형·타원체·원기둥·납작한 판)
inline std::vector<physx::PxVec3> randomCloud(std::mt19937& rng, int kind, int n) {
  using physx::PxVec3;
  std::uniform_real_distribution<float> U(-1.0f, 1.0f), P(0.0f, 1.0f);
  const PxVec3 ax(0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng), 0.05f + 0.5f * P(rng));
  std::vector<PxVec3> v;
  for (int i = 0; i < n; ++i) {
    PxVec3 p;
    switch (kind % 4) {
      case 0: {  // 타원체 표면 근처
        PxVec3 d(U(rng), U(rng), U(rng));
        const float m = d.magnitude();
        d = m > 1e-3f ? d / m : PxVec3(1, 0, 0);
        p = PxVec3(d.x * ax.x, d.y * ax.y, d.z * ax.z) * (0.9f + 0.1f * P(rng));
        break;
      }
      case 1: {  // 상자 모서리 + 떨림
        p = PxVec3((i & 1 ? 1 : -1) * ax.x, (i & 2 ? 1 : -1) * ax.y, (i & 4 ? 1 : -1) * ax.z) + PxVec3(U(rng), U(rng), U(rng)) * 0.01f;
        break;
      }
      case 2: {  // 원기둥 테두리
        const float a = 6.2831853f * float(i) / float(n);
        p = PxVec3(std::cos(a) * ax.x, std::sin(a) * ax.y, (i & 1 ? 1 : -1) * ax.z);
        break;
      }
      default:  // 속이 찬 구름 (내부 점 많음)
        p = PxVec3(U(rng) * ax.x, U(rng) * ax.y, U(rng) * ax.z * 0.2f);
    }
    v.push_back(p);
  }
  return v;
}

// ---- PhysX 모양 -> 우리 모양 (같은 구운 데이터를 가리킴: 호스트 시험용. GPU 는 deepCopy 로 옮긴다)
inline eng::px::PxMeshScale toE(const physx::PxMeshScale& s) {
  eng::px::PxMeshScale o;
  o.scale = eng::px::PxVec3(s.scale.x, s.scale.y, s.scale.z);
  o.rotation = eng::px::PxQuat(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w);
  return o;
}
inline const eng::px::Gu::ConvexHullData* hullOf(const physx::PxConvexMesh* m) {
  static_assert(sizeof(eng::px::Gu::ConvexHullData) == sizeof(physx::Gu::ConvexHullData), "ConvexHullData 배치");
  return reinterpret_cast<const eng::px::Gu::ConvexHullData*>(&static_cast<const physx::Gu::ConvexMesh*>(m)->getHull());
}
inline eng::px::PxConvexMeshGeometry toE(const physx::PxConvexMeshGeometry& g) {
  return eng::px::PxConvexMeshGeometry(hullOf(g.convexMesh), toE(g.scale), g.meshFlags);
}
inline eng::px::PxTransform toE(const physx::PxTransform& t) {
  return eng::px::PxTransform(eng::px::PxVec3(t.p.x, t.p.y, t.p.z), eng::px::PxQuat(t.q.x, t.q.y, t.q.z, t.q.w));
}


// ---- 모양 만들기 (PhysX 쪽 + 우리 쪽 같은 값) 와 PhysX PCM 함수 부르기 (PxcNpBatch 처럼 모양 번호 순서 맞추고 뒤집으면 법선 반전)
struct TestShape {
  physx::PxGeometryHolder p;
  eng::contact::ShapeGeom e;
  float radius = 0.0f;  // 대략 크기
};

inline void makeShape(int type, std::mt19937& rng, const std::vector<physx::PxConvexMesh*>& hulls, TestShape& s) {
  using namespace physx;
  std::uniform_real_distribution<float> P(0.0f, 1.0f);
  s.e = eng::contact::ShapeGeom();
  s.e.type = type;
  switch (type) {
    case 0: { const float r = 0.02f + 0.4f * P(rng); s.p.storeAny(PxSphereGeometry(r)); s.e.sphere = eng::px::PxSphereGeometry(r); s.radius = r; break; }
    case 1: { s.p.storeAny(PxPlaneGeometry()); s.radius = 0.0f; break; }
    case 2: {
      const float r = 0.02f + 0.3f * P(rng), hh = 0.01f + 0.5f * P(rng);
      s.p.storeAny(PxCapsuleGeometry(r, hh)); s.e.capsule = eng::px::PxCapsuleGeometry(r, hh); s.radius = r + hh; break;
    }
    case 3: {
      const PxVec3 he(0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng), 0.02f + 0.4f * P(rng));
      s.p.storeAny(PxBoxGeometry(he)); s.e.box = eng::px::PxBoxGeometry(he.x, he.y, he.z); s.radius = he.magnitude(); break;
    }
    default: {
      PxConvexMesh* m = hulls[size_t(P(rng) * hulls.size()) % hulls.size()];
      PxMeshScale sc = (P(rng) < 0.5f) ? PxMeshScale(1.0f) : PxMeshScale(PxVec3(0.5f + P(rng), 0.5f + P(rng), 0.5f + P(rng)));
      const PxConvexMeshGeometry g(m, sc);
      s.p.storeAny(g); s.e.convex = toE(g);
      s.radius = static_cast<Gu::ConvexMesh*>(m)->getLocalBoundsFast().mExtents.magnitude() * sc.scale.maxElement();
    }
  }
}

inline bool physxPcmOrdered(int t0, int t1, const physx::PxGeometry& a, const physx::PxGeometry& b, const physx::PxTransform32& ta,
                            const physx::PxTransform32& tb, const physx::Gu::NarrowPhaseParams& np, physx::Gu::Cache& c, physx::PxContactBuffer& buf) {
  using namespace physx;
  switch (t0 * 16 + t1) {
    case 0 * 16 + 0: return Gu::pcmContactSphereSphere(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 16 + 1: return Gu::pcmContactSpherePlane(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 16 + 2: return Gu::pcmContactSphereCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 16 + 3: return Gu::pcmContactSphereBox(a, b, ta, tb, np, c, buf, NULL);
    case 0 * 16 + 5: return Gu::pcmContactSphereConvex(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 16 + 2: return Gu::pcmContactPlaneCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 16 + 3: return Gu::pcmContactPlaneBox(a, b, ta, tb, np, c, buf, NULL);
    case 1 * 16 + 5: return Gu::pcmContactPlaneConvex(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 16 + 2: return Gu::pcmContactCapsuleCapsule(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 16 + 3: return Gu::pcmContactCapsuleBox(a, b, ta, tb, np, c, buf, NULL);
    case 2 * 16 + 5: return Gu::pcmContactCapsuleConvex(a, b, ta, tb, np, c, buf, NULL);
    case 3 * 16 + 3: return Gu::pcmContactBoxBox(a, b, ta, tb, np, c, buf, NULL);
    case 3 * 16 + 5: return Gu::pcmContactBoxConvex(a, b, ta, tb, np, c, buf, NULL);
    case 5 * 16 + 5: return Gu::pcmContactConvexConvex(a, b, ta, tb, np, c, buf, NULL);
  }
  return false;
}

// PhysX 쪽 다양체 칸 (PxsContext::createCache 와 같게)
struct PhysxSlot {
  alignas(16) physx::Gu::LargePersistentContactManifold large;
  alignas(16) physx::Gu::SpherePersistentContactManifold sphere;
  int kind = 0;
  physx::Gu::PersistentContactManifold* get() { return kind == 1 ? static_cast<physx::Gu::PersistentContactManifold*>(&sphere) : &large; }
  void init(int t0, int t1) {
    const int a = t0 < t1 ? t0 : t1, b = t0 < t1 ? t1 : t0;
    if (eng::contact::pcmCaching(a, b)) {
      kind = (a == 0 || b == 0) ? 1 : 2;
      get()->clearManifold();
    } else {
      kind = 0;
    }
  }
};

// PxcNpBatch::discreteNarrowPhase 의 뒤집기 규칙대로 PhysX PCM 을 부른다
inline bool physxPcmPair(const TestShape& s0, const TestShape& s1, const physx::PxTransform& t0, const physx::PxTransform& t1, float contactDist,
                         PhysxSlot& slot, physx::PxContactBuffer& buf) {
  using namespace physx;
  int ty0 = s0.e.type, ty1 = s1.e.type;
  const bool flip = ty1 < ty0;
  const TestShape* a = &s0; const TestShape* b = &s1;
  PxTransform32 ta(t0), tb(t1);
  if (flip) { std::swap(ty0, ty1); std::swap(a, b); std::swap(ta, tb); }
  const Gu::NarrowPhaseParams np(contactDist, 0.01f, 1.0f);
  Gu::Cache c;
  if (slot.kind) c.setManifold(slot.get());
  buf.reset();
  const bool r = physxPcmOrdered(ty0, ty1, a->p.any(), b->p.any(), ta, tb, np, c, buf);
  if (flip) for (PxU32 i = 0; i < buf.count; ++i) buf.contacts[i].normal = -buf.contacts[i].normal;
  return r;
}

}  // namespace cxt
