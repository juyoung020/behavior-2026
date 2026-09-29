// 좁은 단계 한 쌍 처리 (호스트·GPU 공용): 모양 쌍 + 두 자세 + 지속 다양체 -> PhysX PCM 접촉 함수 결과.
// PhysX 흐름 (PxcNpBatch.cpp:368 discreteNarrowPhase):
//   모양 번호가 뒤 < 앞이면 두 모양·자세를 바꿔(flip) 부르고 -> g_PCMContactMethodTable[type0][type1] -> 뒤집었으면 법선 부호를 바꾼다
//   (flipContacts:65, 점·분리는 그대로). 다양체 종류는 PxsContext::createCache(PxsContext.cpp:282) 규칙.
// 접촉 관리자(쌍 목록·생성 순서)·재질·패치 압축은 이 위 층에서 (patches.h, pipeline).
#pragma once
#include <new>

#include "core/contact/px/gu.h"

namespace eng {
namespace contact {

namespace G = px::Gu;

// ---- 모양: 좁은 단계가 읽는 기하 (PxGeometryType 번호 그대로)
enum GeomType : int32_t { eSPHERE = 0, ePLANE = 1, eCAPSULE = 2, eBOX = 3, eCONVEXCORE = 4, eCONVEXMESH = 5, eTRIANGLEMESH = 8 };

struct ShapeGeom {
  int32_t type;
  px::PxSphereGeometry sphere;
  px::PxPlaneGeometry plane;
  px::PxCapsuleGeometry capsule;
  px::PxBoxGeometry box;
  px::PxConvexMeshGeometry convex;
  EHD const px::PxGeometry& get() const {
    switch (type) {
      case eSPHERE: return sphere;
      case ePLANE: return plane;
      case eCAPSULE: return capsule;
      case eBOX: return box;
      default: return convex;
    }
  }
};

// PxsContext.cpp:89 gEnablePCMCaching (구·평면·캡슐·상자·볼록코어·볼록메시 6x6 부분; 원본 주석의 "eCONVEX" 칸 = eCONVEXCORE)
EHD bool pcmCaching(int t0, int t1) {
  // 행 = t0, 열 = t1 (대칭)
  const uint64_t bits = 0ull
      | (1ull << (0 * 6 + 5))                                          // 구: 볼록메시
      | (1ull << (1 * 6 + 2)) | (1ull << (1 * 6 + 3)) | (1ull << (1 * 6 + 5))  // 평면: 캡슐·상자·볼록메시
      | (1ull << (2 * 6 + 1)) | (1ull << (2 * 6 + 3)) | (1ull << (2 * 6 + 5))  // 캡슐: 평면·상자·볼록메시
      | (1ull << (3 * 6 + 1)) | (1ull << (3 * 6 + 2)) | (1ull << (3 * 6 + 3)) | (1ull << (3 * 6 + 5))  // 상자
      | (1ull << (5 * 6 + 0)) | (1ull << (5 * 6 + 1)) | (1ull << (5 * 6 + 2)) | (1ull << (5 * 6 + 3)) | (1ull << (5 * 6 + 5));
  if (t0 > 5 || t1 > 5) return false;  // 삼각메시 등은 다중 다양체 (아직)
  return (bits >> (t0 * 6 + t1)) & 1ull;
}

// ---- 지속 다양체 칸 (쌍마다 하나). kind: 0 없음, 1 구(1점), 2 큰(4점)
struct alignas(16) ManifoldSlot {
  alignas(16) unsigned char storage[sizeof(G::LargePersistentContactManifold)];
  int32_t kind;
  EHD G::PersistentContactManifold& get() { return *reinterpret_cast<G::PersistentContactManifold*>(storage); }
};

// PxsContext::createCache 와 같게 (다양체를 새로 만들고 비운다)
EHD void initManifold(ManifoldSlot& m, int t0, int t1) {
  if (pcmCaching(t0, t1)) {
    if (t0 == eSPHERE || t1 == eSPHERE) {
      new (m.storage) G::SpherePersistentContactManifold();
      m.kind = 1;
    } else {
      new (m.storage) G::LargePersistentContactManifold();
      m.kind = 2;
    }
    m.get().clearManifold();
  } else {
    m.kind = 0;
  }
}

// t0 <= t1 인 PCM 접촉 함수 (GuContactMethodImpl.h / PxcContactMethodImpl.cpp g_PCMContactMethodTable)
EHD bool pcmMethod(int t0, int t1, const px::PxGeometry& a, const px::PxGeometry& b, const px::PxTransform32& ta, const px::PxTransform32& tb,
                   const G::NarrowPhaseParams& np, G::Cache& c, px::PxContactBuffer& buf) {
  switch (t0 * 16 + t1) {
    case eSPHERE * 16 + eSPHERE: return G::pcmContactSphereSphere(a, b, ta, tb, np, c, buf, nullptr);
    case eSPHERE * 16 + ePLANE: return G::pcmContactSpherePlane(a, b, ta, tb, np, c, buf, nullptr);
    case eSPHERE * 16 + eCAPSULE: return G::pcmContactSphereCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case eSPHERE * 16 + eBOX: return G::pcmContactSphereBox(a, b, ta, tb, np, c, buf, nullptr);
    case eSPHERE * 16 + eCONVEXMESH: return G::pcmContactSphereConvex(a, b, ta, tb, np, c, buf, nullptr);
    case ePLANE * 16 + eCAPSULE: return G::pcmContactPlaneCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case ePLANE * 16 + eBOX: return G::pcmContactPlaneBox(a, b, ta, tb, np, c, buf, nullptr);
    case ePLANE * 16 + eCONVEXMESH: return G::pcmContactPlaneConvex(a, b, ta, tb, np, c, buf, nullptr);
    case eCAPSULE * 16 + eCAPSULE: return G::pcmContactCapsuleCapsule(a, b, ta, tb, np, c, buf, nullptr);
    case eCAPSULE * 16 + eBOX: return G::pcmContactCapsuleBox(a, b, ta, tb, np, c, buf, nullptr);
    case eCAPSULE * 16 + eCONVEXMESH: return G::pcmContactCapsuleConvex(a, b, ta, tb, np, c, buf, nullptr);
    case eBOX * 16 + eBOX: return G::pcmContactBoxBox(a, b, ta, tb, np, c, buf, nullptr);
    case eBOX * 16 + eCONVEXMESH: return G::pcmContactBoxConvex(a, b, ta, tb, np, c, buf, nullptr);
    case eCONVEXMESH * 16 + eCONVEXMESH: return G::pcmContactConvexConvex(a, b, ta, tb, np, c, buf, nullptr);
    default: return false;  // 평면-평면 등 (PhysX 표에서도 빈 칸)
  }
}

// 한 쌍: 필요하면 뒤집어 부르고, 뒤집었으면 법선 부호를 바꾼다. 반환 = 접촉 함수 반환값. flipped 로 뒤집힘 알림(재질 번호 교환용).
EHD bool pcmPair(const ShapeGeom& s0, const ShapeGeom& s1, const px::PxTransform32& t0, const px::PxTransform32& t1, float contactDist,
                 float meshMargin, float toleranceLength, ManifoldSlot& man, px::PxContactBuffer& buf, bool& flipped) {
  int ty0 = s0.type, ty1 = s1.type;
  flipped = ty1 < ty0;
  const ShapeGeom* a = &s0;
  const ShapeGeom* b = &s1;
  const px::PxTransform32* ta = &t0;
  const px::PxTransform32* tb = &t1;
  if (flipped) {
    const int t = ty0; ty0 = ty1; ty1 = t;
    a = &s1; b = &s0; ta = &t1; tb = &t0;
  }
  const G::NarrowPhaseParams np(contactDist, meshMargin, toleranceLength);
  G::Cache cache;
  if (man.kind) cache.setManifold(man.storage);
  buf.reset();
  const bool r = pcmMethod(ty0, ty1, a->get(), b->get(), *ta, *tb, np, cache, buf);
  if (flipped)
    for (px::PxU32 i = 0; i < buf.count; ++i) buf.contacts[i].normal = -buf.contacts[i].normal;  // PxcNpBatch.cpp:71
  return r;
}

// ---- 시험·일괄 처리용 묶음 (판끼리 공유하는 정적 부분 / 판마다 자세 / 결과)
static constexpr int kMaxOutContacts = 16;

struct ConvexPair {  // (옛 이름 유지) 볼록-볼록 전용
  px::PxConvexMeshGeometry g0, g1;
};
struct ShapePair {
  ShapeGeom s0, s1;
};
struct PairPose {
  px::PxTransform32 tf0, tf1;
};
struct PairResult {
  uint32_t ret;       // 접촉 함수 반환값
  uint32_t count;     // 접촉 수 (PxContactBuffer::count)
  uint32_t overflow;  // count > kMaxOutContacts
  uint32_t flipped;
  px::PxContactPoint c[kMaxOutContacts];
};

EHD void copyOut(const px::PxContactBuffer& buf, PairResult& out) {
  out.count = buf.count;
  out.overflow = buf.count > uint32_t(kMaxOutContacts) ? 1u : 0u;
  const uint32_t n = buf.count < uint32_t(kMaxOutContacts) ? buf.count : uint32_t(kMaxOutContacts);
  for (uint32_t i = 0; i < n; ++i) out.c[i] = buf.contacts[i];
}

// 볼록-볼록 한 쌍 (다양체를 직접 받는 판)
EHD void convexConvexPair(const ConvexPair& s, const PairPose& p, const G::NarrowPhaseParams& np, G::LargePersistentContactManifold& man,
                          px::PxContactBuffer& buf, PairResult& out) {
  G::Cache cache;
  cache.setManifold(&man);
  buf.reset();
  out.ret = G::pcmContactConvexConvex(s.g0, s.g1, p.tf0, p.tf1, np, cache, buf, nullptr) ? 1u : 0u;
  out.flipped = 0;
  copyOut(buf, out);
}

// 아무 모양 쌍
EHD void shapePair(const ShapePair& s, const PairPose& p, float contactDist, float meshMargin, float toleranceLength, ManifoldSlot& man,
                   px::PxContactBuffer& buf, PairResult& out) {
  bool flipped;
  out.ret = pcmPair(s.s0, s.s1, p.tf0, p.tf1, contactDist, meshMargin, toleranceLength, man, buf, flipped) ? 1u : 0u;
  out.flipped = flipped ? 1u : 0u;
  copyOut(buf, out);
}

}  // namespace contact
}  // namespace eng
