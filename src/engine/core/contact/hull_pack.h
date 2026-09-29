// 구운 볼록 데이터(Gu::ConvexHullData + 다각형·꼭짓점 덩어리 + 큰 볼록의 가우스 지도·이웃표)를 한 덩어리로 포장한다.
// 포인터는 "덩어리가 놓일 주소(base)" 기준으로 다시 건다 -> GPU 에 그대로 올리면 장치에서 바로 쓴다 (장면 적재 단계, 호스트).
// 배치는 PhysX 가 구운 바이트 그대로다 (GuConvexMeshData.h:57-160 의 get* 오프셋 규칙, GuBigConvexData.h:53).
#pragma once
#include <cstdint>
#include <cstring>

#include "core/contact/px/gu.h"

namespace eng {
namespace contact {

inline size_t align16(size_t x) { return (x + 15) & ~size_t(15); }

// 다각형 배열에서 꼭짓점 번호표(vertexData8) 끝까지의 바이트 수
inline size_t hullBlobBytes(const px::Gu::ConvexHullData& h) {
  const uint8_t* start = reinterpret_cast<const uint8_t*>(h.mPolygons);
  const uint8_t* vd8 = h.getVertexData8();
  size_t endRef = 0;
  for (uint32_t i = 0; i < h.mNbPolygons; ++i) {
    const size_t e = size_t(h.mPolygons[i].mVRef8) + h.mPolygons[i].mNbVerts;
    if (e > endRef) endRef = e;
  }
  return size_t(vd8 - start) + endRef;
}

inline size_t hullPackedBytes(const px::Gu::ConvexHullData& h) {
  size_t n = align16(sizeof(px::Gu::ConvexHullData)) + align16(hullBlobBytes(h) + 16);
  if (const px::Gu::BigConvexRawData* b = h.mBigConvexRawData) {
    n += align16(sizeof(px::Gu::BigConvexRawData));
    n += align16(size_t(b->mNbSamples) * 2);
    n += align16(size_t(b->mNbVerts) * sizeof(px::Gu::Valency));
    n += align16(size_t(b->mNbAdjVerts));
  }
  return n;
}

// dst(호스트 메모리)에 포장. base = 이 덩어리가 최종적으로 놓일 주소(호스트면 dst 자체, GPU 면 장치 주소).
// 반환: 쓴 바이트 수. 덩어리 첫머리가 ConvexHullData 이다.
inline size_t packHull(const px::Gu::ConvexHullData& h, uint8_t* dst, uintptr_t base) {
  size_t off = 0;
  px::Gu::ConvexHullData* ch = reinterpret_cast<px::Gu::ConvexHullData*>(dst);
  std::memcpy(ch, &h, sizeof(h));
  off = align16(sizeof(h));
  const size_t blob = hullBlobBytes(h);
  std::memset(dst + off, 0, align16(blob + 16));
  std::memcpy(dst + off, h.mPolygons, blob);
  ch->mPolygons = reinterpret_cast<px::Gu::HullPolygonData*>(base + off);
  off += align16(blob + 16);
  ch->mSdfData = nullptr;
  if (const px::Gu::BigConvexRawData* b = h.mBigConvexRawData) {
    px::Gu::BigConvexRawData* bb = reinterpret_cast<px::Gu::BigConvexRawData*>(dst + off);
    std::memcpy(bb, b, sizeof(*b));
    ch->mBigConvexRawData = reinterpret_cast<px::Gu::BigConvexRawData*>(base + off);
    off += align16(sizeof(*b));
    std::memcpy(dst + off, b->mSamples, size_t(b->mNbSamples) * 2);
    bb->mSamples = reinterpret_cast<px::PxU8*>(base + off);
    off += align16(size_t(b->mNbSamples) * 2);
    std::memcpy(dst + off, b->mValencies, size_t(b->mNbVerts) * sizeof(px::Gu::Valency));
    bb->mValencies = reinterpret_cast<px::Gu::Valency*>(base + off);
    off += align16(size_t(b->mNbVerts) * sizeof(px::Gu::Valency));
    std::memcpy(dst + off, b->mAdjacentVerts, size_t(b->mNbAdjVerts));
    bb->mAdjacentVerts = reinterpret_cast<px::PxU8*>(base + off);
    off += align16(size_t(b->mNbAdjVerts));
  }
  return off;
}

}  // namespace contact
}  // namespace eng
