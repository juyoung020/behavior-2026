// joints 층 2: 장면 광선(raycastClosest)을 CUDA 로 — 광선 하나 = 스레드 하나. 판 N 개가 각자 광선을 쏘는 보조 잡기 판정에 쓴다.
// 모양 배열(정적은 판끼리 공유, 동적 자세만 판마다)은 전역 메모리. 코드는 core/joints/scene_query.h 그대로라 층 1 과 비트가 같아야 한다.
#pragma once
#include <cstdint>

#include "core/joints/scene_query.h"

namespace eng {
namespace jcuda {

struct SqRayIn {
  V3 origin, dir;
  float maxDist;
  uint32_t env;  // 판 번호: shapes + env * shapesPerEnv
};
struct SqRayOut {
  int shape;
  sq::RayHit hit;
};

__global__ void kRaycastClosest(const sq::SqShape* shapes, uint32_t shapesPerEnv, const SqRayIn* rays, uint32_t nRays, uint32_t hitFlags,
                                SqRayOut* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRays) return;
  const SqRayIn r = rays[i];
  SqRayOut o;
  o.hit = sq::RayHit{};
  o.shape = sq::raycastClosest(shapes + size_t(r.env) * shapesPerEnv, shapesPerEnv, r.origin, r.dir, r.maxDist, hitFlags, o.hit);
  out[i] = o;
}

}  // namespace jcuda
}  // namespace eng
