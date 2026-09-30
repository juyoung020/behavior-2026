// G1 Sc 입력 그림자의 계산 쪽 (PhysX 헤더 없는 번역 단위 — core/common/aos.h 와 PhysX 의 SSE 정의가 부딪혀서 나눔).
#include "core/contact/narrowphase.h"
#include "core/scene/sc_inputs.h"

// in: shape2Actor[q4 p3], flags[isStatic, idtShape, idtBody2Actor], a2w[q4 p3], b2a[q4 p3] -> pose[q4 p3], bounds[min3 max3]
void g1sc_update(const float* s2a, const unsigned char* flags, const float* a2w, const float* b2a, const eng::contact::ShapeGeom& g, float* pose, float* bounds) {
  namespace ep = eng::px;
  eng::scene::ShapePoseIn in{};
  in.shape2Actor = ep::PxTransform(ep::PxVec3(s2a[4], s2a[5], s2a[6]), ep::PxQuat(s2a[0], s2a[1], s2a[2], s2a[3]));
  in.isStatic = flags[0];
  in.idtShape = flags[1];
  in.idtBody2Actor = flags[2];
  const ep::PxTransform A(ep::PxVec3(a2w[4], a2w[5], a2w[6]), ep::PxQuat(a2w[0], a2w[1], a2w[2], a2w[3]));
  const ep::PxTransform B(ep::PxVec3(b2a[4], b2a[5], b2a[6]), ep::PxQuat(b2a[0], b2a[1], b2a[2], b2a[3]));
  ep::PxTransform out;
  ep::PxBounds3 bb;
  eng::scene::updateShapeCached(in, g.get(), A, B, out, bb);
  const float o[13] = {out.q.x, out.q.y, out.q.z, out.q.w, out.p.x, out.p.y, out.p.z, bb.minimum.x, bb.minimum.y, bb.minimum.z, bb.maximum.x, bb.maximum.y, bb.maximum.z};
  for (int k = 0; k < 7; ++k) pose[k] = o[k];
  for (int k = 0; k < 6; ++k) bounds[k] = o[7 + k];
}
