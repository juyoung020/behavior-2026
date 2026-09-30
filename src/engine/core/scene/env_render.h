// G3: env 물리 상태 -> 렌더 기준 prim 행렬 (RenderBatch 입력, 호스트).
// 기준 prim(렌더 장면의 anchor) 은 USD 강체 prim 경로다(tests/render/capture render_capture.py: RigidBodyAPI prim). omni.physx 는 PhysX 행위자
// 이름을 prim 경로로 두므로 장면 파일 행위자 이름과 경로가 같으면 그 행위자다. 조명 기준 prim(메시 기준 prim 뒤)은 고정.
//   기준 prim 행렬 = aff_from_pose(행위자 전역 자세, 축척)  — 행위자 전역 자세 = PxRigidActor::getGlobalPose = body2World * body2Actor^-1
//   (NpRigidBodyTemplate getGlobalPose -> Sc::BodyCore::getBody2World() * getBody2Actor().getInverse(), 같은 식 pmath Tf)
//   축척 = 렌더 장면 첫 프레임 기준 prim 행렬의 열 길이 (강체 prim 축척은 판 도중 안 바뀜).
//   정적 행위자 = 장면 파일 staticPose, 짝이 없는 기준 prim(조명 등) = 첫 프레임 행렬 그대로.
// 통합 약속(14.6): 스텝 k 관측은 스텝 k-1 물리 결과로 그린다 -> 부르는 쪽이 스텝 뒤 상태로 이 행렬을 만들어 다음 스텝 관측에 쓴다.
#pragma once
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/pmath.h"
#include "core/render/rig.h"
#include "core/scene/env_solve.h"
#include "core/scene/scene_file.h"

namespace eng {
namespace scene {

struct EnvRenderAnchor {
  enum Kind : uint8_t { kFixed = 0, kBody = 1, kLink = 2, kStaticActor = 3 };
  uint8_t kind = kFixed;
  uint32_t body = 0, link = 0;  // kBody: 몸체 번호 / kLink: 관절체 번호 + 링크 생성 번호
  float scale[3] = {1, 1, 1};
  rnd::Aff fixed{};
};

struct EnvRenderMap {
  std::vector<EnvRenderAnchor> a;
  uint32_t n[4] = {0, 0, 0, 0};  // 종류별 수

  // paths: 메시 기준 prim 경로 (렌더 장면 차례), frame0: 렌더 장면 첫 프레임 기준 prim 행렬 (메시 + 조명, n_anchor 개)
  bool build(const SceneFile& f, const std::vector<std::string>& paths, const std::vector<rnd::Aff>& frame0, std::string* err) {
    if (frame0.size() < paths.size()) {
      if (err) *err = "첫 프레임 기준 prim 수가 경로 수보다 적음";
      return false;
    }
    std::unordered_map<std::string, uint32_t> actorOf;
    for (uint32_t i = 0; i < f.actors.size(); ++i) {
      const SceneActor& s = f.actors[i];
      if (s.name < f.names.size()) actorOf[std::string(f.names.data() + s.name)] = i;
    }
    a.assign(frame0.size(), EnvRenderAnchor{});
    for (size_t k = 0; k < frame0.size(); ++k) {
      EnvRenderAnchor& x = a[k];
      x.fixed = frame0[k];
      for (int c = 0; c < 3; ++c) {  // 열 c 길이 (행렬 행 우선 3x4: m[r*4+c])
        const float v0 = frame0[k].m[c], v1 = frame0[k].m[4 + c], v2 = frame0[k].m[8 + c];
        x.scale[c] = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
      }
      if (k >= paths.size()) continue;  // 조명
      auto it = actorOf.find(paths[k]);
      if (it == actorOf.end()) continue;
      const SceneActor& s = f.actors[it->second];
      if (s.kind == kDynamic) {
        x.kind = EnvRenderAnchor::kBody;
        x.body = s.body;
      } else if (s.kind == kLink) {
        x.kind = EnvRenderAnchor::kLink;
        x.body = s.body;
        x.link = s.link;
      } else {
        x.kind = EnvRenderAnchor::kStaticActor;
        const Tf& t = s.staticPose;
        const float q[4] = {t.q.x, t.q.y, t.q.z, t.q.w}, p[3] = {t.p.x, t.p.y, t.p.z};
        x.fixed = rnd::aff_from_pose(q, p, x.scale);
      }
    }
    for (int k = 0; k < 4; ++k) n[k] = 0;
    for (const EnvRenderAnchor& x : a) ++n[x.kind];
    return true;
  }

  // 판 하나의 기준 prim 행렬 (out: a.size() 개)
  void anchors(const EnvSolveImpl& S, rnd::Aff* out) const {
    for (size_t k = 0; k < a.size(); ++k) {
      const EnvRenderAnchor& x = a[k];
      Tf g;
      if (x.kind == EnvRenderAnchor::kBody && x.body < S.bodies.size()) {
        const Body& b = S.bodies[x.body];
        g = b.body2World * inverse(b.body2Actor);
      } else if (x.kind == EnvRenderAnchor::kLink && x.body < S.arts.size()) {
        const art::Articulation& A = S.arts[x.body];
        const uint32_t ll = art::slot(A, x.link);
        if (ll >= A.nLinks) {
          out[k] = x.fixed;
          continue;
        }
        const art::LinkBody& b = A.bodies[ll];
        g = b.body2World * inverse(b.body2Actor);
      } else {
        out[k] = x.fixed;
        continue;
      }
      const float q[4] = {g.q.x, g.q.y, g.q.z, g.q.w}, p[3] = {g.p.x, g.p.y, g.p.z};
      out[k] = rnd::aff_from_pose(q, p, x.scale);
    }
  }
};

}  // namespace scene
}  // namespace eng
