// 장면 파일 하나 -> env 한 스텝 함수 상태 (문서 15.3 닫힌 고리 4단 E1, 리드): PhysX 없이 판을 세운다.
//   넓은 단계·쌍 관리층·좁은 단계 = EnvContact::load (기록 다시 넣기 + 지속 다양체)
//   섬 관리 = IslandStore::load (파일 섬 상태)
//   Sc 입력 조각 = scLoad (파일 선택 절 SCSTATE1, 모양 기하는 틀에서)
//   깸 카운터 표 = 파일 몸체·관절체 값, 노드 번호는 섬 노드의 객체 번호(장면 뜨기 규약: 강체 = 장면 행위자 번호, 관절체 = 0x80000000|관절체 번호)
//   행위자 활성 표 = 정확 섬 노드 활성 (스텝 경계에서 ActorSim::isActive 와 같음 — 순서기 그림자로 확인)
// 풀이 자리(EnvSolve)는 부르는 쪽이 붙인다.
#pragma once
#include <algorithm>
#include <memory>
#include <string>

#include "core/scene/batch.h"
#include "core/scene/env_step.h"
#include "core/scene/sc_state_io.h"

namespace eng {
namespace scene {

struct EnvOwned {
  EnvContact C;
  IslandStore isl;
  ScScene sc;
  EnvStep E;
};

inline bool envLoad(EnvOwned& o, const SceneFile& f, const SceneShared& sh, std::string* err = nullptr) {
  auto fail = [&](const char* w) { if (err) *err = w; return false; };
  if (!f.islands.valid) return fail("섬 상태가 없음");
  if (!f.sc.valid) return fail("Sc 상태 절(SCSTATE1)이 없음 — 이 기능 전에 뜬 파일");
  if (!o.C.load(f, sh, err)) return false;
  if (!o.isl.load(f.islands)) return fail("섬 상태 적재 실패(용량)");
  const uint32_t miss = scLoad(o.sc, f.sc, [&](uint32_t k) -> const contact::ShapeGeom* { return k < sh.shapes.size() ? &sh.shapes[k].geom : nullptr; });
  if (miss) return fail("Sc 모양 기하를 틀에서 못 찾음");
  EnvStep& E = o.E;
  E.C = &o.C;
  E.isl = &o.isl;
  E.sc = &o.sc;
  // 깸 카운터 표
  E.wake = HostWake{};
  std::vector<uint8_t> kinOfNode;
  for (const ScActorRec& a : o.sc.actors)
    if (a.alive && a.kind != 0) {
      const uint32_t id = uint32_t(a.node & 0xffffffffu);
      if (kinOfNode.size() <= id) kinOfNode.resize(size_t(id) + 1, 0);
      kinOfNode[id] = a.kinematic;
    }
  const IslandSimState& acc = f.islands.accurate;
  for (uint32_t id = 0; id < acc.nodes.size(); ++id) {
    const ig::Node& n = acc.nodes[id];
    if (n.flags & ig::N_DELETED) continue;
    if (n.type == ig::eRIGID_BODY_TYPE) {
      if (n.object >= f.actors.size() || f.actors[n.object].kind != kDynamic) continue;
      const Body& b = f.bodies[f.actors[n.object].body];
      HostBodyWake w;
      w.node = id;
      w.wc = b.wakeCounter;
      w.solverWc = b.solverWakeCounter;
      w.kinematic = id < kinOfNode.size() ? kinOfNode[id] : 0;
      E.wake.bodies.push_back(w);
    } else if (n.type == ig::eARTICULATION_TYPE && (n.object & 0x80000000u)) {
      const uint32_t k = n.object & 0x7fffffffu;
      if (k >= f.arts.size()) continue;
      const art::Articulation& A = f.arts[k];
      HostArtWake aw;
      aw.node = id;
      aw.wc = A.wakeCounter;
      for (uint32_t l = 0; l < A.nLinks; ++l) {
        const uint64_t raw = uint64_t(id) | (uint64_t((l << 1) | 1u) << 32);  // PxNodeIndex(관절체 노드, 링크 LL 번호)
        aw.links.push_back(raw);
        HostBodyWake w;
        w.node = raw;
        w.wc = A.bodies[l].wakeCounter;
        w.solverWc = A.bodies[l].wakeCounter;
        w.link = 1;
        E.wake.bodies.push_back(w);
      }
      E.wake.arts.push_back(aw);
    }
  }
  std::sort(E.wake.bodies.begin(), E.wake.bodies.end(), [](const HostBodyWake& x, const HostBodyWake& y) { return x.node < y.node; });
  // 행위자 활성 표 (쌍 관리층 행위자 번호)
  const ss::ScPairs& P = o.C.S->pairs;
  E.active.assign(P.actors.size(), 0);
  for (uint32_t a = 0; a < P.actors.size(); ++a) {
    if (P.actors[a].isStatic()) continue;
    const uint32_t id = uint32_t(P.actors[a].nodeIndex & 0xffffffffu);
    E.active[a] = id < o.isl.M.accurate.nodes.size && (o.isl.M.accurate.nodes.d[id].flags & ig::N_ACTIVE) ? 1 : 0;
  }
  E.bind();
  return true;
}

}  // namespace scene
}  // namespace eng
