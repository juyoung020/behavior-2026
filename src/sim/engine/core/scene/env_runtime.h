// 판 하나의 contact 층 런타임 넘겨받기 (문서 15.3 v1·G2 준비, 리드): 장면 파일 하나로 판의 호스트 모듈 상태를 세운다.
//   넓은 단계: 기록(BpLog)을 처음부터 다시 넣음 — 사용자 자료 = 요소 번호 규약(scene_step.h userOfElem)
//   쌍 관리층: 기록(PairsLog)을 처음부터 다시 넣음 — 거르개 = core/scene/omni_filter.h (판의 표)
//   좁은 단계 칸 캐시: 관리자 모양 쌍(요소 번호)으로 파일의 지속 다양체를 우리 목록 칸에 넣음
//   좁은 단계 입력: 모양 기하(틀 = SceneShared, 볼록 덩어리 주소가 걸린 것)·재질·길이 눈금
// 섬 관리자는 IslandStore(island_state.h)가 따로 넘겨받는다. 입력이 같으면 상태가 같다는 것은 G1 그림자(G1_SCENE_FROM)로 보인다.
// 호스트 전용 (넓은 단계 AABB 관리자 번역본이 호스트용). 판 N 개면 판마다 하나씩.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/contact/scene_step.h"
#include "core/scene/batch.h"
#include "core/scene/bp_log.h"
#include "core/scene/omni_filter.h"
#include "core/scene/pairs_log.h"

namespace eng {
namespace scene {

struct EnvContact {
  std::unique_ptr<BpRuntime> bp;
  std::unique_ptr<contact::ContactScene> S;
  PairsHooks H;
  OmniFilterSpec spec;
  OmniFilterCtx fctx;
  uint64_t sim = 0;  // 넘겨받은 경계 (이 simulate 바로 앞)
  uint32_t actBad = 0, manifoldsIn = 0, manifoldsMiss = 0;

  // f: 장면 파일, sh: 그 파일의 틀(볼록 덩어리 주소가 걸린 것 — Batch::shared(e) 또는 makeShared(f))
  bool load(const SceneFile& f, const SceneShared& sh, std::string* err = nullptr) {
    auto fail = [&](const char* w) { if (err) *err = w; return false; };
    if (!f.bp.valid || !f.pairs.valid) return fail("넓은 단계·쌍 관리층 기록이 없음");
    if (f.shapeElems.size() != f.shapes.size()) return fail("모양별 요소 번호가 없음 (파일 v8 이전)");
    sim = f.h.sim;
    spec = f.omniFilter;
    S.reset(new contact::ContactScene);
    contact::ContactScene& C = *S;
    // 넓은 단계
    bp.reset(new BpRuntime);
    if (!bp->replay(f.bp, true)) return fail("넓은 단계 기록 다시 넣기 실패");
    C.aabb = bp->m.get();
    // 쌍 관리층 (거르개 먼저 — 다시 넣기가 새 겹침을 거른다)
    fctx.spec = &spec;
    setOmniFilter(C.pairs, fctx);
    pairsReplay(C.pairs, H, f.pairs, &actBad);
    // 좁은 단계 입력 (요소 번호별)
    uint32_t maxElem = 0;
    for (uint32_t e : f.shapeElems)
      if (e != kNone && e + 1 > maxElem) maxElem = e + 1;
    C.npShapes.resize(maxElem);
    for (size_t k = 0; k < sh.shapes.size(); ++k) {
      const uint32_t e = f.shapeElems[k];
      if (e == kNone) continue;
      C.npShapes[e].geom = sh.shapes[k].geom;
      C.npShapes[e].material = sh.shapes[k].material;
    }
    C.materials = sh.materials;
    C.npParams.toleranceLength = f.h.lengthScale;
    C.npParams.meshContactMargin = 0.01f * f.h.lengthScale;
    C.npParams.createAveragePoint = (f.h.sceneFlags & (1u << 11)) != 0;  // PxSceneFlag::eENABLE_AVERAGE_POINT (PxSceneDesc.h:217)
    // 지속 다양체: 파일 관리자 (모양0, 모양1) -> 요소 쌍 -> 우리 목록 칸
    std::map<std::pair<int32_t, int32_t>, uint32_t> fileCM;
    for (uint32_t k = 0; k < f.cms.size(); ++k) {
      const SceneCM& c = f.cms[k];
      fileCM[{int32_t(f.shapeElems[c.shape0]), int32_t(f.shapeElems[c.shape1])}] = k;
    }
    for (int nl = 0; nl < 2; ++nl) {
      const contact::sc::NpList& L = nl ? C.pairs.npNew : C.pairs.npMain;
      for (uint32_t slot = 0; slot < L.size(); ++slot) {
        const contact::sc::ContactManager& cm = C.pairs.cmsData[size_t(L.cms[slot])];
        auto it = fileCM.find({cm.shape0, cm.shape1});
        if (it == fileCM.end()) { ++manifoldsMiss; continue; }
        const SceneCM& c = f.cms[it->second];
        if (c.manifold == kNone) continue;
        C.caches.L[nl][slot].man = f.manifolds[c.manifold];  // 대입이 자기 버퍼 포인터를 다시 건다
        C.caches.L[nl][slot].out.statusFlag = L.outputs[slot].statusFlag;
        ++manifoldsIn;
      }
    }
    return true;
  }
};

}  // namespace scene
}  // namespace eng
