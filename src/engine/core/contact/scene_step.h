// contact 장면 단위 한 서브스텝 (docs 19.2 순서를 함수 몇 개로 묶음, 09-30 engine-solver-art).
//   넓은 단계(AABB 관리자 세 단계) -> 쌍 관리(더러운 상호작용·새 겹침 거르기·관리자 만들기) -> [섬 1·2차: 호출자] ->
//   좁은 단계(기존 목록 다음 새 목록, 칸마다 PCM + 압축 스트림) -> 합치기·닿음 사건 -> solver 입력(12.3 형식)
//   풀이 뒤: 사라진 겹침·잃은 닿음·관리자 없애기 (호출자가 섬 3차를 사이에 넣는다).
// 호스트 층 1 (AABB 관리자 번역본이 호스트용). 좁은 단계 칸 계산(np_step.h)은 호스트·GPU 공용 그대로.
// 입력으로 받는 것(Sc 층 몫, 아직 이 함수 밖): 경계 상자 배열·접촉 거리·바뀜 비트맵(모양 = ElementSim 번호 칸), 변환 캐시(모양 세계 자세·얼림).
//   — PhysX 는 이것들을 지난 스텝 끝(afterIntegration)에 움직인 몸체의 모양만 다시 계산한다. 옮기기는 docs 19.6 남은 것.
// PhysX 원본: simulationcontroller/src/ScPipeline.cpp (broadPhase·finishBroadPhase·postBroadPhase·processLostContacts),
//   lowlevel/software/src/PxsNphaseImplementationContext.cpp (processCms·appendContactManagers), PxsContext.cpp:528 (fillManagerTouchEvents).
#pragma once
#include <cstddef>
#include <cstring>
#include <vector>

#include "core/contact/np_step.h"
#include "core/contact/px/aabb.h"
#include "core/contact/sc_pairs.h"
#include "core/solver/solver_io.h"

namespace eng {
namespace contact {

// 좁은 단계 칸 캐시 (지속 다양체 + 출력 스트림) — 쌍 관리층이 목록을 바꿀 때 같이 옮긴다
struct NpSlotData {
  ManifoldSlot man;
  NpSlotOutput<32, 256> out;
};
struct NpSlotCaches : public sc::CacheHooks {
  std::vector<NpSlotData> L[2];
  void create(bool nl, uint32_t slot, int32_t g0, int32_t g1) override {
    NpSlotData& d = L[nl][slot];
    d.out = NpSlotOutput<32, 256>();
    initManifold(d.man, g0 < g1 ? g0 : g1, g0 < g1 ? g1 : g0);
  }
  void move(bool dn, uint32_t dst, bool sn, uint32_t src) override {
    if (dn == sn && dst == src) return;
    copyManifoldSlot(L[dn][dst].man, L[sn][src].man);
    L[dn][dst].out = L[sn][src].out;
  }
  void destroy(bool, uint32_t) override {}
  void resize(bool nl, uint32_t n) override { L[nl].resize(n); }
};

// 장면 (호출자가 채우고 스텝마다 그대로 넘긴다)
struct ContactScene {
  sc::ScPairs pairs;              // 행위자·모양·거르개·섬 갈고리(pairs.islands)는 호출자가 채운다
  NpSlotCaches caches;
  px::Bp::AABBManager* aabb = nullptr;  // 모양 = 경계 번호 = ElementSim 번호, 사용자 자료 = userOfElem(번호) (아래 2 비트는 AABB 관리자가 부피 종류로 씀)
  px::Cm::FlushPool pool;
  px::PxcScratchAllocator scratch;
  // 좁은 단계 입력 (ElementSim 번호 = 변환 캐시 번호로 찾음)
  std::vector<NpShape> npShapes;
  std::vector<MaterialData> materials;
  NpParams npParams;
  // 이번 스텝 겹침 (AABB 관리자 출력을 ElementSim 번호 쌍으로; 사용자 자료 순서 그대로)
  std::vector<int32_t> createdShape, createdTrigger, destroyedShape, destroyedTrigger;
  px::PxContactBuffer buf;
  ContactScene() { pairs.caches = &caches; }
};

// 사용자 자료 <-> 요소 번호. AABB 관리자 VolumeData 가 사용자 자료의 아래 2 비트에 부피 종류를 넣으므로(BpAABBManager.h VolumeData::setVolumeType)
// 4 의 배수여야 한다 (09-30 리드: 번호+1 로 두면 이웃 번호가 같은 값으로 뭉개져 겹침 목록이 틀어졌다).
inline void* userOfElem(uint32_t e) { return reinterpret_cast<void*>((uintptr_t(e) + 1) << 2); }
inline int32_t elemOfUser(void* ud) { return int32_t(reinterpret_cast<uintptr_t>(ud) >> 2) - 1; }

// ---- 1. 넓은 단계 (Sc::Scene::broadPhase -> postBroadPhase): 입력 배열은 AABB 관리자가 가리키는 BoundsArray·접촉 거리·바뀜 비트맵에 호출자가 넣어 둔다.
//      hasContactDistanceChanged = Sc 의 접촉 거리 바뀜 표시. 결과는 S.created*/destroyed* 에 (다음 contactBroadPhaseEnd 까지 유효).
inline void contactBroadPhase(ContactScene& S, bool hasContactDistanceChanged) {
  px::Bp::AABBManager& m = *S.aabb;
  {
    px::EndTask t;
    t.setContinuation(nullptr);
    m.updateBPFirstPass(1, S.pool, hasContactDistanceChanged, &t);
    t.removeReference();
  }
  {
    px::EndTask t;
    t.setContinuation(nullptr);
    m.updateBPSecondPass(&S.scratch, &t);
    t.removeReference();
  }
  {
    px::EndTask t;
    t.setContinuation(nullptr);
    m.postBroadPhase(&t, S.pool);
    t.removeReference();
  }
  auto grab = [&](bool created, px::Bp::ElementType::Enum type, std::vector<int32_t>& out) {
    out.clear();
    px::PxU32 n = 0;
    const px::Bp::AABBOverlap* o = created ? m.getCreatedOverlaps(type, n) : m.getDestroyedOverlaps(type, n);
    for (px::PxU32 i = 0; i < n; ++i) {
      out.push_back(elemOfUser(o[i].mUserData0));
      out.push_back(elemOfUser(o[i].mUserData1));
    }
  };
  grab(true, px::Bp::ElementType::eSHAPE, S.createdShape);
  grab(true, px::Bp::ElementType::eTRIGGER, S.createdTrigger);
  grab(false, px::Bp::ElementType::eSHAPE, S.destroyedShape);
  grab(false, px::Bp::ElementType::eTRIGGER, S.destroyedTrigger);
}

// ---- 2. 쌍 관리 (ScPipeline.cpp:477 finishBroadPhase): 지난 스텝·API 가 표시한 재거르기 먼저(19.2-1), 그다음 새 겹침
inline void contactPairs(ContactScene& S) {
  S.pairs.updateDirtyInteractions();
  S.pairs.finishBroadPhase(S.createdTrigger.data(), uint32_t(S.createdTrigger.size() / 2), S.createdShape.data(), uint32_t(S.createdShape.size() / 2));
}

// ---- 3. [호출자: 섬 1·2차 + 추측 섬에서 활성화된 접촉 간선 순서대로 pairs.activateInteraction]

// ---- 4. 좁은 단계 + 합치기 + 닿음 사건 (19.2-5,6). tc = 변환 캐시(ElementSim 번호), contactDist = 모양별 contactOffset.
//      FTZ·DAZ 는 호출자가 (PhysX PxSIMDGuard 구간).
inline void contactNarrowPhase(ContactScene& S, const CachedTransform* tc, const float* contactDist) {
  sc::ScPairs& M = S.pairs;
  M.beginNarrowPhase();
  for (int nl = 0; nl < 2; ++nl) {  // 기존 목록(1차) 다음 새 목록(2차)
    sc::NpList& L = nl ? M.npNew : M.npMain;
    for (uint32_t slot = 0; slot < L.size(); ++slot) {
      NpSlotData& sd = S.caches.L[nl][slot];
      sd.out.statusFlag = L.outputs[slot].statusFlag;
      const sc::ContactManager& cm = M.cmsData[size_t(L.cms[slot])];
      const NpWorkUnit wu{cm.wuFlags, cm.shape0, cm.shape1};
      discreteNarrowPhasePCM(wu, S.npShapes.data(), tc, contactDist, S.materials.data(), S.npParams, sd.man, S.buf, sd.out);
      M.narrowPhaseResult(nl != 0, slot, sd.out.statusFlag, sd.out.nbPatches);
    }
  }
  M.mergeNarrowPhase();
  M.fillTouchEvents();
  M.processNewTouches();
  M.setEdgesConnected();
}

// ---- 5. solver 입력 (12.3 형식). 행위자 -> 몸체 참조는 호출자 표:
//      actorBody[a] = Body 번호(강체) / 관절체 번호(링크) / NONE(정적), actorLinkP1[a] = 0(강체·정적) 또는 LL 링크 + 1, actorStaticPose[a] = 정적 행위자 자세.
//      cms 는 관리자 풀 번호 칸 배열(섬의 접촉 간선이 이 번호로 가리킨다, 마찰 칸은 solver 가 들고 있으므로 건드리지 않는다).
//      fresh = 이번 스텝 새로 등록(새 관리자·캐시 지움 뒤 재등록)된 관리자 번호 — solver 입력 resetCMs 로 넘기면 마찰 패치 수 0 (17.2 규칙 1·3).
struct SolverInputOut {
  std::vector<sv::ContactPatchIn> patches;
  std::vector<sv::ContactIn> contacts;
};
static_assert(sizeof(sv::ContactPatchIn) <= sizeof(ContactPatch) && offsetof(sv::ContactPatchIn, materialIndex1) == offsetof(ContactPatch, materialIndex1),
              "PxContactPatch 배치 (solver 쪽은 끝 채움 없음 — 앞부분이 같다)");
static_assert(sizeof(sv::ContactIn) == sizeof(Contact), "PxContact 배치");

inline void contactSolverInput(const ContactScene& S, const uint32_t* actorBody, const uint32_t* actorLinkP1, const px::PxTransform* actorStaticPose,
                               sv::SolverCM* cms, uint32_t cmsCap, SolverInputOut& out) {
  const sc::ScPairs& M = S.pairs;
  out.patches.clear();
  out.contacts.clear();
  for (uint32_t slot = 0; slot < M.npMain.size(); ++slot) {
    const int32_t ci = M.npMain.cms[slot];
    if (ci < 0 || uint32_t(ci) >= cmsCap) continue;
    const sc::ContactManager& cm = M.cmsData[size_t(ci)];
    const NpSlotData& sd = S.caches.L[0][slot];
    sv::SolverCM& m = cms[ci];
    const int32_t a0 = M.shapes[size_t(cm.shape0)].actor, a1 = M.shapes[size_t(cm.shape1)].actor;
    m.body0 = actorBody[a0];
    m.artLink0 = actorLinkP1[a0];
    m.body1 = actorBody[a1];
    m.artLink1 = actorLinkP1[a1];
    if (m.body1 == sv::NONE) {
      const px::PxTransform& t = actorStaticPose[a1];
      m.staticPose1 = Tf{Q{t.q.x, t.q.y, t.q.z, t.q.w}, V3{t.p.x, t.p.y, t.p.z}};
    }
    m.npFlags = cm.wuFlags;
    m.restDistance = cm.restDistance;
    m.torsionalPatchRadius = cm.torsionalPatchRadius;
    m.minTorsionalPatchRadius = cm.minTorsionalPatchRadius;
    m.offsetSlop = cm.offsetSlop;
    m.patchStart = uint32_t(out.patches.size());
    m.contactStart = uint32_t(out.contacts.size());
    const bool touch = (sd.out.statusFlag & NpStatus::eHAS_TOUCH) != 0;
    m.nbPatches = touch ? sd.out.nbPatches : 0u;
    m.nbContacts = touch ? sd.out.nbContacts : 0u;
    for (uint32_t k = 0; k < m.nbPatches; ++k) {
      sv::ContactPatchIn p;
      memcpy(&p, &sd.out.stream.patches[k], sizeof(p));
      out.patches.push_back(p);
    }
    for (uint32_t k = 0; k < m.nbContacts; ++k) {
      sv::ContactIn c;
      memcpy(&c, &sd.out.stream.contacts[k], sizeof(c));
      out.contacts.push_back(c);
    }
  }
}

// ---- 6. 풀이 뒤 (19.2-7): 사라진 겹침 -> 잃은 닿음 -> 등록 풀기. afterIsland3 = 섬 3차(호출자) 자리.
template <class Island3>
inline void contactPostSolve(ContactScene& S, const Island3& afterIsland3) {
  sc::ScPairs& M = S.pairs;
  M.processLostContacts(S.destroyedShape.data(), uint32_t(S.destroyedShape.size() / 2), S.destroyedTrigger.data(),
                        uint32_t(S.destroyedTrigger.size() / 2));
  M.processNarrowPhaseLostTouchEventsIslands();
  M.processNarrowPhaseLostTouchEvents();
  M.processLostContacts2();
  M.lostTouchReports();
  M.unregisterInteractions();
  afterIsland3();
  M.destroyManagers();
  M.processLostContacts3();
}
// 스텝 끝: AABB 관리자 작업 버퍼 비우기 (다음 넓은 단계 전에)
inline void contactBroadPhaseEnd(ContactScene& S) {
  S.aabb->freeBuffers();
  S.pool.clear();
}

}  // namespace contact
}  // namespace eng
