// G1 닫힌 고리 (2a): 스텝 사이 API 창 (문서 15.3·G1 설계, 리드, 비계 전용).
// 창 = 한 simulate 의 fetchResults 뒤부터 다음 simulate 앞까지. 공식 평가기(omni·OmniGibson)가 PhysX 에 쓰는 값은 모두 이 창에서 들어간다.
// 창 시작(S0)과 끝(S1)의 PhysX 공개 상태를 같은 목록으로 떠서, 바뀐 칸만 "창 편집"(WindowEdit)으로 낸다.
//   -> 닫힌 고리는 우리 상태에 창 편집만 넣는다(PhysX 가 계산한 값은 받지 않음). 창 시작에서 우리 상태 = S0 인지 비교해 고리를 확인한다.
// 공개 상태 목록 (행위자·관절체·조인트·장면): 쓸 수 있는 칸만 — 아래 표. 행위자 추가·삭제는 sc_scene.h 편집 API 몫이라 여기서는 개수만 센다.
// 켜기: G1_LOOP=1 (G1_LOOP_SHOW=n: 처음 n 개 편집을 찍음). 결과: 칸 종류별 바뀐 수·창 수.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "PxPhysicsAPI.h"
#include "g1_hooks.h"

using namespace physx;

namespace {

// 칸 종류 (공개 API 이름). 값은 32 비트 낱말 묶음으로 비교(비트).
enum Kind : uint16_t {
  // 강체 동체 (운동학 포함)
  kRdPose, kRdLinVel, kRdAngVel, kRdWake, kRdSleeping, kRdFlags, kRdKinTarget, kRdMass, kRdCMass, kRdInertia, kRdLinDamp, kRdAngDamp, kRdMaxDepen,
  kRdSleepThr, kRdStabThr, kRdLockFlags, kRdIters, kRdActorFlags, kRdMaxLinVel, kRdMaxAngVel, kRdContactSlop,
  // 정적
  kRsPose,
  // 관절체
  kArRootPose, kArRootLinVel, kArRootAngVel, kArJointPos, kArJointVel, kArDriveTarget, kArDriveVel, kArDrive, kArWake, kArSleeping, kArFlags, kArIters,
  kArLinkFlags, kArLinkMass, kArLinkCMass, kArLinkInertia, kArLinkDamp, kArJointFriction, kArJointMaxVel, kArJointArmature, kArJointLimit, kArJointMotion,
  // 조인트 (D6)
  kJLocal0, kJLocal1, kJDrivePos, kJDriveVel, kJDrive, kJMotion, kJFlags, kJBreak,
  // 장면
  kScGravity,
  kCount
};
const char* kName[kCount] = {"동체 자세", "동체 선속도", "동체 각속도", "동체 깸 카운터", "동체 잠", "동체 플래그", "동체 운동학 목표", "동체 질량", "동체 질량 중심 자세",
                             "동체 관성", "동체 선감쇠", "동체 각감쇠", "동체 최대 빠져나옴 속도", "동체 잠 문턱", "동체 안정 문턱", "동체 잠금 플래그", "동체 반복 수",
                             "동체 행위자 플래그", "동체 최대 선속도", "동체 최대 각속도", "동체 접촉 느슨함", "정적 자세", "관절체 뿌리 자세", "관절체 뿌리 선속도",
                             "관절체 뿌리 각속도", "관절 위치", "관절 속도", "관절 드라이브 목표", "관절 드라이브 목표 속도", "관절 드라이브 값", "관절체 깸 카운터",
                             "관절체 잠", "관절체 플래그", "관절체 반복 수", "링크 플래그", "링크 질량", "링크 질량 중심 자세", "링크 관성", "링크 감쇠",
                             "관절 마찰", "관절 최대 속도", "관절 아마추어", "관절 한계", "관절 운동 종류", "조인트 국소 자세 0", "조인트 국소 자세 1",
                             "조인트 드라이브 자세", "조인트 드라이브 속도", "조인트 드라이브 값", "조인트 운동 종류", "조인트 플래그", "조인트 끊김 힘", "장면 중력"};

struct Snap {
  // (객체, 종류, 칸 번호) -> 낱말들
  std::map<std::tuple<const void*, uint16_t, uint16_t>, std::vector<uint32_t>> v;
  void put(const void* o, Kind k, uint16_t i, const void* p, size_t bytes) {
    std::vector<uint32_t>& w = v[{o, uint16_t(k), i}];
    w.resize((bytes + 3) / 4);
    memset(w.data(), 0, w.size() * 4);
    memcpy(w.data(), p, bytes);
  }
  template <class T>
  void val(const void* o, Kind k, uint16_t i, const T& x) { put(o, k, i, &x, sizeof(T)); }
};

void snapshot(PxScene* scene, Snap& S) {
  S.v.clear();
  S.val(scene, kScGravity, 0, scene->getGravity());
  const PxU32 na = scene->getNbActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC);
  std::vector<PxActor*> acts(na);
  scene->getActors(PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC, acts.data(), na);
  for (PxActor* a : acts) {
    if (a->getType() == PxActorType::eRIGID_STATIC) {
      S.val(a, kRsPose, 0, static_cast<PxRigidStatic*>(a)->getGlobalPose());
      continue;
    }
    auto* d = static_cast<PxRigidDynamic*>(a);
    const bool kin = d->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
    S.val(a, kRdPose, 0, d->getGlobalPose());
    S.val(a, kRdLinVel, 0, d->getLinearVelocity());
    S.val(a, kRdAngVel, 0, d->getAngularVelocity());
    if (!kin) S.val(a, kRdWake, 0, d->getWakeCounter());
    S.val(a, kRdSleeping, 0, uint32_t(d->isSleeping()));
    S.val(a, kRdFlags, 0, uint32_t(PxU16(d->getRigidBodyFlags())));
    PxTransform kt;
    if (kin && d->getKinematicTarget(kt)) S.val(a, kRdKinTarget, 0, kt);
    S.val(a, kRdMass, 0, d->getMass());
    S.val(a, kRdCMass, 0, d->getCMassLocalPose());
    S.val(a, kRdInertia, 0, d->getMassSpaceInertiaTensor());
    S.val(a, kRdLinDamp, 0, d->getLinearDamping());
    S.val(a, kRdAngDamp, 0, d->getAngularDamping());
    S.val(a, kRdMaxDepen, 0, d->getMaxDepenetrationVelocity());
    S.val(a, kRdSleepThr, 0, d->getSleepThreshold());
    S.val(a, kRdStabThr, 0, d->getStabilizationThreshold());
    S.val(a, kRdLockFlags, 0, uint32_t(PxU8(d->getRigidDynamicLockFlags())));
    PxU32 pi = 0, vi = 0;
    d->getSolverIterationCounts(pi, vi);
    S.val(a, kRdIters, 0, pi | (vi << 16));
    S.val(a, kRdActorFlags, 0, uint32_t(PxU8(d->getActorFlags())));
    S.val(a, kRdMaxLinVel, 0, d->getMaxLinearVelocity());
    S.val(a, kRdMaxAngVel, 0, d->getMaxAngularVelocity());
    S.val(a, kRdContactSlop, 0, d->getContactSlopCoefficient());
  }
  const PxU32 nArt = scene->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate*> arts(nArt);
  scene->getArticulations(arts.data(), nArt);
  for (PxArticulationReducedCoordinate* ar : arts) {
    S.val(ar, kArRootPose, 0, ar->getRootGlobalPose());
    S.val(ar, kArRootLinVel, 0, ar->getRootLinearVelocity());
    S.val(ar, kArRootAngVel, 0, ar->getRootAngularVelocity());
    S.val(ar, kArWake, 0, ar->getWakeCounter());
    S.val(ar, kArSleeping, 0, uint32_t(ar->isSleeping()));
    S.val(ar, kArFlags, 0, uint32_t(PxU8(ar->getArticulationFlags())));
    PxU32 pi = 0, vi = 0;
    ar->getSolverIterationCounts(pi, vi);
    S.val(ar, kArIters, 0, pi | (vi << 16));
    std::vector<PxArticulationLink*> links(ar->getNbLinks());
    ar->getLinks(links.data(), PxU32(links.size()));
    for (uint16_t l = 0; l < links.size(); ++l) {
      PxArticulationLink* lk = links[l];
      S.val(ar, kArLinkFlags, l, uint32_t(PxU16(lk->getRigidBodyFlags())) | (uint32_t(PxU8(lk->getActorFlags())) << 16));
      S.val(ar, kArLinkMass, l, lk->getMass());
      S.val(ar, kArLinkCMass, l, lk->getCMassLocalPose());
      S.val(ar, kArLinkInertia, l, lk->getMassSpaceInertiaTensor());
      const float damp[2] = {lk->getLinearDamping(), lk->getAngularDamping()};
      S.val(ar, kArLinkDamp, l, damp);
      PxArticulationJointReducedCoordinate* j = lk->getInboundJoint();
      if (!j) continue;
      for (int ax = 0; ax < 6; ++ax) {
        const PxArticulationAxis::Enum A = PxArticulationAxis::Enum(ax);
        const PxArticulationMotion::Enum mo = j->getMotion(A);
        const uint16_t k = uint16_t(l * 6 + ax);
        S.val(ar, kArJointMotion, k, uint32_t(mo));
        if (mo == PxArticulationMotion::eLOCKED) continue;
        S.val(ar, kArJointPos, k, j->getJointPosition(A));
        S.val(ar, kArJointVel, k, j->getJointVelocity(A));
        S.val(ar, kArDriveTarget, k, j->getDriveTarget(A));
        S.val(ar, kArDriveVel, k, j->getDriveVelocity(A));
        S.val(ar, kArDrive, k, j->getDriveParams(A));
        S.val(ar, kArJointArmature, k, j->getArmature(A));
        S.val(ar, kArJointLimit, k, j->getLimitParams(A));
      }
      S.val(ar, kArJointFriction, l, j->getFrictionCoefficient());
      S.val(ar, kArJointMaxVel, l, j->getMaxJointVelocity());
    }
  }
  const PxU32 nc = scene->getNbConstraints();
  std::vector<PxConstraint*> cs(nc);
  scene->getConstraints(cs.data(), nc);
  for (PxConstraint* c : cs) {
    PxU32 tid = 0;
    void* ext = c->getExternalReference(tid);
    if (!ext || tid != PxConstraintExtIDs::eJOINT) continue;
    PxJoint* jb = static_cast<PxJoint*>(ext);
    S.val(jb, kJLocal0, 0, jb->getLocalPose(PxJointActorIndex::eACTOR0));
    S.val(jb, kJLocal1, 0, jb->getLocalPose(PxJointActorIndex::eACTOR1));
    S.val(jb, kJFlags, 0, uint32_t(PxU16(jb->getConstraintFlags())));
    float bf[2];
    jb->getBreakForce(bf[0], bf[1]);
    S.val(jb, kJBreak, 0, bf);
    if (jb->getConcreteType() != PxJointConcreteType::eD6) continue;
    PxD6Joint* d6 = static_cast<PxD6Joint*>(jb);
    S.val(jb, kJDrivePos, 0, d6->getDrivePosition());
    PxVec3 lv, av;
    d6->getDriveVelocity(lv, av);
    const PxVec3 v2[2] = {lv, av};
    S.val(jb, kJDriveVel, 0, v2);
    for (int ax = 0; ax < 6; ++ax) S.val(jb, kJMotion, uint16_t(ax), uint32_t(d6->getMotion(PxD6Axis::Enum(ax))));
    for (int dr = 0; dr < 6; ++dr) S.val(jb, kJDrive, uint16_t(dr), d6->getDrive(PxD6Drive::Enum(dr)));
  }
}

// 창 편집 하나 (닫힌 고리가 우리 상태에 넣는 것)
struct WindowEdit {
  const void* obj;  // PxActor* / PxArticulationReducedCoordinate* / PxJoint* / PxScene*
  uint16_t kind, index;
  std::vector<uint32_t> value;
};
struct LoopShadow {
  bool inited = false, on = false, haveS0 = false;
  PxScene* scene = nullptr;
  Snap s0, s1;
  uint64_t windows = 0, windowsEdited = 0, edits = 0, appear = 0, vanish = 0;
  uint64_t perKind[kCount] = {}, perKindWin[kCount] = {};
  std::map<std::string, uint64_t> objEdits;  // 진단: 편집이 많은 객체
  std::vector<WindowEdit> lastEdits;         // 이번 창 편집 (다음 simulate 앞에 채움, 순서 = 객체·종류·칸 순)
  int show = 0;
} LS;

const char* objName(const void* o) {
  // 이름은 PxActor·PxArticulation·PxJoint 모두 getName 이 있으나 형이 달라 종류로 가른다 (칸 종류로 판단)
  return o ? "?" : "-";
}

}  // namespace

// ---- (2b) API 거울
namespace {
struct ApiMirror {
  bool inited = false, persist = false;
  std::unordered_set<const void*> touched;
  std::unordered_map<const void*, std::vector<G1ArtOp>> artOps;
  uint64_t touches = 0, touchOther = 0, drives = 0, wakes = 0;
} AM;
void amInit() {
  if (AM.inited) return;
  AM.inited = true;
  AM.persist = getenv("G1_LOOP_PERSIST") != nullptr;
}
}  // namespace
bool g1_loop_persist() {
  amInit();
  return AM.persist;
}
bool g1_loop_touched(const void* obj) { return AM.touched.count(obj) != 0; }
void g1_loop_take_art_ops(const void* art, std::vector<G1ArtOp>& out) {
  out.clear();
  auto it = AM.artOps.find(art);
  if (it == AM.artOps.end()) return;
  out.swap(it->second);
  AM.artOps.erase(it);
}
void g1_api_touch(PxBase* b) {
  amInit();
  if (!AM.persist || !b) return;
  ++AM.touches;
  if (auto* j = b->is<PxArticulationJointReducedCoordinate>()) AM.touched.insert(&j->getChildArticulationLink().getArticulation());
  else if (auto* l = b->is<PxArticulationLink>()) AM.touched.insert(&l->getArticulation());
  else if (auto* a = b->is<PxArticulationReducedCoordinate>()) AM.touched.insert(a);
  else if (auto* r = b->is<PxRigidActor>()) AM.touched.insert(r);
  else if (auto* s = b->is<PxShape>()) { if (s->getActor()) AM.touched.insert(s->getActor()); }
  else ++AM.touchOther;  // 장면·재질 등 (몸체 상태와 무관 추정)
}
void g1_api_art_drive(PxArticulationJointReducedCoordinate* j, int axis, float v, bool velocity) {
  amInit();
  if (!AM.persist) return;
  ++AM.drives;
  PxArticulationLink& l = j->getChildArticulationLink();
  PxArticulationReducedCoordinate& art = l.getArticulation();
  // 생성 순서 번호 (getLinks 순서; getLinkIndex 는 LL 번호라 다르다)
  static std::vector<PxArticulationLink*> links;
  links.resize(art.getNbLinks());
  art.getLinks(links.data(), PxU32(links.size()));
  uint32_t ci = 0;
  while (ci < links.size() && links[ci] != &l) ++ci;
  AM.artOps[&art].push_back(G1ArtOp{uint8_t(velocity ? 1 : 0), uint8_t(axis), ci, v});
}
void g1_api_art_wake(PxArticulationReducedCoordinate* a, bool sleep) {
  amInit();
  if (!AM.persist) return;
  ++AM.wakes;
  AM.artOps[a].push_back(G1ArtOp{uint8_t(sleep ? 3 : 2), 0, 0, 0.f});
}

void g1_loop_after(PxScene* scene, uint64_t) {
  amInit();
  AM.touched.clear();  // 다음 창부터 새로 (지난 창 것은 이번 simulate 에서 썼음)
  AM.artOps.clear();
  if (!LS.inited) {
    LS.inited = true;
    LS.on = getenv("G1_LOOP") != nullptr;
    LS.show = getenv("G1_LOOP_SHOW") ? atoi(getenv("G1_LOOP_SHOW")) : 0;
  }
  if (!LS.on) return;
  LS.scene = scene;
  snapshot(scene, LS.s0);
  LS.haveS0 = true;
}

void g1_loop_before(PxScene* scene, uint64_t sim) {
  if (const char* want = getenv("G1_TOUCH_NAME")) {  // 진단: 이름에 want 가 든 강체의 simulate 앞 자세 (처음 나온 뒤 3 번)
    static int shown = 0;
    if (shown < 3) {
      const PxU32 n = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      std::vector<PxActor*> as(n);
      scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, as.data(), n);
      bool any = false;
      for (PxActor* x : as)
        if (x->getName() && strstr(x->getName(), want)) {
          const PxTransform t = static_cast<PxRigidActor*>(x)->getGlobalPose();
          fprintf(stderr, "[simulate %llu 앞] %s q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g\n", (unsigned long long)sim, x->getName(), t.q.x, t.q.y, t.q.z, t.q.w, t.p.x,
                  t.p.y, t.p.z);
          any = true;
        }
      if (any) ++shown;
    }
  }
  if (!LS.on || !LS.haveS0 || scene != LS.scene) return;
  snapshot(scene, LS.s1);
  ++LS.windows;
  LS.lastEdits.clear();
  bool any = false;
  bool seenKind[kCount] = {};
  for (auto& kv : LS.s1.v) {
    auto it = LS.s0.v.find(kv.first);
    if (it == LS.s0.v.end()) {  // 새로 생긴 칸 (행위자·조인트 추가) — 값도 편집으로 낸다
      ++LS.appear;
      any = true;
      LS.lastEdits.push_back(WindowEdit{std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), kv.second});
      continue;
    }
    if (it->second == kv.second) continue;
    const uint16_t k = std::get<1>(kv.first);
    ++LS.edits;
    ++LS.perKind[k];
    LS.lastEdits.push_back(WindowEdit{std::get<0>(kv.first), k, std::get<2>(kv.first), kv.second});
    seenKind[k] = true;
    any = true;
    if (LS.show > 0) {
      --LS.show;
      const float* a = reinterpret_cast<const float*>(it->second.data());
      const float* b = reinterpret_cast<const float*>(kv.second.data());
      fprintf(stderr, "[g1 loop 편집] simulate %llu %s 칸 %u: %g %g %g -> %g %g %g (낱말 %zu)\n", (unsigned long long)sim, kName[k], std::get<2>(kv.first), a[0],
              it->second.size() > 1 ? a[1] : 0.f, it->second.size() > 2 ? a[2] : 0.f, b[0], kv.second.size() > 1 ? b[1] : 0.f, kv.second.size() > 2 ? b[2] : 0.f,
              kv.second.size());
    }
  }
  for (auto& kv : LS.s0.v)
    if (!LS.s1.v.count(kv.first)) { ++LS.vanish; any = true; }
  for (int k = 0; k < kCount; ++k)
    if (seenKind[k]) ++LS.perKindWin[k];
  if (any) ++LS.windowsEdited;
  LS.haveS0 = false;
}

void g1_loop_report() {
  if (AM.persist)
    printf("G1 닫힌 고리 (2b) API 거울: 옮긴 관절 드라이브 %" PRIu64 ", 관절체 깨움·재움 %" PRIu64 ", 건드림(다시 맞춤) %" PRIu64 " (몸체와 무관 %" PRIu64 ")\n", AM.drives,
           AM.wakes, AM.touches, AM.touchOther);
  if (!LS.on) return;
  printf("G1 닫힌 고리 (2a) API 창: 창 %" PRIu64 " (편집 있는 창 %" PRIu64 "), 칸 편집 %" PRIu64 ", 새로 생긴 칸 %" PRIu64 ", 없어진 칸 %" PRIu64 "\n", LS.windows,
         LS.windowsEdited, LS.edits, LS.appear, LS.vanish);
  for (int k = 0; k < kCount; ++k)
    if (LS.perKind[k]) printf("  %-24s 편집 %8" PRIu64 " (창 %" PRIu64 ")\n", kName[k], LS.perKind[k], LS.perKindWin[k]);
  (void)objName;
}
