// G1 넓은 단계 그림자 (문서 15.3 v1-b, 리드, 비계 전용): 공식 radio 재생 중 PhysX AABB 관리자(Bp::AABBManager + PABP)와 똑같은 입력을
// 우리 AABB 관리자(core/contact/px/aabb.h, 기계 번역)에 나란히 넣고, 스텝마다 겹침 생성·소멸 목록(모양·트리거)을 순서까지 비교한다.
// 입력 잡기:
//  - 생성: PhysX 가 AABB 관리자를 만들 때(생성자 --wrap) 같은 인자로 우리 것도 만든다. PABP 생성 인자도 --wrap 으로 받는다.
//  - 구조 변경(addBounds·removeBounds·createAggregate·destroyAggregate·reallocateChangedAABBMgActorHandleMap): 모두 가상 함수라
//    --wrap 이 안 걸린다 -> 그 PhysX 관리자 객체의 가상 함수표를 복사해 해당 칸만 우리 가로채기로 바꾼다(원래 함수를 부른 뒤 우리 것에도).
//  - 스텝 입력(경계 상자 배열·접촉 거리·바뀜 비트맵·접촉 거리 바뀜 표시): 작업 "ScScene.broadPhaseFirstPass" 직전(가로채기 디스패처)에
//    Sc 장면의 배열을 통째로 복사하고 우리 세 단계(updateBPFirstPass·SecondPass·postBroadPhase)를 차례로 돈다.
//  - 비교: 작업 "ScScene.postBroadPhaseCont" 직전(PhysX postBroadPhase 끝) 두 관리자의 getCreatedOverlaps·getDestroyedOverlaps.
//    사용자 자료(userData)는 PhysX 가 넘긴 ElementSim 주소를 그대로 우리에게도 넘기므로 주소로 바로 비교한다.
// 켜기: G1_BP=1 (디스패처 필요).
// 넘겨받기(v1-b): G1_DUMP_AT 가 있으면 그 simulate 앞까지의 입력 기록을 모아 장면 파일에 넣고(core/scene/bp_log.h),
//   G1_BP_FROM=<장면 파일> 이면 그 파일의 경계 simulate 에서 우리 관리자를 기록으로 다시 세운 뒤(사용자 자료만 살아 있는 값으로 바꿈) 이어서 비교한다.
#include <cinttypes>
#include <map>
#include <memory>
#include <unordered_map>

#include "../tests/solver/px_internal.h"
#include "BpAABBManager.h"
#include "BpBroadPhase.h"
#include "CmFlushPool.h"
#include "PxcScratchAllocator.h"
#include "core/contact/px/aabb.h"
#include "core/scene/bp_log.h"
#include "core/scene/scene_file.h"
#include "g1_hooks.h"

using namespace physx;
namespace ep = eng::px;

namespace {

struct AbpArgs {
  PxU32 maxOverlaps = 0, maxStatic = 0, maxDynamic = 0;
  PxU64 ctx = 0;
  bool mt = true;
};

struct Mirror {
  Bp::AABBManager* pm = nullptr;
  std::unique_ptr<eng::scene::BpRuntime> rt;  // 우리 관리자 (null = 파일 넘겨받기를 기다림)
  bool ran = false;
};

// 보호 멤버 mVolumeData 읽기·쓰기 (사용자 자료 바꿔 달기)
struct PxVD : Bp::AABBManagerBase {
  static Bp::VolumeData* data(Bp::AABBManagerBase* m) { return static_cast<PxVD*>(m)->mVolumeData.begin(); }
  static PxU32 size(Bp::AABBManagerBase* m) { return static_cast<PxVD*>(m)->mVolumeData.size(); }
};
struct EVD : ep::Bp::AABBManagerBase {
  static ep::Bp::VolumeData* data(ep::Bp::AABBManagerBase* m) { return static_cast<EVD*>(m)->mVolumeData.begin(); }
  static uint32_t size(ep::Bp::AABBManagerBase* m) { return static_cast<EVD*>(m)->mVolumeData.size(); }
};

struct BpShadow {
  bool inited = false, on = false;
  std::unordered_map<const void*, AbpArgs> abpArgs;  // PhysX BroadPhaseABP* -> 생성 인자
  std::unordered_map<const void*, std::unique_ptr<Mirror>> mirrors;  // PhysX AABBManager* -> 우리 것
  Mirror* cur = nullptr;  // 지금 simulate 하는 장면의 것
  PxScene* scene = nullptr;
  uint64_t steps = 0, cmpShape[2] = {0, 0}, cmpTrig[2] = {0, 0}, bad = 0, handleBad = 0, notRan = 0;
  long long firstBad = -1;
  std::string firstWhat;
  uint64_t curSim = 0;
  int show = 0;
  // 넘겨받기 기록 (G1_DUMP_AT 앞까지)
  long long logUntil = -1;
  eng::scene::BpLog log;
  std::string from;
  bool fromDone = false;
  uint64_t fromPatched = 0;
} BS;
bool logging() { return BS.logUntil >= 0 && (long long)BS.curSim < BS.logUntil; }
void logOp(const eng::scene::BpOp& o) {
  if (!logging()) return;
  eng::scene::BpOp x = o;
  x.frame = uint32_t(BS.log.frames.size());
  BS.log.ops.push_back(x);
}

void init() {
  if (BS.inited) return;
  BS.inited = true;
  BS.on = getenv("G1_BP") != nullptr;
  BS.show = getenv("G1_BP_SHOW") ? atoi(getenv("G1_BP_SHOW")) : 0;
  if (const char* a = getenv("G1_DUMP_AT")) BS.logUntil = atoll(a);
  if (const char* f = getenv("G1_BP_FROM")) BS.from = f;
}

// ---- 가상 함수표 가로채기
struct VHook {
  std::vector<void*> table;  // [offset-to-top, RTTI, 함수들...]
  void* orig[64] = {};
};
std::unordered_map<const void*, std::unique_ptr<VHook>> gVHooks;

template <class MFP>
size_t slotOf(MFP f) {
  uintptr_t v;
  static_assert(sizeof(MFP) >= sizeof(uintptr_t), "");
  memcpy(&v, &f, sizeof(v));
  return (v - 1) / sizeof(void*);  // Itanium: 가상 멤버 함수 포인터 = 1 + 표 안 바이트 위치
}

size_t kAdd, kRemove, kCreateAgg, kDestroyAgg, kRealloc;

Mirror* mirrorOf(const void* pm) {
  auto it = BS.mirrors.find(pm);
  return it == BS.mirrors.end() ? nullptr : it->second.get();
}
VHook* hookOf(const void* pm) { return gVHooks[pm].get(); }

bool hkAdd(Bp::AABBManager* self, Bp::BoundsIndex index, PxReal cd, Bp::FilterGroup::Enum g, void* ud, Bp::AggregateHandle agg, Bp::ElementType::Enum vt,
           PxU32 env) {
  auto fn = reinterpret_cast<bool (*)(Bp::AABBManager*, Bp::BoundsIndex, PxReal, Bp::FilterGroup::Enum, void*, Bp::AggregateHandle, Bp::ElementType::Enum, PxU32)>(
      hookOf(self)->orig[kAdd]);
  const bool r = fn(self, index, cd, g, ud, agg, vt, env);
  eng::scene::BpOp o{};
  o.type = eng::scene::BP_ADD; o.index = index; o.contactDistance = cd; o.group = uint32_t(g); o.userData = uint64_t(uintptr_t(ud)); o.agg = agg;
  o.volumeType = uint32_t(vt); o.env = env; o.result = r;
  logOp(o);
  if (Mirror* M = mirrorOf(self))
    if (M->rt) {
      const uint32_t before = M->rt->handleBad;
      M->rt->apply(o);
      BS.handleBad += M->rt->handleBad - before;
    }
  return r;
}
bool hkRemove(Bp::AABBManager* self, Bp::BoundsIndex index) {
  auto fn = reinterpret_cast<bool (*)(Bp::AABBManager*, Bp::BoundsIndex)>(hookOf(self)->orig[kRemove]);
  const bool r = fn(self, index);
  eng::scene::BpOp o{};
  o.type = eng::scene::BP_REMOVE; o.index = index; o.result = r;
  logOp(o);
  if (Mirror* M = mirrorOf(self))
    if (M->rt) {
      const uint32_t before = M->rt->handleBad;
      M->rt->apply(o);
      BS.handleBad += M->rt->handleBad - before;
    }
  return r;
}
Bp::AggregateHandle hkCreateAgg(Bp::AABBManager* self, Bp::BoundsIndex index, Bp::FilterGroup::Enum g, void* ud, PxU32 maxNum, PxAggregateFilterHint hint,
                                PxU32 env) {
  auto fn = reinterpret_cast<Bp::AggregateHandle (*)(Bp::AABBManager*, Bp::BoundsIndex, Bp::FilterGroup::Enum, void*, PxU32, PxAggregateFilterHint, PxU32)>(
      hookOf(self)->orig[kCreateAgg]);
  const Bp::AggregateHandle h = fn(self, index, g, ud, maxNum, hint, env);
  eng::scene::BpOp o{};
  o.type = eng::scene::BP_CREATE_AGG; o.index = index; o.group = uint32_t(g); o.userData = uint64_t(uintptr_t(ud)); o.maxNum = maxNum; o.hint = hint; o.env = env;
  o.result = h;
  logOp(o);
  if (Mirror* M = mirrorOf(self))
    if (M->rt) {
      const uint32_t before = M->rt->handleBad;
      M->rt->apply(o);
      BS.handleBad += M->rt->handleBad - before;
    }
  return h;
}
bool hkDestroyAgg(Bp::AABBManager* self, Bp::BoundsIndex& index, Bp::FilterGroup::Enum& g, Bp::AggregateHandle h) {
  auto fn = reinterpret_cast<bool (*)(Bp::AABBManager*, Bp::BoundsIndex&, Bp::FilterGroup::Enum&, Bp::AggregateHandle)>(hookOf(self)->orig[kDestroyAgg]);
  const bool r = fn(self, index, g, h);
  eng::scene::BpOp o{};
  o.type = eng::scene::BP_DESTROY_AGG; o.agg = h; o.result = r;
  logOp(o);
  if (Mirror* M = mirrorOf(self))
    if (M->rt) {
      const uint32_t before = M->rt->handleBad;
      M->rt->apply(o);
      BS.handleBad += M->rt->handleBad - before;
    }
  return r;
}
void hkRealloc(Bp::AABBManager* self, const PxU32 size) {
  auto fn = reinterpret_cast<void (*)(Bp::AABBManager*, const PxU32)>(hookOf(self)->orig[kRealloc]);
  fn(self, size);
  eng::scene::BpOp o{};
  o.type = eng::scene::BP_REALLOC; o.size = size;
  logOp(o);
  if (Mirror* M = mirrorOf(self))
    if (M->rt) M->rt->apply(o);
}

void installHooks(Bp::AABBManager* pm) {
  kAdd = slotOf(&Bp::AABBManager::addBounds);
  kRemove = slotOf(&Bp::AABBManager::removeBounds);
  kCreateAgg = slotOf(&Bp::AABBManager::createAggregate);
  kDestroyAgg = slotOf(&Bp::AABBManager::destroyAggregate);
  kRealloc = slotOf(&Bp::AABBManager::reallocateChangedAABBMgActorHandleMap);
  const size_t N = 48;  // 표 칸 (AABBManager 가상 함수보다 넉넉히)
  std::unique_ptr<VHook> h(new VHook);
  void** vt = *reinterpret_cast<void***>(pm);
  h->table.assign(vt - 2, vt + N);
  for (size_t k = 0; k < N && k < 64; ++k) h->orig[k] = vt[k];
  void** nv = h->table.data() + 2;
  nv[kAdd] = reinterpret_cast<void*>(&hkAdd);
  nv[kRemove] = reinterpret_cast<void*>(&hkRemove);
  nv[kCreateAgg] = reinterpret_cast<void*>(&hkCreateAgg);
  nv[kDestroyAgg] = reinterpret_cast<void*>(&hkDestroyAgg);
  nv[kRealloc] = reinterpret_cast<void*>(&hkRealloc);
  *reinterpret_cast<void***>(pm) = nv;
  gVHooks[pm] = std::move(h);
}

}  // namespace

// ---- 생성 가로채기 (--wrap)
extern "C" {
void __real__ZN5physx2Bp13BroadPhaseABPC1Ejjjmb(void* self, PxU32 a, PxU32 b, PxU32 c, PxU64 d, bool e);
void __wrap__ZN5physx2Bp13BroadPhaseABPC1Ejjjmb(void* self, PxU32 a, PxU32 b, PxU32 c, PxU64 d, bool e) {
  __real__ZN5physx2Bp13BroadPhaseABPC1Ejjjmb(self, a, b, c, d, e);
  init();
  if (BS.on) BS.abpArgs[self] = AbpArgs{a, b, c, d, e};
}
void __real__ZN5physx2Bp11AABBManagerC1ERNS0_10BroadPhaseERNS0_11BoundsArrayERNS_7PxArrayIfNS_17PxPinnedAllocatorIfEEEEjjRNS_18PxVirtualAllocatorEmNS_19PxPairFilteringMode4EnumESE_(
    Bp::AABBManager*, Bp::BroadPhase&, Bp::BoundsArray&, PxArray<float, PxPinnedAllocator<float>>&, PxU32, PxU32, PxVirtualAllocator&, PxU64, PxPairFilteringMode::Enum,
    PxPairFilteringMode::Enum);
void __wrap__ZN5physx2Bp11AABBManagerC1ERNS0_10BroadPhaseERNS0_11BoundsArrayERNS_7PxArrayIfNS_17PxPinnedAllocatorIfEEEEjjRNS_18PxVirtualAllocatorEmNS_19PxPairFilteringMode4EnumESE_(
    Bp::AABBManager* self, Bp::BroadPhase& bp, Bp::BoundsArray& b, PxArray<float, PxPinnedAllocator<float>>& d, PxU32 maxAgg, PxU32 maxShapes,
    PxVirtualAllocator& alloc, PxU64 ctx, PxPairFilteringMode::Enum kk, PxPairFilteringMode::Enum sk) {
  __real__ZN5physx2Bp11AABBManagerC1ERNS0_10BroadPhaseERNS0_11BoundsArrayERNS_7PxArrayIfNS_17PxPinnedAllocatorIfEEEEjjRNS_18PxVirtualAllocatorEmNS_19PxPairFilteringMode4EnumESE_(
      self, bp, b, d, maxAgg, maxShapes, alloc, ctx, kk, sk);
  init();
  if (!BS.on) return;
  std::unique_ptr<Mirror> M(new Mirror);
  M->pm = self;
  const AbpArgs a = BS.abpArgs.count(&bp) ? BS.abpArgs[&bp] : AbpArgs{};
  if (!BS.abpArgs.count(&bp)) fprintf(stderr, "[g1 bp] PABP 생성 인자를 못 받음 (기본값)\n");
  // 기록 머리 (마지막으로 만든 관리자 것)
  BS.log = eng::scene::BpLog{};
  BS.log.abpMaxOverlaps = a.maxOverlaps; BS.log.abpMaxStatic = a.maxStatic; BS.log.abpMaxDynamic = a.maxDynamic; BS.log.abpMT = a.mt; BS.log.ctx = a.ctx;
  BS.log.maxAggregates = maxAgg; BS.log.maxShapes = maxShapes; BS.log.kineKine = uint32_t(kk); BS.log.staticKine = uint32_t(sk);
  BS.log.valid = true;
  if (BS.from.empty()) {
    M->rt.reset(new eng::scene::BpRuntime);
    M->rt->create(a.maxOverlaps, a.maxStatic, a.maxDynamic, a.ctx, a.mt, maxAgg, maxShapes, uint32_t(kk), uint32_t(sk));
  }
  BS.mirrors[self] = std::move(M);
  installHooks(self);
}
}  // extern "C"

void g1_bp_before(PxScene* scene, uint64_t sim) {
  init();
  if (!BS.on) return;
  BS.scene = scene;
  BS.curSim = sim;
  BS.cur = mirrorOf(static_cast<NpScene*>(scene)->getScScene().getAABBManager());
  if (BS.cur) BS.cur->ran = false;
  if (!BS.from.empty() && !BS.fromDone && BS.cur) {
    static eng::scene::SceneFile ff;
    static int state = 0;
    if (state == 0) {
      std::string err;
      state = eng::scene::readScene(BS.from.c_str(), ff, &err) && ff.bp.valid ? 1 : 2;
      if (state == 2) fprintf(stderr, "[g1 bp] 파일 넓은 단계 기록 읽기 실패: %s\n", err.c_str());
    }
    if (state == 1 && sim == ff.h.sim) {
      BS.fromDone = true;
      std::unique_ptr<eng::scene::BpRuntime> rt(new eng::scene::BpRuntime);
      eng::scene::BpLog lg = ff.bp;
      if (getenv("G1_BP_FROM_NEG")) {  // 음성 대조: 스텝 입력 없이 구조 변경만 (이러면 달라져야 한다)
        lg.frames.clear();
        for (auto& o : lg.ops) o.frame = 0;
      }
      const bool ok = rt->replay(lg);
      // 사용자 자료(ElementSim 주소) = 이번 실행의 살아 있는 값으로 (기록 때 주소는 다른 프로세스 것)
      Bp::VolumeData* pv = PxVD::data(BS.cur->pm);
      ep::Bp::VolumeData* ev = EVD::data(rt->m.get());
      const uint32_t n = std::min<uint32_t>(PxVD::size(BS.cur->pm), EVD::size(rt->m.get()));
      for (uint32_t i = 0; i < n; ++i) {
        static_assert(sizeof(Bp::VolumeData) == sizeof(ep::Bp::VolumeData), "VolumeData 배치");
        memcpy(static_cast<void*>(&ev[i]), &pv[i], sizeof(ev[i]));  // 사용자 자료 + 종류 비트 + 묶음 번호
        ++BS.fromPatched;
      }
      BS.handleBad += rt->handleBad;
      BS.cur->rt = std::move(rt);
      printf("G1 넓은 단계 넘겨받기: simulate %llu 에서 파일 기록(구조 변경 %zu, 스텝 %zu)으로 우리 관리자를 다시 세움 %s, 칸 %u 의 사용자 자료를 바꿔 닮 — 이 뒤로만 비교\n",
             (unsigned long long)sim, ff.bp.ops.size(), ff.bp.frames.size(), ok ? "됨" : "실패", n);
      BS.steps = BS.cmpShape[0] = BS.cmpShape[1] = BS.cmpTrig[0] = BS.cmpTrig[1] = BS.bad = 0;
    }
  }
}

// 가로채기 디스패처가 작업을 돌리기 직전에
void g1_bp_task(const char* name) {
  if (!BS.on || !BS.cur || !BS.scene) return;
  Mirror& M = *BS.cur;
  if (!strcmp(name, "ScScene.broadPhaseFirstPass")) {
    Sc::Scene& sc = static_cast<NpScene*>(BS.scene)->getScScene();
    Bp::BoundsArray& pb = sc.getBoundsArray();
    PxFloatArrayPinnedSafe& pd = *sc.mContactDistance;
    PxBitMapPinned& pc = M.pm->getChangedAABBMgActorHandleMap();
    const bool hasCD = sc.mHasContactDistanceChanged;
    if (logging()) {  // 넘겨받기 기록 (경계 앞 스텝 입력)
      eng::scene::BpFrame F{};
      F.boundsStart = BS.log.bounds.size() / 6;
      F.nBounds = pb.size();
      F.distStart = BS.log.dist.size();
      F.nDist = pd.size();
      F.wordsStart = BS.log.words.size();
      F.nWords = pc.getWordCount();
      F.hasContactDistanceChanged = hasCD;
      F.boundsChanged = pb.hasChanged();
      F.boundsStart *= 6;
      BS.log.bounds.insert(BS.log.bounds.end(), reinterpret_cast<const float*>(pb.begin()), reinterpret_cast<const float*>(pb.begin()) + 6 * size_t(pb.size()));
      BS.log.dist.insert(BS.log.dist.end(), pd.begin(), pd.begin() + pd.size());
      BS.log.words.insert(BS.log.words.end(), pc.getWords(), pc.getWords() + pc.getWordCount());
      BS.log.frames.push_back(F);
    }
    if (!M.rt) return;
    M.rt->step(reinterpret_cast<const ep::PxBounds3*>(pb.begin()), pb.size(), pb.hasChanged(), pd.begin(), pd.size(), pc.getWords(), pc.getWordCount(), hasCD);
    M.ran = true;
  } else if (!strcmp(name, "ScScene.postBroadPhaseCont")) {
    if (!M.ran) { ++BS.notRan; return; }
    ++BS.steps;
    for (int t = 0; t < 2; ++t)
      for (int w = 0; w < 2; ++w) {
        PxU32 nP = 0, nE = 0;
        const Bp::AABBOverlap* oP = w == 0 ? M.pm->getCreatedOverlaps(Bp::ElementType::Enum(t), nP) : M.pm->getDestroyedOverlaps(Bp::ElementType::Enum(t), nP);
        const ep::Bp::AABBOverlap* oE =
            w == 0 ? M.rt->m->getCreatedOverlaps(ep::Bp::ElementType::Enum(t), nE) : M.rt->m->getDestroyedOverlaps(ep::Bp::ElementType::Enum(t), nE);
        bool diff = nP != nE;
        for (PxU32 k = 0; k < PxMin(nP, nE); ++k) {
          (t ? BS.cmpTrig : BS.cmpShape)[w]++;
          diff |= oP[k].mUserData0 != oE[k].mUserData0 || oP[k].mUserData1 != oE[k].mUserData1;
        }
        if (diff) {
          if (!BS.bad) {
            BS.firstBad = (long long)BS.curSim;
            BS.firstWhat = std::string(t ? "트리거 " : "모양 ") + (w ? "소멸" : "생성") + " PhysX " + std::to_string(nP) + " / 우리 " + std::to_string(nE);
          }
          ++BS.bad;
          if (BS.show > 0) {
            --BS.show;
            fprintf(stderr, "[g1 bp 다름] sim %llu %s %s: PhysX %u / 우리 %u\n", (unsigned long long)BS.curSim, t ? "트리거" : "모양", w ? "소멸" : "생성", nP, nE);
          }
        }
      }
    M.rt->endStep();
  }
}

// 장면 파일 뜨기(g1_dump.cpp)가 부른다: 지금까지(경계 앞) 모은 넓은 단계 입력 기록
const eng::scene::BpLog* g1_bp_log() { return BS.on && BS.log.valid ? &BS.log : nullptr; }

void g1_bp_report() {
  if (!BS.on) return;
  printf("G1 넓은 단계 그림자 (PhysX AABB 관리자+PABP 와 같은 입력으로 우리 것): 관리자 %zu, 비교 스텝 %" PRIu64 " (우리가 안 돈 스텝 %" PRIu64 ")\n", BS.mirrors.size(),
         BS.steps, BS.notRan);
  printf("  겹침 비교: 모양 생성 %" PRIu64 " 소멸 %" PRIu64 ", 트리거 생성 %" PRIu64 " 소멸 %" PRIu64 " — 목록 다름 %" PRIu64 "%s, 번호·결과 다름 %" PRIu64 "\n", BS.cmpShape[0],
         BS.cmpShape[1], BS.cmpTrig[0], BS.cmpTrig[1], BS.bad, BS.bad ? ("  첫 다름 simulate " + std::to_string(BS.firstBad) + " " + BS.firstWhat).c_str() : "",
         BS.handleBad);
}
