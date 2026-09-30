// bp.h 가 번역 조각(gen_bp.inc) 앞에 넣는 손으로 짠 바탕: 할당 매크로·프로파일·작업(task) 형·임시 할당기.
// 결과 값·순서에 영향 없는 것들뿐이다 (호스트 전용 층 1).
namespace px {

#define EPX_ALLOC(n, name) malloc(n)
#define EPX_ALLOCATE(type, count, name) reinterpret_cast<type*>(malloc((count) * sizeof(type)))
#define E_FREE(x) do { free(x); x = nullptr; } while (0)
#define EPX_NEW(T) new T
#define E_DELETE(x) do { if (x) { delete x; x = nullptr; } } while (0)
#define EPX_DELETE_ARRAY(x) do { if (x) { delete[] x; x = nullptr; } } while (0)
#define EPX_DELETE_THIS delete this
#define EPX_PROFILE_ZONE(...) ((void)0)
#define EPX_OVERRIDE override
#define EPX_FINAL final
#define EPX_IR(x) ((PxU32&)(x))
#define EPX_SIR(x) ((PxI32&)(x))
#define EPX_FR(x) ((PxReal&)(x))
#define EPX_SIMD_GUARD
#define EPX_CHECK_MSG(...) ((void)0)
#define EPX_CHECK_AND_RETURN(...) ((void)0)

// include/task/PxTask.h 의 작업 형 — 한 스레드로 차례대로 흉내: 참조가 0 이 되는 순간 그 자리에서 run() 하고
// 이어질 작업(continuation)의 참조를 내린다. PABP 의 작업들은 각자 버퍼에 쓰고 끝에 번호 순서로 합치므로
// (BpBroadPhaseABP.cpp addDelayedPairs2) 실행 순서가 결과를 바꾸지 않는다.
class PxTaskManager;
class PxBaseTask {
 public:
  EHDI PxBaseTask() : mContextID(0), mTm(nullptr) {}
  EHDI virtual ~PxBaseTask() {}
  EHDI virtual void run() = 0;
  EHDI virtual const char* getName() const = 0;
  EHDI virtual void addReference() = 0;
  EHDI virtual void removeReference() = 0;
  EHDI virtual int32_t getReference() const = 0;
  EHDI virtual void release() = 0;
  EHDI virtual bool isHighPriority() const { return false; }
  EHDI PxTaskManager* getTaskManager() const { return mTm; }
  EHDI void setContextId(PxU64 id) { mContextID = id; }
  EHDI PxU64 getContextId() const { return mContextID; }
 protected:
  PxU64 mContextID;
  PxTaskManager* mTm;
};
class PxLightCpuTask : public PxBaseTask {
 public:
  EHDI PxLightCpuTask() : mCont(nullptr), mRefCount(0) {}
  EHDI void setContinuation(PxTaskManager& tm, PxBaseTask* c) { mRefCount = 1; mCont = c; mTm = &tm; if (mCont) mCont->addReference(); }
  EHDI void setContinuation(PxBaseTask* c) { mRefCount = 1; mCont = c; if (mCont) { mCont->addReference(); mTm = c->getTaskManager(); } }
  EHDI PxBaseTask* getContinuation() const { return mCont; }
  EHDI void removeReference() override {
    if (--mRefCount == 0) {
      run();
      release();
    }
  }
  EHDI int32_t getReference() const override { return mRefCount; }
  EHDI void addReference() override { ++mRefCount; }
  EHDI void release() override {
    PxBaseTask* c = mCont;
    mCont = nullptr;
    if (c) c->removeReference();
  }
 protected:
  PxBaseTask* mCont;
  int32_t mRefCount;
};

// 시험·파이프라인에서 PABP 를 "작업 경로"로 돌리려고 넘기는 끝 작업 (아무것도 안 함)
class EndTask : public PxLightCpuTask {
 public:
  bool done = false;
  EHDI void run() override { done = true; }
  EHDI const char* getName() const override { return "EndTask"; }
};

// PxcScratchAllocator (lowlevel/common/include/utils/PxcScratchAllocator.h): 층 1 에서는 늘 힙으로
class PxcScratchAllocator {
 public:
  EHDI void* alloc(PxU32 requestedSize, bool fallBackToHeap = false) {
    (void)fallBackToHeap;
    return malloc((requestedSize + 15) & ~15u);
  }
  EHDI void free(void* addr) { ::free(addr); }
};

}  // namespace px

// include/PxBroadPhase.h 의 쓰이는 형 (MBP 영역 기능은 ABP 에서 안 씀)
namespace px {
struct PxBroadPhaseType {
  enum Enum { eSAP, eMBP, eABP, ePABP, eGPU, eLAST };
};
struct PxBroadPhaseRegion {
  PxBounds3 mBounds;
  void* mUserData;
};
struct PxBroadPhaseRegionInfo {
  PxBroadPhaseRegion mRegion;
  PxU32 mNbStaticObjects;
  PxU32 mNbDynamicObjects;
  bool mActive;
  bool mOverlap;
};
struct PxBroadPhaseCaps {
  PxU32 mMaxNbRegions;
};
class PxBroadPhaseRegions {
 public:
  EHDI virtual ~PxBroadPhaseRegions() {}
  EHDI virtual PxU32 getNbRegions() const = 0;
  EHDI virtual PxU32 getRegions(PxBroadPhaseRegionInfo* userBuffer, PxU32 bufferSize, PxU32 startIndex = 0) const = 0;
  EHDI virtual PxU32 addRegion(const PxBroadPhaseRegion& region, bool populateRegion, const PxBounds3* bounds, const PxReal* distances) = 0;
  EHDI virtual bool removeRegion(PxU32 handle) = 0;
  EHDI virtual PxU32 getNbOutOfBoundsObjects() const = 0;
  EHDI virtual const PxU32* getOutOfBoundsObjects() const = 0;
};
namespace Cm {
// common/src/CmUtils.h:278
template <class T>
EHDI T* reserveContainerMemory(PxArray<T>& container, PxU32 nb) {
  const PxU32 maxNbEntries = container.capacity();
  const PxU32 requiredSize = container.size() + nb;
  if (requiredSize > maxNbEntries) {
    const PxU32 naturalGrowthSize = maxNbEntries ? maxNbEntries * 2 : 2;
    const PxU32 newSize = PxMax(requiredSize, naturalGrowthSize);
    container.reserve(newSize);
  }
  T* buf = container.end();
  container.forceSize_Unsafe(requiredSize);
  return buf;
}
}  // namespace Cm
}  // namespace px
