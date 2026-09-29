// aabb.h 가 번역 조각(gen_aabb.inc) 앞에 넣는 손으로 짠 바탕 (호스트 전용 층 1).
// 할당기·작업(task)·잠금·풀(pool) 처럼 결과 값·순서에 영향 없는 것만 둔다.
// 순서에 영향 있는 컨테이너(PxHashMap·PxHashSet·PxCoalescedHashMap·PxBitMap·PxSort)는 원본을 번역한 것을 쓴다.
namespace px {

// foundation/PxAllocator.h: 힙 할당기 (PxHashBase·PxBitMap 이 부름)
class PxAllocator {
 public:
  PxAllocator(const char* = nullptr) {}
  template <class A> PxAllocator(const A&) {}
  void* allocate(size_t size, const char*, int) { return size ? malloc(size) : nullptr; }
  void deallocate(void* ptr) { free(ptr); }
  PxAllocator& getCallback() { return *this; }
};
typedef PxAllocator PxVirtualAllocator;  // foundation/PxVirtualAllocator.h (핀 메모리 = 보통 힙으로)
template <class T> class PxReflectionAllocator : public PxAllocator {
 public:
  PxReflectionAllocator(const char* = nullptr) {}
  template <class A> PxReflectionAllocator(const A&) {}
};

// foundation/PxPinnedArray.h: 핀 배열 = 보통 배열
template <class T> using PxPinnedArray = PxArray<T>;
template <class T> using PxPinnedArraySafe = PxArray<T>;
typedef PxArray<PxBounds3> PxBoundsArrayPinned;
typedef PxArray<PxReal> PxFloatArrayPinnedSafe;
typedef PxArray<PxReal> PxFloatArrayPinned;
typedef PxArray<PxI32> PxInt32ArrayPinnedSafe;
typedef PxArray<PxU32> PxInt32ArrayPinned;

// foundation/PxInlineArray.h: 앞 N 칸 내장 배열 — 크기·순서만 같으면 되므로 보통 배열로
template <class T, PxU32 N> class PxInlineArray : public PxArray<T> {
 public:
  PxInlineArray(const char* = nullptr) { this->reserve(N); }
};

// foundation/PxMutex.h, PxSList.h: 한 스레드라 잠금은 없음, SList 는 LIFO 스택(원본과 같은 LIFO)
class PxMutex {
 public:
  void lock() {}
  void unlock() {}
  class ScopedLock {
   public:
    explicit ScopedLock(PxMutex&) {}
  };
};
struct PxSListEntry {
  PxSListEntry* mNext = nullptr;
  PxSListEntry* next() { return mNext; }
};
class PxSList {
 public:
  void push(PxSListEntry& e) { e.mNext = mHead; mHead = &e; }
  PxSListEntry* pop() { PxSListEntry* e = mHead; if (e) mHead = e->mNext; return e; }
  PxSListEntry* flush() { PxSListEntry* e = mHead; mHead = nullptr; return e; }
 private:
  PxSListEntry* mHead = nullptr;
};

namespace intrinsics {  // foundation/unix/PxUnixIntrinsics.h
inline void* memSet(void* dest, int32_t c, uint32_t count) { return memset(dest, c, count); }
inline void* memCopy(void* dest, const void* src, uint32_t count) { return memcpy(dest, src, count); }
inline void* memMove(void* dest, const void* src, uint32_t count) { return memmove(dest, src, count); }
inline void* memZero(void* dest, uint32_t count) { return memset(dest, 0, count); }
}  // namespace intrinsics
// PxAlloca(스택) -> 힙 배열 (층 1 호스트, 값·순서 영향 없음)
#define EPX_ALLOCA(var, type, count) std::vector<type> var##_allocaVec(count); type* var = var##_allocaVec.data()
template <class A> class PxBitMapBase;
typedef PxBitMapBase<PxAllocator> PxBitMapPinned;  // foundation/PxPinnedBitMap.h

// include/PxFiltering.h:40
struct PxPairFilteringMode {
  enum Enum { eKEEP, eSUPPRESS, eKILL, eDEFAULT = eSUPPRESS };
};
// include/PxAggregate.h:45~ (필터 힌트 비트 배치)
struct PxAggregateType {
  enum Enum { eGENERIC = 0, eSTATIC = 1, eKINEMATIC = 2 };
};
typedef PxU32 PxAggregateFilterHint;
inline PxAggregateFilterHint PxGetAggregateFilterHint(PxAggregateType::Enum type, bool enableSelfCollision) {
  const PxU32 selfCollisionBit = enableSelfCollision ? 1 : 0;
  return PxAggregateFilterHint((PxU32(type) << 1) | selfCollisionBit);
}
inline PxU32 PxGetAggregateSelfCollisionBit(PxAggregateFilterHint hint) { return hint & 1; }
inline PxAggregateType::Enum PxGetAggregateType(PxAggregateFilterHint hint) { return PxAggregateType::Enum(hint >> 1); }

// 시각화·오류 보고 (결과에 영향 없음)
struct PxDebugColor {
  enum Enum { eARGB_BLACK = 0xff000000, eARGB_RED = 0xffff0000, eARGB_GREEN = 0xff00ff00, eARGB_BLUE = 0xff0000ff, eARGB_CYAN = 0xff00ffff };
};
struct PxErrorCode {
  enum Enum { eNO_ERROR = 0, eDEBUG_INFO = 1, eDEBUG_WARNING = 2, eINVALID_PARAMETER = 4, eINVALID_OPERATION = 8, eOUT_OF_MEMORY = 16 };
};
struct PxFoundationStub {
  bool error(PxErrorCode::Enum, const char*, int, const char* msg, ...) { fprintf(stderr, "[eng px] %s", msg); return false; }
};
namespace Cm { inline void renderOutputDebugBox(PxRenderOutput&, const PxBounds3&) {} }  // CmVisualization.h (시각화만)
inline PxFoundationStub& PxGetFoundation() { static PxFoundationStub f; return f; }

namespace Cm {
// common/src/CmTask.h:48 (SIMD 지킴이는 값에 영향 없음: 호출하는 쪽이 FTZ 구간을 정한다)
class Task : public PxLightCpuTask {
 public:
  Task(PxU64 contextId) { mContextID = contextId; }
  void run() override { runInternal(); }
  virtual void runInternal() = 0;
};
template <class T, void (T::*Fn)(PxBaseTask*)>
class DelegateTask : public Cm::Task {
 public:
  DelegateTask(PxU64 contextID, T* obj, const char* name) : Cm::Task(contextID), mObj(obj), mName(name) {}
  void run() override { (mObj->*Fn)(mCont); }
  void runInternal() override { (mObj->*Fn)(mCont); }
  const char* getName() const override { return mName; }
  void setObject(T* obj) { mObj = obj; }
 private:
  T* mObj;
  const char* mName;
};
// common/src/CmFlushPool.h: 스텝 동안 쓰고 clear 때 한꺼번에 버리는 임시 메모리
class FlushPool {
 public:
  ~FlushPool() { clear(); }
  void* allocate(PxU32 size, PxU32 = 16) {
    void* p = aligned_alloc(16, (size + 15) & ~15u);
    mBlocks.push_back(p);
    return p;
  }
  void clear() { for (void* p : mBlocks) free(p); mBlocks.clear(); }
  void clearNotThreadSafe(PxU32 = 0) { clear(); }
 private:
  std::vector<void*> mBlocks;
};
}  // namespace Cm

}  // namespace px
