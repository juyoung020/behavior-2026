// gu.h 가 번역 조각(gen_gu_convex.inc) 앞에 넣는 손으로 짠 바탕: 원본이 끌어오는 무거운 헤더(할당기·직렬화·SDF·PxContact.h)
// 대신 쓰이는 이름만 같은 뜻으로 둔다. 값·연산 순서에 영향 없는 것들뿐이다.
namespace px {

struct PxUserAllocated {};  // foundation/PxUserAllocated.h (할당기 연산자만 있음)

#define PXC_CONTACT_NO_FACE_INDEX 0xffffffff  // PxContact.h:52

// foundation/PxMemory.h (0 바이트면 아무것도 안 함)
EHD void* PxMemZero(void* dest, PxU32 count) { return count ? memset(dest, 0, count) : nullptr; }
EHD void* PxMemSet(void* dest, PxI32 c, PxU32 count) { return count ? memset(dest, c, count) : nullptr; }
EHD void* PxMemCopy(void* dest, const void* src, PxU32 count) { return count ? memcpy(dest, src, count) : nullptr; }
EHD void* PxMemMove(void* dest, const void* src, PxU32 count) {
  // 겹칠 수 있음: 앞/뒤에서부터 한 바이트씩 (장치 코드에 memmove 가 없어서)
  unsigned char* d = static_cast<unsigned char*>(dest);
  const unsigned char* s = static_cast<const unsigned char*>(src);
  if (d < s) { for (PxU32 k = 0; k < count; ++k) d[k] = s[k]; }
  else { for (PxU32 k = count; k > 0; --k) d[k - 1] = s[k - 1]; }
  return dest;
}

// foundation/PxMathUtils.h:154 (정의는 FdMathUtils.cpp). 삼각 메시 질의 경계 변환에만 쓰임 -> 접촉 경로에서 필요해지면 옮긴다.
EHDV PxVec3 PxOptimizeBoundingBox(PxMat33& basis);

template <class T> EHD void flip(T&) {}  // CmSerialize.h 바이트 순서 뒤집기 (직렬화 전용)

// PxMath.h:265 PxAcos = ::acosf(PxClamp(f,-1,1)) -> glibc 2.35 acosf 이식본
EHD float PxAcos(float f) { return eng::glibcx::acosf(PxClamp(f, -1.0f, 1.0f)); }

// 이름공간 수준 상수 표: 호스트 표 + GPU __constant__ 표, name_p() 가 실행 쪽 표를 돌려준다 (translate.py device_tables)
#if defined(__CUDACC__)
#define E_GTABLE(T, name, N, ...)                                 \
  __device__ __constant__ static T name##_d[N] = __VA_ARGS__;     \
  static const T name##_h[N] = __VA_ARGS__;                       \
  EHD const T* name##_p() {                                       \
    E_GTABLE_PICK(name)                                           \
  }
#if defined(__CUDA_ARCH__)
#define E_GTABLE_PICK(name) return name##_d;
#else
#define E_GTABLE_PICK(name) return name##_h;
#endif
#else
#define E_GTABLE(T, name, N, ...)                   \
  static const T name##_h[N] = __VA_ARGS__;         \
  EHD const T* name##_p() { return name##_h; }
#endif
#define E_GTABLE_DECL(T, name) EHD const T* name##_p()

// 할당기 달린 컨테이너(동적 PriorityQueue·IDPool) 는 접촉 경로에서 안 쓴다: 템플릿이 파싱만 되게 이름만 둔다
struct PxDummyAlloc {
  EHD void* allocate(size_t, const char*, int) { return nullptr; }
  EHD void deallocate(void*) {}
};
template <class T> struct PxAllocatorTraits { typedef PxDummyAlloc Type; };
// foundation/PxArray.h 의 쓰이는 부분 (호스트용 동적 배열; 늘리는 규칙은 결과에 영향 없음 — 크기·순서만 같으면 됨)
template <class T> class PxArray {
 public:
  EHDI PxArray() : mData(nullptr), mSize(0), mCapacity(0) {}
  EHDI explicit PxArray(const char*) : mData(nullptr), mSize(0), mCapacity(0) {}
  template <class A> EHDI explicit PxArray(const A&) : mData(nullptr), mSize(0), mCapacity(0) {}
  EHDI PxArray(const PxArray& o) : mData(nullptr), mSize(0), mCapacity(0) { assignFrom(o); }
  EHDI PxArray& operator=(const PxArray& o) { if (this != &o) { mSize = 0; assignFrom(o); } return *this; }
  EHDI ~PxArray() { freeAll(); }
  EHD PxU32 size() const { return mSize; }
  EHD PxU32 capacity() const { return mCapacity; }
  EHD bool empty() const { return mSize == 0; }
  EHD T* begin() { return mData; }
  EHD const T* begin() const { return mData; }
  EHD T* end() { return mData + mSize; }
  EHD const T* end() const { return mData + mSize; }
  EHD T& operator[](PxU32 i) { return mData[i]; }
  EHD const T& operator[](PxU32 i) const { return mData[i]; }
  EHD T& back() { return mData[mSize - 1]; }
  EHD const T& back() const { return mData[mSize - 1]; }
  EHD T& front() { return mData[0]; }
  EHDI T& pushBack(const T& v) {
    if (mSize == mCapacity) {
      const T copy = v;  // v 가 배열 안을 가리킬 수 있음
      grow(mCapacity ? mCapacity * 2 : 1);
      new (mData + mSize) T(copy);
    } else {
      new (mData + mSize) T(v);
    }
    return mData[mSize++];
  }
  EHDI T& insert() { if (mSize == mCapacity) grow(mCapacity ? mCapacity * 2 : 1); new (mData + mSize) T(); return mData[mSize++]; }
  EHDI T popBack() { T t = mData[mSize - 1]; mData[--mSize].~T(); return t; }
  EHDI void reserve(PxU32 n) { if (n > mCapacity) grow(n); }
  EHDI void resize(PxU32 n, const T& v = T()) {
    reserve(n);
    for (PxU32 i = mSize; i < n; ++i) new (mData + i) T(v);
    for (PxU32 i = n; i < mSize; ++i) mData[i].~T();
    mSize = n;
  }
  EHDI void resizeUninitialized(PxU32 n) { reserve(n); mSize = n; }
  EHDI void forceSize_Unsafe(PxU32 n) { mSize = n; }
  EHDI void clear() { for (PxU32 i = 0; i < mSize; ++i) mData[i].~T(); mSize = 0; }
  EHDI void reset() { freeAll(); }
  EHDI void resetOrClear() { clear(); }
  EHDI void shrink() {}
  EHDI void remove(PxU32 i) {  // 뒤를 한 칸씩 당김 (순서 유지)
    for (PxU32 k = i; k + 1 < mSize; ++k) mData[k] = mData[k + 1];
    mData[--mSize].~T();
  }
  EHDI void replaceWithLast(PxU32 i) {  // 마지막 것을 i 자리로 (순서 안 지킴)
    mData[i] = mData[mSize - 1];
    mData[--mSize].~T();
  }
  EHDI bool findAndReplaceWithLast(const T& v) {
    for (PxU32 i = 0; i < mSize; ++i)
      if (mData[i] == v) { replaceWithLast(i); return true; }
    return false;
  }
  EHDI T* find(const T& v) { for (PxU32 i = 0; i < mSize; ++i) if (mData[i] == v) return mData + i; return end(); }

 private:
  T* mData;
  PxU32 mSize, mCapacity;
  EHDI void grow(PxU32 n) {
    T* nd = static_cast<T*>(malloc(sizeof(T) * n));
    for (PxU32 i = 0; i < mSize; ++i) { new (nd + i) T(mData[i]); mData[i].~T(); }
    free(mData);
    mData = nd;
    mCapacity = n;
  }
  EHDI void freeAll() { clear(); free(mData); mData = nullptr; mCapacity = 0; }
  EHDI void assignFrom(const PxArray& o) { reserve(o.mSize); for (PxU32 i = 0; i < o.mSize; ++i) new (mData + i) T(o.mData[i]); mSize = o.mSize; }
};

#define EPX_ALWAYS_ASSERT_MESSAGE(...) ((void)0)
#define E_ALLOCA_N 256  // PxAlloca 대신 쓰는 고정 배열 크기 (볼록 꼭짓점 ≤255, 접촉 버퍼 256)

// include/collision/PxCollisionDefs.h:67
struct PxCache {
  PxU8* mCachedData;
  PxU16 mCachedSize;
  PxU8 mPairData;
  PxU8 mManifoldFlags;
  EHD PxCache() : mCachedData(nullptr), mCachedSize(0), mPairData(0), mManifoldFlags(0) {}
  EHD void reset() { mCachedData = nullptr; mCachedSize = 0; mPairData = 0; mManifoldFlags = 0; }
};

// 디버그 그리기(PCM_LOW_LEVEL_DEBUG, drawManifold 등): 호출되지 않지만 본문이 파싱되게 아무것도 안 하는 판
struct PxMat44 {
  template <class... A> EHD PxMat44(const A&...) {}
};
class PxRenderOutput {
 public:
  enum Primitive { POINTS, LINES, LINESTRIP, TRIANGLES, TRIANGLESTRIP };
  template <class T> EHD PxRenderOutput& operator<<(const T&) { return *this; }
};

namespace Gu {
struct SDF;  // GuSDF.h (볼록 메시에는 없음, 포인터만)
}

// GuConvexSupportTable.cpp:33 boxVertexTable[8] = {BFFFF, BTFFF, BFTFF, BTTFF, BFFTF, BTFTF, BFTTF, BTTTF}
// (번호의 비트 0·1·2 = x·y·z 칸 참) — 전역 표 대신 같은 값을 만든다 (GPU 에서 읽을 수 있게)
EHD aos::BoolV boxVertexTable_get(PxU32 i) {
  const float t = em_u2f(0xFFFFFFFFu);
  return em128{{(i & 1) ? t : 0.0f, (i & 2) ? t : 0.0f, (i & 4) ? t : 0.0f, 0.0f}};
}

}  // namespace px
namespace px {
EHDI int Pxstrcmp(const char* a, const char* b) { while (*a && *a == *b) { ++a; ++b; } return int((unsigned char)*a) - int((unsigned char)*b); }
}
