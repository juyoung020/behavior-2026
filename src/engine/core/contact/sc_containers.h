// 쌍 관리층(sc_pairs.h)이 쓰는 작은 컨테이너 (호스트·GPU 공용, 힙 = malloc/free, GPU 는 장치 힙).
// 순서가 결과에 영향 있는 곳은 원본과 같은 연산(뒤에 붙이기·마지막을 그 자리로)만 쓰므로 늘리는 규칙은 결과와 무관하다.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#if defined(__CUDACC__)
#define SCHD __host__ __device__ inline
#define SCHDV __host__ __device__
#else
#define SCHD inline
#define SCHDV
#endif

namespace eng {
namespace contact {
namespace sc {

SCHD uint32_t ctz32(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return uint32_t(__ffs(int(v)) - 1);
#else
  return uint32_t(__builtin_ctz(v));
#endif
}

template <class T>
class Vec {
 public:
  SCHD Vec() : d_(nullptr), n_(0), cap_(0) {}
  SCHD explicit Vec(uint32_t n) : d_(nullptr), n_(0), cap_(0) { resize(n); }
  SCHD Vec(const Vec& o) : d_(nullptr), n_(0), cap_(0) { copyFrom(o); }
  SCHD Vec& operator=(const Vec& o) {
    if (this != &o) {
      clear();
      copyFrom(o);
    }
    return *this;
  }
  SCHD ~Vec() {
    clear();
    free(d_);
  }
  SCHD uint32_t size() const { return n_; }
  SCHD bool empty() const { return n_ == 0; }
  SCHD T* data() { return d_; }
  SCHD const T* data() const { return d_; }
  SCHD T* begin() { return d_; }
  SCHD T* end() { return d_ + n_; }
  SCHD const T* begin() const { return d_; }
  SCHD const T* end() const { return d_ + n_; }
  SCHD T& operator[](size_t i) { return d_[i]; }
  SCHD const T& operator[](size_t i) const { return d_[i]; }
  SCHD T& back() { return d_[n_ - 1]; }
  SCHD const T& back() const { return d_[n_ - 1]; }
  SCHD void push_back(const T& v) {
    if (n_ == cap_) {
      const T copy = v;  // v 가 배열 안을 가리킬 수 있음
      grow(cap_ ? cap_ * 2 : 4);
      new (d_ + n_) T(copy);
    } else {
      new (d_ + n_) T(v);
    }
    ++n_;
  }
  SCHD void pop_back() { d_[--n_].~T(); }
  SCHD void clear() {
    for (uint32_t i = 0; i < n_; ++i) d_[i].~T();
    n_ = 0;
  }
  SCHD void reserve(uint32_t n) {
    if (n > cap_) grow(n);
  }
  SCHD void resize(uint32_t n) {
    reserve(n);
    for (uint32_t i = n_; i < n; ++i) new (d_ + i) T();
    for (uint32_t i = n; i < n_; ++i) d_[i].~T();
    n_ = n;
  }
  SCHD void resize(uint32_t n, const T& v) {
    reserve(n);
    for (uint32_t i = n_; i < n; ++i) new (d_ + i) T(v);
    for (uint32_t i = n; i < n_; ++i) d_[i].~T();
    n_ = n;
  }
  SCHD void assign(uint32_t n, const T& v) {
    clear();
    reserve(n);
    for (uint32_t i = 0; i < n; ++i) new (d_ + i) T(v);
    n_ = n;
  }

 private:
  T* d_;
  uint32_t n_, cap_;
  SCHD void grow(uint32_t c) {
    T* nd = static_cast<T*>(malloc(sizeof(T) * c));
    for (uint32_t i = 0; i < n_; ++i) {
      new (nd + i) T(d_[i]);
      d_[i].~T();
    }
    free(d_);
    d_ = nd;
    cap_ = c;
  }
  SCHD void copyFrom(const Vec& o) {
    reserve(o.n_);
    for (uint32_t i = 0; i < o.n_; ++i) new (d_ + i) T(o.d_[i]);
    n_ = o.n_;
  }
};

// 64 비트 열쇠 -> 값 (찾기·넣기·빼기만, 순회 없음 — 순서가 결과에 들어가지 않는 표에만 쓴다)
template <class V>
class Map {
 public:
  SCHD Map() : keys_(nullptr), vals_(nullptr), used_(nullptr), cap_(0), n_(0) {}
  SCHD Map(const Map&) = delete;
  SCHD Map& operator=(const Map&) = delete;
  SCHD ~Map() { release(); }
  SCHD V* findPtr(uint64_t k) {
    if (!cap_) return nullptr;
    for (uint32_t i = slot(k);; i = (i + 1) & (cap_ - 1)) {
      if (used_[i] == 0) return nullptr;
      if (used_[i] == 1 && keys_[i] == k) return &vals_[i];
    }
  }
  SCHD const V* findPtr(uint64_t k) const { return const_cast<Map*>(this)->findPtr(k); }
  SCHD V& operator[](uint64_t k) {
    if (V* v = findPtr(k)) return *v;
    if ((n_ + tomb_ + 1) * 2 > cap_) rehash(cap_ ? cap_ * 2 : 64);
    uint32_t i = slot(k);
    while (used_[i] == 1) i = (i + 1) & (cap_ - 1);
    if (used_[i] == 2) --tomb_;
    used_[i] = 1;
    keys_[i] = k;
    new (vals_ + i) V();
    ++n_;
    return vals_[i];
  }
  SCHD void erase(uint64_t k) {
    V* v = findPtr(k);
    if (!v) return;
    const uint32_t i = uint32_t(v - vals_);
    vals_[i].~V();
    used_[i] = 2;
    --n_;
    ++tomb_;
  }
  SCHD void clear() {
    for (uint32_t i = 0; i < cap_; ++i) {
      if (used_[i] == 1) vals_[i].~V();
      used_[i] = 0;
    }
    n_ = 0;
    tomb_ = 0;
  }

 private:
  uint64_t* keys_;
  V* vals_;
  uint8_t* used_;  // 0 빈 칸, 1 사용, 2 지움 표시
  uint32_t cap_, n_, tomb_ = 0;
  SCHD uint32_t slot(uint64_t k) const {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    return uint32_t(k) & (cap_ - 1);
  }
  SCHD void rehash(uint32_t nc) {
    uint64_t* ok = keys_;
    V* ov = vals_;
    uint8_t* ou = used_;
    const uint32_t oc = cap_;
    keys_ = static_cast<uint64_t*>(malloc(sizeof(uint64_t) * nc));
    vals_ = static_cast<V*>(malloc(sizeof(V) * nc));
    used_ = static_cast<uint8_t*>(malloc(nc));
    for (uint32_t i = 0; i < nc; ++i) used_[i] = 0;
    cap_ = nc;
    n_ = 0;
    tomb_ = 0;
    for (uint32_t i = 0; i < oc; ++i)
      if (ou[i] == 1) {
        (*this)[ok[i]] = ov[i];
        ov[i].~V();
      }
    free(ok);
    free(ov);
    free(ou);
  }
  SCHD void release() {
    for (uint32_t i = 0; i < cap_; ++i)
      if (used_[i] == 1) vals_[i].~V();
    free(keys_);
    free(vals_);
    free(used_);
    keys_ = nullptr;
    vals_ = nullptr;
    used_ = nullptr;
    cap_ = n_ = tomb_ = 0;
  }
};

}  // namespace sc
}  // namespace contact
}  // namespace eng
