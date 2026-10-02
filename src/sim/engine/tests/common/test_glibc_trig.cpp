// core/common/glibc_trig.h 시험 (joints 작업자 시험 tests/joints/test_libm_joints* 를 공용판으로 옮김).
// joints 시험: glibc 2.35 atanf/asinf/acosf(float 2^32 전부)와 atan2f(무작위 쌍 + 특수값 격자) 이식본 = 이 PC libm, 비트 비교.
// FTZ/DAZ 끔과 켬(PhysX simulate 안의 PxSIMDGuard 와 같은 MXCSR) 둘 다. NaN 은 비트까지 비교(조용한 NaN 무늬 포함).
//   test_glibc_trig [--pairs N] [--threads T]
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "core/common/glibc_trig.h"

namespace G = eng::glibc;

static uint32_t fu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float uf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

struct Mx {
  unsigned old;
  explicit Mx(bool ftz) {
    old = _mm_getcsr();
    _mm_setcsr(ftz ? (_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)) : _MM_MASK_MASK);
  }
  ~Mx() { _mm_setcsr(old); }
};

// 참조: 진짜 libm (함수 포인터로 불러 컴파일러 대체를 막는다)
static float (*volatile ref_atanf)(float) = ::atanf;
static float (*volatile ref_asinf)(float) = ::asinf;
static float (*volatile ref_acosf)(float) = ::acosf;
static float (*volatile ref_atan2f)(float, float) = ::atan2f;

int main(int argc, char** argv) {
  uint64_t pairs = 1ull << 28;
  int nt = int(std::thread::hardware_concurrency());
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
  }
  bool allOk = true;
  const char* names[3] = {"atanf", "asinf", "acosf"};
  for (int ftz = 0; ftz < 2; ++ftz) {
    for (int f = 0; f < 3; ++f) {
      std::atomic<uint64_t> bad{0};
      std::atomic<int64_t> first{-1};
      std::vector<std::thread> th;
      for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t]() {
          Mx m(ftz != 0);
          const uint64_t n = 1ull << 32, lo = n * uint64_t(t) / uint64_t(nt), hi = n * uint64_t(t + 1) / uint64_t(nt);
          uint64_t b = 0;
          for (uint64_t u = lo; u < hi; ++u) {
            const float x = uf(uint32_t(u));
            const float r = f == 0 ? ref_atanf(x) : (f == 1 ? ref_asinf(x) : ref_acosf(x));
            const float e = f == 0 ? G::atanf(x) : (f == 1 ? G::asinf(x) : G::acosf(x));
            if (fu(r) != fu(e)) {
              if (b++ == 0) {
                int64_t exp = -1;
                first.compare_exchange_strong(exp, int64_t(u));
              }
            }
          }
          bad += b;
        });
      }
      for (auto& x : th) x.join();
      printf("  %-6s FTZ %s: 입력 2^32 개, 비트 다름 %" PRIu64, names[f], ftz ? "켬" : "끔", bad.load());
      if (bad) {
        const float x = uf(uint32_t(first.load()));
        Mx m(ftz != 0);
        const float r = f == 0 ? ref_atanf(x) : (f == 1 ? ref_asinf(x) : ref_acosf(x));
        const float e = f == 0 ? G::atanf(x) : (f == 1 ? G::asinf(x) : G::acosf(x));
        printf("  첫 다름 x=%08x libm=%08x 이식=%08x", fu(x), fu(r), fu(e));
      }
      printf("\n");
      allOk &= bad == 0;
    }
    // atan2f: 특수값 격자 + 무작위 쌍 (보통 값·0·무한·NaN·비정규 섞음)
    std::vector<float> sp = {0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, uf(0x7fc00000u), uf(0xffc00001u), uf(0x7f800001u),
                             uf(0x00000001u), uf(0x80000001u), uf(0x007fffffu), 1e-30f, -1e30f, 3.0f, 0.5f, 2.4375f, 1e20f, -1e-20f};
    uint64_t bad2 = 0, cmp2 = 0;
    {
      Mx m(ftz != 0);
      for (float a : sp)
        for (float b : sp) {
          cmp2++;
          if (fu(ref_atan2f(a, b)) != fu(G::atan2f(a, b))) {
            if (bad2 == 0) printf("  atan2f 특수 첫 다름 y=%08x x=%08x libm=%08x 이식=%08x\n", fu(a), fu(b), fu(ref_atan2f(a, b)), fu(G::atan2f(a, b)));
            bad2++;
          }
        }
    }
    std::atomic<uint64_t> bad3{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
      th.emplace_back([&, t]() {
        Mx m(ftz != 0);
        std::mt19937_64 rng(1234 + uint64_t(t) + 1000ull * uint64_t(ftz));
        const uint64_t cnt = pairs / uint64_t(nt);
        uint64_t b = 0;
        for (uint64_t i = 0; i < cnt; ++i) {
          const uint64_t r = rng();
          uint32_t ya = uint32_t(r), xa = uint32_t(r >> 32);
          // 절반은 비슷한 크기(지수 차이 작게) — 실제 쓰임(4*atan2(swing.y, 1+swing.w))과 비슷한 영역
          if (i & 1) {
            const float yy = uf(ya);
            const float xx = uf((xa & 0x807fffffu) | (fu(fabsf(yy)) & 0x7f800000u));
            ya = fu(yy); xa = fu(xx);
          }
          const float y = uf(ya), x = uf(xa);
          if (fu(ref_atan2f(y, x)) != fu(G::atan2f(y, x))) b++;
        }
        bad3 += b;
      });
    }
    for (auto& x : th) x.join();
    printf("  atan2f FTZ %s: 특수값 %" PRIu64 " 쌍 다름 %" PRIu64 ", 무작위 %" PRIu64 " 쌍 다름 %" PRIu64 "\n", ftz ? "켬" : "끔", cmp2, bad2,
           pairs / uint64_t(nt) * uint64_t(nt), bad3.load());
    allOk &= bad2 == 0 && bad3 == 0;
  }
  printf("%s\n", allOk ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  return allOk ? 0 : 3;
}
