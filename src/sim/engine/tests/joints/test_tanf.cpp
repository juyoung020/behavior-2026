// joints 시험: glibc 2.35 tanf 이식본(core/joints/glibc_tanf.h) = 이 PC libm, float 2^32 전부 비트 비교.
// FTZ/DAZ 끔과 켬(PhysX simulate 안의 PxSIMDGuard 와 같은 MXCSR) 둘 다. NaN 도 비트까지.
// (atanf/atan2f/asinf/acosf 는 공용판 core/common/glibc_trig.h 로 옮겨 tests/common/test_glibc_trig 가 맡는다.)
//   test_tanf [--threads T]
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/joints/glibc_tanf.h"

static uint32_t fu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float uf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
struct Mx {
  unsigned old;
  explicit Mx(bool ftz) { old = _mm_getcsr(); _mm_setcsr(ftz ? (_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6)) : _MM_MASK_MASK); }
  ~Mx() { _mm_setcsr(old); }
};
static float (*volatile ref_tanf)(float) = ::tanf;

int main(int argc, char** argv) {
  int nt = int(std::thread::hardware_concurrency());
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
  bool allOk = true;
  for (int ftz = 0; ftz < 2; ++ftz) {
    std::atomic<uint64_t> bad{0};
    std::atomic<int64_t> first{-1};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        Mx m(ftz != 0);
        const uint64_t n = 1ull << 32, lo = n * uint64_t(t) / uint64_t(nt), hi = n * uint64_t(t + 1) / uint64_t(nt);
        uint64_t b = 0;
        for (uint64_t u = lo; u < hi; ++u) {
          const float x = uf(uint32_t(u));
          if (fu(ref_tanf(x)) != fu(eng::glibc::tanf(x)) && b++ == 0) {
            int64_t e = -1;
            first.compare_exchange_strong(e, int64_t(u));
          }
        }
        bad += b;
      });
    for (auto& x : th) x.join();
    printf("  tanf FTZ %s: 입력 2^32 개, 비트 다름 %" PRIu64, ftz ? "켬" : "끔", bad.load());
    if (bad) {
      Mx m(ftz != 0);
      const float x = uf(uint32_t(first.load()));
      printf("  첫 다름 x=%08x libm=%08x 이식=%08x", fu(x), fu(ref_tanf(x)), fu(eng::glibc::tanf(x)));
    }
    printf("\n");
    allOk &= bad == 0;
  }
  printf("%s\n", allOk ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  return allOk ? 0 : 3;
}
