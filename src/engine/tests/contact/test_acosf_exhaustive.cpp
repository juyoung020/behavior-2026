// glibc 2.35 acosf 이식본(core/contact/px/glibc_acosf.h) = 호스트 libm ::acosf, float 2^32 개 전부 비트 비교.
// MXCSR 기본(반올림 가까운 짝수)과 PhysX 좁은 단계와 같은 FTZ+DAZ 두 경우를 다 본다. NaN 도 비트까지 비교.
//   test_acosf_exhaustive [--threads T]
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/contact/px/glibc_acosf.h"

extern "C" float acosf(float);

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  int nt = int(std::thread::hardware_concurrency());
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
  if (nt < 1) nt = 1;
  int fails = 0;
  for (int mode = 0; mode < 2; ++mode) {
    std::atomic<uint64_t> bad{0};
    std::atomic<uint64_t> firstBad{~0ull};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        if (mode == 1) _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));
        uint64_t b = 0;
        const uint64_t lo = (uint64_t(1) << 32) * t / nt, hi = (uint64_t(1) << 32) * (t + 1) / nt;
        for (uint64_t u = lo; u < hi; ++u) {
          float x;
          const uint32_t ux = uint32_t(u);
          memcpy(&x, &ux, 4);
          const float r0 = acosf(x);
          const float r1 = eng::glibcx::acosf(x);
          if (bits(r0) != bits(r1)) {
            ++b;
            uint64_t cur = firstBad.load();
            while (u < cur && !firstBad.compare_exchange_weak(cur, u)) {}
          }
        }
        bad += b;
      });
    for (auto& x : th) x.join();
    printf("acosf %s: 입력 4294967296 개, 비트 다름 %" PRIu64, mode ? "FTZ+DAZ" : "기본 MXCSR", bad.load());
    if (bad) {
      const uint32_t ux = uint32_t(firstBad.load());
      float x;
      memcpy(&x, &ux, 4);
      printf("  첫 다름 입력 0x%08x (%.9g)", ux, x);
      ++fails;
    }
    printf("\n");
  }
  printf(fails ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return fails ? 1 : 0;
}
