// rcpps/rsqrtps 표 흉내(core/contact/px/approx.h, GPU 판이 쓰는 식) = 이 CPU 의 진짜 명령, float 2^32 개 전부 비트 비교.
// MXCSR 기본·FTZ+DAZ 두 경우. 이 시험이 통과한 CPU 에서만 "가수 위 12 비트만의 함수" 가정이 맞다.
//   test_approx_exhaustive [--threads T]
#include <xmmintrin.h>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/contact/px/approx.h"

int main(int argc, char** argv) {
  int nt = int(std::thread::hardware_concurrency());
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
  if (nt < 1) nt = 1;
  eng::px::ApproxTables tab;
  eng::px::buildApproxTables(tab);
  int fails = 0;
  for (int mode = 0; mode < 2; ++mode) {
    std::atomic<uint64_t> badR{0}, badS{0};
    std::atomic<uint64_t> firstR{~0ull}, firstS{~0ull};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        if (mode == 1) _mm_setcsr(_MM_MASK_MASK | _MM_FLUSH_ZERO_ON | (1 << 6));
        uint64_t br = 0, bs = 0, fr = ~0ull, fs = ~0ull;
        const uint64_t lo = (uint64_t(1) << 32) * t / nt, hi = (uint64_t(1) << 32) * (t + 1) / nt;
        for (uint64_t v = lo; v < hi; ++v) {
          const float x = eng::px::em_u2f(uint32_t(v));
          const uint32_t r0 = eng::px::em_f2u(_mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))));
          const uint32_t r1 = eng::px::em_f2u(eng::px::approxRcp(x, tab));
          if (r0 != r1) { if (!br) fr = v; ++br; }
          const uint32_t s0 = eng::px::em_f2u(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))));
          const uint32_t s1 = eng::px::em_f2u(eng::px::approxRsqrt(x, tab));
          if (s0 != s1) { if (!bs) fs = v; ++bs; }
        }
        badR += br; badS += bs;
        uint64_t c = firstR.load(); while (fr < c && !firstR.compare_exchange_weak(c, fr)) {}
        c = firstS.load(); while (fs < c && !firstS.compare_exchange_weak(c, fs)) {}
      });
    for (auto& x : th) x.join();
    printf("%s: rcpps 비트 다름 %" PRIu64 ", rsqrtps 비트 다름 %" PRIu64, mode ? "FTZ+DAZ" : "기본 MXCSR", badR.load(), badS.load());
    if (badR) printf("  rcp 첫 다름 0x%08" PRIx64, firstR.load());
    if (badS) printf("  rsqrt 첫 다름 0x%08" PRIx64, firstS.load());
    printf("  (입력 4294967296 개씩)\n");
    if (badR || badS) ++fails;
  }
  printf(fails ? "결과: 비트 다름 있음\n" : "결과: 전부 비트 동일\n");
  return fails ? 1 : 0;
}
