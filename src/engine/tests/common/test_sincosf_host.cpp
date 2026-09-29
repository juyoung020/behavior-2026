// glibc sinf/cosf 이식판(core/glibc_sincosf.h) = 이 PC libm (::sinf/::cosf) 인가, float 전부(2^32 개)를 비교한다 (CPU, 32 스레드).
//   test_sincosf_host [--fma 0|1 은 컴파일 때 ENG_GLIBC_FMA 로]
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "core/common/glibc_sincosf.h"

int main() {
  const unsigned nt = std::thread::hardware_concurrency();
  std::atomic<uint64_t> bad_s{0}, bad_c{0};
  std::atomic<int64_t> first_s{-1}, first_c{-1};
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      const uint64_t lo = (uint64_t(1) << 32) * t / nt, hi = (uint64_t(1) << 32) * (t + 1) / nt;
      for (uint64_t u = lo; u < hi; ++u) {
        const uint32_t bits = uint32_t(u);
        float x;
        memcpy(&x, &bits, 4);
        const float a = ::sinf(x), b = eng::glibc::sinf(x);
        const float c = ::cosf(x), d = eng::glibc::cosf(x);
        const bool nan_ok_s = std::isnan(a) && std::isnan(b), nan_ok_c = std::isnan(c) && std::isnan(d);
        if (!nan_ok_s && memcmp(&a, &b, 4)) { if (bad_s++ == 0) first_s = int64_t(u); }
        if (!nan_ok_c && memcmp(&c, &d, 4)) { if (bad_c++ == 0) first_c = int64_t(u); }
      }
    });
  for (auto& x : th) x.join();
  printf("float 2^32 개 전부 (FMA 판=%d): sinf 다름 %" PRIu64 ", cosf 다름 %" PRIu64 "\n", ENG_GLIBC_FMA, bad_s.load(), bad_c.load());
  if (first_s >= 0) { uint32_t b = uint32_t(first_s); float x; memcpy(&x, &b, 4); printf("  sinf 첫 다름 x=%.9g (0x%08x) libm=%.9g 이식=%.9g\n", x, b, ::sinf(x), eng::glibc::sinf(x)); }
  if (first_c >= 0) { uint32_t b = uint32_t(first_c); float x; memcpy(&x, &b, 4); printf("  cosf 첫 다름 x=%.9g (0x%08x) libm=%.9g 이식=%.9g\n", x, b, ::cosf(x), eng::glibc::cosf(x)); }
  return (bad_s || bad_c) ? 3 : 0;
}
