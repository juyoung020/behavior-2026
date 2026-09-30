// sleef_trigf.h 시험용 C 입구
#include <cstdint>
#include "core/common/sleef_trigf.h"
extern "C" void sl_batch(const float* x, float* s, float* c, int64_t n) {
  for (int64_t i = 0; i < n; ++i) { s[i] = eng::sleef::sinf_u10(x[i]); c[i] = eng::sleef::cosf_u10(x[i]); }
}
